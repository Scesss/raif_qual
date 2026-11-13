"""Official SIWC OAuth + streamed Responses. No desktop credentials or browser cookies."""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import secrets
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlencode

ISSUER = 'https://auth.openai.com'
AUTHORIZE = ISSUER + '/api/accounts/authorize'
TOKEN = ISSUER + '/api/accounts/oauth/token'
RESOURCE = 'https://api.openai.com/v1'
SCOPE = 'openid profile email offline_access resource.invoke chatgpt.tokens.use.direct'
RESPONSE_READ_TIMEOUT = 180
RESPONSE_WRITE_TIMEOUT = 60
REQUEST_ATTEMPTS = 3
RETRY_BASE_DELAY = 1


class GatewayError(Exception):
    def __init__(self, code, message, *, request_id=None, transport_error=None):
        super().__init__(message)
        self.code, self.request_id = code, request_id
        self.transport_error = transport_error

    def public(self):
        result = {'code': self.code, 'message': str(self), 'request_id': self.request_id}
        if self.transport_error:
            result['transport_error'] = self.transport_error
        return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def http_client():
    import httpx
    return httpx.Client(timeout=httpx.Timeout(20, connect=10), follow_redirects=False)


class Credentials:
    def __init__(self, directory=None, client_factory=http_client):
        self.directory = Path(directory or os.environ.get('REVIEWER_AUTH_DIR', '~/.config/homework-reviewer')).expanduser()
        self.path = self.directory / 'connections.json'
        self.client_factory = client_factory
        self._lock = threading.RLock()
        self.pending = {}

    @contextmanager
    def locked(self):
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            with (self.directory / 'refresh.lock').open('a') as lock:
                os.chmod(lock.name, 0o600)
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def read(self):
        return load_json(self.path, {'profiles': {}, 'active': None})

    def public(self):
        with self._lock:
            data = self.read()
            profiles = [{'id': key, 'email': p.get('email', ''), 'subject': p['subject'],
                         'connected': bool(p.get('refresh_token') or p.get('access_token')),
                         'plan_enabled': 'chatgpt.tokens.use.direct' in p.get('scopes', [])}
                        for key, p in data['profiles'].items()]
        return {'profiles': profiles, 'active_profile': data['active']}

    def select(self, profile_id):
        with self.locked():
            data = self.read()
            if profile_id not in data['profiles']:
                raise ValueError('Подключение не найдено.')
            data['active'] = profile_id
            atomic_json(self.path, data)

    def start(self, redirect_uri, profile_id=None):
        with self.locked():
            data = self.read()
            profile = data['profiles'].get(profile_id) if profile_id else None
            if profile_id and not profile:
                raise ValueError('Подключение не найдено.')
            if 'host_id' not in data:
                data['host_id'] = 'urn:uuid:' + str(uuid.uuid4())
                atomic_json(self.path, data)
            state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
            self.pending = {k: p for k, p in self.pending.items() if p['expires'] > time.time()}
            self.pending[state] = {'nonce': nonce, 'verifier': verifier, 'redirect_uri': redirect_uri,
                                   'profile_id': profile_id, 'expires': time.time() + 600}
            params = {'client_id': profile_id or 'dynamic_agent_client', 'ext_agent_host_id': data['host_id'],
                      'response_type': 'code', 'redirect_uri': redirect_uri, 'scope': SCOPE,
                      'resource': RESOURCE, 'state': state, 'nonce': nonce, 'code_challenge_method': 'S256',
                      'code_challenge': base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')}
            if profile:
                if profile.get('id_token'):
                    params['id_token_hint'] = profile['id_token']
                if profile.get('email'):
                    params['login_hint'] = profile['email']
            else:
                params['agent_name_hint'] = 'GradeAI'
            return AUTHORIZE + '?' + urlencode(params)

    def _claims(self, token, client_id, nonce=None):
        import jwt
        try:
            with self.client_factory() as client:
                discovery_response = client.get(ISSUER + '/.well-known/openid-configuration')
                discovery_response.raise_for_status()
                discovery = discovery_response.json()
                jwks_uri = discovery['jwks_uri']
                if discovery['issuer'] != ISSUER or not jwks_uri.startswith(ISSUER + '/'):
                    raise ValueError('Unexpected issuer')
                keys_response = client.get(jwks_uri)
                keys_response.raise_for_status()
                keys = jwt.PyJWKSet.from_dict(keys_response.json())
            header = jwt.get_unverified_header(token)
            if header.get('alg') not in ('RS256', 'ES256'):
                raise ValueError('Unsupported JWT algorithm')
            key = next(k for k in keys.keys if k.key_id == header.get('kid'))
            claims = jwt.decode(token, key.key, algorithms=[header['alg']], audience=client_id, issuer=ISSUER,
                                options={'require': ['exp', 'sub', 'iss', 'aud']}, leeway=15)
            if not isinstance(claims['sub'], str) or not claims['sub']:
                raise ValueError('Missing subject')
            if nonce is not None and not secrets.compare_digest(str(claims.get('nonce', '')), nonce):
                raise ValueError('Invalid nonce')
            if claims.get('azp', client_id) != client_id:
                raise ValueError('Invalid authorized party')
            return claims
        except Exception:
            raise GatewayError('AuthRequired', 'Не удалось проверить подпись или параметры ID token. Повторите вход.') from None

    def _token_request(self, fields):
        try:
            with self.client_factory() as client:
                response = client.post(TOKEN, data=fields)
                if response.status_code >= 400:
                    code = 'AuthRequired' if response.status_code < 500 else 'ProviderTransient'
                    raise GatewayError(code, 'Сессия не обновлена. Повторите вход или попробуйте позже.')
                result = response.json()
                if not result.get('access_token') or result.get('token_type', '').lower() != 'bearer':
                    raise GatewayError('AuthRequired', 'Провайдер не выдал Bearer access token.')
                return result
        except GatewayError:
            raise
        except Exception:
            raise GatewayError('ProviderTransient', 'Не удалось связаться с сервером авторизации.') from None

    def finish(self, query):
        state = query.get('state', [''])[0]
        with self.locked():
            pending = self.pending.pop(state, None)
            if not pending or pending['expires'] < time.time():
                raise GatewayError('AuthRequired', 'Вход устарел или state не совпадает. Начните вход заново.')
            if query.get('error'):
                raise GatewayError('AuthRequired', 'Вход отменён или разрешение не выдано.')
            old_id = pending['profile_id']
            client_id = query.get('client_id', [old_id])[0]
            if not client_id or client_id == 'dynamic_agent_client' or (old_id and client_id != old_id) or not query.get('code'):
                raise GatewayError('AuthRequired', 'Регистрация не завершена: неверный client_id или отсутствует код.')
            tokens = self._token_request({'grant_type': 'authorization_code', 'client_id': client_id,
                                         'code': query['code'][0], 'code_verifier': pending['verifier'],
                                         'redirect_uri': pending['redirect_uri'], 'resource': RESOURCE})
            claims = self._claims(tokens.get('id_token', ''), client_id, pending['nonce'])
            data = self.read()
            previous = data['profiles'].get(client_id)
            if previous and previous['subject'] != claims['sub']:
                raise GatewayError('AuthRequired', 'Аккаунт не совпадает с выбранным подключением.')
            data['profiles'][client_id] = {**tokens, 'client_id': client_id, 'subject': claims['sub'],
                'email': claims.get('email', ''), 'scopes': tokens.get('scope', '').split(),
                'expires_at': time.time() + float(tokens.get('expires_in', 3600))}
            data['active'] = client_id
            atomic_json(self.path, data)
            return 'chatgpt.tokens.use.direct' in data['profiles'][client_id]['scopes']

    def access_token(self):
        with self.locked():
            data = self.read()
            profile_id = data['active']
            profile = data['profiles'].get(profile_id)
            if not profile or not profile.get('access_token'):
                raise GatewayError('AuthRequired', 'Подключите ChatGPT в разделе «Агент».')
            if 'chatgpt.tokens.use.direct' not in profile.get('scopes', []):
                raise GatewayError('ScopeMissing', 'Вход выполнен, но использование подписки не разрешено. Подключитесь повторно с разрешением плана.')
            if profile['expires_at'] < time.time() + 60:
                if not profile.get('refresh_token'):
                    raise GatewayError('AuthRequired', 'Сессия истекла. Повторите вход.')
                tokens = self._token_request({'grant_type': 'refresh_token', 'client_id': profile_id,
                                             'refresh_token': profile['refresh_token'], 'resource': RESOURCE})
                if tokens.get('id_token'):
                    claims = self._claims(tokens['id_token'], profile_id)
                    if claims['sub'] != profile['subject']:
                        raise GatewayError('AuthRequired', 'При обновлении изменился аккаунт. Повторите вход.')
                profile.update(tokens)
                profile['expires_at'] = time.time() + float(tokens.get('expires_in', 3600))
                if 'scope' in tokens:
                    profile['scopes'] = tokens['scope'].split()
                atomic_json(self.path, data)
                if 'chatgpt.tokens.use.direct' not in profile['scopes']:
                    raise GatewayError('ScopeMissing', 'Разрешение использовать подписку отозвано.')
            return profile['access_token']

    def logout(self):
        confirmed = False
        with self.locked():
            data = self.read()
            profile = data['profiles'].get(data['active'])
            if profile:
                if profile.get('refresh_token'):
                    try:
                        with self.client_factory() as client:
                            response = client.get(ISSUER + '/.well-known/openid-configuration')
                            response.raise_for_status()
                            endpoint = response.json()['revocation_endpoint']
                            if endpoint.startswith(ISSUER + '/'):
                                for attempt in range(3):
                                    result = client.post(endpoint, data={'token': profile['refresh_token'],
                                        'token_type_hint': 'refresh_token', 'client_id': data['active']})
                                    if result.status_code == 200:
                                        confirmed = True
                                        break
                                    if result.status_code < 500:
                                        break
                                    time.sleep(0.2 * (2 ** attempt))
                    except Exception:
                        pass
                for key in ('access_token', 'refresh_token', 'id_token'):
                    profile.pop(key, None)
                profile['scopes'] = []
                atomic_json(self.path, data)
        return {'signed_out': True, 'remote_revocation_confirmed': confirmed}


def provider_error(status, body, request_id=None):
    error = body.get('error') if isinstance(body, dict) else None
    error = error if isinstance(error, dict) else {}
    code = str(error.get('code', ''))[:120]
    if status == 401 or code in ('invalid_api_key', 'token_expired'):
        category, message = 'AuthRequired', 'Авторизация истекла или отклонена. Повторите подключение.'
    elif status == 429 or 'usage_limit' in code or 'usage_unavailable' in code or 'quota' in code:
        category, message = 'CapacityUnavailable', 'Лимит модели недоступен. Повторите позже; платный режим автоматически не включается.'
    elif code == 'model_not_found':
        category, message = 'ModelUnavailable', 'Выбранная модель недоступна. Обновите список моделей.'
    elif status >= 500:
        category, message = 'ProviderTransient', 'Провайдер временно недоступен.'
    else:
        category, message = 'InputUnsupported', 'Провайдер отклонил запрос. Проверьте модель и совместимость инструментов.'
    result = GatewayError(category, message, request_id=request_id)
    result.provider_code = code
    return result


def parse_sse(lines, cancel, on_event):
    buffer = []
    for line in lines:
        if cancel.is_set():
            raise GatewayError('Canceled', 'Проверка остановлена.')
        if line.startswith('data:'):
            buffer.append(line[5:].lstrip())
        elif not line and buffer:
            raw, buffer = '\n'.join(buffer), []
            if raw == '[DONE]':
                break
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                raise GatewayError('StreamIncomplete', 'Провайдер прислал повреждённое событие.') from None
            kind = event.get('type')
            if kind == 'response.output_text.delta':
                on_event('text_delta', {'text': str(event.get('delta', ''))[:4000]})
            elif kind in ('response.failed', 'error'):
                raise provider_error(400, event.get('response', event))
            elif kind == 'response.incomplete':
                raise GatewayError('StreamIncomplete', 'Провайдер не завершил ответ.')
            elif kind == 'response.completed':
                response = event.get('response', {})
                if response.get('status') != 'completed' or not isinstance(response.get('output'), list):
                    raise GatewayError('StreamIncomplete', 'Ответ не содержит завершённого результата.')
                return response
    raise GatewayError('StreamIncomplete', 'Поток оборвался без response.completed. Черновик не сохранён.')


class ModelGateway:
    def __init__(self, config_path, credentials=None, client_factory=http_client):
        self.config_path = Path(config_path)
        self.credentials = credentials or Credentials()
        self.client_factory = client_factory
        self._lock = threading.Lock()

    def config(self):
        defaults = {'provider': 'subscription', 'model': os.environ.get('REVIEWER_MODEL', '')}
        return {**defaults, **load_json(self.config_path, {})}

    def configure(self, payload):
        provider = payload.get('provider', 'subscription')
        if provider not in ('subscription', 'api', 'replay'):
            raise ValueError('Неизвестный провайдер.')
        if provider == 'replay' and os.environ.get('REVIEWER_DEMO') != '1':
            raise ValueError('Сценарный режим разрешён только в отдельной демонстрационной базе.')
        model = str(payload.get('model', '')).strip()
        if len(model) > 150 or any(ord(c) < 32 for c in model):
            raise ValueError('Некорректное имя модели.')
        with self._lock:
            atomic_json(self.config_path, {'provider': provider, 'model': model})
        return self.status()

    def status(self):
        config = self.config()
        return {**config, **self.credentials.public(), 'api_key_configured': bool(os.environ.get('OPENAI_API_KEY')),
                'demo_allowed': os.environ.get('REVIEWER_DEMO') == '1', 'automatic_paid_fallback': False}

    def token(self, provider):
        if provider == 'subscription':
            return self.credentials.access_token()
        if provider != 'api':
            raise GatewayError('InputUnsupported', 'Сценарный режим не обращается к модели.')
        token = os.environ.get('OPENAI_API_KEY')
        if not token:
            raise GatewayError('AuthRequired', 'Для явно выбранного API-режима задайте OPENAI_API_KEY при запуске сервера.')
        return token

    def models(self):
        config = self.config()
        token = self.token(config['provider'])
        try:
            with self.client_factory() as client:
                response = client.get(RESOURCE + '/models', headers={'Authorization': 'Bearer ' + token})
                if response.status_code >= 400:
                    raise provider_error(response.status_code, {}, response.headers.get('x-request-id'))
                data = response.json()
                if 'models' in data:
                    return [{'id': m['slug'], 'name': m.get('display_name', m['slug'])} for m in data['models'] if m.get('visibility') == 'list']
                return [{'id': m['id'], 'name': m['id']} for m in data.get('data', [])]
        except GatewayError:
            raise
        except Exception:
            raise GatewayError('ProviderTransient', 'Не удалось получить каталог моделей.') from None

    def complete(self, history, instructions, tools, cancel, on_event, *, config=None):
        config = config or self.config()
        # Only a completed response may reach the tool loop. Retrying this request
        # cannot re-execute tools from a discarded, incomplete response.
        for attempt in range(REQUEST_ATTEMPTS):
            try:
                return self._complete_once(history, instructions, tools, cancel, on_event, config=config)
            except GatewayError as error:
                if cancel.is_set():
                    raise GatewayError('Canceled', 'Проверка остановлена.') from None
                if error.code not in {'ModelTimeout', 'ConnectionFailed', 'StreamDisconnected', 'StreamIncomplete'} or attempt + 1 == REQUEST_ATTEMPTS:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                on_event('request_retry', {
                    'message': 'Запрос к модели не завершился. Повторяем текущий шаг; уже прочитанные страницы и результаты инструментов сохранены в контексте.',
                    'attempt': attempt + 2, 'max_attempts': REQUEST_ATTEMPTS, 'delay_seconds': delay,
                    'code': error.code, 'request_id': error.request_id, 'transport_error': error.transport_error})
                if cancel.wait(delay):
                    raise GatewayError('Canceled', 'Проверка остановлена.') from None

    def _complete_once(self, history, instructions, tools, cancel, on_event, *, config):
        import httpx
        if cancel.is_set():
            raise GatewayError('Canceled', 'Проверка остановлена.')
        if not config['model']:
            raise GatewayError('ModelUnavailable', 'Выберите модель в разделе «Агент».')
        token = self.token(config['provider'])
        payload = {'model': config['model'], 'input': history, 'instructions': instructions,
                   'store': False, 'stream': True, 'tools': tools,
                   'include': ['reasoning.encrypted_content']}
        request_id = None
        try:
            with self.client_factory() as client:
                with client.stream('POST', RESOURCE + '/responses', json=payload,
                                   headers={'Authorization': 'Bearer ' + token},
                                   timeout=httpx.Timeout(RESPONSE_READ_TIMEOUT, connect=10,
                                                         write=RESPONSE_WRITE_TIMEOUT, pool=10)) as response:
                    request_id = response.headers.get('x-request-id')
                    if response.status_code >= 400:
                        response.read()
                        try:
                            body = response.json()
                        except ValueError:
                            body = {}
                        raise provider_error(response.status_code, body, request_id)
                    try:
                        return parse_sse(response.iter_lines(), cancel, on_event)
                    except GatewayError as error:
                        error.request_id = error.request_id or request_id
                        raise
        except GatewayError:
            raise
        except httpx.TimeoutException as error:
            if cancel.is_set():
                raise GatewayError('Canceled', 'Проверка остановлена.') from None
            phase = ('передачи изображений' if isinstance(error, httpx.WriteTimeout) else
                     'ожидания данных от модели' if isinstance(error, httpx.ReadTimeout) else 'подключения к модели')
            raise GatewayError('ModelTimeout', f'Истекло время {phase}. Черновик не сохранён.',
                               request_id=request_id, transport_error=type(error).__name__) from None
        except httpx.ConnectError as error:
            if cancel.is_set():
                raise GatewayError('Canceled', 'Проверка остановлена.') from None
            raise GatewayError('ConnectionFailed', 'Не удалось установить соединение с OpenAI. Проверьте интернет, VPN и DNS. Черновик не сохранён.',
                               request_id=request_id, transport_error=type(error).__name__) from None
        except httpx.TransportError as error:
            if cancel.is_set():
                raise GatewayError('Canceled', 'Проверка остановлена.') from None
            raise GatewayError('StreamDisconnected', 'Соединение с моделью оборвалось. Черновик не сохранён.',
                               request_id=request_id, transport_error=type(error).__name__) from None
        except Exception as error:
            if cancel.is_set():
                raise GatewayError('Canceled', 'Проверка остановлена.') from None
            raise GatewayError('GatewayInternalError', f'Ошибка обработки ответа модели ({type(error).__name__}). Черновик не сохранён.', request_id=request_id) from None

    def probe(self):
        import io
        from PIL import Image, ImageDraw
        picture = Image.new('RGB', (200, 120), 'white')
        ImageDraw.Draw(picture).text((65, 20), '7', fill='black', font_size=70)
        buffer = io.BytesIO()
        picture.save(buffer, format='PNG')
        history = [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'Read the digit in the image and call probe with it. Then repeat the receipt returned by that tool.'},
                    {'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}]}]
        tool = {'type': 'namespace', 'name': 'connection_check', 'description': 'Connection test.', 'tools': [
            {'type': 'function', 'name': 'probe', 'description': 'Report the digit visible in the image.',
             'parameters': {'type': 'object', 'properties': {'digit': {'type': 'integer'}}, 'required': ['digit'], 'additionalProperties': False}}]}
        cancel = threading.Event()
        config = self.config()
        response = self.complete(history, 'Perform the connection test using its tool.', [tool], cancel, lambda *_: None, config=config)
        calls = [i for i in response['output'] if i.get('type') == 'function_call']
        if len(calls) != 1 or calls[0].get('name') != 'probe' or json.loads(calls[0]['arguments']).get('digit') != 7:
            raise GatewayError('InputUnsupported', 'Проверка изображения и вызова инструмента не пройдена.')
        receipt = secrets.token_hex(8)
        history.extend(response['output'])
        history.append({'type': 'function_call_output', 'call_id': calls[0]['call_id'], 'output': json.dumps({'receipt': receipt})})
        result = self.complete(history, 'Repeat the tool receipt exactly.', [tool], cancel, lambda *_: None, config=config)
        text = ''.join(c.get('text', '') for i in result['output'] for c in i.get('content', []) if c.get('type') == 'output_text')
        if receipt not in text:
            raise GatewayError('InputUnsupported', 'Модель не использовала результат инструмента.')
        return {'completed': True, 'image': True, 'tool_call': True, 'tool_result_used': True, 'provider': config['provider'], 'model': config['model']}
