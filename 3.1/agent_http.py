"""Local HTTP surface for model settings, OAuth and review runs."""
import base64
import json
import threading
from urllib.parse import parse_qs, urlparse

from agent_tools import Documents
from model_gateway import GatewayError, ModelGateway
from review_agent import RunManager

_lock = threading.Lock()
_runtime = None


def runtime(store):
    global _runtime
    with _lock:
        if _runtime is None or _runtime.store is not store:
            if _runtime is not None:
                _runtime.close()
            gateway = ModelGateway(store.path.parent / 'agent-settings.json')
            _runtime = RunManager(store, gateway)
        return _runtime


def handle(handler, store, path, *, payload=None, query=None):
    import re
    matched = (path.startswith('/api/model/') or path.startswith('/api/agent/') or path == '/auth/callback'
               or re.fullmatch(r'/api/records/[^/]+/(versions|page|pages)', path))
    if not matched:
        return False
    manager = runtime(store)
    gateway = manager.gateway
    query = query or {}
    if payload is None:
        if path == '/api/model/status':
            handler.respond(200, gateway.status())
        elif path == '/api/model/models':
            handler.respond(200, gateway.models())
        elif path == '/api/agent/runs':
            handler.respond(200, manager.list(query.get('homework_id', [None])[0]))
        elif re.fullmatch(r'/api/agent/runs/[^/]+', path):
            handler.respond(200, manager.get(path.rsplit('/', 1)[1], max(0, int(query.get('after', ['0'])[0]))))
        elif path.endswith('/versions'):
            handler.respond(200, store.review_versions(path.split('/')[3]))
        elif path.endswith('/pages'):
            sources = store.source_manifest(path.split('/')[3])['sources']
            handler.respond(200, {'pages': Documents(sources).count('work')})
        elif path.endswith('/page'):
            review_id = path.split('/')[3]
            sources = store.source_manifest(review_id)['sources']
            page = int(query.get('page', ['1'])[0])
            result, image = Documents(sources).page('work', page)
            if image:
                handler.respond(200, base64.b64decode(image.split(',', 1)[1]), 'image/png')
            else:
                handler.respond(200, result['text'].encode(), 'text/plain; charset=utf-8')
        elif path == '/auth/callback':
            if manager.busy():
                raise ValueError('Дождитесь окончания проверок и повторите вход.')
            enabled = gateway.credentials.finish(query)
            message = 'ChatGPT подключён. Вернитесь в приложение и обновите список моделей.' if enabled else 'Вход выполнен, но разрешение использовать подписку не выдано. Повторите подключение с разрешением плана.'
            handler.respond(200, ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Подключение ChatGPT</title><body><h1>' + message + '</h1><a href="/">Вернуться к проверке ДЗ</a></body></html>').encode(), 'text/html; charset=utf-8')
        else:
            handler.error(404, 'Маршрут не найден.')
    else:
        if path == '/api/agent/runs':
            handler.respond(202, manager.start(payload.get('review_ids')))
        elif re.fullmatch(r'/api/agent/runs/[^/]+/cancel', path):
            handler.respond(200, manager.cancel(path.split('/')[4]))
        elif path in ('/api/model/config', '/api/model/auth/start', '/api/model/auth/select', '/api/model/auth/logout', '/api/model/probe'):
            if manager.busy():
                raise ValueError('Дождитесь завершения или остановите проверки перед изменением подключения.')
            if path == '/api/model/config':
                handler.respond(200, gateway.configure(payload))
            elif path == '/api/model/auth/start':
                port = handler.server.server_address[1]
                url = gateway.credentials.start(f'http://127.0.0.1:{port}/auth/callback', payload.get('profile_id'))
                handler.respond(200, {'authorization_url': url})
            elif path == '/api/model/auth/select':
                gateway.credentials.select(payload.get('profile_id'))
                handler.respond(200, gateway.status())
            elif path == '/api/model/auth/logout':
                handler.respond(200, gateway.credentials.logout())
            else:
                handler.respond(200, gateway.probe())
        else:
            handler.error(404, 'Маршрут не найден.')
    return True
