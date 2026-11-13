"""Explicitly scripted, offline UI/queue exercise. This is NOT an LLM or the submitted AI demo."""
import json
import time
import uuid

from model_gateway import GatewayError


class ReplayGateway:
    def complete(self, history, instructions, tools, cancel, on_event, *, config=None):
        if cancel.wait(0.18):
            raise GatewayError('Canceled', 'Сценарная проверка остановлена.')
        results = []
        for item in history:
            if item.get('type') == 'function_call_output':
                results.append(json.loads(item['output'][0]['text']))
        calls = [i for i in history if i.get('type') == 'function_call']

        def call(name, args):
            on_event('text_delta', {'text': '[Без LLM · сценарная проверка интерфейса] ' + name})
            return {'status': 'completed', 'output': [{'type': 'function_call', 'namespace': 'homework',
                'call_id': 'replay_' + uuid.uuid4().hex, 'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}]}

        if not calls:
            return call('read_context', {})
        pages = [(i['source'], i['page']) for i in results if 'source' in i]
        for source in ('statement', 'rubric', 'work'):
            if (source, 1) not in pages:
                return call('read_page', {'source': source, 'page': 1})
        work = next(r['text'] for r in results if r.get('source') == 'work')
        if 'UNCLEAR' in work or 'утрачен' in work:
            if not any(c['name'] == 'read_page' and 'crop' in json.loads(c['arguments']) for c in calls):
                return call('read_page', {'source': 'work', 'page': 1, 'crop': [0.05, 0.20, 0.95, 0.7]})
            if not any(c['name'] == 'request_manual_reading' for c in calls):
                return call('request_manual_reading', {'page': 1, 'reason': 'Символ операции закрыт даже после увеличения. Нужен исходник или пояснение преподавателя.'})
            return {'status': 'completed', 'output': []}
        if not any(c['name'] == 'check_logic' for c in calls):
            left = {'op': 'and' if 'ERROR' in work or 'конъюнкцию' in work else 'implies', 'args': ['P', 'Q']}
            return call('check_logic', {'left': left, 'right': {'op': 'or', 'args': [{'op': 'not', 'args': ['P']}, 'Q']}})
        counterexample = next(r for r in results if 'counterexample' in r)['counterexample']
        if counterexample is not None and sum(c['name'] == 'read_page' and json.loads(c['arguments'])['source'] == 'work' for c in calls) == 1:
            return call('read_page', {'source': 'work', 'page': 1, 'crop': [0.05, 0.2, 0.95, 0.7]})
        if not any(c['name'] == 'calculate_sets' for c in calls):
            return call('calculate_sets', {'operation': 'intersection', 'left': [1, 2], 'right': [2, 3]})
        if not any(c['name'] == 'submit_draft' for c in calls):
            bad = counterexample is not None
            reason = ('Стр. 1: конъюнкция ошибочно отождествлена с импликацией. При P=false,Q=false слева false, справа true.'
                      if bad else 'Стр. 1: импликация корректно преобразована в ¬P∨Q; все четыре набора совпадают.')
            return call('submit_draft', {'draft': {
                'scores': [{'problem': 1, 'score': 0 if bad else 4, 'page': 1, 'reason': reason, 'subparts': []},
                           {'problem': 2, 'score': 4, 'page': 1, 'reason': 'Стр. 1: пересечение {1,2} и {2,3} равно {2}; общий элемент указан с объяснением.', 'subparts': []}],
                'evidence': [{'source': 'work', 'page': 1, 'note': reason}, {'source': 'rubric', 'page': 1, 'note': 'Применена шкала 0 или 4 для каждой задачи.'}],
                'service_comment': 'СЦЕНАРНЫЙ РЕЖИМ БЕЗ LLM. Проверка механики инструментов и интерфейса.',
                'student_comment': 'Уточните различие конъюнкции и импликации в задаче 1.' if bad else 'Обе задачи решены верно и обоснованно.',
                'flags': [], 'defense_questions': ['Чему равна импликация при ложной посылке?'] if bad else []}})
        return {'status': 'completed', 'output': []}
