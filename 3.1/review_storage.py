"""Validation, immutable review versions and optimistic concurrency for the reviewer."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


class ConflictError(ValueError):
    pass


def integer(value, label):
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return int(value)
    raise ValueError(f"{label}: требуется целое число.")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def init_review_schema(db):
    for table, additions in {
        'homeworks': {'problems_json': "TEXT NOT NULL DEFAULT '[]'", 'rubric_path': "TEXT NOT NULL DEFAULT ''"},
        'reviews': {'revision': 'INTEGER NOT NULL DEFAULT 0', 'provenance_json': "TEXT NOT NULL DEFAULT '{}'"},
    }.items():
        columns = {r['name'] for r in db.execute(f'PRAGMA table_info({table})')}
        for name, definition in additions.items():
            if name not in columns:
                db.execute(f'ALTER TABLE {table} ADD COLUMN {name} {definition}')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS review_versions (
            id TEXT PRIMARY KEY, review_id TEXT NOT NULL REFERENCES reviews(id),
            revision INTEGER NOT NULL, action TEXT NOT NULL, payload_json TEXT NOT NULL,
            provenance_json TEXT NOT NULL, created_at TEXT NOT NULL,
            UNIQUE(review_id, revision));
        CREATE TABLE IF NOT EXISTS agent_runs (
            id TEXT PRIMARY KEY, review_id TEXT NOT NULL REFERENCES reviews(id),
            status TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
            expected_revision INTEGER NOT NULL, snapshot_json TEXT NOT NULL,
            error_json TEXT NOT NULL DEFAULT '{}', cancel_requested INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_run_per_review
            ON agent_runs(review_id) WHERE status IN ('queued','running');
        CREATE TABLE IF NOT EXISTS agent_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES agent_runs(id),
            kind TEXT NOT NULL, data_json TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS events_by_run ON agent_events(run_id, seq);
    ''')


class ReviewStorageMixin:
    @staticmethod
    def validate_problem_spec(problems):
        if not isinstance(problems, list) or len(problems) > 100:
            raise ValueError('Перечень задач должен быть списком до 100 задач.')
        cleaned, seen = [], set()
        for item in problems:
            if not isinstance(item, dict):
                raise ValueError('Неверное описание задачи.')
            number = integer(item.get('problem'), 'Номер задачи')
            parts = item.get('subparts', [])
            if number < 1 or number in seen or not isinstance(parts, list) or any(not isinstance(x, str) or not x.strip() for x in parts) or len(parts) != len(set(parts)):
                raise ValueError('Нужны уникальные номера задач и непустые уникальные подпункты.')
            seen.add(number)
            entry = {'problem': number, 'title': str(item.get('title', '')).strip(), 'subparts': parts}
            if 'max_score' in item:
                maximum = integer(item['max_score'], 'Максимальный балл')
                if not 1 <= maximum <= 100:
                    raise ValueError('Максимальный балл должен быть целым числом от 1 до 100.')
                entry['max_score'] = maximum
            if 'subpart_max_scores' in item:
                allocations = item['subpart_max_scores']
                if not isinstance(allocations, dict) or not parts or set(allocations) != set(parts):
                    raise ValueError('Задайте максимумы для всех подпунктов задачи.')
                allocations = {key: integer(value, 'Максимум подпункта') for key, value in allocations.items()}
                if any(not 1 <= value <= 100 for value in allocations.values()) or sum(allocations.values()) != entry.get('max_score', 4):
                    raise ValueError('Максимумы подпунктов должны быть положительными и в сумме давать максимум задачи.')
                entry['subpart_max_scores'] = allocations
            cleaned.append(entry)
        return cleaned

    @staticmethod
    def score_capacity(problems, scores):
        spec = {p['problem']: p for p in problems}
        return sum(spec.get(s['problem'], {}).get('max_score', 4) for s in scores)

    def update_homework_spec(self, homework_id, payload):
        if 'problems' not in payload and 'rubric_path' not in payload:
            return
        existing = self.homework(homework_id)
        cleaned = self.validate_problem_spec(payload.get('problems', existing['problems']))
        with self.session() as db:
            db.execute('UPDATE homeworks SET problems_json=?, rubric_path=? WHERE id=?',
                       (json.dumps(cleaned, ensure_ascii=False), str(payload.get('rubric_path', existing['rubric_path'])).strip(), homework_id))

    def source_manifest(self, review_id):
        review = self.review(review_id)
        if not review:
            raise LookupError('Запись проверки не найдена.')
        hw = self.homework(review['homework_id'])
        sources = {}
        paths = {'statement': hw['statement_path'], 'rubric': hw['rubric_path'],
                 'solution_tex': hw['solution_tex_path'], 'solution_pdf': hw['solution_pdf_path'],
                 'reference': hw['reference_solution_path']}
        paths.update({f'lecture_{i + 1}': p for i, p in enumerate(hw['lecture_paths'])})
        if review['submission_id']:
            work, _ = self.submission_file(review['submission_id'])
            paths['work'] = str(work)
        for key, value in paths.items():
            if value:
                path = Path(value).expanduser().resolve()
                if not path.is_file():
                    raise ValueError(f'Материал {key} недоступен: проверьте путь в карточке ДЗ.')
                if path.stat().st_size > 50 * 1024 * 1024:
                    raise ValueError(f'Материал {key} превышает 50 МБ.')
                sources[key] = {'path': str(path), 'sha256': digest(path)}
        with self.session() as db:
            submission = db.execute('SELECT student_id FROM submissions WHERE id=?', (review['submission_id'],)).fetchone() if review['submission_id'] else None
        if submission and submission['student_id'] != review['student_id']:
            raise ConflictError('Принадлежность работы не совпадает со строкой студента. Нужна ручная сверка.')
        return {'submission_id': review['submission_id'], 'student_id': review['student_id'],
                'homework_id': review['homework_id'], 'problems': hw['problems'], 'sources': sources}

    def validate_draft(self, payload, homework_id=None):
        flag_kinds = {'task_mismatch', 'missing_justification', 'unclear_solution', 'out_of_scope_fact',
                      'sudden_progress', 'similarity', 'generated_text', 'low_readability'}
        if not isinstance(payload, dict):
            raise ValueError('Черновик должен быть объектом.')
        scores = payload.get('scores', [])
        if not isinstance(scores, list) or not scores:
            raise ValueError('Нужен хотя бы один балл по задаче.')
        cleaned, numbers = [], set()
        problems = self.homework(homework_id)['problems'] if homework_id else []
        spec = {p['problem']: p for p in problems}
        evidence = payload.get('evidence', [])
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('Нужны основания со ссылками на страницы.')
        clean_evidence = []
        for e in evidence:
            if not isinstance(e, dict) or any(not isinstance(e.get(k), str) or not e[k].strip() for k in ('source', 'note')):
                raise ValueError('У основания должны быть источник, страница и пояснение.')
            page = integer(e.get('page'), 'Страница основания')
            if page < 1:
                raise ValueError('Страницы нумеруются с 1.')
            clean_evidence.append({**e, 'page': page})
        for item in scores:
            if not isinstance(item, dict):
                raise ValueError('Неверный формат балла.')
            problem = integer(item.get('problem'), 'Номер задачи')
            score = integer(item.get('score'), 'Балл')
            reason = item.get('reason', '')
            if not isinstance(reason, str):
                raise ValueError('Обоснование должно быть непустым текстом.')
            reason = reason.strip()
            maximum = spec.get(problem, {}).get('max_score', 4)
            if problem < 1 or problem in numbers or not 0 <= score <= maximum or not reason:
                raise ValueError(f'Нужны уникальные номера, целые баллы 0–{maximum} и обоснование каждой оценки.')
            entry = {'problem': problem, 'score': score, 'reason': reason}
            if problems:
                page = integer(item.get('page'), 'Страница решения')
                if page < 1 or not any(e['source'] == 'work' and e['page'] == page for e in clean_evidence):
                    raise ValueError('Для каждой оценки нужна ссылка на страницу work в основаниях.')
                parts = item.get('subparts', [])
                if not isinstance(parts, list) or any(not isinstance(p, dict) or not isinstance(p.get('id'), str) or not p['id'].strip() or not isinstance(p.get('reason'), str) or not p['reason'].strip() for p in parts):
                    raise ValueError('Нужны обоснования по подпунктам.')
                ids = [p['id'] for p in parts]
                if len(ids) != len(set(ids)) or set(ids) != set(spec.get(problem, {}).get('subparts', [])):
                    raise ValueError(f'Неполное покрытие подпунктов задачи {problem}.')
                allocations = spec.get(problem, {}).get('subpart_max_scores')
                if allocations:
                    parts = [{**p, 'score': integer(p.get('score'), 'Балл подпункта')} for p in parts]
                    if any(not 0 <= p['score'] <= allocations[p['id']] for p in parts):
                        raise ValueError(f'Балл подпункта задачи {problem} превышает его максимум.')
                    if sum(p['score'] for p in parts) != score:
                        raise ValueError(f'Сумма баллов подпунктов задачи {problem} должна совпадать с баллом задачи.')
                entry.update(page=page, subparts=parts)
            numbers.add(problem)
            cleaned.append(entry)
        if problems and numbers != set(spec):
            raise ValueError('Черновик должен покрывать все задачи из карточки ДЗ.')
        flags = payload.get('flags', [])
        if not isinstance(flags, list):
            raise ValueError('Сигналы передаются списком.')
        clean_flags = []
        for flag in flags:
            if not isinstance(flag, dict):
                raise ValueError('Неверный сигнал.')
            severity = integer(flag.get('severity', 1), 'Серьёзность')
            if flag.get('kind') not in flag_kinds or severity not in (1, 2, 3) or not isinstance(flag.get('evidence'), str) or not flag['evidence'].strip():
                raise ValueError('У сигнала должны быть тип, серьёзность 1–3 и основание.')
            clean_flags.append({**flag, 'severity': severity, 'status': 'needs_manual_review'})
        questions = payload.get('defense_questions', [])
        if not isinstance(questions, list) or any(not isinstance(q, str) for q in questions):
            raise ValueError('Вопросы должны быть списком строк.')
        return {'scores': sorted(cleaned, key=lambda i: i['problem']), 'total': sum(i['score'] for i in cleaned),
                'service_comment': str(payload.get('service_comment', '')).strip(),
                'student_comment': str(payload.get('student_comment', '')).strip(),
                'evidence': clean_evidence, 'flags': clean_flags,
                'defense_questions': [q.strip() for q in questions if q.strip()]}

    def _version(self, db, review_id, action):
        row = dict(db.execute('SELECT * FROM reviews WHERE id=?', (review_id,)).fetchone())
        db.execute('INSERT INTO review_versions VALUES (?,?,?,?,?,?,?)',
                   (str(uuid.uuid4()), review_id, row['revision'], action,
                    json.dumps(row, ensure_ascii=False), row['provenance_json'], now()))

    def save_draft(self, review_id, payload, *, expected_revision=None, provenance=None, run_id=None):
        review = self.review(review_id)
        if not review:
            raise LookupError('Запись проверки не найдена.')
        data = self.validate_draft(payload, review['homework_id'])
        expected = payload.get('expected_revision', review['revision']) if expected_revision is None else expected_revision
        expected = integer(expected, 'Версия')
        manifest = provenance if provenance is not None else self.source_manifest(review_id)
        if provenance is not None and self.source_manifest(review_id) != provenance:
            raise ConflictError('Исходники изменились во время проверки. Запустите её заново.')
        multiplier, reason = self.late_adjustment(payload, review)
        with self.session() as db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT * FROM reviews WHERE id=?', (review_id,)).fetchone()
            if current['status'] == 'approved' or current['revision'] != expected or current['submission_id'] != manifest['submission_id']:
                raise ConflictError('Запись уже изменена или утверждена. Обновите её перед сохранением.')
            self.validate_draft(payload, current['homework_id'])
            if self.source_manifest(review_id) != manifest:
                raise ConflictError('Материалы изменились перед сохранением. Запустите проверку заново.')
            if run_id:
                run = db.execute('SELECT status,cancel_requested FROM agent_runs WHERE id=?', (run_id,)).fetchone()
                if not run or run['cancel_requested'] or run['status'] != 'running':
                    raise ConflictError('Запуск остановлен; черновик не сохранён.')
            if db.execute('SELECT 1 FROM review_versions WHERE review_id=? LIMIT 1', (review_id,)).fetchone() is None:
                self._version(db, review_id, 'before_first_edit')
            db.execute('''UPDATE reviews SET status='draft', suggested_scores_json=?, suggested_total=?,
                service_comment=?, student_comment=?, evidence_json=?, flags_json=?, defense_questions_json=?,
                late_multiplier=?, late_reason=?, provenance_json=?, revision=revision+1, updated_at=? WHERE id=?''',
                (json.dumps(data['scores'], ensure_ascii=False), data['total'], data['service_comment'], data['student_comment'],
                 json.dumps(data['evidence'], ensure_ascii=False), json.dumps(data['flags'], ensure_ascii=False),
                 json.dumps(data['defense_questions'], ensure_ascii=False), multiplier, reason,
                 json.dumps(manifest, ensure_ascii=False), now(), review_id))
            self._version(db, review_id, 'agent_draft' if run_id else 'manual_draft')
            self.audit(db, 'review', review_id, 'draft_saved', {'total': data['total'], 'run_id': run_id})
            if run_id:
                db.execute("UPDATE agent_runs SET status='completed', updated_at=? WHERE id=?", (now(), run_id))
        return self.review(review_id)

    def approve(self, review_id, payload):
        review = self.review(review_id)
        if not review:
            raise LookupError('Запись проверки не найдена.')
        source = {**review, 'scores': review['suggested_scores']}
        source.update({k: payload[k] for k in ('scores', 'service_comment', 'student_comment') if k in payload})
        data = self.validate_draft(source, review['homework_id'])
        if not data['student_comment']:
            raise ValueError('Перед утверждением добавьте комментарий для студента.')
        manifest = self.source_manifest(review_id)
        stored = json.loads(review['provenance_json'])
        if stored and manifest != stored:
            raise ConflictError('Материалы изменились после проверки. Подготовьте новый черновик.')
        multiplier, reason = self.late_adjustment(payload, review)
        expected = integer(payload.get('expected_revision', review['revision']), 'Версия')
        with self.session() as db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT status,revision,submission_id FROM reviews WHERE id=?', (review_id,)).fetchone()
            if current['status'] == 'approved' or current['revision'] != expected or current['submission_id'] != manifest['submission_id']:
                raise ConflictError('Запись уже изменена или утверждена. Обновите её.')
            if db.execute('SELECT 1 FROM review_versions WHERE review_id=? LIMIT 1', (review_id,)).fetchone() is None:
                self._version(db, review_id, 'before_first_edit')
            db.execute('''UPDATE reviews SET status='approved', approved_scores_json=?, approved_total=?,
                approved_service_comment=?, approved_student_comment=?, approved_at=?, updated_at=?,
                approved_late_multiplier=?, approved_late_reason=?, revision=revision+1 WHERE id=?''',
                (json.dumps(data['scores'], ensure_ascii=False), data['total'], data['service_comment'], data['student_comment'],
                 now(), now(), multiplier, reason, review_id))
            self._version(db, review_id, 'approved')
            self.audit(db, 'review', review_id, 'approved', {'total': data['total']})
        return self.review(review_id)

    def review_versions(self, review_id):
        with self.session() as db:
            rows = db.execute('SELECT * FROM review_versions WHERE review_id=? ORDER BY revision DESC', (review_id,)).fetchall()
        return [{**dict(r), 'payload': json.loads(r['payload_json']), 'provenance': json.loads(r['provenance_json'])} for r in rows]
