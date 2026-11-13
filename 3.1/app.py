#!/usr/bin/env python3
"""Local homework-review workspace. No third-party packages required."""

from __future__ import annotations

import csv
import io
import json
import math
import mimetypes
import os
import fcntl
import random
import re
import shutil
import sqlite3
import sys
import unicodedata
import uuid
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
from review_storage import ReviewStorageMixin, ConflictError, init_review_schema
from model_gateway import GatewayError

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DATA = Path(os.environ.get("REVIEWER_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()
DB_PATH = DATA / "reviewer.sqlite3"
HOST, PORT = "127.0.0.1", int(os.environ.get("REVIEWER_PORT", "8765"))

FLAG_LABELS = {
    "task_mismatch": "решается не выданная задача",
    "missing_justification": "не хватает обоснования",
    "unclear_solution": "решение неясное или слишком краткое",
    "out_of_scope_fact": "нетривиальный факт вне лекций",
    "sudden_progress": "резкий скачок уровня",
    "similarity": "сильное сходство с другой работой",
    "generated_text": "признаки шаблонного/сгенерированного текста",
    "low_readability": "нечитаемый или неполный файл",
}
HARD_FLAGS = {"task_mismatch", "similarity", "out_of_scope_fact", "sudden_progress", "generated_text"}
SUBMISSION_EXTENSIONS = {".pdf", ".tex", ".jpg", ".jpeg", ".png"}
FORMAT_PRIORITY = {".tex": 3, ".pdf": 2, ".jpg": 1, ".jpeg": 1, ".png": 1}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).lower().replace("ё", "е")
    return " ".join(re.sub(r"[^a-zа-я0-9]+", " ", value).split())


def json_load(raw: str | None, fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


class Store(ReviewStorageMixin):
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    @contextmanager
    def session(self):
        """Commit or roll back a request and always release the SQLite connection."""
        db = self.connect()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def init_schema(self):
        with self.session() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS students (
                  id TEXT PRIMARY KEY,
                  email TEXT NOT NULL UNIQUE,
                  full_name TEXT NOT NULL,
                  aliases_json TEXT NOT NULL DEFAULT '[]',
                  teacher_notes TEXT NOT NULL DEFAULT '',
                  learning_notes TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS homeworks (
                  id TEXT PRIMARY KEY,
                  number INTEGER NOT NULL,
                  title TEXT NOT NULL DEFAULT '',
                  due_at TEXT NOT NULL DEFAULT '',
                  classroom_url TEXT NOT NULL DEFAULT '',
                  topic TEXT NOT NULL DEFAULT '',
                  statement_path TEXT NOT NULL DEFAULT '',
                  solution_tex_path TEXT NOT NULL DEFAULT '',
                  solution_pdf_path TEXT NOT NULL DEFAULT '',
                  reference_solution_path TEXT NOT NULL DEFAULT '',
                  lecture_paths_json TEXT NOT NULL DEFAULT '[]',
                  import_path TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(number)
                );
                CREATE TABLE IF NOT EXISTS submissions (
                  id TEXT PRIMARY KEY,
                  homework_id TEXT NOT NULL REFERENCES homeworks(id) ON DELETE CASCADE,
                  student_id TEXT REFERENCES students(id) ON DELETE SET NULL,
                  original_name TEXT NOT NULL,
                  stored_path TEXT NOT NULL,
                  file_hash TEXT NOT NULL,
                  match_status TEXT NOT NULL,
                  match_note TEXT NOT NULL DEFAULT '',
                  owner_email TEXT NOT NULL DEFAULT '',
                  imported_at TEXT NOT NULL,
                  UNIQUE(homework_id, file_hash)
                );
                CREATE TABLE IF NOT EXISTS drive_metadata (
                  id TEXT PRIMARY KEY,
                  homework_id TEXT REFERENCES homeworks(id) ON DELETE CASCADE,
                  file_id TEXT NOT NULL,
                  path TEXT NOT NULL DEFAULT '',
                  file_name TEXT NOT NULL,
                  mime_type TEXT NOT NULL DEFAULT '',
                  modified_at TEXT NOT NULL DEFAULT '',
                  last_editor_name TEXT NOT NULL DEFAULT '',
                  last_editor_email TEXT NOT NULL DEFAULT '',
                  owner_name TEXT NOT NULL DEFAULT '',
                  owner_email TEXT NOT NULL DEFAULT '',
                  resolved_email TEXT NOT NULL DEFAULT '',
                  resolved_student_id TEXT REFERENCES students(id) ON DELETE SET NULL,
                  web_url TEXT NOT NULL DEFAULT '',
                  imported_at TEXT NOT NULL,
                  UNIQUE(file_id)
                );
                CREATE TABLE IF NOT EXISTS reviews (
                  id TEXT PRIMARY KEY,
                  homework_id TEXT NOT NULL REFERENCES homeworks(id) ON DELETE CASCADE,
                  student_id TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                  submission_id TEXT REFERENCES submissions(id) ON DELETE SET NULL,
                  status TEXT NOT NULL DEFAULT 'pending',
                  suggested_scores_json TEXT NOT NULL DEFAULT '[]',
                  suggested_total INTEGER,
                  service_comment TEXT NOT NULL DEFAULT '',
                  student_comment TEXT NOT NULL DEFAULT '',
                  evidence_json TEXT NOT NULL DEFAULT '[]',
                  flags_json TEXT NOT NULL DEFAULT '[]',
                  defense_questions_json TEXT NOT NULL DEFAULT '[]',
                  defense_recommendation TEXT NOT NULL DEFAULT 'not_selected',
                  defense_reason TEXT NOT NULL DEFAULT '',
                  approved_scores_json TEXT NOT NULL DEFAULT '[]',
                  approved_total INTEGER,
                  approved_service_comment TEXT NOT NULL DEFAULT '',
                  approved_student_comment TEXT NOT NULL DEFAULT '',
                  approved_at TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(homework_id, student_id)
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                  id TEXT PRIMARY KEY,
                  entity_type TEXT NOT NULL,
                  entity_id TEXT NOT NULL,
                  action TEXT NOT NULL,
                  details_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL
                );
                """
            )
            columns = {row["name"] for row in db.execute("PRAGMA table_info(homeworks)").fetchall()}
            review_columns = {row["name"] for row in db.execute("PRAGMA table_info(reviews)")}
            for name, definition in {
                "late_multiplier": "REAL NOT NULL DEFAULT 1",
                "late_reason": "TEXT NOT NULL DEFAULT ''",
                "approved_late_multiplier": "REAL NOT NULL DEFAULT 1",
                "approved_late_reason": "TEXT NOT NULL DEFAULT ''",
            }.items():
                if name not in review_columns:
                    db.execute(f"ALTER TABLE reviews ADD COLUMN {name} {definition}")
            if "reference_solution_path" not in columns:
                db.execute("ALTER TABLE homeworks ADD COLUMN reference_solution_path TEXT NOT NULL DEFAULT ''")
            submission_columns = {row["name"] for row in db.execute("PRAGMA table_info(submissions)").fetchall()}
            if "owner_email" not in submission_columns:
                db.execute("ALTER TABLE submissions ADD COLUMN owner_email TEXT NOT NULL DEFAULT ''")
            db.execute("""CREATE TABLE IF NOT EXISTS drive_accounts (
                permission_id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL REFERENCES students(id),
                evidence TEXT NOT NULL,
                confirmed_at TEXT NOT NULL
            )""")
            init_review_schema(db)

    def bind_drive_account(self, permission_id, email, evidence):
        """Explicit confirmation only; display names are never account identifiers."""
        permission_id, email, evidence = str(permission_id).strip(), str(email).strip().lower(), str(evidence).strip()
        if not permission_id or not permission_id.isascii() or not permission_id.isdigit() or not evidence:
            raise ValueError("Нужны цифровой permissionId и основание подтверждения.")
        if email in {"lyudasha.2006@gmail.com", "kiriell55@gmail.com"}:
            raise ValueError("Аккаунт сотрудника нельзя связать со студентом.")
        with self.session() as db:
            student = db.execute("SELECT id FROM students WHERE email=?", (email,)).fetchone()
            if not student:
                raise ValueError("Email не найден в реестре студентов.")
            existing = db.execute("SELECT * FROM drive_accounts WHERE permission_id=?", (permission_id,)).fetchone()
            if existing:
                if existing["student_id"] != student["id"]:
                    raise ValueError("Конфликт: аккаунт уже подтверждён для другого студента.")
                return dict(existing)
            db.execute("INSERT INTO drive_accounts VALUES (?,?,?,?)", (permission_id, student["id"], evidence, utcnow()))
            self.audit(db, "drive_account", permission_id, "confirmed", {"student_id": student["id"], "evidence": evidence})
            return dict(db.execute("SELECT * FROM drive_accounts WHERE permission_id=?", (permission_id,)).fetchone())

    def resolve_drive_identity(self, metadata, filename):
        """Resolve verified account IDs/emails; conflicting evidence always blocks."""
        candidates, evidence = {}, []
        staff = {"lyudasha.2006@gmail.com", "kiriell55@gmail.com"}
        for role in ("owner", "last_editor"):
            email = str(metadata.get(role + "_email") or "").strip().lower()
            pid = str(metadata.get(role + "_permission_id") or "").strip()
            if email in staff:
                continue
            with self.session() as db:
                account = db.execute("SELECT s.* FROM drive_accounts a JOIN students s ON s.id=a.student_id WHERE a.permission_id=?", (pid,)).fetchone() if pid else None
                person = db.execute("SELECT * FROM students WHERE email=?", (email,)).fetchone() if email else None
            if account and email and account["email"] != email:
                return None, "ambiguous", "Email и подтверждённый permissionId противоречат друг другу."
            for row in (account, person):
                if row:
                    candidates[row["id"]] = self.student_dict(row)
                    evidence.append(f"{role}: permissionId={pid}, email={email}")
            if email and not person:
                return None, "ambiguous", "Неизвестный email в метаданных: нужна ручная сверка."
        named, name_status, _ = self.find_student_for_filename(filename)
        if named:
            candidates[named["id"]] = named
        if len(candidates) > 1 or name_status == "ambiguous":
            return None, "ambiguous", "ФИО/аккаунты в метаданных указывают на разных студентов."
        if candidates:
            return next(iter(candidates.values())), "drive_matched", "Сквозная привязка; " + "; ".join(evidence) + "; file_id=" + str(metadata.get("file_id", ""))
        return None, "unmatched", "Нет подтверждённого аккаунта/email или ФИО. Имя редактора не используется как email."

    def audit(self, db, entity_type: str, entity_id: str, action: str, details=None):
        db.execute(
            "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), entity_type, entity_id, action, json.dumps(details or {}, ensure_ascii=False), utcnow()),
        )

    def student(self, student_id: str):
        with self.session() as db:
            row = db.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
        return self.student_dict(row) if row else None

    @staticmethod
    def student_dict(row):
        if not row:
            return None
        data = dict(row)
        data["aliases"] = json_load(data.pop("aliases_json"), [])
        return data

    def list_students(self):
        with self.session() as db:
            rows = db.execute("SELECT * FROM students ORDER BY full_name COLLATE NOCASE").fetchall()
        return [self.student_dict(row) for row in rows]

    def create_student(self, payload):
        email = payload.get("email", "").strip().lower()
        name = payload.get("full_name", "").strip()
        if not email or "@" not in email or not name:
            raise ValueError("Укажите корректный email и ФИО.")
        aliases = clean_lines(payload.get("aliases", []))
        now, student_id = utcnow(), str(uuid.uuid4())
        with self.session() as db:
            try:
                db.execute(
                    "INSERT INTO students VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (student_id, email, name, json.dumps(aliases, ensure_ascii=False), payload.get("teacher_notes", "").strip(), payload.get("learning_notes", "").strip(), now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Такой email уже есть в реестре.") from exc
            self.audit(db, "student", student_id, "created")
        # A student added after the first homework still needs a row in every existing session.
        for homework in self.list_homeworks():
            self.ensure_reviews(homework["id"])
        return self.student(student_id)

    def update_student(self, student_id, payload):
        if not self.student(student_id):
            raise LookupError("Студент не найден.")
        name = payload.get("full_name", "").strip()
        if not name:
            raise ValueError("ФИО не может быть пустым.")
        aliases = clean_lines(payload.get("aliases", []))
        with self.session() as db:
            db.execute(
                "UPDATE students SET full_name=?, aliases_json=?, teacher_notes=?, learning_notes=?, updated_at=? WHERE id=?",
                (name, json.dumps(aliases, ensure_ascii=False), payload.get("teacher_notes", "").strip(), payload.get("learning_notes", "").strip(), utcnow(), student_id),
            )
            self.audit(db, "student", student_id, "updated", {"fields": ["full_name", "aliases", "notes"]})
        return self.student(student_id)

    def student_history(self, student_id):
        student = self.student(student_id)
        if not student:
            raise LookupError("Студент не найден.")
        with self.session() as db:
            rows = db.execute(
                """SELECT h.number, h.title, r.status, r.suggested_total, r.approved_total,
                          r.defense_recommendation, r.defense_reason, r.flags_json,
                          r.late_multiplier, r.approved_late_multiplier
                   FROM reviews r JOIN homeworks h ON h.id=r.homework_id
                   WHERE r.student_id=? ORDER BY h.number""", (student_id,)
            ).fetchall()
        history = []
        for row in rows:
            data = dict(row)
            data["flags"] = json_load(data.pop("flags_json"), [])
            self.add_adjusted_totals(data)
            history.append(data)
        return {"student": student, "history": history}

    def homework_dict(self, row):
        data = dict(row)
        data["lecture_paths"] = json_load(data.pop("lecture_paths_json"), [])
        data["problems"] = json_load(data.pop("problems_json"), [])
        return data

    def list_homeworks(self):
        with self.session() as db:
            rows = db.execute("SELECT * FROM homeworks ORDER BY number").fetchall()
        return [self.homework_dict(row) for row in rows]

    def homework(self, homework_id):
        with self.session() as db:
            row = db.execute("SELECT * FROM homeworks WHERE id=?", (homework_id,)).fetchone()
        return self.homework_dict(row) if row else None

    def create_homework(self, payload):
        self.validate_problem_spec(payload.get('problems', []))
        try:
            number = int(payload.get("number"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Номер ДЗ должен быть целым числом.") from exc
        if number < 1:
            raise ValueError("Номер ДЗ должен быть положительным.")
        homework_id, now = str(uuid.uuid4()), utcnow()
        values = (
            homework_id, number, payload.get("title", "").strip(), payload.get("due_at", "").strip(), payload.get("classroom_url", "").strip(),
            payload.get("topic", "").strip(), payload.get("statement_path", "").strip(), payload.get("solution_tex_path", "").strip(),
            payload.get("solution_pdf_path", "").strip(), payload.get("reference_solution_path", "").strip(), json.dumps(clean_lines(payload.get("lecture_paths", [])), ensure_ascii=False), payload.get("import_path", "").strip(), now, now,
        )
        with self.session() as db:
            try:
                db.execute("""INSERT INTO homeworks
                    (id, number, title, due_at, classroom_url, topic, statement_path, solution_tex_path,
                     solution_pdf_path, reference_solution_path, lecture_paths_json, import_path, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", values)
            except sqlite3.IntegrityError as exc:
                raise ValueError("ДЗ с таким номером уже создано.") from exc
            self.audit(db, "homework", homework_id, "created")
        self.ensure_reviews(homework_id)
        self.update_homework_spec(homework_id, payload)
        return self.homework(homework_id)

    def update_homework(self, homework_id, payload):
        existing = self.homework(homework_id)
        if not existing:
            raise LookupError("ДЗ не найдено.")
        self.validate_problem_spec(payload.get('problems', existing['problems']))
        try:
            number = int(payload.get("number", existing["number"]))
        except (ValueError, TypeError) as exc:
            raise ValueError("Номер ДЗ должен быть целым числом.") from exc
        with self.session() as db:
            db.execute(
                """UPDATE homeworks SET number=?, title=?, due_at=?, classroom_url=?, topic=?, statement_path=?, solution_tex_path=?, solution_pdf_path=?, reference_solution_path=?, lecture_paths_json=?, import_path=?, updated_at=? WHERE id=?""",
                (number, payload.get("title", "").strip(), payload.get("due_at", "").strip(), payload.get("classroom_url", "").strip(), payload.get("topic", "").strip(), payload.get("statement_path", "").strip(), payload.get("solution_tex_path", "").strip(), payload.get("solution_pdf_path", "").strip(), payload.get("reference_solution_path", "").strip(), json.dumps(clean_lines(payload.get("lecture_paths", [])), ensure_ascii=False), payload.get("import_path", "").strip(), utcnow(), homework_id),
            )
            self.audit(db, "homework", homework_id, "updated")
        self.update_homework_spec(homework_id, payload)
        return self.homework(homework_id)

    def ensure_reviews(self, homework_id):
        now = utcnow()
        with self.session() as db:
            students = db.execute("SELECT id FROM students").fetchall()
            for student in students:
                db.execute(
                    "INSERT OR IGNORE INTO reviews (id, homework_id, student_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), homework_id, student["id"], now, now),
                )

    def find_student_for_filename(self, name):
        candidate = normalized(Path(name).stem)
        possible = []
        for student in self.list_students():
            names = [student["full_name"], *student["aliases"]]
            tokens = [normalized(item) for item in names if normalized(item)]
            matched = [item for item in tokens if item in candidate or candidate in item]
            if matched:
                possible.append((student, max(len(item.split()) for item in matched)))
        if not possible:
            return None, "unmatched", "ФИО в имени файла не найдено в реестре."
        top = max(score for _, score in possible)
        winners = [student for student, score in possible if score == top]
        if len(winners) != 1:
            return None, "ambiguous", "Имя файла похоже сразу на нескольких студентов."
        return winners[0], "matched", "Сопоставлено по ФИО/алиасу."

    def import_files(self, homework_id, source_path, drive_metadata=None):
        if drive_metadata is not None and (not isinstance(drive_metadata, dict) or any(not isinstance(value, dict) for value in drive_metadata.values())):
            raise ValueError("Метаданные должны быть словарём local_name → запись Drive.")
        homework = self.homework(homework_id)
        if not homework:
            raise LookupError("ДЗ не найдено.")
        source = Path(source_path).expanduser().resolve()
        if not source.is_dir():
            raise ValueError("Укажите существующую папку с PDF.")
        target = DATA / "submissions" / f"hw-{homework['number']:02d}"
        target.mkdir(parents=True, exist_ok=True)
        files = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in SUBMISSION_EXTENSIONS]
        report = {"imported": [], "alternates": [], "duplicates": [], "unmatched": [], "ambiguous": [], "ignored": []}
        import hashlib
        with self.session() as db:
            for source_file in sorted(files, key=lambda item: item.name.casefold()):
                raw_start = source_file.read_bytes()[:1024]
                suffix = source_file.suffix.lower()
                if suffix == ".pdf" and b"%PDF-" not in raw_start:
                    report["ignored"].append({"file": source_file.name, "reason": "Файл не похож на читаемый PDF (не найден заголовок %PDF-)."})
                    continue
                if suffix in {".jpg", ".jpeg"} and not raw_start.startswith(b"\xff\xd8\xff"):
                    report["ignored"].append({"file": source_file.name, "reason": "Файл не похож на JPEG."})
                    continue
                if suffix == ".png" and not raw_start.startswith(b"\x89PNG\r\n\x1a\n"):
                    report["ignored"].append({"file": source_file.name, "reason": "Файл не похож на PNG."})
                    continue
                file_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
                existing = db.execute("SELECT id FROM submissions WHERE homework_id=? AND file_hash=?", (homework_id, file_hash)).fetchone()
                if existing:
                    report["duplicates"].append({"file": source_file.name, "reason": "Точный дубликат уже импортирован."})
                    continue
                if drive_metadata is not None:
                    metadata = drive_metadata.get(source_file.name)
                    if metadata is None:
                        student, status, note = None, "unmatched", "Файл отсутствует в переданном манифесте Drive."
                    else:
                        if metadata.get("sha256") and metadata["sha256"] != file_hash:
                            raise ValueError("SHA256 не совпадает с манифестом Drive: " + source_file.name)
                        student, status, note = self.resolve_drive_identity(metadata, source_file.name)
                else:
                    student, status, note = self.find_student_for_filename(source_file.name)
                safe_name = re.sub(r"[^a-zA-Zа-яА-Я0-9._ -]", "_", source_file.name)
                stored_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
                stored = target / stored_name
                shutil.copy2(source_file, stored)
                submission_id = str(uuid.uuid4())
                db.execute(
                    "INSERT INTO submissions (id, homework_id, student_id, original_name, stored_path, file_hash, match_status, match_note, imported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (submission_id, homework_id, student["id"] if student else None, source_file.name, str(stored), file_hash, status, note, utcnow()),
                )
                if student:
                    review = db.execute("SELECT id FROM reviews WHERE homework_id=? AND student_id=?", (homework_id, student["id"])).fetchone()
                    if review:
                        previous = db.execute("SELECT submission_id, status FROM reviews WHERE id=?", (review["id"],)).fetchone()
                        old = previous["submission_id"]
                        if old:
                            old_file = db.execute("SELECT original_name, stored_path FROM submissions WHERE id=?", (old,)).fetchone()
                            old_suffix = Path(old_file["stored_path"]).suffix.lower()
                            if previous["status"] not in ("draft", "approved") and FORMAT_PRIORITY[suffix] > FORMAT_PRIORITY.get(old_suffix, 0):
                                db.execute("UPDATE submissions SET match_status='alternate_format', match_note='Альтернативная версия: выбрана более приоритетная версия работы.' WHERE id=?", (old,))
                                db.execute("UPDATE reviews SET submission_id=?, status='imported', revision=revision+1, updated_at=? WHERE id=?", (submission_id, utcnow(), review["id"]))
                                note = f"Выбрана версия {suffix}; она приоритетнее {old_suffix or 'предыдущего формата'}."
                                db.execute("UPDATE submissions SET match_status='matched_preferred', match_note=? WHERE id=?", (note, submission_id))
                                report["alternates"].append({"file": source_file.name, "student": student["full_name"], "note": note})
                            else:
                                db.execute("UPDATE submissions SET match_status='alternate_format', match_note='Альтернативная версия сохранена; основная версия имеет не меньший приоритет.' WHERE id=?", (submission_id,))
                                report["alternates"].append({"file": source_file.name, "student": student["full_name"], "note": "Сохранено как альтернативная версия; основная работа не заменена."})
                            continue
                        if previous["status"] not in ("draft", "approved"):
                            db.execute("UPDATE reviews SET submission_id=?, status='imported', revision=revision+1, updated_at=? WHERE id=?", (submission_id, utcnow(), review["id"]))
                self.audit(db, "submission", submission_id, "imported", {"from": str(source_file), "status": status})
                item = {"file": source_file.name, "student": student["full_name"] if student else None, "note": note}
                report["imported"].append(item)
                if status == "unmatched": report["unmatched"].append(item)
                if status == "ambiguous": report["ambiguous"].append(item)
        return report

    def list_submissions(self, homework_id, unresolved_only=False):
        with self.session() as db:
            query = """SELECT sub.*, s.full_name AS student_name FROM submissions sub
                       LEFT JOIN students s ON s.id=sub.student_id WHERE sub.homework_id=?"""
            params = [homework_id]
            if unresolved_only:
                query += " AND sub.student_id IS NULL"
            query += " ORDER BY sub.imported_at DESC"
            rows = db.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def assign_submission(self, submission_id, student_id):
        with self.session() as db:
            db.execute('BEGIN IMMEDIATE')
            submission = db.execute("SELECT * FROM submissions WHERE id=?", (submission_id,)).fetchone()
            student = db.execute("SELECT id FROM students WHERE id=?", (student_id,)).fetchone()
            if not submission or not student:
                raise LookupError("Файл или студент не найдены.")
            linked = db.execute('SELECT student_id FROM reviews WHERE submission_id=?', (submission_id,)).fetchall()
            if any(r['student_id'] != student_id for r in linked):
                raise ConflictError('Файл уже связан с проверкой другого студента. Автоматическое переназначение запрещено.')
            review = db.execute("SELECT * FROM reviews WHERE homework_id=? AND student_id=?", (submission["homework_id"], student_id)).fetchone()
            if not review:
                raise LookupError("Для этого студента нет строки проверки.")
            if review["status"] == "approved":
                raise ValueError("Нельзя подменить файл у уже утверждённой проверки.")
            if review["submission_id"] and review["submission_id"] != submission_id:
                raise ValueError("У этого студента уже есть другая работа в этой сессии.")
            db.execute("UPDATE submissions SET student_id=?, match_status='manual_matched', match_note='Сопоставлено преподавателем вручную.' WHERE id=?", (student_id, submission_id))
            db.execute("UPDATE reviews SET submission_id=?, status='imported', revision=revision+1, updated_at=? WHERE id=?", (submission_id, utcnow(), review["id"]))
            self.audit(db, "submission", submission_id, "manually_assigned", {"student_id": student_id})
        return {"submission_id": submission_id, "student_id": student_id}

    def set_submission_owner(self, submission_id, owner_email):
        """Store Drive owner metadata; link to a roster student when that email is known."""
        owner_email = str(owner_email or "").strip().lower()
        if not owner_email or "@" not in owner_email:
            raise ValueError("Укажите корректную почту владельца.")
        with self.session() as db:
            db.execute('BEGIN IMMEDIATE')
            submission = db.execute("SELECT * FROM submissions WHERE id=?", (submission_id,)).fetchone()
            if not submission:
                raise LookupError("Файл не найден.")
            student = db.execute("SELECT * FROM students WHERE email=?", (owner_email,)).fetchone()
            linked = db.execute('SELECT student_id FROM reviews WHERE submission_id=?', (submission_id,)).fetchall()
            if student and any(r['student_id'] != student['id'] for r in linked):
                raise ConflictError('Указанный владелец противоречит уже связанной проверке. Сначала сверка принадлежности.')
            note = "Почта владельца указана вручную по свойствам файла."
            db.execute("UPDATE submissions SET owner_email=?, match_note=? WHERE id=?", (owner_email, note, submission_id))
            if student:
                review = db.execute("SELECT * FROM reviews WHERE homework_id=? AND student_id=?", (submission["homework_id"], student["id"])).fetchone()
                if review and not review["submission_id"]:
                    db.execute("UPDATE submissions SET student_id=?, match_status='owner_matched' WHERE id=?", (student["id"], submission_id))
                    db.execute("UPDATE reviews SET submission_id=?, status='imported', revision=revision+1, updated_at=? WHERE id=?", (submission_id, utcnow(), review["id"]))
                else:
                    db.execute("UPDATE submissions SET student_id=?, match_status='owner_matched_alternate' WHERE id=?", (student["id"], submission_id))
            else:
                db.execute("UPDATE submissions SET match_status='owner_only' WHERE id=?", (submission_id,))
            self.audit(db, "submission", submission_id, "owner_set", {"owner_email": owner_email, "student_id": student["id"] if student else None})
        return self.list_submissions(submission["homework_id"], True)

    def review_dict(self, row):
        data = dict(row)
        for field in ("suggested_scores", "evidence", "flags", "defense_questions", "approved_scores"):
            data[field] = json_load(data.pop(field + "_json"), [])
        self.add_adjusted_totals(data)
        return data

    @staticmethod
    def add_adjusted_totals(data):
        # Preserve mathematical totals; disciplinary adjustments never change tasks.
        for prefix, coefficient in (("suggested", "late_multiplier"), ("approved", "approved_late_multiplier")):
            total = data.get(prefix + "_total")
            data[prefix + "_adjusted_total"] = None if total is None else float(Decimal(str(total)) * Decimal(str(data.get(coefficient, 1))))

    @staticmethod
    def late_adjustment(payload, review):
        try:
            multiplier = float(payload.get("late_multiplier", review["late_multiplier"]))
        except (ValueError, TypeError):
            raise ValueError("Коэффициент просрочки должен быть числом от 0 до 1.")
        reason = str(payload.get("late_reason", review["late_reason"])).strip()
        if not math.isfinite(multiplier) or not 0 <= multiplier <= 1:
            raise ValueError("Коэффициент просрочки должен быть числом от 0 до 1.")
        if multiplier != 1 and not reason:
            raise ValueError("Укажите основание коэффициента просрочки.")
        return multiplier, reason

    def list_reviews(self, homework_id):
        self.ensure_reviews(homework_id)
        with self.session() as db:
            rows = db.execute(
                """SELECT r.*, s.full_name, s.email, s.teacher_notes, s.learning_notes,
                          sub.original_name, sub.stored_path, sub.match_status, sub.match_note, sub.owner_email
                   FROM reviews r JOIN students s ON s.id=r.student_id
                   LEFT JOIN submissions sub ON sub.id=r.submission_id
                   WHERE r.homework_id=? ORDER BY s.full_name COLLATE NOCASE""", (homework_id,)
            ).fetchall()
        return [self.review_dict(row) for row in rows]

    def review(self, review_id):
        with self.session() as db:
            row = db.execute("SELECT * FROM reviews WHERE id=?", (review_id,)).fetchone()
        return self.review_dict(row) if row else None

    def submission_file(self, submission_id):
        """Return an imported work file only when it belongs to this local storage."""
        with self.session() as db:
            row = db.execute(
                "SELECT id, original_name, stored_path FROM submissions WHERE id=?", (submission_id,)
            ).fetchone()
        if not row:
            raise LookupError("Работа не найдена.")
        path = Path(row["stored_path"]).resolve()
        submissions_root = (DATA / "submissions").resolve()
        if submissions_root not in path.parents or not path.is_file():
            raise LookupError("Локальный файл работы недоступен.")
        return path, row["original_name"]

    def reject(self, review_id, payload):
        review = self.review(review_id)
        if not review:
            raise LookupError("Запись проверки не найдена.")
        if review["status"] == "approved":
            raise ValueError("Утверждённую запись нельзя отклонить этим действием.")
        note = str(payload.get("note", "")).strip()
        with self.session() as db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT revision,status FROM reviews WHERE id=?', (review_id,)).fetchone()
            if current['status'] == 'approved' or current['revision'] != review['revision']:
                raise ConflictError('Запись изменилась. Обновите её перед отклонением.')
            db.execute("UPDATE reviews SET status='rejected', service_comment=?, revision=revision+1, updated_at=? WHERE id=?", (note or review["service_comment"], utcnow(), review_id))
            self._version(db, review_id, 'rejected')
            self.audit(db, "review", review_id, "rejected", {"note": note})
        return self.review(review_id)

    def reopen(self, review_id, payload):
        """Human correction path. Draft creation itself still cannot alter approved rows."""
        review = self.review(review_id)
        if not review:
            raise LookupError("Запись проверки не найдена.")
        if review["status"] != "approved":
            raise ValueError("Открыть на исправление можно только утверждённую запись.")
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise ValueError("Укажите причину повторного открытия.")
        with self.session() as db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT revision,status FROM reviews WHERE id=?', (review_id,)).fetchone()
            if current['status'] != 'approved' or current['revision'] != review['revision']:
                raise ConflictError('Запись изменилась. Обновите её перед исправлением.')
            db.execute("UPDATE reviews SET status='draft', revision=revision+1, updated_at=? WHERE id=?", (utcnow(), review_id))
            self._version(db, review_id, 'reopened')
            self.audit(db, "review", review_id, "reopened_for_correction", {"reason": reason})
        return self.review(review_id)

    def set_defense(self, review_id, payload):
        review = self.review(review_id)
        if not review:
            raise LookupError("Запись проверки не найдена.")
        selected = bool(payload.get("selected"))
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise ValueError("Для ручного решения о защите укажите причину.")
        recommendation = "selected" if selected else "not_selected"
        with self.session() as db:
            db.execute("UPDATE reviews SET defense_recommendation=?, defense_reason=?, revision=revision+1, updated_at=? WHERE id=?", (recommendation, reason, utcnow(), review_id))
            self.audit(db, "review", review_id, "defense_manually_set", {"recommendation": recommendation, "reason": reason})
        return self.review(review_id)

    def history_check(self, review_id, payload):
        review = self.review(review_id)
        if not review: raise LookupError("Запись проверки не найдена.")
        scores = review["suggested_scores"] or review["approved_scores"]
        if not scores: raise ValueError("Сначала сохраните черновик с баллами.")
        current = 100 * sum(item["score"] for item in scores) / self.score_capacity(self.homework(review['homework_id'])['problems'], scores)
        with self.session() as db:
            rows = db.execute(
                """SELECT approved_scores_json, problems_json FROM reviews r JOIN homeworks h ON h.id=r.homework_id
                   WHERE r.student_id=? AND h.number < (SELECT number FROM homeworks WHERE id=?) AND r.status='approved'
                   ORDER BY h.number DESC LIMIT 3""", (review["student_id"], review["homework_id"]),
            ).fetchall()
        previous = []
        for row in rows:
            prior = json_load(row["approved_scores_json"], [])
            if prior: previous.append(100 * sum(item["score"] for item in prior) / self.score_capacity(json_load(row['problems_json'], []), prior))
        if len(previous) < 2:
            return {"candidate": False, "reason": "Недостаточно двух предыдущих утверждённых работ."}
        baseline = median(previous)
        jump = current - baseline
        result = {"candidate": jump >= 25, "current_percent": round(current, 1), "baseline_percent": round(baseline, 1), "jump_points": round(jump, 1)}
        if result["candidate"]:
            result["reason"] = "Рост ≥25 п.п.; прежде чем добавлять сигнал, агент должен подтвердить изменение стиля решения."
            if payload.get("style_change_observed"):
                flag = {"kind": "sudden_progress", "severity": 2, "evidence": f"Результат {current:.1f}% против медианы {baseline:.1f}% по {len(previous)} предыдущим ДЗ; наблюдается изменение стиля решения.", "status": "needs_manual_review"}
                flags = review["flags"] + ([] if any(item["kind"] == "sudden_progress" for item in review["flags"]) else [flag])
                with self.session() as db:
                    db.execute("UPDATE reviews SET flags_json=?, revision=revision+1, updated_at=? WHERE id=?", (json.dumps(flags, ensure_ascii=False), utcnow(), review_id))
                    self.audit(db, "review", review_id, "history_flag_added", result)
                result["flag_added"] = True
        else:
            result["reason"] = "Существенного скачка нет."
        return result

    def defense_recommendations(self, homework_id):
        homework = self.homework(homework_id)
        if not homework: raise LookupError("ДЗ не найдено.")
        reviews = self.list_reviews(homework_id)
        if homework["number"] <= 4:
            selected = {row["id"]: "Все студенты: обязательная защита ДЗ 1–4." for row in reviews}
            quota = len(reviews)
        else:
            fraction = 2 / 5 if homework["number"] <= 8 else 1 / 4 if homework["number"] <= 12 else 1 / 5
            quota = math.ceil(len(reviews) * fraction)
            priority = []
            remainder = []
            for row in reviews:
                flags = row["flags"]
                score = sum(int(flag["severity"]) for flag in flags)
                hard = any(flag["kind"] in HARD_FLAGS for flag in flags)
                (priority if hard or score else remainder).append((row, hard, score))
            priority.sort(key=lambda item: (not item[1], -item[2], item[0]["full_name"].casefold()))
            selected = {}
            for row, hard, score in priority:
                cause = "Существенный сигнал" if hard else "Сигналы требуют уточнения"
                selected[row["id"]] = f"{cause}: {', '.join(FLAG_LABELS[f['kind']] for f in row['flags'])}."
            if len(selected) < quota:
                pool = [item[0] for item in remainder]
                seed = f"{homework['id']}:{len(reviews)}"
                random.Random(seed).shuffle(pool)
                for row in pool[: quota - len(selected)]:
                    selected[row["id"]] = "Добор до целевой доли защиты: воспроизводимая случайная выборка."
        with self.session() as db:
            for row in reviews:
                recommendation = "selected" if row["id"] in selected else "not_selected"
                reason = selected.get(row["id"], "Не вошёл в текущую рекомендацию.")
                db.execute("UPDATE reviews SET defense_recommendation=?, defense_reason=?, revision=revision+1, updated_at=? WHERE id=?", (recommendation, reason, utcnow(), row["id"]))
                self.audit(db, "review", row["id"], "defense_recommended", {"recommendation": recommendation})
        return {"homework_number": homework["number"], "quota": quota, "selected": len(selected), "items": self.list_reviews(homework_id)}

    def context(self, homework_id):
        homework = self.homework(homework_id)
        if not homework: raise LookupError("ДЗ не найдено.")
        return {"homework": homework, "reviews": self.list_reviews(homework_id), "policy": {"scale": "0–4 for each problem", "progress_jump": "+25 p.p. vs median of at least 2 approved prior works", "defense": "1–4 all; 5–8 2/5; 9–12 1/4; 13–16 1/5"}}

    def dashboard(self):
        homeworks = self.list_homeworks()
        with self.session() as db:
            students = db.execute("SELECT COUNT(*) AS n FROM students").fetchone()["n"]
            counts = db.execute("SELECT status, COUNT(*) AS n FROM reviews GROUP BY status").fetchall()
        return {"students": students, "homeworks": homeworks, "review_counts": {row["status"]: row["n"] for row in counts}}

    def export_rows(self, homework_id):
        homework = self.homework(homework_id)
        if not homework: raise LookupError("ДЗ не найдено.")
        rows = []
        for review in self.list_reviews(homework_id):
            if review["status"] == "approved":
                rows.append({"ДЗ": homework["number"], "ФИО": review["full_name"], "Email": review["email"], "Итог без штрафа": review["approved_total"], "Коэффициент просрочки": review["approved_late_multiplier"], "Основание просрочки": review["approved_late_reason"], "Итог (для ручного переноса)": review["approved_adjusted_total"], "Комментарий студенту": review["approved_student_comment"], "Служебный комментарий": review["approved_service_comment"], "Защита": "да" if review["defense_recommendation"] == "selected" else "нет"})
        return rows


def clean_lines(value):
    if isinstance(value, str): value = value.splitlines()
    if not isinstance(value, list): return []
    return [str(item).strip() for item in value if str(item).strip()]


def median(items):
    values = sorted(items)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


STORE = Store()


def xlsx_bytes(rows):
    headers = list(rows[0].keys()) if rows else ["ДЗ", "ФИО", "Email", "Итог (для ручного переноса)", "Комментарий студенту", "Служебный комментарий", "Защита"]
    def cell(value):
        text = str(value if value is not None else "")
        return "<c t=\"inlineStr\"><is><t>" + xml_escape(text) + "</t></is></c>"
    sheet_rows = []
    for index, values in enumerate([headers] + [[row.get(header, "") for header in headers] for row in rows], 1):
        sheet_rows.append(f'<row r="{index}">' + "".join(cell(item) for item in values) + "</row>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        archive.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr("xl/workbook.xml", '<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Classroom export" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", '<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + "".join(sheet_rows) + "</sheetData></worksheet>")
    return buf.getvalue()


def xml_escape(value):
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class App(BaseHTTPRequestHandler):
    server_version = "HomeworkReviewer/1.0"

    def log_message(self, format, *args):
        # OAuth callbacks contain one-time codes; never log query strings.
        print(f"[{self.log_date_time_string()}] {self.command} {urlparse(self.path).path}")

    def local_request(self):
        host = urlparse('http://' + self.headers.get('Host', '')).hostname
        origin = self.headers.get('Origin')
        if host not in ('127.0.0.1', 'localhost', '::1'):
            self.error(403, 'Разрешён только локальный доступ.')
            return False
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            self.error(403, 'Источник запроса не совпадает с приложением.')
            return False
        return True

    def body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length < 0 or length > 2 * 1024 * 1024:
            raise ValueError('Тело запроса превышает 2 МБ.')
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError('Тело запроса должно быть объектом JSON.')
            return body
        except json.JSONDecodeError: raise ValueError("Тело запроса должно быть JSON.")

    def respond(self, status, payload, content_type="application/json; charset=utf-8"):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(raw)

    def error(self, status, message): self.respond(status, {"error": message})

    def do_GET(self):
        try:
            if not self.local_request(): return
            parsed = urlparse(self.path); path = parsed.path
            if path == '/favicon.ico': return self.respond(204, b'', 'image/x-icon')
            from agent_http import handle
            if handle(self, STORE, path, query=parse_qs(parsed.query)): return
            if path == "/": return self.serve_file(STATIC / "index.html", "text/html; charset=utf-8")
            if path.startswith("/static/"):
                requested = (STATIC / path.removeprefix("/static/")).resolve()
                if STATIC.resolve() not in requested.parents: return self.error(404, "Не найдено.")
                return self.serve_file(requested, mimetypes.guess_type(str(requested))[0] or "application/octet-stream")
            if path == "/api/dashboard": return self.respond(200, STORE.dashboard())
            if path == "/api/students": return self.respond(200, STORE.list_students())
            match = re.fullmatch(r"/api/students/([^/]+)/history", path)
            if match: return self.respond(200, STORE.student_history(match.group(1)))
            if path.startswith("/api/students/"):
                student = STORE.student(path.rsplit("/", 1)[1]); return self.respond(200, student) if student else self.error(404, "Студент не найден.")
            if path == "/api/homeworks": return self.respond(200, STORE.list_homeworks())
            match = re.fullmatch(r"/api/homeworks/([^/]+)/(records|context)", path)
            if match:
                homework_id, resource = match.groups()
                return self.respond(200, STORE.list_reviews(homework_id) if resource == "records" else STORE.context(homework_id))
            match = re.fullmatch(r"/api/homeworks/([^/]+)/submissions", path)
            if match: return self.respond(200, STORE.list_submissions(match.group(1), True))
            match = re.fullmatch(r"/api/submissions/([^/]+)/file", path)
            if match:
                file_path, file_name = STORE.submission_file(match.group(1))
                return self.serve_file(file_path, mimetypes.guess_type(file_name)[0] or "application/octet-stream", file_name)
            match = re.fullmatch(r"/api/export/([^/]+)", path)
            if match:
                rows = STORE.export_rows(match.group(1)); fmt = parse_qs(parsed.query).get("format", ["csv"])[0]
                if fmt == "xlsx":
                    return self.respond(200, xlsx_bytes(rows), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()) if rows else ["ДЗ", "ФИО", "Email", "Итог (для ручного переноса)", "Комментарий студенту", "Служебный комментарий", "Защита"]); writer.writeheader(); writer.writerows(rows)
                return self.respond(200, ("\ufeff" + out.getvalue()).encode("utf-8"), "text/csv; charset=utf-8")
            self.error(404, "Маршрут не найден.")
        except ConflictError as exc: self.error(409, str(exc))
        except GatewayError as exc: self.respond(400, {'error': str(exc), **exc.public()})
        except (ValueError, LookupError) as exc: self.error(400, str(exc))
        except Exception as exc:
            print(repr(exc), file=sys.stderr); self.error(500, "Внутренняя ошибка. Подробности — в терминале.")

    def do_POST(self):
        try:
            if not self.local_request(): return
            path, payload = urlparse(self.path).path, self.body()
            from agent_http import handle
            if handle(self, STORE, path, payload=payload): return
            if path == "/api/students": return self.respond(201, STORE.create_student(payload))
            match = re.fullmatch(r"/api/students/([^/]+)", path)
            if match: return self.respond(200, STORE.update_student(match.group(1), payload))
            if path == "/api/homeworks": return self.respond(201, STORE.create_homework(payload))
            match = re.fullmatch(r"/api/homeworks/([^/]+)", path)
            if match: return self.respond(200, STORE.update_homework(match.group(1), payload))
            match = re.fullmatch(r"/api/homeworks/([^/]+)/import", path)
            if match: return self.respond(200, STORE.import_files(match.group(1), payload.get("source_path", ""), payload.get("drive_metadata")))
            match = re.fullmatch(r"/api/homeworks/([^/]+)/defense-recommendations", path)
            if match: return self.respond(200, STORE.defense_recommendations(match.group(1)))
            match = re.fullmatch(r"/api/records/([^/]+)/draft", path)
            if match: return self.respond(200, STORE.save_draft(match.group(1), payload))
            match = re.fullmatch(r"/api/records/([^/]+)/approve", path)
            if match: return self.respond(200, STORE.approve(match.group(1), payload))
            match = re.fullmatch(r"/api/records/([^/]+)/reject", path)
            if match: return self.respond(200, STORE.reject(match.group(1), payload))
            match = re.fullmatch(r"/api/records/([^/]+)/reopen", path)
            if match: return self.respond(200, STORE.reopen(match.group(1), payload))
            match = re.fullmatch(r"/api/records/([^/]+)/defense", path)
            if match: return self.respond(200, STORE.set_defense(match.group(1), payload))
            match = re.fullmatch(r"/api/records/([^/]+)/history-check", path)
            if match: return self.respond(200, STORE.history_check(match.group(1), payload))
            match = re.fullmatch(r"/api/submissions/([^/]+)/assign", path)
            if match: return self.respond(200, STORE.assign_submission(match.group(1), payload.get("student_id", "")))
            match = re.fullmatch(r"/api/submissions/([^/]+)/owner", path)
            if match: return self.respond(200, {"submissions": STORE.set_submission_owner(match.group(1), payload.get("owner_email", ""))})
            self.error(404, "Маршрут не найден.")
        except ConflictError as exc: self.error(409, str(exc))
        except GatewayError as exc: self.respond(400, {'error': str(exc), **exc.public()})
        except (ValueError, LookupError) as exc: self.error(400, str(exc))
        except Exception as exc:
            print(repr(exc), file=sys.stderr); self.error(500, "Внутренняя ошибка. Подробности — в терминале.")

    def serve_file(self, path, content_type, file_name=None):
        if not path.is_file(): return self.error(404, "Файл не найден.")
        raw = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(raw)))
        if file_name:
            ascii_name = re.sub(r"[^A-Za-z0-9._ -]", "_", file_name).replace('"', "'")
            self.send_header("Content-Disposition", f"inline; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(file_name)}")
        self.end_headers(); self.wfile.write(raw)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / '.server.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Эта база уже открыта другим сервером.')
        server = ThreadingHTTPServer((HOST, PORT), App)
        from agent_http import runtime
        manager = runtime(STORE)
        print(f"Проверка ДЗ: http://{HOST}:{PORT}", flush=True)
        print(f"Локальная база: {DB_PATH}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            manager.close()
            server.server_close()


if __name__ == "__main__": main()
