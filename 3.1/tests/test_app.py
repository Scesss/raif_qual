import json
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import app


def draft(scores, comment="Хорошая работа.", flags=None):
    return {
        "scores": [{"problem": index + 1, "score": score, "reason": "Проверено."} for index, score in enumerate(scores)],
        "service_comment": "Служебная заметка.",
        "student_comment": comment,
        "evidence": [{"source": "work", "page": 1, "note": "Решение."}],
        "flags": flags or [],
        "defense_questions": ["Сформулируйте определение."],
    }


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = app.Store(Path(self.temp.name) / "review.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def homework(self, number):
        return self.store.create_homework({"number": number, "title": f"ДЗ {number}"})

    def student(self, name="Иванов Иван Иванович", email="ivanov@example.test"):
        return self.store.create_student({"full_name": name, "email": email, "aliases": ["Иванов И. И."]})

    def test_email_is_unique_and_rows_are_created_for_existing_homework(self):
        hw = self.homework(1)
        person = self.student()
        self.assertEqual(len(self.store.list_reviews(hw["id"])), 1)
        with self.assertRaises(ValueError):
            self.student("Другой Иван", person["email"])

    def test_draft_scores_are_validated_and_approved_draft_is_immutable(self):
        person, hw = self.student(), self.homework(5)
        review = self.store.list_reviews(hw["id"])[0]
        saved = self.store.save_draft(review["id"], draft([4, 3]))
        self.assertEqual(saved["suggested_total"], 7)
        self.assertEqual(self.store.approve(review["id"], {} )["approved_total"], 7)
        with self.assertRaises(ValueError):
            self.store.save_draft(review["id"], draft([0]))
        other = self.store.list_reviews(hw["id"])[0]
        with self.assertRaises(ValueError):
            self.store.save_draft(other["id"], draft([5]))

    def test_signal_requires_evidence(self):
        person, hw = self.student(), self.homework(5)
        review = self.store.list_reviews(hw["id"])[0]
        with self.assertRaises(ValueError):
            self.store.save_draft(review["id"], draft([4], flags=[{"kind": "similarity", "severity": 3, "evidence": ""}]))

    def test_lateness_preserves_raw_scores_and_exports_adjusted_total(self):
        person, hw = self.student(), self.homework(2)
        rid = self.store.list_reviews(hw['id'])[0]['id']
        payload = draft([4, 4, 4, 4, 4])
        payload.update(late_multiplier=0.7, late_reason='Просрочка; указание преподавателя')
        saved = self.store.save_draft(rid, payload)
        self.assertEqual(saved['suggested_total'], 20)
        self.assertEqual(saved['suggested_adjusted_total'], 14)
        # An older agent payload must not silently remove the penalty.
        self.assertEqual(self.store.save_draft(rid, draft([4]*5))['late_multiplier'], 0.7)
        approved = self.store.approve(rid, {})
        self.assertEqual(approved['approved_total'], 20)
        self.assertEqual(approved['approved_adjusted_total'], 14)
        exported = self.store.export_rows(hw['id'])[0]
        self.assertEqual(exported['Итог без штрафа'], 20)
        self.assertEqual(exported['Итог (для ручного переноса)'], 14)
        history = self.store.student_history(person['id'])['history'][0]
        self.assertEqual(history['approved_adjusted_total'], 14)
        with self.assertRaises(ValueError): self.store.approve(rid, {'late_multiplier': 1})
        self.store.reopen(rid, {'reason': 'Коррекция'})
        payload.update(late_multiplier=0.5)
        saved = self.store.save_draft(rid, payload)
        self.assertEqual(saved['suggested_adjusted_total'], 10)
        self.assertEqual(saved['approved_adjusted_total'], 14)

    def test_invalid_late_adjustments(self):
        self.student(); hw = self.homework(2)
        rid = self.store.list_reviews(hw['id'])[0]['id']
        for multiplier in (-1, 1.1, 'nan', 'inf', None):
            with self.assertRaises(ValueError):
                self.store.save_draft(rid, dict(draft([4]), late_multiplier=multiplier, late_reason='test'))
        with self.assertRaises(ValueError):
            self.store.save_draft(rid, dict(draft([4]), late_multiplier=0.7))
        result = self.store.save_draft(rid, dict(draft([3]), late_multiplier=0.7, late_reason='test'))
        self.assertEqual(result['suggested_adjusted_total'], 2.1)

    def test_sudden_progress_needs_two_prior_approved_works_and_style_confirmation(self):
        person = self.student()
        first, second, third = self.homework(1), self.homework(2), self.homework(3)
        for hw in (first, second):
            review = self.store.list_reviews(hw["id"])[0]
            self.store.save_draft(review["id"], draft([1, 1]))
            self.store.approve(review["id"], {})
        review = self.store.list_reviews(third["id"])[0]
        self.store.save_draft(review["id"], draft([4, 4]))
        candidate = self.store.history_check(review["id"], {})
        self.assertTrue(candidate["candidate"])
        self.assertFalse(candidate.get("flag_added", False))
        self.assertTrue(self.store.history_check(review["id"], {"style_change_observed": True})["flag_added"])
        self.assertTrue(any(flag["kind"] == "sudden_progress" for flag in self.store.review(review["id"])["flags"]))

    def test_defense_policy_and_hard_signals_can_exceed_quota(self):
        hw = self.homework(5)
        for number in range(5):
            student = self.student(f"Студент {number} Тестовый", f"s{number}@example.test")
        reviews = self.store.list_reviews(hw["id"])
        for index, review in enumerate(reviews):
            flag = [] if index > 1 else [{"kind": "similarity", "severity": 3, "evidence": "Совпадает структура с другой работой."}]
            self.store.save_draft(review["id"], draft([4], flags=flag))
        result = self.store.defense_recommendations(hw["id"])
        self.assertEqual(result["quota"], 2)
        self.assertGreaterEqual(result["selected"], 2)
        self.assertTrue(all(r["defense_recommendation"] == "selected" for r in self.store.list_reviews(hw["id"])[:2]))

    def test_filename_matching_uses_aliases(self):
        self.student()
        student, status, _ = self.store.find_student_for_filename("ДЗ1 Иванов И. И..pdf")
        self.assertEqual(status, "matched")
        self.assertEqual(student["email"], "ivanov@example.test")

    def test_import_copies_pdf_and_manual_assignment_resolves_queue(self):
        student, hw = self.student(), self.homework(5)
        source = Path(self.temp.name) / "source"
        source.mkdir()
        (source / "Неизвестный человек.pdf").write_bytes(b"%PDF-1.7 pretend pdf")
        (source / "битый.pdf").write_bytes(b"pretend pdf")
        old_data = app.DATA
        try:
            app.DATA = Path(self.temp.name) / "app-data"
            report = self.store.import_files(hw["id"], str(source))
            self.assertEqual(len(report["unmatched"]), 1)
            self.assertEqual(len(report["ignored"]), 1)
            queued = self.store.list_submissions(hw["id"], unresolved_only=True)
            self.assertEqual(len(queued), 1)
            self.store.assign_submission(queued[0]["id"], student["id"])
            review = self.store.list_reviews(hw["id"])[0]
            self.assertEqual(review["status"], "imported")
            self.assertEqual(review["match_status"], "manual_matched")
        finally:
            app.DATA = old_data

    def test_tex_version_is_preferred_over_pdf_for_same_student(self):
        self.student()
        hw = self.homework(7)
        source = Path(self.temp.name) / "source-tex"
        source.mkdir()
        (source / "Иванов И. И..pdf").write_bytes(b"%PDF-1.7 placeholder")
        (source / "Иванов И. И..tex").write_text("\\documentclass{article}\\begin{document}Solution\\end{document}", encoding="utf-8")
        old_data = app.DATA
        try:
            app.DATA = Path(self.temp.name) / "app-data"
            report = self.store.import_files(hw["id"], str(source))
            review = self.store.list_reviews(hw["id"])[0]
            self.assertTrue(review["stored_path"].endswith(".tex"))
            self.assertEqual(len(report["alternates"]), 1)
        finally:
            app.DATA = old_data

    def test_approved_result_requires_audited_reopen_before_editing(self):
        self.student()
        hw = self.homework(6)
        review = self.store.list_reviews(hw["id"])[0]
        self.store.save_draft(review["id"], draft([3]))
        self.store.approve(review["id"], {})
        with self.assertRaises(ValueError):
            self.store.save_draft(review["id"], draft([4]))
        self.store.reopen(review["id"], {"reason": "Исправляю арифметическую ошибку."})
        self.assertEqual(self.store.save_draft(review["id"], draft([4]))["suggested_total"], 4)

    def test_local_http_api_serves_workspace_and_export(self):
        old_store = app.STORE
        app.STORE = self.store
        server = ThreadingHTTPServer(("127.0.0.1", 0), app.App)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        def request(path, payload=None):
            if payload is None:
                return urlopen(base + path, timeout=2)
            return urlopen(Request(base + path, json.dumps(payload).encode(), {"Content-Type": "application/json"}), timeout=2)
        try:
            with request("/") as response:
                self.assertIn(b"\xd0\x9f\xd1\x80\xd0\xbe\xd0\xb2\xd0\xb5\xd1\x80\xd0\xba\xd0\xb0 \xd0\x94\xd0\x97", response.read())
            with request("/api/students", {"email": "web@example.test", "full_name": "Веб Тестов"}) as response:
                self.assertEqual(response.status, 201)
            with request("/api/homeworks", {"number": 8}) as response:
                homework = json.load(response)
            with request(f"/api/homeworks/{homework['id']}/records") as response:
                review = json.load(response)[0]
            with request(f"/api/records/{review['id']}/draft", draft([4])):
                pass
            with request(f"/api/records/{review['id']}/approve", {}):
                pass
            with request(f"/api/export/{homework['id']}?format=csv") as response:
                self.assertIn("Веб Тестов", response.read().decode("utf-8-sig"))
            with request(f"/api/export/{homework['id']}?format=xlsx") as response:
                self.assertTrue(response.read().startswith(b"PK"))
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
            app.STORE = old_store


if __name__ == "__main__":
    unittest.main()
