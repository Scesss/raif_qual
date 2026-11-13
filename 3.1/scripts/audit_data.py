#!/usr/bin/env python3
"""Read-only consistency audit. Prints JSON; exit 1 means findings, not a repair."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def audit(path, expected_problems=None):
    findings = []
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        for row in db.execute('PRAGMA foreign_key_check'):
            findings.append({'kind': 'foreign_key', 'detail': list(row)})
        for row in db.execute('SELECT * FROM submissions'):
            source = Path(row['stored_path'])
            if not source.is_file():
                findings.append({'kind': 'missing_file', 'submission_id': row['id']})
            else:
                with source.open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != row['file_hash']:
                    findings.append({'kind': 'changed_file', 'submission_id': row['id']})
        rows = db.execute('''SELECT r.*, sub.student_id AS file_student,
                             sub.homework_id AS file_homework FROM reviews r
                             LEFT JOIN submissions sub ON sub.id=r.submission_id''').fetchall()
        for row in rows:
            def finding(kind, **detail):
                findings.append({'kind': kind, 'review_id': row['id'], **detail})
            if row['submission_id'] and (row['student_id'] != row['file_student'] or row['homework_id'] != row['file_homework']):
                finding('identity_mismatch')
            for prefix in ('suggested', 'approved'):
                try:
                    scores = json.loads(row[prefix + '_scores_json'])
                    if not isinstance(scores, list):
                        raise ValueError('Scores must be a list.')
                    if any(type(x.get('score')) is not int or x['score'] not in range(5) or
                           type(x.get('problem')) is not int or x['problem'] < 1 for x in scores):
                        raise ValueError('Invalid scores.')
                    numbers = [x['problem'] for x in scores]
                    if len(set(numbers)) != len(numbers):
                        raise ValueError('Duplicate problem numbers.')
                except (ValueError, TypeError, KeyError, AttributeError):
                    finding('invalid_scores', field=prefix)
                    continue
                total = row[prefix + '_total']
                if total is not None and total != sum(x['score'] for x in scores):
                    finding('total_mismatch', field=prefix, stored=total, calculated=sum(x['score'] for x in scores))
                if expected_problems and (scores or total is not None) and set(numbers) != set(range(1, expected_problems + 1)):
                    finding('problem_coverage', field=prefix, actual=numbers, expected=expected_problems)
                if any(not x.get('reason', '').strip() for x in scores):
                    finding('missing_reason', field=prefix)
        return {'reviews': len(rows), 'findings': findings}
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database')
    parser.add_argument('--expected-problems', type=int, help='Only use when all homeworks have this problem count.')
    args = parser.parse_args()
    result = audit(args.database, args.expected_problems)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result['findings']))
