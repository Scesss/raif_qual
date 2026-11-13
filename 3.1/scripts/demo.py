#!/usr/bin/env python3
"""Create a separate synthetic course and run the local product demo."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_pdf(path, title, paragraphs, *, obscure=False):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    font_paths = [os.environ.get('REVIEWER_DEMO_FONT', ''), '/Library/Fonts/Arial Unicode.ttf',
                  '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/arialuni.ttf']
    font = None
    for candidate in font_paths:
        if candidate and Path(candidate).is_file():
            face = TTFont('DemoCandidate', candidate)
            if all(ord(c) in face.face.charToGlyph for c in 'А→¬∨∧∩'):
                font = Path(candidate)
                break
    if not font:
        raise SystemExit('Нужен шрифт с кириллицей и математикой: задайте REVIEWER_DEMO_FONT (Arial Unicode/DejaVuSans TTF).')
    pdfmetrics.registerFont(TTFont('DemoSans', str(font)))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    c.setTitle(title)
    width, height = A4
    c.setFillColor(colors.HexColor('#133c34'))
    c.rect(0, height - 105, width, 105, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont('DemoSans', 19)
    c.drawString(45, height - 48, title)
    c.setFont('DemoSans', 10)
    c.drawString(45, height - 75, 'Дискретная математика · учебный демонстрационный курс')
    c.setFillColor(colors.HexColor('#192227'))
    text = c.beginText(45, height - 145)
    text.setFont('DemoSans', 12)
    text.setLeading(24)
    for paragraph in paragraphs:
        text.textLine(paragraph)
    c.drawText(text)
    if obscure:
        # The hidden operator is genuinely absent from the image and text layer.
        c.setFillColor(colors.HexColor('#deded7'))
        c.rect(78, height - 196, 18, 17, fill=1, stroke=0)
    c.setFont('DemoSans', 9)
    c.setFillColor(colors.HexColor('#617178'))
    c.drawString(45, 36, 'Учебный пример. Персональных данных студентов нет.                         Страница 1')
    c.save()


def prepare(data_directory):
    import app
    assets = ROOT / 'demo'
    make_pdf(assets / 'statement.pdf', 'ДЗ 1. Логика и множества', [
        'Задача 1. Докажите равносильность P → Q и ¬P ∨ Q.',
        'Можно использовать определение импликации или таблицу истинности.', '',
        'Задача 2. Найдите {1, 2} ∩ {2, 3}. Обоснуйте ответ.', '',
        'Проверяйте каждую задачу независимо. Максимум: 8 баллов.'])
    make_pdf(assets / 'rubric.pdf', 'Критерии проверки', [
        'Задача 1: 4 балла за корректное доказательство равносильности.',
        'Если вместо импликации использована конъюнкция и доказано',
        'неверное утверждение — 0 баллов за задачу 1.', '',
        'Задача 2: 4 балла за ответ {2} с указанием общего элемента.',
        'Неверный ответ — 0 баллов. Промежуточных баллов в этом примере нет.', '',
        'Если ключевой символ не читается, запросите ручное чтение.',
        'Не подменяйте нечитаемость оценкой 0.'])
    cases = [
        ('Учебная Анна', 'anna@example.test', 'correct', ['Письменная работа', '',
            '1. По определению P → Q ложно только при P=1, Q=0.',
            '¬P ∨ Q ложно ровно при тех же значениях, значит формулы равносильны.', '',
            '2. {1, 2} ∩ {2, 3} = {2}: число 2 принадлежит обоим множествам,',
            'а 1 и 3 — только одному из них.']),
        ('Учебный Борис', 'boris@example.test', 'error', ['Письменная работа', '',
            '1. Переписываю импликацию как конъюнкцию: P ∧ Q = ¬P ∨ Q.',
            'Эти формулы равносильны при всех P и Q.', '',
            '2. {1, 2} ∩ {2, 3} = {2}, так как общий элемент только 2.']),
        ('Учебная Вера', 'vera@example.test', 'unclear', ['Фотокопия работы', '',
            '1. P     Q = ¬P ∨ Q.',
            'Символ между P и Q утрачен при сканировании.', '',
            '2. {1, 2} ∩ {2, 3} = {2}, общий элемент только 2.']),
    ]
    for name, email, case, paragraphs in cases:
        make_pdf(assets / 'works' / (name + '.pdf'), name + ' · решение', paragraphs, obscure=case == 'unclear')
    marker = data_directory / 'demo-course.json'
    if marker.exists():
        return json.loads(marker.read_text())
    if app.STORE.list_students() or app.STORE.list_homeworks():
        raise SystemExit('Демонстрационная папка уже содержит другой курс. Выберите новую --data-dir.')
    for name, email, _, _ in cases:
        app.STORE.create_student({'full_name': name, 'email': email})
    hw = app.STORE.create_homework({'number': 1, 'title': 'Логика и множества · демонстрация',
        'topic': 'Импликация, таблицы истинности, пересечение множеств',
        'statement_path': str(assets / 'statement.pdf'), 'rubric_path': str(assets / 'rubric.pdf'),
        'problems': [{'problem': 1, 'title': 'Равносильность', 'subparts': []},
                     {'problem': 2, 'title': 'Пересечение', 'subparts': []}]})
    app.STORE.import_files(hw['id'], str(assets / 'works'))
    result = {'homework_id': hw['id'], 'data_dir': str(data_directory)}
    marker.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'demo')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    directory = args.data_dir.expanduser().resolve()
    if directory == (ROOT / 'data').resolve():
        raise SystemExit('Демобазу нельзя создавать в основной папке data. Выберите отдельную папку.')
    os.environ['REVIEWER_DATA_DIR'] = str(directory)
    os.environ['REVIEWER_PORT'] = str(args.port)
    os.environ['REVIEWER_DEMO'] = '1'
    data = prepare(directory)
    print(json.dumps(data, ensure_ascii=False), flush=True)
    if not args.prepare_only:
        import app
        app.main()


if __name__ == '__main__':
    main()
