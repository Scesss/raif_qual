import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from scripts.assemble_submission import prepare, build, read_document, digest
    from pypdf import PdfWriter, PdfReader
    from PIL import Image, ImageDraw
    PDF_TOOLS = bool(shutil.which('pdftoppm'))
except ImportError:
    PDF_TOOLS = False


@unittest.skipUnless(PDF_TOOLS, 'Install requirements-pdf.txt and Poppler to test assembly.')
class AssemblyTests(unittest.TestCase):
    def test_rotation_provenance_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / 'part.pdf'
            writer = PdfWriter()
            writer.add_blank_page(200, 300).rotate(90)
            writer.add_blank_page(400, 500)
            writer.write(pdf)
            photo = root / 'phone.jpg'
            im = Image.new('RGB', (120, 200), 'white')
            ImageDraw.Draw(im).text((10, 10), 'TOP', fill='black')
            exif = Image.Exif()
            exif[274] = 6
            im.save(photo, exif=exif)
            # EXIF 6 must swap width/height before conversion to PDF.
            page = read_document(photo).pages[0]
            self.assertGreater(float(page.mediabox.width), float(page.mediabox.height))
            before = digest(pdf)
            manifest = prepare([pdf, photo], root / 'preview')
            with self.assertRaisesRegex(ValueError, 'Inspect orientation'):
                build(manifest, root / 'unreviewed')
            data = json.loads(manifest.read_text())
            for entry in data['pages']:
                entry['orientation_checked'] = True
            data['pages'][0]['rotate_clockwise'] = 270
            manifest.write_text(json.dumps(data))
            output = build(manifest, root / 'result')
            result = PdfReader(output)
            self.assertEqual(len(result.pages), 3)
            self.assertEqual(result.pages[0].rotation % 360, 0)
            self.assertEqual(digest(pdf), before)
            mapping = json.loads((output.parent / 'page-map.json').read_text())
            self.assertEqual(mapping['pages'][2]['source'], str(photo.resolve()))
            self.assertEqual(mapping['output_sha256'], digest(output))
            self.assertEqual(len(list((output.parent / 'pages').glob('*.png'))), 3)
            with self.assertRaisesRegex(ValueError, 'new output directory'):
                build(manifest, output.parent)
            data['pages'].pop()
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Every source page'):
                build(manifest, root / 'missing')
            pdf.write_bytes(pdf.read_bytes() + b'\nchanged')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                build(manifest, root / 'changed')

    def test_duplicate_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / 'one.pdf'
            writer = PdfWriter()
            writer.add_blank_page(200, 300)
            writer.write(pdf)
            with self.assertRaisesRegex(ValueError, 'Duplicate input'):
                prepare([pdf, pdf], root / 'preview')
