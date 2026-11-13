#!/usr/bin/env python3
"""Prepare previews, then assemble an explicitly ordered and rotated submission."""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageOps
from pypdf import PdfReader, PdfWriter


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_document(path):
    if path.suffix.lower() == '.pdf':
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError(f'Encrypted PDF: {path}')
        return reader
    if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        raise ValueError(f'Unsupported file: {path}')
    with Image.open(path) as original:
        # Correct phone-camera EXIF orientation before making PDF pages.
        picture = ImageOps.exif_transpose(original).convert('RGBA')
        canvas = Image.new('RGB', picture.size, 'white')
        canvas.paste(picture, mask=picture.getchannel('A'))
        stream = io.BytesIO()
        canvas.save(stream, format='PDF', resolution=150, quality=100)
    stream.seek(0)
    return PdfReader(stream)


def render(pdf, directory):
    directory.mkdir()
    subprocess.run(['pdftoppm', '-r', '100', '-png', str(pdf),
                    str(directory / 'page')], check=True, timeout=120)


def prepare(paths, directory):
    paths = [Path(p).resolve(strict=True) for p in paths]
    hashes = [digest(p) for p in paths]
    if len(hashes) != len(set(hashes)):
        raise ValueError('Duplicate input content; specify each source once.')
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError('Use a new output directory; existing results are preserved.')
    writer, entries, sources = PdfWriter(), [], []
    for path, sha in zip(paths, hashes):
        reader = read_document(path)
        sources.append({'path': str(path), 'sha256': sha, 'page_count': len(reader.pages)})
        for number, page in enumerate(reader.pages, 1):
            writer.add_page(page)
            entries.append({'source': str(path), 'page': number,
                            'rotate_clockwise': 0, 'orientation_checked': False})
    if not entries:
        raise ValueError('No pages to assemble.')
    directory.mkdir(parents=True)
    writer.write(directory / 'preview.pdf')
    manifest = {'version': 1, 'sources': sources, 'pages': entries}
    (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    render(directory / 'preview.pdf', directory / 'preview-pages')
    return directory / 'manifest.json'


def build(manifest_path, directory):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError('Use a new output directory; existing results are preserved.')
    if manifest.get('version') != 1:
        raise ValueError('Unsupported manifest version.')
    readers, expected = {}, set()
    for source in manifest['sources']:
        path = Path(source['path']).resolve(strict=True)
        if digest(path) != source['sha256']:
            raise ValueError(f'Source changed since preview: {path}')
        key = str(path)
        if key in readers:
            raise ValueError('Duplicate source in manifest.')
        readers[key] = read_document(path)
        if len(readers[key].pages) != source['page_count']:
            raise ValueError('Source page count changed.')
        expected.update((key, n) for n in range(1, source['page_count'] + 1))
    writer, seen, page_map = PdfWriter(), set(), []
    for output_page, entry in enumerate(manifest['pages'], 1):
        key = (entry['source'], entry['page'])
        rotation = entry['rotate_clockwise']
        if type(entry['page']) is not int or key not in expected or key in seen:
            raise ValueError('Missing, duplicate or invalid source page.')
        if entry.get('orientation_checked') is not True:
            raise ValueError(f'Inspect orientation of output page {output_page} first.')
        if type(rotation) is not int or rotation not in (0, 90, 180, 270):
            raise ValueError('Rotation must be 0, 90, 180 or 270 clockwise.')
        seen.add(key)
        page = writer.add_page(readers[key[0]].pages[key[1] - 1])
        page.rotate(rotation)  # Additional rotation relative to preview, not absolute.
        page_map.append({'output_page': output_page, **entry})
    if seen != expected or not seen:
        raise ValueError('Every source page must occur exactly once.')
    directory.mkdir(parents=True)
    output = directory / 'submission.pdf'
    writer.write(output)
    if len(PdfReader(output).pages) != len(seen):
        raise ValueError('Output page count mismatch.')
    render(output, directory / 'pages')
    report = {'output_sha256': digest(output), 'sources': manifest['sources'],
              'pages': page_map, 'final_visual_check_required': True}
    (directory / 'page-map.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_cmd = sub.add_parser('prepare')
    prepare_cmd.add_argument('--directory', required=True)
    prepare_cmd.add_argument('inputs', nargs='+', help='Explicit file order; no filename guessing.')
    build_cmd = sub.add_parser('build')
    build_cmd.add_argument('--directory', required=True)
    build_cmd.add_argument('manifest')
    args = parser.parse_args()
    try:
        result = (prepare(args.inputs, args.directory) if args.command == 'prepare'
                  else build(args.manifest, args.directory))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'{exc}\n')
    print(result)


if __name__ == '__main__':
    main()
