#!/usr/bin/env python3
"""Confirmed, cross-homework Drive identities; local JSON manifests, no cloud writes."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    bind = commands.add_parser('bind')
    bind.add_argument('--permission-id', required=True)
    bind.add_argument('--email', required=True)
    bind.add_argument('--evidence', required=True)
    imp = commands.add_parser('import')
    imp.add_argument('--homework', type=int, required=True)
    imp.add_argument('--folder', required=True)
    imp.add_argument('--manifest', required=True)
    args = parser.parse_args()
    store = Store()
    if args.command == 'bind':
        result = store.bind_drive_account(args.permission_id, args.email, args.evidence)
    else:
        rows = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
        mapping = {}
        for row in rows:
            name = row['local_name']
            if name in mapping or Path(name).name != name or not row.get('file_id') or not row.get('sha256'):
                raise ValueError('Each manifest entry needs unique local_name, file_id and sha256.')
            mapping[name] = row
        homework = next((h for h in store.list_homeworks() if h['number'] == args.homework), None)
        if not homework:
            raise ValueError('Create the homework in the app first.')
        result = store.import_files(homework['id'], args.folder, mapping)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
