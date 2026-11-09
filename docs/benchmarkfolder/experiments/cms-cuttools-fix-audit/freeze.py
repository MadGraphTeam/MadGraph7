#!/usr/bin/env python3
"""Freeze or verify this read-only CMS source/provenance audit."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INDEX = HERE / 'artifact-index.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error('Run from the MadGraph repository root')
    files = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(HERE.rglob('*')) if p.is_file() and p != INDEX
        and '__pycache__' not in p.parts and p.suffix != '.pyc']
    if args.write:
        record = {'schema': 1,
            'scope': 'Read-only CMS CutTools production-fix outline and exact reconstruction of the originally audited dirty working source.',
            'production_changed': False, 'automatic_repair_qualified': False,
            'files': files}
        INDEX.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    else:
        assert json.loads(INDEX.read_text())['files'] == files, 'CMS audit inventory differs'
    print(json.dumps({'indexed_files': len(files), 'bytes': sum(r['bytes'] for r in files),
        'action': 'written' if args.write else 'verified'}))


if __name__ == '__main__':
    main()
