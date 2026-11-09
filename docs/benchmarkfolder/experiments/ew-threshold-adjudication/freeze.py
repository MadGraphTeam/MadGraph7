#!/usr/bin/env python3
"""Freeze or verify EW adjudication evidence, including ignored raw logs."""
import argparse
import hashlib
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[3]
INDEX = REPORT / 'artifact-index.json'


def inventory():
    paths = [p for p in REPORT.rglob('*') if p.is_file() and p != INDEX
             and '__pycache__' not in p.parts and p.suffix != '.pyc']
    return [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error('Run from the MadGraph repository root')
    files = inventory()
    if args.write:
        if not (REPORT / 'execution.json').exists():
            raise SystemExit('Native trace is still pending; cannot freeze final evidence')
        record = {'schema': 1, 'scope': 'EW threshold source/geometry and separately labeled causal diagnostics',
                  'original_extreme_point_qualified': False,
                  'production_changed': False, 'files': files}
        INDEX.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    elif json.loads(INDEX.read_text())['files'] != files:
        raise SystemExit('EW evidence index differs from current files')
    print(json.dumps({'indexed_files': len(files), 'bytes': sum(p['bytes'] for p in files),
        'logs': sum(p['path'].endswith('.log') for p in files),
        'action': 'written' if args.write else 'verified'}))


if __name__ == '__main__':
    main()
