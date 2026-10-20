#!/usr/bin/env python3
"""Read-only checksum verification of the bounded DP process-timing scope."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
SEAL = HERE / 'evidence-seal.json'

seal = json.loads(SEAL.read_text())
assert seal['schema'] == 'dp-factor-process-timing21-evidence-v1'
observed = set()
for directory in seal['directories']:
    for path in (ROOT / directory).rglob('*'):
        if path.is_file() and path != SEAL and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.o', '.mod'):
            observed.add(path.relative_to(ROOT).as_posix())
assert observed == set(seal['files']), ('inventory mismatch', sorted(observed ^ set(seal['files'])))
for name, expected in seal['files'].items():
    path = ROOT / name
    assert path.stat().st_size == expected['bytes'], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], name
assert sum(value['bytes'] for value in seal['files'].values()) == seal['bytes']
print(json.dumps(dict(complete=True, files=len(observed), bytes=seal['bytes'], engine_calls=0), indent=2))
