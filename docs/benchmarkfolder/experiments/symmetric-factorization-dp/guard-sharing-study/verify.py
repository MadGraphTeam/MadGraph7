#!/usr/bin/env python3
"""Read-only source-census checksum verification; no engine calls."""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
args = argparse.ArgumentParser()
args.add_argument('--sources', action='store_true')
options = args.parse_args()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
seal = json.loads((HERE / 'seal.json').read_text())
for name, expected in seal['files'].items():
    assert sha(HERE / name) == expected, name
if options.sources:
    data = json.loads((HERE / 'analysis.json').read_text())
    for case in data['cases'].values():
        for name, expected in case['source_files'].items():
            assert sha(ROOT / name) == expected, name
    obj = json.loads((HERE / 'object-provenance.json').read_text())
    assert sha(ROOT / obj['path']) == obj['sha256']
print(json.dumps({'verified': True, 'files': len(seal['files']),
                  'source_and_object_recheck': options.sources}))
