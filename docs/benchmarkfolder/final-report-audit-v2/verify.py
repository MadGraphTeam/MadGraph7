#!/usr/bin/env python3
"""Check this immutable audit and optional current source bindings only."""
import argparse,hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
root=next(p for p in here.parents if (p/'madgraph/__init__.py').is_file())
p=argparse.ArgumentParser();p.add_argument('--sources',action='store_true');args=p.parse_args()
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((here/'manifest.json').read_text())
for name,item in manifest['files'].items():
    path=here/name
    assert path.stat().st_size==item['bytes'] and h(path)==item['sha256'],name
record=json.loads((here/'validation.json').read_text())
assert record['passed'] and record['tests_run']==76 and record['all_children_exited']
if args.sources:
    for name,expected in record['source_sha256'].items():
        assert h(root/name)==expected,name
print('Independent 76-test software audit hashes verified; no tests or engines executed.')
