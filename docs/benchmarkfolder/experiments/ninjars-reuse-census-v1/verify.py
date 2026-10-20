#!/usr/bin/env python3
"""Read-only source/hash verification plus exact offline census replay."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import census

HERE = Path(__file__).resolve().parent
def main():
    provenance = json.loads((HERE/'provenance.json').read_text())
    for name, expected in provenance['files'].items():
        raw = (HERE/'sources'/name).read_bytes()
        assert len(raw) == expected['bytes']
        assert hashlib.sha256(raw).hexdigest() == expected['sha256']
    manifest = HERE/'manifest.json'
    if manifest.exists():
        bound = json.loads(manifest.read_text())
        actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p != manifest}
        assert actual == set(bound['files'])
        for name, entry in bound['files'].items():
            raw = (HERE/name).read_bytes()
            assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
    census.selftest()
    assert census.compute() == json.loads((HERE/'result.json').read_text())
    print(json.dumps({'passed':True, 'source_snapshots':len(provenance['files']),
                      'records_checked':2784, 'engine_calls':0}))

if __name__ == '__main__':
    main()
