#!/usr/bin/env python3
"""Freeze this already-prepared text package once; never launches a process."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    assert not (HERE/'index.json').exists(), 'Already frozen; do not rewrite the index'
    tests = json.loads((HERE/'source-tests.json').read_text())
    assert tests['tests'] == 4 and tests['failures'] == tests['errors'] == 0
    files = {}
    for path in sorted(HERE.rglob('*')):
        assert not path.is_symlink(), path
        if path.is_file():
            files[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    index = dict(schema='qp-mass-preflight-fixture-closed-index-v1', files=files,
                 native_artifacts_in_payload=False, newly_launched_native=False,
                 qualification='218 recording-stub caller fixtures; no amplitude workers in captured runs')
    (HERE/'index.json').write_text(json.dumps(index, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(files=len(files), index_sha256=hashlib.sha256((HERE/'index.json').read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
