#!/usr/bin/env python3
"""Verify this source-only proposal; no compiler, native import, or mutation."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError('Verification requires enabled assertions; do not use python -O.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', action='store_true')
    args = parser.parse_args()
    index = json.loads((HERE / 'evidence-index.json').read_text())
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}
    assert actual == set(index['files']) | {'evidence-index.json'}
    for name, entry in index['files'].items():
        path = HERE / name
        assert not path.is_symlink() and '..' not in Path(name).parts
        raw = path.read_bytes()
        assert not raw.startswith(b'\x7fELF') and path.suffix not in ('.pyc', '.o', '.mod', '.a', '.so')
        assert len(raw) == entry['bytes'] and digest(path) == entry['sha256'], name
    bindings = json.loads((HERE / 'source-bindings.json').read_text())
    assert not bindings['compiler_executed'] and not bindings['native_executed']
    if args.sources:
        for section in ('sources', 'evidence'):
            for name, entry in bindings[section].items():
                path = ROOT / name
                assert digest(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], name
    print(json.dumps({'verification': 'pass', 'portable_files': len(index['files']),
                      'source_paths_checked': args.sources, 'native_executed': False}, sort_keys=True))


if __name__ == '__main__':
    main()
