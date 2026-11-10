#!/usr/bin/env python3
"""Read-only inventory/source verification; no numerical code imports."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', action='store_true')
    args = parser.parse_args()
    index = json.loads((HERE / 'evidence-index.json').read_text())
    paths = list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(HERE)) for p in paths if p.is_file()} == set(index['files']) | {'evidence-index.json'}
    for name, entry in index['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        path = HERE / name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
    bindings = json.loads((HERE / 'source-bindings.json').read_text())
    if args.sources:
        for name, entry in bindings['files'].items():
            path = ROOT / name
            assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], name
            for anchor in entry.get('anchors', []):
                assert anchor in path.read_text(), (name, anchor)
    print(json.dumps({'portable_files': len(index['files']), 'external_bindings': len(bindings['files']),
                      'external_checked': args.sources, 'native_executed': False,
                      'production_ready': False}, sort_keys=True))


if __name__ == '__main__':
    main()
