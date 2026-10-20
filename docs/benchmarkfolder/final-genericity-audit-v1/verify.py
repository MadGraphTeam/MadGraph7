#!/usr/bin/env python3
"""Verify the bounded genericity source census without regenerating it."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    files = {str(p.relative_to(HERE)): p for p in HERE.rglob('*')
             if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    assert set(files) == set(manifest['files'])
    for name, path in files.items():
        assert not path.is_symlink()
        assert path.stat().st_size == manifest['files'][name]['bytes']
        assert sha(path) == manifest['files'][name]['sha256'], name
    census = json.loads((HERE / 'source-census.json').read_text())
    assert census['count'] == len(census['files']) == 42
    assert census['new_production_matches'] == []
    assert not any(item['added_or_modified_line'] for item in census['matches_for_manual_review'])
    assert census['builds_started'] == census['tests_started'] == census['reduction_engine_calls'] == 0
    if args.sources:
        for name, item in census['files'].items():
            assert sha(ROOT / name) == item['sha256'], name
    print(json.dumps({'passed': True, 'files': len(files), 'source_paths': 42,
                      'new_production_pattern_matches': 0, 'sources': args.sources,
                      'reduction_engine_calls': 0, 'tests_started': 0}))


if __name__ == '__main__':
    main()
