#!/usr/bin/env python3
"""Verify the complete portable Born-prototype inventory without native calls."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    index_path = HERE / 'evidence-index.json'
    index = json.loads(index_path.read_text())
    expected = index['files']
    found = set()
    for path in HERE.rglob('*'):
        if path.is_symlink():
            raise SystemExit(f'Symlink in portable package: {path}')
        if not path.is_file() or path == index_path:
            continue
        relative = str(path.relative_to(HERE))
        found.add(relative)
        if path.suffix in ('.pyc', '.o', '.a', '.so', '.dylib') or path.read_bytes()[:4] == b'\x7fELF':
            raise SystemExit(f'Compiled artifact in portable package: {relative}')
        entry = expected.get(relative)
        if entry is None:
            raise SystemExit(f'Unindexed file: {relative}')
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise SystemExit(f'Hash/size mismatch: {relative}')
    if found != set(expected):
        raise SystemExit(f'Missing indexed files: {sorted(set(expected) - found)}')
    # Explicitly bind historical failed attempts to their preserved source copy,
    # rather than claiming those old paths still contain the old source bytes.
    old = json.loads((HERE / 'tree-native-1/execution.json').read_text())
    for name in ('tree_driver.f', 'run_tree.py'):
        match = [value for key, value in old['protected_inputs_before'].items()
                 if key.endswith('/qcd-born-fallback-prototype/' + name)]
        assert len(match) == 1 and sha(HERE / 'tree-native-1' / name) == match[0], name
    original = json.loads((HERE / 'revision-1/preparation.json').read_text())
    prefix = 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype/'
    for name, expected_sha in original['files_sha256'].items():
        if name.startswith(prefix):
            assert sha(HERE / 'revision-1' / name[len(prefix):]) == expected_sha, name
    print(json.dumps({'verified_unique_files': len(found), 'index_sha256': sha(index_path),
                      'native_executed': False, 'production_promoted': False}, sort_keys=True))


if __name__ == '__main__':
    main()
