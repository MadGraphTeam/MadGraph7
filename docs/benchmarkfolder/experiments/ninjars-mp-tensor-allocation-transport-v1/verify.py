#!/usr/bin/env python3
"""Offline seal/correctness validation, never numerical execution."""
import hashlib
import json
from pathlib import Path

from summarize import HERE, ROOT, sha, summarize


def main():
    assert Path.cwd() == ROOT
    manifest = json.loads((HERE / 'manifest.json').read_text())
    identity = manifest.pop('identity')
    assert hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == identity
    for name, record in manifest['files'].items():
        path = HERE / name
        assert path.is_file() and not path.is_symlink()
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], name
    actual = {str(path.relative_to(HERE)) for path in HERE.rglob('*') if path.is_file()
              and '__pycache__' not in path.parts and path.name != 'manifest.json'}
    assert actual == set(manifest['files'])
    summary = summarize()
    assert summary == json.loads((HERE / 'summary.json').read_text())
    print(json.dumps({'status': 'passed', 'identity': identity, 'files': len(manifest['files']),
        'tests': summary['tests_passed'], 'golden_lanes': summary['golden_lanes'],
        'records_per_replay': summary['state_replay_records'], 'fields_per_replay': 11136,
        'numerical_binary_executed': False, 'production_changed': False}))


if __name__ == '__main__':
    main()
