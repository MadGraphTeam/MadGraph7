#!/usr/bin/env python3
"""Offline hash/row/statistical verification; never starts a numerical binary."""
import hashlib
import json
from pathlib import Path

from analyze_affinity import analyze_result

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256')
    return digest.hexdigest()


def main():
    assert Path.cwd() == ROOT
    manifest = json.loads((HERE / 'manifest.json').read_text())
    identity = manifest.pop('identity')
    assert hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == identity
    for name, record in manifest['files'].items():
        path = HERE / name
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name
    actual = {str(path.relative_to(HERE)) for path in HERE.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts and path.name != 'manifest.json'}
    assert actual == set(manifest['files'])
    summary = analyze_result(HERE / 'timings/attempt-1/result.json')
    assert summary == json.loads((HERE / 'timings/attempt-1/summary.json').read_text())
    assert summary['all_42_blocks_exact'] and summary['clock_control']['passed']
    print(json.dumps({'status': 'passed', 'identity': identity,
        'files': len(manifest['files']), 'pairs': summary['pairs'],
        'ratio_of_medians': summary['ratio_of_medians'],
        'paired_bootstrap_95ci': summary['paired_bootstrap_95ci'],
        'acceptance_evidence': False, 'numerical_binary_executed': False}))


if __name__ == '__main__':
    main()
