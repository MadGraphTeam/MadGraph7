#!/usr/bin/env python3
"""Check this independent report/reader audit without executing its suites."""
import argparse
import hashlib
import json
from pathlib import Path
import re

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
    result = json.loads((HERE / 'validation.json').read_text())
    assert result['passed'] and result['exit_code'] == 0 and result['tests_run'] == 55
    assert result['all_children_exited'] and result['remaining_process_group_members'] == []
    assert result['source_unchanged_during_tests'] and len(result['source_sha256']) == 13
    assert result['reduction_engine_calls'] == result['builds_of_numerical_code'] == 0
    assert not result['final_goal_claimed']
    for name, expected in result['logs'].items():
        assert sha(HERE / name) == expected
    text = (HERE / 'unittest.stderr').read_text()
    assert re.findall(r'^Ran (\d+) tests? in [0-9.]+s$', text, re.M) == ['55']
    assert re.findall(r'^OK$', text, re.M) == ['OK']
    assert len(re.findall(r'^test_.* \.\.\. ok$', text, re.M)) == 55
    if args.sources:
        for name, expected in result['source_sha256'].items():
            assert sha(ROOT / name) == expected, name
    print(json.dumps({'passed': True, 'tests': 55, 'files': len(files),
                      'sources': args.sources, 'reduction_engine_calls': 0}))


if __name__ == '__main__':
    main()
