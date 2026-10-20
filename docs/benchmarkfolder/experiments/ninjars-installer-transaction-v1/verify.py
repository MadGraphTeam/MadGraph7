#!/usr/bin/env python3
"""Check sealed red/green evidence without executing tests or an installer."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', action='store_true', help='Also check the current worktree source')
    args = parser.parse_args()
    manifest = json.loads((HERE/'manifest.json').read_text())
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.name != 'manifest.json'}
    assert actual == set(manifest['files'])
    for name, record in manifest['files'].items():
        path = HERE/name
        assert not path.is_symlink()
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], name
    result = json.loads((HERE/'validation.json').read_text())
    assert result['passed'] and result['new_regression_tests'] == 7
    assert result['compiler_calls'] == result['numerical_engine_calls'] == 0
    assert result['cpu_affinity'] == [3] and result['quiet_released']
    red, green = result['original_red'], result['final_green']
    assert (red['tests'],red['failures'],red['errors'],red['passed']) == (3,2,1,False)
    assert (green['tests'],green['failures'],green['errors'],green['skipped'],green['passed']) == (37,0,0,0,True)
    assert red['log_sha256'] == sha(HERE/'original-failure-tests.log')
    assert green['log_sha256'] == sha(HERE/'focused-tests.log')
    assert 'FAILED (failures=2, errors=1)' in (HERE/'original-failure-tests.log').read_text()
    assert 'Ran 37 tests' in (HERE/'focused-tests.log').read_text()
    assert (HERE/'focused-tests.log').read_text().rstrip().endswith('OK')
    assert sha(HERE/'original/ninjars_install.py') == result['original_source_sha256'] == 'aabedacfb872d3d3527dab19f162fc2733efc9fe05b52aec42d4b2d4145c06de'
    assert sha(HERE/'collect.py') == result['collector_sha256']
    for name, expected in result['source_files'].items():
        assert sha(HERE/'final'/Path(name).name) == expected
        if args.sources:
            assert sha(ROOT/name) == expected, name
    print(json.dumps({'passed':True,'files':len(actual),'sources':args.sources,
        'original_expected_failures':3,'focused_tests':37,'new_tests':7,'engine_calls':0}))


if __name__ == '__main__':
    main()
