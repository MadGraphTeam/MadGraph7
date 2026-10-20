#!/usr/bin/env python3
"""Recheck the retained audit; no network, compiler or numerical engine calls."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE/name).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', action='store_true',
                        help='Also check the current source inputs against this checkpoint')
    args = parser.parse_args()
    manifest = read('manifest.json')
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.name != 'manifest.json'}
    assert actual == set(manifest['files'])
    for name, info in manifest['files'].items():
        path = HERE/name
        assert not path.is_symlink()
        assert path.stat().st_size == info['bytes']
        assert sha(path) == info['sha256'], name
    checkpoint = read('checkpoint.json')
    assert checkpoint == read('checkpoint-stdout.json')
    assert checkpoint['complete'] and not checkpoint['update_pending']
    assert set(checkpoint['repositories']) == {'ninjars', 'oneloopmaster'}
    for record in checkpoint['repositories'].values():
        assert record['main'] == record['default_head'] == record['pinned']
        assert not record['update_pending'] and not record['tags']
    dependencies = checkpoint['git_dependencies']
    assert len(dependencies) == 5
    assert all(len(item['revision']) == 40 and item['packages'] for item in dependencies)
    validation = read('validation.json')
    assert (validation['tests'], validation['new_tests'], validation['engine_executions']) == (30, 19, 0)
    assert validation['cpu_affinity'] == [3]
    assert validation['ninjars_and_oneloopmaster_current']
    assert validation['symbolica_pin_changed'] is False
    assert all(record['exit_code'] == 0 for record in validation['commands'])
    assert 'Ran 30 tests' in (HERE/'focused-tests.log').read_text()
    assert (HERE/'focused-tests.log').read_text().rstrip().endswith('OK')
    installed = read('installed-proof.log')
    assert installed == {'passed':True, 'files':63, 'live':True,
        'standalone_statuses':[110,120,218], 'nlo_poles':20,
        'compiled_smoke':True, 'engine_executions':0}
    assert (HERE/'symbolica-main-ancestry.txt').read_text().split() == ['596','0']
    assert (HERE/'symbolica-ninjars-pin-ancestry.txt').read_text().split() == ['0','1']
    assert validation['symbolica']['candidate_tested'] is False
    patch = (HERE/'symbolica-9cdc.patch').read_text()
    assert patch.count('diff --git ') == 3
    assert 'fn float_products_round_once()' in patch
    assert 'fn float_unit_factors_are_exact()' in patch
    assert 'fn float_evaluator_keeps_exact_differences()' in patch
    if args.sources:
        for name, expected in validation['source_files'].items():
            assert sha(ROOT/name) == expected, name
    print(json.dumps({'passed':True, 'files':len(actual), 'sources':args.sources,
        'tests':30, 'new_tests':19, 'engine_executions':0}))


if __name__ == '__main__':
    main()
