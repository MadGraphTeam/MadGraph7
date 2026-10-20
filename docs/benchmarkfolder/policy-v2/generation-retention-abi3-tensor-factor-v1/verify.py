#!/usr/bin/env python3
"""Rehash the complete clean-export evidence; optionally verify live builds."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
V2 = HERE.parent
sys.path.insert(0, str(ROOT/'tools'))
import ninjars_benchmark_policy_v2 as policy


def load(path):
    return json.loads(path.read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    seal = load(HERE/'manifest.json')
    for name, expected in seal['files'].items():
        path = ROOT/name
        assert path.stat().st_size == expected['bytes'], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], name
    generation = V2/'generation-abi3-tensor-factor-v1'
    status, summary = load(generation/'status.json'), load(generation/'summary.json')
    assert status['git_revision'] == summary['source_revision'] == 'f4bffc61ed67e5d083b80ff9901768d4d5779989'
    assert summary['core_case_count'] == 8 and summary['control_case_count'] == 5
    assert all(status[k] is True for k in ('complete', 'preserved_existing_inputs', 'dependencies_unchanged'))
    assert len(status['cases']) == 13
    for case, record in status['cases'].items():
        assert record['status'] == 'complete', case
        assert record['compilation']['build_kind'] == 'clean-subprocess', case
        assert record['compilation']['preexisting_subprocess_objects'] == 0, case
        assert record['loop_amplitude_evaluations'] == record['initialized_symbolica_workers'] == 0
    inventory = V2/'final-runtime-audit-abi3-tensor-factor-v1/inventory.json'
    retained = load(inventory)
    assert retained['case_count'] == 13 and retained['single_common_group_per_backend']
    reuse = load(inventory.with_name('reuse-check.json'))
    assert reuse['passed'] and reuse['numerical_evaluations'] == 0
    if args.live:
        actual = module(inventory.with_name('audit.py'), 'live_factor_inventory').audit(generation)
        assert actual == retained
        actual = module(inventory.with_name('check-reuse.py'), 'live_factor_reuse').check()
        assert actual == reuse
    print(json.dumps(dict(verified_files=len(seal['files']), complete_exports=13,
        source_revision=summary['source_revision'], inventory_identity=retained['identity_sha256'],
        exact_model_and_runtime_reuse=True, live=args.live,
        amplitude_evaluations=0, acceptance_evidence=False), indent=2))


if __name__ == '__main__':
    main()
