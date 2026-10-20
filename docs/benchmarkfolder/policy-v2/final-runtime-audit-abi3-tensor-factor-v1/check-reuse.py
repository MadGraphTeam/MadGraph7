#!/usr/bin/env python3
"""Validate the exact reuse boundary after fresh generation, without evaluation."""
from pathlib import Path
import json
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'madgraph/__init__.py').is_file())
sys.path.insert(0, str(ROOT / 'tools'))
import ninjars_benchmark_policy_v2 as policy

V2 = ROOT / 'docs/benchmarkfolder/policy-v2'
GENERATION = V2 / 'generation-abi3-tensor-factor-v1'
HERE = Path(__file__).resolve().parent


def check():
    _, correction, _ = policy.validate_policy(V2 / 'policy-runtime-groups.json')
    old = policy.verified(policy.read(V2 / 'final-runtime-audit-fixed/inventory.json'),
                          'fresh-generated-runtime-inventory-v2')
    current = policy.verified(policy.read(HERE / 'inventory.json'),
                              'fresh-generated-runtime-inventory-v2')
    status = policy.read(policy.check_file(current['status']))
    assert all(status.get(key) is True for key in
               ('complete', 'preserved_existing_inputs', 'dependencies_unchanged'))
    assert set(old['cases']) == set(current['cases']) == set(correction['cases'])
    checked = {}
    for case_id, item in current['cases'].items():
        compiled = policy.read(policy.check_file(item['compiled_manifest']))
        policy.validate_source_role('candidate_corrected', case_id, compiled,
                                    item['generation_source_identity'], correction)
        if (item['signatures'] != old['cases'][case_id]['signatures'] or
                item['runtime_metadata'] != old['cases'][case_id]['runtime_metadata']):
            raise RuntimeError('Runtime/interface metadata changed: ' + case_id +
                '; refresh ABI-level controls and exact group coverage before reusing evidence')
        checked[case_id] = {
            'compiled_manifest': item['compiled_manifest'],
            'model_identity_sha256': policy.bench.content_digest(policy.model_subset(compiled)),
            'signatures': item['signatures'],
        }
    return policy.seal({
        'schema': 'tensor-factor-exact-reuse-check-v1',
        'passed': True, 'numerical_evaluations': 0,
        'correction': policy.bound_file(V2 / 'model-correction.json'),
        'previous_inventory': policy.bound_file(V2 / 'final-runtime-audit-fixed/inventory.json'),
        'fresh_inventory': policy.bound_file(HERE / 'inventory.json'),
        'checker': policy.bound_file(Path(__file__)),
        'cases': checked,
        'scope': 'Exact generated model postimages and unchanged numerical runtime/metadata; '
                 'fresh candidate process proof and controls remain required.',
    })


if __name__ == '__main__':
    output = HERE / 'reuse-check.json'
    if output.exists():
        raise RuntimeError('Preserve previous reuse evidence')
    result = check()
    policy.bench.dump(output, result)
    print(json.dumps(result, indent=2))
