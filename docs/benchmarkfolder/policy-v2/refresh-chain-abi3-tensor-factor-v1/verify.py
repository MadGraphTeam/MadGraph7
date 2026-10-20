#!/usr/bin/env python3
"""Static held-plan verification; never compile, launch a worker or write evidence."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
V2 = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))
import ninjars_benchmark_policy_v2 as policy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(text, record):
    for before, after in record['namespace_mapping'].items():
        text = text.replace(before, after)
    return text.replace(record['old_runtime'], record['new_runtime'])


def verify(require_unlaunched=False):
    if (HERE / 'manifest.json').exists():
        sealed = policy.verified(policy.read(HERE / 'manifest.json'),
                                 'held-tensor-factor-refresh-retention-v1')
        for name, digest in sealed['files'].items():
            assert sha(ROOT / name) == digest, name
        for descriptor in sealed['reused_evidence']:
            policy.check_file(descriptor)
    copies = policy.read(HERE / 'copied-sources.json')
    for entry in copies['copies']:
        before, after = (policy.check_file(entry[key]) for key in ('before', 'after'))
        expected = before.read_text()
        kind = entry['transformation']
        if kind != 'byte-identical':
            if before.name == 'execute.py':
                expected = expected.replace(
                    '.benchmark-work/policy-v2/final-variant-controls-both-precision',
                    '.benchmark-work/policy-v2/final-variant-controls-abi3-tensor-factor-v1')
                expected = expected.replace(copies['old_runtime'], copies['new_runtime'])
                if kind == 'variant-namespace-and-default-cpu':
                    expected = expected.replace('type=int, default=4)', 'type=int, default=5)')
            else:
                expected = transform(expected, copies)
        if kind == 'namespace-and-diagnostic-cpu':
            expected_json = json.loads(expected)
            expected_json['cpu'] = 5
            assert json.loads(after.read_text()) == expected_json
        else:
            assert after.read_text() == expected, str(after)

    folders = [V2 / name for name in copies['namespace_mapping'].values()]
    checked = []
    for folder in folders:
        plan = policy.verified(policy.read(folder / 'plan.json'), 'held-tensor-factor-refresh-phase-v1')
        assert plan['state'].startswith('HELD-') and not plan['numerical_launch']
        assert not plan['acceptance_claim']
        for descriptor in ('policy', 'correction', 'configuration'):
            policy.check_file(plan[descriptor])
        for name, digest in plan['files'].items():
            assert sha(ROOT / name) == digest, name
        for source in folder.rglob('*.py'):
            ast.parse(source.read_text(), filename=str(source))
            checked.append(str(source.relative_to(ROOT)))
        for source in folder.rglob('*.sh'):
            subprocess.run(['bash', '-n', str(source)], check=True, cwd=ROOT)
            assert 'export TMPDIR="$PWD/.ninjars-build/tmp"' in source.read_text()
            checked.append(str(source.relative_to(ROOT)))
        if require_unlaunched:
            assert not any((folder / name).exists() for name in (
                'status.json', 'summary.json', 'inventory.json', 'prepared.json', 'comparison.json'))

    rules, correction, config = policy.validate_policy(V2 / 'policy-runtime-groups.json')
    controls = policy.read(V2 / 'controls.json')
    assert len(config['cases']) == 8 and config['points_per_case'] == 100
    assert config['repetitions'] == 7 and controls['points_per_case'] == 10
    all_cases = [c['id'] for name in ('cases.json', 'controls.json')
                 for c in policy.read(ROOT / 'docs/benchmarkfolder' / name)['cases']]
    assert len(all_cases) == len(set(all_cases)) == 13

    candidate = V2 / 'final-candidate-plan-abi3-tensor-factor-v1'
    wanted = {'ggh': (['control-gg-h'], [1, 6, 7, 8]),
              'mixed': (['control-mixed'], [8]),
              'forced-qp': (['control-forced-qp'], [1, 8]),
              'ninja-near-singular': (['control-forced-qp'], [6])}
    target_count = 0
    for name, (cases, backends) in wanted.items():
        spec = policy.read(candidate / 'specs' / (name + '.json'))
        assert spec['cases'] == cases and spec['backends'] == backends
        assert spec['sources'] == ['candidate'] and spec['cpu'] == 5
        assert spec['fresh_parity'] and spec['representative']
        assert spec['configuration'] == 'docs/benchmarkfolder/policy-v2/controls.json'
        assert spec['output_roots']['baseline'] == '.benchmark-work/outputs/baseline'
        assert spec['output_roots']['candidate'] == '.benchmark-work/policy-v2/generation-abi3-tensor-factor-v1/outputs/candidate'
        if require_unlaunched:
            assert not (ROOT / spec['report']).exists() and not (ROOT / spec['work']).exists()
        target_count += len(cases) * len(backends)
    assert target_count == 8
    for source in (V2 / 'final-controls-plan-abi3-tensor-factor-v1/specs').glob('*.json'):
        assert policy.read(source)['cpu'] == 5
    variant_commands = (V2 / 'final-variant-controls-abi3-tensor-factor-v1/commands.sh').read_text()
    assert '--cpu 5' in variant_commands and '--cpu 4' not in variant_commands
    generation_commands = (V2 / 'generation-abi3-tensor-factor-v1/commands.sh').read_text()
    assert 'GENERATION_CPUS:-7,8' in generation_commands

    old_comparison = policy.read(V2 / 'final-controls-plan-fixed/comparison-spec.json')
    comparison = policy.read(V2 / 'final-controls-plan-abi3-tensor-factor-v1/comparison-spec.json')
    assert comparison['policy'] == old_comparison['policy']
    assert comparison['configuration'] == old_comparison['configuration']
    assert comparison['proof'] == transform(old_comparison['proof'], copies)
    legacy_files = set()
    for case_id, targets in comparison['cases'].items():
        assert len(targets) == 7
        assert {(t['role'], t['backend']) for t in targets} == {
            *[('candidate_corrected', b) for b in (1, 6, 7, 8)],
            *[('original_helper_corrected', b) for b in (1, 6, 7)]}
        for before, after in zip(old_comparison['cases'][case_id], targets):
            assert before['role'] == after['role'] and before['backend'] == after['backend']
            if before['role'] == 'original_helper_corrected':
                assert before == after
                legacy_files.add(after['binding'])
            else:
                assert after['binding'] == transform(before['binding'], copies)
    commands = (V2 / 'final-controls-plan-abi3-tensor-factor-v1/commands.sh').read_text()
    assert 'controls-reference-hh-complex' not in commands
    assert '--correction' in commands  # New candidate binding only.
    for filename in sorted(legacy_files):
        binding = policy.checked_bindings(ROOT / filename, correction)
        assert {item['role'] for item in binding['targets'].values()} == {'original_helper_corrected'}

    core = V2 / 'final-core-plan-abi3-tensor-factor-v1-24h'
    before = policy.read(V2 / 'final-core-plan-fixed-24h/spec.json')
    after = policy.read(core / 'spec.json')
    assert after == json.loads(transform(json.dumps(before), copies))
    assert after['cases'] == [case['id'] for case in config['cases']]
    assert after['cpu'] == 207 and after['timeout'] == 86400
    assert after['backends'] == [1, 6, 7, 8] and after['sources'] == ['baseline', 'candidate']
    assert not after['representative'] and not after['fresh_parity']
    if require_unlaunched:
        assert not (ROOT / after['report']).exists() and not (ROOT / after['work']).exists()
    reference = 'runtime-bindings-core-24h-20261020/corrected-original-core-v2-24h-20261020.json'
    assert reference in (core / 'commands.sh').read_text()
    assert reference in (core / 'publish.sh').read_text()
    legacy = policy.verified(policy.read(V2 / 'legacy-coverage-status.json'),
                             'legacy-representative-coverage-status-v2')
    assert len(legacy['bindings']) == 12
    for descriptor in legacy['bindings']:
        policy.check_file(descriptor)
    return {'schema': 'held-tensor-factor-refresh-static-validation-v1',
            'passed': True, 'numerical_launch': False, 'build_launch': False,
            'required_unlaunched': require_unlaunched,
            'generation_cases': all_cases, 'candidate_representative_targets': 8,
            'reused_legacy_representative_bindings': 12,
            'new_generated_control_targets': 11, 'reused_legacy_control_targets': 12,
            'reused_new_candidate_control_targets': 5, 'core_cpu': 207,
            'representative_cpu': 5, 'core_timeout_seconds': 86400,
            'generated_controls_cpu': 5, 'variants_cpu': 5,
            'generation_default_cpus': [7, 8],
            'legacy_control_bindings': sorted(legacy_files),
            'checked_syntax': checked,
            'copied_sources': policy.bound_file(HERE / 'copied-sources.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-unlaunched', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.require_unlaunched), indent=2))
