#!/usr/bin/env python3
"""Read retained focused controls; never compile or invoke a numerical worker."""
from pathlib import Path
import collections
import datetime
import hashlib
import itertools
import json
import math
import os
import sys
import traceback

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'madgraph/__init__.py').is_file())
BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers

EXPECTED = '8b454519cb6818ef1fafe37378930c276f7e85394a2333ebc9fdb00e75726cff'
REUSE_SHA = '6fe6c7f4f76f46b2f2daee1e501883b7c49ddc9d2e1e954567b2a3cd03336ff6'
PREFIX = 'docs/benchmarkfolder/policy-v2/'


def bound(p):
    p = policy.path(p)
    return {'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}


def write(p, value):
    with p.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    if (BASE / 'receipt.json').exists():
        raise RuntimeError('Retain the existing audit; do not overwrite')
    reuse_path = ROOT / PREFIX / 'final-focused-reference-reuse-v1/receipt.json'
    if bound(reuse_path)['sha256'] != REUSE_SHA:
        raise RuntimeError('Helper reuse receipt changed')
    reuse = policy.read(reuse_path)
    if not reuse['passed']:
        raise RuntimeError('Helper receipt is not successful')
    correction = policy.validate_correction(ROOT / PREFIX / 'model-correction.json')
    inventory_path = ROOT / PREFIX / 'final-core-generation-v1/runtime-inventory.json'
    inventory = policy.verified(policy.read(inventory_path), 'fresh-generated-runtime-inventory-v2')
    signature = inventory['runtime_groups']['8'][EXPECTED]['signature']
    specifications = {
        'complex': (PREFIX + 'final-focused-controls-v1/runtime-bindings/complex.json', 'control-complex', 'candidate_corrected', {1, 6, 7, 8}),
        'mixed-legacy': (PREFIX + 'final-focused-controls-v1/runtime-bindings/mixed-legacy.json', 'control-mixed', 'candidate_corrected', {1, 6, 7}),
        'mixed8': (PREFIX + 'final-representative-v1/runtime-bindings/mixed.json', 'control-mixed', 'candidate_corrected', {8}),
        'complex-helper': (reuse['cases']['control-complex']['binding']['path'], 'control-complex', 'original_helper_corrected', {1, 6, 7}),
        'mixed-helper': (reuse['cases']['control-mixed']['binding']['path'], 'control-mixed', 'original_helper_corrected', {1, 6, 7}),
    }
    cases = {name: {} for name in ('control-complex', 'control-mixed')}
    validations, inputs = {}, {}
    inputs[str(reuse_path.relative_to(ROOT))] = bound(reuse_path)
    inputs[str(inventory_path.relative_to(ROOT))] = bound(inventory_path)
    for label, (filename, case, role, backends) in specifications.items():
        bindings = policy.checked_bindings(filename, correction, live=True)
        report = policy.path(bindings['report'])
        protocol, records = workers.checked_series(report, minimum=2)
        if protocol['repetitions'] != 2:
            raise RuntimeError('Expected the actual complete two-command diagnostic stream')
        selected = {key: item for key, item in bindings['targets'].items() if item['case'] == case and item['role'] == role}
        if {item['backend'] for item in selected.values()} != backends:
            raise RuntimeError('Focused selected backend set differs')
        validations[label] = {'binding': bound(filename), 'live_sources_runtime_executables_verified': True, 'targets': {}}
        inputs[filename] = bound(filename)
        for key, item in selected.items():
            backend = item['backend']
            if role == 'candidate_corrected' and item['reducer_runtime'] != signature:
                raise RuntimeError('Candidate full reducer descriptor differs from final runtime')
            if role != 'candidate_corrected':
                reference = next(t for t in reuse['cases'][case]['selected_targets'] if t['backend'] == backend)
                if (reference['full_reducer_runtime'] != item['reducer_runtime'] or
                        reference['executable_sha256'] != item['executable_sha256']):
                    raise RuntimeError('Selected helper differs from root-revalidated receipt')
            selected_records = []
            for repetition in (0, 1):
                path, data = records[case, 'candidate' if role == 'candidate_corrected' else 'baseline', backend, repetition]
                inputs[str(path.relative_to(ROOT))] = bound(path)
                if data['points_sha256'] != bound(ROOT / 'docs/benchmarkfolder/points' / (case + '.dat'))['sha256']:
                    raise RuntimeError('Actual point stream differs from fixed retained input')
                if data['generation_source_identity'] != item['generation_source_identity']:
                    raise RuntimeError('Actual raw generation identity differs from source binding')
                selected_records.append(data)
            config = protocol['configuration']
            workers.bench.validate_repetition(selected_records[0], selected_records[1], config)
            historical_no_fresh = label == 'complex-helper'
            if historical_no_fresh:
                if protocol['fresh_parity']:
                    raise RuntimeError('Historical complex fresh parity unexpectedly changed')
                parity = {'observed_fresh_parity_enabled': False, 'reason': 'Historical helper collection did not execute a fresh-process command.'}
            else:
                if not protocol['fresh_parity']:
                    raise RuntimeError('Required fresh-process parity was not collected')
                fresh_path = report / 'data' / (key + '-fresh.json')
                fresh = policy.read(fresh_path)
                if fresh['executable_sha256'] != item['fresh_executable_sha256']:
                    raise RuntimeError('Fresh executable differs from bound executable')
                parity = workers.numerical_parity(selected_records[0], fresh, config)
                if not parity['exact_rows']:
                    raise RuntimeError('Actual candidate/helper fresh rows differ')
                inputs[str(fresh_path.relative_to(ROOT))] = bound(fresh_path)
            name = ('candidate' if role == 'candidate_corrected' else 'helper') + '/' + str(backend)
            if name in cases[case]:
                raise RuntimeError('Duplicate physical target')
            cases[case][name] = selected_records
            validations[label]['targets'][name] = {
                'case': case, 'role': role, 'backend': backend,
                'runtime_identity_sha256': item['reducer_runtime']['identity_sha256'],
                'executable_sha256': item['executable_sha256'],
                'generation_revision': item['generation_source_identity'].get('revision'),
                'generation_tree_identity_sha256': item['generation_source_identity']['tree_state_sha256'],
                'answer_dimensions': selected_records[0]['answer_dimensions'],
                'rows_per_command': len(selected_records[0]['rows']),
                'physical_status_counts_per_command': [r['status_counts'] for r in selected_records],
                'status_categories': sorted({r['status'] // 100 for d in selected_records for r in d['rows']}),
                'two_complete_persistent_commands': True, 'fresh_parity': parity,
            }
    comparisons = {}
    for case, targets in cases.items():
        if set(targets) != {'candidate/1', 'candidate/6', 'candidate/7', 'candidate/8', 'helper/1', 'helper/6', 'helper/7'}:
            raise RuntimeError('Missing candidate or helper target')
        repetitions = []
        for repetition in (0, 1):
            groups = {name: records[repetition]['rows'] for name, records in targets.items()}
            data = [records[repetition] for records in targets.values()]
            for field in ('points_sha256', 'parameter_card_sha256', 'answer_dimensions'):
                if any(d[field] != data[0][field] for d in data):
                    raise RuntimeError('Cross-backend physical input differs: ' + field)
            # Collier has no QP numerator toggle; compare all common numerical settings.
            settings = [{k: v for k, v in d['shared_numerical_settings'].items()
                         if k != 'selected_backend_qp_integrand'} for d in data]
            if any(s != settings[0] for s in settings):
                raise RuntimeError('Cross-backend numerical settings differ')
            scales = {}
            for rows in groups.values():
                for row in rows:
                    scales[row['point']] = max(scales.get(row['point'], 0.0), *map(abs, row['coefficients']))
            agreement = policy.compare_rows(groups, scales=scales)
            maximum = {'fraction_of_tolerance': -1.0}
            pair_metrics, scalar_comparisons = [], 0
            for left, right in itertools.combinations(groups, 2):
                worst = {'fraction_of_tolerance': -1.0}
                for a, b in zip(groups[left], groups[right]):
                    for coefficient, (x, y) in enumerate(zip(a['coefficients'], b['coefficients'])):
                        tolerance = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scales[a['point']]
                        delta = abs(x-y)
                        fraction = delta / tolerance if tolerance else (0.0 if not delta else math.inf)
                        item = {'fraction_of_tolerance': fraction, 'targets': [left, right], 'point': a['point'],
                                'order': a['order'], 'coefficient': coefficient, 'values': [x, y],
                                'absolute_difference': delta, 'tolerance': tolerance, 'point_scale': scales[a['point']]}
                        if fraction > worst['fraction_of_tolerance']: worst = item
                        if fraction > maximum['fraction_of_tolerance']: maximum = item
                        scalar_comparisons += 1
                pair_metrics.append({'targets': [left, right], 'maximum': worst})
            repetitions.append({'repetition': repetition, 'agreement': agreement, 'pairs': pair_metrics,
                                'scalar_comparisons': scalar_comparisons, 'maximum': maximum,
                                'common_point_scales': scales})
        comparisons[case] = {'passed': all(r['agreement']['passed'] for r in repetitions),
                            'physical_points': 10, 'coefficient_order': ['Born', 'finite', 'single-pole', 'double-pole'],
                            'order_indices_including_total': sorted({r['order'] for r in next(iter(targets.values()))[0]['rows']}),
                            'repetitions': repetitions}
    result = policy.seal({'schema': 'final-focused-correctness-audit-v1',
        'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'source_only_audit': True,
        'compiler_calls': 0, 'numerical_executions': 0, 'tests_or_worker_reruns': 0,
        'old_candidate_measurements_reused': False,
        'final_representative_mixed8_reused_without_new_evaluation': True,
        'candidate_runtime_identity_sha256': EXPECTED,
        'validation': validations, 'comparisons': comparisons,
        'total_scalar_pairwise_comparisons': sum(r['scalar_comparisons'] for c in comparisons.values() for r in c['repetitions']),
        'maximum_fraction_of_tolerance': max(r['maximum']['fraction_of_tolerance'] for c in comparisons.values() for r in c['repetitions']),
        'passed': all(c['passed'] for c in comparisons.values()),
        'bound_inputs': inputs,
        'auditor_source': bound(__file__),
        'reader_sources': {name: bound(ROOT/'tools'/name) for name in ('ninjars_benchmark_policy_v2.py', 'ninjars_benchmark_workers.py', 'ninjars_benchmark.py')},
        'limitations': ['Ten fixed points per control; this is focused correctness evidence, not the eight-process 100-point gate or a performance claim.',
                        'Historical complex helper data has two persistent commands and no fresh-process parity measurement.']})
    write(BASE/'receipt.json', result)
    print(json.dumps({'passed': result['passed'], 'identity_sha256': result['identity_sha256'],
                      'comparisons': result['total_scalar_pairwise_comparisons'],
                      'maximum_fraction_of_tolerance': result['maximum_fraction_of_tolerance']}, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if not isinstance(error, SystemExit) or error.code:
            if not (BASE/'failure.json').exists():
                write(BASE/'failure.json', {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
        raise
