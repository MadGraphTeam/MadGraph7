#!/usr/bin/env python3
"""Summarize retained attribution; never interpret instrumented runs as a gate."""
import ast
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return {'values': values, 'median': statistics.median(values),
            'minimum': min(values), 'maximum': max(values)}


def function_ast(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.dump(node, include_attributes=False)
    raise ValueError('Missing function: '+name)


def main():
    matrix_path = HERE/'attribution-1/results.json'
    matrix = json.loads(matrix_path.read_text())
    control_path = HERE/'original-controls-1/results.json'
    control = json.loads(control_path.read_text())
    result = {'performance_eligible': False,
              'purpose': 'descriptive diagnostic attribution; no optimization has been applied',
              'sources': {str(p.relative_to(HERE)): digest(p) for p in (matrix_path, control_path)},
              'units': 'microseconds per full amplitude unless otherwise stated',
              'full_amplitude_evaluations_per_batch': matrix['loops']*100,
              'levels': {}, 'stages': {},
              'limits': [
                  'Three repetitions describe stage costs and ranges, not a confidence interval.',
                  'Cross-level differences include timer, instrumentation, DSO and host perturbations.',
                  'Nested bridge clocks overlap the Fortran wrapper stage and must not be added.',
                  'Automatic-array allocation/free and IEEE procedure entry/exit lie outside the adapter timer brackets.',
                  'Empty-clock subtraction estimates only the interval overhead, not all observer overhead.',
                  'Original paired controls are diagnostic; the separate qualified campaign gate remains authoritative.']}
    for level in matrix['levels']:
        runs = [r for r in matrix['runs'] if r['level'] == level]
        summary = {'wall': distribution([r['observation']['seconds_per_point']*1e6 for r in runs]),
                   'maximum_numerical_relative_error': max(r['parity_maximum_relative_error'] for r in runs),
                   'actual_backend_precision_counts': [r['observation']['backend_precision_counts'] for r in runs]}
        if level != 'pristine':
            summary['cpu'] = distribution([r['profile']['clock']['batch_cpu_seconds'] /
                                          (matrix['loops']*100)*1e6 for r in runs])
            summary['empty_interval_nanoseconds'] = distribution([
                r['profile']['clock']['calibration_internal_seconds'] /
                r['profile']['clock']['calibration_iterations']*1e9 for r in runs])
            summary['empty_pair_total_nanoseconds'] = distribution([
                r['profile']['clock']['calibration_wall_seconds'] /
                r['profile']['clock']['calibration_iterations']*1e9 for r in runs])
            summary['bridge_failed_calls'] = sum(r['bridge_failed_calls'] for r in runs)
        result['levels'][level] = summary
        if not level.startswith('1:'):
            continue
        stages = {}
        for stage in range(1, 8):
            raw, adjusted, counts = [], [], []
            for run in runs:
                rows = [r for r in run['profile']['stages'] if r['stage'] == stage]
                raw.append(sum(r['seconds_per_point'] for r in rows)*1e6)
                adjusted.append(sum(r['timer_adjusted_estimate_seconds_per_call']*r['calls']
                                    for r in rows)/(matrix['loops']*100)*1e6)
                counts.append({str(r['ctmode']): r['calls'] for r in rows})
            stages[str(stage)] = {'name': matrix['preparation']['stages'][str(stage)],
                                  'raw': distribution(raw), 'timer_adjusted_estimate': distribution(adjusted),
                                  'calls_by_ctmode': counts}
        result['stages'][level] = stages
    result['all_profile_counts_expected'] = all(r.get('expected_n1_stage_counts_passed', True)
                                               for r in matrix['runs'])
    result['all_profile_values_bit_identical'] = all(r['parity_maximum_relative_error'] == 0
                                                   for r in matrix['runs'])
    result['nonbridge_sum_of_stage_medians'] = sum(result['stages']['1:0'][str(s)]['raw']['median']
                                                  for s in (1,2,3,4,6,7))
    result['original_controls'] = {
        'performance_eligible': False,
        'statistics_note': 'The raw acceptance_eligible field checks pair count only; this diagnostic is not a qualified gate.',
        'paired_statistics': control['statistics'],
        'wall': {name: distribution([v*1e6 for v in values]) for name, values in control['timings'].items()},
        'all_numerical_parity_passed': control['all_numerical_parity_passed'],
        'maximum_numerical_relative_error': max(p['maximum_relative_error'] for p in control['parity']),
        'actual_backend_precision_counts': {name: [r['observation']['backend_precision_counts']
                                                   for r in control['runs'] if r['source'] == name]
                                           for name in ('baseline','candidate')}}
    frozen = HERE/'protocol/tirust_benchmark.py'
    captured = HERE/'protocol/profile-campaign-captured.py'
    result['protocol_proof'] = {
        'historical_runner_matches_record': digest(HERE/'protocol/profile-run-used.py') == matrix['run_source_sha256'],
        'frozen_campaign_sha256': digest(frozen), 'captured_campaign_sha256': digest(captured),
        'used_helpers_ast_identical': {name: function_ast(frozen,name) == function_ast(captured,name)
                                       for name in ('environment','parse_chain','host_load_snapshot','compare_rows')},
        'note': 'The matrix ran the captured campaign helpers. Later reruns use the frozen protocol explicitly.'}
    if not result['all_profile_counts_expected'] or not result['all_profile_values_bit_identical']:
        raise RuntimeError('Invalid numerical/count profile evidence')
    if not all(result['protocol_proof']['used_helpers_ast_identical'].values()):
        raise RuntimeError('Captured protocol functions differ')
    (HERE/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'nonbridge_us': result['nonbridge_sum_of_stage_medians'],
                      'paired_ratio': control['statistics']['geometric_mean_ratio'],
                      'all_profile_values_bit_identical': result['all_profile_values_bit_identical']}, indent=2))


if __name__ == '__main__':
    main()
