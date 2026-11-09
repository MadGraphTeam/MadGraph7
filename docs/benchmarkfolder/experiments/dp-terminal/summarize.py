#!/usr/bin/env python3
"""Summarize the retained isolated experiments without running native code."""
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coefficients(rows):
    return b''.join(struct.pack('!d', value) for row in rows for value in row['coefficients'])


def amplitude(name):
    path = HERE / name / 'results.json'
    result = json.loads(path.read_text())
    groups = {}
    for run in result['runs']:
        assert run['parity_passed'] and run['calibration_parity_passed']
        assert run['expected_backend_usage_passed']
        groups.setdefault(run['repetition'], {})[run['variant']] = run
    for pair in groups.values():
        assert set(pair) == {'baseline', 'original', 'terminal'}
        for phase in ('observation', 'calibration'):
            assert coefficients(pair['original'][phase]['rows']) == coefficients(pair['terminal'][phase]['rows'])
    record = {
        'results': str(path.relative_to(HERE)), 'results_sha256': digest(path),
        'pairs': result['pairs'], 'loops_per_pass': result['loops'],
        'points_per_pass': 100, 'all_calibration_and_observed_coefficients_bit_identical': True,
        'all_expected_backend_precision_counts_pass': True,
        'maximum_baseline_relative_difference': max(run['maximum_relative_error'] for run in result['runs']),
        'median_microseconds_per_amplitude': result['median_microseconds_per_amplitude'],
        'paired_statistics': result['paired_statistics'],
        'candidate_library': result['candidate_library'],
        'performance_eligible': False,
    }
    traces = [run['admission_trace'] for run in result['runs'] if 'admission_trace' in run]
    if traces:
        record['admission_traces'] = traces
    return record


def main():
    summaries = {}
    for variant, directory in [('elemental', HERE), ('packed', HERE / 'packed-scan')]:
        guards = json.loads((directory / 'guard-results.json').read_text())
        micro = json.loads((directory / 'microbenchmark-results.json').read_text())
        assert guards['passed'] and all(row['passed'] for row in guards['results'])
        summaries[variant] = {
            'guard_comparisons': guards['guard_comparisons'],
            'paired_trap_probes': guards['trap_comparison_pairs'],
            'synthetic_kernel_ratios': {row['name']: row['median_terminal_over_dense_ratio']
                                       for row in micro['results']},
            'parity': amplitude('amplitude-' + variant + '-parity-1'),
            'whole_amplitude_diagnostic': amplitude('amplitude-' + variant + '-paired-1'),
        }
    record = {
        'scope': 'Isolated DP terminal rank1x1 generated-amplitude experiment; no production edits',
        'performance_eligible': False,
        'compiler_scope': 'Original polynomial.f special recipe uses -O3; caller recipe has no -O flag. '
                          'No recipe or compiler environment changed. The isolated -O0 kernel result '
                          'is a separate sensitivity diagnostic.',
        'comparison_identity': 'Frozen c91 generated baseline8 and candidate9; original9 and terminal9 '
                               'use the same pinned split-workspace DSO. Existing QP code is unchanged.',
        'variants': summaries,
        'admission': amplitude('amplitude-elemental-trace-parity-1'),
        'retained_failures': [
            'Elemental synthetic kernel is slower at -O0 (median ratio 1.2996).',
            'Initial fixed-form label conversion failed compilation; its complete source/log is retained.'],
        'remaining_before_promotion': ['A general DP generator consumer/alias validator and fallback path.',
                                       'Production generator integration and qualified campaign retesting.'],
        'summarizer_sha256': digest(Path(__file__)),
    }
    (HERE / 'analysis-summary.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
