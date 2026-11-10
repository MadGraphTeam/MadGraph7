#!/usr/bin/env python3
"""Replay closed parameter-integral evidence; never run a quadrature/reducer."""
from decimal import Decimal
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p/'madgraph').is_dir() and (p/'tools').is_dir())
PARSER = HERE.parent/'rank6-feynman-parameter-analysis-proposal-1/compare.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw(name):
    return gzip.decompress((HERE/'data'/name).read_bytes())


def derived():
    spec = importlib.util.spec_from_file_location('rank6_fp_qualification_parser', PARSER)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    data = {name: json.loads(raw(name+'.json.gz')) for name in ('float-result', 'decimal8-result', 'float18-result')}
    snapshot = parser.retained_snapshot()
    # Every actual result is independently admitted and checked before this
    # analysis concatenates their already-closed order records.
    for result in data.values():
        parser.analyze(result, snapshot)
    aggregate = dict(data['float-result'])
    aggregate['orders'] = [*data['float-result']['orders'], *data['float18-result']['orders']]
    analysis = parser.analyze(aggregate, snapshot)
    analysis['aggregation_scope'] = 'offline concatenation of closed binary64 orders; no new numerical integration'
    analysis['raw_result_sha256'] = {name: hashlib.sha256(raw(name+'.json.gz')).hexdigest() for name in data}
    analysis['arithmetic_comparison'] = parser.compare_arithmetic(data['float-result'], data['decimal8-result'])
    analysis['prototype_index_sha256'] = parser.SOURCE_INDEX_SHA256
    analysis['retained_rows_sha256'] = parser.sha(parser.ROWS)
    endpoint = analysis['orders'][-1]
    assert endpoint['n'] == 18
    all_parts = [part for item in endpoint['components'].values() for part in item['parts'].values()]
    selected_parts = [part for name in ('index55', 'index111', 'index161')
                      for part in endpoint['components'][name]['parts'].values()]
    arithmetic_parts = [part for item in analysis['arithmetic_comparison']['orders'][0]['components'].values()
                        for part in item.values()]
    receipts = {name: json.loads((HERE/'data'/name).read_text())
                for name in ('execution.json', 'decimal8-execution.json', 'float18-execution.json')}
    for name, receipt in receipts.items():
        assert receipt['exit_code'] == 0 and receipt['source_index_sha256'] == parser.SOURCE_INDEX_SHA256
        assert receipt['cpu'] == 10 and receipt['cpu_reservation'] is False
        assert receipt.get('native_reducer_calls', receipt.get('numerical_reducer_calls')) == 0
        assert receipt['performance_qualified'] is False
        for basename, expected in receipt['files'].items():
            payload = raw(basename+'.gz') if basename.endswith('.json') else (HERE/'data'/basename).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == expected
    summary = {'schema': 1, 'scope': 'seven Cartesian finite components of synthetic N6/R6 fixture108 only',
               'status': 'independent numerical cross-check corroborates sampled TIRust components',
               'sampled_components': 7, 'sampled_real_imaginary_parts': 14,
               'orders_binary64': [row['n'] for row in analysis['orders']], 'order_decimal80': 8,
               'endpoint': 18,
               'tirust_maximum_observed_symmetric_residual': str(max(Decimal(v['TIRust_QP']['symmetric_residual']) for v in all_parts)),
               'tirust_all_observed_parts_agree_at_3e6': all(v['TIRust_QP']['estimate_agrees_at_3e6'] for v in all_parts),
               'tirust_all_observed_parts_agree_at_1e8': all(v['TIRust_QP']['estimate_agrees_at_1e8'] for v in all_parts),
               'lowrank_convention_checks': endpoint['lowrank_convention_checks_by_estimate'],
               'collier_failed_disputed_parts_at_3e6': sum(not v['COLLIER']['estimate_agrees_at_3e6'] for v in selected_parts),
               'disputed_parts': len(selected_parts),
               'collier_maximum_observed_symmetric_residual': str(max(Decimal(v['COLLIER']['symmetric_residual']) for v in selected_parts)),
               'tirust_maximum_analytic_error_only_upper_residual': str(max(Decimal(v['TIRust_QP']['analytic_error_only_interval']['symmetric_residual_upper']) for v in all_parts)),
               'tirust_all_parts_agree_at_3e6_with_analytic_error_only': all(v['TIRust_QP']['analytic_error_only_interval']['quadrature_error_only_agreement_at_3e6'] for v in all_parts),
               'collier_failed_disputed_parts_with_analytic_error_only': sum(v['COLLIER']['analytic_error_only_interval']['quadrature_error_only_disagreement_at_3e6'] for v in selected_parts),
               'float8_decimal80_8_maximum_symmetric_difference': str(max(Decimal(v['symmetric_residual']) for v in arithmetic_parts)),
               'retained_backend_states': analysis['retained_backend_states'],
               'actual_quadrature_elapsed_seconds': {name: r['elapsed_seconds'] for name, r in receipts.items()},
               'native_reducer_calls_in_this_experiment': 0, 'performance_qualified': False,
               'total_floating_point_error_formally_certified': False,
               'inference': 'Independent direct integration corroborates TIRust for these sampled components. The uncached COLLIER values miss the requested comparison criterion on five target parts, consistent with its retained requested-accuracy flag -1; no specific internal algorithm bug is assigned.',
               'limits': ['no qualification of all 210 tensor components or physical amplitudes',
                          'analytic Gaussian tail has a rigorous bound; finite-precision node/weight/arithmetic uncertainty is separately checked empirically, not globally certified',
                          'same-order Decimal corroboration and convergence are retained observations',
                          'no COLLIER cache modification; known cached N7/R9 issue is not the cause established here',
                          'no precision timing or performance acceptance from quadrature wall time']}
    return analysis, summary


def verify_index():
    index = json.loads((HERE/'index.json').read_text())
    for record in index['files']:
        path = HERE/record['path']
        assert path.resolve().is_relative_to(HERE.resolve())
        data = path.read_bytes()
        assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], str(path)
        if record.get('uncompressed_sha256'):
            data = gzip.decompress(data)
            assert len(data) == record['uncompressed_bytes']
            assert hashlib.sha256(data).hexdigest() == record['uncompressed_sha256']
    for record in index['inputs']:
        path = REPO/record['path']
        assert path.resolve().is_relative_to(REPO.resolve())
        assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], str(path)
    return index


def main():
    index = verify_index()
    analysis, summary = derived()
    assert json.loads(json.dumps(analysis)) == json.loads((HERE/'analysis.json').read_text())
    assert summary == json.loads((HERE/'summary.json').read_text())
    print(json.dumps({'passed': True, 'owned_files': len(index['files']),
                      'orders': summary['orders_binary64'], 'decimal_order': 8,
                      'native_reducer_calls': 0, 'new_physics_evaluations': 0,
                      'sampled_tirust_crosscheck': summary['tirust_all_observed_parts_agree_at_3e6'],
                      'collier_disputed_part_misses': summary['collier_failed_disputed_parts_at_3e6']}))


if __name__ == '__main__':
    main()
