#!/usr/bin/env python3
"""Offline full-component comparisons; no numerical integration or reducer."""
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
PARSER = EXPERIMENTS/'rank6-feynman-parameter-analysis-proposal-1'
CORE = EXPERIMENTS/'rank6-feynman-parameter-proposal-1'
VECTOR = EXPERIMENTS/'rank6-feynman-parameter-all-components-proposal-1'
SELECTED = EXPERIMENTS/'rank6-feynman-parameter-qualification-1'
PINS = {PARSER/'source-index.json': '3978c18474e8b5de1b89b60a814479aef062d73ee024e92b2b683008adb11c54',
        CORE/'source-index.json': '03ffcd9f8b88855b59d78387cde255460a36d35cca251fce093741a9c0065db3',
        VECTOR/'source-index.json': '128df03a4790e25ea49f7bacee5a0a6d5339035ad5e6b73b62df7a7a2dabef27',
        SELECTED/'index.json': 'c2fe806d2996e7ee758e2a7894318d04388b640126b9f636fd23be99650b90d7'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    for path, expected in PINS.items():
        if sha(path) != expected:
            raise ValueError('changed frozen dependency '+str(path))
    for directory, filename in ((PARSER, 'compare.py'), (CORE, 'quadrature.py'), (VECTOR, 'all_components.py')):
        index = json.loads((directory/'source-index.json').read_text())
        binding = next(x for x in index['files'] if x['path'] == filename)
        if sha(directory/filename) != binding['sha256']:
            raise ValueError('changed frozen analysis source '+filename)
    parser = load_module(PARSER/'compare.py', 'all210_exact_parser')
    core = load_module(CORE/'quadrature.py', 'all210_exact_bounds')
    vector = load_module(VECTOR/'all_components.py', 'all210_layout')
    snapshot = parser.retained_snapshot()
    return parser, core, vector, snapshot


def zero_diagnostic(estimate, reference, rank_scale):
    difference = abs(estimate-reference)
    return {'retained_reference_exactly_zero': reference.is_zero(),
            'estimate_exactly_zero': estimate.is_zero(),
            'absolute_difference': str(difference), 'rank_reference_scale': str(rank_scale),
            'rank_scaled_absolute_difference': str(difference/rank_scale) if rank_scale else None,
            'role': 'diagnostic only; no floor, clipping or acceptance-criterion change'}


def summarize_parts(parts, label):
    entries = [(i, p, value[label]) for i, p, value in parts]
    worst = max(entries, key=lambda x: Decimal(x[2]['symmetric_residual']))
    result = {'parts': len(entries),
              'pass_3e6': sum(item['estimate_agrees_at_3e6'] for _, _, item in entries),
              'fail_3e6': sum(not item['estimate_agrees_at_3e6'] for _, _, item in entries),
              'pass_1e8': sum(item['estimate_agrees_at_1e8'] for _, _, item in entries),
              'fail_1e8': sum(not item['estimate_agrees_at_1e8'] for _, _, item in entries),
              'maximum_symmetric_residual': worst[2]['symmetric_residual'],
              'worst_index': worst[0], 'worst_part': worst[1],
              'maximum_absolute_difference': str(max(Decimal(item['absolute_difference']) for _, _, item in entries)),
              'maximum_analytic_error_only_upper_residual': str(max(Decimal(item['analytic_error_only_interval']['symmetric_residual_upper']) for _, _, item in entries)),
              'exact_zero_reference_parts': sum(Decimal(item['retained_value']).is_zero() for _, _, item in entries),
              'analytic_error_only_agree_3e6': sum(item['analytic_error_only_interval']['quadrature_error_only_agreement_at_3e6'] for _, _, item in entries),
              'analytic_error_only_disagree_3e6': sum(item['analytic_error_only_interval']['quadrature_error_only_disagreement_at_3e6'] for _, _, item in entries)}
    result['analytic_error_only_inconclusive_3e6'] = len(entries)-result['analytic_error_only_agree_3e6']-result['analytic_error_only_disagree_3e6']
    result['analytic_error_only_agree_1e8'] = sum(Decimal(item['analytic_error_only_interval']['symmetric_residual_upper']) <= Decimal('1e-8') for _, _, item in entries)
    result['analytic_error_only_disagree_1e8'] = sum(Decimal(item['analytic_error_only_interval']['symmetric_residual_lower']) > Decimal('1e-8') for _, _, item in entries)
    result['analytic_error_only_inconclusive_1e8'] = len(entries)-result['analytic_error_only_agree_1e8']-result['analytic_error_only_disagree_1e8']
    return result


def saved_join(order, tensor_values):
    n = order['n']
    if n not in (8, 18):
        return {'status': 'no historical scalar quadrature at this order; no substitution'}
    name = 'float18-result.json.gz' if n == 18 else 'float-result.json.gz'
    path = SELECTED/'data'/name
    index = json.loads((SELECTED/'index.json').read_text())
    binding = next(x for x in index['files'] if x['path'] == 'data/'+name)
    if sha(path) != binding['sha256']:
        raise ValueError('changed historical scalar quadrature')
    source = json.loads(gzip.decompress(path.read_bytes()))
    row = next(x for x in source['orders'] if x['n'] == n)
    parts = {}
    with localcontext() as context:
        context.prec = 100
        for label, metadata in source['targets'].items():
            i = metadata['index']
            if tensor_values[i]['powers'] != metadata['powers']:
                raise ValueError('saved scalar component mapping changed')
            for part in ('real', 'imaginary'):
                observed = Decimal(tensor_values[i][part]); reference = Decimal.from_float(row['results'][label][part])
                residual = abs(observed-reference)/abs(observed+reference) if observed+reference else Decimal(0) if observed == reference else Decimal('Infinity')
                parts[label+'_'+part] = {'new_extended_value': str(observed), 'saved_binary64_value': str(reference),
                                        'absolute_difference': str(abs(observed-reference)), 'symmetric_residual': str(residual),
                                        'agrees_at_1e12': residual <= Decimal('1e-12')}
    retained = order['saved_selected_comparison'].get('parts')
    if retained is not None and retained != parts:
        raise ValueError('saved scalar comparison differs from the actual retained worker output')
    return {'status': 'all same-order scalar/vector comparisons retained' if retained is not None else 'offline scalar comparison recomputed; original worker comparison unavailable',
            'worker_comparison_available': retained is not None,
            'original_worker_status': order['saved_selected_comparison'].get('status'), 'parts': len(parts),
            'all_agree_at_1e12': all(x['agrees_at_1e12'] for x in parts.values()),
            'maximum_symmetric_residual': str(max(Decimal(x['symmetric_residual']) for x in parts.values()))}


def analyze(result):
    parser, core, vector, snapshot = dependencies()
    by = {row['action']: row for row in snapshot['rows']}
    monomials = [tuple(x) for x in snapshot['monomials']]
    if result['fixture_sha256'] != core.FIXTURE_SHA256 or result['monomials'] != snapshot['monomials']:
        raise ValueError('fixture identity or full component mapping mismatch')
    if result['components'] != 210 or result['primitive_moments'] != 296 or result['native_reducer_calls'] != 0:
        raise ValueError('full-component diagnostic scope mismatch')
    layout = vector.plan(core, monomials)
    refs = {label: [[parser.binary128(row['values'][i][0][offset:offset+2]) for offset in (0, 2)]
                    for i in range(210)] for label, row in (('TIRust_QP', by[3007]), ('COLLIER', by[3013]))}
    scales = {rank: max(value.copy_abs() for label in refs for powers, pair in zip(monomials, refs[label]) if sum(powers) == rank for value in pair)
              for rank in range(7)}
    analysis = {'schema': 1, 'scope': 'all 210 Cartesian finite components of synthetic N6/rank6 fixture108',
                'execution_status': result['status'], 'failure': result.get('failure'),
                'attempted_order': result.get('attempted_order'),
                'unavailable_proposed_orders': [n for n in (8, 16, 18) if n not in [row['n'] for row in result['orders']]],
                'precision': result['extended_dtype'], 'numpy_version': result['numpy_version'],
                'total_floating_point_error_formally_certified': False,
                'retained_backend_states': {'TIRust_QP': by[3007]['state'], 'COLLIER': by[3013]['state']},
                'orders': [], 'successive_order_differences': [],
                'zero_handling': 'strict symmetric criterion unchanged; exact-zero and rank-scaled absolute diagnostics are reported separately'}
    previous = None
    with localcontext() as context:
        context.prec = 180
        for order in result['orders']:
            n = order['n']
            if n not in (8, 16, 18) or order['nodes'] != n**5 or (previous and n <= previous[0]):
                raise ValueError('unreviewed, repeated or unordered quadrature order')
            tensors, primitives = order['tensor_values'], order['primitive_values']
            if len(tensors) != 210 or len(primitives) != 296:
                raise ValueError('incomplete raw tensor or primitive array')
            for (power, powers), primitive in zip(layout['primitives'], primitives):
                if primitive['inverse_F_power'] != power or tuple(primitive['powers']) != powers:
                    raise ValueError('primitive mapping mismatch')
                if any(not Decimal(primitive[part]).is_finite() for part in ('real', 'imaginary')):
                    raise ValueError('nonfinite retained primitive')
            entry = {'n': n, 'nodes': n**5, 'components': [], 'saved_scalar_control': saved_join(order, tensors)}
            flat, current = [], []
            for i, (powers, tensor) in enumerate(zip(monomials, tensors)):
                if tensor['index'] != i or tuple(tensor['powers']) != powers:
                    raise ValueError('tensor component mapping mismatch')
                bounds = core.exact_bounds(n, powers)
                if bounds != tensor['analytic_error_bound_excludes_roundoff']:
                    raise ValueError('analytic quadrature bound changed')
                e = Fraction(bounds['absolute_analytic_error'])
                with localcontext() as rounding:
                    rounding.prec = 180; rounding.rounding = ROUND_CEILING
                    error = Decimal(e.numerator)/Decimal(e.denominator)
                component = {'index': i, 'rank': sum(powers), 'powers': powers,
                             'analytic_error_bound_excludes_roundoff': bounds, 'parts': {}}
                current_pair = []
                for part, offset in (('real', 0), ('imaginary', 1)):
                    estimate = Decimal(tensor[part])
                    if not estimate.is_finite():
                        raise ValueError('nonfinite retained tensor value')
                    current_pair.append(estimate)
                    item = {'estimate': str(estimate)}
                    for label in refs:
                        reference = refs[label][i][offset]
                        item[label] = {'retained_value': str(reference), **parser.distance(estimate, reference),
                                       'analytic_error_only_interval': parser.analytic_interval(estimate, reference, error),
                                       'absolute_scale_diagnostic': zero_diagnostic(estimate, reference, scales[sum(powers)])}
                    component['parts'][part] = item
                    flat.append((i, part, item))
                current.append(current_pair)
                entry['components'].append(component)
            entry['summary'] = {label: summarize_parts(flat, label) for label in refs}
            entry['rank_summary'] = {str(rank): {label: summarize_parts([x for x in flat if sum(monomials[x[0]]) == rank], label)
                                                        for label in refs} for rank in range(7)}
            analysis['orders'].append(entry)
            if previous:
                changes = [{'index': i, 'powers': powers,
                            'real': parser.distance(pair[0], previous[1][i][0]),
                            'imaginary': parser.distance(pair[1], previous[1][i][1])}
                           for i, (powers, pair) in enumerate(zip(monomials, current))]
                analysis['successive_order_differences'].append({'previous_n': previous[0], 'current_n': n, 'components': changes,
                    'maximum_symmetric_difference': str(max(Decimal(item[part]['symmetric_residual']) for item in changes for part in ('real', 'imaginary')))})
            previous = (n, current)
    return analysis


def summary(analysis):
    return {'schema': 1, 'scope': analysis['scope'], 'execution_status': analysis['execution_status'],
            'failure': analysis['failure'], 'precision': analysis['precision'], 'numpy_version': analysis['numpy_version'],
            'attempted_order': analysis['attempted_order'], 'unavailable_proposed_orders': analysis['unavailable_proposed_orders'],
            'orders': [{'n': row['n'], 'nodes': row['nodes'], 'summary': row['summary'], 'rank_summary': row['rank_summary'],
                        'saved_scalar_control': row['saved_scalar_control']} for row in analysis['orders']],
            'successive_order_maximum_symmetric_differences': [{k: value for k, value in row.items() if k != 'components'} for row in analysis['successive_order_differences']],
            'retained_backend_states': analysis['retained_backend_states'],
            'native_reducer_calls_in_experiment': 0, 'performance_qualified': False,
            'total_floating_point_error_formally_certified': False,
            'limits': ['same synthetic fixture only; no physical amplitude, CMS stress or stability/rescue qualification',
                       'analytic Gaussian bounds exclude finite-precision node/weight/arithmetic error',
                       'all original native answers, refusals and flags remain unchanged',
                       'strict symmetric criteria are unchanged; absolute-scale diagnostics do not change acceptance']}
