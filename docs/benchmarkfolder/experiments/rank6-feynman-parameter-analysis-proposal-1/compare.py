#!/usr/bin/env python3
"""Offline, additive comparison of explicitly closed parameter estimates.

No quadrature/reducer is invoked. The root must confirm process termination
before passing --admit-closed-result; this script will not read a result file
without that explicit gate. Neither convergence nor agreement assigns truth.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent/'generated-rank6-poles-proposal/attempt1/common-input-proposal'
ROWS = COMMON/'finite-audit-1/rows.json.gz'
SOURCE_INDEX = HERE.parent/'rank6-feynman-parameter-proposal-1/source-index.json'
SOURCE_INDEX_SHA256 = '03ffcd9f8b88855b59d78387cde255460a36d35cca251fce093741a9c0065db3'
DERIVATION = HERE.parent/'rank6-feynman-parameter-proposal-1/derivation.json'
TARGETS = {
    'scalar': (0, (0, 0, 0, 0)), 'odd_time': (1, (1, 0, 0, 0)),
    'time_time': (5, (2, 0, 0, 0)), 'space_space': (7, (0, 2, 0, 0)),
    'index55': (55, (1, 1, 1, 1)), 'index111': (111, (1, 1, 1, 2)),
    'index161': (161, (3, 1, 1, 1)),
}
CONTROLS = ('scalar', 'odd_time', 'time_time', 'space_space')
DERIVATION_NAMES = {'scalar': 'scalar', 'odd_time': 'q0', 'time_time': 'q0_squared',
                    'space_space': 'q1_squared', 'index55': 'target_55',
                    'index111': 'target_111', 'index161': 'target_161'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary128(words):
    if len(words) != 2 or any(not isinstance(x, str) or re.fullmatch(r'[0-9a-fA-F]{16}', x) is None for x in words):
        raise ValueError('expected two exact hexadecimal binary128 words')
    bits = int(words[0], 16) | (int(words[1], 16) << 64)
    sign, exponent = bits >> 127, (bits >> 112) & 32767
    fraction = bits & ((1 << 112)-1)
    if exponent == 32767:
        raise ValueError('nonfinite retained coefficient')
    with localcontext() as context:
        # Every finite binary128 is an integer times a power of two. The
        # smallest exponent needs at most 11,564 decimal significand digits;
        # retain it exactly rather than rounding subnormals at a fixed180.
        context.prec = 20000
        significand = fraction + ((1 << 112) if exponent else 0)
        shift = (exponent or 1)-16383-112
        magnitude = Decimal(significand << shift) if shift >= 0 else Decimal(significand * 5**(-shift)).scaleb(shift)
        return magnitude.copy_negate() if sign else magnitude


def estimate_number(value, engine):
    if engine == 'float':
        if type(value) is not float:
            raise ValueError('binary64 estimate must round-trip as a JSON float')
        result = Decimal.from_float(value)
    elif engine == 'decimal':
        if not isinstance(value, str):
            raise ValueError('Decimal estimate must preserve its original decimal string')
        result = Decimal(value)
    else:
        raise ValueError('unsupported arithmetic engine')
    if not result.is_finite():
        raise ValueError('nonfinite parameter estimate')
    return result


def symmetric(x, y):
    return abs(x-y)/abs(x+y) if x+y else Decimal(0) if x == y else Decimal('Infinity')


def distance(x, y):
    residual = symmetric(x, y)
    return {'absolute_difference': str(abs(x-y)),
            'symmetric_residual': str(residual),
            'relative_to_reference': str(abs(x-y)/abs(y)) if y else None,
            'estimate_agrees_at_3e6': residual <= Decimal('3e-6'),
            'estimate_agrees_at_1e8': residual <= Decimal('1e-8')}


def analytic_interval(x, y, error):
    """Triangle-inequality enclosure excluding arithmetic error explicitly."""
    if not error.is_finite() or error < 0:
        raise ValueError('analytic error must be finite and nonnegative')
    numerator, denominator, e = abs(Fraction(x)-Fraction(y)), abs(Fraction(x)+Fraction(y)), Fraction(error)
    def display(fraction, rounding):
        with localcontext() as context:
            context.prec = 180
            context.rounding = rounding
            return Decimal(fraction.numerator)/Decimal(fraction.denominator)
    if not denominator+e:
        lower = upper = Decimal(0) if not numerator else Decimal('Infinity')
    else:
        lower = display(max(Fraction(0), numerator-e)/(denominator+e), ROUND_FLOOR)
        upper = display((numerator+e)/(denominator-e), ROUND_CEILING) if denominator > e else Decimal('Infinity')
    return {'symmetric_residual_lower': str(lower), 'symmetric_residual_upper': str(upper),
            'quadrature_error_only_agreement_at_3e6': upper <= Decimal('3e-6'),
            'quadrature_error_only_disagreement_at_3e6': lower > Decimal('3e-6'),
            'excludes_finite_precision_node_weight_arithmetic_error': True}


def retained_snapshot():
    if sha(SOURCE_INDEX) != SOURCE_INDEX_SHA256:
        raise ValueError('reviewed parameter prototype index changed')
    index = json.loads(SOURCE_INDEX.read_text())
    binding = next(record for record in index['inputs'] if record['path'].endswith('/finite-audit-1/rows.json.gz'))
    if sha(ROWS) != binding['sha256'] or len(ROWS.read_bytes()) != binding['bytes']:
        raise ValueError('retained rows differ from the reviewed source binding')
    derivation_binding = next(record for record in index['files'] if record['path'] == 'derivation.json')
    if sha(DERIVATION) != derivation_binding['sha256']:
        raise ValueError('analytic bounds differ from the reviewed source binding')
    snapshot = json.loads(gzip.decompress(ROWS.read_bytes()))
    by = {row['action']: row for row in snapshot['rows']}
    for action, backend in ((3007, 9), (3013, 7)):
        row = by[action]
        if not (row['fixture'] == 108 and row['n'] == row['rank'] == 6
                and row['backend'] == backend and row['finite'] == 1
                and row['mode'] == 0 and row['cache'] == 0):
            raise ValueError('unexpected retained comparison row')
    for index, powers in TARGETS.values():
        if tuple(snapshot['monomials'][index]) != powers:
            raise ValueError('MadLoop component index/powers mismatch')
    return snapshot


def analytic_bound(n, name):
    derivation = json.loads(DERIVATION.read_text())
    row = next((row for row in derivation['analytic_quadrature_bounds'] if row['nodes_per_dimension'] == n), None)
    if row is None:
        raise ValueError('quadrature order has no reviewed bound in the frozen derivation')
    return row['absolute_complex_error_bound'][DERIVATION_NAMES[name]]['exact_rational']


def analyze(result, snapshot):
    if not result['status'].startswith('completed estimates;'):
        raise ValueError('only completed closed estimate sequences may be analyzed')
    if result.get('fixture_sha256') != '180cb642d3886dab8055a70a7f4fef6586e2b2c635e998aab0066e95886f53f4':
        raise ValueError('fixture identity mismatch')
    engine = result['engine']
    if result.get('native_reducer_calls') != 0 or result.get('model_card_initialization') is not False:
        raise ValueError('parameter-only evaluation scope mismatch')
    for name, (index, powers) in TARGETS.items():
        target = result['targets'][name]
        if target['index'] != index or tuple(target['powers']) != powers:
            raise ValueError('estimate target index/powers mismatch')
    by = {row['action']: row for row in snapshot['rows']}
    output = {'schema': 1, 'status': 'descriptive estimates only; total numerical error unqualified',
              'engine': engine, 'finite_truth_assigned': False, 'native_reducer_calls': 0,
              'retained_backend_states': {'TIRust_QP': by[3007]['state'], 'COLLIER': by[3013]['state']},
              'orders': [], 'successive_order_differences': [],
              'interpretation_limits': ['analytic quadrature error excludes node/weight/arithmetic roundoff',
                                        'successive-order convergence is descriptive, not a certified total error',
                                        'COLLIER requested-accuracy miss -1 is retained; neither backend is assumed true']}
    previous = None
    seen = set()
    with localcontext() as context:
        context.prec = 180
        for order in result['orders']:
            n = order['n']
            if type(n) is not int or not 4 <= n <= 20 or n in seen or order['nodes'] != n**5:
                raise ValueError('invalid or duplicate quadrature order')
            seen.add(n)
            entry = {'n': n, 'nodes': n**5, 'components': {}}
            current = {}
            for name, (index, powers) in TARGETS.items():
                item = order['results'][name]
                if Fraction(item['bounds']['absolute_analytic_error']) != Fraction(analytic_bound(n, name)):
                    raise ValueError('analytic bound does not match the frozen derivation')
                rational_error = Fraction(item['bounds']['absolute_analytic_error'])
                with localcontext() as rounding:
                    rounding.prec = 180
                    rounding.rounding = ROUND_CEILING
                    analytic_error = Decimal(rational_error.numerator)/Decimal(rational_error.denominator)
                current[name] = {}
                component = {'index': index, 'powers': powers,
                             'analytic_error_bound_excludes_roundoff': item['bounds']['absolute_analytic_error'],
                             'parts': {}}
                for part, offset in (('real', 0), ('imaginary', 2)):
                    estimate = estimate_number(item[part], engine)
                    current[name][part] = estimate
                    comparisons = {'estimate': str(estimate)}
                    for label, action in (('TIRust_QP', 3007), ('COLLIER', 3013)):
                        value = binary128(by[action]['values'][index][0][offset:offset+2])
                        comparisons[label] = {'retained_value': str(value), **distance(estimate, value),
                                              'analytic_error_only_interval': analytic_interval(estimate, value, analytic_error)}
                    component['parts'][part] = comparisons
                entry['components'][name] = component
            entry['lowrank_convention_checks_by_estimate'] = {
                label: {'all_four_controls_agree_at_3e6': all(
                    entry['components'][name]['parts'][part][label]['estimate_agrees_at_3e6']
                    for name in CONTROLS for part in ('real', 'imaginary')),
                        'all_four_controls_agree_at_1e8': all(
                    entry['components'][name]['parts'][part][label]['estimate_agrees_at_1e8']
                    for name in CONTROLS for part in ('real', 'imaginary'))}
                for label in ('TIRust_QP', 'COLLIER')}
            output['orders'].append(entry)
            if previous is not None:
                output['successive_order_differences'].append({
                    'previous_n': previous[0], 'current_n': n,
                    'components': {name: {part: distance(current[name][part], previous[1][name][part])
                                          for part in ('real', 'imaginary')} for name in TARGETS}})
            previous = (n, current)
    if not output['orders']:
        raise ValueError('no completed quadrature order')
    return output


def compare_arithmetic(left, right):
    """Same-order arithmetic agreement; not a total integration certificate."""
    if {left['engine'], right['engine']} != {'float', 'decimal'}:
        raise ValueError('arithmetic comparison requires binary64 and Decimal results')
    if any(not item['status'].startswith('completed estimates;') for item in (left, right)):
        raise ValueError('both arithmetic results must be closed/completed')
    if left['fixture_sha256'] != right['fixture_sha256']:
        raise ValueError('different fixture inputs')
    output = {'status': 'observed same-order arithmetic agreement; no rigorous roundoff bound', 'orders': []}
    by = {row['n']: row for row in right['orders']}
    with localcontext() as context:
        context.prec = 180
        for row in left['orders']:
            if row['n'] not in by:
                continue
            other = by[row['n']]
            output['orders'].append({'n': row['n'], 'left_engine': left['engine'], 'right_engine': right['engine'],
                'components': {name: {part: {
                    'left_value': str(estimate_number(row['results'][name][part], left['engine'])),
                    'right_value': str(estimate_number(other['results'][name][part], right['engine'])),
                    **distance(estimate_number(row['results'][name][part], left['engine']),
                               estimate_number(other['results'][name][part], right['engine']))}
                    for part in ('real', 'imaginary')} for name in TARGETS}})
    if not output['orders']:
        raise ValueError('no common arithmetic-check order')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paired-result', type=Path)
    parser.add_argument('--admit-closed-result', action='store_true')
    args = parser.parse_args(argv)
    if not args.admit_closed_result:
        parser.error('root confirmation of process termination is required before reading results')
    if args.output.exists():
        parser.error('output must be a new additive file')
    # Deliberately read the result only after the explicit closed-result gate.
    result = json.loads(args.result.read_text())
    analysis = analyze(result, retained_snapshot())
    analysis['result_sha256'] = sha(args.result)
    analysis['retained_rows_sha256'] = sha(ROWS)
    analysis['prototype_index_sha256'] = SOURCE_INDEX_SHA256
    if args.paired_result:
        paired = json.loads(args.paired_result.read_text())
        analyze(paired, retained_snapshot())
        analysis['paired_result_sha256'] = sha(args.paired_result)
        analysis['arithmetic_comparison'] = compare_arithmetic(result, paired)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2)+'\n')
    print(json.dumps({'status': analysis['status'], 'orders': len(analysis['orders']),
                      'output': str(args.output), 'sha256': sha(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
