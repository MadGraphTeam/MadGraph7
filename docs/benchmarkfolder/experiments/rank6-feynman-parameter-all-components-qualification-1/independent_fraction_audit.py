#!/usr/bin/env python3
"""Reproduce the independent exact-Fraction bookkeeping audit, without kernels.

Only explicitly closed result data and historical coefficient words are read.
No numerical integral, Gaussian rule, NumPy or reducer is evaluated. Gzip input
is accepted; result_sha256 always identifies the decompressed original JSON.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
from functools import lru_cache
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE.parent/'generated-rank6-poles-proposal/attempt1/common-input-proposal/finite-audit-1/rows.json.gz'
ROWS_SHA256 = '729abf58745f554684eae4e2020c52280bcb3658becf80f2cec85f3540a3f74a'
FIXTURE_SHA256 = '180cb642d3886dab8055a70a7f4fef6586e2b2c635e998aab0066e95886f53f4'


def decode(words):
    bits = int(words[0], 16)+(int(words[1], 16) << 64)
    exponent = (bits >> 112) & 32767
    mantissa = (bits & ((1 << 112)-1))+(1 << 112 if exponent else 0)
    if exponent == 32767:
        raise ValueError('nonfinite historical coefficient')
    return (-1 if bits >> 127 else 1)*F(mantissa)*F(2)**((exponent or 1)-16383-112)


def display(value):
    if value is None:
        return 'Infinity'
    with localcontext() as context:
        context.prec = 80
        return str(Decimal(value.numerator)/Decimal(value.denominator))


@lru_cache(maxsize=630)
def independent_bound(n, powers):
    """Independent Wick-loop enumeration and exact binomial-series tail."""
    k_taylor = n-4
    rho, center, error = F(118031, 1000000), F(1161, 4), F(0)
    for loops in itertools.product(*(range(0, p+1, 2) for p in powers)):
        pairs = sum(loops)//2
        inverse_power = 4-pairs
        coefficient = F((-1)**(loops[0]//2)*math.factorial(3-pairs), 2**pairs)
        for power, loop_power in zip(powers, loops):
            coefficient *= math.comb(power, loop_power)*math.prod(range(1, loop_power, 2))
        residual = [p-l for p, l in zip(powers, loops)]
        numerator_bound = math.prod(m**p for m, p in zip((3, 4, 5, 5), residual))
        omitted = k_taylor+1
        first = F(math.comb(inverse_power+omitted-1, omitted))*rho**omitted
        ratio = rho*F(inverse_power+omitted, omitted+1)
        error += 2*F(1, 120)*abs(coefficient)*numerator_bound*center**(-inverse_power)*first/(1-ratio)
    return error


def analyze(raw, rows_raw):
    if hashlib.sha256(rows_raw).hexdigest() != ROWS_SHA256:
        raise ValueError('historical rows identity mismatch')
    result = json.loads(raw)
    if not result['status'].startswith('completed estimates;'):
        raise ValueError('this reproduction audit requires the closed completed run')
    if result['fixture_sha256'] != FIXTURE_SHA256:
        raise ValueError('fixture identity mismatch')
    snapshot = json.loads(gzip.decompress(rows_raw))
    by = {r['action']: r for r in snapshot['rows']}
    monomials = snapshot['monomials']
    if result['monomials'] != monomials:
        raise ValueError('full tensor mapping mismatch')
    references = {label: [[decode(z[0][offset:offset+2]) for offset in (0, 2)]
                          for z in by[action]['values']]
                  for label, action in (('TIRust_QP', 3007), ('COLLIER', 3013))}
    thresholds = {'3e-6': F(3, 10**6), '1e-8': F(1, 10**8)}
    output = {'schema': 1,
              'scope': 'independent exact-Fraction offline audit of explicitly closed all-component result',
              'result_sha256': hashlib.sha256(raw).hexdigest(),
              'retained_rows_sha256': hashlib.sha256(rows_raw).hexdigest(),
              'execution_status': result['status'], 'orders': [],
              'quadrature_calls': 0, 'native_reducer_calls': 0,
              'all_recomputed_bounds_match': True}
    for order in result['orders']:
        n = order['n']
        if n not in (8, 16, 18) or order['nodes'] != n**5 or len(order['tensor_values']) != 210:
            raise ValueError('unexpected order or incomplete tensor array')
        entry = {'n': n, 'backends': {}}
        for label in references:
            parts = []
            for i, (powers, tensor) in enumerate(zip(monomials, order['tensor_values'])):
                if tensor['index'] != i or tensor['powers'] != powers:
                    raise ValueError('component mapping mismatch')
                error = independent_bound(n, tuple(powers))
                if error != F(tensor['analytic_error_bound_excludes_roundoff']['absolute_analytic_error']):
                    raise ValueError('independent analytic-bound mismatch')
                for name, axis in (('real', 0), ('imaginary', 1)):
                    estimate, reference = F(tensor[name]), references[label][i][axis]
                    numerator, denominator = abs(estimate-reference), abs(estimate+reference)
                    residual = numerator/denominator if denominator else F(0) if not numerator else None
                    lower = max(F(0), numerator-error)/(denominator+error)
                    upper = (numerator+error)/(denominator-error) if denominator > error else None
                    parts.append({'i': i, 'part': name, 'rank': sum(powers), 'absdiff': numerator,
                                  'r': residual, 'lo': lower, 'hi': upper,
                                  'reference_zero': not reference})
            worst = max(parts, key=lambda z: (z['r'] is None, z['r'] or 0))
            counts = {}
            for name, threshold in thresholds.items():
                point_agrees = lambda z: z['r'] is not None and z['r'] <= threshold
                agrees = lambda z: z['hi'] is not None and z['hi'] <= threshold
                disagrees = lambda z: z['lo'] > threshold
                counts[name] = {
                    'point_agree': sum(point_agrees(z) for z in parts),
                    'point_disagree': sum(not point_agrees(z) for z in parts),
                    'analytic_tail_only_agree': sum(agrees(z) for z in parts),
                    'analytic_tail_only_disagree': sum(disagrees(z) for z in parts),
                    'analytic_tail_only_inconclusive': sum(not agrees(z) and not disagrees(z) for z in parts),
                    'exact_point_threshold_boundary': sum(z['r'] == threshold for z in parts)}
            entry['backends'][label] = {
                'parts': len(parts), 'reference_zero_parts': sum(z['reference_zero'] for z in parts),
                'maximum_symmetric_residual': display(worst['r']),
                'worst_index': worst['i'], 'worst_part': worst['part'],
                'maximum_absolute_difference': display(max(z['absdiff'] for z in parts)),
                'counts': counts,
                'rank_point_counts': {str(rank): {name: sum(z['r'] is not None and z['r'] <= threshold
                                                          for z in parts if z['rank'] == rank)
                                                  for name, threshold in thresholds.items()}
                                      for rank in range(7)}}
        output['orders'].append(entry)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--rows', type=Path, default=ROWS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--admit-closed-result', action='store_true')
    args = parser.parse_args(argv)
    if not args.admit_closed_result:
        parser.error('root process-termination confirmation is required before reading a result')
    if args.output.exists():
        parser.error('output must be a new additive file')
    raw = args.result.read_bytes()
    if raw.startswith(b'\x1f\x8b'):
        raw = gzip.decompress(raw)
    output = analyze(raw, args.rows.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
