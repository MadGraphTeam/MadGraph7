#!/usr/bin/env python3
"""Independent source-only proposal for smooth six-point tensor quadrature.

No MadLoop/reducer module is imported and no shared library is loaded.  The
``proposal`` command performs exact input/analytic-bound checks only.  Actual
fixture evaluation requires the explicit ``run --admit-physics`` command.
Numerical roundoff remains a separate qualification item: this prototype does
not present an observed order difference as a rigorous total error bound.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / 'generated-rank6-poles-proposal/attempt1/common-input-proposal'
FIXTURE = COMMON / 'fixtures.json'
FIXTURE_SHA256 = '180cb642d3886dab8055a70a7f4fef6586e2b2c635e998aab0066e95886f53f4'
TARGETS = {
    'scalar': (0, 0, 0, 0), 'odd_time': (1, 0, 0, 0),
    'time_time': (2, 0, 0, 0), 'space_space': (0, 2, 0, 0),
    'index55': (1, 1, 1, 1), 'index111': (1, 1, 1, 2),
    'index161': (3, 1, 1, 1),
}
INDICES = dict(zip(TARGETS, (0, 1, 5, 7, 55, 111, 161)))
_RULES = {}


def check_deadline(deadline):
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError('explicit aggregate wall-time cap reached')


def trim(p):
    while len(p) > 1 and not p[-1]:
        p.pop()
    return p


def derivative(p):
    return trim([Fraction(i) * p[i] for i in range(1, len(p))] or [Fraction(0)])


def poly_at(p, x):
    value = Fraction(0)
    for coefficient in reversed(p):
        value = value * x + coefficient
    return value


def remainder(a, b, deadline=None):
    a = list(a)
    while len(a) >= len(b) and any(a):
        check_deadline(deadline)
        shift = len(a) - len(b)
        factor = a[-1] / b[-1]
        for j, coefficient in enumerate(b):
            a[shift + j] -= factor * coefficient
        trim(a)
    return a


def sturm(p, deadline=None):
    sequence = [list(p), derivative(p)]
    while any(sequence[-1]):
        check_deadline(deadline)
        r = [-x for x in remainder(sequence[-2], sequence[-1], deadline)]
        if not any(r):
            break
        # A positive scale limits intermediate numerator growth and preserves
        # signs and the Sturm variation count.
        r = [x / abs(r[-1]) for x in r]
        sequence.append(r)
    return sequence


def variations(sequence, x):
    signs = [1 if value > 0 else -1 for p in sequence if (value := poly_at(p, x))]
    return sum(a != b for a, b in zip(signs, signs[1:]))


def jacobi_polynomial(n, alpha):
    """Exact P_n^(alpha,0)(2*t-1), ascending rational coefficients."""
    result = [Fraction(0)] * (n + 1)
    for k in range(n + 1):
        scale = math.comb(n + alpha, k) * math.comb(n, n-k)
        for j in range(n-k + 1):
            result[k+j] += scale * math.comb(n-k, j) * (-1) ** (n-k-j)
    return trim(result)


def isolate_roots(p, bits=220, deadline=None):
    """Exact Sturm counts certify disjoint dyadic/rational root intervals."""
    check_deadline(deadline)
    sequence = sturm(p, deadline)
    pending = [(Fraction(0), Fraction(1))]
    roots = []
    while pending:
        check_deadline(deadline)
        lo, hi = pending.pop()
        count = variations(sequence, lo) - variations(sequence, hi)
        if not count:
            continue
        if count == 1:
            # Opposite signs are an independent simple-root certificate.
            assert poly_at(p, lo) * poly_at(p, hi) < 0
            while hi - lo > Fraction(1, 1 << bits):
                check_deadline(deadline)
                mid = (lo + hi) / 2
                value = poly_at(p, mid)
                if not value:
                    # A rational root (e.g. the Legendre midpoint) is enclosed
                    # without placing a zero at a future Sturm endpoint.
                    radius = min((mid-lo)/2, (hi-mid)/2, Fraction(1, 1 << (bits+2)))
                    lo, hi = mid-radius, mid+radius
                    break
                if poly_at(p, lo) * value < 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo, hi))
        else:
            mid = (lo + hi) / 2
            while not poly_at(p, mid):
                mid = (lo + 2*mid) / 3
            pending.extend(((lo, mid), (mid, hi)))
    roots.sort()
    assert len(roots) == len(p)-1
    assert all(a[1] < b[0] for a, b in zip(roots, roots[1:]))
    return roots


def decimal_fraction(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def gaussian_rule(n, alpha, deadline=None):
    """Gauss-Jacobi for (1-t)^alpha on [0,1]; no SciPy/NumPy dependency.

    Exact root brackets are retained. Weights are evaluated at an 80-digit
    midpoint then rounded to binary64. Thus the rule's root isolation is
    certified; the exported floating weights are not interval certificates.
    """
    if not (1 <= n <= 24 and 0 <= alpha <= 4):
        raise ValueError('prototype rule domain is 1 <= n <= 24, 0 <= alpha <= 4')
    check_deadline(deadline)
    if (n, alpha) in _RULES:
        return _RULES[n, alpha]
    p = jacobi_polynomial(n, alpha)
    dp = derivative(p)
    records = []
    with localcontext() as context:
        context.prec = 80
        coefficients = [decimal_fraction(x) for x in dp]
        for lo, hi in isolate_roots(p, deadline=deadline):
            check_deadline(deadline)
            t = decimal_fraction((lo + hi) / 2)
            d = Decimal(0)
            for coefficient in reversed(coefficients):
                d = d * t + coefficient
            # For beta=0 the transformed Jacobi weight constant is exactly 1.
            weight = 1 / (t * (1-t) * d*d)
            records.append({'node': float(t), 'weight': float(weight),
                            'root_lo': str(lo), 'root_hi': str(hi),
                            'node_decimal': str(t), 'weight_decimal': str(weight)})
    _RULES[n, alpha] = tuple(records)
    return _RULES[n, alpha]


def fixture():
    raw = FIXTURE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError('fixture bytes differ from the reviewed analytic-bound input')
    data = json.loads(raw)
    selected = next(x for x in data['fixtures'] if x['id'] == 108)
    q = selected['shifts']
    reconstructed = [[sum((1 if a == 0 else -1) * (u[a]-v[a])**2 for a in range(4))
                      for v in q] for u in q]
    assert reconstructed == selected['invariants']
    assert selected['n'] == selected['rank'] == 6
    assert selected['masses_squared'] == [[256+8*i, -(i+1)/8] for i in range(6)]
    bernstein = [Fraction(selected['masses_squared'][i][0]+selected['masses_squared'][j][0]-selected['invariants'][i][j], 2)
                 for i in range(6) for j in range(i, 6)]
    assert min(bernstein) == 256 and max(bernstein) == Fraction(649, 2)
    assert tuple(max(abs(p[a]) for p in q) for a in range(4)) == (3, 4, 5, 5)
    return selected


def tensor_terms(powers):
    """(coefficient, residual Cartesian powers, inverse F power).

    MadLoop's odd-rank sign cancels the shift (-P) sign. Paired Cartesian
    indices contribute (-g_mumu/2) and Γ(4-k), with the number of equal-index
    pairings; there are no multinomial component weights.
    """
    if len(powers) != 4 or any(type(p) is not int or p < 0 for p in powers) or sum(powers) > 6:
        raise ValueError('tensor powers must be four nonnegative integers of total rank <=6')
    terms = []
    for pairs in itertools.product(*(range(p//2+1) for p in powers)):
        k = sum(pairs)
        if k > 3:
            raise ValueError('only UV-convergent N=6 tensors through rank six')
        factor = Fraction(math.factorial(3-k), 2**k)
        for axis, (power, pair) in enumerate(zip(powers, pairs)):
            factor *= Fraction(math.factorial(power), math.factorial(power-2*pair) * math.factorial(pair) * 2**pair)
            factor *= (-1 if axis == 0 else 1) ** pair
        terms.append((factor, tuple(p-2*k for p, k in zip(powers, pairs)), 4-k))
    return terms


@lru_cache(maxsize=256)
def numeric_terms(powers, engine):
    convert = float if engine == 'float' else lambda x: Decimal(x.numerator)/Decimal(x.denominator)
    return tuple((convert(c), r, p) for c, r, p in tensor_terms(powers))


def exact_bounds(n, powers):
    """Rigorous analytic quadrature remainder; excludes floating roundoff.

    Taylor degree K=n-ceil((rank+1)/2) guarantees degree <=2n-1 in every
    Jacobi coordinate for numerator*z^K. A fixed conservative rank-six K=n-4
    is used for all channels so the calculation and table share one proof.
    The factor two bounds integral and positive-weight quadrature remainders.
    """
    k = n-4
    if k < 0:
        return None
    center = Fraction(1161, 4)
    rho = Fraction(118031, 1000000)
    assert rho*rho > Fraction(18778, 1161**2)
    pmax = (3, 4, 5, 5)
    error = Fraction(0)
    envelope = Fraction(0)
    for coefficient, residual, power in tensor_terms(powers):
        prefactor = abs(coefficient) * math.prod(m**r for m, r in zip(pmax, residual)) / 120
        omitted = k+1
        term = math.comb(power+omitted-1, omitted) * rho**omitted
        ratio = rho * Fraction(power+omitted, omitted+1)
        assert ratio < 1
        error += 2 * prefactor * center**(-power) * term / (1-ratio)
        envelope += prefactor * Fraction(256)**(-power)
    return {'K': k, 'absolute_analytic_error': str(error),
            'absolute_integral_envelope': str(envelope)}


def stick_breaking(ts):
    remaining = 1.0
    xs = []
    for t in ts:
        xs.append(remaining*t)
        remaining *= 1-t
    return (*xs, remaining)


def point_integrand(xs, selected, targets=TARGETS):
    """Only explicit real multiplies/divides are used for inverse complex F."""
    p = tuple(math.fsum(x*q[a] for x, q in zip(xs, selected['shifts'])) for a in range(4))
    real = math.fsum(x*m[0] for x, m in zip(xs, selected['masses_squared']))
    real -= math.fsum(xs[i]*xs[j]*selected['invariants'][i][j] for i in range(6) for j in range(i+1, 6))
    imaginary = math.fsum(x*m[1] for x, m in zip(xs, selected['masses_squared']))
    denominator = real*real + imaginary*imaginary
    inv = complex(real/denominator, -imaginary/denominator)
    inverse = {1: inv}
    for power in (2, 3, 4):
        a, b = inverse[power-1], inv
        inverse[power] = complex(a.real*b.real-a.imag*b.imag, a.real*b.imag+a.imag*b.real)
    result = {}
    for name, powers in targets.items():
        terms = [c * math.prod(v**power for v, power in zip(p, residual)) * inverse[power]
                 for c, residual, power in numeric_terms(powers, 'float')]
        result[name] = complex(math.fsum(x.real for x in terms), math.fsum(x.imag for x in terms))
    return result


def decimal_point_integrand(xs, selected, targets=TARGETS):
    """Independent Decimal arithmetic path for a small-order cross-check.

    The mathematical formula is shared. This is an arithmetic check, not an
    independent analytic representation or a rigorous roundoff certificate.
    """
    zero = Decimal(0)
    p = tuple(sum((x*shift[a] for x, shift in zip(xs, selected['shifts'])), zero) for a in range(4))
    real = sum((x*mass[0] for x, mass in zip(xs, selected['masses_squared'])), zero)
    real -= sum((xs[i]*xs[j]*selected['invariants'][i][j] for i in range(6) for j in range(i+1, 6)), zero)
    imaginary = sum((x*Decimal(str(mass[1])) for x, mass in zip(xs, selected['masses_squared'])), zero)
    denominator = real*real+imaginary*imaginary
    inv = (real/denominator, -imaginary/denominator)
    inverse = {1: inv}
    for power in (2, 3, 4):
        a, b = inverse[power-1], inv
        inverse[power] = (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])
    result = {}
    for name, powers in targets.items():
        real, imaginary = zero, zero
        for c, residual, power in numeric_terms(powers, 'decimal'):
            prefactor = c
            for v, r in zip(p, residual):
                if r:
                    prefactor *= v**r
            real += prefactor*inverse[power][0]
            imaginary += prefactor*inverse[power][1]
        result[name] = (real, imaginary)
    return result


def integrate(n, selected, deadline, engine='float'):
    if sys.float_info.mant_dig != 53 or sys.float_info.radix != 2:
        raise RuntimeError('prototype fast evaluator requires IEEE binary64')
    rules = [gaussian_rule(n, a, deadline) for a in (4, 3, 2, 1, 0)]
    if time.monotonic() > deadline:
        raise TimeoutError('aggregate cap reached during exact rule preparation')
    if engine == 'decimal':
        return integrate_decimal(n, selected, deadline, rules)
    # The first-coordinate slices bound memory to O(n^4), and fsum avoids
    # a naive n^5-term accumulation. Two fsum stages are retained explicitly.
    partials = {name: [] for name in TARGETS}
    for first in rules[0]:
        arrays = {name: ([], []) for name in TARGETS}
        for others in itertools.product(*rules[1:]):
            if time.monotonic() > deadline:
                raise TimeoutError('explicit aggregate wall-time cap reached')
            records = (first, *others)
            weight = math.prod(r['weight'] for r in records)
            values = point_integrand(stick_breaking(tuple(r['node'] for r in records)), selected)
            for name, value in values.items():
                arrays[name][0].append(value.real*weight)
                arrays[name][1].append(value.imag*weight)
        for name, (real, imaginary) in arrays.items():
            partials[name].append((math.fsum(real), math.fsum(imaginary)))
    return {name: {'real': math.fsum(x[0] for x in parts),
                   'imaginary': math.fsum(x[1] for x in parts),
                   'bounds': exact_bounds(n, TARGETS[name])}
            for name, parts in partials.items()}


def integrate_decimal(n, selected, deadline, rules):
    """Small-order 80-digit check, with no per-node values retained."""
    with localcontext() as context:
        context.prec = 80
        totals = {name: [Decimal(0), Decimal(0)] for name in TARGETS}
        for records in itertools.product(*rules):
            if time.monotonic() > deadline:
                raise TimeoutError('explicit aggregate wall-time cap reached')
            remaining, weight = Decimal(1), Decimal(1)
            xs = []
            for record in records:
                t = Decimal(record['node_decimal'])
                xs.append(remaining*t)
                remaining *= 1-t
                weight *= Decimal(record['weight_decimal'])
            xs.append(remaining)
            values = decimal_point_integrand(xs, selected)
            for name, (real, imaginary) in values.items():
                totals[name][0] += real*weight
                totals[name][1] += imaginary*weight
        return {name: {'real': str(real), 'imaginary': str(imaginary),
                       'bounds': exact_bounds(n, TARGETS[name])}
                for name, (real, imaginary) in totals.items()}


def proposal():
    selected = fixture()
    return {'schema': 1, 'status': 'source-only; actual fixture quadrature not executed',
            'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            'mutually_consistent_4d_invariants': True,
            'qualified_domain': 'N6 massive fixture108 only; no singular integrals',
            'finite_normalization': 'd^4q/(i*pi^2); MadLoop component moments, no multinomial weights',
            'targets': {name: {'index': INDICES[name], 'powers': powers,
                              'terms': [{'coefficient': str(c), 'powers': r, 'inverse_F_power': p}
                                        for c, r, p in tensor_terms(powers)],
                              'n18_analytic_bound': exact_bounds(18, powers),
                              'n20_analytic_bound': exact_bounds(20, powers)}
                        for name, powers in TARGETS.items()},
            'roundoff_status': 'unqualified; analytic bounds exclude floating-node/weight/arithmetic error',
            'runtime_proposal': {'initial_orders': [4, 6, 8, 10],
                                 'optional_analytic_endpoints': [18, 20],
                                 'independent_arithmetic_check': '80-digit Decimal at order8',
                                 'aggregate_seconds': 180, 'max_nodes_per_order': 20**5},
            'native_reducer_calls': 0, 'model_card_initialization': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('proposal')
    run = subs.add_parser('run')
    run.add_argument('--orders', type=int, nargs='+', default=[4, 6, 8, 10])
    run.add_argument('--engine', choices=('float', 'decimal'), default='float')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--admit-physics', action='store_true')
    run.add_argument('--seconds', type=int, default=180)
    args = parser.parse_args(argv)
    if args.command == 'proposal':
        print(json.dumps(proposal(), indent=2))
        return 0
    if not args.admit_physics:
        parser.error('actual fixture evaluation requires root-reviewed --admit-physics')
    if args.output.exists():
        parser.error('output must be a new path; prior results are immutable')
    if not (1 <= args.seconds <= 180) or any(not 4 <= n <= 20 for n in args.orders):
        parser.error('proposal bounds: 1..180 seconds and orders 4..20')
    if args.engine == 'decimal' and any(n > 10 for n in args.orders):
        parser.error('the bounded arithmetic cross-check is restricted to Decimal order <=10')
    result = proposal()
    result['status'] = 'running; floating roundoff unqualified'
    result['orders'] = []
    result['engine'] = args.engine
    result['decimal_precision'] = 80 if args.engine == 'decimal' else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    started = time.monotonic()
    deadline = started + args.seconds
    try:
        for n in args.orders:
            result['attempted_order'] = n
            tick = time.monotonic()
            row = {'n': n, 'nodes': n**5, 'results': integrate(n, fixture(), deadline, args.engine)}
            row['rules'] = [{'alpha': alpha, 'records': gaussian_rule(n, alpha)} for alpha in (4, 3, 2, 1, 0)]
            row['elapsed_seconds'] = time.monotonic()-tick
            result['orders'].append(row)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
        result['status'] = 'completed estimates; total numerical error unqualified'
    except Exception as exc:
        result['status'] = 'failed or timed out; prior completed orders retained'
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    result['elapsed_seconds'] = time.monotonic()-started
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['status'].startswith('completed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
