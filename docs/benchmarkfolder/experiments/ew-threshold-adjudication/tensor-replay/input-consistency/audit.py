#!/usr/bin/env python3
"""Exact-rational audit of captured binary128 shifts versus supplied invariants."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def power2(exponent):
    return Fraction(2 ** exponent) if exponent >= 0 else Fraction(1, 2 ** (-exponent))


def binary128(text):
    """Round a finite decimal token to IEEE binary128, ties to even.

    All nonzero inputs here are normal binary128 values. This reproduces the
    values read from the 36-significant-digit round-trip Fortran records,
    without a binary64 intermediate or a floating-point squared difference.
    """
    value = Fraction(Decimal(text))
    if value == 0:
        return value
    sign = -1 if value < 0 else 1
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < power2(exponent):
        exponent -= 1
    assert -16382 <= exponent <= 16383
    quantum = power2(exponent - 112)
    scaled = value / quantum
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    if twice > scaled.denominator or (twice == scaled.denominator and integer % 2):
        integer += 1
    result = sign * integer * quantum
    assert abs(result - sign * value) <= quantum / 2
    return result


def decimal(value):
    with localcontext() as context:
        context.prec = 90
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def main():
    selected = json.loads((PARENT / 'selected-inputs.json').read_text())
    assert digest(PARENT / 'selected-inputs.dat') == selected['input_stream_sha256']
    rows = []
    for record in selected['records']:
        shifts = {row[0]: [binary128(v) for v in row[1:]] for row in record['shifts']}
        supplied = {(i, j): binary128(v) for i, j, v in record['invariants']}
        differences = []
        for i in range(1, record['n'] + 1):
            for j in range(i + 1, record['n'] + 1):
                delta = [a - b for a, b in zip(shifts[i], shifts[j])]
                exact = delta[0] ** 2 - sum(v ** 2 for v in delta[1:])
                residual = exact - supplied[i, j]
                differences.append({'pair': [i, j], 'supplied_invariant': decimal(supplied[i, j]),
                    'exact_shift_difference_squared': decimal(exact),
                    'exact_minus_supplied': decimal(residual),
                    'absolute_residual': decimal(abs(residual))})
        rows.append({'point': record['point'], 'group': record['group'], 'pairs': differences,
            'max_absolute_residual': max((row['absolute_residual'] for row in differences), key=Decimal)})
    result = {'schema': 1, 'scope': 'Source/input arithmetic audit only; no reducer call or physical point alteration.',
        'input_manifest_sha256': digest(PARENT / 'selected-inputs.json'),
        'input_stream_sha256': digest(PARENT / 'selected-inputs.dat'),
        'audit_script_sha256': digest(Path(__file__)),
        'method': 'Round exact decimal Fractions to113 bits, nearest ties-even; compute Minkowski squared differences as exact rationals; print90 decimal digits.',
        'interpretation': 'Identical supplied shifts and invariants can still be mutually inconsistent after intentional on-shell snapping. Different internal uses of them need independent examination before assigning a near-threshold disagreement to arithmetic precision.',
        'limitation': 'This audit neither changes the standard MadLoop on-shell policy nor identifies which internal recurrence causes an observed coefficient difference.',
        'records': rows}
    (HERE / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps([{'point': row['point'], 'group': row['group'],
        'max_absolute_residual': row['max_absolute_residual']} for row in rows]))


if __name__ == '__main__':
    main()
