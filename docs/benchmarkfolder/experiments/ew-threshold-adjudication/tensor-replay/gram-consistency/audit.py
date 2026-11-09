#!/usr/bin/env python3
"""Exact determinant comparison for fixed binary128 shifts and supplied S."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = HERE.parents[5]
spec = importlib.util.spec_from_file_location('exact_binary128', PARENT / 'input-consistency/audit.py')
exact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact)
D = Decimal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dec(value):
    return D(value.numerator) / D(value.denominator)


def determinant(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def main():
    assert Path.cwd().resolve() == ROOT
    selected = json.loads((PARENT / 'selected-inputs.json').read_text())
    assert sha(PARENT / 'selected-inputs.dat') == selected['input_stream_sha256']
    source_paths = [Path(__file__), PARENT / 'input-consistency/audit.py',
        PARENT / 'selected-inputs.json', PARENT / 'selected-inputs.dat',
        ROOT / '.tirust-build/pole-development/crates/iregi-rust/src/direct/exec.rs',
        ROOT / '.tirust-build/pole-development/crates/iregi-rust/src/direct/numeric.rs']
    rows = []
    with localcontext() as context:
        context.prec = 90
        for record in selected['records']:
            assert record['n'] == 4 and record['rank'] == 3
            shifts = {row[0]: [exact.binary128(v) for v in row[1:]] for row in record['shifts']}
            supplied = {(i, j): exact.binary128(v) for i, j, v in record['invariants']}
            masses = {row[0]: exact.binary128(row[1]) for row in record['masses']}
            assert all(exact.binary128(row[2]) == 0 for row in record['masses'])
            derived = {(i, j): (shifts[i][0] - shifts[j][0]) ** 2 -
                sum((shifts[i][k] - shifts[j][k]) ** 2 for k in range(1, 4))
                for i in shifts for j in shifts}
            variants = {}
            for label, distances in [('authoritative_supplied_S', supplied), ('exact_fixed_shift_geometry', derived)]:
                gram = [[distances[1, i] + distances[1, j] - distances[i, j]
                    for j in (2, 3, 4)] for i in (2, 3, 4)]
                det = determinant(gram)
                hadamard = D(1)
                for row in gram:
                    hadamard *= dec(sum(v * v for v in row)).sqrt()
                relative = dec(abs(det)) / hadamard
                f = [distances[1, i] - masses[i] + masses[1] for i in (2, 3, 4)]
                scale = dec(max([abs(masses[1])] + list(map(abs, f)) +
                    [abs(v) for row in gram for v in row]))
                scaled = dec(abs(det)) ** (D(1) / 3) / scale
                condition = min(relative, scaled)
                variants[label] = {'gram_matrix': [[str(dec(v)) for v in row] for row in gram],
                    'determinant_exact_fraction': str(det), 'determinant': str(dec(det)),
                    'determinant_sign': (det > 0) - (det < 0), 'determinant_is_zero': det == 0,
                    'hadamard_relative_determinant': str(relative), 'recursion_scale': str(scale),
                    'scaled_root_determinant': str(scaled), 'condition': str(condition),
                    'rank_power': 3, 'condition_cubed': str(condition ** 3),
                    'guard_256_2^-136_refuses': condition ** 3 < D(2) ** -136,
                    'guard_512_2^-392_refuses': condition ** 3 < D(2) ** -392,
                    'heuristic_loss_bits': str(-3 * condition.ln() / D(2).ln())}
            authoritative = variants['authoritative_supplied_S']
            if record['historical_status'] != 0:
                captured = D(re.search(r'conditioning ([0-9.e+-]+)', record['historical_error']).group(1))
                residual = abs(D(authoritative['condition']) - captured) / abs(captured)
                assert residual < D('1e-55')
            else:
                captured, residual = None, None
            original_det = D(authoritative['determinant'])
            derived_det = D(variants['exact_fixed_shift_geometry']['determinant'])
            rows.append({'point': record['point'], 'group': record['group'], 'variants': variants,
                'captured_refusal_condition': str(captured) if captured is not None else None,
                'relative_error_against_captured_condition': str(residual) if residual is not None else None,
                'relative_determinant_change_from_supplied_to_shift_geometry': str((derived_det - original_det) / original_det)})
    result = {'schema': 1,
        'scope': 'Pure exact-rational input/source audit. No reducer call, input alteration, generated distance transport or acceptance claim.',
        'sources': {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        'definition': 'Zij = S(1,i)+S(1,j)-S(i,j); min(|det Z|/product(row norms), |det Z|^(1/3)/recursion scale), cubed for coefficient rank3.',
        'precision': 'Decode fixed binary128 inputs exactly; all determinant arithmetic is rational. Roots/logs use90decimal digits. Shift-derived S is exact geometry of those fixed shifts, not newly rounded or supplied reducer input.',
        'limits': ['The normalized determinant is a source-defined heuristic, not a certified condition number or error bound.',
            'Replacing supplied S with shift-derived S would change API inputs and requires a separately labeled diagnostic.',
            'The audit does not establish tensor coefficients, Cartesian consistency, finite amplitude accuracy or any production guard policy.'],
        'records': rows}
    (HERE / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps([{'point': r['point'], 'group': r['group'],
        'relative_determinant_change': r['relative_determinant_change_from_supplied_to_shift_geometry']} for r in rows]))


if __name__ == '__main__':
    main()
