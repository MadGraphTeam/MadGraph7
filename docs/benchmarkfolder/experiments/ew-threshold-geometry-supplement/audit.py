#!/usr/bin/env python3
"""Replay rank-one geometry identities from saved coefficients, without reducers."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TENSORS = HERE.parent / 'ew-threshold-adjudication/tensor-replay'
REPLAY = TENSORS / 'higher-precision/replay-forward'
spec = importlib.util.spec_from_file_location('exact_binary128', TENSORS / 'input-consistency/audit.py')
exact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact)
D = Decimal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dec(value):
    return D(value.numerator) / D(value.denominator)


def determinant(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def solve(a, b):
    n = len(a)
    rows = [[*a[i], b[i]] for i in range(n)]
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(rows[i][j]))
        assert rows[pivot][j] != 0
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [x / scale for x in rows[j]]
        for i in range(n):
            if i != j:
                scale = rows[i][j]
                rows[i] = [x - scale * y for x, y in zip(rows[i], rows[j])]
    return [row[n] for row in rows]


def multiply(a, v):
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def dot(a, b):
    return a[0] * b[0] - sum(a[i] * b[i] for i in range(1, 4))


def complex_text(a):
    return [str(dec(x)) for x in a]


def comparison(a, b):
    absolute = sum(dec(x - y) ** 2 for x, y in zip(a, b)).sqrt()
    norm = max(sum(dec(x) ** 2 for x in v).sqrt() for v in (a, b))
    return {'absolute': str(absolute), 'relative': str(absolute / norm if norm else D(0))}


def main():
    assert Path.cwd().resolve() == ROOT
    selected = json.loads((REPLAY / 'selected-inputs.json').read_text())
    summary = json.loads((REPLAY / 'summary.json').read_text())
    execution = json.loads((REPLAY / 'execution.json').read_text())
    assert execution['complete'] and execution['returncode'] == 0
    assert summary['execution_manifest_sha256'] == sha(REPLAY / 'execution.json')
    assert summary['input_manifest_sha256'] == sha(REPLAY / 'selected-inputs.json')
    assert summary['raw_output_sha256'] == sha(REPLAY / 'output.log')
    sources = [Path(__file__), TENSORS / 'input-consistency/audit.py',
        REPLAY / 'selected-inputs.json', REPLAY / 'summary.json', REPLAY / 'execution.json',
        REPLAY / 'output.log', TENSORS / 'replay.f90',
        ROOT / '.tirust-build/pole-development/crates/iregi-rust/src/direct/exec.rs',
        ROOT / '.tirust-build/pole-development/crates/iregi-rust/src/direct/plan.rs']
    inputs = {(r['point'], r['group']): r for r in selected['records']}
    records = []
    with localcontext() as context:
        context.prec = 100
        for record in summary['records']:
            key = record['point'], record['group']
            data = inputs[key]
            assert data['n'] == 4 and data['rank'] == 3 and record['tir_status'] == 0
            shifts = [[exact.binary128(x) for x in row[1:]] for row in data['shifts']]
            masses = [exact.binary128(row[1]) for row in data['masses']]
            assert all(exact.binary128(row[2]) == 0 for row in data['masses'])
            supplied = {(i - 1, j - 1): exact.binary128(x) for i, j, x in data['invariants']}
            p = [[shifts[i][mu] - shifts[0][mu] for i in (1, 2, 3)] for mu in range(4)]
            vectors = [[p[mu][i] for mu in range(4)] for i in range(3)]
            zs = [[supplied[0, i] + supplied[0, j] - supplied[i, j]
                for j in (1, 2, 3)] for i in (1, 2, 3)]
            zq = [[2 * dot(a, b) for b in vectors] for a in vectors]
            fs = [supplied[0, i] - masses[i] + masses[0] for i in (1, 2, 3)]
            fq = [dot(vectors[i - 1], vectors[i - 1]) - masses[i] + masses[0] for i in (1, 2, 3)]
            axes = max(itertools.combinations(range(4), 3),
                key=lambda axes: abs(determinant([p[mu] for mu in axes])))
            minor = [p[mu] for mu in axes]
            assert determinant(minor) != 0
            observations = {(r['unit'], r['slot']): r for r in record['comparisons']}
            slots = []
            for slot in range(3):
                scalar = [exact.binary128(x) for x in observations[0, slot]['tir']]
                assert observations[0, slot]['tir'] == observations[0, slot]['nrs']
                tir = [[exact.binary128(x) for x in observations[mu + 1, slot]['tir']] for mu in range(4)]
                nrs = [[exact.binary128(x) for x in observations[mu + 1, slot]['nrs']] for mu in range(4)]
                predicted = [[Fraction(0), Fraction(0)] for _ in range(4)]
                reconstructed = [[Fraction(0), Fraction(0)] for _ in range(4)]
                pv = [[Fraction(0), Fraction(0)] for _ in range(3)]
                defect = [[Fraction(0), Fraction(0)] for _ in range(3)]
                for part in range(2):
                    # MadLoop rank-one moment M=-integral(q); untranslate:
                    # integral(q+q0)=P*c=-M+q0*I0.
                    j = [-tir[mu][part] + shifts[0][mu] * scalar[part] for mu in range(4)]
                    c = solve(minor, [j[mu] for mu in axes])
                    common_rhs = multiply(zs, c)
                    rhs_q = [common_rhs[i] + (fs[i] - fq[i]) * scalar[part] for i in range(3)]
                    cq = solve(zq, rhs_q)
                    pc, pcq = multiply(p, c), multiply(p, cq)
                    for mu in range(4):
                        reconstructed[mu][part] = shifts[0][mu] * scalar[part] - pc[mu]
                        predicted[mu][part] = shifts[0][mu] * scalar[part] - pcq[mu]
                    for i in range(3):
                        pv[i][part] = c[i]
                        # Physical contraction residual against the implied
                        # supplied-S recurrence RHS, without pinched values.
                        defect[i][part] = (multiply(zq, c)[i] - common_rhs[i]
                            + (fq[i] - fs[i]) * scalar[part])
                slots.append({'slot': slot, 'scalar_common': complex_text(scalar),
                    'recovered_TIR_PV_coefficients': [complex_text(x) for x in pv],
                    'implied_shift_geometry_contraction_defect': [complex_text(x) for x in defect],
                    'components': [{'axis': mu, 'TIR_recorded': complex_text(tir[mu]),
                        'NRS_recorded': complex_text(nrs[mu]),
                        'TIR_rank1_gather_reconstruction': complex_text(reconstructed[mu]),
                        'geometry_transformed_TIR': complex_text(predicted[mu]),
                        'gather_reconstruction_residual': comparison(reconstructed[mu], tir[mu]),
                        'original_TIR_vs_NRS': comparison(tir[mu], nrs[mu]),
                        'geometry_transformed_vs_NRS': comparison(predicted[mu], nrs[mu])}
                        for mu in range(4)]})
            records.append({'point': key[0], 'group': key[1], 'best_invertible_coordinate_axes': axes,
                'coordinate_minor_determinant': str(dec(determinant(minor))),
                'Gram_supplied': [[str(dec(x)) for x in row] for row in zs],
                'Gram_from_exact_shifts': [[str(dec(x)) for x in row] for row in zq],
                'f_supplied_minus_f_shift': [str(dec(a - b)) for a, b in zip(fs, fq)],
                'slots': slots})
    result = {'schema': 1, 'scope': 'Exact-rational algebra on saved native tensors; no reducer, compiler, alternate-input native evaluation, or production change.',
        'method': 'Recover rank1 coefficients using the coordinate minor with largest absolute determinant; map the same inferred pinched-scalar RHS from supplied-S Gram/f to exact-shift Gram/f. This minor choice is not a certified conditioning bound.',
        'identity': 'Z_s*c_s=C_i-C_0-f_s*I0; c_q=Z_q^-1*(Z_s*c_s+(f_s-f_q)*I0); M_q=q0*I0-P*c_q.',
        'metric': 'Complex abs(a-b)/max(abs(a),abs(b)); zero-zero=0.',
        'sources': {str(path.relative_to(ROOT)): sha(path) for path in sources},
        'limits': ['The inferred RHS uses recorded TIR outputs, so this is a causal representation-consistency diagnostic, not an independent scalar or recurrence correctness proof.',
            'Native coefficients were rounded to113bits. Exact arithmetic preserves those values but cannot recover their unrounded native values; the unused Cartesian coordinate residual is retained.',
            'The alternative geometry is used only in this algebraic identity; no integral with altered invariants is evaluated.',
            'Higher-rank and whole-amplitude accuracy are not established by a rank-one identity.'], 'records': records}
    (HERE / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    for row in records:
        print(row['point'], row['group'], [(s['slot'], max(D(c['original_TIR_vs_NRS']['relative']) for c in s['components']),
            max(D(c['geometry_transformed_vs_NRS']['relative']) for c in s['components'])) for s in row['slots']])


if __name__ == '__main__':
    main()
