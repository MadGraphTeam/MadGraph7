#!/usr/bin/env python3
"""Check saved rank-one box/pinched-triangle identities; no reducer calls."""
import hashlib
import importlib.util
import json
from decimal import Decimal as D, localcontext
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TRACE = HERE.parent / 'ew-threshold-adjudication'
REPLAY = TRACE / 'tensor-replay/higher-precision/replay-forward'
EXACT_PATH = TRACE / 'tensor-replay/input-consistency/audit.py'
PINCHES = {1: ('8', (2, 3, 4)), 2: ('196', (3, 4, 1)),
           3: ('4', (2, 1, 4)), 4: ('16', (1, 2, 3))}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def mul(s, z):
    return tuple(s * x for x in z)


def norm(z):
    return sum(x * x for x in z).sqrt()


def dot(a, b):
    return a[0] * b[0] - sum(a[k] * b[k] for k in (1, 2, 3))


def solve(matrix, vector):
    rows = [[*row, value] for row, value in zip(matrix, vector)]
    n = len(rows)
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(rows[i][k]))
        rows[k], rows[pivot] = rows[pivot], rows[k]
        divisor = rows[k][k]
        assert divisor != 0
        rows[k] = [x / divisor for x in rows[k]]
        for i in range(n):
            if i != k:
                factor = rows[i][k]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[k])]
    return [row[-1] for row in rows]


def pair(z):
    return [str(x) for x in z]


def main():
    assert Path.cwd().resolve() == ROOT
    paths = [Path(__file__), EXACT_PATH, TRACE / 'trace.json',
             TRACE / 'execution.json', TRACE / 'summary.json',
             REPLAY / 'selected-inputs.json', REPLAY / 'selected-inputs.dat',
             REPLAY / 'source-manifest.json', REPLAY / 'execution.json',
             REPLAY / 'output.log', REPLAY / 'summary.json',
             REPLAY / 'analyze.py', REPLAY / 'replay.f90',
             TRACE / 'tensor-replay/higher-precision/source/lib.rs']
    sources = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    spec = importlib.util.spec_from_file_location('pinch_exact_binary128', EXACT_PATH)
    exact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exact)

    def number(token):
        f = exact.binary128(token)
        return D(f.numerator) / D(f.denominator)

    def complex_number(tokens):
        assert len(tokens) == 2
        return tuple(map(number, tokens))

    trace = json.loads((TRACE / 'trace.json').read_text())
    trace_execution = json.loads((TRACE / 'execution.json').read_text())
    selected = json.loads((REPLAY / 'selected-inputs.json').read_text())
    summary = json.loads((REPLAY / 'summary.json').read_text())
    execution = json.loads((REPLAY / 'execution.json').read_text())
    assert trace_execution['complete'] and trace_execution['final_values_identical_to_historical']
    assert execution['complete'] and execution['returncode'] == 0
    assert execution['protected_inputs_unchanged']
    assert execution['output_sha256'] == summary['raw_output_sha256'] == sha(REPLAY / 'output.log')
    assert summary['execution_manifest_sha256'] == sha(REPLAY / 'execution.json')
    assert summary['input_manifest_sha256'] == sha(REPLAY / 'selected-inputs.json')
    assert summary['analyzer_sha256'] == sha(REPLAY / 'analyze.py')
    assert selected['source_manifest_sha256'] == execution['source_manifest_sha256'] == sha(REPLAY / 'source-manifest.json')
    assert selected['input_stream_sha256'] == execution['input_stream_sha256'] == sha(REPLAY / 'selected-inputs.dat')
    for name in ('trace', 'summary', 'execution'):
        assert selected['parent_' + name + '_sha256'] == sha(TRACE / (name + '.json'))

    # Independently bind the summarized comparison tokens to the frozen output.
    raw_comparisons = {}
    current = None
    for line in (REPLAY / 'output.log').read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'BEGIN':
            current = tuple(map(int, fields[1:3]))
        elif fields[0] == 'COMPARE':
            key = (*current, int(fields[1]), int(fields[2]))
            assert key not in raw_comparisons
            raw_comparisons[key] = {'tir': fields[3:5], 'nrs': fields[5:7]}
    for record in summary['records']:
        assert record['tir_status'] == 0
        assert len(record['comparisons']) == 105
        for row in record['comparisons']:
            key = (record['point'], record['group'], row['unit'], row['slot'])
            assert raw_comparisons[key] == {b: row[b] for b in ('tir', 'nrs')}
    assert len(raw_comparisons) == 105 * len(summary['records'])

    results = []
    with localcontext() as context:
        context.prec = 110
        for point in (1, 2):
            inp = next(r for r in selected['records'] if (r['point'], r['group']) == (point, 5))
            out = next(r for r in summary['records'] if (r['point'], r['group']) == (point, 5))
            original = trace[str(point)]['groups']['5']
            assert inp['n'] == 4 and inp['rank'] == 3
            for field in ('shifts', 'masses', 'invariants'):
                assert [[str(x) for x in row] for row in inp[field]] == original[field]
            assert inp['scale'] == original['scale']
            q = {i: list(map(number, row)) for i, *row in inp['shifts']}
            masses = {i: complex_number([re, im]) for i, re, im in inp['masses']}
            assert all(z[1] == 0 for z in masses.values())
            m = {i: z[0] for i, z in masses.items()}
            s = {(i, j): exact.binary128(value) for i, j, value in inp['invariants']}
            sd = {(i, j): number(value) for i, j, value in inp['invariants']}
            pinches, pinch_bindings = {}, []
            for omitted, (group, permutation) in PINCHES.items():
                triangle = trace[str(point)]['groups'][group]
                assert triangle['status'] == 0 and triangle['n'] == 3
                assert exact.binary128(triangle['scale']) == exact.binary128(inp['scale'])
                tm = {int(i): tuple(exact.binary128(x) for x in (re, im))
                      for i, re, im in triangle['masses']}
                sm = {i: tuple(exact.binary128(x) for x in (re, im))
                      for i, re, im in inp['masses']}
                ts = {(int(i), int(j)): exact.binary128(value)
                      for i, j, value in triangle['invariants']}
                assert set(permutation) == set(range(1, 5)) - {omitted}
                assert all(tm[i] == sm[permutation[i - 1]] for i in (1, 2, 3))
                assert all(ts[i, j] == s[permutation[i - 1], permutation[j - 1]]
                           for i in (1, 2, 3) for j in (1, 2, 3))
                raw = {(int(i), int(k)): complex_number([re, im])
                       for i, k, re, im in triangle['raw']}
                assert len(raw) == len(triangle['raw'])
                pinches[omitted] = {0: raw[1, 4], 1: add(raw[1, 1], raw[1, 2]), 2: raw[1, 3]}
                pinch_bindings.append({'omitted_box_denominator': omitted,
                    'trace_group': int(group), 'native_status': triangle['status'],
                    'triangle_order_to_box_indices': list(permutation),
                    'exact_supplied_invariants_masses_and_scale_match': True,
                    'scalar_slots_finite_single_double': [pair(pinches[omitted][slot]) for slot in range(3)]})

            p = [[q[j][k] - q[1][k] for k in range(4)] for j in (2, 3, 4)]
            zq = [[2 * dot(a, b) for b in p] for a in p]
            zs = [[sd[1, i] + sd[1, j] - sd[i, j] for j in (2, 3, 4)] for i in (2, 3, 4)]
            units = {(row['unit'], row['slot']): row for row in out['comparisons']}
            rows = []
            for backend in ('tir', 'nrs'):
                for slot in range(3):
                    for k in range(5):
                        assert out['NRS_units'][str(k)]['status'] == 0
                        assert out['NRS_units'][str(k)]['degree'] == (0 if k == 0 else 1)
                    i0 = complex_number(units[0, slot][backend])
                    moment = [complex_number(units[k, slot][backend]) for k in (1, 2, 3, 4)]
                    jmoment = [add(mul(-1, moment[k]), mul(q[1][k], i0)) for k in range(4)]
                    contractions = [mul(2, sub(mul(pj[0], jmoment[0]),
                        add(add(mul(pj[1], jmoment[1]), mul(pj[2], jmoment[2])),
                            mul(pj[3], jmoment[3])))) for pj in p]
                    coefficients = list(zip(*(solve(zq, [z[d] for z in contractions]) for d in (0, 1))))
                    for index, j in enumerate((2, 3, 4)):
                        rhs = sub(pinches[j][slot], pinches[1][slot])
                        for geometry in ('exact_fixed_shift_geometry', 'authoritative_supplied_Gram'):
                            if geometry == 'exact_fixed_shift_geometry':
                                contraction = contractions[index]
                                offset = dot(p[index], p[index]) + m[1] - m[j]
                            else:
                                contraction = tuple(sum(zs[index][k] * coefficients[k][d] for k in range(3)) for d in (0, 1))
                                offset = sd[1, j] + m[1] - m[j]
                            scalar_term = mul(offset, i0)
                            lhs = add(contraction, scalar_term)
                            residual = sub(lhs, rhs)
                            absolute = norm(residual)
                            rhs_scale = norm(rhs)
                            equation_scale = max(norm(lhs), rhs_scale)
                            term_scale = max(norm(contraction), norm(scalar_term), norm(pinches[j][slot]), norm(pinches[1][slot]))
                            rows.append({'backend': backend, 'slot': slot,
                                'slot_name': ('finite', 'single', 'double')[slot], 'j': j,
                                'geometry': geometry, 'lhs': pair(lhs), 'rhs': pair(rhs),
                                'rhs_is_exactly_zero': rhs == (D(0), D(0)),
                                'vector_contraction': pair(contraction), 'scalar_offset': str(offset),
                                'scalar_offset_times_I0': pair(scalar_term), 'residual': pair(residual),
                                'absolute_residual': str(absolute), 'rhs_scale': str(rhs_scale),
                                'equation_scale': str(equation_scale), 'term_scale': str(term_scale),
                                'relative_to_rhs': str(absolute / rhs_scale) if rhs_scale else None,
                                'relative_to_equation_scale': str(absolute / equation_scale) if equation_scale else None,
                                'relative_to_term_scale': str(absolute / term_scale) if term_scale else None})
            results.append({'point': point, 'group': 5, 'pinch_bindings': pinch_bindings,
                'gram_exact_shifts': [[str(x) for x in row] for row in zq],
                'gram_supplied_invariants': [[str(x) for x in row] for row in zs], 'identities': rows})

    for name, expected in sources.items():
        assert sha(ROOT / name) == expected, name
    result = {'schema': 1, 'scope': 'Saved rank-one box/pinched-scalar consistency; no native execution or input change.',
        'sources_sha256': sources, 'sources_unchanged': True,
        'arithmetic': 'Decode decimal echoes to exact binary128 fractions, then use110decimal digits for contractions and3x3solve.',
        'definitions': {
            'denominator': 'D_i=(ell+q_i)^2-m_i^2, metric(+---). Mass records already contain m_i^2.',
            'MadLoop_moment': 'M^mu=integral(-ell^mu)/product(D); unit1..4 are time,x,y,z.',
            'pinch': 'C_i=scalar triangle omitting D_i, normalization unchanged across topology.',
            'shifted_vector': 'p_j=q_j-q_1; J=-M+q_1*I0. The tiny nonzero captured q_1 is retained.',
            'physical_identity': '2*p_j.J + [p_j^2+m_1^2-m_j^2]*I0 = C_j-C_1.',
            'authoritative_identity': 'Recover c from Zq*c=2*p.J, then test Zs*c+fs*I0=C_j-C_1; Zs_ij=S1i+S1j-Sij and fs_j=S1j+m_1^2-m_j^2.',
            'term_scale': 'max(abs(vector contraction),abs(offset*I0),abs(C_j),abs(C_1)); not an accuracy tolerance.',
            'slots': 'finite=raw4, single=raw1+raw2, double=raw3; before real projection or2/(16pi^2).'},
        'limits': ['The saved triangles use historical TIRust/scalar machinery; this is an algebraic consistency test, not an independent scalar oracle.',
            'Only rank-one identities are checked. Rank-two/three disagreements and full numerator contractions are not adjudicated.',
            'Authoritative S and fixed shifts are intentionally different representations in this evidence; neither geometry is changed or declared the uniquely correct input.',
            'Returning binary128 tensors limits residual precision despite512-bit TIRust internal arithmetic.',
            'Zero RHS and cancellation require absolute and term scales; relative residual alone is misleading.',
            'This does not establish finite amplitude accuracy, stability acceptance, performance, or a production policy.'],
        'records': results}
    destination = HERE / 'pinch-summary.json'
    encoded = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if destination.exists():
        assert destination.read_text() == encoded, 'Refusing to overwrite different pinch evidence'
    else:
        destination.write_text(encoded)
    print('PASS: saved pinch identities bound; no reducer invocation; ' + sha(destination))


if __name__ == '__main__':
    main()
