#!/usr/bin/env python3
"""Source-bound QP input-policy arithmetic; no compiler or reducer invocation."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EW = HERE.parent / 'ew-threshold-adjudication'
GENERATED = ROOT / '.tirust-build/ew-threshold-adjudication'
EXACT = EW / 'tensor-replay/input-consistency/audit.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, text):
    if path.exists():
        assert path.read_text() == text, f'Refusing to change frozen evidence: {path}'
    else:
        path.write_text(text)


def det(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def dot(a):
    return a[0] ** 2 - sum(x ** 2 for x in a[1:])


def main():
    assert Path.cwd().resolve() == ROOT
    spec = importlib.util.spec_from_file_location('policy_binary128', EXACT)
    exact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exact)
    source_ranges = {
        'Source/MODEL/lha_read.f': [(356, 407)],
        'Source/MODEL/param_read.inc': [(20, 30)],
        'Source/MODEL/mp_intparam_definition.inc': [(105, 115)],
        'Source/MODEL/couplings.f': [(15, 55), (134, 155)],
        'Source/MODEL/actualize_mp_ext_params.inc': [(1, 7)],
        'SubProcesses/P0_epem_ttxa/loop_matrix.f': [(2130, 2160)],
        'SubProcesses/P0_epem_ttxa/improve_ps.f': [(42, 151), (230, 251), (609, 622), (654, 662)],
        'SubProcesses/P0_epem_ttxa/CT_interface.f': [(248, 340)],
    }
    old = json.loads((EW / 'prepared.json').read_text())
    bound = {str(p.relative_to(ROOT)): sha(p) for p in (
        Path(__file__).resolve(), EXACT, EW / 'prepared.json', EW / 'trace.json',
        EW / 'geometry/raw.log', EW / 'geometry/summary.json',
        EW / 'artifact-index.json',
        HERE.parent / 'ew-threshold-geometry-supplement/evidence-index.json')}
    excerpts = ['Source excerpts from the unchanged historical generated process.\n',
        'Complete-file hashes below match the original trace preparation manifest; '
        'they are working-file identities, not an assertion that a Git HEAD contains generated files.\n']
    for relative, ranges in source_ranges.items():
        path = GENERATED / relative
        digest = sha(path)
        assert digest == old['original_source_hashes'][relative], relative
        bound[str(path.relative_to(ROOT))] = digest
        lines = path.read_text().splitlines()
        excerpts += [f'\n{relative}\nSHA256 {digest}\n']
        for start, end in ranges:
            excerpts += ['\n'.join(f'{i}: {lines[i-1]}' for i in range(start, end + 1)) + '\n']

    headers = {}
    for line in (EW / 'geometry/raw.log').read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in ('DP_TOP_PROMOTED', 'MP_TOP', 'DP_SQUARE_PROMOTED', 'MP_SQUARE'):
            assert fields[0] not in headers
            headers[fields[0]] = exact.binary128(fields[1])
    mass = headers['MP_TOP']
    old_square = headers['DP_SQUARE_PROMOTED']
    qp_square = headers['MP_SQUARE']
    assert mass == headers['DP_TOP_PROMOTED']
    # A product of this53-bit dyadic value fits exactly within113significand bits.
    assert qp_square == mass * mass
    trace = json.loads((EW / 'trace.json').read_text())
    records = []
    with localcontext() as context:
        context.prec = 100
        def decimal(value):
            return str(Decimal(value.numerator) / Decimal(value.denominator))
        for point, group in ((2, 5), (1, 5), (2, 24)):
            r = trace[str(point)]['groups'][str(group)]
            q = {int(i): list(map(exact.binary128, values)) for i, *values in r['shifts']}
            s = {(int(i), int(j)): exact.binary128(value) for i, j, value in r['invariants']}
            assert r['n'] == 4
            physical = {(i, j): dot([x-y for x, y in zip(q[i], q[j])]) for i, j in s}
            # Algebraic target substitution only: it is NOT a floating-point
            # replay of MP_BUILD_KINEMATIC_MATRIX or a new integral request.
            hypothetical = {key: qp_square if value == old_square else value for key, value in s.items()}
            changed = [list(key) for key in sorted(s) if s[key] != hypothetical[key]]
            grams = {}
            for name, matrix in (('original_S', s), ('QP_shell_target_substitution', hypothetical), ('exact_fixed_shifts', physical)):
                z = [[matrix[1,i] + matrix[1,j] - matrix[i,j] for j in (2,3,4)] for i in (2,3,4)]
                grams[name] = {'determinant': decimal(det(z)), 'determinant_exact': str(det(z)),
                    'max_distance_from_exact_shift_geometry': decimal(max(abs(matrix[key]-physical[key]) for key in matrix))}
            records.append({'point': point, 'group': group, 'substituted_entries_1_based': changed,
                'original_status': r['status'], 'Gram_arithmetic': grams})
        result = {'schema': 1,
            'scope': 'Research-only source/input policy proposal; exact arithmetic on saved data, no native calls or production changes.',
            'source_sha256': bound,
            'input_mass_policy': 'Preserve the existing promoted binary64 parameter; do not reinterpret its original decimal spelling.',
            'mass_arithmetic': {
                'promoted_DP_mass': decimal(mass),
                'mass_minus_exact_decimal_173_3': decimal(mass-Fraction(1733,10)),
                'promoted_DP_square': decimal(old_square),
                'QP_square_of_promoted_mass': decimal(qp_square),
                'QP_square_minus_promoted_DP_square': decimal(qp_square-old_square),
                'QP_square_minus_exact_decimal_mass_square': decimal(qp_square-Fraction(1733,10)**2),
                'QP_square_exact_for_this_fixed_input_mass': True},
            'rank1_exact_identity': 'cq-cs = inverse(Zq)*((Zs-Zq)*cs + (fs-fq)*I0), provided the same pinched scalars and I0 are held fixed.',
            'rank1_error_bound': 'norm(cq-cs) <= norm(inverse(Zq))*(norm(Zs-Zq)*norm(cs)+norm(fs-fq)*abs(I0)); use a consistent operator/vector norm. The native determinant heuristic is not this norm.',
            'limits': [
                'Shell-target substitution is an algebraic sensitivity diagnostic, not a replay of rounded native preprocessing and not evaluated reducer input.',
                'Correcting squared propagator masses also changes scalar integrals; the fixed-scalar rank1 identity does not predict those new tensors.',
                'Finite binary128 repaired shifts retain shell/conservation roundoff. Exact q-distance and exact on-shell identities generally cannot both hold for all represented values.',
                'No precision guard, threshold, cache, physical input, phase-space repair, numerator or production source is changed.',
                'No conclusion about higher ranks, finite amplitude, automatic acceptance, or performance.'],
            'records': records}
    for relative, expected in bound.items():
        assert sha(ROOT / relative) == expected, relative
    save(HERE / 'source-excerpts.txt', '\n'.join(excerpts))
    save(HERE / 'summary.json', json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'mass_square_difference': result['mass_arithmetic']['QP_square_minus_promoted_DP_square'],
        'threshold_original_distance_error': records[0]['Gram_arithmetic']['original_S']['max_distance_from_exact_shift_geometry'],
        'threshold_substituted_distance_error': records[0]['Gram_arithmetic']['QP_shell_target_substitution']['max_distance_from_exact_shift_geometry']}))


if __name__ == '__main__':
    main()
