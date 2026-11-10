#!/usr/bin/env python3
"""Offline exact-rational geometry audit of frozen EW QP mass captures.

This program never loads a reducer, runs a subprocess, or edits an input.
Only --output is created, exclusively. All floating-point input decoding and
matrix arithmetic use integers/Fraction; Decimal is for printed diagnostics.
"""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
if sys.flags.optimize:
    raise RuntimeError('Do not disable evidence assertions')
HERE = Path(__file__).resolve().parent
EXECUTION = HERE.parent
DESIGN = HERE.parents[2]
ROOT = HERE.parents[7]
PARSER_SHA = '5c255423ba0f656df2e1e338beebfab2dee4c7a5f85c2a92ac57d503b143ebb8'
BACKENDS = (8, 9)
ARMS = ('original', 'candidate')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dec(value):
    if value is None:
        return None
    with localcontext() as ctx:
        ctx.prec = 90
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def exact(value):
    return {'fraction': str(value), 'decimal': dec(value)}


def load_parser():
    path = DESIGN / 'analyze.py'
    assert sha(path) == PARSER_SHA
    spec = importlib.util.spec_from_file_location('frozen_mass_capture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rne(value, precision):
    """Nearest-even rounding for finite normal values, exact integer arithmetic."""
    if not value:
        return value
    sign = -1 if value < 0 else 1
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    assert exponent >= (-16382 if precision == 113 else -1022)
    quantum = Fraction(2) ** (exponent - precision + 1)
    x = value / quantum
    integer, remainder = divmod(x.numerator, x.denominator)
    if 2 * remainder > x.denominator or (2 * remainder == x.denominator and integer % 2):
        integer += 1
    return sign * integer * quantum


def values(parser, target, label, real=True):
    result = [parser.row_values(row) for row in target['data'][label]]
    assert all(x is not None for row in result for x in row), label
    if real:
        assert all(len(row) == 1 or row[1] == 0 for row in result), label
        return [row[0] for row in result]
    return result


def vectors(flat):
    assert len(flat) % 4 == 0
    return [flat[i:i + 4] for i in range(0, len(flat), 4)]


def matrix(flat, n):
    assert len(flat) == n * n
    return [[flat[i + n * j] for j in range(n)] for i in range(n)]


def dot(a, b):
    return a[0] * b[0] - sum(a[k] * b[k] for k in range(1, 4))


def distances(shifts):
    return [[dot([x - y for x, y in zip(a, b)], [x - y for x, y in zip(a, b)])
             for b in shifts] for a in shifts]


def determinant(matrix):
    if not matrix:
        return Fraction(1)
    a = [row[:] for row in matrix]
    value = Fraction(1)
    for i in range(len(a)):
        pivot = next((j for j in range(i, len(a)) if a[j][i]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            value = -value
        p = a[i][i]
        value *= p
        for j in range(i + 1, len(a)):
            factor = a[j][i] / p
            for k in range(i + 1, len(a)):
                a[j][k] -= factor * a[i][k]
    return value


def gram(dist, members):
    origin = members[0]
    return [[dist[origin][i] + dist[origin][j] - dist[i][j]
             for j in members[1:]] for i in members[1:]]


def gram_metrics(dist, masses, members):
    z = gram(dist, members)
    det = determinant(z)
    size = len(z)
    origin = members[0]
    f = [dist[origin][i] - masses[i] + masses[origin] for i in members[1:]]
    scale = max([abs(masses[origin])] + list(map(abs, f)) + [abs(x) for row in z for x in row])
    with localcontext() as ctx:
        ctx.prec = 90
        ddet = Decimal(dec(abs(det)))
        hadamard = Decimal(1)
        for row in z:
            hadamard *= Decimal(dec(sum(x * x for x in row))).sqrt()
        relative = ddet / hadamard if hadamard else Decimal(0)
        scaled = (ddet ** (Decimal(1) / size) / Decimal(dec(scale))) if size and scale else Decimal(0)
        condition = min(relative, scaled) if size else None
    return {'members_zero_based': members, 'gram_exact': [[str(x) for x in row] for row in z],
            'determinant': exact(det), 'recursion_scale': exact(scale),
            'hadamard_relative_determinant': str(relative),
            'scale_relative_root_determinant': str(scaled),
            'source_defined_condition': str(condition) if condition is not None else None}, z, det


def comparison(parser, captures):
    pairs = [(f'original-{b}', f'candidate-{b}', 'same_backend') for b in BACKENDS]
    pairs += [(f'{a}-8', f'{a}-9', 'cross_backend') for a in ARMS]
    out = {}
    for left, right, kind in pairs:
        checked = parser.compare_pair(captures[left], captures[right], kind)
        assert checked['nonmass_input_bits_match'], checked['failures']
        out[left + '_vs_' + right] = {
            'all_scoped_input_bits_match': True,
            'difference_counts': dict(Counter(x['label'] for x in checked['differences'])),
            'mass_square_changes': len(checked['mass_square_changes']),
            'scope': 'Input point, actual repaired/consumed QP point, model values, numerator, PL and shifts; '
                     'within-backend scale. Cross-backend masses and native invariants also match. '
                     'Cross-backend native scale is intentionally excluded because of the preserved square/promotion convention.',
        }
    return out


def validate_mass_path(parser, captures, mapping):
    changed = Counter()
    records = []
    for point in (1, 2):
        p = captures['candidate-9']['points'][point]
        model = {symbol: values(parser, p, 'MODEL_MP__' + symbol)[0]
                 for symbol in ('MDL_MT', 'MDL_MZ', 'MDL_MW', 'MDL_MH')}
        model['ZERO'] = Fraction(0)
        for g in p['groups']:
            group = g['key'][1]
            n = g['key'][4]
            original = captures['original-9']['points'][point]['groups'][group - 1]
            symbols = [x['model_symbol'] for x in mapping[group]['mass_pairs']]
            assert len(symbols) == n
            for i, symbol in enumerate(symbols, 1):
                assert values(parser, g, 'MP_ARG_M' + str(i)) == [model[symbol]]
                dp = values(parser, g, 'ARG_M' + str(i))[0]
                assert model[symbol] == dp, 'This actual fixed card promotes external real model entries'
            symbols = symbols[-1:] + symbols[:-1]
            new = values(parser, g, 'QP_M2L')
            old = values(parser, original, 'DP_M2L')
            assert new == [rne(model[s] ** 2, 113) for s in symbols]
            assert old == [rne(model[s] ** 2, 53) for s in symbols]
            for label in ('ENTRY_M2_QP', 'ORDERED_MASSES', 'NATIVE_MASSES'):
                assert values(parser, g, label) == new, label
            for label in ('ENTRY_M2_DP', 'ORDERED_MASSES', 'NATIVE_MASSES'):
                assert values(parser, original, label) == old, label
            for i, (a, b, symbol) in enumerate(zip(old, new, symbols)):
                if a != b:
                    changed[symbol] += 1
        for symbol, value in model.items():
            records.append({'point': point, 'symbol': symbol, 'actual_model_mass': exact(value),
                            'dp_square_promoted': exact(rne(value ** 2, 53)),
                            'qp_square': exact(rne(value ** 2, 113)),
                            'qp_minus_promoted_dp_square': exact(rne(value ** 2, 113) - rne(value ** 2, 53))})
    return {'all_400_groups_match_actual_model_and_cyclic_mass_mapping': True,
            'all_400_groups_entry_ordered_and_native_mass_arrays_match': True,
            'square_semantics': 'Exact integer nearest-even checks of captured real masses: 53-bit old square, 113-bit candidate square.',
            'changed_mass_records_by_symbol_over_two_points': dict(changed), 'model_records': records}


def geometry(parser, captures):
    records = []
    internal = {}
    for arm in ARMS:
        for point in (1, 2):
            p = captures[arm + '-9']['points'][point]
            ps = vectors(values(parser, p, 'CONSUMED_MP_PS'))
            model_mass = values(parser, p, 'MODEL_MP__MDL_MT')[0]
            external = [{'leg_one_based': i + 1, 'mass_shell_residual_against_actual_model_square':
                         exact(dot(v, v) - (model_mass ** 2 if i in (2, 3) else 0))}
                        for i, v in enumerate(ps)]
            for g in p['groups']:
                group, n, rank = g['key'][1], g['key'][4], g['key'][5]
                shifts = vectors(values(parser, g, 'NATIVE_SHIFTS'))
                cart = distances(shifts)
                inv = matrix(values(parser, g, 'NATIVE_INVARIANTS'), n)
                smat = matrix(values(parser, g, 'KINEMATIC_S'), n)
                masses = values(parser, g, 'NATIVE_MASSES')
                assert all(inv[i][j] == inv[j][i] for i in range(n) for j in range(n))
                assert all(inv[i][i] == 0 for i in range(n))
                pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        residue = cart[i][j] - inv[i][j]
                        scale = max(abs(cart[i][j]), abs(inv[i][j]), Fraction(1))
                        modes = []
                        if inv[i][j] == 0:
                            modes.append('matches_zero_shell')
                        if masses[i] and smat[i][j] == -masses[j]:
                            modes.append('matches_mass_i_snap')
                        if masses[j] and smat[i][j] == -masses[i]:
                            modes.append('matches_mass_j_snap')
                        pairs.append({'pair_zero_based': [i, j], 'native_invariant': exact(inv[i][j]),
                                      'cartesian_invariant': exact(cart[i][j]),
                                      'cartesian_minus_native': exact(residue),
                                      'relative_to_max_abs_pair_or_1': dec(abs(residue) / scale),
                                      'S_plus_masses_minus_native': exact(smat[i][j] + masses[i] + masses[j] - inv[i][j]),
                                      'captured_S_matches': modes or ['no_exact_mass_or_zero_shell_match']})
                variants = {'whole_tensor': list(range(n))}
                native_refused = g['native'][0]['status'] != 0
                captured_condition = None
                if native_refused:
                    assert not any(label.startswith('RAW_') for label in g['data'])
                    assert len(g['errors']) == 1
                    match = re.search(r'ill-conditioned Gram matrix for \[([^]]+)\]: conditioning ([\deE.+-]+)', g['errors'][0])
                    assert match, g['errors']
                    members = [int(x.strip()) for x in match[1].split(',')]
                    variants['captured_refusal_subtopology'] = members
                    captured_condition = match[2]
                grams = {}
                for name, members in variants.items():
                    supplied, zs, ds = gram_metrics(inv, masses, members)
                    derived, zc, dc = gram_metrics(cart, masses, members)
                    difference = [[zc[i][j] - zs[i][j] for j in range(len(zs))] for i in range(len(zs))]
                    origin = members[0]
                    f_residual = [cart[origin][i] - inv[origin][i] for i in members[1:]]
                    grams[name] = {'supplied': supplied, 'cartesian': derived,
                                   'cartesian_minus_supplied_gram_exact': [[str(x) for x in row] for row in difference],
                                   'max_abs_gram_entry_difference': dec(max([abs(x) for row in difference for x in row] or [Fraction(0)])),
                                   'relative_determinant_change_cartesian_minus_supplied_over_supplied': dec((dc - ds) / ds) if ds else None,
                                   'rank_one_kinematic_f_cartesian_minus_supplied': [exact(x) for x in f_residual]}
                    if name == 'captured_refusal_subtopology':
                        with localcontext() as ctx:
                            ctx.prec = 90
                            stored = Decimal(captured_condition)
                            calc = Decimal(supplied['source_defined_condition'])
                            err = abs(calc - stored) / abs(stored)
                        grams[name]['captured_condition'] = captured_condition
                        grams[name]['relative_difference_from_captured_condition'] = str(err)
                        assert err < Decimal('1e-50'), (arm, group, err)
                r = {'arm': arm, 'point': point, 'group': group, 'n': n, 'rank': rank,
                     'native_refused': native_refused, 'native_error': g['errors'], 'pairs': pairs, 'grams': grams,
                     'max_absolute_pair_residual': dec(max([abs(cart[i][j] - inv[i][j]) for i in range(n) for j in range(n)])),
                     'rank_one_tensor_identity': {'evaluated': False,
                         'reason': 'Native refusal: no valid RAW tensor outputs exist.' if native_refused else
                         'Not evaluated here: a full identity needs input-matched pinched scalar masters, not just parent tensor outputs.'}}
                if group == 5:
                    r['actual_external_mass_shell_residuals'] = external
                records.append(r)
                internal[arm, point, group] = (inv, cart)
    comparisons = []
    for point in (1, 2):
        for group in range(1, 201):
            old, cart = internal['original', point, group]
            new, cart2 = internal['candidate', point, group]
            assert cart == cart2
            n = len(cart)
            a = max(abs(cart[i][j] - old[i][j]) for i in range(n) for j in range(n))
            b = max(abs(cart[i][j] - new[i][j]) for i in range(n) for j in range(n))
            comparisons.append({'point': point, 'group': group,
                                'original_max_abs_invariant_residual': dec(a),
                                'candidate_max_abs_invariant_residual': dec(b),
                                'candidate_over_original': dec(b / a) if a else None})
    return records, comparisons


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert Path.cwd().resolve() == ROOT
    parser = load_parser()
    paths = [Path(__file__), DESIGN / 'analyze.py', DESIGN / 'attempt2/analyze.py',
             EXECUTION / 'analysis.json', EXECUTION / 'summary.json',
             DESIGN.parent / 'group-mapping.json', DESIGN / 'historical-controls.json']
    captures = {}
    for arm in ARMS:
        for backend in BACKENDS:
            path = EXECUTION / 'raw' / f'{arm}-{backend}' / 'mass-observer.hex'
            captures[f'{arm}-{backend}'] = parser.parse(path, arm, backend)
            paths.append(path)
            ignored = ROOT / '.tirust-build/ew-qp-mass-prototype/observer-native-design/attempt2/runs' / f'{arm}-{backend}' / 'mass-observer.hex'
            assert sha(path) == sha(ignored), 'Retained log differs from ignored original'
    paths += [EXECUTION / 'raw/candidate-9/MadLoop5_resources/MadLoopParams.dat']
    for arm in ARMS:
        process = DESIGN / 'sources' / arm
        for name in ('CT_interface.f', 'TIRUST_interface.f', 'loop_CT_calls_1.f'):
            paths.append(process / name)
    paths.append(ROOT / '.tirust-build/ew-qp-mass-prototype/original/SubProcesses/P0_epem_ttxa/improve_ps.f')
    oldaudit = ROOT / 'docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay'
    paths += [oldaudit / x for x in ('input-consistency/audit.py', 'gram-consistency/audit.py', 'gram-consistency/summary.json')]
    paths += [ROOT / '.tirust-build/pole-development/crates/iregi-rust/src/direct' / x for x in ('exec.rs', 'numeric.rs')]
    assert all(path.is_file() and not path.is_symlink() for path in paths)
    mapping = {g['loop_group']: g for g in json.loads((DESIGN.parent / 'group-mapping.json').read_text())['groups']}
    assert set(mapping) == set(range(1, 201))
    comparisons = comparison(parser, captures)
    mass_path = validate_mass_path(parser, captures, mapping)
    records, changes = geometry(parser, captures)
    controls = json.loads((DESIGN / 'historical-controls.json').read_text())['cases']
    historical = {name: parser.historical_control(captures[name], control) for name, control in controls.items()}
    assert all(x['exact_retained_binary64_status_and_available_host_flag_parity'] for x in historical.values())
    native_summary = {}
    for name, c in captures.items():
        native_summary[name] = {'footer': c['footer'], 'points': []}
        for point, p in c['points'].items():
            assert all(len(g['native']) == 1 for g in p['groups'])
            native_summary[name]['points'].append({'point': point, 'groups': len(p['groups']),
                'refused_groups': [g['key'][1] for g in p['groups'] if g['native'][0]['status']],
                'public_status': p['data']['PUBLIC_STATUS'], 'accuracy_bits': p['data']['ACCURACY']})
    result = {'schema': 1, 'scope': 'Offline exact arithmetic on frozen fixed-CTMode4 unchecked captures; no native call or input change.',
        'sources_and_inputs_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'retained_logs_match_original_ignored_logs': True, 'input_comparisons': comparisons,
        'historical_controls': historical, 'native_summary': native_summary, 'mass_path': mass_path,
        'definitions': {'invariant': 'd_ij=(q_i-q_j)^2, Minkowski signature +---.',
            'S': 'KINEMATIC_S is the mass-subtracted matrix; NATIVE_INVARIANTS is REAL(S_ij+m_i^2+m_j^2) after binary128 arithmetic.',
            'Gram': 'Z_ij=d_oi+d_oj-d_ij=2(q_i-q_o).(q_j-q_o), for the stated zero-based subset origin o.',
            'rank_one_kinematic': 'f_i=d_oi-m_i^2+m_o^2; with l_prime=l+q_o, 2*l_prime.(q_i-q_o)=D_i-D_o-f_i. Reported residual is f_cartesian-f_supplied, not a numerical tensor residual.',
            'rounding': 'Raw little-endian IEEE words decoded exactly. Determinants and matrix residuals use Fraction. Printed square roots and condition ratios use90 decimal digits.',
            'conditioning': 'Same source-defined min(Hadamard-relative determinant, scale-relative determinant root); heuristic only, not a condition number or error bound.',
            'snap_labels': 'Labels report exact matches to source snap expressions, not an instrumented branch trace.'},
        'limitations': ['Both arms retain37 TIRust native Gram refusals at threshold; failed RAW outputs are not read or inferred.',
            'Fixed4 public120/accuracy-1 is unchecked. Aggregate TIRust threshold amplitudes omit refused groups and do not establish physical accuracy.',
            'This audit does not change supplied invariants to Cartesian ones and makes no new reducer evaluation.',
            'Successful native status and exact kinematic consistency do not establish tensor recurrence correctness.',
            'Only two frozen real-model points and explicit backends8/9 are covered; CT5, checked rescue, derived masses and complex masses remain unqualified.',
            'Historical NinjaRS host flags were unavailable; only retained fields can be compared.'],
        'original_candidate_invariant_residuals': changes, 'geometry_records': records}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'records': len(records), 'output': str(args.output), 'sha256': sha(args.output),
                      'group5': [{k: r[k] for k in ('arm', 'point', 'group', 'max_absolute_pair_residual')} for r in records if r['group'] == 5]}, indent=2))


if __name__ == '__main__':
    main()
