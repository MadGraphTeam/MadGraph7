#!/usr/bin/env python3
"""Offline rank-one residuals from successful, frozen current-arm RAW records.

No native code or subprocess is invoked. The default creates rank-one.json
exclusively; --check recomputes it in memory and verifies the existing bytes.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
from itertools import combinations, permutations
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if sys.flags.optimize:
    raise RuntimeError('Evidence assertions must remain enabled')
HERE = Path(__file__).resolve().parent
EXECUTION = HERE.parent
DESIGN = HERE.parents[2]
ROOT = HERE.parents[7]
PARSER_SHA = '5c255423ba0f656df2e1e338beebfab2dee4c7a5f85c2a92ac57d503b143ebb8'
INDEX_SHA = '4012228d621d39956944027d5682033781c88791c6dc11f90abc4a6fe30e1f7b'
SOURCE_PINS = {
    'direct/exec.rs': '255b250f6997cfd08f017a9962ac79e6a4a87421f1f2d9f7ddb6a8bc1129eace',
    'convention.rs': '2346b0ba2e8b8000d99a626849d8d466f1b3492518499cecb8dfa398a56f43ad',
    'direct/plan.rs': 'f24b05ca3fbd74c68fde719da0efb6280d341b0a75e6eee7cc9ca42711626b2a',
}
SLOTS = ('finite', 'single_total', 'double', 'uv1', 'ir1')
ZERO = (Fraction(0), Fraction(0))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decimal(x):
    return Decimal(x.numerator) / Decimal(x.denominator)


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def scale(a, b):
    return a[0] * b, a[1] * b


def mul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def dot(a, b):
    return a[0] * b[0] - sum(a[k] * b[k] for k in range(1, 4))


def magnitude(a):
    return decimal(a[0] ** 2 + a[1] ** 2).sqrt()


def pack(a):
    return {'exact': [str(x) for x in a], 'decimal': [str(decimal(x)) for x in a]}


def raw(parser, g, slot, index):
    names = {'finite': ('RAW_FINITE',), 'single_total': ('RAW_UV1', 'RAW_IR1'),
             'double': ('RAW_IR2',), 'uv1': ('RAW_UV1',), 'ir1': ('RAW_IR1',)}
    result = ZERO
    for label in names[slot]:
        values = parser.row_values(g['data'][label][index])
        assert len(values) == 2 and all(v is not None for v in values)
        result = add(result, values)
    return result


def shifts(parser, g):
    flat = [parser.row_values(row)[0] for row in g['data']['NATIVE_SHIFTS']]
    assert all(v is not None for v in flat)
    return [flat[i:i + 4] for i in range(0, len(flat), 4)]


def signature(g, order):
    """Full supplied invariant matrix, masses and scale; exact storage words."""
    a, n = g['data'], g['key'][4]
    return (tuple(a['NATIVE_SCALE'][0][2:]),
            tuple(tuple(a['NATIVE_MASSES'][i][2:]) for i in order),
            tuple(tuple(a['NATIVE_INVARIANTS'][j * n + i][2:]) for j in order for i in order))


def canonical(g, keep):
    return min(signature(g, p) for p in permutations(keep))


def pinch_details(parser, parent, drop, matches):
    keep = [i for i in range(parent['key'][4]) if i != drop]
    ps = shifts(parser, parent)
    target = signature(parent, keep)
    output = []
    for child in sorted(matches, key=lambda g: g['key'][1]):
        cs = shifts(parser, child)
        mappings = []
        for p in permutations(range(len(keep))):
            if signature(child, p) != target:
                continue
            translations = [[cs[c][k] - ps[i][k] for k in range(4)] for i, c in zip(keep, p)]
            translated = all(t == translations[0] for t in translations)
            mappings.append({'child_order_1_based': [c + 1 for c in p],
                             'exact_common_translation': translated,
                             'translation_if_common': [str(v) for v in translations[0]] if translated else None})
        assert mappings
        compatible = any(m['exact_common_translation'] for m in mappings)
        # Record all admissible permutations, including any alternative routing.
        chosen_mapping = min(mappings, key=lambda m: (not m['exact_common_translation'], m['child_order_1_based']))
        output.append({'group': child['key'][1], 'native_status': child['native'][0]['status'],
                       'scalar_key': 'full supplied-invariant/mass/scale bit match',
                       'routing_classification': 'retained parent shifts up to exact common translation'
                       if compatible else 'supplied-invariant-only match; no exact common translation',
                       'any_exact_common_translation': compatible, 'selected_mapping': chosen_mapping,
                       'all_matching_permutations': mappings,
                       'scalar': {slot: pack(raw(parser, child, slot, 0)) for slot in SLOTS}})
    spreads = {}
    for slot in SLOTS:
        values = [raw(parser, g, slot, 0) for g in matches]
        differences = [magnitude(sub(a, b)) for a, b in combinations(values, 2)]
        spread = max(differences, default=Decimal(0))
        norm = max(map(magnitude, values), default=Decimal(0))
        spreads[slot] = {'max_pairwise_absolute': str(spread),
                         'max_pairwise_over_max_scalar_magnitude': str(spread / norm if norm else Decimal(0))}
    return {'omitted_parent_1_based': drop + 1, 'retained_parent_order_1_based': [i + 1 for i in keep],
            'selected_lowest_group': output[0]['group'], 'matches': output, 'alternative_scalar_spread': spreads}


def equation(parser, parent, i, j, pinch_i, pinch_j):
    p = shifts(parser, parent)
    delta = [a - b for a, b in zip(p[i], p[j])]
    mi = tuple(parser.row_values(parent['data']['NATIVE_MASSES'][i]))
    mj = tuple(parser.row_values(parent['data']['NATIVE_MASSES'][j]))
    coefficient = add((dot(p[i], p[i]) - dot(p[j], p[j]), Fraction(0)), sub(mj, mi))
    result = {'pair_1_based': [i + 1, j + 1], 'pinch_i_group': pinch_i['key'][1],
              'pinch_j_group': pinch_j['key'][1], 'coefficient_exact': [str(v) for v in coefficient], 'slots': {}}
    for slot in SLOTS:
        vector_terms = [scale(raw(parser, parent, slot, k + 1), 2 * delta[k] * (1 if k == 0 else -1))
                        for k in range(4)]
        lhs = ZERO
        for term in vector_terms:
            lhs = add(lhs, term)
        pi, pj = raw(parser, pinch_i, slot, 0), raw(parser, pinch_j, slot, 0)
        mass_term = mul(coefficient, raw(parser, parent, slot, 0))
        rhs = add(sub(pj, pi), mass_term)
        residual = sub(lhs, rhs)
        norm = magnitude(residual)
        terms_norm = sum((magnitude(t) for t in vector_terms + [pi, pj, mass_term]), Decimal(0))
        equation_norm = max(magnitude(lhs), magnitude(rhs))
        assert terms_norm or not norm
        assert equation_norm or not norm
        result['slots'][slot] = {'lhs': pack(lhs), 'rhs': pack(rhs), 'residual': pack(residual),
            'residual_absolute': str(norm), 'primitive_term_magnitude_sum': str(terms_norm),
            'equation_side_magnitude_max': str(equation_norm),
            'residual_over_primitive_terms': str(norm / terms_norm if terms_norm else Decimal(0)),
            'residual_over_equation_sides': str(norm / equation_norm if equation_norm else Decimal(0))}
    return result


def cohort(parser, capture, arm, point):
    groups = capture['points'][point]['groups']
    lookup = defaultdict(list)
    for g in groups:
        assert len(g['native']) == 1
        if g['native'][0]['status'] == 0:
            lookup[g['key'][4], canonical(g, range(g['key'][4]))].append(g)
        else:
            assert not any(label.startswith('RAW_') for label in g['data'])
    availability, records = [], []
    for g in groups:
        n, rank = g['key'][4:6]
        if n == 1:
            continue
        matches = [lookup.get((n - 1, canonical(g, [k for k in range(n) if k != drop])), []) for drop in range(n)]
        complete = all(matches)
        success = g['native'][0]['status'] == 0
        availability.append({'group': g['key'][1], 'n': n, 'rank': rank,
            'native_status': g['native'][0]['status'], 'all_pinches_matched': complete,
            'scalar_match_groups_by_omitted_propagator': [sorted(x['key'][1] for x in m) for m in matches],
            'evaluated_identity': bool(success and complete),
            'exclusion': None if success and complete else 'native refusal; parent RAW undefined' if not success
            else 'not every pinch has a successful exact scalar-key match'})
        if not success or not complete:
            continue
        assert rank >= 1
        details = [pinch_details(parser, g, drop, m) for drop, m in enumerate(matches)]
        selected = [min(m, key=lambda x: x['key'][1]) for m in matches]
        records.append({'group': g['key'][1], 'n': n, 'rank': rank, 'native_status': 0,
            'first_shift_exact': [str(x) for x in shifts(parser, g)[0]], 'pinches': details,
            'all_selected_pinches_exact_common_translation': all(p['matches'][0]['any_exact_common_translation'] for p in details),
            'equations': [equation(parser, g, i, j, selected[i], selected[j]) for i, j in combinations(range(n), 2)]})
    assert len(records) == (33 if point == 1 else 17)
    refused = [g['key'][1] for g in groups if g['native'][0]['status'] != 0]
    assert len(refused) == (0 if point == 1 else 37)
    maxima = {}
    for slot in SLOTS:
        maxima[slot] = {}
        for metric in ('residual_absolute', 'residual_over_primitive_terms', 'residual_over_equation_sides'):
            entries = [(Decimal(eq['slots'][slot][metric]), r['group'], eq['pair_1_based']) for r in records for eq in r['equations']]
            value, group, pair = max(entries)
            maxima[slot][metric] = {'value': str(value), 'group': group, 'pair_1_based': pair}
    return {'arm': arm, 'point': point, 'point_label': 'ordinary' if point == 1 else 'threshold',
        'native_successes': len(groups) - len(refused), 'native_refused_groups': refused,
        'refused_with_complete_pinches': [x['group'] for x in availability if x['native_status'] and x['all_pinches_matched']],
        'complete_pinches_successful_parents': len(records),
        'complete_parent_equations': sum(len(r['equations']) for r in records),
        'selected_scalar_routing_counts': dict(Counter(p['matches'][0]['routing_classification'] for r in records for p in r['pinches'])),
        'maxima_diagnostics_only': maxima, 'availability': availability, 'records': records}


def build():
    assert Path.cwd().resolve() == ROOT
    parser_path = DESIGN / 'analyze.py'
    assert sha(parser_path) == PARSER_SHA
    index_path = EXECUTION / 'evidence-index.json'
    assert sha(index_path) == INDEX_SHA
    index = json.loads(index_path.read_text())
    spec = importlib.util.spec_from_file_location('frozen_rank_one_parser', parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    paths = [Path(__file__), parser_path, index_path]
    source_root = ROOT / '.tirust-build/pole-development/crates/iregi-rust/src'
    for relative, expected in SOURCE_PINS.items():
        path = source_root / relative
        assert sha(path) == expected
        paths.append(path)
    paths += [source_root / 'basis.rs', ROOT / 'vendor/tirust/src/lib.rs',
              DESIGN / 'sources/original/TIRUST_interface.f', DESIGN / 'sources/candidate/TIRUST_interface.f']
    cohorts = []
    for arm in ('original', 'candidate'):
        raw_path = EXECUTION / 'raw' / (arm + '-9') / 'mass-observer.hex'
        card_path = raw_path.parent / 'MadLoop5_resources/MadLoopParams.dat'
        for path in (raw_path, card_path):
            assert sha(path) == index['files'][str(path.relative_to(EXECUTION))]
            paths.append(path)
        card = card_path.read_text()
        assert '#LoopPoleMode\n0\n' in card and '#TIRustComputeFinite\n.TRUE.\n' in card
        capture = parser.parse(raw_path, arm, 9)
        for point in (1, 2):
            assert capture['points'][point]['data']['PUBLIC_STATUS'][0][1] == '120'
            assert all(parser.row_values(r) == [Fraction(-1)] for r in capture['points'][point]['data']['ACCURACY'])
            cohorts.append(cohort(parser, capture, arm, point))
    return {'schema': 1, 'scope': 'Offline exact arithmetic on current-arm frozen successful TIRust RAW records only.',
        'accuracy_criterion': None, 'physical_accuracy_qualified': False, 'native_execution_performed': False,
        'sources_and_inputs_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'formula': '2 (p_i-p_j).R = P_j-P_i + (p_i^2-p_j^2-m_i^2+m_j^2) S; metric (+---); S=RAW[1], R=RAW[2:5].',
        'derivation': 'D_i-D_j = 2q.(p_i-p_j)+p_i^2-p_j^2-m_i^2+m_j^2; divide by product D and integrate. MadLoop rank-one RAW is minus the OneLOop q moment; scalar RAW is unchanged.',
        'convention_source_anchors': ['basis.rs:13-27: increasing rank, then time/x/y/z order',
            'convention.rs:70-82: MadLoop only multiplies by (-1)^rank; no N-dependent factor',
            'direct/exec.rs:3622-3630: direct MadLoop output same four slots with odd-rank minus',
            'direct/plan.rs:1123-1142: parent and pinched PV terms have the displayed signs',
            'vendor/tirust/src/lib.rs:355-362: MadLoop, Components, SuppliedInvariants, MuSquared'],
        'numerical_method': 'Exact IEEE binary128 Fractions, complex rational products and residuals. Euclidean magnitudes and ratios printed with 90-digit Decimal arithmetic. Zero/zero ratios are defined as zero. No residual rounding to binary128.',
        'normalization': {'primitive_terms': 'sum magnitudes of four 2*metric_mu*(p_i-p_j)_mu*R_mu terms, P_i, P_j, and c*S',
            'equation_sides': 'max(|2*(p_i-p_j).R|, |P_j-P_i+c*S|) after exact summation'},
        'selection': 'Every successful parent with every pinch matched; all unordered propagator pairs. Lowest matching current-arm/same-point group ID supplies each scalar. All alternative IDs, permutations and scalar spreads retained. No partial-pinch equations evaluated.',
        'limitations': ['Matches are external scalar evaluations, not captured internal master states.',
            'Scalar keys omit native shifts by supplied-invariant scalar translation/permutation invariance; exact common-translation routing checks are separately retained for every match.',
            'A supplied-invariant-only match cannot establish exact Cartesian denominator-product consistency. Even a common-translation match retains the supplied-versus-Cartesian invariant discrepancy.',
            'All 37 refused threshold parents remain undefined, including 14 with complete matched pinches; none receives a numeric identity.',
            'Residues include supplied/Cartesian geometry inconsistency and output rounding, and may include reduction errors; this audit does not separate or bound them.',
            'UV and IR single slots are also shown separately; the merged single pole is computed by exact addition of the recorded split slots.',
            'Fixed mode 4, status120 and accuracy -1 remain unchecked. No physical accuracy, complete tensor reconstruction, amplitude acceptance, historical NinjaRS replay, or performance conclusion follows.'],
        'cohorts': cohorts}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--check', action='store_true')
    args = cli.parse_args()
    with localcontext() as context:
        context.prec = 90
        result = build()
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    path = HERE / 'rank-one.json'
    if args.check:
        assert path.read_text() == text
    else:
        with path.open('x') as out:
            out.write(text)
    print(json.dumps({'state': 'reproduction_verified' if args.check else 'offline_appendix_created',
        'output': str(path.relative_to(ROOT)), 'sha256': sha(path),
        'cohorts': [{'arm': c['arm'], 'point': c['point'], 'parents': c['complete_pinches_successful_parents'],
            'equations': c['complete_parent_equations'], 'maxima': c['maxima_diagnostics_only']}
            for c in result['cohorts']]}))


if __name__ == '__main__':
    main()
