#!/usr/bin/env python3
"""Offline exact-rational assessment of retained native-policy study records."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICIES = ['Interpreted', 'NativeDenominators', 'FullNative']
FAMILIES = ['T2Denominators', 'T3Denominators', 'BubbleSourceProjection',
            'TriangleBubbleCorrection', 'TadpoleSourceProjection',
            'OtherReduction', 'Integration', 'Accumulation']
TOLERANCE = Fraction(1, 10**31)
SPLIT_TOLERANCE = Fraction(1, 10**60)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def exact(component):
    assert component['finite'] is True
    numerator, denominator = int(component['numerator']), int(component['denominator'])
    assert denominator > 0 and component['precision'] >= 1
    value = Fraction(numerator, denominator)
    assert value == 0 or (value < 0) == component['negative']
    return value


def rational(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator),
            'approximate': float(value)}


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows and rows[-1]['kind'] == 'summary' and rows[-1]['complete'] is True
    assert all(r['kind'] != 'summary' for r in rows[:-1])
    return rows[:-1], rows[-1]


def assess_arm(arm):
    ok = arm['status'] == 0 and arm['input_unchanged'] and len(arm['output']) == 8
    ok = ok and arm['raw'] is not None and len(arm['raw']) == 10 and all(c['finite'] for c in arm['raw'])
    if not ok:
        return {'successful': False, 'observation': arm}
    values = [exact(c) for c in arm['raw']]
    norm = max(map(abs, values[:6]))
    split = [values[i]-values[6+i]-values[8+i] for i in (0, 1)]
    split_bound = SPLIT_TOLERANCE * norm
    return {'successful': True, 'laurent_norm': rational(norm),
            'finite_split_defect': [rational(v) for v in split],
            'finite_split_bound': rational(split_bound),
            'finite_split_passed': all(abs(v) <= split_bound for v in split),
            'tracked_precisions': [v['precision'] for v in arm['raw']],
            'negative_zero_indices': [i for i, v in enumerate(arm['raw']) if values[i] == 0 and v['negative']]}


def compare(left, right, tolerance=TOLERANCE):
    if not assess_arm(left)['successful'] or not assess_arm(right)['successful']:
        return {'comparable': False, 'value_screen_passed': False}
    a, b = [[exact(c) for c in arm['raw']] for arm in (left, right)]
    norm = max(map(abs, b[:6]))
    bound = tolerance * norm
    differences = [abs(x-y) for x, y in zip(a, b)]
    precision_changes = [i for i, (x, y) in enumerate(zip(left['raw'], right['raw'])) if x['precision'] != y['precision']]
    sign_changes = [i for i, (x, y) in enumerate(zip(left['raw'], right['raw'])) if x['negative'] != y['negative']]
    return {'comparable': True, 'common_laurent_norm': rational(norm), 'bound': rational(bound),
            'absolute_differences_all_finite_poles_cc_rational': [rational(v) for v in differences],
            'value_screen_passed': all(d <= bound for d in differences[:6]),
            'rational_screen_passed': all(d <= bound for d in differences[8:]),
            'nonzero_value_indices': [i for i, d in enumerate(differences) if d],
            'precision_changed_indices': precision_changes, 'sign_changed_indices': sign_changes,
            'formatted_changed_indices': [i for i, (x, y) in enumerate(zip(left['output'], right['output'])) if x != y],
            'requires_review': bool(any(differences) or precision_changes or sign_changes or left['output'] != right['output'])}


def replay(path):
    rows, summary = read(path)
    assert len(rows) == 1392 and summary['records'] == 1392 and summary['arm_fields_each'] == 11136
    result = []
    backends = {}
    status = {p: Counter() for p in POLICIES}
    for index, row in enumerate(rows):
        assert row['kind'] == 'request' and row['sequence'] == index
        assert row['input']['status'] == 0 and len(row['input']['output']) == 8
        assert [a['policy'] for a in row['arms']] == POLICIES
        assert all(a['working_bits'] == 256 for a in row['arms'])
        baseline = row['arms'][0]
        baseline_exact = baseline['status'] == row['input']['status'] and baseline['output'] == row['input']['output']
        assert row['baseline_exact_historical'] == baseline_exact
        for arm in row['arms']:
            status[arm['policy']][str(arm['status'])] += 1
            if arm['backends'] is not None:
                assert [s['family'] for s in arm['backends']] == FAMILIES
                native = set()
                if arm['policy'] != 'Interpreted':
                    native.update(FAMILIES[:2])
                if arm['policy'] == 'FullNative':
                    native.update(FAMILIES[2:5])
                    if row['input']['rank'] == row['input']['n'] + 2:
                        native.remove('TriangleBubbleCorrection')
                assert all(s['backend'] == ('NativeRust' if s['family'] in native else 'SymbolicaInterpreter')
                           for s in arm['backends']), ('unexpected factory coverage', index, arm['policy'])
                shape = f"{arm['policy']}:{row['input']['n']}:{row['input']['rank']}"
                if shape in backends:
                    assert backends[shape] == arm['backends']
                backends[shape] = arm['backends']
        comparisons = [compare(arm, baseline) for arm in row['arms'][1:]]
        result.append({'sequence': index, 'n': row['input']['n'], 'rank': row['input']['rank'],
                       'baseline_exact_historical': baseline_exact,
                       'arms': [assess_arm(arm) for arm in row['arms']],
                       'comparisons_vs_interpreted': comparisons})
    return {'stage': 'replay', 'complete': True, 'input_sha256': sha(path), 'record_count': 1392,
            'fields_per_arm': 11136, 'engine_calls': 0, 'performance_claim': False,
            'all_baseline_controls_exact': all(r['baseline_exact_historical'] for r in result),
            'status_counts': status, 'factory_backend_reports_not_invocation_counts': backends,
            'all_success_finite': all(a['successful'] for r in result for a in r['arms']),
            'all_laurent_value_screens_pass': all(c['value_screen_passed'] for r in result for c in r['comparisons_vs_interpreted']),
            'records_requiring_convergence_review': [r['sequence'] for r in result
                if any(c.get('requires_review', True) for c in r['comparisons_vs_interpreted'])],
            'records': result}


def convergence(path, replay_path):
    rows, summary = read(path)
    replay_rows, _ = read(replay_path)
    selected = {r['sequence'] for r in replay_rows
                if any(r['native_any_value_precision_sign_or_formatted_difference'])
                or any(not assess_arm(a)['successful'] for a in r['arms'])}
    selected.add('physical_complex_mass_pentagon')
    assert summary['requests_per_precision'] == len(selected)
    by_key = {}
    for row in rows:
        assert row['kind'] == 'convergence' and row['sequence'] in selected
        key = (row['sequence'], row['working_bits'])
        assert key not in by_key and key[1] in (256, 384)
        assert [a['policy'] for a in row['arms']] == POLICIES
        assert all(a['working_bits'] == key[1] for a in row['arms'])
        by_key[key] = row
    assert set(by_key) == {(s, bits) for s in selected for bits in (256, 384)}
    result = []
    for sequence in sorted(selected, key=str):
        low, high = by_key[sequence, 256], by_key[sequence, 384]
        assert low['input'] == high['input']
        result.append({'sequence': sequence,
            'convergence_per_policy': [compare(a, b) for a, b in zip(low['arms'], high['arms'])],
            'cross_policy_384': [compare(a, high['arms'][0]) for a in high['arms'][1:]],
            'arms256': [assess_arm(a) for a in low['arms']],
            'arms384': [assess_arm(a) for a in high['arms']]})
    return {'stage': 'convergence', 'complete': True, 'input_sha256': sha(path),
            'replay_sha256': sha(replay_path), 'engine_calls': 0, 'requests_per_precision': len(selected),
            'all_laurent_convergence_screens_pass': all(c['value_screen_passed'] for r in result for c in r['convergence_per_policy']),
            'all_cross_policy384_laurent_screens_pass': all(c['value_screen_passed'] for r in result for c in r['cross_policy_384']),
            'automatic_sign_or_precision_difference_acceptance': False, 'records': result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('replay', 'convergence'))
    parser.add_argument('input', type=Path)
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    result = replay(args.input) if args.stage == 'replay' else convergence(args.input, args.replay)
    result['analyzer_sha256'] = sha(Path(__file__))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
