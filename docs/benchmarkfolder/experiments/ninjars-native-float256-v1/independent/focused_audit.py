#!/usr/bin/env python3
"""Read retained focused controls only; never invoke a numerical engine."""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

POLICIES = ('Interpreted', 'NativeDenominators', 'FullNative')
TOLERANCE = Fraction(1, 10**31)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction_json(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def recover_binary(value, bits=113):
    """Independent nearest-even rational rounding of one finite input record."""
    value = Fraction(value)
    if not value:
        return value
    sign = -1 if value < 0 else 1
    value = abs(value)
    n, d = value.numerator, value.denominator
    exponent = n.bit_length() - d.bit_length()
    if (n < d << exponent) if exponent >= 0 else (n << -exponent < d):
        exponent -= 1
    shift = exponent - bits + 1
    unit = Fraction(1 << shift) if shift >= 0 else Fraction(1, 1 << -shift)
    scaled = value / unit
    integer, residual = divmod(scaled.numerator, scaled.denominator)
    twice = residual * 2
    if twice > scaled.denominator or (twice == scaled.denominator and integer % 2):
        integer += 1
    return sign * integer * unit


def validate_rounding():
    # Exact binary inputs and halfway cases, including a carry into the exponent.
    for bits in (2, 53, 113, 256):
        assert recover_binary('0', bits) == 0
        assert recover_binary('-0', bits) == 0
        assert recover_binary('0.125', bits) == Fraction(1, 8)
        step = Fraction(1, 2**(bits-1))
        assert recover_binary(1 + step/2, bits) == 1
        assert recover_binary(1 + 3*step/2, bits) == 1 + 2*step
        assert recover_binary(2 - step/2, bits) == 2
        assert recover_binary(-1 - step/2, bits) == -1
    assert recover_binary('1.000000000000000000000000000000788860905') != 1


def analytic_poles(input_record, arm, existing):
    coefficient = [recover_binary(v) for v in input_record['coefficients']]
    assert len(coefficient) == 2 and coefficient[1] == 0
    if input_record['n'] == 1:
        mass = [recover_binary(v) for v in input_record['masses']]
        expected = [coefficient[0]*mass[0], coefficient[0]*mass[1]]
    else:
        assert input_record['n'] == 2
        expected = [coefficient[0], Fraction(0)]
    actual = [existing.exact(v) for v in arm['raw']]
    bound = TOLERANCE * sum(map(abs, expected))
    differences = [abs(actual[2+i]-v) for i, v in enumerate(expected)]
    return {'independently_recovered_113_bit_single_pole': [fraction_json(v) for v in expected],
            'single_pole_absolute_difference': [fraction_json(v) for v in differences],
            'single_pole_bound': fraction_json(bound),
            'single_pole_passed': all(v <= bound for v in differences),
            'double_pole_exact_zero': actual[4] == actual[5] == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    validate_rounding()
    experiment = args.experiment.resolve(strict=True)
    analyzer = experiment/'analyze.py'
    analyzer_hash = sha(analyzer)
    spec = importlib.util.spec_from_file_location('native_existing_analysis', analyzer)
    existing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(existing)
    stage = experiment/'run-focused'
    result_path = stage/'result.json'
    result = json.loads(result_path.read_text())
    assert result['complete'] and result['returncode'] == 0 and result['stage'] == 'focused'
    raw = stage/'raw.jsonl'
    assert result['files']['raw.jsonl'] == sha(raw)
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert rows[-1] == result['summary']
    assert rows[-1]['kind'] == 'summary' and rows[-1]['complete']
    assert rows[-1]['failed_controls'] == 0
    controls = rows[:-1]
    assert len(controls) == 90
    observed = {}
    records = []
    for start in range(0, len(controls), 3):
        analytic, invalid, short = controls[start:start+3]
        assert [r['kind'] for r in (analytic, invalid, short)] == [
            'analytic_rebind', 'invalid_then_recover', 'short_shape_then_recover']
        reused, fresh = analytic['reused'], analytic['fresh']
        key = (analytic['index'], reused['policy'], reused['working_bits'])
        assert key not in observed and key[0] in range(5)
        assert key[1] in POLICIES and key[2] in (256, 384)
        assert reused == fresh and analytic['passed'] is True
        assert existing.assess_arm(reused)['successful']
        bad_input = dict(analytic['input'])
        bad_input['masses'] = list(bad_input['masses'])
        bad_input['masses'][1] = '1'
        assert invalid['input'] == bad_input
        recovery = []
        for kind, row in (('invalid', invalid), ('short', short)):
            rejected = row['rejected']
            assert rejected['status'] == 2 and rejected['output'] == [] and rejected['raw'] is None
            assert rejected['input_unchanged'] is True
            assert (rejected['policy'], rejected['working_bits']) == key[1:]
            assert row['recovered'] == fresh and row['passed'] is True
            is_panic = rejected.get('panic', False)
            if kind == 'invalid':
                assert not is_panic and rejected['error'] == 'invalid MP squared scale or mass width'
            else:
                assert is_panic and 'Option::unwrap()' in rejected['panic_message']
            recovery.append({'kind': kind, 'status': rejected['status'],
                'caught_panic': is_panic, 'entire_private_state_replaced': is_panic,
                'retained_state_error_recovery_proven': not is_panic,
                'reported_error': rejected.get('error', rejected.get('panic_message'))})
        observed[key] = analytic
        records.append({'fixture': key[0], 'policy': key[1], 'working_bits': key[2],
            'arm': existing.assess_arm(reused),
            'analytic_poles': analytic_poles(analytic['input'], reused, existing),
            'recovery': recovery})
    assert set(observed) == {(i,p,b) for i in range(5) for p in POLICIES for b in (256,384)}
    convergence = []
    cross_policy = []
    for index in range(5):
        for policy in POLICIES:
            low, high = [observed[index,policy,b] for b in (256,384)]
            assert low['input'] == high['input']
            convergence.append({'fixture': index, 'policy': policy,
                'comparison_256_vs_384': existing.compare(low['reused'], high['reused'])})
        for policy in POLICIES[1:]:
            arm = observed[index,policy,384]
            reference = observed[index,'Interpreted',384]
            assert arm['input'] == reference['input']
            cross_policy.append({'fixture': index, 'policy': policy,
                'comparison_vs_interpreted_384': existing.compare(arm['reused'], reference['reused'])})
    assert sha(analyzer) == analyzer_hash and sha(raw) == result['files']['raw.jsonl']
    output = {'schema': 'independent-native-focused-audit-v1', 'complete': True,
        'engine_calls': 0, 'performance_claim': False,
        'audit_source_sha256': sha(Path(__file__)), 'shared_analyzer_sha256': analyzer_hash,
        'raw_sha256': sha(raw), 'run_result_sha256': sha(result_path),
        'analytic_records': len(records), 'invalid_recovery_records': 30,
        'panic_and_state_replacement_records': 30,
        'all_analytic_poles_pass': all(r['analytic_poles']['single_pole_passed'] and r['analytic_poles']['double_pole_exact_zero'] for r in records),
        'all_finite_split_checks_pass': all(r['arm']['finite_split_passed'] for r in records),
        'all_full_laurent_convergence_pass': all(r['comparison_256_vs_384']['value_screen_passed'] for r in convergence),
        'all_cross_policy_384_laurent_pass': all(r['comparison_vs_interpreted_384']['value_screen_passed'] for r in cross_policy),
        'automatic_sign_or_precision_difference_acceptance': False,
        'post_entry_wrong_coefficient_count_gate': 'NOT TESTED: short input panics during pairs(), then replaces state',
        'records': records, 'convergence': convergence, 'cross_policy_384': cross_policy}
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
