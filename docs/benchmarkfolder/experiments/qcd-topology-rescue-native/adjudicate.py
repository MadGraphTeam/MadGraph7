#!/usr/bin/env python3
"""Offline normalization and exact supplied/Cartesian consistency diagnostics."""
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import json
import sys

from common import ROOT, HERE, OLD, ARMS, sha, rel, require, pins

sys.dont_write_bytecode = True
sys.path.insert(0, str(OLD))
from decode_trace import load_trace
from analyze import one, reals, value

PI = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')


def fraction(record):
    return Fraction(int(record['numerator']), int(record['denominator']))


def dec(q):
    q = Fraction(q)
    return Decimal(q.numerator)/Decimal(q.denominator)


def main():
    pins()
    output = HERE/'analysis1/adjudication.json'
    require(not output.exists(), 'Do not replace an adjudication attempt.')
    result = {'schema': 1, 'native_executed': False, 'physics_accepted': False,
              'normalization': {}, 'remaining_tirust_consistency': {},
              'bindings': {rel(Path(__file__)): sha(Path(__file__)), rel(OLD/'decode_trace.py'): sha(OLD/'decode_trace.py')},
              'limitations': ['The universal double pole is one independent normalization check, not finite/single-pole truth.',
                             'Checked host322 is recorded, not promoted to cross-backend physical acceptance.',
                             'The existing Born helper promotes DP parameters; QP loop/model parameter consistency is a separate audit.',
                             'Input-condition indicators are heuristics; no precision or native calculation is changed here.']}
    for arm in ('original-b', *ARMS):
        for backend in (9, 8):
            directory = OLD/'attempt1'/f'backend-{backend}' if arm == 'original-b' else HERE/'attempt1'/arm/f'backend-{backend}'
            execution = json.loads((directory/'execution.json').read_text())
            require(execution['complete'] and execution['protected_inputs_unchanged'], 'Incomplete native record.')
            require(sha(directory/'trace.dat') == execution['trace_sha256'] and
                    sha(directory/'observed.log') == execution['log_sha256'], 'Native output hash changed.')
            trace = load_trace(directory/'trace.dat', directory/'observed.log')
            point = next(p for p in trace['points'] if p['seq'] == 2)
            ans = reals(one(point['events'], 'POINT_HOST_ANS'), trace['byte_order'])
            g = reals(one(point['events'], 'POINT_G_DP'), trace['byte_order'])[0]
            require(len(ans) == 8 and ans[:4] == ans[4:], 'Split-order host answers differ.')
            with localcontext() as ctx:
                ctx.prec = 90
                # u,u~,g,g contribute 2*C_F+2*C_A=26/3; colourless Z contributes0.
                coefficient = -Decimal(26)/Decimal(3)*dec(g)*dec(g)/(8*PI*PI)
                prediction = dec(ans[0])*coefficient
                actual = dec(ans[3])
                relative = abs(actual-prediction)/abs(prediction)
                symmetric = 2*abs(actual-prediction)/(abs(actual)+abs(prediction))
            result['normalization'][f'{arm}:{backend}'] = {
                'host_code': point['host_code'], 'born': value(ans[0]), 'finite': value(ans[1]),
                'single_pole': value(ans[2]), 'double_pole': value(ans[3]), 'g_dp': value(g),
                'universal_coefficient': str(coefficient), 'predicted_double_pole_from_actual_host_born': str(prediction),
                'relative_error': str(relative), 'symmetric_error': str(symmetric),
                'passes_3e_6_double_pole_only': relative <= Decimal('3e-6'),
                'independent_finite_single_truth_available': False}
            for name in ('execution.json', 'observed.log', 'trace.dat'):
                result['bindings'][rel(directory/name)] = sha(directory/name)
    matrix_dir = HERE/'analysis1/matrices/topology-c'
    grams = json.loads((matrix_dir/'gram.json').read_text())
    failed = {(g['ctmode'], g['group']): g for g in grams}
    pairs = [json.loads(line) for line in (matrix_dir/'pairs.jsonl').read_text().splitlines()]
    selected = [p for p in pairs if p['backend'] == 9 and (p['ctmode'], p['group']) in failed]
    groups = []
    for key, gram in failed.items():
        pp = [p for p in selected if (p['ctmode'], p['group']) == key]
        records = []
        for pair in pp:
            initial, final = fraction(pair['initial_s_real']), fraction(pair['final_s_real'])
            mi, mj = fraction(pair['mass_i']['real']), fraction(pair['mass_j']['real'])
            diffsq = fraction(pair['diffsq_real'])
            supplied = fraction(pair['native_distance'])
            cartesian = fraction(pair['cartesian_exact_squared'])
            physical = fraction(pair['literal_physical_channel_squared'])
            decomposition = {
                'routing_rounding_cartesian_minus_exact_C_channel': cartesian-physical,
                'diffsq_rounding_minus_fixed_shift_cartesian': diffsq-cartesian,
                'S_mass_subtraction_rounding': initial-(diffsq-mi-mj),
                'actual_clamp_change': final-initial,
                'distance_reconstruction_rounding': supplied-(final+mi+mj)}
            require(sum(decomposition.values(), Fraction()) == supplied-physical, 'Error decomposition failed.')
            records.append({'pair_one_based': [pair['i'], pair['j']],
                            'classification': pair['routing']['classification'],
                            'target_composite': pair['routing']['target_p1_minus_p4'],
                            'clamp_branches': [c['branch'] for c in pair['taken_clamps']],
                            'native_distance': pair['native_distance'],
                            'fixed_shift_cartesian': pair['cartesian_exact_squared'],
                            'exact_C_channel': pair['literal_physical_channel_squared'],
                            'native_minus_fixed_shift_cartesian': value(supplied-cartesian),
                            'native_minus_exact_C_channel': value(supplied-physical),
                            'decomposition': {name: value(v) for name, v in decomposition.items()}})
        sg, cg = gram['variants']['supplied'], gram['variants']['exact_cartesian']
        with localcontext() as ctx:
            ctx.prec = 90
            det_change = (dec(fraction(sg['determinant']))-dec(fraction(cg['determinant'])))/dec(fraction(cg['determinant']))
        groups.append({'ctmode': key[0], 'group': key[1], 'n': gram['n'], 'rank': gram['rank'],
                       'reported_subset': gram['members_zero_based'], 'native_condition': gram['captured_condition'],
                       'exact_cartesian_condition': cg['condition'],
                       'supplied_minus_cartesian_determinant_over_cartesian': str(det_change),
                       'internal_mass_values': sorted({str(fraction(p['mass_i']['real'])) for p in pp} |
                                                      {str(fraction(p['mass_j']['real'])) for p in pp}),
                       'pair_records': records})
    result['remaining_tirust_consistency'] = {
        'scope': 'All pairs in each of the51 refused topology-c calls; topology-b native inputs are independently bit-identical.',
        'calls': len(groups), 'all_reported_ill_conditioned': all(g['reported_kind'] == 'ill-conditioned' for g in grams),
        'supplied_exact_singular': sum(g['variants']['supplied']['determinant_is_zero'] for g in grams),
        'cartesian_exact_singular': sum(g['variants']['exact_cartesian']['determinant_is_zero'] for g in grams),
        'remaining_clamp_branches': dict(Counter(str(c['branch']) for p in selected for c in p['taken_clamps'])),
        'groups': groups}
    for name in ('gram.json', 'pairs.jsonl'):
        result['bindings'][rel(matrix_dir/name)] = sha(matrix_dir/name)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({key: {'code': v['host_code'], 'double_pole_relative_error': v['relative_error']}
                      for key, v in result['normalization'].items()}, sort_keys=True))


if __name__ == '__main__':
    main()
