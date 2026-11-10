#!/usr/bin/env python3
"""Portable raw-log analysis; no compilation or numerical execution."""
import argparse
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def parse_row(line):
    v = line.split()
    if len(v) != 16 or v[0] != 'ROW':
        raise ValueError(f'malformed ROW: {line!r}')
    return {'id': v[1], 'mode': v[2], 'numerator': v[3], 'denominator': v[4],
            **dict(zip(('seed', 'requested_round', 'requested_traps', 'before_flags',
                        'after_flags', 'before_round', 'after_round', 'before_traps',
                        'after_traps'), map(int, v[5:14]))), 'tag': v[14], 'bits': v[15]}

def analyze(raw_dir):
    spec = json.loads((HERE / 'spec.json').read_text())
    patterns = spec['numerator_and_denominator_bit_patterns']
    rounds = list(spec['rounding_modes'].values())
    seeds = spec['initial_flag_seeds']
    results = {'schema': 'cayley-zero-guard-extracted-results-v1', 'passed': True,
               'optimization_levels': [], 'limitations': spec['qualification']}
    for opt in spec['optimization_levels']:
        with gzip.open(raw_dir / f'opt{opt}-masked.txt.gz', 'rt') as f:
            rows = [parse_row(line) for line in f if line.strip()]
        grouped = {}
        expected_keys = {(n, d, r, s) for n in patterns for d in patterns for r in rounds for s in seeds}
        for r in rows:
            key = (r['numerator'], r['denominator'], r['requested_round'], r['seed'])
            assert key in expected_keys
            assert r['mode'] in ('baseline', 'candidate')
            pair = grouped.setdefault(key, {})
            assert r['mode'] not in pair
            pair[r['mode']] = r
            assert r['requested_traps'] == r['before_traps'] == r['after_traps'] == 0
            assert r['requested_round'] == r['before_round'] == r['after_round']
            assert r['before_flags'] == r['seed']
            assert r['after_flags'] & r['before_flags'] == r['before_flags']
        assert set(grouped) == expected_keys
        zero = nonzero = changed_flags = infinity_finite_somezero = 0
        for (n, d, _, _), pair in grouped.items():
            assert set(pair) == {'baseline', 'candidate'}
            b, c = pair['baseline'], pair['candidate']
            assert (b['tag'], b['bits']) == (c['tag'], c['bits'])
            if int(d, 16) & 0x7fffffffffffffff == 0:
                zero += 1
                assert c['tag'] == 'none'
                assert c['after_flags'] == c['before_flags']
                changed_flags += b['after_flags'] != c['after_flags']
            else:
                nonzero += 1
                assert b['after_flags'] == c['after_flags']
            if int(d, 16) & 0x7fffffffffffffff == 0x7ff0000000000000 and int(n, 16) & 0x7ff0000000000000 != 0x7ff0000000000000:
                infinity_finite_somezero += 1
                assert c['tag'] == 'some' and int(c['bits'], 16) & 0x7fffffffffffffff == 0
        assert zero == 320 and nonzero == 2880 and changed_flags > 0
        trap_records = [json.loads(x) for x in (raw_dir / f'opt{opt}-traps.jsonl').read_text().splitlines()]
        trap_pairs = {}
        for rec in trap_records:
            assert rec['stderr'] == ''
            assert rec['returncode'] in (0, -8)
            lines = rec['stdout'].splitlines()
            assert lines[0].startswith('ENTER ')
            entry = lines[0].split()
            assert entry[1:3] == [rec['id'], rec['mode']]
            assert tuple(map(int, entry[5:])) == (0, 0, spec['trap_mask'])
            if rec['returncode'] == 0:
                assert len(lines) == 2
                row = parse_row(lines[1])
                assert row['requested_round'] == row['before_round'] == row['after_round'] == 0
                assert row['requested_traps'] == row['before_traps'] == row['after_traps'] == spec['trap_mask']
                rec['row'] = row
            else:
                assert len(lines) == 1
            trap_pairs.setdefault(rec['id'], {})[rec['mode']] = rec
        assert set(trap_pairs) == {x['id'] for x in spec['trap_cases']}
        avoided = 0
        for case in spec['trap_cases']:
            b, c = (trap_pairs[case['id']][k] for k in ('baseline', 'candidate'))
            if int(case['denominator'], 16) & 0x7fffffffffffffff == 0:
                assert c['returncode'] == 0 and c['row']['tag'] == 'none'
                assert c['row']['after_flags'] == 0
                avoided += b['returncode'] == -8
            else:
                assert b['returncode'] == c['returncode']
                if b['returncode'] == 0:
                    assert {k: v for k, v in b['row'].items() if k != 'mode'} == {k: v for k, v in c['row'].items() if k != 'mode'}
        for case in ('one_zero', 'zero_zero', 'overflow_control'):
            assert trap_pairs[case]['baseline']['returncode'] == -8
        assert trap_pairs['overflow_control']['candidate']['returncode'] == -8
        assert avoided == 6
        results['optimization_levels'].append({'opt_level': opt, 'masked_pairs': len(grouped),
            'zero_denominator_pairs': zero, 'nonzero_denominator_pairs_bit_and_flag_exact': nonzero,
            'intentional_zero_divisor_flag_differences': changed_flags,
            'finite_over_infinity_some_signedzero_pairs': infinity_finite_somezero,
            'trap_workers': len(trap_records), 'avoided_zero_divisor_sigfpe': avoided,
            'other_trap_decisions_unchanged': True})
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = json.dumps(analyze(args.raw_dir), indent=2) + '\n'
    if args.output:
        args.output.write_text(out)
    else:
        print(out, end='')
