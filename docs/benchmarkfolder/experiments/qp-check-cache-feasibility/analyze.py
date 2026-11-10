#!/usr/bin/env python3
"""Recount exact check-tensor input repeats in already frozen physical traces."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DECODER = BASE / 'qcd-loop-kinematics-observer-native'
NATIVE = BASE / 'qcd-topology-rescue-native'
SOURCE = BASE / 'qcd-topology-rescue-prototype'
PINS = {
    DECODER: '6a2e48f62ee48ebd2a892da18f797249e686be660225e9d05c66415f98e8f5ac',
    NATIVE: 'c7d185c4991fdfacf5c76710d1790b7ccf31c5c2ec0f9270cc4f72ec3dfdfc32',
    SOURCE: '0af03d980acdac032abcb767713bf5906fa7fa53cf8b3c57da75b764281e3015',
}
TAGS = ('SHIFTS', 'DISTANCES', 'MASS2_ORDERED', 'NATIVE_SCALE')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(folder, expected):
    assert sha(folder / 'evidence-index.json') == expected
    for name, entry in json.loads((folder / 'evidence-index.json').read_text())['files'].items():
        path = folder / name
        assert not path.is_symlink() and not Path(name).is_absolute() and '..' not in Path(name).parts
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], str(path)


def event(call, tag):
    values = [e for e in call['events'] if e['tag'] == tag]
    assert len(values) == 1
    return values[0]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def key(call):
    # These are exact binary128 components before deterministic ES64.39E4
    # packing, retaining sign bits even on zero. Numerator/group/SQSO are not
    # tensor-integral inputs. Directions are analyzed in disjoint cohorts.
    return digest({'n': call['n'], 'rank': call['rank'], 'pole_mode': 0,
                   'compute_finite': True, 'input_bits': 113, 'working_bits': 256,
                   'fields': {tag: event(call, tag)['words'] for tag in TAGS}})


def compute():
    for folder, expected in PINS.items():
        inventory(folder, expected)
    spec = importlib.util.spec_from_file_location('frozen_check_cache_decoder', DECODER / 'decode_trace.py')
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    bindings = {str((folder / 'evidence-index.json').relative_to(ROOT)): expected for folder, expected in PINS.items()}
    bindings[str((DECODER / 'decode_trace.py').relative_to(ROOT))] = sha(DECODER / 'decode_trace.py')
    rows = []
    for label, folder in [('original-b', DECODER / 'attempt1/backend-9'),
                          ('topology-c', NATIVE / 'attempt1/topology-c/backend-9')]:
        record = json.loads((folder / 'execution.json').read_text())
        assert record['complete'] and record['protected_inputs_unchanged']
        assert record['runtime_sha256_before'] == record['runtime_sha256_after']
        for filename, field in [('trace.dat', 'trace_sha256'), ('observed.log', 'log_sha256')]:
            assert sha(folder / filename) == record[field]
        for filename in ['execution.json', 'trace.dat', 'observed.log']:
            bindings[str((folder / filename).relative_to(ROOT))] = sha(folder / filename)
        trace = decoder.load_trace(folder / 'trace.dat', folder / 'observed.log')
        for point in trace['points']:
            for mode in (4, 5):
                calls = [call for call in point['calls'] if call['ctmode'] == mode]
                grouped = {}
                for call in calls:
                    grouped.setdefault(key(call), []).append(call)
                duplicates = []
                for input_key, same in grouped.items():
                    if len(same) < 2:
                        continue
                    success = [call for call in same if call['native_status'] == 0]
                    moments = [event(call, 'NATIVE_MOMENTS')['words'] for call in success]
                    duplicates.append({
                        'input_key_sha256': input_key,
                        'n': same[0]['n'], 'rank': same[0]['rank'],
                        'call_ordinals': [c['call'] for c in same],
                        'groups': [c['group'] for c in same],
                        'sqso': [c['sqso'] for c in same],
                        'native_statuses': [c['native_status'] for c in same],
                        'stable': [bool(c['stable']) for c in same],
                        'wrapper_fpe_entry': [c['fpe_entry'] for c in same],
                        'wrapper_fpe_exit': [c['fpe'] for c in same],
                        'successful_moments_sha256': [digest(words) for words in moments],
                        'successful_moments_bit_identical': len({digest(words) for words in moments}) <= 1,
                        'candidate_successful_repeats': max(0, len(success) - 1),
                        'different_numerator_words': len({digest(event(c, 'NUM_ORIGINAL')['words']) for c in same}) > 1,
                    })
                rows.append({'arm': label, 'point_sequence': point['seq'], 'point_id': point['point'],
                             'host_code': point['host_code'], 'ctmode': mode,
                             'check_pass': mode == 5, 'calls': len(calls),
                             'distinct_input_keys': len(grouped),
                             'extra_repeated_inputs': len(calls) - len(grouped),
                             'candidate_successful_repeats': sum(r['candidate_successful_repeats'] for r in duplicates),
                             'group_visit_counts': {str(k): v for k, v in sorted(Counter(c['group'] for c in calls).items())},
                             'sqso_counts': {str(k): v for k, v in sorted(Counter(c['sqso'] for c in calls).items())},
                             'duplicate_sets': duplicates})
    assert all(r['successful_moments_bit_identical'] for row in rows for r in row['duplicate_sets'])
    return {'schema': 1, 'native_executed': False, 'cache_implemented': False,
            'timing_observed': False, 'physics_accepted': False,
            'key_contract': 'Exact binary128 tensor inputs and fixed N/rank/mode/precision, separate CT4 and CT5 cohorts; no numerator or group identity.',
            'rows': rows, 'bindings': bindings,
            'limits': ['This is input/output-repeat counting, not measured cache hits or speedup.',
                       'The retained ordinary points use DP and contain zero QP calls.',
                       'All observed QP calls have SQSO1 and one visit per group/direction; this corpus does not measure split-order or helicity repetition.',
                       'CT4 primary calls may already use the historical cache; CT5 bypasses it, so CT5 duplicate native outputs are independent repeated evaluations.',
                       'Wrapper FPE records are sticky common flags, not per-kernel MPFR or complete host IEEE delta snapshots.',
                       'Both stress amplitudes retain host420 and are not accepted physical results.',
                       'No cross-direction, cross-rotation, cross-event or failed-result reuse is proposed or counted.']}


if __name__ == '__main__':
    answer = compute()
    output = HERE / 'counts.json'
    if '--verify' in sys.argv:
        assert answer == json.loads(output.read_text())
    else:
        assert not output.exists()
        output.write_text(json.dumps(answer, indent=2, sort_keys=True) + '\n')
    print(json.dumps([{'arm': r['arm'], 'ctmode': r['ctmode'], 'calls': r['calls'],
                       'keys': r['distinct_input_keys'], 'success_repeats': r['candidate_successful_repeats']}
                      for r in answer['rows'] if r['calls']], sort_keys=True))
