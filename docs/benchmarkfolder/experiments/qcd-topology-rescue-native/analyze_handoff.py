#!/usr/bin/env python3
"""Offline coordinate/handoff and topology-input comparison; no reducers run.

Consumes all six completed private replays and both retained original-b controls.
Writes one compact report to a new --output path; never overwrites an artifact.
"""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
from common import ROOT, HERE, DESIGN, OLD, ARMS, sha, rel, require, pins, controller_pins

ALL_ARMS = ('original-b', *ARMS)
BASE_TAGS = {'EDGE_INITIAL', 'MASS2_DP', 'NUM_ORIGINAL', 'EDGE_ORDERED', 'MASS2_ORDERED',
             'CLOSURE_SUM', 'CLOSURE_ABS', 'CLOSURE_REF', 'SHIFTS', 'NUM_NATIVE', 'NATIVE_SCALE',
             'PAIR_DIFFSQ', 'PAIR_S_INITIAL', 'PAIR_E_REFERENCE'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()


def load_decoder():
    spec = importlib.util.spec_from_file_location('handoff_frozen_decoder', OLD / 'decode_trace.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def promote64(word, byte_order):
    """Exact finite IEEE binary64 -> binary128, including the zero sign."""
    bits = int(word, 16)
    sign, exponent, tail = bits >> 63, (bits >> 52) & 2047, bits & ((1 << 52) - 1)
    require(exponent != 2047, 'Nonfinite user coordinate cannot certify a finite promotion.')
    if exponent:
        quad = (sign << 127) | ((exponent - 1023 + 16383) << 112) | (tail << 60)
    elif tail:
        leading = tail.bit_length() - 1
        quad = (sign << 127) | ((leading - 1074 + 16383) << 112) | ((tail - (1 << leading)) << (112 - leading))
    else:
        quad = sign << 127
    words = [format(quad & ((1 << 64) - 1), '016X'), format(quad >> 64, '016X')]
    return words if byte_order == 'little' else words[::-1]


def event_one(events, tag):
    selected = [event for event in events if event['tag'] == tag]
    require(len(selected) == 1, 'Expected exactly one ' + tag)
    return selected[0]


def word_record(words):
    return {'word_count': len(words), 'words_sha256': digest(words)}


def coordinate_diff(left, right, byte_order, decoder):
    require(len(left) == len(right) == 40, 'Coordinate vector must have twenty binary128 values.')
    result = []
    for index in range(20):
        a, b = left[2 * index:2 * index + 2], right[2 * index:2 * index + 2]
        if a == b:
            continue
        before, after = decoder.decode_binary128(a, byte_order), decoder.decode_binary128(b, byte_order)
        item = {'mu': index % 4, 'leg': index // 4 + 1, 'left_words': a, 'right_words': b,
                'left_bits_hex': before['bits_hex'], 'right_bits_hex': after['bits_hex']}
        if before['finite'] and after['finite']:
            delta = decoder.finite_fraction(after) - decoder.finite_fraction(before)
            item['right_minus_left_exact'] = {'numerator_hex': hex(delta.numerator),
                                              'denominator_hex': hex(delta.denominator)}
        result.append(item)
    return result


def call_meta(call):
    fields = ('call', 'backend', 'ctmode', 'group', 'sqso', 'n', 'rank',
              'native_status', 'stable', 'fpe', 'fpe_entry', 'outcome', 'stage_statuses')
    return {**{key: call['events'][0][key] for key in ('seq', 'point')},
            **{key: call[key] for key in fields}}


def nonmatrix(call):
    return [{'tag': event['tag'], 'i': event['i'], 'j': event['j'], 'value': event['value'],
             'words': event['words']} for event in call['events']
            if event['tag'] in BASE_TAGS or event['tag'].startswith('CONTEXT_')]


def analyze_point(point, trace, arm, decoder):
    events = point['events']
    user = event_one(events, 'POINT_P_USER')['words']
    promoted = [part for word in user for part in promote64(word, trace['byte_order'])]
    before = [event for event in events if event['tag'] == 'POINT_MP_BEFORE']
    after = [event for event in events if event['tag'] == 'POINT_MP_AFTER']
    prepared_p = [event for event in events if event['tag'] == 'POINT_MP_P']
    triplets = []
    for b, a, p in zip(before, after, prepared_p):
        triplets.append({'ordinals': [b['ordinal'], a['ordinal'], p['ordinal']],
                         'before_matches_exact_user_promotion': b['words'] == promoted,
                         'after_matches_prepared_p': a['words'] == p['words'],
                         'before': word_record(b['words']), 'after': word_record(a['words']),
                         'prepared_p': word_record(p['words']),
                         'repair_coordinate_changes': coordinate_diff(b['words'], a['words'], trace['byte_order'], decoder)})
    expected_count = 1 if point['seq'] == 2 else 0
    requirements = {
        'setter_triplet_count_expected': len(before) == len(after) == len(prepared_p) == expected_count,
        'all_before_exact_original_input_promotion': all(t['before_matches_exact_user_promotion'] for t in triplets),
        'all_after_equals_prepared_p': all(t['after_matches_prepared_p'] for t in triplets),
    }
    calls = []
    c = after[0]['words'] if len(after) == 1 else None
    for call in point['calls']:
        ps = event_one(call['events'], 'CONTEXT_MP_PS')['words']
        p = event_one(call['events'], 'CONTEXT_MP_P')['words']
        matches = c is not None and ps == c and p == c
        calls.append({'metadata_status': call_meta(call),
                      'context_mp_ps': word_record(ps), 'context_mp_p': word_record(p),
                      'both_contexts_equal_single_setter_c': matches,
                      'setter_finished_before_call': len(after) == 1 and after[0]['ordinal'] < call['events'][0]['ordinal'],
                      'nonmatrix_inputs_sha256': digest(nonmatrix(call))})
    requirements['every_qp_call_uses_single_c'] = all(call['both_contexts_equal_single_setter_c'] for call in calls)
    requirements['setter_precedes_every_qp_call'] = all(call['setter_finished_before_call'] for call in calls)
    requirements['stress_has_qp_calls'] = bool(calls) if point['seq'] == 2 else True
    expected_groups = Counter((mode, group, 1) for mode in (4, 5) for group in range(1, 117)) if point['seq'] == 2 else Counter()
    requirements['exact_qp_group_coverage'] = Counter((c['ctmode'], c['group'], c['sqso']) for c in point['calls']) == expected_groups
    return {'seq': point['seq'], 'point': point['point'], 'host_code': point['host_code'],
            'born_coordinate_policy': 'C via frozen source handoff' if arm.endswith('-c') else 'B via frozen source promotion',
            'point_user': word_record(user), 'point_user_exact_words': user,
            'promoted_user_b': word_record(promoted), 'setter_triplet_count': len(triplets),
            'triplets': triplets, 'single_c_exact_words': c,
            'requirements': requirements, 'all_handoff_requirements_pass': all(requirements.values()),
            'calls': calls, 'native_summary': point['summary'], 'health': point['health']}


def call_map(trace):
    result, counts = {}, Counter()
    for point in trace['points']:
        for call in point['calls']:
            base = (point['seq'], call['ctmode'], call['group'], call['sqso'])
            counts[base] += 1
            key = '/'.join(map(str, (*base, counts[base])))
            result[key] = call
    return result


def compare_inputs(left, right):
    lm, rm = call_map(left), call_map(right)
    rows, differences = [], 0
    for key in sorted(set(lm) | set(rm), key=lambda key: tuple(map(int, key.split('/')))):
        a, b = lm.get(key), rm.get(key)
        if a is None or b is None:
            rows.append({'key': key, 'left_present': a is not None, 'right_present': b is not None,
                         'inputs_bit_exact': False})
            differences += 1
            continue
        ap, bp = nonmatrix(a), nonmatrix(b)
        row = {'key': key, 'left_metadata_status': call_meta(a), 'right_metadata_status': call_meta(b),
               'left_inputs_sha256': digest(ap), 'right_inputs_sha256': digest(bp),
               'inputs_bit_exact': ap == bp,
               'shape_metadata_equal': all(a[field] == b[field] for field in ('backend', 'n', 'rank'))}
        if ap != bp:
            # Each selected tag/i/j is unique in these frozen adapters.
            am = {(e['tag'], e['i'], e['j']): e for e in ap}
            bm = {(e['tag'], e['i'], e['j']): e for e in bp}
            require(len(am) == len(ap) and len(bm) == len(bp), 'Duplicate selected input event.')
            row['mismatches'] = [{'tag_i_j': list(tag), 'left': word_record(am[tag]['words']) if tag in am else None,
                                  'right': word_record(bm[tag]['words']) if tag in bm else None,
                                  'left_value': am[tag]['value'] if tag in am else None,
                                  'right_value': bm[tag]['value'] if tag in bm else None}
                                 for tag in sorted(set(am) | set(bm)) if am.get(tag) != bm.get(tag)]
            row['event_order_equal'] = [(e['tag'], e['i'], e['j']) for e in ap] == [(e['tag'], e['i'], e['j']) for e in bp]
        if not row['inputs_bit_exact'] or not row['shape_metadata_equal']:
            differences += 1
        rows.append(row)
    return {'key_fields': ['point_sequence', 'ctmode', 'group', 'squared_order', 'occurrence'],
            'left_calls': len(lm), 'right_calls': len(rm), 'differing_calls': differences,
            'all_selected_inputs_bit_exact': differences == 0, 'calls': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to replace an analysis artifact.')
    pinned = pins()
    pinned.update(controller_pins())
    arm_sources = json.loads((DESIGN / 'arms.json').read_text())['arms']
    decoder = load_decoder()
    traces, records, inputs = {}, {}, {}
    for arm in ALL_ARMS:
        for backend in (8, 9):
            directory = (OLD / 'attempt1' if arm == 'original-b' else HERE / 'attempt1' / arm) / ('backend-' + str(backend))
            manifest_path = directory / 'execution.json'
            manifest = json.loads(manifest_path.read_text())
            require(manifest['complete'] and manifest['protected_inputs_unchanged'], 'Replay is incomplete: ' + rel(directory))
            require(manifest['protected_inputs_before'] == manifest['protected_inputs_after'], 'Replay protected bytes differ.')
            require(manifest['runtime_sha256_before'] == manifest['runtime_sha256_after'], 'Replay runtime bytes differ.')
            for filename, field in (('trace.dat', 'trace_sha256'), ('observed.log', 'log_sha256')):
                require(sha(directory / filename) == manifest[field], 'Retained trace/log hash differs.')
            build_path = directory.parent / 'build.json'
            build = json.loads(build_path.read_text())
            require(build['complete'] and build['protected_inputs_unchanged'], 'Incomplete source build.')
            if arm != 'original-b':
                require(manifest['build_sha256'] == sha(build_path), 'Replay/build binding differs.')
                for name, entry in arm_sources[arm]['sources'].items():
                    matched = [value for path, value in build['artifacts_sha256'].items() if Path(path).name == name]
                    require(matched == [entry['sha256']], 'Built source differs from selected arm: ' + name)
            for path in (manifest_path, build_path, directory / 'trace.dat', directory / 'observed.log'):
                inputs[rel(path)] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
            trace = decoder.load_trace(directory / 'trace.dat', directory / 'observed.log')
            require(all(call['backend'] == backend for point in trace['points'] for call in point['calls']), 'Wrong traced backend.')
            key = arm + '/backend-' + str(backend)
            traces[key] = trace
            records[key] = {'arm': arm, 'backend': backend, 'byte_order': trace['byte_order'],
                            'source_manifest_sha256': digest(arm_sources[arm]),
                            'points': [analyze_point(point, trace, arm, decoder) for point in trace['points']]}
    comparisons, coordinate_comparisons = {}, {}
    for backend in (8, 9):
        suffix = '/backend-' + str(backend)
        for control, variant in (('original-b', 'topology-b'), ('original-c', 'topology-c')):
            comparisons[variant + '_vs_' + control + suffix] = compare_inputs(traces[control + suffix], traces[variant + suffix])
        historical = records['original-b' + suffix]['points'][1]
        for variant in ARMS:
            changed = records[variant + suffix]['points'][1]
            before, after = historical['single_c_exact_words'], changed['single_c_exact_words']
            coordinate_comparisons[variant + '_vs_original-b' + suffix] = {
                'original_user_input_bit_exact': historical['point_user_exact_words'] == changed['point_user_exact_words'],
                'both_have_one_setter_c': before is not None and after is not None,
                'setter_c_bit_exact': before == after if before is not None and after is not None else False,
                'historical_c': word_record(before) if before is not None else None,
                'new_c': word_record(after) if after is not None else None,
                'changed_coordinates': coordinate_diff(before, after, records[variant + suffix]['byte_order'], decoder)
                                       if before is not None and after is not None else None}
    report = {'schema': 1, 'offline_analysis_complete': True, 'physics_accepted': False,
              'native_executed_by_analysis': False, 'script': {'path': rel(Path(__file__)), 'sha256': sha(Path(__file__))},
              'decoder_sha256': sha(OLD / 'decode_trace.py'), 'frozen_pin_count': len(pinned),
              'frozen_pins_sha256': digest(pinned), 'inputs': inputs, 'arms': records,
              'topology_input_comparisons': comparisons, 'historical_vs_new_coordinates': coordinate_comparisons,
              'all_handoff_requirements_pass': all(point['all_handoff_requirements_pass'] for record in records.values() for point in record['points']),
              'all_topology_only_input_comparisons_pass': all(item['all_selected_inputs_bit_exact'] for item in comparisons.values()),
              'limitations': [
                  'Born arguments are not observed directly: C use is established by the pinned wrapper call/setter/copy source and the observed single-setter handoff.',
                  'The B control keeps promoted original DP coordinates for Born; loops use the actual setter output C.',
                  'The C arms retain the same promoted-DP Born parameter policy; actual MP loop parameter differences remain unresolved.',
                  'No shell/conservation repair success, numerical accuracy, physical acceptance, or performance conclusion follows from coordinate/input identity.',
                  'Native status/stability/FPE and matrix/output changes are recorded independently of the nonmatrix-input equality requirement.',
                  'Word digests hash canonical JSON word arrays, not numerical floating-point conversions.'
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write('\n')
    print(json.dumps({'output': rel(args.output.resolve()), 'sha256': sha(args.output),
                      'all_handoff_requirements_pass': report['all_handoff_requirements_pass'],
                      'all_topology_only_input_comparisons_pass': report['all_topology_only_input_comparisons_pass']}))


if __name__ == '__main__':
    main()
