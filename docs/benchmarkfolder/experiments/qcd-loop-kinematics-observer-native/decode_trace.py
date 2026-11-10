#!/usr/bin/env python3
"""Strict decoder for the frozen QCD observer schema; no native evaluation.

load_trace(trace_path, stdout_path) returns JSON-serializable points/calls with
exact hexadecimal words and structural summaries, or raises TraceError.
decode_payload(event, byte_order) decodes components to exact dyadic rationals
on demand. finite_fraction(component) returns a Fraction, never a float.
Binary128 decimal conversion is deliberately avoided, including for subnormals.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re


class TraceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise TraceError(message)


META = ('kind seq point call backend ctmode group sqso n rank i j value word_count '
        'word_first error_index native_present native_status stable fpe tag_length').split()
HEADER = [1, 200000, 2000000, 4096, 1024, 64, 21, 64]
HEX_WORD = re.compile(r'[0-9A-F]{16}\Z')
EXPECTED_POINTS = [[1, 1], [2, 2], [3, 1]]
FORTRAN_REAL = re.compile(r'(?:[+-]?(?:\d+\.?\d*|\.\d+)(?:[ED][+-]?\d+)?|[+-]?(?:NaN|Inf(?:inity)?))\Z', re.IGNORECASE)
CONTEXT = ['CONTEXT_MP_PS', 'CONTEXT_MP_P']
CONTEXT += [f'CONTEXT_{mass}_{kind}' for kind in ('DP', 'MP') for mass in ('MZ', 'MT', 'MB')]
CONTEXT += ['CONTEXT_G_DP', 'CONTEXT_G_MP', 'CONTEXT_MU_DP', 'CONTEXT_MU_MP']
CONTEXT += [f'CONTEXT_GC{coupling}_{kind}' for kind in ('DP', 'MP') for coupling in (4, 5, 22, 23)]


def integers(tokens, context):
    require(all(re.fullmatch(r'-?\d+', token) for token in tokens), f'{context}: invalid integer')
    return [int(token) for token in tokens]


def ieee_component(bits, width, rational=True):
    """Decode IEEE binary64/128 bits; rational=False omits the optional fraction."""
    require(width in (64, 128), 'unsupported IEEE width')
    require(0 <= bits < (1 << width), 'IEEE bit pattern outside width')
    fraction_bits, exponent_bits, bias = (52, 11, 1023) if width == 64 else (112, 15, 16383)
    sign = bits >> (width - 1)
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits) - 1)
    tail = bits & ((1 << fraction_bits) - 1)
    result = {'width': width, 'bits_hex': f'{bits:0{width // 4}X}', 'sign': sign,
              'exponent_field': exponent, 'fraction_bits_hex': hex(tail)}
    if exponent == (1 << exponent_bits) - 1:
        result['classification'] = ('infinity' if tail == 0 else
                                    'quiet_nan' if tail & (1 << (fraction_bits - 1)) else 'signaling_nan')
        result['finite'] = False
        return result
    result['finite'] = True
    result['classification'] = 'normal' if exponent else 'subnormal' if tail else 'zero'
    significand = tail + ((1 << fraction_bits) if exponent else 0)
    result['significand_hex'] = hex(-significand if sign else significand)
    result['exponent2'] = (exponent - bias if exponent else 1 - bias) - fraction_bits
    result['negative_zero'] = bool(sign and significand == 0)
    if rational:
        value = finite_fraction(result)
        result['rational'] = {'numerator_hex': hex(value.numerator), 'denominator_hex': hex(value.denominator)}
    return result


def finite_fraction(component):
    require(component.get('finite') is True, 'nonfinite IEEE value has no Fraction')
    significand, exponent = int(component['significand_hex'], 16), component['exponent2']
    return Fraction(significand << exponent, 1) if exponent >= 0 else Fraction(significand, 1 << -exponent)


def decode_binary64(word):
    return ieee_component(int(word, 16) if isinstance(word, str) else word, 64)


def decode_binary128(words, byte_order='little'):
    require(len(words) == 2, 'binary128 requires exactly two INT64 words')
    values = [int(word, 16) if isinstance(word, str) else word for word in words]
    require(all(isinstance(word, int) and 0 <= word < (1 << 64) for word in values),
            'binary128 input word outside INT64 bit width')
    require(byte_order in ('little', 'big'), 'unknown native byte order')
    bits = values[0] | (values[1] << 64) if byte_order == 'little' else (values[0] << 64) | values[1]
    return ieee_component(bits, 128)


def payload_schema(event):
    """Return (real-component width, complex flag, expected words, shape)."""
    tag, n, rank, backend = event['tag'], event['n'], event['rank'], event['backend']
    nc = (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) // 24
    if tag in ('EDGE_INITIAL', 'EDGE_ORDERED', 'SHIFTS'):
        return 128, False, 8 * n, [4, n]
    if tag == 'MASS2_DP':
        return 64, True, 2 * n, [n]
    if tag == 'MASS2_ORDERED':
        return 128, True, 4 * n, [n]
    if tag in ('NUM_ORIGINAL', 'NUM_NATIVE'):
        return 128, True, 4 * nc, [nc]
    if tag in ('CLOSURE_SUM', 'CLOSURE_ABS'):
        return 128, False, 8, [4]
    if tag in ('CLOSURE_REF', 'NATIVE_SCALE', 'PAIR_E_REFERENCE'):
        return 128, False, 2, []
    if tag == 'S_COMPLEX':
        return 128, True, 4 * n * n, [n, n]
    if tag == 'DISTANCES':
        return 128, False, 2 * n * n, [n, n]
    if tag in ('PAIR_DIFFSQ', 'PAIR_S_INITIAL', 'PAIR_S_CLAMPED', 'PAIR_S_FINAL',
               'CONTRACTION_RAW', 'CONTRACTION_PROJECTED'):
        return 128, True, 4, []
    if tag == 'NATIVE_CONTRACTED':
        return 128, True, 12, [3]
    if tag == 'NATIVE_MOMENTS':
        return 128, True, 16 * nc, [nc, 4]
    if tag == 'MAPPED_MOMENTS':
        return 128, True, 12 * nc, [nc, 3]
    if tag == 'DP_CONVERTED':
        return (64, True, 6, [3]) if backend == 8 else (64, True, 2, [])
    if tag in ('CONTEXT_MP_PS', 'CONTEXT_MP_P', 'POINT_MP_BEFORE', 'POINT_MP_AFTER', 'POINT_MP_P'):
        return 128, False, 40, [4, 5]
    if tag == 'POINT_P_USER':
        return 64, False, 20, [4, 5]
    if tag == 'POINT_HOST_ANS':
        return 64, False, 8, [4, 2]
    if tag == 'POINT_HOST_ACC':
        return 64, False, 2, [2]
    if tag in ('POINT_G_DP', 'POINT_MU_DP'):
        return 64, False, 1, []
    if re.fullmatch(r'CONTEXT_(MZ|MT|MB|G|MU)_DP', tag):
        return 64, False, 1, []
    if re.fullmatch(r'CONTEXT_(MZ|MT|MB|G)_MP', tag):
        return 128, False, 2, []
    if tag == 'CONTEXT_MU_MP':  # MP__MU_R is COMPLEX*32 in the frozen model.
        return 128, True, 4, []
    if re.fullmatch(r'CONTEXT_GC(4|5|22|23)_DP', tag):
        return 64, True, 2, []
    if re.fullmatch(r'CONTEXT_GC(4|5|22|23)_MP', tag):
        return 128, True, 4, []
    raise TraceError(f'unknown payload tag: {tag}')


def components(event, byte_order, rational):
    require(byte_order in ('little', 'big'), 'unknown native byte order')
    width, is_complex, count, _ = payload_schema(event)
    require(len(event['words']) == count, f'{event["tag"]}: wrong payload size')
    step = width // 64
    values = [int(word, 16) for word in event['words']]
    for offset in range(0, count, step):
        bits = values[offset]
        if step == 2:
            bits = (bits | (values[offset + 1] << 64)) if byte_order == 'little' else ((bits << 64) | values[offset + 1])
        yield ieee_component(bits, width, rational), (offset // step), is_complex


def decode_payload(event, byte_order='little'):
    """Return exact real components, paired for complex elements, in Fortran order."""
    values = [value for value, _, _ in components(event, byte_order, True)]
    if payload_schema(event)[1]:
        return [{'real': values[i], 'imag': values[i + 1]} for i in range(0, len(values), 2)]
    return values


def payload_summary(event, byte_order):
    width, is_complex, count, shape = payload_schema(event)
    classes, nonfinite, negative_zeros = Counter(), [], 0
    for value, index, _ in components(event, byte_order, False):
        classes[value['classification']] += 1
        negative_zeros += int(value.get('negative_zero', False))
        if not value['finite']:
            nonfinite.append({'component_index': index, 'element_index': index // 2 if is_complex else index,
                              'part': ('real' if index % 2 == 0 else 'imag') if is_complex else 'real',
                              'classification': value['classification'], 'bits_hex': value['bits_hex']})
    return {'component_width': width, 'complex': is_complex, 'shape': shape, 'order': 'Fortran column-major',
            'words': count, 'all_finite': not nonfinite, 'classes': dict(sorted(classes.items())),
            'negative_zero_components': negative_zeros, 'nonfinite_components': nonfinite}


class Lines:
    def __init__(self, text):
        self.lines = text.splitlines()
        self.index = 0

    def next(self):
        require(self.index < len(self.lines), 'unexpected end of trace')
        line = self.lines[self.index]
        self.index += 1
        require(line.strip(), f'trace line {self.index}: empty record')
        return line

    def numeric(self, tag, length=None):
        tokens = self.next().split()
        require(tokens and tokens[0] == tag, f'line {self.index}: expected {tag}')
        values = integers(tokens[1:], tag)
        require(length is None or len(values) == length, f'{tag}: wrong field count')
        return values


def parse_point(lines, byte_order, seen_sequences):
    head = lines.numeric('QO_POINT', 13)
    seq, point_id, call_count, event_count, word_count, error_count = head[:6]
    require(seq > 0 and seq not in seen_sequences, 'duplicate/nonpositive point sequence')
    require(seq == len(seen_sequences) + 1, 'point sequence is not contiguous')
    seen_sequences.add(seq)
    require(0 <= call_count <= event_count <= HEADER[1] and 0 <= word_count <= HEADER[2]
            and 0 <= error_count <= HEADER[3], 'point counts exceed bounds')
    require(head[6:] == [0] * 7, f'point {seq}: observer fault/drop counters are nonzero')
    events, used_words, references = [], 0, []
    for ordinal in range(1, event_count + 1):
        parts = lines.next().split(maxsplit=23)
        require(len(parts) in (23, 24) and parts[0] == 'QO_EVENT', 'malformed event record')
        values = integers(parts[1:23], 'QO_EVENT')
        require(values[0] == ordinal, 'event ordinal gap/duplicate')
        event = dict(zip(META, values[1:]))
        event.update({'ordinal': ordinal, 'tag': parts[23].rstrip(' ') if len(parts) == 24 else '', 'words': []})
        require(event['seq'] == seq and event['point'] == point_id, 'event point context differs')
        require(len(event['tag']) == event['tag_length'] <= 64 and all(32 <= ord(c) <= 126 for c in event['tag']),
                'event tag length/encoding mismatch')
        require(event['kind'] in range(1, 9), 'observer fault/abandoned/unknown event kind')
        count = event['word_count']
        require(count >= 0 and count <= word_count - used_words, 'invalid payload word count')
        require(event['word_first'] == (used_words + 1 if count else 0), 'payload offset gap/overlap')
        require((event['kind'] == 4) or count == 0, 'non-payload event contains words')
        if count:
            tokens = lines.next().split()
            require(len(tokens) == count + 3 and tokens[0] == 'QO_WORDS', 'missing/malformed word record')
            require(integers(tokens[1:3], 'QO_WORDS') == [ordinal, count], 'word record refers to wrong event')
            require(all(HEX_WORD.fullmatch(word) for word in tokens[3:]), 'noncanonical INT64 hex word')
            event['words'] = tokens[3:]
            used_words += count
        if event['kind'] == 4:
            require(count > 0, 'frozen payload tag has an empty payload')
            event['numerical_summary'] = payload_summary(event, byte_order)
        if event['kind'] == 7:
            references.append(event['error_index'])
        else:
            require(event['error_index'] == 0, 'non-error event references an error')
        events.append(event)
    require(used_words == word_count, 'unaccounted payload words')
    require(references == list(range(1, error_count + 1)), 'error reference gap/duplicate')
    errors = []
    for index in range(1, error_count + 1):
        descriptor = lines.numeric('QO_ERROR', 3)
        require(descriptor[0] == index and 0 <= descriptor[1] == descriptor[2] <= 1024, 'invalid/truncated error text')
        codes = lines.numeric('QO_ERROR_BYTES')
        require(codes and codes[0] == index and len(codes) == descriptor[1] + 1 and all(0 <= c <= 255 for c in codes[1:]),
                'error bytes/count mismatch')
        raw = bytes(codes[1:])
        errors.append({'index': index, 'bytes_hex': raw.hex(), 'text': raw.decode('utf-8', errors='replace')})
    foot = lines.numeric('QO_POINT_END', 13)
    require(foot[:2] == head[:2] and foot[3:] == head[3:], 'footer does not match point header/counters')
    for event in events:
        if event['kind'] == 7:
            error = errors[event['error_index'] - 1]
            require(event['value'] == len(bytes.fromhex(error['bytes_hex'])), 'error event length mismatch')
            event['error'] = error
    point = {'seq': seq, 'point': point_id, 'host_code': foot[2], 'declared_calls': call_count,
             'event_count': event_count, 'word_count': word_count, 'error_count': error_count,
             'observer_fault_counters': head[6:], 'events': [], 'calls': []}
    validate_scopes(events, point)
    return point


def validate_scopes(events, point):
    require(events and events[0]['kind'] == 1 and events[-1]['kind'] == 2, 'missing point boundary events')
    current, calls = None, []
    for event in events:
        kind, tag = event['kind'], event['tag']
        point_scope = kind in (1, 2) or (kind in (4, 5) and tag.startswith('POINT_'))
        if point_scope:
            require(all(event[key] == 0 for key in ('call', 'backend', 'ctmode', 'group', 'sqso', 'n', 'rank',
                                                   'native_present', 'native_status')) and event['stable'] == event['fpe'] == -1,
                    'point-scoped event contains call context')
            if kind in (1, 2):
                require(current is None, 'point boundary inside active call')
                require(event['i'] == event['j'] == 0, 'point boundary has nonzero indices')
            point['events'].append(event)
            continue
        if kind == 3:
            require(current is None and tag == 'CALL_BEGIN', 'nested/invalid call begin')
            require(event['call'] == len(calls) + 1, 'call ordinal gap/duplicate')
            require(event['backend'] in (8, 9) and event['ctmode'] in (4, 5) and 1 <= event['group'] <= 116
                    and event['sqso'] == 1 and 2 <= event['n'] <= 5 and 0 <= event['rank'] <= 4,
                    'call context outside frozen QCD export')
            current = {key: event[key] for key in ('call', 'backend', 'ctmode', 'group', 'sqso', 'n', 'rank')}
            current.update({'events': [], 'native_status': None})
        require(current is not None, 'call event outside an active call')
        require(all(event[key] == current[key] for key in ('call', 'backend', 'ctmode', 'group', 'sqso', 'n', 'rank')),
                'call context changes within a call')
        if kind == 6:
            require(tag == 'NATIVE_STATUS' and current['native_status'] is None, 'duplicate/malformed native status')
            require(0 <= event['value'] <= (2 if current['backend'] == 8 else 3), 'unknown backend status')
            current['native_status'] = event['value']
        status = current['native_status']
        require(event['native_present'] == int(status is not None) and event['native_status'] == (status or 0),
                'native status presence/value snapshot mismatch')
        if kind != 8:
            require(event['stable'] == event['fpe'] == -1, 'final flags appear before call end')
        else:
            require(tag == 'CALL_END' and event['stable'] in (0, 1) and event['fpe'] in (0, 1), 'invalid call end')
            current['stable'], current['fpe'] = event['stable'], event['fpe']
        current['events'].append(event)
        if kind == 8:
            validate_call(current)
            calls.append(current)
            current = None
    require(current is None and len(calls) == point['declared_calls'], 'unclosed/missing call')
    point['calls'] = calls
    cursor = Cursor(point['events'])
    cursor.take('POINT_BEGIN', 1, value=0)
    cursor.take('POINT_P_USER', 4)
    while cursor.peek() == 'POINT_MP_BEFORE':
        for tag in ('POINT_MP_BEFORE', 'POINT_MP_AFTER', 'POINT_MP_P'):
            cursor.take(tag, 4)
    for tag in ('POINT_HOST_ANS', 'POINT_HOST_ACC', 'POINT_G_DP', 'POINT_MU_DP'):
        cursor.take(tag, 4)
    for tag in ('POINT_INVALID', 'POINT_DIVZERO', 'POINT_OVERFLOW'):
        require(cursor.take(tag, 5)['value'] in (0, 1), 'invalid point IEEE flag')
    cursor.take('POINT_END', 2, value=point['host_code'])
    require(cursor.done(), 'unrecognized point-scope record')
    if calls:
        user_input = next(e['ordinal'] for e in point['events'] if e['tag'] == 'POINT_P_USER')
        host_output = next(e['ordinal'] for e in point['events'] if e['tag'] == 'POINT_HOST_ANS')
        require(user_input < calls[0]['events'][0]['ordinal']
                and calls[-1]['events'][-1]['ordinal'] < host_output,
                'call occurs outside the user-input/host-output interval')


class Cursor:
    def __init__(self, events):
        self.events, self.index = events, 0

    def peek(self):
        return self.events[self.index]['tag'] if self.index < len(self.events) else None

    def done(self):
        return self.index == len(self.events)

    def take(self, tag, kind, i=0, j=0, value=None):
        require(self.index < len(self.events), f'missing stage {tag}')
        event = self.events[self.index]
        require(event['tag'] == tag and event['kind'] == kind and event['i'] == i and event['j'] == j,
                f'event {event["ordinal"]}: expected {tag}/{kind}/{i}/{j}, got {event["tag"]}/{event["kind"]}/{event["i"]}/{event["j"]}')
        if value is not None:
            require(event['value'] == value, f'{tag}: metadata value differs')
        elif kind in (1, 2, 3, 4, 8):
            require(event['value'] == 0, f'{tag}: unexpected metadata value')
        self.index += 1
        return event


def validate_call(call):
    cursor = Cursor(call['events'])
    pairs, markers = [], []
    cursor.take('CALL_BEGIN', 3)
    call['fpe_entry'] = cursor.take('FPE_ENTRY', 5)['value']
    require(call['fpe_entry'] in (0, 1), 'invalid FPE entry flag')
    for tag in ('EDGE_INITIAL', 'MASS2_DP', 'NUM_ORIGINAL', *CONTEXT):
        cursor.take(tag, 4)

    def finish(reason, early=False):
        cursor.take('CALL_END', 8)
        require(cursor.done(), 'events after call end')
        if early:
            require(call['stable'] == 0 and call['fpe'] == 1, 'early refusal missing failure flags')
        require(call['stable'] or call['fpe'] == 1, 'unstable call missing final FPE flag')
        require(call['fpe'] >= call['fpe_entry'], 'call cleared the sticky wrapper FPE flag')
        call['outcome'] = reason
        call['pair_records'] = pairs
        call['stage_statuses'] = markers
        call['numerical_summaries'] = {event['tag']: event['numerical_summary'] for event in call['events']
                                       if event['tag'] in ('NUM_ORIGINAL', 'NUM_NATIVE', 'NATIVE_MOMENTS', 'NATIVE_CONTRACTED')}
        call['taken_clamps'] = [{'i': pair['i'], 'j': pair['j'], 'branches': pair['branches']}
                                for pair in pairs if pair['branches']]

    if call['backend'] == 9:
        cursor.take('RANK_SUPPORT_CHECK', 5, value=call['rank'])
        if cursor.peek() == 'CALL_END':
            return finish('rank_or_support_refusal_before_native', True)
        for tag in ('INIT_STATUS', 'BUDGET_STATUS'):
            event = cursor.take(tag, 5)
            require(event['value'] in (0, 1, 2, 3), f'{tag}: unknown bridge status')
            markers.append({'tag': tag, 'status': event['value']})
            if event['value'] != 0:
                return finish(tag.lower() + '_refusal_before_native', True)
    for tag in ('EDGE_ORDERED', 'MASS2_ORDERED'):
        cursor.take(tag, 4)
    cursor.take('ORDER_DONE', 5, value=0)
    for tag in ('CLOSURE_SUM', 'CLOSURE_ABS', 'CLOSURE_REF'):
        cursor.take(tag, 4)
    cursor.take('CLOSURE_CHECK', 5, value=0)
    if cursor.peek() == 'CALL_END':
        return finish('closure_refusal_before_native', True)
    cursor.take('SHIFTS', 4)
    for i in range(1, call['n'] + 1):
        for j in range(1, call['n'] + 1):
            pair = {'i': i, 'j': j, 'branches': [], 'event_ordinals': []}

            def take_pair(tag, kind, value=None):
                event = cursor.take(tag, kind, i, j, value)
                pair['event_ordinals'].append(event['ordinal'])
                return event

            if i == j:
                take_pair('PAIR_DIAGONAL', 5, 0)
            else:
                take_pair('PAIR_DIFFSQ', 4)
                take_pair('PAIR_S_INITIAL', 4)
                for branch in (1, 2):
                    if cursor.peek() == 'PAIR_CLAMP' and cursor.events[cursor.index]['value'] == branch:
                        take_pair('PAIR_CLAMP', 5, branch)
                        take_pair('PAIR_S_CLAMPED', 4)
                        pair['branches'].append(branch)
                take_pair('PAIR_E_REFERENCE', 4)
                if cursor.peek() == 'PAIR_CLAMP':
                    take_pair('PAIR_CLAMP', 5, 3)
                    take_pair('PAIR_S_CLAMPED', 4)
                    pair['branches'].append(3)
            take_pair('PAIR_S_FINAL', 4)
            pairs.append(pair)
    for tag in ('S_COMPLEX', 'DISTANCES'):
        cursor.take(tag, 4)
    if call['backend'] == 8:
        cursor.take('NUM_NATIVE', 4)
    cursor.take('NATIVE_SCALE', 4)
    cursor.take('NATIVE_BEGIN', 5, value=0)
    cursor.take('NATIVE_STATUS', 6, value=call['native_status'])
    if call['native_status'] != 0:
        cursor.take('BACKEND_ERROR', 7)
        cursor.take('NATIVE_RETURN', 5, value=0)
        return finish('native_refusal', True)
    native = cursor.take('NATIVE_CONTRACTED' if call['backend'] == 8 else 'NATIVE_MOMENTS', 4)
    cursor.take('NATIVE_RETURN', 5, value=0)
    if cursor.peek() == 'CALL_END':
        return finish('adapter_refusal_after_native_success', True)
    if call['backend'] == 8:
        converted = [cursor.take('DP_CONVERTED', 4)]
    else:
        cursor.take('MAPPED_MOMENTS', 4)
        converted = []
        for i in (1, 2, 3):
            cursor.take('CONTRACTION_RAW', 4, i)
            cursor.take('CONTRACTION_PROJECTED', 4, i)
            converted.append(cursor.take('DP_CONVERTED', 4, i))
    if call['stable']:
        require(native['numerical_summary']['all_finite'] and all(e['numerical_summary']['all_finite'] for e in converted),
                'stable call has nonfinite native/converted output')
    finish('stable' if call['stable'] else 'post_native_contraction_or_conversion_refusal')


def validate_stdout(text, points):
    health, begins, values, records, complete = [], [], [], [], 0
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == 'QO_HEALTH':
            require(len(tokens) == 4, 'malformed health acknowledgement')
            health.append(integers(tokens[1:], 'QO_HEALTH'))
            records.append(('health', health[-1][0]))
        elif tokens[0] == 'AMPLITUDE_BEGIN':
            require(len(tokens) == 3, 'malformed amplitude begin')
            begins.append(integers(tokens[1:], 'AMPLITUDE_BEGIN'))
            records.append(('begin', begins[-1][0]))
        elif tokens[0] == 'AMPLITUDE_VALUE':
            require(len(tokens) == 13, 'malformed amplitude value record')
            require(all(FORTRAN_REAL.fullmatch(t) for t in tokens[5:10]), 'malformed Fortran real in amplitude value')
            require(all(t in ('T', 'F') for t in tokens[10:]), 'malformed amplitude IEEE flag')
            identifiers = integers(tokens[1:5], 'AMPLITUDE_VALUE')
            values.append({'identifiers': identifiers, 'numeric_tokens': tokens[5:10],
                           'flags': [int(t == 'T') for t in tokens[10:]]})
            records.append(('value', identifiers[0], identifiers[2]))
        elif tokens[0] == 'AMPLITUDE_DIAGNOSTIC_COMPLETE':
            require(len(tokens) == 1, 'malformed driver completion marker')
            complete += 1
            records.append(('complete',))
    require(health == [[p['seq'], 0, 0] for p in points], 'missing/extra/nonzero/out-of-order health acknowledgement')
    require(begins == [[p['seq'], p['point']] for p in points], 'stdout/trace point scopes differ')
    require([v['identifiers'] for v in values] == [[p['seq'], p['point'], order, p['host_code']]
                                                 for p in points for order in (0, 1)],
            'stdout amplitude order/host-code coverage differs')
    require(complete == 1, 'missing/duplicate completed-driver marker')
    expected_records = [record for p in points for record in
                        [('begin', p['seq']), ('health', p['seq']), ('value', p['seq'], 0), ('value', p['seq'], 1)]]
    require(records == expected_records + [('complete',)], 'stdout observer/driver chronology differs')
    for index, point in enumerate(points):
        flags = [e['value'] for e in point['events'] if e['tag'] in ('POINT_INVALID', 'POINT_DIVZERO', 'POINT_OVERFLOW')]
        rows = values[2 * index:2 * index + 2]
        require(all(row['flags'] == flags for row in rows), 'stdout/trace IEEE flags differ')
        point['stdout_values'] = rows
        point['health'] = {'faults': 0, 'io_failures': 0, 'independent_acknowledgement': True}


def parse_trace(text, stdout_text):
    lines = Lines(text)
    require(lines.numeric('QO_HEADER', 8) == HEADER, 'unsupported observer schema/capacities')
    byte_marker = lines.numeric('QO_BYTE_ORDER', 8)
    require(byte_marker in (list(range(8, 0, -1)), list(range(1, 9))), 'unsupported native byte order')
    byte_order = 'little' if byte_marker[0] == 8 else 'big'
    points, sequences = [], set()
    while lines.index < len(lines.lines):
        points.append(parse_point(lines, byte_order, sequences))
    require(points, 'trace contains no complete measured point')
    require([[p['seq'], p['point']] for p in points] == EXPECTED_POINTS,
            'missing/extra/reordered point relative to frozen three-point driver')
    validate_stdout(stdout_text, points)
    outcomes, statuses, clamps = Counter(), Counter(), Counter()
    for point in points:
        for call in point['calls']:
            outcomes[call['outcome']] += 1
            statuses['not_called' if call['native_status'] is None else str(call['native_status'])] += 1
            for pair in call['taken_clamps']:
                clamps.update(map(str, pair['branches']))
        point['summary'] = {'calls': len(point['calls']),
                            'outcomes': dict(sorted(Counter(c['outcome'] for c in point['calls']).items())),
                            'native_statuses': dict(sorted(Counter('not_called' if c['native_status'] is None
                                                                  else str(c['native_status']) for c in point['calls']).items()))}
    return {'schema': 1, 'valid_trace': True, 'byte_order': byte_order, 'points': points,
            'summary': {'points': len(points), 'calls': sum(len(p['calls']) for p in points),
                        'outcomes': dict(sorted(outcomes.items())), 'native_statuses': dict(sorted(statuses.items())),
                        'taken_clamp_branches': dict(sorted(clamps.items()))},
            'limitations': [
                'Structural validity is not physical amplitude acceptance or numerical qualification.',
                'Native status is the public Fortran wrapper status; failed outputs are not interpreted as computed zeros.',
                'Exact words are retained; decode_payload returns Fraction-based values on demand.',
                'DP_CONVERTED is captured before possible final output reset.',
                'Pole/finite/cache/QP-integrand options and OSTHRES require separately bound configuration; they are not inferred here.',
            ]}


def load_trace(path, stdout):
    path, stdout = Path(path), Path(stdout)
    raw, output = path.read_bytes(), stdout.read_bytes()
    try:
        result = parse_trace(raw.decode('ascii'), output.decode('utf-8'))
    except UnicodeError as error:
        raise TraceError(f'trace/stdout encoding error: {error}') from error
    result['inputs'] = {str(source): {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                        for source, data in ((path, raw), (stdout, output))}
    code = Path(__file__).read_bytes()
    result['decoder'] = {'sha256': hashlib.sha256(code).hexdigest(), 'bytes': len(code)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--stdout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = load_trace(args.trace, args.stdout)
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    except (TraceError, OSError) as error:
        parser.exit(1, f'trace decoding failed: {error}\n')
    print(json.dumps(result['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
