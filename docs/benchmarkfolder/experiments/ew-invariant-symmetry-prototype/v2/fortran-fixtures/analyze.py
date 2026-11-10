#!/usr/bin/env python3
"""Integer-only retained-frame comparison; importing this module has no effects."""
from itertools import product
import signal

SIZES = (1, 2, 3, 6)
CASES = {1: range(1, 7), 2: range(1, 9)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bits(words):
    return sum(int(word, 16) << (64 * index) for index, word in enumerate(words))


def finite(words):
    value = bits(words)
    fraction, exponent = (52, 11) if len(words) == 1 else (112, 15)
    return ((value >> fraction) & ((1 << exponent) - 1)) != (1 << exponent) - 1


def equal(a, b):
    # All callers have established finiteness: only opposite zeros have
    # different bit patterns but equal numerical values in one IEEE format.
    mask = (1 << (64 * len(a) - 1)) - 1
    return a == b or (bits(a) & mask == 0 and bits(b) & mask == 0)


def status(matrix):
    if not all(finite(value) for value in matrix.values()):
        return 1
    for (i, j), value in matrix.items():
        if i < j and not equal(value, matrix[j, i]):
            return 2
    return 0


def parse(text, partial=False):
    frames, rounds, skips = [], {}, set()
    frame = None
    complete = False
    for raw in text.splitlines():
        words = raw.split()
        if not words:
            continue
        tag = words[0]
        require(not complete, 'records after COMPLETE')
        if tag == 'ROUND':
            require(frame is None and len(words) == 3, 'ROUND schema')
            key, value = map(int, words[1:])
            require(key in range(1, 5) and value in (0, 1) and key not in rounds, 'ROUND identity')
            rounds[key] = value
        elif tag == 'SKIP_FLAG':
            require(frame is None and len(words) == 3, 'SKIP_FLAG schema')
            item = tuple(map(int, words[1:]))
            require(item not in skips and item[0] in range(1, 5) and item[1] in range(1, 6), 'flag skip identity')
            skips.add(item)
        elif tag == 'H':
            require(frame is None and len(words) == 9, 'H schema/nesting')
            header = tuple(map(int, words[1:]))
            precision, route, case, n, rounding, initial, trap, arm = header
            require(precision in (8, 16) and route in CASES and case in CASES[route]
                    and n in SIZES and rounding in range(1, 5) and initial in range(6)
                    and trap in range(6) and arm in (0, 1), 'H range')
            frame = {'header': header, 'A': {}, 'W': {}, 'order': []}
        elif tag in ('A', 'W'):
            require(frame is not None, 'matrix outside frame')
            count = frame['header'][0] // 8
            require(len(words) == 3 + count, 'matrix word count')
            ij = tuple(map(int, words[1:3]))
            require(all(1 <= x <= frame['header'][3] for x in ij) and ij not in frame[tag], 'matrix indices')
            require(all(len(x) == 16 and all(c in '0123456789ABCDEF' for c in x) for x in words[3:]), 'matrix hex')
            frame[tag][ij] = tuple(words[3:])
            frame['order'].append(tag)
        elif tag in ('I', 'J'):
            require(frame is not None and tag not in frame, 'input frame/schema')
            require(len(words) - 1 == 6 * frame['header'][3] * (frame['header'][0] // 8), 'input word count')
            require(all(len(x) == 16 and all(c in '0123456789ABCDEF' for c in x) for x in words[1:]), 'input hex')
            frame[tag] = tuple(words[1:]); frame['order'].append(tag)
        elif tag in ('E', 'X'):
            require(frame is not None and tag not in frame and len(words) == 12, 'state schema')
            values = tuple(map(int, words[1:]))
            require(all(x in (0, 1) for x in values[:10]) and values[10] in range(1, 5), 'state values')
            frame[tag] = values; frame['order'].append(tag)
        elif tag == 'S':
            require(frame is not None and 'S' not in frame and len(words) == 2, 'status schema')
            frame['S'] = int(words[1]); frame['order'].append(tag)
        elif tag == 'END':
            require(frame is not None and len(words) == 1, 'END schema')
            n = frame['header'][3]
            require(frame['order'] == ['A']*(n*n)+['I','E','X','J']+['W']*(n*n)+['S'], 'frame sequence/coverage')
            require(frame['S'] == status(frame['W']), 'stub status differs from independent word classification')
            frames.append(frame); frame = None
        elif tag == 'COMPLETE':
            require(frame is None and len(words) == 1, 'incomplete final frame')
            complete = True
        elif partial and frame is not None and 'E' in frame:
            # Retain process diagnostics in the raw log; do not treat them as
            # Fortran observation records or evidence of a particular signal.
            continue
        else:
            raise ValueError('unknown record: ' + raw[:160])
    if not partial:
        require(complete and frame is None, 'missing COMPLETE')
    return {'frames': frames, 'rounds': rounds, 'skips': skips,
            'partial_frame': frame, 'complete': complete}


def pair(original, candidate):
    require(original['header'][:-1] == candidate['header'][:-1]
            and original['header'][-1] == 0 and candidate['header'][-1] == 1, 'pair identity')
    header = original['header']
    problems = []
    expected_entry = tuple(int(header[5] == i+1) for i in range(5)) + \
                     tuple(int(header[6] == i+1) for i in range(5)) + (header[4],)
    if original['E'] != expected_entry or candidate['E'] != expected_entry:
        problems.append('unexpected entry state')
    if original['E'] != candidate['E'] or original['X'] != candidate['X']:
        problems.append('IEEE state differs')
    if original['X'][5:] != original['E'][5:] or candidate['X'][5:] != candidate['E'][5:]:
        problems.append('halting or rounding changed')
    if not (original['I'] == original['J'] == candidate['I'] == candidate['J']):
        problems.append('input words changed')
    if original['A'] != candidate['A']:
        problems.append('different starting matrices')
    if header[1] == 2 and original['W'] != original['A']:
        problems.append('standalone original changed input matrix')
    expected = dict(original['W'])
    if all(finite(value) for value in expected.values()):
        for (i, j), value in list(expected.items()):
            if i < j and not equal(value, expected[j, i]):
                expected[j, i] = value
    if candidate['W'] != expected:
        problems.append('matrix policy differs')
    return {'header': list(header[:-1]), 'passed': not problems, 'problems': problems,
            'original_status': original['S'], 'candidate_status': candidate['S'],
            'matrix_changed': original['W'] != candidate['W']}


def bulk(text, precision):
    parsed = parse(text)
    require(set(parsed['rounds']) == {1, 2, 3, 4}, 'rounding support coverage')
    expected = []
    for rounding in range(1, 5):
        if not parsed['rounds'][rounding]:
            continue
        for initial in range(6):
            if (rounding, initial) in parsed['skips']:
                continue
            for n in SIZES:
                for route in (1, 2):
                    for case in CASES[route]:
                        for arm in (0, 1):
                            expected.append((precision, route, case, n, rounding, initial, 0, arm))
    require([f['header'] for f in parsed['frames']] == expected and expected, 'exact bulk stream/order')
    pairs = [pair(*parsed['frames'][i:i+2]) for i in range(0, len(expected), 2)]
    failures = [p for p in pairs if not p['passed']]
    return {'passed': not failures, 'frames': len(expected), 'pairs': len(pairs),
            'rounding_support': parsed['rounds'], 'skipped_flags': sorted(parsed['skips']),
            'changed_matrix_pairs': sum(p['matrix_changed'] for p in pairs), 'failures': failures}


def trap(original_text, candidate_text, original_code, candidate_code):
    if original_text.strip() == candidate_text.strip() == 'UNSUPPORTED_TRAP':
        require(original_code == candidate_code == 0, 'unsupported trap failed')
        return {'supported': False, 'equivalent': None}
    a, b = parse(original_text, partial=True), parse(candidate_text, partial=True)
    if original_code == candidate_code == 0:
        require(a['complete'] and b['complete'] and len(a['frames']) == len(b['frames']) == 1, 'trap completion')
        result = pair(a['frames'][0], b['frames'][0])
        return {'supported': True, 'equivalent': result['passed'], 'returned': True, 'comparison': result}
    fa = a['partial_frame'] or (a['frames'][0] if a['frames'] else None)
    fb = b['partial_frame'] or (b['frames'][0] if b['frames'] else None)
    require(fa is not None and fb is not None and 'E' in fa and 'E' in fb, 'trap lacks pre-call evidence')
    same_entry = fa['header'][:-1] == fb['header'][:-1] and fa['I'] == fb['I'] \
                 and fa['A'] == fb['A'] and fa['E'] == fb['E']
    return {'supported': True, 'equivalent': bool(same_entry and original_code == candidate_code == -signal.SIGFPE),
            'returned': False, 'original_exit': original_code, 'candidate_exit': candidate_code,
            'same_input_and_entry_state': same_entry,
            'scope': 'matching SIGFPE termination only; no post-call words/state available'}
