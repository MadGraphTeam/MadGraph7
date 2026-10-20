#!/usr/bin/env python3
"""Independent raw replay parsing and exact-value contrasts; no engine calls."""
import sys
sys.dont_write_bytecode = True
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPLAY = HERE.parent
CAPTURE = REPLAY.parent
spec = importlib.util.spec_from_file_location('independent_capture', CAPTURE / 'independent-inventory/check.py')
integer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integer)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def parse():
    cells, current, sequence = {}, None, []
    tags = {'C': 1, 'S': 2, 'P': 2, 'M': 1, 'U': 0, 'R': 1, 'R1': 0}
    completed = False
    for line in (REPLAY / 'raw.hex').read_text().splitlines():
        f = line.split()
        assert not completed
        if f[0] == 'BEGIN':
            assert current is None and len(f) == 9
            index, point, direction, group, order, n, rank, cell = map(int, f[1:])
            key = (point, direction, group, order, n, rank)
            assert 1 <= cell <= 8 and point in (44, 80) and direction in (1, 2)
            assert 1 <= group <= 116 and order == 1 and 1 <= n <= 5 and 0 <= rank <= 4
            assert (index, cell) == (len(sequence) // 8 + 1, len(sequence) % 8 + 1)
            current = {'key': key, 'index': index, 'cell': cell, 'input': {}, 'output': {}}
        elif f[0] == 'INPUT':
            assert current is not None and 'bits' not in current
            current['bits'] = int(f[1])
            assert current['bits'] == (64 if current['cell'] in (1, 2, 4, 5) else 128)
        elif f[0] == 'RESULT':
            assert current is not None and 'status' not in current and int(f[1]) == current['bits']
            current['status'] = int(f[2])
            assert current['status'] == 0
        elif f[0] == 'END':
            assert f == ['END', str(current['index']), str(current['cell'])]
            n, rank = current['key'][-2:]
            expected = {f'C:{i}' for i in range(math.comb(rank + 4, 4))}
            expected |= {f'S:{i},{j}' for i in range(1, n+1) for j in range(1, n+1)}
            expected |= {f'P:{i},{j}' for i in range(4) for j in range(1, n+1)}
            expected |= {f'M:{j}' for j in range(1, n+1)} | {'U:'}
            assert set(current['input']) == expected
            assert set(current['output']) == {'R:0', 'R:1', 'R:2', 'R1:'}
            identity = (current['key'], current['cell'])
            assert identity not in cells
            cells[identity] = current
            sequence.append(identity)
            current = None
        elif f[0] == 'COMPLETE':
            assert current is None and f == ['COMPLETE', '464', '3712', '0']
            completed = True
        else:
            tag = f[0]
            count = tags[tag]
            name = tag + ':' + ','.join(str(int(x)) for x in f[1:1+count])
            words = f[1+count:]
            width = current['bits'] // 64
            assert len(words) == width * (2 if tag in ('C', 'M', 'R', 'R1') else 1)
            assert all(re.fullmatch('[0-9A-F]{16}', w) for w in words)
            for i in range(0, len(words), width):
                integer.decode(words[i:i+width])  # rejects nonfinite fields
            target = current['output'] if 'status' in current else current['input']
            assert name not in target
            target[name] = words
    assert completed and len(cells) == 3712
    return cells


def decoded(words, bits):
    width = bits // 64
    return [integer.decode(words[i:i+width]) for i in range(0, len(words), width)]


def compute():
    cells = parse()
    records = integer.parse_independently(CAPTURE / 'attempt-2/raw/tensor-capture.hex')
    captured = {}
    for r in records:
        key = (r['point'], r['ctmode'] - (3 if r['stage'] == 3 else 0),
               r['loop_id'], r['order'], r['n'], r['rank'])
        captured[key, {1: 1, 2: 2, 3: 8}[r['stage']]] = r
    assert len(captured) == 1392
    keys = sorted({key for key, cell in cells})
    assert len(keys) == 464
    for point in (44, 80):
        for direction in (1, 2):
            assert sorted(k[2] for k in keys if k[:2] == (point, direction)) == list(range(1, 117))
    for (key, cell), r in captured.items():
        actual = cells[key, cell]
        assert actual['status'] == r['status']
        assert all(actual['output'][name] == r['data'][name] for name in actual['output'])
        assert all(actual['input'][name] == r['data'][name] for name in actual['input'])
    for key in keys:
        base, qp = cells[key, 1], cells[key, 8]
        for cell in range(1, 9):
            row = cells[key, cell]
            for name, words in row['input'].items():
                if cell == 8 or (cell == 7 and name.startswith('C:')):
                    assert words == qp['input'][name]
                elif cell in (4, 5, 6) and name.startswith('C:'):
                    rounded = [integer.round64(value, negative) for value, negative in decoded(qp['input'][name], 128)]
                    if row['bits'] == 64:
                        assert words == rounded
                    else:
                        assert decoded(words, 128) == decoded(rounded, 64)
                else:
                    assert decoded(words, row['bits']) == decoded(base['input'][name], 64)
                    if row['bits'] == 64:
                        assert words == base['input'][name]
    retained = json.loads((REPLAY / 'contrasts.json').read_text())
    summaries = {}
    for name, contrast in retained['contrasts'].items():
        left, right = contrast['left_cell'], contrast['reference_cell']
        ours = {}
        for key in keys:
            def values(cell):
                row = cells[key, cell]
                return [value for pole in range(3) for value, sign in decoded(row['output'][f'R:{pole}'], row['bits'])]
            a, b = values(left), values(right)
            differences = [abs(x-y) for x, y in zip(a, b)]
            ours[key] = (max(differences), max(map(abs, b)), sum(x != y for x, y in zip(a, b)))
        assert contrast['valid_records'] == 464 and not contrast['invalid_keys']
        assert len(contrast['all_records']) == 464
        for item in contrast['all_records']:
            difference, norm, changed = ours[tuple(item['key'])]
            for field, value in [('maximum_absolute_difference', difference), ('reference_maximum_component', norm)]:
                assert Fraction(int(item[field]['numerator']), int(item[field]['denominator'])) == value
            assert item['different_components'] == changed
        assert contrast['different_components'] == sum(item[2] for item in ours.values())
        worst = max(ours, key=lambda k: ours[k][0])
        difference, norm, changed = ours[worst]
        summaries[name] = {'maximum_key': worst, 'maximum_absolute': str(difference),
                           'maximum_absolute_approximate': float(difference),
                           'same_key_reference_norm': str(norm)}
    return {'passed': True, 'engine_calls': 0, 'raw_sha256': sha(REPLAY/'raw.hex'),
            'checker_sha256': sha(Path(__file__)), 'keys': 464, 'cells': 3712,
            'all_success_finite': True, 'exact_capture_controls': 1392,
            'actual_input_constructions_verified_including_zero_sign': True,
            'all_exact_per_record_contrast_differences_and_norms_match': True,
            'contrasts': summaries,
            'scope': 'Stored per-loop diagnostic only; no amplitude reconstruction, provider/precision causality or mixed-retry acceptance.'}


if __name__ == '__main__':
    print(json.dumps(compute(), indent=2))
