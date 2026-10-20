#!/usr/bin/env python3
"""Independent exact-dyadic audit of stored capture; never calls an engine."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
TAGS = ('C', 'S', 'P', 'M', 'U')
CLASSES = ('same_sign_nonzero', 'zero_to_nonzero', 'nonzero_to_zero',
           'opposite_sign_nonzero', 'both_zero_same_sign', 'both_zero_changed_sign')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(words):
    """Return exact finite value and sign; binary128 words are low, high."""
    assert len(words) in (1, 2)
    raw = sum(int(word, 16) << (64 * i) for i, word in enumerate(words))
    width, fraction_bits, exponent_bits, bias = ((64, 52, 11, 1023) if len(words) == 1
                                                else (128, 112, 15, 16383))
    negative = bool(raw >> (width - 1))
    exponent = (raw >> fraction_bits) & ((1 << exponent_bits) - 1)
    significand = raw & ((1 << fraction_bits) - 1)
    assert exponent != (1 << exponent_bits) - 1
    if exponent:
        significand += 1 << fraction_bits
    power = (exponent or 1) - bias - fraction_bits
    value = Fraction(significand << power, 1) if power >= 0 else Fraction(significand, 1 << -power)
    return -value if negative else value, negative


def round64(value, negative_zero=False):
    """Exact integer-quotient rounding to binary64, nearest with ties to even."""
    sign = value < 0 or (not value and negative_zero)
    value = abs(value)
    if not value:
        return f'{int(sign) << 63:016X}'
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if exponent >= 0:
        if value.numerator < (value.denominator << exponent):
            exponent -= 1
    elif (value.numerator << -exponent) < value.denominator:
        exponent -= 1
    power = max(exponent - 52, -1074)
    scaled = value / (1 << power) if power >= 0 else value * (1 << -power)
    integer, residue = divmod(scaled.numerator, scaled.denominator)
    if 2 * residue > scaled.denominator or (2 * residue == scaled.denominator and integer & 1):
        integer += 1
    if exponent < -1022:
        raw = integer  # The carry 2**52 encodes the smallest normal.
    else:
        if integer == 1 << 53:
            integer >>= 1
            exponent += 1
        assert exponent <= 1023, 'overflow is outside this capture audit'
        assert 1 << 52 <= integer < 1 << 53
        raw = ((exponent + 1023) << 52) | (integer - (1 << 52))
    return f'{raw | (int(sign) << 63):016X}'


def selftest():
    examples = {
        '0000000000000000': Fraction(0), '8000000000000000': Fraction(0),
        '3FF0000000000000': Fraction(1), 'C000000000000000': Fraction(-2),
        '0000000000000001': Fraction(1, 1 << 1074),
        '0010000000000000': Fraction(1, 1 << 1022),
        '7FEFFFFFFFFFFFFF': Fraction((1 << 53) - 1) * (1 << 971),
    }
    for word, expected in examples.items():
        value, sign = decode([word])
        assert value == expected and round64(value, sign) == word
    assert decode(['0000000000000000', '3FFF000000000000'])[0] == 1
    assert decode(['0000000000000001', '3FFF000000000000'])[0] == 1 + Fraction(1, 1 << 112)
    assert round64(1 + Fraction(1, 1 << 53)) == '3FF0000000000000'
    assert round64(1 + Fraction(3, 1 << 53)) == '3FF0000000000002'
    assert round64(Fraction(1, 1 << 1075)) == '0000000000000000'
    assert round64(Fraction(3, 1 << 1075)) == '0000000000000002'
    assert round64(-Fraction(1, 1 << 1075)) == '8000000000000000'


def parse_independently(path):
    """Parse tagged integer rows without importing the production capture parser."""
    lines = path.read_text().splitlines()
    assert [line.split() for line in lines[:3]] == [
        ['LAYOUT', '3FF0000000000000', '0000000000000000', '3FFF000000000000'],
        ['COMPLEX64', '3FF0000000000000', '4000000000000000'],
        ['COMPLEX128', '0000000000000000', '3FFF000000000000', '0000000000000000', '4000000000000000']]
    records, active = [], None
    index_counts = {'C': 1, 'S': 2, 'P': 2, 'M': 1, 'U': 0, 'R': 1, 'R1': 0}
    for line in lines[3:]:
        fields = line.split()
        if fields[0] == 'BEGIN':
            assert active is None and len(fields) == 9
            seq, point, mode, loop_id, order, n, rank, stage = map(int, fields[1:])
            assert seq == len(records) + 1 and point in (44, 80) and order == 1
            assert 1 <= loop_id <= 116 and 1 <= n <= 5 and 0 <= rank <= 4
            assert stage in (1, 2, 3) and mode in ((4, 5) if stage == 3 else (1, 2))
            active = {'sequence': seq, 'point': point, 'ctmode': mode, 'loop_id': loop_id,
                      'order': order, 'n': n, 'rank': rank, 'stage': stage, 'data': {}}
        elif fields[0] == 'PAYLOAD':
            assert active is not None and 'bits' not in active and len(fields) == 5
            seq, bits, status, nc = map(int, fields[1:])
            assert seq == active['sequence'] and nc == math.comb(active['rank'] + 4, 4)
            assert bits == (128 if active['stage'] == 3 else 64) and status in (0, 1, 2)
            active.update(bits=bits, status=status)
        elif fields[0] == 'END':
            assert active is not None and fields == ['END', str(active['sequence'])]
            n, rank = active['n'], active['rank']
            expected = {f'C:{i}' for i in range(math.comb(rank + 4, 4))}
            expected |= {f'S:{i},{j}' for i in range(1, n+1) for j in range(1, n+1)}
            expected |= {f'P:{i},{j}' for i in range(4) for j in range(1, n+1)}
            expected |= {f'M:{j}' for j in range(1, n+1)} | {'U:', 'R:0', 'R:1', 'R:2', 'R1:'}
            assert set(active['data']) == expected
            records.append(active)
            active = None
        else:
            assert active is not None and 'bits' in active
            tag = fields[0]
            count = index_counts[tag]
            key = tag + ':' + ','.join(str(int(x)) for x in fields[1:1+count])
            words = fields[1+count:]
            width = active['bits'] // 64
            assert len(words) == width * (2 if tag in ('C', 'M', 'R', 'R1') else 1)
            assert key not in active['data'] and all(re.fullmatch('[0-9A-F]{16}', x) for x in words)
            for i in range(0, len(words), width):
                decode(words[i:i+width])
            active['data'][key] = words
    assert active is None and len(records) == 1392
    return records


def quantiles(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    result = {'count': len(values), 'minimum': values[0], 'maximum': values[-1]}
    for label, num, den in [('p50', 1, 2), ('p90', 9, 10), ('p95', 19, 20), ('p99', 99, 100)]:
        result[label] = values[(num * len(values) + den - 1) // den - 1]
    result['buckets'] = dict(Counter('0' if v == 0 else '1' if v == 1 else '2-4' if v <= 4 else
        '5-16' if v <= 16 else '17-256' if v <= 256 else '257-4096' if v <= 4096 else '4097+' for v in values))
    return result


def rational(value):
    with localcontext() as context:
        context.prec = 40
        display = str(Decimal(value.numerator) / Decimal(value.denominator))
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator), 'decimal_diagnostic': display}


def compute():
    selftest()
    raw = BASE / 'attempt-2/raw/tensor-capture.hex'
    records = parse_independently(raw)
    spec = importlib.util.spec_from_file_location('retained_capture_parser', BASE / 'parse_capture_v2.py')
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    assert records == parser.parse(raw)
    matching = parser.match(records)
    assert matching == json.loads((BASE / 'attempt-2/matching.json').read_text())
    tables = {stage: {} for stage in (1, 2, 3)}
    for record in records:
        stage = record['stage']
        key = (record['point'], record['ctmode'] - (3 if stage == 3 else 0),
               record['loop_id'], record['order'], record['n'], record['rank'])
        assert key not in tables[stage]
        tables[stage][key] = record
    assert set(tables[1]) == set(tables[2]) == set(tables[3]) and len(tables[1]) == 464
    for stage in tables:
        for point in (44, 80):
            for direction in (1, 2):
                assert sorted(key[2] for key in tables[stage] if key[:2] == (point, direction)) == list(range(1, 117))
    summaries = {tag: Counter() for tag in TAGS}
    distances = {tag: {category: [] for category in CLASSES} for tag in TAGS}
    maxima = {tag: {'absolute': Fraction(0), 'relative': Fraction(0)} for tag in TAGS}
    output_words = Counter()
    per_record = []
    for key in sorted(tables[1]):
        native, compensated, qp = (tables[stage][key] for stage in (1, 2, 3))
        assert all(record['status'] == 0 for record in (native, compensated, qp))
        local = {tag: Counter() for tag in TAGS}
        for field, words in native['data'].items():
            tag = field.split(':')[0]
            if tag not in TAGS:
                output_words[tag] += sum(a != b for a, b in zip(words, compensated['data'][field], strict=True))
                continue
            assert words == compensated['data'][field], (key, field)
            for component, word in enumerate(words):
                value, negative = decode([word])
                high, high_negative = decode(qp['data'][field][2*component:2*component+2])
                rounded_word = round64(high, high_negative)
                rounded, rounded_negative = decode([rounded_word])
                local[tag]['real_components'] += 1
                local[tag]['exact_value_differences'] += value != high
                local[tag]['rounded_word_differences'] += word != rounded_word
                local[tag]['exact_value_equal_sign_different'] += value == high and negative != high_negative
                a, b = int(word, 16), int(rounded_word, 16)
                ordinal_a = (-1 if negative else 1) * (a & ((1 << 63) - 1))
                ordinal_b = (-1 if rounded_negative else 1) * (b & ((1 << 63) - 1))
                if not value and not rounded:
                    category = 'both_zero_changed_sign' if negative != rounded_negative else 'both_zero_same_sign'
                elif not value:
                    category = 'zero_to_nonzero'
                elif not rounded:
                    category = 'nonzero_to_zero'
                elif negative != rounded_negative:
                    category = 'opposite_sign_nonzero'
                else:
                    category = 'same_sign_nonzero'
                distances[tag][category].append(abs(ordinal_a - ordinal_b))
                difference = abs(high - value)
                norm = max(abs(high), abs(value))
                for name, candidate in [('absolute', difference), ('relative', difference / norm if norm else Fraction(0))]:
                    if candidate > maxima[tag][name]:
                        maxima[tag][name] = candidate
                        maxima[tag][name + '_at'] = {'key': list(key), 'field': field, 'component': component,
                            'dp_word': word, 'qp_words': qp['data'][field][2*component:2*component+2]}
        for tag in TAGS:
            summaries[tag].update(local[tag])
            summaries[tag]['records_with_exact_differences'] += bool(local[tag]['exact_value_differences'])
            summaries[tag]['records_with_rounded_differences'] += bool(local[tag]['rounded_word_differences'])
        per_record.append({'key': list(key), 'by_field': {tag: dict(local[tag]) for tag in TAGS}})
    ulps = {tag: {'classes': {c: quantiles(values) for c, values in distances[tag].items()},
                  'same_sign_nonzero_changed': quantiles([v for v in distances[tag]['same_sign_nonzero'] if v])} for tag in TAGS}
    owner_ulp = json.loads((BASE / 'ulp-inventory.json').read_text())
    for tag in TAGS:
        for name in ('classes', 'same_sign_nonzero_changed'):
            assert ulps[tag][name] == owner_ulp['by_field'][tag][name], (tag, name)
    binding = json.loads((BASE / 'bindings.json').read_text())
    rows = []
    for line in (BASE / 'attempt-2/raw/values.dat').read_text().splitlines():
        fields = line.split()
        assert len(fields) == 8
        row = {'point': int(fields[0]), 'order': int(fields[1]), 'status': int(fields[2]),
               'coefficients': [float(x) for x in fields[3:7]], 'accuracy': float(fields[7])}
        assert all(math.isfinite(x) for x in row['coefficients'] + [row['accuracy']])
        rows.append(row)
    assert rows == binding['reference_rows']
    assert [(row['point'], row['order']) for row in rows] == [(i, order) for i in range(1, 101) for order in (0, 1)]
    timed = [tuple(map(int, line.split())) for line in (BASE / 'attempt-2/raw/timed_status.dat').read_text().splitlines()]
    assert len({a for a, _ in timed}) == len(timed)
    assert dict(timed) == {int(k): v for k, v in binding['reference_status_counts'].items()}
    capture = json.loads((BASE / 'attempt-2/capture-result.json').read_text())
    for filename, info in capture['raw'].items():
        path = BASE / 'attempt-2/raw' / filename
        assert sha(path) == info['sha256'] and path.stat().st_size == info['bytes']
    source_files = ['parse_capture_v2.py', 'inventory.py', 'input-inventory.json', 'inventory_ulp.py',
        'ulp-inventory.json', 'capture_source.py', 'run_capture_v2.py', 'bindings.json', 'prepared.json',
        'attempt-2/capture-result.json', 'attempt-2/matching.json', 'attempt-2/ldd.txt']
    source_files += [f'attempt-2/raw/{name}' for name in capture['raw']]
    return {'schema': 'independent-stored-capture-audit-v1', 'complete': True, 'engine_calls': 0,
        'script_sha256': sha(Path(__file__)), 'bound_files': {p: sha(BASE / p) for p in source_files},
        'independent_raw_parse_equals_hardened_parser': True,
        'full_100_rows_statuses_accuracies_match_reference': True,
        'matched_keys': len(tables[1]), 'native_compensated_inputs_bit_identical': True,
        'all_capture_statuses_zero': True, 'native_compensated_changed_output_words': dict(output_words),
        'by_field': {tag: dict(summaries[tag]) for tag in TAGS}, 'rounded_ulp': ulps,
        'ulp_distributions_equal_owner_inventory': True,
        'exact_component_difference_maxima': {tag: {name: rational(maxima[tag][name]) | {'location': maxima[tag].get(name + '_at')}
            for name in ('absolute', 'relative')} for tag in TAGS}, 'per_record': per_record,
        'scope': 'Exact stored input-bundle differences, not amplitude errors or isolated numerator arithmetic. QP uses improved kinematics and recomputed MP numerators. No engine replay performed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = compute()
    output = HERE / 'result.json'
    if args.write:
        assert not output.exists()
        output.write_text(json.dumps(result, indent=2) + '\n')
    else:
        assert json.loads(output.read_text()) == result
    print(json.dumps({key: result[key] for key in ('complete', 'engine_calls', 'matched_keys',
        'native_compensated_changed_output_words', 'by_field', 'ulp_distributions_equal_owner_inventory')}, indent=2))
