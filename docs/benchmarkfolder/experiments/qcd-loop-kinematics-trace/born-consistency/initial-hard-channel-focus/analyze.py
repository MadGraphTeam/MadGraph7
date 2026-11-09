#!/usr/bin/env python3
"""Exact offline analysis of captured IEEE words; no native calls."""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import struct

ROOT = Path.cwd().resolve()
HERE = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-trace/born-consistency'
OUT = HERE / 'native-1'
FALLBACK = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype'
POINTS = ROOT / 'docs/benchmarkfolder/experiments/physical-stress/points/qcd_n3_uux_zgg/selected.dat'
PI = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode128(word):
    value = int(word, 16)
    sign = -1 if value >> 127 else 1
    exp = (value >> 112) & 0x7fff
    mantissa = value & ((1 << 112) - 1)
    assert exp != 0x7fff, 'Nonfinite binary128 cannot support this arithmetic comparison'
    if exp:
        mantissa += 1 << 112
        power = exp - 16383 - 112
    else:
        power = 1 - 16383 - 112
    return sign * Fraction(mantissa) * Fraction(2) ** power


def dec(value):
    with localcontext() as context:
        context.prec = 100
        return Decimal(value.numerator) / Decimal(value.denominator)


def exact(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator),
            'decimal': str(dec(value))}


def square(vector):
    return vector[0] ** 2 - sum(x ** 2 for x in vector[1:])


def change(before, after):
    denominator = abs(before) + abs(after)
    return {'before': exact(before), 'after': exact(after),
            'signed_difference': exact(after - before),
            'symmetric_relative_change': None if not denominator else str(dec(2 * abs(after - before) / denominator)),
            'fractional_change_over_before': None if not before else str(dec((after - before) / before))}


def main():
    execution = json.loads((OUT / 'execution.json').read_text())
    assert execution['complete'] and execution['protected_inputs_unchanged']
    for name, expected in execution['protected_inputs_before'].items():
        assert sha(ROOT / name) == expected, name
    assert sha(OUT / 'output.log') == execution['logs_sha256']['output.log']
    data = {}
    exceptions = []
    for line in (OUT / 'output.log').read_text().splitlines():
        fields = line.split()
        if len(fields) == 6 and len(fields[4]) == 32:
            try:
                value = decode128(fields[4])
            except ValueError:
                continue
            key = fields[0], int(fields[1]), int(fields[2]), int(fields[3])
            assert key not in data, key
            data[key] = value
        if fields and fields[0] in ('REPAIR_BADFLAGS', 'BORN_BADFLAGS'):
            assert fields[-3:] == ['F', 'F', 'F'], line
            exceptions.append(line)
    assert len(exceptions) == 6
    original = POINTS.read_text().split()
    assert int(original[0]) == 2 and len(original) == 41
    historical = json.loads((FALLBACK / 'amplitude-native-1/adjudication.json').read_text())
    report = {
        'schema': 1, 'native_executed_by_analyzer': False,
        'scope': 'Same supplied physical record, two explicitly different effective kinematics. Neither C nor this Born-only replay is a checked loop acceptance.',
        'source_bindings': {str(p.relative_to(ROOT)): sha(p) for p in
                            (HERE / 'analyze.py', OUT / 'execution.json', OUT / 'output.log',
                             HERE / 'driver.f', HERE / 'set_mp_ps.f', POINTS,
                             FALLBACK / 'amplitude-native-1/adjudication.json')},
        'all_named_bad_flags_false': True,
        'coefficient_conventions': 'Physical p1,p2 incoming; p3,p4,p5 outgoing. Subset channels sum (+p1,+p2,-p3,-p4,-p5).',
        'points': [],
        'limits': [
            'B/C use the identical reviewed QP Born aggregate with DP parameters promoted exactly.',
            'C is actual unmodified SET_MP_PS repair, not an unchanged numerical input.',
            'No loop reducer was called and no clamp was observed by this Born-only replay.',
            'Full QP numerator uses its existing MP coupling conventions; these are not replaced by the Born diagnostic.',
            'Double-pole comparisons use rejected historical loop outputs and are diagnostic only.',
            'Exact fractions quantify input invariants; they do not reproduce each rounded loop prefix or matrix-builder operation.',
        ],
    }
    for point in (1, 2):
        coords = {kind: [[data[(kind + '_COORD', point, leg, mu)] for mu in range(4)]
                         for leg in range(1, 6)] for kind in ('B', 'C')}
        for leg in range(5):
            for mu in range(4):
                token = original[1 + (point - 1) * 20 + leg * 4 + mu]
                assert coords['B'][leg][mu] == Fraction(float(token.replace('D', 'E').replace('d', 'e')))
        entry = {'point': point, 'B_exactly_promotes_original_binary64': True,
                 'external_shell_residuals': [], 'channels': []}
        max_coordinate_change = max(abs(c - b) for bv, cv in zip(coords['B'], coords['C'])
                                    for b, c in zip(bv, cv))
        entry['max_absolute_coordinate_change'] = exact(max_coordinate_change)
        entry['conservation_residuals'] = {
            kind: [exact(sum((1 if leg < 2 else -1) * coords[kind][leg][mu] for leg in range(5)))
                   for mu in range(4)] for kind in ('B', 'C')}
        for leg in range(1, 6):
            item = {'leg': leg, 'physical_mass_class': 'Z' if leg == 3 else 'massless',
                    'mass_DP_promoted': exact(data['MASS_DP', point, leg, 0]),
                    'mass_MP': exact(data['MASS_MP', point, leg, 0]),
                    'DP_mass_square_promoted': exact(data['SQUARE_DP', point, leg, 0]),
                    'MP_mass_square_binary128': exact(data['SQUARE_MP', point, leg, 0])}
            for kind in ('B', 'C'):
                p2 = square(coords[kind][leg - 1])
                item[kind] = {'p_squared': exact(p2),
                             'minus_DP_mass_square': exact(p2 - data['SQUARE_DP', point, leg, 0]),
                             'minus_MP_mass_square': exact(p2 - data['SQUARE_MP', point, leg, 0])}
            entry['external_shell_residuals'].append(item)
        for size in range(1, 5):
            for subset in itertools.combinations(range(1, 6), size):
                channel = {'legs': list(subset), 'topological_class':
                           'direct single leg' if size == 1 else
                           'single-leg complement' if size == 4 else 'composite'}
                values = {}
                for kind in ('B', 'C'):
                    vector = [sum((1 if leg < 3 else -1) * coords[kind][leg - 1][mu]
                                  for leg in subset) for mu in range(4)]
                    values[kind] = square(vector)
                channel.update(change(values['B'], values['C']))
                entry['channels'].append(channel)
                if subset == (1, 5):
                    entry['p1_minus_p5'] = channel
        born_b = data['B_BORN', point, 0, 0]
        born_c = data['C_BORN', point, 0, 0]
        assert born_b == data['B_BORN', point, 1, 0]
        assert born_c == data['C_BORN', point, 1, 0]
        entry['born_change'] = change(born_b, born_c)
        entry['double_pole_comparisons'] = {}
        with localcontext() as context:
            context.prec = 90
            g = dec(data['G_DP', point, 0, 0])
            prefactor = -Decimal(26) / Decimal(3) * g * g / (8 * PI * PI)
            predictions = {'B': dec(born_b) * prefactor, 'C': dec(born_c) * prefactor}
            entry['analytic_double_poles'] = {name: str(value) for name, value in predictions.items()}
            for backend in ('8', '9'):
                rows = historical['backends'][backend]['prototype']['rows']
                row = next(row for row in rows if row['point'] == point and row['split_order'] == 0)
                word = row['raw_binary64_hex'][3]
                native = Fraction(struct.unpack('>d', bytes.fromhex(word))[0])
                value = dec(native)
                entry['double_pole_comparisons'][backend] = {
                    'historical_status': row['code'], 'historical_double_pole': exact(native),
                    'relative_error_vs_B': str(abs(value - predictions['B']) / abs(predictions['B'])),
                    'relative_error_vs_C': str(abs(value - predictions['C']) / abs(predictions['C'])),
                }
                assert float(born_b) == row['born'] or point == 1
        report['points'].append(entry)
    (OUT / 'analysis.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    stress = report['points'][1]
    print(json.dumps({'stress_born_change': stress['born_change']['fractional_change_over_before'],
                      'stress_channel_change': stress['p1_minus_p5']['fractional_change_over_before'],
                      'stress_channel_before': stress['p1_minus_p5']['before']['decimal'],
                      'stress_channel_after': stress['p1_minus_p5']['after']['decimal'],
                      'loop_pole_comparisons': stress['double_pole_comparisons']}, indent=2))


if __name__ == '__main__':
    main()
