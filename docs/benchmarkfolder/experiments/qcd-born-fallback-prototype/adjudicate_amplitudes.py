#!/usr/bin/env python3
"""Classify actual host results independently of process exit success."""
from pathlib import Path
import hashlib
import json
import math
import struct

ROOT = Path.cwd().resolve()
REPORT = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype'
OUT = REPORT / 'amplitude-native-1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    rows, counts, normalization = [], {}, {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'AMPLITUDE_VALUE':
            assert len(fields) == 13, fields
            sequence, point, so, code = map(int, fields[1:5])
            values = list(map(float, fields[5:10]))
            rows.append({'sequence': sequence, 'point': point, 'split_order': so,
                         'code': code, 'h': code // 100, 't': code // 10 % 10,
                         'u': code % 10, 'born': values[0], 'finite': values[1],
                         'single_pole': values[2], 'double_pole': values[3], 'accuracy': values[4],
                         'finite_output': all(map(math.isfinite, values)),
                         'ieee_invalid_divzero_overflow': [x == 'T' for x in fields[10:]],
                         'raw_binary64_hex': [struct.pack('>d', x).hex() for x in values]})
        elif fields[0] == 'BORN_RESCUE_COUNTS':
            counts[int(fields[1])] = list(map(int, fields[2:]))
        elif fields[0] == 'LOOP_NORMALIZATION':
            normalization[int(fields[1])] = list(map(float, fields[2:]))
    for row in rows:
        if row['sequence'] in normalization and math.isfinite(row['born']):
            mu, coupling = normalization[row['sequence']]
            expected = -(26. / 3.) * coupling**2 / (8. * math.pi**2) * row['born']
            row['analytic_double_pole'] = expected
            row['double_pole_relative_error'] = abs(row['double_pole'] - expected) / max(abs(expected), 1e-300)
    return {'rows': rows, 'rescue_counts': counts, 'normalization': normalization, 'log_sha256': sha(path)}


def main():
    report = {'schema': 1, 'scope': 'Private same-input Born fallback physical host adjudication.',
              'physics_gate': 3e-6, 'host_stability_threshold': 1e-6,
              'analytic_double_pole_convention': '-Born * (2*C_F+2*C_A) * G^2/(8*pi^2), C_F=4/3, C_A=3.',
              'ieee_interpretation': 'Record caller-visible sticky flags separately. MadLoop acceptance uses its per-reduction FPE/failure flag and checked status; a sticky flag alone is not an independent physics error estimator.',
              'backends': {}, 'cross_backend': [], 'formal_performance_eligible': False}
    for directory in sorted(OUT.glob('backend-*')):
        execution_path = directory / 'execution.json'
        if not execution_path.is_file():
            continue
        execution = json.loads(execution_path.read_text())
        backend = int(directory.name.split('-')[1])
        item = {'execution_complete': execution['complete'], 'execution_sha256': sha(execution_path)}
        for mode in ('original', 'prototype'):
            path = directory / (mode + '.log')
            if path.exists():
                item[mode] = read(path)
        if 'original' in item and 'prototype' in item:
            original = {(r['sequence'], r['split_order']): r for r in item['original']['rows']}
            ordinary = [r for r in item['prototype']['rows'] if r['point'] == 1]
            item['ordinary_bits_status_flags_identical'] = len(ordinary) == 4 and all(
                (r['raw_binary64_hex'], r['code'], r['ieee_invalid_divzero_overflow']) ==
                (original[(r['sequence'], r['split_order'])]['raw_binary64_hex'],
                 original[(r['sequence'], r['split_order'])]['code'],
                 original[(r['sequence'], r['split_order'])]['ieee_invalid_divzero_overflow'])
                for r in ordinary)
            item['ordinary_checked_pass'] = len(ordinary) == 4 and all(
                r['h'] in (2, 3) and r['t'] in (1, 2) and r['finite_output']
                and r['accuracy'] < 1e-6
                and r.get('double_pole_relative_error', math.inf) <= 3e-6 for r in ordinary)
            stress = [r for r in item['prototype']['rows'] if r['point'] == 2]
            expected_u = {1: 9, 8: 2, 9: 3}[backend]
            item['stress_checked_qp_and_poles_pass'] = len(stress) == 2 and all(
                r['h'] == 3 and r['t'] == 2 and r['u'] == expected_u and r['finite_output']
                and r['accuracy'] < 1e-6
                and r.get('double_pole_relative_error', math.inf) <= 3e-6 for r in stress)
        report['backends'][str(backend)] = item
    backend_ids = sorted(report['backends'])
    for i, first in enumerate(backend_ids):
        for second in backend_ids[i+1:]:
            one = report['backends'][first].get('prototype', {}).get('rows', [])
            two = report['backends'][second].get('prototype', {}).get('rows', [])
            for x in one:
                if x['point'] != 2 or x['split_order'] != 0:
                    continue
                for y in two:
                    if y['point'] == 2 and y['split_order'] == 0:
                        errors = {name: 2 * abs(x[name] - y[name]) / max(abs(x[name]) + abs(y[name]), 1e-300)
                                  for name in ('born', 'finite', 'single_pole', 'double_pole')}
                        report['cross_backend'].append({'backends': [first, second], 'symmetric_relative_errors': errors})
    # JSON records explicit nonfinite values as strings, not nonstandard NaN.
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {str(k): finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        return value
    (OUT / 'adjudication.json').write_text(json.dumps(finite_json(report), indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: {n: v[n] for n in ('ordinary_bits_status_flags_identical', 'stress_checked_qp_and_poles_pass') if n in v}
                      for k, v in report['backends'].items()}, indent=2))


if __name__ == '__main__':
    main()
