#!/usr/bin/env python3
"""Offline exact-dyadic contrasts; no amplitude reconstruction or engine calls."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import matrix

HERE = Path(__file__).resolve().parent
PAIRS = {
    'native_original_vs_generic_original': (1, 3),
    'compensated_original_vs_generic_original': (2, 3),
    'native_rounded_qp_vs_generic_rounded_qp': (4, 6),
    'compensated_rounded_qp_vs_generic_rounded_qp': (5, 6),
    'generic_original_vs_generic_rounded_qp': (3, 6),
    'generic_rounded_qp_vs_generic_full_qp_coefficients': (6, 7),
    'generic_original_geometry_vs_generic_full_qp_geometry': (7, 8),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator), 'approximate': float(value)}


def values(row):
    width = row['bits'] // 64
    return [matrix.integer.decode(row['output'][f'R:{pole}'][component * width:(component + 1) * width])[0]
            for pole in range(3) for component in range(2)]


def contrast(left, right):
    differences = [abs(a - b) for a, b in zip(left, right)]
    numerator = max(differences)
    denominator = max(map(abs, right))
    return {'maximum_absolute_difference': exact(numerator),
            'reference_maximum_component': exact(denominator),
            'scaled_maximum_difference': float(numerator / denominator) if denominator else (0.0 if not numerator else None),
            'nonzero_difference_with_zero_reference': bool(numerator and not denominator),
            'different_components': sum(a != b for a, b in zip(left, right))}


def compute():
    run = json.loads((HERE / 'result.json').read_text())
    assert run['complete'] and run['interpretation_permitted']
    for name, digest in run['files'].items():
        assert sha(HERE / name) == digest, name
    analysis = matrix.check(HERE / 'raw.hex')
    assert analysis['all_reproduction_controls_exact'] and analysis['all_actual_input_words_match']
    rows = {(tuple(row['key']), row['cell']): row for row in analysis['cells']}
    keys = sorted({key for key, cell in rows})
    assert len(keys) == 464 and len(rows) == 3712
    result = {'complete': True, 'engine_calls': 0, 'physical_amplitude_or_retry_acceptance': False,
              'input_sha256': sha(HERE / 'raw.hex'), 'analysis_sha256': sha(HERE / 'analysis.json'),
              'analyzer_sha256': sha(Path(__file__)),
              'status_counts': {str(cell): dict(sorted(Counter(row['status'] for row in analysis['cells']
                              if row['cell'] == cell).items())) for cell in range(1, 9)},
              'metric': 'For each loop: max absolute difference over real/imag of all three Laurent coefficients divided by max absolute reference component. No tolerance gate. Rational split excluded.',
              'confounds': ['Generic and native master on-shell thresholds differ.',
                            'Captured QP coefficients include changed external-point and parameter/Born construction.',
                            'Historical MP runtime canonicalizes signed zero during input widening.',
                            'No full-amplitude assembly, normalization, or MadLoop acceptance is inferred.'],
              'contrasts': {}, 'raw_laurent_sums': {}}
    for name, (left_cell, right_cell) in PAIRS.items():
        records, invalid = [], []
        for key in keys:
            left, right = rows[key, left_cell], rows[key, right_cell]
            if left['status'] or right['status'] or left['nonfinite_components'] or right['nonfinite_components']:
                invalid.append(list(key))
                continue
            records.append({'key': list(key), **contrast(values(left), values(right))})
        scaled = sorted(record['scaled_maximum_difference'] for record in records if record['scaled_maximum_difference'] is not None)
        records.sort(key=lambda record: Fraction(int(record['maximum_absolute_difference']['numerator']),
                                                int(record['maximum_absolute_difference']['denominator'])), reverse=True)
        result['contrasts'][name] = {'left_cell': left_cell, 'reference_cell': right_cell,
            'valid_records': len(records), 'invalid_keys': invalid,
            'different_components': sum(record['different_components'] for record in records),
            'scaled_difference_quantiles': {str(q): scaled[round((len(scaled) - 1) * q)] for q in (0, .5, .9, .99, 1)},
            'largest_absolute_difference_records': records[:10],
            'all_records': records}
    # Formal unweighted sums are diagnostic outputs, not asserted physical amplitudes.
    for point in (44, 80):
        for direction in (1, 2):
            group = [key for key in keys if key[:2] == (point, direction)]
            assert len(group) == 116
            target = result['raw_laurent_sums'][f'{point}:{direction}'] = {}
            for cell in range(1, 9):
                chosen = [rows[key, cell] for key in group]
                assert all(not row['status'] and not row['nonfinite_components'] for row in chosen)
                decoded = [values(row) for row in chosen]
                target[str(cell)] = {'unweighted_sum': [exact(sum((row[index] for row in decoded), Fraction(0))) for index in range(6)],
                                    'sum_absolute_components': [exact(sum((abs(row[index]) for row in decoded), Fraction(0))) for index in range(6)]}
    return result


if __name__ == '__main__':
    with (HERE / 'contrasts.json').open('x') as stream:
        json.dump(compute(), stream, indent=2)
        stream.write('\n')
