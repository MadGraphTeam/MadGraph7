#!/usr/bin/env python3
"""Compact the frozen tensor replay results without evaluating a reducer."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize():
    data = json.loads((HERE / 'summary.json').read_text())
    execution = json.loads((HERE / 'execution.json').read_text())
    cases = []
    for record in data['records']:
        slots = {}
        for slot, name in enumerate(('finite', 'single', 'double')):
            rows = [row for row in record['comparisons'] if row['slot'] == slot]
            if rows:
                worst = {metric: max(rows, key=lambda row: Decimal(row['difference'][metric]))
                         for metric in ('absolute', 'relative')}
                slots[name] = {metric: {'value': row['difference'][metric], 'unit': row['unit'],
                    'tir': row['tir'], 'nrs': row['nrs']} for metric, row in worst.items()}
                slots[name]['rows_above_3e_minus_6_relative_diagnostic_only'] = sum(
                    Decimal(row['difference']['relative']) > Decimal('3e-6') for row in rows)
                slots[name]['scalar_unit'] = next(row for row in rows if row['unit'] == 0)
        historical = record['TIR_historical_differences']
        cases.append({'point': record['point'], 'group': record['group'],
            'TIR_status': record['tir_status'], 'TIR_error': record['tir_error'],
            'NRS_status_counts': dict(Counter(u['status'] for u in record['NRS_units'].values())),
            'comparison_count': len(record['comparisons']), 'slot_metrics': slots,
            'historical_TIR_max_absolute_difference':
                str(max(Decimal(row['difference']['absolute']) for row in historical)) if historical else None})
    return {'scope': 'Bounded tensor diagnostic; no full-amplitude or performance acceptance.',
        'metric': data['complex_metric'],
        'coefficient_scale_limit': 'Absolute maxima can refer to different tensor degrees with different units. Tiny absolute pole differences can have large relative ratios near zero.',
        'relative_count_limit': 'The optional 3e-6 count uses this complex max-norm diagnostic metric, not the campaign abs(a-b)/abs(a+b) acceptance metric.',
        'cases': cases, 'native_complete': execution['complete'], 'native_returncode': execution['returncode'],
        'protected_inputs_unchanged': execution['protected_inputs_unchanged'],
        'summary_sha256': digest(HERE / 'summary.json'),
        'execution_sha256': digest(HERE / 'execution.json'),
        'summarizer_sha256': digest(Path(__file__))}


if __name__ == '__main__':
    (HERE / 'metrics.json').write_text(json.dumps(summarize(), indent=2, sort_keys=True) + '\n')
