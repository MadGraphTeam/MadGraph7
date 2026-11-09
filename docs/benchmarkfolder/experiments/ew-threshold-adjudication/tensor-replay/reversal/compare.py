#!/usr/bin/env python3
"""Compare completed forward/reverse unit tensors without native evaluation."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(a, b):
    ar, ai, br, bi = map(Decimal, [*a, *b])
    absolute = ((ar-br)**2 + (ai-bi)**2).sqrt()
    norm = max((ar**2 + ai**2).sqrt(), (br**2 + bi**2).sqrt())
    return {'absolute': str(absolute), 'relative': str(absolute/norm if norm else Decimal(0))}


def compare():
    forward = json.loads((HERE.parent / 'summary.json').read_text())
    reverse = json.loads((HERE / 'summary.json').read_text())
    a = next(record for record in forward['records'] if (record['point'], record['group']) == (2, 5))
    assert len(reverse['records']) == 1
    b = reverse['records'][0]
    assert (b['point'], b['group']) == (2, 5)
    rows = []
    for i in range(35):
        au, bu = a['NRS_units'][str(i)], b['NRS_units'][str(i)]
        assert au['degree'] == bu['degree'] and au['mapping'] == bu['mapping']
        if au['status'] == 0 and bu['status'] == 0:
            for slot in range(3):
                av, bv = au['results'][str(slot)], bu['results'][str(slot)]
                rows.append({'unit': i, 'degree': au['degree'], 'slot': slot,
                    'forward': av, 'reverse': bv, 'difference': distance(av, bv)})
    maxima = {}
    for slot, label in enumerate(('finite', 'single', 'double')):
        subset = [row for row in rows if row['slot'] == slot]
        if subset:
            maxima[label] = {metric: max(subset, key=lambda row: Decimal(row['difference'][metric]))
                            for metric in ('absolute', 'relative')}
    return {'scope': 'Denominator-order diagnostic, unchanged 256-bit bridges; no accuracy certificate.',
        'metric': 'abs(a-b)/max(abs(a),abs(b)); zero-zero=0', 'forward_TIR_status': a['tir_status'],
        'reverse_TIR_status': b['tir_status'], 'forward_TIR_error': a['tir_error'],
        'reverse_TIR_error': b['tir_error'], 'NRS_order_comparisons': rows, 'NRS_slot_maxima': maxima,
        'forward_summary_sha256': digest(HERE.parent / 'summary.json'),
        'reverse_summary_sha256': digest(HERE / 'summary.json'), 'analyzer_sha256': digest(Path(__file__))}


if __name__ == '__main__':
    with localcontext() as context:
        context.prec = 100
        result = compare()
    (HERE / 'comparison.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
