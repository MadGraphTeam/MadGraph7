#!/usr/bin/env python3
"""Independent exact-rational audit of fixed native replay; no engine calls."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path

POLICIES = ('Interpreted', 'NativeDenominators', 'FullNative')
FAMILIES = ('T2Denominators', 'T3Denominators', 'BubbleSourceProjection',
    'TriangleBubbleCorrection', 'TadpoleSourceProjection', 'OtherReduction',
    'Integration', 'Accumulation')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dyadic(component):
    assert component['finite'] is True
    assert isinstance(component['negative'], bool)
    n, d = int(component['numerator']), int(component['denominator'])
    p = component['precision']
    assert isinstance(p, int) and p > 0 and d > 0 and d & (d-1) == 0
    assert not n or ((n < 0) == component['negative'])
    if n:
        magnitude = abs(n)
        significant = magnitude >> ((magnitude & -magnitude).bit_length()-1)
        assert significant.bit_length() <= p
    return Fraction(n, d)


def ratio(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator),
        'approximate': float(value)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    root = next(p for p in experiment.parents if (p/'madgraph/__init__.py').is_file())
    prepared = json.loads((experiment/'prepared.json').read_text())
    capture = root/prepared['capture']['path']
    assert sha(capture) == prepared['capture']['sha256']
    with gzip.open(capture, 'rb') as stream:
        original_bytes = stream.read()
    assert hashlib.sha256(original_bytes).hexdigest() == prepared['capture']['uncompressed_sha256']
    inputs = [json.loads(line) for line in original_bytes.splitlines()]
    assert len(inputs) == 1392
    stage = experiment/'run-replay'
    result = json.loads((stage/'result.json').read_text())
    raw = stage/'raw.jsonl'
    assert result['complete'] and result['returncode'] == 0 and result['stage'] == 'replay'
    assert sha(raw) == result['files']['raw.jsonl']
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert rows[-1] == result['summary'] and len(rows) == 1393
    summary = rows.pop()
    assert summary['complete'] and summary['kind'] == 'summary'
    status = {p: Counter() for p in POLICIES}
    tracked = {p: Counter() for p in POLICIES}
    counts = {p: Counter() for p in POLICIES[1:]}
    backend_shapes = {}
    split_failures = []
    discrepancies = []
    max_ratios = {p: Fraction(0) for p in POLICIES[1:]}
    for sequence, (row, source) in enumerate(zip(rows, inputs)):
        assert row['kind'] == 'request' and row['sequence'] == sequence and row['input'] == source
        assert source['status'] == 0 and len(source['output']) == 8
        arms = row['arms']
        assert len(arms) == 3 and [a['policy'] for a in arms] == list(POLICIES)
        values = []
        for arm in arms:
            policy = arm['policy']
            assert arm['working_bits'] == 256 and arm['input_unchanged'] is True
            assert arm['status'] == 0 and len(arm['output']) == 8
            assert arm['formatted_buffer_bytes'] == 8*64 and len(arm['raw']) == 10
            status[policy][str(arm['status'])] += 1
            value = [dyadic(c) for c in arm['raw']]
            values.append(value)
            tracked[policy].update(str(c['precision']) for c in arm['raw'])
            norm = max(map(abs, value[:6]))
            defect = [value[i]-value[6+i]-value[8+i] for i in (0,1)]
            if any(abs(v) > Fraction(1,10**60)*norm for v in defect):
                split_failures.append({'sequence': sequence, 'policy': policy, 'defect': [ratio(v) for v in defect]})
            native = set()
            if policy != 'Interpreted': native.update(FAMILIES[:2])
            if policy == 'FullNative':
                native.update(FAMILIES[2:5])
                if source['rank'] == source['n']+2: native.remove('TriangleBubbleCorrection')
            expected = [{'family':f,'backend':'NativeRust' if f in native else 'SymbolicaInterpreter'} for f in FAMILIES]
            assert arm['backends'] == expected
            shape = f"{policy}:{source['n']}:{source['rank']}"
            assert backend_shapes.setdefault(shape, expected) == expected
        baseline = arms[0]
        assert baseline['status'] == source['status'] and baseline['output'] == source['output']
        assert row['baseline_exact_historical'] is True
        norm = max(map(abs, values[0][:6]))
        for index in (1,2):
            policy, arm, value = POLICIES[index], arms[index], values[index]
            differences = [abs(x-y) for x,y in zip(value,values[0])]
            changed_value = [i for i,d in enumerate(differences) if d]
            changed_sign = [i for i,(a,b) in enumerate(zip(arm['raw'],baseline['raw'])) if a['negative'] != b['negative']]
            changed_precision = [i for i,(a,b) in enumerate(zip(arm['raw'],baseline['raw'])) if a['precision'] != b['precision']]
            changed_text = [i for i,(a,b) in enumerate(zip(arm['output'],baseline['output'])) if a != b]
            different = bool(changed_value or changed_sign or changed_precision or changed_text)
            assert row['native_any_value_precision_sign_or_formatted_difference'][index-1] == different
            counts[policy]['different_records'] += different
            counts[policy]['value_different_records'] += bool(changed_value)
            counts[policy]['sign_different_records'] += bool(changed_sign)
            counts[policy]['precision_different_records'] += bool(changed_precision)
            counts[policy]['formatted_different_records'] += bool(changed_text)
            screen_failed = any(d > Fraction(1,10**31)*norm for d in differences[:6])
            rational_failed = any(d > Fraction(1,10**31)*norm for d in differences[8:])
            counts[policy]['laurent_screen_failed_records'] += screen_failed
            counts[policy]['rational_screen_failed_records'] += rational_failed
            if norm: max_ratios[policy] = max(max_ratios[policy],max(differences[:6])/norm)
            if different:
                discrepancies.append({'sequence':sequence,'policy':policy,'value_indices':changed_value,
                    'sign_indices':changed_sign,'precision_indices':changed_precision,'formatted_indices':changed_text,
                    'laurent_screen_failed':screen_failed,'rational_screen_failed':rational_failed})
    assert summary['records'] == 1392 and summary['arm_fields_each'] == 11136
    assert summary['failed_arms'] == summary['baseline_changed_records'] == 0
    assert summary['native_different_records'] == [counts[p]['different_records'] for p in POLICIES[1:]]
    print(json.dumps({'schema':'independent-native-replay-audit-v1','complete':True,
        'engine_calls':0,'performance_claim':False,'audit_source_sha256':sha(Path(__file__)),
        'run_result_sha256':sha(stage/'result.json'),'raw_sha256':sha(raw),
        'capture_compressed_sha256':sha(capture),'record_count':1392,'fields_per_arm':11136,
        'all_baseline_ascii_fields_exact':True,'status_counts':status,'tracked_precision_counts':tracked,
        'backend_shape_count':len(backend_shapes),'backend_reports':backend_shapes,'difference_counts':counts,
        'largest_laurent_component_difference_over_baseline_norm':{p:ratio(v) for p,v in max_ratios.items()},
        'finite_split_failures':split_failures,'discrepancies':discrepancies},indent=2))


if __name__ == '__main__':
    main()
