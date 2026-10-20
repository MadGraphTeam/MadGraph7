#!/usr/bin/env python3
"""Recompute the fixed paired kernel result from every retained raw interval."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(binding):
    path = ROOT / binding['path']
    assert sha(path) == binding['sha256'] and path.stat().st_size == binding['bytes']
    return path


def quantile(values, probability):
    data = sorted(values)
    index = probability * (len(data) - 1)
    lower = int(index)
    upper = min(lower + 1, len(data) - 1)
    return data[lower] * (upper - index) + data[upper] * (index - lower) if upper != lower else data[lower]


def spread(values):
    median = statistics.median(values)
    return {'median_ns': median, 'minimum_ns': min(values), 'maximum_ns': max(values),
            'q25_ns': quantile(values, .25), 'q75_ns': quantile(values, .75),
            'median_absolute_deviation_ns': statistics.median(abs(v - median) for v in values)}


def analyze(result_path):
    result = json.loads(result_path.read_text())
    assert result['status'] == 'passed' and result['action'] == 'kernel'
    plan = json.loads(bound(result['plan']).read_text())
    assert plan['pairs'] == 21 and plan['bootstrap']['samples'] == 100000 and plan['bootstrap']['seed'] == 20261020
    build_path = bound(result['build'])
    build = json.loads(build_path.read_text())
    assert build['status'] == 'compiled-not-executed' and build['exit_code'] == 0
    expected_kernel = build['executables']['tensor-kernel']
    assert result['executable'] == {'path': expected_kernel['path'], 'sha256': expected_kernel['sha256'],
                                   'bytes': expected_kernel['bytes']}
    for relative, expected in build['sources'].items():
        assert sha(build_path.parent / 'source' / relative) == expected
    assert build['sources']['src/original.rs'] == plan['original_tensor_sha256']
    assert build['sources']['src/allocation.rs'] == plan['prototype_tensor_sha256']
    prepared = json.loads(bound(result['trace_preparation']).read_text())
    assert prepared['status'] == 'passed' and prepared['action'] == 'capture'
    assert prepared['build'] == result['build'] and prepared['plan'] == result['plan']
    expected_capture = build['executables']['tensor-capture']
    assert prepared['executable'] == {'path': expected_capture['path'], 'sha256': expected_capture['sha256'],
                                     'bytes': expected_capture['bytes']}
    trace = bound(prepared['retained_trace'])
    with gzip.open(trace, 'rb') as stream:
        data = stream.read()
    assert hashlib.sha256(data).hexdigest() == result['trace_sha256'] == prepared['trace_sha256']
    trace_rows = [json.loads(line) for line in data.splitlines()]
    assert len(trace_rows) == 1392
    original_capture = ROOT / plan['input_capture']['path']
    assert sha(original_capture) == plan['input_capture']['sha256']
    with gzip.open(original_capture, 'rb') as stream:
        original_data = stream.read()
    assert hashlib.sha256(original_data).hexdigest() == plan['input_capture']['uncompressed_sha256']
    original_rows = [json.loads(line) for line in original_data.splitlines()]
    assert len(original_rows) == 1392
    for index, (trace_row, original_row) in enumerate(zip(trace_rows, original_rows)):
        assert trace_row['bridge_record'] == index
        assert original_row['status'] == 0
        assert (trace_row['n'], trace_row['rank']) == (original_row['n'], original_row['rank'])
        assert len(trace_row['final_ascii']) == 8 and trace_row['final_ascii'] == original_row['output']
    shapes = [(r['n'], r['rank']) for r in trace_rows]
    raw = [json.loads(line) for line in bound(result['blocks']).read_text().splitlines()]
    assert raw[0]['event'] == 'loaded'
    assert raw[-1] == {'event': 'complete', 'pairs': 21, 'exact': True}
    expected_events = ['loaded', 'warm', 'warm'] + ['block-start', 'block'] * 42 + ['clock-control'] * 21 + ['complete']
    assert [row['event'] for row in raw] == expected_events
    for warm, arm in zip(raw[1:3], ['original', 'allocation']):
        assert warm['arm'] == arm
        assert warm['result'] == {'records': 1392, 'checked_complex_outputs': 86436, 'exact': True,
            'control_only': False, 'total_ns': 0, 'intervals_ns': []}
    blocks = [row for row in raw if row['event'] == 'block']
    starts = [row for row in raw if row['event'] == 'block-start']
    values = {'original': [], 'allocation': []}
    shape_values = {shape: {'original': [], 'allocation': []} for shape in sorted(set(shapes))}
    for index, (start, row) in enumerate(zip(starts, blocks)):
        pair, position = divmod(index, 2)
        arm = plan['pair_order'][pair][position]
        assert {key: row[key] for key in ['pair', 'position', 'arm']} == {'pair': pair, 'position': position, 'arm': arm}
        assert {key: start[key] for key in ['pair', 'position', 'arm']} == {'pair': pair, 'position': position, 'arm': arm}
        measurement = row['result']
        assert measurement['records'] == 1392 and measurement['checked_complex_outputs'] == 86436
        assert measurement['exact'] is True and measurement['control_only'] is False
        intervals = measurement['intervals_ns']
        assert len(intervals) == 1392 and all(type(v) is int and v > 0 for v in intervals)
        assert sum(intervals) == measurement['total_ns']
        values[arm].append(measurement['total_ns'])
        for shape in shape_values:
            shape_values[shape][arm].append(sum(v for v, actual in zip(intervals, shapes) if actual == shape))
    controls = [row for row in raw if row['event'] == 'clock-control']
    overhead = []
    for index, row in enumerate(controls):
        assert row['index'] == index
        measurement = row['result']
        assert measurement['records'] == 1392 and measurement['checked_complex_outputs'] == 0
        assert measurement['exact'] is False and measurement['control_only'] is True
        assert len(measurement['intervals_ns']) == 1392
        assert all(type(v) is int and v > 0 for v in measurement['intervals_ns'])
        assert sum(measurement['intervals_ns']) == measurement['total_ns']
        overhead.append(measurement['total_ns'])
    original, allocation = values['original'], values['allocation']
    assert len(original) == len(allocation) == len(overhead) == 21
    rng = random.Random(20261020)
    bootstrap = []
    for _ in range(100000):
        indices = rng.choices(range(21), k=21)
        bootstrap.append(statistics.median(allocation[i] for i in indices) / statistics.median(original[i] for i in indices))
    interval = [quantile(bootstrap, .025), quantile(bootstrap, .975)]
    ratio = statistics.median(allocation) / statistics.median(original)
    overhead_fraction = statistics.median(overhead) / statistics.median(original)
    return {'schema': 'pure-tensor-allocation-kernel-result-v1', 'acceptance_evidence': False,
        'production_changed': False, 'plan_identity': plan['identity'], 'result_sha256': sha(result_path),
        'trace_sha256': result['trace_sha256'], 'pairs': 21, 'records_per_block': 1392,
        'exact_complex_outputs_per_block': 86436, 'all_42_blocks_exact': True,
        'original': spread(original), 'allocation': spread(allocation),
        'original_totals_ns': original, 'allocation_totals_ns': allocation,
        'ratio_of_medians': ratio, 'paired_ratios': [a / o for a, o in zip(allocation, original)],
        'paired_bootstrap_95ci': interval,
        'bootstrap': {'samples': 100000, 'seed': 20261020, 'rng': 'Python random.Random',
            'resampling': 'complete adjacent pair indices', 'percentile': 'linear interpolation at (n-1)*p'},
        'clock_control': {'totals_ns': overhead, 'median_fraction': overhead_fraction,
            'maximum_fraction': .01, 'subtracted': False, 'passed': overhead_fraction <= .01},
        'kernel_statistical_gain': interval[1] < 1 and overhead_fraction <= .01,
        'quiet_window_context_requires_independent_review': True,
        'full_process_speedup_claim': False,
        'shapes_descriptive_only': {str(shape): {'records': shapes.count(shape),
            'ratio_of_medians': statistics.median(rows['allocation']) / statistics.median(rows['original']),
            'original': spread(rows['original']), 'allocation': spread(rows['allocation'])}
            for shape, rows in shape_values.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summary = analyze(args.result.resolve())
    if args.output:
        with args.output.open('x') as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write('\n')
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
