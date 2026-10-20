#!/usr/bin/env python3
"""Independently reparse every raw output and apply the fixed paired estimator."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import math
from pathlib import Path
import random
import statistics
from build import HERE, ROOT, binding, sha, write
sys.path[:0] = [str(ROOT / 'tools')]
import ninjars_benchmark_workers as workers
from comparison import compare

def percentile(values,q):
    pos = (len(values)-1)*q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    return values[lower]+(pos-lower)*(values[upper]-values[lower])

def summary(a,b):
    rng = random.Random(20261020)
    n = len(a)
    ratios = []
    for _ in range(100000):
        indices = rng.choices(range(n),k=n)
        ratios.append(statistics.median(b[i] for i in indices)/statistics.median(a[i] for i in indices))
    ratios.sort()
    def describe(values):
        med = statistics.median(values)
        return {'samples':values,'median':med,'minimum':min(values),'maximum':max(values),
            'stdev':statistics.stdev(values),'median_absolute_deviation':statistics.median(abs(x-med) for x in values)}
    return {'ratio':statistics.median(b)/statistics.median(a),
        'confidence_interval_95':[percentile(ratios,.025),percentile(ratios,.975)],
        'original':describe(a),'allocation':describe(b),
        'bootstrap':{'kind':'ratio of medians, paired percentile, linear interpolation','resamples':100000,'seed':20261020}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt',type=Path)
    args = parser.parse_args()
    dest = args.attempt.resolve()
    result = json.loads((dest / 'result.json').read_text())
    assert result['status'] == 'complete-unanalysed' and len(result['samples']) == 56
    assert all(r['busy_fraction'] <= .05 for r in result['preflight'])
    all_samples = {}
    for item in result['samples']:
        path = ROOT / item['path']
        assert binding(path) == item
        data = json.loads(path.read_text())
        assert data['exit_code'] == 0 and data['loops'] == 1
        parsed = workers.parse_outputs(path.parent,8,100,1)
        for k in parsed:
            assert parsed[k] == data[k], (path,k)
        for name,record in data['raw_files'].items():
            assert binding(path.parent / name) == record
        key = data['case'],data['pair'],data['arm']
        assert key not in all_samples
        all_samples[key] = (path,data)
    answer = {'schema':'allocation-process-analysis-v1','source_result':binding(dest / 'result.json'),
        'controller':binding(Path(__file__)),'acceptance_eligible':False,'cases':{}}
    for case,count in [('qcd-3',21),('ew-2',7)]:
        a,b = [],[]
        ref_path,ref = all_samples[case,0,'original']
        checks = 0
        for pair in range(count):
            for arm,times in [('original',a),('allocation',b)]:
                path,data = all_samples[case,pair,arm]
                assert data['rows'] == ref['rows']
                for name in ['dimensions.dat','values.dat','timed_status.dat']:
                    assert (path.parent / name).read_bytes() == (ref_path.parent / name).read_bytes()
                comp = compare({'reference':ref['rows'],'current':data['rows']})
                assert comp['passed']
                checks += comp['comparisons']
                times.append(data['seconds_per_point'])
        stats = summary(a,b)
        upper = stats['confidence_interval_95'][1]
        stats.update(pairs=count,exact_rows_and_raw_values=True,physical_component_comparisons=checks,
            status_counts=ref['status_counts'],timed_status_counts=ref['timed_status_counts'],
            decision=('gain-demonstrated' if upper < 1 else 'gain-unresolved-or-absent') if case == 'qcd-3'
                else ('regression-gate-passed' if upper <= 1.05 else 'regression-gate-unresolved-or-failed'))
        answer['cases'][case] = stats
    answer['promotion_gates_passed'] = answer['cases']['qcd-3']['confidence_interval_95'][1] < 1 and answer['cases']['ew-2']['confidence_interval_95'][1] <= 1.05
    write(dest / 'analysis.json',answer)
    print(json.dumps(answer,indent=2),flush=True)

if __name__ == '__main__':
    main()
