#!/usr/bin/env python3
"""Read-only fixed-block timing analysis; all retained samples participate."""
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(report):
    plan=json.loads((report/'plan.json').read_text())
    data=json.loads((report/'measurements.json').read_text())
    assert plan['complete'] and data['complete'] and data['all_samples_retained']
    assert data['plan_sha256']==sha(report/'plan.json')
    assert data['preflight_sha256']==sha(report/'preflight.json')
    assert plan['runner_sha256']==sha(report/'run.py')
    assert len(plan['configurations'])==12 and len(data['results'])==36
    configurations={r['key']:r for r in plan['configurations']}
    expected={(r['key'],f['case']) for r in plan['configurations'] for f in r['fixtures']}
    assert {(r['key'],r['case']) for r in data['results']}==expected
    rng=np.random.default_rng(20261020)
    bootstrap=rng.integers(0,21,size=(10000,21))
    summary=dict(schema='fixed-dp-kernel-analysis-v1',complete=False,scope=plan['scope'],
                 full_process_claim=False,plan_sha256=sha(report/'plan.json'),
                 measurements_sha256=sha(report/'measurements.json'),
                 analyzer_sha256=sha(Path(__file__)),
                 statistics='Median of21 within-block candidate/reference ratios; percentile95% paired bootstrap CI,10000 fixed-seed resamples; no exclusions',
                 bootstrap_seed=20261020,bootstrap_samples=10000,results=[])
    for record in data['results']:
        config=configurations[record['key']]
        work=report/record['key']
        assert sha(work/f'case-{record["case"]}.log')==record['log_sha256']
        for name,value in config['generated_sources'].items():
            assert sha(work/name)==value
        for f in config['fixtures']:
            assert sha(work/f['file'])==f['sha256']
        samples=record['samples']
        assert len(samples)==84 and [(r['pair'],r['position']) for r in samples]==[(p,o) for p in range(1,22) for o in range(1,5)]
        metrics={}
        for clock in ('cpu_seconds','wall_seconds'):
            arms={i:np.array([next(s[clock] for s in samples if s['pair']==p and s['arm']==i) for p in range(1,22)]) for i in range(4)}
            assert all(np.isfinite(values).all() and (values>0).all() for values in arms.values())
            comparisons={}
            for candidate,reference in ((1,0),(2,0),(3,0),(2,1),(3,1)):
                ratios=arms[candidate]/arms[reference]
                distribution=np.median(ratios[bootstrap],axis=1)
                interval=np.quantile(distribution,[.025,.975],method='linear')
                key=plan['arms'][candidate]+'/'+plan['arms'][reference]
                comparisons[key]=dict(median=float(np.median(ratios)),ci95=interval.tolist(),ratios=ratios.tolist())
            metrics[clock]=dict(median_seconds={plan['arms'][i]:float(np.median(values)) for i,values in arms.items()},
                                minimum_seconds={plan['arms'][i]:float(np.min(values)) for i,values in arms.items()},comparisons=comparisons)
        summary['results'].append(dict(key=record['key'],case=record['case'],
            fixture=next(f for f in config['fixtures'] if f['case']==record['case']),metrics=metrics))
    summary['complete']=True
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    print(json.dumps(analyze(BASE/args.run_id),indent=2,sort_keys=True))
