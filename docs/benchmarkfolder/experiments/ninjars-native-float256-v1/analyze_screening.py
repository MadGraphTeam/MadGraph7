#!/usr/bin/env python3
"""Recompute the fixed request-screen series and all output controls offline."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ORDERS=((0,1,2),(1,2,0),(2,0,1),(0,2,1),(2,1,0),(1,0,2),(0,1,2))
POLICIES=('Interpreted','NativeDenominators','FullNative')


def percentile(values,fraction):
    position=(len(values)-1)*fraction
    left=int(position)
    right=min(left+1,len(values)-1)
    return values[left]+(values[right]-values[left])*(position-left)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    run=args.run.resolve(strict=True)
    result=json.loads((run/'result.json').read_text())
    assert result['complete'] and result['returncode']==0 and result['stage']=='screening'
    assert result['thread_environment']=={name:'1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS','MKL_NUM_THREADS')}
    assert result['cpu'] not in (2,207) and result['controller_affinity']==[2]
    for name,digest in result['files'].items():
        assert sha(run/name)==digest,name
    handoff_path=Path(result['handoff']['path'])
    if not handoff_path.is_absolute():handoff_path=HERE.parents[3]/handoff_path
    assert sha(handoff_path)==result['handoff']['sha256']
    handoff=json.loads(handoff_path.read_text())
    preflight_path=HERE.parents[3]/handoff['preflight_path']
    assert sha(preflight_path)==handoff['preflight_sha256']
    preflight=json.loads(preflight_path.read_text())
    assert handoff['root_authorized'] is True and handoff['phase']=='native-screening'
    assert handoff['cpu']==preflight['cpu']==result['cpu'] and preflight['complete'] and preflight['qualified']
    assert handoff['executable_sha256']==result['executable_sha256']
    assert handoff['screening_plan_sha256']==preflight['plan_sha256']==sha(HERE/'SCREENING-PLAN.md')
    raw=run/'raw.jsonl'
    assert sha(raw)==result['files']['raw.jsonl']
    activity_path=run/'activity.jsonl'
    assert sha(activity_path)==result['files']['activity.jsonl']
    activity=[json.loads(line) for line in activity_path.read_text().splitlines()]
    assert len(activity)>2 and all(a['monotonic_ns']<b['monotonic_ns'] for a,b in zip(activity,activity[1:]))
    replay_path=HERE/'run-replay/raw.jsonl'
    assert sha(replay_path)==result['controls_sha256']
    replay=[json.loads(line) for line in replay_path.read_text().splitlines()]
    assert len(replay)==1393 and replay[-1]['kind']=='summary'
    expected={row['sequence']:row['arms'] for row in replay[:-1]}
    sample_outputs=Counter()
    warmup_outputs=Counter()
    measurements=[]
    warmups=[]
    failures=[]
    summary=None
    with raw.open() as stream:
        for line in stream:
            row=json.loads(line)
            kind=row['kind']
            if kind in ('sample_output','warmup_output'):
                arm,sequence=row['arm'],row['sequence']
                observed=row['result']; control=expected[sequence][arm]
                good=(observed['status']==control['status']==0 and observed['output']==control['output']
                    and observed['output_bytes']==8*64)
                assert observed['passed']==good
                if not good:failures.append(row)
                if kind=='sample_output':
                    assert ORDERS[row['triplet']][row['position']]==arm
                    key=(row['triplet'],row['position'],arm)
                    assert sequence==sample_outputs[key]
                    sample_outputs[key]+=1
                else:
                    assert sequence==warmup_outputs[arm]
                    warmup_outputs[arm]+=1
            elif kind in ('sample','warmup'):
                assert row['policy']==POLICIES[row['arm']] and row['requests']==1392
                assert int(row['elapsed_ns'])>0
                (measurements if kind=='sample' else warmups).append(row)
            elif kind=='summary':
                assert summary is None
                summary=row
            else:raise AssertionError(kind)
    assert summary==result['summary'] and summary['complete'] and summary['input_unchanged']
    assert summary['samples']==21 and summary['triplets']==7 and summary['requests_per_sample']==1392
    assert summary['warmup_streams']==3
    assert summary['failed_outputs']==len(failures) and not failures
    assert list(warmup_outputs.items())==[(0,1392),(1,1392),(2,1392)]
    assert [r['arm'] for r in warmups]==[0,1,2]
    assert all(r['performance_sample'] is False for r in warmups)
    expected_samples=[(triplet,position,arm) for triplet,order in enumerate(ORDERS) for position,arm in enumerate(order)]
    assert [(r['triplet'],r['position'],r['arm']) for r in measurements]==expected_samples
    assert sample_outputs==Counter({key:1392 for key in expected_samples})
    values={arm:[int(next(r for r in measurements if r['triplet']==triplet and r['arm']==arm)['elapsed_ns'])/1e9
                 for triplet in range(7)] for arm in range(3)}
    warmup_seconds={r['policy']:int(r['elapsed_ns'])/1e9 for r in warmups}
    timed_total=sum(sum(v) for v in values.values())
    untimed_lifecycle_remainder=result['elapsed_seconds']-timed_total-sum(warmup_seconds.values())
    assert untimed_lifecycle_remainder>=0
    comparisons={}
    for arm in (1,2):
        median_ratio=statistics.median(values[arm])/statistics.median(values[0])
        # Identical fixed index resamples per native arm preserve pairing and
        # make every ratio reconstructible from the retained21observations.
        rng=random.Random(20261021)
        ratios=[]
        for _ in range(100000):
            indices=[rng.randrange(7) for _ in range(7)]
            ratios.append(statistics.median(values[arm][i] for i in indices)/statistics.median(values[0][i] for i in indices))
        ratios.sort()
        interval=[percentile(ratios,.025),percentile(ratios,.975)]
        comparisons[POLICIES[arm]]={'ratio_of_medians':median_ratio,'paired_bootstrap_95_interval':interval,
            'demonstrated_request_scope_gain':interval[1]<1}
    print(json.dumps({'complete':True,'engine_calls':0,'analysis_source_sha256':sha(Path(__file__)),
        'raw_sha256':sha(raw),'result_sha256':sha(run/'result.json'),'controls_sha256':sha(replay_path),
        'plan_sha256':sha(HERE/'SCREENING-PLAN.md'),'samples':21,'triplets':7,
        'host_activity_sha256':sha(activity_path),'host_activity_snapshots':len(activity),
        'records_per_sample':1392,'timed_outputs_checked':21*1392,'warmup_outputs_checked':3*1392,
        'all_output_status_controls_exact':True,'all_samples_retained':True,
        'seconds_per_complete_1392_request_stream':{POLICIES[a]:values[a] for a in range(3)},
        'median_seconds':{POLICIES[a]:statistics.median(values[a]) for a in range(3)},
        'warmup_seconds_not_performance_samples':warmup_seconds,
        'controller_elapsed_seconds':result['elapsed_seconds'],
        'untimed_lifecycle_remainder_seconds':untimed_lifecycle_remainder,
        'untimed_lifecycle_remainder_scope':'Controller wall minus measured evaluation and warmup intervals; includes initialization, validation, serialization, output release, teardown and host-monitor completion; no subtraction from reported samples',
        'comparisons':comparisons,'bootstrap':{'resamples':100000,'seed':20261021,'unit':'triplet indices','percentiles':'linear2.5%97.5%'},
        'scope':'privateMpState::evaluate requestscreen only; includesASCII parse/tensor/reduction/masters/format; excludesFortran/controller/managerIPC',
        'full_MadLoop_performance_claim':False,'original_Ninja_or_5_percent_gate_claim':False},indent=2))


if __name__=='__main__':main()
