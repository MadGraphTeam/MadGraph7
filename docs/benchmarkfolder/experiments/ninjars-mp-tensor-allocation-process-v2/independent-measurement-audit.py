#!/usr/bin/env python3
"""Reparse every retained sample and recompute the frozen paired estimator."""
import ast
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
ATTEMPT = HERE/'timings/attempt-1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def bound(record):
    path = ROOT/record['path']
    assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], path
    return path


def parse(directory):
    cold,seconds = map(float,(directory/'timing.dat').read_text().split())
    assert all(math.isfinite(x) and x > 0 for x in (cold,seconds))
    ndim,nso,nborn = map(int,(directory/'dimensions.dat').read_text().split())
    assert ndim == max(nso,nborn) and nborn <= nso
    rows,counts = [],{}
    for line in (directory/'values.dat').read_text().splitlines():
        f = line.split(); assert len(f) == 8
        p,order,status = map(int,f[:3]); values = list(map(float,f[3:]))
        assert status in (218,322) and all(math.isfinite(x) for x in values) and values[-1] >= 0
        rows.append(dict(point=p,order=order,status=status,coefficients=values[:4],accuracy=values[4]))
        if order == 0:
            counts[str(status)] = counts.get(str(status),0)+1
    assert [(r['point'],r['order']) for r in rows] == [(p,o) for p in range(1,101) for o in range(nso+1)]
    timed = {}
    for line in (directory/'timed_status.dat').read_text().splitlines():
        code,count = map(int,line.split()); assert code in (218,322) and count > 0 and str(code) not in timed
        timed[str(code)] = count
    assert counts == timed and sum(timed.values()) == 100
    return dict(cold_seconds=cold,seconds=seconds,seconds_per_point=seconds/100,
        rows=rows,status_counts=counts,timed_status_counts=timed,
        answer_dimensions=dict(allocated=ndim,loop_orders=nso,born_orders=nborn))


def summary(a,b):
    rng = random.Random(20261020)
    ratios = []
    for _ in range(100000):
        ids = rng.choices(range(len(a)),k=len(a))
        ratios.append(statistics.median(b[i] for i in ids)/statistics.median(a[i] for i in ids))
    ratios.sort()
    def percentile(q):
        x = (len(ratios)-1)*q; lo,hi = math.floor(x),math.ceil(x)
        return ratios[lo]+(x-lo)*(ratios[hi]-ratios[lo])
    return statistics.median(b)/statistics.median(a),[percentile(.025),percentile(.975)]


def main():
    os.sched_setaffinity(0,{3})
    frozen_path = HERE/'frozen-bindings.json'
    assert sha(frozen_path) == '64766a546d751705993e4b074f830d9ff32260c89e4ecfd6f6e4aa9114e2871e'
    frozen = load(frozen_path)
    # The controller's asset reader executes no compiler, worker or amplitude.
    sys.path[:0] = [str(HERE),str(ROOT),str(ROOT/'tools')]
    spec = importlib.util.spec_from_file_location('allocation_bound_asset_reader',HERE/'run-processes.py')
    control = importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    actual = control.assets()
    assert all(actual[key] == frozen[key] for key in actual)
    result,analysis = load(ATTEMPT/'result.json'),load(ATTEMPT/'analysis.json')
    assert result['status'] == 'complete-unanalysed' and len(result['samples']) == 56
    assert result['frozen_bindings']['sha256'] == sha(frozen_path)
    assert result['affinity'] == [234] and result['cpu'] == 234
    assert len(result['preflight']) == 3 and all(x['busy_fraction'] <= .05 for x in result['preflight'])
    sequence = [(case,pair,arm) for case,n in [('qcd-3',21),('ew-2',7)]
                for pair in range(n) for arm in (['original','allocation'] if pair%2 == 0 else ['allocation','original'])]
    samples,raw_count,pids = {},0,set()
    for record,key in zip(result['samples'],sequence):
        path = bound(record); data = load(path)
        assert (data['case'],data['pair'],data['arm']) == key and key not in samples
        assert data['exit_code'] == 0 and data['backend'] == 8 and data['loops'] == 1 and data['cpu'] == 234
        assert data['worker_pid'] not in pids;pids.add(data['worker_pid'])
        for name,value in parse(path.parent).items():assert data[name] == value,(path,name)
        for raw in data['raw_files'].values():bound(raw);raw_count += 1
        prepared = frozen['builds'][data['case']][data['arm']]
        for name in ('executable','points','parameter_card'):
            assert data[name+'_sha256'] == prepared[name]['sha256']
        assert data['shared_numerical_settings'] == prepared['shared_numerical_settings']
        for name,value in frozen['thread_environment'].items():assert data['execution_environment'][name] == value
        samples[key] = (path,data)
    outcomes,checks = {},0
    for case,count in [('qcd-3',21),('ew-2',7)]:
        reference_path,reference = samples[case,0,'original'];a,b=[],[];case_checks=0
        for pair in range(count):
            for arm,times in [('original',a),('allocation',b)]:
                path,data = samples[case,pair,arm]
                assert data['rows'] == reference['rows']
                for name in ('dimensions.dat','values.dat','timed_status.dat'):
                    assert (path.parent/name).read_bytes() == (reference_path.parent/name).read_bytes()
                case_checks += sum(len(row['coefficients']) for row in data['rows'])
                times.append(data['seconds_per_point'])
        ratio,ci = summary(a,b);prior=analysis['cases'][case]
        assert a == prior['original']['samples'] and b == prior['allocation']['samples']
        assert ratio == prior['ratio'] and ci == prior['confidence_interval_95']
        assert case_checks == prior['physical_component_comparisons']
        checks += case_checks
        outcomes[case] = dict(pairs=count,ratio=ratio,confidence_interval_95=ci,
            component_comparisons=case_checks,status_counts=reference['status_counts'])
    assert checks == 44800
    assert outcomes['qcd-3']['confidence_interval_95'][1] < 1
    assert outcomes['ew-2']['confidence_interval_95'][1] <= 1.05
    assert analysis['promotion_gates_passed'] and not analysis['acceptance_eligible']
    harness = ROOT/'docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1'
    parsed = []
    for path in sorted(harness.glob('*.py')):
        ast.parse(path.read_text(),filename=str(path));parsed.append(str(path.relative_to(ROOT)))
    record = dict(schema='allocation-independent-measurement-audit-v2',passed=True,
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_affinity=[3],
        reviewer_script_sha256=sha(Path(__file__)),frozen_bindings_sha256=sha(frozen_path),
        source_result_sha256=sha(ATTEMPT/'result.json'),source_analysis_sha256=sha(ATTEMPT/'analysis.json'),
        workers=56,raw_file_bindings=raw_count,physical_component_checks=checks,cases=outcomes,
        actual_assets_equal_pre_sample_freeze=True,historical_samples_reused=False,
        harness_ast_parse=dict(passed=True,files=parsed),
        test_compiler_or_engine_executions=0,all_owned_children_exited=True,
        limit='Allocation promotion gates only; final full-process acceptance remains separate.')
    with (HERE/'independent-measurement-audit.json').open('x') as stream:
        json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
