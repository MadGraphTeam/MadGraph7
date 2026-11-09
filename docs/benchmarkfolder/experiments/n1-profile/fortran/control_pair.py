#!/usr/bin/env python3
"""Diagnostic paired original n1 workers; no instrumentation or acceptance claim."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT=Path(__file__).resolve().parents[5]
HERE=Path(__file__).resolve().parent
WORK=ROOT/'.tirust-build/n1-profile/fortran'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE/'protocol'))
import tirust_benchmark as campaign
campaign.ROOT=ROOT
campaign.REPORT=ROOT/'docs/benchmarkfolder'
campaign.WORK=ROOT/'.benchmark-work/tirust'
campaign.BASELINE=campaign.WORK/'baseline'
from tirust_benchmark_worker import Worker,parse_outputs,paired_statistics
from prepare import clone,digest
from madgraph.various.banner import MadLoopParam


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--cpu',type=int,default=1)
    parser.add_argument('--loops',type=int,default=200)
    parser.add_argument('--pairs',type=int,default=7)
    args=parser.parse_args()
    if not args.name.replace('-','').replace('_','').isalnum(): raise ValueError('Invalid output name')
    report=HERE/args.name
    report.mkdir(exist_ok=False)
    frozen=ROOT/'docs/benchmarkfolder/runs/qcd_n1_uux_z/tirust-first-attempt-3'
    points=ROOT/'docs/benchmarkfolder/points/qcd_n1_uux_z.dat'
    result={'performance_eligible':False,'purpose':'original uninstrumented paired controls for stage attribution',
            'cpu':args.cpu,'loops':args.loops,'pairs':args.pairs,'points_sha256':digest(points),
            'controller_sha256':digest(__file__),'worker_controller_sha256':digest(HERE/'protocol/tirust_benchmark_worker.py'),
            'inputs':{},'runs':[]}
    with ExitStack() as stack:
        lock=stack.enter_context((ROOT/'.tirust-build/numerical-tests.lock').open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX)
        workers={}
        chains={}
        for source in ('baseline','candidate'):
            inputs=json.loads((frozen/(source+'-inputs.json')).read_text())
            original=ROOT/'.benchmark-work/tirust/outputs'/source/'qcd_n1_uux_z'
            original_process=original/'SubProcesses/P0_uux_z'
            if digest(original_process/'check')!=inputs['compilation']['executable_sha256']:
                raise RuntimeError('Original executable changed since qualified gate')
            if digest(original_process/'check_sa.f')!=inputs['compilation']['driver_sha256']:
                raise RuntimeError('Original worker source changed since qualified gate')
            if digest(points)!=inputs['points_sha256']:
                raise RuntimeError('Frozen points changed')
            if digest(original/'Cards/param_card.dat')!=inputs['param_card_sha256']:
                raise RuntimeError('Frozen parameter card changed')
            output=WORK/(args.name+'-'+source)
            clone(original,output)
            directory=output/'SubProcesses/P0_uux_z'
            path=output/'Cards/MadLoopParams.dat'
            card=MadLoopParam(str(path))
            for key,value in inputs['settings'].items(): card[key]=value
            card.write(str(path),template=str(path))
            for local in (output/'SubProcesses').rglob('MadLoopParams.dat'):
                if local.is_symlink(): local.unlink()
                shutil.copy2(path,local)
            for pattern in ('*HelFilter.dat','*LoopFilter.dat'):
                for local in (output/'SubProcesses').rglob(pattern): local.unlink()
            shutil.copy2(path,report/(source+'-MadLoopParams.dat'))
            chains[source]=campaign.parse_chain(inputs['settings']['mlreductionlib'])
            runtime=campaign.runtime_identity(directory)
            if runtime['libraries']!=inputs['runtime']['libraries']:
                dump(report/(source+'-runtime-mismatch.json'),runtime)
                raise RuntimeError('Original runtime libraries changed since qualified gate')
            result['inputs'][source]={'frozen_input_sha256':digest(frozen/(source+'-inputs.json')),
                'settings':inputs['settings'],'executable_sha256':digest(directory/'check'),
                'driver_sha256':digest(directory/'check_sa.f'),'param_card_sha256':digest(output/'Cards/param_card.dat'),
                'compilation':inputs['compilation'],'runtime':runtime}
            worker=Worker(['taskset','-c',str(args.cpu),'./check',str(points)],directory,
                          report/(source+'-worker.log'),env=campaign.environment())
            stack.callback(worker.close)
            workers[source]=worker
        times={source:[] for source in workers}
        paired_rows={}
        # One sustained untimed-for-statistics traversal per worker mitigates
        # initial frequency/load transients; retain it separately.
        sequence=[('calibrate',source) for source in workers]
        sequence += [(i,source) for i in range(args.pairs)
                     for source in (('baseline','candidate') if i%2==0 else ('candidate','baseline'))]
        for repetition,source in sequence:
            worker=workers[source]
            target=report/str(repetition)/source
            target.mkdir(parents=True)
            before=campaign.host_load_snapshot()
            worker.run(args.loops)
            for filename in ('timing.dat','values.dat','timed_status.dat','dimensions.dat'):
                shutil.copy2(worker.directory/filename,target/filename)
            observation=parse_outputs(target,chains[source],100,args.loops)
            row={'repetition':repetition,'source':source,'observation':observation,
                 'host_load_before':before,'host_load_after':campaign.host_load_snapshot()}
            result['runs'].append(row)
            dump(target/'observation.json',row)
            if repetition!='calibrate':
                times[source].append(observation['seconds_per_point'])
                paired_rows.setdefault(repetition,{})[source]=observation['rows']
            dump(report/'results.json',result)
        result['parity']=[campaign.compare_rows(pair['baseline'],pair['candidate'],1e-8)
                          for pair in paired_rows.values()]
        result['statistics']=paired_statistics(times['baseline'],times['candidate'],threshold=1.05,minimum_pairs=7)
        result['median_seconds_per_point']={s:statistics.median(x) for s,x in times.items()}
        result['timings']=times
        result['all_numerical_parity_passed']=all(x['passed'] for x in result['parity'])
        dump(report/'results.json',result)
    print(json.dumps({'statistics':result['statistics'],'median_seconds_per_point':result['median_seconds_per_point'],
                      'all_numerical_parity_passed':result['all_numerical_parity_passed']},indent=2))


if __name__=='__main__': main()
