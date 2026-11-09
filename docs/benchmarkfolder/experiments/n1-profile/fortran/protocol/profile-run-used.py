#!/usr/bin/env python3
"""Run isolated n1 attribution under the campaign numerical lock; not a gate."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.tirust-build/n1-profile/fortran'
sys.path.insert(0, str(ROOT/'tools'))
import tirust_benchmark as campaign
from tirust_benchmark_worker import Worker, parse_outputs


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def parse_profile(directory, points):
    clock = {}
    for line in (directory/'fortran-clock.txt').read_text().splitlines():
        name, *values = line.split()
        clock[name] = [float(x) for x in values] if len(values)>1 else float(values[0])
    rows = []
    overhead = clock['calibration_internal_seconds']/clock['calibration_iterations']
    for line in (directory/'fortran-profile.tsv').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        a,b,c,d,e,count,seconds = line.split()
        count,seconds = int(count),float(seconds)
        rows.append({'stage':int(a), 'ctmode':int(b), 'propagators':int(c),
                     'rank':int(d), 'precision':int(e), 'calls':count, 'seconds':seconds,
                     'seconds_per_call':seconds/count, 'seconds_per_point':seconds/points,
                     'timer_adjusted_estimate_seconds_per_call':max(0.,seconds/count-overhead)})
    return {'clock':clock, 'stages':rows,
            'bridge':json.loads((directory/'bridge-profile.json').read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--loops', type=int, default=200)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--name', required=True)
    parser.add_argument('--levels', default='pristine,0:0,1:0,0:1,0:2,1:2')
    parser.add_argument('--input-record', type=Path,
                        default=ROOT/'docs/benchmarkfolder/runs/qcd_n1_uux_z/tirust-first/candidate-inputs.json')
    args = parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0) or min(args.loops,args.repetitions)<1:
        raise ValueError('Invalid CPU or batch dimensions')
    if not args.name.replace('-','').replace('_','').isalnum():
        raise ValueError('Name must be a simple new directory name')
    levels = args.levels.split(',')
    allowed = {'pristine','0:0','1:0','0:1','0:2','1:1','1:2'}
    if len(set(levels)) != len(levels) or not set(levels)<=allowed:
        raise ValueError('Invalid or duplicate profile levels')
    report = HERE / args.name
    report.mkdir(exist_ok=False)
    inputs = json.loads(args.input_record.read_text())
    chain = campaign.parse_chain(inputs['settings']['mlreductionlib'])
    points = ROOT/'docs/benchmarkfolder/points/qcd_n1_uux_z.dat'
    specification = {'performance_eligible':False, 'purpose':'diagnostic stage attribution',
                     'cpu':args.cpu, 'loops':args.loops, 'repetitions':args.repetitions,
                     'levels':levels, 'input_record':str(args.input_record),
                     'input_record_sha256':digest(args.input_record),
                     'points_sha256':digest(points), 'chain':chain,
                     'run_source_sha256':digest(__file__), 'preparation':json.loads((WORK/'preparation.json').read_text()),
                     'measurement_limits':['instrumented attribution, not performance acceptance',
                        'per-stage wall clocks include calibrated timer overhead',
                        'control comparisons include DSO and instrumentation perturbations',
                        'batch CPU time excludes profile serialization and timer calibration'],
                     'settings':inputs['settings'], 'runs':[]}
    write_json(report/'specification.json', specification)
    from madgraph.various.banner import MadLoopParam
    reference = None
    profile_library=Path(specification['preparation']['bridge']['path'])
    actual_profile_hash=digest(profile_library)
    expected_profile_hash=specification['preparation']['bridge']['sha256']
    if expected_profile_hash is not None and expected_profile_hash!=actual_profile_hash:
        raise RuntimeError('Profiling DSO changed after preparation')
    specification['used_profile_library_sha256']=actual_profile_hash
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Preserve actual settings; do not alter clone cards/filters while a
        # previous diagnostic worker owns the shared numerical lock.
        for label in ('pristine','instrumented'):
            output = WORK/label
            card = MadLoopParam(str(output/'Cards/MadLoopParams.dat'))
            for key,value in inputs['settings'].items():
                card[key] = value
            card.write(str(output/'Cards/MadLoopParams.dat'))
            for card_path in (output/'SubProcesses').rglob('MadLoopParams.dat'):
                if card_path.is_symlink():
                    card_path.unlink()
                card.write(str(card_path))
            for pattern in ('*HelFilter.dat','*LoopFilter.dat'):
                for path in (output/'SubProcesses').rglob(pattern):
                    path.unlink()
            shutil.copy2(output/'Cards/MadLoopParams.dat', report/(label+'-MadLoopParams.dat'))
        # Run one worker at a time, avoiding extra live Symbolica sessions.
        for repetition in range(args.repetitions):
            order = levels if repetition%2==0 else list(reversed(levels))
            for level in order:
                label = 'pristine' if level=='pristine' else 'instrumented'
                directory = WORK/label/'SubProcesses/P0_uux_z'
                env = campaign.environment()
                if level!='pristine':
                    left,right=level.split(':')
                    env['ML_TIRUST_FORTRAN_PROFILE']=left
                    env['ML_TIRUST_BRIDGE_PROFILE']=right
                    profile_library=Path(specification['preparation']['bridge']['path'])
                    env['LD_LIBRARY_PATH']=str(profile_library.parent)+os.pathsep+env.get('LD_LIBRARY_PATH','')
                else:
                    profile_library=ROOT/'HEPTools/tirust/lib/libmadloop_tirust.so'
                target = report/f'repetition-{repetition}'/level.replace(':','-')
                target.mkdir(parents=True)
                linkage=subprocess.check_output(['ldd',str(directory/'check')],env=env,text=True,
                                                stderr=subprocess.STDOUT)
                (target/'ldd.txt').write_text(linkage)
                actual=[line.split('=>',1)[1].strip().split()[0] for line in linkage.splitlines()
                        if line.strip().startswith('libmadloop_tirust.so =>')]
                if len(actual)!=1 or Path(actual[0]).resolve()!=profile_library.resolve():
                    raise RuntimeError('Runtime resolved the wrong TIRust DSO; inspect retained ldd output')
                load_before=campaign.host_load_snapshot()
                with Worker(['taskset','-c',str(args.cpu),'./check',str(points)],
                            directory,target/'worker.log',env=env) as worker:
                    worker.run(args.loops)
                    for filename in ('timing.dat','values.dat','timed_status.dat','dimensions.dat'):
                        shutil.copy2(directory/filename,target/filename)
                    observation=parse_outputs(target,chain,100,args.loops)
                    if reference is None:
                        reference=observation['rows']
                    parity=campaign.compare_rows(reference,observation['rows'],1e-8)
                    if not parity['passed']:
                        write_json(target/'parity.json',parity)
                        raise RuntimeError('Profile changed numerical output')
                    record={'repetition':repetition,'level':level,'observation':observation,
                            'parity_maximum_relative_error':parity['maximum_relative_error'],
                            'host_load_before':load_before,'host_load_after':campaign.host_load_snapshot(),
                            'executable_sha256':digest(directory/'check'),
                            'tirust_runtime':{'path':actual[0],'sha256':digest(actual[0])}}
                    if level!='pristine':
                        for filename in ('fortran-clock.txt','fortran-profile.tsv','bridge-profile.json'):
                            shutil.copy2(directory/filename,target/filename)
                        record['profile']=parse_profile(target,args.loops*100)
                        record['bridge_failed_calls']=sum(x['failed_calls'] for x in record['profile']['bridge']['rows'])
                        if record['bridge_failed_calls']:
                            write_json(target/'failed-bridge-calls.json',record)
                            raise RuntimeError('Failed bridge calls make stage decomposition incomplete')
                        if level.startswith('1:'):
                            rows=record['profile']['stages']
                            expected={(stage,ctmode,3,2,1):args.loops*100
                                      for stage in range(1,8) for ctmode in (1,2)}
                            actual={(x['stage'],x['ctmode'],x['propagators'],x['rank'],x['precision']):x['calls']
                                    for x in rows}
                            record['expected_n1_stage_counts_passed']=actual==expected
                            if actual!=expected:
                                write_json(target/'unexpected-counts.json',record)
                                raise RuntimeError('Unexpected n1 call mix; retain and investigate before attribution')
                    specification['runs'].append(record)
                    write_json(target/'observation.json',record)
                    write_json(report/'results.json',specification)
    summaries={}
    for level in levels:
        rows=[x for x in specification['runs'] if x['level']==level]
        summaries[level]={'median_wall_seconds_per_point':statistics.median(
            x['observation']['seconds_per_point'] for x in rows),
            'maximum_numerical_relative_error':max(x['parity_maximum_relative_error'] for x in rows)}
        if level!='pristine':
            summaries[level]['median_cpu_seconds_per_point']=statistics.median(
                x['profile']['clock']['batch_cpu_seconds']/(args.loops*100) for x in rows)
    specification['summary']=summaries
    write_json(report/'results.json',specification)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
