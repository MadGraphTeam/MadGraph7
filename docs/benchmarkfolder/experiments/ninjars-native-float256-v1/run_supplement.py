#!/usr/bin/env python3
"""Run supplemental unit/control stages, or the separately handed-off screen."""
import sys
sys.dont_write_bytecode = True
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT/'.benchmark-work/experiments/ninjars-native-float256-v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
TESTS = {'focused':'native_study_tests::supplement::returned_shape_error_retains_entries_and_late_policy',
         'screening':'native_study_tests::supplement::fixed_request_screening_seven_triplets'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('unit',*TESTS))
    parser.add_argument('--cpu',type=int,required=True)
    parser.add_argument('--handoff',type=Path)
    args = parser.parse_args()
    assert Path.cwd() == ROOT and args.cpu != 207
    build_path = HERE/'supplement/build-completed.json'
    build = json.loads(build_path.read_text())
    assert build['complete'] and not build['executed']
    executable = ROOT/build['executables'][0]['path']
    assert sha(executable) == build['executables'][0]['sha256']
    for name,digest in build['source_hashes'].items(): assert sha(WORK/'supplement/bridge'/name)==digest,name
    original = json.loads((HERE/'build.json').read_text())
    assert sha(ROOT/original['executables'][0]['path']) == original['executables'][0]['sha256']
    source = WORK/'actual-mp-requests.jsonl'
    prepared = json.loads((HERE/'prepared.json').read_text())
    assert sha(source) == prepared['capture']['uncompressed_sha256']
    if args.stage != 'unit': assert json.loads((HERE/'supplement/run-unit/result.json').read_text())['complete']
    if args.stage == 'screening':
        assert args.cpu != 2, 'The screening controller uses CPU2'
        assert json.loads((HERE/'supplement/run-focused/result.json').read_text())['complete']
        assert args.handoff is not None
        handoff = json.loads(args.handoff.read_text())
        assert handoff['root_authorized'] is True and handoff['phase']=='native-screening'
        assert handoff['cpu']==args.cpu and handoff['screening_plan_sha256']==sha(HERE/'SCREENING-PLAN.md')
        assert handoff['executable_sha256']==sha(executable)
        preflight_path=ROOT/handoff['preflight_path']
        assert sha(preflight_path)==handoff['preflight_sha256']
        preflight=json.loads(preflight_path.read_text())
        assert preflight['complete'] and preflight['qualified'] and preflight['cpu']==args.cpu
        assert preflight['plan_sha256']==sha(HERE/'SCREENING-PLAN.md')
        assert preflight['interval_count']==3 and preflight['interval_seconds']==3
        assert preflight['maximum_busy_fraction']==.05 and preflight['performance_samples_collected']==0
    destination = HERE/'supplement'/('run-'+args.stage)
    destination.mkdir(exist_ok=False)
    command = ['taskset','-c',str(args.cpu),str(executable)]
    if args.stage != 'unit': command += ['--ignored','--exact',TESTS[args.stage]]
    command += ['--test-threads=1','--nocapture']
    env = dict(os.environ)
    env.update(TMPDIR=str(ROOT/'.ninjars-build/tmp'),
               SYMBOLICA_LOCK_DIR=str(ROOT/'.ninjars-build/symbolica-locks'),
               NINJARS_NATIVE_STUDY_INPUT=str(source),
               NINJARS_NATIVE_STUDY_OUTPUT=str(destination/'raw.jsonl'),
               NINJARS_NATIVE_STUDY_REPLAY=str(HERE/'run-replay/raw.jsonl'))
    thread_environment={name:'1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS','MKL_NUM_THREADS')}
    env.update(thread_environment)
    ldd = subprocess.check_output(['ldd',str(executable)],text=True)
    assert 'not found' not in ldd
    (destination/'ldd.txt').write_text(ldd)
    libraries = {}
    for name in re.findall(r'(?:=>\s+)?(/\S+)',ldd):
        path=Path(name).resolve(strict=True)
        libraries[name]={'resolved':str(path),'sha256':sha(path)}
    result={'complete':False,'stage':args.stage,'production_modified':False,'command':command,
        'cpu':args.cpu,'timeout_seconds':3600,'runner_sha256':sha(Path(__file__)),
        'build_completed_sha256':sha(build_path),'executable_sha256':sha(executable),
        'input_sha256':sha(source),'controls_sha256':sha(HERE/'run-replay/raw.jsonl'),
        'shared_libraries':libraries,'full_MadLoop_performance_claim':False,
        'thread_environment':thread_environment,
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if args.handoff: result['handoff']={'path':str(args.handoff),'sha256':sha(args.handoff)}
    monitor_stop=threading.Event()
    monitor=None
    if args.stage=='screening':
        # Keep the controller/host sampler off the measured CPU. A taskset
        # child may explicitly expand its own affinity to the chosen CPU.
        os.sched_setaffinity(0,{2})
        result['controller_affinity']=[2]
        def snapshot():
            names={f'cpu{args.cpu}','cpu207'}
            counters={}
            for line in Path('/proc/stat').read_text().splitlines():
                fields=line.split()
                if fields[0] in names:counters[fields[0]]=[int(v) for v in fields[1:]]
            frequency={}
            for cpu in (args.cpu,207):
                path=Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq')
                frequency[str(cpu)]=path.read_text().strip() if path.exists() else None
            return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'monotonic_ns':time.monotonic_ns(),'cpu_jiffies':counters,
                'loadavg':Path('/proc/loadavg').read_text().strip(),'frequency_khz':frequency}
        def watch():
            with (destination/'activity.jsonl').open('x') as stream:
                while not monitor_stop.is_set():
                    stream.write(json.dumps(snapshot())+'\n');stream.flush()
                    monitor_stop.wait(5)
                stream.write(json.dumps(snapshot())+'\n')
        monitor=threading.Thread(target=watch)
        monitor.start()
    process=None
    start=time.monotonic()
    try:
        with (destination/'stdout').open('x') as out,(destination/'stderr').open('x') as err:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
            result['pid']=process.pid
            (destination/'launch.json').write_text(json.dumps(result,indent=2)+'\n')
            try: result['returncode']=process.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                raise
        assert result['returncode']==0
        stdout=(destination/'stdout').read_text()
        assert ('18 passed; 0 failed;' if args.stage=='unit' else '1 passed; 0 failed;') in stdout
        if args.stage!='unit':
            # Only read the final JSON line here; complete raw-output analysis
            # is independent, and the request timer may retain many megabytes.
            with (destination/'raw.jsonl').open() as stream:
                final=None
                for line in stream: final=line
            summary=json.loads(final)
            assert summary['kind']=='summary' and summary['complete'] is True
            result['summary']=summary
        assert sha(executable)==result['executable_sha256'] and sha(source)==result['input_sha256']
        assert sha(ROOT/original['executables'][0]['path'])==original['executables'][0]['sha256']
        for name,digest in build['source_hashes'].items(): assert sha(WORK/'supplement/bridge'/name)==digest,name
        for name,record in libraries.items():
            assert str(Path(name).resolve(strict=True))==record['resolved']
            assert sha(Path(record['resolved']))==record['sha256']
        result['complete']=True
    except BaseException as error:
        result['failure']=repr(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            result['exceptional_owned_process_cleanup']=True
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            result['cleanup_returncode']=process.returncode
        if monitor is not None:
            monitor_stop.set()
            monitor.join()
        result['elapsed_seconds']=time.monotonic()-start
        result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        result['files']={name:sha(destination/name) for name in ('stdout','stderr','raw.jsonl','ldd.txt','activity.jsonl') if (destination/name).is_file()}
        with (destination/'result.json').open('x') as stream:
            json.dump(result,stream,indent=2)
            stream.write('\n')


if __name__=='__main__':
    def interrupted(signum,_frame):
        raise SystemExit(f'Controller interrupted by signal{signum}; owned process cleanup follows')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGHUP,interrupted)
    main()
