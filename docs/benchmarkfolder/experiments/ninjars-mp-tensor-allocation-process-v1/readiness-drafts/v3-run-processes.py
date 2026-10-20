#!/usr/bin/env python3
"""Fixed paired process study, requiring a separate explicit quiet-window grant."""
import sys
sys.dont_write_bytecode = True
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from build import HERE, ROOT, WORK, binding, sources, sha, write
sys.path[:0] = [str(ROOT),str(ROOT / 'tools')]
import ninjars_benchmark as bench
import ninjars_benchmark_workers as workers
import ninjars_benchmark_policy_v2 as policy
from ninjars_report_v2 import compare

PREP = HERE / 'processes-final/preparation.json'
ARMS = ['original','allocation']
COUNTS = {'qcd-3':21,'ew-2':7}
THREAD_ENV = {n:'1' for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS','MKL_NUM_THREADS']}

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def assets():
    prep = json.loads(PREP.read_text())
    assert prep['complete'] and not prep['amplitudes_evaluated']
    package_path = ROOT / prep['build_result']['path']
    assert binding(package_path) == prep['build_result']
    package = json.loads(package_path.read_text())
    build_path = ROOT / package['build']['path']
    assert binding(build_path) == package['build']
    build = json.loads(build_path.read_text())
    assert sources(WORK / 'bridge') == build['allocation_sources']
    assert binding(ROOT / package['allocation_archive']['path']) == package['allocation_archive']
    assert binding(ROOT / package['baseline_archive']['path']) == package['baseline_archive']
    answer = {'preparation':binding(PREP), 'plan':binding(HERE / 'PLAN.md'),
        'controller':binding(Path(__file__)), 'analyzer':binding(HERE / 'analyze-processes.py'),
        'readiness_verifier':binding(HERE / 'verify-ready.py'),
        'package':binding(package_path),'rust_build':binding(build_path),
        'bridge_sources':build['allocation_sources'],
        'support_tools':{n:binding(ROOT / 'tools' / n) for n in ['ninjars_benchmark_policy_v2.py','ninjars_report_v2.py']},
        'thread_environment':THREAD_ENV,
        'protocol':workers.protocol_descriptor(),'builds':{}}
    for case,arms in prep['builds'].items():
        answer['builds'][case] = {}
        for arm,b in arms.items():
            output,directory = ROOT / b['output'],ROOT / b['directory']
            assert binding(directory / 'check') == b['executable']
            assert bench.generated_source_manifest(output)['files_sha256'] == b['generated_sources_sha256']
            assert policy.library_hashes(output) == b['libraries']
            meta = {n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in policy.runtime_metadata_files(output).items()}
            assert meta == b['runtime_metadata']
            assert {n:sha(output / n) for n in policy.RUNTIME_LINK_INPUTS} == b['link_inputs']
            for n in ['parameter_card','madloop_card','points']:
                assert binding(ROOT / b[n]['path']) == b[n]
            assert sha(output / 'lib/ninjars_lib/ninjars.h') == b['header_sha256']
            assert sha(output / 'lib/ninjars_lib/mninjars.f90') == b['module_source_sha256']
            workers.validate_param_card(directory,(output / 'Cards/param_card.dat').read_bytes())
            answer['builds'][case][arm] = b
    return answer

def cpu_stat(cpu):
    lines = Path('/proc/stat').read_text().splitlines()
    row = next(s.split()[1:] for s in lines if s.startswith(f'cpu{cpu} '))
    values = list(map(int,row))
    # guest counters are already included in user/nice; omit double counting.
    return {'values':values,'total':sum(values[:8]),'idle':values[3]+values[4]}

def validate_assets(frozen):
    actual = assets()
    assert all(actual[k] == frozen[k] for k in actual)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--cpu',type=int)
    parser.add_argument('--authorized-run')
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    if args.freeze:
        assert not args.authorized_run
        data = assets()
        data.update(schema='allocation-process-frozen-bindings-v1',numerical_execution=False,
            repetitions=COUNTS,loops=1,points=100,bootstrap_resamples=100000,bootstrap_seed=20261020,
            frozen_utc=now())
        write(HERE / 'frozen-bindings.json',data)
        print(json.dumps({'frozen':True,'binding':binding(HERE / 'frozen-bindings.json')}),flush=True)
        return
    assert args.cpu in os.sched_getaffinity(0) and args.cpu != 207
    assert args.authorized_run and args.authorized_run.strip()
    os.sched_setaffinity(0,{args.cpu})
    frozen_path = HERE / 'frozen-bindings.json'
    frozen = json.loads(frozen_path.read_text())
    current = assets()
    assert all(current[k] == frozen[k] for k in current)
    attempts = HERE / 'timings'
    attempts.mkdir(exist_ok=True)
    for previous in attempts.iterdir():
        prior_result = previous / 'result.json'
        assert prior_result.is_file(), 'Unfinished previous attempt requires explicit investigation'
        prior_data = json.loads(prior_result.read_text())
        assert prior_data['status'] == 'failed' and not prior_data['samples']
        assert not list(previous.glob('*/pair-*')), 'Never replace a started numerical series'
    dest = attempts / f'attempt-{len(list(attempts.iterdir()))+1}'
    dest.mkdir()
    started = {'schema':'allocation-process-execution-v1','status':'preflight','started_utc':now(),
        'authorized_run':args.authorized_run,'cpu':args.cpu,'affinity':sorted(os.sched_getaffinity(0)),
        'frozen_bindings':binding(frozen_path),'controller':binding(Path(__file__)),
        'repetitions':COUNTS,'reference_exception':'Existing corrected-original collector CPU207 only',
        'topology':subprocess.check_output(['lscpu','-e=CPU,NODE,SOCKET,CORE,ONLINE'],text=True),
        'samples':[],'preflight':[],'acceptance_eligible':False,'production_changed':False}
    write(dest / 'started.json',started)
    result = dict(started)
    stop = threading.Event()
    monitor = None
    try:
        before = cpu_stat(args.cpu)
        for index in range(3):
            time.sleep(3)
            after = cpu_stat(args.cpu)
            total = after['total']-before['total']
            assert total > 0
            busy = 1-(after['idle']-before['idle'])/total
            result['preflight'].append({'index':index,'interval_seconds':3,'before':before,'after':after,'busy_fraction':busy})
            before = after
        write(dest / 'preflight.json',{'cpu':args.cpu,'intervals':result['preflight'],
            'passed':all(r['busy_fraction'] <= .05 for r in result['preflight'])})
        assert all(r['busy_fraction'] <= .05 for r in result['preflight']), 'CPU failed idle preflight; zero numerical samples launched'
        def host_monitor():
            with (dest / 'host-activity.jsonl').open('x') as stream:
                while True:
                    stream.write(json.dumps({'recorded_utc':now(),'selected_cpu':cpu_stat(args.cpu),'reference_cpu207':cpu_stat(207)})+'\n')
                    stream.flush()
                    if stop.wait(5):
                        break
        monitor = threading.Thread(target=host_monitor,daemon=True)
        monitor.start()
        references = {}
        result['status'] = 'running'
        config = json.loads((ROOT / json.loads(PREP.read_text())['config']['path']).read_text())
        prep = json.loads(PREP.read_text())
        for case,pairs in COUNTS.items():
            references[case] = None
            for pair in range(pairs):
                for arm in (ARMS if pair % 2 == 0 else list(reversed(ARMS))):
                    b = prep['builds'][case][arm]
                    output,directory = ROOT / b['output'],ROOT / b['directory']
                    job = output.parents[2]
                    sample = dest / case / f'pair-{pair:02d}-{arm}'
                    sample.mkdir(parents=True)
                    # Recreate identical cards and remove generated filters before each fresh process.
                    with workers.isolated_harness(job,sample / 'configuration'):
                        settings = bench.configure('candidate',b['case'],8,config=config)
                    assert settings == b['shared_numerical_settings']
                    validate_assets(frozen)
                    workers.validate_param_card(directory,(output / 'Cards/param_card.dat').read_bytes())
                    worker = None
                    worker_env = workers.environment(job)
                    worker_env.update(THREAD_ENV)
                    execution_environment = {n:worker_env[n] for n in [*THREAD_ENV,'LD_LIBRARY_PATH','SYMBOLICA_LOCK_DIR']}
                    begin = time.monotonic()
                    try:
                        worker = workers.Worker(['taskset','-c',str(args.cpu),'./check',str(ROOT / b['points']['path'])],
                            directory,sample / 'worker.log',env=worker_env,timeout=3600.)
                        assert worker.points == 100
                        assert worker.nso >= 1 and worker.nborn >= 1
                        wall = worker.run(1)
                        parsed = workers.parse_outputs(directory,8,100,1)
                        assert parsed['answer_dimensions'] == {'allocated':worker.ndim,'loop_orders':worker.nso,'born_orders':worker.nborn}
                        assert sum(parsed['status_counts'].values()) == sum(parsed['timed_status_counts'].values()) == 100
                        assert parsed['status_counts'] == parsed['timed_status_counts']
                        for name in ['dimensions.dat',*workers.OUTPUTS]:
                            shutil.copy2(directory / name,sample / name)
                    finally:
                        if worker is not None:
                            worker.close()
                    assert worker.process.returncode == 0
                    payload = {'case':case,'pair':pair,'arm':arm,'backend':8,'cpu':args.cpu,'loops':1,
                        'recorded_utc':now(),'worker_pid':worker.process.pid,'exit_code':worker.process.returncode,
                        'command_wall_seconds':wall,'total_lifecycle_seconds':time.monotonic()-begin,
                        'execution_environment':execution_environment,
                        'executable_sha256':b['executable']['sha256'],'points_sha256':b['points']['sha256'],
                        'parameter_card_sha256':b['parameter_card']['sha256'],
                        'shared_numerical_settings':settings,**parsed,
                        'raw_files':{p.name:binding(p) for p in sorted(sample.iterdir()) if p.is_file()}}
                    write(sample / 'result.json',payload)
                    ref = references[case]
                    if ref is None:
                        references[case] = sample
                    else:
                        ref_data = json.loads((ref / 'result.json').read_text())
                        assert payload['rows'] == ref_data['rows'], 'Any coefficient/status/accuracy change fails whole fixed series'
                        for name in ['dimensions.dat','values.dat','timed_status.dat']:
                            assert (sample / name).read_bytes() == (ref / name).read_bytes(), name
                        assert compare({'reference':ref_data['rows'],'current':parsed['rows']})['passed']
                    validate_assets(frozen)
                    result['samples'].append(binding(sample / 'result.json'))
                    print(case,pair,arm,parsed['seconds_per_point'],parsed['timed_status_counts'],flush=True)
        assert len(result['samples']) == 56
        result['status'] = 'complete-unanalysed'
    except BaseException as exc:
        result['status'] = 'failed'
        result['failure'] = repr(exc)
        raise
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        result['finished_utc'] = now()
        write(dest / 'result.json',result)
    print(json.dumps({'status':result['status'],'result':str((dest / 'result.json').relative_to(ROOT))}),flush=True)

if __name__ == '__main__':
    main()
