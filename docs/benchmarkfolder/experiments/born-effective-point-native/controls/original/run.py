#!/usr/bin/env python3
"""Verify or execute the five prepared control fixtures; never compile.

Default invocation verifies completed build/source identities only. --execute
is parent-owned, append-only, CPU14 serial, with one inclusive 60-second limit.
The declared per-job timeout/taskset recipes are retained as provenance; actual
execution passes only each executable to the reviewed bounded process owner.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BUILD=HERE.parent/'attempt1'
HELPER=ROOT/'docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/attempt2/control.py'
HELPER_SHA='acf00431793415817c29f26d997ae272da968f4b024482cfb72c60333b61ed50'
MANIFEST_SHA='07413855fd2545af7e2fd9d575117c02533f9a6e229dbae81095501f7ba9f0d4'
JOBS_SHA='28613b4a4f1f8cf5e1685715cb623edcc98d557edf3236ca130b6aa6e1c47229'
EXPECTED_SHA='a36336eb92aea09fc48df21586afe70e70edbf138858049994bea4faa8c2026a'
TOTAL_SECONDS=60.0
CPU=14
NAMES=('scalar_n3','scalar_n5','split_n3','split_n5','repair')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def load_owner():
    require(sha(HELPER)==HELPER_SHA,'Reviewed process owner changed')
    spec=importlib.util.spec_from_file_location('control_stub_bounded_owner',HELPER)
    owner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    require(owner.ROOT==ROOT,'Process owner repository differs')
    owner.CPU=CPU
    return owner


def verify(owner):
    """Bind current exact sources to five successful parent compiler events."""
    owner.check_interrupt()
    require(Path.cwd().resolve()==ROOT,'Run from the MadGraph repository root')
    require(sha(HERE/'manifest.json')==MANIFEST_SHA,'Frozen source manifest changed')
    require(sha(HERE/'compiler-jobs.json')==JOBS_SHA,'Declared compiler/run recipe changed')
    require(sha(HERE/'expected.json')==EXPECTED_SHA,'Expected case records changed')
    require(sha(HELPER)==HELPER_SHA,'Process owner changed')
    manifest=json.loads((HERE/'manifest.json').read_text())
    jobs=json.loads((HERE/'compiler-jobs.json').read_text())['jobs']
    expected=json.loads((HERE/'expected.json').read_text())['cases']
    require(tuple(j['name'] for j in jobs)==NAMES,'Unexpected fixture sequence')
    require(set(expected)==set(NAMES) and sum(map(len,expected.values()))==51,'Expected 51 cases')
    require(sha(HERE/'prepare.py')==manifest['generator_sha256'],'Source generator changed')
    require(manifest['expected_sha256']==EXPECTED_SHA and manifest['compiler_jobs_sha256']==JOBS_SHA,'Manifest recipe bindings differ')
    for name,want in manifest['inputs'].items():
        require(sha(ROOT/name)==want,'Changed frozen source dependency: '+name)
    for name,want in manifest['files'].items():
        require(sha(HERE/name)==want,'Changed generated source: '+name)
    build_path=BUILD/'build.json'
    build_bytes=build_path.read_bytes()
    build=json.loads(build_bytes)
    build_sha=hashlib.sha256(build_bytes).hexdigest()
    require(build['state']=='complete' and not build.get('changed_inputs'),'Parent build is not successfully complete')
    require(build['cpu']==CPU and build['max_compiler_jobs']==1,'Parent compiler ownership differs')
    require(build['native_executed'] is False,'Unexpected native work in compiler report')
    require(build['helper_sha256']==HELPER_SHA,'Parent used another process owner')
    require(sha(BUILD/'build.py')==build['controller_sha256'],'Parent build controller changed')
    require(sha(BUILD/'build-inputs.json')==build['build_inputs_sha256'],'Parent build inputs changed')
    require(sha(BUILD/'build.log')==build['build_log_sha256'],'Parent compiler log changed')
    require(build['environment']['FC']==str(owner.COMPILER),'Unpinned parent compiler')
    require(sha(owner.COMPILER)==owner.COMPILER_SHA256,'Pinned compiler binary changed')
    build_inputs=json.loads((BUILD/'build-inputs.json').read_text())
    source_prefix=rel(HERE/'sources')+'/'
    compiled_source_bindings={name:want for name,want in build_inputs.items() if name.startswith(source_prefix)}
    generated_source_bindings={rel(HERE/name):want for name,want in manifest['files'].items()}
    require(compiled_source_bindings==generated_source_bindings,'Parent did not compile this exact complete source package')
    commands={}
    for event in build['commands']:
        name=event.get('name','')
        if name.startswith('control-stub/'):
            require(name not in commands,'Repeated compiler job: '+name)
            commands[name]=event
    require(set(commands)=={'control-stub/'+name for name in NAMES},'All five unique compiler jobs required')
    artifact_prefix=rel(HERE/'work')+'/'
    artifacts={name:want for name,want in build['artifacts'].items() if name.startswith(artifact_prefix)}
    require(artifacts,'No bound control artifacts')
    for name,want in artifacts.items():
        path=ROOT/name
        require(path.is_file() and not path.is_symlink() and sha(path)==want,'Built control artifact changed: '+name)
    for job in jobs:
        name=job['name'];event=commands['control-stub/'+name]
        require(owner.successful(event),'Failed compiler job: '+name)
        require(event['cwd']=='.','Wrong compiler working directory: '+name)
        declared=job['compile_argv'];position=declared.index('gfortran')
        actual=event['command']
        require(Path(actual[0]).name=='taskset' and actual[1:3]==['-c',str(CPU)],'Compiler CPU wrapper differs')
        require(actual[3:]==[str(owner.COMPILER),*declared[position+1:]],'Actual compiler argv differs: '+name)
        require(job['run_argv'][:5]==['timeout','15s','taskset','-c',str(CPU)],'Declared runtime recipe differs')
        executable=Path(job['run_argv'][-1])
        require(executable==HERE/'work'/name/'control.exe','Unexpected executable location')
        require(rel(executable) in artifacts and os.access(executable,os.X_OK),'Missing executable binding')
    return jobs,expected,{
        'build_report_sha256':build_sha,'build_inputs_sha256':build['build_inputs_sha256'],
        'build_log_sha256':build['build_log_sha256'],'build_controller_sha256':build['controller_sha256'],
        'source_manifest_sha256':MANIFEST_SHA,'declared_jobs_sha256':JOBS_SHA,'expected_sha256':EXPECTED_SHA,
        'compiled_source_bindings':compiled_source_bindings,'artifact_bindings':artifacts,
        'scope':'Every control-stub source and built artifact; unrelated process-export artifacts are outside this run.',
        'successful_compiler_events':commands}


def validate_output(text,wanted):
    rows=[line.strip() for line in text.splitlines() if line.strip()]
    names=[line[5:] for line in rows if line.startswith('PASS ')]
    counts=Counter(names)
    missing=[name for name in wanted if counts[name]==0]
    duplicate={name:count for name,count in counts.items() if count!=1}
    unexpected=[name for name in counts if name not in wanted]
    failed=[line for line in rows if line.startswith('FAIL')]
    all_pass=[i for i,line in enumerate(rows) if line=='ALL PASS']
    pass_after_terminal=(bool(all_pass) and any(line.startswith('PASS ') for line in rows[all_pass[0]+1:]))
    passed=not(missing or duplicate or unexpected or failed or pass_after_terminal) and len(all_pass)==1
    return {'passed':passed,'expected_count':len(wanted),'observed_pass_names':names,
            'missing':missing,'duplicate_counts':duplicate,'unexpected':unexpected,
            'failure_lines':failed,'all_pass_count':len(all_pass),
            'pass_after_terminal':pass_after_terminal,'last_nonempty_line':rows[-1] if rows else None}


def projection(env):
    keys=('PATH','PWD','LANG','LC_ALL','TZ','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX')
    prefixes=('NIX_','LD_','OMP_','OPENBLAS_','RAYON_','MKL_','NUMEXPR_','GFORTRAN_')
    return {k:v for k,v in sorted(env.items()) if k in keys or k.startswith(prefixes)}


def execute(owner):
    target=HERE/'native-runs'
    report_path=HERE/'native.json'
    require(not target.exists() and not report_path.exists(),'Prior native attempt exists; no retry or overwrite')
    # This single deadline includes preflight, every launch, parsing and final
    # verification. No per-job reset, nested timeout process, or retry exists.
    start=time.monotonic();started_unix=time.time();deadline=start+TOTAL_SECONDS
    report={'schema':1,'state':'preflight','cpu':CPU,'serial':True,'inclusive_budget_seconds':TOTAL_SECONDS,
        'started_unix':started_unix,'deadline_monotonic':deadline,'deadline_unix':started_unix+TOTAL_SECONDS,
        'controller_sha256':sha(Path(__file__)),'helper_sha256':HELPER_SHA,
        'native_executed':False,'commands':[],'qualification':'controlled Fortran only; no reducer or full-amplitude claims'}
    owner.reserve(report_path,report)
    try:
        with owner.signal_cleanup():
            jobs,expected,binding=verify(owner)
            report['bindings_before']=binding
            env=owner.environment({},reducers=False)
            env['PWD']=str(ROOT)
            report['environment']=projection(env)
            taskset=shutil.which('taskset')
            require(taskset is not None,'taskset unavailable')
            report['taskset']={'path':taskset,'sha256':sha(taskset)}
            report['declared_commands']={job['name']:job['run_argv'] for job in jobs}
            report['execution_override']='Actual argv is executable only; reviewed owner applies CPU14 and one shared inclusive60s deadline. Original nested timeout/taskset recipes are provenance only.'
            target.mkdir(exist_ok=False)
            report['state']='running';owner.atomic_json(report_path,report)
            for job in jobs:
                owner.check_interrupt()
                _,_,current=verify(owner)
                require(current==binding,'Inputs/build bindings changed between jobs')
                require(time.monotonic()<deadline,'Inclusive runtime budget exhausted before launch')
                name=job['name'];executable=Path(job['run_argv'][-1]);log_path=target/(name+'.log')
                with log_path.open('x') as log:
                    event=owner.bounded([str(executable)],ROOT,log,deadline,env)
                event.update(name=name,effective_argv=[str(executable)],declared_argv=job['run_argv'],
                    executable_sha256=sha(executable),log=rel(log_path),log_sha256=sha(log_path),
                    log_bytes=log_path.stat().st_size)
                event['output_validation']=validate_output(log_path.read_text(errors='replace'),expected[name])
                report['commands'].append(event)
                report['native_executed']=any('pid' in item for item in report['commands'])
                owner.atomic_json(report_path,report)
                require(owner.successful(event),'Runtime command failed; stop: '+name)
                require(event['output_validation']['passed'],'Output case validation failed; stop: '+name)
            owner.check_interrupt()
            _,_,after=verify(owner)
            report['bindings_after']=after
            require(after==binding,'Source/build/artifact bindings changed during execution')
            require(time.monotonic()<=deadline,'Inclusive runtime budget exhausted during final verification')
            require(sum(e['output_validation']['expected_count'] for e in report['commands'])==51,'Incomplete case execution')
            report['state']='complete';report['passed_cases']=51
    except BaseException as error:
        report['state']='failed'
        report['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        report['finished_unix']=time.time()
        owner.atomic_json(report_path,report)
    print(json.dumps({'state':report['state'],'native_executed':report['native_executed'],
        'commands':len(report['commands']),'passed_cases':report.get('passed_cases',0),
        'elapsed_seconds':report['elapsed_seconds'],'error':report.get('error')}))
    return 0 if report['state']=='complete' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Parent-owned execution after successful complete build; one attempt only')
    args=parser.parse_args()
    owner=load_owner()
    if args.execute:
        return execute(owner)
    jobs,expected,binding=verify(owner)
    print(json.dumps({'state':'verified-no-native','jobs':len(jobs),
        'cases':sum(map(len,expected.values())),'build_sha256':binding['build_report_sha256']}))
    return 0


if __name__=='__main__':sys.exit(main())
