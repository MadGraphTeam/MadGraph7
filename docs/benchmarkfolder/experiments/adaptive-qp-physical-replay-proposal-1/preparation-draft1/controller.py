"""One copied-runtime physical diagnostic; no compiler, relink, or installation."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
WORK=ROOT/'.tirust-build/adaptive-qp-physical-replay-proposal-1/attempt1'
MASS=ROOT/'.tirust-build/qp-mass-general-implementation/combined-qualification'
ORIGINAL=MASS/'amplitude/runs/guarded-qp-9'
EXECUTABLE=MASS/'amplitude/build/guarded-qp'
BRIDGE=ROOT/'.tirust-build/adaptive-qp-rescue-release-build/attempt1/runtime/libmadloop_tirust_adaptive_private.so'
BRIDGE_SHA='c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7'
SUPERVISOR=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py'
SUPERVISOR_SHA='ed25a9fc96e2faf2b545a9bf70a91b34315c01e534a6ebe68d7520ec2171aa42'
ELF=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-design/elf_metadata.py'
PARSER=ROOT/'docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/analyze.py'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    temporary.replace(path)


def inputs():
    assert Path.cwd().resolve()==ROOT
    prepared=json.loads((MASS/'prepared.json').read_text())
    checks=[c for c in prepared['runtime_checks'] if c['executable']==str(EXECUTABLE.relative_to(ROOT))]
    assert len(checks)==2 and checks[0]['expected']==checks[1]['expected']
    original=next(j for s in prepared['stages'] for j in s['jobs'] if j['name']=='guarded-qp-9')
    assert original['backend']==9 and original['mode']==4 and original['point_count']==2
    assert original['expected_outputs']==['mass-observer.hex','observer-run.log','param.log','stdout.log']
    return original,checks[0]


def source_prepare():
    """Freeze hashes/spec only. Does not copy or execute an ELF or acquire a lock."""
    assert not (HERE/'prepared.json').exists() and not (HERE/'source-index.json').exists()
    original,check=inputs();elf=load('adaptive_physical_source_elf',ELF)
    old=elf.inspect(MASS/'current-runtime/libmadloop_tirust.so');new=elf.inspect(BRIDGE)
    binary=elf.inspect(EXECUTABLE)
    assert new['sha256']==BRIDGE_SHA
    for key in ('needed','version_requirements'):
        assert old['metadata'][key]==new['metadata'][key],key
    assert set(old['metadata']['exported_ml_tirust_symbols']) <= set(new['metadata']['exported_ml_tirust_symbols'])
    assert old['metadata']['soname']=='libmadloop_tirust.so' and new['metadata']['soname'] is None
    assert 'libmadloop_tirust.so' in binary['metadata']['needed'] and binary['metadata']['rpath'] is None
    runtime={}
    for name,info in check['expected'].items():
        key=Path(name).name
        assert key not in runtime and sha(info['path'])==info['sha256']
        runtime[key]=info
    assert len(runtime)==16 and 'libquadmath.so.0' in runtime
    names=[MASS/'prepared.json',MASS/'build.json',MASS/'amplitude/prepared.json',
           MASS/'amplitude/source-checks.json',MASS/'amplitude/analyze.py',MASS/'loader-correction/continuation/run.json',
           EXECUTABLE,BRIDGE,SUPERVISOR,ELF,PARSER]
    names.extend(Path(p) for p in original['input_sha256'])
    names.extend(ORIGINAL/p for p in original['expected_outputs'])
    names.extend(p for p in (MASS/'amplitude/src').iterdir() if p.is_file())
    names.extend(Path(info['path']) for info in runtime.values())
    # Keep independent/lower-precision evidence and the separately qualified
    # transport policy bound, without extending their scientific scope.
    for folder,index_name in [('ew-qp-mass-prototype/combined-qualification-1','evidence-index.json'),
                              ('adaptive-qp-release-fortran-qualification-v2/runtime1','evidence-index.json')]:
        base=ROOT/'docs/benchmarkfolder/experiments'/folder;index=base/index_name
        names.append(index)
        for name,value in json.loads(index.read_text())['files'].items():
            path=base/name;expected=value['sha256'] if isinstance(value,dict) else value
            assert sha(path)==expected
            names.append(path)
    dependencies={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in sorted(set(names))}
    assert sha(SUPERVISOR)==SUPERVISOR_SHA
    spec={'schema':'adaptive-physical-source-only-proposal-v1','original_job':original,
          'runtime_inputs':runtime,'elf_metadata':{'executable':binary,'original_bridge':old,'candidate_bridge':new},
          'original_executable_sha256':sha(EXECUTABLE),'candidate_bridge_sha256':BRIDGE_SHA,
          'cpu':12,'cpu_reserved':False,'rlimit_as_bytes':4294967296,'one_attempt':True,
          'execution_seconds_max':180,'loader_seconds_max':10,'per_call_precision_telemetry':False,
          'dependencies':dependencies,'source_only_preparation':True,'native_launched':False,
          'root_admission_required_before_run':True}
    save(HERE/'prepared.json',spec)
    sources={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='source-index.json'}
    save(HERE/'source-index.json',{'schema':'adaptive-physical-source-freeze-v1','sources':sources,
                                 'dependencies':dependencies,'native_launched':False})
    print(json.dumps({'source_only_prepared':True,'dependencies':len(dependencies),'native_launched':False}))


def verify():
    assert Path.cwd().resolve()==ROOT
    index=json.loads((HERE/'source-index.json').read_text())
    for name,digest in index['sources'].items():assert sha(HERE/name)==digest,name
    for name,digest in index['dependencies'].items():assert sha(ROOT/name if not name.startswith('/') else name)==digest,name
    spec=json.loads((HERE/'prepared.json').read_text())
    assert spec['dependencies']==index['dependencies']
    assert spec['cpu']==12 and spec['execution_seconds_max']==180 and spec['rlimit_as_bytes']==4294967296
    assert spec['candidate_bridge_sha256']==BRIDGE_SHA and spec['one_attempt']
    assert sha(SUPERVISOR)==SUPERVISOR_SHA
    return spec


def runtime_listing(text,expected,loader):
    import re
    assert 'not found' not in text
    paths={str(Path(p).resolve()) for p in re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(0x[0-9a-f]+\)',text,re.M)}
    assert paths==set(expected),{'unexpected':sorted(paths-set(expected)),'missing':sorted(set(expected)-paths)}
    assert str(Path(loader).resolve()) in paths
    assert {p:sha(p) for p in paths}==expected
    return {'actual_listed_runtime_sha256':expected,'full_closure_exact':True}


def remaining(deadline,reserve=0):
    value=deadline-time.monotonic()-reserve
    if value<=0:raise TimeoutError('Shared180second diagnostic deadline exhausted; no retry')
    return value


def run():
    spec=verify()
    assert 12 in os.sched_getaffinity(0),'CPU12 unavailable'
    # A busy lock neither waits nor consumes the unique attempted directory.
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(75)
        assert not WORK.exists(),'One attempt only; retain an existing failure without retry'
        WORK.mkdir(parents=True)
        start=time.monotonic();deadline=start+180
        report={'complete':False,'native_launched':False,'source_index_sha256':sha(HERE/'source-index.json'),
                'prepared_sha256':sha(HERE/'prepared.json'),'cpu':12,'cpu_reservation':False,
                'execution_seconds_max':180,'automatic_retry':False,'commands':[],'process_ownership_reused':SUPERVISOR_SHA,
                'scope':'Same two physical points, copied original executable and dependencies; bridge bytes only changed',
                'started_unix':time.time(),'loadavg_before':Path('/proc/loadavg').read_text().strip()}
        old_handlers={}
        def interrupted(signum,frame):raise InterruptedError('Physical replay interrupted by signal%d'%signum)
        for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP,signal.SIGALRM):
            old_handlers[signum]=signal.signal(signum,interrupted)
        signal.setitimer(signal.ITIMER_REAL,remaining(deadline))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(4294967296,4294967296))
        os.sched_setaffinity(0,{12})
        try:
            source=SUPERVISOR.read_text()
            assert source.count("affinity!=['11']")==source.count('differs from CPU11.')==1
            source=source.replace("affinity!=['11']","affinity!=['12']").replace('differs from CPU11.','differs from CPU12.')
            signals='SIGNALS=(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)'
            assert source.count(signals)==1
            source=source.replace(signals,'SIGNALS=(signal.SIGINT,signal.SIGTERM,signal.SIGHUP,signal.SIGALRM)')
            helper={'__name__':'adaptive_physical_owned_supervisor','__file__':str(SUPERVISOR)}
            exec(compile(source,str(SUPERVISOR),'exec'),helper)
            report['adapted_supervisor_sha256']=hashlib.sha256(source.encode()).hexdigest()
            runtime=WORK/'runtime';runtime.mkdir();case=WORK/'case';case.mkdir()
            executable=WORK/'guarded-qp';shutil.copyfile(EXECUTABLE,executable);executable.chmod(0o700)
            assert sha(executable)==spec['original_executable_sha256']
            loader=None;expected={}
            for name,info in spec['runtime_inputs'].items():
                if name=='ld-linux-x86-64.so.2':
                    loader=info['path'];expected[str(Path(loader).resolve())]=info['sha256'];continue
                donor=BRIDGE if name=='libmadloop_tirust.so' else Path(info['path'])
                destination=runtime/name;shutil.copyfile(donor,destination)
                expected[str(destination.resolve())]=sha(destination)
                assert expected[str(destination.resolve())]==(BRIDGE_SHA if name=='libmadloop_tirust.so' else info['sha256'])
            assert loader and len(expected)==16
            for name,digest in spec['original_job']['input_sha256'].items():
                rel=Path(name).relative_to(ORIGINAL);destination=case/rel;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(name,destination);assert sha(destination)==digest
            before={str(p.relative_to(case)):sha(p) for p in case.rglob('*') if p.is_file()}
            env={key:os.environ[key] for key in ('PATH','HOME','TMPDIR') if key in os.environ}
            env.update(LD_LIBRARY_PATH=str(runtime),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1',
                       MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',LC_ALL='C',PYTHONDONTWRITEBYTECODE='1')
            report.update(environment=env,expected_runtime_sha256=expected,executable_sha256=sha(executable),
                          input_sha256=before,bridge_mapped_filename_is_abi_alias=True,
                          bridge_byte_identity=BRIDGE_SHA,rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)))
            save(WORK/'execution.json',report)
            item={'kind':'loader-list','argv':[loader,'--list',str(executable)],'progress':{}};report['commands'].append(item)
            with (WORK/'loader.log').open('xb') as log:
                item['exit_code']=helper['run_plain_capped'](item['argv'],log,cwd=case,env=env,progress=item['progress'],seconds=min(10,remaining(deadline,5)))
            assert item['exit_code']==0
            report['loader_closure']=runtime_listing((WORK/'loader.log').read_text(),expected,loader)
            save(WORK/'execution.json',report)
            item={'kind':'one-physical-worker','argv':[str(executable),str(case/'selected.dat')],'progress':{}}
            report['commands'].append(item);report['native_launched']=True;save(WORK/'execution.json',report)
            with (case/'stdout.log').open('xb') as log:
                item['exit_code'],report['actual_native_mapping']=helper['run_captured'](
                    item['argv'],log,cwd=case,env=env,expected_runtime=expected,executable=str(executable.resolve()),
                    selected_library=str((runtime/'libmadloop_tirust.so').resolve()),maps_path=WORK/'native-maps.txt',
                    progress=item['progress'],seconds=remaining(deadline,5))
            assert item['exit_code']==0 and item['progress']['process_reaped']
            after={str(p.relative_to(case)):sha(p) for p in case.rglob('*') if p.is_file()}
            assert all(after.get(p)==digest for p,digest in before.items())
            assert set(after)-set(before)==set(spec['original_job']['expected_outputs'])
            report['output_sha256']={p:after[p] for p in sorted(set(after)-set(before))}
            # Reuse the exact fixed-CTMode4 warmup/point-completion validator.
            validator=load('adaptive_physical_original_completion',MASS/'amplitude/analyze.py')
            job=dict(spec['original_job']);job['cwd']=str(case);job['input_sha256']={str(case/p):v for p,v in before.items()}
            observed=validator.check_run(job)
            parser=load('adaptive_physical_frozen_journal',PARSER)
            old=parser.parse(ORIGINAL/'mass-observer.hex','candidate',9)
            new=parser.parse(case/'mass-observer.hex','candidate',9)
            assert observed==[int(new['points'][n]['data']['PUBLIC_STATUS'][0][1]) for n in (1,2)]
            analysis=load('adaptive_physical_same_input_analysis',HERE/'analyze.py')
            result=analysis.compare(old,new,(ORIGINAL/'stdout.log').read_text(),(case/'stdout.log').read_text())
            save(WORK/'analysis.json',result);report['analysis_sha256']=sha(WORK/'analysis.json')
            report['invariants_passed']=result['invariants_passed']
            verify();assert {p:sha(p) for p in expected}==expected
            assert sha(executable)==spec['original_executable_sha256'];remaining(deadline)
            report['complete']=bool(result['invariants_passed'])
        except BaseException as error:
            report['failure']={'type':type(error).__name__,'message':str(error)}
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for signum,handler in old_handlers.items():signal.signal(signum,handler)
            # Closure after a failed/deadline attempt is administrative evidence;
            # it never extends or resumes any scientific execution allowance.
            try:
                verify();report['protected_inputs_after']='unchanged'
                if 'expected' in locals():
                    report['runtime_sha256_after']={p:sha(p) for p in expected}
                    if report['runtime_sha256_after']!=expected:raise ValueError('Copied runtime changed')
                if 'case' in locals():
                    report['all_case_files_after']={str(p.relative_to(case)):sha(p) for p in case.rglob('*') if p.is_file()}
            except BaseException as error:
                report['complete']=False;report['protected_inputs_after']='FAILED: '+str(error)
            report['elapsed_seconds']=time.monotonic()-start
            report['finished_unix']=time.time();report['loadavg_after']=Path('/proc/loadavg').read_text().strip()
            report['physical_accuracy_qualified']=False;report['performance_qualified']=False
            report['post_deadline_report_is_administrative_only']=time.monotonic()>deadline
            save(WORK/'execution.json',report)
        print(json.dumps({'complete':report['complete'],'native_launched':report['native_launched'],
                          'failure':report.get('failure'),'work':str(WORK)},sort_keys=True))
        if not report['complete']:raise SystemExit(1)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('action',choices=('prepare','verify','run'))
    action=cli.parse_args().action
    if action=='prepare':source_prepare()
    elif action=='verify':print(json.dumps({'verified':bool(verify()),'native_launched':False}))
    else:run()
