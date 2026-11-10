"""Private one-main build and one bounded checked/trap diagnostic, no retries.

prepare/verify read and freeze source only. build and run require separate root
admission. None of these actions installs a library or edits production input.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import struct
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
WORK=ROOT/'.tirust-build/adaptive-qp-checked-trap-proposal-1'
BUILD=WORK/'build1';RUN=WORK/'run1'
MASS=ROOT/'.tirust-build/qp-mass-general-implementation/combined-qualification'
ORIGINAL=MASS/'amplitude/runs/guarded-qp-9'
OLD_EXECUTABLE=MASS/'amplitude/build/guarded-qp'
EXE=BUILD/'checked'
BRIDGES={'original':MASS/'current-runtime/libmadloop_tirust.so',
         'adaptive':ROOT/'.tirust-build/adaptive-qp-rescue-release-build/attempt1/runtime/libmadloop_tirust_adaptive_private.so'}
BRIDGE_HASHES={'original':'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9',
               'adaptive':'c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7'}
SUPERVISOR=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py'
SUPERVISOR_HASH='ed25a9fc96e2faf2b545a9bf70a91b34315c01e534a6ebe68d7520ec2171aa42'
ELF=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-design/elf_metadata.py'


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def save(path,value):
    # An interrupted write remains as additive evidence; later terminal
    # publication must not overwrite or collide with that partial temporary.
    temporary=path.with_name(path.name+'.tmp-'+str(time.monotonic_ns()))
    with temporary.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    temporary.replace(path)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def relative(path):
    path=Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def replace_card(text,name,old,new):
    lines=text.splitlines(True);indices=[i for i,line in enumerate(lines) if line.strip()=='#'+name]
    assert len(indices)==1
    index=indices[0]+1;assert lines[index].strip()==old,(name,lines[index])
    lines[index]=new+'\n';return ''.join(lines)


def source_prepare():
    """No compiler, loader, lock or native child is created by preparation."""
    assert Path.cwd().resolve()==ROOT and not (HERE/'prepared.json').exists()
    parent=json.loads((MASS/'prepared.json').read_text())
    old=list(next(job for job in parent['build_jobs'] if job['name']=='link-guarded-qp')['argv'])
    old_main=next(arg for arg in old if arg.endswith('/check_sa.o'))
    assert old.count(old_main)==1 and old.count(str(OLD_EXECUTABLE))==1 and old[0]==parent['compiler']
    numerical=[arg for arg in old if arg.endswith('.o') and arg!=old_main]
    assert len(numerical)==25
    link=[str(EXE) if arg==str(OLD_EXECUTABLE) else str(BUILD/'driver.o') if arg==old_main else arg for arg in old]
    link.append('-Wl,-Map,'+str(BUILD/'checked.map'))
    assert [arg for arg in link if arg.endswith('.o') and arg!=str(BUILD/'driver.o')]==numerical
    module=Path(old_main).parent
    coupl=module/'coupl.inc';coupl_text=coupl.read_text()
    assert 'DOUBLE PRECISION MU_R, ALL_MU_R' in coupl_text and 'COMMON/RSCALE/ MU_R, ALL_MU_R' in coupl_text
    compile_argv=[old[0],'-w','-fPIC','-O0','-fno-fast-math','-frounding-math','-fsignaling-nans',
                  '-ffree-line-length-none','-J'+str(BUILD),'-I'+str(module),'-c',str(HERE/'driver.f90'),'-o',str(BUILD/'driver.o')]
    checks=[check for check in parent['runtime_checks'] if check['executable']==relative(OLD_EXECUTABLE)]
    assert len(checks)==2 and checks[0]['expected']==checks[1]['expected']
    runtime={Path(name).name:info for name,info in checks[0]['expected'].items()}
    assert len(runtime)==15 and 'libquadmath.so.0' in runtime
    old_job=next(job for stage in parent['stages'] for job in stage['jobs'] if job['name']=='guarded-qp-9')
    fixtures=HERE/'fixtures';fixtures.mkdir(exist_ok=False)
    original_inputs={}
    for name,digest in old_job['input_sha256'].items():
        path=Path(name);assert sha(path)==digest
        target=fixtures/path.relative_to(ORIGINAL);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);original_inputs[relative(path)]=digest
    card=fixtures/'MadLoop5_resources/MadLoopParams.dat'
    original_card=card.read_text();candidate_card=replace_card(original_card,'CTModeRun','4','-1')
    candidate_card=replace_card(candidate_card,'NRotations_QP','0','1')
    assert replace_card(replace_card(candidate_card,'CTModeRun','-1','4'),'NRotations_QP','1','0')==original_card
    card.write_text(candidate_card)
    elf=load('checked_source_elf',ELF)
    metadata={key:elf.inspect(path) for key,path in BRIDGES.items()}
    assert all(metadata[key]['sha256']==BRIDGE_HASHES[key] for key in BRIDGES)
    for field in ('needed','version_requirements'):
        assert metadata['original']['metadata'][field]==metadata['adaptive']['metadata'][field]
    assert set(metadata['original']['metadata']['exported_ml_tirust_symbols'])<=set(metadata['adaptive']['metadata']['exported_ml_tirust_symbols'])
    dependencies=[MASS/'prepared.json',MASS/'build.json',MASS/'amplitude/prepared.json',
                  OLD_EXECUTABLE,SUPERVISOR,ELF,coupl,module/'ew_mass_observer.mod',
                  HERE.parent/'ew-qp-mass-prototype/observer-native-design/attempt2/check_observed.f',
                  HERE.parent/'adaptive-qp-physical-replay-qualification-1/evidence-index.json',
                  HERE.parent/'adaptive-qp-physical-replay-independent-8-audit-1/evidence-index.json']
    dependencies.extend(BRIDGES.values());dependencies.extend(Path(info['path']) for info in runtime.values())
    dependencies.extend(Path(arg) for arg in old if arg.startswith('/') and Path(arg).is_file())
    dependencies.extend(Path(name) for name in old_job['input_sha256'])
    dependencies.extend(Path(name) for name in parent['system_files'])
    bindings={relative(path):sha(path) for path in sorted(set(dependencies))}
    assert sha(SUPERVISOR)==SUPERVISOR_HASH
    spec={'schema':'private-adaptive-checked-trap-source-only-v1','scope':'Old genuine-MP-mass private candidate, distinct from invariant-symmetry-1',
          'compile_argv':compile_argv,'link_argv':link,'original_link_argv':old,'original_main':old_main,
          'unchanged_numerical_objects':{arg:sha(arg) for arg in numerical},
          'compiler_environment':parent['compiler_environment'],'original_inputs':original_inputs,
          'runtime_inputs':runtime,'loader':checks[0]['loader'],'bridge_metadata':metadata,
          'dependencies':bindings,'build_seconds_max':180,'execution_seconds_max':600,
          'physical_worker_seconds_max':120,'fault_worker_seconds_max':10,'loader_seconds_max':10,
          'cpu':12,'cpu_reservation':False,'rlimit_as_bytes':4294967296,'automatic_retry':False,
          'physical_workers':[['original','masked'],['adaptive','masked'],['original','trapped'],['adaptive','trapped']],
          'fault_controls':['invalid','divide_by_zero','overflow'],'source_only_prepared':True,
          'compiler_loader_or_native_launched':False,'separate_root_admission_before_build_and_run':True}
    save(HERE/'prepared.json',spec)
    sources={str(path.relative_to(HERE)):sha(path) for path in sorted(HERE.rglob('*'))
             if path.is_file() and path.name!='source-index.json'}
    save(HERE/'source-index.json',{'schema':'private-checked-trap-source-freeze-v1','sources':sources,
                                 'dependencies':bindings,'native_launched':False})
    print(json.dumps({'source_only_prepared':True,'sources':len(sources),'dependencies':len(bindings),'native_launched':False}))


def verify():
    assert Path.cwd().resolve()==ROOT
    index=json.loads((HERE/'source-index.json').read_text())
    for name,digest in index['sources'].items():assert sha(HERE/name)==digest,name
    for name,digest in index['dependencies'].items():assert sha(ROOT/name if not name.startswith('/') else name)==digest,name
    spec=json.loads((HERE/'prepared.json').read_text());assert spec['dependencies']==index['dependencies']
    assert spec['cpu']==12 and not spec['cpu_reservation'] and spec['rlimit_as_bytes']==4294967296
    assert spec['execution_seconds_max']==600 and spec['physical_worker_seconds_max']==120
    assert spec['fault_worker_seconds_max']==10 and not spec['automatic_retry']
    assert len(spec['unchanged_numerical_objects'])==25
    return spec


def supervisor():
    assert sha(SUPERVISOR)==SUPERVISOR_HASH
    source=SUPERVISOR.read_text();assert source.count("affinity!=['11']")==source.count('differs from CPU11.')==1
    source=source.replace("affinity!=['11']","affinity!=['12']").replace('differs from CPU11.','differs from CPU12.')
    signals='SIGNALS=(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)';assert source.count(signals)==1
    source=source.replace(signals,'SIGNALS=(signal.SIGINT,signal.SIGTERM,signal.SIGHUP,signal.SIGALRM)')
    namespace={'__name__':'private_checked_owned_supervisor','__file__':str(SUPERVISOR)}
    exec(compile(source,str(SUPERVISOR),'exec'),namespace)
    return namespace,hashlib.sha256(source.encode()).hexdigest()


def remaining(deadline,reserve=0):
    seconds=deadline-time.monotonic()-reserve
    if seconds<=0:raise TimeoutError('Shared admitted deadline exhausted; no retry')
    return seconds


def handlers(deadline):
    def interrupt(signum,frame):raise InterruptedError('Checked/trap stage interrupted by signal%d'%signum)
    previous={number:signal.signal(number,interrupt) for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP,signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL,remaining(deadline));return previous


def restore(previous):
    signal.setitimer(signal.ITIMER_REAL,0)
    for number,handler in previous.items():signal.signal(number,handler)


def timed_call(action,seconds):
    start=time.monotonic();deadline=start+seconds;previous=handlers(deadline)
    try:return action(start,deadline)
    finally:restore(previous)


def publish_final(path,record,deadline,start,complete_key,closure,*,claim_key=None):
    """Complete hashes/publication inside cap; late closure only records failure."""
    try:
        remaining(deadline);closure();remaining(deadline)
        record['elapsed_seconds']=time.monotonic()-start
        record['post_deadline_closure_is_administrative_only']=False
        save(path,record);remaining(deadline)
    except BaseException as error:
        # No new scientific or native work follows an expired stage. Preserve
        # any interrupted temporary and publish an explicitly failed receipt.
        signal.setitimer(signal.ITIMER_REAL,0)
        record[complete_key]=False
        if claim_key:record[claim_key]=False
        record['closure_failure']={'type':type(error).__name__,'message':str(error)}
        record['elapsed_seconds']=time.monotonic()-start
        record['post_deadline_closure_is_administrative_only']=time.monotonic()>deadline
        save(path,record)


def base_environment(spec,*,compiler=False):
    if compiler:
        env=dict(spec['compiler_environment'])
        for name in ('HOME','TMPDIR'):
            if name in os.environ:env[name]=os.environ[name]
    else:env={name:os.environ[name] for name in ('PATH','HOME','TMPDIR') if name in os.environ}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
               LC_ALL='C',PYTHONDONTWRITEBYTECODE='1',GFORTRAN_ERROR_BACKTRACE='1')
    return env


def limits():
    assert 12 in os.sched_getaffinity(0),'CPU12 unavailable'
    os.sched_setaffinity(0,{12});resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_AS,(4294967296,4294967296))


def build():return timed_call(build_stage,180)


def build_stage(start,deadline):
    spec=verify();helper,adapted=supervisor();assert not BUILD.exists(),'One build attempt only'
    BUILD.mkdir(parents=True);limits()
    record={'complete':False,'source_index_sha256':sha(HERE/'source-index.json'),'cpu':12,'cpu_reservation':False,
            'build_seconds_max':180,'commands':[],'unchanged_numerical_objects':spec['unchanged_numerical_objects'],
            'adapted_supervisor_sha256':adapted,'native_evaluation_or_loader_launched':False,
            'environment':base_environment(spec,compiler=True)}
    try:
        for name,argv in [('compile-main',spec['compile_argv']),('relink-only',spec['link_argv'])]:
            verify();row={'name':name,'argv':argv,'progress':{}};record['commands'].append(row);save(BUILD/'build.json',record)
            with (BUILD/(name+'.log')).open('xb') as log:
                row['exit_code']=helper['run_plain_capped'](argv,log,cwd=ROOT,env=record['environment'],progress=row['progress'],seconds=remaining(deadline,5))
            assert row['exit_code']==0 and row['progress']['process_reaped'],name
        actual={str(Path(name).resolve()) for name in re.findall(r'^LOAD\s+(.+)$',(BUILD/'checked.map').read_text(),re.M)}
        assert str(BUILD/'driver.o') in actual and set(spec['unchanged_numerical_objects'])<=actual
        assert spec['original_main'] not in actual
        record['actual_link_inputs']={name:sha(name) for name in sorted(actual)}
        verify();remaining(deadline)
        record['artifacts']={p.name:sha(p) for p in sorted(BUILD.iterdir()) if p.is_file() and p.name!='build.json'}
        assert set(record['artifacts'])=={'driver.o','checked','checked.map','compile-main.log','relink-only.log'}
        record['complete']=True
    except BaseException as error:record['failure']={'type':type(error).__name__,'message':str(error)}
    finally:
        def closure():
            verify();record['protected_inputs_after']='unchanged'
        publish_final(BUILD/'build.json',record,deadline,start,'complete',closure)
    print(json.dumps({'build_complete':record['complete'],'native_evaluation_launched':False,'failure':record.get('failure')}))
    if not record['complete']:raise SystemExit(1)


def built():
    spec=verify();record=json.loads((BUILD/'build.json').read_text())
    assert record['complete'] and record['source_index_sha256']==sha(HERE/'source-index.json')
    assert len(record['commands'])==2 and all(row['exit_code']==0 and row['progress']['process_reaped'] for row in record['commands'])
    assert [row['argv'] for row in record['commands']]==[spec['compile_argv'],spec['link_argv']]
    for name,digest in record['artifacts'].items():assert sha(BUILD/name)==digest,name
    for name,digest in record['actual_link_inputs'].items():assert sha(name)==digest,name
    return spec,record


def interpreter(executable):
    data=executable.read_bytes();assert data[:6]==b'\x7fELF\x02\x01'
    start=struct.unpack_from('<Q',data,32)[0];size,count=struct.unpack_from('<HH',data,54);result=[]
    for index in range(count):
        entry=struct.unpack_from('<IIQQQQQQ',data,start+index*size)
        if entry[0]==3:result.append(data[entry[2]:entry[2]+entry[5]].rstrip(b'\0').decode())
    assert len(result)==1;return result[0]


def listed(text,expected):
    assert 'not found' not in text
    paths={str(Path(name).resolve()) for name in re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(0x[0-9a-f]+\)',text,re.M)}
    assert paths==set(expected),{'extra':sorted(paths-set(expected)),'missing':sorted(set(expected)-paths)}
    assert {name:sha(name) for name in paths}==expected
    return {'exact':True,'actual_runtime_sha256':expected}


def compare_policy_outputs(left,right):
    """Same executable/library/input under two exception policies."""
    failures=[];group_comparable=True
    if not left['complete'] or not right['complete']:failures.append(['partial_capture'])
    if set(left['points'])!={1,2} or set(right['points'])!={1,2}:
        return {'outputs_exact':False,'failures':[['point_set']],'group_outputs_comparable':False}
    for number in (1,2):
        a,b=left['points'][number],right['points'][number]
        for label in ('AMPLITUDE','ACCURACY','PUBLIC_STATUS'):
            first=a['data'].get(label);second=b['data'].get(label)
            if label=='PUBLIC_STATUS':
                first=[row[:2] for row in first] if first else first
                second=[row[:2] for row in second] if second else second
            if first!=second:failures.append([number,label])
        if [g['key'] for g in a['groups']]!=[g['key'] for g in b['groups']]:
            group_comparable=False;failures.append([number,'group_sequence']);continue
        for x,y in zip(a['groups'],b['groups'],strict=True):
            for label in ('RAW_UV1','RAW_IR1','RAW_IR2','RAW_FINITE','GROUP_RESULT','GROUP_STABLE'):
                first=x['data'].get(label);second=y['data'].get(label)
                if label=='GROUP_STABLE':
                    first=[row[:2] for row in first] if first else first
                    second=[row[:2] for row in second] if second else second
                if first!=second:failures.append([number,x['key'][0],label])
    return {'outputs_exact':not failures,'failures':failures,'group_outputs_comparable':group_comparable,
            'hardware_flag_observations_compared_separately':True,'physical_truth_qualified':False}


def output_closure(before,after,kind,exit_code):
    snapshots={'first-read.txt','last-read.txt','first-nonempty.txt','last-nonempty.txt',
               'first-incomplete.txt','last-incomplete.txt','complete.txt','mismatch.txt'}
    allowed={'stdout.log','checked-control.log','mapping-ready','native-maps.txt'}
    allowed|={'native-maps.snapshots/'+name for name in snapshots}
    if kind=='physical':allowed|={'mass-observer.hex','param.log'}
    added=set(after)-set(before);unexpected=sorted(added-allowed)
    required={'stdout.log','checked-control.log','mapping-ready','native-maps.txt','native-maps.snapshots/complete.txt'}
    if kind=='physical' and exit_code==0:required|={'mass-observer.hex','param.log'}
    return {'unexpected':unexpected,'missing_required':sorted(required-added),
            'matched':not unexpected and required<=added}


def transport_complete(workers,*,top_failure=False):
    if top_failure or len(workers)!=7:return False
    for row in workers.values():
        mapping=row.get('actual_mapping',{});expected=row.get('expected_runtime_sha256',{})
        if not (row['progress'].get('process_reaped',False) and mapping and 'process_failure' not in row and
                row.get('exit_code') in (0,-signal.SIGFPE) and not row['changed_inputs'] and
                row.get('native_output_closure',{}).get('matched',False)):
            return False
        if (len(expected)!=15 or mapping.get('expected_runtime_sha256')!=expected or
                mapping.get('mapped_files_sha256_after_exit')!=expected or
                set(mapping.get('mapped_elf_paths',[]))!=set(expected)|{mapping.get('executable')}):
            return False
    return True


def run():return timed_call(run_stage,600)


def run_stage(start,deadline):
    spec,build_record=built();helper,adapted=supervisor()
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(75)
        assert not RUN.exists(),'One runtime attempt only; retain any failure'
        RUN.mkdir();limits()
        record={'execution_complete':False,'source_index_sha256':sha(HERE/'source-index.json'),'build_sha256':sha(BUILD/'build.json'),
                'scope':spec['scope'],'cpu':12,'cpu_reservation':False,'rlimit_as':[4294967296]*2,
                'execution_seconds_max':600,'automatic_retry':False,'workers':{},'loaders':{},
                'candidate_checked_and_trap_coverage_qualified':False,
                'adapted_supervisor_sha256':adapted,'physical_accuracy_or_performance_qualified':False,
                'loadavg_before':Path('/proc/loadavg').read_text().strip()}
        expected_by_arm={};captures={};runtime_results={};pending_candidate=False
        controls_module=load('private_checked_control_log',HERE/'control_log.py')
        journal_module=load('private_checked_pass_parser',HERE/'parser.py')
        def checkpoint():save(RUN/'execution.json',record)
        try:
            assert interpreter(EXE)==spec['loader']
            for arm in BRIDGES:
                directory=RUN/('runtime-'+arm);directory.mkdir();expected={}
                for name,info in spec['runtime_inputs'].items():
                    if name=='ld-linux-x86-64.so.2':expected[str(Path(info['path']).resolve())]=info['sha256'];continue
                    donor=BRIDGES[arm] if name=='libmadloop_tirust.so' else Path(info['path'])
                    target=directory/name;shutil.copyfile(donor,target)
                    expected[str(target.resolve())]=sha(target)
                    assert sha(target)==(BRIDGE_HASHES[arm] if name=='libmadloop_tirust.so' else info['sha256'])
                assert len(expected)==15;expected_by_arm[arm]=expected
                env=base_environment(spec);env['LD_LIBRARY_PATH']=str(directory)
                row={'argv':[spec['loader'],'--list',str(EXE)],'progress':{},'environment':env}
                record['loaders'][arm]=row;checkpoint()
                with (RUN/('loader-'+arm+'.log')).open('xb') as log:
                    row['exit_code']=helper['run_plain_capped'](row['argv'],log,cwd=ROOT,env=env,progress=row['progress'],seconds=min(10,remaining(deadline,5)))
                assert row['exit_code']==0;row['closure']=listed((RUN/('loader-'+arm+'.log')).read_text(),expected);checkpoint()
            jobs=[('fault-'+name,'original','fault',name) for name in spec['fault_controls']]
            jobs.extend((arm+'-'+policy,arm,'physical',policy) for arm,policy in spec['physical_workers'])
            for name,arm,kind,policy in jobs:
                built();remaining(deadline,5)
                directory=RUN/name;directory.mkdir()
                if kind=='physical':
                    for source in (HERE/'fixtures').rglob('*'):
                        if source.is_file():
                            target=directory/source.relative_to(HERE/'fixtures');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                before={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}
                env=base_environment(spec);env['LD_LIBRARY_PATH']=str(RUN/('runtime-'+arm))
                argv=[str(EXE),'--fault',policy] if kind=='fault' else [str(EXE),str(directory/'selected.dat'),policy]
                row={'kind':kind,'arm':arm,'policy':policy,'argv':argv,'progress':{},'inputs_before':before,
                     'environment':env,'expected_runtime_sha256':expected_by_arm[arm]}
                record['workers'][name]=row;checkpoint()
                original_capture=helper['capture_maps']
                def acknowledged(*args,**kwargs):
                    mapping=original_capture(*args,**kwargs)
                    if mapping is not None:
                        ack=directory/'mapping-ready'
                        if not ack.exists():ack.write_text('Actual full ELF mapping captured before numerical work\n')
                    return mapping
                helper['capture_maps']=acknowledged
                try:
                    with (directory/'stdout.log').open('xb') as log:
                        row['exit_code'],row['actual_mapping']=helper['run_captured'](
                            argv,log,cwd=directory,env=env,expected_runtime=expected_by_arm[arm],executable=str(EXE.resolve()),
                            selected_library=str((RUN/('runtime-'+arm)/'libmadloop_tirust.so').resolve()),
                            maps_path=directory/'native-maps.txt',progress=row['progress'],
                            seconds=min(10 if kind=='fault' else 120,remaining(deadline,5)))
                except BaseException as error:
                    if isinstance(error,InterruptedError):raise
                    row['process_failure']={'type':type(error).__name__,'message':str(error)}
                finally:
                    helper['capture_maps']=original_capture
                    after={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}
                    row['files_after']=after
                    row['changed_inputs']={p:{'before':digest,'after':after.get(p)} for p,digest in before.items() if after.get(p)!=digest}
                    checkpoint()
                assert not row['changed_inputs'],'Private case input changed'
                code=row.get('exit_code',row['progress'].get('returncode'))
                row['native_output_closure']=output_closure(before,after,kind,code)
                try:
                    parsed=controls_module.parse(directory/'checked-control.log',allow_partial=code!=0)
                    row['control_assessment']=controls_module.assess(parsed,exit_code=code)
                    save(directory/'named-control.json',parsed)
                    if kind=='physical':
                        journal=journal_module.parse(directory/'mass-observer.hex',allow_partial=code!=0)
                        captures[name]=journal
                        summary=journal_module.summarize(journal);save(directory/'journal-summary.json',summary)
                        omissions=sum('##W03' in line for line in (directory/'stdout.log').read_text().splitlines())
                        qualification=journal_module.qualify_checked(journal,parsed['observed'],omissions=omissions)
                        qualification['IEEE_policy_worker_completed']=row['control_assessment']['physical_worker_completed_with_requested_IEEE_policy']
                        row['checked_assessment']=qualification;runtime_results[name]=qualification
                except Exception as error:
                    row['offline_assessment_failure']={'type':type(error).__name__,'message':str(error)}
                checkpoint()
            comparisons={}
            for left,right in [('original-masked','adaptive-masked'),('original-trapped','adaptive-trapped'),('adaptive-masked','adaptive-trapped')]:
                if left in captures and right in captures:
                    comparisons[left+'__'+right]=journal_module.compare_inputs_and_born(captures[left],captures[right])
            parity={'outputs_exact':False,'reason':'Missing candidate policy capture'}
            if 'adaptive-masked' in captures and 'adaptive-trapped' in captures:
                parity=compare_policy_outputs(captures['adaptive-masked'],captures['adaptive-trapped'])
            comparisons['adaptive_masked_trapped_output_parity']=parity
            save(RUN/'comparisons.json',comparisons)
            faults=[record['workers']['fault-'+name].get('control_assessment',{}).get('fault_control_demonstrated',False) for name in spec['fault_controls']]
            record['all_trap_controls_demonstrated']=all(faults)
            input_parity=comparisons.get('adaptive-masked__adaptive-trapped',{})
            pending_candidate=all(faults) and parity['outputs_exact'] and input_parity.get('inputs_and_born_exact',False) and input_parity.get('all_input_records_comparable',False) and all(
                runtime_results.get('adaptive-'+policy,{}).get('checked_outputs_qualified',False) and
                runtime_results.get('adaptive-'+policy,{}).get('qp_coverage_observed',False) and
                runtime_results.get('adaptive-'+policy,{}).get('IEEE_policy_worker_completed',False)
                for policy in ('masked','trapped'))
            record['working_precision_per_call_observed']=False
            built();remaining(deadline)
        except BaseException as error:record['failure']={'type':type(error).__name__,'message':str(error)}
        finally:
            def closure():
                built();record['protected_inputs_after']='unchanged'
                record['runtime_sha256_after']={arm:{p:sha(p) for p in expected} for arm,expected in expected_by_arm.items()}
                assert record['runtime_sha256_after']==expected_by_arm
                for name,row in record['workers'].items():
                    row['complete_case_files_sha256']={relative(p):sha(p) for p in (RUN/name).rglob('*') if p.is_file()}
                record['loadavg_after']=Path('/proc/loadavg').read_text().strip()
                record['execution_complete']=transport_complete(record['workers'],top_failure='failure' in record)
                record['candidate_checked_and_trap_coverage_qualified']=bool(record['execution_complete'] and pending_candidate)
            publish_final(RUN/'execution.json',record,deadline,start,'execution_complete',closure,
                          claim_key='candidate_checked_and_trap_coverage_qualified')
        print(json.dumps({'execution_complete':record['execution_complete'],
                          'candidate_checked_and_trap_coverage_qualified':record.get('candidate_checked_and_trap_coverage_qualified',False),
                          'physical_truth_or_performance_qualified':False,'failure':record.get('failure')},sort_keys=True))
        if not record['execution_complete']:raise SystemExit(1)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('action',choices=('prepare','verify','build','run'))
    action=cli.parse_args().action
    if action=='prepare':source_prepare()
    elif action=='verify':print(json.dumps({'verified':bool(verify()),'native_launched':False}))
    elif action=='build':build()
    else:run()
