"""One new four-lepton process, five explicit reducers, no acceptance gate.

Private copies only. The unchanged worker learns100 points, then times complete
100-point traversals. Source/prepare/RUN are separately bound and append-only.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
REPORT=ROOT/'docs/benchmarkfolder/experiments/current-source-four-lepton-timing-1'
WORK=ROOT/'.tirust-build/current-source-four-lepton-timing-work-1'
CAMPAIGN=None
ORIGINAL=ROOT/'.tirust-build/current-source-four-lepton-preparation-1'
CASES=('qcd_n4_uux_epem_mupmum',)
BACKENDS=(1,6,7,8,9)
CPU=12;CASE_SECONDS=600.;TOTAL_SECONDS=600.;CLEANUP_SECONDS=60.;TARGET_SECONDS=2.
LIMITS={'address_space_bytes':16*1024**3,'file_bytes':512*1024**2,'core_bytes':0}
SUPPORT={
 'cleanup':('docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py','240d1067a7b7098f056f0f367097def8f374d7979c3001d249c8b1f7cb0bbd84'),
 'loader':('docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-3/loader_checks.py','4b8d035797b0ad742a0b0903228a20cc4bddca922146c4c5ea02ad4f3a011e55'),
}
sys.path.insert(0,str(ROOT/'tools'))
import tirust_benchmark as bench
from tirust_benchmark_provisional import alarm_deadline,BudgetExpired,Subreaper,private_execution
from tirust_benchmark_worker import RAW_OUTPUTS
from tirust_benchmark_warmup import parse_warmup
EXPECTED_SOURCE=None;EXPECTED_PREPARED=None

def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as out:json.dump(value,out,indent=2,sort_keys=True,allow_nan=False);out.write('\n')
def load(name,path,checksum):
    require(digest(path)==checksum,'Pinned helper changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
def local(name):
    return load('four_lepton_'+name,HERE/(name+'.py'),read(HERE/'source-index.json')['files'][name+'.py']['sha256'])
def support(name):
    path,checksum=SUPPORT[name];return load('four_lepton_'+name,ROOT/path,checksum)

def verify_source():
    require(EXPECTED_SOURCE and digest(HERE/'source-index.json')==EXPECTED_SOURCE,'Exact reviewed source seal required')
    seal=read(HERE/'source-index.json')
    require({str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}==set(seal['files'])|{'source-index.json'},'Source inventory changed')
    for name,row in seal['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe source path')
        require((HERE/name).stat().st_size==row['bytes'] and digest(HERE/name)==row['sha256'],'Controller source changed')
    for name,checksum in seal['support'].items():require(digest(ROOT/name)==checksum,'Frozen protocol/support changed: '+name)
    bench.CAMPAIGN_ID=bench.CAMPAIGN_MANIFEST=bench.CAMPAIGN_MANIFEST_SHA256=None
    bench.WORK=ORIGINAL/'work';bench.REPORT=ORIGINAL/'report';bench.MODEL_ROOT=bench.WORK/'models'
    bench.TIRUST_PREFIX=ROOT/'.tirust-build/qp-transport-install/mg/HEPTools/tirust'
    local('preparation').shared_bindings(bench)
    require(CPU in os.sched_getaffinity(0),'Nonexclusive CPU12 unavailable')

def preflight():
    verify_source();originals=local('preparation').collect_originals(bench)
    references={case:reference_identity(case,originals[case]) for case in CASES}
    return {'source_index_sha256':EXPECTED_SOURCE,'status':'pure-preflight-passed','originals':originals,'references':references,
            'native_executions':0,'compiler_invocations':0,'case_caps_seconds':CASE_SECONDS,'total_cap_seconds':TOTAL_SECONDS}

def reference_identity(case,original):
    require(case==CASES[0],'Unknown fresh-reference case')
    return {'mode':'fresh-backend-1-full100-learning','historical_reference':None,'backend':1,
            'points':100,'points_sha256':original['points_sha256'],
            'executable_sha256':original['executable_sha256'],'acquisition_pending':True}

def learning_check(backend,learned,reference,tolerance):
    """Form CutTools reference first; other backends compare only after it exists."""
    if backend==1:
        require(reference is None,'Fresh CutTools reference may be formed only once')
        return learned,{'passed':True,'ordinary_target_passed':True,
            'role':'fresh-reference-acquisition','independent_accuracy_claim':False}
    require(reference is not None,'Fresh full100 CutTools reference must precede other backends')
    return reference,bench.compare_rows(reference['rows'],learned['rows'],tolerance)

def expected_runtime(original):
    """Use actual ELF PT_INTERP; retain but exclude only the old ldd alias."""
    loader=support('loader');directory=Path(original['original_output'])/'SubProcesses'/original['subprocess']
    actual=loader._support('run.py',{'interpreter'}).interpreter(directory/'check')
    old=original['runtime'];files={str(Path(p).resolve()):h for p,h in old['libraries'].items()};aliases={}
    for left,right in re.findall(r'^\s*(/[^\s]+)\s*=>\s*(/[^\s]+)\s*\(',old['ldd'],re.M):
        if Path(left).resolve()==actual and Path(right).resolve()!=actual:
            name=str(Path(right).resolve());require('ld-linux' in Path(name).name and name in files,'Unknown loader alias')
            aliases[name]=files.pop(name)
    require(str(actual) in files,'Actual PT_INTERP absent from closed runtime pin')
    require(any(h=='eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9' for h in files.values()),'Wrong native TIRust')
    return {'files':files,'actual_interpreter':str(actual),'historical_ldd_only_aliases':aliases,
            'runtime_verified_for_original':False,
            'scope':'Expected providers prebound from closed n1; actual new4l loader/maps are required during execution'}

def prepare():
    before=preflight();require(not REPORT.exists() and not WORK.exists(),'Append-only preparation already exists')
    REPORT.mkdir(parents=True);protocol=REPORT/'protocol';protocol.mkdir()
    seal=read(HERE/'source-index.json')
    for name in seal['support']:
        if name.startswith('tools/'):
            shutil.copy2(ROOT/name,protocol/Path(name).name)
    shutil.copy2(HERE/'source-index.json',REPORT/'source-index.json')
    for name in seal['files']:
        target=REPORT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(HERE/name,target)
    shutil.copy2(bench.REPORT/'cases.json',REPORT/'cases.json')
    shutil.copy2(ROOT/'docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/manifest.json',REPORT/'source-manifest.json')
    for case in CASES:
        for suffix in ('.dat','.json'):
            target=REPORT/'points'/(case+suffix);target.parent.mkdir(exist_ok=True)
            shutil.copy2(bench.REPORT/'points'/(case+suffix),target)
    prepared=local('preparation').prepare(bench,WORK/'clones',REPORT)
    require(before['originals']==prepared['originals'],'Original inputs changed during preparation')
    dump(REPORT/'reference-admission.json',before['references'])
    runtime={case:expected_runtime(prepared['originals'][case]) for case in CASES}
    locale=Path('/nix/store/ig5ylfz7qcgpmvzip4pxfkp2pfi76s2j-glibc-locales-2.44-25/lib/locale/locale-archive')
    locale_hash='53851ab7caa5a0f893056762a6abff4e383ea06c0e6b276ee16876499e712d27'
    require(digest(locale)==locale_hash,'Frozen locale map changed')
    spec={'schema':'madloop-current-four-lepton-five-backend-descriptive-v1','source_index_sha256':EXPECTED_SOURCE,
        'created_utc':bench.utc_now(),'case_ids':list(CASES),'backends':list(BACKENDS),'cpu':CPU,'cpu_reservation':False,
        'machine':bench.machine_identity(CPU),'cycles':local('analyze').schedule(),'points_per_batch':100,'point_subset':None,
        'target_batch_seconds':TARGET_SECONDS,'case_cap_seconds':CASE_SECONDS,'total_cap_seconds':TOTAL_SECONDS,
        'cleanup_reserve_seconds':CLEANUP_SECONDS,'resource_limits':LIMITS,'preparation_sha256':digest(REPORT/'preparation.json'),
        'actual_runtime':runtime,'allowed_extra_mappings':{str(locale):locale_hash},
        'compiler_or_relink_performed':False,'native_evaluations_during_preparation':0,'performance_acceptance':False,
        'clock':'Fortran SYSTEM_CLOCK wall elapsed / (100 * loops)','initialization_scope':'First amplitude + full100 learning; excludes SETPARA/process startup',
        'frozen_files':{str(p.relative_to(REPORT)):digest(p) for p in sorted(REPORT.rglob('*')) if p.is_file()}}
    dump(REPORT/'specification.json',spec)
    print(json.dumps({'prepared_specification_sha256':digest(REPORT/'specification.json'),'clones':5,'native_executions':0}),flush=True)

def verify_prepared(check_clones=True):
    verify_source();require(EXPECTED_PREPARED and digest(REPORT/'specification.json')==EXPECTED_PREPARED,'Exact reviewed prepared specification required')
    spec=read(REPORT/'specification.json');require(spec['source_index_sha256']==EXPECTED_SOURCE,'Prepared/source mismatch')
    for name,checksum in spec['frozen_files'].items():require(digest(REPORT/name)==checksum,'Prepared evidence changed: '+name)
    prepared=read(REPORT/'preparation.json');helper=local('preparation')
    require(helper.collect_originals(bench)==prepared['originals'],'Original source/object/card/runtime changed')
    require({case:reference_identity(case,prepared['originals'][case]) for case in CASES}==read(REPORT/'reference-admission.json'),
            'Fresh CutTools acquisition plan changed')
    if check_clones:
        for item in prepared['clones'].values():helper.verify_clone(bench,Path(item['private_output']),item)
    return spec,prepared

def loader_query(exe,expected,destination):
    loader=support('loader')
    def callback(argv,label):
        require(argv==[expected['actual_interpreter'],'--list',str(exe.resolve())],'Unexpected loader invocation')
        dump(destination/(label+'-command.json'),{'argv':argv,'executable_sha256':digest(exe),'invokes_fortran_main':False})
        text=subprocess.check_output(argv,env=bench.environment(),text=True,stderr=subprocess.STDOUT)
        (destination/(label+'.txt')).write_text(text);return text
    return loader.query_runtime(exe,expected['files'],expected['files'],callback)

def mapped_files(worker,expected,spec):
    text=Path(f'/proc/{worker.process.pid}/maps').read_text();files={}
    for line in text.splitlines():
        fields=line.split(None,5)
        if len(fields)==6 and fields[-1].startswith('/'):
            require(not fields[-1].endswith(' (deleted)'),'Deleted runtime mapping')
            path=Path(fields[-1]).resolve();files[str(path)]=digest(path)
    exe=(worker.directory/'check').resolve()
    required={**expected['files'],str(exe):digest(exe)}
    require(all(files.get(p)==h for p,h in required.items()),'Expected executable/provider absent from maps')
    allowed={**required,**spec['allowed_extra_mappings']}
    require(all(allowed.get(p)==h for p,h in files.items()),'Unexpected mapped file/provider')
    return {'pid':worker.process.pid,'maps':text,'files':files}

def capture_last(case,prepared):
    for backend in BACKENDS:
        item=prepared['clones'][case+'/'+str(backend)];directory=Path(item['private_output'])/'SubProcesses'/item['subprocess']
        lane=REPORT/'cases'/case/str(backend)
        if not (lane/'started-worker.json').exists():continue
        target=lane/'last-raw';target.mkdir(exist_ok=True)
        for name in RAW_OUTPUTS:
            path=directory/name
            if path.is_file():shutil.copy2(path,target/name)

def case_child(case_id,deadline):
    spec,prepared=verify_prepared();case=next(c for c in bench.load_config()['cases'] if c['id']==case_id)
    config=bench.load_config();folder=REPORT/'cases'/case_id;admission=read(folder/'admission.json')
    require(admission['work_deadline_monotonic']==deadline and admission['prepared_sha256']==EXPECTED_PREPARED,'Missing exact case admission')
    for kind,name in ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_FSIZE,'file_bytes'),(resource.RLIMIT_CORE,'core_bytes')):
        resource.setrlimit(kind,(LIMITS[name],LIMITS[name]))
    def remaining():
        value=deadline-time.monotonic()
        if value<=0:raise BudgetExpired('Case work deadline reached')
        return value
    workers={};inputs={};settings={};ready={};calibration={};checks={str(b):[] for b in BACKENDS};cycles=[]
    reference=None
    try:
        with ExitStack() as stack:
            lock=stack.enter_context((ROOT/'.tirust-build/numerical-tests.lock').open('a'))
            while True:
                remaining()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(min(.1,remaining()))
            dump(folder/'lock-acquired.json',{'utc':bench.utc_now(),'path':str(Path(lock.name))})
            for backend in BACKENDS:
                key=str(backend);lane=folder/key;lane.mkdir()
                workspace=WORK/'clones'/key;item=prepared['clones'][case_id+'/'+key]
                with private_execution(workspace):
                    local('preparation').verify_clone(bench,Path(item['private_output']),item)
                    exe=bench.subprocess_dir('candidate',case)/'check'
                    dump(lane/'actual-runtime-before.json',loader_query(exe,spec['actual_runtime'][case_id],lane/'loader-before'))
                    dump(lane/'started-worker.json',{'utc':bench.utc_now(),'points':100,'requested_backend':backend})
                    workers[key],settings[key]=bench.start_worker(stack,'candidate',case,[backend],config,CPU,lane,
                        timeout=bench.case_worker_timeout(case),points_path=REPORT/'points'/(case_id+'.dat'))
                    inputs[key]=read(lane/'candidate-inputs.json')
                ready[key]={name:getattr(workers[key],name) for name in ('points','ndim','nso','nborn','cold_seconds')}
                dump(lane/'ready.json',ready[key]);dump(lane/'maps-before.json',mapped_files(workers[key],spec['actual_runtime'][case_id],spec))
                learned=parse_warmup(lane/'candidate-initialization',[backend],100,ready[key]);dump(lane/'learning.json',learned)
                reference,check=learning_check(backend,learned,reference,config['relative_tolerance'])
                if backend==1:
                    dump(folder/'fresh-cuttools-reference.json',{'observation':reference,
                        'origin':'backend1-full100-learning','point_sha256':item['points_sha256'],
                        'executable_sha256':item['executable_sha256'],'independent_accuracy_claim':False})
                checks[key].append(check)
                dump(lane/'learning-comparison.json',check);require(check['passed'],'Full100 numerical learning disagrees')
                require(settings[key]['looppolemode']==0 and settings[key]['colliercomputeuvpoles'] is True and settings[key]['colliercomputeirpoles'] is True
                    and settings[key]['collierusecacheforpoles'] is False and settings[key]['mlstabthres']==1e-6,'Pole/cache/stability policy mismatch')
                remaining()
            common=set(settings['1'])-{'mlreductionlib'}
            require(all({k:settings[str(b)][k] for k in common}=={k:settings['1'][k] for k in common} for b in BACKENDS),'Backend cards differ beyond selection')
            def observe(backend,loops,tag):
                key=str(backend);worker=workers[key];worker.timeout=min(bench.case_worker_timeout(case),remaining())
                before=bench.host_load_snapshot();target=folder/key/'raw'/tag
                value=bench.observe(worker,[backend],loops,target,count=100)
                dump(target/'host.json',{'before':before,'after':bench.host_load_snapshot()})
                check=bench.compare_rows(reference['rows'],value['rows'],config['relative_tolerance']);dump(target/'comparison.json',check)
                checks[key].append(check);require(check['passed'],'Timed traversal numerical disagreement');remaining();return value
            for backend in BACKENDS:
                key=str(backend);first=observe(backend,1,'calibration-0');count=max(1,math.ceil(TARGET_SECONDS/first['seconds']))
                second=observe(backend,count,'calibration-1');loops=max(1,math.ceil(count*TARGET_SECONDS/second['seconds']))
                calibration[key]={'target_seconds':TARGET_SECONDS,'loops':loops,'initial':first,'sustained_loops':count,'sustained':second}
                dump(folder/key/'calibration.json',calibration[key])
            for row in local('analyze').schedule():
                observations={}
                for backend in row['order']:
                    key=str(backend);loops=calibration[key]['loops']
                    observations[key]={'loops':loops,'observation':observe(backend,loops,'cycle-'+str(row['cycle']))}
                value={**row,'observations':observations};cycles.append(value);dump(folder/('cycle-'+str(row['cycle'])+'.json'),value)
            for backend in BACKENDS:
                key=str(backend);dump(folder/key/'maps-after.json',mapped_files(workers[key],spec['actual_runtime'][case_id],spec))
        # Every worker is closed before live-card/source/runtime postchecks.
        for backend in BACKENDS:
            key=str(backend);item=prepared['clones'][case_id+'/'+key]
            with private_execution(WORK/'clones'/key):
                dump(folder/key/'inputs-after.json',bench.verify_observation_inputs('candidate',case,inputs[key]))
                local('preparation').verify_clone(bench,Path(item['private_output']),item)
            calibration[key].update(numerical_regression_pass=all(c['passed'] for c in checks[key]),
                                    ordinary_target_passed=all(c['ordinary_target_passed'] for c in checks[key]))
        summary=local('analyze').summarize_case(case_id,cycles,ready,calibration)
        dump(folder/'result.json',{'status':'complete','case':case,'summary':summary,'numerical_checks':checks,
            'source_index_sha256':EXPECTED_SOURCE,'prepared_sha256':EXPECTED_PREPARED,
            'performance_acceptance':False,'source_scope':'new4l afb6/current0fcd numerical source with eb3 candidate only; fresh CT reference; no9228 speed comparison'})
    except BaseException as error:
        dump(folder/'failure.json',{'error_class':type(error).__name__,'error':repr(error),'performance_acceptance':False});raise
    finally:capture_last(case_id,prepared)

def execute():
    start=time.monotonic();end=start+TOTAL_SECONDS;rows=[];spec=None
    try:
        with alarm_deadline(end):
            spec,prepared=verify_prepared();dump(REPORT/'run-start.json',{'utc':bench.utc_now(),'total_deadline_monotonic':end,'performance_acceptance':False})
            cleanup=support('cleanup')
            with cleanup.SignalRequests() as requests,Subreaper() as reaper:
                class Tracked(cleanup.OwnedChild):
                    def sample(self):super().sample();reaper.adopt(self,cleanup.identity)
                for case in CASES:
                    requests.check();finish=min(end,time.monotonic()+CASE_SECONDS);work_until=finish-CLEANUP_SECONDS;cleanup_until=finish-2.
                    if work_until<=time.monotonic():rows.append({'case':case,'status':'not_run_budget'});continue
                    folder=REPORT/'cases'/case;folder.mkdir(parents=True,exist_ok=False)
                    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child',case,'--deadline',repr(work_until),
                             '--source',EXPECTED_SOURCE,'--prepared',EXPECTED_PREPARED]
                    dump(folder/'admission.json',{'case':case,'argv':command,'work_deadline_monotonic':work_until,'finish_deadline_monotonic':finish,
                         'source_sha256':EXPECTED_SOURCE,'prepared_sha256':EXPECTED_PREPARED,'cpu_reservation':False})
                    owned=process=None;record=None;state='unavailable'
                    try:
                        with alarm_deadline(cleanup_until):
                            try:
                                with alarm_deadline(work_until):
                                    with (folder/'console.log').open('xb') as output:
                                        process=subprocess.Popen(command,cwd=ROOT,env=bench.environment(),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                                        owned=Tracked(process);print('START '+case+' controller='+str(process.pid),flush=True)
                                        while True:
                                            owned.sample();requests.check()
                                            if process.poll() is not None:owned.sample();break
                                            time.sleep(.1)
                                    state='complete' if process.returncode==0 and (folder/'result.json').is_file() else 'failed'
                            except BudgetExpired:state='unavailable_deadline'
                            finally:
                                if owned is not None:
                                    record=owned.stop(cooperative_seconds=45.,terminate_seconds=5.,kill_seconds=5.);reaper.reap(owned)
                                    require(not record['survivors'] and not owned.live(),'Unresolved owned cleanup')
                                else:record=reaper.stop_unassigned(cleanup);require(not record['survivors'],'Unresolved spawn cleanup')
                                dump(folder/'cleanup.json',record)
                            capture_last(case,prepared);requests.check();verify_prepared()
                    except BaseException:
                        with alarm_deadline(finish):
                            if owned is not None:
                                record=owned.stop(cooperative_seconds=0.,terminate_seconds=0.,kill_seconds=min(1.,max(0.,finish-time.monotonic())));reaper.reap(owned)
                            else:record=reaper.stop_unassigned(cleanup)
                            dump(folder/'cleanup-interruption.json',record)
                        raise
                    with alarm_deadline(finish):
                        requests.check()
                        row={'case':case,'status':state,'returncode':process.returncode if process else None,'cleanup':record,
                             'performance_acceptance':False,'result_sha256':digest(folder/'result.json') if (folder/'result.json').exists() else None}
                        dump(folder/'completion.json',row);rows.append(row)
                    print('CLOSED '+case+' '+state,flush=True)
                requests.check()
                dump(REPORT/'summary.json',{'schema':'madloop-current-four-lepton-five-backend-descriptive-v1','cases':rows,
                     'performance_acceptance':False,'cpu_reservation':False,'source_sha256':EXPECTED_SOURCE,'prepared_sha256':EXPECTED_PREPARED})
    except BaseException as error:
        dump(REPORT/'administrative-failure.json',{'error':repr(error),'completed':rows,'performance_acceptance':False,
             'scope':'Postdeadline administrative failure publication may be outside cap; unfinished samples remain unavailable'})
        raise

def main():
    global EXPECTED_SOURCE,EXPECTED_PREPARED
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true')
    mode.add_argument('--child',choices=CASES);parser.add_argument('--deadline',type=float);parser.add_argument('--source',required=True);parser.add_argument('--prepared')
    args=parser.parse_args();EXPECTED_SOURCE=args.source;EXPECTED_PREPARED=args.prepared
    require(re.fullmatch('[0-9a-f]{64}',EXPECTED_SOURCE),'Exact source SHA required')
    if args.preflight:print(json.dumps(preflight(),sort_keys=True,indent=2));return
    if args.prepare:prepare();return
    require(EXPECTED_PREPARED and re.fullmatch('[0-9a-f]{64}',EXPECTED_PREPARED),'Reviewed prepared SHA required')
    if args.child:require(args.deadline and math.isfinite(args.deadline),'Bounded child deadline required');case_child(args.child,args.deadline)
    else:execute()
if __name__=='__main__':main()
