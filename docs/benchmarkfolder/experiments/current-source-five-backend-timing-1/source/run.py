"""Three fixed current-source SM processes, five explicit reducers, no acceptance gate.

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
REPORT=ROOT/'docs/benchmarkfolder/experiments/current-source-five-backend-timing-1'
WORK=ROOT/'.tirust-build/current-source-five-backend-timing-work-1'
CAMPAIGN='entry-latch-cache-guard-1'
CASES=('qcd_n1_uux_z','qcd_n2_gg_ttx','ew_n2_epem_mupmum')
BACKENDS=(1,6,7,8,9)
CPU=12;CASE_SECONDS=600.;TOTAL_SECONDS=1800.;CLEANUP_SECONDS=60.;TARGET_SECONDS=2.
LIMITS={'address_space_bytes':16*1024**3,'file_bytes':512*1024**2,'core_bytes':0}
SUPPORT={
 'cleanup':('docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py','240d1067a7b7098f056f0f367097def8f374d7979c3001d249c8b1f7cb0bbd84'),
 'loader':('docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-3/loader_checks.py','4b8d035797b0ad742a0b0903228a20cc4bddca922146c4c5ea02ad4f3a011e55'),
}
sys.path.insert(0,str(ROOT/'tools'))
import tirust_benchmark as bench
from tirust_benchmark_campaign import activate_campaign
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
    return load('five_backend_'+name,HERE/(name+'.py'),read(HERE/'source-index.json')['files'][name+'.py']['sha256'])
def support(name):
    path,checksum=SUPPORT[name];return load('five_backend_'+name,ROOT/path,checksum)

def verify_source():
    require(EXPECTED_SOURCE and digest(HERE/'source-index.json')==EXPECTED_SOURCE,'Exact reviewed source seal required')
    seal=read(HERE/'source-index.json')
    require({str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}==set(seal['files'])|{'source-index.json'},'Source inventory changed')
    for name,row in seal['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe source path')
        require((HERE/name).stat().st_size==row['bytes'] and digest(HERE/name)==row['sha256'],'Controller source changed')
    for name,checksum in seal['support'].items():require(digest(ROOT/name)==checksum,'Frozen protocol/support changed: '+name)
    activate_campaign(bench,CAMPAIGN)
    require(bench.CAMPAIGN_MANIFEST_SHA256==seal['manifest_sha256'],'Wrong campaign manifest')
    require(CPU in os.sched_getaffinity(0),'Nonexclusive CPU12 unavailable')

def preflight():
    verify_source();originals=local('preparation').collect_originals(bench)
    references={case:reference_identity(case,originals[case]) for case in CASES}
    return {'source_index_sha256':EXPECTED_SOURCE,'status':'pure-preflight-passed','originals':originals,'references':references,
            'native_executions':0,'compiler_invocations':0,'case_caps_seconds':CASE_SECONDS,'total_cap_seconds':TOTAL_SECONDS}

def reference_identity(case,original):
    directory=bench.REPORT/'experiments/numerical'/('full100-warmup-'+case+'-1')/'candidate-1'
    inputs,after,result=(read(directory/name) for name in ('candidate-inputs.json','inputs-after.json','result.json'))
    require(result.get('status')=='complete' and result.get('backend')==1 and result.get('source')=='candidate','Incomplete CutTools reference')
    output=Path(original['original_output'])
    require(inputs['generation']==read(output/'generation.json') and inputs['compilation']==read(output/'compilation.json')
            and inputs['campaign_binding']==bench.campaign_binding(),'CutTools reference build/source mismatch')
    require(inputs['points_sha256']==original['points_sha256'] and inputs['points_metadata_sha256']==original['points_metadata_sha256']
            and inputs['param_card_sha256']==original['physical_card_sha256'],'CutTools reference point/card mismatch')
    require(inputs['settings']['mlreductionlib']=='1','CutTools reference selection mismatch')
    for key in ('libraries','archives','metadata'):
        require(inputs['runtime'][key]==original['runtime'][key]==after['runtime'][key],'CutTools reference runtime mismatch')
    require(after['executable_sha256']==original['executable_sha256'] and after['points_sha256']==original['points_sha256']
            and after['numerical_cards']==inputs['numerical_cards'],'CutTools reference postcheck mismatch')
    raw=directory/'raw';observed=parse_warmup(raw,[1],100,read(raw/'ready.json'))
    require(observed==read(raw/'observation.json')==result['observation'],'CutTools reference raw replay differs')
    require(result['comparison']['passed'] is True,'CutTools reference regression failed')
    return {'path':str(directory),'files':{str(p.relative_to(directory)):digest(p) for p in sorted(directory.rglob('*')) if p.is_file()},
            'replayed_points':100,'backend':1,'source_build_points_runtime_matched':True}

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
    return {'files':files,'actual_interpreter':str(actual),'historical_ldd_only_aliases':aliases}

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
    for name in ('manifest.json','cases.json'):shutil.copy2(bench.REPORT/name,REPORT/name)
    for case in CASES:
        for suffix in ('.dat','.json'):
            target=REPORT/'points'/(case+suffix);target.parent.mkdir(exist_ok=True)
            shutil.copy2(bench.REPORT/'points'/(case+suffix),target)
        reference=bench.REPORT/'experiments/numerical'/('full100-warmup-'+case+'-1')/'candidate-1'
        target=REPORT/'reference'/case;target.mkdir(parents=True)
        for name in ('candidate-inputs.json','inputs-after.json','result.json'):
            shutil.copy2(reference/name,target/name)
        shutil.copytree(reference/'raw',target/'raw')
    prepared=local('preparation').prepare(bench,WORK/'clones',REPORT)
    require(before['originals']==prepared['originals'],'Original inputs changed during preparation')
    dump(REPORT/'reference-admission.json',before['references'])
    runtime={case:expected_runtime(prepared['originals'][case]) for case in CASES}
    locale=Path('/nix/store/ig5ylfz7qcgpmvzip4pxfkp2pfi76s2j-glibc-locales-2.44-25/lib/locale/locale-archive')
    locale_hash='53851ab7caa5a0f893056762a6abff4e383ea06c0e6b276ee16876499e712d27'
    require(digest(locale)==locale_hash,'Frozen locale map changed')
    spec={'schema':'madloop-current-five-backend-descriptive-v1','source_index_sha256':EXPECTED_SOURCE,
        'created_utc':bench.utc_now(),'case_ids':list(CASES),'backends':list(BACKENDS),'cpu':CPU,'cpu_reservation':False,
        'machine':bench.machine_identity(CPU),'cycles':local('analyze').schedule(),'points_per_batch':100,'point_subset':None,
        'target_batch_seconds':TARGET_SECONDS,'case_cap_seconds':CASE_SECONDS,'total_cap_seconds':TOTAL_SECONDS,
        'cleanup_reserve_seconds':CLEANUP_SECONDS,'resource_limits':LIMITS,'preparation_sha256':digest(REPORT/'preparation.json'),
        'actual_runtime':runtime,'allowed_extra_mappings':{str(locale):locale_hash},
        'compiler_or_relink_performed':False,'native_evaluations_during_preparation':0,'performance_acceptance':False,
        'clock':'Fortran SYSTEM_CLOCK wall elapsed / (100 * loops)','initialization_scope':'First amplitude + full100 learning; excludes SETPARA/process startup',
        'frozen_files':{str(p.relative_to(REPORT)):digest(p) for p in sorted(REPORT.rglob('*')) if p.is_file()}}
    dump(REPORT/'specification.json',spec)
    print(json.dumps({'prepared_specification_sha256':digest(REPORT/'specification.json'),'clones':15,'native_executions':0}),flush=True)

def verify_prepared(check_clones=True):
    verify_source();require(EXPECTED_PREPARED and digest(REPORT/'specification.json')==EXPECTED_PREPARED,'Exact reviewed prepared specification required')
    spec=read(REPORT/'specification.json');require(spec['source_index_sha256']==EXPECTED_SOURCE,'Prepared/source mismatch')
    for name,checksum in spec['frozen_files'].items():require(digest(REPORT/name)==checksum,'Prepared evidence changed: '+name)
    prepared=read(REPORT/'preparation.json');helper=local('preparation')
    require(helper.collect_originals(bench)==prepared['originals'],'Original source/object/card/runtime changed')
    require({case:reference_identity(case,prepared['originals'][case]) for case in CASES}==read(REPORT/'reference-admission.json'),
            'CutTools reference changed')
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
    reference=read(REPORT/'reference'/case_id/'raw/observation.json')
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
                check=bench.compare_rows(reference['rows'],learned['rows'],config['relative_tolerance']);checks[key].append(check)
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
            'performance_acceptance':False,'source_scope':'current registered0fcd/eb3 candidate only; no9228 speed comparison'})
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
                dump(REPORT/'summary.json',{'schema':'madloop-current-five-backend-descriptive-v1','cases':rows,
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
