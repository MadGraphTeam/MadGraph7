"""Validation-only successor; preserve failed compilation-stage closure exactly."""
import argparse,ast,fcntl,hashlib,importlib.util,json,os,re,resource,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
WORK=ROOT/'.tirust-build/rank6-ir-triangle-validation-work-1'
ORIGINAL=ROOT/'.tirust-build/rank6-ir-triangle-controller-1'
COMPILED=ROOT/'.tirust-build/rank6-ir-triangle-work-1/build'
SOURCE1='4f496ebf3071e8d405e1e2e72b4bcff3ab7b210ef2338740cdbe314b0cfcb2b0'
FAILED1='e1d442dbc86c04606a5aba1ee5f83614b00c64d6ac9ef0a07558277fc5f2ad77'
EXTRA='/nix/store/chqq8mpmpyfi9kgsngya71akv5xicn03-gcc-15.2.0-lib/lib/libquadmath.so.0.0.0'
EXTRA_SHA='78d384074a271a26fd8bd466aedb4f285e56547daf596aa898c3bfff4bcaa6b5'
CPU=12;CAPS={'validate':180.};CLEANUP=60.
LIMITS={'address_space_bytes':4294967296,'file_bytes':536870912,'core_bytes':0}
EXPECTED_SOURCE=None;REVIEWED_BUILD=None

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def require(x,m):
    if not x:raise ValueError(m)
def module(name,path,pin):
    require(digest(path)==pin,'Changed imported source')
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj;spec.loader.exec_module(obj);return obj

def original():
    index=read(ORIGINAL/'source-index.json')
    require(digest(ORIGINAL/'source-index.json')==SOURCE1,'Changed original source seal')
    obj=module('ir_validate_original',ORIGINAL/'controller.py',index['files']['controller.py']['sha256'])
    obj.EXPECTED_SOURCE=SOURCE1;obj.REVIEWED_BUILD=None
    return obj

def support(name):return original().support(name)
def environment():return original().environment()
def original_inputs():return read(ORIGINAL/'input-pins.json')

def check_files(base,files):
    for name,w in files.items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe package path')
        q=base/p;require(q.is_file() and not q.is_symlink(),'Missing indexed regular file')
        require(q.stat().st_size==w['bytes'] and digest(q)==w['sha256'],'Indexed source differs '+name)
def verify_source():
    require(EXPECTED_SOURCE and digest(HERE/'source-index.json')==EXPECTED_SOURCE,'Exact reviewed source SHA256 required')
    own=read(HERE/'source-index.json');check_files(HERE,own['files'])
    require({str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}==set(own['files'])|{'source-index.json'},'Validation source inventory differs')
    baseline=read(HERE/'input-pins.json')
    for path,w in baseline['protected_files'].items():
        require(Path(path).is_file() and Path(path).stat().st_size==w['bytes'] and digest(path)==w['sha256'],'Protected original artifact differs '+path)
    require({str(p.relative_to(COMPILED)) for p in COMPILED.rglob('*') if p.is_file()}==set(baseline['failed_build_files']),'Original failed BUILD inventory differs')
    original().verify_source()
    return {'protected_files_checked':len(baseline['protected_files']),'original_source_sha256':SOURCE1,'original_failed_terminal_sha256':FAILED1}

def join_record():
    """New post-failure manifest, explicitly distinct from the original helper."""
    pins=original_inputs();old=read(COMPILED/'prelink.json')
    require(digest(COMPILED/'terminal.json')==FAILED1,'Different failed BUILD')
    failure=read(COMPILED/'terminal.json')
    require(failure['status']=='failed' and failure['returncode']==1 and not failure['cleanup']['survivors'],'Failed BUILD closure differs')
    require('Expected runtime provider was not prebound' in (COMPILED/'console.log').read_text(),'Different administrative failure')
    require(not (COMPILED/'result.json').exists(),'Failed BUILD unexpectedly gained a result')
    labels={p.name.removesuffix('.terminal.json') for p in (COMPILED/'commands').glob('*.terminal.json')}
    require(labels=={'compiler-version','driver-compile','link'}|{'prelink-query-'+str(i).zfill(3) for i in range(51)},'Original54 tool command set differs')
    for label in labels:
        command=read(COMPILED/'commands'/(label+'.command.json'));terminal=read(COMPILED/'commands'/(label+'.terminal.json'))
        require(terminal['returncode']==0 and terminal['argv']==command['argv'],'Original command failed or differs')
        require(terminal['stdout_sha256']==digest(COMPILED/'commands'/(label+'.stdout.txt')),'Original tool log differs')
        require(command['argv'][0]==pins['tools']['gfortran'],'Unexpected original executable')
    missing={name:pin for name,pin in pins['expected_runtime'].items() if old['bound'].get(name)!=pin}
    require(missing=={EXTRA:EXTRA_SHA},'Missing runtime binding is no longer exactly the scoped quadmath file')
    require(pins['files'].get(EXTRA)==EXTRA_SHA and digest(EXTRA)==EXTRA_SHA,'Extra runtime was not in original protected source manifest')
    source=ORIGINAL/'controller.py';tree=ast.parse(source.read_text())
    child=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='child')
    calls=[(n.lineno,n.func.id) for n in ast.walk(child) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    require(min(n for n,k in calls if k=='verify_source')<min(n for n,k in calls if k=='build_locked'),'Original control flow does not verify source before compilation')
    admission=read(COMPILED/'admission.json')
    require(admission['stage']=='build' and admission['source_index_sha256']==SOURCE1,'Original BUILD source admission differs')
    require(read(COMPILED/'lock.json')['acquired_utc']<read(COMPILED/'commands/driver-compile.command.json')['started_utc'],'Original execution ordering differs')
    combined=dict(old['bound']);combined[EXTRA]=EXTRA_SHA
    return {'schema':'rank6-post-failure-runtime-prebinding-join-v1','bound':combined,
      'record_created_after_failed_build':True,'original_helper_bound_unchanged':True,'original_helper_prelink_sha256':digest(COMPILED/'prelink.json'),
      'added_bindings':missing,'origin':'Original SOURCE1 protected manifest guarded before version/compile/link commands; this is a new provenance join, not a claim that the original helper map contained the transitive provider.',
      'original_source_index_sha256':SOURCE1,'original_input_manifest_sha256':digest(ORIGINAL/'input-pins.json'),'original_controller_sha256':digest(source),
      'original_failed_terminal_sha256':FAILED1,'original_54_commands_all_zero':True,
      'source_guard_evidence':{'kind':'Immutable source/admission/control-flow and observed subsequent commands; no retrospective hash measurement invented.',
       'source_verification_precedes_build_call':True,'protected_provider_path':EXTRA,'protected_provider_sha256':EXTRA_SHA}}

def preflight():
    before=verify_source();join_record();require(CPU in os.sched_getaffinity(0),'CPU12 unavailable')
    return {'status':'validation-source-preflight-passed','source_index_sha256':EXPECTED_SOURCE,'protected':before,'native_tensor_calls':0,'compiler_calls':0,'loader_queries':0,'caps_seconds':CAPS}
def remaining(directory):
    left=read(directory/'admission.json')['work_deadline_monotonic']-time.monotonic();require(left>0,'Original validation work deadline expired');return left
def command(argv,directory,label):
    pins=original_inputs();loaders=[name for name in pins['expected_runtime'] if Path(name).name=='ld-linux-x86-64.so.2']
    require(len(loaders)==1,'Expected exactly one actual loader')
    allowed=[[loaders[0],'--list',str(COMPILED/'check')],[pins['tools']['nm'],'--defined-only',str(COMPILED/'check')]]
    require(list(map(str,argv)) in allowed,'Forbidden validation command; compiler/linker/main execution is disabled')
    logs=directory/'commands';logs.mkdir(exist_ok=True);cap=min(30.,remaining(directory))
    record={'argv':list(map(str,argv)),'cwd':str(ROOT),'timeout_seconds':cap,'started_utc':now()};dump(logs/(label+'.command.json'),record)
    with (logs/(label+'.stdout.txt')).open('xb') as output:
        try:p=subprocess.run(argv,cwd=ROOT,env=environment(),stdout=output,stderr=subprocess.STDOUT,timeout=cap);record['returncode']=p.returncode
        except BaseException as error:record.update(error=repr(error),finished_utc=now());dump(logs/(label+'.terminal.json'),record);raise
    record.update(finished_utc=now(),stdout_sha256=digest(logs/(label+'.stdout.txt')));dump(logs/(label+'.terminal.json'),record)
    require(record['returncode']==0,'Validation command failed '+label)
    return (logs/(label+'.stdout.txt')).read_text()
def validate_locked(directory):
    pins=original_inputs();loader=support('loader');amended=join_record();dump(directory/'amended-prebound.json',amended)
    actual=loader.requireactualmap((COMPILED/'link.map').read_text(),ROOT,amended,COMPILED/'check')
    expected={str(Path(p).resolve()):h for p,h in pins['old_link_inputs'].items() if p!=pins['old_driver_object']}
    expected[str(COMPILED/'driver.o')]=digest(COMPILED/'driver.o');require(actual==expected,'Original actual LOAD set differs')
    runtime=loader.query_runtime(COMPILED/'check',pins['expected_runtime'],amended,lambda argv,label:command(argv,directory,label))
    symbols=command([pins['tools']['nm'],'--defined-only',str(COMPILED/'check')],directory,'symbols')
    names=re.findall(r'^[0-9a-fA-F]+\s+[A-Za-z]\s+(\S+)$',symbols,re.M)
    for name in ('cap6_tirust_tensor_diagnostic_','cap6_mp_tirust_tensor_diagnostic_','__wrap_ml_tirust_evaluate_dp','__wrap_ml_tirust_evaluate_mp','__mtirust_MOD_tensor_mp'):
        require(names.count(name)==1,'Missing/ambiguous compiled diagnostic/module/observer '+name)
    verify_source()
    dump(directory/'result.json',{'schema':'rank6-diagnostic-validation-only-result-v1','status':'passed','source_index_sha256':EXPECTED_SOURCE,
      'original_source_index_sha256':SOURCE1,'original_failed_terminal_sha256':FAILED1,'failed_build_preserved':True,'original54commands_all_zero':True,
      'actual_link_inputs':actual,'runtime':runtime,'amended_prebound_sha256':digest(directory/'amended-prebound.json'),
      'executable':str(COMPILED/'check'),'executable_sha256':digest(COMPILED/'check'),'driver_object_sha256':digest(COMPILED/'driver.o'),'link_map_sha256':digest(COMPILED/'link.map'),
      'compiler_calls':0,'linker_calls':0,'native_tensor_calls':0,'numerical_rows_executed':0,'RUN_admitted':False,
      'scope':'Metadata validation of unchanged compiled historical generated tensor diagnostic; no numerical qualification or performance claim.'})
def set_limits():
    for kind,key in ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_FSIZE,'file_bytes'),(resource.RLIMIT_CORE,'core_bytes')):
        value=LIMITS[key];_,hard=resource.getrlimit(kind);require(hard==resource.RLIM_INFINITY or hard>=value,'Inherited limit too small');resource.setrlimit(kind,(value,value))
def child(name):
    require(name=='validate','Only validation is implemented');set_limits();directory=WORK/name;admit=read(directory/'admission.json')
    require(admit['stage']==name and admit['source_index_sha256']==EXPECTED_SOURCE,'Missing exact validation admission')
    dump(directory/'limits.json',{'actual':{key:list(resource.getrlimit(kind)) for kind,key in ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_FSIZE,'file_bytes'),(resource.RLIMIT_CORE,'core_bytes'))},'pid':os.getpid()})
    verify_source()
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);dump(directory/'lock.json',{'acquired_utc':now(),'path':str(Path(lock.name))});validate_locked(directory)


def cleanup_owned(owned,end):
    budget=min(55.,max(0.,end-time.monotonic()))
    result=owned.stop(cooperative_seconds=budget*45/55,terminate_seconds=budget*5/55,kill_seconds=budget*5/55)
    require(not result['survivors'],'Owned cleanup unresolved')
    return result


class StageDeadline:
    """One cap covers preflight, execution, cleanup and administrative closure."""
    def __enter__(self):
        self.end=time.monotonic()+self.cap
        self.work_expired=False;self.handoff=False;self.deferred=False
        self.previous=signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM,self.expired)
        signal.setitimer(signal.ITIMER_REAL,self.cap-CLEANUP)
        return self
    def __init__(self,cap):self.cap=cap
    def expired(self,number,frame):
        if self.handoff:self.deferred=True;return
        if not self.work_expired:
            self.work_expired=True
            signal.setitimer(signal.ITIMER_REAL,max(.001,self.end-time.monotonic()))
            raise TimeoutError('Unit work cutoff reached; original cleanup reserve starts')
        raise TimeoutError('Original unit all-in cap reached')
    def __exit__(self,*error):
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,self.previous)


def raw_cleanup(process,record,cleanup,end):
    """Failure-only path if OwnedChild construction raised after Popen."""
    events=[]
    if record is None:
        record=cleanup.identity(process.pid)
    if record is None and process.poll() is not None:
        return {'failure_only':True,'returncode':process.returncode,
                'cleanup_scope':'Root exited before identity capture; descendant closure unqualified',
                'uncertainty_blocks_later_stages':True}
    require(record is not None and record['pid']==record['process_group']==record['session'],
            'Raw child session identity unavailable; cleanup uncertainty blocks continuation')
    # The raw handle is retained before ownership construction. Only this
    # positively matched original child session is targeted on this failure path.
    for sig,seconds in ((signal.SIGTERM,2.),(signal.SIGKILL,2.)):
        if process.poll() is not None:break
        require(cleanup.matches(record),'Raw child start identity changed')
        os.killpg(record['process_group'],sig);events.append(signal.Signals(sig).name)
        try:process.wait(timeout=min(seconds,max(.001,end-time.monotonic())))
        except subprocess.TimeoutExpired:continue
    require(process.poll() is not None,'Raw child survived bounded termination')
    return {'failure_only':True,'events':events,'root':record,
            'cleanup_scope':'Raw identified child session; no qualification pass on ownership construction failure',
            'returncode':process.poll(),'uncertainty_blocks_later_stages':True}


def stage(name):
    directory=WORK/name
    entry={'stage':name,'cap_seconds':CAPS[name],'cleanup_reserve_seconds':CLEANUP,'started_utc':now(),'status':'running'}
    owned=None;process=None;raw_record=None;cleaned=False;created=False
    with StageDeadline(CAPS[name]) as deadline:
        cleanup=support('cleanup')
        with cleanup.SignalRequests() as requests:
            entry['deadline_monotonic']=deadline.end
            try:
                WORK.mkdir(exist_ok=True);directory.mkdir(exist_ok=False);created=True
                requests.check();preflight()
                argv=[original_inputs()['tools']['taskset'],'-c',str(CPU),sys.executable,'-B',str(Path(__file__).relative_to(ROOT)),
                      '--child',name,'--source',EXPECTED_SOURCE]
                if REVIEWED_BUILD:argv+=['--reviewed-build',REVIEWED_BUILD]
                entry['command']=argv
                dump(directory/'admission.json',{'stage':name,'source_index_sha256':EXPECTED_SOURCE,'reviewed_build_sha256':REVIEWED_BUILD,'command':argv,'resource_limits':LIMITS,'deadline_monotonic':deadline.end,'work_deadline_monotonic':deadline.end-CLEANUP})
                with (directory/'console.log').open('xb') as output:
                    deadline.handoff=True
                    try:
                        process=subprocess.Popen(argv,cwd=ROOT,env=environment(),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                        raw_record=cleanup.identity(process.pid)
                        owned=cleanup.OwnedChild(process)
                    finally:
                        deadline.handoff=False
                        if deadline.deferred:deadline.expired(None,None)
                    entry['child_pid']=process.pid;dump(directory/'started.json',entry)
                    print('START '+name+' child_pid='+str(process.pid),flush=True)
                    entry['returncode']=owned.wait(requests);requests.check()
                    entry['cleanup']=cleanup_owned(owned,deadline.end);cleaned=True
                require(entry['returncode']==0,'Tensor stage failed')
                verify_source();requests.check()
                entry.update(status='passed',finished_utc=now(),result_sha256=digest(directory/'result.json'))
                dump(directory/'terminal.json',entry)
            except BaseException as error:
                if process is not None and not cleaned:
                    try:
                        entry['cleanup']=cleanup_owned(owned,deadline.end) if owned is not None else raw_cleanup(process,raw_record,cleanup,deadline.end)
                    except BaseException as cleanup_error:entry['cleanup_error']=repr(cleanup_error)
                entry.update(status='failed',error=repr(error),finished_utc=now(),publication_scope='Administrative failure closure, never a qualification pass')
                
                if created and not (directory/'terminal.json').exists():dump(directory/'terminal.json',entry)
                raise


def main():
    global EXPECTED_SOURCE
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--preflight',action='store_true');m.add_argument('--validate',action='store_true');m.add_argument('--child',choices=('validate',))
    p.add_argument('--source',required=True);a=p.parse_args();EXPECTED_SOURCE=a.source
    require(re.fullmatch('[0-9a-f]{64}',EXPECTED_SOURCE),'Exact reviewed source SHA256 required')
    if a.child:child(a.child)
    elif a.preflight:print(json.dumps(preflight(),indent=2,sort_keys=True))
    else:stage('validate')
if __name__=='__main__':main()
