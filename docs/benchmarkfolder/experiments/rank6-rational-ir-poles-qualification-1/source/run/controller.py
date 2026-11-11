"""RUN-only glue for the exact compiled and separately validated tensor driver."""
import argparse,ast,fcntl,hashlib,importlib.util,json,os,re,resource,signal,subprocess,sys,time,types
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
WORK=ROOT/'.tirust-build/rank6-ir-triangle-run-work-1'
SCI=ROOT/'.tirust-build/rank6-ir-triangle-controller-1'
VALIDATOR=ROOT/'.tirust-build/rank6-ir-triangle-validation-controller-1'
VALIDATED=ROOT/'.tirust-build/rank6-ir-triangle-validation-work-1/validate'
COMPILED=ROOT/'.tirust-build/rank6-ir-triangle-work-1/build'
SCI_SHA='4f496ebf3071e8d405e1e2e72b4bcff3ab7b210ef2338740cdbe314b0cfcb2b0'
VALIDATOR_SHA='73f311c041c176fe80c00b53f51935d8283521672717247ee73d26c855ee0640'
VALIDATION_SHA='9c6d48c1db7e725db8a7579d9d6dc1ea623c5f22f4603232ebf3540042f4715c'
VALIDATION_RESULT_SHA='d34d15d59ed16cdab5927fd219146979f0bd77d567dcd3ab831d547ff121dc79'
AMENDED_SHA='af27f81ff8e853595ad83e7e9be2a96a6d8df974486fab3d2617b89e0ff97d70'
CPU=12;CAPS={'run':180.};CLEANUP=60.
LIMITS={'address_space_bytes':4294967296,'file_bytes':536870912,'core_bytes':0}
EXPECTED_SOURCE=None;REVIEWED_VALIDATION=None

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

def scientific():
    require(digest(SCI/'source-index.json')==SCI_SHA,'Scientific source seal differs')
    index=read(SCI/'source-index.json');obj=module('ir_run_scientific',SCI/'controller.py',index['files']['controller.py']['sha256'])
    obj.EXPECTED_SOURCE=SCI_SHA;obj.REVIEWED_BUILD=None
    return obj

def validator():
    require(digest(VALIDATOR/'source-index.json')==VALIDATOR_SHA,'Validator source differs')
    index=read(VALIDATOR/'source-index.json');obj=module('ir_run_validator',VALIDATOR/'controller.py',index['files']['controller.py']['sha256'])
    obj.EXPECTED_SOURCE=VALIDATOR_SHA
    return obj

def original_inputs():return read(SCI/'input-pins.json')
def support(name):return scientific().support(name)
def environment():return scientific().environment()
def check_files(base,files):
    for name,w in files.items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe indexed path')
        q=base/p;require(q.is_file() and not q.is_symlink() and q.stat().st_size==w['bytes'] and digest(q)==w['sha256'],'Indexed file differs '+name)
def verify_source():
    require(EXPECTED_SOURCE and digest(HERE/'source-index.json')==EXPECTED_SOURCE,'Exact reviewed RUN source SHA256 required')
    index=read(HERE/'source-index.json');check_files(HERE,index['files'])
    require({str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}==set(index['files'])|{'source-index.json'},'RUN source inventory differs')
    pins=read(HERE/'input-pins.json')
    for path,w in pins['protected_files'].items():
        require(Path(path).is_file() and Path(path).stat().st_size==w['bytes'] and digest(path)==w['sha256'],'Protected scientific/compiled/validation input differs '+path)
    for base,key in ((VALIDATED,'validated_files'),(COMPILED,'failed_build_files')):
        require({str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}==set(pins[key]),'Ancestor output inventory differs')
    validator().verify_source()
    return {'protected_files_checked':len(pins['protected_files']),'scientific_source_sha256':SCI_SHA,'validation_source_sha256':VALIDATOR_SHA}
def verify_validated():
    require(REVIEWED_VALIDATION==VALIDATION_SHA,'Exact separately reviewed successful validation required')
    require(digest(VALIDATED/'terminal.json')==VALIDATION_SHA and digest(VALIDATED/'result.json')==VALIDATION_RESULT_SHA,'Validation closure differs')
    terminal=read(VALIDATED/'terminal.json');result=read(VALIDATED/'result.json')
    require(terminal['status']=='passed' and terminal['returncode']==0 and not terminal['cleanup']['survivors'],'Validation is not closed successfully')
    require(terminal['result_sha256']==VALIDATION_RESULT_SHA and result['source_index_sha256']==VALIDATOR_SHA,'Validation source/result join differs')
    require(digest(VALIDATED/'amended-prebound.json')==AMENDED_SHA and result['amended_prebound_sha256']==AMENDED_SHA,'Amended provenance record differs')
    require(result['runtime']['files']==original_inputs()['expected_runtime'],'Validation runtime differs from scientific inputs')
    require(result['compiler_calls']==result['linker_calls']==result['native_tensor_calls']==result['numerical_rows_executed']==0,'Unexpected validation execution')
    require(result['executable']==str(COMPILED/'check') and digest(COMPILED/'check')==result['executable_sha256'],'Compiled ELF differs')
    require(digest(COMPILED/'driver.o')==result['driver_object_sha256'] and digest(COMPILED/'link.map')==result['link_map_sha256'],'Compiled object/map differs')
    for p,h in {**result['actual_link_inputs'],**result['runtime']['files']}.items():require(digest(p)==h,'Validated actual LOAD/provider changed')
    return result

def scientific_run_namespace(directory):
    """Reuse unchanged numerical body with explicit build/provenance gates."""
    origin=scientific();source=(SCI/'controller.py').read_text();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_locked')
    body=ast.get_source_segment(source,node)
    before="'reviewed_build_sha256':REVIEWED_BUILD";after="'reviewed_validation_sha256':REVIEWED_VALIDATION"
    require(body.count(before)==1,'Numerical RUN owner field anchor differs');body=body.replace(before,after)
    require(digest(COMPILED/'prelink.json')==read(VALIDATED/'amended-prebound.json')['original_helper_prelink_sha256'],'Original helper record differs')
    bound=read(VALIDATED/'amended-prebound.json');plain_loader=origin.support('loader')
    def joined_runtime(exe,expected,old_record,callback):
        verify_validated()
        require(Path(exe)==COMPILED/'check' and expected==original_inputs()['expected_runtime'],'Unreviewed runtime query')
        require(old_record==read(COMPILED/'prelink.json'),'Original helper-map argument differs')
        return plain_loader.query_runtime(exe,expected,bound,callback)
    def curated_support(name):
        if name=='loader':return types.SimpleNamespace(query_runtime=joined_runtime)
        require(name=='capture','Unsupported numerical helper');return origin.support(name)
    def loader_command(argv,cwd,output,label,timeout=30.):
        loaders=[p for p in original_inputs()['expected_runtime'] if Path(p).name=='ld-linux-x86-64.so.2']
        require(len(loaders)==1 and list(map(str,argv))==[loaders[0],'--list',str(COMPILED/'check')],'Only the exact pre-main loader query is allowed')
        require(Path(cwd)==ROOT and Path(output)==directory,'Unreviewed metadata working directory')
        return origin.command(argv,cwd,output,label,timeout=timeout)
    ns=dict(origin.__dict__)
    ns.update(EXPECTED_SOURCE=EXPECTED_SOURCE,REVIEWED_VALIDATION=REVIEWED_VALIDATION,
      verify_build=verify_validated,support=curated_support,command=loader_command,protected=verify_source)
    # The original body executes the exact old ELF under its old WORK prefix;
    # only its output directory argument belongs to this RUN-only owner.
    exec(compile(body,str(SCI/'controller.py')+'::run_locked-reviewed-glue','exec'),ns)
    return ns

def preflight():
    before=verify_source();verify_validated();require(CPU in os.sched_getaffinity(0),'CPU12 unavailable')
    scientific().preflight()
    return {'status':'RUN-source-preflight-passed','source_index_sha256':EXPECTED_SOURCE,'reviewed_validation_sha256':REVIEWED_VALIDATION,'protected':before,'rows_pending':24,'compiler_calls':0,'loader_queries':0,'native_tensor_calls':0,'caps_seconds':CAPS}
def run_locked(directory):
    verify_validated();sys.path.insert(0,str(SCI))
    ns=scientific_run_namespace(directory);ns['run_locked'](directory)
    result=read(directory/'result.json')
    require(result['status']=='passed' and result['source_index_sha256']==EXPECTED_SOURCE and result['reviewed_validation_sha256']==REVIEWED_VALIDATION,'RUN result owner/validation join differs')
    dump(directory/'owner-closure.json',{'owner_source_sha256':EXPECTED_SOURCE,'scientific_source_sha256':SCI_SHA,'validation_source_sha256':VALIDATOR_SHA,'validation_terminal_sha256':REVIEWED_VALIDATION,
      'original_failed_BUILD_preserved':True,'numerical_body_change':'One provenance field rename only; validated-build and amended-runtime callbacks supplied by this reviewed glue.',
      'result_sha256':digest(directory/'result.json'),'scope':'Two standalone historical generated rank6 tensor fixtures; no physical amplitude/stress/performance claim.'})
def set_limits():
    for kind,key in ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_FSIZE,'file_bytes'),(resource.RLIMIT_CORE,'core_bytes')):
        value=LIMITS[key];_,hard=resource.getrlimit(kind);require(hard==resource.RLIM_INFINITY or hard>=value,'Inherited limit too small');resource.setrlimit(kind,(value,value))
def child(name):
    require(name=='run','Only numerical RUN is implemented');set_limits();directory=WORK/name;admit=read(directory/'admission.json')
    require(admit['stage']==name and admit['source_index_sha256']==EXPECTED_SOURCE and admit['reviewed_validation_sha256']==REVIEWED_VALIDATION,'Exact RUN source/validation admission missing')
    dump(directory/'limits.json',{'actual':{key:list(resource.getrlimit(kind)) for kind,key in ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_FSIZE,'file_bytes'),(resource.RLIMIT_CORE,'core_bytes'))},'pid':os.getpid()})
    verify_source();verify_validated()
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);dump(directory/'lock.json',{'acquired_utc':now(),'path':str(Path(lock.name))});run_locked(directory)


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
                if REVIEWED_VALIDATION:argv+=['--reviewed-validation',REVIEWED_VALIDATION]
                entry['command']=argv
                dump(directory/'admission.json',{'stage':name,'source_index_sha256':EXPECTED_SOURCE,'reviewed_validation_sha256':REVIEWED_VALIDATION,'command':argv,'resource_limits':LIMITS,'deadline_monotonic':deadline.end,'work_deadline_monotonic':deadline.end-CLEANUP})
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
    global EXPECTED_SOURCE,REVIEWED_VALIDATION
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--preflight',action='store_true');m.add_argument('--run',action='store_true');m.add_argument('--child',choices=('run',))
    p.add_argument('--source',required=True);p.add_argument('--reviewed-validation',required=True);a=p.parse_args()
    EXPECTED_SOURCE=a.source;REVIEWED_VALIDATION=a.reviewed_validation
    require(re.fullmatch('[0-9a-f]{64}',EXPECTED_SOURCE),'Exact RUN source SHA256 required')
    require(REVIEWED_VALIDATION==VALIDATION_SHA,'Exact reviewed validation SHA256 required')
    if a.child:child(a.child)
    elif a.preflight:print(json.dumps(preflight(),indent=2,sort_keys=True))
    else:stage('run')
if __name__=='__main__':main()
