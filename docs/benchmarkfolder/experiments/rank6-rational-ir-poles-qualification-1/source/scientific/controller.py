"""Two small diagnostic fixtures. SOURCE first; BUILD/RUN separately admitted."""
import argparse,fcntl,hashlib,importlib.util,json,os,re,resource,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path.cwd()
HERE=Path(__file__).resolve().parent
WORK=ROOT/'.tirust-build/rank6-ir-triangle-work-1'
CPU=12
CAPS={'build':180.,'run':180.}
CLEANUP=60.
LIMITS={'address_space_bytes':4294967296,'file_bytes':536870912,'core_bytes':0}
EXPECTED_SOURCE=None
REVIEWED_BUILD=None
SUPPORT={
 'cleanup':('docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py','240d1067a7b7098f056f0f367097def8f374d7979c3001d249c8b1f7cb0bbd84'),
 'loader':('docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-3/loader_checks.py','4b8d035797b0ad742a0b0903228a20cc4bddca922146c4c5ea02ad4f3a011e55'),
 'capture':('docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py','ed25a9fc96e2faf2b545a9bf70a91b34315c01e534a6ebe68d7520ec2171aa42'),
}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def require(x,m):
    if not x:raise ValueError(m)
def load(name,path,checksum):
    require(digest(path)==checksum,'Changed helper '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
def support(name):
    path,pin=SUPPORT[name]
    if name!='capture':return load('ir_tensor_'+name,ROOT/path,pin)
    require(digest(ROOT/path)==pin,'Changed native capture helper')
    source=(ROOT/path).read_text()
    require(source.count("affinity!=['11']")==source.count('differs from CPU11.')==1,'Capture adaptation anchors')
    source=source.replace("affinity!=['11']","affinity!=['12']").replace('differs from CPU11.','differs from CPU12.')
    ns={'__name__':'ir_tensor_capture','__file__':str(ROOT/path)};exec(compile(source,str(ROOT/path),'exec'),ns)
    import types
    return types.SimpleNamespace(**ns)
def check_files(base,files):
    for name,w in files.items():
        rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe indexed path')
        q=base/rel;require(q.is_file() and not q.is_symlink(),'Missing indexed file')
        require(q.stat().st_size==w['bytes'] and digest(q)==w['sha256'],'Indexed source differs: '+name)
def protected():
    pins=read(HERE/'input-pins.json')
    for name,pin in pins['files'].items():require(digest(name)==pin,'Protected dependency differs: '+name)
    return {'input_pins_sha256':digest(HERE/'input-pins.json'),'files_checked':len(pins['files'])}
def verify_source():
    require(EXPECTED_SOURCE and digest(HERE/'source-index.json')==EXPECTED_SOURCE,'Exact reviewed source hash required')
    index=read(HERE/'source-index.json');check_files(HERE,index['files'])
    require({str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}==set(index['files'])|{'source-index.json'},'Source inventory differs')
    for _,(path,pin) in SUPPORT.items():require(digest(ROOT/path)==pin,'Support source differs')
    return protected()
def source_module(name):
    index=read(HERE/'source-index.json')
    return load('ir_tensor_'+name,HERE/(name+'.py'),index['files'][name+'.py']['sha256'])
def environment():
    pins=read(HERE/'input-pins.json');env=os.environ.copy()
    for key in list(env):
        if key.startswith(('LD_','NIX_')) or key in ('GLIBC_TUNABLES','GCONV_PATH','BASH_ENV','ENV'):env.pop(key,None)
    for key,value in pins['compiler_environment'].items():
        if value is None:env.pop(key,None)
        else:env[key]=value
    env.update(pins['runtime_environment'])
    env.update(PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',LD_PRELOAD='',
      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1',MKL_NUM_THREADS='1',
      LC_ALL='C',LANG='C',TZ='UTC',RUST_MIN_STACK='134217728',SYMBOLICA_HIDE_BANNER='1',
      SYMBOLICA_LOCK_DIR=str(ROOT/'.tirust-build/symbolica-locks'),GFORTRAN_ERROR_BACKTRACE='0')
    return env
def preflight():
    before=verify_source();require(CPU in os.sched_getaffinity(0),'CPU12 unavailable')
    rows=read(HERE/'schedule.json');require(len(rows)==24 and [r['action'] for r in rows]==list(range(4101,4125)),'Schedule coverage')
    keys=('action','fixture','backend','ct','mode','finite','cache','budget','clear','n','rank')
    require((HERE/'schedule.dat').read_text()=='24\n'+''.join(' '.join(str(r[k]) for k in keys)+'\n' for r in rows),'Schedule text differs')
    require({(r['fixture'],r['ct'],r['mode'],r['finite']) for r in rows}=={(f,c,m,t) for f in (110,111) for c in (1,4) for m in range(3) for t in (0,1)},'Matrix differs')
    require(all(r['backend']==9 and r['n']==(3 if r['fixture']==110 else 2) and r['rank']==6 and r['cache']==0 and r['clear']==1 and r['budget']==268435456 for r in rows),'Changed numerical policy')
    sys.path.insert(0,str(HERE))
    from oracle import monomials,triangle_poles,scaleless_bubble_poles
    mapping=ROOT/'docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/materialized/ew/monomials.json'
    require(read(mapping)==[list(a) for a in monomials(6)],'Historical Cartesian mapping ordering differs')
    for name,fn in [('future-triangle-fixture.json',triangle_poles),('future-scaleless-fixture.json',scaleless_bubble_poles)]:
        fixture=read(HERE/name)
        require(len(fixture['shifts'])==len(fixture['external_pl'])==len(fixture['masses_squared'])==fixture['n'],'Fixture dimensions')
        expected=[{'index':i,'powers':list(a),'UV1':p[0].json(),'IR1':p[1].json(),'IR2':p[2].json()} for i,a in enumerate(monomials(6)) for p in [fn(a)]]
        require(fixture['components']==expected,'Exact rational fixture differs')
        require(all(sum(q[mu] for q in fixture['external_pl'])==0 for mu in range(4)),'Corners do not close exactly')
        n=fixture['n'];q=fixture['shifts']
        require(fixture['external_pl']==[[q[(i+1)%n][mu]-q[i][mu] for mu in range(4)] for i in range(n)],'Corner/routing transformation differs')
        require(fixture['invariants']==[[sum((x-y)**2*(1 if mu==0 else -1) for mu,(x,y) in enumerate(zip(u,v))) for v in q] for u in q],'Exact invariants differ')
    return {'status':'source-preflight-passed','source_index_sha256':EXPECTED_SOURCE,'protected':before,'rows':24,'native_calls':0,'compiler_calls':0,'caps_seconds':CAPS,'limits':LIMITS}
def work_remaining(directory):
    end=read(directory/'admission.json')['work_deadline_monotonic']
    left=end-time.monotonic();require(left>0,'Original work deadline expired');return left
def command(argv,cwd,directory,label,timeout=30.):
    logs=directory/'commands';logs.mkdir(exist_ok=True)
    cap=min(timeout,work_remaining(directory));record={'argv':list(map(str,argv)),'cwd':str(cwd),'started_utc':now(),'timeout_seconds':cap}
    dump(logs/(label+'.command.json'),record)
    with (logs/(label+'.stdout.txt')).open('xb') as out:
        try:
            p=subprocess.run(argv,cwd=cwd,env=environment(),stdout=out,stderr=subprocess.STDOUT,timeout=cap);record['returncode']=p.returncode
        except BaseException as e:
            record['failure']=repr(e);record['finished_utc']=now();dump(logs/(label+'.terminal.json'),record);raise
    record.update(finished_utc=now(),stdout_sha256=digest(logs/(label+'.stdout.txt')));dump(logs/(label+'.terminal.json'),record)
    require(record['returncode']==0,'Command failed '+label)
    return (logs/(label+'.stdout.txt')).read_text()
def build_locked(directory):
    pins=read(HERE/'input-pins.json');loader=support('loader');commands=pins['build_commands']
    require('15.2.0' in command([pins['tools']['gfortran'],'--version'],ROOT,directory,'compiler-version'),'Compiler version differs')
    command(commands[0],ROOT,directory,'driver-compile')
    callback=lambda argv,label:command(argv,ROOT,directory,label)
    prebound=loader.prelink(commands[1],ROOT,environment(),callback);dump(directory/'prelink.json',prebound)
    command(commands[1],ROOT,directory,'link')
    loads=loader.requireactualmap((directory/'link.map').read_text(),ROOT,prebound,directory/'check')
    expected={str(Path(x).resolve()):h for x,h in pins['old_link_inputs'].items() if x!=pins['old_driver_object']}
    expected[str(directory/'driver.o')]=digest(directory/'driver.o')
    require(loads==expected,'Actual old/new link input set differs beyond the private driver')
    runtime=loader.query_runtime(directory/'check',pins['expected_runtime'],prebound,callback)
    symbols=command([pins['tools']['nm'],'--defined-only',str(directory/'check')],ROOT,directory,'symbols')
    names=re.findall(r'^[0-9a-fA-F]+\s+[A-Za-z]\s+(\S+)$',symbols,re.M)
    require(names.count('cap6_tirust_tensor_diagnostic_')==1 and names.count('cap6_mp_tirust_tensor_diagnostic_')==1,'Ambiguous/missing generated diagnostics')
    require(names.count('__wrap_ml_tirust_evaluate_mp')==1 and names.count('__wrap_ml_tirust_evaluate_dp')==1,'Observer missing or duplicated')
    protected()
    result={'status':'passed','source_index_sha256':EXPECTED_SOURCE,'scope':'One private driver compile and relink of historical generated rank6 diagnostic; no main executed.',
      'driver_sha256':digest(HERE/'driver.f'),'actual_link_inputs':loads,'runtime':runtime,'prelink_sha256':digest(directory/'prelink.json'),
      'objects':{'driver.o':digest(directory/'driver.o')},'executable_sha256':digest(directory/'check'),'link_map_sha256':digest(directory/'link.map'),
      'native_calls':0,'physical_amplitude_calls':0,'performance_acceptance':False}
    dump(directory/'result.json',result)
def verify_build():
    directory=WORK/'build';require(REVIEWED_BUILD and digest(directory/'terminal.json')==REVIEWED_BUILD,'Separate reviewed BUILD terminal required')
    closed=read(directory/'terminal.json');require(closed['status']=='passed' and closed['returncode']==0 and not closed['cleanup']['survivors'],'BUILD not closed')
    result=read(directory/'result.json');require(digest(directory/'result.json')==closed['result_sha256'] and result['source_index_sha256']==EXPECTED_SOURCE,'BUILD source/result mismatch')
    require(digest(directory/'check')==result['executable_sha256'] and digest(directory/'link.map')==result['link_map_sha256'] and digest(directory/'prelink.json')==result['prelink_sha256'],'BUILD artifact changed')
    for name,pin in {**result['actual_link_inputs'],**result['runtime']['files']}.items():require(digest(name)==pin,'Admitted link/runtime input changed')
    return result
def run_locked(directory):
    built=verify_build();pins=read(HERE/'input-pins.json');capture=support('capture');loader=support('loader')
    for name,origin in pins['runtime_inputs'].items():
        dest=directory/name;dest.parent.mkdir(parents=True,exist_ok=True)
        require(not dest.exists(),'Existing private runtime input');dest.write_bytes(Path(origin).read_bytes())
    require(digest(directory/'schedule.dat')==digest(HERE/'schedule.dat'),'Schedule copy mismatch')
    callback=lambda argv,label:command(argv,ROOT,directory,label)
    runtime=loader.query_runtime(WORK/'build/check',pins['expected_runtime'],read(WORK/'build/prelink.json'),callback)
    require(runtime['files']==built['runtime']['files'],'Runtime changed after BUILD')
    progress={};record={'source_index_sha256':EXPECTED_SOURCE,'reviewed_build_sha256':REVIEWED_BUILD,'runtime':runtime,'native_progress':progress,
      'argv':[str(WORK/'build/check')],'scope':'24 standalone tensor requests; no physical amplitude/stress/timing claim'}
    dump(directory/'native-start.json',record)
    seconds=min(90.,work_remaining(directory)-15.);require(seconds>0,'No bounded native budget left')
    try:
        with (directory/'native-console.log').open('xb') as out:
            code,mapping=capture.run_captured(record['argv'],out,cwd=directory,env=environment(),expected_runtime=runtime['files'],executable=str(WORK/'build/check'),
              selected_library=pins['selected_library'],maps_path=directory/'maps.txt',progress=progress,seconds=seconds)
        record.update(returncode=code,mapping=mapping);require(code==0,'Native tensor worker failed')
        analyzer=source_module('analyze');result=analyzer.analyze(directory);dump(directory/'analysis.json',result)
        record['analysis_sha256']=digest(directory/'analysis.json');require(result['status']=='passed','Tensor numerical/IEEE qualification failed')
        for name,origin in pins['runtime_inputs'].items():require(digest(directory/name)==digest(origin),'Runtime input changed')
        verify_build();protected();record['status']='passed';dump(directory/'result.json',record)
    except BaseException as e:
        record.update(status='failed',error=repr(e));raise
    finally:dump(directory/'native-terminal.json',record)
def set_limits():
    for kind,key in ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_FSIZE,'file_bytes'),(resource.RLIMIT_CORE,'core_bytes')):
        value=LIMITS[key];_,hard=resource.getrlimit(kind);require(hard==resource.RLIM_INFINITY or hard>=value,'Inherited limit too small');resource.setrlimit(kind,(value,value))
def child(name):
    set_limits();directory=WORK/name;admit=read(directory/'admission.json')
    require(admit['source_index_sha256']==EXPECTED_SOURCE and admit['stage']==name and admit['reviewed_build_sha256']==REVIEWED_BUILD,'Exact stage admission missing')
    dump(directory/'limits.json',{'actual':{key:list(resource.getrlimit(kind)) for kind,key in ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_FSIZE,'file_bytes'),(resource.RLIMIT_CORE,'core_bytes'))},'pid':os.getpid(),'session':os.getsid(0),'process_group':os.getpgrp()})
    verify_source()
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);dump(directory/'lock.json',{'acquired_utc':now(),'path':str(Path(lock.name))})
        if name=='build':build_locked(directory)
        else:run_locked(directory)


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
                if name=='run':verify_build()
                argv=[read(HERE/'input-pins.json')['tools']['taskset'],'-c',str(CPU),sys.executable,'-B',str(Path(__file__).relative_to(ROOT)),
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
    global EXPECTED_SOURCE,REVIEWED_BUILD
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--build',action='store_true');mode.add_argument('--run',action='store_true')
    mode.add_argument('--child',choices=('build','run'));parser.add_argument('--source',required=True);parser.add_argument('--reviewed-build')
    args=parser.parse_args();EXPECTED_SOURCE=args.source;REVIEWED_BUILD=args.reviewed_build
    require(re.fullmatch('[0-9a-f]{64}',EXPECTED_SOURCE),'Exact source SHA256 required')
    if args.run or args.child=='run':require(REVIEWED_BUILD and re.fullmatch('[0-9a-f]{64}',REVIEWED_BUILD),'Separate reviewed BUILD terminal SHA required')
    if args.child:child(args.child);return
    if args.preflight:print(json.dumps(preflight(),sort_keys=True,indent=2));return
    stage('build' if args.build else 'run')

if __name__=='__main__':main()
