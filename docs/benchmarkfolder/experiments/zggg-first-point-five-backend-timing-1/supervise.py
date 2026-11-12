"""Existing owned-child/deadline protocol; build300s or one-point run600s."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,resource,subprocess,sys,time
ROOT=Path.cwd();HERE=ROOT/'.tirust-build/zggg-first-point-five-backend-timing-1'
assert ROOT==Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def dump(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('build','run'));parser.add_argument('arguments',nargs=argparse.REMAINDER);args=parser.parse_args()
cleanup=module(ROOT/'docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py','zggg_one_cleanup')
ownership=module(ROOT/'.tirust-build/generic-owned-snapshot-qualification-proposal-2/ownership.py','zggg_one_ownership')
for kind,value in ((resource.RLIMIT_AS,32*1024**3),(resource.RLIMIT_FSIZE,2*1024**3),(resource.RLIMIT_CORE,0)):resource.setrlimit(kind,(value,value))
start=time.monotonic();cap=300. if args.stage=='build' else 600.;child=owned=raw=None
folder=HERE/('supervision-'+args.stage);folder.mkdir(exist_ok=False)
record={'stage':args.stage,'status':'running','cap_seconds':cap,'cleanup_reserve_seconds':60,'controller_sha256':sha(HERE/'pilot.py'),'scope':'one explicitly diagnosed point; no process-average/main-campaign acceptance','cpu_reserved':False,'performance_acceptance':False}
lock=(ROOT/'.tirust-build/numerical-tests.lock').open('a')
try:
 if args.stage=='run':fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with ownership.StageDeadline(cap,absolute_end=start+cap) as deadline,cleanup.SignalRequests() as requests:
  try:
   env=os.environ.copy();env.update(ZGGG_POINT_ABSOLUTE_END=repr(deadline.end),ZGGG_POINT_LOCK_OWNER_PID=str(os.getpid()))
   arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
   command=[sys.executable,'-B',str(HERE/'pilot.py'),args.stage,*arguments]
   dump(folder/'admission.json',{**record,'command':command,'cwd':str(ROOT),'native_authorization':'coordinator GO required before invoking build/run'})
   with (folder/'output.log').open('xb') as output:
    deadline.handoff=True
    try:
     child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True);raw=cleanup.identity(child.pid);owned=cleanup.OwnedChild(child)
    finally:deadline.handoff=False
    dump(folder/'started.json',{'pid':child.pid,'identity':owned.root});print('START '+args.stage+' pid='+str(child.pid),flush=True)
    record['returncode']=owned.wait(requests);record['cleanup']=ownership.cleanup_owned(owned,deadline.end)
   if record['returncode']:raise RuntimeError('Stage failed; partial evidence retained')
   record['status']='build-only-complete' if args.stage=='build' else 'observations-collected'
   if args.stage=='run':record['physics_regression_pass']=json.loads((HERE/'observations/result.json').read_text()).get('physics_regression_pass',False)
  except BaseException as error:
   record.update(status='failed',error=repr(error))
   if owned is not None and 'cleanup' not in record:record['cleanup']=ownership.cleanup_owned(owned,deadline.end)
   elif child is not None and owned is None:record['cleanup']=ownership.raw_cleanup(child,raw,cleanup,deadline.end)
   raise
  finally:
   record.update(elapsed_seconds=time.monotonic()-start,log_sha256=sha(folder/'output.log') if (folder/'output.log').exists() else None);dump(folder/'terminal.json',record)
finally:
 lock.close()
print('CLOSED '+args.stage+' '+record['status'],flush=True)
