"""Owned, all-inclusive3600s supervisor; invoke only after the coordinating numerical GO."""
from pathlib import Path
import hashlib, importlib.util, json, os, subprocess, sys, time
ROOT=Path.cwd();HERE=ROOT/'.tirust-build/zggg-five-backend-timing-1'
assert ROOT==Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def dump(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
cleanup=module(ROOT/'docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py','zggg_super_cleanup')
ownership=module(ROOT/'.tirust-build/generic-owned-snapshot-qualification-proposal-2/ownership.py','zggg_super_ownership')
start=time.monotonic();cap=3600.;child=owned=raw=None
record={'status':'running','cap_seconds':cap,'cleanup_reserve_seconds':60,'cwd':str(ROOT),'controller_sha256':sha(HERE/'numerical.py'),'cpu_reservation':False,'performance_acceptance':False}
with ownership.StageDeadline(cap,absolute_end=start+cap) as deadline,cleanup.SignalRequests() as requests:
 try:
  env=os.environ.copy();env['ZGGG_ABSOLUTE_END']=repr(deadline.end)
  with (HERE/'numerical-console.log').open('xb') as output:
   deadline.handoff=True
   try:
    child=subprocess.Popen([sys.executable,'-B',str(HERE/'numerical.py')],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True);raw=cleanup.identity(child.pid);owned=cleanup.OwnedChild(child)
   finally:deadline.handoff=False
   dump(HERE/'numerical-started.json',{'pid':child.pid,'identity':owned.root,'controller_sha256':record['controller_sha256']});print('START requested-zggg numerical pid='+str(child.pid),flush=True)
   record['returncode']=owned.wait(requests);record['cleanup']=ownership.cleanup_owned(owned,deadline.end)
  if record['returncode']:raise RuntimeError('Numerical action failed; all partial evidence retained')
  observed=json.loads((HERE/'measurements/observations.json').read_text());record['physics_regression_pass']=observed.get('physics_regression_pass',False);record['status']='observations-collected';record['performance_acceptance']=False
 except BaseException as error:
  record.update(status='failed',error=repr(error))
  if owned is not None and 'cleanup' not in record:record['cleanup']=ownership.cleanup_owned(owned,deadline.end)
  elif child is not None and owned is None:record['cleanup']=ownership.raw_cleanup(child,raw,cleanup,deadline.end)
  raise
 finally:
  record.update(elapsed_seconds=time.monotonic()-start,console_sha256=sha(HERE/'numerical-console.log') if (HERE/'numerical-console.log').exists() else None)
  dump(HERE/'numerical-terminal.json',record)
print('CLOSED requested-zggg numerical status='+record['status'],flush=True)
