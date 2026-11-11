#!/usr/bin/env python3
"""One bounded admitted standalone Rust-hook attempt; no retry or physics code."""
import fcntl,hashlib,json,os,resource,signal,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
pre=P/'preflight.json'
assert hashlib.sha256(pre.read_bytes()).hexdigest()=='a39aa7aa0821b8f6957c1c15833c75d9a864f1d5018cea2aaff0ccf84c798da5'
spec=json.loads(pre.read_text());limits=spec['limits']
def verify(pin):
 f=Path(pin['path']);assert f.stat().st_size==pin['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==pin['sha256']
for pin in [spec['source_index'],spec['source_file'],spec['extraction']['module'],spec['extraction']['source'],spec['rustc'],spec['cc']]:verify(pin)
closure={'passed':False,'native_commands_run':0,'commands':[],'source_index_sha256':spec['source_index']['sha256'],'scientific_qualification':False,'fenv_or_trap_qualification':False,'library_or_physics_build':False,'network_commands':0,'lock_acquired':False,'cleanup':[]}
lock=open(limits['mutex'],'a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 closure.update(blocked='shared mutex busy; no native command and no automatic retry',cleanup_complete=True)
 (P/'closure.json').write_text(json.dumps(closure,indent=2)+'\n');print(json.dumps(closure,indent=2));raise SystemExit(0)
closure['lock_acquired']=True;started=time.monotonic();deadline=started+limits['work_wall_seconds'];owned={}
env=os.environ.copy();env.update(RUST_TEST_THREADS='1',RUST_MIN_STACK='2097152',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CARGO_NET_OFFLINE='true')
closure['child_environment_overrides']={x:env[x] for x in ['RUST_TEST_THREADS','RUST_MIN_STACK','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','CARGO_NET_OFFLINE']}
closure['path']=env.get('PATH','');closure['limits']=limits
cpu=limits['nonexclusive_cpu']
def child_setup():
 os.sched_setaffinity(0,{cpu});resource.setrlimit(resource.RLIMIT_AS,(limits['address_space_bytes_per_process'],)*2);resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
def proc_state(pid):
 try:
  text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rindex(')')+2:].split()
  return {'pid':pid,'state':fields[0],'ppid':int(fields[1]),'pgrp':int(fields[2]),'start_ticks':int(fields[19]),'rss_bytes':int(fields[21])*os.sysconf('SC_PAGE_SIZE')}
 except (FileNotFoundError,ProcessLookupError,PermissionError):return None
def descendants(pid):
 seen={};todo=[pid]
 while todo:
  q=todo.pop();s=proc_state(q)
  if not s or q in seen:continue
  seen[q]=s
  try:todo.extend(int(x) for x in Path(f'/proc/{q}/task/{q}/children').read_text().split())
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return seen
def group_exists(pgid):
 try:os.killpg(pgid,0);return True
 except ProcessLookupError:return False
try:
 for item in spec['commands']:
  assert closure['native_commands_run']<4
  if time.monotonic()>=deadline:raise RuntimeError('60-second work deadline before next command')
  name=item['name'];log=P/'logs'/(name+'.txt');t=time.monotonic()
  record={'name':name,'argv':item['argv'],'cpu':cpu,'start_offset_seconds':t-started,'peak_sampled_aggregate_rss_bytes':0,'process_samples':[]}
  with log.open('wb') as out:
   proc=subprocess.Popen(item['argv'],cwd=P,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=child_setup)
   closure['native_commands_run']+=1;owned[proc.pid]=proc;record['pid']=proc.pid
   failure=None
   while proc.poll() is None:
    states=descendants(proc.pid);record['process_samples'].append(list(states.values()))
    rss=sum(s['rss_bytes'] for s in states.values());record['peak_sampled_aggregate_rss_bytes']=max(rss,record['peak_sampled_aggregate_rss_bytes'])
    if rss>limits['aggregate_rss_sampled_cap_bytes']:failure='aggregate RSS sampled cap exceeded';break
    if time.monotonic()>=deadline:failure='60-second work deadline reached';break
    time.sleep(.02)
   if failure:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
   else:proc.wait()
  record.update(returncode=proc.returncode,elapsed_seconds=time.monotonic()-t,failure=failure,log=str(log.relative_to(P)),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
  (P/'logs'/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');closure['commands'].append({k:v for k,v in record.items() if k!='process_samples'})
  if failure or proc.returncode!=0:raise RuntimeError(f'{name}: '+str(failure or f'exit {proc.returncode}'))
  if name.startswith('compile-'):
   elf=Path(item['argv'][item['argv'].index('-o')+1]);assert elf.read_bytes()[:4]==b'\x7fELF'
   closure.setdefault('elfs',[]).append({'path':str(elf.relative_to(P)),'bytes':elf.stat().st_size,'sha256':hashlib.sha256(elf.read_bytes()).hexdigest()})
  if name=='run-enabled':
   text=log.read_text();assert '2 passed; 0 failed' in text and 'running 2 tests' in text
  if name=='run-disabled':
   text=log.read_text();assert '0 passed; 0 failed' in text and 'running 0 tests' in text
 closure['passed']=True
except Exception as e:closure['failure']=repr(e)
finally:
 cleanup_deadline=time.monotonic()+30
 for pgid,proc in owned.items():
  before=group_exists(pgid)
  if before:
   try:os.killpg(pgid,signal.SIGTERM)
   except ProcessLookupError:pass
   grace_deadline=min(cleanup_deadline,time.monotonic()+.2)
   while group_exists(pgid) and time.monotonic()<grace_deadline:time.sleep(.01)
   if group_exists(pgid):
    try:os.killpg(pgid,signal.SIGKILL)
    except ProcessLookupError:pass
  try:proc.wait(timeout=max(.01,cleanup_deadline-time.monotonic()))
  except subprocess.TimeoutExpired:closure['passed']=False
  after=group_exists(pgid);closure['cleanup'].append({'pgid':pgid,'existed_at_cleanup':before,'exists_after_cleanup':after,'parent_reaped':proc.returncode is not None})
  if after:closure['passed']=False
 closure['cleanup_complete']=all(not x['exists_after_cleanup'] and x['parent_reaped'] for x in closure['cleanup'])
 closure['elapsed_work_and_cleanup_seconds']=time.monotonic()-started
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close();closure['lock_released']=True
 (P/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
print(json.dumps({k:v for k,v in closure.items() if k not in ['commands','path']},indent=2))
