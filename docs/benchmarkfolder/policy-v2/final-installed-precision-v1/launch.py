import datetime,json,os
from pathlib import Path
import subprocess,sys,time
root=Path.cwd();base=root/'docs/benchmarkfolder/policy-v2/final-installed-precision-v1'
if os.getpgrp()!=os.getpid(): os.setsid()
if os.sched_getaffinity(0)!={190}: raise RuntimeError('CPU190 required')
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def identity(pid):
 stat=(Path('/proc')/str(pid)/'stat').read_text()
 return {'pid':pid,'pgid':os.getpgid(pid),'start_ticks':stat[stat.rfind(')')+2:].split()[19]}
def write(path,value):
 with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
launch={'schema':'final-installed-precision-launch-v1','started_utc':stamp(),'controller':identity(os.getpid()),'parent_pid':os.getppid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'root_mp_handoff':True,'revision':'b0ea105a04afa34304d437d2956e4d4672360670','runtime_identity':'8b454519cb6818ef1fafe37378930c276f7e85394a2333ebc9fdb00e75726cff','expected_public_calls':64}
write(base/'launch.json',launch)
print(json.dumps(launch),flush=True)
env=os.environ.copy();env.update(TMPDIR=str(root/'.ninjars-build/tmp'),SYMBOLICA_LOCK_DIR=str(root/'.ninjars-build/symbolica-locks'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1')
tool='tools/ninjars_precision_v2.py';spec=base/'precision-spec.json';run=base/'run-v1'
commands=[('prepare',[sys.executable,'-B',tool,'prepare',str(spec)]),('execute',[sys.executable,'-B',tool,'execute',str(run/'prepared.json'),'--cpu','190','--execute-numerical']),('validate',[sys.executable,'-B',tool,'validate',str(run/'result.json')])]
for stage,command in commands:
 start=time.monotonic()
 with (base/(stage+'.stdout.log')).open('x') as out,(base/(stage+'.stderr.log')).open('x') as err:
  process=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,start_new_session=True)
  record={'stage':stage,'started_utc':stamp(),'process':identity(process.pid),'command':command}
  write(base/(stage+'-start.json'),record)
  print(stage+' PID '+str(process.pid),flush=True)
  code=process.wait(timeout=86400.)
 children=[]
 for line in subprocess.check_output(['ps','-eo','pid=,pgid=,stat=,args='],text=True).splitlines():
  row=line.split(None,3)
  if len(row)==4 and int(row[1])==process.pid:children.append(line.strip())
 record.update(exit_code=code,wall_seconds=time.monotonic()-start,remaining_process_group_members=children,all_children_exited=not children)
 write(base/(stage+'-process.json'),record)
 if code!=0 or children:raise RuntimeError(stage+' failed; retain logs and first result')
 print(stage+' complete',flush=True)
result=json.loads((run/'result.json').read_text())
write(base/'launch-result.json',{'finished_utc':stamp(),'status':result['summary']['status'],'actual_public_calls':len(result['summary']['checks']),'all_children_exited':True,'mp_slot_released':True})
print(json.dumps({'status':result['summary']['status'],'actual_public_calls':len(result['summary']['checks']),'all_children_exited':True,'mp_slot_released':True}),flush=True)
