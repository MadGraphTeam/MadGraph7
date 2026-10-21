import datetime,hashlib,json,os
from pathlib import Path
import shutil,subprocess,sys,threading,time,traceback
root=Path.cwd();base=root/'docs/benchmarkfolder/policy-v2/final-representative-v1';work=root/'.benchmark-work/policy-v2/final-representative-v1'
if base.exists() or work.exists():raise RuntimeError('Preserve existing representative attempts')
base.mkdir(parents=True);work.mkdir(parents=True)
if os.getpgrp()!=os.getpid():os.setsid()
if os.sched_getaffinity(0)!={190}:raise RuntimeError('CPU190 required')
os.environ.update(TMPDIR=str(root/'.ninjars-build/tmp'),SYMBOLICA_LOCK_DIR=str(root/'.ninjars-build/symbolica-locks'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def bound(path):return {'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
def identity(pid):
 stat=(Path('/proc')/str(pid)/'stat').read_text()
 return {'pid':pid,'pgid':os.getpgid(pid),'start_ticks':stat[stat.rfind(')')+2:].split()[19]}
write(base/'launch.json',{'schema':'final-representative-launch-v1','started_utc':stamp(),'controller':identity(os.getpid()),'parent_pid':os.getppid(),'revision':'b0ea105a04afa34304d437d2956e4d4672360670','cpu_affinity':sorted(os.sched_getaffinity(0)),'root_mp_handoff':True,'targets':8,'persistent_commands_per_target':2,'fresh_parity':True})
shutil.copy2(__file__,base/'launch.py')
print(json.dumps({'controller':identity(os.getpid()),'report':str(base.relative_to(root))}),flush=True)
ORIGINAL_POPEN=subprocess.Popen;lock=threading.Lock()
def tracked_popen(*args,**kwargs):
 process=ORIGINAL_POPEN(*args,**kwargs)
 try:row=identity(process.pid)
 except (FileNotFoundError,ProcessLookupError):row={'pid':process.pid,'already_exited':True}
 cmd=kwargs.get('args',args[0] if args else None)
 if isinstance(cmd,(list,tuple)):cmd=[str(x) for x in cmd]
 row.update(started_utc=stamp(),command=cmd,cwd=str(kwargs.get('cwd',root)))
 with lock:
  with (base/'descendant-starts.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
 return process
subprocess.Popen=tracked_popen
sys.path[:0]=[str(root),str(root/'tools')]
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_references_v2 as references
expected='8b454519cb6818ef1fafe37378930c276f7e85394a2333ebc9fdb00e75726cff'
inventory=policy.verified(policy.read(root/'docs/benchmarkfolder/policy-v2/final-core-generation-v1/runtime-inventory.json'),'fresh-generated-runtime-inventory-v2')
if set(inventory['runtime_groups']['8'])!={expected}:raise RuntimeError('Final core runtime group changed')
signature=inventory['runtime_groups']['8'][expected]['signature']
precision=policy.read(root/'docs/benchmarkfolder/policy-v2/final-installed-precision-v1/run-v1/prepared.json')
if precision['candidate_runtime_signature']!=signature:raise RuntimeError('Installed64 proof differs from final core runtime')
correction=root/'docs/benchmarkfolder/policy-v2/model-correction.json'
configuration='docs/benchmarkfolder/policy-v2/controls.json'
def freeze():
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!='b0ea105a04afa34304d437d2956e4d4672360670':raise RuntimeError('HEAD changed')
 subprocess.run(['git','diff','--exit-code','HEAD','--',*bench.GENERATOR_PATHS,'vendor/ninjars'],cwd=root,stdout=subprocess.DEVNULL,check=True)
 if hashlib.sha256((root/'HEPTools/ninjars/lib/libninjars.a').read_bytes()).hexdigest()!='800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42':raise RuntimeError('Main archive changed')
freeze()
collections=[('ggh',['control-gg-h'],[1,6,7,8],'docs/benchmarkfolder/points'),('mixed',['control-mixed'],[8],'docs/benchmarkfolder/points'),('forced-qp',['control-forced-qp'],[1,8],'docs/benchmarkfolder/points'),('ninja-near-singular',['control-forced-qp'],[6],'docs/benchmarkfolder/policy-v2/representative-point-streams/near-singular')]
results={};runtime_bindings=[]
for name,cases,backends,points in collections:
 spec={'run_id':'final-representative-'+name+'-b0ea','configuration':configuration,'report':str((base/'runs'/name).relative_to(root)),'work':str((work/name).relative_to(root)),'output_roots':{'baseline':'.benchmark-work/outputs/baseline','candidate':'.benchmark-work/policy-v2/final-representative-generation-v1/outputs/candidate'},'points_root':points,'cases':cases,'sources':['candidate'],'backends':backends,'cpu':190,'representative':True,'fresh_parity':True,'timeout':7200,'minimum_seconds':1e-12}
 spec_path=base/'specs'/(name+'.json');write(spec_path,spec)
 print(name+' started',flush=True);start=time.monotonic()
 try:
  freeze();binding_spec=references.collect_timing(spec_path)
  run=root/spec['report'];role_path=run/'role-bindings.json';runtime_path=base/'runtime-bindings'/(name+'.json')
  policy.bind_run(binding_spec['report'],binding_spec['role_map'],binding_spec['outputs'],correction,role_path)
  policy.refresh_runtime_bindings(role_path,correction,runtime_path)
  bound_run=policy.checked_bindings(runtime_path,policy.validate_correction(correction),live=True)
  for target in bound_run['targets'].values():
   if target['reducer_runtime']!=signature:raise RuntimeError('Actual representative runtime differs from final core')
  protocol,rows=references.workers.checked_series(run,minimum=2)
  targets={}
  for key,target in bound_run['targets'].items():
   first=policy.read(run/'data'/(key+'-0.json'));fresh=policy.read(run/'data'/(key+'-fresh.json'))
   parity=references.workers.numerical_parity(first,fresh,protocol['configuration'])
   targets[key]={'status_counts':first['status_counts'],'runtime':target['reducer_runtime']['identity_sha256'],'executable':target['executable_sha256'],'generation':target['generation_source_identity']['tree_state_sha256'],'parity':parity}
  results[name]={'status':'passed','seconds':time.monotonic()-start,'runtime_binding':bound(runtime_path),'targets':targets,'records':len(rows)}
  runtime_bindings.append(str(runtime_path.relative_to(root)))
  freeze();write(base/(name+'-result.json'),results[name]);print(name+' passed',flush=True)
 except BaseException as error:
  failure={'status':'failed','error_type':type(error).__name__,'error':str(error),'seconds':time.monotonic()-start}
  write(base/(name+'-failure.json'),failure);(base/(name+'-traceback.txt')).write_text(traceback.format_exc());raise
write(base/'candidate-results.json',{'schema':'final-candidate-representative-results-v1','finished_utc':stamp(),'runtime':expected,'collections':results,'runtime_bindings':runtime_bindings,'targets':sum(len(x['targets']) for x in results.values()),'status':'passed','final_role_proof_pending':True})
print(json.dumps({'status':'passed','targets':sum(len(x['targets']) for x in results.values()),'runtime':expected,'final_role_proof_pending':True}),flush=True)
