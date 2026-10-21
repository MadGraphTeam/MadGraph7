import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback

ROOT=Path.cwd()
REVISION='b0ea105a04afa34304d437d2956e4d4672360670'
WORK=ROOT/'.benchmark-work/policy-v2/final-representative-generation-v1'
REPORT=ROOT/'docs/benchmarkfolder/policy-v2/final-representative-generation-v1'
if WORK.exists() or REPORT.exists(): raise RuntimeError('Preserve previous control attempts; fresh namespaces required')
WORK.mkdir(parents=True);REPORT.mkdir(parents=True)
if os.getpgrp()!=os.getpid(): os.setsid()
if os.sched_getaffinity(0)!={194,195}: raise RuntimeError('Control generation requires CPUs194,195')
LOCK=threading.RLock()
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def bound(path): return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size}
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def identity(pid):
 path=Path('/proc')/str(pid);stat=(path/'stat').read_text()
 return {'pid':pid,'pgid':os.getpgid(pid),'start_ticks':stat[stat.rfind(')')+2:].split()[19]}
write(REPORT/'launch.json',{'schema':'final-control-generation-launch-v1','started_utc':stamp(),'revision':REVISION,'controller':identity(os.getpid()),'parent_pid':os.getppid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'jobs':2,'amplitude_calls':0,'mp_calls':0})
shutil.copy2(__file__,REPORT/'launch.py')
ORIGINAL_POPEN=subprocess.Popen
def tracked_popen(*args,**kwargs):
 process=ORIGINAL_POPEN(*args,**kwargs)
 try: record=identity(process.pid)
 except (FileNotFoundError,ProcessLookupError): record={'pid':process.pid,'already_exited':True}
 command=kwargs.get('args',args[0] if args else None)
 if isinstance(command,(tuple,list)): command=[str(x) for x in command]
 record.update(started_utc=stamp(),command=command,cwd=str(kwargs.get('cwd',ROOT)))
 with LOCK:
  with (REPORT/'descendant-starts.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
 return process
subprocess.Popen=tracked_popen
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
bench.WORK=WORK;bench.REPORT=REPORT
config=json.loads((ROOT/'docs/benchmarkfolder/policy-v2/controls.json').read_text())
correction=policy.verified(json.loads((ROOT/'docs/benchmarkfolder/policy-v2/model-correction.json').read_text()),policy.CORRECTION)
installation=json.loads((ROOT/'docs/benchmarkfolder/policy-v2/final-installed-precision-v1/installation.json').read_text())
selected=[case for case in config['cases'] if case['id'] in ('control-gg-h','control-mixed','control-forced-qp')]
if len(selected)!=3: raise RuntimeError('Exact three control definitions required')
write(REPORT/'controls.json',config)
STATUS={'schema':'final-control-generation-status-v1','revision':REVISION,'started_utc':stamp(),'status':'started','cpu_affinity':sorted(os.sched_getaffinity(0)),'cases':{c['id']:{'status':'pending','case':c} for c in selected},'amplitude_evaluations':0,'initialized_symbolica_workers':0,'installation':bound(ROOT/'docs/benchmarkfolder/policy-v2/final-installed-precision-v1/installation.json')}
def update(cid,record):
 with LOCK: STATUS['cases'][cid]=record;write(REPORT/'status.json',STATUS)
 print(cid+' '+record['status'],flush=True)
def freeze():
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=REVISION: raise RuntimeError('HEAD changed during control generation')
 if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='madloop_ninjars': raise RuntimeError('Wrong branch')
 subprocess.run(['git','diff','--exit-code','HEAD','--',*bench.GENERATOR_PATHS,'vendor/ninjars'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 if subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*bench.GENERATOR_PATHS,'vendor/ninjars'],cwd=ROOT,text=True).strip(): raise RuntimeError('Uncommitted numerical source additions')
 for name,expected in installation['files'].items():
  if sha(ROOT/'HEPTools/ninjars'/name)!=expected: raise RuntimeError('Installed asset changed: '+name)
freeze();initial=bench.generator_tree_state();initial_sha=bench.content_digest(initial)
write(REPORT/'generator-state.json',initial)
write(REPORT/'status.json',STATUS)
def case_job(case):
 cid=case['id'];output=bench.output_dir('candidate',case);original=ROOT/'.benchmark-work/outputs/baseline'/cid
 record={'case':case,'status':'started','output':str(output.relative_to(ROOT))}
 update(cid,record)
 try:
  freeze()
  if bench.content_digest(bench.generator_tree_state())!=initial_sha: raise RuntimeError('Generator fingerprint changed')
  if output.exists(): raise RuntimeError('Control export already exists')
  bench.generate('candidate',case)
  generated=bench.generated_source_manifest(output)
  if policy.model_subset(generated)!=policy.model_subset(correction['cases'][cid]['fresh_after_manifest']): raise RuntimeError('Fresh control MODEL differs from correction postimage')
  card=output/'Cards/param_card.dat';original_card=original/'Cards/param_card.dat'
  if card.read_bytes()!=original_card.read_bytes(): raise RuntimeError('Fresh control physical card differs from retained original')
  shutil.copy2(original_card,card)
  points=ROOT/'docs/benchmarkfolder/points'/(cid+'.dat');metadata=points.with_suffix('.json')
  for path in (points,metadata):
   destination=REPORT/'points'/path.name;destination.parent.mkdir(exist_ok=True);shutil.copy2(path,destination)
  write(REPORT/'manifests'/(cid+'-generated.json'),generated)
  record.update(status='building',generation=json.loads((output/'generation.json').read_text()),parameter_card=bound(card),points=bound(points),point_metadata=bound(metadata))
  update(cid,record)
  masses=bench.build('candidate',case)
  saved=json.loads(metadata.read_text())
  if saved['masses']!=masses or saved['points']!=10 or saved['sha256']!=sha(points): raise RuntimeError('Control masses/point inventory changed')
  compilation=json.loads((output/'compilation.json').read_text())
  if compilation['build_kind']!='clean-subprocess': raise RuntimeError('Control unexpectedly reused subprocess objects')
  directory=bench.subprocess_dir('candidate',case)
  if bench.content_digest(bench.generator_tree_state())!=initial_sha: raise RuntimeError('Generator changed during build')
  freeze()
  record.update(status='complete',masses=masses,dimensions=(directory/'dimensions.dat').read_text(),compilation=compilation,compiled_manifest=bound(output/'compiled-sources.json'),amplitude_evaluations=0,initialized_symbolica_workers=0)
 except BaseException as error:
  record.update(status='failed',error_type=type(error).__name__,error=str(error))
  target=REPORT/'logs'/('failure-'+cid+'.txt');target.parent.mkdir(exist_ok=True);target.write_text(traceback.format_exc());record['failure']=bound(target)
 finally:
  for stage in ('generate','compile','masses'):
   log=WORK/'logs'/(stage+'-candidate-'+cid+'.log')
   if log.exists():
    destination=REPORT/'logs'/log.name;destination.parent.mkdir(exist_ok=True);shutil.copy2(log,destination)
  update(cid,record)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(case_job,c) for c in selected]
 for future in concurrent.futures.as_completed(futures): future.result()
freeze()
STATUS.update(status='complete' if all(r['status']=='complete' for r in STATUS['cases'].values()) else 'failed',finished_utc=stamp(),generator_tree_state_sha256=initial_sha)
write(REPORT/'status.json',STATUS)
print(json.dumps({'status':STATUS['status'],'cases':{k:v['status'] for k,v in STATUS['cases'].items()},'controller':os.getpid(),'pgid':os.getpgrp(),'amplitude_evaluations':0,'initialized_symbolica_workers':0}),flush=True)
if STATUS['status']!='complete': raise SystemExit(1)
