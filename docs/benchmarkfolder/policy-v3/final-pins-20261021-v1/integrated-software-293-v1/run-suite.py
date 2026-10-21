from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time
W=Path(__file__).resolve().parent;ROOT=W.parents[1];SOURCE=W/'source-v4';OUT=W/'attempt-4';OUT.mkdir(exist_ok=False);modules=json.loads((W/'modules.json').read_text());prep=json.loads((W/'preparation-v4.json').read_text());records=[]
env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(SOURCE),TMPDIR=str(W/'tmp'),XDG_CACHE_HOME=str(W/'cache'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1')
for p in ['tmp','cache','scratch']:(W/p).mkdir(exist_ok=True)
scratch=W/'scratch'/OUT.name;scratch.mkdir()
for name in modules:
 stem=name.rsplit('.',1)[1]
 for relative in ['.benchmark-work','.ninjars-build']:
  owned=SOURCE/relative
  if owned.exists():shutil.move(str(owned),scratch/(stem+'-previous-'+relative[1:]))
 result=OUT/(stem+'.json');log=OUT/(stem+'.log');command=[sys.executable,'-B',str(W/'run-module.py'),str(SOURCE),name,str(result)];started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with log.open('xb') as stream:
  child=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=child.wait(timeout=180)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,15)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,9);child.wait(timeout=10)
   code=child.returncode
 try:os.killpg(child.pid,0);absent=False
 except ProcessLookupError:absent=True
 row=json.loads(result.read_text()) if result.exists() else {'module':name,'passed':False,'wrapper_failed':True,'tests_run':0,'failures':[],'errors':[{'test':name,'traceback':'Result not written; retain full log'}],'skipped':[]}
 row.update(command=command,exit_code=code,started_utc=started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),log_bytes=log.stat().st_size,pid=child.pid,child_process_group_absent=absent);row['expected_tests']=prep['test_counts'][name];row['expected_count_matched']=row['tests_run']==row['expected_tests'];records.append(row);print(json.dumps({k:row.get(k) for k in ['module','tests_run','expected_count_matched','passed','exit_code','child_process_group_absent']}),flush=True)
state=json.loads((W/'source-state-before-v4.json').read_text());changed={}
for name,old in state['files'].items():
 p=SOURCE/name;actual={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} if p.is_file() else None
 if actual!=old:changed[name]={'before':old,'after':actual}
live_overlay_changes={}
for name,old in prep['overlays'].items():
 p=ROOT/name;actual={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} if p.is_file() else None
 if actual!=old:live_overlay_changes[name]={'before':old,'after':actual}
summary={'schema':'integrated-final-selected-software-run-v1','records':records,'modules':len(records),'tests_run':sum(x['tests_run'] for x in records),'passed':all(x['passed'] and x['exit_code']==0 and x['expected_count_matched'] and x['child_process_group_absent'] for x in records) and not changed and not live_overlay_changes,'failures':sum(len(x['failures']) for x in records),'errors':sum(len(x['errors']) for x in records),'skips':sum(len(x['skipped']) for x in records),'source_files_changed':changed,'live_overlay_changes':live_overlay_changes,'cpu_affinity':sorted(os.sched_getaffinity(0)),'actual_compiler_or_reducer_invocations':0,'scope':'Twelve current relevant Python software modules, pristine selected tracked source plus exact current overlays; ignored parents absent before every module; compiler/Cargo/native/shell subprocesses blocked; onlyPython/Git/read-only fc-list queries allowed by runner. Prior Fortran/native evidence not rerun.'};(OUT/'result.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True);raise SystemExit(0 if summary['passed'] else 1)
