from datetime import datetime,timezone
import hashlib,json,os,pathlib,shutil,subprocess,sys,time
work=pathlib.Path(__file__).resolve().parent;root=work.parents[1];source=work/'source';out=work/'actual-run-v1';out.mkdir(exist_ok=False);modules=json.loads((work/'modules.json').read_text());records=[]
for name in ['tmp','cache']:(work/name).mkdir(exist_ok=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(work/'tmp'),XDG_CACHE_HOME=str(work/'cache'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1')
for name in modules:
 for relative in ['.benchmark-work','.ninjars-build']:
  owned=source/relative
  if owned.exists():shutil.rmtree(owned)
 stem=name.rsplit('.',1)[1];result=out/(stem+'.json');log=out/(stem+'.log');command=[sys.executable,'-B',str(work/'run-module.py'),str(source),name,str(result)];began=datetime.now(timezone.utc).isoformat()
 with log.open('wb') as stream:
  process=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);returncode=process.wait(timeout=180)
 try:os.killpg(process.pid,0);reaped=False
 except ProcessLookupError:reaped=True
 row=json.loads(result.read_text()) if result.exists() else {'module':name,'passed':False,'wrapper_failed':True}
 row.update(command=command,cwd=str(root),exit_code=returncode,started_utc=began,finished_utc=datetime.now(timezone.utc).isoformat(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),log_bytes=log.stat().st_size,process_id=process.pid,child_process_group_reaped=reaped);records.append(row);print(json.dumps({k:row.get(k) for k in ['module','tests_run','passed','exit_code','skipped','wrapper_failed','child_process_group_reaped']}),flush=True)
summary={'schema':'ninjars-targeted-software-run-v1','records':records,'modules':len(records),'tests_run':sum(r.get('tests_run',0) for r in records),'passed':all(r['passed'] and r['exit_code']==0 and r['child_process_group_reaped'] for r in records),'failures':sum(len(r.get('failures',[])) for r in records),'errors':sum(len(r.get('errors',[])) for r in records),'skips':sum(len(r.get('skipped',[])) for r in records),'cpu_affinity':sorted(os.sched_getaffinity(0)),'native_calls':0,'scope':'Six frozen software-only modules; fresh source archive plus explicit overlays, ignored fixture parents absent before every module; no real generated-program/native reducer/build calls.'};(out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
raise SystemExit(0 if summary['passed'] else 1)
