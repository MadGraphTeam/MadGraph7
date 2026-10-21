from datetime import datetime,timezone
import hashlib,json,os,pathlib,shutil,subprocess,sys,time
work=pathlib.Path(__file__).resolve().parent;root=work.parents[1];source=work/"source";phase=sys.argv[1];out=work/phase;out.mkdir(exist_ok=False);modules=json.loads((work/"modules-current.json").read_text());records=[]
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",TMPDIR=str(work/"tmp"),XDG_CACHE_HOME=str(work/"cache"),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",RAYON_NUM_THREADS="1")
for name in modules:
 for relative in (".benchmark-work",".ninjars-build"):
  owned=source/relative;assert owned.parent==source and owned.resolve().is_relative_to(work)
  if owned.exists():shutil.rmtree(owned)
 stem=name.rsplit(".",1)[1];result=out/(stem+".json");log=out/(stem+".log");command=[sys.executable,"-B",str(work/"run-module.py"),str(source),name,str(result)];began=datetime.now(timezone.utc).isoformat()
 with log.open("wb") as stream:run=subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
 record=json.loads(result.read_text()) if result.exists() else {"module":name,"passed":False,"wrapper_failed":True}
 record.update(command=command,cwd=str(root),exit_code=run.returncode,started_utc=began,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest());records.append(record);print(json.dumps({k:record.get(k) for k in ("module","tests_run","passed","exit_code","skipped","wrapper_failed")}),flush=True)
summary={"phase":phase,"records":records,"modules":len(records),"tests_run":sum(x.get("tests_run",0) for x in records),"passed":all(x["passed"] and x["exit_code"]==0 for x in records),"failures":sum(len(x.get("failures",[])) for x in records),"errors":sum(len(x.get("errors",[])) for x in records),"skips":sum(len(x.get("skipped",[])) for x in records),"cpu_affinity":sorted(os.sched_getaffinity(0)),"native_library_or_install_calls":0,"scope":"Software-only test modules in a tracked source archive, ignored fixture parents absent before every module; fake Python protocol children and Typst are permitted."};(out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps({k:v for k,v in summary.items() if k!="records"}),flush=True)
