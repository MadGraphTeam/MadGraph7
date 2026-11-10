"""Bind a new actual-install replay to old095e without rewriting old evidence."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path.cwd(); report=root/'docs/benchmarkfolder/qualification/tirust-install/qp-transport-cache'
old=root/'.tirust-build/current-install/mg/HEPTools/tirust/lib/libmadloop_tirust.so'
new=root/'.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib/libmadloop_tirust.so'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
record_path=report/'dp-replay-execution.json'
if record_path.exists(): raise RuntimeError('Existing evidence cannot be overwritten')
command=[sys.executable,str(report/'validate_release.py'),'--original-library',str(old),'--split-library',str(new),'--output',str(report/'release-validation.json'),'--timeout','900']
checks=json.loads((report/'check-installation.json').read_text())
if sha(new)!=checks['native_artifacts']['libmadloop_tirust.so']['sha256']: raise RuntimeError('Published DSO changed')
if sha(old)!='095e957990c2e7ed87ed9ea73fee71eb132f00b330e0d38fd423163c173e358c': raise RuntimeError('Frozen old095e DSO changed')
record={'schema_version':1,'state':'queued-or-running','started_utc':now(),'command':command,'cwd':str(root),'source_sha256':sha(Path(__file__)),'runner_sha256':sha(report/'validate_release.py'),'old_library_sha256':sha(old),'new_library_sha256':sha(new),'shared_lock':str(root/'.tirust-build/numerical-tests.lock'),'scope':'90-call raw DP ABI parity; separate Fortran archive tests cover changed boundary'}
record_path.write_text(json.dumps(record,indent=2)+'\n')
with (report/'dp-replay-console.log').open('w') as log:
    result=subprocess.run(command,cwd=root,env={k:v for k,v in dict(os.environ,PYTHONDONTWRITEBYTECODE='1').items() if k not in ('LD_PRELOAD','LD_LIBRARY_PATH','LD_AUDIT')},stdout=log,stderr=subprocess.STDOUT)
record.update(state='complete' if result.returncode==0 else 'failed',finished_utc=now(),returncode=result.returncode)
record_path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
raise SystemExit(result.returncode)
