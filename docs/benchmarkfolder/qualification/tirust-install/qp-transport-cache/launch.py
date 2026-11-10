import datetime, hashlib, json, os, subprocess
from pathlib import Path
root=Path.cwd(); work=root/'.tirust-build/qp-transport-install'
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
command=['taskset','-c','2-5','bash',str(work/'run.sh')]
record={'state':'running','started_utc':now(),'command':command,'cwd':str(root),'compile_affinity_cpus':[2,3,4,5],'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'run_script_sha256':hashlib.sha256((work/'run.sh').read_bytes()).hexdigest(),'observer_sha256':hashlib.sha256((work/'observer/sitecustomize.py').read_bytes()).hexdigest()}
(work/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
with (work/'cli.log').open('w') as log:
    result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
record.update(state='completed' if result.returncode==0 else 'failed',finished_utc=now(),returncode=result.returncode)
(work/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
raise SystemExit(result.returncode)
