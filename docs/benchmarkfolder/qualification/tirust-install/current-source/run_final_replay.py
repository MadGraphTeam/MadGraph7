"""Run the previously held final-library parity replay after recovery release."""
import datetime, hashlib, json, os, subprocess, sys
from pathlib import Path
root=Path.cwd(); report=root/'docs/benchmarkfolder/qualification/tirust-install/current-source'
original=root/'HEPTools/tirust/lib/libmadloop_tirust.so'
final=root/'.tirust-build/current-install/mg/HEPTools/tirust/lib/libmadloop_tirust.so'
record_path=report/'replay-execution.json'
if record_path.exists(): raise RuntimeError('Existing replay record must not be overwritten')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
command=[sys.executable,str(report/'validate_release.py'),'--original-library',str(original),'--split-library',str(final),'--output',str(report/'release-validation.json'),'--timeout','900']
record={'schema_version':1,'requested_by':'Root agent recovery release: resume prepared replay behind original QCD4 backend9 using the shared numerical lock','recovery_observation_utc':'2026-11-09T22:59:00Z','recovery_observation_precision':'approximate time reported by root','reported_host_load_average_one_minute':210,'reported_filesystem_state':'reads immediate','policy':'Only correctness replay released; formal timing acceptance remains held. Preserve original shared-lock queue. No rebuild or compiler work.','started_utc':now(),'command':command,'cwd':str(root),'launcher_sha256':sha(Path(__file__)),'runner_sha256':sha(report/'validate_release.py'),'original_library_sha256':sha(original),'final_library_sha256':sha(final),'shared_lock':str(root/'.tirust-build/numerical-tests.lock'),'stdout_stderr':'replay-console.log','state':'queued-or-running'}
expected=json.loads((report/'check-installation.json').read_text())['native_artifacts']['libmadloop_tirust.so']['sha256']
if record['final_library_sha256']!=expected: raise RuntimeError('Final library changed before replay')
record_path.write_text(json.dumps(record,indent=2)+'\n')
with (report/'replay-console.log').open('w') as output:
    result=subprocess.run(command,cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=output,stderr=subprocess.STDOUT)
record.update(finished_utc=now(),returncode=result.returncode,state='completed' if result.returncode==0 else 'failed')
record_path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'state':record['state'],'returncode':result.returncode,'final_library_sha256':record['final_library_sha256']}))
raise SystemExit(result.returncode)
