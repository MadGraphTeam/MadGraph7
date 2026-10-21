#!/usr/bin/env python3
"""Retain an actual serialized installed ABI process and revalidate raw observations."""
import datetime, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import ninjars_precision_v2 as precision
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark as bench
import ninjars_supplement_v2 as supplement
report=ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/installed-precision/run-v1'
prepared_path=report/'prepared.json'
prepared=policy.verified(precision.read(prepared_path),precision.SCHEMA)
if (report/'result.json').exists(): raise ValueError('Already executed')
executable=precision.checked(prepared['executable'])
stream=precision.checked(prepared['request_stream'])
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS','MKL_NUM_THREADS'):
    if os.environ.get(variable)!='1': raise ValueError(variable)
prefix=ROOT/'.benchmark-work/policy-v3/final-pins-20261021-v1/install-export/source/HEPTools/ninjars'
image={str(p.relative_to(prefix)):supplement.digest(p) for p in prefix.rglob('*') if p.is_file()}
start=time.monotonic(); started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stream.open() as stdin,(report/'stdout.log').open('x') as stdout,(report/'stderr.log').open('x') as stderr:
    child=subprocess.Popen(['taskset','-c','190',str(executable)],cwd=ROOT,stdin=stdin,stdout=stdout,stderr=stderr,start_new_session=True)
    ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split(')',1)[1].split()[19])
    active={'pid':child.pid,'process_group_id':child.pid,'start_ticks':ticks,'cpu':190,'started_utc':started,'executable':prepared['executable']}
    bench.dump(report/'active-process.json',active)
    print('ACTIVE '+json.dumps(active),flush=True)
    try: exit_code=child.wait(timeout=1800)
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
loaded={}
for line in (report/'stderr.log').read_text().splitlines():
    if not line.startswith('NINJARS_LOADED '):continue
    name=line[len('NINJARS_LOADED '):];path=Path(name)
    if name.startswith('/'):loaded[name]={'sha256':supplement.digest(path),'bytes':path.stat().st_size}
    elif name=='linux-vdso.so.1':loaded[name]={'virtual_kernel_object':True}
    else:raise ValueError('Unresolved runtime library '+name)
bench.dump(report/'loaded-libraries.json',loaded)
summary=precision.execution_summary(precision.read(precision.checked(prepared['requests'])),(report/'stdout.log').read_text(),exit_code)
result=policy.seal({'schema':precision.SCHEMA,'status':'executed','prepared':supplement.bound(prepared_path),'cpu':190,'exit_code':exit_code,'stdout':supplement.bound(report/'stdout.log'),'stderr':supplement.bound(report/'stderr.log'),'loaded_libraries':supplement.bound(report/'loaded-libraries.json'),'summary':summary})
bench.dump(report/'result.json',result)
validated=precision.validate(report/'result.json',runtime_identity=prepared['candidate_runtime_identity_sha256'],runtime_signature=prepared['candidate_runtime_signature'])
bench.dump(report/'validation.json',validated)
remaining=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        f=(p/'stat').read_text().split(')',1)[1].split()
        if int(f[2])==child.pid:remaining.append({'pid':int(p.name),'state':f[0],'start_ticks':int(f[19])})
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
after={str(p.relative_to(prefix)):supplement.digest(p) for p in prefix.rglob('*') if p.is_file()}
terminal={**active,'schema':'serialized-installed-precision-handoff-v1','exit_code':exit_code,'wall_seconds_diagnostic_only':time.monotonic()-start,'result':supplement.bound(report/'result.json'),'validation':supplement.bound(report/'validation.json'),'child_pid_absent':not Path(f'/proc/{child.pid}').exists(),'remaining_process_group_members':remaining,'all_children_exited':not remaining,'installed_image_sha256_before':image,'installed_image_sha256_after':after,'installed_image_unchanged':image==after,'installed_image_files':len(image),'performance_claim':False}
bench.dump(report/'runtime-handoff.json',terminal)
print('TERMINAL '+json.dumps({'exit_code':exit_code,'status':validated['status'],'calls':len(validated['raw_call_checks']),'all_children_exited':not remaining,'image_unchanged':image==after,'handoff':supplement.bound(report/'runtime-handoff.json')}),flush=True)
if exit_code or validated['status']!='passed' or remaining or image!=after:raise SystemExit(1)
