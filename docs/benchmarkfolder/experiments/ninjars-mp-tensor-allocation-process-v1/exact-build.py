#!/usr/bin/env python3
"""Match the actual installer Cargo invocation before any numerical sample."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from build import HERE, ROOT, WORK, binding, sources, write

assert Path.cwd() == ROOT
os.sched_setaffinity(0,{234})
prior = json.loads((HERE / 'build-retry-1/result.json').read_text())
assert prior['status'] == 'compiled-not-executed'
assert sources(WORK / 'bridge') == prior['allocation_sources']
dest = HERE / 'exact-build'
dest.mkdir()
baseline_fp = ROOT / '.ninjars-build/cargo/build/release/.fingerprint/madloop-ninjars-ab2a04765db6f7c1/lib-madloop_ninjars.json'
shutil.copy2(baseline_fp,dest / 'baseline-fingerprint.json')
baseline_data = json.loads(baseline_fp.read_text())
assert baseline_data['profile'] == 1760205775310677080 and baseline_data['features'] == '[]' and baseline_data['rustflags'] == []
env = dict(os.environ,CARGO_HOME=str(Path.home() / '.cargo'),
    CARGO_TARGET_DIR=str(WORK / 'exact-target'),CARGO_BUILD_BUILD_DIR=str(WORK / 'build'),CARGO_BUILD_JOBS='1')
command = ['cargo','rustc','--manifest-path',str(WORK / 'bridge/Cargo.toml'),
    '--release','--locked','--offline','--lib','--message-format=json','--','--print','native-static-libs']
flags = {k:v for k,v in env.items() if k in ['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER'] or k.startswith('CARGO_PROFILE_') or k.startswith('CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_')}
assert flags == {}, flags
record = dict(prior)
record.update(status='building',controller=binding(Path(__file__)),command=command,
    correction='Use installer cargo rustc with --print native-static-libs, matching final profile; initial cargo build retained as diagnostic only',
    prior_diagnostic_build=binding(HERE / 'build-retry-1/result.json'),
    baseline_fingerprint=binding(dest / 'baseline-fingerprint.json'),effective_flag_environment=flags,
    environment={k:env[k] for k in ['CARGO_HOME','CARGO_TARGET_DIR','CARGO_BUILD_BUILD_DIR','CARGO_BUILD_JOBS','TMPDIR']},
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
for key in ['allocation_archive','prior_failed_result','elapsed_seconds','finished_utc','exit_code']:
    record.pop(key,None)
write(dest / 'started.json',record)
start = time.monotonic()
try:
    with (dest / 'cargo.jsonl').open('x') as stdout,(dest / 'cargo.stderr').open('x') as stderr:
        child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        write(dest / 'pids.json',{'controller':os.getpid(),'cargo':child.pid})
        print(json.dumps({'controller':os.getpid(),'cargo':child.pid}),flush=True)
        record['exit_code'] = child.wait()
    assert record['exit_code'] == 0
    assert sources(WORK / 'bridge') == prior['allocation_sources']
    fp = WORK / 'build/release/.fingerprint/madloop-ninjars-ab2a04765db6f7c1/lib-madloop_ninjars.json'
    data = json.loads(fp.read_text())
    assert data['profile'] == baseline_data['profile'] and data['features'] == baseline_data['features'] and data['rustflags'] == baseline_data['rustflags']
    for key in ['rustc','target','declared_features','path','deps','config','compile_kind']:
        assert data[key] == baseline_data[key],key
    shutil.copy2(fp,dest / 'allocation-fingerprint.json')
    record['allocation_fingerprint'] = binding(dest / 'allocation-fingerprint.json')
    record['allocation_archive'] = binding(WORK / 'exact-target/release/libmadloop_ninjars.a')
    record['status'] = 'compiled-not-executed'
except BaseException as exc:
    record['status'] = 'failed'
    record['failure'] = repr(exc)
    raise
finally:
    record['elapsed_seconds'] = time.monotonic()-start
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write(dest / 'result.json',record)
print(json.dumps({'status':record['status'],'archive':record['allocation_archive']}),flush=True)
