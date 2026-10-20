#!/usr/bin/env python3
"""Retry the preserved pre-compiler failure using the existing Cargo cache."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from build import HERE, ROOT, PRIOR, WORK, binding, sources, sha, write

assert Path.cwd() == ROOT
os.sched_setaffinity(0, {234})
prior = json.loads((HERE / 'build-result.json').read_text())
assert prior['status'] == 'failed' and prior['exit_code'] == 101
assert sources(WORK / 'bridge') == prior['allocation_sources'] == sources(HERE / 'source')
assert sources(PRIOR / 'original') == prior['original_sources']
baseline = ROOT / prior['baseline_archive']['path']
assert binding(baseline) == prior['baseline_archive']
attempt = HERE / 'build-retry-1'
attempt.mkdir()
env = dict(os.environ, CARGO_HOME=str(Path.home() / '.cargo'),
    CARGO_TARGET_DIR=str(WORK / 'target'), CARGO_BUILD_BUILD_DIR=str(WORK / 'build'),
    CARGO_BUILD_JOBS='1')
result = dict(prior)
result.update(status='building', prior_failed_result=binding(HERE / 'build-result.json'),
    correction='CARGO_HOME is existing dependency cache; private target/build directories retained',
    controller=binding(Path(__file__)),
    environment={n:env[n] for n in ['CARGO_HOME','CARGO_TARGET_DIR','CARGO_BUILD_BUILD_DIR','CARGO_BUILD_JOBS','TMPDIR']},
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
for n in ['failure', 'elapsed_seconds', 'finished_utc', 'exit_code']:
    result.pop(n, None)
write(attempt / 'started.json', result)
start = time.monotonic()
try:
    with (attempt / 'cargo.jsonl').open('x') as stdout, (attempt / 'cargo.stderr').open('x') as stderr:
        child = subprocess.Popen(result['command'], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        write(attempt / 'pids.json', {'controller':os.getpid(),'cargo':child.pid})
        print(json.dumps({'controller':os.getpid(),'cargo':child.pid}), flush=True)
        result['exit_code'] = child.wait()
    assert result['exit_code'] == 0
    assert sources(WORK / 'bridge') == prior['allocation_sources']
    assert binding(baseline) == prior['baseline_archive']
    result['allocation_archive'] = binding(WORK / 'target/release/libmadloop_ninjars.a')
    result['status'] = 'compiled-not-executed'
except BaseException as exc:
    result['status'] = 'failed'
    result['failure'] = repr(exc)
    raise
finally:
    result['elapsed_seconds'] = time.monotonic() - start
    result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write(attempt / 'result.json', result)
print(json.dumps({'status':result['status'],'archive':result['allocation_archive']}),flush=True)
