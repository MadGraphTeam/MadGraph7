#!/usr/bin/env python3
"""Guarded private build recipe. Do not invoke before root's source review."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
WORK=ROOT/'.tirust-build/qp-scalar-shadow-profile'

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root-reviewed-source', action='store_true')
    args=ap.parse_args()
    if not args.root_reviewed_source:
        raise SystemExit('Build is deferred: requires root review and an explicit coordinated launch.')
    assert Path.cwd().resolve()==ROOT
    affinity=sorted(os.sched_getaffinity(0))
    assert affinity==[2,3], 'Authorized build affinity is exactly CPUs2,3; use taskset.'
    frozen=json.loads((HERE/'source-index.json').read_text())
    for rel,expected in frozen['files'].items():assert digest(HERE/rel)==expected,rel
    prepared=json.loads((HERE/'prepared.json').read_text())
    for rel,sha in prepared['isolated_source_sha256'].items():
        assert digest(WORK/rel)==sha,rel
    for rel,sha in prepared['frozen_fortran_runtime_sha256'].items():assert digest(ROOT/rel)==sha,rel
    env=os.environ.copy()
    # Cargo fingerprinted dependency objects are shared; the private crate has
    # a unique library name. Installed prefixes and acceptance outputs are not
    # build destinations. Cargo owns all writes to this ignored cache.
    env['CARGO_HOME']=str(ROOT/'.tirust-build/cargo-home')
    env['CARGO_TARGET_DIR']=str(ROOT/'.tirust-build/cargo-build')
    env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    command=['cargo','rustc','--offline','--locked','--release','--lib','--features','poles-only',
             '--manifest-path',str(WORK/'bridge/Cargo.toml'),'-j','2','--',
             '-C','link-arg=-Wl,-soname,libmadloop_tirust.so','-C','link-arg=-Wl,--exclude-libs,ALL']
    record={'scope':'Private diagnostic compilation only','command':command,
            'cap_seconds':900,'jobs':2,'actual_inherited_affinity':affinity,
            'source_manifest_sha256':digest(HERE/'prepared.json')}
    start=time.monotonic()
    with (WORK/'build.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: record['returncode']=proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            record['timed_out']=True;record['returncode']=proc.returncode
    record['elapsed_seconds']=time.monotonic()-start
    record['log_sha256']=digest(WORK/'build.log')
    (HERE/'build-result.json').write_text(json.dumps(record,indent=2)+'\n')
    if record['returncode']!=0:raise SystemExit('Build failed or timed out; no automatic retry.')
    source=Path(env['CARGO_TARGET_DIR'])/'release/libmadloop_tirust_qp_shadow.so'
    target=WORK/'runtime/libmadloop_tirust.so';target.parent.mkdir(exist_ok=True)
    target.write_bytes(source.read_bytes())
    record['library_sha256']=digest(target)
    runtime=ROOT/'.tirust-build/ew-threshold-tensor-replay-runtime'
    historical=runtime/'libmadloop_tirust.so'
    assert digest(historical)=='49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'
    assert digest(runtime/'mtirust.f90')=='12295dca6b64a96dbd4f184cf67ed702f38c9fd82658e196fa4c5b7135fee10e'
    record['driver_commands']=[]
    for name,library,extra in [('original',historical,[]),('instrumented',target,['-DSHADOW'])]:
        command=[shutil.which('gfortran'),'-O2','-frounding-math','-ffree-line-length-none','-cpp',*extra,
                 str(HERE/'replay.F90'),'-I'+str(runtime),str(runtime/'libtirust.a'),
                 '-lgcc_s',str(library),'-Wl,-rpath,'+str(library.parent),'-lquadmath',
                 '-o',str(WORK/('replay-'+name))]
        remaining=900-(time.monotonic()-start)
        if remaining<=0:raise SystemExit('Total build cap reached before driver compilation; no retry.')
        with (WORK/('build-'+name+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        record['driver_commands'].append(command)
    record['replay_executable_sha256']={name:digest(WORK/('replay-'+name)) for name in ('original','instrumented')}
    record['historical_library_sha256']=digest(historical)
    record['elapsed_seconds']=time.monotonic()-start
    (HERE/'build-result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(target)

if __name__=='__main__':main()
