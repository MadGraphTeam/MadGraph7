#!/usr/bin/env python3
"""Separately guarded replay; never compile and never retry a timed-out worker."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from analyze import analyze

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
WORK=ROOT/'.tirust-build/qp-scalar-shadow-profile'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def blocks(text):
    result=[];current=None
    tags={'INPUT_P','INPUT_S','INPUT_M','INPUT_MU','STATUS','HOST_FLAGS','OUTPUT','ERROR','CACHE','END'}
    for line in text.splitlines():
        if line.startswith('BEGIN '):
            assert current is None;current=[line]
        elif current is not None and line.split() and line.split()[0] in tags:
            current.append(line)
            if line.startswith('END '):result.append(current);current=None
    assert current is None and len(result)==36, 'Incomplete replay output retained as failure'
    expected=json.loads((HERE/'expected-input-bits.json').read_text())['records']
    selected=json.loads((HERE/'selected-inputs.json').read_text())['records']
    for i,block in enumerate(result):
        ref=expected[i if i<24 else (0 if i<34 else 16)]
        fields={line.split()[0]:line.split(maxsplit=1)[1].strip() for line in block[1:]}
        assert len(fields)==len(block)-1,'Duplicate tagged output line'
        assert tags-{'ERROR'}<=fields.keys(),'Missing required replay field'
        for name in ('INPUT_P','INPUT_S','INPUT_M','INPUT_MU'):
            assert fields[name]==ref[name],(i+1,name,'saved binary128 input mismatch')
        if i<24:assert int(fields['STATUS'])==selected[i]['historical_status'],(i+1,'historical status changed')
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root-authorized-native-window',action='store_true')
    ap.add_argument('--cpu',type=int,required=True)
    args=ap.parse_args()
    if not args.root_authorized_native_window:
        raise SystemExit('Native launch deferred: source/build review is not replay authorization.')
    assert Path.cwd().resolve()==ROOT
    frozen=json.loads((HERE/'source-index.json').read_text())
    for rel,expected in frozen['files'].items():assert digest(HERE/rel)==expected,rel
    assert args.cpu in os.sched_getaffinity(0)
    build=json.loads((HERE/'build-result.json').read_text())
    prepared=json.loads((HERE/'prepared.json').read_text())
    for rel,sha in prepared['frozen_fortran_runtime_sha256'].items():assert digest(ROOT/rel)==sha,rel
    assert build['returncode']==0 and not build.get('timed_out')
    for name,value in build['replay_executable_sha256'].items():assert digest(WORK/('replay-'+name))==value
    assert digest(WORK/'runtime/libmadloop_tirust.so')==build['library_sha256']
    results=HERE/'results'
    if results.exists():raise SystemExit('Refusing to overwrite an earlier replay')
    results.mkdir()
    record={'scope':'Bounded diagnostic tensor replay, not amplitude timing','cpu':args.cpu,
            'build_manifest_sha256':digest(HERE/'build-result.json'),'source_manifest_sha256':digest(HERE/'prepared.json'),
            'individual_timeout_seconds':180,'process_limit':3,'runs':[]}
    lock=open(ROOT/'.tirust-build/numerical-tests.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        for label,exe,mode in [('original','original','0'),('off','instrumented','0'),('on','instrumented','1')]:
            cwd=WORK/('run-'+label);cwd.mkdir()
            env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
            env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
            command=['taskset','-c',str(args.cpu),str(WORK/('replay-'+exe)),str(HERE/'selected-inputs.dat'),mode]
            item={'label':label,'command':command,'host_load_before':os.getloadavg()}
            linkage=subprocess.check_output(['ldd',str(WORK/('replay-'+exe))],text=True,env=env,cwd=ROOT)
            (results/(label+'-ldd.txt')).write_text(linkage)
            resolved=[line.split('=>',1)[1].strip().split()[0] for line in linkage.splitlines()
                      if 'libmadloop_tirust.so =>' in line]
            assert len(resolved)==1
            expected_library=(ROOT/'.tirust-build/ew-threshold-tensor-replay-runtime/libmadloop_tirust.so'
                              if exe=='original' else WORK/'runtime/libmadloop_tirust.so')
            assert Path(resolved[0]).resolve()==expected_library.resolve(),'Unexpected runtime DSO'
            item['actual_dso_sha256']=digest(expected_library)
            started=time.monotonic()
            with (results/(label+'.log')).open('w') as log:
                proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:item['returncode']=proc.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    item['timeout']=True;item['returncode']=proc.returncode
            item['elapsed_seconds_not_a_benchmark']=time.monotonic()-started
            assert digest(expected_library)==item['actual_dso_sha256'],'DSO changed during replay'
            item['log_sha256']=digest(results/(label+'.log'));record['runs'].append(item)
            for name in ('shadow.json','mpfr-flags.dat'):
                if (cwd/name).exists():(results/(label+'-'+name)).write_bytes((cwd/name).read_bytes())
            (results/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
            if item['returncode']!=0:raise SystemExit('Replay failure retained; no automatic retry.')
        parsed={label:blocks((results/(label+'.log')).read_text()) for label in ('original','off','on')}
        assert parsed['original']==parsed['off']==parsed['on'],'Input/output/status/error/cache/host-flag parity failed'
        assert (results/'off-mpfr-flags.dat').read_bytes()==(results/'on-mpfr-flags.dat').read_bytes(),'MPFR flag parity failed'
        observed=json.loads((results/'on-shadow.json').read_text())
        assert observed['key_flag_changes']==observed['output_flag_changes']==0,'Observation changed MPFR flags'
        public_status={str(i+1):int(next(line.split()[1] for line in block if line.startswith('STATUS ')))
                       for i,block in enumerate(parsed['on'])}
        if observed['uniqueness_complete']:
            assert sum(e['kind']=='parent_begin' for e in observed['events'])==36,'Bridge/public parent identity mismatch'
        observed['public_status_by_parent']=public_status
        summary=analyze(observed)
        summary['original_off_on_exact_parity']=True;summary['off_on_mpfr_flag_parity']=True
        (results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        record['complete']=True
        (results/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
