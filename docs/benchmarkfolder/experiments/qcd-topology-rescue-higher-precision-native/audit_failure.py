#!/usr/bin/env python3
"""Reproduce the failed runtime-capture audit without any native execution."""
from pathlib import Path
import hashlib
import json
import struct
import sys

sys.dont_write_bytecode=True
ROOT=Path.cwd().resolve()
HERE=Path(__file__).resolve().parent
RUNNER=HERE.with_name('qcd-topology-rescue-higher-precision-runner')
RUNNER_SHA='db98fc5fa2fc23ed7bc46f01555c0d5a4b21ef817cf925c5b0de8c1b5f2015ca'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value,message):
    if not value:
        raise RuntimeError(message)


def interpreter(path):
    raw=path.read_bytes()
    require(raw[:6]==b'\x7fELF\x02\x01','Expected ELF64 little-endian executable.')
    header=struct.unpack_from('<16sHHIQQQIHHHHHH',raw)
    offset,size,count=header[5],header[9],header[10]
    require(size==56 and offset+size*count<=len(raw),'Malformed program-header table.')
    found=[]
    for i in range(count):
        ph=struct.unpack_from('<IIQQQQQQ',raw,offset+i*size)
        if ph[0]==3:
            start,length=ph[2],ph[5]
            require(length>1 and start+length<=len(raw),'Malformed PT_INTERP.')
            payload=raw[start:start+length]
            require(payload[-1:]==b'\0' and b'\0' not in payload[:-1],'Invalid interpreter string.')
            found.append(payload[:-1].decode())
    require(len(found)==1 and Path(found[0]).is_absolute(),'Expected one absolute PT_INTERP.')
    target=Path(found[0]).resolve()
    return {'pt_interp':found[0],'resolved_path':str(target),'sha256':sha(target)}


def compute():
    require(sha(RUNNER/'source-index.json')==RUNNER_SHA,'Runner revision changed.')
    sys.path.insert(0,str(RUNNER))
    from controller import preflight
    proposal,pins,historical,_,_=preflight()
    directory=HERE/'attempt1'
    record=json.loads((directory/'execution.json').read_text())
    require(not record['complete'] and record['native_executed'],'Failure classification changed.')
    require(record['source_index_sha256']==RUNNER_SHA,'Wrong executed controller.')
    require(record['error']=={'type':'RuntimeError','message':'Process exited before complete actual mapping capture.'},'Unexpected failure.')
    require(record['native_launch_progress']['returncode']==0 and record['native_launch_progress']['process_reaped'],'Child was not reaped successfully.')
    for prefix in ('protected_inputs','runtime_sha256','resource_sha256'):
        require(record[prefix+'_unchanged'] and record[prefix+'_before']==record[prefix+'_after'],'Protected bytes changed.')
    require(record['protected_inputs_before']==pins,'Executed pin set differs from reviewed source.')
    for name,prefix in (('observed.log','log'),('trace.dat','trace'),('ldd.txt','ldd'),('process-maps.txt','maps')):
        p=directory/name
        require(sha(p)==record[prefix+'_sha256'] and p.stat().st_size==record[prefix+'_bytes'],'Captured file changed.')
    require((directory/'process-maps.txt').stat().st_size==0 and 'actual_process_mapping' not in record,'Unexpected retained mapping proof.')
    executable=Path(record['command'][3])
    require(sha(executable)==pins[str(executable.relative_to(ROOT))],'Executable changed.')
    interp=interpreter(executable)
    expected=record['runtime_sha256_before']
    loaders={p:h for p,h in expected.items() if Path(p).name=='ld-linux-x86-64.so.2'}
    require(len(loaders)==1 and interp['resolved_path'] not in expected,'Expected loader mismatch absent.')
    ldd=(directory/'ldd.txt').read_text()
    require(any(interp['pt_interp']+' =>' in line and '0d8g8n0a11v6f5m2h416ajyxmnkwc3md-glibc-2.42-67' in line for line in ldd.splitlines()),'Retained ldd substitution differs.')
    reference=HERE.with_name('qcd-topology-rescue-native')/'attempt1/topology-c/backend-9'
    rows=historical.compare_rows((directory/'observed.log').read_text(),(reference/'observed.log').read_text())
    decoder=historical.load_decoder()
    trace=decoder.load_trace(directory/'trace.dat',directory/'observed.log')
    old=decoder.load_trace(reference/'trace.dat',reference/'observed.log')
    compared=historical.compare_traces(trace,old,9)
    stress=trace['points'][1]
    require(stress['host_code']==323 and len(stress['calls'])==232 and all(c['native_status']==0 and c['stable'] for c in stress['calls']),'Provisional raw result changed.')
    return {'schema':1,'native_executed_by_auditor':False,'capture_complete':False,
            'actual_runtime_identity_proven':False,'physics_accepted':False,
            'native_exit_code':0,'native_process_wall_seconds':record['native_launch_progress']['process_wall_seconds'],
            'executable':{'path':str(executable.relative_to(ROOT)),'sha256':sha(executable)},
            'interpreter':interp,'ldd_expected_loader':loaders,
            'static_capture_predicate_mismatch':True,
            'cause_scope':'The expected ldd loader is incompatible with the pinned normal direct-exec PT_INTERP. Empty retained maps cannot prove the actual loaded DSO or CPU affinity.',
            'raw_scope':'Provisional numerical trace only; neither runtime qualification nor a replacement successful attempt.',
            'ordinary_numeric_rows_exact':rows['ordinary_numeric_rows_exact'],
            'ordinary_raw_words_exact':compared['ordinary_raw_words_exact'],
            'health':compared['health'],'point_summaries':compared['point_summaries'],
            'stress_numeric_rows':rows['stress_numeric_rows'],
            'bindings':{str(p.relative_to(ROOT)):sha(p) for p in
                        (Path(__file__),RUNNER/'source-index.json',directory/'execution.json',directory/'observed.log',
                         directory/'trace.dat',directory/'ldd.txt',directory/'process-maps.txt',reference/'execution.json',
                         reference/'observed.log',reference/'trace.dat')}}


if __name__=='__main__':
    result=compute()
    if sys.argv[1:]==['--write']:
        path=HERE/'failure-audit.json'
        require(not path.exists(),'Do not replace a retained audit.')
        path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    else:
        require(not sys.argv[1:],'Unknown arguments.')
        require(result==json.loads((HERE/'failure-audit.json').read_text()),'Audit differs from retained result.')
        print(json.dumps({'verification':'pass','native_executed':False,'capture_complete':False,'physics_accepted':False}))
