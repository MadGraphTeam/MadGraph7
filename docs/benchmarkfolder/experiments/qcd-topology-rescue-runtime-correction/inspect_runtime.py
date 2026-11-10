#!/usr/bin/env python3
"""One bounded exact-interpreter dependency listing; no amplitude entrypoint."""
from pathlib import Path
import hashlib
import json
import signal
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path.cwd().resolve()
HERE=Path(__file__).resolve().parent
FAILED=HERE.with_name('qcd-topology-rescue-higher-precision-native')
FAILED_SHA='d5b67551d3c9353dbd472a0d272710b63bf220bdcab6e186944713ed07e9b14c'
RUNNER=HERE.with_name('qcd-topology-rescue-higher-precision-runner')
RUNNER_SHA='db98fc5fa2fc23ed7bc46f01555c0d5a4b21ef817cf925c5b0de8c1b5f2015ca'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value,message):
    if not value:
        raise RuntimeError(message)


def interrupted(signum,frame):
    raise InterruptedError('Inspection interrupted by signal '+str(signum))


def main():
    require(sha(FAILED/'evidence-index.json')==FAILED_SHA,'Failed-attempt inventory changed.')
    require(sha(RUNNER/'source-index.json')==RUNNER_SHA,'Reviewed supervisor changed.')
    runner_entries=json.loads((RUNNER/'source-index.json').read_text())['files']
    runner_paths=list(RUNNER.rglob('*'))
    require(not any(p.is_symlink() for p in runner_paths),'Linked runner member.')
    require({str(p.relative_to(RUNNER)) for p in runner_paths if p.is_file()}==set(runner_entries)|{'source-index.json'},'Runner inventory differs.')
    for name,entry in runner_entries.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe runner member.')
        p=RUNNER/name
        require(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'Runner member changed before import: '+name)
    sys.path.insert(0,str(RUNNER))
    from controller import preflight,inventory,validate,snapshot,runtime_map,write
    proposal,pins,_,run_plain,_=preflight()
    pins.update(inventory(FAILED,'evidence-index.json'))
    sys.path.insert(0,str(FAILED))
    from audit_failure import interpreter
    runtime=proposal['runtime']
    executable=Path(runtime['command'][3])
    interp=interpreter(executable)
    require(interp==json.loads((FAILED/'failure-audit.json').read_text())['interpreter'],'Previously bound interpreter changed.')
    old=runtime['expected_loaded_files_sha256']
    old_loader=[p for p in old if Path(p).name=='ld-linux-x86-64.so.2']
    require(len(old_loader)==1,'Expected exactly one ldd-tool loader.')
    expected={p:h for p,h in old.items() if p not in old_loader}
    expected[interp['resolved_path']]=interp['sha256']
    pins[interp['resolved_path']]=interp['sha256']
    pins[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    validate(pins)
    output=HERE/'inspection1'
    require(not output.exists(),'Inspection is one-shot; preserve previous attempts.')
    output.mkdir()
    command=['taskset','-c','11',interp['resolved_path'],'--list',str(executable)]
    record={'schema':1,'complete':False,'amplitude_executed':False,'compiler_executed':False,
            'scope':'Exact PT_INTERP --list dependency inspection only; no program entrypoint or numerical evaluation.',
            'command':command,'environment':runtime['environment'],'cpu':11,'seconds_cap':30,
            'automatic_retry_count':0,'interpreter':interp,'expected_native_runtime_sha256':expected,
            'ldd_tool_runtime_sha256':old,'protected_inputs_before':pins,'started_unix':time.time(),
            'launch_progress':{'process_spawned':False,'process_reaped':False}}
    write(output/'inspection.json',record)
    handlers={}
    try:
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            handlers[sig]=signal.signal(sig,interrupted)
        with (output/'interpreter-list.txt').open('x') as log:
            code=run_plain(command,log,cwd=ROOT,env=runtime['environment'],progress=record['launch_progress'],seconds=30)
        record['returncode']=code
        require(code==0,'Exact-interpreter listing failed.')
        actual=runtime_map((output/'interpreter-list.txt').read_text())
        record['listed_runtime_sha256']=actual
        require(actual==expected,'Native listing differs beyond declared loader correction.')
        record['complete']=True
    except BaseException as exc:
        record['error']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        try:
            after,errors=snapshot(pins)
            record['protected_inputs_after']=after
            record['protected_inputs_unchanged']=after==pins and not errors
            if errors:
                record['snapshot_errors']=errors
            record['complete']=record['complete'] and record['protected_inputs_unchanged']
            log=output/'interpreter-list.txt'
            if log.is_file():
                record['listing_sha256']=sha(log);record['listing_bytes']=log.stat().st_size
            record['finished_unix']=time.time()
            write(output/'inspection.json',record)
        finally:
            for sig,handler in handlers.items():
                signal.signal(sig,handler)
    require(record['complete'],'Incomplete inspection.')
    print(json.dumps({'inspection':'pass','listed_files':len(actual),'amplitude_executed':False,
                      'interpreter':interp,'native_runtime_delta':'Only PT_INTERP replaces ldd tool loader.'},sort_keys=True))


if __name__=='__main__':
    main()
