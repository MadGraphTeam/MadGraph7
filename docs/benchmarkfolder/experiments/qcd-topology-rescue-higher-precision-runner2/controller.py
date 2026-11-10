#!/usr/bin/env python3
"""ONE proposed512 replay; invoke only after explicit execution clearance."""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path.cwd().resolve()
HERE=Path(__file__).resolve().parent
DESIGN=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-design'
DESIGN_SHA='51f45d9e6f2c13227dc504fa284cba967f947db0f073dfb18bb9bf924c28c298'
NATIVE=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-native'
FAILED=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-native'
FAILED_SHA='d5b67551d3c9353dbd472a0d272710b63bf220bdcab6e186944713ed07e9b14c'
CORRECTION=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-runtime-correction'
CORRECTION_SHA='c9752a8caeea6de84e63fb90dd8407528b72a92e6c5d3b58c9f66651a8114c05'
OUTPUT=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-native2'
WORK=ROOT/'.tirust-build/qcd-topology-rescue-higher-precision/attempt2'


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temporary.replace(path)


def inventory(directory,name):
    entries=read(directory/name)['files']
    paths=list(directory.rglob('*'))
    require(not any(path.is_symlink() for path in paths),'Linked inventory path.')
    require({str(path.relative_to(directory)) for path in paths if path.is_file()}==set(entries)|{name},'Inventory has missing or additional paths.')
    result={str((directory/name).relative_to(ROOT)):sha(directory/name)}
    for item,entry in entries.items():
        path=directory/item
        require(not path.is_symlink() and not Path(item).is_absolute() and '..' not in Path(item).parts,'Unsafe controller/design inventory.')
        require(path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'Controller/design bytes changed: '+item)
        result[str(path.relative_to(ROOT))]=entry['sha256']
    return result


def validate(pins):
    for name,digest in pins.items():
        require(sha(ROOT/name)==digest,'Protected input changed: '+name)


def snapshot(pins):
    result,errors={},{}
    for name in pins:
        try:
            result[name]=sha(ROOT/name)
        except (OSError,ValueError) as exc:
            result[name]=None;errors[name]=type(exc).__name__+': '+str(exc)
    return result,errors


def preflight():
    require(sha(DESIGN/'evidence-index.json')==DESIGN_SHA,'Wrong frozen proposal.')
    pins=inventory(HERE,'source-index.json')
    pins.update(inventory(DESIGN,'evidence-index.json'))
    proposal=read(DESIGN/'proposal.json')
    require(not proposal['native_executed'] and not proposal['compiler_or_linker_required'],'Unexpected source proposal.')
    require((proposal['proposed_processes'],proposal['backend'],proposal['cpu'],proposal['thread_count'],
             proposal['seconds_per_process_cap'],proposal['automatic_retry_count'])==(1,9,11,1,180,0),'Execution limits changed.')
    for name,digest in proposal['protected_inputs'].items():
        require(name not in pins or pins[name]==digest,'Conflicting protected pin.')
        pins[name]=digest
    validate(pins)
    require(sha(NATIVE/'evidence-index.json')==pins[str((NATIVE/'evidence-index.json').relative_to(ROOT))], 'Historical helpers changed.')
    inventory(NATIVE,'evidence-index.json')
    require(sha(FAILED/'evidence-index.json')==FAILED_SHA,'Failed attempt changed.')
    require(sha(CORRECTION/'evidence-index.json')==CORRECTION_SHA,'Runtime correction changed.')
    pins.update(inventory(FAILED,'evidence-index.json'))
    pins.update(inventory(CORRECTION,'evidence-index.json'))
    inspection=read(CORRECTION/'inspection1/inspection.json')
    require(inspection['complete'] and inspection['protected_inputs_unchanged'] and not inspection['amplitude_executed'],'Invalid interpreter inspection.')
    require(inspection['protected_inputs_before']==inspection['protected_inputs_after'],'Inspection pin drift.')
    for name,digest in inspection['protected_inputs_before'].items():
        require(name not in pins or pins[name]==digest,'Conflicting correction pin.')
        pins[name]=digest
    runtime=proposal['runtime']
    require(inspection['ldd_tool_runtime_sha256']==runtime['expected_loaded_files_sha256'],'Wrong ldd-tool closure.')
    require(inspection['environment']==runtime['environment'],'Inspection environment changed.')
    require(inspection['command']==['taskset','-c','11',inspection['interpreter']['resolved_path'],'--list',runtime['command'][3]],'Wrong inspected executable/interpreter.')
    expected=dict(runtime['expected_loaded_files_sha256'])
    loaders=[p for p in expected if Path(p).name=='ld-linux-x86-64.so.2']
    require(len(loaders)==1,'Wrong ldd-tool loader count.')
    del expected[loaders[0]]
    expected[inspection['interpreter']['resolved_path']]=inspection['interpreter']['sha256']
    require(inspection['listed_runtime_sha256']==inspection['expected_native_runtime_sha256']==expected,'Runtime delta exceeds the interpreter correction.')
    require(sha(CORRECTION/'inspection1/interpreter-list.txt')==inspection['listing_sha256'],'Native dependency listing changed.')
    runtime['expected_ldd_tool_files_sha256']=runtime['expected_loaded_files_sha256']
    runtime['expected_loaded_files_sha256']=expected
    old_resources=Path(runtime['command'][6])
    require(runtime['environment']['TMPDIR']==str(old_resources.parent/'tmp'),'Unexpected original temporary directory.')
    runtime['command'][6]=str(WORK/'MadLoop5_resources')
    runtime['command'][7]=str(OUTPUT/'attempt1/trace.dat')
    runtime['environment']['TMPDIR']=str(WORK/'tmp')
    validate(pins)
    # Imported only after exact source hashes are checked. No imported main is run.
    from supervise import run_captured,run_plain_capped
    sys.path.insert(0,str(NATIVE))
    import run as historical
    return proposal,pins,historical,run_plain_capped,run_captured


def runtime_map(text):
    require('not found' not in text,'Unresolved runtime dependency.')
    result={}
    for line in text.splitlines():
        match=re.search(r'(?:=>\s*)?(/\S+)\s+\(',line)
        if match:
            path=Path(match[1]).resolve()
            result[str(path)]=sha(path)
    require(result,'No runtime mappings found.')
    return result


def interrupted(signum,frame):
    raise InterruptedError('Controller interrupted by signal '+str(signum))


def main():
    # Nonblocking admission is distinct from a native attempt. A busy shared
    # lane consumes no process/work/attempt and never starts an automatic retry.
    OUTPUT.mkdir(exist_ok=True)
    admissions=OUTPUT/'admissions';admissions.mkdir(exist_ok=True)
    admission_path=admissions/(str(time.time_ns())+'.json')
    admission={'schema':1,'native_executed':False,'automatic_retry_count':0,
               'controller_index_sha256':sha(HERE/'source-index.json'),'started_unix':time.time()}
    lock=(ROOT/'.tirust-build/numerical-tests.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        admission.update({'admitted':False,'reason':'shared numerical lock busy','attempt_created':False})
        write(admission_path,admission)
        lock.close()
        print('Shared numerical lane busy; no native attempt created and no retry scheduled.')
        return 75
    result=None;record=None;pins={};resources_before={};old_handlers={}
    started=False;protocol_passed=False
    try:
        admission.update({'admitted':True,'lock_acquired_unix':time.time(),'attempt_created':False})
        write(admission_path,admission)
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            old_handlers[sig]=signal.signal(sig,interrupted)
        proposal,pins,historical,run_capped,run_captured=preflight()
        require(not WORK.exists() and not (OUTPUT/'attempt1').exists(),'One-shot attempt already exists; no retry.')
        result=OUTPUT/'attempt1';result.mkdir()
        WORK.mkdir(parents=True)
        (WORK/'tmp').mkdir()
        record={'schema':1,'scope':'Private512 arithmetic+guard diagnostic; not physical acceptance.',
                'complete':False,'native_executed':False,'compiler_executed':False,
                'formal_performance_eligible':False,'physics_accepted':False,
                'cpu':11,'thread_count':1,'timeout_per_process_seconds':180,'automatic_retry_count':0,
                'working_bits':512,'input_bits':113,'guard_exponent':-392,
                'source_index_sha256':sha(HERE/'source-index.json'),'proposal_index_sha256':DESIGN_SHA,
                'runtime_correction_index_sha256':CORRECTION_SHA,'prior_incomplete_attempt_index_sha256':FAILED_SHA,
                'admission_path':str(admission_path.relative_to(ROOT)),
                'protected_inputs_before':pins,'started_unix':time.time()}
        admission['attempt_created']=True;write(admission_path,admission)
        write(result/'execution.json',record)
        runtime=proposal['runtime'];env=runtime['environment'];command=runtime['command']
        require(command[:3]==['taskset','-c','11'] and len(command)==8,'Command shape changed.')
        require(Path(command[6])==WORK/'MadLoop5_resources' and Path(command[7])==result/'trace.dat','Unexpected output/resource paths.')
        resources=WORK/'MadLoop5_resources';resources.mkdir()
        for name,digest in runtime['resource_copies_sha256'].items():
            source=Path(runtime['resource_source'])/name
            require(sha(source)==digest,'Resource source changed.')
            shutil.copyfile(source,resources/name)
            resources_before[str((resources/name).relative_to(ROOT))]=digest
        validate(resources_before);validate(pins)
        record.update({'command':command,'environment':env,'resource_sha256_before':resources_before,
                       'expected_runtime_sha256':runtime['expected_loaded_files_sha256']})
        record['ldd_launch_progress']={'process_spawned':False,'process_reaped':False}
        with (result/'ldd.txt').open('x') as log:
            code=run_capped(['taskset','-c','11','ldd',command[3]],log,cwd=ROOT,env=env,
                           progress=record['ldd_launch_progress'],seconds=30)
        require(code==0,'Runtime dependency inspection failed; no native amplitude started.')
        tool_runtime=runtime_map((result/'ldd.txt').read_text())
        require(tool_runtime==runtime['expected_ldd_tool_files_sha256'],'ldd tool closure differs from the retained inspection.')
        record['ldd_tool_runtime_sha256_before']=tool_runtime
        actual_runtime=runtime['expected_loaded_files_sha256']
        record['runtime_sha256_before']=actual_runtime
        validate(pins);validate(resources_before);validate(actual_runtime)
        expected_executable=str(Path(command[3]).resolve())
        selected=[name for name in actual_runtime if Path(name).name=='libmadloop_tirust.so']
        require(len(selected)==1,'Expected exactly one TIRust runtime.')
        record['spawn_attempted']=True;record['native_launch_attempt_unix']=time.time();record['host_load_start']=os.getloadavg()
        progress={'process_spawned':False,'process_reaped':False}
        record['native_launch_progress']=progress
        write(result/'execution.json',record)
        start=time.monotonic()
        try:
            with (result/'observed.log').open('x') as log:
                code,mapping=run_captured(command,log,cwd=ROOT,env=env,
                    expected_runtime=actual_runtime,executable=expected_executable,
                    selected_library=selected[0],maps_path=result/'process-maps.txt',progress=progress,seconds=180)
            record['returncode']=code;record['actual_process_mapping']=mapping
            record['native_process_wall_seconds']=mapping['process_wall_seconds']
        finally:
            started=progress['process_spawned'];record['native_executed']=started
            if started:
                record['native_start_unix']=progress['spawn_unix']
            record['native_end_unix']=time.time();record['wall_seconds']=time.monotonic()-start
            record['host_load_end']=os.getloadavg()
        require(record['returncode']==0,'Native512 diagnostic failed; no retry.')
        failed=FAILED/'attempt1'
        require(historical.selected_rows((result/'observed.log').read_text())==historical.selected_rows((failed/'observed.log').read_text()),'Raw numerical/status/IEEE rows differ from incomplete attempt.')
        require((result/'trace.dat').read_bytes()==(failed/'trace.dat').read_bytes(),'Full raw trace differs from incomplete attempt.')
        record['incomplete_attempt_all_numeric_rows_exact']=True
        record['incomplete_attempt_full_trace_bytes_exact']=True
        reference=NATIVE/'attempt1/topology-c/backend-9'
        record.update(historical.compare_rows((result/'observed.log').read_text(),(reference/'observed.log').read_text()))
        decoder=historical.load_decoder()
        trace=decoder.load_trace(result/'trace.dat',result/'observed.log')
        old=decoder.load_trace(reference/'trace.dat',reference/'observed.log')
        record.update(historical.compare_traces(trace,old,9))
        record['remaining_gate']='Offline exact group/input/handoff/mass/matrix comparison and Laurent diagnostics; no physical acceptance from execution completion.'
        protocol_passed=True
    except BaseException as exc:
        admission['error']={'type':type(exc).__name__,'message':str(exc)}
        if record is not None:
            record['error']=admission['error']
        raise
    finally:
        try:
            if record is not None:
                unchanged=True
                for prefix,before in (('protected_inputs',pins),('resource_sha256',resources_before),
                                      ('runtime_sha256',record.get('runtime_sha256_before',{})),
                                      ('ldd_tool_runtime_sha256',record.get('ldd_tool_runtime_sha256_before',{}))):
                    after,errors=snapshot(before)
                    record[prefix+'_after']=after;record[prefix+'_unchanged']=after==before and not errors
                    unchanged=unchanged and record[prefix+'_unchanged']
                    if errors:
                        record[prefix+'_snapshot_errors']=errors
                record['complete']=protocol_passed and unchanged
                for name,prefix in (('observed.log','log'),('trace.dat','trace'),('ldd.txt','ldd'),('process-maps.txt','maps')):
                    path=result/name
                    if path.is_file():
                        record[prefix+'_sha256']=sha(path);record[prefix+'_bytes']=path.stat().st_size
                snapshots=result/'process-maps.snapshots'
                if snapshots.exists():
                    record['mapping_snapshots']={str(p.relative_to(result)):{'sha256':sha(p),'bytes':p.stat().st_size}
                                                 for p in sorted(snapshots.iterdir()) if p.is_file()}
                record['finished_unix']=time.time();write(result/'execution.json',record)
            admission['native_executed']=started;admission['finished_unix']=time.time()
            write(admission_path,admission)
        finally:
            for sig,handler in old_handlers.items():
                signal.signal(sig,handler)
            lock.close()
    require(record and record['complete'],'Incomplete private diagnostic capture.')
    print('Private512 protocol capture complete; ordinary parity passed; stress physics remains unqualified.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
