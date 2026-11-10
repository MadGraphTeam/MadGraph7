"""Run real source/ABI admission with writes, children, loaders and locks denied."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    module=types.ModuleType('guarded_capture_readiness')
    module.__file__=str(here/'controller.py')
    exec(compile((here/'controller.py').read_bytes(),module.__file__,'exec'),module.__dict__)
    if module.WORK.exists() or module.WORK.is_symlink():
        raise ValueError('Readiness requires the native work namespace to be absent')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Use a new readiness receipt')
    active=[True]
    counts={'read_opens':0,'write_attempts':0,'native_attempts':0,'lock_attempts':0}
    write_flags=os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND
    def audit(event,values):
        if not active[0]: return
        if event=='open':
            if values[2]&write_flags:
                counts['write_attempts']+=1
                raise PermissionError('Read-only readiness attempted a write')
            counts['read_opens']+=1
        if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.symlink','os.link','os.chmod','os.chown','os.utime'):
            counts['write_attempts']+=1
            raise PermissionError('Read-only readiness attempted a mutation')
        if event in ('subprocess.Popen','os.system','os.exec','os.posix_spawn','os.fork','os.forkpty','ctypes.dlopen'):
            counts['native_attempts']+=1
            raise PermissionError('Readiness attempted a child or loader')
    def no_native(*args,**kwargs):
        counts['native_attempts']+=1
        raise PermissionError('Readiness attempted Popen/CDLL')
    def no_lock(*args,**kwargs):
        counts['lock_attempts']+=1
        raise PermissionError('Readiness attempted a numerical lock')
    old_popen=subprocess.Popen;old_cdll=ctypes.CDLL;old_flock=fcntl.flock
    subprocess.Popen=no_native;ctypes.CDLL=no_native;fcntl.flock=no_lock
    sys.addaudithook(audit)
    try:
        result=module.readonly_inputs()
        if module.WORK.exists() or module.WORK.is_symlink():
            raise AssertionError('Readiness created the native work namespace')
        if any(counts[key] for key in ('write_attempts','native_attempts','lock_attempts')):
            raise AssertionError('Readiness guard observed a prohibited operation')
    finally:
        active[0]=False
        subprocess.Popen=old_popen;ctypes.CDLL=old_cdll;fcntl.flock=old_flock
    proof={'schema':'madloop-private-tirust-capture-readiness-v1','verified':True,
           'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'controller_sha256':hashlib.sha256((here/'controller.py').read_bytes()).hexdigest(),
           'guard':counts,'native_work_absent':True,'readonly_inputs':result}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(proof,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({'verified':True,'source_index_sha256':result['source_index_sha256'],
                      'proof_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'cases':len(result['cases']),'guard':counts},indent=2))


if __name__=='__main__':
    main()
