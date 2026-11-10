"""Set target loader variables only at the final execve; never fork or run ld.

The outer taskset, prlimit and bound isolated Python start in a clean environment.
execve preserves the owned PID/session and makes /proc/<pid>/exe the target.
Tests inject an execve fake; they do not execute an ELF or a loader.
"""
from pathlib import Path
import hashlib
import os
import struct
import sys


def interpreter(data):
    if len(data)<64 or data[:7]!=b'\x7fELF\x02\x01\x01':
        raise ValueError('Expected little-endian ELF64 target')
    header=struct.unpack_from('<16sHHIQQQIHHHHHH',data)
    if header[2]!=62 or header[9]!=56 or not 0<header[10]<65535:
        raise ValueError('Unsupported target ELF program headers')
    offset,count=header[5],header[10]
    if offset+count*56>len(data): raise ValueError('Truncated target ELF program headers')
    found=[]
    for index in range(count):
        ph=struct.unpack_from('<IIQQQQQQ',data,offset+index*56)
        if ph[0]!=3: continue
        start,size=ph[2],ph[5]
        if not size or start+size>len(data): raise ValueError('Invalid PT_INTERP payload')
        raw=data[start:start+size]
        if raw[-1:]!=b'\0' or b'\0' in raw[:-1]: raise ValueError('Invalid PT_INTERP string')
        value=raw[:-1].decode('ascii')
        if not Path(value).is_absolute(): raise ValueError('Nonabsolute target interpreter')
        found.append(value)
    if len(found)!=1: raise ValueError('Expected exactly one target interpreter')
    return found[0]


def target_environment(base,library_path,trace):
    paths=library_path.split(':')
    if not paths or any(not value or not Path(value).is_absolute() for value in paths):
        raise ValueError('Target library path must contain only absolute directories')
    result={key:value for key,value in base.items() if not key.startswith('LD_')}
    result['LD_LIBRARY_PATH']=library_path
    if trace: result['LD_TRACE_LOADED_OBJECTS']='1'
    return result


def dispatch(arguments,*,environ=None,execve=None):
    if len(arguments)<5: raise ValueError('Expected mode,target,hash,interpreter,library-path')
    mode,target,expected_hash,expected_interpreter,library_path,*extra=arguments
    if mode not in ('run','trace'): raise ValueError('Unknown exec mode')
    if mode=='trace' and extra: raise ValueError('Loader trace has no numerical arguments')
    if mode=='run' and (len(extra)!=3 or extra[0] not in ('1','6','8','9') or extra[1] not in ('4','5') or extra[2] not in ('0','1')):
        raise ValueError('Unexpected numerical arm')
    path=Path(target)
    if not path.is_absolute() or path.is_symlink(): raise ValueError('Target must be an absolute regular executable')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_hash: raise ValueError('Target bytes changed before execve')
    if interpreter(data)!=expected_interpreter: raise ValueError('Target PT_INTERP differs from bound donor')
    environment=target_environment(os.environ if environ is None else environ,library_path,mode=='trace')
    # No child, shell, direct-loader launch, or code after the real execve.
    return (os.execve if execve is None else execve)(str(path),[str(path),*extra],environment)


if __name__=='__main__': dispatch(sys.argv[1:])
