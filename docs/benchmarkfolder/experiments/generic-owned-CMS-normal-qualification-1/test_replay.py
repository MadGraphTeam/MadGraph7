"""Small portable tests; temporary mutation fixtures never execute native code."""
from contextlib import ExitStack
import builtins
import ctypes
import gzip
import _ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from unittest import mock

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def require(value,message):
    if not value:raise AssertionError(message)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root,name):
    spec=importlib.util.spec_from_file_location(name,root/'replay.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def rejected(call,label):
    try:call()
    except (ValueError,RuntimeError):return
    raise AssertionError(label+' was accepted')


class DeadmanReached(BaseException):pass
def deadman(*args,**kwargs):raise DeadmanReached('Effect escaped portable guard')


def guard_tests(module,root,outside):
    memory={str(p.resolve()):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    owned=(root/'README.md').resolve();wanted=memory[str(owned)];forbidden=outside/'must-not-exist'
    targets=[(builtins,'open'),(io,'open'),(Path,'read_bytes'),(Path,'read_text'),(Path,'exists'),
             (Path,'write_bytes'),(os,'mkdir'),(subprocess,'Popen'),(ctypes,'CDLL'),(_ctypes,'dlopen'),
             (os,'killpg'),(signal,'setitimer'),(socket,'socket')]
    with ExitStack() as stack:
        for obj,name in targets:stack.enter_context(mock.patch.object(obj,name,deadman))
        with module.blocked_apis(memory) as attempts:
            require(owned.read_bytes()==wanted and owned.read_text()==wanted.decode(),'Contained memory read changed')
            require(not attempts,'Contained reads attempted external effects')
            probes=[('external-read',lambda:forbidden.read_bytes()),('open',lambda:builtins.open(forbidden,'rb')),
                ('metadata',lambda:forbidden.exists()),('process',lambda:subprocess.Popen(['must-not-run'])),
                ('CDLL',lambda:ctypes.CDLL('must-not-load')),('dlopen',lambda:_ctypes.dlopen('must-not-load',0)),
                ('write',lambda:forbidden.write_bytes(b'bad')),('mkdir',lambda:os.mkdir(forbidden)),
                ('signal',lambda:os.killpg(999999,9)),('timer',lambda:signal.setitimer(signal.ITIMER_REAL,1)),
                ('network',lambda:socket.socket())]
            for label,call in probes:
                before=len(attempts)
                try:call()
                except DeadmanReached as error:raise AssertionError(label+' escaped guard') from error
                except RuntimeError:pass
                else:raise AssertionError(label+' allowed a forbidden operation')
                require(len(attempts)>before,label+' denial not recorded')
            count=len(attempts)
    require(not forbidden.exists(),'Forbidden path was created')
    return count


def main():
    root=HERE;expected=digest(root/'index.json');module=load(root,'cmsnormal_portable_test_original')
    require(module.verify(root,expected)['verified'],'Original replay failed')
    with tempfile.TemporaryDirectory(prefix='cmsnormal-portable-',dir='/tmp') as tmp:
        scratch=Path(tmp);relocated=scratch/'relocated';shutil.copytree(root,relocated)
        require(load(relocated,'cmsnormal_portable_test_relocated').verify(relocated,expected)['verified'],'Relocated replay failed')
        damaged=scratch/'payload';shutil.copytree(root,damaged)
        index=json.loads((damaged/'index.json').read_text());name=index['logical_files']['run/evidence/candidate-controls/driver.raw64']['path']
        path=damaged/name;path.write_bytes(path.read_bytes()+b'BAD')
        rejected(lambda:module.verify(damaged,expected),'Changed raw container')
        changed=scratch/'claims';shutil.copytree(root,changed);idx=json.loads((changed/'index.json').read_text())
        idx['claims']['completed_outer_calls']+=1;(changed/'index.json').write_text(json.dumps(idx,sort_keys=True,indent=2)+'\n')
        rejected(lambda:module.verify(changed,digest(changed/'index.json')),'False rehashed claim')
        validity=scratch/'invented-validity';shutil.copytree(root,validity);idx=json.loads((validity/'index.json').read_text())
        idx['claims']['all_requests_numerically_valid']=True
        (validity/'index.json').write_text(json.dumps(idx,sort_keys=True,indent=2)+'\n')
        rejected(lambda:module.verify(validity,digest(validity/'index.json')),'Invented numerical-validity claim')
        mapping=scratch/'raw-map';shutil.copytree(root,mapping);idx=json.loads((mapping/'index.json').read_text())
        entry=idx['logical_files']['run/evidence/candidate-controls/maps-ready.json'];path=mapping/entry['path']
        raw=gzip.decompress(path.read_bytes()) if entry['compression']=='gzip' else path.read_bytes();mapped=json.loads(raw)
        mapped['raw']=mapped['raw'].replace('/nix/store/','/nix/bogus/',1);raw=(json.dumps(mapped,sort_keys=True,indent=2)+'\n').encode()
        payload=gzip.compress(raw,mtime=0) if entry['compression']=='gzip' else raw;path.write_bytes(payload)
        entry['raw_bytes']=len(raw);entry['raw_sha256']=hashlib.sha256(raw).hexdigest()
        idx['files'][entry['path']]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
        (mapping/'index.json').write_text(json.dumps(idx,sort_keys=True,indent=2)+'\n')
        rejected(lambda:module.verify(mapping,digest(mapping/'index.json')),'Rehashed raw kernel-map mismatch')
        native=scratch/'native-identity';shutil.copytree(root,native);idx=json.loads((native/'index.json').read_text())
        idx['excluded_binaries']['build/legacy/check']['sha256']='0'*64
        (native/'index.json').write_text(json.dumps(idx,sort_keys=True,indent=2)+'\n')
        rejected(lambda:module.verify(native,digest(native/'index.json')),'Changed rehashed ELF identity')
        linked=scratch/'symlink';shutil.copytree(root,linked);(linked/'bad-link').symlink_to('/must-not-be-read')
        rejected(lambda:module.verify(linked,expected),'Symlink member')
        attempts=guard_tests(module,root,scratch)
    print(json.dumps({'tests_passed':9,'in_place':True,'relocation':True,'payload_tamper_rejected':True,
        'false_claim_rejected':True,'native_identity_tamper_rejected':True,'symlink_rejected':True,
        'invented_numerical_validity_rejected':True,'raw_kernel_map_mismatch_rejected':True,
        'intentional_guard_denials':attempts,'new_native_executions':0,'public_writes':False},sort_keys=True))


if __name__=='__main__':main()
