"""Standalone EW capsule tests; only contained text replay is evaluated."""
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
    root=HERE;expected=digest(root/'index.json');module=load(root,'EW_final_capsule_test')
    require(module.verify(root,expected)['verified'],'Original final EW replay failed')
    with tempfile.TemporaryDirectory(prefix='EW-final-portable-',dir='/tmp') as tmp:
        scratch=Path(tmp);relocated=scratch/'relocated';shutil.copytree(root,relocated)
        require(load(relocated,'EW_final_capsule_relocated').verify(relocated,expected)['verified'],'Relocated final EW replay failed')
        raw=scratch/'raw';shutil.copytree(root,raw);index=json.loads((raw/'index.json').read_text())
        state=json.loads((raw/'ordinary-section/collection-state.json').read_text());entry=state['logical_files']['ordinary/candidate/exact-words.dat'];path=raw/'ordinary-section'/entry['path'];path.write_bytes(path.read_bytes()+b'BAD')
        rejected(lambda:module.verify(raw,expected),'Changed ordinary raw container')
        for label,key in [('invented-QP','changed_QP_mass_path_observed'),('invented-truth','independent_absolute_accuracy_qualified')]:
            changed=scratch/label;shutil.copytree(root,changed);index=json.loads((changed/'index.json').read_text());index['claims'][key]=True;(changed/'index.json').write_text(json.dumps(index,sort_keys=True,indent=2)+'\n')
            rejected(lambda changed=changed:module.verify(changed,digest(changed/'index.json')),'Rehashed '+label)
        reviewed=scratch/'review';shutil.copytree(root,reviewed);index=json.loads((reviewed/'index.json').read_text());entry=index['review_text_members']['reviews/ordinary-pair/receipt.json'];path=reviewed/entry['path'];rawdata=gzip.decompress(path.read_bytes()) if entry['compression']=='gzip' else path.read_bytes();obj=json.loads(rawdata);obj['qualification']['changed_QP_mass_path']=True
        rawdata=(json.dumps(obj,sort_keys=True,indent=2)+'\n').encode();payload=gzip.compress(rawdata,mtime=0) if entry['compression']=='gzip' else rawdata;path.write_bytes(payload);entry['raw_sha256']=hashlib.sha256(rawdata).hexdigest();entry['raw_bytes']=len(rawdata);index['files'][entry['path']]={'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)};(reviewed/'index.json').write_text(json.dumps(index,sort_keys=True,indent=2)+'\n')
        rejected(lambda:module.verify(reviewed,digest(reviewed/'index.json')),'Rehashed closed review change')
        linked=scratch/'symlink';shutil.copytree(root,linked);(linked/'bad-link').symlink_to('/must-not-be-read');rejected(lambda:module.verify(linked,expected),'Final EW symlink')
        denials=guard_tests(module,root,scratch)
        memory={str(p.resolve()):p.read_bytes() for p in root.rglob('*') if p.is_file()};directories={str(root.resolve())}|{str(p.resolve()) for p in root.rglob('*') if p.is_dir()}
        with module.contained_reads(memory,directories) as attempts:
            require((root/'README.md').read_bytes()==memory[str(root/'README.md')] and root.resolve()==root and bool(list(root.rglob('*'))),'Contained root metadata/read failed')
            for probe in (lambda:Path('/outside-capsule').resolve(),lambda:Path('/outside-capsule').is_file(),lambda:list(root.rglob('*.py'))):
                before=len(attempts);rejected(probe,'Outside/nonexact metadata operation');require(len(attempts)>before,'Outer metadata denial not recorded')
            metadata_denials=len(attempts)
    print(json.dumps({'tests_passed':9,'relocation':True,'raw_tamper_rejected':True,'invented_QP_coverage_rejected':True,'invented_absolute_accuracy_rejected':True,
        'closed_review_tamper_rejected':True,'symlink_rejected':True,'intentional_effect_denials':denials,'intentional_metadata_denials':metadata_denials,
        'new_native_executions':0,'public_writes':False},sort_keys=True))

if __name__=='__main__':main()
