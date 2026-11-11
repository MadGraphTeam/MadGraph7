"""Portable replay, clean relocation, and reindexed corruption controls."""
import builtins
from contextlib import contextmanager,ExitStack
import ctypes
import gzip
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest import mock

# Preload every approved standard module before tests deny filesystem reads.
import replay_measurements
import verify_evidence

HERE=Path(__file__).resolve().parent
CAPSULE=HERE if (HERE/'evidence-index.json').is_file() else HERE/'package'
EXPECTED_INDEX=None
INDEX='evidence-index.json'


def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def snapshot(root):return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def reader(files,calls=None):
    def read(name):
        if calls is not None:calls.append(name)
        p=PurePosixPath(name)
        if not isinstance(name,str) or not name or '\\' in name or p.is_absolute() or '..' in p.parts or str(p)!=name or name not in files:
            raise AssertionError('Unindexed/outside capsule read:'+str(name))
        return files[name]
    read.paths=set(files)
    return read


@contextmanager
def guarded(root=None):
    attempts=[];root=Path(root).absolute() if root is not None else None
    original_open=builtins.open;original_io=io.open;original_os_open=os.open
    def denied(*args,**kwargs):attempts.append('native/process/mutation');raise AssertionError('Forbidden native/process/mutation API')
    def check(path,mode='r',flags=0):
        p=Path(path).absolute()
        if root is None or not p.is_relative_to(root) or any(c in str(mode) for c in 'wax+') or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            attempts.append('outside/mutation');raise AssertionError('Outside/mutating capsule access:'+str(path))
    def file_open(path,mode='r',*a,**k):check(path,mode);return original_open(path,mode,*a,**k)
    def io_open(path,mode='r',*a,**k):check(path,mode);return original_io(path,mode,*a,**k)
    def os_open(path,flags,*a,**k):check(path,flags=flags);return original_os_open(path,flags,*a,**k)
    with ExitStack() as stack:
        for name in ('Popen','run','call','check_call','check_output'):stack.enter_context(mock.patch.object(subprocess,name,denied))
        for name in ('CDLL','PyDLL','WinDLL','OleDLL'):
            if hasattr(ctypes,name):stack.enter_context(mock.patch.object(ctypes,name,denied))
        for name in ('system','fork','forkpty','posix_spawn','posix_spawnp','spawnv','spawnve','spawnvp','spawnvpe','execv','execve','execvp','execvpe','execl','execle','execlp','execlpe','kill','killpg','chdir','mkdir','rmdir','remove','unlink','rename','replace','symlink','link','truncate','chmod'):
            if hasattr(os,name):stack.enter_context(mock.patch.object(os,name,denied))
        stack.enter_context(mock.patch.object(builtins,'open',file_open));stack.enter_context(mock.patch.object(io,'open',io_open));stack.enter_context(mock.patch.object(os,'open',os_open))
        yield attempts


def replace_logical(files,name,value,update_origin=True):
    files=dict(files);index=json.loads(files[INDEX]);row=index['payloads'][name]
    stored=gzip.compress(value,compresslevel=9,mtime=0) if row['encoding']=='gzip+utf8' else value
    files[row['path']]=stored;row.update(sha256=sha(value),bytes=len(value));index['files'][row['path']].update(sha256=sha(stored),bytes=len(stored));files[INDEX]=encoded(index)
    if update_origin and name!='retention.json':
        meta=index['payloads']['retention.json'];raw=files[meta['path']];retention=json.loads(gzip.decompress(raw) if meta['encoding']=='gzip+utf8' else raw)
        if name in retention['origins']:
            retention['origins'][name].update(sha256=sha(value),bytes=len(value));files=replace_logical(files,'retention.json',encoded(retention),False)
    return files


def logical_bytes(files,name):
    row=json.loads(files[INDEX])['payloads'][name];raw=files[row['path']]
    return gzip.decompress(raw) if row['encoding']=='gzip+utf8' else raw


class CapsuleChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files=snapshot(CAPSULE);cls.index_sha=sha(cls.files[INDEX])
        if EXPECTED_INDEX is not None and cls.index_sha!=EXPECTED_INDEX:raise ValueError('Different expected index')
        with guarded() as attempts:
            cls.result=verify_evidence.verify(read_bytes=reader(cls.files),expected_index_sha256=cls.index_sha)
        if attempts:raise ValueError('Unexpected guarded replay operation')

    @classmethod
    def tearDownClass(cls):
        if snapshot(CAPSULE)!=cls.files:raise ValueError('Capsule changed during tests')

    def replay(self,files):
        with guarded() as attempts:result=verify_evidence.verify(read_bytes=reader(files))
        self.assertEqual(attempts,[])
        return result

    def test_full_callback_replay_without_filesystem_native_or_process(self):
        self.assertEqual(self.result['status'],'PASS')
        counts=self.result['summary']['witness_counts']
        self.assertEqual(counts['timed_amplitude_calls'],358400)
        self.assertEqual(counts['learning_traces'],8)
        self.assertEqual(counts['timed_batches'],28)
        self.assertFalse(self.result['performance_acceptance'])

    def test_wrong_index_stops_after_one_read(self):
        calls=[]
        with guarded(),self.assertRaises(ValueError):verify_evidence.verify(read_bytes=reader(self.files,calls),expected_index_sha256='0'*64)
        self.assertEqual(calls,[INDEX])

    def test_stored_corruption_missing_and_extra_rejected(self):
        name=next(iter(json.loads(self.files[INDEX])['files']))
        for mode in ('changed','missing','extra'):
            files=dict(self.files)
            if mode=='changed':files[name]+=b'corrupt'
            elif mode=='missing':del files[name]
            else:files['unexpected.txt']=b'unindexed\n'
            with self.assertRaises((ValueError,AssertionError,KeyError)):self.replay(files)

    def test_unsafe_and_duplicate_payload_aliases_rejected_early(self):
        for unsafe in ('../outside','/tmp/outside','a//outside','a\\outside',INDEX):
            files=dict(self.files);index=json.loads(files[INDEX]);first=next(iter(index['files']));index['files'][unsafe]=index['files'].pop(first);files[INDEX]=encoded(index);calls=[]
            with guarded(),self.assertRaises((ValueError,AssertionError)):verify_evidence.verify(read_bytes=reader(files,calls))
            self.assertEqual(calls,[INDEX])
        files=dict(self.files);index=json.loads(files[INDEX]);first=next(iter(index['payloads']));index['payloads']['duplicate.txt']=dict(index['payloads'][first]);files[INDEX]=encoded(index)
        with self.assertRaises(ValueError):self.replay(files)

    def test_reindexed_native_nul_and_false_claim_rejected(self):
        for native in (b'\x7fELF',b'!<arch>\n',b'bad\0nul'):
            files=replace_logical(self.files,'README.md',native)
            with self.assertRaises((ValueError,UnicodeError)):self.replay(files)
        files=dict(self.files);index=json.loads(files[INDEX]);index['performance_acceptance']=True;files[INDEX]=encoded(index)
        with self.assertRaises(ValueError):self.replay(files)

    def test_reindexed_raw_answer_and_clock_corruption_rejected(self):
        name='results/profile/baseline/cycle-1/dp-timed-exact.dat';lines=logical_bytes(self.files,name).decode().splitlines();fields=lines[0].split();fields[3]=f'{int(fields[3],16)^1:016X}';lines[0]=' '.join(fields)
        files=replace_logical(self.files,name,('\n'.join(lines)+'\n').encode())
        with self.assertRaises(ValueError):self.replay(files)
        name='results/profile/observed-2/cycle-1/dp-timers.tsv';text=logical_bytes(self.files,name).decode();fields=text.splitlines()[4].split();fields[3]=str(int(fields[3])+1);lines=text.splitlines();lines[4]=' '.join(fields)
        with self.assertRaises(ValueError):self.replay(replace_logical(self.files,name,('\n'.join(lines)+'\n').encode()))

    def test_reindexed_learning_stage_and_parser_corruption_rejected(self):
        for name in ('learning/parity/baseline.dat','results/profile/result.json','source/analyze.py'):
            value=logical_bytes(self.files,name)+b'\nchanged\n'
            with self.assertRaises((ValueError,json.JSONDecodeError)):self.replay(replace_logical(self.files,name,value))

    def test_clean_relocation_local_reader_only(self):
        with tempfile.TemporaryDirectory(prefix='host-repair-capsule-') as temp:
            relocated=Path(temp)/'relocated';shutil.copytree(CAPSULE,relocated)
            with guarded(relocated) as attempts:
                result=verify_evidence.verify(root=relocated,expected_index_sha256=self.index_sha)
            self.assertEqual(attempts,[]);self.assertEqual(result['summary'],self.result['summary'])

    def test_symlink_capsule_member_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='host-repair-symlink-') as temp:
            relocated=Path(temp)/'relocated';shutil.copytree(CAPSULE,relocated);name=next(iter(json.loads(self.files[INDEX])['files']));p=relocated/name;outside=Path(temp)/'outside';outside.write_bytes(p.read_bytes());p.unlink();p.symlink_to(outside)
            with guarded(relocated),self.assertRaises(ValueError):verify_evidence.verify(root=relocated)

    def test_process_native_and_mutation_guards_reject_probes(self):
        actions=(lambda:subprocess.run(['never-executed']),lambda:ctypes.CDLL(None),lambda:builtins.open('/tmp/never-written','w'),lambda:os.system('never-executed'),lambda:os.mkdir('/tmp/never-created'))
        with guarded() as attempts:
            for action in actions:
                with self.assertRaises(AssertionError):action()
        self.assertEqual(len(attempts),len(actions))


if __name__=='__main__':unittest.main(verbosity=2)
