"""Independent file-only capsule tests; run Python -B, pytest cache disabled.

The initial complete replay is shared. Decoded gzip bytes are memoized across
tests, avoiding repeated inflation of the114MB native wire text. All mutations
use memory or a disposable /tmp copy; the original capsule is never written.
"""
import builtins
from contextlib import contextmanager
import ctypes
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import types

import pytest

HERE = Path(__file__).resolve().parent
CAPSULE = HERE/"capsule" if (HERE/"capsule").is_dir() else HERE
INDEX = "evidence-index.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()}


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2)+"\n").encode()


def reader(files,calls=None):
    def read(name):
        if calls is not None:calls.append(name)
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or str(p)!=name or name not in files:
            raise RuntimeError('Attempted outside/unindexed read: '+name)
        return files[name]
    return read


@contextmanager
def no_live_reads():
    def blocked(*args,**kwargs):
        raise RuntimeError('In-memory replay attempted a live file read')
    with pytest.MonkeyPatch.context() as patch:
        for name in ('read_bytes','read_text'):patch.setattr(Path,name,blocked)
        patch.setattr(builtins,'open',blocked);patch.setattr(io,'open',blocked)
        yield


def assert_summary(result):
    assert result['status']=='passed'
    assert result['learning_workers']==5 and result['learning_points']==500
    assert result['post_learning_parity_calls']==306
    assert result['all_public_coefficient_accuracy_status_IEEE_words_equal'] is True
    assert result['original_wire_observer']['native_QP_calls']==3248
    assert result['original_wire_observer']['raw_records']==818720
    assert result['timing_batches']==36 and result['timing_calls_including_calibration']==260
    assert result['candidate_promoted'] is result['performance_acceptance'] is False
    assert result['portable_verifier_native_executions']==0


@pytest.fixture(scope='session')
def packet():
    files=snapshot(CAPSULE)
    verifier_path=HERE/'verify_evidence.py'
    module=types.ModuleType('independent_physical_capsule_verifier')
    module.__file__=str(verifier_path)
    exec(compile(verifier_path.read_bytes(),str(verifier_path),'exec'),module.__dict__)
    patch=pytest.MonkeyPatch()
    def forbidden(*args,**kwargs):raise AssertionError('Process/native execution forbidden')
    for name in ('Popen','run','call','check_call','check_output'):patch.setattr(subprocess,name,forbidden)
    for name in ('CDLL','PyDLL','WinDLL','OleDLL'):
        if hasattr(ctypes,name):patch.setattr(ctypes,name,forbidden)
    for name in ('system','fork','forkpty','posix_spawn','posix_spawnp','spawnv','spawnve','spawnvp','spawnvpe','execv','execve','execvp','execvpe'):
        if hasattr(os,name):patch.setattr(os,name,forbidden)
    decompress=gzip.decompress;decoded={}
    def cached(raw):
        if raw not in decoded:decoded[raw]=decompress(raw)
        return decoded[raw]
    patch.setattr(gzip,'decompress',cached)
    index_sha=sha(files[INDEX])
    with no_live_reads():
        summary=module.verify(read_bytes=reader(files),derive_summary=False,expected_index_sha256=index_sha)
    assert_summary(summary)
    yield types.SimpleNamespace(files=files,verify=module.verify,summary=summary,index_sha=index_sha)
    patch.undo()
    assert snapshot(CAPSULE)==files,'Original capsule changed during independent tests'


def test_blocked_callback_full_replay(packet):
    assert_summary(packet.summary)


def test_expected_index_mismatch_rejected_first(packet):
    calls=[]
    with pytest.raises(ValueError,match='Different package index'):
        packet.verify(read_bytes=reader(packet.files,calls),derive_summary=True,expected_index_sha256='0'*64)
    assert calls==[INDEX]


def test_payload_byte_mutation_rejected(packet):
    files=dict(packet.files);index=json.loads(files[INDEX]);name=next(iter(index['files']))
    raw=files[name];files[name]=(bytes([raw[0]^1])+raw[1:]) if raw else b'mutation'
    with pytest.raises(ValueError,match='Changed package payload'):
        packet.verify(read_bytes=reader(files),derive_summary=True)


def test_reindexed_gzip_decoded_identity_mismatch(packet):
    files=dict(packet.files);index=json.loads(files[INDEX]);logical=next(iter(index['compressed_text']))
    row=index['compressed_text'][logical];path=row['path']
    files[path]=gzip.compress(gzip.decompress(files[path])+b' ',mtime=0)
    index['files'][path]['size']=len(files[path]);index['files'][path]['sha256']=sha(files[path])
    files[INDEX]=encoded(index)
    with pytest.raises(ValueError,match='Changed decoded gzip text'):
        packet.verify(read_bytes=reader(files),derive_summary=True)


def test_unsafe_and_self_index_names_rejected_before_payload_reads(packet):
    for unsafe in ('/tmp/outside','../outside','a/../outside','a//outside','a\\outside',INDEX):
        files=dict(packet.files);index=json.loads(files[INDEX]);old=next(iter(index['files']))
        index['files'][unsafe]=index['files'].pop(old);files[INDEX]=encoded(index);calls=[]
        with pytest.raises(ValueError):packet.verify(read_bytes=reader(files,calls),derive_summary=True)
        assert calls==[INDEX]


def test_compressed_alias_rejected_before_payload_reads(packet):
    files=dict(packet.files);index=json.loads(files[INDEX]);first=next(iter(index['compressed_text']))
    index['compressed_text']['unrelated-alias.txt']=dict(index['compressed_text'][first])
    files[INDEX]=encoded(index);calls=[]
    with pytest.raises(ValueError,match='Aliased or colliding'):
        packet.verify(read_bytes=reader(files,calls),derive_summary=True)
    assert calls==[INDEX]


def test_missing_indexed_file_rejected(packet):
    index=json.loads(packet.files[INDEX]);name=next(iter(index['files']))
    def missing(relative):
        if relative==name:raise FileNotFoundError(relative)
        return packet.files[relative]
    with pytest.raises((ValueError,FileNotFoundError)):
        packet.verify(read_bytes=missing,derive_summary=True)


def test_symlink_root_rejected(packet):
    with tempfile.TemporaryDirectory(prefix='physical-capsule-link-',dir='/tmp') as temp:
        alias=Path(temp)/'alias';alias.symlink_to(CAPSULE,target_is_directory=True)
        with pytest.raises(ValueError,match='Symlink package root'):
            packet.verify(alias,derive_summary=True)


def test_additional_file_rejected(packet):
    with tempfile.TemporaryDirectory(prefix='physical-capsule-extra-',dir='/tmp') as temp:
        copy=Path(temp)/'packet';shutil.copytree(CAPSULE,copy)
        (copy/'unindexed-extra.txt').write_bytes(b'extra\n')
        with pytest.raises(ValueError,match='inventory changed'):
            packet.verify(copy,derive_summary=True)


def test_relocated_full_replay_blocks_recorded_live_paths(packet):
    with tempfile.TemporaryDirectory(prefix='physical-capsule-relocated-',dir='/tmp') as temp:
        copy=Path(temp)/'packet';shutil.copytree(CAPSULE,copy)
        original_bytes=Path.read_bytes;original_text=Path.read_text
        def allowed(path):
            path=Path(path).absolute()
            if not path.is_relative_to(copy) or path.is_symlink():
                raise RuntimeError('Relocation attempted a recorded live path: '+str(path))
        def read_bytes(path,*a,**k):allowed(path);return original_bytes(path,*a,**k)
        def read_text(path,*a,**k):allowed(path);return original_text(path,*a,**k)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path,'read_bytes',read_bytes);patch.setattr(Path,'read_text',read_text)
            result=packet.verify(copy,derive_summary=False,expected_index_sha256=packet.index_sha)
        assert result==packet.summary
        assert snapshot(copy)==packet.files


def test_reindexed_observation_corruption_remains_origin_bound(packet):
    files=dict(packet.files);index=json.loads(files[INDEX]);logical='parity/original-parity/warmup.json'
    row=index['compressed_text'][logical];path=row['path'];value=json.loads(gzip.decompress(files[path]))
    value['rows'][0]['status']=216;raw=encoded(value);files[path]=gzip.compress(raw,mtime=0)
    row['raw_bytes']=len(raw);row['raw_sha256']=sha(raw)
    index['files'][path]['size']=len(files[path]);index['files'][path]['sha256']=sha(files[path]);files[INDEX]=encoded(index)
    with pytest.raises(ValueError,match='Original retained bytes changed'):
        packet.verify(read_bytes=reader(files),derive_summary=True)


def test_runtime_proof_and_scope_remain_recorded_only(packet):
    assert packet.summary['binary_bytes_retained'] is False
    assert 'no fresh native verification' in packet.summary['runtime_proof']
    assert 'No convincing useful measured benefit' in packet.summary['decision']
