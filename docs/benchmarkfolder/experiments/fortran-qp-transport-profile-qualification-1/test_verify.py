"""File-only closure tests; no compilers, loaders or scientific execution."""
import gzip
import importlib.util
import io
import json
from pathlib import Path
import shutil

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('portable_qp_verify', HERE/'verify.py')
V = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(V)


def write_index(root):
    files = {}
    for path in root.rglob('*'):
        if not path.is_file() or path == root/'index.json':
            continue
        name = path.relative_to(root).as_posix(); raw = path.read_bytes()
        row = {'bytes':len(raw),'sha256':V.sha(raw),'kind':'utf8-text'}
        if name.endswith('.gz'):
            decoded = gzip.decompress(raw)
            row.update(kind='gzip-utf8-text',decoded_path=name[:-3],decoded_bytes=len(decoded),decoded_sha256=V.sha(decoded))
        files[name] = row
    (root/'index.json').write_text(json.dumps({'schema':V.SCHEMA,'files':files,'file_count':len(files),
        'total_bytes':sum(row['bytes'] for row in files.values()),'performance_acceptance':False}))


@pytest.fixture(scope='module')
def capsule(tmp_path_factory):
    root = tmp_path_factory.mktemp('portable-qp-capsule')/'copy'
    shutil.copytree(HERE, root, ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','index.json'))
    # Nested immutable indexes must be copied, unlike a provisional root index.
    for path in HERE.rglob('index.json'):
        if path.parent != HERE:
            target = root/path.relative_to(HERE); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
    write_index(root)
    return root


def test_complete_replay_from_relocated_capsule_without_host_reads(capsule, monkeypatch):
    original = io.open
    def confined(path, *args, **kwargs):
        assert Path(path).is_relative_to(capsule), 'Attempted host filesystem read'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(io,'open',confined)
    result = V.verify(capsule)
    assert result['verified'] and result['historical_failures_retained'] == 2
    assert result['native_executions_by_verifier'] == 0 and result['performance_acceptance'] is False
    assert [m['anchors'] for m in result['modes']] == [2,2,2]


@pytest.mark.parametrize('change', ['anchor','flags','timer','shadow','extra_map','failure','parser','closure','protected'])
def test_rehashed_payload_still_fails_semantic_replay(capsule,monkeypatch,change):
    index,files,checksum = V.load_capsule(capsule); files = dict(files)
    if change == 'anchor':
        name='mode-0/raw/profile-values.dat'; lines=files[name].decode().splitlines()
        for i,line in enumerate(lines):
            if line.strip() and not line.startswith('#'):
                fields=line.split();fields[4]='0.0';lines[i]=' '.join(fields);break
        files[name]=('\n'.join(lines)+'\n').encode()
    elif change == 'flags':
        name='mode-1/raw/ieee-flags.dat'; lines=files[name].decode().splitlines()
        for i,line in enumerate(lines):
            if line.strip() and not line.startswith('#'):
                fields=line.split();fields[4]='1';lines[i]=' '.join(fields);break
        files[name]=('\n'.join(lines)+'\n').encode()
    elif change == 'timer':
        name='mode-1/raw/point-16.timers.tsv';lines=files[name].decode().splitlines()
        for i,line in enumerate(lines):
            if line.strip() and not line.startswith('#'):
                fields=line.split();fields[-1]=str(int(fields[-1])+1);lines[i]=' '.join(fields);break
        files[name]=('\n'.join(lines)+'\n').encode()
    elif change == 'shadow':
        name='mode-2/raw/point-16.shadow.hex';files[name]+=b'999999 0 FF\n'
    elif change == 'extra_map':
        name='mode-2/maps-after.json';data=json.loads(files[name]);path='/unadmitted/libextra.so'
        data['maps']+='123-456 r-xp 0 00:00 1 '+path+'\n';data['files'][path]='0'*64
        files[name]=json.dumps(data).encode()
    elif change == 'failure':
        name='failure-mode0/terminal.json';data=json.loads(files[name]);data['status']='passed';files[name]=json.dumps(data).encode()
    elif change == 'parser':
        files['protocol/tirust_benchmark_worker.py']+=b'\n# edited\n'
    elif change == 'closure':
        name='run-closure.json';data=json.loads(files[name]);data['total_recorded_warmup_public_calls']=300
        files[name]=json.dumps(data).encode()
    else:
        name='mode-0/result.json';data=json.loads(files[name]);data['protected']['files_checked']-=1
        files[name]=json.dumps(data).encode();terminal=json.loads(files['mode-0/terminal.json'])
        terminal['result_sha256']=V.sha(files[name]);files['mode-0/terminal.json']=json.dumps(terminal).encode()
    monkeypatch.setattr(V,'load_capsule',lambda _: (index,files,checksum))
    with pytest.raises((ValueError,KeyError)):
        V.verify(capsule)


def test_snapshot_rejects_outside_or_unindexed_reads():
    Snap=V.snapshot_paths({'a':b'hello'})
    assert Snap('/capsule/a').read_text() == 'hello'
    for path in ('/etc/passwd','/capsule/../etc/passwd','/capsule/missing'):
        with pytest.raises(ValueError): Snap(path).read_bytes()
    with pytest.raises(ValueError): Snap('/capsule/a').open('w')


def test_unpinned_analysis_cannot_execute_before_rejection(capsule,monkeypatch):
    index,files,checksum=V.load_capsule(capsule);files=dict(files)
    files['analyze.py']=files['analyze.py'].replace(b'def analyze(root):',
        b'def analyze(root):\n    raise AssertionError("untrusted function executed")')
    monkeypatch.setattr(V,'load_capsule',lambda _: (index,files,checksum))
    with pytest.raises(ValueError,match='index differs: analyze.py'):
        V.verify(capsule)


def test_typed_inventory_rejects_unindexed_file_symlink_and_bad_decoding(tmp_path):
    (tmp_path/'file.txt').write_text('hello');write_index(tmp_path)
    (tmp_path/'extra').write_text('extra')
    with pytest.raises(ValueError,match='membership'): V.load_capsule(tmp_path)
    (tmp_path/'extra').unlink();(tmp_path/'link').symlink_to('/etc/passwd')
    with pytest.raises(ValueError,match='symlink'): V.load_capsule(tmp_path)
    (tmp_path/'link').unlink();(tmp_path/'map.gz').write_bytes(gzip.compress(b'LOAD fixture.o\n'));write_index(tmp_path)
    index=json.loads((tmp_path/'index.json').read_text());index['files']['map.gz']['decoded_sha256']='0'*64
    (tmp_path/'index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError,match='Decoded member'): V.load_capsule(tmp_path)


def test_utf8_label_does_not_admit_binary_nul(tmp_path):
    (tmp_path/'binary.txt').write_bytes(b'valid utf8\0binary');write_index(tmp_path)
    with pytest.raises(ValueError,match='Binary NUL'): V.load_capsule(tmp_path)
