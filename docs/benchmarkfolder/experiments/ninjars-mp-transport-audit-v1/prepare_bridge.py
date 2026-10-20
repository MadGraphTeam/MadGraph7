#!/usr/bin/env python3
"""Prepare an isolated vendor crate with widening and tests only; no compile/run."""
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
WORK=ROOT/'.benchmark-work/experiments/ninjars-mp-transport-audit-v1'
BRIDGE=WORK/'bridge'
REVISION='f4bffc61ed67e5d083b80ff9901768d4d5779989'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert Path.cwd()==ROOT
    green=json.loads((HERE/'scalar-result.json').read_text())
    assert green['complete'] and green['engine_calls']==0
    BRIDGE.mkdir(parents=True,exist_ok=False)
    (HERE/'original').mkdir()
    originals={}
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',REVISION,'vendor/ninjars'],text=True,cwd=ROOT).splitlines()
    for name in names:
        relative=Path(name).relative_to('vendor/ninjars')
        data=subprocess.check_output(['git','show',REVISION+':'+name],cwd=ROOT)
        assert (ROOT/name).read_bytes()==data,('numerical source changed',name)
        for destination in (BRIDGE/relative,HERE/'original'/relative):
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
        originals[str(relative)]=hashlib.sha256(data).hexdigest()
    original=(BRIDGE/'src/lib.rs').read_text()
    old=(HERE/'old-parser.rs').read_text().rstrip('\n')
    new=(HERE/'proposed-parser.rs').read_text().rstrip('\n')
    assert original.count(old)==1
    updated=original.replace(old,new)
    updated+='\n#[cfg(test)]\nmod transport_contract_tests;\n#[cfg(test)]\nmod transport_replay_tests;\n'
    (BRIDGE/'src/lib.rs').write_text(updated)
    test='''use super::*;
fn record(text:&str)->[u8;RECORD] {
    assert!(text.len()<=RECORD);
    let mut out=[b' ';RECORD];out[..text.len()].copy_from_slice(text.as_bytes());out
}
'''+(HERE/'proposed_tests.rs').read_text()
    (BRIDGE/'src/transport_contract_tests.rs').write_text(test)
    (BRIDGE/'src/transport_replay_tests.rs').write_bytes((HERE/'transport_replay_tests.rs').read_bytes())
    (HERE/'bridge.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),
        fromfile='original/src/lib.rs',tofile='proposed/src/lib.rs')))
    (HERE/'proposed').mkdir()
    proposed={}
    for path in sorted(BRIDGE.rglob('*')):
        if path.is_file():
            relative=path.relative_to(BRIDGE)
            target=HERE/'proposed'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
            proposed[str(relative)]=sha(path)
    assert {name for name in originals if originals[name]!=proposed[name]}=={'src/lib.rs'}
    frozen=HERE.parent/'ninjars-mp-tensor-allocation-v1'
    manifest=json.loads((frozen/'manifest.json').read_text())
    provenance_path=frozen/'provenance.json'
    expected=manifest['files']['provenance.json']
    digest=expected['sha256'] if isinstance(expected,dict) else expected
    assert sha(provenance_path)==digest
    capture=json.loads(provenance_path.read_text())['capture']
    source=ROOT/capture['path'];assert sha(source)==capture['sha256']
    with gzip.open(source,'rb') as compressed,(WORK/'actual-mp-requests.jsonl').open('xb') as out:
        while block:=compressed.read(1024*1024):out.write(block)
    assert sha(WORK/'actual-mp-requests.jsonl')==capture['uncompressed_sha256']
    with (HERE/'bridge-prepared.json').open('x') as out:
        json.dump({'complete':True,'compiled':False,'executed':False,'production_modified':False,
            'numerical_revision':REVISION,'original_files':originals,'proposed_files':proposed,
            'only_existing_source_changed':'src/lib.rs','extra_files':['src/transport_contract_tests.rs','src/transport_replay_tests.rs'],
            'capture':capture,'scalar_result_sha256':sha(HERE/'scalar-result.json'),
            'prepare_sha256':sha(Path(__file__))},out,indent=2);out.write('\n')

if __name__=='__main__':main()
