#!/usr/bin/env python3
"""Capture only the terminal run5 and immutable controller; never follow DSOs."""
from pathlib import Path
import gzip
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
BASE=HERE.parent
ATTEMPT=ROOT/'.tirust-build/qcd-n3-genuine-qp-diagnostic/run5'
SOURCE=BASE/'source-proposal5'


def digest(raw):return hashlib.sha256(raw).hexdigest()
def dump(name,value):(HERE/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def main():
    if (HERE/'evidence-index.json').exists():raise FileExistsError('Closed retention is immutable')
    result=json.loads((ATTEMPT/'result.json').read_bytes())
    if result['passed'] is not True or result['inherited_workers']!=10 or result['new_workers_parsed']!=6:
        raise ValueError('Unexpected terminal result; preserve it with an appropriate failed-attempt collector')
    originals={};links={};directories=[]
    def capture(original,logical):
        if original.is_symlink():raise ValueError('No symlink target capture')
        raw=original.read_bytes()
        if raw.startswith(b'\x7fELF') or raw.startswith(b'!<arch>\n'):raise ValueError('No binary/archive capture')
        compressed=len(raw)>1000000;member=logical+('.gz' if compressed else '')
        output=gzip.compress(raw,compresslevel=9,mtime=0) if compressed else raw
        target=HERE/member;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.read_bytes()!=output:raise ValueError('Previously copied bytes changed')
        else:
            with target.open('xb') as stream:stream.write(output)
        originals[logical]={'original_path':str(original.relative_to(ROOT)),'member':member,
            'codec':'gzip' if compressed else 'identity','raw_bytes':len(raw),'raw_sha256':digest(raw),
            'stored_bytes':len(output),'stored_sha256':digest(output)}
    for p in sorted(ATTEMPT.rglob('*')):
        name=p.relative_to(ATTEMPT).as_posix()
        if p.is_symlink():links[name]=str(p.readlink())
        elif p.is_dir():directories.append(name)
        elif p.is_file():capture(p,'actual-run5/'+name)
        else:raise ValueError('Unknown attempt member type')
    source_index=json.loads((SOURCE/'source-index.json').read_bytes())
    for name in [*source_index['files'],'source-index.json']:capture(SOURCE/name,'controller/'+name)
    dump('retention-inventory.json',{'schema':'closed-qcd-n3-run5-retention-inventory-v1',
        'original_files':originals,'actual_attempt_symlinks':links,'actual_attempt_directories':directories,
        'symlink_targets_retained_as_metadata_only':True,'no_dso_or_executable_copies':True,
        'original_paths_are_metadata_not_replay_inputs':True})
    dump('prefix-dependency.json',{'schema':'fixed-closed-prefix-dependency-v1','relative_sibling':'failed-run4-retention-1',
        'index_file':'evidence-index.json','index_sha256':'fcd4f75340f7db9261e91d6f0c37e7a1ba85b3ff68b8b45a639ecbcc49ddf361',
        'scope':'Original failed run4 remains failed; ten native workers reinterpreted through sealed source5 parser'})
    import verify
    payload={name:(gzip.decompress((HERE/r['member']).read_bytes()) if r['codec']=='gzip' else
                   (HERE/r['member']).read_bytes()) for name,r in originals.items()}
    summary=verify.replay(payload,json.loads((HERE/'retention-inventory.json').read_bytes()),BASE/'failed-run4-retention-1')
    dump('summary.json',summary)
    files={}
    for p in sorted(HERE.rglob('*')):
        if p.is_file():
            raw=p.read_bytes();files[p.relative_to(HERE).as_posix()]={'bytes':len(raw),'sha256':digest(raw)}
    dump('evidence-index.json',{'schema':'qcd-n3-genuine-qp-run5-combined-retention-v1','files':files,
        'scope':'Ten inherited native workers plus six new; pure full sixteen-arm replay, unchecked coefficient consistency only',
        'required_prefix_index_sha256':'fcd4f75340f7db9261e91d6f0c37e7a1ba85b3ff68b8b45a639ecbcc49ddf361'})
    print(json.dumps(verify.verify(),sort_keys=True,indent=2))


if __name__=='__main__':main()
