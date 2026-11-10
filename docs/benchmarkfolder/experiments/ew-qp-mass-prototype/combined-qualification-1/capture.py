#!/usr/bin/env python3
"""Retain completed mass qualification envelopes and exact UTF-8 journals."""
import gzip,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
BASE=ROOT/'.tirust-build/qp-mass-general-implementation/combined-qualification'
CONT=BASE/'loader-correction/continuation'
FIX=CONT/'additive-offline-analysis'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

def main():
    assert Path.cwd()==ROOT and not (HERE/'payload.json.gz').exists()
    files={};blobs={};native={};direct={}
    def key(p):
        p=Path(p);p=p if p.is_absolute() else ROOT/p
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    def add(p,expected=None):
        p=Path(p);p=p if p.is_absolute() else ROOT/p;name=key(p)
        if name in files or name in native:
            if expected:assert (files.get(name) or native[name])['sha256']==expected,name
            return
        assert p.is_file(),p
        with p.open('rb') as stream:head=stream.read(16)
        binary=head.startswith((b'\x7fELF',b'!<arch>\n')) or p.suffix in ('.o','.a','.so','.mod','.smod','.pyc','.pkl')
        if binary:
            digest=expected or sha(p)
            native[name]={'sha256':digest,'bytes':p.stat().st_size,'scope':'Identity only; native bytes not distributed. Hash is bound by original manifest/execution.'}
            return
        data=p.read_bytes();text=data.decode('utf-8');assert b'\0' not in data
        digest=hashlib.sha256(data).hexdigest()
        if expected:assert digest==expected,name
        files[name]={'sha256':digest,'bytes':len(data)};blobs[digest]=text
    def top(directory):
        for p in sorted(directory.iterdir()):
            if p.is_file() and p.suffix in ('.py','.json','.md','.log','.diff','.patch'):add(p)
    def manifest(path):
        add(path);data=json.loads(path.read_text())
        for name,entry in data.get('inputs',{}).items():
            if isinstance(entry,str):add(name,entry)
        return data
    for directory in (BASE,BASE/'loader-correction',CONT,CONT/'additive',BASE/'current-original-qp',FIX):top(directory)
    for p in sorted((FIX/'history').iterdir()):add(p)
    for name in ('prepared.json','loader-correction/prepared.json','loader-correction/continuation/prepared.json','loader-correction/continuation/additive/prepared.json','current-original-qp/prepared.json'):manifest(BASE/name)
    for directory in (BASE/'fixture',BASE/'amplitude'):top(directory)
    for reportpath in (CONT/'run.json',CONT/'additive/run.json'):
        report=json.loads(reportpath.read_text());add(reportpath)
        for event in report['jobs']:
            add(event['log'],event['log_sha256'])
            cwd=Path(event['cwd']);cwd=cwd if cwd.is_absolute() else ROOT/cwd
            for name,digest in event.get('output_files',{}).items():add(cwd/name,digest)
    child=json.loads((BASE/'current-original-qp/child-prepared.json').read_text())
    for field in ('frozen_parser','frozen_support'):add(child[field]['path'],child[field]['sha256'])
    for entry in child['historical_oracles'].values():
        for arm in ('original','candidate'):add(entry[arm]['path'],entry[arm]['sha256'])
    for counterpart in child['counterparts'].values():
        for name,digest in counterpart['input_sha256'].items():add(name,digest)
    for run in child['runs']:
        for name,digest in run['input_sha256'].items():add(name,digest)
    initial=ROOT/'docs/benchmarkfolder/experiments/qp-mass-general-preflight-private/index.json'
    add(initial,'a079a55fd2055948e5cb2c259135a4e20da6efa3a36de72aa9bcf8efb2680753')
    add(FIX/'source-index.json','4b2e0cd422cc094c47e6440b10e0cd7a586efaf91edbcdf1e5eb12432ef6823f')
    for source,target in [(CONT/'run.json','continuation-run.json'),(CONT/'additive/run.json','additive-failed-run.json'),(FIX/'warning-audit.json','warning-audit.json'),(FIX/'analysis-execution.json','offline-analysis-execution.json'),(FIX/'lineage.json','offline-lineage.json'),(FIX/'parent-path.patch','parent-path.patch')]:
        shutil.copyfile(source,HERE/target);direct[target]={'original':key(source),'sha256':sha(source)}
    payload={'schema':'mass-qualification-utf8-evidence-v1','files':files,'blobs':blobs}
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    (HERE/'payload.json.gz').write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
    dump(HERE/'payload-inventory.json',{'files':files,'unique_text_contents':len(blobs),'uncompressed_bytes':len(raw),'gzip_sha256':sha(HERE/'payload.json.gz')})
    dump(HERE/'native-identities.json',{'scope':'Native identities from pinned preparation/execution; no binaries or links shipped. This package does not re-evaluate native code.','files':native})
    dump(HERE/'direct-copies.json',direct)
    print(json.dumps({'text_paths':len(files),'unique_text_contents':len(blobs),'native_identities':len(native),'gzip_bytes':(HERE/'payload.json.gz').stat().st_size}))

if __name__=='__main__':main()
