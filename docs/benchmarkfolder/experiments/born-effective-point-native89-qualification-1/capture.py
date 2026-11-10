#!/usr/bin/env python3
"""Capture closed real-Z backend8/9 controls as exact text; never run native code."""
import gzip,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
WORK=ROOT/'.tirust-build/born-effective-runtime/native89-controller2'
PRIOR=ROOT/'docs/benchmarkfolder/experiments/born-effective-point-native'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def main():
    assert Path.cwd()==ROOT and not (HERE/'payload.json.gz').exists()
    assert sha(WORK/'source-index.json')=='55af12e109954110be3326e1a0d52dd28cb51bdf419585e2c12ffbd2fd1c03b4'
    assert sha(PRIOR/'evidence-index.json')=='923f9028c8aca151672cbe932a0e6321607b0545f8c8af569ad491c2bc0291c9'
    files={};blobs={};direct={}
    def add(name,data):
        text=data.decode('utf-8');assert b'\0' not in data and not data.startswith((b'\x7fELF',b'!<arch>\n'))
        h=hashlib.sha256(data).hexdigest()
        if name in files:assert files[name]['sha256']==h;return
        files[name]={'sha256':h,'bytes':len(data)};blobs[h]=text
    def actual(path):
        assert not path.is_symlink();add(str(path.relative_to(ROOT)),path.read_bytes())
    for path in sorted(WORK.rglob('*')):
        if path.is_file():
            assert '__pycache__' not in path.parts and path.suffix not in ('.o','.a','.so','.mod','.pyc')
            actual(path)
    prepared=json.loads((WORK/'prepared.json').read_text())
    # The original milestone already binds the complete implementation and all
    # generated exports. Preserve its exact real-Z source and generator text
    # here too, so this package's parser/source proof is independently readable.
    old=json.loads(gzip.decompress((PRIOR/'sources.json.gz').read_bytes()))
    prefixes=('.tirust-build/born-effective-implementation/',
              '.tirust-build/born-effective-exporter-addendum/',
              '.tirust-build/general-kinematics-consumer/',
              '.tirust-build/qp-mass-general-implementation/combined-qualification/src/',
              '.tirust-build/born-effective-runtime/attempt1/exports/real_uux_z/')
    for name,record in old['files'].items():
        if name.startswith(prefixes):add(name,old['blobs'][record['sha256']].encode('utf-8'))
    for name in prepared['source_bindings']:
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        if path.suffix not in ('.so',) and path.read_bytes()[:4]!=b'\x7fELF':
            if path.is_relative_to(ROOT):add(str(path.relative_to(ROOT)),path.read_bytes())
    for command in prepared['commands']:
        for name,want in command['inputs'].items():
            if name.startswith('Cards/'):continue
            path=ROOT/command['cwd']/name;assert sha(path)==want;actual(path)
    native={}
    for name,want in (prepared['input_bindings']|prepared['source_bindings']).items():
        if name not in files:native[name]={'sha256':want,'scope':'Original bound dependency/artifact identity; bytes not distributed here.'}
    for stage,count in (('ordinary',8),('fixed-qp',16)):
        path=WORK/('execution-'+stage)/'summary.json';data=json.loads(path.read_text())
        assert data['state']=='complete' and len(data['runs'])==count and not data['changed_bound_files']
        target=stage+'-summary.json';shutil.copyfile(path,HERE/target);direct[target]={'original':str(path.relative_to(ROOT)),'sha256':sha(path)}
    for name in ('prepared.json','source-index.json','lineage.json'):
        target='native89-'+name;shutil.copyfile(WORK/name,HERE/target);direct[target]={'original':str((WORK/name).relative_to(ROOT)),'sha256':sha(WORK/name)}
    dump(HERE/'prior-milestone.json',{'path':str(PRIOR.relative_to(ROOT)),'index_sha256':sha(PRIOR/'evidence-index.json'),
        'source_bundle_sha256':sha(PRIOR/'sources.json.gz'),'scope':'Prior milestone remains unchanged. Relevant exact generated/source text is copied into this independent payload.'})
    value={'schema':'real-z-born-native89-utf8-v1','files':files,'blobs':blobs}
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode();(HERE/'payload.json.gz').write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
    dump(HERE/'payload-inventory.json',{'files':files,'unique_text_contents':len(blobs),'uncompressed_bytes':len(raw),'gzip_sha256':sha(HERE/'payload.json.gz')})
    dump(HERE/'other-bound-identities.json',{'scope':'Dependency and artifact identities not embedded as text; native bytes and symlinks excluded. Full prepared bindings remain unchanged.','files':native})
    dump(HERE/'direct-copies.json',direct)
    print(json.dumps({'text_paths':len(files),'unique_contents':len(blobs),'other_bound_identities':len(native),'gzip_bytes':(HERE/'payload.json.gz').stat().st_size}))

if __name__=='__main__':main()
