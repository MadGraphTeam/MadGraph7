"""Capture closed text/gzip only. Never compile, load or execute unit programs."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent;OUT=HERE/'capsule'
PREFIX=ROOT/'.tirust-build'
SOURCES={
    'proposal':(PREFIX/'fortran-qp-output-bulk-proposal-1','index.json','600fb55cae10c8f105998c6ae689c09992526765734f33036dd79936261f5150'),
    'source2':(PREFIX/'fortran-qp-output-bulk-unit-controller-2','source-index.json','ad63b2e1128c70085fb282b1a74e1b84aa282c7e5be95514a79a558845817fe4'),
    'source3':(PREFIX/'fortran-qp-output-bulk-unit-controller-3','source-index.json','511225ea5bb4523cbf84012ea424a0710ef6f86b3ec6964ba8e8b41ea08a76d6'),
    'source4':(PREFIX/'fortran-qp-output-bulk-unit-controller-4','source-index.json','9d159331439f106df0a7571c69633c40b5900cb5c4162d6f9850521a3db8f219'),
}
WORKS={
    'build2':PREFIX/'fortran-qp-output-bulk-unit-work-2/build',
    'build3':PREFIX/'fortran-qp-output-bulk-unit-work-3/build',
    'run3':PREFIX/'fortran-qp-output-bulk-unit-work-3/run',
    'run4':PREFIX/'fortran-qp-output-bulk-unit-work-4/run',
}


def sha(raw):return hashlib.sha256(raw).hexdigest()
def identity(raw):return {'bytes':len(raw),'sha256':sha(raw)}
def read(p):return json.loads(Path(p).read_text())
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def main():
    assert not OUT.exists(),'Fresh capture only'
    OUT.mkdir();compressed={}
    def save(name,raw):
        raw.decode('utf-8');assert b'\0' not in raw
        if len(raw)>8192 and name not in {'verify_evidence.py','README.md','capture.py','qualification-summary.json'}:
            physical='compressed/'+name+'.gz';data=gzip.compress(raw,compresslevel=9,mtime=0)
            compressed[name]={'path':physical,'raw_bytes':len(raw),'raw_sha256':sha(raw)}
        else:physical=name;data=raw
        path=OUT/physical;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(data)
    groups={};origin={}
    for group,(base,index_name,index_sha) in SOURCES.items():
        seal_raw=(base/index_name).read_bytes();assert sha(seal_raw)==index_sha
        seal=json.loads(seal_raw);names={**seal['files'],index_name:identity(seal_raw)}
        groups[group]={'origin':str(base),'files':{},'binary_hashes':{}}
        for name,want in names.items():
            raw=(base/name).read_bytes();assert identity(raw)==want,name
            save(group+'/'+name,raw);groups[group]['files'][name]=identity(raw)
        origin[group]={'index_name':index_name,'index_sha256':index_sha}
    for group,base in WORKS.items():
        groups[group]={'origin':str(base),'files':{},'binary_hashes':{}}
        for path in sorted(base.rglob('*')):
            if not path.is_file():continue
            assert not path.is_symlink()
            name=str(path.relative_to(base));raw=path.read_bytes();meta=identity(raw)
            try:
                raw.decode('utf-8');assert b'\0' not in raw
                save(group+'/'+name,raw);groups[group]['files'][name]=meta
            except (UnicodeDecodeError,AssertionError):
                assert path.suffix in ('.o','.mod') or path.name=='check',name
                groups[group]['binary_hashes'][name]=meta
    # Preserve the source-defined native-error basis, not only an observation.
    for rel,target in [('vendor/tirust/src/lib.rs','production/bridge-lib.rs'),
                       ('vendor/tirust/mtirust.f90','production/mtirust.f90')]:
        raw=(ROOT/rel).read_bytes();save(target,raw)
    proof={}
    build=read(WORKS['build3']/'result.json')
    for variant,row in build['executables'].items():
        base=WORKS['build3']/variant;pre=read(base/'prelink.json')
        aliases=[]
        for spelling in re.findall(r'^LOAD\s+(.+)$',(base/'link.map').read_text(),re.M):
            q=Path(spelling);q=q if q.is_absolute() else base/q
            resolved=str(q.resolve());checksum=sha(q.read_bytes())
            assert row['actual_link_loads'][resolved]==pre['bound'][resolved]==checksum
            aliases.append({'spelling':spelling,'resolved':resolved,'sha256':checksum})
        runtime_aliases=[]
        for line in row['runtime']['listing'].splitlines():
            line=line.strip()
            if not line or re.fullmatch(r'linux-vdso\.so\.1\s+\(0x[0-9a-fA-F]+\)',line):continue
            match=re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)',line)
            assert match,line
            spelling=match[1];q=Path(spelling);resolved=str(q.resolve());checksum=sha(q.read_bytes())
            assert row['runtime']['files'][resolved]==pre['bound'][resolved]==checksum
            runtime_aliases.append({'spelling':spelling,'resolved':resolved,'sha256':checksum})
        proof[variant]={'map_aliases':aliases,'runtime_aliases':runtime_aliases,
                        'scope':'Post-run read-only alias resolution against unchanged recorded prebound and actual closures; binaries retained by hash only'}
    save('link-runtime-resolution.json',encoded(proof))
    save('closed-combined-summary.json',(PREFIX/'fortran-qp-output-bulk-unit-combined-summary-1.json').read_bytes())
    save('root-build-review.json',(PREFIX/'fortran-qp-output-bulk-root-build-review-1.json').read_bytes())
    save('capture.py',Path(__file__).read_bytes())
    save('verify_evidence.py',(HERE/'verify_evidence.py').read_bytes())
    save('README.md',(HERE/'README.md').read_bytes())
    retention={'schema':'madloop-tirust-bulk-unit-retention-v1','groups':groups,'sealed_sources':origin,
               'source3_failed_status_preserved':True,'native_or_compiler_executions_during_capture':0,
               'binaries':'Source/object/module/executable bytes are omitted; identities and closed link/runtime receipts retained',
               'publication_scope':'Recording C-provider parser qualification only; candidate not promoted; real physical/performance gates pending'}
    save('retention.json',encoded(retention))
    def index():
        files={str(p.relative_to(OUT)):{'kind':'file','size':p.stat().st_size,'sha256':sha(p.read_bytes())}
               for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='evidence-index.json'}
        data={'schema_version':1,'kind':'madloop-tirust-bulk-unit-qualification-v1','files':files,
              'compressed_text':compressed,'native_execution':False,'performance_acceptance':False}
        (OUT/'evidence-index.json').write_bytes(encoded(data))
    index()
    spec=importlib.util.spec_from_file_location('bulk_capture_verify',HERE/'verify_evidence.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    summary=verifier.verify(OUT,derive_summary=True)
    save('qualification-summary.json',encoded(summary));index()
    assert verifier.verify(OUT)==summary
    print(json.dumps({'index_sha256':sha((OUT/'evidence-index.json').read_bytes()),
                      'files':len(read(OUT/'evidence-index.json')['files'])+1,
                      'compressed_payloads':len(compressed),
                      'stored_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
                      'summary':summary},sort_keys=True,indent=2))


if __name__=='__main__':main()
