"""Capture closed physical observations as UTF-8/gzip; never invoke native code."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path.cwd(); HERE=Path(__file__).resolve().parent; OUT=HERE/'capsule'
PREFIX=ROOT/'.tirust-build'
SOURCE=PREFIX/'fortran-qp-output-bulk-physical-proposal-1'
WORK=PREFIX/'fortran-qp-output-bulk-physical-work-1'
UNIT=PREFIX/'fortran-qp-output-bulk-unit-retention-1/capsule'
SOURCE_PIN='8e19ec4f1e422e5022cb3161cfe248bbc3a11cc6a2e3912e3693718ccea571ce'
UNIT_PIN='d0d816c0bb7ebe4cd5b8e051aaf206552597c65c8a1e26fdf6d678d8ae685546'
VARIANTS=('original-parity','candidate-parity','wire-parity','original-timing','candidate-timing')
AUDITS={
 'build':('fortran-qp-output-bulk-physical-build-independent-audit-1','d1f5802b8bf51cba5740513de3620c6431b5f67e728a5189c4bbb086c69dba36'),
 'parity':('fortran-qp-output-bulk-physical-parity-independent-audit-1','e5a3577d3a68dabf2226728844c968d97c54e6dc678cee5a1a2915c36d61d4aa'),
}

def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def load(path):return json.loads(path.read_text())
def identity(raw):return {'bytes':len(raw),'sha256':sha(raw)}

def main():
    assert not OUT.exists(),'Fresh capture only'
    OUT.mkdir(); packed={}; origins={}
    def save(name,raw):
        raw.decode('utf-8');assert b'\0' not in raw and not raw.startswith(b'\x7fELF')
        if len(raw)>8192 and name not in {'verify_evidence.py','README.md','capture.py','test_verify.py','qualification-summary.json'}:
            target='compressed/'+name+'.gz';data=gzip.compress(raw,compresslevel=9,mtime=0)
            packed[name]={'path':target,'raw_bytes':len(raw),'raw_sha256':sha(raw)}
        else:target=name;data=raw
        path=OUT/target;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(data)
    def copy(name,path,want=None):
        raw=path.read_bytes()
        if want is not None:assert sha(raw)==want,(name,'Changed original')
        if name in origins:
            assert identity(raw)=={k:origins[name][k] for k in ('bytes','sha256')}
            return
        save(name,raw);origins[name]={'original_path':str(path),'sha256':sha(raw),'bytes':len(raw)}
    seal=(SOURCE/'source-index.json').read_bytes();assert sha(seal)==SOURCE_PIN
    for name,want in {**json.loads(seal)['files'],'source-index.json':identity(seal)}.items():
        raw=(SOURCE/name).read_bytes();assert identity(raw)==want;save('source/'+name,raw)
    inputs=load(SOURCE/'input-pins.json');case=Path(inputs['case_path'])
    for name,want in inputs['original_sources'].items():copy('donor/'+name,case/name,want)
    for name in ('SubProcesses/P0_uux_zgg/makefile','SubProcesses/make_opts','compilation.json','generation.json'):
        path=case/name
        if path.exists():copy('donor/'+name,path,inputs['files'].get(str(path.relative_to(ROOT))))
    for name,want in inputs['expected_numerical_cards'].items():copy('donor/'+name,case/name,want)
    # Physical card includes unprofiled filters/defaults; these are frozen separately.
    for name in ('Cards/param_card.dat','Cards/MadLoopParams.dat'):
        if not ('donor/'+name) in origins:copy('donor/'+name,case/name)
    copy('inputs/points.dat',Path(inputs['point_path']),inputs['point_sha256'])
    aliases={}
    for phase in ('build','parity','timing'):
        base=WORK/phase
        for path in sorted(base.iterdir()):
            if path.is_file():copy(phase+'/'+path.name,path)
        if phase=='build':
            for path in sorted((base/'commands').iterdir()):copy('build/commands/'+path.name,path)
            build=load(base/'result.json')
            for variant in VARIANTS:
                proc=base/variant/'SubProcesses/P0_uux_zgg';row=build['executables'][variant]
                for name in ('prelink.json','private-link.map','makefile','mtirust.f90','check_sa.f'):
                    copy('build/'+variant+'/'+name,proc/name)
                if variant=='wire-parity':copy('build/'+variant+'/wire_capture.c',proc/'wire_capture.c')
                for name in inputs['expected_numerical_cards']:
                    copy('build/'+variant+'/'+name,base/variant/name)
                pre=load(proc/'prelink.json');mapping=[]
                for spelling in re.findall(r'^LOAD\s+(.+)$',(proc/'private-link.map').read_text(),re.M):
                    q=Path(spelling);q=q if q.is_absolute() else proc/q
                    resolved=str(q.resolve());want=row['actual_link_loads'][resolved]
                    assert pre['bound'][resolved]==want
                    mapping.append({'spelling':spelling,'resolved':resolved,'sha256':want})
                runtime=[]
                for line in row['runtime']['listing'].splitlines():
                    line=line.strip()
                    if not line or re.fullmatch(r'linux-vdso\.so\.1\s+\(0x[0-9a-fA-F]+\)',line):continue
                    m=re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)',line);assert m,line
                    resolved=str(Path(m[1]).resolve());want=row['runtime']['files'][resolved];assert pre['bound'][resolved]==want
                    runtime.append({'spelling':m[1],'resolved':resolved,'sha256':want})
                aliases[variant]={'map_aliases':mapping,'runtime_aliases':runtime}
        else:
            for variant in (('original-parity','candidate-parity','wire-parity') if phase=='parity' else ('original','candidate')):
                base2=base/variant
                for path in sorted(base2.rglob('*')):
                    if path.is_file() and 'export' not in path.relative_to(base2).parts:
                        copy(phase+'/'+variant+'/'+str(path.relative_to(base2)),path)
    save('link-runtime-resolution.json',encoded(aliases))
    # Compact cryptographic reference to the already published complete unit packet.
    ui=(UNIT/'evidence-index.json').read_bytes();assert sha(ui)==UNIT_PIN;save('unit-reference/evidence-index.json',ui)
    unitindex=json.loads(ui)
    selected=['retention.json','qualification-summary.json','proposal/index.json','proposal/proposal.diff',
              'proposal/formatter-source.json','proposal/bindings.json','proposal/EQUIVALENCE.md',
              'source2/source-index.json','source3/source-index.json','source4/source-index.json',
              'build2/terminal.json','build3/terminal.json','run3/terminal.json','run4/terminal.json']
    joined={}
    for name in selected:
        if name in unitindex['compressed_text']:
            meta=unitindex['compressed_text'][name];raw=gzip.decompress((UNIT/meta['path']).read_bytes())
            assert identity(raw)=={'bytes':meta['raw_bytes'],'sha256':meta['raw_sha256']}
        else:
            raw=(UNIT/name).read_bytes();meta=unitindex['files'][name];assert len(raw)==meta['size'] and sha(raw)==meta['sha256']
        save('unit-reference/'+name,raw);joined[name]=identity(raw)
    save('unit-reference/join.json',encoded({'index_sha256':UNIT_PIN,'published_relative_path':'../fortran-qp-output-bulk-parser-qualification-1',
       'files':joined,'scope':'Exact indexed subset and full unit index bind prior source/failures. Full149-case replay remains in the referenced unit packet; not rerun here.'}))
    for stage,(directory,pin) in AUDITS.items():
        base=PREFIX/directory;copy('audits/'+stage+'/receipt.json',base/'receipt.json',pin)
        copy('audits/'+stage+'/review.py',base/'review.py')
    # Timing audit must be supplied explicitly after closure; no audit is fabricated.
    timing=HERE/'timing-audit.json';assert timing.exists(),'Await independent TIMING audit pin'
    t=load(timing);base=PREFIX/t['directory'];copy('audits/timing/receipt.json',base/'receipt.json',t['sha256'])
    copy('audits/timing/review.py',base/'review.py')
    profile=ROOT/'docs/benchmarkfolder/experiments/fortran-qp-transport-profile-qualification-1'
    for name in ('index.json','analysis.json'):
        copy('profile-reference/'+name,profile/name,inputs['files'].get(str((profile/name).relative_to(ROOT))))
    for name in ('capture.py','verify_evidence.py','replay_warmup.py','test_verify.py','README.md'):
        copy(name,HERE/name)
    save('retention.json',encoded({'schema':'madloop-tirust-bulk-physical-retention-v1',
       'source_index_sha256':SOURCE_PIN,'unit_packet_index_sha256':UNIT_PIN,'origins':origins,
       'binary_bytes_retained':False,'binary_identity_scope':'Executable/object/provider hashes, actual prebound LOAD/runtime listings and independently rehashed closure receipts only; no binary payloads.',
       'native_or_compiler_executions_during_capture':0,'candidate_promoted':False,'performance_acceptance':False}))
    def index():
        files={str(q.relative_to(OUT)):{'kind':'file','size':q.stat().st_size,'sha256':sha(q.read_bytes())}
               for q in sorted(OUT.rglob('*')) if q.is_file() and q.name!='evidence-index.json'}
        (OUT/'evidence-index.json').write_bytes(encoded({'schema_version':1,'kind':'madloop-tirust-bulk-physical-qualification-v1',
           'files':files,'compressed_text':packed,'native_execution':False,'performance_acceptance':False}))
    index();spec=importlib.util.spec_from_file_location('capture_verify',HERE/'verify_evidence.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    summary=m.verify(OUT,derive_summary=True);save('qualification-summary.json',encoded(summary));index();assert m.verify(OUT)==summary
    print(json.dumps({'index_sha256':sha((OUT/'evidence-index.json').read_bytes()),'files':len(load(OUT/'evidence-index.json')['files'])+1,
       'compressed_payloads':len(packed),'stored_bytes':sum(q.stat().st_size for q in OUT.rglob('*') if q.is_file()),'summary':summary},indent=2,sort_keys=True))

if __name__=='__main__':main()
