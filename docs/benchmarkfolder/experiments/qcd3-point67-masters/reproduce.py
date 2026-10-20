#!/usr/bin/env python3
"""Build/run only an ignored scalar-oracle diagnostic; never edits dependencies."""
from pathlib import Path
import argparse, decimal, hashlib, json, os, resource, shutil, subprocess, time
ROOT=Path(__file__).resolve().parents[4]
DOC=Path(__file__).resolve().parent
WORK=ROOT/'.benchmark-work/experiments/qcd3-point67-masters'
INPUT=DOC.parent/'qcd3-point67-dp/oracle-inputs.json'
ORACLE=DOC.parent/'qcd3-point67-dp/oracle-results.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def build():
    crate=WORK/'bridge'; (crate/'src').mkdir(parents=True,exist_ok=True)
    for name in ['Cargo.toml','Cargo.lock']:shutil.copy2(DOC/name,crate/name)
    shutil.copy2(DOC/'main.rs',crate/'src/main.rs')
    env=os.environ.copy(); env['CARGO_TARGET_DIR']=str(WORK/'target')
    command=['cargo','build','--release','--locked','--manifest-path',str(crate/'Cargo.toml')]
    metadata={'command':command,'sources':{n:sha(DOC/n) for n in ['Cargo.toml','Cargo.lock','main.rs']},'input_sha256':sha(INPUT),'full_oracle_sha256':sha(ORACLE)}
    with (DOC/'build.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    metadata['executable_sha256']=sha(WORK/'target/release/point67-master-oracle')
    save(DOC/'build.json',metadata)
def run(cpu):
    exe=WORK/'target/release/point67-master-oracle'
    env=os.environ.copy(); env.setdefault('SYMBOLICA_LOCK_DIR',str(ROOT/'.ninjars-build/symbolica-locks'))
    command=['taskset','-c',str(cpu),str(exe),str(INPUT),str(DOC/'results.json')]
    metadata={'command':command,'source_sha256':sha(DOC/'main.rs'),'executable_sha256':sha(exe),'input_sha256':sha(INPUT),'full_oracle_sha256':sha(ORACLE),'cpu':cpu}
    save(DOC/'run.json',metadata)
    def stack():
        _,hard=resource.getrlimit(resource.RLIMIT_STACK)
        resource.setrlimit(resource.RLIMIT_STACK,(256*1024*1024,hard))
    start=time.monotonic()
    with (DOC/'run.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,preexec_fn=stack)
    metadata['elapsed_seconds']=time.monotonic()-start; metadata['complete']=True
    metadata['results_sha256']=sha(DOC/'results.json');save(DOC/'run.json',metadata)
    summarize()
def summarize():
    decimal.getcontext().prec=90;D=decimal.Decimal
    oracle={(r['loop'],r['direction']):r for r in json.loads(ORACLE.read_text())['rows']}
    rows=[];families={};requests=0
    for row in json.loads((DOC/'results.json').read_text()):
        ref=oracle[row['loop'],row['direction']]
        item={'loop':row['loop'],'direction':row['direction']}
        for mode in ['native','substituted']:
            values=[D.from_float(v) for z in row[mode]['laurent'] for v in z]
            item[mode+'_errors']=[str(v-D(w)) for v,w in zip(values,ref['oracle_laurent_components'])]
        item['substitution_shift']=[str(D.from_float(v)-D.from_float(w)) for z,y in zip(row['substituted']['laurent'],row['native']['laurent']) for v,w in zip(z,y)]
        assert row['native']['masters']==row['substituted']['masters']
        for rec in row['native']['masters']:
            requests+=1; assert rec['native_ok']
            for out in rec['outputs']:
                assert out['high_error'] is None
                stat=families.setdefault(out['integral'],{'count':0,'max_absolute_component_error':'0','max_relative_complex_error':'0'})
                stat['count']+=1
                for z,h in zip(out['native'],out['high_decimal']):
                    dz=[D.from_float(v)-D(w) for v,w in zip(z,h)]
                    norm=sum(D(w)**2 for w in h).sqrt();err=sum(x*x for x in dz).sqrt()
                    stat['max_absolute_component_error']=str(max(D(stat['max_absolute_component_error']),*[abs(x) for x in dz]))
                    if norm:stat['max_relative_complex_error']=str(max(D(stat['max_relative_complex_error']),err/norm))
        rows.append(item)
    save(DOC/'summary.json',{'rows':rows,'master_requests':requests,'family_statistics':families,'native_replay_exact':True,'substitution_requests_exact':True,'onshell_threshold':1e-9,'high_precision_bits':256,'full_oracle_note':'Previously retained full Float256 reduction of the same frozen binary64 tensor inputs.'})
    for r in rows:print(r['loop'],r['direction'],'native finite error',r['native_errors'][0],'substituted',r['substituted_errors'][0])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','run','summarize']);p.add_argument('--cpu',type=int,default=207);a=p.parse_args()
    if a.action=='build':build()
    elif a.action=='run':run(a.cpu)
    else:summarize()
