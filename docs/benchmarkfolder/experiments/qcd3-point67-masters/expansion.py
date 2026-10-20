#!/usr/bin/env python3
"""Keep fast cut geometry, isolate numerator callback arithmetic."""
from pathlib import Path
import argparse,decimal,hashlib,json,os,resource,shutil,subprocess,time
ROOT=Path(__file__).resolve().parents[4];DOC=Path(__file__).resolve().parent
WORK=ROOT/'.benchmark-work/experiments/qcd3-point67-masters';INPUT=DOC.parent/'qcd3-point67-dp/oracle-inputs.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def build():
 crate=WORK/'bridge';(crate/'src').mkdir(parents=True,exist_ok=True)
 for name in ['expansion.rs','tensor.rs']:shutil.copy2(DOC/name,crate/'src'/name)
 shutil.copy2(DOC/'Cargo.lock',crate/'Cargo.lock')
 manifest=(DOC/'Cargo.toml').read_text()+'\n[[bin]]\nname = "expansion-oracle"\npath = "src/expansion.rs"\n';(crate/'Cargo.toml').write_text(manifest)
 env=os.environ.copy();env['CARGO_TARGET_DIR']=str(WORK/'target');cmd=['cargo','build','--release','--locked','--bin','expansion-oracle','--manifest-path',str(crate/'Cargo.toml')]
 meta={'command':cmd,'manifest':manifest,'lock_sha256':sha(crate/'Cargo.lock'),'source_sha256':sha(DOC/'expansion.rs'),'tensor_sha256':sha(DOC/'tensor.rs')}
 with (DOC/'expansion-build.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 meta['executable_sha256']=sha(WORK/'target/release/expansion-oracle');save(DOC/'expansion-build.json',meta)
def run(cpu):
 exe=WORK/'target/release/expansion-oracle';cmd=['taskset','-c',str(cpu),str(exe),str(INPUT),str(DOC/'expansion-results.json')]
 meta={'command':cmd,'executable_sha256':sha(exe),'input_sha256':sha(INPUT)};save(DOC/'expansion-run.json',meta)
 env=os.environ.copy();env.setdefault('SYMBOLICA_LOCK_DIR',str(ROOT/'.ninjars-build/symbolica-locks'))
 def stack():
  _,hard=resource.getrlimit(resource.RLIMIT_STACK);resource.setrlimit(resource.RLIMIT_STACK,(256*1024*1024,hard))
 start=time.monotonic()
 with (DOC/'expansion-run.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,preexec_fn=stack)
 meta.update(elapsed_seconds=time.monotonic()-start,complete=True,results_sha256=sha(DOC/'expansion-results.json'));save(DOC/'expansion-run.json',meta);analyze()
def analyze():
 decimal.getcontext().prec=90;D=decimal.Decimal
 oracle={(r['loop'],r['direction']):r for r in json.loads((DOC.parent/'qcd3-point67-dp/oracle-results.json').read_text())['rows']};summary=[]
 for r in json.loads((DOC/'expansion-results.json').read_text()):
  ref=oracle[r['loop'],r['direction']]
  for m in r['results']:
   errs=[str(D.from_float(v)-D(w)) for v,w in zip([x for c in m['laurent'] for x in c],ref['oracle_laurent_components'])]
   summary.append({'loop':r['loop'],'direction':r['direction'],'mode':m['mode'],'errors':errs})
 save(DOC/'expansion-summary.json',{'rows':summary})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','run','analyze']);p.add_argument('--cpu',type=int,default=207);a=p.parse_args()
 if a.action=='run':run(a.cpu)
 else:globals()[a.action]()
