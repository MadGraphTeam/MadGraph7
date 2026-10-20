#!/usr/bin/env python3
"""Expose final integration coefficients with DP unit-master requests."""
from pathlib import Path
import argparse,decimal,hashlib,json,os,shutil,subprocess
ROOT=Path(__file__).resolve().parents[4];DOC=Path(__file__).resolve().parent
WORK=ROOT/'.benchmark-work/experiments/qcd3-point67-masters'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def build():
 crate=WORK/'bridge';shutil.copy2(DOC/'weights.rs',crate/'src/weights.rs')
 manifest=(DOC/'Cargo.toml').read_text()+'\n[[bin]]\nname = "master-weights"\npath = "src/weights.rs"\n';(crate/'Cargo.toml').write_text(manifest)
 env=os.environ.copy();env['CARGO_TARGET_DIR']=str(WORK/'target')
 command=['cargo','build','--release','--locked','--bin','master-weights','--manifest-path',str(crate/'Cargo.toml')]
 meta={'command':command,'source_sha256':sha(DOC/'weights.rs'),'manifest':manifest,'lock_sha256':sha(crate/'Cargo.lock')}
 with (DOC/'weights-build.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 meta['executable_sha256']=sha(WORK/'target/release/master-weights');save(DOC/'weights-build.json',meta)
def run():
 exe=WORK/'target/release/master-weights';inp=DOC.parent/'qcd3-point67-dp/oracle-inputs.json'
 command=[str(exe),str(inp),str(DOC/'results.json'),str(DOC/'weights.json')]
 meta={'command':command,'executable_sha256':sha(exe),'source_sha256':sha(DOC/'weights.rs'),'inputs_sha256':sha(inp),'capture_sha256':sha(DOC/'results.json')}
 subprocess.run(command,cwd=ROOT,check=True);meta['results_sha256']=sha(DOC/'weights.json');save(DOC/'weights-run.json',meta);analyze()
def analyze():
 decimal.getcontext().prec=90;D=decimal.Decimal
 mul=lambda a,b:[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]]
 oracle={(r['loop'],r['direction']):r for r in json.loads((DOC.parent/'qcd3-point67-dp/oracle-results.json').read_text())['rows']}
 rows=[]
 for w,row in zip(json.loads((DOC/'weights.json').read_text()),json.loads((DOC/'results.json').read_text())):
  assert (w['loop'],w['direction'])==(row['loop'],row['direction'])
  masters=[o for r in row['native']['masters'] for o in r['outputs']];assert len(masters)==len(w['weights'])
  out={'loop':w['loop'],'direction':w['direction']};sums=[[D(0),D(0)] for _ in range(3)];abs_sums=[[D(0),D(0)] for _ in range(3)]
  for z,m in zip(w['weights'],masters):
   coeff=list(map(D.from_float,z))
   for k,master in enumerate(m['native']):
    term=mul(coeff,list(map(D.from_float,master)))
    for j in range(2):sums[k][j]+=term[j];abs_sums[k][j]+=abs(term[j])
  for j in range(2):sums[0][j]+=D.from_float(w['rational'][j]);abs_sums[0][j]+=abs(D.from_float(w['rational'][j]))
  exact=[v for pair in sums for v in pair];native=[D.from_float(v) for pair in row['native']['laurent'] for v in pair]
  truth=list(map(D,oracle[w['loop'],w['direction']]['oracle_laurent_components']))
  out['exact_sum_dp_coefficients_and_dp_masters']=[str(v) for v in exact]
  out['native_minus_exact_sum']=[str(v-w) for v,w in zip(native,exact)]
  out['exact_sum_minus_full_oracle']=[str(v-w) for v,w in zip(exact,truth)]
  out['sum_absolute_terms']=[str(v) for pair in abs_sums for v in pair];rows.append(out)
  print(w['loop'],w['direction'],'final accumulation error',out['native_minus_exact_sum'][0],'earlier arithmetic error',out['exact_sum_minus_full_oracle'][0])
 save(DOC/'weights-summary.json',{'rows':rows,'note':'Exact decimal products/sum of DP integration weights and DP masters; weights isolated through one unit finite master at a time. This also removes rounding of the final complex products, so it bounds more than summation alone.'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','run','analyze']);a=p.parse_args();globals()[a.action]()
