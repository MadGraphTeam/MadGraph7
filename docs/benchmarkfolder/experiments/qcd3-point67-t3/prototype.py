#!/usr/bin/env python3
"""DP-only arithmetic variants with previously captured high-precision targets."""
from pathlib import Path
import argparse,decimal,hashlib,json,math,os,shutil,struct,subprocess
ROOT=Path(__file__).resolve().parents[4];DOC=Path(__file__).resolve().parent;WORK=ROOT/'.benchmark-work/experiments/qcd3-point67-t3';BASE=DOC.parent/'qcd3-point67-masters';INPUT=DOC.parent/'qcd3-point67-dp/oracle-inputs.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def build():
 crate=WORK/'bridge';(crate/'src').mkdir(parents=True,exist_ok=True)
 for f in ['main.rs','t3.rs']:shutil.copy2(DOC/f,crate/'src'/f)
 for f in ['Cargo.toml','Cargo.lock']:shutil.copy2(DOC/f,crate/f)
 env=os.environ.copy();env['CARGO_TARGET_DIR']=str(WORK/'target');cmd=['cargo','rustc','--release','--locked','--bin','point67-t3','--manifest-path',str(crate/'Cargo.toml'),'--','-C','lto=off']
 meta={'command':cmd,'source_sha256':{f:sha(DOC/f)for f in ['main.rs','t3.rs','Cargo.toml','Cargo.lock']},'note':'Root executable uses lto=off for fast diagnostic iteration; no timing claim. Unchanged native mode must reproduce every frozen Laurent component exactly.'}
 with (DOC/'build.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 meta['executable_sha256']=sha(WORK/'target/release/point67-t3');save(DOC/'build.json',meta)
def run():
 exe=WORK/'target/release/point67-t3';cmd=[str(exe),str(INPUT),str(BASE/'results.json'),str(BASE/'expansion-results.json'),str(DOC/'results.json')]
 meta={'command':cmd,'executable_sha256':sha(exe),'input_sha256':sha(INPUT),'master_capture_sha256':sha(BASE/'results.json'),'callback_capture_sha256':sha(BASE/'expansion-results.json')};save(DOC/'run.json',meta)
 subprocess.run(cmd,cwd=ROOT,check=True);meta['results_sha256']=sha(DOC/'results.json');meta['complete']=True;save(DOC/'run.json',meta);analyze()
def ordered(x):
 b=struct.unpack('>Q',struct.pack('>d',x))[0];return ~b & ((1<<64)-1) if b>>63 else b | (1<<63)
def analyze():
 decimal.getcontext().prec=90;D=decimal.Decimal
 oracle={(r['loop'],r['direction']):r for r in json.loads((DOC.parent/'qcd3-point67-dp/oracle-results.json').read_text())['rows']}
 stats={m:{'values':0,'exact':0,'better':0,'worse':0,'same':0,'max_ulp':0,'sum_ulp':0}for m in range(7)};full=[];original=[];mode3=[]
 for r in json.loads((DOC/'results.json').read_text()):
  ref=oracle[r['loop'],r['direction']]
  for m in r['full']:
   errors=[str(D.from_float(x)-D(y))for x,y in zip([v for z in m['laurent']for v in z],ref['oracle_laurent_components'])];full.append({'loop':r['loop'],'direction':r['direction'],'mode':m['mode'],'errors':errors})
  for cb in r['callbacks']:
   modes=[{'mode':0,'values':cb['native']}]+cb['variants']
   for variant in modes:
    s=stats[variant['mode']]
    for z,o,n in zip(variant['values'],cb['oracle'],cb['native']):
     for v,target,native in zip(z,o,n):
      err=abs(ordered(v)-ordered(target));old=abs(ordered(native)-ordered(target));s['values']+=1;s['exact']+=err==0;s['better']+=err<old;s['worse']+=err>old;s['same']+=err==old;s['max_ulp']=max(s['max_ulp'],err);s['sum_ulp']+=err
    if variant['mode']==3:mode3.extend(x for z in variant['values']for x in z)
 for line in (DOC/'original-output.txt').read_text().splitlines():original.extend(map(float,line.split()[1:]))
 assert len(original)==len(mode3);mismatches=[{'index':i,'rust':x,'ninja':y}for i,(x,y)in enumerate(zip(mode3,original))if struct.pack('>d',x)!=struct.pack('>d',y)]
 # Diagnostic direction aggregation replaces only these three loop groups.
 oldlines=(DOC.parent/'qcd3-point67-dp/normal/normal-dp-decision.dat').read_text().splitlines();old={int(t[1]):list(map(float,t[2:]))for t in map(str.split,oldlines[3:])if t[0]=='0'}
 scale=2/(16*math.pi**2)/72;aggregate=[];results=json.loads((DOC/'results.json').read_text())
 for mode in range(7):
  vals={k:v.copy()for k,v in old.items()}
  for r in results:
   base=r['full'][0]['laurent'];new=r['full'][mode]['laurent'];direction=r['direction']
   for j in range(3):vals[direction][j]+=(new[j][0]-base[j][0])*scale
  avg=[(x+y)/2 for x,y in zip(vals[1],vals[2])];acc=sum(abs(x-y)for x,y in zip(vals[1],vals[2]))/(abs(sum(avg))/3)
  aggregate.append({'mode':mode,'directions':vals,'predicted_stability_estimate':acc})
 summary={'callback_statistics':stats,'full_reduction_errors':full,'original_ninja_order_mismatches':mismatches,'original_ninja_order_components':len(original),'direction_aggregate':aggregate,'aggregate_note':'Diagnostic substitution of only loops3/52/71 using known2/(16pi²)/72 normalization; not a generated-process evaluation or an acceptance result.'};save(DOC/'summary.json',summary)
 print(json.dumps(stats,indent=2));print('Original-order C++ mismatches:',len(mismatches));print('Predicted aggregate:',[(r['mode'],r['predicted_stability_estimate'])for r in aggregate])
 for r in full:print('finite error',r['mode'],r['loop'],r['direction'],r['errors'][0])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','run','analyze']);a=p.parse_args();globals()[a.action]()
