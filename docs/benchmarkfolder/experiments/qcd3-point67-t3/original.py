#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[4];DOC=Path(__file__).resolve().parent;WORK=ROOT/'.benchmark-work/experiments/qcd3-point67-t3';BASE=DOC.parent/'qcd3-point67-masters'
p=argparse.ArgumentParser();p.add_argument('--quad',action='store_true');args=p.parse_args()
name='quad-original' if args.quad else 'original'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=json.loads((DOC.parent/'qcd3-point67-dp/oracle-inputs.json').read_text());capture=json.loads((BASE/'expansion-results.json').read_text());rows=[]
for v,r in zip(inputs,capture):
 for c in r['results'][0]['callbacks']:
  if c['kind']!='t3':continue
  i=c['inputs']
  for row in range(len(i['lanes'])):
   pairs=v['coefficients']+i['a'][row]+i['e3'][row]+i['e4'][row]+[i['param'][row]];rows.append(f"{v['n']} {v['rank']} {len(v['coefficients'])}\n"+'\n'.join(' '.join(format(x,'.17e')for x in z)for z in pairs))
(DOC/'original-inputs.dat').write_text(str(len(rows))+'\n'+'\n'.join(rows)+'\n')
prefix=ROOT/'.benchmark-work/original/HEPTools/ninja'
compile=['g++','-O3','-I'+str(prefix/'include'),'-c',str(DOC/f'{name}.cc'),'-o',str(WORK/f'{name}.o')]
link=['gfortran',str(WORK/f'{name}.o'),'-L'+str(prefix/'lib'),'-lninja','-lstdc++','-Wl,-rpath,'+str(prefix/'lib'),'-o',str(WORK/f'{name}-callback')]
for cmd in [compile,link]:subprocess.run(cmd,cwd=ROOT,check=True)
meta={'build_commands':[compile,link],'source_sha256':sha(DOC/f'{name}.cc'),'executable_sha256':sha(WORK/f'{name}-callback'),'input_sha256':sha(DOC/'original-inputs.dat'),'libninja_sha256':sha(prefix/'lib/libninja.so')}
with (DOC/'original-inputs.dat').open()as inp,(DOC/f'{name}-output.txt').open('w')as out:subprocess.run([str(WORK/f'{name}-callback')],cwd=ROOT,stdin=inp,stdout=out,check=True)
meta['output_sha256']=sha(DOC/f'{name}-output.txt');(DOC/f'{name}-run.json').write_text(json.dumps(meta,indent=2)+'\n')
