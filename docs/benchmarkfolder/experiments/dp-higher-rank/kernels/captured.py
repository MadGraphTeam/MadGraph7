#!/usr/bin/env python3
"""Replay captured operands with their observed flags, then warm-repeat timing."""
from pathlib import Path
import hashlib,json,os,shutil,statistics,subprocess
from madgraph.various.q_polynomial import FortranPolynomialRoutines
from madgraph.iolibs.file_writers import FortranWriter
ROOT=Path.cwd();HERE=ROOT/'docs/benchmarkfolder/experiments/dp-higher-rank/kernels';WORK=ROOT/'.tirust-build/dp-higher-rank/kernels/captured'
records=[];commands=[]
def fixed(p,s):
    with FortranWriter(str(p)) as w:w.downcase=False;w.writelines(s)
def run(cmd,w,tag):
    p=subprocess.run(cmd,cwd=w,capture_output=True,text=True,timeout=40)
    r={'command':list(map(str,cmd)),'cwd':str(w),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};commands.append(r)
    (HERE/(tag+'.log')).write_text(json.dumps(r,indent=2)+'\n');assert p.returncode==0,r
    return p.stdout
for rank in (2,3):
    w=WORK/('r%d'%rank);w.mkdir(parents=True,exist_ok=True)
    for f in ('coef_specs.inc','loop_max_coefs.inc','public.f'):
        shutil.copy2(HERE/f,w/f)
    (w/'loop_max_coefs.inc').write_text('      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=70)\n')
    gen=FortranPolynomialRoutines(4,updater_max_rank=1,proc_prefix='TEST_',sub_prefix='TEST_')
    fixed(w/'constants.f',gen.write_polynomial_constant_module())
    shutil.copy2(HERE/('private-r%d.f'%rank),w/'private.f')
    driver=(HERE/'captured-template.f90').read_text().replace('RANK_ID',str(rank))
    (w/'driver.f90').write_text(driver)
    for name in ('constants.f','loop_max_coefs.inc','driver.f90'):
        shutil.copy2(w/name,HERE/('captured-r%d-'%rank+name))
    run(['gfortran','-O3','-fPIC','-ffixed-line-length-132','-ffree-line-length-none','constants.f','public.f','private.f','driver.f90','-o','captured'],w,'captured-r%d-build'%rank)
    for sample in (1,2):
        capture=ROOT/('docs/benchmarkfolder/experiments/dp-higher-rank/capture/attempt3/capture-r%d-%d.txt'%(rank,sample))
        for pair in range(7):
            for method in ([0,1] if pair%2==0 else [1,0]):
                out=run([str(w/'captured'),str(capture),str(method)],w,'captured-r%d-s%d-p%d-m%d'%(rank,sample,pair,method))
                fields=out.split();assert fields[0]=='CAPTURE_PASS',out
                row={'rank':rank,'sample':sample,'pair':pair,'method':method,'captured_inexact':fields[1]=='T','warm_inexact':fields[2]=='T','seconds':float(fields[3]),'iterations':500,'capture_sha256':hashlib.sha256(capture.read_bytes()).hexdigest()}
                assert row['seconds']<.1,row
                records.append(row)
summary=[]
for rank in (2,3):
 for sample in (1,2):
  ratios=[]
  for pair in range(7):
   vals={r['method']:r['seconds'] for r in records if r['rank']==rank and r['sample']==sample and r['pair']==pair}
   ratios.append(vals[1]/vals[0])
  summary.append({'rank':rank,'sample':sample,'candidate_over_dense_median':statistics.median(ratios),'paired_ratios':ratios})
result={'scope':'Exact observed-flag captured-input parity; timing is warm repeated operands, includes candidate IEEE_GET_FLAG and all guards, original O3 polynomial flags without bounds checking; no amplitude or acceptance claim','records':records,'summary':summary,'commands':commands,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('captured*') if p.suffix in ('.f','.f90','.py','.inc')}}
(HERE/'captured-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary),flush=True)
