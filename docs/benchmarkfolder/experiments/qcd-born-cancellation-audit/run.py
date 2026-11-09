#!/usr/bin/env python3
"""Compile/run a bounded CPU11 Born-only diagnostic, using immutable archives."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
ROOT=Path.cwd().resolve();REPORT=ROOT/'docs/benchmarkfolder/experiments/qcd-born-cancellation-audit';WORK=ROOT/'.tirust-build/qcd-born-cancellation-audit'
PROC=ROOT/'.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg';SP=PROC/'SubProcesses/P0_uux_zgg'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def name(p):return str(p.relative_to(ROOT))
meta=json.loads((REPORT/'preparation.json').read_text())
checks=meta['source_and_link_inputs']
checks[name(REPORT/'run.py')]=sha(REPORT/'run.py')
for rel,d in checks.items():assert sha(ROOT/rel)==d,rel
for f in ('driver.f','qp_born.f'):assert sha(WORK/f)==sha(REPORT/f)
assert sha(WORK/'points.dat')==sha(ROOT/'docs/benchmarkfolder/experiments/physical-stress/points/qcd_n3_uux_zgg/selected.dat')
assert sha(WORK/'param_card.dat')==sha(PROC/'Cards/param_card.dat')
if (REPORT/'execution.json').exists():raise SystemExit('Refusing overwrite')
fc=shutil.which('gfortran');assert fc
base=['taskset','-c','11',fc,'-w','-fPIC','-ffixed-line-length-132','-I'+str(PROC/'Source/MODEL'),'-I'+str(PROC/'Source/DHELAS'),'-I'+str(SP)]
commands=[[*base,'-c',str(WORK/'driver.f'),'-o',str(WORK/'driver.o')],[*base,'-c',str(WORK/'qp_born.f'),'-o',str(WORK/'qp_born.o')],[*base,str(WORK/'driver.o'),str(WORK/'qp_born.o'),str(SP/'born_matrix.o'),str(PROC/'lib/libdhelas.a'),str(PROC/'lib/libmodel.a'),'-o',str(WORK/'born-audit')]]
record={'scope':meta['scope'],'commands':commands,'cpu':11,'source_and_link_inputs_before':checks,'complete':False,'compiler_version':subprocess.check_output([fc,'--version'],text=True).splitlines()[0],'run_script_sha256':sha(REPORT/'run.py'),'native_reducers_called':False,'symbolica_loaded':False,'performance_eligible':False}
try:
 with (REPORT/'build.log').open('x') as log:
  for cmd in commands:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
 record['artifacts_sha256']={name(WORK/f):sha(WORK/f) for f in ('driver.o','qp_born.o','born-audit')}
 ldd=subprocess.check_output(['ldd',str(WORK/'born-audit')],text=True)
 (REPORT/'ldd.txt').write_text(ldd)
 assert 'libmadloop_tirust' not in ldd and 'ninjars' not in ldd
 record['execution_command']=['taskset','-c','11',str(WORK/'born-audit'),str(WORK/'param_card.dat'),str(WORK/'points.dat')]
 with (REPORT/'output.log').open('x') as log:
  subprocess.run(record['execution_command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
 rows=(REPORT/'output.log').read_text().splitlines()
 assert sum(x.startswith('DP_BORN') for x in rows)==2,'MissingDPBornrows'
 assert sum(x.startswith('QP_BORN') for x in rows)==6,'MissingQPBornrows'
 assert sum(x.startswith('DENOM') for x in rows)==6,'Missingdenominatorrows'
 record['complete']=True
except BaseException as e:
 record['error']={'type':type(e).__name__,'message':str(e)}
 raise
finally:
 record['source_and_link_inputs_after']={rel:sha(ROOT/rel) for rel in checks}
 record['protected_inputs_unchanged']=record['source_and_link_inputs_after']==checks
 record['logs_sha256']={f:sha(REPORT/f) for f in ('build.log','ldd.txt','output.log') if (REPORT/f).exists()}
 (REPORT/'execution.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
 assert record['protected_inputs_unchanged']
