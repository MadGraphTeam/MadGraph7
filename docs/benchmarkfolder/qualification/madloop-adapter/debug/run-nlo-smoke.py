from pathlib import Path
import json, subprocess, sys
sys.path.insert(0,str(Path.cwd()))
from madgraph.various.banner import MadLoopParam
root=Path.cwd()
work=root/'.tirust-build/adapter-export/nlo-internal/SubProcesses/P0_uux_z/V0_uux_z'
card=root/'.tirust-build/adapter-export/nlo-internal/SubProcesses/MadLoopParams.dat'
original=card.read_text()
records=[]
try:
 for backend,mode in [(1,1),(9,1),(9,4)]:
  params=MadLoopParam(str(card))
  params['MLReductionLib']=str(backend)
  params['CTModeRun']=mode
  params['CTModeInit']=mode
  params['CheckCycle']=1
  params['TIRustComputeFinite']=True
  params.write(str(card))
  output=work/'result.dat'
  output.unlink(missing_ok=True)
  ret=subprocess.run([str(work/'check')],cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
  log=root/('.tirust-build/adapter-export/nlo-check-%d-%d.log'%(backend,mode))
  log.write_text(ret.stdout)
  result=output.read_text() if output.exists() else ''
  (root/('.tirust-build/adapter-export/nlo-result-%d-%d.dat'%(backend,mode))).write_text(result)
  records.append(dict(backend=backend,mode=mode,status=ret.returncode,result=result))
  print(backend,mode,ret.returncode,result[-400:],flush=True)
  if ret.returncode or not result: raise RuntimeError(str(log))
finally:
 card.write_text(original)
 (root/'.tirust-build/adapter-export/nlo-smoke.json').write_text(json.dumps(records,indent=2))

original=card.read_text()
try:
 params=MadLoopParam(str(card))
 params['MLReductionLib']='9'
 params['TIRustComputeFinite']=False
 params.write(str(card))
 ret=subprocess.run([str(work/'check')],cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
 Path('.tirust-build/adapter-export/nlo-partial-rejected.log').write_text(ret.stdout)
 assert ret.returncode==1,ret.stdout
 assert 'Poles-only tensors are not complete amplitudes' in ret.stdout
finally: card.write_text(original)
