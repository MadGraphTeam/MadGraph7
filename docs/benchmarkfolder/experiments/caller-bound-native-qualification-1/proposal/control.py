"""One-shot bounded standalone build/run; call only after explicit admission."""
import argparse,hashlib,json,os,shutil,signal,time
from pathlib import Path
import prepare,analyze,supervise
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WORK=ROOT/'.tirust-build/caller-bound-native-qualification-1'
def binding(p):
 b=p.read_bytes();return {'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def dump(p,d):p.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
def source_verified():
 idx=json.loads((HERE/'index.json').read_text())
 assert set(p.name for p in HERE.iterdir())==set(idx['files'])|{'index.json'}
 for name,b in idx['files'].items():assert not (HERE/name).is_symlink() and binding(HERE/name)==b,name
 prepare.verify_factory()
 for path,b in idx['tools'].items():assert binding(Path(path))==b,path
 return idx

def clean_env():
 e={k:os.environ[k] for k in ('PATH','HOME','TMPDIR','TMP','TEMP') if k in os.environ}
 e.update(LC_ALL='C',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');return e

def interrupt(sig,frame):raise InterruptedError('Controller interrupted: '+str(sig))

def stage(action):
 idx=source_verified();start=time.monotonic();budget=60 if action=='build' else 10;deadline=start+budget
 out=WORK/action
 if action=='run':
  prior=json.loads((WORK/'build'/'result.json').read_text());assert prior['passed']
  assert prior['source_index_sha256']==binding(HERE/'index.json')['sha256']
  assert binding(WORK/'build'/'probe')==prior['executable']
 out.mkdir(parents=True,exist_ok=False)
 receipt={'stage':action,'passed':False,'budget_seconds':budget,'source_index_sha256':binding(HERE/'index.json')['sha256'],'commands':[],'tensor_calls':0}
 old={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
 def command(argv,cwd,label):
  left=deadline-time.monotonic()
  if left<=0:raise TimeoutError('Aggregate stage budget exceeded')
  record={'argv':argv,'cwd':str(cwd),'progress':{}};receipt['commands'].append(record)
  try:
   with (out/(label+'.log')).open('wb') as log:code=supervise.run_plain_capped(argv,log,cwd=cwd,env=clean_env(),progress=record['progress'],seconds=left)
   record['returncode']=code
   if code!=0:raise RuntimeError('Failed '+label)
  finally:record['log']=binding(out/(label+'.log')) if (out/(label+'.log')).exists() else None
 try:
  if action=='build':
   for name in ('caller_bounds.f90','cms_bounds_data.inc','environment_token.c'):shutil.copyfile(prepare.FACTORY/name,out/name)
   shutil.copyfile(HERE/'driver.f90',out/'driver.f90')
   c=idx['compiler_paths']['gcc'];f=idx['compiler_paths']['gfortran'];limit=idx['compiler_paths']['prlimit']
   prefix=[limit,'--as=4294967296','--']
   command(prefix+[c,'-O2','-Wall','-Wextra','-std=c11','-fno-fast-math','-c','environment_token.c','-o','environment_token.o'],out,'01-c')
   ff=['-O2','-Wall','-Wextra','-fcheck=all','-fno-fast-math']
   command(prefix+[f,*ff,'-c','caller_bounds.f90','-o','caller_bounds.o'],out,'02-factory')
   command(prefix+[f,*ff,'-c','driver.f90','-o','driver.o'],out,'03-driver')
   command(prefix+[f,*ff,'caller_bounds.o','environment_token.o','driver.o','-o','probe'],out,'04-link')
   receipt['executable']=binding(out/'probe')
   receipt['objects']={n:binding(out/n) for n in ('caller_bounds.o','environment_token.o','driver.o')}
   for name in ('caller_bounds.f90','cms_bounds_data.inc','environment_token.c'):assert binding(out/name)==binding(prepare.FACTORY/name)
   assert binding(out/'driver.f90')==binding(HERE/'driver.f90')
  else:
   for name in ('inputs.txt','tokens.txt'):shutil.copyfile(HERE/name,out/name)
   receipt['executable']=binding(WORK/'build'/'probe')
   command([idx['compiler_paths']['prlimit'],'--as=4294967296','--',str(WORK/'build'/'probe')],out,'native')
   receipt['qualification']=analyze.parse((out/'native.log').read_bytes(),json.loads((HERE/'expected.json').read_text()),json.loads((HERE/'tokens.json').read_text()))
   for name in ('inputs.txt','tokens.txt'):assert binding(out/name)==binding(HERE/name)
   assert binding(WORK/'build'/'probe')==receipt['executable']
  source_verified()
  if time.monotonic()>deadline:raise TimeoutError('Aggregate stage budget exceeded after final verification')
  receipt['passed']=True
 except BaseException as ex:
  receipt['error_type']=type(ex).__name__;receipt['error']=str(ex);raise
 finally:
  receipt['elapsed_seconds']=time.monotonic()-start;dump(out/'result.json',receipt)
  for s,h in old.items():signal.signal(s,h)
 print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','run']);stage(p.parse_args().stage)
