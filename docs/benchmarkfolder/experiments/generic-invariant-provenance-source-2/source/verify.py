"""Authenticate the private source packet; optional pure tests / explicit live pins."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,importlib.util,io,json,os,sys,types,unittest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def require(ok,why):
 if not ok:raise ValueError(why)
def digest(b):return hashlib.sha256(b).hexdigest()
def safe(name):
 p=PurePosixPath(name);require(not p.is_absolute() and '..' not in p.parts and str(p)==name,'unsafe payload name');return p

def verify(live=False):
 from bindings import verify_parents
 parent_proof=verify_parents(live)
 b=(HERE/'source-index.json').read_bytes();index=json.loads(b)
 for n,m in index['files'].items():
  safe(n);p=HERE/n
  require(not any(x.is_symlink() for x in [p,*p.parents] if x.is_relative_to(HERE)),'payload symlink')
  raw=p.read_bytes();require(digest(raw)==m['sha256'] and len(raw)==m['bytes'],'payload mismatch '+n);raw.decode('utf8')
 parents=json.loads((HERE/'parent-pins.json').read_bytes())
 for n,h in parents.items():require(digest((ROOT/n).read_bytes())==h,'immutable parent mismatch '+n)
 inputs=json.loads((HERE/'input-bindings.json').read_bytes())
 if live:
  for n,m in inputs.items():
   raw=(ROOT/n).read_bytes();require(digest(raw)==m['sha256'] and len(raw)==m['bytes'],'explicit live input changed '+n)
 closed=json.loads((HERE/'closed-results.json').read_bytes())
 require(set(closed)=={'gmu','loop_sm'},'closed case set')
 for c,r in closed.items():
  require(r['producer']['reference_complete'] and not r['producer']['complete'],'whole-producer scope')
  require(all(v['complete'] and not v['scientific_admission'] for v in r['momentum_windows'].values()),'closed prefix scope')
 generated=json.loads((HERE/'generation-results.json').read_bytes());require(len(generated)==7,'generation count')
 require(sum(x['has_born'] for x in generated)==5,'Born/loop-induced scope')
 for r in generated:
  for w in r['momentum_environment_contract'].values():
   require(w['prefix']['complete'] and not w['scientific_admission'],'generated prefix scope')
   require(w['complete']==r['has_born'],'unsupported loop-induced ABI limitation changed')
 return {'passed':True,'source_index_sha256':digest(b),'files':len(index['files']),
         'parent_proof':parent_proof,'live_input_count':len(inputs) if live else None,'closed_windows':4,'actual_generation_cases':7,
         'scientific_admission':False,'native_execution':False}

def tests():
 # Only authenticated new source is imported; no MG/model execution.
 for name in ('madgraph','madgraph.loop'):
  mod=types.ModuleType(name);mod.__path__=[str(HERE/'src/madgraph/loop')];sys.modules[name]=mod
 sys.modules['madgraph'].loop=sys.modules['madgraph.loop']
 sys.dont_write_bytecode=True
 allowed=(HERE,Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve())
 def guard(event,args):
  if event in ('subprocess.Popen','os.system','os.exec','os.posix_spawn','ctypes.dlopen'):raise AssertionError('external execution forbidden')
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   mode=args[1];flags=args[2]
   require(not (isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)),'pure replay write forbidden')
   p=Path(os.fsdecode(args[0])).resolve();require(any(p.is_relative_to(a) for a in allowed),'pure replay outside source/python environment')
 sys.addaudithook(guard)
 suite=unittest.TestSuite()
 for f in ('test_environment.py','test_window.py'):
  spec=importlib.util.spec_from_file_location('replay_'+f[:-3],HERE/f);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
  suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(m))
 log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=1).run(suite)
 require(result.wasSuccessful(),log.getvalue());require(result.testsRun==40,'test count')
 return {'tests':result.testsRun,'passed':True,'external_execution':False,'writes':False}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--live',action='store_true');ap.add_argument('--tests',action='store_true');a=ap.parse_args()
 result=verify(a.live)
 if a.tests:result['pure_tests']=tests()
 print(json.dumps(result,sort_keys=True,indent=2))
