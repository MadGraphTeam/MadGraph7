"""Pure clean-relocation replay with workspace/process/native-library reads blocked."""
import json,shutil,sys,tempfile
from pathlib import Path
import verify

def main(index_name='evidence-index.json'):
 source=Path(__file__).resolve().parent
 expected=verify.verify(source,index_name)
 with tempfile.TemporaryDirectory(prefix='caller-bound-portable-') as tmp:
  allowed=Path(tmp).resolve();target=allowed/source.name;shutil.copytree(source,target)
  own=json.loads((target/index_name).read_text())
  needed={
   'current-kinematic-trace-build-qualification-1':['results/derived/index.json','results/derived/cms-source-bindings.json','results/generation.json','helpers/.tirust-build/kinematic-provenance-current-proposal-1/proposal/candidate-cms.mg5','source/run.py'],
   'current-kinematic-trace-runtime-qualification-1':['analysis/decoder/decode_trace.py','results/ordinary-terminal-closure.json','results/stress-terminal-closure.json',
       *[f'results/runs/cms-point{p}-mode{m}-enabled/trace.dat' for p in (1,20,21) for m in (1,4)]]}
  for sibling,logical in needed.items():
   src=source.parent/sibling;dest=allowed/sibling;dest.mkdir();shutil.copyfile(src/'evidence-index.json',dest/'evidence-index.json')
   idx=json.loads((dest/'evidence-index.json').read_text())
   for name in logical:
    actual=idx['compressed_text'].get(name,{}).get('path',name);p=dest/actual;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src/actual,p)
  blocked=[];guard_active=[True]
  def audit(event,args):
   if event=='open' and guard_active[0]:
    name=args[0]
    if isinstance(name,(str,bytes)):
     p=Path(name.decode() if isinstance(name,bytes) else name).resolve()
     if not p.is_relative_to(allowed):blocked.append(str(p));raise RuntimeError('Workspace/outside read blocked: '+str(p))
   if event.startswith(('subprocess.','ctypes.dlopen','os.exec','os.posix_spawn')) or event=='os.system':
    raise RuntimeError('Native/process operation forbidden: '+event)
  sys.addaudithook(audit)
  actual=verify.verify(target,index_name);assert actual==expected
  try:verify.Store(target,index_name).read('../factory/index.json')
  except ValueError:pass
  else:raise AssertionError('Unsafe logical path admitted')
  p=target/'factory'/'caller_bounds.f90';old=p.read_bytes();p.write_bytes(old+b'! tamper\n')
  try:verify.verify(target,index_name)
  except ValueError:pass
  else:raise AssertionError('Changed factory source admitted')
  p.write_bytes(old)
  assert verify.verify(target,index_name)==expected
  guard_active[0]=False  # Allow tempfile's fd-relative directory cleanup after all guarded verification.
  print(json.dumps({'pure_clean_relocation_passed':True,'comparison_exact':True,'source_tamper_rejected':True,'unsafe_path_rejected':True,
    'workspace_reads_attempted':blocked,'process_and_dlopen_calls_forbidden':True,'compiled_cases_replayed':actual['compiled_cases'],'new_native_calls':0},sort_keys=True))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'evidence-index.json')
