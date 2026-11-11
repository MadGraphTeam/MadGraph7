#!/usr/bin/env python3
"""Pure portable retention verification. Never compile/run a native ELF."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def inventory(folder, manifest, omissions=()):
 omitted=set(omissions)
 for item in manifest['files']:
  rel=Path(item['path']);assert not rel.is_absolute() and '..' not in rel.parts
  if str(rel) in omitted:continue
  f=folder/rel;assert f.is_file() and f.stat().st_size==item['bytes'] and digest(f)==item['sha256'],str(f)
outer=json.loads((ROOT/'index.json').read_text());inventory(ROOT,outer)
assert {str(x.relative_to(ROOT)) for x in ROOT.rglob('*') if x.is_file()}=={'index.json'}|{x['path'] for x in outer['files']}
source=ROOT/'source-proposal';hook=ROOT/'hook-native'
assert digest(source/'index.json')=='5902b9b2b6e378d7d2617b991399f22f86cf3d1813a51432e05ba33c4638b8e2'
assert digest(hook/'index.json')=='4a06fcf2b3d37ec95183d0271345734c05b0f17fda955bcde14e69c22ecf7d72'
inventory(source,json.loads((source/'index.json').read_text()))
closure=json.loads((hook/'closure.json').read_text());omitted=json.loads((hook/'omitted-native-binaries.json').read_text())['omitted']
assert omitted==closure['elfs'] and {x['path'] for x in omitted}=={'bin/hook-enabled','bin/hook-disabled'}
old_hook_index=json.loads((hook/'index.json').read_text());old_files={x['path']:x for x in old_hook_index['files']}
for item in omitted:
 assert item==old_files[item['path']] and not (hook/item['path']).exists()
inventory(hook,old_hook_index,{x['path'] for x in omitted})
assert (hook/'inputs/source-proposal-index.json').read_bytes()==(source/'index.json').read_bytes()
assert (hook/'inputs/candidate-dilog.rs').read_bytes()==(source/'candidate/sources/oneloopmaster/src/native/dilog.rs').read_bytes()
for f in ROOT.rglob('*'):
 if f.is_file():
  with f.open('rb') as stream:assert stream.read(4)!=b'\x7fELF',str(f)
assert sum(x['elapsed_seconds'] for x in closure['commands'])<=60
assert closure['elapsed_work_and_cleanup_seconds']<=90
assert closure['limits']['max_commands']==4 and closure['limits']['cpu_count']==1
assert closure['limits']['mutex_wait'] is False and closure['limits']['address_space_bytes_per_process']==1073741824
reports={}
with tempfile.TemporaryDirectory(prefix='retained-guard-pure-verify-') as d:
 for key,folder in [('source',source),('hook',hook)]:
  result=subprocess.run([sys.executable,'-B',str(folder/'verify.py')],cwd=d,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
  assert result.returncode==0,(key,result.returncode,result.stdout,result.stderr)
  reports[key]=json.loads(result.stdout)
assert reports['source']['passed'] and reports['source']['checks']==127 and reports['source']['native_executions']==0
assert reports['hook']['passed'] and reports['hook']['native_commands_rerun']==0
assert reports['hook']['compiled_hook_tests_passed']==2 and reports['hook']['feature_disabled_tests']==0
assert reports['hook']['admitted_commands_completed']==4 and reports['hook']['hook_source_equivalence']
print(json.dumps({'passed':True,'retained_payloads':len(outer['files']),'source_checks':127,'standalone_hook_enabled_tests':2,'standalone_hook_disabled_tests':0,'retained_native_command_count':4,'native_commands_rerun':0,'native_ELFs_retained_publicly':0,'numerical_guard_integration_built':False,'scientific_qualification':False,'fenv_or_trap_qualification':False},indent=2))
