#!/usr/bin/env python3
"""Pure read-only verification; never runs a compiled executable."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
def digest(f):
 with f.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
source_index=P/'inputs/source-proposal-index.json'
assert digest(source_index)=='5902b9b2b6e378d7d2617b991399f22f86cf3d1813a51432e05ba33c4638b8e2'
original=json.loads(source_index.read_text());entry=next(x for x in original['files'] if x['path']=='candidate/sources/oneloopmaster/src/native/dilog.rs')
source=P/'inputs/candidate-dilog.rs';assert digest(source)==entry['sha256'] and source.stat().st_size==entry['bytes']
pre=json.loads((P/'preflight.json').read_text());e=pre['extraction'];module=source.read_bytes()[e['start_byte']:e['end_byte']]
assert module==(P/'extracted-module.rs').read_bytes()
assert hashlib.sha256(module).hexdigest()=='58958f74ed2c2e6966fc2d665c515777d6a8c27561b6dc620f01ba40cbeb12b0'
assert (P/'hook.rs').read_bytes()==bytes.fromhex(e['wrapper_prefix_hex'])+module+bytes.fromhex(e['wrapper_suffix_hex'])
assert module.count(b'#[test]')==2 and b'unsafe' not in module
closure=json.loads((P/'closure.json').read_text())
assert closure['passed'] and closure['native_commands_run']==4
assert closure['cleanup_complete'] and closure['lock_acquired'] and closure['lock_released']
assert closure['elapsed_work_and_cleanup_seconds']<=90
assert closure['network_commands']==0 and not closure['scientific_qualification'] and not closure['fenv_or_trap_qualification'] and not closure['library_or_physics_build']
expected=['compile-enabled','run-enabled','compile-disabled','run-disabled'];assert [x['name'] for x in closure['commands']]==expected
assert [x['argv'] for x in closure['commands']]==[x['argv'] for x in pre['commands']]
for x in closure['commands']:
 assert x['returncode']==0 and x['failure'] is None and digest(P/x['log'])==x['log_sha256']
 assert x['peak_sampled_aggregate_rss_bytes']<=1073741824
for x in closure['cleanup']:assert x['parent_reaped'] and not x['exists_after_cleanup']
enabled=(P/'logs/run-enabled.txt').read_text();disabled=(P/'logs/run-disabled.txt').read_text()
assert 'running 2 tests' in enabled and '2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' in enabled
for name in ['default_nesting_reentry_and_thread_isolation','restores_action_and_probe_panics']:assert ('test checked_rounding::tests::'+name+' ... ') in enabled
assert 'action unwind' in enabled and 'probe unwind' in enabled
assert 'running 0 tests' in disabled and '0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' in disabled
for x in closure['elfs']:
 f=P/x['path']
 if f.exists():assert f.stat().st_size==x['bytes'] and digest(f)==x['sha256'] and f.read_bytes()[:4]==b'\x7fELF'
 else:
  omitted=json.loads((P/'omitted-native-binaries.json').read_text());assert x in omitted['omitted']
if (P/'index.json').exists():
 idx=json.loads((P/'index.json').read_text())
 omitted_paths=set()
 if (P/'omitted-native-binaries.json').exists():omitted_paths={x['path'] for x in json.loads((P/'omitted-native-binaries.json').read_text())['omitted']}
 for x in idx['files']:
  if x['path'] in omitted_paths:continue
  f=P/x['path'];assert f.stat().st_size==x['bytes'] and digest(f)==x['sha256']
print(json.dumps({'passed':True,'native_commands_rerun':0,'compiled_hook_tests_passed':2,'feature_disabled_tests':0,'admitted_commands_completed':4,'hook_source_equivalence':True,'scientific_qualification':False,'fenv_or_trap_qualification':False},indent=2))
