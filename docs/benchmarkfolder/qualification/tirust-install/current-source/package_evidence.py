"""Record the actual committed-source CLI publication after its native smoke succeeds."""
import datetime, hashlib, json, shutil, subprocess, sys
from pathlib import Path
root=Path.cwd(); work=root/'.tirust-build/current-install'; mg=work/'mg'; installed=mg/'HEPTools/tirust'; lib=installed/'lib'
sys.path.insert(0,str(mg))
from madgraph.various import tirust_install

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def require(ok,message):
    if not ok: raise RuntimeError(message)

config=tirust_install.read_config(lib)
require(sha(lib/'mtirust.mod')==(work/'compiled-module.sha256').read_text().split()[0],'Installed module differs from freshly compiled smoke-visible module')
require(tirust_install.is_available(lib),'Installed native artifacts fail availability validation')
exported=tirust_install.export_files(lib)
require(all(name in exported for name in config['dependency_notices']['files']),'Expected notice files are not exportable')
require(config['finite_terms'] and config['poles_only'] and config['quad_precision'],'Missing required capability')
require('TIRust successfully installed' in (work/'cli.log').read_text(),'Missing actual CLI publication success')
log=(installed/'install.log').read_text()
for marker in ['TIRust poles-only evaluation passed','TIRust ABI1: DP, binary128 input, UV/IR poles and cache checks passed']:
    require(marker in log,'Missing native smoke marker: '+marker)
require(str(lib) in (mg/'input/mg7_configuration.txt').read_text(),'Isolated configuration missing published path')
source=json.loads((work/'source-inventory.json').read_text()); revision=json.loads((work/'source-revision.json').read_text())
require(all(sha(mg/name)==value for name,value in source.items()),'Captured implementation snapshot changed')
before=json.loads((work/'production-before.json').read_text()); after={name:sha(root/name) for name in before}
require(all(before[name]==after[name] for name in before if name!='input/mg7_configuration.txt'),'Protected production native artifact changed')
write(work/'production-after.json',after)
observer=[json.loads(line) for line in (work/'execution-observer.jsonl').read_text().splitlines()]
require(any(x['event']=='serialized-smoke' for x in observer),'Native smoke serialization missing')
report=root/'docs/benchmarkfolder/qualification/tirust-install/current-source'; report.mkdir(exist_ok=False)
for name in ['run.sh','install.mg5','cli.log','prepare_snapshot.py','snapshot-result.json','source-revision.json','source-inventory.json','install-start.json','host-toolchain.txt','production-before.json','production-after.json','production-drift-observed.json','config-drift-search.json','config-drift-attribution.json','production-main-configuration-observed.txt','allow-compilation','administrative-hold.json','host-overload-context.json','cpu-affinity.txt','compile-affinity.json','compiled-module.sha256','resolved-source-hashes.json','execution-observer.jsonl']:
    shutil.copy2(work/name,report/name)
shutil.copy2(work/'observer/sitecustomize.py',report/'execution-observer.py')
shutil.copy2(Path(__file__),report/'package_evidence.py')
shutil.copy2(root/'docs/benchmarkfolder/experiments/n1-profile/production/validate_release.py',report/'validate_release.py')
for name in ['install.log','Cargo.toml','Cargo.lock']:
    shutil.copy2(installed/name,report/name)
for name in [tirust_install.CONFIG_NAME,'tirust-dependency-provenance.json','tirust-NOTICES.txt']:
    shutil.copy2(lib/name,report/name)
shutil.copy2(mg/'input/mg7_configuration.txt',report/'isolated-configuration-after.txt')
commands=[]
for argv,filename in [(['nm','-D','--defined-only',str(lib/'libmadloop_tirust.so')],'native-exports.txt'),(['readelf','-d',str(lib/'libmadloop_tirust.so')],'dynamic-section.txt')]:
    (report/filename).write_text(subprocess.check_output(argv,text=True)); commands.append({'argv':argv,'returncode':0,'output':filename})
exports=(report/'native-exports.txt').read_text().splitlines()
require(len(exports)==12 and all(x.split()[-1].startswith('ml_tirust_') for x in exports),'Unexpected C ABI export surface')
checks={'status':'passed','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_revision':revision['revision'],'source_files':len(source),'source_snapshot_rechecked':True,'source_overrides':False,'cache_policy':'Existing common Cargo caches reused; no clean-cache claim','available':True,'serialized_real_native_smoke':True,'isolated_configuration_saved':True,'protected_production_native_hashes_unchanged':True,'main_configuration_unchanged':before['input/mg7_configuration.txt']==after['input/mg7_configuration.txt'],'native_artifacts':{name:{'sha256':sha(lib/name),'bytes':(lib/name).stat().st_size} for name in tirust_install.QUALIFIED_ARTIFACTS},'notice_files':{name:{'sha256':sha(lib/name),'bytes':(lib/name).stat().st_size} for name in config['dependency_notices']['files']},'native_export_count':len(exports),'commands':commands,'source_patch_sha256':config['local_sources']['tirust']['patch_sha256'],'build_provenance':config['build_provenance'],'bridge_source_sha256':source['vendor/tirust/src/lib.rs'],'fortran_source_sha256':source['vendor/tirust/mtirust.f90']}
write(report/'check-installation.json',checks)
(report/'README.md').write_text('''# Committed current-source installer qualification

The actual CLI completed `install tirust --jobs=4` from Git revision `606603e720aff7e96df0c9c7981e7bafd09b5d1c`, including the split DP workspace bridge. The source was obtained directly with `git archive`; `prepare_snapshot.py`, `source-revision.json` and `source-inventory.json` record the filtered implementation snapshot, materialized symlinks and omitted dangling generated-file symlinks. No untracked experiments or working-tree edits entered it.

This run uses pinned public source downloads without local source overrides and reuses the common Cargo dependency/build caches. It is a cached installer qualification, not a second clean-cache claim. The ordinary installer created its own fresh randomized stage and performed Cargo compilation, installed-artifact checks, real DP/QP/poles/cache smoke, transactional publication and configuration saving. No precompiled artifacts were transplanted into the new package.

The actual CLI log records publication and configuration saving under the isolated copied CLI tree; compilation uses the recorded common Cargo caches. The four recorded production library and metadata hashes remain unchanged. The separate main MG configuration changed at 22:40:22 UTC while this CLI was blocked in its smoke subprocess; its initial bytes were not retained, so an exact before/after text diff is unavailable. The observed hashes, mtime and later bytes are preserved. Retained concurrent export logs show six saving `set` commands per invocation falling back to the main install config; `config-drift-attribution.json` records their hashes and relevant lines. No claim of global configuration immutability is made across concurrent CLI activity. Scheduling instrumentation delays `make install` until the timing window releases and serializes only the real final smoke with the shared numerical lock. The waiting CLI was restricted to allowed CPUs2–5 before compilation, so subsequent compiler children inherit that affinity and leave benchmark CPU9 unused; `cpu-affinity.txt` records the operation, and `compile-affinity.json` verifies the running make/Cargo/rustc children inherited it. No compiler flags or source bytes were changed. No installation performance claim is made. Native binaries are not included in this evidence bundle.

The final smoke completed and the CLI published successfully during a period of severe unrelated host load. The queued smoke wrapper had already exited when an administrative hold was attempted, so no signal was sent and its test was not altered. `host-overload-context.json` and `administrative-hold.json` preserve this context. The separate 90-call replay is held until recovery; a completed correct smoke is retained as correctness evidence, with no runtime claim.

`check-installation.json` binds successful installer qualification to the actual published native artifact hashes, source revision, bridge source hash, final pole patch and resolved Cargo lock. The native smoke is narrower than the physical-amplitude acceptance campaign; any campaign or replay using this package must bind these actual artifact bytes separately. Dependency notice hashes were validated in the installed package; the text provenance and notice files are included here.
''')
write(report/'index.json',{'schema_version':1,'status':'passed','source_revision':revision['revision'],'artifacts':{p.relative_to(report).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(report.rglob('*')) if p.is_file()}})
outer=report.parent/'index.json'; index=json.loads(outer.read_text()); readme=report.parent/'README.md'
readme.write_text(readme.read_text()+'\nThe current split-workspace bridge also has an actual committed-source CLI publication in `current-source/index.json`. That run snapshots Git revision `606603e720`, reuses normal Cargo caches, downloads pinned dependencies without local overrides, and passes the real native smoke before isolated publication. Its native artifact hashes are recorded separately from earlier packages.\n')
index['artifacts']['README.md']={'sha256':sha(readme),'size':readme.stat().st_size}
for p in sorted(report.rglob('*')):
    if p.is_file(): index['artifacts'][p.relative_to(report.parent).as_posix()]={'sha256':sha(p),'size':p.stat().st_size}
index.setdefault('additional_qualifications',[]).append({'record':'current-source/index.json','status':'passed','source_revision':revision['revision'],'production_native_artifacts_unchanged':True,'main_configuration_changed_during_concurrent_activity':True,'cache_reuse':True})
write(outer,index)
print(json.dumps(checks,indent=2))
