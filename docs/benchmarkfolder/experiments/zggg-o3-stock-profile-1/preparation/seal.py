"""Seal only this closed actual-profile capture; pure file work, no native replay."""
from pathlib import Path
import collections,fcntl,hashlib,importlib.util,json,os,re,shlex,shutil,time
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent;PACK=HERE/'closed-packet';BASE=HERE/'build/SAVEDTMP_CHECK_uux_zggg'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
assert not PACK.exists();PACK.mkdir()
# Both retained attempts are necessary: first successfully compiled the model module.
build=read(HERE/'build.json');commands=[]
for label in ('build-run','build-run-2'):
 for line in (HERE/label/'console.log').read_text().splitlines():
  if '/bin/gfortran ' in line and ' -c ' in line:
   tokens=shlex.split(line);assert '-O3' in tokens and not any(t in tokens for t in ('-O0','-O1','-O2','-Ofast','-ffast-math','-funsafe-math-optimizations'))
   output=tokens[tokens.index('-o')+1];commands.append({'build_log':label+'/console.log','command':line,'object_basename':Path(output).name})
objects={n:v for n,v in build['objects'].items() if n.endswith('.o')}
assert len(objects)==174 and len({Path(n).name for n in objects})==174
proof={}
for name,identity in objects.items():
 matches=[r for r in commands if r['object_basename']==Path(name).name];assert matches,name
 assert sha(BASE/name)==identity['sha256'],name
 proof[name]={**identity,'optimization':'-O3','compile_command_verified':True,'observed_compile_commands':matches}
compiler=build['effective_flags']['scopes']['process']['recipes']['loop_matrix.f']['compiler'][0]
dump(PACK/'build-proof.json',{'all_generated_objects_verified_O3':True,'generated_object_count':174,'objects':proof,'compiler':compiler,'unsafe_math_flags':False,'full_build_manifest_sha256':sha(HERE/'build.json'),'shared_reducer_libraries_rebuilt':False,'scope':'Every generated numerical model,DHELAS,process,driver object rebuilt at-O3; installed release reducer libraries unchanged. Subsequent stock driver/loop_matrix recompiles inherit same make_opts and FFLAGS.','initial_failed_parallel_model_dependency_race_retained':True})
copy(HERE/'build.json',PACK/'build.json')
for label in ('build-run','build-run-2'):
 for n in ('console.log','terminal.json','started.json','shared-before.json'):copy(HERE/label/n,PACK/label/n)
rows=[]
for backend in (1,6,7,8,9):
 lane=HERE/'runs'/str(backend);terminal=read(lane/'terminal.json');assert terminal['status']=='child-complete' and terminal['shared_unchanged'] and not terminal['cleanup']['survivors']
 rows.append(read(lane/'summary.json'))
 for p in lane.rglob('*'):
  if p.is_file() and p.parts[len(lane.parts)] not in ('symbolica-locks','tmp','config','cache'):copy(p,PACK/'runs'/str(backend)/p.relative_to(lane))
 saved=HERE/'outputs'/str(backend)/BASE.name/'SavedRunInfos'
 for p in saved.rglob('*'):
  if p.is_file() and p.suffix in ('.f','.json','.dat','.log','.txt','.pkl'):copy(p,PACK/'stages'/str(backend)/p.relative_to(saved))
dump(PACK/'summary.json',{'schema':'zggg-actual-stock-profile-O3-five-backend-v1','source_mg_revision':read(HERE/'prepared.json')['source_revision'],'rows':rows,'build_proof':'build-proof.json','performance_acceptance':False,'scope':'Actual stock check profile, single sequential capture per backend; full helicity-sum FULL ms/point. Unequal stage lengths, no paired confidence or main-campaign acceptance claim. Separate1-point stability phase is outside timings; no per-timed-point status/route proof.','timing_policy':'CTModeRun/Init1, legacy LoopPoleMode-1, learned filters,HFL2,finite and UV/IR enabled. TIRust DP interface can use internal local tensor tiers.','report_headline':'unpolarized_full_ms; FULL-SKIP is estimated reduction time only'})
spec=importlib.util.spec_from_file_location('o3_endpoint',HERE/'endpoint.py');ep=importlib.util.module_from_spec(spec);spec.loader.exec_module(ep);dump(PACK/'endpoint-consistency.json',ep.endpoint_consistency(PACK))
copy(HERE/'prepared.json',PACK/'prepared.json');copy(HERE/'profile-campaign-deadline.json',PACK/'profile-campaign-deadline.json')
for name in ('run.py','verify.py','seal.py','build.sh','endpoint.py'):copy(HERE/name,PACK/'preparation'/name)
for name in ('madgraph/various/process_checks.py','madgraph/various/misc.py','madgraph/interface/madevent_interface.py','madgraph/iolibs/template_files/loop/check_sa.inc','Template/loop_material/StandAlone/Cards/MadLoopParams.dat'):
 copy(HERE/'mg'/name,PACK/'stock-source'/name)
copy(BASE/'Source/make_opts',PACK/'generated-makefiles/Source/make_opts')
for name in ('pmass.inc','proc_prefix.txt','nexternal.inc','nsquaredSO.inc'):
 copy(BASE/'SubProcesses/P0_uux_zggg'/name,PACK/'generated-process'/name)
copy(BASE/'SubProcesses/P0_uux_zggg/makefile',PACK/'generated-makefiles/SubProcesses/P0_uux_zggg/makefile')
copy(BASE/'SubProcesses/MadLoop_makefile_definitions',PACK/'generated-makefiles/SubProcesses/MadLoop_makefile_definitions')
spec=importlib.util.spec_from_file_location('o3_seal_cleanup',ROOT/'docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py');clean=importlib.util.module_from_spec(spec);spec.loader.exec_module(clean)
identities=[i for b in (1,6,7,8,9) for i in read(HERE/'runs'/str(b)/'terminal.json')['cleanup']['observed_owned_processes']]
assert not any(clean.matches(i) for i in identities),'An observed owned process identity remains alive'
with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 dump(PACK/'native-release.json',{'all_five_profile_children_closed':True,'owned_survivors':[],'mutex_nonblocking_acquired':True,'source_files_unchanged':all(sha(HERE/'mg'/p)==v for p,v in read(HERE/'prepared.json')['source_files'].items()),'shared_unchanged_all_profiles':all(read(HERE/'runs'/str(b)/'terminal.json')['shared_unchanged'] for b in (1,6,7,8,9))})
files={p.relative_to(PACK).as_posix():sha(p) for p in sorted(PACK.rglob('*')) if p.is_file()};dump(PACK/'index.json',{'schema':'zggg-stock-profile-O3-closed-packet-v1','files':files,'performance_acceptance':False})
spec=importlib.util.spec_from_file_location('closed_o3_verify',PACK/'preparation/verify.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.verify(PACK)
print('SEALED '+str(PACK)+' index_sha256='+sha(PACK/'index.json'),flush=True)
