"""Explicit, bounded, isolated first-point diagnostic. Run only after root grants numerical window."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,resource,shutil,stat,struct,subprocess,sys,time
ROOT=Path.cwd();assert ROOT==Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE=ROOT/'.tirust-build/tirust-initialization-profile-full-o2-1'
O2_BUILD=ROOT/'.tirust-build/zggg-fortran-o2-model-experiment-1'
BASE_PROFILE=ROOT/'.tirust-build/tirust-initialization-profile-1'
ORIGINAL=ROOT/'.tirust-build/zggg-five-backend-timing-1'
PREFIX=ROOT/'.tirust-build/tirust-oem-signed-prefix-1/HEPTools/tirust'
EXPECTED_DSO='85ee0cdf860e5ad4c7a5c26163802673de22404690a10c7c53fb73ebe1ff5bb0'
EXPECTED_EXE='c4b5a2bdc7ed555939239360e357fbcc09fbae06cee35e816f3133c5848648f2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def clone():
 src=O2_BUILD/'output';dest=HERE/'output'
 def ignored(directory,names):
  return [n for n in names if n in ('worker-raw','build-history','masses.dat','warmup.dat','dimensions.dat','timing.dat','values.dat','timed_status.dat') or (Path(directory)==src and n=='lib') or stat.S_ISFIFO((Path(directory)/n).lstat().st_mode)]
 shutil.copytree(src,dest,symlinks=True,ignore=ignored);(dest/'lib').symlink_to(src/'lib',target_is_directory=True)
 for path in dest.rglob('*'):
  if path.is_symlink():
   target=path.resolve()
   if target.is_relative_to(src) and not target.is_relative_to(src/'lib'):
    new=dest/target.relative_to(src);path.unlink();path.symlink_to(os.path.relpath(new,path.parent),target_is_directory=new.is_dir())
 return dest

parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--run-native',action='store_true');parser.add_argument('--run-label',default='o2-1');args=parser.parse_args()
if args.prepare_only==args.run_native:parser.error('Choose --prepare-only or --run-native explicitly')
if Path(args.run_label).name!=args.run_label or not args.run_label.replace('-','').isalnum():parser.error('Invalid run label')
sys.path[:0]=[str(ROOT/'tools'),str(HERE)]
import tirust_benchmark as bench
from tirust_benchmark_worker import Worker,raw_outputs
from summarize import summarize
for k in list(os.environ):
 if k.startswith(('LD_','NIX_','SYMBOLICA_TEST_')) or k in ('SYMBOLICA_LICENSE','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','MAKEFLAGS','PYTHONPATH'):os.environ.pop(k,None)
os.environ.update(read(ROOT/'.tirust-build/tirust-oem-upstream-environment-1/environment.json'))
bench.TIRUST_PREFIX=PREFIX
ENV=bench.environment();ENV['MAKEFLAGS']='-j2'
source_manifest=read(ORIGINAL/'compilation-pending.json')
closure=read(HERE/'build-bindings/closure.json')
assert closure['executable_sha256']==EXPECTED_EXE
for binding,digest in read(HERE/'build-binding-index.json').items():assert sha(HERE/'build-bindings'/binding)==digest
assert sha(PREFIX/'lib/libmadloop_tirust.so')==EXPECTED_DSO
output=HERE/'output'
if args.prepare_only:
 assert not output.exists()
 output=clone()
 bench.output_dir=lambda source,case:output
 config=read(ORIGINAL/'measurements/configuration.json');case=config['cases'][0]
 settings=bench.configure('candidate',case,[9],config)
 full=ORIGINAL/'measurements/points/qcd_n4_uux_zggg-first10.dat'
 lines=full.read_text().splitlines();assert lines[0].strip()=='10'
 points=HERE/'first-point.dat';points.write_bytes((BASE_PROFILE/'first-point.dat').read_bytes());assert sha(points)=='8880598b2d448930eedda40d9797fd137a5845c5289345d1e31cbcd22ff10c3b'
 directory=next((output/'SubProcesses').glob('P*_*'))
 assert sha(directory/'check')==EXPECTED_EXE
 assert sha(output/'Cards/param_card.dat')==source_manifest['parameter_card_sha256']
 assert bench.generated_sources(output)==source_manifest['sources']
 assert bench.dependency_inputs(directory)==read(HERE/'build-bindings/link-inputs-after.json')
 assert bench.numerical_card_inputs(output)==bench.numerical_card_inputs(BASE_PROFILE/'output')
 for p in [output/'Cards',*(output/'Cards').rglob('*'),*((output/'SubProcesses').rglob('MadLoopParams.dat'))]:assert p.resolve().is_relative_to(output),'Mutable card escapes profile clone'
 dump(HERE/'preparation.json',{'status':'prepared-not-executed','directory':str(directory),'points_sha256':sha(points),'source_cohort_sha256':sha(full),'point_membership_one_based':[1],'executable_sha256':EXPECTED_EXE,'tirust_dso_sha256':EXPECTED_DSO,'settings':settings,'native_deadline_seconds':270.,'standard_shutdown_and_replay_reserve_seconds':30.,'all_inclusive_supervisor_cap_seconds':360.,'outer_cleanup_reserve_seconds':60.,'optimized_fortran_build_closure_sha256':sha(HERE/'build-bindings/closure.json'),'optimized_fortran_flag_policy':closure['scope'],'effective_compiler_flags_sha256':sha(HERE/'build-bindings/effective-compiler-flags.json'),'baseline_source_index_sha256':sha(BASE_PROFILE/'source-index-baseline-2.json'),'baseline_profile_protected':read(HERE/'baseline2-preservation-before.json'),'shared_state':{p:sha(p) for p in read(ORIGINAL/'shared-before.json')},'profile_scope':'Fortran O2 control, unchanged polynomial O3, MODEL also rebuilt O2; same first frozen point, first amplitude plus one-point warmup until READY or 270s timeout; same C ABI shim','instrumentation_is_benchmark':False})
 print('PREPARED only; no native amplitude, master integral or SDK was run')
 raise SystemExit(0)

prep=read(HERE/'preparation.json');directory=Path(prep['directory']);destination=HERE/'runs'/args.run_label
destination.mkdir(parents=True,exist_ok=False)
assert sha(directory/'check')==EXPECTED_EXE and sha(PREFIX/'lib/libmadloop_tirust.so')==EXPECTED_DSO
assert sha(HERE/'first-point.dat')==prep['points_sha256']
assert {p:sha(p) for p in prep['shared_state']}==prep['shared_state']
assert {p:sha(p) for p in prep['baseline_profile_protected']}==prep['baseline_profile_protected']
assert bench.generated_sources(output)==source_manifest['sources']
assert bench.dependency_inputs(directory)==read(HERE/'build-bindings/link-inputs-after.json')
ENV.update(LD_PRELOAD=str(HERE/'libtirust_phase_profile.so'),ML_TIRUST_PROFILE=str(destination/'phases.jsonl'))
CPU=12;assert CPU in os.sched_getaffinity(0)
result={'status':'starting','native_work_cap_seconds':270.,'standard_shutdown_and_replay_reserve_seconds':30.,'cpu':CPU,'cpu_reservation':False,'source_executable_sha256':EXPECTED_EXE,'tirust_dso_sha256':EXPECTED_DSO,'point_sha256':prep['points_sha256'],'profile_sources':{p.name:sha(p) for p in [HERE/'trace.c',HERE/'libtirust_phase_profile.so',HERE/'profile.py',HERE/'summarize.py']},'host_load_before':bench.host_load_snapshot(),'precision_and_stability_policy':'unchanged','optimized_fortran_build_closure_sha256':sha(HERE/'build-bindings/closure.json'),'effective_compiler_flags_sha256':sha(HERE/'build-bindings/effective-compiler-flags.json'),'baseline_source_index_sha256':sha(BASE_PROFILE/'source-index-baseline-2.json')}
worker=None;start=time.monotonic()
with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  for kind,value in ((resource.RLIMIT_AS,32*1024**3),(resource.RLIMIT_FSIZE,2*1024**3),(resource.RLIMIT_CORE,0)):resource.setrlimit(kind,(value,value))
  executable=directory/'check';raw=executable.read_bytes();assert raw[:6]==b'\x7fELF\x02\x01'
  offset=struct.unpack_from('<Q',raw,32)[0];size,count=struct.unpack_from('<HH',raw,54);loaders=[]
  for index in range(count):
   entry=offset+index*size
   if struct.unpack_from('<I',raw,entry)[0]==3:
    pos,length=struct.unpack_from('<Q',raw,entry+8)[0],struct.unpack_from('<Q',raw,entry+32)[0];loaders.append(Path(raw[pos:pos+length].rstrip(b'\0').decode()).resolve())
  assert len(loaders)==1
  listing=subprocess.check_output([str(loaders[0]),'--list',str(executable)],cwd=directory,env=ENV,text=True,timeout=10.)
  (destination/'loader-list.txt').write_text(listing)
  paths=re.findall(r'(?:=>\s*)?(/[^\s]+)',listing)
  result['loader_tirust']={p:sha(p) for p in paths if Path(p).name=='libmadloop_tirust.so'}
  assert result['loader_tirust']=={str((PREFIX/'lib/libmadloop_tirust.so').resolve()):EXPECTED_DSO}
  result['loader_identity']={'path':str(loaders[0]),'sha256':sha(loaders[0]),'listing_sha256':sha(destination/'loader-list.txt')}
  dump(destination/'started.json',result)
  worker=Worker(['taskset','-c',str(CPU),'./check',str(HERE/'first-point.dat')],directory,destination/'output.log',env=ENV,timeout=min(270.,max(.001,float(os.environ.get('PROFILE_ABSOLUTE_END',time.monotonic()+360.))-90.-time.monotonic())))
  result.update(status='ready',initialization={k:getattr(worker,k) for k in ('points','ndim','nso','nborn','cold_seconds')},pid=worker.process.pid)
  paths=sorted({line.split()[-1] for line in Path('/proc/'+str(worker.process.pid)+'/maps').read_text().splitlines() if len(line.split())>=6 and line.split()[-1].startswith('/')})
  result['mapped_tirust']={p:sha(p) for p in paths if Path(p).name=='libmadloop_tirust.so'}
  assert result['mapped_tirust']=={str((PREFIX/'lib/libmadloop_tirust.so').resolve()):EXPECTED_DSO}
 except Exception as e:
  result.update(status='failed-or-work-cap-reached',error_type=type(e).__name__,error=repr(e))
 finally:
  if worker is not None:
   try:worker.close(graceful=False)
   except Exception as e:result['cleanup_error']=repr(e)
  result['elapsed_seconds']=time.monotonic()-start
  for name,contents in raw_outputs(directory).items():(destination/name).write_text(contents)
  events=destination/'phases.jsonl'
  maps=destination/'phases.jsonl.maps'
  if maps.exists():
   paths=sorted({line.split()[-1] for line in maps.read_text().splitlines() if len(line.split())>=6 and line.split()[-1].startswith('/')})
   result['first_cabi_mapped_tirust']={p:sha(p) for p in paths if Path(p).name=='libmadloop_tirust.so'}
   result['first_cabi_maps_sha256']=sha(maps)
   assert result['first_cabi_mapped_tirust']=={str((PREFIX/'lib/libmadloop_tirust.so').resolve()):EXPECTED_DSO}
  if events.exists():
   summary=summarize(events);dump(destination/'phase-summary.json',summary);result['phase_summary_sha256']=sha(destination/'phase-summary.json');result['phase_events_sha256']=sha(events)
  result['host_load_after']=bench.host_load_snapshot()
  result['shared_after']={p:sha(p) for p in prep['shared_state']};result['shared_unchanged']=result['shared_after']==prep['shared_state']
  result['baseline_profile_unchanged']={p:sha(p) for p in prep['baseline_profile_protected']}==prep['baseline_profile_protected']
  dump(destination/'result.json',result)
  assert result['shared_unchanged'] and result['baseline_profile_unchanged']
print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'result':str(destination/'result.json')}))
