"""One requested additional process; existing MadLoop persistent worker, all failures retained."""
from pathlib import Path
from contextlib import ExitStack
import collections, fcntl, hashlib, importlib.util, json, math, os, resource, shutil, stat, struct, subprocess, sys, time
ROOT=Path.cwd();HERE=ROOT/'.tirust-build/zggg-five-backend-timing-1';OUT=HERE/'output';REPORT=HERE/'measurements'
assert ROOT==Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT))
import tirust_benchmark as bench
from tirust_benchmark_worker import Worker
from tirust_benchmark_warmup import parse_warmup
from tirust_benchmark_points import generate_points,write_points
PREFIX=ROOT/'.tirust-build/tirust-oem-signed-prefix-1/HEPTools/tirust'
bench.WORK=HERE/'work';bench.REPORT=REPORT;bench.TIRUST_PREFIX=PREFIX
BACKENDS=(1,6,7,8,9);POINTS=10;SEED=20261112;TARGET=1.1
EXPECTED_DSO='85ee0cdf860e5ad4c7a5c26163802673de22404690a10c7c53fb73ebe1ff5bb0'
sha=bench.digest;dump=bench.dump
read=lambda p:json.loads(Path(p).read_text())
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ownership=module(ROOT/'.tirust-build/generic-owned-snapshot-qualification-proposal-2/ownership.py','zggg_numerical_ownership')
controls=read(ROOT/'.tirust-build/tirust-oem-upstream-environment-1/environment.json')
for k in list(os.environ):
 if k.startswith(('LD_','NIX_','SYMBOLICA_TEST_')) or k in ('SYMBOLICA_LICENSE','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','MAKEFLAGS','PYTHONPATH'):os.environ.pop(k,None)
os.environ.update(controls);os.environ.update(PYTHONDONTWRITEBYTECODE='1')
ENV=bench.environment();ENV['MAKEFLAGS']='-j2'
CPU=12 if 12 in os.sched_getaffinity(0) else min(os.sched_getaffinity(0))
rotations=[list(BACKENDS[i:]+BACKENDS[:i]) for i in range(5)]
SCHEDULE=[{'cycle':i+1,'order':r} for i,r in enumerate(rotations+[list(reversed(r)) for r in rotations])]
start=time.monotonic();cap=3600.;workers={};lanes={};checks={str(b):[] for b in BACKENDS};cycles=[];reference=None
CASE=read(HERE/'specification.json')['case'];pending=read(HERE/'compilation-pending.json');DIRECTORY=Path(pending['directory'])
config=read(ROOT/'docs/benchmarkfolder/cases.json');config.update(points_per_case=10,cases=[CASE],seed=SEED)
config['madloop_parameters']['COLLIERUseCacheForPoles']=False
config['madloop_parameters']['COLLIERComputeUVpoles']=True;config['madloop_parameters']['COLLIERComputeIRpoles']=True
config['candidate_parameters'].update(TIRustComputeFinite=True,UseQPIntegrandForTIRust=True,TIRustGlobalCache=-1,TIRustCacheMaxBytes=268435456)
REPORT.mkdir(exist_ok=False)
shutil.copy2(HERE/'shared-before.json',REPORT/'shared-before.json')
dump(REPORT/'configuration.json',config)
shared_before=read(HERE/'shared-before.json')
def shared():return {p:sha(p) for p in shared_before}
def remaining():
 value=deadline.end-60-time.monotonic()
 if value<=0:raise TimeoutError('Requested zggg numerical window expired')
 return value

def mappings(worker):
 paths=sorted({line.split()[-1] for line in Path('/proc/'+str(worker.process.pid)+'/maps').read_text().splitlines() if len(line.split())>=6 and line.split()[-1].startswith('/')})
 result={p:sha(p) for p in paths if Path(p).is_file()}
 matches={p:v for p,v in result.items() if Path(p).name=='libmadloop_tirust.so'}
 assert matches=={str((PREFIX/'lib/libmadloop_tirust.so').resolve()):EXPECTED_DSO},'Live worker did not map exact signed TIRust S DSO'
 return {'pid':worker.process.pid,'mapped_files':result,'signed_tirust_dso':matches}

def interpreter(executable):
 raw=Path(executable).read_bytes();assert raw[:6]==b'\x7fELF\x02\x01'
 offset=struct.unpack_from('<Q',raw,32)[0];size,count=struct.unpack_from('<HH',raw,54);found=[]
 for index in range(count):
  entry=offset+index*size
  if struct.unpack_from('<I',raw,entry)[0]==3:
   pos,length=struct.unpack_from('<Q',raw,entry+8)[0],struct.unpack_from('<Q',raw,entry+32)[0];found.append(raw[pos:pos+length].rstrip(b'\0').decode())
 assert len(found)==1 and Path(found[0]).is_absolute();return Path(found[0]).resolve()

def clone(backend):
 dest=HERE/'lanes'/str(backend)/'output'
 def ignored(directory,names):
  return [n for n in names if n in ('worker-raw','build-history') or (Path(directory)==OUT and n=='lib') or stat.S_ISFIFO((Path(directory)/n).lstat().st_mode)]
 shutil.copytree(OUT,dest,symlinks=True,ignore=ignored);(dest/'lib').symlink_to(OUT/'lib',target_is_directory=True)
 for path in dest.rglob('*'):
  if path.is_symlink():
   target=path.resolve()
   if target.is_relative_to(OUT) and not target.is_relative_to(OUT/'lib'):
    new=dest/target.relative_to(OUT);path.unlink();path.symlink_to(os.path.relpath(new,path.parent),target_is_directory=new.is_dir())
 return dest

def settings(output,backend):
 bench.output_dir=lambda source,case:output
 value=bench.configure('candidate',CASE,[backend],config)
 assert value['looppolemode']==0 and value['tirustcomputefinite'] is True and value['colliercomputeuvpoles'] is True and value['colliercomputeirpoles'] is True
 assert value['collierusecacheforpoles'] is False and value['tirustglobalcache']==-1
 for p in [output/'Cards',*(output/'Cards').rglob('*'),*((output/'SubProcesses').rglob('MadLoopParams.dat'))]:assert p.resolve().is_relative_to(output),'Mutable card escapes lane'
 return value

def observe(backend,loops,tag):
 lane=lanes[str(backend)];worker=workers[str(backend)];worker.timeout=min(300.,remaining());destination=REPORT/'lanes'/str(backend)/'raw'/tag
 before=bench.host_load_snapshot();value=bench.observe(worker,[backend],loops,destination,count=POINTS)
 dump(destination/'host.json',{'before':before,'after':bench.host_load_snapshot()})
 comparison=bench.compare_rows(reference['rows'],value['rows'],3e-6)
 dump(destination/'comparison.json',comparison);checks[str(backend)].append({'path':str(destination/'comparison.json'),'result':comparison})
 return value

with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with ownership.StageDeadline(cap,absolute_end=float(os.environ.get('ZGGG_ABSOLUTE_END',start+cap))) as deadline:
  try:
   for kind,value in ((resource.RLIMIT_AS,32*1024**3),(resource.RLIMIT_FSIZE,2*1024**3),(resource.RLIMIT_CORE,0)):resource.setrlimit(kind,(value,value))
   assert shared()==shared_before and read(HERE/'preparation.json')['status']=='compiled-awaiting-numerical-window'
   assert sha(DIRECTORY/'check')==pending['executable_sha256'] and bench.generated_sources(OUT)==pending['sources']
   assert bench.dependency_inputs(DIRECTORY)==pending['link_inputs_after']
   with (REPORT/'mass-probe.log').open('xb') as f:subprocess.run(['./check','--masses'],cwd=DIRECTORY,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=remaining(),check=True,start_new_session=True)
   masses=list(map(float,(DIRECTORY/'masses.dat').read_text().split()));assert len(masses)==6
   inputs={'masses':masses[2:],'incoming_masses':masses[:2],'seed':SEED,'count':100,'energy':1000.}
   fixture=generate_points(**inputs);points=REPORT/'points';points.mkdir();full=points/'qcd_n4_uux_zggg-100.dat';pilot=points/'qcd_n4_uux_zggg-first10.dat'
   write_points(full,fixture);write_points(pilot,fixture[:POINTS])
   dump(full.with_suffix('.json'),{'inputs':inputs,'sha256':sha(full),'points':fixture})
   dump(pilot.with_suffix('.json'),{'inputs':dict(inputs,count=POINTS),'sha256':sha(pilot),'points':fixture[:POINTS],'full_fixture_sha256':sha(full),'membership_one_based':list(range(1,11)),'selection':'unselected deterministic prefix frozen before amplitude evaluation'})
   loader=interpreter(DIRECTORY/'check');listing=subprocess.check_output([str(loader),'--list',str(DIRECTORY/'check')],cwd=DIRECTORY,env=ENV,text=True,timeout=remaining());(REPORT/'loader-list.txt').write_text(listing);dump(REPORT/'loader.json',{'path':str(loader),'sha256':sha(loader),'listing_sha256':sha(REPORT/'loader-list.txt')})
   runtime=bench.runtime_identity(DIRECTORY);assert runtime['libraries'].get(str(PREFIX/'lib/libmadloop_tirust.so'))==EXPECTED_DSO or EXPECTED_DSO in runtime['libraries'].values()
   compilation=dict(pending,status='compiled-and-real-mass-probe-complete',masses=masses,numerical_build_from_sources=True)
   dump(REPORT/'compilation.json',compilation)
   specification={**read(HERE/'specification.json'),'status':'running','points_per_traversal':POINTS,'cycles':SCHEDULE,'calibration_target_seconds':TARGET,'cpu':CPU,'machine':bench.machine_identity(CPU),'host_load_before':bench.host_load_snapshot(),'cpu_reservation':False,'host_exclusivity':'Not assumed; descriptive shared-host measurements','runtime':runtime,'full_points_sha256':sha(full),'pilot_points_sha256':sha(pilot),'frozen_membership_one_based':list(range(1,11)),'mass_probe_completed_before_points':True,'madloop_configuration':config,'worker_template_sha256':pending['worker_template_sha256'],'numerical_controller_sha256':sha(__file__),'compiler':pending['compiler'],'compiler_flags':pending['effective_compiler_flags'],'initialization_scope':'Fortran READY SYSTEM_CLOCK includes first amplitude plus all warmup, excludes process launch and SETPARA','finite_and_poles':'full finite+UV+IR; ordinary amplitudes','performance_acceptance':False}
   dump(REPORT/'specification.json',specification)
   for backend in BACKENDS:
    key=str(backend);folder=REPORT/'lanes'/key;folder.mkdir(parents=True);record={'backend':backend,'status':'initializing'};lanes[key]=record
    try:
     output=clone(backend);directory=next((output/'SubProcesses').glob('P*_*'));record['output']=str(output);record['settings']=settings(output,backend)
     assert sha(directory/'check')==pending['executable_sha256']
     assert sha(output/'Cards/param_card.dat')==pending['parameter_card_sha256']
     dump(folder/'inputs.json',{'settings':record['settings'],'physical_card_sha256':sha(output/'Cards/param_card.dat'),'cards':bench.numerical_card_inputs(output),'executable_sha256':sha(directory/'check'),'points_sha256':sha(pilot),'compiler':pending['compiler'],'compiler_flags':pending['effective_compiler_flags'],'runtime':bench.runtime_identity(directory)})
     print('INITIALIZE backend='+key,flush=True)
     worker=Worker(['taskset','-c',str(CPU),'./check',str(pilot)],directory,folder/'output.log',env=ENV,timeout=min(600.,remaining()));workers[key]=worker
     ready={name:getattr(worker,name) for name in ('points','ndim','nso','nborn','cold_seconds')};record['initialization']=ready;dump(folder/'ready.json',ready)
     dump(folder/'maps-before.json',mappings(worker))
     startup=folder/'startup';startup.mkdir()
     for name in ('warmup.dat','dimensions.dat'):shutil.copy2(directory/name,startup/name)
     learning=parse_warmup(startup,[backend],POINTS,ready);dump(folder/'learning.json',learning)
     if backend==1:reference=learning
     if reference is not None:
      check=bench.compare_rows(reference['rows'],learning['rows'],3e-6);dump(folder/'learning-comparison.json',check);checks[key].append({'path':str(folder/'learning-comparison.json'),'result':check})
     record['status']='ready';dump(folder/'status.json',record);print('READY backend='+key+' seconds='+str(ready['cold_seconds']),flush=True)
    except Exception as e:
     record.update(status='failed-initialization',error_type=type(e).__name__,error=repr(e));dump(folder/'failure.json',record);
     for name,value in getattr(e,'raw_outputs',{}).items():
      if Path(name).name==name:
       target=folder/'failed-startup-raw'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(value)
     print('FAILED backend='+key+' '+repr(e),flush=True)
     if key in workers:workers.pop(key).close(graceful=False)
   if reference is None:raise RuntimeError('No valid CutTools reference; all timed lanes remain unavailable')
   ready_settings=[value['settings'] for value in lanes.values() if value.get('status')=='ready']
   if ready_settings:
    common=set(ready_settings[0])-{'mlreductionlib'}
    assert all({k:value[k] for k in common}=={k:ready_settings[0][k] for k in common} for value in ready_settings),'Backend cards differ beyond backend selection'
   for backend in BACKENDS:
    key=str(backend)
    if key not in workers:continue
    try:
     initial=observe(backend,1,'calibration-0');loops=max(1,math.ceil(TARGET/initial['seconds']));sustained=observe(backend,loops,'calibration-1');loops=max(1,math.ceil(loops*TARGET/sustained['seconds']))
     lanes[key]['calibration']={'target_seconds':TARGET,'loops':loops,'initial':initial,'sustained':sustained};dump(REPORT/'lanes'/key/'calibration.json',lanes[key]['calibration']);print('CALIBRATED backend='+key+' loops='+str(loops),flush=True)
    except Exception as e:
     lanes[key].update(status='failed-calibration',error_type=type(e).__name__,error=repr(e));dump(REPORT/'lanes'/key/'failure.json',lanes[key]);workers.pop(key).close(graceful=False)
   for row in SCHEDULE:
    values={}
    for position,backend in enumerate(row['order'],1):
     key=str(backend)
     if key not in workers:continue
     try:
      loops=lanes[key]['calibration']['loops'];value=observe(backend,loops,'cycle-'+str(row['cycle']));values[key]={'loops':loops,'position':position,'observation':value}
      print('CYCLE '+str(row['cycle'])+' backend='+key+' ms='+str(value['seconds_per_point']*1000),flush=True)
     except Exception as e:
      lanes[key].update(status='failed-cycle',failed_cycle=row['cycle'],error_type=type(e).__name__,error=repr(e));dump(REPORT/'lanes'/key/'failure.json',lanes[key]);workers.pop(key).close(graceful=False)
    cycle={**row,'observations':values};cycles.append(cycle);dump(REPORT/('cycle-'+str(row['cycle'])+'.json'),cycle)
   for key,worker in workers.items():dump(REPORT/'lanes'/key/'maps-after.json',mappings(worker));lanes[key]['status']='complete'
   physics_pass=all(lanes[str(b)]['status']=='complete' and checks[str(b)] and all(v['result']['passed'] for v in checks[str(b)]) for b in BACKENDS)
   dump(REPORT/'observations.json',{'lanes':lanes,'checks':checks,'cycles':cycles,'physics_regression_pass':physics_pass,'performance_acceptance':False})
  except BaseException as e:
   dump(REPORT/'failure.json',{'error_type':type(e).__name__,'error':repr(e),'elapsed_seconds':time.monotonic()-start,'lanes':lanes,'checks':checks,'cycles':cycles,'performance_acceptance':False});raise
  finally:
   for worker in workers.values():
    try:worker.close()
    except Exception as e:dump(REPORT/('cleanup-failure-'+str(worker.process.pid)+'.json'),{'error':repr(e)})
   after=shared();dump(REPORT/'shared-after.json',after)
   if after!=shared_before:
    dump(REPORT/'shared-integrity-failure.json',{'error':'Shared numerical dependencies/config changed','before':shared_before,'after':after});raise RuntimeError('Shared numerical dependencies/config changed')
print('CLOSED numerical observations; pure replay summary is next',flush=True)
