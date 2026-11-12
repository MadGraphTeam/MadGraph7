"""One diagnosed physical point, five backends, seven matched timing cycles.

prepare performs file work only. build and run require separate coordinator GO.
No original source, fixture, output, installation or historical result is mutated.
"""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,math,os,re,resource,shutil,stat,struct,subprocess,sys,time
ROOT=Path.cwd();assert ROOT==Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE=ROOT/'.tirust-build/zggg-first-point-five-backend-timing-1';OLD=ROOT/'.tirust-build/zggg-five-backend-timing-1';SEALED=OLD/'closed-packet'
POINT='8880598b2d448930eedda40d9797fd137a5845c5289345d1e31cbcd22ff10c3b'
OLD_INDEX='e401cf5d4efb7708c277172eaa3be9f29bcadf54c5b3b484e7311109cf9a26cb'
BACKENDS=(1,6,7,8,9);ORDERS=((1,6,7,8,9),(7,1,8,9,6),(6,8,1,7,9),(9,7,1,8,6),(7,1,9,6,8),(8,9,6,1,7),(6,9,8,7,1))
ROUTES=('Direct','MethodChoice','Tier106','Tier256','Tier512','Refused','unreported/reduction-error','attempted106','attempted256','attempted512')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def common():
 sys.path[:0]=[str(ROOT/'tools'),str(ROOT)]
 import tirust_benchmark as bench
 for k in list(os.environ):
  if k.startswith(('LD_','NIX_','SYMBOLICA_TEST_')) or k in ('SYMBOLICA_LICENSE','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','MAKEFLAGS','PYTHONPATH'):os.environ.pop(k,None)
 os.environ.update(read(ROOT/'.tirust-build/tirust-oem-upstream-environment-1/environment.json'))
 bench.WORK=HERE/'work';bench.REPORT=HERE/'observations';return bench

def prepare():
 assert sha(SEALED/'index.json')==OLD_INDEX
 points=ROOT/'.tirust-build/tirust-initialization-clean-pair-1/first-point.dat';assert sha(points)==POINT
 assert points.read_text()=='1\n'+'\n'.join((OLD/'measurements/points/qcd_n4_uux_zggg-first10.dat').read_text().splitlines()[1:7])+'\n'
 shutil.copy2(points,HERE/'first-point.dat')
 original=SEALED/'generated-source/SubProcesses/P0_uux_zggg/check_sa.f';driver=original.read_text()
 assert sha(original)==read(OLD/'compilation-pending.json')['driver_sha256']
 driver=driver.replace('      IMPLICIT NONE\n','      USE MTIRUST, ONLY: TIRUST_DP_ROUTE_STATISTICS, TIRUST_SUCCESS\n      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T\n      IMPLICIT NONE\n',1)
 driver=driver.replace('      INTEGER PREVIOUS_SEQUENCE\n','      INTEGER PREVIOUS_SEQUENCE\n      INTEGER TIR_ROUTE_STATUS_BEFORE,TIR_ROUTE_STATUS_AFTER\n      INTEGER(C_INT64_T) TIR_ROUTES_BEFORE(20),TIR_ROUTES_AFTER(20)\n',1)
 before='''      TIMED_CODES=0
      CALL TIRUST_DP_ROUTE_STATISTICS(TIR_ROUTES_BEFORE,
     $ TIR_ROUTE_STATUS_BEFORE)
      IF(TIR_ROUTE_STATUS_BEFORE.NE.TIRUST_SUCCESS) STOP 6
      CALL SYSTEM_CLOCK(T0)
'''
 assert driver.count('      TIMED_CODES=0\n      CALL SYSTEM_CLOCK(T0)\n')==1
 driver=driver.replace('      TIMED_CODES=0\n      CALL SYSTEM_CLOCK(T0)\n',before)
 after='''      CALL SYSTEM_CLOCK(T1)
      CALL TIRUST_DP_ROUTE_STATISTICS(TIR_ROUTES_AFTER,
     $ TIR_ROUTE_STATUS_AFTER)
      IF(TIR_ROUTE_STATUS_AFTER.NE.TIRUST_SUCCESS) STOP 6
      IF(ANY(TIR_ROUTES_AFTER.LT.TIR_ROUTES_BEFORE)) STOP 6
      OPEN(76,FILE='tirust_routes.dat',STATUS='REPLACE')
      WRITE(76,*) 1,SEQUENCE,NLOOPS,NPOINTS,
     $ TIR_ROUTE_STATUS_BEFORE,TIR_ROUTE_STATUS_AFTER
      WRITE(76,*) TIR_ROUTES_BEFORE
      WRITE(76,*) TIR_ROUTES_AFTER
      WRITE(76,*) TIR_ROUTES_AFTER-TIR_ROUTES_BEFORE
      CLOSE(76)
      ELAPSED=DBLE(T1-T0)/DBLE(RATE)
'''
 assert driver.count('      CALL SYSTEM_CLOCK(T1)\n      ELAPSED=DBLE(T1-T0)/DBLE(RATE)\n')==1
 driver=driver.replace('      CALL SYSTEM_CLOCK(T1)\n      ELAPSED=DBLE(T1-T0)/DBLE(RATE)\n',after)
 (HERE/'worker-route-observer.f').write_text(driver)
 # Remove the additions and require byte-identical legacy driver/math/loops.
 restored=driver.replace('      USE MTIRUST, ONLY: TIRUST_DP_ROUTE_STATISTICS, TIRUST_SUCCESS\n      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T\n','').replace('      INTEGER TIR_ROUTE_STATUS_BEFORE,TIR_ROUTE_STATUS_AFTER\n      INTEGER(C_INT64_T) TIR_ROUTES_BEFORE(20),TIR_ROUTES_AFTER(20)\n','').replace(before,'      TIMED_CODES=0\n      CALL SYSTEM_CLOCK(T0)\n').replace(after,'      CALL SYSTEM_CLOCK(T1)\n      ELAPSED=DBLE(T1-T0)/DBLE(RATE)\n')
 assert restored==original.read_text()
 positions={str(b):[sum(order[p]==b for order in ORDERS) for p in range(5)] for b in BACKENDS}
 precedence={str(a)+':'+str(b):sum(order.index(a)<order.index(b) for order in ORDERS) for i,a in enumerate(BACKENDS) for b in BACKENDS[i+1:]}
 assert all(set(v)<=set((1,2)) for v in positions.values()) and set(precedence.values())<=set((3,4))
 dump(HERE/'specification.json',{'schema':'madloop-zggg-diagnosed-first-point-five-backend-v1','status':'prepared-file-only-awaiting-final-prefix-and-GO','case':read(OLD/'specification.json')['case'],'scope':'one explicitly selected diagnosed physical fixture; no process-average or main-campaign performance claim','points':1,'point_sha256':POINT,'membership_one_based':[1],'original100_sha256':read(OLD/'measurements/specification.json')['full_points_sha256'],'original10_sha256':read(OLD/'measurements/specification.json')['pilot_points_sha256'],'diagnosis_fixture':str(points),'original_packet_index_sha256':OLD_INDEX,'original_mg_revision':read(OLD/'specification.json')['mg_revision'],'original_driver_sha256':sha(original),'driver_sha256':sha(HERE/'worker-route-observer.f'),'cycles':[{'cycle':i+1,'order':list(row)} for i,row in enumerate(ORDERS)],'position_counts':positions,'pairwise_precedence_counts':precedence,'route_blocks':['primary','independent-check'],'route_names':list(ROUTES),'route_scope':'cumulative full-DP bridge execution final routes and actual tier attempts; Fortran coefficient-cache hits are outside these counters; unreported bucket makes attempt counts lower bounds','counter_clock_policy':'before T0 and after T1; writes outside timed interval; no math/loop changes','historical_status_scope':'H2/U9 identifies DP-input dispatch; internal tiers are not genuine binary128-input QP and are not native-DP-only','minimum_batch_seconds':1.1,'relative_tolerance':3e-6,'ordinary_target_tolerance':1e-8,'cpu':12,'cpu_reserved':False,'performance_acceptance':False,'no_compiler_or_native_executed':True})
 print('PREPARED point/identical-math route-observer driver/seven balanced matched orders; no compilation or numerical evaluation')

def clone(src,dest):
 def ignored(directory,names):return [n for n in names if n in ('worker-raw','build-history','masses.dat','warmup.dat','dimensions.dat','timing.dat','values.dat','timed_status.dat','tirust_routes.dat') or stat.S_ISFIFO((Path(directory)/n).lstat().st_mode)]
 shutil.copytree(src,dest,symlinks=True,ignore=ignored)
 for p in dest.rglob('*'):
  if p.is_symlink():
   target=p.resolve()
   if target.is_relative_to(src):
    replacement=dest/target.relative_to(src);p.unlink();p.symlink_to(os.path.relpath(replacement,p.parent),target_is_directory=replacement.is_dir())
 return dest

def objects(output):return {p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.suffix in ('.o','.a') and p.name!='check_sa.o'}
def validate_qualification(prefix,path,dso):
 # Authenticate the actual final installer receipt chain; file reads only.
 path=path.resolve();base=path.parent;published=read(base/'published.json');qualified=read(base/'qualified.json');compiled=read(base/'compiled.json');live=read(base/'live-abi.json')
 assert path.name in ('published.json','qualified.json')
 assert published['status']=='published-private-candidate' and Path(published['library_directory']).resolve()==prefix/'lib'
 assert qualified['status']=='sixteen-tests-native-routes-genuine-qp-poles-smoke-passed' and qualified['sdk_smoke_executed'] is True and qualified['release_tests_passed']>=16
 assert published['qualified_receipt_sha256']==sha(base/'qualified.json') and qualified['compiled_receipt_sha256']==sha(base/'compiled.json') and qualified['live_abi_receipt_sha256']==sha(base/'live-abi.json')
 assert live['status']=='passed' and live['abi_version']==1 and live['dp_route_statistics'] is True and live['count']==20 and live['preserves_prior_error_on_success'] is True and live['mapped_library_sha256']==dso
 assert compiled['immutable_core_sources_reused'] is True and compiled['precision_tiers'] is True and compiled['partial_rescue'] is False
 assert published['candidate_source_sha256']==qualified['candidate_source_sha256']==compiled['candidate_source_sha256']==live['source_sha256']
 for name in ('lib/libmadloop_tirust.so','lib/libtirust.a','lib/mtirust.mod'):
  assert qualified['artifacts'][name]==published['artifacts'][name]==sha(prefix/name)
 for name,checksum in published['artifacts'].items():
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  candidate=prefix/name;assert candidate.is_file() and sha(candidate)==checksum
 assert qualified['qualification_log_sha256']==sha(base/'qualification.log') and 'TIRust tensor-route statistics passed' in (base/'qualification.log').read_text()
 return {'status':'final-published-qualified-artifacts-authenticated','prefix':str(prefix),'receipt_sha256':{name:sha(base/name) for name in ('published.json','qualified.json','compiled.json','live-abi.json','qualification.log')},'getter_count':20,'getter_preserves_prior_error':True,'dso_sha256':dso,'candidate_source_sha256':compiled['candidate_source_sha256']}

def build(args):
 assert sha(HERE/'worker-route-observer.f')==read(HERE/'specification.json')['driver_sha256']
 bench=common();from tirust_benchmark_compiler import capture_effective_flags
 from madgraph.various import tirust_install
 assert bench.generated_sources(OLD/'output')==read(OLD/'compilation-pending.json')['sources']
 assert sha(OLD/'output/SubProcesses/P0_uux_zggg/check')=='ffb951914a087327ae05ed63e90fa9ecfa3603137a254cca2781fc431791ffa3'
 prefix=args.prefix.resolve();assert prefix.is_dir() and args.expected_dso and args.qualification.is_file()
 assert prefix not in ((ROOT/'.tirust-build/initialization-adaptive-tiers-build-1/install').resolve(),(ROOT/'.tirust-build/tirust-oem-signed-prefix-1/HEPTools/tirust').resolve())
 qualification=validate_qualification(prefix,args.qualification,args.expected_dso)
 assert sha(prefix/'lib/libmadloop_tirust.so')==args.expected_dso and tirust_install.supports_dp_route_statistics(str(prefix/'lib'))
 bench.TIRUST_PREFIX=prefix;env=bench.environment();env['MAKEFLAGS']='-j2 SHELL='+shutil.which('bash',path=env['PATH']);output=clone(OLD/'output',HERE/'build/output');directory=output/'SubProcesses/P0_uux_zggg'
 oldlib=str(ROOT/'.tirust-build/tirust-oem-signed-prefix-1/HEPTools/tirust/lib');newlib=str(prefix/'lib')
 definitions=output/'SubProcesses/MadLoop_makefile_definitions';text=definitions.read_text();assert oldlib in text;definitions.write_text(text.replace(oldlib,newlib))
 for name in ('tirust_include','tirust_lib'):
  link=output/'lib'/name;assert link.is_symlink();link.unlink();link.symlink_to(prefix/'lib',target_is_directory=True)
 before_objects=objects(output);shutil.copy2(HERE/'worker-route-observer.f',directory/'check_sa.f')
 before_sources=bench.generated_sources(output);before_flags=capture_effective_flags(directory,env=env);assert bench.same_compiler_policy(before_flags,read(OLD/'compilation-pending.json')['effective_compiler_flags'])
 shared={**bench.dependency_inputs(directory)['archives'],**bench.dependency_inputs(directory)['metadata'],str(ROOT/'input/mg7_configuration.txt'):sha(ROOT/'input/mg7_configuration.txt'),str(prefix/'lib/libmadloop_tirust.so'):args.expected_dso}
 with (HERE/'build.log').open('xb') as log:subprocess.run(['make','-j2','check'],cwd=directory,env={**env,'PWD':str(directory)},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180.)
 assert objects(output)==before_objects and bench.generated_sources(output)==before_sources and bench.same_compiler_policy(before_flags,capture_effective_flags(directory,env=env))
 assert {p:sha(p) for p in shared}==shared
 record={'status':'built-only-no-numerical-evaluation','output':str(output),'directory':str(directory),'prefix':str(prefix),'dso_sha256':args.expected_dso,'qualification':qualification,'executable_sha256':sha(directory/'check'),'driver_sha256':sha(directory/'check_sa.f'),'sources':before_sources,'objects_excluding_driver':before_objects,'compiler_flags':before_flags,'shared':shared,'parameter_card_sha256':sha(output/'Cards/param_card.dat'),'original_parameter_card_sha256':read(OLD/'compilation-pending.json')['parameter_card_sha256'],'old_math_objects_unchanged':True,'new_source_binding':{p.name:sha(p) for p in (prefix/'lib/tirust-config.json',prefix/'lib/libtirust.a',prefix/'lib/mtirust.mod')}}
 assert record['parameter_card_sha256']==record['original_parameter_card_sha256'];dump(HERE/'build.json',record);print('BUILT getter-observer driver only; old O0 numerical objects/flags/cards unchanged')

def loader(exe):
 raw=exe.read_bytes();assert raw[:6]==b'\x7fELF\x02\x01';offset=struct.unpack_from('<Q',raw,32)[0];size,count=struct.unpack_from('<HH',raw,54);found=[]
 for i in range(count):
  entry=offset+i*size
  if struct.unpack_from('<I',raw,entry)[0]==3:
   pos,length=struct.unpack_from('<Q',raw,entry+8)[0],struct.unpack_from('<Q',raw,entry+32)[0];found.append(Path(raw[pos:pos+length].rstrip(b'\0').decode()).resolve())
 assert len(found)==1;return found[0]

def route_delta(path,sequence,loops):
 values=list(map(int,path.read_text().split()));assert len(values)==66
 header=values[:6];before=values[6:26];after=values[26:46];delta=values[46:66]
 assert header==[1,sequence,loops,1,0,0] and all(a>=b>=0 and d==a-b for a,b,d in zip(after,before,delta))
 blocks={name:dict(zip(ROUTES,delta[i*10:(i+1)*10])) for i,name in enumerate(('primary','independent-check'))}
 return {'schema':'madloop-tirust-timed-route-delta-v1','header':header,'before':before,'after':after,'delta':delta,'blocks':blocks,'counter_clock_policy':'getter before T0 and after T1; writes outside clock','precision_scope':'DP-input local tensor arithmetic; genuine QP is separately identified by amplitude status','fortran_coefficient_cache_hits_outside_scope':True,'attempt_counts_lower_bounds':any(block['unreported/reduction-error'] for block in blocks.values())}

# Build/run entrypoints are inert until invoked after coordinator GO.
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','build','run'));parser.add_argument('--prefix',type=Path);parser.add_argument('--expected-dso');parser.add_argument('--qualification',type=Path);args=parser.parse_args()
 if args.stage=='prepare':prepare()
 elif args.stage=='build':build(args)
 else:run()

def run():
 bench=common();from tirust_benchmark_worker import Worker,raw_outputs
 from tirust_benchmark_warmup import parse_warmup
 bound=read(HERE/'build.json');spec=read(HERE/'specification.json');prefix=Path(bound['prefix']);bench.TIRUST_PREFIX=prefix;env=bench.environment();env['MAKEFLAGS']='-j2 SHELL='+shutil.which('bash',path=env['PATH'])
 assert int(os.environ['ZGGG_POINT_LOCK_OWNER_PID'])==os.getppid()
 end=float(os.environ['ZGGG_POINT_ABSOLUTE_END'])
 def remaining():
  value=end-60.-time.monotonic();assert value>0;return value
 assert sha(HERE/'first-point.dat')==POINT and sha(prefix/'lib/libmadloop_tirust.so')==bound['dso_sha256']
 assert {p:sha(p) for p in bound['shared']}==bound['shared'] and sha(Path(bound['directory'])/'check')==bound['executable_sha256']
 result=HERE/'observations';result.mkdir(exist_ok=False)
 config=read(OLD/'measurements/configuration.json');config['points_per_case']=1;case=config['cases'][0]
 cpu=spec['cpu'];assert cpu in os.sched_getaffinity(0)
 records={};workers={};checks={str(b):[] for b in BACKENDS};cycles=[];reference=None
 for kind,value in ((resource.RLIMIT_AS,32*1024**3),(resource.RLIMIT_FSIZE,2*1024**3),(resource.RLIMIT_CORE,0)):resource.setrlimit(kind,(value,value))
 expected={str((prefix/'lib/libmadloop_tirust.so').resolve()):bound['dso_sha256']}
 def maps(worker):
  raw=Path('/proc/'+str(worker.process.pid)+'/maps').read_text();assert '(deleted)' not in raw
  paths={line.split()[-1] for line in raw.splitlines() if len(line.split())>=6 and line.split()[-1].startswith('/')}
  actual={p:sha(p) for p in paths if Path(p).name=='libmadloop_tirust.so'};assert actual==expected
  return {'pid':worker.process.pid,'signed_tirust_dso':actual,'raw_maps':raw}
 def observe(backend,loops,tag):
  key=str(backend);worker=workers[key];worker.timeout=min(60.,remaining());target=result/'lanes'/key/'raw'/tag;route_file=worker.directory/'tirust_routes.dat';route_file.unlink(missing_ok=True)
  before=bench.host_load_snapshot()
  try:value=bench.observe(worker,[backend],loops,target,count=1)
  finally:
   if route_file.is_file():shutil.copy2(route_file,target/'tirust_routes.dat')
  routes=route_delta(target/'tirust_routes.dat',worker.sequence,loops)
  if backend!=9:assert routes['delta']==[0]*20,'Unused TIRust getter initialized/mutated a context'
  dump(target/'routes.json',routes);dump(target/'host.json',{'before':before,'after':bench.host_load_snapshot()})
  comparison=bench.compare_rows(reference['rows'],value['rows'],3e-6);dump(target/'comparison.json',comparison);checks[key].append({'path':str(target/'comparison.json'),'result':comparison})
  return {'observation':value,'routes':routes,'dispatch_scope':'Historical parser fields describe MadLoop input/dispatch precision, not local tensor arithmetic'}
 try:
  dump(result/'binding.json',{'specification_sha256':sha(HERE/'specification.json'),'build_sha256':sha(HERE/'build.json'),'controller_sha256':sha(__file__),'point_sha256':POINT,'prefix':str(prefix),'expected_dso':expected,'machine':bench.machine_identity(cpu),'config':config,'old_packet_index_sha256':OLD_INDEX,'performance_acceptance':False})
  for backend in BACKENDS:
   key=str(backend);folder=result/'lanes'/key;folder.mkdir(parents=True);record={'backend':backend,'status':'initializing'};records[key]=record
   try:
    output=clone(Path(bound['output']),HERE/'lanes'/key/'output');directory=output/'SubProcesses/P0_uux_zggg';record['directory']=str(directory)
    bench.output_dir=lambda source,case:output;settings=bench.configure('candidate',case,[backend],config);record['settings']=settings
    assert settings['tirustcomputefinite'] is True and settings['looppolemode']==0 and settings['colliercomputeuvpoles'] is True and settings['colliercomputeirpoles'] is True and settings['tirustglobalcache']==-1
    assert sha(directory/'check')==bound['executable_sha256'] and sha(output/'Cards/param_card.dat')==bound['parameter_card_sha256']
    assert objects(output)==bound['objects_excluding_driver'] and bench.generated_sources(output)==bound['sources']
    localenv=dict(env,SYMBOLICA_LOCK_DIR=str(folder/'symbolica-locks'));(folder/'symbolica-locks').mkdir(mode=0o700)
    interpreter=loader(directory/'check');listing=subprocess.check_output([str(interpreter),'--list',str(directory/'check')],cwd=directory,env=localenv,text=True,timeout=min(10.,remaining()));(folder/'loader-list.txt').write_text(listing)
    loaded={p:sha(p) for p in re.findall(r'(?:=>\s*)?(/[^\s]+)',listing) if Path(p).name=='libmadloop_tirust.so'};assert loaded==expected
    dump(folder/'inputs.json',{'settings':settings,'parameter_card_sha256':sha(output/'Cards/param_card.dat'),'numerical_cards':bench.numerical_card_inputs(output),'executable_sha256':sha(directory/'check'),'driver_sha256':sha(directory/'check_sa.f'),'compiler_flags':bound['compiler_flags'],'runtime':bench.runtime_identity(directory),'point_sha256':POINT,'loader_verified_dso':loaded})
    print('INITIALIZE backend='+key,flush=True)
    worker=Worker(['taskset','-c',str(cpu),'./check',str(HERE/'first-point.dat')],directory,folder/'output.log',env=localenv,timeout=min(120.,remaining()));workers[key]=worker
    ready={name:getattr(worker,name) for name in ('points','ndim','nso','nborn','cold_seconds')};assert ready['points']==1;record['initialization']=ready;dump(folder/'ready.json',ready);dump(folder/'maps-before.json',maps(worker))
    startup=folder/'startup';startup.mkdir()
    for name in ('warmup.dat','dimensions.dat'):shutil.copy2(directory/name,startup/name)
    learning=parse_warmup(startup,[backend],1,ready);dump(folder/'learning.json',learning)
    if backend==1:reference=learning
    if reference is not None:
     comparison=bench.compare_rows(reference['rows'],learning['rows'],3e-6);checks[key].append({'path':str(folder/'learning-comparison.json'),'result':comparison});dump(folder/'learning-comparison.json',comparison)
    record['status']='ready';dump(folder/'status.json',record);print('READY backend='+key+' seconds='+str(ready['cold_seconds']),flush=True)
   except Exception as error:
    record.update(status='failed-initialization',error_type=type(error).__name__,error=repr(error));dump(folder/'failure.json',record)
    for name,value in getattr(error,'raw_outputs',{}).items():
     if Path(name).name==name:
      p=folder/'failed-startup-raw'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    if key in workers:workers.pop(key).close(graceful=False)
  assert reference is not None,'No CutTools reference; no timing comparison is qualified'
  settings=[records[str(b)]['settings'] for b in BACKENDS if str(b) in workers];assert all({k:v for k,v in row.items() if k!='mlreductionlib'}=={k:v for k,v in settings[0].items() if k!='mlreductionlib'} for row in settings)
  for backend in BACKENDS:
   key=str(backend)
   if key not in workers:continue
   try:
    observed=observe(backend,1,'calibration');loops=max(1,math.ceil(spec['minimum_batch_seconds']/observed['observation']['seconds']));records[key]['loops']=loops;dump(result/'lanes'/key/'calibration.json',{'loops':loops,**observed})
   except Exception as error:
    records[key].update(status='failed-calibration',error=repr(error));dump(result/'lanes'/key/'failure.json',records[key]);workers.pop(key).close(graceful=False)
  for row in spec['cycles']:
   observed={}
   for position,backend in enumerate(row['order'],1):
    key=str(backend)
    if key not in workers:continue
    try:
     loops=records[key]['loops'];value=observe(backend,loops,'cycle-'+str(row['cycle']));observed[key]={'position':position,'loops':loops,**value};print('CYCLE '+str(row['cycle'])+' backend='+key+' ms='+str(value['observation']['seconds_per_point']*1000),flush=True)
    except Exception as error:
     records[key].update(status='failed-cycle',cycle=row['cycle'],error=repr(error));dump(result/'lanes'/key/'failure.json',records[key]);workers.pop(key).close(graceful=False)
   cycle={**row,'observations':observed};cycles.append(cycle);dump(result/('cycle-'+str(row['cycle'])+'.json'),cycle)
  for key,worker in workers.items():dump(result/'lanes'/key/'maps-after.json',maps(worker));records[key]['status']='complete'
  dump(result/'result.json',{'status':'observations-collected','lanes':records,'checks':checks,'cycles':cycles,'point_sha256':POINT,'scope':spec['scope'],'physics_regression_pass':all(records[str(b)]['status']=='complete' and all(check['result']['passed'] for check in checks[str(b)]) for b in BACKENDS),'performance_acceptance':False})
 except BaseException as error:
  dump(result/'failure.json',{'error_type':type(error).__name__,'error':repr(error),'lanes':records,'checks':checks,'cycles':cycles,'performance_acceptance':False});raise
 finally:
  for key,worker in workers.items():
   try:worker.close(graceful=True);dump(result/'lanes'/key/'STOP-BYE.json',{'closed':True})
   except Exception as error:dump(result/'cleanup-failure-'+key+'.json',{'error':repr(error)})
  after={p:sha(p) for p in bound['shared']};dump(result/'shared-after.json',after)
  if after!=bound['shared']:dump(result/'shared-integrity-failure.json',{'before':bound['shared'],'after':after});raise RuntimeError('Shared dependency/config changed')
 print('CLOSED one diagnosed point; observations only, no process-average acceptance',flush=True)

if __name__=='__main__':main()
