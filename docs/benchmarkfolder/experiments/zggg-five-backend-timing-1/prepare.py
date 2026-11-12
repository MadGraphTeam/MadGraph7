from pathlib import Path
import hashlib, importlib.util, json, os, re, resource, shutil, subprocess, sys, tarfile, time
ROOT=Path.cwd(); HERE=ROOT/'.tirust-build/zggg-five-backend-timing-1'; MG=HERE/'mg'; OUT=HERE/'output'
assert ROOT==Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
REV='cd0145922977fcfdc4e3cb3183ddc974c31a7897'
CASE={'id':'qcd_n4_uux_zggg','model':'loop_sm','process':'u u~ > z g g g [virt=QCD]','nfinal':4,'role':'additional-timing-control'}
PREFIX=ROOT/'.tirust-build/tirust-oem-signed-prefix-1/HEPTools/tirust'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
cleanup=module(ROOT/'docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py','zggg_cleanup')
ownership=module(ROOT/'.tirust-build/generic-owned-snapshot-qualification-proposal-2/ownership.py','zggg_ownership')
controls=json.loads((ROOT/'.tirust-build/tirust-oem-upstream-environment-1/environment.json').read_text())
env=os.environ.copy()
for k in list(env):
 if k.startswith(('LD_','NIX_','SYMBOLICA_TEST_')) or k in ('SYMBOLICA_LICENSE','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','MAKEFLAGS','PYTHONPATH'):env.pop(k,None)
env.update(controls);env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1',MAKEFLAGS='-j2 SHELL='+shutil.which('bash',path=env['PATH']),XDG_CONFIG_HOME=str(HERE/'config'))
env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,(ROOT/'HEPTools/ninja/lib',ROOT/'HEPTools/collier',ROOT/'HEPTools/ninjars/lib',PREFIX/'lib')))
shared=[ROOT/'input/mg7_configuration.txt']+[ROOT/'vendor/CutTools/includects'/name for name in ('libcts.a','mpmodule.mod','compiler_version.log')]
for d in (ROOT/'HEPTools/ninja/lib',ROOT/'HEPTools/collier',ROOT/'HEPTools/ninjars/lib',PREFIX/'lib'):
 shared.extend(p for p in d.iterdir() if p.is_file() and (p.suffix in ('.a','.so','.mod','.json','.txt') or '.so.' in p.name))
def snapshot():return {str(p):sha(p) for p in sorted(set(shared))}
start=time.monotonic();cap=7200.;before=snapshot();dump(HERE/'shared-before.json',before)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==REV
record={'status':'running','stage':'compile-only','case':CASE,'mg_revision':REV,'native_numerical_executions':0,'cap_seconds':cap,'shared_before':before,'seed':20261112,'fixture_points':100,'pilot_points':10,'cycles':10,'minimum_batch_seconds':1.,'performance_acceptance':False}
dump(HERE/'preparation.json',record)
def run(argv,log,cwd=ROOT):
 env['PWD']=str(cwd)
 with log.open('xb') as stream:
  deadline.handoff=True
  try:
   p=subprocess.Popen(argv,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);raw=cleanup.identity(p.pid);owned=cleanup.OwnedChild(p)
  finally:deadline.handoff=False
  dump(log.with_suffix('.started.json'),{'pid':p.pid,'identity':owned.root,'argv':argv,'cwd':str(cwd)})
  print('START '+log.name+' pid='+str(p.pid),flush=True)
  try:
   rc=owned.wait(requests);closure=ownership.cleanup_owned(owned,deadline.end)
  except BaseException:
   ownership.cleanup_owned(owned,deadline.end);raise
  if rc:raise RuntimeError(str(argv)+' exit='+str(rc))
  dump(log.with_suffix('.terminal.json'),{'returncode':rc,'cleanup':closure,'sha256':sha(log)})
  print('PASS '+log.name,flush=True)
with ownership.StageDeadline(cap,absolute_end=start+cap) as deadline,cleanup.SignalRequests() as requests:
 try:
  for kind,value in ((resource.RLIMIT_AS,32*1024**3),(resource.RLIMIT_FSIZE,2*1024**3),(resource.RLIMIT_CORE,0)):resource.setrlimit(kind,(value,value))
  pass # Exact additional archive already extracted; private sources unchanged
  for name in ('libcts.a','mpmodule.mod','compiler_version.log'):
   dst=MG/'vendor/CutTools/includects'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'vendor/CutTools/includects'/name,dst)
  shutil.rmtree(MG/'vendor/IREGI',ignore_errors=True)
  gf='/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran'
  cpp='/nix/store/788mx070y81zjlg5ipcl0cra3afviw9k-gcc-wrapper-15.2.0/bin/c++'
  lines=['set automatic_html_opening False --no_save','set auto_update 0 --no_save','set nb_core 2 --no_save','set loop_optimized_output True --no_save','set output_dependencies external --no_save','set complex_mass_scheme False --no_save',f'set fortran_compiler {gf} --no_save',f'set cpp_compiler {cpp} --no_save',f'set ninja {ROOT}/HEPTools/ninja/lib --no_save',f'set collier {ROOT}/HEPTools/collier --no_save',f'set ninjars {ROOT}/HEPTools/ninjars/lib --no_save',f'set tirust {PREFIX}/lib --no_save','import model loop_sm','generate '+CASE['process'],f'save processes {HERE}/processes.pkl',f'output standalone {OUT}','quit']
  command=HERE/'generate.mg5';command.write_text('\n'.join(lines)+'\n');dump(HERE/'specification.json',{'case':CASE,'seed':20261112,'fixture_points':100,'pilot_points':10,'cycles':10,'minimum_batch_seconds':1.,'mg_revision':REV,'tirust_revision':'10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1','tirust_dso_sha256':sha(PREFIX/'lib/libmadloop_tirust.so'),'command_sha256':sha(command),'scope':'shared-host-pilot','performance_acceptance':False})
  # Actual generation receipt retained; resume the compile-only stage below.
  if 'Error detected' in (HERE/'generation.log').read_text() or not (OUT/'Cards/param_card.dat').is_file():raise RuntimeError('MG CLI output failed; inspect generation.log')
  sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT))
  import tirust_benchmark as bench
  bench.WORK=HERE/'work';bench.REPORT=HERE/'measurements';bench.TIRUST_PREFIX=PREFIX
  from madgraph.models.check_param_card import ParamCard
  card=ParamCard(str(OUT/'Cards/param_card.dat'))
  for decay in card['decay']:decay.value=0.
  card.write(str(OUT/'Cards/param_card.dat'))
  dirs=sorted((OUT/'SubProcesses').glob('P*_*'));assert len(dirs)==1;directory=dirs[0]
  original=(directory/'check_sa.f').read_text();prefix=re.search(r'CALL\s+(\w*)GET_ANSWER_DIMENSION',original,re.I).group(1)
  born=directory/'nsqso_born.inc';nborn=int(re.search(r'NSQSO_BORN\s*=\s*(\d+)',born.read_text(),re.I).group(1)) if born.exists() else 0
  parameters={'prefix':prefix,'nexternal':6,'nborn':nborn};template=ROOT/'tools/tirust_benchmark_worker.f'
  (directory/'check_sa.f').write_text(bench.render_worker(template.read_text(),parameters))
  from tirust_benchmark_compiler import capture_effective_flags
  flags=capture_effective_flags(directory,env=env);sources=bench.generated_sources(OUT)
  links_before=bench.dependency_inputs(directory)
  # MODEL and DHELAS full source rebuilds already succeeded; retained logs bound below.
  run(['make','--always-make','-j2','check'],HERE/'compile-process.log',directory)
  links_after=bench.dependency_inputs(directory)
  # Internal model/helicity archives are intentionally rebuilt; external inputs must be exact.
  internal={str((OUT/'lib'/n).resolve()) for n in ('libmodel.a','libdhelas.a')}
  strip=lambda x:{'metadata':x['metadata'],'archives':{k:v for k,v in x['archives'].items() if k not in internal}}
  assert strip(links_before)==strip(links_after)
  assert bench.generated_sources(OUT)==sources
  assert bench.same_compiler_policy(flags,capture_effective_flags(directory,env=env))
  pending={'status':'compiled-awaiting-real-mass-probe','case':CASE,'sources':sources,'driver_sha256':sha(directory/'check_sa.f'),'worker_template_sha256':sha(template),'worker_render_parameters':parameters,'executable_sha256':sha(directory/'check'),'effective_compiler_flags':flags,'link_inputs_before':links_after,'link_inputs_after':links_after,'native_numerical_executions':0,'compiler':subprocess.check_output([gf,'--version'],text=True).splitlines()[0],'parameter_card_sha256':sha(OUT/'Cards/param_card.dat'),'directory':str(directory)}
  dump(HERE/'compilation-pending.json',pending)
  record.update(status='compiled-awaiting-numerical-window',subprocess=str(directory),executable_sha256=pending['executable_sha256'],elapsed_seconds=time.monotonic()-start)
 except BaseException as error:
  record.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-start);raise
 finally:
  after=snapshot();dump(HERE/'shared-after.json',after);record['shared_unchanged']=after==before;dump(HERE/'preparation.json',record)
  assert after==before,'Shared source/config/libraries changed'
print('READY compiled; numerical executions=0',flush=True)
