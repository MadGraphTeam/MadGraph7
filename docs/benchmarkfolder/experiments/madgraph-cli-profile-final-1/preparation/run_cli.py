"""Bounded stock `check profile` invocation; root numerical GO is external."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,resource,shutil,stat,subprocess,sys,time
ROOT=Path.cwd();assert ROOT==Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE=Path(__file__).resolve().parent;MG=HERE/'mg'
parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('n1','n2'),default='n1');parser.add_argument('--backend',type=int,choices=(8,9),required=True);parser.add_argument('--run-label',default='1');parser.add_argument('--reuse-export',type=Path);parser.add_argument('--tirust-prefix',type=Path);args=parser.parse_args()
assert args.run_label.replace('-','').isalnum() and Path(args.run_label).name==args.run_label
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def dump(p,v):
 with p.open('x') as stream:json.dump(v,stream,indent=2,sort_keys=True);stream.write('\n')
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
cleanup=module(ROOT/'docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py','stock_profile_cleanup')
ownership=module(ROOT/'.tirust-build/generic-owned-snapshot-qualification-proposal-2/ownership.py','stock_profile_ownership')
prepared=read(HERE/'prepared.json')
assert prepared['status']=='prepared-not-executed' and prepared['schema']=='madgraph-stock-profile-preparation-v2'
assert sha(MG/'madgraph/various/process_checks.py')==prepared['process_checks_sha256']
source_before={str(MG/name):sha(MG/name) for name in prepared['source_files']}
source_command=HERE/f'{args.case}-{args.backend}.mg5'
assert sha(source_command)==prepared['commands'][source_command.name]['sha256']
prefix=(args.tirust_prefix or Path(prepared['tirust_prefix'])).resolve()
sys.path.insert(0,str(MG));from madgraph.various import tirust_install
assert prefix==Path(prepared['tirust_prefix']).resolve(), 'A changed prefix requires fresh preparation'
assert tirust_install.is_available(prefix/'lib') and tirust_install.supports_dp_route_statistics(prefix/'lib')
assert sha(prefix/'lib/libmadloop_tirust.so')==prepared['tirust_dso_sha256']
from observe_result import observe_result
identifier=f'{args.case}-{args.backend}-{args.run_label}'
CAPSULE=HERE/'runs'/identifier;CAPSULE.mkdir(parents=True,exist_ok=False)
OUT=HERE/'outputs'/identifier;OUT.mkdir(parents=True,exist_ok=False)
if args.reuse_export:
 source=args.reuse_export.resolve()
 assert source.is_relative_to(HERE/'outputs') and source.name.startswith('SAVEDTMP_CHECK_')
 assert (source/'SavedRunInfos').is_dir()
 previous=read(HERE/'runs'/source.parent.name/'terminal.json')
 assert previous['status']=='profile-complete'
 assert previous['tirust_prefix']==str(prefix), 'Changed dependency prefix requires a fresh export'
 assert previous['tirust_dso_sha256']==sha(prefix/'lib/libmadloop_tirust.so'), 'Changed runtime requires a fresh export'
 destination=OUT/source.name
 def ignore(directory,names):
  return [name for name in names if stat.S_ISFIFO((Path(directory)/name).lstat().st_mode)]
 shutil.copytree(source,destination,symlinks=True,ignore=ignore)
 for path in destination.rglob('*'):
  if path.is_symlink():
   target=path.resolve()
   if target.is_relative_to(source):
    new=destination/target.relative_to(source);path.unlink();path.symlink_to(os.path.relpath(new,path.parent),target_is_directory=new.is_dir())
command=source_command.read_text();original_line='set tirust '+prepared['tirust_prefix']+'/lib --no_save';replacement='set tirust '+str(prefix/'lib')+' --no_save'
assert command.count(original_line)==1;command=command.replace(original_line,replacement)
command_path=CAPSULE/'commands.mg5';command_path.write_text(command)
controls=read(ROOT/'.tirust-build/tirust-oem-upstream-environment-1/environment.json')
env=os.environ.copy()
for key in list(env):
 if key.startswith(('LD_','NIX_','SYMBOLICA_TEST_')) or key in ('SYMBOLICA_LICENSE','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','MAKEFLAGS','PYTHONPATH'):
  env.pop(key,None)
env.update(controls)
for name in ('symbolica-locks','tmp','config','cache'):(CAPSULE/name).mkdir(mode=0o700)
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(MG),SYMBOLICA_LOCK_DIR=str(CAPSULE/'symbolica-locks'),TMPDIR=str(CAPSULE/'tmp'),XDG_CONFIG_HOME=str(CAPSULE/'config'),XDG_CACHE_HOME=str(CAPSULE/'cache'),MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1',MAKEFLAGS='-j2 SHELL='+shutil.which('bash',path=env['PATH']),PWD=str(OUT))
env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,(ROOT/'HEPTools/ninja/lib',ROOT/'HEPTools/collier',ROOT/'HEPTools/ninjars/lib',prefix/'lib')))
shared=[ROOT/'input/mg7_configuration.txt']+[ROOT/'vendor/CutTools/includects'/name for name in ('libcts.a','mpmodule.mod','compiler_version.log')]
for directory in (ROOT/'HEPTools/ninja/lib',ROOT/'HEPTools/collier',ROOT/'HEPTools/ninjars/lib',prefix/'lib'):
 shared.extend(path for path in directory.iterdir() if path.is_file() and (path.suffix in ('.a','.so','.mod','.json','.txt') or '.so.' in path.name))
snapshot=lambda:{str(path):sha(path) for path in sorted(set(shared))}
before=snapshot();dump(CAPSULE/'shared-before.json',before)
CPU=12;assert CPU in os.sched_getaffinity(0)
start=time.monotonic();cap=3600.;child=owned=raw=None
record={'status':'running','case':args.case,'backend':args.backend,'run_label':args.run_label,'source_mg_revision':prepared['base_mg_revision'],'process_checks_sha256':prepared['process_checks_sha256'],'command_sha256':sha(command_path),'controller_sha256':sha(__file__),'output_directory':str(OUT),'cli_root':str(MG),'tirust_prefix':str(prefix),'tirust_dso_sha256':sha(prefix/'lib/libmadloop_tirust.so'),'cpu':CPU,'cpu_reservation':False,'cap_seconds':cap,'cleanup_reserve_seconds':60,'performance_acceptance':False,'precision_scope':'stock DP-interface timing with local tensor-tier policy; timed native routes and per-evaluation statuses unavailable; external stability analysis separate','route_statistics':prepared['route_statistics'],'timing_clocks':prepared['timing_clocks'],'source_manifest_sha256':sha(HERE/'prepared.json'),'host_load_before':list(os.getloadavg())}
with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with ownership.StageDeadline(cap,absolute_end=start+cap) as deadline,cleanup.SignalRequests() as requests:
  try:
   for kind,value in ((resource.RLIMIT_AS,32*1024**3),(resource.RLIMIT_FSIZE,2*1024**3),(resource.RLIMIT_CORE,0)):resource.setrlimit(kind,(value,value))
   dump(CAPSULE/'admission.json',record)
   argv=['taskset','-c',str(CPU),sys.executable,'-B',str(MG/'bin/madgraph'),str(command_path)]
   with (CAPSULE/'console.log').open('xb') as output:
    deadline.handoff=True
    try:
     child=subprocess.Popen(argv,cwd=OUT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True);raw=cleanup.identity(child.pid);owned=cleanup.OwnedChild(child)
    finally:deadline.handoff=False
    dump(CAPSULE/'started.json',{'pid':child.pid,'identity':owned.root,'argv':argv,'cwd':str(OUT)})
    print('START stock check profile '+identifier+' pid='+str(child.pid),flush=True)
    record['returncode']=owned.wait(requests);record['cleanup']=ownership.cleanup_owned(owned,deadline.end)
   if record['returncode']:raise RuntimeError('Stock profile exited unsuccessfully')
   text=(CAPSULE/'console.log').read_text(errors='replace')
   if 'Error detected' in text or 'Traceback (most recent call last)' in text:raise RuntimeError('MG CLI reported an error; retained console')
   profiles=list(OUT.glob('profile_optimized_*.log'));assert len(profiles)==1
   paths=list(OUT.glob('SAVEDTMP_CHECK_*/SavedRunInfos/profile_timing_metadata.json'));assert len(paths)==1
   metadata=read(paths[0]);assert metadata['reduction_backends']==[args.backend]
   assert metadata['backend_labels']==['TIRust' if args.backend==9 else 'NinjaRS']
   selection=metadata['selected_timing_helicity'];assert selection['filter_policy']=='all-helicities-explicit-poles' and selection['learned_filter_used'] is False
   assert selection['count_kind']=='configured' and selection['polarized_contribution_verified'] is False
   stages={row['name']:row for row in metadata['stages']}
   assert set(('boot','polarized_full','unpolarized_full','unpolarized_skip','polarized_skip')).issubset(stages)
   for stage in stages.values():
    effective={key.lower():value for key,value in stage['effective_madloop_parameters'].items()}
    assert effective['looppolemode']==0 and effective['ctmoderun']==1
    assert effective['tirustcomputefinite'] and effective['useqpintegrandfortirust'] and effective['useqpintegrandforninjars']
   result=metadata.get('profile_timing_result',metadata['timing_result'])
   summary={'schema':'madgraph-stock-profile-result-v1','backend':args.backend,'backend_label':metadata['backend_labels'][0],'process':prepared['commands'][source_command.name]['process'],'source_mg_revision':prepared['base_mg_revision'],'initialization_seconds':result.get('Initialization'),'boot_seconds':result['Booting_time'],'unpolarized_full_ms':result['run_unpolarized_total']*1000,'unpolarized_skip_ms':result['run_unpolarized_coefs']*1000,'unpolarized_full_minus_skip_estimate_ms':(result['run_unpolarized_total']-result['run_unpolarized_coefs'])*1000,'polarized_full_ms':result['run_polarized_total']*1000,'polarized_skip_ms':result['run_polarized_coefs']*1000,'stage_points':{name:row['actual_npoints'] for name,row in stages.items()},'precision_scope':record['precision_scope'],'per_evaluation_statuses_available':False,'performance_acceptance':False,'metadata_path':str(paths[0]),'metadata_sha256':sha(paths[0]),'profile_log_sha256':sha(profiles[0]),'tirust_dso_sha256':record['tirust_dso_sha256']}
   summary.update(route_statistics=prepared['route_statistics'],timing_clocks=prepared['timing_clocks'],unpolarized_last_result=observe_result(stages['unpolarized_full']),polarized_last_result=observe_result(stages['polarized_full']),polarized_scope='configured, contribution unverified; diagnostic only',stage_wall_seconds={name:row['wall_seconds'] for name,row in stages.items()})
   assert result['timing_helicity_selection']=='configured-unverified'
   dump(CAPSULE/'summary.json',summary)
   shutil.copy2(profiles[0],CAPSULE/'profile.log')
   record.update(status='profile-complete',summary_sha256=sha(CAPSULE/'summary.json'),metadata_sha256=sha(paths[0]),saved_export=str(paths[0].parents[1]))
  except BaseException as error:
   record.update(status='failed',error=repr(error))
   if owned is not None and 'cleanup' not in record:record['cleanup']=ownership.cleanup_owned(owned,deadline.end)
   elif child is not None and owned is None:record['cleanup']=ownership.raw_cleanup(child,raw,cleanup,deadline.end)
   raise
  finally:
   record['shared_unchanged']=snapshot()==before
   record['source_changed']=[path for path,value in source_before.items() if sha(path)!=value]
   record.update(elapsed_seconds=time.monotonic()-start,console_sha256=sha(CAPSULE/'console.log') if (CAPSULE/'console.log').exists() else None,host_load_after=list(os.getloadavg()))
   dump(CAPSULE/'terminal.json',record)
   assert record['shared_unchanged'] and not record['source_changed']
print('CLOSED stock check profile '+identifier+' status='+record['status'],flush=True)
