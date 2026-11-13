"""Bounded actual stock profile, generated Fortran rebuilt at O3; private evidence."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,re,resource,shutil,stat,subprocess,sys,tarfile,time
ROOT=Path.cwd();assert ROOT==Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE=Path(__file__).resolve().parent;MG=HERE/'mg';BASE=HERE/'build/SAVEDTMP_CHECK_uux_zggg'
DONOR=ROOT/'.tirust-build/zggg-first-point-five-backend-timing-1/build/output'
PREFIX=ROOT/'.tirust-build/tirust-native-tiers-final-1/HEPTools/tirust'
REV='cda704e28d634e893fabc3ac358d9a27789f93aa';DSO='d427667a0f2f55d0bd96778c1259ffcf52e3925341890b81c38f5f94e1eb7052'
FC='/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran'
CPP='/nix/store/788mx070y81zjlg5ipcl0cra3afviw9k-gcc-wrapper-15.2.0/bin/c++'
LABELS={1:'CutTools',6:'Ninja',7:'COLLIER',8:'NinjaRS',9:'TIRust'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
cleanup=module(ROOT/'docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py','o3_cleanup')
ownership=module(ROOT/'.tirust-build/generic-owned-snapshot-qualification-proposal-2/ownership.py','o3_ownership')
def clone(src,dst):
 def ignore(directory,names):return [n for n in names if stat.S_ISFIFO((Path(directory)/n).lstat().st_mode)]
 shutil.copytree(src,dst,symlinks=True,ignore=ignore)
 for p in dst.rglob('*'):
  if p.is_symlink() and p.resolve().is_relative_to(src):
   target=dst/p.resolve().relative_to(src);p.unlink();p.symlink_to(os.path.relpath(target,p.parent),target_is_directory=target.is_dir())
 return dst
def environment(capsule):
 env=os.environ.copy()
 for k in list(env):
  if k.startswith(('LD_','NIX_','SYMBOLICA_TEST_')) or k in ('SYMBOLICA_LICENSE','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','MAKEFLAGS','PYTHONPATH','FFLAGS'):env.pop(k,None)
 env.update(read(ROOT/'.tirust-build/tirust-oem-upstream-environment-1/environment.json'))
 for n in ('symbolica-locks','tmp','config','cache'):(capsule/n).mkdir(mode=0o700,parents=True,exist_ok=True)
 env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(MG),SYMBOLICA_LOCK_DIR=str(capsule/'symbolica-locks'),TMPDIR=str(capsule/'tmp'),XDG_CONFIG_HOME=str(capsule/'config'),XDG_CACHE_HOME=str(capsule/'cache'),MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1',FFLAGS='-w -fPIC -O3',MAKEFLAGS='-j2 SHELL='+shutil.which('bash',path=env['PATH']))
 env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,(ROOT/'HEPTools/ninja/lib',ROOT/'HEPTools/collier',ROOT/'HEPTools/ninjars/lib',PREFIX/'lib')))
 return env
def shared():
 paths=[ROOT/'input/mg7_configuration.txt']
 for d in (ROOT/'vendor/CutTools/includects',ROOT/'HEPTools/ninja/lib',ROOT/'HEPTools/collier',ROOT/'HEPTools/ninjars/lib',PREFIX/'lib',DONOR/'lib'):
  paths.extend(p for p in d.iterdir() if p.is_file() and (p.suffix in ('.a','.so','.mod','.json','.txt') or '.so.' in p.name))
 return {str(p):sha(p) for p in sorted(set(paths))}
def execute(argv,cwd,capsule,env,cap):
 capsule.mkdir(parents=True,exist_ok=True);before=shared();dump(capsule/'shared-before.json',before)
 env={**env,'PWD':str(cwd)};record={'argv':argv,'cwd':str(cwd),'cap_seconds':cap,'cpu_reserved':False,'start_load':list(os.getloadavg())};child=owned=raw=None;started=time.monotonic()
 with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with ownership.StageDeadline(cap,absolute_end=started+cap) as deadline,cleanup.SignalRequests() as requests:
   try:
    for k,v in ((resource.RLIMIT_AS,32*1024**3),(resource.RLIMIT_FSIZE,2*1024**3),(resource.RLIMIT_CORE,0)):resource.setrlimit(k,(v,v))
    with (capsule/'console.log').open('xb') as output:
     deadline.handoff=True
     try:child=subprocess.Popen(argv,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True);raw=cleanup.identity(child.pid);owned=cleanup.OwnedChild(child)
     finally:deadline.handoff=False
     dump(capsule/'started.json',{'pid':child.pid,'identity':owned.root,'argv':argv});print('START '+capsule.name+' pid='+str(child.pid),flush=True)
     record['returncode']=owned.wait(requests);record['cleanup']=ownership.cleanup_owned(owned,deadline.end)
    if record['returncode']:raise RuntimeError('Child returned '+str(record['returncode']))
    record['status']='child-complete'
   except BaseException as e:
    record.update(status='failed',error=repr(e))
    if owned is not None and 'cleanup' not in record:record['cleanup']=ownership.cleanup_owned(owned,deadline.end)
    elif child is not None:record['cleanup']=ownership.raw_cleanup(child,raw,cleanup,deadline.end)
    raise
   finally:
    record.update(elapsed_seconds=time.monotonic()-started,console_sha256=sha(capsule/'console.log'),shared_unchanged=shared()==before,end_load=list(os.getloadavg()))
    dump(capsule/'terminal.json',record);assert record['shared_unchanged']
 print('CLOSED '+capsule.name,flush=True);return record
def prepare():
 assert not MG.exists() and sha(PREFIX/'lib/libmadloop_tirust.so')==DSO
 MG.mkdir();archive=HERE/'mg-source.tar'
 subprocess.run(['git','archive','--format=tar','--output='+str(archive),REV,'madgraph','aloha','Template','bin','models','input','vendor','mg5decay','MadSpin'],check=True)
 with tarfile.open(archive) as f:f.extractall(MG,filter='data')
 for n in ('libcts.a','mpmodule.mod','compiler_version.log'):
  p=MG/'vendor/CutTools/includects'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'vendor/CutTools/includects'/n,p)
 shutil.rmtree(MG/'vendor/IREGI',ignore_errors=True)
 clone(DONOR,BASE);sys.path.insert(0,str(MG))
 from madgraph.iolibs.file_writers import FortranWriter
 template=MG/'madgraph/iolibs/template_files/loop_optimized/check_sa.inc'
 values=dict(nexternal=6,nincoming=2,proc_prefix='ML5_0_',print_so_born_results='',print_so_loop_results='',write_so_born_results='',write_so_loop_results='',set_coupling_target='')
 driver=BASE/'SubProcesses/P0_uux_zggg/check_sa.f';writer=FortranWriter(str(driver));writer.writelines(template.read_text()%values);writer.close()
 assert 'SYSTEM_CLOCK' not in driver.read_text() and 'NPSPOINTS' in driver.read_text()
 card=MG/'Template/loop_material/StandAlone/Cards/MadLoopParams.dat';shutil.copy2(card,BASE/'Cards/MadLoopParams.dat')
 # Fresh native learning; preserve HelConfigs/color resources, discard old diagnostic observations.
 removed=[]
 for p in list(BASE.rglob('*')):
  if p.is_file() and (p.suffix in ('.o','.mod') or p.name in ('libmodel.a','libdhelas.a','check','StabilityCheckDriver') or 'HelFilter' in p.name or 'LoopFilter' in p.name):
   if p.is_symlink() and not p.resolve().is_relative_to(BASE):continue
   assert p.resolve().is_relative_to(BASE);removed.append(p.relative_to(BASE).as_posix());p.unlink()
 for p in BASE.rglob('SavedRunInfos'):
  if p.is_dir():shutil.rmtree(p)
 # Retain the exact existing floating-point flags, with O3 the only added optimization.
 opts=BASE/'Source/make_opts';original=opts.read_text();assert original.count('FFLAGS= -w -fPIC\n')==1
 opts.write_text(original.replace('FFLAGS= -w -fPIC\n','FFLAGS= -w -fPIC -O3\n'))
 assert not any(p.is_file() and not p.is_symlink() and p.suffix in ('.o','.mod') for p in BASE.rglob('*'))
 dump(HERE/'prepared.json',{'schema':'zggg-actual-stock-profile-o3-v1','source_revision':REV,'source_archive_sha256':sha(archive),'driver_template_sha256':sha(template),'restored_driver_sha256':sha(driver),'restored_stock_card_sha256':sha(card),'removed_generated_artifacts':removed,'make_opts_before':original,'make_opts_after':opts.read_text(),'tirust_dso_sha256':DSO,'source_files':{p.relative_to(MG).as_posix():sha(p) for p in MG.rglob('*') if p.is_file() and not p.is_symlink()},'performance_acceptance':False})
 print('PREPARED canonical stock driver/card and complete generated-object clean',flush=True)
def build():
 caps=HERE/'build-run-2';env=environment(caps)
 script=HERE/'build.sh';script.write_text('#!/bin/bash\nset -euo pipefail\nmake -C \"'+str(BASE/'Source/MODEL')+'\" -j1 flavor_couplings.o\nmake -C "'+str(BASE/'Source')+'" -j2 all\nmake -C "'+str(BASE/'SubProcesses/P0_uux_zggg')+'" -j2 check\n')
 execute([shutil.which('bash',path=env['PATH']),str(script)],ROOT,caps,env,2700.)
 log=(caps/'console.log').read_text(errors='replace');commands=[line for line in log.splitlines() if FC in line and ' -c ' in line]
 assert commands and all(' -O3' in line and not any(flag in line for flag in ('-Ofast','-ffast-math','-funsafe-math','-O0','-O1','-O2')) for line in commands)
 sys.path.insert(0,str(ROOT/'tools'));from tirust_benchmark_compiler import capture_effective_flags
 flags=capture_effective_flags(BASE/'SubProcesses/P0_uux_zggg',env=env)
 objects={p.relative_to(BASE).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in BASE.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in ('.o','.a','.mod')}
 assert any(n.startswith('Source/MODEL/') and n.endswith('.o') for n in objects) and any(n.startswith('Source/DHELAS/') and n.endswith('.o') for n in objects)
 dump(HERE/'build.json',{'status':'complete-generated-Fortran-O3-rebuild','source_revision':REV,'objects':objects,'compiler_commands':commands,'effective_flags':flags,'build_log_sha256':sha(caps/'console.log'),'shared_reducer_libraries_rebuilt':False,'unsafe_math_flags':False,'source_model_process_optimization':'-O3','driver_sha256':sha(BASE/'SubProcesses/P0_uux_zggg/check_sa.f')})
 print('O3 BUILD COMPLETE compiler_commands='+str(len(commands))+' objects='+str(len(objects)),flush=True)
def profile(backend):
 assert backend in LABELS and read(HERE/'build.json')['status']=='complete-generated-Fortran-O3-rebuild'
 capsule=HERE/'runs'/str(backend);capsule.mkdir(parents=True,exist_ok=False)
 out=HERE/'outputs'/str(backend);out.mkdir(parents=True,exist_ok=False);export=clone(BASE,out/BASE.name)
 env=environment(capsule)
 lines=['set automatic_html_opening False --no_save','set auto_update 0 --no_save','set nb_core 2 --no_save','set loop_optimized_output True --no_save','set output_dependencies external --no_save','set complex_mass_scheme False --no_save',f'set fortran_compiler {FC} --no_save',f'set cpp_compiler {CPP} --no_save',f'set ninja {ROOT}/HEPTools/ninja/lib --no_save',f'set collier {ROOT}/HEPTools/collier --no_save',f'set ninjars {ROOT}/HEPTools/ninjars/lib --no_save',f'set tirust {PREFIX}/lib --no_save','import model loop_sm',f'check profile -reuse 1 u u~ > z g g g [virt=QCD] --reduction={backend} --loop_pole_mode=-1 --split_orders=-1 --energy=1000 --seed=20261112','quit']
 command=capsule/'commands.mg5';command.write_text('\n'.join(lines)+'\n')
 assert 12 in os.sched_getaffinity(0)
 cap=min(3600.,read(HERE/'profile-campaign-deadline.json')['end_monotonic']-time.monotonic());assert cap>90
 execute(['taskset','-c','12',sys.executable,'-B',str(MG/'bin/madgraph'),str(command)],out,capsule,env,cap)
 text=(capsule/'console.log').read_text(errors='replace')
 assert 'Error detected' not in text and 'Traceback (most recent call last)' not in text
 path=export/'SavedRunInfos/profile_timing_metadata.json';metadata=read(path)
 assert metadata['reduction_backends']==[backend]
 stages={r['name']:r for r in metadata['stages']}
 for stage in stages.values():
  e={k.lower():v for k,v in stage['effective_madloop_parameters'].items()}
  assert e['looppolemode']==-1 and e['ctmoderun']==1 and e['helicityfilterlevel']==2 and e['useloopfilter'] is True
  assert e['tirustcomputefinite'] and e['colliercomputeuvpoles'] and e['colliercomputeirpoles']
 selection=metadata['selected_timing_helicity'];assert selection['learned_filter_used'] is True
 result=metadata.get('profile_timing_result',metadata['timing_result']);logs=list(out.glob('profile_optimized_*.log'));assert len(logs)==1
 shutil.copy2(logs[0],capsule/'profile.log');shutil.copy2(path,capsule/'profile_timing_metadata.json')
 for p in sorted(export.rglob('*')):
  if p.is_file() and ('HelFilter' in p.name or 'LoopFilter' in p.name or p.name=='MadLoopParams.dat'):
   target=capsule/'observed-cards-filters'/p.relative_to(export);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 summary={'schema':'zggg-stock-profile-O3-result-v1','source_mg_revision':REV,'backend':backend,'backend_label':LABELS[backend],'process':'u u~ > z g g g [virt=QCD]','unpolarized_full_ms':1000*result['run_unpolarized_total'],'unpolarized_skip_ms':1000*result['run_unpolarized_coefs'],'unpolarized_full_minus_skip_estimate_ms':1000*(result['run_unpolarized_total']-result['run_unpolarized_coefs']),'initialization_seconds':result.get('Initialization'),'boot_seconds':result['Booting_time'],'stage_points':{n:r['actual_npoints'] for n,r in stages.items()},'stage_wall_seconds':{n:r['wall_seconds'] for n,r in stages.items()},'selected_timing_helicity':selection,'timing_result':result,'profile_log_sha256':sha(capsule/'profile.log'),'metadata_sha256':sha(path),'tirust_dso_sha256':DSO,'build_manifest_sha256':sha(HERE/'build.json'),'per_evaluation_statuses_available':False,'precision_scope':'Stock fixed-DP-interface timing CTModeRun/Init=1; whole-amplitude checked rescues disabled. TIRust may use DP-input local106/256/512 tensor tiers. Timed per-evaluation statuses/routes unavailable. Separate one-point stability phase is outside timing.','timing_clocks':{'unpolarized_full':'stock ProcessTimer monotonic waiter completion wall time minus boot, divided by stock stage points','other_stages':'stock Python time.time subprocess wall minus boot, divided by stock stage points'},'performance_acceptance':False,'profile_metadata_path':str(path)}
 dump(capsule/'summary.json',summary);print('RESULT backend='+str(backend)+' FULL_ms='+str(summary['unpolarized_full_ms']),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','build','profile'));p.add_argument('--backend',type=int);a=p.parse_args()
 if a.stage=='prepare':prepare()
 elif a.stage=='build':build()
 else:profile(a.backend)
