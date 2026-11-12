"""Source-only preparation for the final stock CLI profile; execute only after source GO."""
from pathlib import Path
import argparse,ast,hashlib,json,re,shutil,subprocess,tarfile
ROOT=Path.cwd();assert ROOT==Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE=Path(__file__).resolve().parent;MG=HERE/'mg'
parser=argparse.ArgumentParser();parser.add_argument('--revision',required=True);parser.add_argument('--tirust-prefix',type=Path,required=True);parser.add_argument('--delivery-receipt',type=Path,default=ROOT/'.tirust-build/tirust-native-tiers-final-1/delivery-receipt.json');args=parser.parse_args()
REV=args.revision;PREFIX=args.tirust_prefix.resolve();EXPECTED_DSO='d427667a0f2f55d0bd96778c1259ffcf52e3925341890b81c38f5f94e1eb7052'
assert re.fullmatch('[0-9a-f]{40}',REV)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
 with p.open('x') as stream:json.dump(v,stream,indent=2,sort_keys=True);stream.write('\n')
assert not MG.exists()
delivery=json.loads(args.delivery_receipt.read_text())
assert delivery['shared_library_sha256']==EXPECTED_DSO==sha(PREFIX/'lib/libmadloop_tirust.so')
assert delivery['dp_route_statistics'] is True and not delivery['owned_survivors']
assert delivery['release_tests_failed']==0 and delivery['live_abi_capability_and_error_preservation_passed'] is True
MG.mkdir();archive_path=HERE/'mg-source.tar'
subprocess.run(['git','archive','--format=tar','--output='+str(archive_path),REV,
                'madgraph','aloha','Template','bin','models','input','vendor','mg5decay','MadSpin'],check=True)
with tarfile.open(archive_path) as archive:archive.extractall(MG,filter='data')
for name,key in [('vendor/tirust/src/lib.rs','rust_bridge_source_sha256'),('vendor/tirust/mtirust.f90','fortran_module_source_sha256'),('vendor/tirust/tirust.h','c_header_sha256')]:
 assert sha(MG/name)==delivery[key],name
source=MG/'madgraph/various/process_checks.py'
assert source.read_bytes()==subprocess.check_output(['git','show',REV+':madgraph/various/process_checks.py'])
assert 'configured-unverified' in source.read_text() and 'polarized_contribution_verified' in source.read_text()
misc=(MG/'madgraph/various/misc.py').read_text();tree=ast.parse(misc)
timer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ProcessTimer')
assert 'perf_counter' in ast.get_source_segment(misc,timer) and '_completion' in ast.get_source_segment(misc,timer)
interface=(MG/'madgraph/interface/madevent_interface.py').read_text();tree=ast.parse(interface)
owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MadLoopInitializer')
method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='make_and_run')
assert 'ptimer.wait(.2)' in ast.get_source_segment(interface,method)
for name in ('libcts.a','mpmodule.mod','compiler_version.log'):
 dst=MG/'vendor/CutTools/includects'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'vendor/CutTools/includects'/name,dst)
shutil.rmtree(MG/'vendor/IREGI',ignore_errors=True)
commands={};fc='/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran';cpp='/nix/store/788mx070y81zjlg5ipcl0cra3afviw9k-gcc-wrapper-15.2.0/bin/c++'
for case,process in [('n1','u u~ > z [virt=QCD]'),('n2','u u~ > d d~ QED=0 [virt=QCD]')]:
 for backend in (8,9):
  lines=['set automatic_html_opening False --no_save','set auto_update 0 --no_save','set nb_core 2 --no_save','set loop_optimized_output True --no_save','set output_dependencies external --no_save','set complex_mass_scheme False --no_save',f'set fortran_compiler {fc} --no_save',f'set cpp_compiler {cpp} --no_save',f'set ninja {ROOT}/HEPTools/ninja/lib --no_save',f'set collier {ROOT}/HEPTools/collier --no_save',f'set ninjars {ROOT}/HEPTools/ninjars/lib --no_save',f'set tirust {PREFIX}/lib --no_save','import model loop_sm',f'check profile -reuse 10 {process} --reduction={backend} --loop_pole_mode=0 --tirust_qp_integrand=true --tirust_cache=-1 --seed=20261112','quit']
  path=HERE/f'{case}-{backend}.mg5';path.write_text('\n'.join(lines)+'\n');commands[path.name]={'sha256':sha(path),'process':process,'backend':backend}
files={p.relative_to(MG).as_posix():sha(p) for p in sorted(MG.rglob('*')) if p.is_file() and not p.is_symlink()}
config=json.loads((PREFIX/'lib/tirust-config.json').read_text());assert config.get('dp_route_statistics') is True
record={'schema':'madgraph-stock-profile-preparation-v2','status':'prepared-not-executed','base_mg_revision':REV,'source_archives':{'mg-source.tar':sha(archive_path)},'source_files':files,'process_checks_sha256':sha(source),'timer_source_sha256':{'misc.py':sha(MG/'madgraph/various/misc.py'),'madevent_interface.py':sha(MG/'madgraph/interface/madevent_interface.py')},'commands':commands,'tirust_prefix':str(PREFIX),'tirust_dso_sha256':EXPECTED_DSO,'delivery_receipt_path':str(args.delivery_receipt.resolve()),'delivery_receipt_sha256':sha(args.delivery_receipt),'installed_configuration':config,'native_executions':0,'stability_points':10,'performance_acceptance':False,'timing_policy':'stock DP-interface check profile; all-helicity FULL headline; FULL-SKIP estimate; internal local tensor tiers possible; timed per-point statuses/routes unavailable','timing_clocks':{'unpolarized_full':'ProcessTimer monotonic waiter completion, RSS sampling does not define end time','other_stages':'Python time.time subprocess wall durations'},'route_statistics':{'installed_capability':True,'stock_driver_emits_counts':False,'timed_counts':None,'missing_counts_are_not_zero':True},'initialization_policy':'stock behavior: first export fresh bootstrap, cloned second export reuses it; null Initialization is not zero','polarized_policy':'configured selection has unverified contribution, diagnostic only'}
dump(HERE/'prepared.json',record);shutil.copyfile(args.delivery_receipt,HERE/'delivery-receipt.json')
print('PREPARED final source/commands only; no CLI, compiler, SDK or numerical execution',flush=True)
