"""Prepare stock CLI commands and isolated source only; never execute MG or a reducer."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
ROOT=Path.cwd();assert ROOT==Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE=Path(__file__).resolve().parent;MG=HERE/'mg'
REV='cafb35d7987436661c3346e39c724d329846a1f2'
PROPOSAL=ROOT/'.tirust-build/madgraph-profile-integration-proposal-1'
PREFIX=ROOT/'.tirust-build/tirust-oem-signed-prefix-1/HEPTools/tirust'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
assert not MG.exists();MG.mkdir()
archive_path=HERE/'mg-source.tar'
subprocess.run(['git','archive','--format=tar','--output='+str(archive_path),REV,
                'madgraph','aloha','Template','bin','models','input','vendor','mg5decay','MadSpin'],check=True)
archives={'mg-source.tar':sha(archive_path)}
with tarfile.open(archive_path) as archive:archive.extractall(MG,filter='data')
source=MG/'madgraph/various/process_checks.py'
expected=subprocess.check_output(['git','show',REV+':madgraph/various/process_checks.py'])
assert source.read_bytes()==expected
index=json.loads((PROPOSAL/'source-index.json').read_text())
assert sha(source)==index['candidate_sources']['madgraph/various/process_checks.py']
for name in ('libcts.a','mpmodule.mod','compiler_version.log'):
 dst=MG/'vendor/CutTools/includects'/name;dst.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(ROOT/'vendor/CutTools/includects'/name,dst)
shutil.rmtree(MG/'vendor/IREGI',ignore_errors=True)
commands={}
fc='/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran'
cpp='/nix/store/788mx070y81zjlg5ipcl0cra3afviw9k-gcc-wrapper-15.2.0/bin/c++'
for case,process in [('n1','u u~ > z [virt=QCD]'),('n2','u u~ > d d~ QED=0 [virt=QCD]')]:
 for backend in (8,9):
  lines=['set automatic_html_opening False --no_save','set auto_update 0 --no_save','set nb_core 2 --no_save','set loop_optimized_output True --no_save','set output_dependencies external --no_save','set complex_mass_scheme False --no_save',f'set fortran_compiler {fc} --no_save',f'set cpp_compiler {cpp} --no_save',f'set ninja {ROOT}/HEPTools/ninja/lib --no_save',f'set collier {ROOT}/HEPTools/collier --no_save',f'set ninjars {ROOT}/HEPTools/ninjars/lib --no_save',f'set tirust {PREFIX}/lib --no_save','import model loop_sm',f'check profile -reuse 10 {process} --reduction={backend} --loop_pole_mode=0 --tirust_qp_integrand=true --tirust_cache=-1 --seed=20261112','quit']
  path=HERE/f'{case}-{backend}.mg5';path.write_text('\n'.join(lines)+'\n');commands[path.name]={'sha256':sha(path),'process':process,'backend':backend}
files={p.relative_to(MG).as_posix():sha(p) for p in sorted(MG.rglob('*')) if p.is_file() and not p.is_symlink()}
dump(HERE/'prepared.json',{'schema':'madgraph-stock-profile-preparation-v1','status':'prepared-not-executed','base_mg_revision':REV,'source_archives':archives,'source_patch_sha256':None,'profile_fix_commit':REV,'profile_fix_proposal_sha256':index['patch_sha256'],'process_checks_sha256':sha(source),'source_files':files,'commands':commands,'tirust_prefix':str(PREFIX),'tirust_dso_sha256':sha(PREFIX/'lib/libmadloop_tirust.so'),'native_executions':0,'timing_policy':'stock check profile; separate explicit backend8/backend9; LoopPoleMode0 fullUVIR; stock CTModeRun1 fixedDP; FULL-SKIP estimate only; timing pointcounts chosen by stock timer; no per-evaluation statuses','stability_points':10,'performance_acceptance':False})
print('PREPARED source and commands only; no CLI, compiler, SDK or reduction executed',flush=True)
