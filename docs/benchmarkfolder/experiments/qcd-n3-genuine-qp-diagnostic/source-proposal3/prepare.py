"""Read-only donor binding and private text fixture preparation; no subprocesses.
Run only before source review/seal. The output namespace must be new.
"""
from pathlib import Path
import hashlib
import json
import re
import shlex
import shutil
import shapes

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
DONOR=ROOT/'.benchmark-work/tirust/campaigns/invariant-symmetry-1/outputs/candidate/qcd_n3_uux_zgg'
PROCESS=DONOR/'SubProcesses/P0_uux_zgg'
META=ROOT/'docs/benchmarkfolder/campaigns/invariant-symmetry-1/experiments/numerical/full100-warmup-qcd_n3_uux_zgg-1/candidate-9/candidate-inputs.json'
SPEC=ROOT/'docs/benchmarkfolder/experiments/invariant-symmetry-qcd-n3-target-analysis/qp-specification1'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj): path.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
def check(path,expected):
    if sha(path)!=expected: raise ValueError('Frozen donor mismatch: '+str(path))

def prepare():
    check(META,'2cfa41878014841fa3a87b75a70a9a82881504477fbaf12d869e6c349e014b28')
    check(SPEC/'index.json','741813a73a19891091f5597d8032033e2e1723866c45b10f7f8ebacbe6c1c9a9')
    metadata=json.loads(META.read_text()); selected=json.loads((SPEC/'selected-inputs.json').read_text())
    for name,item in json.loads((SPEC/'index.json').read_text())['files'].items():
        check(SPEC/name,item['sha256'])
    dependencies={str(META):sha(META),str(SPEC/'index.json'):sha(SPEC/'index.json')}
    # Compare every closed generated-source byte before any donor object reuse.
    for name,value in metadata['compilation']['sources'].items():
        path=DONOR/name;check(path,value);dependencies[str(path)]=value
    for category in ('archives','libraries','metadata'):
        for name,value in metadata['runtime'][category].items():
            check(Path(name),value); dependencies[name]=value
    # Objects were not individually hashed by the historical build record. Bind
    # current donor bytes now, without mislabelling them as historical object proof.
    patterns=['MadLoopParamReader.f','MadLoopCommons.f','polynomial.f','loop_matrix.f','improve_ps.f',
      'born_matrix.f','CT_interface.f','loop_num.f','helas_calls*.f','jamp?_calls_*.f',
      'mp_born_amps_and_wfs.f','mp_compute_loop_coefs.f','mp_helas_calls*.f','coef_construction_*.f',
      'loop_CT_calls_*.f','mp_coef_construction_*.f','TIR_interface.f','GOLEM_interface.f',
      'COLLIER_interface.f','TIRUST_interface.f','compute_color_flows.f','mg5_citation.f']
    objects=[str(p.with_suffix('.o')) for pattern in patterns for p in sorted(PROCESS.glob(pattern))]
    assert len(objects)==len(set(objects)) and len(objects)==20
    for name in objects: dependencies[name]=sha(Path(name))
    check(PROCESS/'check',metadata['compilation']['executable_sha256'])
    dependencies[str(PROCESS/'check')]=metadata['compilation']['executable_sha256']
    fc=shlex.split(metadata['compilation']['effective_compiler_flags']['scopes']['process']['expanded_make_output'].splitlines()[0])[0]
    programs={k:str(Path(shutil.which(v)).resolve()) for k,v in [('cc','gcc'),('taskset','taskset'),('prlimit','prlimit'),('ldd','ldd')]}
    programs['fc']=fc
    for value in programs.values(): dependencies[value]=sha(Path(value))
    extra=['HEPTools/ninja/Ninja/src/mninja.F90','HEPTools/ninjars/include/mninjars.f90',
           '.tirust-build/qp-transport-install/mg/HEPTools/tirust/include/mtirust.f90',
           'vendor/CutTools/src/cts/cts_cutroutines.f90','vendor/CutTools/includects/cts_mpc.h',
           'vendor/CutTools/includects/cts_mprec.h']
    for name in extra: dependencies[str(ROOT/name)]=sha(ROOT/name)
    shape=shapes.verify(*[(PROCESS/name).read_text() for name in ('coupl.inc','mp_coupl.inc','loop_matrix.f','mp_compute_loop_coefs.f','mp_helas_calls_ampb_1.f','loop_max_coefs.inc')],(HERE/'observer.f90').read_text())
    fixtures=HERE/'fixtures';fixtures.mkdir(exist_ok=False)
    points_path=ROOT/'docs/benchmarkfolder/campaigns/invariant-symmetry-1/points/qcd_n3_uux_zgg.dat'
    check(points_path,'c1dc86e4e0cdd8a049d7a9a87b98831af74f3b6b79fc63fd0bbfe16a7fe55450')
    lines=points_path.read_bytes().splitlines(keepends=True)
    (fixtures/'points.dat').write_bytes(b'3\n'+b''.join(b''.join(lines[1+5*(p-1):1+5*p]) for p in (1,93,39)))
    shutil.copyfile(SPEC/'selected-inputs.json',fixtures/'selected-inputs.json')
    param=Path(metadata['resolved_parameter_card']);check(param,metadata['param_card_sha256'])
    (fixtures/'nominal').mkdir();shutil.copyfile(param,fixtures/'nominal/param_card.dat')
    ident=DONOR/'Cards/ident_card.dat'
    if not ident.exists(): ident=PROCESS/'ident_card.dat'
    shutil.copyfile(ident,fixtures/'nominal/ident_card.dat');dependencies[str(ident)]=sha(ident)
    card=DONOR/'Cards/MadLoopParams.dat';check(card,metadata['numerical_cards']['Cards/MadLoopParams.dat'])
    (fixtures/'MadLoopParams.dat').write_bytes(card.read_bytes())
    for suffix in ('ColorDenomFactors','ColorNumFactors','HelConfigs'):
        source=DONOR/f'SubProcesses/MadLoop5_resources/ML5_0_{suffix}.dat'
        shutil.copyfile(source,fixtures/source.name); dependencies[str(source)]=sha(source)
    definitions=(DONOR/'SubProcesses/MadLoop_makefile_definitions').read_text()
    linker=next(x.split('=',1)[1].strip() for x in definitions.splitlines() if x.startswith('LINK_LOOP_LIBS ='))
    linker=shlex.split(linker.replace('$(LIBDIR)',str(DONOR/'lib')+'/'))
    supervisor=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py'
    check(supervisor,'ed25a9fc96e2faf2b545a9bf70a91b34315c01e534a6ebe68d7520ec2171aa42')
    old=supervisor.read_text();new=old.replace("['11']","['12']").replace('CPU11','CPU12')
    assert old.count("['11']")==1 and old.count('CPU11')==1
    (HERE/'supervise.py').write_text(new);dependencies[str(supervisor)]=sha(supervisor)
    dump(HERE/'bindings.json',{'schema':'closed-qcd-n3-source-proposal-v1','donor':str(DONOR),
      'process':str(PROCESS),'dependencies':dependencies,'objects':objects,'programs':programs,
      'runtime':metadata['runtime']['libraries'],'link_loop_libs':linker,
      'physical_card_sha256':metadata['param_card_sha256'],
      'historical_object_identity_claimed':False,'historical_final_executable_rehashed':True,
      'scope':'Only closed QCDn3 sources/objects and bound reducers; no active output reads.',
      'shapes':shape,
      'commands_executed':[]})
    print('Prepared source fixtures/bindings only; no compiler/loader/native subprocess.')

if __name__=='__main__': prepare()
