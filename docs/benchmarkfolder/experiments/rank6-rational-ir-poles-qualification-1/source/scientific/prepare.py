"""Pure file binding only. No compiler, loader, executable or subprocess call."""
import hashlib,json
from pathlib import Path
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
OLD=ROOT/'.tirust-build/generated-rank6-poles-1/common-input1'
WORK=ROOT/'.tirust-build/rank6-ir-triangle-work-1'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    build=json.loads((OLD/'build1/build.json').read_text());run=json.loads((OLD/'run1/execution.json').read_text())
    assert build['complete'] and run['complete'] and run['exit_code']==0
    bulk=json.loads((ROOT/'.tirust-build/fortran-qp-output-bulk-unit-controller-2/input-pins.json').read_text())
    tools={k:bulk['tools'][k] for k in ('gfortran','nm','python','taskset')}
    old_driver=str(OLD/'build1/driver.o');old_source=str(ROOT/'docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/common-input-proposal/driver.f')
    commands=[list(r['argv']) for r in build['commands']]
    assert len(commands)==2 and commands[0][0]==commands[1][0]==tools['gfortran']
    replace={old_source:str(HERE/'driver.f'),old_driver:str(WORK/'build/driver.o'),'-J'+str(OLD/'build1'):'-J'+str(WORK/'build'),str(OLD/'build1/rank6-cms'):str(WORK/'build/check'),'-Wl,-Map,'+str(OLD/'build1/link.map'):'-Wl,-Map,'+str(WORK/'build/link.map')}
    commands=[[replace.get(a,a) for a in cmd] for cmd in commands]
    assert old_source not in commands[0] and old_driver not in commands[1]
    files=dict(build['actual_link_inputs'])
    for path,pin in files.items():assert digest(path)==pin,path
    for path,pin in run['runtime_sha256'].items():assert digest(path)==pin,path
    files.update(run['runtime_sha256'])
    ancestor=ROOT/'docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1'
    original=json.loads((ancestor/'build1/build.json').read_text())
    materialized=json.loads((ancestor/'materialized-index.json').read_text())
    assert build['parent_build_sha256']==digest(ancestor/'build1/build.json')
    assert original['complete'] and original['sources_and_dependencies_unchanged']
    for name in ('cap_tirust.o','cap_helpers.o','cap_polynomial.o','observer.o'):
        assert original['artifacts']['ew/'+name]['sha256']==digest(ROOT/'.tirust-build/generated-rank6-poles-1/ew'/name)
    for name in ('cap_tirust.f','cap_helpers.f','cap_polynomial.f','monomials.json'):
        assert materialized['files']['materialized/ew/'+name]==digest(ancestor/'materialized/ew'/name)
    extra=[ancestor/'build1/build.json',ancestor/'build1/archive.json',ancestor/'materialized-index.json',
     ROOT/'.tirust-build/rank6-rational-pole-oracle-proposal-1/source-index.json',
     ROOT/'.tirust-build/rank6-rational-pole-oracle-proposal-1/oracle.py',
     ROOT/'.tirust-build/rank6-rational-pole-oracle-proposal-1/future-triangle-fixture.json',OLD/'build1/build.json',OLD/'run1/execution.json',ROOT/'docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/observer.c',
     ROOT/'docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/materialized/ew/cap_tirust.f',
     ROOT/'docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/materialized/ew/cap_helpers.f',
     ROOT/'docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/materialized/ew/cap_polynomial.f',
     ROOT/'docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/materialized/ew/monomials.json',
     ROOT/'docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/analyze.py',
     ROOT/'.tirust-build/fortran-qp-output-bulk-unit-controller-2/controller.py',
     ROOT/'docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py',
     ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py',
     ROOT/'docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-3/loader_checks.py']
    support=ROOT/'docs/benchmarkfolder/experiments/current-cms-replay-supervisor-1/ownership'
    extra += [support/n for n in ('linker_bind.py','run.py','legacy_qualify.py')]
    include_dirs=[Path(x[2:]) for x in commands[0] if x.startswith('-I')]
    for folder in include_dirs:
        assert folder.is_dir(),folder
        extra += [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.inc','.mod')]
    runtime_inputs={'schedule.dat':str(HERE/'schedule.dat'),'nominal/param_card.dat':str(ROOT/'.tirust-build/generated-rank6-poles-1/ew/nominal/param_card.dat'),'nominal/ident_card.dat':str(ROOT/'.tirust-build/generated-rank6-poles-1/ew/nominal/ident_card.dat')}
    extra += [Path(v) for v in runtime_inputs.values()]+[Path(v) for v in tools.values()]
    for p in extra:files[str(p)]=digest(p)
    selected=str(ROOT/'.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib/libmadloop_tirust.so')
    assert files[selected]=='eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9'
    result={'schema':'rank6-IR-tensor-diagnostic-input-pins-v1','files':dict(sorted(files.items())),
      'tools':tools,'compiler_environment':build['environment'],'runtime_environment':run['environment_overrides'],
      'build_commands':commands,'old_link_inputs':build['actual_link_inputs'],'old_driver_object':old_driver,
      'runtime_inputs':runtime_inputs,'expected_runtime':run['runtime_sha256'],'selected_library':selected,
      'historical_source_scope':'CAP6 synthetic-capacity extension and exact old generated helper/mapping; no current full-amplitude qualification',
      'native_calls':0,'compiler_calls':0}
    path=HERE/'input-pins.json'
    with path.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'files':len(files),'pins_sha256':digest(path),'native_calls':0,'compiler_calls':0}))
if __name__=='__main__':main()
