#!/usr/bin/env python3
"""Prepare one additive link and two current-runtime original-QP input sets."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

if sys.flags.optimize:raise RuntimeError('Source validation requires assertions')
ROOT=Path.cwd().resolve()
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
HISTORY=ROOT/'docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design'
OLD=ROOT/'.tirust-build/ew-qp-mass-prototype/observer-native-design/attempt2/original/SubProcesses/P0_epem_ttxa'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert ROOT.name=='madgraph7_madloop_tirust'
    assert not (HERE/'child-prepared.json').exists()
    parent=json.loads((PARENT/'prepared.json').read_text())
    built=json.loads((PARENT/'build.json').read_text())
    child=json.loads((PARENT/'amplitude/prepared.json').read_text())
    assert built['state']=='complete' and built['manifest_sha256']==sha(PARENT/'prepared.json')
    inputs={};system={}
    def bind(path,expected=None):
        path=Path(path);digest=sha(path)
        if expected is not None:assert digest==expected,('Changed source input',str(path))
        (inputs if path.is_relative_to(ROOT) else system)[str(path)]=digest
        return path
    for name in ('prepared.json','build.json','evidence-index.json','amplitude/prepared.json','amplitude/analyze.py'):
        bind(PARENT/name)
    for name in ('analyze.py','attempt2/execution-1/evidence-index.json','attempt2/execution-1/run.json'):
        bind(HISTORY/name)
    frozen_inputs={str(ROOT/name):digest for name,digest in parent['inputs'].items()}
    frozen_inputs.update(parent['system_files'])
    previous=[job for job in parent['build_jobs'] if job['name']=='link-guarded-qp']
    assert len(previous)==1
    job=copy.deepcopy(previous[0]);argv=job['argv'];job['name']='link-current-original-qp'
    build=HERE/'build';build.mkdir(exist_ok=False)
    exe=build/'current-original-qp'
    argv[argv.index('-o')+1]=str(exe)
    removed=[str(PARENT/'amplitude/build/helpers.o'),str(PARENT/'amplitude/build/adapters.o')]
    for item in removed:assert argv.count(item)==1;argv.remove(item)
    guarded_caller=str(PARENT/'amplitude/build/loop_CT_calls_1.o')
    assert argv.count(guarded_caller)==1
    argv[argv.index(guarded_caller)]=str(OLD/'loop_CT_calls_1.o')
    objects=[p for p in argv if p.endswith('.o')]
    assert len(objects)==24 and all(Path(p).parent==OLD for p in objects)
    assert objects[0]==str(OLD/'check_sa.o')
    for arg in argv:
        if arg.startswith('/') and Path(arg).is_file():
            expected=frozen_inputs.get(arg) or child['readonly_sha256'].get(arg)
            assert expected is not None,('Unbound link input',arg)
            bind(arg,expected)
    old_prepared=json.loads((HISTORY/'attempt2/prepared.json').read_text())
    bind(OLD/'loop_CT_calls_1.o',old_prepared['artifacts'][str((OLD/'loop_CT_calls_1.o').relative_to(ROOT))])
    proof={'inherited_link_job':'link-guarded-qp','removed_objects':removed,
           'restored_object':str(OLD/'loop_CT_calls_1.o'),'reused_original_object_count':24,
           'original_driver_object':str(OLD/'check_sa.o'),'fortran_compiles':0,
           'original_link_recipe':previous[0]['argv']}
    staged=HERE/'runs';staged.mkdir(exist_ok=False)
    runs=[];checks=[];counterparts={};oracles={}
    for backend in (8,9):
        name=f'current-original-qp-{backend}'
        source=next(r for r in child['runs'] if r['name']==f'guarded-qp-{backend}')
        aggregate=next(j for stage in parent['stages'] for j in stage['jobs'] if j['name']==source['name'])
        directory=staged/name;directory.mkdir()
        source_dir=Path(source['cwd'])
        for item,digest in source['input_sha256'].items():
            sourcepath=bind(item,digest);relative=sourcepath.relative_to(source_dir)
            target=directory/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(sourcepath,target)
        run=copy.deepcopy(aggregate)
        run.update(name=name,arm='original',backend=backend,mode=4,point_count=2,
                   argv=[str(exe),str(directory/'selected.dat')],cwd=str(directory),
                   log_file=str(directory/'stdout.log'),timeout_seconds=120,
                   input_sha256={str(p):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()})
        run['expected_outputs']=['mass-observer.hex','observer-run.log','param.log','stdout.log']
        run['scientific_outputs']=['mass-observer.hex','observer-run.log','param.log']
        run['runtime_mutable_inputs']=[]
        runs.append(run)
        for path,digest in run['input_sha256'].items():bind(path,digest)
        runtime=copy.deepcopy(next(check for check in parent['runtime_checks'] if check['job']['name']==f'loader-guarded-qp-{backend}'))
        runtime['executable']=str(exe)
        runtime['job'].update(name=f'loader-current-original-qp-{backend}',argv=[runtime['loader'],'--list',str(exe)],
                              cwd=str(directory),log_file=str(directory/'loader-list.txt'),timeout_seconds=5)
        checks.append(runtime)
        for entry in runtime['expected'].values():bind(entry['path'],entry['sha256'])
        assert runtime['expected']['libmadloop_tirust.so']['sha256']=='eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9'
        counterparts[str(backend)]={'run':aggregate,'journal':str(source_dir/'mass-observer.hex'),
            'input_sha256':source['input_sha256'],'executable':aggregate['argv'][0],
            'executable_sha256':built['outputs'][str(Path(aggregate['argv'][0]).relative_to(ROOT))]}
        bind(aggregate['argv'][0],counterparts[str(backend)]['executable_sha256'])
        oracle=bind(HISTORY/f'attempt2/execution-1/raw/original-{backend}/mass-observer.hex')
        candidate=bind(HISTORY/f'attempt2/execution-1/raw/candidate-{backend}/mass-observer.hex')
        oracles[str(backend)]={'original':{'path':str(oracle),'sha256':sha(oracle)},
                              'candidate':{'path':str(candidate),'sha256':sha(candidate)},
                              'runtime_identity':'historical49af; secondary reference, not same-current-runtime control'}
    result={'schema':'additive-current-original-qp-source-v1','state':'source_prepared_not_linked_or_executed',
        'build_jobs':[job],'build_outputs':[str(exe)],'runs':runs,'runtime_checks':checks,
        'runtime_library_directories':parent['runtime_library_directories'],
        'compiler':parent['compiler'],'compiler_environment':parent['compiler_environment'],
        'python':parent['python'],'taskset':parent['taskset'],'inputs':inputs,'system_files':system,
        'counterparts':counterparts,'historical_oracles':oracles,'link_delta_proof':proof,
        'frozen_parser':{'path':str(HISTORY/'analyze.py'),'sha256':sha(HISTORY/'analyze.py')},
        'frozen_support':{'path':str(PARENT/'amplitude/analyze.py'),'sha256':sha(PARENT/'amplitude/analyze.py')},
        'parent_prepared_sha256':sha(PARENT/'prepared.json'),'parent_build_sha256':sha(PARENT/'build.json'),
        'required_prior_run':{'path':str(PARENT/'run.json'),'state':'complete','guarded_qp_backends':[8,9],
                              'must_match_parent_prepared_sha256':sha(PARENT/'prepared.json')},
        'caps':{'link_seconds':30,'loader_seconds_total':10,'process_seconds':120,'processes':2,'cpu':8,'jobs':1},
        'scope':'Same-current-runtime mass-transport comparison only. No broader graph/Born, accuracy, host-stability or performance qualification.'}
    with (HERE/'child-prepared.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({'child_prepared_sha256':sha(HERE/'child-prepared.json'),'link_jobs':1,'fortran_compiles':0,'native_executions':0}))


if __name__=='__main__':main()
