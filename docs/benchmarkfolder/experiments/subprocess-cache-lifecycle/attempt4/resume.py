#!/usr/bin/env python3
"""Fresh private drivers with complete model-card inputs; no numerical edits."""
import argparse
import difflib
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import shutil
import time

ROOT=Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
BASE=ROOT/'docs/benchmarkfolder/experiments/subprocess-cache-lifecycle'
PRIOR=BASE/'attempt3'
REPORT=BASE/'attempt4'
WORK=ROOT/'.tirust-build/subprocess-cache-lifecycle/attempt4'
SUB=ROOT/'.tirust-build/subprocess-cache-lifecycle/export/SubProcesses'
ORIGINAL=ROOT/'.tirust-build/subprocess-cache-lifecycle/export'
INDEX=REPORT/'source-freeze.json'
RUNS=[('plain-cache0','plain',0),('plain-cache-all','plain',-1),
      ('observed-cache0','observed',0),('observed-cache-all','observed',-1)]
GUARD=None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True)
        stream.write('\n')


def qualifier():
    if digest(BASE/'source-freeze.json')!='2f971799fa749a7591736967da1e119655670b21fd304b01e452d601bb131c50':
        raise RuntimeError('Original protocol freeze changed')
    original=json.loads((BASE/'source-freeze.json').read_text())
    if digest(BASE/'qualify.py')!=original['protocol']['qualify.py']:
        raise RuntimeError('Original qualifier changed')
    spec=importlib.util.spec_from_file_location('original_qualifier',BASE/'qualify.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.frozen()
    module.REPORT=REPORT
    if GUARD is not None: GUARD.bind(module)
    return module


def analyzer():
    spec=importlib.util.spec_from_file_location('frozen_analyzer',BASE/'analyze.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze():
    q=qualifier()
    q.verify_paths(json.loads((BASE/'generated-source-freeze.json').read_text())['source_inventory'],ORIGINAL)
    baseline=json.loads((PRIOR/'build.json').read_text())
    q.verify_hashes(baseline['numerical_objects'])
    q.verify_hashes(baseline['link_inputs_before_after'])
    WORK.mkdir(exist_ok=False)
    roles=json.loads((BASE/'generation.json').read_text())['roles']
    rendered={}
    for template,name in [('driver.f.in','lifecycle_driver_attempt4.f'),
                          ('card_smoke.f.in','lifecycle_card_smoke_attempt4.f')]:
        text=(REPORT/template).read_text()
        for key,value in {'PREFIX_A':roles['101']['prefix'],'PREFIX_B':roles['202']['prefix'],
                          'CAPACITY':str(roles['101']['capacity'])}.items():
            text=text.replace('@'+key+'@',value)
        if re.search(r'@[A-Z_]+@',text):
            raise RuntimeError('Unexpanded driver placeholder')
        with (SUB/name).open('x') as stream: stream.write(text)
        with (REPORT/name).open('x') as stream: stream.write(text)
        rendered[str(SUB/name)]=digest(SUB/name)
    previous=(BASE/'driver.f.in').read_text()
    expected=previous.replace("SETPARA('param_card.dat')","SETPARA('nominal/param_card.dat')")
    expected=expected.replace("SETPARA('param_alternate.dat')","SETPARA('alternate/param_card.dat')")
    if (REPORT/'driver.f.in').read_text()!=expected:
        raise RuntimeError('New physical driver contains more than the declared path substitutions')
    patch=''.join(difflib.unified_diff(previous.splitlines(True),expected.splitlines(True),
        fromfile='../driver.f.in',tofile='driver.f.in'))
    with (REPORT/'driver-paths.patch').open('x') as stream: stream.write(patch)
    prior_files=[BASE/'source-freeze.json',BASE/'generated-source-freeze.json',
        BASE/'generation.json',BASE/'cli-bootstrap-provenance.json',
        PRIOR/'build.json',PRIOR/'source-freeze.json',PRIOR/'resume-freeze.json',
        PRIOR/'native-failure.json',PRIOR/'execution/native-plain-cache0.txt',
        PRIOR/'execution/native-plain-cache0.json',PRIOR/'runs/plain-cache0/inputs.json',
        PRIOR/'runs/plain-cache0/param_card.dat',PRIOR/'runs/plain-cache0/param_alternate.dat',
        PRIOR/'runs/plain-cache0/MadLoopParams.dat',ORIGINAL/'Cards/ident_card.dat',
        BASE/'attempt2/execution/link-plain.txt',PRIOR/'execution/link-plain.txt',
        PRIOR/'execution/link-observed.txt',BASE/'attempt2/execution/link-plain.json',
        PRIOR/'execution/link-plain.json',PRIOR/'execution/link-observed.json']
    save(INDEX,{'schema_version':1,'kind':'private card-path/input completion only',
        'sources':{name:digest(REPORT/name) for name in ('resume.py','driver.f.in','card_smoke.f.in',
            'driver-paths.patch','AMENDMENT.md','process_guard.py','test_smoke.py','synthetic-smoke-tests.txt',
            'test_process_guard.py','synthetic-process-guard-tests.txt')},
        'rendered_sources':rendered,'prior_inputs':{str(path):digest(path) for path in prior_files},
        'unchanged_numerical_objects':{path:value for path,value in baseline['numerical_objects'].items()
                                       if Path(path).name!='lifecycle_driver.o'},
        'prior_link_inputs':baseline['link_inputs_before_after'],
        'models':'Exact nominal/alternate bytes from the stopped attempt3 worker; each directory gets the original ident_card.dat.',
        'declared_outputs':['records.txt','trace.txt','param.log'],
        'native_execution_authorized':False,'cpu':10,'build_cap_seconds':600,
        'card_smoke_cap_seconds':30,'native_worker_cap_seconds':180})


def verified():
    state=json.loads(INDEX.read_text())
    for name,value in state['sources'].items():
        if digest(REPORT/name)!=value: raise RuntimeError('Attempt4 source changed')
    q=qualifier()
    for key in ('rendered_sources','prior_inputs','unchanged_numerical_objects','prior_link_inputs'):
        q.verify_hashes(state[key])
    q.verify_paths(json.loads((BASE/'generated-source-freeze.json').read_text())['source_inventory'],ORIGINAL)
    q.verify_paths(json.loads((BASE/'materialized-source.json').read_text())['files'],q.MG)
    return state,q


def argv_from_log(path,needle):
    lines=[line for line in path.read_text().splitlines() if needle in line]
    if len(lines)!=1: raise RuntimeError('Actual compiler recipe is not unique: '+needle)
    return shlex.split(lines[0])


def build():
    state,q=verified()
    baseline=json.loads((PRIOR/'build.json').read_text())
    for compiler in baseline['compilers'].values():
        if digest(compiler['path'])!=compiler['sha256']: raise RuntimeError('Compiler changed')
    deadline=time.monotonic()+600
    for reference in (BASE/'attempt2/execution/link-plain.json',
                      PRIOR/'execution/link-plain.json',PRIOR/'execution/link-observed.json'):
        expected=json.loads(reference.read_text())['compiler_environment']
        if {key:q.env().get(key) for key in expected}!=expected:
            raise RuntimeError('Compiler environment differs from retained actual recipe: '+str(reference))
    def command(argv,name):
        return q.command(argv,name,SUB,q.remaining(deadline,600))
    compile_base=argv_from_log(BASE/'attempt2/execution/link-plain.txt',
        ' -c lifecycle_driver.f -o lifecycle_driver.o ')
    objects={}
    for label,stem in [('driver','lifecycle_driver_attempt4'),('card-smoke','lifecycle_card_smoke_attempt4')]:
        argv=[stem+'.f' if item=='lifecycle_driver.f' else stem+'.o' if item=='lifecycle_driver.o'
              else item for item in compile_base]
        command(argv,'compile-'+label)
        objects[label]={'path':str(SUB/(stem+'.o')),'sha256':digest(SUB/(stem+'.o'))}
        q.verify_hashes(state['unchanged_numerical_objects'])
        q.verify_hashes(state['prior_link_inputs'])
    bound=dict(state['prior_link_inputs'])
    bound.update({item['path']:item['sha256'] for item in objects.values()})
    executables={}
    for variant in ('plain','observed','card-smoke'):
        original_variant='observed' if variant=='card-smoke' else variant
        argv=argv_from_log(PRIOR/f'execution/link-{original_variant}.txt',
            f' -o lifecycle_{original_variant} lifecycle_driver.o ')
        executable=WORK/('lifecycle_'+variant)
        map_path=WORK/('lifecycle_'+variant+'.map')
        argv=[str(executable) if item=='lifecycle_'+original_variant else
              objects['card-smoke' if variant=='card-smoke' else 'driver']['path'] if item=='lifecycle_driver.o' else
              '-Wl,-Map,'+str(map_path) if item=='-Wl,-Map,lifecycle_'+original_variant+'.map' else item
              for item in argv]
        q.verify_hashes(bound)
        command(argv,'link-'+variant)
        q.verify_hashes(bound)
        loads={}
        for loaded in re.findall(r'^LOAD (.+)$',map_path.read_text(),re.M):
            path=Path(loaded)
            if not path.is_absolute(): path=SUB/path
            name=str(path.resolve())
            if name not in bound or digest(path)!=bound[name]:
                raise RuntimeError('Actual LOAD input absent from before-link bindings: '+loaded)
            loads[name]=bound[name]
        actual_runtime=q.runtime(executable,deadline)
        if actual_runtime['files']!=baseline['executables'][original_variant]['runtime']['files']:
            raise RuntimeError('Runtime library set changed')
        executables[variant]={'path':str(executable),'sha256':digest(executable),'runtime':actual_runtime,
                             'resolved_link_inputs':loads,'map_path':str(map_path),'map_sha256':digest(map_path)}
    verified()
    q.remaining(deadline)
    save(REPORT/'build.json',{'sources_freeze_sha256':digest(INDEX),'objects':objects,
        'unchanged_numerical_objects':state['unchanged_numerical_objects'],
        'all_link_inputs_before_after':bound,'executables':executables,
        'compilers':baseline['compilers'],'scope':'Only private driver and card smoke were compiled; all33 generated/common numerical objects retained unchanged.'})


def input_layout(directory,cache,q):
    shutil.copytree(SUB/'MadLoop5_resources',directory/'resources')
    for folder,card in [('nominal','param_card.dat'),('alternate','param_alternate.dat')]:
        (directory/folder).mkdir()
        shutil.copy2(PRIOR/'runs/plain-cache0'/card,directory/folder/'param_card.dat')
        shutil.copy2(ORIGINAL/'Cards/ident_card.dat',directory/folder/'ident_card.dat')
    base=(PRIOR/'runs/plain-cache0/MadLoopParams.dat').read_text()
    (directory/'resources/MadLoopParams.dat').write_text(q.edit_card(base,{'TIRustGlobalCache':cache}))


def execute(label,variant,cache,timeout,build_state,q):
    directory=WORK/'runs'/label
    evidence=REPORT/'runs'/label
    directory.mkdir(parents=True,exist_ok=False)
    evidence.mkdir(parents=True,exist_ok=False)
    input_layout(directory,cache,q)
    before=q.inventory(directory)
    executable=build_state['executables'][variant]
    if digest(executable['path'])!=executable['sha256']: raise RuntimeError('Executable changed')
    if q.runtime(executable['path'])['files']!=executable['runtime']['files']: raise RuntimeError('Runtime changed')
    save(evidence/'inputs.json',{'label':label,'cache':cache,'variant':variant,'input_files':before,
        'build_sha256':digest(REPORT/'build.json'),'executable':executable})
    shutil.copytree(directory/'nominal',evidence/'nominal')
    shutil.copytree(directory/'alternate',evidence/'alternate')
    shutil.copy2(directory/'resources/MadLoopParams.dat',evidence/'MadLoopParams.dat')
    lock_path=ROOT/'.tirust-build/numerical-tests.lock'
    failure=None
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            q.command([executable['path'],str(cache)],'native-'+label,directory,timeout)
        except BaseException as error:
            failure=error
        finally:
            for name in ('records.txt','trace.txt','param.log'):
                path=directory/name
                if path.is_file(): shutil.copy2(path,evidence/name)
    after=q.inventory(directory)
    outputs={name:after.pop(name) for name in ('records.txt','trace.txt','param.log') if name in after}
    save(evidence/'outputs.json',{'declared_outputs':outputs,'input_files_after':after,
        'inputs_unchanged':after==before,'execution_exception':repr(failure) if failure else None})
    if after!=before: raise RuntimeError('A runtime input changed or undeclared output appeared')
    if set(outputs)!=set(('records.txt','trace.txt','param.log')):
        raise RuntimeError('Driver stopped without all declared output records')
    verified()
    q.verify_hashes(build_state['all_link_inputs_before_after'])
    if digest(executable['path'])!=executable['sha256']: raise RuntimeError('Executable changed after run')
    if q.runtime(executable['path'])['files']!=executable['runtime']['files']: raise RuntimeError('Runtime changed after run')
    generation=json.loads((BASE/'generation.json').read_text())
    if digest(ROOT/'input/mg7_configuration.txt')!=generation['main_configuration_sha256']:
        raise RuntimeError('Main configuration changed')
    q.verify_paths(generation['main_model_before_after'],ROOT/'models/loop_sm')
    if failure is not None: raise failure
    return evidence


def check_smoke(records_path,trace_path):
    expected=[['HEADER','1'],['INITIAL','-1','-1','0'],
        ['CARD','1','nominal/param_card.dat','4056CC083126E979','-1','-1','-1','-1','0'],
        ['CARD','2','alternate/param_card.dat','4057000000000000','-1','-1','-1','-1','0'],
        ['CARD','3','nominal/param_card.dat','4056CC083126E979','-1','-1','-1','-1','0'],
        ['DONE','3','-1','-1','0'],['CLOSED','-1','-1','0']]
    records=[line.split() for line in records_path.read_text().splitlines()]
    if records!=expected: raise RuntimeError('Card-only records differ from the frozen complete schema')
    a=analyzer()
    trace=a.parse_trace(trace_path)
    if (not trace['observed'] or trace['order']!=[1,2,3] or len(trace['records'])!=3
            or [len(f['records']) for f in trace['flushes']]!=[1,1,1,0]
            or any(f['token']!=-1 for f in trace['flushes'])):
        raise RuntimeError('Card-only trace is incomplete or contains a native reducer export')
    for action,row in enumerate(trace['records'],1):
        expected_fields=dict(sequence=action,action=action,process=0,kind=1,
            token_before=-1,token_after=-1,status=0,version=0,
            before=[0]*15,after=[0]*15)
        expected_fields.update({name:a.NA for name in a.TRACE_NAMES[7:13]+a.TRACE_NAMES[14:18]})
        if any(row[name]!=value for name,value in expected_fields.items()):
            raise RuntimeError('Card-only BEGIN carries unexpected native metadata')
        if len({row['f'+str(i)] for i in range(4)})!=1 or len({row['r'+str(i)] for i in range(4)})!=1:
            raise RuntimeError('Card-only observer altered the floating environment')
    return {'passed':True,'records':records,'trace_records':3,'native_exports_observed':0,
            'model_sequence':['nominal','alternate','nominal'],'context_generations':[-1,-1,-1]}


def run(stage):
    state,q=verified()
    built=json.loads((REPORT/'build.json').read_text())
    q.verify_hashes(built['all_link_inputs_before_after'])
    if stage=='smoke':
        evidence=execute('card-smoke','card-smoke',0,30,built,q)
        save(REPORT/'card-smoke-result.json',{
            'analysis':check_smoke(evidence/'records.txt',evidence/'trace.txt'),
            'evidence':q.inventory(evidence),'build_sha256':digest(REPORT/'build.json'),
            'execution_metadata_sha256':digest(REPORT/'execution/native-card-smoke.json')})
        return
    admit_smoke(built,q)
    for label,variant,cache in RUNS:
        evidence=execute(label,variant,cache,180,built,q)
        analyzer().parse_records(evidence/'records.txt')
        analyzer().parse_trace(evidence/'trace.txt')
    result=analyzer().analyze(REPORT/'runs')
    save(REPORT/'results.json',result)
    if not result['passed']: raise RuntimeError('Retained lifecycle analysis failed; see results.json')


def admit_smoke(built,q):
    evidence=REPORT/'runs/card-smoke'
    result=json.loads((REPORT/'card-smoke-result.json').read_text())
    if (result['evidence']!=q.inventory(evidence)
            or result['build_sha256']!=digest(REPORT/'build.json')
            or result['execution_metadata_sha256']!=digest(REPORT/'execution/native-card-smoke.json')
            or result['analysis']!=check_smoke(evidence/'records.txt',evidence/'trace.txt')
            or built['sources_freeze_sha256']!=digest(INDEX)):
        raise RuntimeError('Smoke evidence/build bindings changed')
    inputs=json.loads((evidence/'inputs.json').read_text())
    outputs=json.loads((evidence/'outputs.json').read_text())
    execution=json.loads((REPORT/'execution/native-card-smoke.json').read_text())
    if (inputs['build_sha256']!=result['build_sha256']
            or inputs['executable']!=built['executables']['card-smoke']
            or (inputs['label'],inputs['variant'],inputs['cache'])!=('card-smoke','card-smoke',0)
            or outputs['input_files_after']!=inputs['input_files']
            or not outputs['inputs_unchanged'] or outputs['execution_exception'] is not None
            or execution['returncode']!=0 or execution['timed_out']
            or execution['interruption'] is not None or execution['cleanup_exception'] is not None
            or execution['argv']!=['taskset','-c','10',built['executables']['card-smoke']['path'],'0']
            or execution['cwd']!=str(WORK/'runs/card-smoke')):
        raise RuntimeError('Smoke prerequisite execution or input inventory failed')
    for name,value in outputs['declared_outputs'].items():
        if value['kind']!='file' or digest(evidence/name)!=value['sha256']:
            raise RuntimeError('Retained smoke output changed')
    if set(outputs['declared_outputs'])!={'records.txt','trace.txt','param.log'}:
        raise RuntimeError('Smoke output inventory incomplete')
    current=q.inventory(WORK/'runs/card-smoke')
    for name,value in outputs['declared_outputs'].items():
        if current.pop(name)!=value: raise RuntimeError('Private smoke output changed')
    if current!=inputs['input_files']: raise RuntimeError('Private smoke inputs changed')
    executable=built['executables']['card-smoke']
    if digest(executable['path'])!=executable['sha256'] or q.runtime(executable['path'])['files']!=executable['runtime']['files']:
        raise RuntimeError('Smoke executable/runtime changed')


def main():
    global GUARD
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','build','smoke','run'])
    args=parser.parse_args()
    if Path.cwd()!=ROOT: parser.error('Run only from repository root')
    spec=importlib.util.spec_from_file_location('attempt4_process_guard',REPORT/'process_guard.py')
    guard_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard_module)
    with guard_module.ProcessGuard() as GUARD:
        try:
            {'freeze':freeze,'build':build,'smoke':lambda:run('smoke'),'run':lambda:run('run')}[args.stage]()
        except BaseException as error:
            save(REPORT/(args.stage+'-failure.json'),{'stage':args.stage,'exception':repr(error),
                'note':'Stopped; no automatic retry or continuation.'})
            raise


if __name__=='__main__':
    main()
