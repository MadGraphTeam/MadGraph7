#!/usr/bin/env python3
"""Two separately reviewed private gates; default is source-only inspection."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'.tirust-build/ew-refusal-replay-executor-1'
WORK=BASE/'work'
RESULTS=BASE/'results'
BUILD_GUARD=ROOT/'.tirust-build/kinematic-provenance-current-executor-1/build-admission-1'
BUILD_GUARD_SHA='c30fefa33c13dc04cbeed5b12a6a44284d1fc4858c12aa569382ea8a8f8bf813'
R=G=B=PLAN=None


def require(ok,message):
    if not ok:raise ValueError(message)


def pin(path):
    raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def authenticate(path,expected):
    raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==expected,'Reviewed index differs')
    for name,value in json.loads(raw)['files'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe index entry')
        require(pin(Path(path).parent/p)==value,'Source packet changed: '+name)


def verify_source(expected):
    authenticate(HERE/'index.json',expected)
    e=load('projection_exact_inputs',HERE/'extract_inputs.py')
    bindings=json.loads((HERE/'bindings.json').read_bytes())
    for name,value in bindings['indices'].items():e.seal(ROOT/name,value)
    for name,value in bindings['files'].items():e.checked(ROOT/name,value)
    for key in ('toolchain','fixed_supplemental_providers'):
        for name,value in bindings[key].items():e.checked(name,value)
    for value in list(bindings['traces'].values())+list(bindings['closures'].values()):e.checked(ROOT/value['path'],value)
    certificate=json.loads((HERE/'input-certificate.json').read_bytes())
    require(len(certificate['requests'])==2,'Replay request count differs')
    for request in certificate['requests']:e.checked(ROOT/request['input']['path'],request['input'])
    guard=load('projection_original_build_guard',BUILD_GUARD/'guard.py')
    controller,toolchain,protected=guard.prepare(BUILD_GUARD_SHA)
    return controller,guard,bindings,json.loads((HERE/'stage-plan.json').read_bytes()),protected


def setup_owned():
    R.WORK,R.RESULTS=WORK,RESULTS
    R.setup_owned_helpers();L=R.L
    L.STAGES=('build','run')
    L.CPU=12
    original=L.command
    def bounded(args,name,cwd=ROOT,timeout=600,cpu=True):
        # Reuse the existing owner; apply the same private process limits to
        # compiler/linker/loader/helper launches without another supervisor.
        return original([sys.executable,'-B',HERE/'limit_exec.py',*args],name,cwd,timeout,cpu)
    L.command=bounded
    return L


def runtime(executable,bound,label,closed=None):
    L=R.L
    receipt={'executable':str(executable),'passed':False,'loader_query_started':False}
    failure=None
    try:
        receipt['executable_pin']=pin(executable)
        original_toolchain=json.loads((BUILD_GUARD/'bindings.json').read_bytes())
        declared=G.compiler_pins(original_toolchain)
        actual=L.interpreter(executable);receipt['actual_interpreter']=str(actual)
        require(actual==declared,'PT_INTERP differs before loader query')
        L.q.verify_hashes(bound)
        for name,value in B['fixed_supplemental_providers'].items():G.verify(name,value)
        if closed is not None:L.q.verify_hashes(closed['files'])
        receipt['loader_query_started']=True
        result=L.runtime(executable);receipt['runtime']=result
        G.providers(result,bound,B['fixed_supplemental_providers'])
        if closed is not None:
            require(result['loader']==closed['loader'] and result['loader_sha256']==closed['loader_sha256'] and
                    result['files']==closed['files'],'Runtime differs from reviewed standalone build')
        L.q.verify_hashes(bound);G.compiler_pins(original_toolchain)
        for name,value in B['fixed_supplemental_providers'].items():G.verify(name,value)
        require(pin(executable)==receipt['executable_pin'],'Executable changed during loader query')
        receipt['passed']=True
        return result
    except BaseException as exc:
        failure=exc;raise
    finally:
        receipt['failure']=None if failure is None else repr(failure)
        G.save(RESULTS/(label+'-runtime.json'),receipt)


def build():
    directory=WORK/'build';require(not directory.exists(),'Fresh build only; no retry');directory.mkdir(parents=True)
    stage=PLAN['build_gate']
    R.command(stage['compile_argv'],'compile-driver',directory,60)
    object_path=directory/'replay.o'
    donor=ROOT/'.tirust-build/kinematic-provenance-current-executor-1/results/builds/cms-unmodified/prelink.json'
    bound=dict(json.loads(donor.read_bytes())['bound'])
    bound[str(object_path.resolve())]=pin(object_path)['sha256']
    for name,value in B['toolchain'].items():bound[name]=value['sha256']
    R.L.q.verify_hashes(bound)
    G.save(RESULTS/'prelink.json',{'bound':bound,'completed_before_link':True,'driver_object':pin(object_path),
                                'donor_prelink':pin(donor)})
    R.command(stage['link_argv'],'link-driver',directory,60)
    R.L.q.verify_hashes(bound)
    loads={}
    for name in re.findall(r'^LOAD (.+)$',(directory/'replay.map').read_text(),re.M):
        p=Path(name);p=p if p.is_absolute() else directory/p
        R.L.linker.require_prebound_load(p,bound);loads[str(p.resolve())]=pin(p)
    executable=directory/'replay'
    actual=runtime(executable,bound,'build')
    result={'schema':'private-ew-refusal-standalone-build-v1','executable':str(executable),
            'executable_pin':pin(executable),'driver_object':pin(object_path),'prelink_bound':bound,
            'actual_link_loads':loads,'runtime':actual,'fortran_main_runs':0,'tensor_calls':0}
    G.save(RESULTS/'build.json',result)
    return result


def reviewed_build(expected):
    require(expected,'Separate reviewed build closure hash required')
    p=RESULTS/'build-closure.json';require(pin(p)['sha256']==expected,'Reviewed build closure differs')
    closure=json.loads(p.read_bytes());require(closure['passed'],'Build closure did not pass')
    raw=G.verify(RESULTS/'build.json',closure['build_receipt'])
    result=json.loads(raw)
    G.verify(result['executable'],result['executable_pin'])
    R.L.q.verify_hashes(result['prelink_bound']);R.L.q.verify_hashes(result['runtime']['files'])
    return result


def run(expected,source_index):
    built=reviewed_build(expected)
    requests=json.loads((HERE/'input-certificate.json').read_bytes())['requests']
    require(len(requests)==2 and all(r['point']==2 and r['variant']=='stock' for r in requests),'Only two stock EW requests allowed')
    completed=[];failure=None
    try:
        for request in requests:
            R.L.remaining(30)
            label=request['id'];cwd=WORK/'workers'/label
            require(not cwd.exists(),'Fresh worker only; no retry');cwd.mkdir(parents=True)
            evidence=RESULTS/'runs'/label;require(not evidence.exists(),'Fresh evidence directory required')
            evidence.mkdir(parents=True)
            local_input=cwd/'input.words';source_input=ROOT/request['input']['path']
            G.verify(source_input,request['input']);shutil.copyfile(source_input,local_input)
            G.verify(local_input,request['input'])
            executable=Path(built['executable']);G.verify(executable,built['executable_pin'])
            observed=runtime(executable,built['prelink_bound'],label,closed=built['runtime'])
            expected_runtime={**observed['files'],str(executable.resolve()):built['executable_pin']['sha256']}
            specification={'schema':'private-ew-refusal-worker-v1','proposal_index_sha256':source_index,
                'request':request,'executable':str(executable),'executable_pin':built['executable_pin'],
                'input_file':str(local_input),'input_pin':pin(local_input),'expected_runtime':expected_runtime,
                'cwd':str(cwd),'evidence':str(evidence)}
            spec=evidence/'specification.json';G.save(spec,specification)
            R.command([sys.executable,'-B',HERE/'worker.py',spec,pin(spec)['sha256']],
                      'worker-'+label,cwd,30)
            result=json.loads((evidence/'result.json').read_bytes())
            require(result['orchestration_complete'],'Replay worker did not close correctly')
            completed.append({'id':label,'point':request['point'],'mode':request['mode'],
                              'variant':request['variant'],'fortran_status':result['fortran_status']})
            if request['variant']=='stock':
                expected_status=1
                require(result['fortran_status']==expected_status,'Stock status differs; reproduction incomplete')
        G.save(RESULTS/'responses.json',{'completed':completed,'success_status_is_not_accuracy_proof':True,
              'diagnostic_original_inputs_only':True,'coefficient_buffers_inspected':False})
    except BaseException as exc:
        failure=repr(exc);raise
    finally:
        G.save(RESULTS/'request-accounting.json',{'completed':completed,'failure':failure,
            'not_completed':[r['id'] for r in requests if r['id'] not in {x['id'] for x in completed}]})


def main():
    global R,G,B,PLAN
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',choices=('build','run'))
    parser.add_argument('--reviewed-source-index',required=True)
    parser.add_argument('--reviewed-build-sha256')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();require(Path.cwd()==ROOT,'Run from repository root')
    R,G,B,PLAN,before=verify_source(args.reviewed_source_index)
    save=G.save
    if not args.execute:
        print(json.dumps({'source_only':True,'native_commands':0,'stages':PLAN,'source_index':args.reviewed_source_index},indent=2));return
    require(args.stage,'One explicit stage required')
    if args.stage=='run':require(args.reviewed_build_sha256,'Separate reviewed build closure required')
    marker=RESULTS/('stage-'+args.stage+'.json');require(not marker.exists(),'Append-only stage; no retry')
    RESULTS.mkdir(parents=True,exist_ok=True);save(RESULTS/(args.stage+'-before.json'),before)
    L=setup_owned();L.CURRENT_STAGE=args.stage
    started=time.monotonic();failure=postfailure=None;lock=None;active=None
    with L.cleanup.SignalRequests() as signals:
        L.REQUESTS=R.BudgetRequests(signals)
        try:
            lock=R.acquire(signals);active=time.monotonic()
            with L.Deadline(PLAN[args.stage+'_gate']['work_cap_seconds']):
                if args.stage=='build':build()
                else:run(args.reviewed_build_sha256,args.reviewed_source_index)
        except BaseException as exc:
            failure=exc
        finally:
            if lock is not None:lock.close()
            try:
                _,_,_,_,after=verify_source(args.reviewed_source_index)
                require(after==before,'Protected original inputs changed')
                save(RESULTS/(args.stage+'-after.json'),after)
            except BaseException as exc:
                postfailure=exc
            result={'stage':args.stage,'passed':failure is None and postfailure is None,
                'failure':None if failure is None else repr(failure),'postcheck_failure':None if postfailure is None else repr(postfailure),
                'source_index_sha256':args.reviewed_source_index,'reviewed_build_sha256':args.reviewed_build_sha256,
                'elapsed_seconds':time.monotonic()-started,'elapsed_after_lock':None if active is None else time.monotonic()-active,
                'independent_accuracy_qualified':False,'performance_measurement':False}
            save(marker,result)
            if args.stage=='build':
                save(RESULTS/'build-closure.json',{**result,
                    'build_receipt':pin(RESULTS/'build.json') if (RESULTS/'build.json').exists() else None,
                    'before':pin(RESULTS/'build-before.json'),
                    'after':pin(RESULTS/'build-after.json') if (RESULTS/'build-after.json').exists() else None})
    if failure is not None:raise failure
    if postfailure is not None:raise postfailure


if __name__=='__main__':main()
