#!/usr/bin/env python3
"""Ordinary-only derivative admission; default is a file-only preflight."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[1]
RESULTS=BASE/'results'
OUT=RESULTS/'ordinary-admission-1'
BUILD_GUARD_SHA='c30fefa33c13dc04cbeed5b12a6a44284d1fc4858c12aa569382ea8a8f8bf813'
G=None


def require(ok,message):
    if not ok:raise ValueError(message)


def authenticate(path,expected):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected,'Reviewed index differs: '+str(path))
    for name,pin in json.loads(raw)['files'].items():
        p=Path(name)
        require(not p.is_absolute() and '..' not in p.parts,'Unsafe index path')
        data=(Path(path).parent/p).read_bytes()
        require(len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256'],
                'Sealed bytes differ: '+str(p))


def setup_guard(expected):
    global G
    authenticate(HERE/'index.json',expected)
    authenticate(BASE/'build-admission-1/index.json',BUILD_GUARD_SHA)
    spec=importlib.util.spec_from_file_location('ktrace_closed_build_guard',BASE/'build-admission-1/guard.py')
    G=importlib.util.module_from_spec(spec);sys.modules[spec.name]=G;spec.loader.exec_module(G)


def verify_hashes(files):
    for name,checksum in files.items():
        require(G.record(name)['sha256']==checksum,'Closed input bytes differ: '+name)


def prepare(expected):
    setup_guard(expected)
    pins=json.loads((HERE/'closed-build-bindings.json').read_bytes())
    for name,pin in pins['files'].items():G.verify(ROOT/name,pin)
    r,toolchain,protected=G.prepare(BUILD_GUARD_SHA)
    require(json.loads((RESULTS/'stage-build.json').read_bytes())['passed'],'Build did not pass')
    result=json.loads((RESULTS/'build-admission-1/result.json').read_bytes())
    require(result['passed'] and result['amplitude_evaluations']==0,'Guarded build did not pass')
    builds={}
    for label in pins['build_labels']:
        build=json.loads((RESULTS/'builds'/label/'build.json').read_bytes())
        G.verify(build['executable'],build['executable_pin'])
        r.exact_inventory(Path(build['output']),build['sources'])
        verify_hashes(build['prelink_bound']);verify_hashes(build['runtime']['files'])
        require(build['runtime']['loader']==toolchain['loader'],'Closed build loader differs')
        builds[label]=build
    requests=json.loads((BASE/'source/inputs/requests.json').read_bytes())['requests']
    requests=[item for item in requests if item['stage']=='ordinary']
    expected_ids={case+'-point1-mode'+str(mode)+'-'+variant for case in ('cms','ew')
                  for mode in (1,4) for variant in ('unmodified','disabled','enabled')}
    require(len(requests)==12 and {item['id'] for item in requests}==expected_ids,
            'Ordinary stage is not the exact admitted twelve requests')
    return r,toolchain,builds,requests,{'guard_index':G.record(HERE/'index.json'),
        'closed_build_bindings':G.record(HERE/'closed-build-bindings.json'),
        'closed_files':pins['files'],'protected':protected,'ordinary_ids':sorted(expected_ids),
        'maximum_workers':12,'maximum_outer_amplitude_entries':24,'stress_admitted':False}


def install_hook(r,toolchain,builds,requests,out):
    original=r.L.runtime
    expected={str((BASE/'work/workers'/item['id']/'SubProcesses'/r.CASES[item['case']][1]/'check').resolve()):item
              for item in requests}
    seen=set()
    def guarded(executable,deadline=None):
        executable=Path(executable).resolve()
        require(str(executable) in expected,'Runtime admission outside ordinary worker set')
        request=expected[str(executable)];label=request['id']
        require(label not in seen,'Repeated runtime admission; no retry')
        seen.add(label)
        variant='unmodified' if request['variant']=='unmodified' else 'instrumented'
        build=builds[request['case']+'-'+variant]
        receipt={'worker_id':label,'executable':str(executable),'passed':False,
                 'loader_query_started':False,'closed_runtime':build['runtime']['files']}
        failure=None
        try:
            G.verify(executable,build['executable_pin'])
            loader=G.compiler_pins(toolchain)
            actual=r.L.interpreter(executable)
            receipt['actual_interpreter']=str(actual)
            require(actual==loader and str(actual)==build['runtime']['loader'],
                    'Worker PT_INTERP differs from closed build/declared loader')
            verify_hashes(build['runtime']['files']);verify_hashes(build['prelink_bound'])
            receipt['loader_query_started']=True
            runtime=original(executable,deadline)
            receipt['runtime']=runtime
            require(runtime['loader']==build['runtime']['loader'] and
                    runtime['loader_sha256']==build['runtime']['loader_sha256'] and
                    runtime['files']==build['runtime']['files'],'Worker runtime differs from closed build')
            verify_hashes(build['runtime']['files']);verify_hashes(build['prelink_bound'])
            G.verify(executable,build['executable_pin']);G.compiler_pins(toolchain)
            receipt['passed']=True
            return runtime
        except BaseException as exc:
            failure=exc
            raise
        finally:
            receipt['failure']=None if failure is None else repr(failure)
            G.save(out/(label+'-runtime.json'),receipt)
    r.L.runtime=guarded
    return seen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-guard-index',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    require(Path.cwd()==ROOT,'Run from repository root')
    r,toolchain,builds,requests,before=prepare(args.reviewed_guard_index)
    if not args.execute:
        print(json.dumps({'passed':True,'native_commands':0,'before':before},sort_keys=True,indent=2));return
    require(not OUT.exists() and not (RESULTS/'stage-ordinary.json').exists(),'Append-only ordinary stage; no retry')
    OUT.mkdir();G.save(OUT/'before.json',before)
    original_setup=r.setup_owned_helpers
    seen=set()
    def setup():
        nonlocal seen
        original_setup();seen=install_hook(r,toolchain,builds,requests,OUT)
    r.setup_owned_helpers=setup
    oldargv=sys.argv;failure=postfailure=None
    try:
        sys.argv=[str(BASE/'source/run.py'),'ordinary','--execute',
            '--reviewed-source-index',G.SOURCE_SHA,'--reviewed-derived-index',G.DERIVED_SHA]
        r.main()
        require(len(seen)==12,'Ordinary stage did not finish twelve runtime admissions')
        require(json.loads((RESULTS/'ordinary-parity.json').read_bytes())['passed'],
                'Ordinary U/D/E parity did not pass')
    except BaseException as exc:
        failure=exc
    finally:
        sys.argv=oldargv
        try:
            _,_,_,_,after=prepare(args.reviewed_guard_index)
            require(after==before,'Protected build or original inputs changed')
            G.save(OUT/'after.json',after)
        except BaseException as exc:
            postfailure=exc
        G.save(OUT/'result.json',{'passed':failure is None and postfailure is None,
            'failure':None if failure is None else repr(failure),
            'postcheck_failure':None if postfailure is None else repr(postfailure),
            'attempted_runtime_admissions':sorted(seen),'maximum_workers':12,
            'maximum_outer_amplitude_entries':24,'stress_admitted':False,
            'guard_index_sha256':args.reviewed_guard_index,'build_guard_index_sha256':BUILD_GUARD_SHA})
    if failure is not None:raise failure
    if postfailure is not None:raise postfailure


if __name__=='__main__':main()
