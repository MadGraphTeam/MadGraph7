#!/usr/bin/env python3
"""Stress-only successor; reuses the sealed ordinary provider admission hook."""
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
OUT=RESULTS/'stress-admission-1'
ORDINARY_SHA='325eb3205b7990e5e8ec367b25ef0af48b41d7271e227dbbd15bdc878be560cf'
O=None


def require(ok,message):
    if not ok:raise ValueError(message)


def authenticate(path,expected):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected,'Reviewed index differs: '+str(path))
    for name,pin in json.loads(raw)['files'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe index path')
        data=(Path(path).parent/p).read_bytes()
        require(len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256'],
                'Sealed bytes differ: '+str(p))


def select_requests(all_requests):
    requests=[item for item in all_requests if item['stage']=='stress']
    expected={case+'-point'+str(point)+'-mode'+str(mode)+'-'+variant
              for case,point in (('cms',20),('cms',21),('ew',2))
              for mode in (1,4) for variant in ('unmodified','disabled','enabled')}
    require(len(requests)==18 and {r['id'] for r in requests}==expected,'Stress request set differs')
    for item in requests:
        case,point,mode,variant=item['case'],item['point'],item['requested_mode'],item['variant']
        require(item['id']==case+'-point'+str(point)+'-mode'+str(mode)+'-'+variant and
                (case,point) in (('cms',20),('cms',21),('ew',2)) and mode in (1,4) and
                item['bootstrap']==case+'-point1.dat' and item['bootstrap_point']==1 and
                item['bootstrap_outer_entries']==1 and item['maximum_outer_amplitude_entries']==2 and
                item['measurement']==case+'-point'+str(point)+'.dat' and
                item['trace_enabled']==(variant=='enabled'),'Stress request fields or bootstrap differ')
    return requests


def prepare(expected):
    global O
    authenticate(HERE/'index.json',expected)
    authenticate(BASE/'ordinary-admission-1/index.json',ORDINARY_SHA)
    spec=importlib.util.spec_from_file_location('ktrace_ordinary_guard_reuse',BASE/'ordinary-admission-1/guard.py')
    O=importlib.util.module_from_spec(spec);sys.modules[spec.name]=O;spec.loader.exec_module(O)
    r,toolchain,builds,_,protected=O.prepare(ORDINARY_SHA)
    pins=json.loads((HERE/'closed-ordinary-bindings.json').read_bytes())
    for name,pin in pins['files'].items():O.G.verify(ROOT/name,pin)
    closure=json.loads((RESULTS/'ordinary-terminal-closure.json').read_bytes())
    require(closure['terminal_exit_code']==0 and closure['stage']['passed'] and closure['guard_result']['passed']
            and closure['parity']['passed'] and closure['actual_native_workers']==12
            and closure['actual_outer_entries_observed']==24,'Closed ordinary admission did not pass')
    for worker in closure['workers'].values():
        for name,pin in worker['files'].items():O.G.verify(ROOT/name,pin)
    for name,item in closure['commands'].items():
        O.G.verify(ROOT/name,item['receipt']);O.G.verify((ROOT/name).with_suffix('.txt'),item['log'])
    requests=select_requests(json.loads((BASE/'source/inputs/requests.json').read_bytes())['requests'])
    return r,toolchain,builds,requests,{'guard_index':O.G.record(HERE/'index.json'),
        'closed_ordinary_bindings':O.G.record(HERE/'closed-ordinary-bindings.json'),
        'closed_files':pins['files'],'protected':protected,
        'stress_ids':sorted(item['id'] for item in requests),'maximum_workers':18,
        'maximum_outer_amplitude_entries':36,'further_native_stage_admitted':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-guard-index',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();require(Path.cwd()==ROOT,'Run from repository root')
    r,toolchain,builds,requests,before=prepare(args.reviewed_guard_index)
    # Keep the initialized writer even if a failing postcheck replaces global O.
    save=O.G.save
    if not args.execute:
        print(json.dumps({'passed':True,'native_commands':0,'before':before},sort_keys=True,indent=2));return
    require(not OUT.exists() and not (RESULTS/'stage-stress.json').exists(),'Append-only stress stage; no retry')
    OUT.mkdir();save(OUT/'before.json',before)
    original_setup=r.setup_owned_helpers;seen=set()
    def setup():
        nonlocal seen
        original_setup();seen=O.install_hook(r,toolchain,builds,requests,OUT)
    r.setup_owned_helpers=setup
    oldargv=sys.argv;failure=postfailure=None
    try:
        sys.argv=[str(BASE/'source/run.py'),'stress','--execute',
            '--reviewed-source-index',O.G.SOURCE_SHA,'--reviewed-derived-index',O.G.DERIVED_SHA]
        r.main();require(len(seen)==18,'Stress stage did not finish eighteen runtime admissions')
        require(json.loads((RESULTS/'stress-parity.json').read_bytes())['passed'],'Stress U/D/E parity did not pass')
    except BaseException as exc:
        failure=exc
    finally:
        sys.argv=oldargv
        try:
            _,_,_,_,after=prepare(args.reviewed_guard_index)
            require(after==before,'Protected ordinary/build/original inputs changed')
            save(OUT/'after.json',after)
        except BaseException as exc:
            postfailure=exc
        save(OUT/'result.json',{'passed':failure is None and postfailure is None,
            'failure':None if failure is None else repr(failure),
            'postcheck_failure':None if postfailure is None else repr(postfailure),
            'attempted_runtime_admissions':sorted(seen),'maximum_workers':18,
            'maximum_outer_amplitude_entries':36,'further_native_stage_admitted':False,
            'guard_index_sha256':args.reviewed_guard_index,'ordinary_guard_index_sha256':ORDINARY_SHA})
    if failure is not None:raise failure
    if postfailure is not None:raise postfailure


if __name__=='__main__':main()
