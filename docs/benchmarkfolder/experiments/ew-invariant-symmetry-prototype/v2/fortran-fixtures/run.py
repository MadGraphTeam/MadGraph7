#!/usr/bin/env python3
"""Explicit future compiler/stub fixture. Default only displays the plan.

No reducers, amplitudes, model libraries, installer or campaign are invoked.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import runpy
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(plan):
    for value in plan['inputs']:
        path = ROOT / value['path']
        require(path.is_file() and not path.is_symlink(), 'Input missing or symlink: ' + str(path))
        require(sha(path) == value['sha256'], 'Input changed: ' + str(path))
    compiler = plan['compiler']
    require(sha(compiler['path']) == compiler['sha256'], 'Compiler changed')
    require(str(Path(compiler['path']).resolve()) == compiler['resolved_path'], 'Compiler resolution changed')
    require(sys.byteorder == 'little', 'Fixture raw-word encoding requires little endian')
    return {x['path']: x['sha256'] for x in plan['inputs']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Compile and run the bounded stub corpus; requires separate task authorization')
    parser.add_argument('--label', default='attempt1')
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    plan_path = HERE / 'plan.json'
    plan = json.loads(plan_path.read_text())
    require(Path.cwd().resolve() == ROOT, 'Run from repository')
    protected = verify(plan)
    if not args.execute:
        print(json.dumps({'source_only': True, 'plan_sha256': sha(plan_path),
                          'cpu': plan['cpu'], 'whole_seconds': plan['whole_seconds'],
                          'configs': plan['configs'], 'reducers': False,
                          'command': 'python3 -I -B ' + str(Path(__file__).relative_to(ROOT))
                          + ' --execute --label LABEL --expected-plan-sha256 ' + sha(plan_path)}, indent=2))
        return
    require(args.expected_plan_sha256 == sha(plan_path), 'Explicit frozen plan hash required')
    require(re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,48}', args.label), 'Unsafe label')
    require(plan['cpu'] in os.sched_getaffinity(0), 'Bound CPU unavailable')
    for key in ('LD_PRELOAD', 'LD_AUDIT', 'LD_DEBUG', 'GFORTRAN_OPTIONS',
                'GFORTRAN_INCLUDE_PATH', 'COMPILER_PATH', 'LIBRARY_PATH', 'GCC_EXEC_PREFIX'):
        require(not os.environ.get(key), 'Compiler/runtime injection rejected: ' + key)
    work = ROOT / '.tirust-build/ew-invariant-symmetry-fixtures-v2' / args.label
    require(not work.exists() and not work.is_symlink(), 'Attempt already exists')
    work.parent.mkdir(parents=True, exist_ok=True)
    require(work.parent.resolve() == work.parent, 'Work parent is symlinked')
    work.mkdir()
    lifecycle_path = ROOT / plan['lifecycle_helper']['path']
    lifecycle = runpy.run_path(str(lifecycle_path), run_name='invariant_fixture_lifecycle')
    analysis = runpy.run_path(str(HERE / 'analyze.py'), run_name='invariant_fixture_analysis')
    deadline = lifecycle['Deadline'](plan['whole_seconds'])
    old_handlers = {}
    record = {'schema': 'ew-invariant-compiled-fixture-execution-v2', 'status': 'running',
              'plan_sha256': sha(plan_path), 'protected_before': protected, 'commands': [],
              'bulk': [], 'traps': [], 'mutations': [], 'reducers': False,
              'physical_amplitude': False, 'performance_eligible': False,
              'environment': {k:v for k,v in sorted(os.environ.items())
                              if k.startswith(('NIX_', 'LD_', 'OMP_', 'GFORTRAN_'))
                              or k in ('PATH', 'LANG', 'LC_ALL', 'TZ')},
              'artifacts': {}, 'loader_lists': []}
    # Same small explicit environment used by the reviewed component builder;
    # ambient NIX flags cannot silently amend the selected numerical flags.
    env = {'PATH': str(Path(plan['compiler']['path']).parent) + os.pathsep + os.defpath,
           'LANG': 'C', 'LC_ALL': 'C', 'TZ': 'UTC', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TMPDIR': str(work/'tmp')}
    (work/'tmp').mkdir()
    record['effective_environment'] = env
    error = None

    def interrupted(signum, frame):
        raise InterruptedError('Fixture interrupted by signal ' + str(signum))

    def command(argv, cwd, label, allow_signal=False):
        deadline.check()
        event = {'argv': [plan['taskset'], '-c', str(plan['cpu'])] + list(map(str, argv)),
                 'cwd': str(cwd), 'label': label}
        log = work / (label + '.stdout')
        event['stdout'] = str(log)
        record['commands'].append(event)
        child = lifecycle['OwnedChild'](deadline, event)
        active_error = None
        try:
            with log.open('xb') as output:
                child.spawn(event['argv'], cwd, env, output)
                while not child.exited():
                    time.sleep(min(.01, deadline.check()))
                code = child.finish()
            require(event.get('process_reaped') is True, 'Child not reaped')
            if not allow_signal:
                require(code == 0, 'Command failed: ' + label)
            return code, log.read_text(errors='strict')
        except BaseException as exc:
            active_error = exc
            child.cleanup_after(exc)
            raise
        finally:
            try:
                if log.exists():
                    event.update(stdout_sha256=sha(log), stdout_size=log.stat().st_size)
                (work / 'progress.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
            except BaseException as retention_error:
                if active_error is None:
                    raise
                active_error.add_note('Retaining command metadata failed: ' + repr(retention_error))

    def runtime_list(executable, directory, label):
        _, text = command([plan['runtime_loader'], '--list', executable], directory, label)
        paths = re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(', text, flags=re.M)
        require(paths and 'not found' not in text, 'Unresolved fixture runtime')
        resolved = {}
        for name in paths:
            path = Path(name).resolve(strict=True)
            digest = sha(path)
            require(plan['runtime_anchors'].get(str(path)) == digest,
                    'Fixture loader input outside retained anchors: ' + str(path))
            resolved[str(path)] = digest
        record['loader_lists'].append({'executable': str(executable), 'resolved_files': resolved,
            'scope': 'Explicit retained loader --list; no amplitude or reducer initialization'})

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old_handlers[signum] = signal.signal(signum, interrupted)
        compiler = plan['compiler']['path']
        for config in plan['configs']:
            name, opt = config['name'], config['additional_flags']
            directory = work / name
            directory.mkdir()
            fixed = plan['fixed_flags'] + opt + ['-I', str(HERE)]
            free = plan['free_flags'] + opt
            for source, target in [('shared-kinematics.f','shared.o'), ('wrappers.f','wrappers.o')]:
                command([compiler] + fixed + ['-c', str(HERE/source), '-o', target], directory, name+'-'+target)
            for precision, kind in [('dp', 8), ('qp', 16)]:
                obj, executable = 'driver_'+precision+'.o', directory/('driver_'+precision)
                command([compiler] + free + ['-c', str(HERE/('driver_'+precision+'.f90')), '-o', obj], directory, name+'-'+obj)
                command([compiler, '-w', '-fPIC'] + opt + ['shared.o','wrappers.o',obj,'-o',str(executable)], directory, name+'-'+precision+'-link')
                record['artifacts'][str(executable)] = sha(executable)
                runtime_list(executable,directory,name+'-'+precision+'-loader-list')
                _, stdout = command([executable,'bulk'], directory, name+'-'+precision+'-bulk')
                result = analysis['bulk'](stdout, kind)
                deadline.check()
                record['bulk'].append({'config':name, 'precision':precision, **result})
                for route, caseid, trapid in plan['trap_cases']:
                    for rounding in range(1,5):
                        observations=[]
                        for arm in (0,1):
                            label=f'{name}-{precision}-trap-{route}-{caseid}-{rounding}-{trapid}-{arm}'
                            observations.append(command([executable,'trap',route,caseid,rounding,trapid,arm], directory,label,True))
                        (oa,ta),(ob,tb)=observations
                        result=analysis['trap'](ta,tb,oa,ob)
                        deadline.check()
                        record['traps'].append({'config':name,'precision':precision,
                            'route':route,'case':caseid,'rounding':rounding,'trap':trapid,**result})
            if name == 'historical-flags':
                for mutation in ('no-finite','no-equality'):
                    obj='mutation-'+mutation+'.o'; executable=directory/('mutation-'+mutation)
                    command([compiler]+fixed+['-c',str(HERE/('mutation-'+mutation+'.f')),'-o',obj],directory,name+'-'+obj)
                    command([compiler,'-w','-fPIC','shared.o',obj,'driver_dp.o','-o',str(executable)],directory,name+'-'+mutation+'-link')
                    record['artifacts'][str(executable)]=sha(executable)
                    runtime_list(executable,directory,name+'-'+mutation+'-loader-list')
                    _,stdout=command([executable,'bulk'],directory,name+'-'+mutation+'-bulk')
                    result=analysis['bulk'](stdout,8)
                    deadline.check()
                    discriminates=any('matrix policy differs' in failure['problems'] for failure in result['failures'])
                    record['mutations'].append({'mutation':mutation,'detected_matrix_failure':discriminates,**result})
        require(all(value['detected_matrix_failure'] for value in record['mutations'])
                and len(record['mutations'])==2, 'Negative control did not discriminate')
        deadline.check()
        record['matrix_and_ieee_parity_passed']=all(value['passed'] for value in record['bulk'])
        record['supported_trap_parity_passed']=all(value['equivalent'] for value in record['traps'] if value['supported'])
        record['status']='complete'
        record['passed']=record['matrix_and_ieee_parity_passed'] and record['supported_trap_parity_passed']
    except BaseException as exc:
        error=exc
        record.update(status='failed',passed=False,error=repr(exc))
        raise
    finally:
        for signum,handler in old_handlers.items():
            signal.signal(signum,handler)
        try:
            record['protected_after']=verify(plan)
            require(record['protected_before']==record['protected_after'],'Protected source drift')
            for path,digest in record['artifacts'].items():
                require(sha(path)==digest,'Executable drift: '+path)
        except BaseException as final_error:
            record.update(status='failed',passed=False,final_verification_error=repr(final_error))
            if error is None:
                error=final_error
            else:
                error.add_note('Final verification failed: '+repr(final_error))
        try:
            (work/'execution.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
        except BaseException as retention_error:
            if error is None:
                raise
            error.add_note('Final evidence retention failed: '+repr(retention_error))
        if error is not None and sys.exc_info()[0] is None:
            raise error
    print(json.dumps({'execution':str(work/'execution.json'),'passed':record['passed'],'reducers':False}))
    if not record['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
