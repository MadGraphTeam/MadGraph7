"""One separately admitted stopped-operand replay; verify is read/hash only."""
import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.tirust-build/post-cayley-sigfpe-register-proposal-1'
RUN = WORK / 'attempt1'
OLD = ROOT / 'docs/benchmarkfolder/experiments/adaptive-qp-checked-trap-proposal-1/controller.py'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()

def old_module():
    spec = importlib.util.spec_from_file_location('frozen_operand_owner_helpers',OLD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify():
    assert Path.cwd().resolve() == ROOT
    index = json.loads((HERE / 'source-index.json').read_text())
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and p != HERE / 'source-index.json'}
    assert actual == set(index['sources'])
    for name, digest in index['sources'].items():
        assert sha(HERE / name) == digest, name
    plan = json.loads((HERE / 'plan.json').read_text())
    for name, digest in plan['inputs'].items():
        assert sha(name) == digest, name
    return plan

def subreaper(enabled):
    libc = ctypes.CDLL(None,use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(36,int(enabled),0,0,0) != 0:
        raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER')

def reap_adopted(helper, row):
    """After GDB leader reaping, reserve each remaining child before kill/wait."""
    row['adopted_children'] = []
    with helper['deferred_signals']() as pending:
        while True:
            children = Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()
            if not children:
                break
            for child_text in children:
                child = int(child_text)
                # Default SIGCHLD/no other reaper reserves this child PID.
                status = os.waitid(os.P_PID,child,os.WEXITED|os.WNOHANG|os.WNOWAIT)
                if status is None:
                    os.kill(child,signal.SIGKILL)
                waited, code = os.waitpid(child,0)
                assert waited == child
                row['adopted_children'].append({'pid':child,'wait_status':code,'reaped':True})
        row['all_owned_children_reaped'] = True
    if pending:
        raise InterruptedError('Interrupted during inferior cleanup: '+repr(pending))

def run_stage(start,deadline):
    plan = verify()
    old = old_module()
    helper, adapted = old.supervisor()
    with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(75)
        assert not RUN.exists(), 'One admitted attempt only; no automatic retry'
        RUN.mkdir()
        old.limits()
        subreaper(True)
        record = {'schema':'post-cayley-stopped-operands-attempt-v1','complete':False,
            'root_admission_required':True,'source_index_sha256':sha(HERE / 'source-index.json'),
            'cpu':12,'cpu_exclusive':False,'address_space_bytes':4294967296,
            'aggregate_seconds':300,'automatic_retry':False,'workers':{},
            'adapted_supervisor_sha256':adapted,'protected_inputs_before':plan['inputs'],
            'accuracy_stability_trap_completion_or_performance_qualified':False}
        try:
            for case in plan['cases']:
                verify()
                old.remaining(deadline,5)
                directory = RUN / case['name']
                directory.mkdir()
                for source_name, digest in case['fixtures'].items():
                    source = Path(source_name)
                    target = directory / source.relative_to(Path(case['fixture_root']))
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(source,target)
                    assert sha(target) == digest
                launch = {'debugger':plan['debugger'],'executable':plan['executable'],
                    'target_argv':[plan['executable'],str(directory / 'selected.dat'),'trapped'],
                    'target_environment':case['target_environment'],
                    'capture_script':str(HERE / 'capture.py'),
                    'mapping_helper':plan['mapping_helper'],
                    'mapping_helper_sha256':plan['inputs'][plan['mapping_helper']],
                    'expected_runtime_sha256':plan['expected_runtime_sha256'],
                    'selected_library':plan['selected_library']}
                old.save(directory / 'launch.json',launch)
                row = {'argv':[plan['python'],'-I','-B',str(HERE / 'attach_wrapper.py')],
                       'progress':{},'physical_policy':'trapped','mode':case['mode']}
                record['workers'][case['name']] = row
                old.save(RUN / 'execution.json',record)
                try:
                    with (directory / 'debugger.log').open('xb') as log:
                        row['exit_code'] = helper['run_plain_capped'](row['argv'],log,
                            cwd=directory,env=plan['debugger_environment'],progress=row['progress'],
                            seconds=min(120,old.remaining(deadline,5)))
                finally:
                    reap_adopted(helper,row)
                    old.save(RUN / 'execution.json',record)
                capture = json.loads((directory / 'capture.json').read_text())
                row['stopped_operands_captured'] = bool(row['exit_code']==0 and capture['captured'] and
                    capture.get('inferior_pid_after_kill')==0 and 'kill_failure' not in capture)
                row['files_after'] = {str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}
                assert row['stopped_operands_captured'], 'No qualified stopped operands; no retry'
            record['complete'] = True
        except BaseException as error:
            record['failure'] = {'type':type(error).__name__,'message':str(error)}
        finally:
            def closure():
                verify()
                record['protected_inputs_after'] = 'unchanged'
                assert not Path(f'/proc/self/task/{os.getpid()}/children').read_text().strip()
                subreaper(False)
            old.publish_final(RUN / 'execution.json',record,deadline,start,'complete',closure)
        print(json.dumps({'complete':record['complete'],'accuracy_or_performance_qualified':False}))
        if not record['complete']:
            raise SystemExit(1)

if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('verify','run'))
    if cli.parse_args().action == 'verify':
        print(json.dumps({'verified':bool(verify()),'debugger_or_native_executed':False}))
    else:
        old_module().timed_call(run_stage,300)
