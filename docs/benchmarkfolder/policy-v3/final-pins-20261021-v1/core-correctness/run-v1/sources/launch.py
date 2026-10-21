#!/usr/bin/env python3
"""Owned, retained launcher; requires the later explicit root handoff."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import core_collect as core


def identity(pid):
    path=Path('/proc')/str(pid)
    try:
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        result={'pid':pid,'ppid':int(fields[1]),'pgid':int(fields[2]),'start_ticks':int(fields[19]),
                'state':fields[0],'affinity':sorted(os.sched_getaffinity(pid)),
                'command':(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
        confirm=(path/'stat').read_text().rsplit(')',1)[1].split()
        return result if int(confirm[19])==result['start_ticks'] else None
    except (FileNotFoundError,ProcessLookupError,PermissionError): return None


def save(path,value):
    with path.open('x') as stream: json.dump(value,stream,indent=2);stream.write('\n')


def cleanup_verified_workers(report,attempt):
    """Emergency fallback only. Never signal a PID whose start ticks changed."""
    ledger=report/'process-ledger.jsonl'; actions=[]
    if not ledger.exists(): return actions
    for event in reversed([json.loads(line) for line in ledger.read_text().splitlines()]):
        if event['event']=='spawn':
            recorded={key:event[key] for key in ('pid','pgid','start_ticks')}
        elif event['event'] in ('worker-start','build-start'):
            recorded=event['identity']
        else:continue
        actual=identity(recorded['pid'])
        if actual is None: continue
        if actual['start_ticks']!=recorded['start_ticks'] or actual['pgid']!=recorded['pid']:
            actions.append({'action':'refused-stale-identity','recorded':recorded,'actual':actual});continue
        try: os.killpg(actual['pgid'],signal.SIGTERM)
        except ProcessLookupError: continue
        actions.append({'action':'TERM','identity':actual})
        until=time.monotonic()+10
        while time.monotonic()<until:
            current=identity(actual['pid'])
            if current is None or current['state']=='Z':break
            time.sleep(.1)
        current=identity(actual['pid'])
        if current and current['state']!='Z' and current['start_ticks']==actual['start_ticks']:
            try: os.killpg(actual['pgid'],signal.SIGKILL)
            except ProcessLookupError: continue
            actions.append({'action':'KILL','identity':current})
    save(attempt/'emergency-cleanup.json',actions)
    return actions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specification',type=Path);parser.add_argument('--handoff',required=True,type=Path)
    parser.add_argument('--attempt',required=True,type=Path);parser.add_argument('--execute-numerical',action='store_true')
    args=parser.parse_args();spec=core.policy.verified(core.read(args.specification),core.SCHEMA)
    if not args.execute_numerical:raise ValueError('No numerical launch without explicit root handoff')
    if not {6,7,190}.issubset(os.sched_getaffinity(0)):
        raise ValueError('Launch with taskset -c6,7,190; numerical workers alone are pinned190')
    for item in spec['sources'].values():core.supplement.checked(item)
    handoff=core.read(args.handoff)
    if (handoff.get('root_authorized') is not True or handoff.get('cpu')!=190 or
            handoff.get('phase')!='final-pins-core-correctness' or
            handoff.get('specification_sha256')!=core.supplement.digest(args.specification)):
        raise ValueError('Handoff does not authorize this exact spec')
    args.attempt.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-B',str(HERE/'core_collect.py'),'collect',str(args.specification),
             '--execute-numerical','--handoff',str(args.handoff)]
    child=None;owned=None;started=time.monotonic();interruptions=[];defer_interruptions=True
    thread_environment=dict(core.THREAD_ENV)
    child_environment=os.environ.copy();child_environment.update(thread_environment)
    def interrupted(signum,_frame):
        interruptions.append({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'signal':signum})
        # Signals may arrive even between Popen returning and STORE_FAST child.
        # Finish ownership and durable recording before raising the interruption.
        if not defer_interruptions:
            raise KeyboardInterrupt('Launcher interrupted; retained owned cleanup follows')
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,interrupted)
    try:
        with (args.attempt/'stdout').open('xb') as out,(args.attempt/'stderr').open('xb') as err:
            child=subprocess.Popen(command,cwd=core.ROOT,env=child_environment,stdout=out,stderr=err,start_new_session=True)
            owned=identity(child.pid)
            save(args.attempt/'launch.json',{'status':'started','controller_identity':owned,
                'launcher_identity':identity(os.getpid()),'command':command,'specification':core.supplement.bound(args.specification),
                'handoff':core.supplement.bound(args.handoff),'orchestration_head_at_launch':subprocess.check_output(
                    ['git','rev-parse','HEAD'],cwd=core.ROOT,text=True).strip(),
                'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'thread_environment':thread_environment,
                'worker_cpu':190,'worker_phase_timeout_seconds':86400,'build_timeout_seconds':21600,
                'automatic_retry':False,'performance_acceptance_gate':False})
            defer_interruptions=False
            if interruptions:
                raise KeyboardInterrupt('Deferred launcher interruption after durable ownership record')
            code=child.wait()
            if code:
                cleanup_verified_workers(core.policy.path(spec['report']),args.attempt)
    except BaseException:
        # Further signals are retained rather than interrupting bounded cleanup.
        defer_interruptions=True
        if child is not None and child.poll() is None:
            actual=identity(child.pid)
            if actual and (owned is None or actual['start_ticks']==owned['start_ticks']):
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.send_signal(signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
        cleanup_verified_workers(core.policy.path(spec['report']),args.attempt)
        code=child.returncode if child is not None else None
        raise
    finally:
        if child is not None:
            save(args.attempt/'result.json',{'status':'complete' if child.returncode==0 else 'execution-not-complete',
                'returncode':child.returncode,'elapsed_seconds':time.monotonic()-started,
                'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'interruptions':interruptions,'thread_environment':thread_environment,
                'automatic_retry':False,'performance_acceptance_gate':False})
    return code


if __name__=='__main__':raise SystemExit(main())
