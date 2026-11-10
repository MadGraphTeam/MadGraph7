"""Own one bounded Linux process group and retain actual ELF mappings."""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import os
import signal
import subprocess
import time

SIGNALS=(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)


@contextmanager
def deferred_signals():
    """Atomic parent handler transitions; original mask restored before yield."""
    previous={};pending=[]
    def defer(signum,frame):
        pending.append(signum)
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:
        try:
            for signum in SIGNALS:
                previous[signum]=signal.signal(signum,defer)
        except BaseException:
            for signum,handler in previous.items():
                signal.signal(signum,handler)
            raise
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    try:
        yield pending
    finally:
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
        try:
            for signum,handler in previous.items():
                signal.signal(signum,handler)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK,mask)


def capture_maps(pid,expected,executable,selected_library,maps_path,state):
    """Sample loaded files and leader affinity, without loading any library."""
    try:
        raw=Path(f'/proc/{pid}/maps').read_text()
    except FileNotFoundError:
        return None
    # Preserve first, nonempty and incomplete snapshots independently. A final
    # empty zombie read must not erase the last live mapping evidence.
    snapshots=maps_path.with_suffix('.snapshots')
    snapshots.mkdir(exist_ok=True)
    state['successful_reads']=state.get('successful_reads',0)+1
    digest=hashlib.sha256(raw.encode()).hexdigest()
    state['last_read_sha256']=digest
    state['last_read_bytes']=len(raw.encode())
    state['last_read_unix']=time.time()
    (snapshots/'last-read.txt').write_text(raw)
    if not (snapshots/'first-read.txt').exists():
        (snapshots/'first-read.txt').write_text(raw)
    if raw:
        maps_path.write_text(raw)
        (snapshots/'last-nonempty.txt').write_text(raw)
        if not (snapshots/'first-nonempty.txt').exists():
            (snapshots/'first-nonempty.txt').write_text(raw)
    def incomplete(reason):
        state['last_incomplete_reason']=reason
        (snapshots/'last-incomplete.txt').write_text(raw)
        if not (snapshots/'first-incomplete.txt').exists():
            (snapshots/'first-incomplete.txt').write_text(raw)
        return None
    def mismatch(reason):
        state['mismatch_reason']=reason
        (snapshots/'mismatch.txt').write_text(raw)
        raise RuntimeError(reason)
    mapped={}
    for line in raw.splitlines():
        columns=line.split(None,5)
        if len(columns)!=6 or not columns[5].startswith('/'):
            continue
        path=columns[5]
        if path.endswith(' (deleted)'):
            mismatch('A file mapping was deleted during replay: '+path)
        mapped.setdefault(str(Path(path).resolve()),[]).append(line)
    bridges=[name for name in mapped if Path(name).name=='libmadloop_tirust.so']
    if executable in mapped and bridges and bridges!=[selected_library]:
        mismatch('Wrong or multiple actual TIRust mappings: '+repr(bridges))
    if selected_library not in mapped or executable not in mapped or set(expected)-set(mapped):
        return incomplete({'missing_expected_paths':sorted(set(expected)-set(mapped)),
                           'executable_present':executable in mapped,'selected_bridge_present':selected_library in mapped})
    status=Path(f'/proc/{pid}/status').read_text()
    affinity=[line.split(':',1)[1].strip() for line in status.splitlines() if line.startswith('Cpus_allowed_list:')]
    if affinity!=['12']:
        mismatch('Native leader affinity differs from CPU12.')
    elves=set()
    for name in mapped:
        with Path(name).open('rb') as stream:
            if stream.read(4)==b'\x7fELF':
                elves.add(name)
    if elves!=set(expected)|{executable}:
        mismatch('Actual mapped ELF set differs from declared runtime plus executable.')
    (snapshots/'complete.txt').write_text(raw)
    state['complete_snapshot_sha256']=digest
    return {'pid':pid,'maps_text':raw,'maps_sha256':hashlib.sha256(raw.encode()).hexdigest(),
            'mapped_paths':sorted(mapped),'mapped_elf_paths':sorted(elves),'expected_runtime_sha256':expected,
            'executable':executable,'selected_library':selected_library,
            'cpu_allowed_list':affinity[0],'capture_unix':time.time(),
            'observation_scope':'One successful mapping and leader-affinity snapshot, not continuous monitoring.'}


def run_captured(command,log,*,cwd,env,expected_runtime,executable,
                 selected_library,maps_path,progress,seconds=180,capture_required=True):
    """Observe exits without reaping, then finish ownership under deferred signals."""
    if signal.getsignal(signal.SIGCHLD)!=signal.SIG_DFL:
        raise RuntimeError('Default SIGCHLD disposition is required to retain child ownership.')
    process=None
    owned=False
    start=time.monotonic()
    mapping=None if capture_required else {}
    capture_state={}
    if capture_required:
        progress['mapping_capture']=capture_state

    def finish_owned():
        nonlocal owned
        with deferred_signals() as pending:
            if not owned:
                return process.returncode,pending
            # waitid(WNOWAIT) never reaps. Even after exit the zombie reserves
            # the leader PID, so its group identity cannot have been recycled.
            # Also kill any surviving members on normal leader exit.
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            code=process.wait()
            # Signals remain deferred across reaping and this ownership update.
            owned=False
            progress['process_reaped']=True
            progress['returncode']=code
            return code,pending

    try:
        with deferred_signals() as pending:
            process=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            owned=True
            progress.update({'process_spawned':True,'pid':process.pid,'spawn_unix':time.time()})
        if pending:
            raise InterruptedError('Interrupted during owned process creation: '+repr(pending))
        while True:
            remaining=seconds-(time.monotonic()-start)
            if remaining<=0:
                raise subprocess.TimeoutExpired(command,seconds)
            if mapping is None:
                mapping=capture_maps(process.pid,expected_runtime,executable,selected_library,maps_path,capture_state)
            status=os.waitid(os.P_PID,process.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
            # Mapping I/O and exit observation are included in the deadline.
            # Never accept an observed exit after those operations cross it.
            remaining=seconds-(time.monotonic()-start)
            if remaining<=0:
                raise subprocess.TimeoutExpired(command,seconds)
            if status is not None:
                if status.si_pid!=process.pid:
                    raise RuntimeError('Unexpected child exit identity.')
                progress['exit_observed_unix']=time.time()
                progress['process_wall_seconds']=time.monotonic()-start
                if mapping is None:
                    raise RuntimeError('Process exited before complete actual mapping capture.')
                break
            time.sleep(min(0.01,remaining))
        code,pending=finish_owned()
        if pending:
            raise InterruptedError('Interrupted during final group cleanup/reaping: '+repr(pending))
    except BaseException:
        if process is not None and owned:
            # Repeated handled signals are deferred through kill, wait, and
            # ownership release, so the caller cannot unlock a live child.
            finish_owned()
        raise
    # No active child ownership remains here. A hash/I/O failure must never
    # signal the now-reusable numeric PID or process-group identifier.
    actual={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in expected_runtime}
    if actual!=expected_runtime:
        raise RuntimeError('An actual mapped runtime file differs from its prelaunch hash.')
    mapping['mapped_files_sha256_after_exit']=actual
    mapping['process_wall_seconds']=progress['process_wall_seconds']
    mapping['process_exit_observed_unix']=progress['exit_observed_unix']
    return code,mapping


def run_plain_capped(command,log,*,cwd,env,progress,seconds=30):
    """Use the same owned-group cleanup for the read-only ldd helper."""
    code,_=run_captured(command,log,cwd=cwd,env=env,expected_runtime={},
                        executable='',selected_library='',maps_path=None,
                        progress=progress,seconds=seconds,capture_required=False)
    return code
