"""Own one bounded process group and retain its actual ELF mappings."""
from pathlib import Path
import hashlib
import os
import signal
import subprocess
import time


def capture_maps(pid, expected, executable, selected_library, maps_path):
    """Return None until the selected executable and bridge are mapped."""
    try:
        raw=Path(f'/proc/{pid}/maps').read_text()
    except FileNotFoundError:
        return None
    mapped={}
    for line in raw.splitlines():
        columns=line.split(None,5)
        if len(columns)!=6 or not columns[5].startswith('/'):
            continue
        path=columns[5]
        if path.endswith(' (deleted)'):
            raise RuntimeError('A file mapping was deleted during replay: '+path)
        mapped.setdefault(str(Path(path).resolve()),[]).append(line)
    bridges=[name for name in mapped if Path(name).name=='libmadloop_tirust.so']
    if executable in mapped and bridges:
        maps_path.write_text(raw)
        if bridges!=[selected_library]:
            raise RuntimeError('Wrong or multiple actual TIRust mappings: '+repr(bridges))
    if selected_library not in mapped or executable not in mapped:
        return None
    missing=set(expected)-set(mapped)
    if missing:
        # The loader may still be mapping the other dependencies.
        return None
    status=Path(f'/proc/{pid}/status').read_text()
    affinity=[line.split(':',1)[1].strip() for line in status.splitlines() if line.startswith('Cpus_allowed_list:')]
    if affinity!=['11']:
        raise RuntimeError('Native CPU affinity differs from CPU11.')
    elves=set()
    for name in mapped:
        with Path(name).open('rb') as stream:
            if stream.read(4)==b'\x7fELF':
                elves.add(name)
    if elves!=set(expected)|{executable}:
        raise RuntimeError('Actual mapped ELF set differs from the declared runtime plus executable.')
    return {'pid':pid,'maps_text':raw,'maps_sha256':hashlib.sha256(raw.encode()).hexdigest(),
            'mapped_paths':sorted(mapped),'mapped_elf_paths':sorted(elves),'expected_runtime_sha256':expected,
            'executable':executable,'selected_library':selected_library,
            'cpu_allowed_list':affinity[0],'capture_unix':time.time()}


def run_captured(command, log, *, cwd, env, expected_runtime, executable,
                 selected_library, maps_path, seconds=180):
    """Retain the reviewed owned-handle spawn/kill/reap protections."""
    process=None
    previous,pending={},[]
    start=time.monotonic()
    mapping=None

    def defer(signum,frame):
        pending.append(signum)

    try:
        # Same protected ownership window as frozen common.run_capped.
        # Caught handlers reset on exec; native signals/traps are not masked.
        try:
            for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
                previous[signum]=signal.signal(signum,defer)
            process=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        finally:
            for signum,handler in previous.items():
                signal.signal(signum,handler)
        if pending:
            raise InterruptedError('Interrupted during process creation: '+repr(pending))
        while mapping is None:
            remaining=seconds-(time.monotonic()-start)
            if remaining<=0:
                raise subprocess.TimeoutExpired(command,seconds)
            mapping=capture_maps(process.pid,expected_runtime,executable,selected_library,maps_path)
            if mapping is not None:
                # Capture before waiting, so even a subsequent failure retains
                # the actual loader evidence. This only writes from the parent.
                maps_path.write_text(mapping['maps_text'])
                break
            if process.poll() is not None:
                raise RuntimeError('Native process exited before complete actual mapping capture.')
            time.sleep(min(0.01,remaining))
        remaining=seconds-(time.monotonic()-start)
        if remaining<=0:
            raise subprocess.TimeoutExpired(command,seconds)
        code=process.wait(timeout=remaining)
        mapping['process_wall_seconds']=time.monotonic()-start
        mapping['process_exit_observed_unix']=time.time()
        # Hash the captured actual paths after the process exits, keeping large
        # file reads outside its deadline-sensitive supervision loop. The same
        # paths were hash-checked before launch and are checked again by caller.
        actual={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in expected_runtime}
        if actual!=expected_runtime:
            raise RuntimeError('An actual mapped runtime file differs from its prelaunch hash.')
        mapping['mapped_files_sha256_after_exit']=actual
        return code,mapping
    except BaseException:
        if process is not None:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
