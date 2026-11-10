"""Read-only source/pin helpers for the private QCD topology replay."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys

ROOT = Path.cwd().resolve()
HERE = ROOT / 'docs/benchmarkfolder/experiments/qcd-topology-rescue-native'
DESIGN = ROOT / 'docs/benchmarkfolder/experiments/qcd-topology-rescue-prototype'
OLD = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer-native'
PROC = ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg'
SP = PROC / 'SubProcesses/P0_uux_zgg'
WORKBASE = ROOT / '.tirust-build/qcd-topology-rescue-native/attempt1'
OUTBASE = HERE / 'attempt1'
ARMS = ('topology-b', 'original-c', 'topology-c')
DESIGN_SHA = '0af03d980acdac032abcb767713bf5906fa7fa53cf8b3c57da75b764281e3015'
OLD_SHA = '6a2e48f62ee48ebd2a892da18f797249e686be660225e9d05c66415f98e8f5ac'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def add_pin(mapping, path, digest):
    name = rel(Path(path))
    require(name not in mapping or mapping[name] == digest, 'Conflicting pin: ' + name)
    mapping[name] = digest


def indexed_pins(directory, expected=None):
    index = directory / 'evidence-index.json'
    if expected:
        require(sha(index) == expected, 'Frozen index changed: ' + str(index))
    entries = json.loads(index.read_text())['files']
    result = {rel(index): sha(index)}
    for name, entry in entries.items():
        path = directory / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe indexed path.')
        require(not path.is_symlink() and path.is_file(), 'Missing/linked indexed path: ' + str(path))
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                'Indexed bytes changed: ' + str(path))
        add_pin(result, path, entry['sha256'])
    return result


def validate_pins(mapping):
    for name, expected in mapping.items():
        require(sha(ROOT / name) == expected, 'Protected input changed: ' + name)


def pins():
    result = indexed_pins(DESIGN, DESIGN_SHA)
    for name, digest in indexed_pins(OLD, OLD_SHA).items():
        add_pin(result, ROOT / name, digest)
    protocol = json.loads((DESIGN / 'protocol.json').read_text())
    for entry in protocol['frozen_inputs'].values():
        path = ROOT / entry['path']
        require(path.stat().st_size == entry['bytes'], 'Frozen input size changed.')
        add_pin(result, path, entry['sha256'])
    for name, digest in protocol['historical_protected_inputs'].items():
        add_pin(result, ROOT / name, digest)
    validate_pins(result)
    return result


def controller_pins():
    index = HERE / 'source-index.json'
    entries = json.loads(index.read_text())['files']
    result = {rel(index): sha(index)}
    for name, entry in entries.items():
        path = HERE / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not path.is_symlink(),
                'Unsafe controller path.')
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                'Controller source changed: ' + name)
        add_pin(result, path, entry['sha256'])
    return result


def frozen_verify():
    done = subprocess.run([sys.executable, '-I', '-B', str(DESIGN / 'verify.py'), '--live'],
                          cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    require(done.returncode == 0, 'Frozen source verifier failed:\n' + done.stdout)
    return done.stdout


def run_capped(command, log, env=None, seconds=180):
    """One process group, killed at the wall cap; no retries or trap changes."""
    process = None
    previous, pending = {}, []

    def defer(signum, frame):
        pending.append(signum)

    try:
        # Defer Python interruptions until the new process handle is owned.
        # Otherwise an exception between Popen returning and assignment can
        # orphan a just-spawned native process. Caught signal handlers reset on
        # exec, so this does not mask signals in the native child.
        try:
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                previous[signum] = signal.signal(signum, defer)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        if pending:
            raise InterruptedError('Interrupted during process creation: ' + repr(pending))
        return process.wait(timeout=seconds)
    except BaseException:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
