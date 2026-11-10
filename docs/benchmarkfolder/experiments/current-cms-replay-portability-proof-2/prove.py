#!/usr/bin/env python3
"""Copy sealed CMS2 to /tmp and prove callback-only verification under I/O gates."""
import builtins
from contextlib import ExitStack
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
CAPSULE = HERE.with_name('current-cms-replay-retention-2')
PINS = {
    'evidence-index.json': 'aa39809c5d6cde25367454bb1fd9c3500b9aaa9b14b976fc1ff42bb0783714d3',
    'verify_evidence.py': '1cf318978a06d1afedff084d9d1e727a8b25610b82d163b41220a5e5906eb0b8',
    'summary.json': 'a84cf40930f9ef8d824ab993df6a1d4eb6f0be8694306d828777d6b8a0cb5cce',
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def snapshot(root):
    result = {}
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink() and (path.is_file() or path.is_dir()), str(path)
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


class Denied(RuntimeError):
    pass


class MemoryReader:
    def __init__(self, files):
        self.files = files
        self.paths = tuple(sorted(files))
        self.calls = []

    def __call__(self, name):
        assert isinstance(name, str) and '\\' not in name and '\x00' not in name
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts and str(path) == name
        self.calls.append(name)
        return self.files[name]


def main():
    if sys.flags.optimize != 0:
        raise RuntimeError('Run without Python optimization; proof assertions must be enabled')
    output = HERE / 'receipt.json'
    assert not output.exists(), 'Append-only proof; use a new proof namespace for another run'
    started = time.monotonic()
    source = snapshot(CAPSULE)
    for name, checksum in PINS.items():
        assert digest(source[name]) == checksum, 'Source capsule pin differs: ' + name
    temporary_root = Path(tempfile.mkdtemp(prefix='cms2-portability-', dir='/tmp'))
    copied = temporary_root / 'capsule'
    shutil.copytree(CAPSULE, copied, symlinks=True)
    files = snapshot(copied)
    assert files == source, 'Copied capsule differs'
    reader = MemoryReader(files)

    # Load this exact reviewed pure verifier from the relocated copy. All its
    # stdlib dependencies are preloaded before access gates become active.
    namespace = {'__name__': '_copied_cms2_portable_verifier', '__file__': str(copied / 'verify_evidence.py')}
    exec(compile(files['verify_evidence.py'], str(copied / 'verify_evidence.py'), 'exec'), namespace)

    state = {'active': False, 'phase': 'probes'}
    attempts = []
    patched = []

    def denied(operation):
        def stop(*args, **kwargs):
            attempts.append({'phase': state['phase'], 'operation': operation})
            raise Denied(operation)
        return stop

    def audit(event, args):
        if not state['active']:
            return
        filesystem = (event == 'open' or event in ('os.listdir', 'os.scandir', 'os.chdir', 'os.fchdir',
                      'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink',
                      'os.truncate', 'os.chmod', 'os.chown', 'os.utime'))
        process = event in ('subprocess.Popen', 'os.system', 'os.fork', 'os.forkpty', 'os.exec',
                            'os.posix_spawn', 'os.spawn') or event.startswith('os.exec')
        dynamic_library = event.startswith('ctypes.dlopen') or event.startswith('ctypes.dlsym')
        if filesystem or process or dynamic_library:
            denied('audit:' + event)()

    sys.addaudithook(audit)
    old_cwd = Path.cwd()
    os.chdir(copied)
    probes = {}
    verified = None
    try:
        with ExitStack() as stack:
            groups = [
                (builtins, ('open',)), (io, ('open', 'open_code')),
                (os, ('open', 'read', 'write', 'stat', 'lstat', 'listdir', 'scandir', 'readlink', 'access',
                      'getcwd', 'chdir', 'fchdir', 'mkdir', 'makedirs', 'remove', 'unlink', 'rmdir',
                      'rename', 'replace', 'link', 'symlink', 'truncate', 'chmod', 'chown', 'utime',
                      'system', 'fork', 'forkpty', 'execv', 'execve', 'execl', 'execle', 'execlp',
                      'execlpe', 'execvp', 'execvpe', 'spawnl', 'spawnle', 'spawnlp', 'spawnlpe',
                      'spawnv', 'spawnve', 'spawnvp', 'spawnvpe', 'posix_spawn', 'posix_spawnp')),
                (Path, ('open', 'read_bytes', 'read_text', 'write_bytes', 'write_text', 'stat', 'lstat',
                        'iterdir', 'glob', 'rglob', 'resolve')),
                (subprocess, ('Popen', 'run', 'call', 'check_call', 'check_output')),
                (ctypes, ('CDLL', 'PyDLL', '_dlopen')),
                (ctypes.cdll, ('LoadLibrary',)), (ctypes.pydll, ('LoadLibrary',)),
            ]
            for owner, names in groups:
                for name in names:
                    if hasattr(owner, name):
                        label = getattr(owner, '__name__', type(owner).__name__) + '.' + name
                        stack.enter_context(patch.object(owner, name, denied(label)))
                        patched.append(label)
            state['active'] = True
            operations = {
                'filesystem_builtin_open': lambda: builtins.open('/etc/passwd', 'rb'),
                'filesystem_path_read': lambda: Path('/etc/passwd').read_bytes(),
                'filesystem_os_open': lambda: os.open('/etc/passwd', os.O_RDONLY),
                'filesystem_stat': lambda: os.stat('/etc/passwd'),
                'filesystem_directory': lambda: os.listdir('/'),
                'subprocess': lambda: subprocess.Popen(['/bin/true']),
                'shell': lambda: os.system('/bin/true'),
                'CDLL': lambda: ctypes.CDLL(None),
                'PyDLL': lambda: ctypes.PyDLL(None),
                'library_loader': lambda: ctypes.cdll.LoadLibrary('/tmp/not-opened.so'),
                'audit_open': lambda: sys.audit('open', '/etc/passwd', 'r', 0),
                'audit_subprocess': lambda: sys.audit('subprocess.Popen', '/bin/true', [], None, None),
                'audit_dlopen': lambda: sys.audit('ctypes.dlopen', '/tmp/not-opened.so'),
            }
            for name, operation in operations.items():
                try:
                    operation()
                except Denied:
                    probes[name] = True
                else:
                    raise AssertionError('Access gate did not reject probe: ' + name)
            state['phase'] = 'verify'
            verified = namespace['verify'](root=copied, read_bytes=reader)
            assert verified['verified'] and verified['index_sha256'] == PINS['evidence-index.json']
            assert encoded(verified['summary']) == files['summary.json']
            assert set(reader.calls) == set(reader.paths), 'Incomplete callback inventory consumption'
            assert not any(item['phase'] == 'verify' for item in attempts), 'Verifier attempted external access'
            state['active'] = False
    finally:
        state['active'] = False
        os.chdir(old_cwd)

    assert snapshot(copied) == files and snapshot(CAPSULE) == source, 'Proof changed copied/source capsule'
    file_hashes = {name: digest(raw) for name, raw in files.items()}
    receipt = {
        'schema_version': 1, 'kind': 'copied-capsule-callback-portability-proof', 'status': 'passed',
        'source_capsule': str(CAPSULE), 'temporary_copy': str(copied), 'copy_preserved_for_review': True,
        'pins': PINS, 'copied_files': len(files), 'copied_bytes': sum(map(len, files.values())),
        'copied_inventory_sha256': digest(encoded(file_hashes)),
        'source_and_copy_identical_before_after': True,
        'compiled_verifier_filename': namespace['__file__'], 'verification_cwd': str(copied),
        'callback_paths': len(reader.paths), 'callback_reads': len(reader.calls),
        'callback_unique_reads': len(set(reader.calls)), 'callback_is_complete_io_boundary': True,
        'patched_entrypoints': patched, 'audit_hook_active_during_verification': True,
        'negative_probes': probes, 'blocked_probe_attempts': attempts,
        'external_access_attempts_during_verify': 0, 'new_subprocesses': 0, 'new_scientific_native_executions': 0,
        'verification': {'verified': True, 'index_sha256': verified['index_sha256'],
                         'summary_sha256': digest(encoded(verified['summary'])),
                         'overall_status': verified['summary']['overall_status'],
                         'completed_requests': verified['summary']['completed_requests'],
                         'full_final_streams': verified['summary']['full_final_streams']},
        'scope': 'Relocated pure callback verification after stdlib initialization; no original qualification/workspace paths or live binaries read',
        'gate_scope': 'Reviewed pinned Python verifier under Python entrypoint/audit gates; not a general hostile-code OS sandbox',
        'python_optimization': sys.flags.optimize,
        'elapsed_seconds': time.monotonic() - started,
        'harness_sha256': digest(Path(__file__).read_bytes()),
    }
    output.write_bytes(encoded(receipt))
    print(json.dumps({'status': receipt['status'], 'temporary_copy': str(copied),
                      'callback_reads': len(reader.calls), 'negative_probes': len(probes),
                      'receipt_sha256': digest(encoded(receipt))}, sort_keys=True))


if __name__ == '__main__':
    main()
