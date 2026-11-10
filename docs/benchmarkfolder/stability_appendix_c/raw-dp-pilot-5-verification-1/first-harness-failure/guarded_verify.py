"""Relocate a sealed capsule, then forbid external byte reads and mutations."""
import argparse
import ctypes
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import types
import zlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--expected-index', required=True)
    args = parser.parse_args()
    home = Path(__file__).resolve().parent
    original = args.source.resolve()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(original / 'index.json') == args.expected_index
    relocated = home / 'relocated-capsule'
    shutil.copytree(original, relocated)
    assert digest(relocated / 'index.json') == args.expected_index
    source = (relocated / 'verify.py').read_bytes()
    active = [True]
    inside_open = [False]
    stats = {'byte_reads': 0, 'directory_opens': 0, 'mutations': 0,
             'native_events': 0, 'external_reads': 0}
    old_open = os.open
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def check(path, flags=0, *, directory=False):
        if isinstance(path, int):
            path = os.readlink('/proc/self/fd/' + str(path))
        target = Path(os.fsdecode(path)).absolute().resolve()
        if flags & write_flags:
            stats['mutations'] += 1
            raise PermissionError('Verifier attempted a write')
        if directory:
            if target == relocated or relocated in target.parents or target in relocated.parents:
                stats['directory_opens'] += 1
                return
        elif target == relocated or relocated in target.parents:
            stats['byte_reads'] += 1
            return
        stats['external_reads'] += 1
        raise PermissionError('Verifier attempted external byte read: ' + str(target))

    def guarded_open(path, flags, mode=0o777, *, dir_fd=None):
        if not active[0]:
            return old_open(path, flags, mode, dir_fd=dir_fd)
        target = Path(os.fsdecode(path))
        if not target.is_absolute() and dir_fd is not None:
            target = Path(os.readlink('/proc/self/fd/' + str(dir_fd))) / target
        check(target, flags, directory=bool(flags & os.O_DIRECTORY))
        inside_open[0] = True
        try:
            return old_open(path, flags, mode, dir_fd=dir_fd)
        finally:
            inside_open[0] = False

    mutation_events = {'os.remove', 'os.rmdir', 'os.rename', 'os.mkdir', 'os.link',
                       'os.symlink', 'os.chmod', 'os.chown', 'os.utime', 'os.truncate'}
    native_events = {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.fork',
                     'os.forkpty', 'os.exec', 'ctypes.dlopen'}

    def audit(event, values):
        if not active[0]:
            return
        if event in mutation_events:
            stats['mutations'] += 1
            raise PermissionError('Verifier attempted mutation: ' + event)
        if event in native_events:
            stats['native_events'] += 1
            raise PermissionError('Verifier attempted native execution/load: ' + event)
        if event == 'open' and not inside_open[0]:
            check(values[0], values[2])

    os.open = guarded_open
    sys.addaudithook(audit)
    negative_probes = {}
    for label, probe in (
        ('outside_read', lambda: Path('/etc/passwd').read_bytes()),
        ('write', lambda: (relocated / 'forbidden').write_bytes(b'x')),
        ('child', lambda: sys.audit('subprocess.Popen', '/never-executed', [], None, None)),
        ('dlopen', lambda: sys.audit('ctypes.dlopen', '/never-loaded')),
    ):
        try:
            probe()
        except PermissionError:
            negative_probes[label] = 'rejected-before-operation'
        else:
            raise AssertionError('Guard probe passed: ' + label)
    stats.update(dict.fromkeys(stats, 0))
    started = time.monotonic()
    module = types.ModuleType('relocated_closed_raw_verifier')
    module.__file__ = str(relocated / 'verify.py')
    try:
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        result = module.verify(relocated)
    finally:
        active[0] = False
        os.open = old_open
    assert result['verified'] and result['index_sha256'] == args.expected_index
    assert all(stats[key] == 0 for key in ('mutations', 'native_events', 'external_reads'))
    assert digest(original / 'index.json') == args.expected_index
    proof = {'schema': 'madloop-raw-stability-relocated-verification-v1',
             'runner_sha256': digest(Path(__file__)),
             'data_index_sha256': args.expected_index,
             'verified': True, 'read_only_native_guard': stats,
             'negative_probes': negative_probes,
             'elapsed_seconds': time.monotonic() - started,
             'summary': result['summary'],
             'scope': 'capsule-local relocated byte replay; no original host paths opened; directory handles on relocation ancestors only'}
    (home / 'proof.json').write_text(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'verified': True, 'proof_sha256': digest(home / 'proof.json'),
                      'data_index_sha256': args.expected_index, 'guard': stats}, indent=2))


if __name__ == '__main__':
    main()
