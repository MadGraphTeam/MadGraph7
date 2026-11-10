#!/usr/bin/env python3
"""Guarded current-source closed-case preflight, without compilation/native I/O."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Fresh readiness receipt required')
    here = Path(__file__).resolve().parent
    blob = (here / 'controller.py').read_bytes()
    source_index = json.loads((here / 'source-index.json').read_bytes())
    checksum = hashlib.sha256(blob).hexdigest()
    if checksum != source_index['files']['controller.py']['sha256']:
        raise ValueError('Current controller differs from source seal')
    controller = types.ModuleType('guarded_current_qcd3_controller')
    controller.__file__ = str(here / 'controller.py')
    exec(compile(blob, controller.__file__, 'exec'), controller.__dict__)
    enabled = True; denied = []; read_opens = 0
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    mutations = {'os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.symlink',
                 'os.link', 'os.chmod', 'os.chown', 'os.utime', 'os.truncate', 'os.chdir'}
    def audit(event, arguments):
        nonlocal read_opens
        if not enabled:
            return
        bad = event in mutations or event.startswith(('subprocess.', 'os.exec', 'os.spawn', 'os.fork'))
        bad = bad or event == 'ctypes.dlopen'
        if event == 'open':
            mode, flags = arguments[1:3]
            bad = bad or (isinstance(mode, str) and any(char in mode for char in 'wax+'))
            bad = bad or (isinstance(flags, int) and bool(flags & write_flags))
            if not bad:
                read_opens += 1
        if bad:
            denied.append(event)
            raise PermissionError('Read-only current pilot admission denied ' + event)
    original_flock = fcntl.flock
    def no_lock(*args, **kwargs):
        denied.append('fcntl.flock')
        raise PermissionError('Read-only current pilot admission denied numerical lock')
    sys.addaudithook(audit); fcntl.flock = no_lock
    try:
        result = controller.readonly()
        if controller.WORK.exists() or controller.WORK.is_symlink():
            raise ValueError('Read-only preflight wrote/encountered a work namespace')
        proof = {'verified': True, 'controller_sha256': checksum,
                 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'source_index_sha256': result['source_index_sha256'], 'admission': result,
                 'read_opens': read_opens, 'denied_attempts': denied, 'work_namespace_absent': True,
                 'compiler_invocations': 0, 'loader_invocations': 0, 'point_generation_calls': 0,
                 'amplitude_evaluations': 0, 'numerical_lock_attempts': 0}
    finally:
        enabled = False; fcntl.flock = original_flock
    with args.output.open('x') as stream:
        json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
