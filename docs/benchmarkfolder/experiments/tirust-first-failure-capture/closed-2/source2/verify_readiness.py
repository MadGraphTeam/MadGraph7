#!/usr/bin/env python3
"""Run additive admission against actual closed inputs, denying side effects."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Readiness proof cannot overwrite history')
    module_path = Path(__file__).with_name('replay_admission.py')
    spec = importlib.util.spec_from_file_location('additive_replay_readiness', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    enabled = True
    denied = []
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    mutations = {'os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.symlink',
                 'os.link', 'os.chmod', 'os.chown', 'os.utime', 'os.truncate'}

    def deny(event, arguments):
        if not enabled:
            return
        bad = event in mutations or event.startswith(('subprocess.', 'os.exec', 'os.spawn', 'os.fork'))
        bad = bad or event == 'ctypes.dlopen'
        if event == 'open':
            mode, flags = arguments[1:3]
            bad = bad or (isinstance(mode, str) and any(char in mode for char in 'wax+'))
            bad = bad or (isinstance(flags, int) and bool(flags & write_flags))
        if bad:
            denied.append(event)
            raise PermissionError('Read-only admission denied ' + event)

    original_flock = fcntl.flock
    def no_lock(*args, **kwargs):
        denied.append('fcntl.flock')
        raise PermissionError('Read-only admission denied numerical lock')
    sys.addaudithook(deny)
    fcntl.flock = no_lock
    try:
        result = module.readonly_inputs()
        if module.WORK.exists():
            raise ValueError('Fresh standalone replay work must remain absent')
        proof = {'verified': True, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'controller_sha256': hashlib.sha256(module_path.read_bytes()).hexdigest(),
                 'admission': result, 'denied_attempts': denied, 'work2_absent': True,
                 'commands_executed': 0, 'compiler_invocations': 0, 'loader_invocations': 0,
                 'amplitude_evaluations': 0, 'standalone_tensor_evaluations': 0}
    finally:
        enabled = False
        fcntl.flock = original_flock
    with args.output.open('x') as stream:
        json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
