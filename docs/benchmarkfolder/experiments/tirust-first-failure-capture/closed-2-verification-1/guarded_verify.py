#!/usr/bin/env python3
"""Verify a closed compact packet after relocation with external I/O denied."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import types


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verifier', type=Path, required=True)
    parser.add_argument('--verifier-sha256', required=True)
    parser.add_argument('--capsule', type=Path, required=True)
    parser.add_argument('--index-sha256', required=True)
    args = parser.parse_args()
    namespace = Path(__file__).resolve().parent
    relocated = namespace / 'relocated'
    output = namespace / 'proof.json'
    if relocated.exists() or output.exists():
        raise ValueError('Fresh verification destination required')
    blob = args.verifier.read_bytes()
    if sha(blob) != args.verifier_sha256 or sha((args.capsule / 'index.json').read_bytes()) != args.index_sha256:
        raise ValueError('Verifier or packet differs from admission')
    module = types.ModuleType('guarded_compact_verifier')
    module.__file__ = str(args.verifier.resolve())
    exec(compile(blob, module.__file__, 'exec'), module.__dict__)
    original = module.verify(args.capsule)
    shutil.copytree(args.capsule, relocated, symlinks=True)
    if sha((relocated / 'index.json').read_bytes()) != args.index_sha256:
        raise ValueError('Relocated packet differs')
    # Warm Python's lazy standard-library imports before denying external I/O.
    warm = module.verify(relocated)
    physical = {str(path.relative_to(relocated)): path.read_bytes()
                for path in relocated.rglob('*') if path.is_file()}
    denied = []
    enabled = True
    root = relocated.resolve()
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    mutations = {'os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.symlink',
                 'os.link', 'os.chmod', 'os.chown', 'os.utime', 'os.truncate', 'os.chdir'}

    def inside(path):
        return isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve().is_relative_to(root)

    def audit(event, arguments):
        if not enabled:
            return
        bad = event in mutations or event.startswith(('subprocess.', 'os.exec', 'os.spawn', 'os.fork'))
        bad = bad or event == 'ctypes.dlopen'
        if event in ('os.listdir', 'os.scandir'):
            bad = bad or not inside(arguments[0])
        if event == 'open':
            path, mode, flags = arguments[:3]
            bad = bad or not inside(path)
            bad = bad or (isinstance(mode, str) and any(char in mode for char in 'wax+'))
            bad = bad or (isinstance(flags, int) and bool(flags & write_flags))
        if bad:
            denied.append({'event': event, 'path': str(arguments[0]) if arguments else None})
            raise PermissionError('Portable verification denied ' + event)

    original_flock = fcntl.flock
    def no_lock(*args, **kwargs):
        denied.append({'event': 'fcntl.flock'})
        raise PermissionError('Portable verification denied numerical lock')
    sys.addaudithook(audit)
    fcntl.flock = no_lock
    try:
        guarded = module.verify(relocated)
        def reader(name):
            return physical[name]
        reader.paths = set(physical)
        callback = module.verify(read_bytes=reader)
        if not (original == warm == guarded == callback):
            raise ValueError('Original, relocated and callback verification disagree')
    finally:
        enabled = False
        fcntl.flock = original_flock
    proof = {'verified': True, 'runner_sha256': sha(Path(__file__).read_bytes()),
             'verifier_sha256': args.verifier_sha256, 'index_sha256': args.index_sha256,
             'original_and_relocated_equal': True, 'callback_equal': True,
             'denied_attempts': denied, 'physical_payload_count': len(physical) - 1,
             'compiler_invocations': 0, 'loader_invocations': 0, 'amplitude_evaluations': 0,
             'standalone_tensor_evaluations': 0, 'numerical_lock_attempts': 0,
             'verification': guarded}
    with output.open('x') as stream:
        json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(sha(output.read_bytes()))


if __name__ == '__main__':
    main()
