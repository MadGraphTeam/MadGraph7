"""Guarded original, isolated-copy and callback replay; zero numerical work."""
import argparse
import ast
import bisect
import collections
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import types
import zlib

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
CAPSULE = ROOT / 'docs/benchmarkfolder/stability_appendix_c/current-source-qcd3-raw-1'
INDEX_SHA = 'b2589cfcd71f9940a6dea98376be16b93e2b056aae4596d86a6fd6d5747dce30'
SOURCE_SHA = 'cff6f1858787fc7ac2719b0a24463e74fe2aa4494a3a9368b850f668c04d0b92'
VERIFY_SHA = '3acab391e5849b500dc1e2fc1e0de022a0f50c6ea5c5abe255851e0ee52f1218'


def sha(blob): return hashlib.sha256(blob).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)


def main():
    own = Path(__file__).resolve(); work = own.parent
    require(not (work / 'proof.json').exists() and not (work / 'isolated').exists(), 'Fresh proof required')
    require(sha((CAPSULE / 'index.json').read_bytes()) == INDEX_SHA, 'Exact admitted index differs')
    code = (CAPSULE / 'verify.py').read_bytes(); require(sha(code) == VERIFY_SHA, 'Verifier differs')
    index = json.loads((CAPSULE / 'index.json').read_bytes())
    isolated = work / 'isolated'; shutil.copytree(CAPSULE, isolated)
    module = types.ModuleType('guarded_current_source_verifier'); module.__file__ = str(CAPSULE / 'verify.py')
    exec(compile(code, module.__file__, 'exec'), module.__dict__)
    allowed_stdlib = tuple(Path(sysconfig.get_paths()[key]).resolve() for key in ('stdlib', 'platstdlib'))
    state = {'active': False, 'root': None, 'reads': 0, 'denied': [], 'native_attempts': 0}
    def inside(path, directory):
        return path == directory or directory in path.parents
    def audit(event, args):
        if not state['active']: return
        forbidden = ('subprocess.', 'ctypes.dlopen', 'os.system', 'os.exec', 'os.fork', 'os.posix_spawn', 'fcntl.flock', 'fcntl.lockf')
        if any(event.startswith(prefix) for prefix in forbidden):
            state['native_attempts'] += 1; state['denied'].append(event); raise PermissionError(event)
        if event == 'open':
            name, mode, flags = args
            if isinstance(name, int):
                state['denied'].append('descriptor-open'); raise PermissionError('Unexpected descriptor')
            path = Path(os.path.abspath(os.fsdecode(name)))
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            allowed = inside(path, state['root']) or any(inside(path, directory) for directory in allowed_stdlib)
            if writing or not allowed:
                state['denied'].append(str(path)); raise PermissionError('Unadmitted open: ' + str(path))
            state['reads'] += 1
        if event in ('os.scandir', 'os.listdir'):
            path = Path(os.path.abspath(os.fsdecode(args[0])))
            if not inside(path, state['root']):
                state['denied'].append(str(path)); raise PermissionError('Unadmitted traversal')
        if event.startswith(('os.remove', 'os.rename', 'os.rmdir', 'os.mkdir', 'os.link', 'os.symlink', 'os.chmod', 'os.utime', 'shutil.')):
            state['denied'].append(event); raise PermissionError('Write forbidden')
    sys.addaudithook(audit)
    results = []
    for label, directory in [('original', CAPSULE), ('isolated-copy', isolated)]:
        state.update(active=True, root=directory, reads=0)
        try: result = module.verify(directory)
        finally: state['active'] = False
        require(result['verified'] and result['index_sha256'] == INDEX_SHA, 'Replay failed')
        results.append({'mode': label, 'read_opens': state['reads'], 'result': result})
    mapping = {name: (isolated / name).read_bytes() for name in (*index['files'], 'index.json')}
    def reader(name): return mapping[name]
    reader.paths = set(mapping)
    state.update(active=True, root=work / 'no-filesystem-access', reads=0)
    try: result = module.verify(read_bytes=reader)
    finally: state['active'] = False
    require(state['reads'] == 0 and result == results[0]['result'] == results[1]['result'], 'Callback/relocated result differs')
    results.append({'mode': 'callback', 'read_opens': 0, 'result': result})
    require(not state['denied'] and state['native_attempts'] == 0, 'Guarded action attempted')
    proof = {'schema': 'madloop-current-source-pilot-portable-proof-v1', 'verified': True,
             'data_index_sha256': INDEX_SHA, 'retention_source_index_sha256': SOURCE_SHA,
             'runner_sha256': sha(own.read_bytes()), 'verifier_sha256': VERIFY_SHA,
             'actual_capture': {'session_id': 10449, 'exit_code': 0},
             'actual_native_run': {'session_id': 29683, 'exit_code': 0},
             'results': results, 'denied_attempts': state['denied'], 'native_attempts': state['native_attempts'],
             'scope': 'pure local/isolated/callback journal, qualified+legacy parser, point-metric replay; no host-path following, native calls, locks or historical reclassification'}
    blob = (json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    (work / 'proof.json').write_bytes(blob)
    print(json.dumps({'verified': True, 'proof_sha256': sha(blob), 'proof_bytes': len(blob),
                      'read_opens': [item['read_opens'] for item in results], 'data_index_sha256': INDEX_SHA}, indent=2))


if __name__ == '__main__': main()
