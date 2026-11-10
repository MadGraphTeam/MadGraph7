"""Offline replay, relocation and negative checks of the closed stress proof."""
import builtins
from contextlib import ExitStack
import ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CAPSULE = BASE / 'physical-stress-current-portable-5-terminal-1'
EXPECTED = 'a520199cf577d7b35b26b054016607020ebe0a533e5afabb4c1a279812c64266'
SIBLINGS = ('physical-stress-current-portable-4-terminal-2', 'physical-stress-current-portable-4-preflight-1')


def require(value, message):
    if not value: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def main():
    require(sha((CAPSULE / 'evidence-index.json').read_bytes()) == EXPECTED, 'Unexpected actual proof index')
    spec = importlib.util.spec_from_file_location('qualified_closed_stress_verifier', CAPSULE / 'verify_evidence.py')
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    payloads = {}
    for prefix, root in (('', CAPSULE), ('ancestor/', BASE / SIBLINGS[0]), ('preparation/', BASE / SIBLINGS[1])):
        paths = list(root.rglob('*'))
        require(not any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths), 'Unexpected proof object')
        for path in paths:
            if path.is_file(): payloads[prefix + path.relative_to(root).as_posix()] = path.read_bytes()
    def reader(name): return payloads[name]
    reader.paths = set(payloads)
    checks = []
    def record(name, action):
        action(); checks.append({'name': name, 'passed': True})
    def blocked(*args, **kwargs): raise AssertionError('Native execution is forbidden in evidence replay')
    def full_callback():
        with patch.object(subprocess, 'Popen', side_effect=blocked), patch.object(ctypes, 'CDLL', side_effect=blocked), patch.object(os, 'system', side_effect=blocked):
            proof = verifier.verify(read_bytes=reader)
        require(proof['verified'] and proof['index_sha256'] == EXPECTED, 'Offline callback replay failed')
        summary = proof['summary']; aggregate = summary['aggregate']
        require(summary['completed_orchestration'] == 20 and summary['host_checked_complete_streams'] == 8,
                'Continuation counts differ')
        require(summary['state_counts'] == {'host_checked_complete': 8, 'physical_rejected': 11, 'controller_failed': 1}, 'Continuation states differ')
        require(aggregate['completed_orchestration'] == 24 and aggregate['host_checked_complete_streams'] == 12
                and len(aggregate['comparisons']) == 22 and sum(not v['passed'] for v in aggregate['comparisons']) == 12
                and len(aggregate['inherited_failing_comparisons']) == 5, 'Aggregate counts/failures differ')
    record('offline_callback_with_native_entrypoints_blocked', full_callback)

    def relocation():
        with tempfile.TemporaryDirectory(prefix='tirust-closed-stress-relocation-') as temporary:
            temporary = Path(temporary)
            for name in (CAPSULE.name, *SIBLINGS): shutil.copytree(BASE / name, temporary / name)
            original_builtin, original_io = builtins.open, io.open
            def guard(original):
                def opened(file, mode='r', *args, **kwargs):
                    if not isinstance(file, int):
                        target = Path(file).absolute()
                        require(target.is_relative_to(temporary), 'Read outside relocated proof closure: ' + str(target))
                    require(not any(flag in mode for flag in ('w', 'a', 'x', '+')), 'Replay attempted a write')
                    return original(file, mode, *args, **kwargs)
                return opened
            with ExitStack() as stack:
                stack.enter_context(patch.object(builtins, 'open', side_effect=guard(original_builtin)))
                stack.enter_context(patch.object(io, 'open', side_effect=guard(original_io)))
                stack.enter_context(patch.object(subprocess, 'Popen', side_effect=blocked))
                stack.enter_context(patch.object(ctypes, 'CDLL', side_effect=blocked))
                stack.enter_context(patch.object(os, 'system', side_effect=blocked))
                proof = verifier.verify(temporary / CAPSULE.name)
            require(proof['verified'] and proof['index_sha256'] == EXPECTED, 'Relocated replay changed identity')
    record('relocated_filesystem_with_external_reads_writes_and_native_calls_blocked', relocation)

    def rejected(name, changed=None, paths=None):
        def altered(path): return changed[path] if changed is not None and path in changed else payloads[path]
        if paths is not None: altered.paths = paths
        try: verifier.verify(read_bytes=altered)
        except (ValueError, KeyError, json.JSONDecodeError): return
        raise AssertionError('Negative proof accepted: ' + name)
    record('missing_reader_paths_rejected_before_read', lambda: rejected('missing_paths'))
    record('extra_reader_path_rejected', lambda: rejected('extra_paths', paths=set(payloads) | {'unindexed.txt'}))
    record('missing_declared_reader_path_rejected', lambda: rejected('missing_path', paths=set(payloads) - {'retention.json'}))
    record('changed_retention_bytes_rejected', lambda: rejected('changed_retention', {'retention.json': b'changed'}, set(payloads)))
    record('changed_ancestor_index_rejected', lambda: rejected('ancestor_index', {'ancestor/evidence-index.json': b'{}'}, set(payloads)))
    record('changed_preparation_index_rejected', lambda: rejected('preparation_index', {'preparation/evidence-index.json': b'{}'}, set(payloads)))
    index = json.loads(payloads['evidence-index.json'])
    alias = next(iter(index['compressed_text']))
    physical = index['compressed_text'][alias]['path']
    record('changed_compressed_text_rejected', lambda: rejected('gzip', {physical: b'changed'}, set(payloads)))
    summary = json.loads(payloads['qualification-summary.json']); summary['host_checked_complete_streams'] = 20
    changed = (json.dumps(summary, sort_keys=True, indent=2) + '\n').encode()
    forged_index = json.loads(payloads['evidence-index.json'])
    forged_index['files']['qualification-summary.json'].update(size=len(changed), sha256=sha(changed))
    record('forged_saved_qualification_count_rejected_by_replay', lambda: rejected('saved_count', {
        'qualification-summary.json': changed, 'evidence-index.json': (json.dumps(forged_index, sort_keys=True, indent=2) + '\n').encode()}, set(payloads)))
    result = {'schema_version': 1, 'verified_index_sha256': EXPECTED, 'native_execution': False,
              'closure_physical_files': len(payloads), 'checks': checks, 'passed': all(c['passed'] for c in checks)}
    output = HERE / 'results.json'
    with output.open('x') as stream: json.dump(result, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__': main()
