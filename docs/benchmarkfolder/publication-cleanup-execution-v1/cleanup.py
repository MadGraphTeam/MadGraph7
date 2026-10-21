#!/usr/bin/env python3
"""One authorized removal of nine exact disposable paths; no general cleanup."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
READY = ROOT/'docs/benchmarkfolder/publication-cleanup-readiness-v1'
EXPECTED_READY_MANIFEST = '4927a8d3b657ffb099eadca907461e153efc307a804b1593470c3603225630f9'
EXPECTED_FILES = [
'docs/benchmarkfolder/experiments/ninjars-native-float256-v1/convergence-analysis.json',
'docs/benchmarkfolder/experiments/ninjars-native-float256-v1/replay-analysis.json',
'docs/benchmarkfolder/experiments/ninjars-native-float256-v1/run-convergence/raw.jsonl',
'docs/benchmarkfolder/experiments/ninjars-native-float256-v1/run-replay/raw.jsonl',
'docs/benchmarkfolder/experiments/ninjars-native-float256-v1/screening-analysis.json',
'docs/benchmarkfolder/experiments/ninjars-native-float256-v1/supplement/run-screening/raw.jsonl']
EXPECTED_DIRS = ['.benchmark-work/final-regression-review-v1',
'.benchmark-work/final-regression-fks-review-v1',
'.benchmark-work/final-regression-fks-review-v2-absent']


def need(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_seal(directory, exclude=()):
    manifest = json.loads((directory/'manifest.json').read_text())
    excluded = {'manifest.json', *exclude}
    actual = {str(path.relative_to(directory)) for path in directory.rglob('*')
              if path.is_file() and str(path.relative_to(directory)) not in excluded}
    need(actual == set(manifest['files']), 'Durable proof file set changed: '+str(directory))
    for name, expected in manifest['files'].items():
        path = directory/name
        need(not path.is_symlink() and path.stat().st_size == expected['bytes']
             and sha(path) == expected['sha256'], 'Durable proof changed: '+str(path))
    return {'path': str((directory/'manifest.json').relative_to(ROOT)),
            'sha256': sha(directory/'manifest.json'), 'files': len(manifest['files']), 'passed': True}


def directory_identity(path):
    need(path.is_dir() and not path.is_symlink() and path.resolve() == ROOT/str(path.relative_to(ROOT)),
         'Unexpected scratch directory or symlink ancestor: '+str(path))
    device = path.stat().st_dev
    values = {}
    for item in sorted(path.rglob('*')):
        info = item.lstat()
        need(info.st_dev == device and not stat.S_ISLNK(info.st_mode), 'Scratch contains a symlink/mount: '+str(item))
        name = str(item.relative_to(path))
        if stat.S_ISREG(info.st_mode):
            values[name] = {'bytes': info.st_size, 'sha256': sha(item)}
        else:
            need(stat.S_ISDIR(info.st_mode), 'Unexpected scratch entry: '+str(item))
    return {'files': len(values), 'bytes': sum(item['bytes'] for item in values.values()),
            'full_file_map_sha256': identity(values)}


def known_workers():
    scripts = {'capture_io.py', 'capture_fks_io.py', 'capture_fks_absent_io.py', 'run_supported_fks.py'}
    modules = {'tests.unit_tests.loop.test_loop_exporters', 'tests.unit_tests.iolibs.test_export_fks',
               'tests.unit_tests.various.test_q_polynomial_compiled', 'tests.unit_tests.various.test_q_polynomial_factorized'}
    matches, unreadable = [], 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = [item.decode() for item in (proc/'cmdline').read_bytes().split(b'\0') if item]
            if not args or 'python' not in Path(args[0]).name:
                continue
            selected = [arg for arg in args[1:] if arg in modules or ('\n' not in arg and '/' in arg and Path(arg).name in scripts)]
            if selected:
                matches.append({'pid': int(proc.name), 'selected_worker_arguments': selected})
        except (FileNotFoundError, ProcessLookupError):
            pass
        except (PermissionError, UnicodeError):
            unreadable += 1
    return {'matches': matches, 'unreadable_same_uid_commands': unreadable,
            'scope': 'Only exact completed regression collector/harness script names and selected unittest module arguments; no unrelated command arguments retained'}


def execute(handoff):
    need(len(handoff.strip()) >= 8, 'Explicit root cleanup handoff is required')
    need(not (HERE/'cleanup-result.json').exists() and not (HERE/'cleanup-started.json').exists(),
         'Cleanup evidence already exists; do not repeat or overwrite')
    need(sha(READY/'manifest.json') == EXPECTED_READY_MANIFEST, 'Authorized readiness manifest changed')
    load('ninjars_cleanup_readiness_verifier', READY/'verify.py').verify()
    ready = json.loads((READY/'result.json').read_text())
    need([row['path'] for row in ready['candidates']] == EXPECTED_FILES+EXPECTED_DIRS,
         'Authorized nine-path inventory changed')
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    native = ROOT/'docs/benchmarkfolder/experiments/ninjars-native-float256-v1'
    native_manifest = json.loads((native/'manifest.json').read_text())
    durable = [check_seal(native, native_manifest['materialized_plaintext_exclusions']),
               check_seal(ROOT/'docs/benchmarkfolder/policy-v2/final-regressions-v1')]
    candidates = []
    for row in ready['candidates']:
        name = row['path']
        path = ROOT/name
        need(name not in tracked and not any(item.startswith(name+'/') for item in tracked if item),
             'Cleanup candidate contains tracked files: '+name)
        if name in EXPECTED_FILES:
            need(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                 'Plaintext duplicate differs from retained checksum: '+name)
            compressed = ROOT/row['compressed_evidence']['path']
            need(str(compressed.relative_to(ROOT)) in tracked, 'Compressed replacement is not committed')
            item = {'path': name, 'kind': 'file', 'bytes': row['bytes'], 'sha256': row['sha256'],
                    'durable_compressed_record': row['compressed_evidence']}
        else:
            checksum = directory_identity(path)
            need(checksum['files'] == row['files'] and checksum['bytes'] == row['bytes'],
                 'Completed comparison scratch count/size changed: '+name)
            item = {'path': name, 'kind': 'directory', **checksum}
        candidates.append(item)
    scanner = load('ninjars_cleanup_observer', READY/'capture.py')
    observation = scanner.process_uses(EXPECTED_FILES+EXPECTED_DIRS)
    workers = known_workers()
    need(not observation['uses'], 'Observed a process using an exact cleanup candidate')
    need(not workers['matches'], 'Completed-regression worker is still live')
    # Preserve the root-approved six exact rules; no broad native/source rule.
    proposed = (READY/'proposed-native-ignore-rules.txt').read_text().splitlines()
    need(proposed == ['/'+name for name in EXPECTED_FILES], 'Exact ignore proposal changed')
    ignore = ROOT/'.gitignore'
    ignore_before = sha(ignore)
    text = ignore.read_text()
    need(not any(rule in text.splitlines() for rule in proposed), 'Exact native rules already exist; review before repeating')
    # A second full directory hash pass immediately before deletion protects
    # against a changed scratch tree between capture and the no-use observation.
    for item in candidates:
        path = ROOT/item['path']
        if item['kind'] == 'directory':
            need(directory_identity(path) == {key: item[key] for key in ('files', 'bytes', 'full_file_map_sha256')},
                 'Scratch changed before deletion: '+item['path'])
        else:
            need(sha(path) == item['sha256'] and not path.is_symlink(), 'Materialization changed before deletion')
    record = {'schema': 'publication-cleanup-execution-v1', 'started_utc': now(),
              'root_handoff': handoff, 'readiness_manifest_sha256': EXPECTED_READY_MANIFEST,
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'candidates_before': candidates, 'durable_proofs_before': durable,
              'fresh_same_uid_process_observation': observation, 'known_completed_workers': workers,
              'permission_limit': 'The same-UID observation records permission-denied/vanished processes and cannot guarantee absence of all inaccessible or later opens. Root explicitly approved this bounded cleanup with that disclosed limit.',
              'engine_calls': 0, 'compiler_calls': 0, 'removed': [],
              'preexisting_vendor_StdHEP_fail_touched': False}
    write(HERE/'cleanup-started.json', record)
    try:
        for item in candidates:
            path = ROOT/item['path']
            if item['kind'] == 'directory':
                shutil.rmtree(path)
            else:
                path.unlink()
            need(not path.exists() and not path.is_symlink(), 'Exact candidate survived deletion')
            record['removed'].append(item['path'])
        with ignore.open('a') as stream:
            stream.write('\n# Recreated plaintext copies of sealed native-study gzip evidence.\n')
            stream.write('\n'.join(proposed)+'\n')
        record.update(status='completed', completed_utc=now(), removed_count=len(record['removed']),
                      removed_bytes=sum(item['bytes'] for item in candidates),
                      exact_ignore_rules_added=proposed,
                      gitignore_before_sha256=ignore_before, gitignore_after_sha256=sha(ignore),
                      durable_proofs_after=[check_seal(native, native_manifest['materialized_plaintext_exclusions']),
                                           check_seal(ROOT/'docs/benchmarkfolder/policy-v2/final-regressions-v1')])
        load('ninjars_cleanup_post_readiness_verifier', READY/'verify.py').verify()
    except BaseException as error:
        record.update(status='partial-failure', completed_utc=now(), error_type=type(error).__name__, error=str(error))
        write(HERE/'cleanup-result.json', record)
        raise
    write(HERE/'cleanup-result.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-handoff', required=True)
    args = parser.parse_args()
    result = execute(args.root_handoff)
    print(json.dumps({key: result[key] for key in ('status', 'removed_count', 'removed_bytes',
                     'gitignore_before_sha256', 'gitignore_after_sha256')}, indent=2))
