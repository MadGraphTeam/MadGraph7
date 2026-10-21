#!/usr/bin/env python3
"""Read-only bounded cleanup candidates and untracked-publication inventory."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NATIVE = ROOT/'docs/benchmarkfolder/experiments/ninjars-native-float256-v1'
SCRATCH = ['.benchmark-work/final-regression-review-v1',
           '.benchmark-work/final-regression-fks-review-v1',
           '.benchmark-work/final-regression-fks-review-v2-absent']


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bound(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'bytes': path.stat().st_size}


def read_json(path):
    return json.loads(path.read_text())


def under(path, candidate):
    return path == candidate or path.startswith(candidate+'/')


def process_uses(candidates):
    """Observe current same-UID cwd/fds/mappings, never signal a process.

    This is a point-in-time same-UID check, not a guarantee about later opens or
    processes outside this account. Do not expose unrelated process arguments.
    """
    uid = os.getuid()
    roots = [str(ROOT/name) for name in candidates]
    result, scanned, vanished, denied = [], 0, 0, 0
    for proc in sorted(Path('/proc').iterdir(), key=lambda p: p.name):
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid != uid:
                continue
            scanned += 1
            hits = []
            try:
                cwd = os.readlink(proc/'cwd')
                if any(under(cwd, root) for root in roots):
                    hits.append({'kind': 'cwd', 'path': cwd})
                for item in (proc/'fd').iterdir():
                    try:
                        target = os.readlink(item).removesuffix(' (deleted)')
                        if any(under(target, root) for root in roots):
                            hits.append({'kind': 'fd', 'descriptor': item.name, 'path': target})
                    except FileNotFoundError:
                        pass
                with (proc/'maps').open() as stream:
                    for line in stream:
                        fields = line.rstrip('\n').split(maxsplit=5)
                        if len(fields) == 6:
                            target = fields[5].removesuffix(' (deleted)')
                            if any(under(target, root) for root in roots):
                                hits.append({'kind': 'mapping', 'path': target})
            except FileNotFoundError:
                vanished += 1
            except PermissionError:
                denied += 1
            if hits:
                result.append({'pid': int(proc.name), 'uses': hits})
        except FileNotFoundError:
            vanished += 1
        except PermissionError:
            denied += 1
    return {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'uid': uid, 'same_uid_processes_examined': scanned,
            'processes_vanished_during_scan': vanished, 'permission_denied': denied,
            'uses': result, 'scope': 'Same UID cwd, file descriptors and mapped paths only; no command arguments, signals or future-use guarantee'}


def capture():
    revision = git('rev-parse', 'HEAD').strip()
    tracked = set(git('ls-files', '-z').split('\0'))
    native_manifest_path = NATIVE/'manifest.json'
    native_manifest = read_json(native_manifest_path)
    index_path = NATIVE/'compressed-artifacts.json'
    index = read_json(index_path)
    require(digest(index_path) == native_manifest['compressed_index_sha256'], 'Native compressed index changed')
    require(set(index['files']) == set(native_manifest['materialized_plaintext_exclusions'])
            and len(index['files']) == 6, 'Expected six exact native materializations')
    candidates, rules = [], []
    for name, record in sorted(index['files'].items()):
        plain = NATIVE/name
        compressed = NATIVE/record['compressed']['path']
        rel = str(plain.relative_to(ROOT))
        archive_rel = str(compressed.relative_to(ROOT))
        require(rel not in tracked, 'Plain materialization was accidentally tracked')
        require(archive_rel in tracked, 'Compressed evidence is not committed')
        require(record['compressed']['path'] in native_manifest['files'], 'Compressed evidence not in native seal')
        for path, expected in [(plain, record['uncompressed']), (compressed, record['compressed'])]:
            require(path.is_file() and not path.is_symlink(), 'Expected ordinary materialization/evidence file: '+str(path))
            require(path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], 'Native materialization/archive changed')
        candidates.append({'path': rel, 'kind': 'materialized-plaintext-duplicate', 'bytes': plain.stat().st_size,
                           'sha256': record['uncompressed']['sha256'], 'compressed_evidence': bound(compressed),
                           'native_manifest': bound(native_manifest_path),
                           'git_payload_manifested': False, 'regenerable': True,
                           'regenerate_command': 'python -B docs/benchmarkfolder/experiments/ninjars-native-float256-v1/archive_large.py materialize'})
        rules.append('/'+rel)
    regression = ROOT/'docs/benchmarkfolder/policy-v2/final-regressions-v1'
    manifest = read_json(regression/'manifest.json')
    require(digest(regression/'validation.json') == manifest['files']['validation.json']['sha256'], 'Regression proof identity changed')
    for name in SCRATCH:
        root = ROOT/name
        require(root.is_dir() and not root.is_symlink(), 'Scratch path is not an ordinary directory: '+name)
        require(not any(under(item, name) for item in tracked if item), 'Scratch path contains tracked files')
        count, size, links = 0, 0, 0
        for path in root.rglob('*'):
            info = path.lstat()
            if stat.S_ISREG(info.st_mode):
                count += 1
                size += info.st_size
            elif stat.S_ISLNK(info.st_mode):
                links += 1
        candidates.append({'path': name, 'kind': 'completed-regression-comparison-scratch',
                           'files': count, 'bytes': size, 'symlinks': links,
                           'git_payload_manifested': False,
                           'retained_proof': bound(regression/'manifest.json'),
                           'scope': 'Private regenerated comparison trees and commands; durable inputs/logs/scripts survive in sealed final-regressions-v1, whose verifier does not read these scratch trees'})
    uses = process_uses([row['path'] for row in candidates])
    for row in candidates:
        absolute = str(ROOT/row['path'])
        row['observed_process_uses'] = [match for match in uses['uses'] if any(under(hit['path'], absolute) for hit in match['uses'])]
        row['recommendation'] = 'Root may remove this exact path after a fresh no-open-process check; preserve compressed/durable proof inputs' if not row['observed_process_uses'] else 'HOLD: observed process use'
    untracked = [name for name in git('ls-files', '--others', '--exclude-standard', '-z').split('\0') if name]
    install_base = ROOT/'docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1'
    install_manifest = read_json(install_base/'manifest.json')
    classes, rows = collections.Counter(), []
    disposable = {row['path'] for row in candidates}
    build_findings = []
    for name in untracked:
        path = ROOT/name
        if name in disposable:
            label = 'six-native-materializations: remove-or-ignore-exactly'
        elif name.startswith(str(install_base.relative_to(ROOT))+'/'):
            item = str(path.relative_to(install_base))
            require(item == 'manifest.json' or item in install_manifest['files'], 'Unknown final-install file: '+name)
            if item != 'manifest.json':
                require(digest(path) == install_manifest['files'][item]['sha256'], 'Final installation evidence changed: '+name)
            label = 'required-sealed-final-install-export-evidence'
        elif name.startswith('docs/benchmarkfolder/policy-v2/main-install-publication-v1/'):
            label = 'required-main-install-transition-preparation: retain-through-publication'
        elif name.startswith('docs/benchmarkfolder/policy-v2/reference-runs/'):
            label = 'required-reference-evidence: append-only-owner-review'
        elif name.startswith('docs/benchmarkfolder/policy-v2/reference-resume-20261021T0002Z/'):
            label = 'required-reference-pause-resume-ledger'
        elif name.startswith('docs/benchmarkfolder/publication-cleanup-readiness-v1/'):
            label = 'this-readonly-cleanup-audit'
        elif name == 'vendor/StdHEP/fail':
            label = 'preexisting-user-workspace-file: preserve'
        else:
            label = 'unclassified: preserve-for-root-owner-review'
        info = path.lstat()
        rows.append({'path': name, 'classification': label, 'bytes': info.st_size})
        classes[label] += 1
        if stat.S_ISREG(info.st_mode):
            with path.open('rb') as stream:
                magic = stream.read(8)
            if magic.startswith((b'\x7fELF', b'!<arch>\n')) or path.suffix in ('.o', '.a', '.so', '.mod', '.pyc', '.pyo') or '__pycache__' in path.parts:
                build_findings.append(name)
    return {'schema': 'publication-cleanup-readiness-v1', 'revision_at_start': revision,
            'revision_at_end': git('rev-parse', 'HEAD').strip(),
            'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'cleanup_performed': False, 'source_changed': False, 'engine_calls': 0, 'compiler_calls': 0,
            'candidates': candidates, 'candidate_count': len(candidates),
            'candidate_bytes': sum(row['bytes'] for row in candidates),
            'same_uid_process_observation': uses,
            'untracked_count': len(untracked), 'untracked_classification_counts': dict(sorted(classes.items())),
            'untracked_files': rows, 'untracked_build_findings': build_findings,
            'proposed_exact_native_ignore_rules': rules,
            'native_materialization_index': bound(index_path),
            'required_installation_manifest': bound(install_base/'manifest.json'),
            'scope': 'No cleanup; only nine exact candidates, current untracked paths, their relevant seals and a point-in-time same-UID process-use check'}


if __name__ == '__main__':
    result = capture()
    with (HERE/'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    with (HERE/'cleanup-candidate-paths.txt').open('x') as stream:
        stream.write(''.join(row['path']+'\n' for row in result['candidates']))
    with (HERE/'proposed-native-ignore-rules.txt').open('x') as stream:
        stream.write(''.join(rule+'\n' for rule in result['proposed_exact_native_ignore_rules']))
    print(json.dumps({key: result[key] for key in ('revision_at_start', 'revision_at_end', 'candidate_count',
          'candidate_bytes', 'untracked_count', 'untracked_classification_counts', 'untracked_build_findings')}, indent=2))
    print(json.dumps(result['same_uid_process_observation'], indent=2))
