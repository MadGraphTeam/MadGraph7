#!/usr/bin/env python3
"""Prepare an exact validated installation copy; publish only on root handoff.

Never build, invoke a numerical engine, change source configuration, clean an old
prefix or alter historical validation records. Publication uses MG7's tested
transaction and retains both the independent old copy and transaction backup.
"""
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
ROOT = HERE.parents[3]
WORK = ROOT/'.benchmark-work/policy-v2/main-install-publication-v1'
PREFIX = ROOT/'HEPTools'
DESTINATION = PREFIX/'ninjars'
PRIVATE = ROOT/'docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1'
INSTALLATION = PRIVATE/'installation.json'
NEW_ARCHIVE = '800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42'
OLD_ARCHIVE = '1199bc84570bee76a6b162a86fd5c191479236b2c0617fe4bf250d5de0d9df32'
REQUIRED = ('Cargo.lock', 'Cargo.toml', 'include/mninjars.f90', 'include/ninjars.h',
            'install.log', 'lib/libninjars.a', 'lib/mninjars.mod',
            'lib/ninjars-config.json', 'lib/ninjars-link-before.txt',
            'lib/ninjars-native-libs.txt')
ALIASES = ('libninjars.a', 'mninjars.mod', 'ninjars-config.json',
           'ninjars-native-libs.txt', 'ninjars-link-before.txt')


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def bound(path):
    path = Path(path)
    return {'path': relative(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def branch():
    require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
            == 'madloop_ninjars', 'Publication must stay on madloop_ninjars')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def tree(path):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), 'Installation tree must be an ordinary directory: '+str(path))
    entries = {}
    for item in [path, *sorted(path.rglob('*'))]:
        key = '.' if item == path else str(item.relative_to(path))
        info = item.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if stat.S_ISLNK(info.st_mode):
            entries[key] = {'kind': 'symlink', 'mode': mode, 'target': os.readlink(item)}
        elif stat.S_ISDIR(info.st_mode):
            entries[key] = {'kind': 'directory', 'mode': mode}
        elif stat.S_ISREG(info.st_mode):
            entries[key] = {'kind': 'file', 'mode': mode, 'bytes': info.st_size, 'sha256': digest(item)}
        else:
            raise RuntimeError('Unsupported installation entry: '+str(item))
    return {'entries': entries, 'identity_sha256': content_digest(entries),
            'files': sum(row['kind'] == 'file' for row in entries.values()),
            'bytes': sum(row.get('bytes', 0) for row in entries.values())}


def aliases():
    common = PREFIX/'lib'
    result = {}
    for name in ALIASES:
        path = common/name
        if path.is_symlink():
            result[name] = {'kind': 'symlink', 'target': os.readlink(path)}
        elif path.is_dir():
            result[name] = {'kind': 'user-directory', 'tree': tree(path)}
        elif path.is_file():
            info = path.stat()
            result[name] = {'kind': 'user-file', 'bytes': info.st_size,
                            'mode': stat.S_IMODE(info.st_mode), 'sha256': digest(path)}
        else:
            require(not path.exists(), 'Unsupported common alias entry: '+str(path))
            result[name] = {'kind': 'missing'}
    return result


def installation():
    record = json.loads(INSTALLATION.read_text())
    require(record.get('passed') is True and record.get('exit_code') == 0
            and record.get('compiled_smoke_passed') is True and record.get('abi_version') == 3,
            'Private actual installation did not pass its compiled smoke')
    require(record.get('reference_source_overrides') is False,
            'Private installation must use pinned sources, not local source overrides')
    require(set(record['files']) == set(REQUIRED), 'Private installation artifact inventory changed')
    require(record['files']['lib/libninjars.a'] == NEW_ARCHIVE, 'Unexpected validated archive')
    source = (ROOT/record['installed']).resolve()
    require(source.is_relative_to(ROOT), 'Private installation must stay inside the repository')
    candidates = [source, source.with_name('ninjars-unavailable')]
    existing = [path for path in candidates if path.is_dir() and not path.is_symlink()]
    require(len(existing) == 1, 'Exactly one ordinary or hidden private installation tree is required')
    source = existing[0]
    for name in REQUIRED:
        path = source/name
        require(path.is_file() and not path.is_symlink() and digest(path) == record['files'][name],
                'Private installation artifact changed: '+name)
    result = tree(source)
    require(result['files'] == 10 and not any(row['kind'] == 'symlink' for row in result['entries'].values()),
            'Validated source must contain precisely ten regular artifacts and ordinary directories')
    config = json.loads((source/'lib/ninjars-config.json').read_text())
    require(config.get('abi_version') == 3 and config.get('quad_precision') is True
            and not config.get('local_sources'), 'Unexpected private ABI/precision/source metadata')
    return record, source, result


def load_installer(expected):
    path = ROOT/'madgraph/various/ninjars_install.py'
    require(digest(path) == expected, 'Tested publication transaction source changed')
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('ninjars_main_publication_installer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(module._publish), 'Tested publication transaction is unavailable')
    return module


def prepare():
    revision = branch()
    require(not WORK.exists(), 'Preparation work already exists; preserve the earlier attempt')
    require(not (HERE/'preparation.json').exists(), 'Publication preparation is immutable')
    record, source, new_tree = installation()
    old_tree = tree(DESTINATION)
    require(digest(DESTINATION/'lib/libninjars.a') == OLD_ARCHIVE, 'Main installation is no longer the expected old archive')
    installer = ROOT/'madgraph/various/ninjars_install.py'
    require(digest(installer) == record['archived_installer_sha256'], 'Main and validated installer transaction differ')
    old_aliases = aliases()
    WORK.mkdir(parents=True)
    old_copy = WORK/'old-prefix-copy'
    staged = WORK/'transaction/install'
    staged.parent.mkdir()
    shutil.copytree(DESTINATION, old_copy, symlinks=True)
    require(tree(old_copy) == old_tree, 'Complete old-prefix preservation differs')
    shutil.copytree(source, staged, symlinks=True)
    require(tree(staged) == new_tree, 'Prepared installation copy differs from validated tree')
    require(tree(DESTINATION) == old_tree and aliases() == old_aliases,
            'Main installation/aliases changed during copy-only preparation')
    shutil.copy2(INSTALLATION, HERE/'private-installation.json')
    value = {'schema': 'main-ninjars-publication-preparation-v1', 'status': 'prepared-not-published',
             'prepared_utc': now(), 'preparation_revision': revision, 'installation_revision': record['revision'],
             'private_installation_record': bound(INSTALLATION), 'private_source_at_preparation': relative(source),
             'copied_installation_record': bound(HERE/'private-installation.json'),
             'main_destination': relative(DESTINATION), 'common_alias_directory': relative(PREFIX/'lib'),
             'old_prefix_backup_copy': relative(old_copy), 'transaction_stage': relative(staged.parent),
             'prepared_installation': relative(staged), 'old_tree': old_tree, 'new_tree': new_tree,
             'old_archive_sha256': OLD_ARCHIVE, 'new_archive_sha256': NEW_ARCHIVE,
             'aliases_before': old_aliases, 'installer_transaction': bound(installer),
             'helper': bound(Path(__file__)), 'publication_authorized': False,
             'engine_calls': 0, 'compiler_calls': 0,
             'scope': 'Copies only; old main installation and common aliases remain unchanged. Publication awaits private SA/NLO proof and an explicit root command.'}
    write(HERE/'preparation.json', value)
    return value


def prepared():
    value = json.loads((HERE/'preparation.json').read_text())
    require(value['schema'] == 'main-ninjars-publication-preparation-v1'
            and value['status'] == 'prepared-not-published', 'Wrong preparation status')
    require(digest(INSTALLATION) == value['private_installation_record']['sha256'], 'Private installation record changed')
    require(digest(HERE/'private-installation.json') == value['copied_installation_record']['sha256'],
            'Retained installation record changed')
    require(digest(Path(__file__)) == value['helper']['sha256'], 'Prepared publication helper changed')
    require(tree(ROOT/value['old_prefix_backup_copy']) == value['old_tree'], 'Preserved old-prefix copy changed')
    require(tree(ROOT/value['prepared_installation']) == value['new_tree'], 'Prepared validated tree changed')
    require(tree(DESTINATION) == value['old_tree'] and aliases() == value['aliases_before'],
            'Main prefix/common aliases changed before authorized publication')
    return value


def publish(handoff, proof_manifest_sha256):
    require(handoff is not None and len(handoff.strip()) >= 8, 'An explicit root handoff identifier is required')
    require(proof_manifest_sha256 is not None and len(proof_manifest_sha256) == 64,
            'Root must bind the final private SA/NLO proof manifest SHA256')
    branch()
    require(not (HERE/'transition.json').exists(), 'Do not overwrite a publication transition')
    value = prepared()
    manifest = PRIVATE/'manifest.json'
    require(manifest.is_file() and digest(manifest) == proof_manifest_sha256,
            'Private SA/NLO proof is absent or its final manifest differs')
    require((PRIVATE/'runtime-results.json').is_file(), 'Private SA/NLO runtime proof is not complete')
    # Existing verifier only reads sealed logs/runtime bytes; it invokes no engine.
    log = HERE/'private-proof-verification.log'
    with log.open('x') as stream:
        run = subprocess.run([sys.executable, '-B', str(PRIVATE/'verify.py'), '--live'], cwd=ROOT,
                             stdout=stream, stderr=subprocess.STDOUT)
    require(run.returncode == 0, 'Private SA/NLO proof verification failed; publication remains held')
    runtime = json.loads((PRIVATE/'runtime-results.json').read_text())
    require(runtime.get('nlo_cli_success') is True and runtime.get('pole_checks_passed') == 20
            and runtime.get('pole_checks_tested') == 20, 'Private NLO gate did not pass')
    require({name: item['return_code'] for name, item in runtime['standalone'].items()}
            == {'dp': 110, 'qp': 120, 'stable': 218}, 'Private standalone DP/QP/stability gate did not pass')
    # Recheck all mutable preconditions after the read-only verifier and before
    # using the same tested transaction as the actual CLI installer.
    value = prepared()
    installer = load_installer(value['installer_transaction']['sha256'])
    stage = ROOT/value['transaction_stage']
    installed = ROOT/value['prepared_installation']
    start = now()
    try:
        alias_stage = installer._publish(installed, DESTINATION, stage, PREFIX)
    except BaseException as error:
        write(HERE/'publication-failed.json', {'schema': 'main-ninjars-publication-failure-v1',
              'started_utc': start, 'root_handoff': handoff, 'error_type': type(error).__name__,
              'error': str(error), 'old_copy_retained': value['old_prefix_backup_copy'],
              'transaction_stage_retained': value['transaction_stage'],
              'note': 'The tested transaction attempted rollback. No recovery tree is cleaned by this helper.'})
        raise
    require(tree(DESTINATION) == value['new_tree'], 'Published main tree differs from validated copy')
    require(tree(stage/'previous') == value['old_tree'], 'Transaction old-prefix backup differs')
    require(tree(ROOT/value['old_prefix_backup_copy']) == value['old_tree'], 'Independent old copy changed')
    after_aliases = aliases()
    for name, previous in value['aliases_before'].items():
        if previous['kind'] in ('user-file', 'user-directory'):
            require(after_aliases[name] == previous, 'User common-library entry changed: '+name)
        else:
            require(after_aliases[name] == {'kind': 'symlink', 'target': os.path.relpath(
                DESTINATION/'lib'/name, (PREFIX/'lib').resolve())}, 'Published alias differs: '+name)
    result = {'schema': 'main-ninjars-publication-transition-v1', 'status': 'published',
              'started_utc': start, 'published_utc': now(), 'root_handoff': handoff,
              'preparation': bound(HERE/'preparation.json'), 'private_proof_manifest': bound(manifest),
              'private_proof_verification': bound(log), 'private_runtime_results': bound(PRIVATE/'runtime-results.json'),
              'installation_revision': value['installation_revision'], 'main_destination': relative(DESTINATION),
              'old_tree': value['old_tree'], 'new_tree': tree(DESTINATION), 'aliases_before': value['aliases_before'],
              'aliases_after': after_aliases, 'transaction_old_prefix_backup': relative(stage/'previous'),
              'independent_old_prefix_backup': value['old_prefix_backup_copy'],
              'retained_alias_stage': relative(alias_stage),
              'main_artifact_sha256': {name: digest(DESTINATION/name) for name in REQUIRED},
              'engine_calls': 0, 'compiler_calls': 0,
              'historical_proof_policy': 'Earlier proofs retain their own archive identities. Proofs formerly pointing to the old main prefix are historical; its exact entire tree now survives in both recorded backups. No old proof or source input is rewritten.'}
    write(HERE/'transition.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--verify-prepared', action='store_true')
    actions.add_argument('--publish', action='store_true')
    parser.add_argument('--root-handoff')
    parser.add_argument('--private-proof-manifest-sha256')
    args = parser.parse_args()
    if args.prepare:
        result = prepare()
    elif args.verify_prepared:
        result = prepared()
    else:
        result = publish(args.root_handoff, args.private_proof_manifest_sha256)
    print(json.dumps({key: result[key] for key in ('status', 'main_destination') if key in result}, indent=2))
