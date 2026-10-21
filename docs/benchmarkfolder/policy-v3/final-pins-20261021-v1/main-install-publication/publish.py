#!/usr/bin/env python3
"""Publish the exact CLI-validated final-pin image without building or evaluating.

The retained policy-v2 helper supplies tree/alias inventory and the production
installer supplies its independently tested recoverable publication transaction.
Historical proofs, both backups, the alias journal and the private source remain
untouched. This operation establishes installed-image custody, not an SA/NLO or
scientific/performance qualification.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = ROOT/'.benchmark-work/policy-v3/final-pins-20261021-v1/main-install-publication'
PREFIX = ROOT/'HEPTools'
DESTINATION = PREFIX/'ninjars'
PRIVATE = ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/install-export'
INSTALLATION = PRIVATE/'installation.json'
FREEZE = ROOT/'.ninjars-build/final-dependency-candidate-20261021/final-source-freeze-v4.json'
REVISION = '856e76e669618ec8d0f799b9c8e6f86819dcd986'
RECORD_SHA = 'df329e3b705d6793753531fab84671056f654849a3d033325882539da55abd5b'
FREEZE_SHA = 'b9488707deb57c8573fe34cd1fbbce69740fab4b024845d625ca1dbfe1060933'
OLD_ARCHIVE = '800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42'
NEW_ARCHIVE = 'b0d69deb8c16deb13293176296afc5e18dbeeebe778a20fa39b61804fba89d4c'
NEW_CONFIG = 'a2b427c4e1c766180cec068ca716e2ff3d035226b06864c50156202bcfa4617f'
NEW_LOCK = 'b8c2eddffe0dd5cb3a88b3b934133c3a0146638fa33de5e5f9fa4d65ee124f18'
REQUIRED = ('Cargo.lock', 'Cargo.toml', 'include/mninjars.f90', 'include/ninjars.h',
            'install.log', 'lib/libninjars.a', 'lib/mninjars.mod',
            'lib/ninjars-config.json', 'lib/ninjars-link-before.txt',
            'lib/ninjars-native-libs.txt')
ALIASES = ('libninjars.a', 'mninjars.mod', 'ninjars-config.json',
           'ninjars-native-libs.txt', 'ninjars-link-before.txt')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


sys.path.insert(0, str(ROOT))
HELPER = ROOT/'docs/benchmarkfolder/policy-v2/main-install-publication-v1/publish.py'
old = module(HELPER, 'ninjars_retained_publication_inventory')
require, digest, tree, bound, write = old.require, old.digest, old.tree, old.bound, old.write


def alias_image(directory):
    result = {}
    for name in ALIASES:
        path = directory/name
        if path.is_symlink():
            result[name] = {'kind': 'symlink', 'target': os.readlink(path)}
        elif path.is_dir():
            result[name] = {'kind': 'user-directory', 'tree': tree(path)}
        elif path.is_file():
            result[name] = {'kind': 'user-file', 'sha256': digest(path),
                            'bytes': path.stat().st_size,
                            'mode': stat.S_IMODE(path.stat().st_mode)}
        else:
            require(not path.exists(), 'Unsupported alias entry: '+str(path))
            result[name] = {'kind': 'missing'}
    return result


def validate_source():
    require(old.branch() == REVISION, 'Final-pin generation source revision changed')
    require(digest(INSTALLATION) == RECORD_SHA, 'Accepted installation record changed')
    record = json.loads(INSTALLATION.read_text())
    require(record.get('revision') == REVISION and record.get('passed') is True
            and record.get('actual_cli_update') is True and record.get('exit_code') == 0
            and record.get('compiled_smoke_passed') is True and record.get('abi_version') == 3
            and record.get('reference_source_overrides') is False
            and record.get('local_sources') == {}, 'Actual pinned CLI update did not pass')
    expected = record['files']
    require(set(expected) == set(REQUIRED) and expected == record['installed_file_sha256'],
            'Accepted complete installed artifact map changed')
    require(expected['lib/libninjars.a'] == NEW_ARCHIVE
            and expected['lib/ninjars-config.json'] == NEW_CONFIG
            and expected['Cargo.lock'] == NEW_LOCK, 'Wrong final artifact identity')
    source = ROOT/record['installed']
    require(source.resolve().is_relative_to(ROOT) and source.is_dir() and not source.is_symlink(),
            'Private installed image is absent or outside the repository')
    image = tree(source)
    require(image['files'] == 10 and not any(x['kind'] == 'symlink' for x in image['entries'].values()),
            'Private installation must be exactly ten regular files without symlinks')
    for name, sha in expected.items():
        require(digest(source/name) == sha, 'Installed image changed: '+name)
    require(digest(FREEZE) == FREEZE_SHA, 'Final45 source-v4 receipt changed')
    freeze = json.loads(FREEZE.read_text())
    require(len(freeze['files']) == 45 and freeze['files'] == record['archived_bridge_source_sha256'],
            'Installed source45 inventory does not match final source-v4')
    for name, sha in freeze['files'].items():
        for base in (ROOT/'vendor/ninjars', source.parents[1]/'vendor/ninjars'):
            require(digest(base/name) == sha, 'Final45 source changed: '+str(base/name))
    installer_path = ROOT/'madgraph/various/ninjars_install.py'
    require(digest(installer_path) == record['archived_installer_sha256'],
            'Current transaction source differs from the validated installer')
    installer = module(installer_path, 'ninjars_final_pinned_publication_installer')
    config = installer.read_config(source/'lib')
    require(config.get('quad_precision') is True and config.get('local_sources') == {}
            and config.get('build_provenance') == installer._dependency_provenance(source)
            and config['build_provenance'] == record['build_provenance']
            and config['build_provenance'] == record['expected_build_provenance'],
            'Resolved pinned manifest/lock/Git provenance differs')
    return record, source, image, installer


def publish(handoff):
    require(handoff == 'ROOT_AUTHORIZED_FINAL_PIN_MAIN_PUBLICATION_20261021',
            'This one-shot publication requires the recorded explicit root handoff')
    require(not WORK.exists() and not (HERE/'transition.json').exists()
            and not (HERE/'preparation.json').exists(), 'Never overwrite an earlier publication attempt')
    record, source, new_tree, installer = validate_source()
    old_tree = tree(DESTINATION)
    old_transition = ROOT/'docs/benchmarkfolder/policy-v2/main-install-publication-v1/transition.json'
    require(digest(DESTINATION/'lib/libninjars.a') == OLD_ARCHIVE
            and old_tree == json.loads(old_transition.read_text())['new_tree'],
            'Main old800 image differs from its retained publication proof')
    old_aliases = alias_image(PREFIX/'lib')
    WORK.mkdir(parents=True)
    stage = WORK/'transaction'
    stage.mkdir()
    independent = WORK/'old-prefix-copy'
    old_links = WORK/'old-common-aliases'
    old_links.mkdir()
    shutil.copytree(DESTINATION, independent, symlinks=True)
    require(tree(independent) == old_tree, 'Independent old800 backup differs')
    for name in ALIASES:
        path = PREFIX/'lib'/name
        if path.is_symlink():
            (old_links/name).symlink_to(os.readlink(path))
        elif path.is_dir():
            shutil.copytree(path, old_links/name, symlinks=True)
        elif path.is_file():
            shutil.copy2(path, old_links/name)
    require(alias_image(old_links) == old_aliases, 'Independent common-alias image differs')
    candidate = stage/'install'
    shutil.copytree(source, candidate, symlinks=True)
    require(tree(candidate) == new_tree, 'Copied validated image differs')
    shutil.copy2(INSTALLATION, HERE/'accepted-installation.json')
    shutil.copy2(FREEZE, HERE/'final-source-freeze-v4.json')
    preparation = {'schema': 'main-ninjars-final-pin-preparation-v1', 'status': 'prepared-authorized',
        'prepared_utc': old.now(), 'root_handoff': handoff, 'revision': REVISION,
        'accepted_installation': bound(INSTALLATION), 'copied_installation': bound(HERE/'accepted-installation.json'),
        'source_freeze': bound(HERE/'final-source-freeze-v4.json'), 'installer': bound(ROOT/'madgraph/various/ninjars_install.py'),
        'helper': bound(Path(__file__)), 'inventory_helper': bound(HELPER),
        'main_destination': old.relative(DESTINATION), 'private_source': old.relative(source),
        'old_tree': old_tree, 'new_tree': new_tree, 'aliases_before': old_aliases,
        'independent_old_prefix_backup': old.relative(independent),
        'independent_old_alias_backup': old.relative(old_links), 'transaction_stage': old.relative(stage),
        'engine_calls': 0, 'compiler_calls': 0,
        'qualification_scope': 'Actual pinned CLI update and compiled installed smoke passed; standalone/NLO, final scientific and timing gates remain separate.'}
    write(HERE/'preparation.json', preparation)
    record2, source2, image2, _ = validate_source()
    require(record2 == record and source2 == source and image2 == new_tree
            and tree(DESTINATION) == old_tree and alias_image(PREFIX/'lib') == old_aliases,
            'Mutable publication preconditions changed during backup/copy')
    started = old.now()
    try:
        alias_stage = installer._publish(candidate, DESTINATION, stage, PREFIX)
    except BaseException as error:
        write(HERE/'publication-failed.json', {'schema': 'main-ninjars-final-pin-failure-v1',
            'started_utc': started, 'root_handoff': handoff, 'error_type': type(error).__name__,
            'error': str(error), 'independent_old_prefix_backup': old.relative(independent),
            'transaction_stage': old.relative(stage), 'cleanup_attempted': False})
        raise
    require(tree(DESTINATION) == new_tree and tree(stage/'previous') == old_tree
            and tree(independent) == old_tree and alias_image(old_links) == old_aliases,
            'Published image or retained old800/alias image differs')
    aliases_after = alias_image(PREFIX/'lib')
    for name, previous in old_aliases.items():
        expected = previous if previous['kind'] in ('user-file', 'user-directory') else {
            'kind': 'symlink', 'target': os.path.relpath(DESTINATION/'lib'/name, (PREFIX/'lib').resolve())}
        require(aliases_after[name] == expected, 'Published alias differs: '+name)
    require(tree(source) == new_tree and digest(INSTALLATION) == RECORD_SHA,
            'Original private validated installation was changed')
    after = {name: digest(DESTINATION/name) for name in REQUIRED}
    require(after == record['files'], 'Complete main afterimage differs from accepted ten-file image')
    transition = {'schema': 'main-ninjars-final-pin-transition-v1', 'status': 'published',
        'started_utc': started, 'published_utc': old.now(), 'root_handoff': handoff, 'revision': REVISION,
        'preparation': bound(HERE/'preparation.json'), 'accepted_installation': bound(INSTALLATION),
        'source_freeze': bound(HERE/'final-source-freeze-v4.json'),
        'main_destination': old.relative(DESTINATION), 'old_tree': old_tree, 'new_tree': tree(DESTINATION),
        'main_artifact_sha256': after, 'aliases_before': old_aliases, 'aliases_after': aliases_after,
        'independent_old_prefix_backup': old.relative(independent),
        'transaction_old_prefix_backup': old.relative(stage/'previous'),
        'independent_old_alias_backup': old.relative(old_links), 'retained_alias_stage': old.relative(alias_stage),
        'engine_calls': 0, 'compiler_calls': 0, 'private_source_unchanged': True,
        'qualification_scope': preparation['qualification_scope'],
        'historical_proof_policy': 'Old800 complete image survives in independent and rename backups; all old proof/source inputs and the private final image remain unchanged. Earlier science/runtime observations retain their original identities.'}
    write(HERE/'transition.json', transition)
    return transition


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-handoff', required=True)
    result = publish(parser.parse_args().root_handoff)
    print(json.dumps({key: result[key] for key in ('status', 'revision', 'main_artifact_sha256')}, indent=2))
