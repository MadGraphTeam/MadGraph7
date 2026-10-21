#!/usr/bin/env python3
"""Check retained image-publication evidence; --live also rehashes owned images.

No compiler, Cargo process, Fortran program or numerical engine is invoked.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ninjars_final_publication', HERE/'publish.py')
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)
ROOT = pub.ROOT


def read(path):
    return json.loads(Path(path).read_text())


def verify(live):
    transition = read(HERE/'transition.json')
    pub.require(transition['status'] == 'published' and transition['engine_calls'] == 0
                and transition['compiler_calls'] == 0 and transition['private_source_unchanged'] is True,
                'Publication is not a clean copy-only transition')
    for key in ('preparation', 'accepted_installation', 'source_freeze'):
        binding = transition[key]
        pub.require(pub.digest(ROOT/binding['path']) == binding['sha256'], 'Changed binding: '+key)
    preparation = read(ROOT/transition['preparation']['path'])
    accepted = read(HERE/'accepted-installation.json')
    freeze = read(HERE/'final-source-freeze-v4.json')
    pub.require(pub.digest(HERE/'accepted-installation.json') == pub.RECORD_SHA
                and pub.digest(HERE/'final-source-freeze-v4.json') == pub.FREEZE_SHA,
                'Accepted source/installation custody changed')
    pub.require(accepted['files'] == accepted['installed_file_sha256'] == transition['main_artifact_sha256']
                and set(accepted['files']) == set(pub.REQUIRED), 'Complete ten-file map differs')
    pub.require(accepted['passed'] and accepted['actual_cli_update'] and accepted['compiled_smoke_passed']
                and accepted['exit_code'] == 0 and accepted['local_sources'] == {}, 'CLI update qualification differs')
    pub.require(len(freeze['files']) == 45 and freeze['files'] == accepted['archived_bridge_source_sha256'],
                'Final45 source-v4 differs')
    pub.require(transition['main_artifact_sha256']['lib/libninjars.a'] == pub.NEW_ARCHIVE
                and transition['main_artifact_sha256']['lib/ninjars-config.json'] == pub.NEW_CONFIG
                and transition['main_artifact_sha256']['Cargo.lock'] == pub.NEW_LOCK,
                'Final installed dependency identity differs')
    pub.require(transition['old_tree'] == preparation['old_tree']
                and transition['new_tree'] == preparation['new_tree'], 'Prepared/published trees differ')
    previous = read(ROOT/'docs/benchmarkfolder/policy-v2/main-install-publication-v1/transition.json')
    pub.require(transition['old_tree'] == previous['new_tree'], 'Old800 historical image differs')
    for key in ('old_tree', 'new_tree'):
        image = transition[key]
        pub.require(pub.old.content_digest(image['entries']) == image['identity_sha256'],
                    'Tree inventory identity differs: '+key)
    new_files = {name: item['sha256'] for name, item in transition['new_tree']['entries'].items()
                 if item['kind'] == 'file'}
    pub.require(new_files == accepted['files'], 'New image entries differ from accepted file map')
    for key in ('helper', 'inventory_helper', 'installer'):
        binding = preparation[key]
        pub.require(pub.digest(ROOT/binding['path']) == binding['sha256'], 'Changed helper/source: '+key)
    private_live_checked = False
    if live:
        pub.require(pub.tree(pub.DESTINATION) == transition['new_tree'], 'Current main image differs')
        for key in ('independent_old_prefix_backup', 'transaction_old_prefix_backup'):
            pub.require(pub.tree(ROOT/transition[key]) == transition['old_tree'], 'Old800 backup differs: '+key)
        pub.require(pub.alias_image(pub.PREFIX/'lib') == transition['aliases_after'], 'Main aliases differ')
        pub.require(pub.alias_image(ROOT/transition['independent_old_alias_backup']) == transition['aliases_before'],
                    'Independent old common aliases differ')
        alias_journal = ROOT/transition['retained_alias_stage']/'previous-links'
        for name, entry in transition['aliases_before'].items():
            if entry['kind'] == 'symlink':
                pub.require(pub.alias_image(alias_journal)[name] == entry, 'Transaction alias journal differs: '+name)
        source = ROOT/accepted['installed']
        if source.is_dir():
            pub.require(pub.tree(source) == transition['new_tree'], 'Private installed image differs')
            private_live_checked = True
        # Relocation qualification may deliberately hide the private prefix;
        # its source archive remains present and the main/two old images above
        # are always mandatory. The initial live receipt records visibility.
        for name, sha in freeze['files'].items():
            for base in (ROOT/'vendor/ninjars', source.parents[1]/'vendor/ninjars'):
                pub.require(pub.digest(base/name) == sha, 'Final45 source changed: '+str(base/name))
    return {'schema': 'main-ninjars-final-pin-verification-v1', 'passed': True, 'live': live,
            'transition_sha256': pub.digest(HERE/'transition.json'), 'installed_files': 10,
            'bridge_source_files': 45, 'main_archive_sha256': pub.NEW_ARCHIVE,
            'main_config_sha256': pub.NEW_CONFIG, 'main_lock_sha256': pub.NEW_LOCK,
            'old_archive_sha256': pub.OLD_ARCHIVE, 'old_prefix_backups': 2,
            'private_live_checked': private_live_checked, 'engine_calls': 0, 'compiler_calls': 0,
            'qualification_scope': transition['qualification_scope']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    print(json.dumps(verify(parser.parse_args().live), indent=2, sort_keys=True))
