#!/usr/bin/env python3
"""Bind the byte-exact root publication to its original successful CLI proof.

Read/copy evidence only: no installer, compiler, numerical executable or update.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'madgraph/__init__.py').is_file())
HERE = Path(__file__).resolve().parent
PRIVATE = ROOT / 'docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1'
TRANSITION = ROOT / 'docs/benchmarkfolder/policy-v2/main-install-publication-v1/transition.json'
UNIT_PROOF = ROOT / 'docs/benchmarkfolder/policy-v2/final-rust-abi3-v1/proof-v2/unit-proof.json'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
            'bytes': path.stat().st_size}


def main():
    targets = ('installation.json', 'install.mg5', 'install-cli.log')
    require(not any((HERE / name).exists() for name in targets),
            'Preserve the existing publication-derived record; use a new namespace')
    source_path = PRIVATE / 'installation.json'
    source = json.loads(source_path.read_text())
    transition = json.loads(TRANSITION.read_text())
    proof = json.loads(UNIT_PROOF.read_text())
    require(transition['schema'] == 'main-ninjars-publication-transition-v1'
            and transition['status'] == 'published'
            and transition['main_destination'] == 'HEPTools/ninjars',
            'The explicit root main-prefix publication is not complete')
    require(source['revision'] == transition['installation_revision'] == proof['revision']
            and source['archived_bridge_source_sha256'] == proof['bridge_source_sha256']
            and len(proof['bridge_source_sha256']) == 44,
            'Private CLI, main publication and final Rust source do not agree')
    require(source['exit_code'] == 0 and source['passed'] is True
            and source['compiled_smoke_passed'] is True and source['abi_version'] == 3
            and source['reference_source_overrides'] is False and proof['exit_code'] == 0,
            'Original actual-CLI/Rust validation did not pass')
    require(source['files'] == transition['main_artifact_sha256']
            and len(source['files']) == 10,
            'Main publication does not contain all ten exact private installation assets')
    destination = ROOT / transition['main_destination']
    for name, expected in source['files'].items():
        require(digest(destination / name) == expected,
                'Published main installation asset changed: ' + name)
    old_manifest = transition['private_proof_manifest']
    require(digest(ROOT / old_manifest['path']) == old_manifest['sha256'],
            'Original private proof manifest changed')
    copies = {}
    for name, key in (('install.mg5', 'cli_card_sha256'), ('install-cli.log', 'cli_log_sha256')):
        original = PRIVATE / name
        require(digest(original) == source[key], 'Original actual-CLI evidence changed: ' + name)
        shutil.copy2(original, HERE / name)
        copies[name] = bound(HERE / name)
    derived = copy.deepcopy(source)
    derived['installed'] = transition['main_destination']
    derived['publication_provenance'] = {
        'schema': 'publication-derived-installation-v1',
        'original_installation': bound(source_path),
        'original_installed_path': source['installed'],
        'transition': bound(TRANSITION),
        'final_unit_proof': bound(UNIT_PROOF),
        'derivation_helper': bound(Path(__file__).resolve()),
        'byte_exact_all_ten_assets': True,
        'inherited_fields': 'All original CLI fields unchanged except installed path; command/time/smoke refer to original private invocation',
        'exact_copied_cli_evidence': copies,
        'new_cli_invocations': 0, 'compiler_calls': 0, 'engine_calls': 0,
    }
    with (HERE / 'installation.json').open('x') as stream:
        json.dump(derived, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'installation': bound(HERE / 'installation.json'),
                      'archive_sha256': source['files']['lib/libninjars.a'],
                      'new_cli_invocations': 0, 'compiler_calls': 0, 'engine_calls': 0}))


if __name__ == '__main__':
    main()
