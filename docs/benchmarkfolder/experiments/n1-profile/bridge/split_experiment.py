#!/usr/bin/env python3
"""Build an isolated direction-separated DP-workspace experiment."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[5]
ORIGINAL = ROOT / '.tirust-build/n1-profile/bridge'
BUILD = ROOT / '.tirust-build/n1-profile/split-bridge'
EVIDENCE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    assert Path.cwd().resolve() == ROOT
    original_metadata = json.loads((ORIGINAL / 'build-metadata.json').read_text())
    for name, expected in original_metadata['input_sha256'].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in original_metadata['source_sha256'].items():
        assert digest(ROOT / name) == expected, name
    assert digest(ORIGINAL / 'libmadloop_tirust.so') == original_metadata['library_sha256']
    before = (ORIGINAL / 'src/lib.rs').read_text()
    after = before
    changes = [
        ('dp: HashMap<Key, TensorWorkspace<f64>>,',
         'dp: HashMap<(Key, bool), TensorWorkspace<f64>>,'),
        ('        if !self.dp.contains_key(&key) {',
         '        let workspace_key = (key, mode.check);\n        if !self.dp.contains_key(&workspace_key) {'),
        ('self.dp.insert(key, workspace);', 'self.dp.insert(workspace_key, workspace);'),
        ('self.dp.get_mut(&key).unwrap()', 'self.dp.get_mut(&workspace_key).unwrap()'),
        ('                self.dp.remove(&key);',
         '                self.dp.remove(&(key, false));\n                self.dp.remove(&(key, true));'),
    ]
    for old, new in changes:
        assert after.count(old) == 1, old
        after = after.replace(old, new)
    (BUILD / 'src').mkdir(parents=True, exist_ok=True)
    (BUILD / 'src/lib.rs').write_text(after)
    shutil.copyfile(ORIGINAL / 'src/profile.rs', BUILD / 'src/profile.rs')
    diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='original-instrumented/lib.rs', tofile='split-instrumented/lib.rs'))
    (EVIDENCE / 'split-workspaces.patch').write_text(diff)
    command = [arg.replace(str(ORIGINAL), str(BUILD)) for arg in original_metadata['command']]
    metadata = {
        'scope': 'Isolated full-DP workspaces separated by check_pass; all other behavior and timers unchanged.',
        'production_modified': False,
        'original_build_metadata_sha256': digest(ORIGINAL / 'build-metadata.json'),
        'original_library_sha256': original_metadata['library_sha256'],
        'input_sha256': original_metadata['input_sha256'],
        'source_sha256': {str(path.relative_to(ROOT)): digest(path) for path in
            (Path(__file__).resolve(), BUILD / 'src/lib.rs', BUILD / 'src/profile.rs',
             EVIDENCE / 'split-workspaces.patch')},
        'command': command,
        'rustc': original_metadata['rustc'],
        'failure_policy': 'A failed full-DP call clears the full store and removes both direction workspaces.',
        'check_policy': 'The independent check workspace still receives no tensor store.',
    }
    (BUILD / 'preparation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    shutil.copyfile(BUILD / 'preparation.json', EVIDENCE / 'split-preparation.json')
    if not args.build:
        return
    with (BUILD / 'build.log').open('w') as log:
        log.write(json.dumps(command) + '\n'); log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    shutil.copyfile(BUILD / 'build.log', EVIDENCE / 'split-build.log')
    if result.returncode:
        raise SystemExit(f'Compiler failed ({result.returncode}), inspect {BUILD / "build.log"}')
    metadata['library_sha256'] = digest(BUILD / 'libmadloop_tirust.so')
    (BUILD / 'build-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    shutil.copyfile(BUILD / 'build-metadata.json', EVIDENCE / 'split-build-metadata.json')
    print(BUILD / 'libmadloop_tirust.so')


if __name__ == '__main__':
    main()
