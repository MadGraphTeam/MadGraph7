#!/usr/bin/env python3
"""Verify sealed publication inventory without requiring disposable executables."""
import collections
import fnmatch
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify():
    manifest = json.loads((HERE/'manifest.json').read_text())
    files = manifest['files']
    identity = digest(json.dumps(files, sort_keys=True, separators=(',', ':')).encode())
    require(identity == manifest['identity_sha256'], 'Audit manifest identity changed')
    actual = {str(path.relative_to(HERE)) for path in HERE.rglob('*')
              if path.is_file() and path.name != 'manifest.json'}
    require(actual == set(files), 'Audit file set changed')
    for name, item in files.items():
        data = (HERE/name).read_bytes()
        require(len(data) == item['bytes'] and digest(data) == item['sha256'],
                'Audit evidence changed: '+name)
    spec = json.loads((HERE/'runtime-assets.json').read_text())
    require(spec['schema'] == 'publication-runtime-assets-audit-v1', 'Unknown inventory schema')
    groups = spec['asset_groups']
    require(len(groups) == 2, 'Expected exactly two binding namespaces')
    assets = []
    for group in groups:
        require(group['files'] == len(group['assets']) == 68, 'Expected 68 assets per namespace')
        require(group['bytes'] == sum(row['bytes'] for row in group['assets']) == 416123,
                'Runtime metadata byte count changed')
        require(group['module_records'] == sum(row['fortran_module_metadata'] for row in group['assets']) == 65,
                'Runtime module metadata count changed')
        for record in group['binding_records']:
            current = (ROOT/record['path']).read_bytes()
            require(digest(current) == record['sha256'], 'Binding changed: '+record['path'])
            historical = subprocess.check_output(
                ['git', 'show', spec['revision']+':'+record['path']], cwd=ROOT)
            require(current == historical, 'Recorded committed binding does not match checkpoint')
        for row in group['assets']:
            path = ROOT/row['path']
            require(not path.is_symlink(), 'Runtime asset must be an ordinary file: '+row['path'])
            data = path.read_bytes()
            require(len(data) == row['bytes'] and digest(data) == row['sha256'] == path.name,
                    'Runtime metadata changed: '+row['path'])
            require(not row['committed_at_snapshot'] and not row['indexed_at_snapshot'],
                    'Initial publication disposition changed')
            assets.append(row)
    require(len(assets) == spec['asset_files'] == 136, 'Asset inventory is incomplete')
    require(sum(row['bytes'] for row in assets) == spec['asset_bytes'] == 832246,
            'Total runtime metadata bytes changed')
    expected_paths = ''.join(row['path']+'\n' for row in assets)
    require((HERE/'runtime-asset-paths.txt').read_text() == expected_paths,
            'Exact asset staging list changed')
    executables = spec['local_executable_exclusions']
    require(len(executables) == 195 and len({row['path'] for row in executables}) == 195,
            'Executable exclusion inventory changed')
    require(not any(row['committed'] for row in executables), 'Inventory contains a tracked executable')
    rules = spec['portable_ignore_proposal']
    require(len(rules) == 8, 'Expected eight portable ignore rules')
    for row in executables:
        require(row['kind'] == 'ELF' and row['portable_rule'] in rules,
                'Unclassified local executable')
        require(fnmatch.fnmatchcase(row['path'], row['portable_rule'].lstrip('/')),
                'Portable rule does not cover '+row['path'])
    require((HERE/'local-executable-paths.txt').read_text() ==
            ''.join(row['path']+'\n' for row in executables), 'Exact executable list changed')
    require((HERE/'portable-ignore-rules.txt').read_text() == ''.join(rule+'\n' for rule in rules),
            'Portable ignore proposal changed')
    return {'status': 'verified', 'manifest_identity_sha256': identity,
            'files': len(files), 'bytes': sum(item['bytes'] for item in files.values()),
            'runtime_assets': len(assets), 'runtime_asset_bytes': 832246,
            'runtime_module_metadata_records': sum(row['fortran_module_metadata'] for row in assets),
            'local_ELF_exclusions': 195,
            'ELF_basename_counts': dict(sorted(collections.Counter(row['basename'] for row in executables).items())),
            'portable_ignore_rules': len(rules),
            'scope': 'No cleanup, process inspection, numerical evaluation or compiler invocation'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True))
