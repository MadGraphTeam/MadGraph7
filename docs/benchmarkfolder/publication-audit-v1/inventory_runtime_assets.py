#!/usr/bin/env python3
"""After quiet timing, inventory only bound metadata and explicit local ELFs.

No source, ignore rule, process, or existing evidence is modified. The one output
JSON is created exclusively; reruns must choose a new output path.
"""
import argparse
import collections
import datetime
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAMESPACES = ('runtime-bindings-abi3-tensor-factor-v1',
              'runtime-bindings-controls-abi3-tensor-factor-v1')
BASENAMES = ('fallback', 'traps', 'timing', 'oracle-check', 'guards', 'sticky', 'zero-work', 'kernel')


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inventory():
    before = git('rev-parse', 'HEAD').decode().strip()
    committed = {name for name in git('ls-tree', '-r', '--name-only', '-z', 'HEAD').decode().split('\0') if name}
    indexed = {name for name in git('ls-files', '-z').decode().split('\0') if name}
    groups = []
    for name in NAMESPACES:
        prefix = 'docs/benchmarkfolder/policy-v2/'+name+'/'
        asset_prefix = prefix+'runtime-assets/'
        references = {}
        labels = collections.defaultdict(set)
        records = []
        def walk(value, label=''):
            if isinstance(value, dict):
                if isinstance(value.get('path'), str) and value['path'].startswith(asset_prefix):
                    path = value['path']
                    digest = value['sha256']
                    if path in references and references[path] != digest:
                        raise RuntimeError('Conflicting asset reference: '+path)
                    references[path] = digest
                    labels[path].add(label)
                for key, child in value.items():
                    walk(child, key)
            elif isinstance(value, list):
                for child in value:
                    walk(child, label)
        for path in sorted(p for p in committed if p.startswith(prefix) and p.endswith('.json') and '/runtime-assets/' not in p):
            raw = (ROOT/path).read_bytes()
            if git('show', before+':'+path) != raw:
                raise RuntimeError('Existing committed binding was modified: '+path)
            walk(json.loads(raw))
            records.append({'path': path, 'sha256': sha(raw)})
        if len(references) != 68:
            raise RuntimeError('Expected 68 exact runtime assets in '+name)
        assets = []
        for path, digest in sorted(references.items()):
            item = ROOT/path
            if not re.fullmatch('[0-9a-f]{64}', item.name) or item.name != digest or item.is_symlink():
                raise RuntimeError('Invalid content-addressed asset: '+path)
            raw = item.read_bytes()
            if sha(raw) != digest:
                raise RuntimeError('Changed runtime asset: '+path)
            assets.append({'path': path, 'sha256': digest, 'bytes': len(raw),
                           'metadata_names': sorted(labels[path]),
                           'fortran_module_metadata': any(label.endswith('.mod') for label in labels[path]),
                           'committed_at_snapshot': path in committed,
                           'indexed_at_snapshot': path in indexed})
        groups.append({'namespace': name, 'binding_records': records, 'assets': assets,
                       'files': len(assets), 'bytes': sum(item['bytes'] for item in assets),
                       'module_records': sum(item['fortran_module_metadata'] for item in assets)})
    excludes = []
    for line in (ROOT/'.git/info/exclude').read_text().splitlines():
        if not line.startswith('/docs/benchmarkfolder/'):
            continue
        path = ROOT/line[1:]
        if path.is_file():
            with path.open('rb') as stream:
                magic = stream.read(4)
            if magic != b'\x7fELF':
                raise RuntimeError('Local executable exclusion changed kind: '+line)
            if path.name not in BASENAMES:
                raise RuntimeError('Unexpected executable basename: '+line)
            excludes.append({'path': line[1:], 'bytes': path.stat().st_size,
                             'basename': path.name, 'kind': 'ELF',
                             'committed': line[1:] in committed,
                             'portable_rule': '/docs/benchmarkfolder/experiments/**/'+path.name})
    rules = ['/docs/benchmarkfolder/experiments/**/'+name for name in BASENAMES]
    if any(not fnmatch.fnmatchcase(row['path'], row['portable_rule'].lstrip('/')) for row in excludes):
        raise RuntimeError('Portable rule does not cover its exact executable')
    after = git('rev-parse', 'HEAD').decode().strip()
    if before != after:
        raise RuntimeError('HEAD changed during audit; repeat in a new output namespace')
    return {'schema': 'publication-runtime-assets-audit-v1',
            'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'revision': before, 'asset_groups': groups,
            'asset_files': sum(group['files'] for group in groups),
            'asset_bytes': sum(group['bytes'] for group in groups),
            'local_executable_exclusions': excludes,
            'portable_ignore_proposal': rules,
            'scope': 'Only already committed runtime binding metadata and explicit local executable exclusions; no cleanup or numerical evaluation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runtime-assets.json')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refuse to overwrite an existing audit output')
    result = inventory()
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'asset_files': result['asset_files'], 'asset_bytes': result['asset_bytes'],
                      'local_executable_exclusions': len(result['local_executable_exclusions']),
                      'output': str(args.output)}, indent=2))
