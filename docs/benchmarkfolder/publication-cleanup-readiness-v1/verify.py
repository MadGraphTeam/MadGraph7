#!/usr/bin/env python3
"""Check retained cleanup readiness; never scan processes or remove files."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def need(value, reason):
    if not value:
        raise RuntimeError(reason)


def check(record):
    path = ROOT/record['path']
    need(path.is_file() and not path.is_symlink(), 'Required ordinary evidence is missing: '+record['path'])
    need(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
         'Required retained evidence changed: '+record['path'])


def verify():
    manifest = json.loads((HERE/'manifest.json').read_text())
    files = manifest['files']
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    need(identity == manifest['identity_sha256'], 'Cleanup audit manifest identity changed')
    actual = {str(path.relative_to(HERE)) for path in HERE.rglob('*') if path.is_file() and path.name != 'manifest.json'}
    need(actual == set(files), 'Cleanup audit file set changed')
    for name, item in files.items():
        path = HERE/name
        need(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Cleanup audit changed: '+name)
    data = json.loads((HERE/'result.json').read_text())
    need(data['cleanup_performed'] is False and data['source_changed'] is False
         and data['engine_calls'] == data['compiler_calls'] == 0, 'Audit scope changed')
    need(data['candidate_count'] == len(data['candidates']) == 9, 'Expected nine exact candidates')
    need(data['candidate_bytes'] == sum(row['bytes'] for row in data['candidates']) == 172173207,
         'Candidate byte counts changed')
    check(data['native_materialization_index'])
    check(data['required_installation_manifest'])
    duplicates = [row for row in data['candidates'] if row['kind'] == 'materialized-plaintext-duplicate']
    scratch = [row for row in data['candidates'] if row['kind'] == 'completed-regression-comparison-scratch']
    need(len(duplicates) == 6 and len(scratch) == 3, 'Exact candidate classes changed')
    for row in duplicates:
        check(row['compressed_evidence'])
        check(row['native_manifest'])
        plain = ROOT/row['path']
        if plain.exists():
            need(plain.is_file() and not plain.is_symlink() and plain.stat().st_size == row['bytes']
                 and sha(plain) == row['sha256'], 'Existing materialization changed: '+row['path'])
    for row in scratch:
        check(row['retained_proof'])
    need((HERE/'cleanup-candidate-paths.txt').read_text() == ''.join(row['path']+'\n' for row in data['candidates']),
         'Exact cleanup candidate list changed')
    need((HERE/'proposed-native-ignore-rules.txt').read_text() == ''.join(rule+'\n' for rule in data['proposed_exact_native_ignore_rules']),
         'Exact ignore recommendation changed')
    return {'status': 'verified', 'identity_sha256': identity, 'candidate_count': 9,
            'candidate_bytes': 172173207, 'native_plaintext_duplicates': 6,
            'completed_regression_scratch_roots': 3, 'cleanup_performed': False,
            'process_scan_rerun': False, 'scope': 'Historical readiness only; root must decide cleanup and recheck live use'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True))
