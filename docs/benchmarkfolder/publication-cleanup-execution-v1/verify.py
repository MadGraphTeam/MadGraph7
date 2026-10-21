#!/usr/bin/env python3
"""Check immutable cleanup evidence and durable inputs; no deletion or signals."""
import hashlib
import importlib.util
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


def verify():
    manifest = json.loads((HERE/'manifest.json').read_text())
    files = manifest['files']
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    need(identity == manifest['identity_sha256'], 'Cleanup execution manifest identity changed')
    actual = {str(path.relative_to(HERE)) for path in HERE.rglob('*') if path.is_file() and path.name != 'manifest.json'}
    need(actual == set(files), 'Cleanup execution evidence set changed')
    for name, item in files.items():
        path = HERE/name
        need(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Cleanup evidence changed: '+name)
    result = json.loads((HERE/'cleanup-result.json').read_text())
    journal = json.loads((HERE/'cleanup-started.json').read_text())
    need(result['status'] == 'completed' and result['removed_count'] == len(result['removed']) == 9,
         'Exact authorized cleanup did not finish')
    need(result['removed_bytes'] == sum(row['bytes'] for row in result['candidates_before']) == 172173207,
         'Recorded cleanup byte total changed')
    need(result['removed'] == [row['path'] for row in result['candidates_before']], 'Removed-path inventory changed')
    need(result['durable_proofs_before'] == result['durable_proofs_after'], 'Durable proof changed during cleanup')
    need(not result['fresh_same_uid_process_observation']['uses'] and not result['known_completed_workers']['matches'],
         'Observed process use was not cleared')
    need(result['engine_calls'] == result['compiler_calls'] == 0 and result['preexisting_vendor_StdHEP_fail_touched'] is False,
         'Cleanup exceeded its recorded scope')
    for key, value in journal.items():
        if key != 'removed':
            need(result[key] == value, 'Before-deletion journal differs: '+key)
    need(journal['removed'] == [], 'Before-deletion journal was rewritten')
    module_spec = importlib.util.spec_from_file_location('ninjars_exact_cleanup_helpers', HERE/'cleanup.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    native = ROOT/'docs/benchmarkfolder/experiments/ninjars-native-float256-v1'
    native_manifest = json.loads((native/'manifest.json').read_text())
    durable = [module.check_seal(native, native_manifest['materialized_plaintext_exclusions']),
               module.check_seal(ROOT/'docs/benchmarkfolder/policy-v2/final-regressions-v1')]
    need(durable == result['durable_proofs_after'], 'Preserved native/regression evidence changed')
    deferral = json.loads((HERE/'ignore-deferral.json').read_text())
    need(deferral['cleanup_result_unchanged_sha256'] == sha(HERE/'cleanup-result.json'), 'Cleanup result was rewritten')
    need(deferral['deferred_patch_sha256'] == sha(HERE/'deferred-native-ignore.patch'), 'Deferred exact patch changed')
    need(deferral['restored_ignore_sha256'] == result['gitignore_before_sha256']
         and deferral['other_ignore_lines_changed'] is False, 'Ignore restoration record changed')
    return {'status': 'verified', 'identity_sha256': identity,
            'removed_paths': 9, 'reclaimed_bytes': 172173207,
            'currently_absent_removed_paths': sum(not (ROOT/name).exists() for name in result['removed']),
            'native_and_regression_proofs_unchanged': True,
            'exact_ignore_rules_deferred': len(deferral['exact_rules']),
            'observed_permission_denied': result['fresh_same_uid_process_observation']['permission_denied'],
            'process_scan_rerun': False, 'cleanup_rerun': False, 'engine_calls': 0}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True))
