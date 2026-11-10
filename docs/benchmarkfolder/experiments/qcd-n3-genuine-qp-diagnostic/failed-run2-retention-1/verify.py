#!/usr/bin/env python3
"""Replay only the closed loader-only failure. Never open original paths."""
from pathlib import Path
import hashlib
import json
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify(root=None):
    root = Path(root or Path(__file__).resolve().parent)
    index = json.loads((root / 'evidence-index.json').read_bytes())
    payload = {}
    observed = set()
    for p in root.rglob('*'):
        require(not p.is_symlink(), 'Symlink in retained package')
        if p.is_file():
            observed.add(p.relative_to(root).as_posix())
    require(observed == set(index['files']) | {'evidence-index.json'}, 'Package membership changed')
    for name, item in index['files'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and path.as_posix() == name,
                'Unsafe package member')
        raw = (root / path).read_bytes()
        require(len(raw) == item['bytes'] and digest(raw) == item['sha256'], 'Changed payload: ' + name)
        payload[name] = raw
    j = lambda name: json.loads(payload[name])
    result = j('actual-run2/result.json')
    receipt = j('actual-run2/loader.json')
    build = j('inherited-build2.json')
    binding = j('inherited-source2-bindings.json')
    identities = j('retention-inventory.json')
    require(result['passed'] is False and result['workers_completed'] == 0, 'Unexpected worker result')
    require(result['failure']['type'] == 'ValueError' and
            result['failure']['message'].startswith('Loader closure differs from closed binding: '),
            'Unexpected failure phase')
    require(result['source_index_sha256'] == digest(payload['inherited-source2-index.json']),
            'Source index mismatch')
    require(result['build_sha256'] == digest(payload['inherited-build2.json']) and build['passed'] is True,
            'Inherited build mismatch')
    require(build['source_index_sha256'] == result['source_index_sha256'], 'Build/source mismatch')
    require(receipt['native'] is False and receipt['returncode'] == 0 and
            receipt['process_spawned'] is True and receipt['process_reaped'] is True and
            receipt.get('failure') is None and not receipt.get('timed_out'), 'Loader trace did not close cleanly')
    require(receipt['seconds_cap'] <= 30 and receipt['argv'][1:3] == ['-c', '12'] and
            receipt['argv'][4:6] == ['--as=4294967296', '--'], 'Resource policy changed')
    require(receipt['target_loader_environment_at_execve'] ==
            {'LD_LIBRARY_PATH': binding['native_library_path'], 'LD_TRACE_LOADED_OBJECTS': '1'},
            'Target environment changed')
    require(build['executable'] in receipt['argv'] and build['executable_sha256'] in receipt['argv'] and
            binding['target_interpreter'] in receipt['argv'], 'Unexpected traced executable')
    require(identities['actual_attempt_files'] == ['loader.json', 'loader.log', 'result.json'],
            'Attempt was not loader-only')
    raw_paths = set(re.findall(r'/[^\s()]+', payload['actual-run2/loader.log'].decode()))
    require('not found' not in payload['actual-run2/loader.log'].decode() and len(raw_paths) == 15,
            'Missing/extra traced runtime')
    aliases = identities['resolved_aliases_at_retention']
    require(raw_paths <= set(aliases) and set(binding['runtime']) <= set(aliases), 'Missing retained alias identity')
    actual = {aliases[p] for p in raw_paths}
    expected = {aliases[p] for p in binding['runtime']}
    extra, missing = sorted(actual - expected), sorted(expected - actual)
    require(len(extra) == len(missing) == 3, 'Failure was not exactly three compiler-support substitutions')
    require(result['failure']['message'] ==
            'Loader closure differs from closed binding: ' + repr((extra, missing)), 'Failure replay differs')
    for original, record in identities['original_copies'].items():
        require(record['member'] in payload and digest(payload[record['member']]) == record['sha256'] and
                len(payload[record['member']]) == record['bytes'], 'Captured original differs: ' + original)
    replay = {'passed': True, 'scope': 'retention verification, not numerical qualification',
              'loader_trace_executed': True, 'loader_trace_returncode': 0,
              'strict_loader_validation_passed': False, 'attempted_workers': 0,
              'diagnostic_calls': 0, 'bootstrap_calls': 0, 'scientific_results': False,
              'actual_runtime_count': 15, 'compiler_support_substitutions': 3,
              'unexpected_actual_paths': extra, 'missing_expected_paths': missing}
    require(replay == j('summary.json'), 'Summary replay differs')
    return replay


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True, indent=2))
