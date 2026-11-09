"""Bind the separately executed final native replay into installer evidence."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path.cwd()
report = root / 'docs/benchmarkfolder/qualification/tirust-install/current-source'
final_lib = root / '.tirust-build/current-install/mg/HEPTools/tirust/lib'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


execution = json.loads((report / 'replay-execution.json').read_text())
require(execution['returncode'] == 0 and execution['state'] == 'completed',
        'Replay execution did not complete successfully')
require(execution['launcher_sha256'] == sha(report / 'run_final_replay.py'),
        'Replay launcher changed after execution')
require(execution['runner_sha256'] == sha(report / 'validate_release.py'),
        'Replay runner changed after execution')
validation = json.loads((report / 'release-validation.json').read_text())
require(validation['source_sha256'] == execution['runner_sha256'],
        'Validation source does not match executed runner')
require(validation['comparison']['passed'] and
        validation['comparison']['cache_statistics_identical'], 'Final native replay failed')
require(validation['call_counts'] == {'original': 90, 'split': 90}, 'Wrong final replay size')
checks = json.loads((report / 'check-installation.json').read_text())
before = json.loads((report / 'production-before.json').read_text())
original_hash = before['HEPTools/tirust/lib/libmadloop_tirust.so']
final_hash = checks['native_artifacts']['libmadloop_tirust.so']['sha256']
for label, expected, execution_key in [('original', original_hash, 'original_library_sha256'),
                                        ('split', final_hash, 'final_library_sha256')]:
    require(execution[execution_key] == expected and
            validation['identities'][label]['library_sha256'] == expected,
            'Executed/validated library identity differs for ' + label)
for name, record in checks['native_artifacts'].items():
    require(sha(final_lib / name) == record['sha256'],
            'Final native artifact changed after replay: ' + name)
for name, record in checks['notice_files'].items():
    require(sha(final_lib / name) == record['sha256'],
            'Final dependency notice changed after replay: ' + name)
require(sha(final_lib / 'tirust-config.json') == sha(report / 'tirust-config.json'),
        'Final installed metadata changed after replay')
require(all(sha(root / name) == expected for name, expected in before.items()
            if name != 'input/mg7_configuration.txt'),
        'Protected production native hash changed during replay')
workers = {}
for label, item in validation['workers'].items():
    require(item['returncode'] == 0, 'Failed worker: ' + label)
    for field in ('result', 'log'):
        path = (root / item[field]).resolve()
        require(path.is_relative_to(report.resolve()), 'Worker artifact escaped evidence folder')
        require(sha(path) == item[field + '_sha256'],
                'Worker artifact changed after validation: ' + item[field])
    worker = json.loads((root / item['result']).read_text())
    require(worker['state'] == 'complete', 'Incomplete worker: ' + label)
    require(worker['identity'] == validation['identities'][label],
            'Worker identity differs from validation: ' + label)
    workers[label] = worker
require(workers['original']['events'] == workers['split']['events'],
        'Full event/cache records differ')
require(len(workers['original']['events']) == 110, 'Unexpected event count')
require(workers['original']['output_payloads'] == workers['split']['output_payloads'],
        'Native output payloads differ')
outcomes = {}
for label, worker in workers.items():
    calls = [event for event in worker['events'] if event['operation'] == 'evaluate_dp']
    successful = sum(event['status'] == 0 for event in calls)
    failures = [event for event in calls if event['status'] != 0]
    require(successful == 80 and len(failures) == 10 and
            all(event['expected'] in ('error', 'failure') and
                event['failure_sentinel_unchanged'] for event in failures),
            'Unexpected native success/failure cohort: ' + label)
    outcomes[label] = {'successful_evaluations': successful,
                       'deliberate_invalid_input_or_overflow_calls': len(failures)}
checks['final_native_replay'] = {
    'record': 'release-validation.json', 'calls_per_library': 90,
    'event_records_identical': len(workers['original']['events']),
    'call_outcomes': outcomes,
    'output_bytes_identical': True, 'cache_snapshots_identical': True,
    'worker_and_log_hashes_verified': True,
    'execution_source_and_library_identities_verified': True,
    'installed_native_metadata_and_notice_hashes_rechecked': True,
    'scope': 'Synthetic raw-ABI DP lifecycle/cache/failure parity only; no full-amplitude '
             'or timing claim. The actual installer smoke separately covers binary128 input and pole mode.'}
write(report / 'check-installation.json', checks)
shutil.copy2(Path(__file__), report / 'bind_replay_evidence.py')
readme = report / 'README.md'
previous = readme.read_text().split('\nThe published package also passed a separate serialized raw-ABI replay', 1)[0].replace(
    'The separate 90-call replay is held until recovery;',
    'The separate 90-call replay was held until root confirmed recovery;')
readme.write_text(previous + '\nThe published package also passed a separate serialized raw-ABI '
    'replay against the unchanged original `49af` release: 90 calls per library and 110 complete '
    'event records, output bytes and cache snapshots identical. Each library served 80 valid evaluations; '
    '10 deliberate invalid-input or overflow calls returned their expected failures with output sentinels untouched. '
    '`release-validation.json` binds '
    'both actual library hashes. This synthetic DP lifecycle/cache/failure check adds no '
    'complete-amplitude or timing claim. The frozen replay runner was invoked with explicit '
    'final-library and fresh output paths; original qualification evidence was not changed. '
    '`replay-execution.json` records the explicit administrative recovery release and actual '
    'command, preserving the earlier hold evidence. Worker/log digests, executed source and '
    'library identities, and the current installed artifact hashes were checked before indexing.\n')
index_path = report / 'index.json'
index = json.loads(index_path.read_text())
index['final_native_replay'] = checks['final_native_replay']
index['artifacts'] = {
    path.relative_to(report).as_posix(): {'sha256': sha(path), 'bytes': path.stat().st_size}
    for path in sorted(report.rglob('*')) if path.is_file() and path != index_path}
write(index_path, index)
outer = report.parent / 'index.json'
data = json.loads(outer.read_text())
for path in sorted(report.rglob('*')):
    if path.is_file():
        data['artifacts'][path.relative_to(report.parent).as_posix()] = {
            'sha256': sha(path), 'size': path.stat().st_size}
write(outer, data)
print(json.dumps(checks['final_native_replay'], indent=2))
