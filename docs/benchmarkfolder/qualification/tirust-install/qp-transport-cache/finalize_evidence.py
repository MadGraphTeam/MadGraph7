"""Freeze successful package qualification after all native workers have exited."""
import datetime
import hashlib
import json
import runpy
from pathlib import Path
import shutil

ROOT = Path.cwd()
WORK = ROOT / '.tirust-build/qp-transport-install'
REPORT = ROOT / 'docs/benchmarkfolder/qualification/tirust-install/qp-transport-cache'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def require(value, message):
    if not value:
        raise RuntimeError(message)


checks = read(REPORT / 'check-installation.json')
prefix = Path(checks['prefix'])
before = read(WORK / 'protected-before.json')
final = {}
for label in ('main49af', 'previous095e'):
    base = Path(before[label]['prefix'])
    files = {str(p.relative_to(base)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(base.rglob('*')) if p.is_file()}
    require(files == before[label]['files'], 'Protected native tree changed after qualification: ' + label)
    final[label] = {'prefix': str(base), 'files': files}
for label in ('main', 'previous095e'):
    entry = before[label + '-configuration']
    path = Path(entry['path'])
    require(sha(path) == entry['sha256'], 'Protected configuration changed after qualification')
    final[label + '-configuration'] = {'path': str(path), 'sha256': sha(path),
                                      'mtime_ns': path.stat().st_mtime_ns}
for group in ('native_artifacts', 'notice_files'):
    for name, record in checks[group].items():
        require(sha(prefix / 'lib' / name) == record['sha256'], 'Published artifact changed: ' + name)
require(sha(prefix / 'lib/tirust-config.json') == checks['metadata_sha256'], 'Published metadata changed')
require(sha(prefix / 'include/mtirust.f90') == checks['fortran_source_sha256'], 'Published wrapper changed')
require(sha(prefix / 'Cargo.lock') == checks['build_provenance']['lock_sha256'], 'Published lock changed')
require(sha(prefix / 'Cargo.toml') == checks['build_provenance']['manifest_sha256'], 'Published manifest changed')
require(sha(WORK / 'mg/input/mg7_configuration.txt') == sha(REPORT / 'isolated-configuration-after.txt'),
        'Published private configuration changed during qualification')
for name, record in read(REPORT / 'published-aliases.json').items():
    path = Path(record['path'])
    require(path.is_symlink() == record['is_symlink'] and str(path.resolve()) == record['resolved']
            and sha(path) == record['sha256'], 'Published alias changed: ' + name)
source = read(WORK / 'source-inventory.json')
require(all(sha(WORK / 'mg' / name) == expected for name, expected in source.items()), 'Source snapshot changed')
write(REPORT / 'protected-final.json', final)
archive = read(REPORT / 'archive-tests/results.json')
require(archive['state'] == 'complete' and archive['finite_underflow_logs_identical'], 'Archive tests incomplete')
for name, result in archive['results'].items():
    if not name.endswith('real-provider'):
        require(result['passed'], 'Archive check failed: ' + name)
for command in archive['commands']:
    require(sha(REPORT / command['log']) == command['log_sha256'], 'Archive log changed')
    if 'executable_sha256' in command:
        require(command['executable_sha256'] == command['executable_sha256_after'] ==
                sha(Path(command['command'][0])), 'Executed archive driver changed')
for name, expected in archive['source_sha256'].items():
    require(sha(REPORT / 'archive-tests' / name) == expected, 'Archive source changed')
for label, identity in archive['identities'].items():
    for name, expected in identity['artifacts'].items():
        require(sha(Path(identity['prefix']) / name) == expected, 'Archive identity changed: ' + label)
execution = read(REPORT / 'dp-replay-execution.json')
require(execution['state'] == 'complete' and execution['returncode'] == 0, 'DP replay incomplete')
require(execution['runner_sha256'] == sha(REPORT / 'validate_release.py'), 'DP runner changed')
require(execution['source_sha256'] == sha(WORK / 'run_dp_replay.py'), 'DP launcher changed')
dp = read(REPORT / 'release-validation.json')
require(dp['state'] == 'complete' and dp['comparison']['passed'] and
        dp['comparison']['cache_statistics_identical'], 'DP parity failed')
require(dp['call_counts'] == {'original': 90, 'split': 90}, 'Unexpected DP cohort')
workers = {}
for label, record in dp['workers'].items():
    require(record['returncode'] == 0, 'DP worker failed')
    for field in ('result', 'log'):
        path = (ROOT / record[field]).resolve()
        require(path.is_relative_to(REPORT.resolve()) and sha(path) == record[field + '_sha256'],
                'DP worker artifact changed')
    worker = read(ROOT / record['result'])
    require(worker['state'] == 'complete' and worker['identity'] == dp['identities'][label], 'DP worker identity mismatch')
    calls = [event for event in worker['events'] if event['operation'] == 'evaluate_dp']
    require(sum(event['status'] == 0 for event in calls) == 80, 'Wrong DP success count')
    require(sum(event['status'] != 0 and event['expected'] in ('error', 'failure') and
                event['failure_sentinel_unchanged'] for event in calls) == 10, 'Wrong DP failure count')
    workers[label] = worker
require(workers['original']['events'] == workers['split']['events'] and len(workers['original']['events']) == 110,
        'DP complete event records differ')
require(workers['original']['output_payloads'] == workers['split']['output_payloads'], 'DP output bytes differ')
require(execution['old_library_sha256'] == dp['identities']['original']['library_sha256'] ==
        before['previous095e']['files']['lib/libmadloop_tirust.so']['sha256'], 'Old DP library identity differs')
require(execution['new_library_sha256'] == dp['identities']['split']['library_sha256'] ==
        checks['native_artifacts']['libmadloop_tirust.so']['sha256'], 'New DP library identity differs')

# The dedicated public-Fortran success replay has its own manifest and parser.
# Its exact schema is bound and checked here after that runner is frozen.
success_path = REPORT / 'success-parity/results.json'
success = read(success_path)
require(success.get('state') == 'complete' and success.get('passed') is True,
        'Public Fortran success-path parity incomplete')
success_dir = success_path.parent
protocol = success_dir / 'protocol-sources'
frozen = read(protocol / 'freeze.json')
for name, expected in frozen['files'].items():
    require(sha(protocol / name) == expected and sha(WORK / 'success-parity' / name) == expected,
            'Public Fortran protocol changed: ' + name)
specification = read(protocol / 'specification.json')
require(sha(ROOT / specification['source_fixture_path']) == specification['source_fixture_sha256'],
        'Public Fortran fixture source changed')
require(sha(protocol / 'fixtures.dat') == specification['input_sha256'], 'Public Fortran input stream changed')
inputs = read(success_dir / 'inputs-before.json')
require(inputs == read(success_dir / 'inputs-after.json'), 'Public Fortran inputs changed')
require(inputs['protocols'] == frozen['files'], 'Public Fortran protocol bindings differ')
for label, records in inputs['installations'].items():
    expected_prefix = Path(before['previous095e']['prefix']) if label == 'old' else prefix
    for name, record in records.items():
        path = Path(name)
        require(path.is_relative_to(expected_prefix) and sha(path) == record['sha256'] and
                path.stat().st_size == record['bytes'], 'Public Fortran installed identity changed')
execution_success = read(success_dir / 'execution.json')
require(execution_success['passed'] is True and execution_success['cpu_affinity'] == [2, 3, 4, 5],
        'Public Fortran execution did not pass with recorded affinity')
require(execution_success['native_lock'] == str(ROOT / '.tirust-build/numerical-tests.lock'),
        'Public Fortran used another numerical lock')
parser = runpy.run_path(str(protocol / 'validate.py'))['qualify']
for label, worker in execution_success['workers'].items():
    expected_prefix = Path(before['previous095e']['prefix']) if label == 'old' else prefix
    require(execution_success['prefixes'][label] == str(expected_prefix), 'Public Fortran prefix mismatch')
    for operation in ('build', 'linkage', 'run'):
        command = worker[operation]
        log = Path(command['log'])
        require(log.is_relative_to(success_dir) and sha(log) == command['log_sha256'] and
                command['returncode'] == 0 and not command['timed_out'], 'Public Fortran command/log failed')
    executable = Path(worker['run']['argv'][0])
    require(worker['executable_sha256'] == worker['executable_after_sha256'] == sha(executable),
            'Public Fortran executable changed')
    require(worker['resolved_tirust_dso'] == str(expected_prefix / 'lib/libmadloop_tirust.so') and
            worker['resolved_tirust_dso_sha256'] == sha(expected_prefix / 'lib/libmadloop_tirust.so'),
            'Public Fortran resolved the wrong DSO')
    require(worker['runtime_before'] == worker['runtime_after'], 'Public Fortran runtime changed')
    for name, record in worker['runtime_after'].items():
        path = Path(name)
        require(str(path.resolve()) == record['resolved'] and sha(path) == record['sha256'] and
                path.stat().st_size == record['bytes'], 'Public Fortran runtime identity drift')
    frames = success_dir / label / 'frames.txt'
    qualification = success_dir / label / 'qualification.json'
    require(sha(frames) == worker['frames_sha256'] == success[label + '_frames_sha256'] and
            sha(qualification) == worker['qualification_sha256'], 'Public Fortran result digest differs')
    parsed = parser(frames.read_text(), specification)
    require(parsed == read(qualification) and parsed['passed'] is True and parsed['calls'] == 65,
            'Independent public Fortran parser replay differs')
require((success_dir / 'old/frames.txt').read_bytes() == (success_dir / 'new/frames.txt').read_bytes(),
        'Public Fortran output/cache records differ')
post_analysis = runpy.run_path(str(success_dir / 'post-analysis.py'))['verify']()
require(post_analysis == read(success_dir / 'post-analysis.json') and post_analysis['passed'],
        'Offline successful-row counter proof differs')
boundary = read(REPORT / 'archive-tests/real-boundary-analysis.json')
require(boundary['passed'] and boundary['native_boundary_reached'] and
        boundary['source_sha256'] == sha(REPORT / 'archive-tests/analyze_real_boundary.py'),
        'Native boundary analysis binding failed')
for name, expected in boundary['input_artifacts'].items():
    require(sha(REPORT / 'archive-tests' / name) == expected, 'Native boundary analysis input changed')

checks.update(state='passed', final_checks_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              raw_dp_calls_per_library=90, raw_dp_successes_per_library=80,
              raw_dp_expected_failures_per_library=10, raw_dp_event_records_identical=110,
              installed_archive_tests='archive-tests/results.json',
              public_fortran_success_parity='success-parity/results.json',
              full_protected_state_rechecked_after_qualification=True)
write(REPORT / 'check-installation.json', checks)
for name in ('run_dp_replay.py', 'finalize_evidence.py'):
    shutil.copy2(WORK / name, REPORT / name)
index = {'schema_version': 1, 'state': 'passed', 'performance_eligible': False,
         'source_revision': checks['source_revision'], 'prefix': str(prefix),
         'native_artifacts': checks['native_artifacts'], 'metadata_sha256': checks['metadata_sha256'],
         'scope': 'Genuine cached CLI publication, DP and public Fortran DP/QP/full/poles/cache parity, '
                  'installed-archive public transport invalidation and old-archive negative control.',
         'artifacts': {p.relative_to(REPORT).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
                       for p in sorted(REPORT.rglob('*')) if p.is_file() and p.name != 'index.json'}}
write(REPORT / 'index.json', index)
outer_path = REPORT.parent / 'index.json'
outer = read(outer_path)
for path in sorted(REPORT.rglob('*')):
    if path.is_file():
        name = path.relative_to(REPORT.parent).as_posix()
        require(name not in outer['artifacts'], 'New evidence name already exists in outer index')
        outer['artifacts'][name] = {'sha256': sha(path), 'size': path.stat().st_size}
outer.setdefault('additional_qualifications', []).append({
    'record': 'qp-transport-cache/index.json', 'status': 'passed',
    'source_revision': checks['source_revision'], 'cache_reuse': True,
    'immutable_previous_installations_and_configurations_unchanged': True})
write(outer_path, outer)
print(json.dumps({'state': 'passed', 'artifacts': len(index['artifacts']),
                  'index_sha256': sha(REPORT / 'index.json'), 'prefix': str(prefix)}, indent=2))
