#!/usr/bin/env python3
"""One final-source libtest build and serialized ABI3 evidence collection.

Preparation only until root supplies the final commit, input hash and explicit
resource release. Never modify production sources or overwrite an attempt.
"""
import argparse
import datetime
from decimal import Decimal
from fractions import Fraction
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'madgraph/__init__.py').is_file())
PREFIX = 'vendor/ninjars/'
REPLAY_TEST = 'final_precision_tests::retained_requests_replay_and_converge'
FIXTURE_TEST = 'final_precision_tests::seven_physical_fixtures_all_modes'
MANDATORY = (
    'tadpole_rebinds_coefficients_masses_and_scale',
    'invalid_input_preserves_outputs_and_reports_error',
    'version_and_malformed_outputs_are_transactional',
    'every_short_buffer_is_rejected_before_access_or_output',
    'mp_transport_preserves_binary128_information',
    'complete_mp_tadpole_retains_input_precision',
    'mp_bubbles_cover_massless_orderings_and_context_rebinding',
    'complex_mass_pentagon_converges_at_binary128_output_precision',
    'mp_transport_preserves_binary128_finite_dyadics_and_sign',
    'mp_transport_preserves_complex_branch_lips',
    'panic_is_contained_and_success_clears_calling_thread_error',
    'analytic_poles_reuse_and_returned_shape_errors',
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def bound(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
            'bytes': path.stat().st_size}


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT)


def verify_revision(revision):
    require(git('rev-parse', 'HEAD').decode().strip() == revision,
            'HEAD differs from the explicit final committed revision')
    require(git('branch', '--show-current').decode().strip() == 'madloop_ninjars',
            'Final collection requires branch madloop_ninjars')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'vendor/ninjars'],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    require(not git('ls-files', '--others', '--exclude-standard', '--',
                    'vendor/ninjars').strip(), 'Untracked bridge source remains')
    controller = str(Path(__file__).resolve().relative_to(ROOT))
    require(git('show', revision + ':' + controller) == Path(__file__).read_bytes(),
            'Proof controller itself is not the committed reviewed source')


def snapshot(revision, work):
    """Extract only committed tracked bridge bytes, never reference checkouts."""
    archive = git('archive', '--format=tar', revision, 'vendor/ninjars')
    (work / 'source.tar').write_bytes(archive)
    source = work / 'source'
    source.mkdir()
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as stream:
        for entry in stream:
            path = PurePosixPath(entry.name)
            require(not path.is_absolute() and '..' not in path.parts,
                    'Unsafe archived source path')
            if entry.isdir():
                continue
            require(entry.isfile() and entry.name.startswith(PREFIX),
                    'Final bridge archive contains an unsupported link/object')
            data = stream.extractfile(entry).read()
            name = entry.name[len(PREFIX):]
            require(name and name not in sources, 'Duplicate archived source')
            target = source / entry.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(entry.mode & 0o777)
            sources[name] = hashlib.sha256(data).hexdigest()
    expected = {name[len(PREFIX):] for name in git(
        'ls-tree', '-r', '--name-only', revision, '--', 'vendor/ninjars').decode().splitlines()}
    require(set(sources) == expected and bool(sources),
            'Complete committed bridge archive inventory is required')
    return source / 'vendor/ninjars', sources


def verify_source(bridge, sources):
    for name, expected in sources.items():
        require(sha(bridge / name) == expected, 'Archived bridge source changed: ' + name)


def run(command, cwd, report, label, env, timeout):
    stdout, stderr = report / (label + '.stdout'), report / (label + '.stderr')
    start = time.monotonic()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with stdout.open('x') as out, stderr.open('x') as err:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out,
                                   stderr=err, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            # Kill the complete child process group before returning the MP slot.
            import signal
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    children = []
    for line in subprocess.check_output(
            ['ps', '-eo', 'pid=,pgid=,stat=,args='], text=True).splitlines():
        row = line.split(None, 3)
        if len(row) == 4 and int(row[1]) == process.pid:
            children.append(line.strip())
    result = {'command': command, 'cwd': str(cwd.relative_to(ROOT)),
              'exit_code': code, 'started_utc': stamp,
              'wall_seconds': time.monotonic() - start,
              'stdout': bound(stdout), 'stderr': bound(stderr),
              'process_id': process.pid, 'remaining_process_group_members': children,
              'all_children_exited': not children}
    dump(report / (label + '-process.json'), result)
    require(not children, 'Child processes remain after ' + label)
    require(code == 0, label + ' failed; retain this attempt and its logs')
    return result


def libtest(process, executable):
    text = (ROOT / process['stdout']['path']).read_text()
    totals = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; '
                        r'(\d+) ignored; (\d+) measured; (\d+) filtered out;', text)
    require(len(totals) == 1, 'Exactly one successful libtest footer per run is required')
    passed, failed, ignored, measured, filtered = map(int, totals[0])
    headers = list(re.finditer(r'^test ([^\n ]+) \.\.\. ', text, re.M))
    names, passed_names, ignored_names = [], [], []
    for index, header in enumerate(headers):
        tail = text[header.end():headers[index + 1].start() if index + 1 < len(headers)
                    else text.index('test result: ok.')]
        name = header.group(1)
        names.append(name)
        if tail.startswith('ignored'):
            ignored_names.append(name)
        else:
            require(re.search(r'\bok\s*$', tail) is not None,
                    'Missing actual successful test status: ' + name)
            passed_names.append(name)
    require(len(names) == len(set(names)) and len(passed_names) == passed
            and len(ignored_names) == ignored and failed == measured == 0,
            'Libtest names and actual successful footer counts disagree')
    return {**process, 'executable': executable,
            'observed_test_names': names, 'passed_test_names': passed_names,
            'ignored_test_names': ignored_names,
            'tests': {'passed': passed, 'failed': failed, 'ignored': ignored,
                      'filtered_out': filtered}}


def tagged(text, marker):
    return [json.loads(line.split(marker, 1)[1])
            for line in text.splitlines() if marker in line]


def fixture_summary(text):
    summaries = tagged(text, 'NINJARS_FIXTURE_SUMMARY ')
    records = tagged(text, 'NINJARS_FIXTURE_RECORD ')
    require(len(summaries) == 1, 'Exactly one complete fixture summary is required')
    summary = summaries[0]
    require(all(summary.get(key) == value for key, value in
                {'fixtures': 7, 'lanes': 21, 'calls': 63,
                 'failed_calls': 0, 'passed': True}.items()), 'Fixture suite failed/incomplete')
    require(len(records) == 63 and len({(r['case'], r['lane'], r['mode'])
                for r in records}) == 63 and len({r['case'] for r in records}) == 7
            and len({(r['case'], r['lane']) for r in records}) == 21,
            'Fixture records do not cover seven cases and all21 lanes/modes')
    for row in records:
        require(row['status'] == 0 and row['passed'] is True
                and row['mode'] in (0, 1, 2) and row['tolerance'] == 1e-9,
                'Failed or relabelled fixture call')
        require(len(row['compared_component_errors']) == (3 if row['mode'] == 2 else 4),
                'Incomplete fixture component comparisons')
        require(all(Decimal(str(value)).is_finite() and Decimal(str(value)) <= Decimal('1e-9')
                    for value in row['compared_component_errors']), 'Fixture discrepancy failed')
    return {key: summary[key] for key in ('fixtures', 'lanes', 'calls', 'failed_calls', 'passed')}


def raw_values(observation):
    require(observation['status'] == 0 and observation['input_unchanged'] is True
            and observation['output_bytes'] == 512 and len(observation['output']) == 8,
            'Failed/malformed MP observation')
    require(all(Decimal(value).is_finite() for value in observation['output']),
            'Nonfinite transported MP output')
    raw = observation['raw']
    require(len(raw) == 10, 'Missing complete raw Laurent/CC/rational result')
    values = []
    for item in raw:
        require(item['finite'] is True and type(item['precision']) is int
                and item['precision'] >= 113 and type(item['negative']) is bool,
                'Invalid raw Float observation')
        value = Fraction(int(item['numerator']), int(item['denominator']))
        require(value == 0 or (value < 0) == item['negative'], 'Raw sign/value disagreement')
        values.append(value)
    return values


def capture_summaries(path, original):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(rows) == 2785 and rows[-1]['kind'] == 'summary',
            'Full1392 input history at each precision plus summary is required')
    low, high, summary = rows[:1392], rows[1392:2784], rows[-1]
    changed_records = changed_fields = sign_changed = failed = 0
    for sequence, (left, right, reference) in enumerate(zip(low, high, original)):
        require(left['kind'] == 'replay' and right['kind'] == 'convergence'
                and left['sequence'] == right['sequence'] == sequence
                and left['input'] == right['input'] == reference,
                'Replay/convergence omits, reorders or changes original input history')
        require(left['historical_binary128_values_and_zero_signs_equal'] is True,
                'Final public binary128 output/sign differs from historical replay')
        actual, expected = left['observed']['output'], reference['output']
        difference = sum(a != b for a, b in zip(actual, expected))
        require(left['historical_ascii_equal'] == (difference == 0),
                'ASCII difference annotation is inconsistent')
        changed_records += bool(difference)
        changed_fields += difference
        a, b = raw_values(left['observed']), raw_values(right['observed_384'])
        require(left['observed']['working_bits'] == 256
                and right['observed_384']['working_bits'] == 384,
                'Actual working precision changed')
        norm = max(map(abs, b[:6]))
        differences = [abs(x - y) for x, y in zip(a, b)]
        laurent = all(value <= norm / 10**31 for value in differences[:6])
        rational = all(value <= norm / 10**31 for value in differences[8:])
        split = all(abs(values[i] - values[i + 6] - values[i + 8]) <= norm / 10**60
                    for values in (a, b) for i in (0, 1))
        comparison = right['comparison']
        require(comparison['comparable'] is True and comparison['passed'] is True
                and laurent and rational and split, 'Raw dyadic convergence/split gate failed')
        signs = [i for i, (x, y) in enumerate(zip(left['observed']['raw'], right['observed_384']['raw']))
                 if x['negative'] != y['negative']]
        require(signs == comparison['sign_changed_indices'], 'Raw sign-change inventory differs')
        sign_changed += bool(signs)
        failed += not (laurent and rational and split)
    require(all(summary.get(key) == value for key, value in {
        'complete': True, 'records': 1392, 'production_bits': 256,
        'reference_bits': 384, 'convergence_tolerance': '1e-31',
        'fields_per_precision': 11136, 'failed_binary128_replay_records': 0,
        'failed_convergence_records': failed,
        'historical_ascii_changed_records_retained': changed_records,
        'sign_changed_convergence_records_require_review': sign_changed,
        'single_production_backend_no_selector': True, 'performance_claim': False}.items()),
        'Retained raw summary differs from independently recomputed observations')
    return ({'passed': True, 'records': 1392, 'fields': 11136,
             'historical_binary128_failures': 0,
             'historical_ascii_changed_records': changed_records,
             'historical_ascii_changed_fields': changed_fields},
            {'passed': failed == 0, 'records': 1392, 'fields_per_precision': 11136,
             'production_bits': 256, 'reference_bits': 384, 'tolerance': '1e-31',
             'failed_records': failed, 'sign_changed_records': sign_changed,
             'raw_comparisons_recomputed': True,
             'normalization': 'max abs complete Laurent fields0..5; excludes CC/rational',
             'finite_split_tolerance': '1e-60 times the same physical Laurent scale',
             'sign_changes_require_review': sign_changed != 0})


def collect(args, report, work):
    require(os.sched_getaffinity(0) == {190}, 'Run the entire controller pinned to CPU190')
    require(args.resources_released, 'Root must explicitly release compilation and the MP slot')
    verify_revision(args.revision)
    retained_input = args.replay_input.resolve()
    require(sha(retained_input) == args.replay_input_sha256, 'Root-supplied original replay hash differs')
    payload = retained_input.read_bytes()
    if retained_input.suffix == '.gz':
        require(args.replay_uncompressed_sha256 is not None,
                'The original gzip capture also requires its exact uncompressed SHA256')
        payload = gzip.decompress(payload)
        replay_input = work / 'original-replay.jsonl'
        replay_input.write_bytes(payload)
    else:
        replay_input = retained_input
    decoded_hash = args.replay_uncompressed_sha256 or args.replay_input_sha256
    require(hashlib.sha256(payload).hexdigest() == decoded_hash,
            'Original uncompressed replay bytes differ from the root-supplied hash')
    original = [json.loads(line) for line in payload.decode('utf-8').splitlines()]
    require(len(original) == 1392, 'Original retained input history must contain1392 requests')
    bridge, sources = snapshot(args.revision, work)
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith(('NINJARS_CONVERGENCE_', 'NINJARS_TRANSPORT_REPLAY')):
            del env[name]
    env.update(CARGO_BUILD_JOBS='1', CARGO_TARGET_DIR=str(work / 'target'),
               CARGO_BUILD_BUILD_DIR=str(work / 'target/build'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1',
               SYMBOLICA_LOCK_DIR=str(work / 'symbolica-locks'), TMPDIR=str(work / 'tmp'))
    (work / 'symbolica-locks').mkdir(mode=0o700)
    (work / 'tmp').mkdir()
    dump(report / 'request.json', {
        'schema': 'final-rust-abi3-request-v1', 'revision': args.revision,
        'cpu': 190, 'resources_released': True, 'replay_input': bound(retained_input),
        'decoded_replay_input': bound(replay_input),
        'bridge_source_sha256': sources, 'source_archive': bound(work / 'source.tar'),
        'controller': bound(Path(__file__)), 'numerical_threads': 1,
        'cargo_build_jobs': 1, 'policy_selector': False,
        'build_environment': {name: value for name, value in env.items()
             if name.startswith(('CARGO_', 'RUSTFLAGS', 'RUSTC_', 'RUSTDOCFLAGS', 'SYMBOLICA_LOCK'))}})
    build_command = ['cargo', 'test', '--release', '--locked', '--lib', '--no-run',
                     '--features', 'fenv-test-support', '--message-format=json']
    building = run(build_command, bridge, report, 'build', env, args.timeout)
    artifacts = []
    for line in (ROOT / building['stdout']['path']).read_text().splitlines():
        message = json.loads(line)
        if message.get('reason') == 'compiler-artifact' and message.get('executable'):
            artifacts.append(message['executable'])
    require(len(artifacts) == 1, 'Exactly one final libtest executable must be built')
    executable_path = work / 'final-libtest'
    shutil.copy2(artifacts[0], executable_path)
    executable = bound(executable_path)
    verify_source(bridge, sources)
    verify_revision(args.revision)
    dump(report / 'build-record.json', {
        'schema': 'final-rust-abi3-build-v1', 'revision': args.revision,
        'bridge_source_sha256': sources, 'executable': executable,
        'source_archive': bound(work / 'source.tar'), 'process': building,
        'cpu': 190, 'cargo_build_jobs': 1, 'single_libtest_executable': True})
    ordinary_process = run([str(executable_path), '--test-threads=1', '--nocapture'],
                           bridge, report, 'ordinary', env, args.timeout)
    ordinary = libtest(ordinary_process, executable)
    ordinary.update(revision=args.revision, bridge_source_sha256=sources)
    require(ordinary['tests']['passed'] >= 18, 'Incomplete ordinary Rust suite')
    for name in MANDATORY:
        require(any(test.endswith('::' + name) or test == name
                    for test in ordinary['passed_test_names']), 'Mandatory test did not pass: ' + name)
    ordinary_text = (ROOT / ordinary['stdout']['path']).read_text()
    sentinel = tagged(ordinary_text, 'NINJARS_CONVERGENCE ')
    require(len(sentinel) == 1,
            'Existing precision reader requires exactly one ordinary convergence sentinel')
    sentinel = sentinel[0]
    require(sentinel['production_bits'] == 256 and sentinel['reference_bits'] == 384
            and sentinel['convergence_tolerance'] == '1e-31',
            'Ordinary convergence precision or threshold changed')
    low, high = sentinel['result_256'], sentinel['result_384']
    require(len(low) == len(high) == 8
            and all(Decimal(value).is_finite() for value in low + high),
            'Ordinary convergence has missing/nonfinite fields')
    low, high = list(map(Fraction, low)), list(map(Fraction, high))
    scale = max(map(abs, high[:6]))
    require(scale > 0 and all(abs(x - y) <= scale / 10**31 for x, y in zip(low, high)),
            'Ordinary convergence failed the physical Laurent normalization')
    fixture_process = run([str(executable_path), '--ignored', '--exact', FIXTURE_TEST,
                           '--test-threads=1', '--nocapture'], bridge, report, 'fixtures', env, args.timeout)
    fixtures = libtest(fixture_process, executable)
    fixtures.update(revision=args.revision, bridge_source_sha256=sources)
    require(fixtures['passed_test_names'] == [FIXTURE_TEST], 'Wrong ignored fixture entrypoint')
    fixtures['summary'] = fixture_summary((ROOT / fixtures['stdout']['path']).read_text())
    capture = report / 'replay-convergence.jsonl'
    env.update(NINJARS_CONVERGENCE_INPUT=str(replay_input),
               NINJARS_CONVERGENCE_OUTPUT=str(capture))
    capture_process = run([str(executable_path), '--ignored', '--exact', REPLAY_TEST,
                           '--test-threads=1', '--nocapture'], bridge, report, 'replay-convergence', env, args.timeout)
    replay_run = libtest(capture_process, executable)
    replay_run.update(revision=args.revision, bridge_source_sha256=sources)
    require(replay_run['passed_test_names'] == [REPLAY_TEST], 'Wrong ignored convergence entrypoint')
    replay_summary, convergence_summary = capture_summaries(capture, original)
    printed = tagged((ROOT / replay_run['stdout']['path']).read_text(),
                     'NINJARS_CONVERGENCE_CAPTURE_SUMMARY ')
    require(len(printed) == 1 and printed[0] == json.loads(capture.read_text().splitlines()[-1]),
            'Printed and retained capture summaries differ')
    require(sha(executable_path) == executable['sha256'], 'Executed binary changed during runs')
    verify_source(bridge, sources)
    verify_revision(args.revision)
    proof = {'schema': 'final-rust-abi3-v1', 'revision': args.revision,
             'exit_code': 0, 'stdout': ordinary['stdout'], 'stderr': ordinary['stderr'],
             'build_record': bound(report / 'build-record.json'), 'executable': executable,
             'bridge_source_sha256': sources,
             'named_runs': {'ordinary': ordinary, 'fixtures': fixtures,
                 'replay': {**replay_run, 'capture': bound(capture), 'summary': replay_summary},
                 'convergence': {**replay_run, 'capture': bound(capture), 'summary': convergence_summary}},
             'unique_executed_runs': 3,
             'total_executed_test_passes': ordinary['tests']['passed'] + 2,
             'all_children_exited': True,
             'source_unchanged_during_tests': True,
             'sign_changes_require_review': convergence_summary['sign_changes_require_review'],
             'physical_process_acceptance_claim': False,
             'performance_claim': False}
    dump(report / 'unit-proof.json', proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--replay-input', type=Path, required=True)
    parser.add_argument('--replay-input-sha256', required=True)
    parser.add_argument('--replay-uncompressed-sha256',
                        help='Exact original uncompressed SHA256; required for a gzip capture')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--resources-released', action='store_true', required=True)
    parser.add_argument('--timeout', type=float, default=86400.)
    args = parser.parse_args()
    require(re.fullmatch('[0-9a-f]{40}', args.revision) is not None, 'Provide final full Git revision')
    require(re.fullmatch('[0-9a-f]{64}', args.replay_input_sha256) is not None, 'Provide original input SHA256')
    require(args.replay_uncompressed_sha256 is None or
            re.fullmatch('[0-9a-f]{64}', args.replay_uncompressed_sha256) is not None,
            'Provide full original uncompressed SHA256')
    require(args.timeout > 0, 'Timeout must be positive')
    work, report = args.work.resolve(), args.report.resolve()
    require(work.is_relative_to(ROOT / '.benchmark-work')
            and report.is_relative_to(ROOT / 'docs/benchmarkfolder'), 'Keep all work inside the repository')
    require(not work.exists() and not report.exists(), 'Retain existing attempts; choose fresh namespaces')
    work.mkdir(parents=True)
    report.mkdir(parents=True)
    shutil.copy2(__file__, report / 'collect.py')
    try:
        result = collect(args, report, work)
    except BaseException as error:
        dump(report / 'failure.json', {'schema': 'final-rust-abi3-failure-v1',
             'revision': args.revision, 'error_type': type(error).__name__, 'error': str(error),
             'passed': False, 'final_goal_claimed': False})
        raise
    finally:
        files = {str(path.relative_to(report)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(report.rglob('*')) if path.is_file() and path.name != 'proof-manifest.json'}
        dump(report / 'proof-manifest.json', {'schema': 'final-rust-abi3-proof-manifest-v1',
             'revision': args.revision, 'files': files, 'count': len(files)})
    print(json.dumps({'proof': bound(report / 'unit-proof.json'),
                      'manifest': bound(report / 'proof-manifest.json'),
                      'total_executed_test_passes': result['total_executed_test_passes'],
                      'all_children_exited': True,
                      'sign_changes_require_review': result['sign_changes_require_review']}))


if __name__ == '__main__':
    main()
