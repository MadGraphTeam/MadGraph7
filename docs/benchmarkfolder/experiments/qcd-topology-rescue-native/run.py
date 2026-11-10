#!/usr/bin/env python3
"""One private, capped arm/backend replay; completion is not physics acceptance.

Prepared for review only. Invoking main executes one native process and therefore
requires the separate execution authorization described by the frozen protocol.
"""
from pathlib import Path
import argparse
import fcntl
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

from common import (ROOT, HERE, DESIGN, OLD, PROC, WORKBASE, OUTBASE, ARMS,
                    sha, rel, require, add_pin, pins, controller_pins,
                    validate_pins, frozen_verify, run_capped)

TIRUST_SHA = '49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'
RESOURCE_NAMES = ('ML5_0_ColorDenomFactors.dat', 'ML5_0_ColorNumFactors.dat',
                  'ML5_0_HelConfigs.dat')
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS', 'TBB_NUM_THREADS')
ORDINARY_TAGS = ('POINT_P_USER', 'POINT_HOST_ANS', 'POINT_HOST_ACC', 'POINT_G_DP',
                 'POINT_MU_DP', 'POINT_INVALID', 'POINT_DIVZERO', 'POINT_OVERFLOW')


def write_record(result, record):
    temporary = result / 'execution.json.tmp'
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    temporary.replace(result / 'execution.json')


def snapshot(expected):
    """Missing/unreadable files remain observable without hiding a primary error."""
    actual, errors = {}, {}
    for name in expected:
        try:
            actual[name] = sha(ROOT / name)
        except (OSError, ValueError) as exc:
            actual[name] = None
            errors[name] = type(exc).__name__ + ': ' + str(exc)
    return actual, errors


def merge_pins(destination, source):
    for name, digest in source.items():
        add_pin(destination, ROOT / name, digest)


def selected_rows(text):
    return [line.strip() for line in text.splitlines()
            if line.strip().startswith(('AMPLITUDE_', 'BORN_RESCUE_COUNTS', 'LOOP_NORMALIZATION'))]


def compare_rows(output, reference):
    """Ordinary rows are exact; stress numbers/status/flags are deliberately raw."""
    observed, expected = selected_rows(output), selected_rows(reference)

    def rows(lines, tag):
        return [line for line in lines if line.split()[0] == tag]

    values = rows(observed, 'AMPLITUDE_VALUE')
    old_values = rows(expected, 'AMPLITUDE_VALUE')
    identities = [(1, 1, 0), (1, 1, 1), (2, 2, 0), (2, 2, 1), (3, 1, 0), (3, 1, 1)]
    require(len(values) == len(old_values) == 6, 'Expected six amplitude rows.')
    require(all(len(line.split()) == 13 for line in values), 'Malformed amplitude row.')
    require([tuple(map(int, line.split()[1:4])) for line in values] == identities,
            'Amplitude row identity/order changed.')
    ordinary = [line for line in values if line.split()[1] in ('1', '3')]
    old_ordinary = [line for line in old_values if line.split()[1] in ('1', '3')]
    require(ordinary == old_ordinary, 'Ordinary amplitude value/status/IEEE rows changed.')
    require([line.split()[2:] for line in values[:2]] == [line.split()[2:] for line in values[4:]],
            'Repeated ordinary amplitude differs.')
    for tag in ('AMPLITUDE_WARMUP_BEGIN', 'AMPLITUDE_WARMUP', 'AMPLITUDE_BEGIN',
                'AMPLITUDE_DIAGNOSTIC_COMPLETE', 'LOOP_NORMALIZATION'):
        require(rows(observed, tag) == rows(expected, tag), 'Frozen driver rows changed: ' + tag)
    counts = [list(map(int, line.split()[1:])) for line in rows(observed, 'BORN_RESCUE_COUNTS')]
    require(len(counts) == 3 and all(len(row) == 3 for row in counts), 'Malformed Born rescue counters.')
    require([row[0] for row in counts] == [1, 2, 3] and counts[0][1:] == [0, 0],
            'Ordinary first point entered Born rescue or counters are reordered.')
    require(all(value >= 0 for row in counts for value in row), 'Negative Born rescue counter.')
    require(counts[2][1:] == counts[1][1:], 'Repeated ordinary point entered Born rescue.')
    # Stress counters are cumulative and may change with the private rescue result.
    return {'observed_numeric_rows': observed, 'reference_numeric_rows': expected,
            'ordinary_numeric_rows_exact': True, 'ordinary_repeated_rows_exact': True,
            'born_rescue_counts': counts,
            'stress_numeric_rows': values[2:4],
            'stress_rows_interpreted_as_physics_acceptance': False}


def load_decoder():
    path = OLD / 'decode_trace.py'
    spec = importlib.util.spec_from_file_location('frozen_qcd_observer_decoder', path)
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module


def compare_traces(observed, reference, backend):
    """Compare literal integer payloads, without floating-point conversion."""
    require(observed['byte_order'] == reference['byte_order'], 'Trace byte order changed.')
    require(all(call['backend'] == backend for point in observed['points'] for call in point['calls']),
            'Unexpected backend in observed reducer calls.')

    def ordinary(point):
        events = [event for event in point['events'] if event['tag'] in ORDINARY_TAGS]
        require([event['tag'] for event in events] == list(ORDINARY_TAGS), 'Incomplete ordinary raw payloads.')
        return {'point': point['point'], 'host_code': point['host_code'],
                'payloads': [{key: event[key] for key in ('tag', 'i', 'j', 'value', 'words')}
                             for event in events]}

    actual = [ordinary(point) for point in observed['points'] if point['seq'] in (1, 3)]
    expected = [ordinary(point) for point in reference['points'] if point['seq'] in (1, 3)]
    require(len(actual) == 2 and actual == expected, 'Ordinary raw input/output/normalization/status/flags differ.')
    require(actual[0] == actual[1], 'Repeated ordinary raw payloads differ.')
    return {'trace_structure_valid': True, 'ordinary_raw_words_exact': True,
            'ordinary_repeated_raw_words_exact': True, 'ordinary_raw_payloads': actual,
            'trace_summary': observed['summary'],
            'point_summaries': [{'seq': point['seq'], 'point': point['point'],
                                 'host_code': point['host_code'], 'summary': point['summary'],
                                 'health': point['health']} for point in observed['points']],
            'decoder': observed['decoder'], 'health': [[1, 0, 0], [2, 0, 0], [3, 0, 0]]}


def runtime_files(text):
    require('not found' not in text, 'An executable runtime dependency is missing.')
    result = {}
    for line in text.splitlines():
        match = re.search(r'(?:=>\s*)?(/\S+)\s+\(', line)
        if match:
            path = Path(match[1]).resolve()
            result[str(path)] = sha(path)
    require(result, 'No runtime dependency paths found.')
    tirust = re.findall(r'^\s*libmadloop_tirust\.so\s+=>\s+(\S+)\s+\(', text, flags=re.M)
    require(len(tirust) == 1, 'Expected exactly one resolved TIRust DSO.')
    actual = Path(tirust[0]).resolve()
    require(actual == (ROOT / 'HEPTools/tirust/lib/libmadloop_tirust.so').resolve(),
            'Unexpected actual TIRust runtime path.')
    require(sha(actual) == TIRUST_SHA, 'Historical TIRust runtime changed.')
    return result, str(actual)


def interrupted(signum, frame):
    # Raising lets run_capped kill/reap the entire child process group, and lets
    # main preserve partial evidence. SIGKILL cannot be handled by any process.
    raise InterruptedError('Replay interrupted by signal ' + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--backend', type=int, choices=(8, 9), required=True)
    args = parser.parse_args()
    result = OUTBASE / args.arm / ('backend-' + str(args.backend))
    work = WORKBASE / args.arm
    require(result.parent.is_dir(), 'Build this private arm before replay.')
    # Atomic directory creation makes failures consume the one attempt, too.
    result.mkdir()
    record = {'schema': 1, 'complete': False, 'native_executed': False,
              'arm': args.arm, 'backend': args.backend, 'cpu': 11, 'thread_count': 1,
              'timeout_per_process_seconds': 180, 'automatic_retry_count': 0,
              'formal_performance_eligible': False, 'physics_accepted': False,
              'interpretation_ready': False,
              'scope': 'One private ordinary/stress/ordinary replay; completion means protocol capture only. '
                       'Ordinary raw words/rows/status/flags must match history; stress changes are retained without acceptance.'}
    write_record(result, record)
    checks, runtime_before, resource_before = {}, {}, {}
    lock = None
    old_handlers = {}
    protocol_passed = False
    try:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            old_handlers[sig] = signal.signal(sig, interrupted)
        checks = pins()
        controller = controller_pins()
        merge_pins(checks, controller)
        build_path = result.parent / 'build.json'
        build = json.loads(build_path.read_text())
        require(build['complete'] and build['protected_inputs_unchanged'] and not build['native_executed'],
                'Private build is incomplete or changed.')
        require(build['arm'] == args.arm and build['cpu'] == 11, 'Wrong private build arm/CPU.')
        require(build['protected_inputs_before'] == build['protected_inputs_after'], 'Build input snapshots differ.')
        require(build['private_copies_unchanged']
                and build['private_copies_expected'] == build['private_copies_after'],
                'Build private copies differ from the frozen arm inputs.')
        require(build['controller_index_sha256'] == sha(HERE / 'source-index.json'), 'Build used a different controller index.')
        require(build['historical_build_sha256'] == sha(OLD / 'attempt1/build.json'), 'Historical build differs.')
        for name, digest in controller.items():
            require(build['protected_inputs_before'].get(name) == digest, 'Controller was not pinned at build time: ' + name)
        merge_pins(checks, build['protected_inputs_before'])
        merge_pins(checks, build['artifacts_sha256'])
        merge_pins(checks, build['private_copies_expected'])
        merge_pins(checks, build['linked_objects_sha256'])
        require(set(build['linked_objects']) == set(build['linked_objects_sha256']), 'Missing linked object hashes.')
        add_pin(checks, build_path, sha(build_path))
        executable = work / 'amplitude-private'
        require(rel(executable) in build['artifacts_sha256'], 'Private executable lacks a build-time hash.')
        protocol = json.loads((DESIGN / 'protocol.json').read_text())
        require((protocol['cpu'], protocol['thread_count'], protocol['seconds_per_process_cap'],
                 protocol['automatic_retry_count']) == (11, 1, 180, 0), 'Frozen execution limits changed.')
        for name in ('param_card.dat', 'ident_card.dat', 'points.dat'):
            require(build['artifacts_sha256'].get(rel(work / name)) == protocol['frozen_inputs'][name]['sha256'],
                    'Private input differs from frozen protocol: ' + name)
        old_dir = OLD / 'attempt1' / ('backend-' + str(args.backend))
        old = json.loads((old_dir / 'execution.json').read_text())
        require(old['complete'] and old['protected_inputs_unchanged'], 'Historical reference is incomplete.')
        require(old['runtime_sha256_before'] == old['runtime_sha256_after'], 'Historical runtime changed during replay.')
        require(sha(old_dir / 'observed.log') == old['log_sha256']
                and sha(old_dir / 'trace.dat') == old['trace_sha256'], 'Historical output hashes differ.')
        card = ROOT / protocol['frozen_inputs']['backend' + str(args.backend) + '_card']['path']
        parameters = dict(re.findall(r'^#(\w+)\s*\n([^!#\n]+)', card.read_text(), flags=re.M))
        for name, expected in protocol['numerical_policy'][str(args.backend)].items():
            require(parameters.get(name, '').strip().lower() == expected.lower(), 'Frozen numerical option changed: ' + name)
        record['reference_log_sha256'] = old['log_sha256']
        record['reference_trace_sha256'] = old['trace_sha256']
        record['build_sha256'] = sha(build_path)
        record['controller_index_sha256'] = sha(HERE / 'source-index.json')
        record['protected_inputs_before'] = checks
        # The historical runner did not retain its entire environment. Preserve
        # its known setup with a positive allowlist, not unrecorded runtime/FPE
        # overrides or loader injection inherited from the calling shell.
        native = [ROOT / 'HEPTools/tirust/lib', ROOT / 'HEPTools/ninja/lib', ROOT / 'HEPTools/collier']
        env = {'PATH': os.environ['PATH'], 'LANG': 'C', 'LC_ALL': 'C',
               'LD_LIBRARY_PATH': ':'.join(map(str, native)),
               'SYMBOLICA_LOCK_DIR': str(ROOT / '.tirust-build/symbolica-locks'),
               'SYMBOLICA_HIDE_BANNER': '1', 'RUST_MIN_STACK': '134217728',
               'TMPDIR': str(work / ('backend-' + str(args.backend)) / 'tmp')}
        if os.environ.get('HOME'):
            env['HOME'] = os.environ['HOME']
        require(Path(env['SYMBOLICA_LOCK_DIR']).is_dir(), 'Expected shared Symbolica lock directory is missing.')
        env.update({name: '1' for name in THREAD_VARIABLES})
        env['OMP_DYNAMIC'] = 'FALSE'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        record['execution_environment'] = env
        record['execution_environment_policy'] = 'Positive allowlist; historical environment was not retained. No inherited loader/FPE policy overrides.'
        lock = (ROOT / '.tirust-build/numerical-tests.lock').open('a+')
        record['lock_wait_started_unix'] = time.time()
        write_record(result, record)
        fcntl.flock(lock, fcntl.LOCK_EX)
        record['lock_acquired_unix'] = time.time()
        validate_pins(checks)
        (result / 'frozen-verifier.log').write_text(frozen_verify())
        private = work / ('backend-' + str(args.backend))
        private.mkdir()
        (private / 'tmp').mkdir()
        resources = private / 'MadLoop5_resources'
        resources.mkdir()
        for name in RESOURCE_NAMES:
            source = PROC / 'SubProcesses/MadLoop5_resources' / name
            require(rel(source) in checks, 'Unpinned shared resource: ' + name)
            shutil.copyfile(source, resources / name)
            resource_before[rel(resources / name)] = checks[rel(source)]
        shutil.copyfile(card, resources / 'MadLoopParams.dat')
        resource_before[rel(resources / 'MadLoopParams.dat')] = protocol['frozen_inputs']['backend' + str(args.backend) + '_card']['sha256']
        validate_pins(resource_before)
        record['resource_sha256_before'] = resource_before
        with (result / 'ldd.txt').open('x') as log:
            code = run_capped(['taskset', '-c', '11', 'ldd', str(executable)], log, env=env, seconds=30)
        require(code == 0, 'Runtime dependency inspection failed.')
        runtime_before, actual_dso = runtime_files((result / 'ldd.txt').read_text())
        record['runtime_sha256_before'] = runtime_before
        record['runtime_tirust_path'] = actual_dso
        require(runtime_before == old['runtime_sha256_before'], 'Actual runtime differs from the historical backend runtime.')
        validate_pins(checks)
        validate_pins(resource_before)
        validate_pins(runtime_before)
        command = ['taskset', '-c', '11', str(executable), str(work / 'param_card.dat'),
                   str(work / 'points.dat'), str(resources), str(result / 'trace.dat')]
        record['command'] = command
        record['host_load_start'] = os.getloadavg()
        record['native_executed'] = True
        record['native_start_unix'] = time.time()
        write_record(result, record)
        start = time.monotonic()
        try:
            with (result / 'observed.log').open('x') as log:
                record['returncode'] = run_capped(command, log, env=env, seconds=180)
        except subprocess.TimeoutExpired:
            record['timed_out'] = True
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - start
            record['host_load_end'] = os.getloadavg()
            record['native_end_unix'] = time.time()
        require(record['returncode'] == 0, 'Private process failed; no retry.')
        record.update(compare_rows((result / 'observed.log').read_text(), (old_dir / 'observed.log').read_text()))
        decoder = load_decoder()
        decoded = decoder.load_trace(result / 'trace.dat', result / 'observed.log')
        reference = decoder.load_trace(old_dir / 'trace.dat', old_dir / 'observed.log')
        record.update(compare_traces(decoded, reference, args.backend))
        record['remaining_gate'] = 'Independent stress trace/kinematic/model-parameter analysis; no physical amplitude acceptance is inferred.'
        protocol_passed = True
    except BaseException as exc:
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        try:
            unchanged = True
            for label, before in (('protected_inputs', checks), ('runtime_sha256', runtime_before),
                                  ('resource_sha256', resource_before)):
                after, errors = snapshot(before)
                record[label + '_after'] = after
                same = after == before and not errors
                record[label + '_unchanged'] = same
                unchanged = unchanged and same
                if errors:
                    record[label + '_snapshot_errors'] = errors
            record['protected_inputs_before'] = checks
            for name, prefix in (('observed.log', 'log'), ('trace.dat', 'trace'), ('ldd.txt', 'ldd')):
                path = result / name
                if path.is_file():
                    record[prefix + '_sha256'] = sha(path)
                    record[prefix + '_bytes'] = path.stat().st_size
            if 'resources' in locals():
                record['resource_sha256_all_after'] = {rel(path): sha(path)
                                                       for path in sorted(resources.iterdir()) if path.is_file()}
            record['complete'] = protocol_passed and unchanged
            if not unchanged and 'error' not in record:
                record['error'] = {'type': 'ProtectedInputChange', 'message': 'A protected input/runtime/resource changed.'}
            record['finished_unix'] = time.time()
            write_record(result, record)
        finally:
            if lock is not None:
                lock.close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    require(record['complete'], 'Replay protocol failed; retained evidence is not a successful replay.')
    print(args.arm + ' backend ' + str(args.backend) + ': protocol capture complete; ordinary parity passed; stress physics unqualified.')


if __name__ == '__main__':
    main()
