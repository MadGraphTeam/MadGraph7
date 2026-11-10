"""Single full-100 EW replay using the already-qualified worker protocol."""
import argparse
import collections
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
CAUSE = ROOT / 'docs/benchmarkfolder/experiments/dp-rescue-cause-observer'
CAMPAIGN = ROOT / 'docs/benchmarkfolder/campaigns/split-dp-terminal-1'
CASE = 'ew_n3_epem_ttxa'
BUILD = ROOT / '.tirust-build/ew-invariant-symmetry-prototype-v2/attempt1/build/build.json'
BUILD_SHA = 'bd0b33f69a552851feac60e61573442bfefad5c17f45dd9713d40d6a9ac6094a'
FIXTURE = ROOT / '.tirust-build/ew-invariant-symmetry-fixtures-v2/attempt1/execution.json'
RUN = ROOT / '.tirust-build/ew-invariant-symmetry-prototype-v2/runtime-1/attempt1'
HELPER_SHA = '6a0512b620f0f9723af8811bbbd128f3b1f0990aac64e3aaf1da60ec7e5cab10'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def inputs():
    assert Path.cwd() == ROOT and sha(BUILD) == BUILD_SHA
    assert sha(CAUSE / 'controller/run.py') == HELPER_SHA
    helper = module('ew_symmetry_owned_controller', CAUSE / 'controller/run.py')
    # Same compiled worker and protocol; only the nonexclusive CPU is changed.
    helper.CPU = 12
    helper.ARM_SECONDS = 600
    build = read(BUILD)
    assert build['status'] == 'complete'
    assert build['protected_before'] == build['protected_after']
    fixture = read(FIXTURE)
    assert fixture['passed'] and fixture['status'] == 'complete'
    assert len(fixture['bulk']) == 4 and len(fixture['traps']) == 48
    assert all(row['passed'] for row in fixture['bulk'])
    assert all(row['equivalent'] for row in fixture['traps'] if row['supported'])
    assert len(fixture['commands']) == 124
    assert all(row['process_reaped'] for row in fixture['commands'])
    assert fixture['protected_before'] == fixture['protected_after']
    helper.verify_pins(fixture['protected_after'])
    original = read(ROOT / '.tirust-build/dp-rescue-cause-observer/cause-2-qualified/build/build.json')
    snapshot = read(CAUSE / 'preparation/source-1/snapshot.json')
    arm = copy.deepcopy(original['cases'][CASE]['arms']['original'])
    assert arm['runtime'] == build['runtime']
    arm.update(executable=build['executable'], interpreter=build['interpreter'],
               launch_prefix=build['launch_prefix'], runtime_environment=build['runtime_environment'])
    pins = helper.collect_pins(build)
    pins.update(fixture['protected_after'])
    helper.collect_pins(original['cases'][CASE]['arms']['original'], pins)
    helper.collect_pins(snapshot['cases'][CASE], pins)
    for path in (BUILD, FIXTURE, Path(__file__), CAUSE / 'controller/run.py',
                 ROOT / '.tirust-build/dp-rescue-cause-observer/cause-2-qualified/build/build.json',
                 CAUSE / 'preparation/source-1/snapshot.json'):
        pins[str(path)] = sha(path)
    raw = CAMPAIGN / ('experiments/numerical/full100-warmup-' + CASE + '-1')
    for lane in ('baseline-1', 'candidate-1', 'candidate-6', 'candidate-7', 'candidate-8', 'candidate-9'):
        for name in ('warmup.dat', 'dimensions.dat', 'ready.json'):
            path = raw / lane / 'raw' / name
            pins[str(path)] = sha(path)
    for path in (CAUSE / 'analyze.py', CAUSE / 'protocol.py', ROOT / 'tools/tirust_benchmark.py',
                 ROOT / 'tools/tirust_benchmark_warmup.py', ROOT / 'tools/tirust_benchmark_worker.py'):
        pins[str(path)] = sha(path)
    helper.verify_pins(pins)
    return helper, build, snapshot['cases'][CASE], arm, pins


def analyze(directory, helper):
    analyzer = helper.analyzer_module()
    candidate = analyzer.read_arm(directory)
    old = analyzer.read_arm(ROOT / '.tirust-build/dp-rescue-cause-observer/cause-2-qualified/controller-native/attempt1' / CASE / 'original')
    assert len(candidate['calls']) == len(old['calls']) == 101
    for before, after in zip(old['calls'], candidate['calls']):
        assert before['input_words'] == after['input_words']
        assert before['entry_state'][5:] == after['entry_state'][5:]
        assert after['entry_state'][5:] == after['exit_state'][5:]
        if before['status'] == 219:
            assert after['status'] == 219
            assert before['answer_words'] == after['answer_words']
            assert before['accuracy_words'] == after['accuracy_words']
    sys.path.insert(0, str(ROOT / 'tools'))
    import tirust_benchmark as bench
    from tirust_benchmark_warmup import parse_warmup
    protocol = (Path(directory) / 'protocol.raw').read_text().splitlines()
    assert len(protocol) == 2 and protocol[1] == 'BYE'
    fields = protocol[0].split()
    assert fields[:6] == ['READY', '1', '100', '1', '1', '1']
    ready = dict(points=100, ndim=1, nso=1, nborn=1, cold_seconds=float(fields[6]))
    observation = parse_warmup(directory, [9], 100, ready)
    comparisons = {}
    raw = CAMPAIGN / ('experiments/numerical/full100-warmup-' + CASE + '-1')
    for lane in ('baseline-1', 'candidate-1', 'candidate-6', 'candidate-7', 'candidate-8', 'candidate-9'):
        folder = raw / lane / 'raw'
        reference = parse_warmup(folder, [int(lane.split('-')[1])], 100, read(folder / 'ready.json'))
        comparisons[lane] = bench.compare_rows(reference['rows'], observation['rows'], 3e-6)
    transitions = collections.Counter((a['status'], b['status']) for a, b in zip(old['calls'][1:], candidate['calls'][1:]))
    ieee_differences = [{'point': b['point'], 'before_entry': a['entry_state'], 'after_entry': b['entry_state'],
                         'before_exit': a['exit_state'], 'after_exit': b['exit_state']}
                        for a, b in zip(old['calls'], candidate['calls'])
                        if a['entry_state'] != b['entry_state'] or a['exit_state'] != b['exit_state']]
    return {'schema': 'ew-symmetry-full100-results-v2',
            'regression_passed': all(row['passed'] for row in comparisons.values()),
            'ordinary_target_passed': all(row['ordinary_target_passed'] for row in comparisons.values()),
            'status_counts': observation['status_counts'],
            'status_transitions': [{'old': a, 'new': b, 'points': n} for (a, b), n in sorted(transitions.items())],
            'comparisons': comparisons, 'ieee_state_differences': ieee_differences,
            'input_words_exact': True, 'previously_accepted_DP_words_exact': True,
            'halting_and_rounding_unchanged': True,
            'warning_W03_lines': [line for line in (Path(directory) / 'stdout.log').read_text().splitlines() if '##W03' in line],
            'performance_eligible': False,
            'scope': 'Same full100 physical stream; independent host stability retained; no performance or global qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify', 'run'))
    args = parser.parse_args()
    helper, build, case, arm, pins = inputs()
    prepared = HERE / 'prepared.json'
    if args.stage == 'prepare':
        assert not prepared.exists()
        helper.write(prepared, {'pins': pins, 'cpu': 12, 'worker_seconds': 600, 'full_points': 100,
                                'library_sha256': next(v for p, v in arm['runtime'].items() if Path(p).name == 'libmadloop_tirust.so')})
    assert read(prepared)['pins'] == pins
    if args.stage != 'run':
        print(json.dumps({'verified': True, 'pins': len(pins), 'native_executed': False}))
        return
    with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(75)
        assert not RUN.exists()
        os.sched_setaffinity(0, {12})
        previous = {s: signal.signal(s, helper.interrupted) for s in helper.SIGNALS}
        record = {'complete': False, 'prepared_sha256': sha(prepared), 'performance_eligible': False}
        try:
            taskset = {'path': '/nix/store/156fzvaz2nsq2hk0fkzn9jwk90bnzyn9-util-linux-2.42.2-bin/bin/taskset'}
            execution = helper.run_arm(case, arm, 'candidate', RUN, build, taskset, helper.Deadline(650))
            record['worker'] = execution
            results = analyze(RUN, helper)
            helper.write(RUN / 'results.json', results)
            helper.verify_pins(pins)
            record.update(complete=True, regression_passed=results['regression_passed'], protected_inputs_unchanged=True)
            assert results['regression_passed'], 'New full100 regression mismatch retained'
        except BaseException as error:
            record['failure'] = repr(error)
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            if RUN.exists():
                helper.write(RUN / 'qualification.json', record)
    print(json.dumps({'complete': True, 'regression_passed': results['regression_passed'], 'status_counts': results['status_counts']}))


if __name__ == '__main__':
    main()
