"""One current-library replay of the frozen synthetic stability fixture."""
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / 'docs/benchmarkfolder/qualification/tirust-bridge/stability-audit'
BUILD = ROOT / '.tirust-build/default-stability-current-1'
OUT = BUILD / 'execution-1'
BRIDGE_SHA = 'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9'
SUPERVISOR = ROOT / 'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py'
SUPERVISOR_SHA = 'ed25a9fc96e2faf2b545a9bf70a91b34315c01e534a6ebe68d7520ec2171aa42'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child():
    # The supervisor holds the repository lock throughout the child's lifetime.
    # Suppress only the historical runner's redundant acquisition/release.
    fcntl.flock = lambda *_: None
    load = ctypes.CDLL

    def observed_load(*args, **kwargs):
        library = load(*args, **kwargs)
        maps = Path('/proc/self/maps').read_text()
        (OUT / 'maps.txt').write_text(maps)
        paths = {line.split()[-1] for line in maps.splitlines() if '/' in line}
        paths = {p for p in paths if p.startswith('/') and Path(p).is_file()}
        hashes = {p: sha(p) for p in sorted(paths)}
        native = {p: h for p, h in hashes.items() if Path(p).name == 'libmadloop_tirust.so'}
        assert len(native) == 1 and set(native.values()) == {BRIDGE_SHA}, native
        metadata = json.loads((BUILD / 'build.json').read_text())
        for build in metadata['build_metadata'].values():
            path = str(Path(build['library']).resolve())
            expected = build.get('input_sha256', build.get('source_hashes'))[str(Path(path).relative_to(ROOT))]
            matching = {p: h for p, h in hashes.items() if Path(p).name == Path(path).name}
            assert matching == {path: expected}, matching
        (OUT / 'runtime.json').write_text(json.dumps({
            'mapped_files_sha256': hashes, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'pid': os.getpid()}, indent=2, sort_keys=True) + '\n')
        return library

    ctypes.CDLL = observed_load
    file = OLD / 'run_host_rescue.py'
    spec = importlib.util.spec_from_file_location('retained_host_replay', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(file), '--library', str(BUILD / 'host/libtirust_host_rescue.so'),
                '--native-report', str(OLD / 'release-raw-and-native.json'),
                '--output', str(OUT / 'rows.json')]
    module.main()


def adjudicate():
    previous = sys.argv
    try:
        sys.argv = [str(OLD / 'adjudicate_host.py'), str(OUT / 'rows.json'),
                    str(OUT / 'adjudication.json')]
        runpy.run_path(str(OLD / 'adjudicate_host.py'), run_name='__main__')
    finally:
        sys.argv = previous
    current = json.loads((OUT / 'rows.json').read_text())['rows']
    historical = json.loads((OLD / 'host-dispatch-release.json').read_text())['rows']
    key = lambda row: (row['threshold'], row['rotations'], row['projection'])
    old = {key(row): row for row in historical}
    assert len({key(row) for row in current}) == len(old) == 8
    comparison = []
    for row in current:
        before = old[key(row)]
        assert row['modes'] == before['modes']
        assert row['return_code'] == before['return_code']
        errors = [abs(a-b)/(abs(a)+abs(b)) if a or b else 0.0
                  for a, b in zip(row['result'], row['fixture_truth'])]
        if row['threshold'] == 1e-6:
            assert all(error <= 3e-6 for error in errors), errors
        comparison.append({'case': key(row), 'truth_symmetric_errors': errors,
                           'historical_result_bit_equal': [struct.pack('d', a) == struct.pack('d', b)
                               for a, b in zip(row['result'], before['result'])],
                           'historical_accuracy_bit_equal': struct.pack('d', row['accuracy']) ==
                               struct.pack('d', before['accuracy'])})
    (OUT / 'historical-comparison.json').write_text(json.dumps({
        'historical_sha256': sha(OLD / 'host-dispatch-release.json'),
        'adjudicator_sha256': sha(OLD / 'adjudicate_host.py'),
        'rows': comparison}, indent=2, sort_keys=True) + '\n')


def main():
    assert Path.cwd() == ROOT
    if sys.argv[1:] == ['--child']:
        child()
        return
    assert not sys.argv[1:]
    assert sha(SUPERVISOR) == SUPERVISOR_SHA
    supervisor = runpy.run_path(str(SUPERVISOR))
    metadata = json.loads((BUILD / 'build.json').read_text())
    assert metadata['complete']
    for build in metadata['build_metadata'].values():
        for path, expected in build.get('input_sha256', build.get('source_hashes')).items():
            assert sha(ROOT / path) == expected, path
    os.sched_setaffinity(0, {14})
    with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Numerical slot busy; no attempt created.')
            raise SystemExit(75)
        OUT.mkdir(exist_ok=False)
        record = {'complete': False, 'cpu': 14, 'timeout_seconds': 120,
                  'runner_sha256': sha(__file__), 'build_sha256': sha(BUILD / 'build.json'),
                  'supervisor_sha256': SUPERVISOR_SHA,
                  'reference_sha256': sha(OLD / 'release-raw-and-native.json'),
                  'historical_runner_sha256': sha(OLD / 'run_host_rescue.py')}
        start = time.monotonic()
        handlers = {}
        def interrupt(signum, frame):
            raise InterruptedError('Replay interrupted by signal %d' % signum)
        try:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                handlers[sig] = signal.signal(sig, interrupt)
            argv = [sys.executable, '-B', str(Path(__file__).resolve()), '--child']
            with (OUT / 'console.log').open('wb') as stream:
                record['process'] = {}
                record['exit_code'] = supervisor['run_plain_capped'](
                    argv, stream, cwd=ROOT, env=dict(os.environ),
                    progress=record['process'], seconds=120)
            assert record['exit_code'] == 0
            rows = json.loads((OUT / 'rows.json').read_text())
            assert rows['state'] == 'complete' and len(rows['rows']) == 8
            assert rows['all_default_kept_dp'] and rows['all_diagnostic_rescued_qp']
            adjudicate()
            record['complete'] = True
        except BaseException as error:
            record['failure'] = repr(error)
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            record['elapsed_seconds'] = time.monotonic() - start
            (OUT / 'execution.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
