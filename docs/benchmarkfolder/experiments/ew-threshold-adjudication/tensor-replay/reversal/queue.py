#!/usr/bin/env python3
"""Wait for admission, then invoke the unchanged nonblocking replay launcher."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=int, default=9)
    parser.add_argument('--wait-seconds', type=float, default=2100)
    args = parser.parse_args()
    assert Path.cwd().resolve() == ROOT
    evidence = HERE / 'queue-execution.json'
    if evidence.exists():
        raise ValueError('Refusing to overwrite queue evidence')
    runner = HERE / 'run.py'
    command = [sys.executable, '-B', str(runner), 'run', '--cpu', str(args.cpu)]
    record = {
        'schema': 1,
        'queue_source_sha256': digest(Path(__file__)),
        'frozen_runner_sha256': digest(runner),
        'command': command,
        'queue_timeout_seconds': args.wait_seconds,
        'native_timeout_seconds': 180,
        'performance_eligible': False,
        'admission_policy': 'Observe the normal flock free for one second, releasing each probe immediately; then invoke the unchanged nonblocking runner. This is not an atomic reservation. A final acquisition failure is preserved by that runner.',
        'state': 'waiting',
        'native_launcher_started': False,
    }
    evidence.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    started = time.monotonic()
    free_since = None
    try:
        with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
            while time.monotonic() - started < args.wait_seconds:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    free_since = None
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                    if free_since is None:
                        free_since = time.monotonic()
                    if time.monotonic() - free_since >= 1:
                        break
                time.sleep(.25)
            else:
                raise TimeoutError('No admission within the bounded queue wait')
        record['queue_elapsed_seconds'] = time.monotonic() - started
        assert digest(runner) == record['frozen_runner_sha256']
        record['native_launcher_started'] = True
        record['state'] = 'launcher_running'
        evidence.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        completed = subprocess.run(command, cwd=ROOT, check=False)
        record['returncode'] = completed.returncode
        record['state'] = 'complete' if completed.returncode == 0 else 'launcher_failed'
        if (HERE / 'execution.json').exists():
            record['native_execution_sha256'] = digest(HERE / 'execution.json')
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)
    except BaseException as exc:
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        record['state'] = 'failed'
        raise
    finally:
        record['total_elapsed_seconds_not_a_benchmark'] = time.monotonic() - started
        record['frozen_runner_unchanged'] = digest(runner) == record['frozen_runner_sha256']
        record['queue_source_unchanged'] = digest(Path(__file__)) == record['queue_source_sha256']
        evidence.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
