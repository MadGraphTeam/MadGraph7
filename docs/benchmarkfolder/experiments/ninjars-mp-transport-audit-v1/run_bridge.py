#!/usr/bin/env python3
"""HELD: requires root's exclusive MP handoff; preserves every replay field."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.benchmark-work/experiments/ninjars-mp-transport-audit-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd() == ROOT
    build = json.loads((HERE / 'bridge-build.json').read_text())
    prepared = json.loads((HERE / 'bridge-prepared.json').read_text())
    assert build['complete'] and not build['executed']
    assert sha(HERE / 'bridge-prepared.json') == build['preparation_sha256']
    assert len(build['executables']) == 1
    executable = ROOT / build['executables'][0]['path']
    assert sha(executable) == build['executables'][0]['sha256']
    for name, digest in prepared['proposed_files'].items():
        assert sha(WORK / 'bridge' / name) == digest, name
    capture = WORK / 'actual-mp-requests.jsonl'
    assert sha(capture) == prepared['capture']['uncompressed_sha256']
    env = dict(os.environ)
    env.update(NINJARS_TRANSPORT_REPLAY=str(capture),
               NINJARS_TRANSPORT_REPLAY_OUTPUT=str(HERE / 'actual-replay.jsonl'),
               RUST_BACKTRACE='1')
    commands = [('unit', ['taskset', '-c', '2', str(executable), '--test-threads=1', '--nocapture'], 600),
                ('replay', ['taskset', '-c', '2', str(executable), '--ignored', '--exact',
                            'transport_replay_tests::all_stored_requests_retain_outputs_after_transport_fix',
                            '--test-threads=1', '--nocapture'], 900)]
    result = {'complete': False, 'production_modified': False, 'allocation_optimization_present': False,
              'build_sha256': sha(HERE / 'bridge-build.json'), 'runner_sha256': sha(Path(__file__)),
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'runs': []}
    try:
        for name, command, limit in commands:
            start = time.monotonic()
            row = {'name': name, 'command': command, 'timeout_seconds': limit}
            result['runs'].append(row)
            with (HERE / f'{name}.stdout').open('x') as out, (HERE / f'{name}.stderr').open('x') as err:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
                row['pid'] = process.pid
                try:
                    row['returncode'] = process.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    row['returncode'] = process.returncode
                    raise
                finally:
                    row['elapsed_seconds'] = time.monotonic() - start
            assert row['returncode'] == 0, 'raw failures/differences remain retained; never overwrite expected results'
        rows = [json.loads(line) for line in (HERE / 'actual-replay.jsonl').read_text().splitlines()]
        assert len(rows) == 1393
        summary = rows[-1]
        assert summary == {'complete': True, 'records': 1392, 'changed_records': 0, 'changed_fields': 0, 'failures': 0}
        for index, row in enumerate(rows[:-1]):
            assert row['record'] == index and row['status'] == 0
            assert row['expected'] == row['actual'] and len(row['actual']) == 8
            assert row['changed_fields'] == []
        result['replay_summary'] = summary
        assert sha(executable) == build['executables'][0]['sha256']
        for name, digest in prepared['proposed_files'].items():
            assert sha(WORK / 'bridge' / name) == digest, name
        result['complete'] = True
    except BaseException as error:
        result['failure'] = repr(error)
        raise
    finally:
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result['files'] = {name: sha(HERE / name) for name in
                           ('unit.stdout', 'unit.stderr', 'replay.stdout', 'replay.stderr', 'actual-replay.jsonl')
                           if (HERE / name).is_file()}
        with (HERE / 'bridge-result.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
