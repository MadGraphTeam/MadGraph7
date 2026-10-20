#!/usr/bin/env python3
"""Recover the preserved orphan Cargo job; never pretend the lost parent waited."""
import sys
sys.dont_write_bytecode = True
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.benchmark-work/experiments/ninjars-native-float256-v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def identity(pid):
    path = Path('/proc') / str(pid)
    if not path.exists():
        return None
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
            'start_ticks': int(fields[19]),
            'exit_code': int(fields[49]),
            'command': (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()}


def main():
    assert Path.cwd() == ROOT
    launch = json.loads((HERE / 'build-launch.json').read_text())
    paused_path = ROOT / '.benchmark-work/pause-native-20261021-second.json'
    paused = json.loads(paused_path.read_text())
    expected = {row['pid']: row for row in paused['processes']}
    assert identity(2962687) is None and identity(2962690) is None
    before = [identity(pid) for pid in (3063368, 2962840)]
    for record in before:
        assert record is not None and record['state'] == 'T'
        assert record['start_ticks'] == expected[record['pid']]['start_ticks']
        assert record['command'] == expected[record['pid']]['command']
    prepared = json.loads((HERE / 'prepared.json').read_text())
    for name, digest in prepared['study_files'].items():
        assert sha(WORK / 'bridge' / name) == digest, name
    result = {'complete': False, 'controller_lost': True,
              'original_parent_wait_status_available': False,
              'pause_record_sha256': sha(paused_path), 'before': before,
              'script_sha256': sha(Path(__file__)),
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'performance_claim': False}
    (HERE / 'build-recovery-launch.json').write_text(json.dumps(result, indent=2) + '\n')
    for record in before:
        current = identity(record['pid'])
        assert current == record
        os.kill(record['pid'], signal.SIGCONT)
    started = time.monotonic()
    try:
        while True:
            observed = [identity(pid) for pid in (3063368, 2962840)]
            for record in observed:
                if record is not None:
                    assert record['start_ticks'] == expected[record['pid']]['start_ticks']
            if all(record is None or record['state'] == 'Z' for record in observed):
                result['after'] = observed
                break
            assert time.monotonic() - started < 7200, 'Recovery timeout; inspect preserved job before further action'
            time.sleep(5)
        rows = [json.loads(line) for line in (HERE / 'build-1.stdout').read_text().splitlines()]
        finished = [row for row in rows if row.get('reason') == 'build-finished']
        assert len(finished) == 1 and finished[0]['success'] is True
        artifacts = [{'path': str(Path(row['executable']).relative_to(ROOT)),
                      'sha256': sha(Path(row['executable']))}
                     for row in rows if row.get('reason') == 'compiler-artifact' and row.get('executable')]
        assert len(artifacts) == 1
        for record in result['after']:
            if record is not None:
                assert record['exit_code'] == 0, record
        for name, digest in prepared['study_files'].items():
            assert sha(WORK / 'bridge' / name) == digest, name
        result['complete'] = True
        result['cargo_build_finished'] = finished[0]
        result['executables'] = artifacts
        # The runner needs a completed build identity. Keep the original launch
        # and all logs, disclose its lost parent, and bind the recovery record.
        build = dict(launch, complete=True, executed=False, controller_lost=True,
                     original_parent_wait_status_available=False,
                     recovery='build-recovery-result.json', executables=artifacts,
                     completion_evidence='Cargo JSON build-finished success and exact artifact hash; observed orphan exit fields')
        build['logs'] = {name: sha(HERE / name) for name in ('build-1.stdout', 'build-1.stderr')}
        with (HERE / 'build.json').open('x') as stream:
            json.dump(build, stream, indent=2)
            stream.write('\n')
    except BaseException as error:
        result['failure'] = repr(error)
        raise
    finally:
        result['elapsed_active_recovery_seconds'] = time.monotonic() - started
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with (HERE / 'build-recovery-result.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
