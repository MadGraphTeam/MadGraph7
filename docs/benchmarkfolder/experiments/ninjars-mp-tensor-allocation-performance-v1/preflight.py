#!/usr/bin/env python3
"""Host-only CPU212 survey; does not launch a benchmark or MP context."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CPU = 212


def read():
    line = next(v for v in Path('/proc/stat').read_text().splitlines() if v.startswith(f'cpu{CPU} '))
    return [int(x) for x in line.split()[1:9]]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    assert Path.cwd() == ROOT and len(sys.argv) == 2
    destination = Path(sys.argv[1]).resolve()
    assert destination.is_relative_to(HERE) and not destination.exists()
    record = {'schema': 'pure-tensor-host-preflight-v1', 'cpu': CPU,
              'started_utc': now(), 'interval_seconds': 3, 'interval_count': 3,
              'maximum_busy_fraction': .05, 'numerical_execution': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'allowed_background': 'Unchanged corrected-original reference collector on CPU207; no task-owned compiler or other evaluation during the requested timing window.'}
    topology = Path(f'/sys/devices/system/cpu/cpu{CPU}/topology')
    record['topology'] = {name: (topology / name).read_text().strip()
                          for name in ['thread_siblings_list', 'core_id', 'physical_package_id']}
    record['process_locations_before'] = subprocess.check_output(['ps', '-eo', 'pid,ppid,psr,comm'], text=True)
    before = read()
    samples = []
    for _ in range(3):
        time.sleep(3)
        after = read()
        delta = [a - b for a, b in zip(after, before)]
        assert all(v >= 0 for v in delta) and sum(delta) > 0
        busy = 1 - (delta[3] + delta[4]) / sum(delta)
        samples.append({'ticks_before': before, 'ticks_after': after, 'busy_fraction': busy})
        before = after
    record['samples'] = samples
    record['process_locations_after'] = subprocess.check_output(['ps', '-eo', 'pid,ppid,psr,comm'], text=True)
    record['passed'] = record['topology']['thread_siblings_list'] == str(CPU) and all(s['busy_fraction'] <= .05 for s in samples)
    record['finished_utc'] = now()
    with destination.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'passed': record['passed'], 'cpu': CPU, 'busy_fractions': [s['busy_fraction'] for s in samples]}))


if __name__ == '__main__':
    main()
