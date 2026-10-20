#!/usr/bin/env python3
"""Survey candidate singleton CPUs without launching any numerical work."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CPUS = tuple(cpu for cpu in sorted(os.sched_getaffinity(0)) if cpu != 207 and Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip() == str(cpu))


def read():
    selected = {f'cpu{cpu}': cpu for cpu in CPUS}
    return {selected[words[0]]: [int(x) for x in words[1:9]]
            for line in Path('/proc/stat').read_text().splitlines()
            if (words := line.split()) and words[0] in selected}


def main():
    assert Path.cwd() == ROOT and len(sys.argv) == 2
    destination = Path(sys.argv[1]).resolve()
    assert destination.is_relative_to(HERE) and not destination.exists()
    record = {'schema': 'pure-tensor-all-singleton-survey-v1', 'cpus': CPUS,
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'interval_seconds': 3, 'interval_count': 3, 'maximum_busy_fraction': .05,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numerical_execution': False}
    record['topology'] = {str(cpu): {name: Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/{name}').read_text().strip()
        for name in ['thread_siblings_list', 'core_id', 'physical_package_id']} for cpu in CPUS}
    before = read()
    samples = []
    for _ in range(3):
        time.sleep(3)
        after = read()
        row = {}
        for cpu in CPUS:
            delta = [a - b for a, b in zip(after[cpu], before[cpu])]
            assert all(v >= 0 for v in delta) and sum(delta) > 0
            row[str(cpu)] = {'ticks_before': before[cpu], 'ticks_after': after[cpu],
                            'busy_fraction': 1 - (delta[3] + delta[4]) / sum(delta)}
        samples.append(row)
        before = after
    record['samples'] = samples
    record['process_locations'] = subprocess.check_output(['ps', '-eo', 'pid,ppid,psr,comm'], text=True)
    record['eligible'] = [cpu for cpu in CPUS if record['topology'][str(cpu)]['thread_siblings_list'] == str(cpu)
                          and all(row[str(cpu)]['busy_fraction'] <= .05 for row in samples)]
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with destination.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    best = sorted(record['eligible'], key=lambda cpu: (max(row[str(cpu)]['busy_fraction'] for row in samples), sum(row[str(cpu)]['busy_fraction'] for row in samples), cpu))[:12]
    print(json.dumps({'eligible_count': len(record['eligible']), 'best_eligible': {str(cpu): [row[str(cpu)]['busy_fraction'] for row in samples] for cpu in best}}))


if __name__ == '__main__':
    main()
