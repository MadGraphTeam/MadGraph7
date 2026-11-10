"""One bounded compiler+integer-check job. Requires separate root authorization."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path.cwd()
HERE = ROOT / '.tirust-build/general-kinematics-integer-check'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(index):
    for name, expected in index['files'].items():
        assert sha(HERE / name) == expected, name
    for name, expected in index['inputs'].items():
        assert sha(ROOT / name) == expected, name
    old = ROOT / '.tirust-build/general-kinematics-classification'
    for name, expected in json.loads((old / 'source-index.json').read_text())['files'].items():
        assert sha(old / name) == expected['sha256'], name
    assert sha(Path(index['compiler']['path'])) == index['compiler']['sha256']


def main():
    assert not (HERE / 'execution.json').exists(), 'No automatic retry or overwrite.'
    index = json.loads((HERE / 'source-index.json').read_text())
    check(index)
    os.sched_setaffinity(0, {13})
    (HERE / 'tmp').mkdir(exist_ok=True)
    compiler = index['compiler']['path']
    env = {'PATH': os.environ['PATH'], 'LC_ALL': 'C', 'LANG': 'C',
           'TMPDIR': str(HERE / 'tmp'), 'OMP_NUM_THREADS': '1'}
    command = [compiler, '-O0', '-g', '-fcheck=all', '-fbacktrace',
               str(HERE / 'driver.f90'), str(HERE / 'qcd_lookup.f'),
               str(HERE / 'boundary_lookup.f'), '-o', str(HERE / 'lookup-check')]
    start = time.monotonic()
    results = []
    evidence = {'source_index_sha256': sha(HERE / 'source-index.json'),
                'cpu': 13, 'total_timeout_seconds': 120, 'amplitude_executed': False,
                'commands': results, 'complete': False, 'environment': env}
    try:
        for phase, args in [('compile', command), ('integer', [str(HERE / 'lookup-check'), str(HERE / 'cases.dat')])]:
            remaining = 120 - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError('Combined job exceeded 120 seconds')
            with (HERE / (phase + '.log')).open('wb') as log:
                process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise TimeoutError('Hard process-group deadline')
            results.append({'phase': phase, 'argv': args, 'returncode': code,
                            'log_sha256': sha(HERE / (phase + '.log'))})
            assert code == 0, phase
        expected = json.loads((HERE / 'preparation.json').read_text())['case_count']
        assert (HERE / 'integer.log').read_text().strip() == f'INTEGER_LOOKUP_PASS {expected}'
        check(index)
        evidence['protected_inputs_unchanged'] = True
        evidence['executable_sha256'] = sha(HERE / 'lookup-check')
        evidence['complete'] = True
    finally:
        evidence['elapsed_seconds'] = time.monotonic() - start
        (HERE / 'execution.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'complete': evidence['complete'], 'cases': expected,
                      'execution_sha256': sha(HERE / 'execution.json')}))


if __name__ == '__main__':
    main()
