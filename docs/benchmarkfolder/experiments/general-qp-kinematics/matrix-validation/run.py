"""One authorized CPU13 standalone Fortran compile/test job; no reducers."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path.cwd()
HERE = ROOT / '.tirust-build/general-kinematics-consumer-validation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(index, protocol):
    for name, expected in index['files'].items():
        assert sha(HERE / name) == expected, name
    for name, expected in protocol['protected_inputs'].items():
        assert sha(ROOT / name) == expected, name
    assert sha(Path(protocol['compiler']['path'])) == protocol['compiler']['sha256']


def main():
    assert not (HERE / 'execution.json').exists(), 'No overwrite or automatic retry.'
    index = json.loads((HERE / 'source-index.json').read_text())
    protocol = json.loads((HERE / 'protocol.json').read_text())
    check(index, protocol)
    os.sched_setaffinity(0, {13})
    (HERE / 'tmp').mkdir(exist_ok=True)
    env = {'PATH': os.environ['PATH'], 'LC_ALL': 'C', 'LANG': 'C',
           'TMPDIR': str(HERE / 'tmp'), 'OMP_NUM_THREADS': '1'}
    result = {'complete': False, 'cpu': 13, 'deadline_seconds': 120,
              'source_index_sha256': sha(HERE / 'source-index.json'),
              'environment': env, 'steps': [], 'reducer_or_amplitude_executed': False}
    start = time.monotonic()
    try:
        for flavor in protocol['flavors']:
            binary = HERE / flavor / 'matrix-check'
            compile_command = [protocol['compiler']['path'], '-O0', '-g', '-fcheck=all', '-fbacktrace',
                '-I' + str(HERE / flavor), str(HERE / 'driver.f90'), str(HERE / 'matrix.f'),
                str(HERE / 'off_shell.f'), str(HERE / flavor / 'model_context.f'), '-o', str(binary)]
            for phase, command in [('compile', compile_command), ('matrix', [str(binary), str(HERE / 'cases.dat')])]:
                remaining = 120 - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError('Combined job deadline exceeded')
                logpath = HERE / flavor / (phase + '.log')
                with logpath.open('wb') as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise TimeoutError('Process group killed at combined deadline')
                result['steps'].append({'flavor': flavor, 'phase': phase, 'argv': command,
                                       'returncode': code, 'log_sha256': sha(logpath)})
                assert code == 0, (flavor, phase)
            lines = (HERE / flavor / 'matrix.log').read_text().splitlines()
            assert lines[-1] == 'MATRIX_POLICY_PASS ' + str(protocol['cases_per_flavor'])
            assert sum(line.startswith('CASE_PASS ') for line in lines) == protocol['cases_per_flavor']
            result[flavor] = {'cases_passed': protocol['cases_per_flavor'], 'binary_sha256': sha(binary)}
        check(index, protocol)
        result['protected_inputs_unchanged'] = True
        result['complete'] = True
    finally:
        result['elapsed_seconds'] = time.monotonic() - start
        (HERE / 'execution.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'complete': result['complete'], 'cases_per_flavor': protocol['cases_per_flavor'],
                      'execution_sha256': sha(HERE / 'execution.json')}))


if __name__ == '__main__':
    main()
