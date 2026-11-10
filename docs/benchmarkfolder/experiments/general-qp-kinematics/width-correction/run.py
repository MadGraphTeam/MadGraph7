"""Approved command-only width correction; frozen first attempt remains intact."""
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
    assert not (HERE / 'execution-2.json').exists(), 'No overwrite or automatic retry.'
    assert sha(HERE / 'source-index.json') == '85f235a0219b3c5f4c47237d785ce75657c0cbca010aee83932d312b3eac3a37'
    assert sha(HERE / 'execution.json') == 'c2031cef0a4c32dade815d7df5f6d4e24266e2c7f5a8f2a9883d2b809d9c36f6'
    assert sha(HERE / 'real/compile.log') == 'a145d850a53771b66bc06fe35993313d61b85499e1b4f5853baa93b39ccbeba6'
    index = json.loads((HERE / 'source-index.json').read_text())
    protocol = json.loads((HERE / 'protocol.json').read_text())
    check(index, protocol)
    os.sched_setaffinity(0, {13})
    (HERE / 'tmp').mkdir(exist_ok=True)
    env = {'PATH': os.environ['PATH'], 'LC_ALL': 'C', 'LANG': 'C',
           'TMPDIR': str(HERE / 'tmp'), 'OMP_NUM_THREADS': '1'}
    result = {'complete': False, 'cpu': 13, 'deadline_seconds': 120,
              'source_index_sha256': sha(HERE / 'source-index.json'),
              'controller_sha256': sha(Path(__file__)),
              'failed_attempt_sha256': sha(HERE / 'execution.json'),
              'mechanical_correction': 'Add -ffixed-line-length-132 only; use distinct output names',
              'environment': env, 'steps': [], 'reducer_or_amplitude_executed': False}
    start = time.monotonic()
    try:
        for flavor in protocol['flavors']:
            binary = HERE / flavor / 'matrix-check-2'
            compile_command = [protocol['compiler']['path'], '-O0', '-g', '-fcheck=all', '-fbacktrace', '-ffixed-line-length-132',
                '-I' + str(HERE / flavor), str(HERE / 'driver.f90'), str(HERE / 'matrix.f'),
                str(HERE / 'off_shell.f'), str(HERE / flavor / 'model_context.f'), '-o', str(binary)]
            for phase, command in [('compile', compile_command), ('matrix', [str(binary), str(HERE / 'cases.dat')])]:
                remaining = 120 - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError('Combined job deadline exceeded')
                logpath = HERE / flavor / (phase + '-2.log')
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
            lines = (HERE / flavor / 'matrix-2.log').read_text().splitlines()
            assert lines[-1] == 'MATRIX_POLICY_PASS ' + str(protocol['cases_per_flavor'])
            assert sum(line.startswith('CASE_PASS ') for line in lines) == protocol['cases_per_flavor']
            result[flavor] = {'cases_passed': protocol['cases_per_flavor'], 'binary_sha256': sha(binary)}
        check(index, protocol)
        result['protected_inputs_unchanged'] = True
        result['complete'] = True
    finally:
        try:
            check(index, protocol)
            result['protected_inputs_unchanged'] = True
        except Exception as exc:
            result['protected_inputs_unchanged'] = False
            result['protection_error'] = repr(exc)
        result['elapsed_seconds'] = time.monotonic() - start
        (HERE / 'execution-2.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'complete': result['complete'], 'cases_per_flavor': protocol['cases_per_flavor'],
                      'execution_sha256': sha(HERE / 'execution-2.json')}))


if __name__ == '__main__':
    main()
