#!/usr/bin/env python3
"""Run the authorized fixed old-runtime replay, retaining every result/failure."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
sys.dont_write_bytecode = True
import matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = ROOT / '.benchmark-work/experiments/ninjars-qp-numerator-dp-probe-v1/replay'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd() == ROOT
    prepared = json.loads((HERE / 'prepared.json').read_text())
    assert prepared['complete'] and prepared['engine_calls'] == 0
    assert sha(HERE.parent / 'manifest.json') == prepared['capture_manifest_sha256']
    seal = json.loads((HERE.parent / 'manifest.json').read_text())
    for name, record in seal['files'].items():
        assert sha(HERE.parent / name) == record['sha256'], name
    expected = {WORK / 'replay': prepared['executable_sha256'],
                WORK / 'inputs.hex': prepared['inputs_sha256'],
                WORK / 'driver.f90': prepared['driver_sha256'],
                HERE / 'matrix.py': prepared['matrix_sha256'],
                HERE / 'PLAN.md': prepared['protocol_sha256']}
    expected.update({WORK / 'lib' / name: digest for name, digest in prepared['runtime_assets'].items()})
    for path, digest in expected.items():
        assert sha(path) == digest, path
    ldd = subprocess.check_output(['ldd', str(WORK / 'replay')], text=True)
    assert 'not found' not in ldd
    (HERE / 'ldd.txt').write_text(ldd)
    shared = {}
    for line in ldd.splitlines():
        matches = re.findall(r'(?:=>\s+)?(/[^\s]+)', line)
        for name in matches:
            path = Path(name).resolve(strict=True)
            shared[name] = {'resolved': str(path), 'sha256': sha(path)}
    command = ['taskset', '-c', '2', str(WORK / 'replay'), str(WORK / 'inputs.hex'), str(HERE / 'raw.hex')]
    result = {'complete': False, 'interpretation_permitted': False,
              'production_modified': False, 'runtime': 'captured original ABI3, including signed-zero defect',
              'prepared_sha256': sha(HERE / 'prepared.json'), 'runner_sha256': sha(Path(__file__)),
              'command': command, 'timeout_seconds': 900, 'maximum_engine_calls': 3712,
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'shared_libraries': shared, 'cpu': 2, 'performance_claim': False}
    start = time.monotonic()
    try:
        with (HERE / 'run.stdout').open('x') as out, (HERE / 'run.stderr').open('x') as err:
            process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, start_new_session=True)
            result['pid'] = process.pid
            (HERE / 'launch.json').write_text(json.dumps(result, indent=2) + '\n')
            try:
                result['returncode'] = process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                result['returncode'] = process.returncode
                raise
        assert result['returncode'] == 0
        for path, digest in expected.items():
            assert sha(path) == digest, path
        for name, record in shared.items():
            assert str(Path(name).resolve(strict=True)) == record['resolved']
            assert sha(Path(record['resolved'])) == record['sha256']
        analysis = matrix.check(HERE / 'raw.hex')
        for cell in analysis['cells']:
            cell['valid_for_comparison'] = cell['status'] == 0 and not cell['nonfinite_components']
        (HERE / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
        assert analysis['records'] == 3712
        assert len(analysis['reproduction_controls']) == 1392
        assert analysis['all_actual_input_words_match']
        assert analysis['all_reproduction_controls_exact'], 'retain complete rows; interpretation withheld'
        result['interpretation_permitted'] = True
        result['complete'] = True
    except BaseException as error:
        result['failure'] = repr(error)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - start
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result['files'] = {name: sha(HERE / name) for name in
                           ('raw.hex', 'run.stdout', 'run.stderr', 'analysis.json', 'ldd.txt')
                           if (HERE / name).is_file()}
        with (HERE / 'result.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
