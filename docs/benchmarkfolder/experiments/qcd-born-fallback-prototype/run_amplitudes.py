#!/usr/bin/env python3
"""Bounded private host replay with exact historical cards and shared lock."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path.cwd().resolve()
REPORT = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype'
OUT = REPORT / 'amplitude-native-1'
WORK = ROOT / '.tirust-build/qcd-born-fallback-prototype/amplitude-native-1'
PROC = ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg'
PHYSICAL = ROOT / 'docs/benchmarkfolder/experiments/physical-stress'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def relative(p):
    return str(p.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=int, choices=(1, 8, 9), required=True)
    args = parser.parse_args()
    build = json.loads((OUT / 'build.json').read_text())
    assert build['complete'] and build['protected_inputs_unchanged']
    result_dir = OUT / ('backend-' + str(args.backend))
    result_dir.mkdir()
    checks = dict(build['protected_inputs_before'])
    checks.update(build['artifacts_sha256'])
    checks[relative(OUT / 'build.json')] = sha(OUT / 'build.json')
    checks[relative(REPORT / 'run_amplitudes.py')] = sha(REPORT / 'run_amplitudes.py')
    card = PHYSICAL / f'runs/qcd_n3_uux_zgg/candidate-{args.backend}-auto/MadLoopParams.dat'
    checks[relative(card)] = sha(card)
    env = os.environ.copy()
    native = [ROOT / 'HEPTools/tirust/lib', ROOT / 'HEPTools/ninja/lib', ROOT / 'HEPTools/collier']
    env['LD_LIBRARY_PATH'] = ':'.join(map(str, native)) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    record = {'schema': 1, 'complete': False, 'backend': args.backend, 'cpu': 11,
              'scope': 'Checked original/prototype host replay; no changed kinematics or stability policy.',
              'formal_performance_eligible': False, 'timeout_per_process_seconds': 180,
              'protected_inputs_before': checks, 'runs': [], 'runner_sha256': sha(REPORT / 'run_amplitudes.py')}
    # Real reducers may initialize Symbolica; every subprocess is serialized.
    with (ROOT / '.tirust-build/numerical-tests.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for p, expected in checks.items():
            assert sha(ROOT / p) == expected, p
        try:
            for mode in ('original', 'prototype'):
                private = WORK / f'backend-{args.backend}-{mode}'
                private.mkdir()
                resources = private / 'MadLoop5_resources'
                resources.mkdir()
                for name in ('ML5_0_ColorDenomFactors.dat', 'ML5_0_ColorNumFactors.dat', 'ML5_0_HelConfigs.dat'):
                    path = PROC / 'SubProcesses/MadLoop5_resources' / name
                    checks[relative(path)] = sha(path)
                    shutil.copy2(path, resources / name)
                shutil.copy2(card, resources / 'MadLoopParams.dat')
                command = ['taskset', '-c', '11', str(WORK / ('amplitude-' + mode)),
                           str(WORK / 'param_card.dat'), str(WORK / 'points.dat'), str(resources)]
                runtime = subprocess.check_output(['ldd', str(WORK / ('amplitude-' + mode))], env=env, text=True)
                (result_dir / (mode + '-ldd.txt')).write_text(runtime)
                actual_dso = next(line.split('=>', 1)[1].split('(')[0].strip() for line in runtime.splitlines()
                                  if 'libmadloop_tirust.so =>' in line)
                assert Path(actual_dso).resolve() == (ROOT / 'HEPTools/tirust/lib/libmadloop_tirust.so').resolve()
                run = {'mode': mode, 'command': command, 'host_load_start': os.getloadavg(),
                       'runtime_tirust_path': actual_dso, 'runtime_tirust_sha256': sha(Path(actual_dso))}
                record['runs'].append(run)
                start = time.monotonic()
                try:
                    with (result_dir / (mode + '.log')).open('x') as log:
                        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, timeout=180)
                    run['returncode'] = completed.returncode
                    output = (result_dir / (mode + '.log')).read_text()
                    run['all_calls_returned'] = 'AMPLITUDE_DIAGNOSTIC_COMPLETE' in output
                    assert completed.returncode == 0 and run['all_calls_returned'], mode
                    assert sum(line.startswith('AMPLITUDE_VALUE') for line in output.splitlines()) == 6
                finally:
                    run['wall_seconds'] = time.monotonic() - start
                    run['host_load_end'] = os.getloadavg()
                    run['log_sha256'] = sha(result_dir / (mode + '.log'))
                    run['resource_sha256'] = {relative(p): sha(p) for p in sorted(resources.iterdir()) if p.is_file()}
            record['complete'] = True
        except BaseException as e:
            record['error'] = {'type': type(e).__name__, 'message': str(e)}
            raise
        finally:
            record['protected_inputs_after'] = {p: sha(ROOT / p) for p in checks}
            record['protected_inputs_unchanged'] = record['protected_inputs_after'] == checks
            (result_dir / 'execution.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
            assert record['protected_inputs_unchanged']
    print('Completed actual host replay; numerical adjudication is separate from execution success.')


if __name__ == '__main__':
    main()
