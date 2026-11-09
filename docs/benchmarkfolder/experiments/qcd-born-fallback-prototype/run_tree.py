#!/usr/bin/env python3
"""Qualify only Born fallback arithmetic/IEEE behavior on isolated CPU 11."""
from pathlib import Path
import hashlib
import json
import os
import resource
import shutil
import subprocess
import time

ROOT = Path.cwd().resolve()
REPORT = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype'
RESULT = REPORT / 'tree-native-2'
WORK = ROOT / '.tirust-build/qcd-born-fallback-prototype/tree-native-2'
PROOF = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-cancellation-audit'
PROC = ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg'
SP = PROC / 'SubProcesses/P0_uux_zgg'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def main():
    # This script is intentionally a one-shot namespace; failed attempts remain.
    RESULT.mkdir()
    WORK.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    prep = json.loads((REPORT / 'preparation.json').read_text())
    protected = dict(prep['files_sha256'])
    protected.update(json.loads((PROOF / 'preparation.json').read_text())['source_and_link_inputs'])
    protected[relative(REPORT / 'preparation.json')] = sha(REPORT / 'preparation.json')
    for file in ('tree_driver.f', 'run_tree.py'):
        protected[relative(REPORT / file)] = sha(REPORT / file)
    for path, expected in protected.items():
        assert sha(ROOT / path) == expected, path
    for file in ('tree_driver.f', 'born_fallback.f'):
        shutil.copy2(REPORT / file, WORK / file)
    for file in ('ident_card.dat', 'param_card.dat'):
        shutil.copy2(PROC / 'Cards' / file, WORK / file)
    shutil.copy2(ROOT / 'docs/benchmarkfolder/experiments/physical-stress/points/qcd_n3_uux_zgg/selected.dat', WORK / 'points.dat')
    fc = shutil.which('gfortran')
    assert fc
    common = ['taskset', '-c', '11', fc, '-w', '-fPIC', '-ffixed-line-length-132',
              '-I' + str(PROC / 'Source/MODEL'), '-I' + str(PROC / 'Source/DHELAS'), '-I' + str(SP)]
    commands = [[*common, '-c', str(WORK / file), '-o', str(WORK / (Path(file).stem + '.o'))]
                for file in ('tree_driver.f', 'born_fallback.f')]
    commands.append([*common, str(WORK / 'tree_driver.o'), str(WORK / 'born_fallback.o'),
                     str(SP / 'born_matrix.o'), str(PROC / 'lib/libdhelas.a'), str(PROC / 'lib/libmodel.a'),
                     '-o', str(WORK / 'tree-qualify')])
    record = {'schema': 1, 'scope': 'Standalone Born-only qualification; no native reducers or Symbolica.',
              'complete': False, 'commands': commands, 'cpu': 11,
              'compiler_version': subprocess.check_output([fc, '--version'], text=True).splitlines()[0],
              'protected_inputs_before': protected, 'native_reducers_called': False,
              'formal_performance_eligible': False, 'host_load_start': os.getloadavg(), 'runs': []}
    try:
        with (RESULT / 'build.log').open('x') as log:
            for command in commands:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
        record['artifacts'] = {relative(WORK / file): sha(WORK / file)
                               for file in ('tree_driver.o', 'born_fallback.o', 'tree-qualify')}
        ldd = subprocess.check_output(['ldd', str(WORK / 'tree-qualify')], text=True)
        (RESULT / 'ldd.txt').write_text(ldd)
        assert 'libmadloop_tirust' not in ldd and 'ninjars' not in ldd
        for mode in ('qualify', 'trap-original', 'trap-fallback', 'timing'):
            command = ['taskset', '-c', '11', str(WORK / 'tree-qualify'),
                       str(WORK / 'param_card.dat'), str(WORK / 'points.dat'), mode]
            start = time.monotonic()
            with (RESULT / (mode + '.log')).open('x') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            output = (RESULT / (mode + '.log')).read_text()
            run = {'mode': mode, 'command': command, 'returncode': result.returncode,
                   'wall_seconds': time.monotonic() - start, 'host_load_after': os.getloadavg(),
                   'log_sha256': sha(RESULT / (mode + '.log'))}
            record['runs'].append(run)
            if mode.startswith('trap-'):
                assert result.returncode == -8, (mode, result.returncode)
                assert 'TRAP_START' in output and 'TRAP_RETURNED_UNEXPECTEDLY' not in output
            else:
                assert result.returncode == 0, (mode, result.returncode)
            if mode == 'qualify':
                assert sum(line.startswith('ORDINARY_PARITY') for line in output.splitlines()) == 147
                assert sum(line.startswith('STRESS_RESCUE') for line in output.splitlines()) == 25
                for token in ('ENTRY_FLAG_PRESERVED', 'INVALID_QP_REJECTED',
                              'QP_CONVERSION_OVERFLOW_REJECTED', 'TREE_QUALIFICATION_PASS'):
                    assert token in output, token
            if mode == 'timing':
                assert sum(line.startswith('BORN_ONLY_TIMING') for line in output.splitlines()) == 7
        record['complete'] = True
    except BaseException as error:
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        record['protected_inputs_after'] = {path: sha(ROOT / path) for path in protected}
        record['protected_inputs_unchanged'] = record['protected_inputs_after'] == protected
        record['host_load_end'] = os.getloadavg()
        (RESULT / 'execution.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        assert record['protected_inputs_unchanged']
    print('Born-only qualification complete; whole-amplitude and host edge qualification remain pending.')


if __name__ == '__main__':
    main()
