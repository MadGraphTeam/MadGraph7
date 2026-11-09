#!/usr/bin/env python3
"""Bounded isolated Born-only original/promoted versus repaired-input replay."""
from pathlib import Path
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path.cwd().resolve()
HERE = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-trace/born-consistency'
DESIGN = HERE.parent / 'design'
FALLBACK = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype'
PROC = ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg'
SP = PROC / 'SubProcesses/P0_uux_zgg'
WORK = ROOT / '.tirust-build/qcd-loop-kinematics-trace/born-consistency/native-1'
OUT = HERE / 'native-1'
POINTS = ROOT / 'docs/benchmarkfolder/experiments/physical-stress/points/qcd_n3_uux_zgg/selected.dat'
MLCARD = ROOT / 'docs/benchmarkfolder/experiments/physical-stress/runs/qcd_n3_uux_zgg/candidate-9-auto/MadLoopParams.dat'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(ROOT))


def main():
    # Never replace an unsuccessful attempt or a frozen package.
    OUT.mkdir()
    WORK.mkdir(parents=True)
    assert sys.byteorder == 'little', 'Driver hex transport is pinned to x86_64 little endian'
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # Verify frozen designs and reviewed arithmetic sources before using them.
    for base in (DESIGN, FALLBACK):
        index = json.loads((base / 'evidence-index.json').read_text())
        for name, entry in index['files'].items():
            assert sha(base / name) == entry['sha256'], name
    bindings = json.loads((DESIGN / 'source-bindings.json').read_text())
    for name, entry in bindings['source_inputs'].items():
        assert sha(ROOT / name) == entry['sha256'], name
    frozen_card = ROOT / 'docs/benchmarkfolder/experiments/physical-stress/inputs/candidate/qcd_n3_uux_zgg/param_card.dat'
    assert sha(PROC / 'Cards/param_card.dat') == sha(frozen_card)
    old_tree = json.loads((FALLBACK / 'tree-native-2/execution.json').read_text())
    old_host = json.loads((FALLBACK / 'amplitude-native-1/build.json').read_text())
    historical = {}
    for path in (SP / 'born_matrix.o', SP / 'improve_ps.o',
                 PROC / 'lib/libdhelas.a', PROC / 'lib/libmodel.a',
                 PROC / 'Source/MODEL/coupl.inc',
                 PROC / 'Source/MODEL/model_object.mod',
                 PROC / 'Source/DHELAS/aloha_object.mod'):
        expected = old_tree['protected_inputs_before'].get(rel(path))
        if expected is None:
            expected = old_host['protected_inputs_before'][rel(path)]
        assert sha(path) == expected, rel(path)
        historical[rel(path)] = expected
    source = (SP / 'loop_matrix.f').read_text()
    start = source.index('      SUBROUTINE ML5_0_SET_MP_PS(P)')
    end = source.index('\n      END\n', start) + len('\n      END\n')
    extracted = source[start:end]
    assert 'CALL ML5_0_MP_IMPROVE_PS_POINT_PRECISION' in extracted
    (HERE / 'set_mp_ps.f').write_text(extracted)
    copies = {
        'driver.f': HERE / 'driver.f',
        'set_mp_ps.f': HERE / 'set_mp_ps.f',
        'born_fallback.f': FALLBACK / 'born_fallback.f',
        'MadLoopParamReader.f': PROC / 'SubProcesses/MadLoopParamReader.f',
        'points.dat': POINTS,
        'param_card.dat': PROC / 'Cards/param_card.dat',
        'ident_card.dat': PROC / 'Cards/ident_card.dat',
        'MadLoopParams.dat': MLCARD,
    }
    for name, path in copies.items():
        shutil.copy2(path, WORK / name)
    paths = list(copies.values()) + [
        HERE / 'run.py', DESIGN / 'evidence-index.json',
        FALLBACK / 'evidence-index.json', SP / 'loop_matrix.f',
        SP / 'born_matrix.f', SP / 'born_matrix.o',
        SP / 'improve_ps.f', SP / 'improve_ps.o',
        PROC / 'SubProcesses/MadLoopParams.inc',
        PROC / 'Source/MODEL/coupl.inc', PROC / 'Source/MODEL/mp_coupl.inc',
        PROC / 'Source/MODEL/couplings.f', PROC / 'Source/MODEL/actualize_mp_ext_params.inc',
        PROC / 'Source/MODEL/model_object.mod', PROC / 'Source/DHELAS/aloha_object.mod',
        PROC / 'lib/libdhelas.a', PROC / 'lib/libmodel.a',
        ROOT / 'HEPTools/tirust/lib/libmadloop_tirust.so',
        FALLBACK / 'amplitude-native-1/adjudication.json',
    ]
    protected = {rel(path): sha(path) for path in paths}
    fc = shutil.which('gfortran')
    assert fc
    common = ['taskset', '-c', '11', fc, '-w', '-fPIC', '-ffixed-line-length-132',
              '-I' + str(PROC / 'Source/MODEL'), '-I' + str(PROC / 'Source/DHELAS'),
              '-I' + str(SP), '-I' + str(PROC / 'SubProcesses')]
    sources = ('driver', 'set_mp_ps', 'born_fallback', 'MadLoopParamReader')
    commands = [[*common, '-c', str(WORK / (name + '.f')), '-o', str(WORK / (name + '.o'))]
                for name in sources]
    exe = WORK / 'born-consistency'
    commands.append([*common, *[str(WORK / (name + '.o')) for name in sources],
                     str(SP / 'born_matrix.o'), str(SP / 'improve_ps.o'),
                     str(PROC / 'lib/libdhelas.a'), str(PROC / 'lib/libmodel.a'),
                     '-o', str(exe)])
    record = {
        'schema': 1, 'scope': 'Born-only B/C effective-kinematics diagnostic; repaired C is not unchanged-input acceptance.',
        'complete': False, 'native_reducers_called': False, 'cpu': 11,
        'timeout_per_subprocess_seconds': 180, 'performance_eligible': False,
        'commands': commands, 'protected_inputs_before': protected,
        'compiler_version': subprocess.check_output([fc, '--version'], text=True, timeout=30).splitlines()[0],
        'host_load_start': os.getloadavg(),
        'historical_native_inputs_reverified': historical,
        'set_mp_ps_extraction': {'source': rel(SP / 'loop_matrix.f'),
                                 'first_line': source[:start].count('\n') + 1,
                                 'sha256': sha(HERE / 'set_mp_ps.f'), 'modified': False},
    }
    try:
        with (OUT / 'build.log').open('x') as log:
            for command in commands:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=180)
        record['artifacts_sha256'] = {rel(path): sha(path) for path in
                                     [exe, *[WORK / (name + '.o') for name in sources]]}
        ldd = subprocess.check_output(['ldd', str(exe)], text=True, timeout=30)
        (OUT / 'ldd.txt').write_text(ldd)
        assert 'libmadloop_tirust' not in ldd and 'ninjars' not in ldd
        command = ['taskset', '-c', '11', str(exe), str(WORK / 'param_card.dat'),
                   str(WORK / 'points.dat'), str(WORK / 'MadLoopParams.dat')]
        record['execution_command'] = command
        started = time.monotonic()
        with (OUT / 'output.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
        record['wall_seconds'] = time.monotonic() - started
        lines = (OUT / 'output.log').read_text().splitlines()
        assert any(line.startswith('CONSISTENCY_COMPLETE') for line in lines)
        assert sum(line.startswith('B_COORD ') for line in lines) == 40
        assert sum(line.startswith('C_COORD ') for line in lines) == 40
        assert sum(line.startswith('B_BORN ') for line in lines) == 4
        assert sum(line.startswith('C_BORN ') for line in lines) == 4
        record['complete'] = True
    except BaseException as error:
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        record['protected_inputs_after'] = {name: sha(ROOT / name) for name in protected}
        record['protected_inputs_unchanged'] = record['protected_inputs_after'] == protected
        record['host_load_end'] = os.getloadavg()
        record['logs_sha256'] = {name: sha(OUT / name) for name in
                                ('build.log', 'ldd.txt', 'output.log') if (OUT / name).exists()}
        (OUT / 'execution.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        assert record['protected_inputs_unchanged']
    print('Bounded Born-only B/C replay complete; no reducer or physical acceptance.')


if __name__ == '__main__':
    main()
