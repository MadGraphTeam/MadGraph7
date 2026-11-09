#!/usr/bin/env python3
"""Trace historical EW threshold QP tensors without changing their arithmetic.

prepare is filesystem-only; build/run require explicit team lane coordination.
The old campaign and numerical objects remain untouched. No timing is eligible.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[3]
OLD = ROOT / 'docs/benchmarkfolder/experiments/physical-stress'
DIAGNOSTIC = OLD / 'diagnostics/ew_n3_epem_ttxa/candidate-9-qp-init-dp'
ORIGIN = ROOT / '.tirust-build/physical-stress/unchecked/ew_n3_epem_ttxa/candidate-9-qp-init-dp'
WORK = ROOT / '.tirust-build/ew-threshold-adjudication'
PROCESS = Path('SubProcesses/P0_epem_ttxa')
EXPECTED_DSO = '49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'
PRIVATE_DSO = ROOT / '.tirust-build/ew-threshold-geometry/private-libraries/libmadloop_tirust.so'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def inventory(path):
    return {str(p.relative_to(path)): digest(p)
            for p in sorted(list(path.rglob('*.o')) + list(path.rglob('*.a')))}


def checked_replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Expected exactly one anchor: ' + before)
    return text.replace(before, after)


def prepare():
    if WORK.exists() or (REPORT / 'prepared.json').exists():
        raise ValueError('Refusing to overwrite prepared evidence')
    old = json.loads((DIAGNOSTIC / 'diagnostic-compilation.json').read_text())
    for category in ('numerical_objects_after', 'sources'):
        for p, expected in old[category].items():
            if digest(ORIGIN / p) != expected:
                raise ValueError('Historical input differs: ' + p)
    if digest(PRIVATE_DSO) != EXPECTED_DSO:
        raise ValueError('Historical bridge snapshot mismatch')
    before = inventory(ORIGIN)
    shutil.copytree(ORIGIN, WORK, symlinks=True,
                    ignore=shutil.ignore_patterns('worker-raw', 'worker-protocol.fifo'))
    proc = WORK / PROCESS
    runtime = WORK / 'private-libraries'
    runtime.mkdir()
    shutil.copy2(PRIVATE_DSO, runtime / PRIVATE_DSO.name)
    for name in ('param_card.dat', 'MadLoopParams.dat'):
        dest = proc / name if name == 'param_card.dat' else proc / 'MadLoop5_resources' / name
        dest.unlink(missing_ok=True)
        shutil.copy2(DIAGNOSTIC / name, dest)
        shutil.copy2(DIAGNOSTIC / name, REPORT / name)
    points = OLD / 'points/ew_n3_epem_ttxa/selected.dat'
    shutil.copy2(points, REPORT / 'selected.dat')
    shutil.copy2(points, proc / 'selected.dat')
    sources = REPORT / 'sources'
    sources.mkdir()
    for name in ('check_sa.f', 'TIRUST_interface.f'):
        p = proc / name
        oldtext = p.read_text()
        (sources / (name + '.original')).write_text(oldtext)
        if name == 'check_sa.f':
            anchor = '            CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)'
            newtext = checked_replace(oldtext, anchor,
                '            CALL EW_TRACE_BEGIN(K)\n' + anchor +
                '\n            CALL EW_TRACE_END(P,ANS,ACC,CODE)')
        else:
            # Modify only the MP routine; DP initialization stays identical.
            marker = '      SUBROUTINE ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC'
            head, tail = oldtext.split(marker, 1)
            anchor = '     $ ,TIRUSTCOMPUTEFINITE,CHECK_PASS,CACHE_ENABLED)\n      IF(STATUS.NE.TIRUST_SUCCESS) RETURN'
            tail = checked_replace(tail, anchor, anchor.split('\n')[0] +
                '\n      CALL EW_TRACE_TENSOR(CTMODE,NLOOPLINE,RANK,SHIFTS,\n'
                '     $ INVARIANTS,MASSES,SCALE,LOOPMAXCOEFS,NC,RAW,STATUS)\n'
                '      IF(STATUS.NE.TIRUST_SUCCESS) RETURN')
            newtext = head + marker + tail
        p.unlink()
        p.write_text(newtext)
        (sources / name).write_text(newtext)
    shutil.copy2(REPORT / 'trace.f', proc / 'ew_trace.f')
    overlay = ('include makefile\n'
        'LINKLIBS := $(subst -lmadloop_tirust,' + str(runtime / PRIVATE_DSO.name) + ',$(LINKLIBS))\n'
        'trace-build: check_sa.o TIRUST_interface.o ew_trace.o\n'
        '\t$(FC) $(FFLAGS) -o check check_sa.o $(PROCESS) ew_trace.o $(LINKLIBS)\n')
    (proc / 'trace.mk').write_text(overlay)
    (REPORT / 'trace.mk').write_text(overlay)
    controller = OLD / 'protocol-sources/tirust_benchmark_worker.py'
    shutil.copy2(controller, sources / controller.name)
    dump(REPORT / 'prepared.json', {
        'origin': str(ORIGIN.relative_to(ROOT)), 'work': str(WORK.relative_to(ROOT)),
        'original_compilation_sha256': digest(DIAGNOSTIC / 'diagnostic-compilation.json'),
        'original_source_hashes': old['sources'], 'objects_archives_before': before,
        'point_sha256': digest(points), 'trace_sha256': digest(REPORT / 'trace.f'),
        'run_script_sha256': digest(Path(__file__)),
        'patched_source_hashes': {name: digest(proc / name) for name in ('check_sa.f', 'TIRUST_interface.f')},
        'card_sha256': {p: digest(REPORT / p) for p in ('param_card.dat', 'MadLoopParams.dat')},
        'private_dso_sha256': digest(runtime / PRIVATE_DSO.name),
        'controller_sha256': digest(sources / controller.name),
        'overlay_sha256': digest(REPORT / 'trace.mk'),
        'scope': 'Logging only; historical unchecked fixed-QP inputs and arithmetic retained.',
        'performance_eligible': False})


def verify(record):
    if inventory(ORIGIN) != record['objects_archives_before']:
        raise ValueError('Original numerical objects changed')
    for p, expected in record['original_source_hashes'].items():
        if digest(ORIGIN / p) != expected:
            raise ValueError('Original numerical source changed: ' + p)
    proc = WORK / PROCESS
    for p, expected in record['patched_source_hashes'].items():
        if digest(proc / p) != expected:
            raise ValueError('Diagnostic source changed: ' + p)
    for p, expected in record['card_sha256'].items():
        dest = proc / p if p == 'param_card.dat' else proc / 'MadLoop5_resources' / p
        if digest(dest) != expected:
            raise ValueError('Diagnostic card changed: ' + p)
    if digest(proc / 'selected.dat') != record['point_sha256']:
        raise ValueError('Point stream changed')
    if digest(WORK / 'private-libraries' / PRIVATE_DSO.name) != record['private_dso_sha256']:
        raise ValueError('Runtime snapshot changed')


def build():
    record = json.loads((REPORT / 'prepared.json').read_text())
    verify(record)
    proc = WORK / PROCESS
    command = ['make', '-f', 'trace.mk', 'SHELL=' + shutil.which('bash'), '-j1', 'trace-build']
    with (REPORT / 'build.log').open('x') as log:
        result = subprocess.run(command, cwd=proc, env=dict(os.environ, PWD=str(proc)),
                                stdout=log, stderr=subprocess.STDOUT, timeout=90)
    dump(REPORT / 'build-status.json', {'command': command, 'returncode': result.returncode})
    result.check_returncode()
    after = inventory(WORK)
    added = sorted(after.keys() - record['objects_archives_before'].keys())
    changed = sorted(p for p, v in record['objects_archives_before'].items() if after.get(p) != v)
    allowed = sorted(str(PROCESS / p) for p in ('check_sa.o', 'TIRUST_interface.o'))
    if changed != allowed or added != [str(PROCESS / 'ew_trace.o')]:
        raise ValueError('Unexpected changed numerical objects: ' + str((changed, added)))
    verify(record)
    dump(REPORT / 'compilation.json', {'objects_archives_after': after,
        'changed_original_objects': changed, 'added_private_objects': added,
        'original_objects_sources_unchanged': True,
        'executable_sha256': digest(proc / 'check'), 'build_log_sha256': digest(REPORT / 'build.log'),
        'inherited_effective_compiler_flags': json.loads((DIAGNOSTIC / 'diagnostic-compilation.json').read_text())['effective_compiler_flags']})


def run(cpu):
    record = json.loads((REPORT / 'prepared.json').read_text())
    compiled = json.loads((REPORT / 'compilation.json').read_text())
    verify(record)
    proc = WORK / PROCESS
    if inventory(WORK) != compiled['objects_archives_after'] or digest(proc / 'check') != compiled['executable_sha256']:
        raise ValueError('Diagnostic compiled inputs changed')
    controller = REPORT / 'sources/tirust_benchmark_worker.py'
    if digest(controller) != record['controller_sha256']:
        raise ValueError('Controller changed')
    spec = importlib.util.spec_from_file_location('ew_frozen_worker', controller)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = dict(os.environ, LD_LIBRARY_PATH=str(WORK / 'private-libraries') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    resolved = subprocess.check_output(['ldd', './check'], cwd=proc, env=env, text=True)
    (REPORT / 'ldd.txt').write_text(resolved)
    if str(WORK / 'private-libraries' / PRIVATE_DSO.name) not in resolved:
        raise ValueError('Historical private bridge not resolved')
    outcome = {'cpu': cpu, 'performance_eligible': False, 'stability_checked': False,
        'fixed_mode': 4, 'initialization_mode': 1, 'private_dso_sha256': record['private_dso_sha256'],
        'ldd_sha256': digest(REPORT / 'ldd.txt')}
    worker = None
    try:
        with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            with module.Worker(['taskset', '-c', str(cpu), './check', str(proc / 'selected.dat')],
                    proc, REPORT / 'output.log', env=env, timeout=180.) as worker:
                worker.run(1)
                shutil.copytree(worker.last_raw_directory, REPORT / 'raw')
        outcome['complete'] = True
    except BaseException as exc:
        outcome.update(complete=False, error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        outputs = ['ew-tensor-trace.dat']
        if worker is not None and worker.last_raw_directory is not None:
            outputs += ['values.dat', 'dimensions.dat', 'timed_status.dat']
        for name in outputs:
            if (proc / name).exists():
                shutil.copy2(proc / name, REPORT / name)
        verify(record)
        outcome['original_objects_sources_unchanged'] = True
        if (REPORT / 'values.dat').exists():
            old_values = DIAGNOSTIC / 'raw/values.dat'
            outcome['final_values_identical_to_historical'] = (REPORT / 'values.dat').read_bytes() == old_values.read_bytes()
            outcome['values_sha256'] = digest(REPORT / 'values.dat')
            outcome['historical_values_sha256'] = digest(old_values)
        dump(REPORT / 'execution.json', outcome)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'build', 'run'))
    parser.add_argument('--cpu', type=int, default=9)
    args = parser.parse_args()
    {'prepare': prepare, 'build': build, 'run': lambda: run(args.cpu)}[args.action]()
