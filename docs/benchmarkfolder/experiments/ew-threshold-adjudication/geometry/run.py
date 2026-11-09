#!/usr/bin/env python3
"""Link a private geometry driver to unchanged generated numerical objects.

Run `prepare` first; `execute` requires the coordinated native-job quiet window.
No amplitude or scalar library API is called by the driver.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[4]
ORIGIN = ROOT / '.tirust-build/physical-stress/unchecked/ew_n3_epem_ttxa/candidate-9-qp-init-dp'
WORK = ROOT / '.tirust-build/ew-threshold-geometry'
PROCESS = Path('SubProcesses/P0_epem_ttxa')
OLD = ROOT / 'docs/benchmarkfolder/experiments/physical-stress'
DIAGNOSTIC = OLD / 'diagnostics/ew_n3_epem_ttxa/candidate-9-qp-init-dp'
DSO = ROOT / 'HEPTools/tirust/lib/libmadloop_tirust.so'
EXPECTED_DSO = '49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def inventory(path):
    return {str(p.relative_to(path)): digest(p)
            for p in sorted(list(path.rglob('*.o')) + list(path.rglob('*.a')))}


def prepare():
    if WORK.exists() or (REPORT / 'prepared.json').exists():
        raise ValueError('Refusing to overwrite prepared evidence')
    old = json.loads((DIAGNOSTIC / 'diagnostic-compilation.json').read_text())
    for p, expected in old['numerical_objects_after'].items():
        if digest(ORIGIN / p) != expected:
            raise ValueError('Historical object differs: ' + p)
    for p, expected in old['sources'].items():
        if digest(ORIGIN / p) != expected:
            raise ValueError('Historical source differs: ' + p)
    if digest(DSO) != EXPECTED_DSO:
        raise ValueError('Expected historical bridge DSO is not installed')
    before = inventory(ORIGIN)
    shutil.copytree(ORIGIN, WORK, symlinks=True)
    proc = WORK / PROCESS
    private = WORK / 'private-libraries'
    private.mkdir()
    shutil.copy2(DSO, private / DSO.name)
    shutil.copy2(REPORT / 'probe.f', proc / 'geometry_probe.f')
    points = OLD / 'points/ew_n3_epem_ttxa/selected.dat'
    shutil.copy2(points, REPORT / 'selected.dat')
    shutil.copy2(points, proc / 'selected.dat')
    for name in ('param_card.dat', 'MadLoopParams.dat'):
        # Replace only the private clone's card symlink, if present.
        (proc / name).unlink(missing_ok=True)
        shutil.copy2(DIAGNOSTIC / name, proc / name)
        shutil.copy2(DIAGNOSTIC / name, REPORT / name)
    overlay = ('include makefile\n'
        'LINKLIBS := $(subst -lmadloop_tirust,' + str(private / DSO.name) + ',$(LINKLIBS))\n'
        'geometry-probe:\n'
        '\t$(FC) $(FFLAGS) -c geometry_probe.f -o geometry_probe.o $(LOOP_INCLUDE)\n'
        '\t$(FC) $(FFLAGS) -o geometry-probe geometry_probe.o $(PROCESS) $(LINKLIBS)\n')
    (proc / 'geometry.mk').write_text(overlay)
    (REPORT / 'geometry.mk').write_text(overlay)
    dump(REPORT / 'prepared.json', {
        'origin': str(ORIGIN.relative_to(ROOT)), 'work': str(WORK.relative_to(ROOT)),
        'original_compilation_sha256': digest(DIAGNOSTIC / 'diagnostic-compilation.json'),
        'original_source_hashes': old['sources'],
        'objects_archives_before': before,
        'point_sha256': digest(points), 'driver_sha256': digest(REPORT / 'probe.f'),
        'run_script_sha256': digest(Path(__file__)),
        'card_sha256': {p: digest(REPORT / p) for p in ('param_card.dat', 'MadLoopParams.dat')},
        'private_dso_sha256': digest(private / DSO.name),
        'overlay_sha256': digest(REPORT / 'geometry.mk'),
        'scope': 'Geometry only; canonical representative shifts, not per-diagram captured topology.'})


def execute():
    record = json.loads((REPORT / 'prepared.json').read_text())
    proc = WORK / PROCESS
    if inventory(ORIGIN) != record['objects_archives_before']:
        raise ValueError('Original numerical inputs changed')
    if inventory(WORK) != record['objects_archives_before']:
        raise ValueError('Private numerical inputs changed')
    if digest(proc / 'geometry_probe.f') != record['driver_sha256']:
        raise ValueError('Private driver changed')
    if digest(WORK / 'private-libraries' / DSO.name) != record['private_dso_sha256']:
        raise ValueError('Private DSO changed')
    command = ['make', '-f', 'geometry.mk', 'SHELL=' + shutil.which('bash'), '-j1', 'geometry-probe']
    env = dict(os.environ, PWD=str(proc))
    with (REPORT / 'build.log').open('x') as log:
        result = subprocess.run(command, cwd=proc, env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=60)
    dump(REPORT / 'build-status.json', {'command': command, 'returncode': result.returncode})
    result.check_returncode()
    after = inventory(WORK)
    added = sorted(after.keys() - record['objects_archives_before'].keys())
    changed = [p for p, v in record['objects_archives_before'].items() if after.get(p) != v]
    if changed or added != [str(PROCESS / 'geometry_probe.o')]:
        raise ValueError('Unexpected changes to numerical objects')
    # The linked DSO is inert: no amplitude, tensor, or scalar evaluation occurs.
    command = [str(proc / 'geometry-probe')]
    env['LD_LIBRARY_PATH'] = str(WORK / 'private-libraries') + ':' + env.get('LD_LIBRARY_PATH', '')
    with (REPORT / 'raw.log').open('x') as log:
        result = subprocess.run(command, cwd=proc, env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=30)
    unchanged = inventory(ORIGIN) == record['objects_archives_before']
    dump(REPORT / 'execution.json', {
        'command': command, 'returncode': result.returncode,
        'cwd': str(proc.relative_to(ROOT)), 'origin_numerical_inputs_unchanged': unchanged,
        'changed_original_objects': changed, 'added_private_objects': added,
        'objects_archives_after': after, 'executable_sha256': digest(proc / 'geometry-probe'),
        'driver_sha256': digest(proc / 'geometry_probe.f'),
        'private_dso_sha256': digest(WORK / 'private-libraries' / DSO.name),
        'raw_log_sha256': digest(REPORT / 'raw.log'), 'build_log_sha256': digest(REPORT / 'build.log'),
        'point_sha256': digest(proc / 'selected.dat'),
        'card_sha256': {p: digest(proc / p) for p in ('param_card.dat', 'MadLoopParams.dat')},
        'no_reducer_api_calls': True})
    result.check_returncode()
    if not unchanged:
        raise ValueError('Original inputs changed during execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'execute'))
    args = parser.parse_args()
    {'prepare': prepare, 'execute': execute}[args.action]()
