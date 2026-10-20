#!/usr/bin/env python3
"""Append the byte-identical installed Fortran wrapper object to a private copy."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from build import HERE, ROOT, WORK, binding, sha, write

assert Path.cwd() == ROOT
os.sched_setaffinity(0, {234})
build = json.loads((HERE / 'build-retry-1/result.json').read_text())
assert build['status'] == 'compiled-not-executed'
baseline = ROOT / build['baseline_archive']['path']
rust_archive = ROOT / build['allocation_archive']['path']
assert binding(baseline) == build['baseline_archive']
assert binding(rust_archive) == build['allocation_archive']
names = subprocess.check_output(['ar','t',str(baseline)],text=True).splitlines()
assert names.count('mninjars.o') == 1
assert 'mninjars.o' not in subprocess.check_output(['ar','t',str(rust_archive)],text=True).splitlines()
out = WORK / 'packaged'
out.mkdir()
obj = out / 'mninjars.o'
obj.write_bytes(subprocess.check_output(['ar','p',str(baseline),'mninjars.o']))
archive = out / 'libninjars.a'
shutil.copy2(rust_archive,archive)
subprocess.run(['ar','rcs',str(archive),str(obj)],check=True)
assert subprocess.check_output(['ar','p',str(archive),'mninjars.o']) == obj.read_bytes()
assert subprocess.check_output(['ar','t',str(archive)],text=True).splitlines().count('mninjars.o') == 1
fingerprints = []
for p in sorted((WORK / 'build/release/.fingerprint').glob('madloop-ninjars-*/lib-madloop_ninjars.json')):
    data = json.loads(p.read_text())
    fingerprints.append({'file':binding(p),'contents':data})
    shutil.copy2(p,HERE / ('fingerprint-'+p.parent.name+'.json'))
write(HERE / 'package.json', {'schema':'allocation-process-packaged-archive-v1',
    'controller':binding(Path(__file__)),'build':binding(HERE / 'build-retry-1/result.json'),
    'baseline_archive':binding(baseline),'bare_rust_archive':binding(rust_archive),
    'wrapper_object':binding(obj),'allocation_archive':binding(archive),
    'wrapper_policy':'Exact installed mninjars.o extracted from baseline; no Fortran recompilation',
    'cargo_fingerprints':fingerprints,'production_changed':False,'numerical_execution':False})
print(json.dumps({'packaged_archive':binding(archive),'wrapper_object':binding(obj)}),flush=True)
