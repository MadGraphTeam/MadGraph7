#!/usr/bin/env python3
"""Resume native qualification after the retained smoke module-shadowing failure."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
BUILD = ROOT / '.tirust-build/tirust-split-release'
PREFIX = ROOT / '.tirust-build/tirust-split-install'
OLD = ROOT / 'HEPTools/tirust'
assert Path.cwd().resolve() == ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


config = json.loads((PREFIX / 'lib/tirust-config.json').read_text())
record = config['bridge_rebuild_provenance']
assert record['builder_sha256'] == digest(HERE / 'build_release_attempt1.py')
assert record['source_sha256'] == digest(ROOT / 'vendor/tirust/src/lib.rs')
assert record['library_sha256'] == digest(PREFIX / 'lib/libmadloop_tirust.so')
record['resumer_sha256'] = digest(Path(__file__))
record['resumption_reason'] = ('The original smoke compiler selected an ignored stale module beside vendor/tirust/smoke.f90. Copying the exact source into the isolated build directory lets the compiler resolve the explicitly requested installed module. No installed native artifacts changed.')
record['failed_smoke_build_log_sha256'] = digest(HERE / 'release-smoke-build-attempt1.log')
record['shadowing_module_sha256'] = digest(ROOT / 'vendor/tirust/mtirust.mod')
smoke_source = BUILD / 'smoke.f90'
shutil.copyfile(ROOT / 'vendor/tirust/smoke.f90', smoke_source)
smoke = BUILD / 'smoke'
command = [os.environ.get('FC', 'gfortran'), '-O3', '-fPIC', str(smoke_source),
           '-I' + str(PREFIX / 'lib'), '-lgcc_s', str(PREFIX / 'lib/libtirust.a'),
           '-L' + str(PREFIX / 'lib'), '-lmadloop_tirust',
           '-Wl,-rpath,' + str(PREFIX / 'lib'), '-o', str(smoke)]
with (HERE / 'release-smoke-build.log').open('w') as log:
    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
assert result.returncode == 0
record['smoke_build_command'] = command
record['smoke_source_sha256'] = digest(smoke_source)
record['smoke_executable_sha256'] = digest(smoke)
with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    with (HERE / 'release-smoke.log').open('w') as log:
        result = subprocess.run([str(smoke), '--require-poles'], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT)
record['smoke_exit_code'] = result.returncode
record['smoke_log_sha256'] = digest(HERE / 'release-smoke.log')
record['old_prefix_unchanged'] = (
    digest(OLD / 'lib/libmadloop_tirust.so') == record['input_sha256']['HEPTools/tirust/lib/libmadloop_tirust.so']
    and digest(OLD / 'lib/tirust-config.json') == record['old_config_sha256'])
assert record['old_prefix_unchanged']
assert record['source_sha256'] == digest(ROOT / 'vendor/tirust/src/lib.rs')
(HERE / 'release-build-metadata.json').write_text(json.dumps(record, indent=2) + '\n')
assert result.returncode == 0
print(json.dumps({'prefix': str(PREFIX), 'library_sha256': record['library_sha256'],
                  'smoke_exit_code': result.returncode}))
