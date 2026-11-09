#!/usr/bin/env python3
"""Rebuild the production bridge into a new prefix, preserving the c91 install."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = Path(__file__).resolve().parent
BUILD = ROOT / '.tirust-build/tirust-split-release'
PREFIX = ROOT / '.tirust-build/tirust-split-install'
OLD = ROOT / 'HEPTools/tirust'
assert Path.cwd().resolve() == ROOT
assert not PREFIX.exists(), 'Use a new prefix; never overwrite a qualified artifact.'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


previous = json.loads((EVIDENCE.parent / 'bridge/build-metadata.json').read_text())
for name, expected in previous['input_sha256'].items():
    assert digest(ROOT / name) == expected, name
source = ROOT / 'vendor/tirust/src/lib.rs'
assert source.read_bytes() == (ROOT / '.tirust-build/bridge-split-production/src/lib.rs').read_bytes()
(BUILD / 'src').mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, BUILD / 'src/lib.rs')
command = [arg.replace(str(ROOT / '.tirust-build/n1-profile/bridge'), str(BUILD))
           for arg in previous['command']]
record = {
    'scope': 'Production bridge, without diagnostic instrumentation; cached exact release dependencies.',
    'command': command, 'rustc': previous['rustc'],
    'source_sha256': digest(source), 'builder_sha256': digest(Path(__file__)),
    'input_sha256': previous['input_sha256'],
    'old_prefix': str(OLD.relative_to(ROOT)), 'new_prefix': str(PREFIX.relative_to(ROOT)),
    'old_config_sha256': digest(OLD / 'lib/tirust-config.json'),
    'fortran_wrapper_reused': 'Unchanged installed archive/module, identified by the original config hashes.',
}
(EVIDENCE / 'release-preparation.json').write_text(json.dumps(record, indent=2) + '\n')
with (EVIDENCE / 'release-build.log').open('w') as log:
    log.write(json.dumps(command) + '\n'); log.flush()
    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
assert result.returncode == 0, 'Inspect release-build.log'
library = BUILD / 'libmadloop_tirust.so'


def exports(path):
    output = subprocess.check_output(['nm', '-D', '--defined-only', str(path)], text=True)
    return output, {line.split()[-1] for line in output.splitlines() if line.strip()}


output, actual = exports(library)
_, expected = exports(OLD / 'lib/libmadloop_tirust.so')
assert actual == expected
assert all(name.startswith('ml_tirust_') and 'profile' not in name for name in actual)
(EVIDENCE / 'release-exports.txt').write_text(output)
shutil.copytree(OLD, PREFIX)
shutil.copyfile(library, PREFIX / 'lib/libmadloop_tirust.so')
config = json.loads((PREFIX / 'lib/tirust-config.json').read_text())
config['artifact_sha256']['libmadloop_tirust.so'] = digest(library)
config['production_source_validation']['bridge_sources_sha256']['src/lib.rs'] = digest(source)
record['library_sha256'] = digest(library)
record['source_unchanged_after_build'] = digest(source) == record['source_sha256']
assert record['source_unchanged_after_build']
config['bridge_rebuild_provenance'] = record
(PREFIX / 'lib/tirust-config.json').write_text(json.dumps(config, indent=2) + '\n')
for name, expected in config['artifact_sha256'].items():
    assert digest(PREFIX / 'lib' / name) == expected
smoke = BUILD / 'smoke'
smoke_source = BUILD / 'smoke.f90'
shutil.copyfile(ROOT / 'vendor/tirust/smoke.f90', smoke_source)
smoke_command = [os.environ.get('FC', 'gfortran'), '-O3', '-fPIC',
    str(smoke_source), '-I' + str(PREFIX / 'lib'),
    '-lgcc_s', str(PREFIX / 'lib/libtirust.a'), '-L' + str(PREFIX / 'lib'),
    '-lmadloop_tirust', '-Wl,-rpath,' + str(PREFIX / 'lib'), '-o', str(smoke)]
with (EVIDENCE / 'release-smoke-build.log').open('w') as log:
    result = subprocess.run(smoke_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
assert result.returncode == 0, 'Inspect release-smoke-build.log'
record['smoke_build_command'] = smoke_command
record['smoke_source_sha256'] = digest(ROOT / 'vendor/tirust/smoke.f90')
record['smoke_executable_sha256'] = digest(smoke)
with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    with (EVIDENCE / 'release-smoke.log').open('w') as log:
        result = subprocess.run([str(smoke), '--require-poles'], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT)
record['smoke_exit_code'] = result.returncode
record['smoke_log_sha256'] = digest(EVIDENCE / 'release-smoke.log')
record['old_prefix_unchanged'] = (
    digest(OLD / 'lib/libmadloop_tirust.so') == previous['input_sha256']['HEPTools/tirust/lib/libmadloop_tirust.so']
    and digest(OLD / 'lib/tirust-config.json') == record['old_config_sha256'])
assert record['old_prefix_unchanged']
(EVIDENCE / 'release-build-metadata.json').write_text(json.dumps(record, indent=2) + '\n')
assert result.returncode == 0, 'Inspect release-smoke.log'
print(json.dumps({'prefix': str(PREFIX), 'library_sha256': record['library_sha256'],
                  'smoke_exit_code': result.returncode}))
