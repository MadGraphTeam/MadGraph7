#!/usr/bin/env python3
"""Qualify the new release prefix without modifying either installed library."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
NEW = ROOT / '.tirust-build/tirust-split-install/lib'
OLD = ROOT / 'HEPTools/tirust/lib'
assert Path.cwd().resolve() == ROOT
sys.path.insert(0, str(ROOT))
from madgraph.various import tirust_install


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


metadata = json.loads((HERE / 'release-build-metadata.json').read_text())
assert metadata['smoke_exit_code'] == 0
assert digest(NEW / 'libmadloop_tirust.so') == metadata['library_sha256']
assert digest(ROOT / 'vendor/tirust/src/lib.rs') == metadata['source_sha256']
assert tirust_install.is_available(NEW)
assert tirust_install.is_available(OLD)
old_hashes = {p.name: digest(p) for p in OLD.iterdir() if p.is_file()}
record = {'scope': 'Release ABI/cache parity and native artifact qualification; no timing or full physics claim.',
          'qualifier_sha256': digest(Path(__file__)),
          'release_build_metadata_sha256': digest(HERE / 'release-build-metadata.json'),
          'production_source_sha256': metadata['source_sha256'],
          'original_library_sha256': digest(OLD / 'libmadloop_tirust.so'),
          'new_library_sha256': digest(NEW / 'libmadloop_tirust.so'),
          'original_config_sha256': digest(OLD / 'tirust-config.json'),
          'new_config_sha256': digest(NEW / 'tirust-config.json'),
          'new_installation_available': True, 'original_installation_available': True}
for name, directory in [('original', OLD), ('new', NEW)]:
    for tool, arguments in [('readelf', ['-d']), ('ldd', [])]:
        result = subprocess.run([tool] + arguments + [str(directory / 'libmadloop_tirust.so')],
                                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode == 0
        path = HERE / (name + '-' + tool + '.txt')
        path.write_text(result.stdout)
        record[path.name + '_sha256'] = digest(path)
command = [sys.executable, str(HERE / 'validate_release.py'),
           '--output', str(HERE / 'release-validation.json')]
record['validation_command'] = command
result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
assert result.returncode == 0
validation = json.loads((HERE / 'release-validation.json').read_text())
assert validation['comparison']['passed']
assert validation['comparison']['cache_statistics_identical']
assert validation['call_counts'] == {'original': 90, 'split': 90}
workers = {name: json.loads((ROOT / result['result']).read_text())
           for name, result in validation['workers'].items()}
assert workers['original']['events'] == workers['split']['events']
assert workers['original']['output_payloads'] == workers['split']['output_payloads']
record['all_event_records_identical'] = True
record['compared_event_records'] = len(workers['original']['events'])
record['validation_sha256'] = digest(HERE / 'release-validation.json')
record['validation_passed'] = True
record['old_prefix_unchanged'] = old_hashes == {p.name: digest(p) for p in OLD.iterdir() if p.is_file()}
assert record['old_prefix_unchanged']
assert digest(ROOT / 'vendor/tirust/src/lib.rs') == metadata['source_sha256']
assert digest(NEW / 'libmadloop_tirust.so') == metadata['library_sha256']
(HERE / 'release-qualification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'qualified': True, 'new_library_sha256': metadata['library_sha256'],
                  'old_prefix_unchanged': record['old_prefix_unchanged']}))
