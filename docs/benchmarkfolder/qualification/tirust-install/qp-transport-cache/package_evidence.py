"""Bind a genuine isolated CLI publication to immutable-source evidence."""
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path.cwd()
WORK = ROOT / '.tirust-build/qp-transport-install'
MG = WORK / 'mg'
PREFIX = MG / 'HEPTools/tirust'
LIB = PREFIX / 'lib'
REPORT = ROOT / 'docs/benchmarkfolder/qualification/tirust-install/qp-transport-cache'
sys.path.insert(0, str(MG))
from madgraph.various import tirust_install


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def require(value, message):
    if not value:
        raise RuntimeError(message)


execution = json.loads((WORK / 'execution.json').read_text())
require(execution['state'] == 'completed' and execution['returncode'] == 0, 'CLI did not finish')
require(execution['observer_sha256'] == sha(WORK / 'observer/sitecustomize.py'), 'Observer changed')
require(execution['run_script_sha256'] == sha(WORK / 'run.sh'), 'Launcher command changed')
require(execution['launcher_sha256'] == sha(WORK / 'launch.py'), 'Launcher changed')
config = tirust_install.read_config(LIB)
require(tirust_install.is_available(LIB), 'Published package fails availability checks')
require(config['quad_precision'] and config['finite_terms'] and config['poles_only'], 'Missing capability')
require(str(LIB) in (MG / 'input/mg7_configuration.txt').read_text(), 'Private config not saved')
require('TIRust successfully installed' in (WORK / 'cli.log').read_text(), 'CLI failed publication')
install_log = (PREFIX / 'install.log').read_text()
for marker in ['TIRust poles-only evaluation passed',
               'TIRust ABI1: DP, binary128 input, UV/IR poles and cache checks passed']:
    require(marker in install_log, 'Missing actual native smoke marker')
observer = [json.loads(x) for x in (WORK / 'execution-observer.jsonl').read_text().splitlines()]
require(any(x['event'] == 'serialized-smoke' for x in observer), 'Smoke was not serialized')
module_records = [x for x in observer if x['event'] == 'smoke-module-identity']
require(len(module_records) == 1 and set(module_records[0]['sha256'].values()) == {sha(LIB / 'mtirust.mod')},
        'Smoke and installed module identities differ')
require(all(x['returncode'] == 0 for x in observer if x['event'] == 'completed'), 'Installer subprocess failed')
source = json.loads((WORK / 'source-inventory.json').read_text())
revision = json.loads((WORK / 'source-revision.json').read_text())
require(revision['revision'] == '059bb9c979194bfda783cad83443223def8df280', 'Wrong committed snapshot')
require(all(sha(MG / name) == expected for name, expected in source.items()), 'Snapshot changed')
require(sha(PREFIX / 'include/mtirust.f90') == source['vendor/tirust/mtirust.f90'], 'Installed wrapper source differs')
require(sha(PREFIX / 'Cargo.lock') == config['build_provenance']['lock_sha256'], 'Lock metadata mismatch')
require(sha(PREFIX / 'Cargo.lock') == 'c4f59f296c7d2838a63b62bd4d47aa0e6e5eaaefd22b2d28d30e94d9f26179e4', 'Resolved locked dependencies changed')
old_config = json.loads((ROOT / '.tirust-build/current-install/mg/HEPTools/tirust/lib/tirust-config.json').read_text())
require(config['build_provenance']['git_dependencies'] == old_config['build_provenance']['git_dependencies'], 'Pinned git dependencies changed')
require(config['local_sources']['tirust'] == old_config['local_sources']['tirust'], 'Pinned TIRust source/patch provenance changed')
require({k: v for k, v in config['local_sources']['pole_kernels'].items() if k != 'source'} == {k: v for k, v in old_config['local_sources']['pole_kernels'].items() if k != 'source'}, 'Bundled pole-kernel bytes changed')
require(sha(PREFIX / 'Cargo.toml') == config['build_provenance']['manifest_sha256'], 'Manifest metadata mismatch')
require(config['local_sources']['tirust']['patch_sha256'] ==
        '80ef17b367511419e734f760edca4593f15a324996c4830b815c03cf884cbab5', 'Pole patch changed')
before = json.loads((WORK / 'protected-before.json').read_text())
after = {}
for label in ('main49af', 'previous095e'):
    base = Path(before[label]['prefix'])
    files = {str(p.relative_to(base)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(base.rglob('*')) if p.is_file()}
    after[label] = {'prefix': str(base), 'files': files}
    require(files == before[label]['files'], 'Immutable installation changed: ' + label)
for label in ('main', 'previous095e', 'isolated'):
    entry = before[label + '-configuration']
    path = Path(entry['path'])
    shutil.copy2(path, WORK / (label + '-configuration-after.txt'))
    after[label + '-configuration'] = {'path': str(path), 'sha256': sha(path),
        'mtime_ns': path.stat().st_mtime_ns, 'unchanged': sha(path) == entry['sha256']}
    (WORK / (label + '-configuration.diff')).write_text(''.join(difflib.unified_diff(
        (WORK / entry['snapshot']).read_text().splitlines(keepends=True),
        path.read_text().splitlines(keepends=True), fromfile=label + '-before', tofile=label + '-after')))
write(WORK / 'protected-after.json', after)
require(after['main-configuration']['unchanged'] and after['previous095e-configuration']['unchanged'], 'Protected configuration changed during installation')
aliases = {}
for path in sorted((MG / 'HEPTools/lib').glob('*tirust*')):
    aliases[path.name] = {'path': str(path), 'is_symlink': path.is_symlink(),
                          'resolved': str(path.resolve()), 'sha256': sha(path)}
require(aliases and all(Path(item['resolved']).is_relative_to(PREFIX) for item in aliases.values()),
        'Published aliases do not resolve inside isolated prefix')
write(WORK / 'published-aliases.json', aliases)
exports = tirust_install.export_files(LIB)
require(all(name in exports for name in config['dependency_notices']['files']), 'Notices not exportable')
commands = []
for argv, filename in [(['nm', '-D', '--defined-only', str(LIB / 'libmadloop_tirust.so')], 'native-exports.txt'),
                       (['readelf', '-d', str(LIB / 'libmadloop_tirust.so')], 'dynamic-section.txt')]:
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (REPORT / filename).write_text(result.stdout)
    commands.append({'argv': argv, 'returncode': result.returncode, 'log': filename})
    require(result.returncode == 0, 'Native metadata command failed')
native_exports = (REPORT / 'native-exports.txt').read_text().splitlines()
require(len(native_exports) == 12 and all(x.split()[-1].startswith('ml_tirust_') for x in native_exports),
        'Unexpected native export surface')
require('(SONAME)' in (REPORT / 'dynamic-section.txt').read_text() and
        '[libmadloop_tirust.so]' in (REPORT / 'dynamic-section.txt').read_text(), 'Unexpected SONAME')
for path in sorted(WORK.iterdir()):
    if path.is_file() and path.suffix in ('.py', '.json', '.jsonl', '.txt', '.log', '.sh', '.mg5', '.diff'):
        shutil.copy2(path, REPORT / path.name)
for name in ('allow-compilation',):
    shutil.copy2(WORK / name, REPORT / name)
shutil.copy2(WORK / 'observer/sitecustomize.py', REPORT / 'execution-observer.py')
for name in ('install.log', 'Cargo.toml', 'Cargo.lock'):
    shutil.copy2(PREFIX / name, REPORT / name)
for name in (tirust_install.CONFIG_NAME, 'tirust-NOTICES.txt', 'tirust-dependency-provenance.json'):
    shutil.copy2(LIB / name, REPORT / name)
checks = {'schema_version': 1, 'state': 'published-qualification-pending',
    'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_revision': revision['revision'], 'source_file_count': len(source),
    'prefix': str(PREFIX), 'source_overrides': False, 'cache_policy': 'Common Cargo caches reused; not a clean-cache claim',
    'available': True, 'quad_precision': True, 'finite_terms': True, 'poles_only': True,
    'serialized_installer_smoke_passed': True, 'smoke_module_identity_verified': True,
    'transactional_private_configuration_saved': True, 'immutable_installation_hashes_unchanged': True,
    'main_configuration_unchanged': after['main-configuration']['unchanged'],
    'old095e_configuration_unchanged': after['previous095e-configuration']['unchanged'],
    'native_artifacts': {name: {'sha256': sha(LIB / name), 'bytes': (LIB / name).stat().st_size}
                         for name in tirust_install.QUALIFIED_ARTIFACTS},
    'metadata_sha256': sha(LIB / tirust_install.CONFIG_NAME),
    'notice_files': {name: {'sha256': sha(LIB / name), 'bytes': (LIB / name).stat().st_size}
                     for name in config['dependency_notices']['files']},
    'bridge_source_sha256': source['vendor/tirust/src/lib.rs'],
    'fortran_source_sha256': source['vendor/tirust/mtirust.f90'],
    'patch_sha256': config['local_sources']['tirust']['patch_sha256'],
    'build_provenance': config['build_provenance'], 'commands': commands,
    'scope': 'Actual CLI publication and installed ABI correctness; no amplitude or performance acceptance claim.'}
write(REPORT / 'check-installation.json', checks)
print(json.dumps(checks, indent=2))
