"""Package the observed clean install only after publication and smoke succeed."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path.cwd()
work = root / '.tirust-build/clean-install'
mg = work / 'mg'
installed = mg / 'HEPTools/tirust'
lib = installed / 'lib'
sys.path.insert(0, str(mg))
from madgraph.various import tirust_install


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


config = tirust_install.read_config(lib)
require(tirust_install.is_available(lib), 'Published library fails availability validation')
require(config['finite_terms'] and config['poles_only'] and config['quad_precision'], 'Missing required capabilities')
cli = (work / 'cli-attempt-2-network.log').read_text()
require('TIRust successfully installed' in cli, 'Missing CLI publication success')
observed = [json.loads(line) for line in (work / 'execution-observer.jsonl').read_text().splitlines()]
require(any(item['event'] == 'serialized-smoke' for item in observed), 'Smoke serialization was not observed')
smoke_log = (installed / 'install.log').read_text()
require('TIRust poles-only evaluation passed' in smoke_log, 'Missing successful poles smoke')
require('TIRust ABI1: DP, binary128 input, UV/IR poles and cache checks passed' in smoke_log, 'Missing successful full ABI smoke')
require(str(lib) in (mg / 'input/mg7_configuration.txt').read_text(), 'Copied configuration did not save the library')
before = json.loads((work / 'production-before.json').read_text())
after = {name: sha(root / name) for name in before}
require(before == after, 'Production installation or configuration changed')
sources = json.loads((work / 'source-inventory.json').read_text())
require(all(sha(mg / name) == value for name, value in sources.items()), 'Copied source snapshot changed')
initial = json.loads((work / 'clean-install-start.json').read_text())
empty_caches = {name: initial['initial_cache_state'][name] == 0 for name in ('cargo-home', 'cargo-target', 'cargo-build')}
require(all(empty_caches.values()), 'Initial cache record was not empty')
compiled_module = (work / 'compiled-module.sha256').read_text().split()[0]
require(sha(lib / 'mtirust.mod') == compiled_module, 'Installed module differs from freshly compiled smoke-visible module')
write(work / 'production-after.json', after)

report = root / 'docs/benchmarkfolder/qualification/tirust-install/clean-download'
report.mkdir(exist_ok=False)
names = ['cli-attempt-1.log', 'cli-attempt-2-network.log', 'install-attempt-1-failed-dns.log',
         'run.sh', 'execution-observer.jsonl', 'install.mg5', 'source-inventory.json',
         'clean-install-start.json', 'host-toolchain.txt', 'resolved-source-hashes.txt',
         'fetched-upstream-revision.txt', 'production-before.json', 'production-after.json',
         'allow-compilation', 'compiled-module.sha256', 'source-revision-comparison.json']
for name in names:
    shutil.copy2(work / name, report / name)
source_revision = json.loads((work / 'source-revision-comparison.json').read_text())
for name in source_revision['snapshot_files_different_from_base']:
    target = report / 'source-snapshot-amendments' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mg / name, target)
shutil.copy2(work / 'observer/sitecustomize.py', report / 'execution-observer.py')
shutil.copy2(Path(__file__), report / 'package_evidence.py')
for name in ['install.log', 'Cargo.toml', 'Cargo.lock']:
    shutil.copy2(installed / name, report / name)
shutil.copy2(lib / tirust_install.CONFIG_NAME, report / tirust_install.CONFIG_NAME)
shutil.copy2(mg / 'input/mg7_configuration.txt', report / 'isolated-configuration-after.txt')
for path in (work / 'cargo-build/release/build').glob('gmp-mpfr-sys-*/output'):
    shutil.copy2(path, report / 'gmp-mpfr-source-build.log')
commands = []
for command, filename in [(['nm', '-D', '--defined-only', str(lib / 'libmadloop_tirust.so')], 'native-exports.txt'),
                          (['readelf', '-d', str(lib / 'libmadloop_tirust.so')], 'dynamic-section.txt')]:
    output = subprocess.check_output(command, text=True)
    (report / filename).write_text(output)
    commands.append({'argv': command, 'returncode': 0, 'output': filename})
exports = (report / 'native-exports.txt').read_text().splitlines()
require(len(exports) == 12 and all(line.split()[-1].startswith('ml_tirust_') for line in exports), 'Unexpected native export surface')
checks = {'status': 'passed', 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'actual_cli_source_overrides': False, 'available': True, 'production_artifacts_and_config_unchanged': True,
          'isolated_configuration_saved': True, 'serialized_real_native_smoke': True,
          'installed_module_identical_to_smoke_visible_build_module': True,
          'empty_initial_caches': empty_caches, 'copied_source_snapshot_rechecked': True,
          'native_artifacts': {name: {'sha256': sha(lib / name), 'bytes': (lib / name).stat().st_size}
                               for name in tirust_install.QUALIFIED_ARTIFACTS},
          'native_export_count': len(exports), 'commands': commands,
          'source_scope': 'Byte-copied pre-split-workspace implementation; source-inventory.json identifies the snapshot. Later bridge edits are not qualified by this run.',
          'reproduction_base_revision': source_revision['reproduction_base_revision'],
          'source_patch_sha256': config['local_sources']['tirust']['patch_sha256'],
          'build_provenance': config['build_provenance']}
write(report / 'check-installation.json', checks)
(report / 'README.md').write_text('''# Clean network installation qualification

The actual copied `bin/madgraph` CLI completed `install tirust --jobs=4` with no local source overrides. Its task-specific Cargo home, target, build, XDG cache and configuration directories started empty. The public pinned TIRust checkout, Symbolica and oneloopmaster dependencies were downloaded, the reviewed final pole patch was applied, dependencies were built from source, and the installed DP/QP/poles smoke passed before publication and configuration saving.

This run uses the byte-copied implementation identified by `source-inventory.json`, captured before the later split-workspace bridge optimization. It qualifies clean installation of that snapshot; it does not qualify later bridge edits or replace the physical-amplitude acceptance campaign. The five recorded production native-artifact and configuration hashes are identical before and after. No native binaries are included here.

`source-revision-comparison.json` compares all 3,227 copied file contents with Git revision `25e513487e3cf152c45070a4fff9fbd5cdc85446`. The isolated copy materializes 43 Git symlinks. Its only different file is `.gitattributes`, retained under `source-snapshot-amendments/`; all copied implementation files match that revision. This supplies a reproducible source base without claiming the original workspace was clean.

The first attempt failed because sandbox DNS could not resolve GitHub. The approved network retry completed successfully. Both CLI logs and the failed installation log are retained; the CLI's zero shell exit on the first attempt is not classified as success. That failure also wrote the ordinary ignored `MG5_debug` diagnostic in the mandated original working directory.

`execution-observer.py` leaves installer source bytes and the underlying command arguments unchanged. It delays `make install` until the separate timing window is released, then prefixes only the real final smoke command with `flock` for the shared numerical-test lock. `execution-observer.jsonl` records those actions. The recorded coordination wait is unrelated to build performance; no installation speed claim is made.

`run.sh` and `install.mg5` retain the exact invocation. They ran from the main MadGraph working directory against the copied CLI, with private configuration and prefix. The command uses the pinned public downloads, not the sibling reference repositories. `host-toolchain.txt` records actual executable versions, `Cargo.lock` identifies resolved dependencies, and `gmp-mpfr-source-build.log` records the native dependency source build. Historical absolute paths describe the original task workspace.
''')
write(report / 'index.json', {'schema_version': 1, 'status': 'passed', 'scope': checks['source_scope'],
    'artifacts': {p.relative_to(report).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
                  for p in sorted(report.rglob('*')) if p.is_file()}})
outer = report.parent / 'index.json'
index = json.loads(outer.read_text())
readme = report.parent / 'README.md'
readme.write_text(readme.read_text() + '\nThe subsequent genuine no-override network installation is recorded in `clean-download/index.json`. It starts with empty task-specific Cargo and native dependency caches, builds downloaded pinned sources, and passes the actual serialized DP/QP/poles smoke and isolated configuration publication. Its copied source snapshot predates the later split-workspace bridge optimization; those later changes require their separate qualification. The original production artifacts remain unchanged.\n')
index['artifacts']['README.md'] = {'sha256': sha(readme), 'size': readme.stat().st_size}
for p in sorted(report.rglob('*')):
    if p.is_file():
        index['artifacts'][p.relative_to(report.parent).as_posix()] = {'sha256': sha(p), 'size': p.stat().st_size}
index.setdefault('additional_qualifications', []).append({'record': 'clean-download/index.json', 'status': 'passed',
    'source_scope': checks['source_scope'], 'production_artifacts_unchanged': True})
write(outer, index)
print(json.dumps(checks, indent=2))
