"""Build and inspect the pinned TIRust Fortran interface.

The implementation lives in vendor/tirust. Cargo fetches its pinned numerical
dependencies; local source overrides are copied into an isolated build directory
so installing never modifies a developer's reference checkouts.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from madgraph import MadGraph5Error, MG5DIR

logger = logging.getLogger('madgraph.tirust_install')
CONFIG_NAME = 'tirust-config.json'
ABI_VERSION = 1
UPSTREAM_URL = "https://github.com/huashengshao/TIRust"
UPSTREAM_REVISION = "a5d4afbf87a0afc155dc57d2ac633af77a932a16"
RUNTIME_ARTIFACTS = ("libmadloop_tirust.so",)
QUALIFIED_ARTIFACTS = ('libtirust.a', 'mtirust.mod') + RUNTIME_ARTIFACTS


def _artifact_hashes(directory):
    """Identify the exact native artifacts exercised by the installed smoke test."""
    hashes = {}
    for name in QUALIFIED_ARTIFACTS:
        path = Path(directory) / name
        if name == 'mtirust.mod' and not path.is_file():
            path = Path(directory).parent / 'include' / name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1048576), b''):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return hashes


def read_config(libdir):
    """Read installed link and precision information, checking the bridge ABI."""
    try:
        with open(os.path.join(libdir, CONFIG_NAME)) as stream:
            config = json.load(stream)
    except (OSError, TypeError, ValueError) as error:
        raise MadGraph5Error(
            "Cannot read TIRust installation metadata in %s: %s. "
            "Run 'install tirust --update'." % (libdir, error))
    if (not isinstance(config, dict) or
            type(config.get('abi_version')) is not int or
            config.get('abi_version') != ABI_VERSION):
        raise MadGraph5Error("Unsupported TIRust bridge ABI in %s; "
                             "run 'install tirust --update'." % libdir)
    flags = config.get('native_libs')
    if (not isinstance(flags, list) or
            any(not isinstance(flag, str) or '\n' in flag or '\r' in flag
                for flag in flags)):
        raise MadGraph5Error("Invalid TIRust native link metadata in %s; "
                             "run 'install tirust --update'." % libdir)
    before = config.get('link_before')
    if (type(config.get('link_order_version')) is not int or
            config.get('link_order_version') != 1 or
            not isinstance(before, list) or not before or
            any(not isinstance(flag, str) or '\n' in flag or '\r' in flag
                for flag in before)):
        raise MadGraph5Error("Missing or invalid TIRust compiler-runtime link metadata "
                             "in %s; run 'install tirust --update'." % libdir)
    if config.get('runtime_artifacts') != list(RUNTIME_ARTIFACTS):
        raise MadGraph5Error("Missing TIRust shared-runtime metadata; run 'install tirust --update'.")
    return config


def is_available(libdir):
    """Check the installed archive, Fortran interface and native link metadata."""
    if not isinstance(libdir, (str, os.PathLike)) or not libdir:
        return False
    directory = Path(libdir)
    if not directory.is_absolute():
        directory = Path(MG5DIR) / directory
    try:
        archive = directory / 'libtirust.a'
        if archive.stat().st_size <= 8:
            return False
        with archive.open('rb') as stream:
            if stream.read(8) != b'!<arch>\n':
                return False
        config = read_config(directory)
        if not all((directory / name).is_file() and (directory / name).stat().st_size
                   for name in RUNTIME_ARTIFACTS):
            return False
        if not any(path.is_file() and path.stat().st_size for path in (
                directory / 'mtirust.mod', directory.parent / 'include' / 'mtirust.mod')):
            return False
        return (config.get('artifact_sha256') == _artifact_hashes(directory) and
                shlex.split((directory / 'tirust-native-libs.txt').read_text()) == config['native_libs'] and
                shlex.split((directory / 'tirust-link-before.txt').read_text()) == config['link_before'])
    except (OSError, ValueError, MadGraph5Error):
        return False


def export_files(libdir):
    """Return the complete installed runtime needed by an internal export."""
    from madgraph.various import rust_dependency_notices

    config = read_config(libdir)
    directory = Path(libdir)
    artifacts = ['libtirust.a', 'mtirust.mod', CONFIG_NAME,
                 'tirust-native-libs.txt', 'tirust-link-before.txt'] + list(RUNTIME_ARTIFACTS)
    # Installations may keep the compiler module alongside their public source.
    if not (directory / 'mtirust.mod').is_file():
        artifacts[1] = str((directory.parent / 'include/mtirust.mod').resolve())
    for name in ('mtirust.f90', 'tirust.h'):
        for candidate in (directory / name, directory.parent / 'include' / name):
            if candidate.is_file():
                artifacts.append(str(candidate.resolve()))
                break
    artifacts.extend(rust_dependency_notices.export_files(
        libdir, 'tirust', expected=config.get('dependency_notices')))
    return artifacts


def pre_link_flags(libdir):
    """Resolve the Fortran compiler runtime before Rust's builtin symbols."""
    return read_config(libdir)['link_before']


def native_link_flags(libdir):
    """Return shared-library dependencies for the Fortran wrapper."""
    return read_config(libdir)['native_libs']


def supports_quad_precision(libdir):
    if not is_available(libdir):
        return False
    return read_config(libdir).get('quad_precision') is True


def supports_dp_route_statistics(libdir):
    """Optional ABI-1 diagnostics; older installations remain usable."""
    return (is_available(libdir) and
            read_config(libdir).get('dp_route_statistics') is True)


def build_parser():
    parser = argparse.ArgumentParser(prog='install tirust',
        description='Install the pinned TIRust/oneloopmaster Fortran interface.')
    parser.add_argument('--source', metavar='DIRECTORY',
                        help='local TIRust checkout (copied before building)')
    parser.add_argument('--oneloop-source', metavar='DIRECTORY',
                        help='local oneloopmaster checkout (copied before building)')
    parser.add_argument('--prefix', metavar='DIRECTORY',
                        help='override the configured HEPTools installation prefix')
    parser.add_argument('--jobs', '-j', type=int, default=None)
    parser.add_argument('--update', '--force', dest='force', action='store_true',
                        help='rebuild using this MadGraph checkout\'s pinned dependencies; '
                             'replace an existing installation after validation')
    return parser


def _copy_source(source, destination, required):
    source = Path(source).expanduser().resolve()
    destination = Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise MadGraph5Error('TIRust source staging cannot be inside its source checkout.')
    for filename in required:
        if not (source / filename).is_file():
            raise MadGraph5Error("Invalid source checkout %s: missing %s" %
                                 (source, filename))
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        '.git', 'target', 'references', '__pycache__', '.direnv'))
    revision = None
    # Do not report an enclosing MadGraph checkout's HEAD for an extracted
    # dependency tarball. Git worktrees have a .git file instead of a directory.
    if (source / '.git').exists():
        try:
            revision = subprocess.check_output(
                ['git', '-C', str(source), 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    # A revision alone cannot identify a modified checkout, or a source tarball.
    # Bind precisely the files copied for this build without relying on .git.
    files = {path.relative_to(destination).as_posix():
             hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(destination.rglob('*')) if path.is_file()}
    content = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return {'source': str(source), 'revision': revision,
            'source_tree_sha256': hashlib.sha256(content).hexdigest(),
            'source_files': len(files)}


def _dependency_provenance(bridge):
    """Identify the actual locked sources, including transitive Git patches."""
    import tomllib

    manifest = (bridge / 'Cargo.toml').read_bytes()
    lock = (bridge / 'Cargo.lock').read_bytes()
    try:
        packages = tomllib.loads(lock.decode())['package']
        git_dependencies = [
            {'name': package['name'], 'version': package['version'],
             'source': package['source']}
            for package in packages
            if package.get('source', '').startswith('git+')]
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError,
            TypeError, AttributeError) as error:
        raise MadGraph5Error('Invalid resolved TIRust Cargo.lock: %s' % error) from error
    return {'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
            'lock_sha256': hashlib.sha256(lock).hexdigest(),
            'git_dependencies': sorted(git_dependencies,
                                       key=lambda item: (item['name'], item['source']))}


def _prepare_overrides(bridge, stage, options):
    """Create Cargo patches pointing only inside the isolated build directory."""
    patches = []
    provenance = {}
    if options.source:
        target = stage / 'sources' / 'tirust'
        provenance['tirust'] = _copy_source(options.source, target, [
            'Cargo.toml', 'crates/iregi-rust/Cargo.toml',
            'crates/iregi-masters/Cargo.toml'])
        patches.append('[patch."https://github.com/huashengshao/TIRust"]')
        for name in ('iregi-rust', 'iregi-masters'):
            patches.append('%s = { path = %s }' % (
                name, json.dumps(str(target / 'crates' / name))))
    if options.oneloop_source:
        target = stage / 'sources' / 'oneloopmaster'
        provenance['oneloopmaster'] = _copy_source(
            options.oneloop_source, target, ['Cargo.toml'])
        patches.extend([
            '[patch."https://github.com/alphal00p/oneloopmaster.git"]',
            'oneloop = { path = %s }' % json.dumps(str(target))])
    if patches:
        (bridge / '.cargo').mkdir(exist_ok=True)
        (bridge / '.cargo' / 'config.toml').write_text('\n'.join(patches) + '\n')
    return provenance


def _validate_pole_patch(bridge, source):
    """Bind the reviewed patch to its pristine upstream inputs before application."""
    patch = bridge / 'patches' / '0001-direct-poles-only.patch'
    try:
        manifest = json.loads((bridge / 'patches' / 'manifest.json').read_text())
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        sources = manifest['source_sha256']
        if (manifest.get('schema_version') != 1 or
                manifest.get('upstream_revision') != UPSTREAM_REVISION or
                manifest.get('patch_sha256') != digest or not isinstance(sources, dict) or not sources):
            raise ValueError('manifest or patch checksum does not match')
        for filename, expected in sources.items():
            relative = Path(filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('invalid patch source path')
            path = source / relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if actual != expected:
                raise ValueError('upstream input differs: %s' % filename)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise MadGraph5Error('Cannot validate the pinned TIRust poles-only patch: %s' % error) from error
    return patch, digest


def _run(command, cwd, log, environment):
    logger.info("TIRust: %s", shlex.join(command))
    with log.open('a') as stream:
        stream.write('\n$ %s\n' % shlex.join(command))
        stream.flush()
        try:
            subprocess.run(command, cwd=str(cwd), env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise MadGraph5Error("TIRust installation failed: %s. "
                                 "See %s" % (error, log)) from error


def _publish(installed, destination, stage, prefix, on_publish=None):
    """Publish the validated tree and owned aliases as one recoverable update."""
    from madgraph.various import rust_dependency_notices

    backup = stage / 'previous'
    common_lib = prefix / 'lib'
    alias_stage = None
    changed_links = []
    published = created_common_lib = False
    try:
        existed = common_lib.exists() or common_lib.is_symlink()
        common_lib.mkdir(exist_ok=True)
        created_common_lib = not existed
        # This directory shares the aliases' filesystem even when prefix/lib
        # is a symlink or mount. Rename must also work during rollback.
        alias_stage = Path(tempfile.mkdtemp(prefix='.tirust-links-', dir=str(common_lib)))
        saved_links = alias_stage / 'previous-links'
        new_links = alias_stage / 'links'
        saved_links.mkdir()
        new_links.mkdir()
        notices = tuple(rust_dependency_notices.export_files(installed / 'lib', 'tirust'))
        if destination.exists() or destination.is_symlink():
            os.replace(destination, backup)
        os.replace(installed, destination)
        published = True
        for name in ('libtirust.a', 'mtirust.mod', CONFIG_NAME,
                     'tirust-native-libs.txt', 'tirust-link-before.txt') + RUNTIME_ARTIFACTS + notices:
            link = common_lib / name
            # Existing regular user files and directories belong to the user.
            if not link.is_symlink() and link.exists():
                continue
            previous = None
            if link.is_symlink():
                previous = saved_links / name
                # Preserve relative and dangling targets verbatim.
                previous.symlink_to(os.readlink(link))
            temporary = new_links / name
            temporary.symlink_to(os.path.relpath(destination / 'lib' / name,
                                                common_lib.resolve()))
            # Journal before rename: cancellation can arrive just after the
            # atomic move, before the next Python instruction executes.
            changed_links.append((link, previous, temporary))
            os.replace(temporary, link)
        # Configuration saving is part of publication: preserve the old tree
        # and alias journal until the caller has accepted the new installation.
        if on_publish is not None:
            on_publish(str(destination / 'lib'))
    except BaseException as error:
        failures = []
        for link, previous, temporary in reversed(changed_links):
            if temporary.is_symlink():
                continue  # The candidate was never published.
            try:
                if previous is None:
                    link.unlink()
                else:
                    os.replace(previous, link)
            except OSError as restore_error:
                failures.append('%s: %s' % (link, restore_error))
        if published or not installed.exists():
            try:
                os.replace(destination, installed)
            except OSError as restore_error:
                failures.append('%s: %s' % (destination, restore_error))
        if backup.exists() or backup.is_symlink():
            try:
                os.replace(backup, destination)
            except OSError as restore_error:
                failures.append('%s: %s' % (backup, restore_error))
        if not failures and alias_stage is not None:
            try:
                shutil.rmtree(alias_stage)
            except OSError as restore_error:
                failures.append('%s: %s' % (alias_stage, restore_error))
        if not failures and created_common_lib:
            try:
                common_lib.rmdir()  # Never remove an existing or nonempty directory.
            except OSError as restore_error:
                failures.append('%s: %s' % (common_lib, restore_error))
        if failures:
            raise MadGraph5Error(
                "TIRust publication failed (%s); rollback could not restore %s. "
                "Recovery files and installation logs are retained in %s."
                % (error, '; '.join(failures), ', '.join(str(p) for p in
                    (stage, alias_stage) if p is not None))) from error
        raise
    return alias_stage


def save_configuration(command, config_file, libdir):
    """Publish the configured library path without truncating a working card."""
    from madgraph.various import misc

    destination = Path(config_file or misc.install_config_file(MG5DIR)).resolve()
    base = destination if destination.is_file() else Path(MG5DIR) / 'input' / misc.CONFIG_TEMPLATE_NAME
    descriptor, name = tempfile.mkstemp(prefix='.tirust_configuration_', dir=str(destination.parent))
    os.close(descriptor)
    temporary = Path(name)
    try:
        command.write_configuration(str(temporary), str(base), MG5DIR, {'tirust': libdir})
        if destination.exists():
            os.chmod(temporary, destination.stat().st_mode & 0o777)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def install(mg_root, prefix, arguments=(), fortran_compiler=None, jobs=None,
            on_publish=None):
    """Install, smoke-test and publish TIRust; return its library directory.

    An optional ``on_publish(libdir)`` saves caller configuration before the
    old installation and alias backups are discarded. Its failure rolls back
    publication. It also runs when reusing an already validated installation.
    """
    from madgraph.various import rust_dependency_notices

    try:
        options = build_parser().parse_args(list(arguments))
    except SystemExit as error:
        if error.code == 0:
            return None
        raise MadGraph5Error("Invalid 'install tirust' options.") from error
    jobs = options.jobs if options.jobs is not None else jobs
    jobs = 1 if jobs is None else jobs
    if jobs < 1:
        raise MadGraph5Error('TIRust --jobs must be a positive integer.')
    prefix = Path(options.prefix or prefix).expanduser().resolve()
    destination = prefix / 'tirust'
    for local_source in (options.source, options.oneloop_source):
        if local_source:
            local_source = Path(local_source).expanduser().resolve()
            if prefix == local_source or local_source in prefix.parents:
                raise MadGraph5Error('TIRust installation prefix cannot be inside a source checkout.')
            if destination == local_source or destination in local_source.parents:
                raise MadGraph5Error('TIRust publication would replace a source checkout; use a separate prefix.')
    if destination.exists() and not options.force:
        read_config(destination / 'lib')
        if not is_available(destination / 'lib'):
            raise MadGraph5Error("Incomplete TIRust installation: "
                                 "run 'install tirust --update'.")
        logger.info("TIRust is already installed in %s; use --update to rebuild "
                    "with this checkout's pinned dependencies.",
                    destination)
        if on_publish is not None:
            on_publish(str(destination / 'lib'))
        return str(destination / 'lib')

    compiler = fortran_compiler or 'gfortran'
    for program in ('cargo', 'make', 'git', 'patch', compiler):
        if not shutil.which(program):
            raise MadGraph5Error("Installing TIRust requires %s on PATH. "
                "Install Rust >= 1.96 and a Fortran compiler with REAL(16) support."
                % program)

    prefix.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.tirust-build-', dir=str(prefix)))
    bridge = stage / 'bridge'
    installed = stage / 'install'
    log = stage / 'install.log'
    shutil.copytree(Path(mg_root) / 'vendor' / 'tirust', bridge,
                   ignore=shutil.ignore_patterns(
                       'target', '.git', '*.o', '*.mod', '*.a', '*.so', '*.dylib', 'build.log',
                       'tirust-native-libs.txt', 'tirust-link-before.txt', 'smoke'))
    environment = os.environ.copy()
    environment['CARGO_BUILD_JOBS'] = str(jobs)
    # Honor an explicitly selected Cargo cache. By default keep it under the
    # installation prefix so retries can reuse compiled pinned dependencies.
    environment['CARGO_TARGET_DIR'] = str(Path(
        environment.get('CARGO_TARGET_DIR', prefix / '.tirust-build-cache')
    ).expanduser().resolve())
    # A global Cargo build-dir may otherwise hash the temporary workspace path,
    # defeating reuse even when CARGO_TARGET_DIR is stable.
    environment.setdefault('CARGO_BUILD_BUILD_DIR', str(
        Path(environment['CARGO_TARGET_DIR']) / 'build'))
    fetched_source = not options.source
    if fetched_source:
        upstream = stage / 'upstream'
        upstream.mkdir()
        _run(['git', 'init', '--quiet'], upstream, log, environment)
        _run(['git', 'fetch', '--depth=1', UPSTREAM_URL, UPSTREAM_REVISION],
             upstream, log, environment)
        _run(['git', 'checkout', '--detach', 'FETCH_HEAD'], upstream, log, environment)
        options.source = str(upstream)
    provenance = _prepare_overrides(bridge, stage, options)
    if fetched_source:
        provenance['tirust']['source'] = UPSTREAM_URL
    source = stage / 'sources' / 'tirust'
    patch = bridge / 'patches' / '0001-direct-poles-only.patch'
    kernels = bridge / 'pole-kernels'
    if not patch.is_file() or not (kernels / 'Cargo.toml').is_file():
        raise MadGraph5Error('TIRust installation requires the bundled poles-only patch and kernels.')
    patch, patch_digest = _validate_pole_patch(bridge, source)
    provenance['pole_kernels'] = _copy_source(
        kernels, source / 'madloop-pole-kernels', ['Cargo.toml', 'src/lib.rs'] +
        ['src/generated_%s.rs' % family for family in ('a0', 'b0', 'c0', 'd0', 'constants')] +
        ['src/generated_manifest.txt'])
    _run(['patch', '-p1', '--batch', '--forward', '--fuzz=0', '-i', str(patch)],
         source, log, environment)
    provenance['tirust']['patch_sha256'] = patch_digest
    if provenance:
        # Resolve changed path sources while retaining other locked versions.
        _run(['cargo', 'metadata', '--format-version=1'], bridge, log, environment)
    _run(['make', 'install', 'PREFIX=' + str(installed), 'FC=' + compiler,
          'CARGO_FEATURES=poles-only'],
         bridge, log, environment)

    libdir = installed / 'lib'
    notices = rust_dependency_notices.package(bridge, libdir, 'tirust', environment)
    try:
        native_libs = shlex.split((libdir / 'tirust-native-libs.txt').read_text())
        link_before = shlex.split((libdir / 'tirust-link-before.txt').read_text())
    except (OSError, ValueError) as error:
        raise MadGraph5Error("TIRust build did not produce native link metadata. "
                             "See %s" % log) from error
    config = {'abi_version': ABI_VERSION, 'quad_precision': True,
              'dp_route_statistics': True,
              'native_libs': native_libs, 'local_sources': provenance,
              'fortran_compiler': compiler, 'link_before': link_before,
              'link_order_version': 1,
              'compiler_runtime': 'gnu-linux-shared-rust',
              'build_provenance': _dependency_provenance(bridge),
              'runtime_artifacts': list(RUNTIME_ARTIFACTS),
              'finite_terms': True, 'poles_only': True,
              'dependency_notices': notices,
              'artifact_sha256': _artifact_hashes(libdir)}
    (libdir / CONFIG_NAME).write_text(json.dumps(config, indent=2) + '\n')
    # This exercises the installed Fortran module and both numerical domains.
    smoke = stage / 'smoke'
    _run([compiler, '-I' + str(libdir), str(bridge / 'smoke.f90')] +
         link_before + [str(libdir / 'libtirust.a'), '-L' + str(libdir),
                        '-Wl,-rpath,' + str(libdir)] + native_libs + ['-o', str(smoke)],
         bridge, log, environment)
    _run([str(smoke), '--require-poles'], bridge, log, environment)
    shutil.copy2(log, installed / 'install.log')
    shutil.copy2(bridge / 'Cargo.lock', installed / 'Cargo.lock')
    shutil.copy2(bridge / 'Cargo.toml', installed / 'Cargo.toml')

    alias_stage = _publish(installed, destination, stage, prefix, on_publish=on_publish)
    # Publication has committed. A cleanup error must not turn a working
    # installation into a reported failure or attempt to restore a backup
    # that cleanup may already have partly removed.
    for temporary in (alias_stage, stage):
        try:
            shutil.rmtree(temporary)
        except OSError as error:
            logger.warning("TIRust is installed successfully, but staging cleanup "
                           "failed in %s: %s", temporary, error)
    logger.info("TIRust successfully installed in %s.", destination)
    return str(destination / 'lib')
