"""Build and inspect the pinned NinjaRS Fortran interface.

The implementation lives in vendor/ninjars. Cargo fetches its pinned numerical
dependencies; local source overrides are copied into an isolated build directory
so installing never modifies a developer's reference checkouts.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from madgraph import MadGraph5Error

logger = logging.getLogger('madgraph.ninjars_install')
CONFIG_NAME = 'ninjars-config.json'
ABI_VERSION = 2


def read_config(libdir):
    """Read installed link and precision information, checking the bridge ABI."""
    try:
        with open(os.path.join(libdir, CONFIG_NAME)) as stream:
            config = json.load(stream)
    except (OSError, TypeError, ValueError) as error:
        raise MadGraph5Error(
            "Cannot read NinjaRS installation metadata in %s: %s. "
            "Run 'install ninjars'." % (libdir, error))
    if not isinstance(config, dict) or config.get('abi_version') != ABI_VERSION:
        raise MadGraph5Error("Unsupported NinjaRS bridge ABI in %s; "
                             "run 'install ninjars --force'." % libdir)
    flags = config.get('native_libs')
    if (not isinstance(flags, list) or
            any(not isinstance(flag, str) or '\n' in flag or '\r' in flag
                for flag in flags)):
        raise MadGraph5Error("Invalid NinjaRS native link metadata in %s" % libdir)
    before = config.get('link_before')
    if (config.get('link_order_version') != 1 or
            not isinstance(before, list) or not before or
            any(not isinstance(flag, str) or '\n' in flag or '\r' in flag
                for flag in before)):
        raise MadGraph5Error("Missing or invalid NinjaRS compiler-runtime link metadata "
                             "in %s; run 'install ninjars --force'." % libdir)
    return config


def pre_link_flags(libdir):
    """Resolve the Fortran compiler runtime before Rust's builtin symbols."""
    return read_config(libdir)['link_before']


def native_link_flags(libdir):
    """Return Cargo's platform-specific static-library dependencies."""
    return read_config(libdir)['native_libs']


def supports_quad_precision(libdir):
    if not libdir or not os.path.isfile(os.path.join(libdir, 'libninjars.a')):
        return False
    return read_config(libdir).get('quad_precision') is True


def build_parser():
    parser = argparse.ArgumentParser(prog='install ninjars',
        description='Install the pinned NinjaRS/oneloopmaster Fortran interface.')
    parser.add_argument('--source', metavar='DIRECTORY',
                        help='local NinjaRS checkout (copied before building)')
    parser.add_argument('--oneloop-source', metavar='DIRECTORY',
                        help='local oneloopmaster checkout (copied before building)')
    parser.add_argument('--prefix', metavar='DIRECTORY',
                        help='override the configured HEPTools installation prefix')
    parser.add_argument('--jobs', '-j', type=int, default=None)
    parser.add_argument('--force', action='store_true',
                        help='replace an existing NinjaRS installation after validation')
    return parser


def _copy_source(source, destination, required):
    source = Path(source).expanduser().resolve()
    for filename in required:
        if not (source / filename).is_file():
            raise MadGraph5Error("Invalid source checkout %s: missing %s" %
                                 (source, filename))
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        '.git', 'target', 'references', '__pycache__', '.direnv'))
    revision = None
    try:
        revision = subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {'source': str(source), 'revision': revision}


def _prepare_overrides(bridge, stage, options):
    """Create Cargo patches pointing only inside the isolated build directory."""
    patches = []
    provenance = {}
    if options.source:
        target = stage / 'sources' / 'ninjars'
        provenance['ninjars'] = _copy_source(options.source, target, [
            'Cargo.toml', 'crates/ninjars/Cargo.toml',
            'crates/ninjars-symbolica/Cargo.toml'])
        patches.append('[patch."https://github.com/alphal00p/ninjars"]')
        for name in ('ninjars', 'ninjars-symbolica'):
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


def _run(command, cwd, log, environment):
    logger.info("NinjaRS: %s", shlex.join(command))
    with log.open('a') as stream:
        stream.write('\n$ %s\n' % shlex.join(command))
        stream.flush()
        try:
            subprocess.run(command, cwd=str(cwd), env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise MadGraph5Error("NinjaRS installation failed: %s. "
                                 "See %s" % (error, log)) from error


def install(mg_root, prefix, arguments=(), fortran_compiler=None, jobs=None):
    """Install, smoke-test and publish NinjaRS; return its library directory."""
    try:
        options = build_parser().parse_args(list(arguments))
    except SystemExit as error:
        if error.code == 0:
            return None
        raise MadGraph5Error("Invalid 'install ninjars' options.") from error
    jobs = options.jobs if options.jobs is not None else jobs
    jobs = 1 if jobs is None else jobs
    if jobs < 1:
        raise MadGraph5Error('NinjaRS --jobs must be a positive integer.')
    prefix = Path(options.prefix or prefix).expanduser().resolve()
    destination = prefix / 'ninjars'
    if destination.exists() and not options.force:
        read_config(destination / 'lib')
        if not all((destination / 'lib' / name).is_file() for name in
                   ('libninjars.a', 'mninjars.mod')):
            raise MadGraph5Error("Incomplete NinjaRS installation: "
                                 "run 'install ninjars --force'.")
        logger.info("NinjaRS is already installed in %s; use --force to rebuild.",
                    destination)
        return str(destination / 'lib')

    compiler = fortran_compiler or 'gfortran'
    for program in ('cargo', 'make', compiler):
        if not shutil.which(program):
            raise MadGraph5Error("Installing NinjaRS requires %s on PATH. "
                "Install Rust >= 1.96 and a Fortran compiler with REAL(16) support."
                % program)

    prefix.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.ninjars-build-', dir=str(prefix)))
    bridge = stage / 'bridge'
    installed = stage / 'install'
    log = stage / 'install.log'
    shutil.copytree(Path(mg_root) / 'vendor' / 'ninjars', bridge,
                   ignore=shutil.ignore_patterns(
                       'target', '.git', '*.o', '*.mod', '*.a', 'build.log',
                       'ninjars-native-libs.txt', 'ninjars-link-before.txt', 'smoke'))
    provenance = _prepare_overrides(bridge, stage, options)
    environment = os.environ.copy()
    environment['CARGO_BUILD_JOBS'] = str(jobs)
    # Honor an explicitly selected Cargo cache. By default keep it under the
    # installation prefix so retries can reuse compiled pinned dependencies.
    environment['CARGO_TARGET_DIR'] = str(Path(
        environment.get('CARGO_TARGET_DIR', prefix / '.ninjars-build-cache')
    ).expanduser().resolve())
    # A global Cargo build-dir may otherwise hash the temporary workspace path,
    # defeating reuse even when CARGO_TARGET_DIR is stable.
    environment.setdefault('CARGO_BUILD_BUILD_DIR', str(
        Path(environment['CARGO_TARGET_DIR']) / 'build'))
    if provenance:
        # Resolve changed path sources while retaining other locked versions.
        _run(['cargo', 'metadata', '--format-version=1'], bridge, log, environment)
    _run(['make', 'install', 'PREFIX=' + str(installed), 'FC=' + compiler],
         bridge, log, environment)

    libdir = installed / 'lib'
    try:
        native_libs = shlex.split((libdir / 'ninjars-native-libs.txt').read_text())
        link_before = shlex.split((libdir / 'ninjars-link-before.txt').read_text())
    except (OSError, ValueError) as error:
        raise MadGraph5Error("NinjaRS build did not produce native link metadata. "
                             "See %s" % log) from error
    config = {'abi_version': ABI_VERSION, 'quad_precision': True,
              'native_libs': native_libs, 'local_sources': provenance,
              'fortran_compiler': compiler, 'link_before': link_before,
              'link_order_version': 1,
              'compiler_runtime': 'gnu-linux-libgcc_s-before-rust'}
    (libdir / CONFIG_NAME).write_text(json.dumps(config, indent=2) + '\n')
    # This exercises the installed Fortran module and both numerical domains.
    smoke = stage / 'smoke'
    _run([compiler, '-I' + str(libdir), str(bridge / 'smoke.f90')] +
         link_before + [str(libdir / 'libninjars.a')] + native_libs + ['-o', str(smoke)],
         bridge, log, environment)
    _run([str(smoke)], bridge, log, environment)
    shutil.copy2(log, installed / 'install.log')
    shutil.copy2(bridge / 'Cargo.lock', installed / 'Cargo.lock')
    shutil.copy2(bridge / 'Cargo.toml', installed / 'Cargo.toml')

    # Preserve an earlier working installation until the new build has passed.
    backup = stage / 'previous'
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(installed, destination)
    except OSError:
        if backup.exists():
            os.replace(backup, destination)
        raise
    common_lib = prefix / 'lib'
    common_lib.mkdir(exist_ok=True)
    for name in ('libninjars.a', 'mninjars.mod', CONFIG_NAME,
                 'ninjars-native-libs.txt', 'ninjars-link-before.txt'):
        link = common_lib / name
        if link.is_symlink() or not link.exists():
            temporary = common_lib / ('.' + name + '.ninjars')
            if temporary.is_symlink():
                temporary.unlink()
            temporary.symlink_to(Path('..') / 'ninjars' / 'lib' / name)
            os.replace(temporary, link)
    shutil.rmtree(stage)
    logger.info("NinjaRS successfully installed in %s.", destination)
    return str(destination / 'lib')
