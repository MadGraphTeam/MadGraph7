#!/usr/bin/env python3
"""Staged, bounded lifecycle qualification. No stage runs implicitly."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
REPORT = ROOT / 'docs/benchmarkfolder/experiments/subprocess-cache-lifecycle'
WORK = ROOT / '.tirust-build/subprocess-cache-lifecycle'
MG = WORK / 'mg'
OUTPUT = WORK / 'export'
INSTALL = ROOT / '.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib'
REVISION = '059bb9c979194bfda783cad83443223def8df280'
EXPECTED = {
    'libmadloop_tirust.so': 'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9',
    'libtirust.a': '0c86349a4fd0dadd918476731e899529ce1c7b17f5892a4af2c22dd762250c1a',
    'tirust-config.json': '25182cfb31e68f8ff6246717e6819f0ec86232de88742f6e47e73676e6a41af9',
}
PROTOCOL = ['DESIGN.md', 'generate.mg5.in', 'driver.f.in', 'observer.c', 'analyze.py',
            'qualify.py', 'test_analyze.py']
WRAPS = ['context_create', 'context_destroy', 'clear_cache', 'register_plan', 'evaluate_dp', 'evaluate_mp']
CPU = 10
RUNS = [('plain-cache0', 'plain', 0), ('plain-cache-all', 'plain', -1),
        ('observed-cache0', 'observed', 0), ('observed-cache-all', 'observed', -1)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def inventory(directory, *, generated=False):
    records = {}
    for path in sorted(directory.rglob('*')):
        if not (path.is_file() or path.is_symlink()):
            continue
        if generated and (path.suffix in ('.o', '.mod', '.pyc', '.a', '.so') or
                          '__pycache__' in path.parts or path.name.startswith('lifecycle_') or
                          path.name.startswith('observer_')):
            continue
        name = str(path.relative_to(directory))
        item = {'kind': 'link' if path.is_symlink() else 'file'}
        if path.is_symlink():
            item['target'] = os.readlink(path)
        if path.is_file():
            item.update(size=path.stat().st_size, sha256=digest(path))
        records[name] = item
    return records


def verify_paths(records, directory):
    for name, item in records.items():
        path = directory / name
        if item['kind'] == 'link' and (not path.is_symlink() or os.readlink(path) != item['target']):
            raise RuntimeError(f'Symlink changed: {path}')
        if 'sha256' in item and (not path.is_file() or digest(path) != item['sha256']):
            raise RuntimeError(f'Bound file changed: {path}')


def installation():
    for name, expected in EXPECTED.items():
        if digest(INSTALL / name) != expected:
            raise RuntimeError(f'Unqualified native artifact: {name}')
    config = json.loads((INSTALL/'tirust-config.json').read_text())
    for name, expected in config['artifact_sha256'].items():
        if digest(INSTALL/name) != expected:
            raise RuntimeError(f'Artifact differs from qualified configuration: {name}')
    # Explicitly bind the live compiler module and public source/header too.
    records = {}
    for path in sorted(INSTALL.iterdir()):
        if path.is_file():
            records[str(path)] = {'sha256': digest(path), 'size': path.stat().st_size}
    for name in ('mtirust.mod', 'mtirust.f90', 'tirust.h'):
        choices = [INSTALL / name, INSTALL.parent / 'include' / name]
        path = next((p for p in choices if p.is_file()), None)
        if path is None:
            raise RuntimeError(f'Missing qualified interface {name}')
        records[str(path)] = {'sha256': digest(path), 'size': path.stat().st_size}
    return records


def env():
    result = os.environ.copy()
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT'):
        result.pop(key, None)
    result.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1',
                  XDG_CONFIG_HOME=str(WORK / 'xdg/config'),
                  XDG_CACHE_HOME=str(WORK / 'xdg/cache'),
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MAKEFLAGS='-j1', MPLCONFIGDIR=str(WORK / 'matplotlib'))
    return result


def terminate_owned_group(process):
    """Terminate only the new session/process group started by this runner."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def capture(args, cwd=ROOT, timeout=30):
    """Bounded read-only provenance query with the same owned-group cleanup."""
    process = subprocess.Popen(list(map(str,args)), cwd=cwd, env=env(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as error:
        cleanup_error = None
        try:
            terminate_owned_group(process)
        except BaseException as cleanup:
            cleanup_error = repr(cleanup)
        try:
            save(REPORT/'execution'/f'query-failure-{time.time_ns()}.json', {
                'argv':list(map(str,args)), 'cwd':str(cwd), 'deadline_seconds':timeout,
                'exception':repr(error), 'returncode':process.returncode,
                'cleanup_exception':cleanup_error})
        except BaseException:
            pass  # Preserve the original interruption even if evidence I/O fails.
        raise
    if process.returncode != 0:
        raise RuntimeError(f'Provenance query failed: {args}: {stderr.decode(errors="replace")}')
    return stdout.decode()


def command(args, name, cwd=ROOT, timeout=600, cpu=True):
    """Preserve argv/cwd/env and every failed attempt; never retry a command."""
    argv = ['taskset', '-c', str(CPU)] + list(map(str, args)) if cpu else list(map(str, args))
    log = REPORT / 'execution' / (name + '.txt')
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    interruption, cleanup_error = None, None
    with log.open('xb') as stream:
        process = subprocess.Popen(argv, cwd=cwd, env=env(), stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code, timed_out = process.wait(timeout=timeout), False
        except BaseException as error:
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            interruption = None if timed_out else error
            try:
                terminate_owned_group(process)
            except BaseException as cleanup:
                cleanup_error = repr(cleanup)
            code = process.returncode
    metadata = {
        'argv': argv, 'cwd': str(cwd), 'started_unix': started,
        'elapsed_seconds': time.time()-started, 'returncode': code,
        'timed_out': timed_out, 'deadline_seconds': timeout,
        'interruption':repr(interruption) if interruption is not None else None,
        'cleanup_exception':cleanup_error,
        'environment': {k: env()[k] for k in ('PYTHONHASHSEED', 'PYTHONDONTWRITEBYTECODE',
            'XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MAKEFLAGS')},
        'neutralized_loader_environment': {key: os.environ.get(key) for key in
            ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT')},
        'compiler_environment': {key: env().get(key) for key in
            ('PATH', 'FC', 'CC', 'CXX', 'FFLAGS', 'CFLAGS', 'CPPFLAGS', 'LDFLAGS',
             'CPATH', 'C_INCLUDE_PATH', 'LIBRARY_PATH', 'COMPILER_PATH', 'GCC_EXEC_PREFIX',
             'NIX_CFLAGS_COMPILE', 'NIX_CFLAGS_COMPILE_FOR_TARGET', 'NIX_CFLAGS_LINK',
             'NIX_LDFLAGS', 'NIX_LDFLAGS_FOR_TARGET', 'NIX_ENFORCE_NO_NATIVE',
             'NIX_HARDENING_ENABLE', 'GFORTRAN_INCLUDE_PATH', 'GFORTRAN_CONVERT_UNIT')},
        'log_sha256': digest(log)}
    try:
        save(log.with_suffix('.json'), metadata)
    finally:
        if interruption is not None:
            raise interruption
    if timed_out or code != 0:
        raise RuntimeError(f'{name} failed: returncode={code}, timeout={timed_out}; retained {log}')
    return log.read_text(errors='replace')


def frozen():
    record = json.loads((REPORT / 'source-freeze.json').read_text())
    for name, value in record['protocol'].items():
        if digest(REPORT / name) != value:
            raise RuntimeError(f'Frozen protocol changed: {name}')
    if installation() != record['installation']:
        raise RuntimeError('Qualified installation changed')
    return record


def freeze():
    save(REPORT / 'source-freeze.json', {
        'schema_version': 1, 'implementation_revision': REVISION, 'cpu': CPU,
        'protocol': {p: digest(REPORT / p) for p in PROTOCOL},
        'installation': installation(), 'native_execution_authorized': False,
        'note': 'Source freeze is provenance, not permission; root peer-review clearance is required separately.'})


def materialize():
    frozen()
    if MG.exists():
        raise RuntimeError('Private committed source already exists; no overwrite')
    MG.mkdir(parents=True)
    archive = subprocess.run(['git', 'archive', REVISION], cwd=ROOT,
                             check=True, stdout=subprocess.PIPE).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(MG, filter='data')
    original = inventory(MG)
    # Prebuilt external reducers are copied into private paths, never rebuilt.
    copied = {}
    for vendor in ('CutTools', 'IREGI'):
        for source in sorted((ROOT / 'vendor' / vendor).rglob('*')):
            if not source.is_file():
                continue
            if source.suffix not in ('.a', '.mod', '.so') and source.name != 'compiler_version.log':
                continue
            relative = source.relative_to(ROOT)
            destination = MG / relative
            if destination.exists():
                if digest(destination) != digest(source):
                    raise RuntimeError(f'Prebuilt artifact would change tracked source: {relative}')
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied[str(relative)] = {'source': str(source), 'sha256': digest(source)}
    verify_paths(original, MG)
    save(REPORT / 'materialized-source.json', {
        'revision': REVISION, 'archive_sha256': hashlib.sha256(archive).hexdigest(),
        'files': original, 'copied_prebuilt_artifacts': copied})
    for directory in ('xdg/config', 'xdg/cache', 'matplotlib'):
        (WORK / directory).mkdir(parents=True, exist_ok=True)


def generate():
    frozen()
    source = json.loads((REPORT / 'materialized-source.json').read_text())
    verify_paths(source['files'], MG)
    if OUTPUT.exists():
        raise RuntimeError('Generated output exists; no overwrite')
    main_config = ROOT / 'input/mg7_configuration.txt'
    config_before = main_config.read_bytes()
    main_model = inventory(ROOT / 'models/loop_sm')
    script = (REPORT / 'generate.mg5.in').read_text()
    replacements = {'INSTALL_LIB': str(INSTALL), 'OUTPUT': str(OUTPUT),
                    'NINJA_LIB': str(ROOT / 'HEPTools/ninja/lib'),
                    'COLLIER_LIB': str(ROOT / 'HEPTools/collier'),
                    'NINJARS_LIB': str(ROOT / 'HEPTools/ninjars/lib')}
    for key, value in replacements.items():
        script = script.replace('@'+key+'@', value)
    if re.search(r'@[A-Z_]+@', script):
        raise RuntimeError('Unexpanded CLI placeholder')
    script_path = REPORT / 'generate.mg5'
    with script_path.open('x') as stream:
        stream.write(script)
    command([sys.executable, MG / 'bin/madgraph', script_path], 'generate')
    if main_config.read_bytes() != config_before or inventory(ROOT / 'models/loop_sm') != main_model:
        raise RuntimeError('Protected main configuration/model changed')
    verify_paths(source['files'], MG)
    processes = sorted(OUTPUT.glob('SubProcesses/P*_*'))
    if len(processes) != 2:
        raise RuntimeError(f'Expected two subprocesses, found {processes}')
    roles = {}
    for identifier, spelling in ((101, 'uux_z'), (202, 'ddx_z')):
        matches = [p for p in processes if p.name == f'P{identifier}_{spelling}']
        if len(matches) != 1:
            raise RuntimeError(f'Unexpected process directory for {identifier}/{spelling}')
        directory = matches[0]
        prefix = (directory / 'proc_prefix.txt').read_text().strip()
        capacity = int(re.search(r'LOOPMAXCOEFS\s*=\s*(\d+)',
            (directory / 'loop_max_coefs.inc').read_text(), re.I).group(1))
        roles[str(identifier)] = {'directory': str(directory), 'prefix': prefix, 'capacity': capacity,
                                 'process_info': (directory / 'process_info.inc').read_text(),
                                 'loop_max_coefs': (directory / 'loop_max_coefs.inc').read_text()}
    if len({v['prefix'] for v in roles.values()}) != 2:
        raise RuntimeError('Subprocess prefixes alias')
    if len({v['capacity'] for v in roles.values()}) != 1 or min(v['capacity'] for v in roles.values()) < 15:
        raise RuntimeError('Unexpected generated diagnostic capacity')
    save(REPORT / 'generation.json', {'implementation_revision': REVISION, 'roles': roles,
        'source_inventory': inventory(OUTPUT, generated=True),
        'main_configuration_sha256': hashlib.sha256(config_before).hexdigest(),
        'main_model_before_after': main_model, 'private_source_after': inventory(MG),
        'cli_sha256': digest(script_path), 'hash_seed': '0'})


def prepare_build():
    frozen()
    generated = json.loads((REPORT / 'generation.json').read_text())
    verify_paths(generated['source_inventory'], OUTPUT)
    roles = generated['roles']
    directory = OUTPUT / 'SubProcesses'
    source = (REPORT / 'driver.f.in').read_text()
    for name, value in {'PREFIX_A': roles['101']['prefix'], 'PREFIX_B': roles['202']['prefix'],
                        'CAPACITY': str(roles['101']['capacity'])}.items():
        source = source.replace('@'+name+'@', value)
    if re.search(r'@[A-Z_]+@', source):
        raise RuntimeError('Unexpanded driver placeholder')
    (directory / 'lifecycle_driver.f').write_text(source)
    shutil.copy2(REPORT / 'observer.c', directory / 'observer.c')
    header = next(p for p in (INSTALL / 'tirust.h', INSTALL.parent / 'include/tirust.h') if p.is_file())
    shutil.copy2(header, directory / 'tirust.h')
    flags = ' '.join('-Wl,--wrap=ml_tirust_'+name for name in WRAPS)
    makefile = '''include makefile
lifecycle_plain: lifecycle_driver.o observer_plain.o $(OLP_PROCESS) $(LIBS) mg5_citation.o
\t$(FC) $(FFLAGS) -o $@ lifecycle_driver.o observer_plain.o $(OLP_PROCESS) mg5_citation.o $(LINKLIBS) $(RPATH_LIBS) -Wl,-Map,lifecycle_plain.map
lifecycle_observed: lifecycle_driver.o observer_observed.o $(OLP_PROCESS) $(LIBS) mg5_citation.o
\t$(FC) $(FFLAGS) -o $@ lifecycle_driver.o observer_observed.o $(OLP_PROCESS) mg5_citation.o $(LINKLIBS) $(RPATH_LIBS) @WRAP@ -Wl,-Map,lifecycle_observed.map
observer_plain.o: observer.c tirust.h
\t$(CC) -std=c11 -O2 -fno-fast-math -I. -c observer.c -o $@
observer_observed.o: observer.c tirust.h
\t$(CC) -std=c11 -O2 -fno-fast-math -DLC_OBSERVED=1 -I. -c observer.c -o $@
lifecycle-objects: lifecycle_driver.o observer_plain.o observer_observed.o $(OLP_PROCESS) $(LIBS) mg5_citation.o
lifecycle-list:
\t@echo $(OLP_PROCESS) mg5_citation.o
lifecycle-link-info:
\t@echo LIBS $(LIBS)
\t@echo LINKLIBS $(LINKLIBS)
\t@echo INCLUDES $(LOOP_INCLUDE)
\t@echo FC $(FC)
\t@echo CC $(CC)
\t@echo FFLAGS $(FFLAGS)
\t@echo LDFLAGS $(LDFLAGS)
'''.replace('@WRAP@', flags)
    (directory / 'lifecycle.mk').write_text(makefile)
    # The exact rendered sources are frozen before any compilation.
    for name in ('lifecycle_driver.f', 'observer.c', 'tirust.h', 'lifecycle.mk'):
        destination = REPORT / 'rendered' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write((directory / name).read_bytes())
    save(REPORT / 'generated-source-freeze.json', {'source_inventory': inventory(OUTPUT, generated=True),
        'rendered': {name: digest(directory/name) for name in
                     ('lifecycle_driver.f', 'observer.c', 'tirust.h', 'lifecycle.mk')},
        'installation': installation(), 'wraps': WRAPS})


def object_inputs(directory, objects):
    return {str((directory / name).resolve()): digest(directory/name) for name in objects}


def remaining(deadline, maximum=30):
    value = maximum if deadline is None else min(maximum, deadline-time.monotonic())
    if value <= 0:
        raise RuntimeError('The shared stage deadline expired')
    return value


def runtime(executable, deadline=None):
    text = capture(['ldd', executable], timeout=remaining(deadline))
    if 'not found' in text:
        raise RuntimeError('Unresolved shared dependency')
    paths = set(re.findall(r'(/[^\s()]+)', text))
    matches = re.findall(r'libmadloop_tirust\.so\s+=>\s+(/[^\s]+)', text)
    if (len(matches) != 1 or Path(matches[0]).resolve() != (INSTALL/'libmadloop_tirust.so').resolve()
            or digest(matches[0]) != EXPECTED['libmadloop_tirust.so']):
        raise RuntimeError('Runtime did not resolve the qualified TIRust DSO')
    return {'ldd': text, 'files': {p: digest(p) for p in sorted(paths) if Path(p).is_file()}}


def external_inputs(directory, expanded, compiler, deadline):
    paths, directories = set(), set()
    tokens = shlex.split(expanded)
    for index, token in enumerate(tokens):
        if token in ('-L', '-I') and index+1 < len(tokens):
            directories.add(tokens[index+1])
        elif token.startswith(('-L', '-I')) and len(token) > 2:
            directories.add(token[2:])
        elif token.startswith('/') and Path(token).is_file():
            paths.add(Path(token).resolve())
    for name in directories:
        container = Path(name)
        if not container.is_absolute():
            container = directory/container
        if container.is_dir():
            for path in container.iterdir():
                if path.is_file() and (path.suffix in ('.a', '.mod') or '.so' in path.name):
                    paths.add(path.resolve())
    # Resolve compiler-selected implicit runtime objects/libraries in advance.
    names = {token[2:] for token in tokens if token.startswith('-l')}
    names.update(('gfortran', 'quadmath', 'gcc', 'gcc_s', 'm', 'c'))
    filenames = {f'lib{name}{suffix}' for name in names for suffix in ('.a', '.so')}
    filenames.update(('crt1.o','Scrt1.o','crti.o','crtn.o','crtbegin.o','crtbeginS.o','crtend.o','crtendS.o'))
    for name in sorted(filenames):
        resolved = capture([*compiler, '-print-file-name='+name], cwd=directory,
            timeout=remaining(deadline, 10)).strip()
        path = Path(resolved)
        if path.is_absolute() and path.is_file():
            paths.add(path.resolve())
            if '.so' in path.name:
                paths.update(p.resolve() for p in path.parent.glob(path.name+'*') if p.is_file())
    # GNU linker scripts can name a real SONAME file with an absolute path.
    for path in list(paths):
        if path.stat().st_size < 65536:
            raw = path.read_bytes()
            if b'\0' not in raw:
                for name in re.findall(rb'/[^\s()]+', raw):
                    candidate = Path(os.fsdecode(name))
                    if candidate.is_file():
                        paths.add(candidate.resolve())
    return {str(path): digest(path) for path in sorted(paths)}


def verify_hashes(records):
    for path, expected in records.items():
        if not Path(path).is_file() or digest(path) != expected:
            raise RuntimeError('Bound link/compiler input changed: '+path)


def build():
    frozen()
    state = json.loads((REPORT / 'generated-source-freeze.json').read_text())
    verify_paths(state['source_inventory'], OUTPUT)
    directory = OUTPUT / 'SubProcesses'
    for name, value in state['rendered'].items():
        if digest(directory/name) != value:
            raise RuntimeError('Rendered source changed')
    deadline = time.monotonic()+600
    def bounded(args, name, cwd=directory):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise RuntimeError('The shared 600-second build deadline expired')
        return command(args, name, cwd, remaining)
    expanded = bounded(['make','--no-print-directory','-s','-f','lifecycle.mk','lifecycle-link-info'], 'link-info')
    fields = {line.split()[0]: shlex.split(line)[1:] for line in expanded.splitlines() if line.strip()}
    forbidden = {'-Ofast','-ffast-math','-funsafe-math-optimizations','-ffinite-math-only'}
    if forbidden.intersection(fields['FC']+fields['FFLAGS']+fields['LDFLAGS']):
        raise RuntimeError('Floating-environment qualification requires ordinary IEEE compiler flags')
    compiler_records = {}
    for key in ('FC', 'CC'):
        argv = fields[key]
        executable = shutil.which(argv[0], path=env()['PATH'])
        if executable is None:
            raise RuntimeError('Cannot resolve '+key)
        compiler_records[key] = {'argv': argv, 'path': str(Path(executable).resolve()),
            'sha256': digest(executable),
            'version': bounded([*argv,'--version'], 'compiler-'+key)}
    external_before = external_inputs(directory, expanded, fields['FC'], deadline)
    bounded(['make', '-j1'], 'build-model-dhelas', OUTPUT / 'Source')
    verify_hashes(external_before)
    recipe = bounded(['make','-n','-f','lifecycle.mk','lifecycle_plain','lifecycle_observed'], 'build-recipes')
    objects = bounded(['make','--no-print-directory','-s','-f','lifecycle.mk','lifecycle-list'], 'object-list').split()
    if len(objects) != len(set(objects)) or any('check_sa' in name for name in objects):
        raise RuntimeError('Duplicate numerical objects or generated program in common link')
    bounded(['make','-j1','-f','lifecycle.mk','lifecycle-objects'], 'build-process-objects')
    verify_hashes(external_before)
    verify_paths(state['source_inventory'], OUTPUT)
    objects_before = object_inputs(directory, objects+['lifecycle_driver.o'])
    link_before = external_inputs(directory, expanded, fields['FC'], deadline)
    executable_records, link_map_inputs = {}, {}
    for variant in ('plain', 'observed'):
        bounded(['make','-j1','-f','lifecycle.mk','lifecycle_'+variant], 'link-'+variant)
        verify_hashes(objects_before)
        verify_hashes(link_before)
        executable = directory/('lifecycle_'+variant)
        executable_records[variant] = {'path':str(executable), 'sha256':digest(executable),
                                       'runtime':runtime(executable, deadline)}
        map_path = directory/('lifecycle_'+variant+'.map')
        loads = {}
        for name in re.findall(r'^LOAD (.+)$', map_path.read_text(), re.M):
            path = Path(name)
            if not path.is_absolute():
                path = directory/path
            if path.is_file():
                loads[str(path.resolve())] = digest(path)
        # Require explicit external numerical libraries to have prior hashes.
        for path,value in loads.items():
            if path.endswith(('.a','.so','.so.1')) and path not in link_before:
                raise RuntimeError('Unbound resolved library in link map: '+path)
        link_map_inputs[variant] = loads
        shutil.copy2(map_path, REPORT/'execution')
    bounded(['nm','-u',INSTALL/'libtirust.a'], 'archive-undefined')
    bounded(['objdump','-r',INSTALL/'libtirust.a'], 'archive-relocations')
    bounded(['nm',directory/'lifecycle_observed'], 'observed-symbols')
    verify_paths(state['source_inventory'], OUTPUT)
    verify_hashes(objects_before)
    verify_hashes(link_before)
    remaining(deadline)
    save(REPORT/'build.json', {'numerical_objects':objects_before,'ordered_objects':objects,
        'compiler_recipes':str(REPORT/'execution/build-recipes.txt'),
        'recipe_sha256':hashlib.sha256(recipe.encode()).hexdigest(), 'compilers':compiler_records,
        'executables':executable_records,'installation':installation(),
        'external_before':external_before,'link_inputs_before_after':link_before,
        'resolved_link_map_inputs':link_map_inputs, 'library_inputs':inventory(OUTPUT/'lib'),
        'generated_source_freeze_sha256':digest(REPORT/'generated-source-freeze.json')})


def edit_card(text, changes):
    lines = text.splitlines()
    for name, value in changes.items():
        indices = [i for i,line in enumerate(lines) if line.strip() == '#'+name]
        if len(indices) != 1:
            raise RuntimeError(f'Missing or duplicate parameter {name}')
        i = indices[0] + 1
        while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith('!')):
            i += 1
        if i == len(lines) or lines[i].lstrip().startswith('#'):
            raise RuntimeError(f'No parameter value for {name}')
        lines[i] = str(value)
    return '\n'.join(lines)+'\n'


def run():
    frozen()
    build_state = json.loads((REPORT / 'build.json').read_text())
    generated = json.loads((REPORT/'generated-source-freeze.json').read_text())
    verify_paths(generated['source_inventory'], OUTPUT)
    for name, value in generated['rendered'].items():
        if digest(OUTPUT/'SubProcesses'/name) != value:
            raise RuntimeError('Rendered source changed before native execution')
    verify_paths(json.loads((REPORT/'materialized-source.json').read_text())['files'], MG)
    for path, value in build_state['numerical_objects'].items():
        if digest(path) != value:
            raise RuntimeError('Numerical object changed')
    verify_paths(build_state['library_inputs'], OUTPUT/'lib')
    verify_hashes(build_state['link_inputs_before_after'])
    for record in build_state['compilers'].values():
        if digest(record['path']) != record['sha256']:
            raise RuntimeError('Compiler identity changed')
    cards = OUTPUT/'Cards'
    nominal = (cards/'param_card.dat').read_text()
    # Change only mass-block PDG23, preserving every other physical card entry.
    block = None
    changed = []
    nchange = 0
    for line in nominal.splitlines():
        head = line.split('#', 1)[0].strip().split()
        if head and head[0].lower() == 'block':
            block = head[1].lower()
        elif head and head[0].lower() == 'decay':
            block = None
        if block == 'mass' and len(head) >= 2 and head[0] == '23':
            line = '  23  9.2000000000000000e+01 # MZ lifecycle alternate'
            nchange += 1
        changed.append(line)
    if nchange != 1:
        raise RuntimeError('Cannot uniquely change physical MZ')
    alternate = '\n'.join(changed)+'\n'
    ml_base = (OUTPUT/'SubProcesses/MadLoop5_resources/MadLoopParams.dat').read_text()
    for label, variant, cache in RUNS:
        directory = WORK/'runs'/label
        directory.mkdir(parents=True, exist_ok=False)
        evidence = REPORT/'runs'/label
        evidence.mkdir(parents=True, exist_ok=False)
        shutil.copytree(OUTPUT/'SubProcesses/MadLoop5_resources', directory/'resources')
        (directory/'param_card.dat').write_text(nominal)
        (directory/'param_alternate.dat').write_text(alternate)
        ml = edit_card(ml_base, {'MLReductionLib':9, 'CTModeInit':1, 'CTModeRun':-1,
            'MLStabThres':'1.0e-6', 'TIRustGlobalCache':cache, 'TIRustComputeFinite':'.TRUE.',
            'UseQPIntegrandForTIRust':'.TRUE.', 'LoopPoleMode':0,
            'TIRustCacheMaxBytes':268435456, 'WriteOutFilters':'.FALSE.'})
        (directory/'resources/MadLoopParams.dat').write_text(ml)
        for name in ('param_card.dat', 'param_alternate.dat'):
            shutil.copy2(directory/name, evidence/name)
        (evidence/'MadLoopParams.dat').write_text(ml)
        executable = Path(build_state['executables'][variant]['path'])
        if digest(executable) != build_state['executables'][variant]['sha256']:
            raise RuntimeError('Executable changed')
        if runtime(executable)['files'] != build_state['executables'][variant]['runtime']['files']:
            raise RuntimeError('Runtime changed')
        input_inventory = inventory(directory)
        save(evidence/'inputs.json', {'variant': variant, 'cache': cache,
            'executable': build_state['executables'][variant],
            'input_files': input_inventory, 'build_sha256': digest(REPORT/'build.json')})
        lock_path = ROOT/'.tirust-build/numerical-tests.lock'
        # Lock wait is external scheduling, not part of the 180s native deadline.
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                command([executable, str(cache)], 'native-'+label, directory, 180)
            finally:
                for name in ('records.txt', 'trace.txt'):
                    if (directory/name).is_file():
                        shutil.copy2(directory/name, evidence/name)
        verify_paths(input_inventory, directory)
        after_inputs = {name:item for name,item in inventory(directory).items()
                        if name not in ('records.txt','trace.txt')}
        if after_inputs != input_inventory:
            raise RuntimeError('Native execution changed or added a runtime input')
        verify_paths(generated['source_inventory'], OUTPUT)
        verify_paths(json.loads((REPORT/'materialized-source.json').read_text())['files'], MG)
        generation = json.loads((REPORT/'generation.json').read_text())
        if digest(ROOT/'input/mg7_configuration.txt') != generation['main_configuration_sha256']:
            raise RuntimeError('Protected main configuration changed')
        verify_paths(generation['main_model_before_after'], ROOT/'models/loop_sm')
        if digest(executable) != build_state['executables'][variant]['sha256']:
            raise RuntimeError('Executable changed after execution')
        if runtime(executable)['files'] != build_state['executables'][variant]['runtime']['files']:
            raise RuntimeError('Runtime changed after execution')
        for path, value in build_state['numerical_objects'].items():
            if digest(path) != value:
                raise RuntimeError('Numerical object changed after execution')
        verify_paths(build_state['library_inputs'], OUTPUT/'lib')
        verify_hashes(build_state['link_inputs_before_after'])
        if installation() != build_state['installation']:
            raise RuntimeError('Installation changed after execution')
    command([sys.executable, REPORT/'analyze.py', REPORT/'runs', '--output', REPORT/'results.json'],
            'analyze', timeout=60, cpu=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'materialize', 'generate', 'prepare-build', 'build', 'run'])
    args = parser.parse_args()
    if Path.cwd() != ROOT:
        parser.error('All work must start from the MadGraph repository root')
    WORK.mkdir(parents=True, exist_ok=True)
    {'freeze':freeze, 'materialize':materialize, 'generate':generate,
     'prepare-build':prepare_build, 'build':build, 'run':run}[args.stage]()

if __name__ == '__main__':
    main()
