#!/usr/bin/env python3
"""Explicit, append-only current-source replay of the successful lifecycle protocol.
Default is a source-only plan. Every real stage needs --execute and coordinator admission.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import time
import types

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
REPORT = ROOT / '.tirust-build/subprocess-cache-successor-results-1'
WORK = ROOT / '.tirust-build/subprocess-cache-successor-work-1'
MG, OUTPUT = WORK / 'mg', WORK / 'export'
REVISION = '0fcd2cd873d61d134837e1d9e87afb13a47824fd'
MANIFEST_SHA = '4a27dda7a0977cd1b0ed0d779b30c403da50968bf494aa96d31cf18fc9fe2076'
GENERATOR_SHA = 'afb6e0bea64ea0932751ff4869c206bcea9477d40a39a54bb99d0f06d85b75a6'
CPU = 12
STAGES = ('freeze', 'materialize', 'generate', 'prepare-build', 'build', 'smoke', 'run')
CAPS = dict.fromkeys(STAGES, 600)
CAPS.update(smoke=30, run=750)
RUNS = [('plain-cache0', 'plain', 0), ('plain-cache-all', 'plain', -1),
        ('observed-cache0', 'observed', 0), ('observed-cache-all', 'observed', -1)]
OUTPUTS = {'records.txt', 'trace.txt', 'param.log', 'runtime-maps-open.txt', 'runtime-maps-close.txt'}
ACTIVE = None
REQUESTS = None
QUERY = 0
CURRENT_STAGE = None
PROTOCOL = ('driver.f.in', 'card_smoke.f.in', 'observer.c', 'generate.mg5.in', 'analyze.py')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cleanup = load('current_lifecycle_cleanup', HERE / 'process_cleanup.py')
q = load('current_lifecycle_legacy', HERE / 'legacy_qualify.py')
analyzer = load('current_lifecycle_analysis', HERE / 'analyze.py')
smoke_parser = load('current_lifecycle_smoke', HERE / 'smoke.py')
linker = load('current_lifecycle_linker_binding', HERE / 'linker_bind.py')
q.ROOT, q.REPORT, q.WORK, q.MG, q.OUTPUT = ROOT, REPORT, WORK, MG, OUTPUT
q.REVISION, q.CPU = REVISION, CPU
smoke_parser.analyzer = lambda: analyzer


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return q.digest(path)


def save(path, value):
    q.save(path, value)


def now():
    return datetime.now(timezone.utc).isoformat()


class Deadline:
    """Signal-bounded synchronous work; no blocked signal mask reaches a child."""
    def __init__(self, seconds):
        self.seconds, self.handoff, self.deferred = seconds, False, False

    def expired(self, *_):
        if self.handoff:
            self.deferred = True
        else:
            raise TimeoutError('Inclusive work deadline reached; cleanup reserve begins')

    def __enter__(self):
        global ACTIVE
        self.parent = ACTIVE
        self.end = min(time.monotonic() + self.seconds,
                       ACTIVE.end if ACTIVE is not None else float('inf'))
        self.previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self.expired)
        signal.setitimer(signal.ITIMER_REAL, max(.001, self.end - time.monotonic() - 10))
        ACTIVE = self
        return self

    def __exit__(self, kind, *_):
        global ACTIVE
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)
        ACTIVE = self.parent
        if ACTIVE is not None:
            signal.setitimer(signal.ITIMER_REAL, max(.001, ACTIVE.end - time.monotonic() - 10))
        if kind is None and time.monotonic() > self.end:
            raise TimeoutError('Inclusive deadline exceeded')


def remaining(maximum=600):
    REQUESTS.check()
    require(ACTIVE is not None, 'External action needs a stage deadline')
    value = min(maximum, ACTIVE.end - time.monotonic() - 10)
    if value <= 0:
        raise TimeoutError('Stage work deadline exhausted')
    return value


def environment(cwd):
    result = q.env()
    shell = shutil.which('bash', path=result['PATH'])
    require(shell is not None, 'Pinned Bash missing')
    result.update(PWD=str(cwd), MAKEFLAGS='-j1 SHELL=' + str(Path(shell).resolve()),
                  RAYON_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return result


def recorded_environment(env):
    names = ('PATH', 'PWD', 'FC', 'CC', 'CXX', 'FFLAGS', 'CFLAGS', 'CXXFLAGS',
             'LDFLAGS', 'MAKEFLAGS', 'PYTHONHASHSEED', 'PYTHONDONTWRITEBYTECODE',
             'XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'OMP_NUM_THREADS',
             'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS', 'MKL_NUM_THREADS',
             'NIX_CFLAGS_COMPILE', 'NIX_LDFLAGS')
    return {name: env[name] for name in names if name in env}


def command(args, name, cwd=ROOT, timeout=600, cpu=True):
    """All child creation, waiting, failure retention and cleanup has one owner."""
    argv = list(map(str, args))
    env = environment(cwd)
    if argv[0] == 'make':
        argv.append('SHELL=' + str(Path(shutil.which('bash', path=env['PATH'])).resolve()))
    if cpu:
        argv = ['taskset', '-c', str(CPU)] + argv
    cap = remaining(timeout)
    path = REPORT / 'execution' / (name + '.txt')
    path.parent.mkdir(parents=True, exist_ok=True)
    start, stamp = time.monotonic(), now()
    process = owned = None
    failure = None
    closed = None
    try:
        with path.open('xb') as stream:
            ACTIVE.handoff = True
            try:
                process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                owned = cleanup.OwnedChild(process)
            finally:
                ACTIVE.handoff = False
            if ACTIVE.deferred:
                ACTIVE.expired()
            end = start + cap
            while True:
                owned.sample()
                REQUESTS.check()
                if time.monotonic() >= end:
                    raise TimeoutError('Child command deadline exhausted')
                try:
                    code = process.wait(timeout=min(.1, max(.001, end - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    pass
            require(code == 0, 'Command returned ' + str(code))
    except BaseException as error:
        failure = error
    finally:
        # The outer ALRM must not interrupt cleanup of our already-owned child.
        prior = signal.getsignal(signal.SIGALRM)
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_failure = None
        try:
            if owned is not None:
                closed = owned.stop(cooperative_seconds=0, terminate_seconds=4, kill_seconds=4)
                require(not closed['survivors'], 'Owned child survived cleanup')
            elif process is not None:
                # Popen succeeded but identity acquisition failed: retain and reap
                # that exact still-owned Popen child without PID-based guessing.
                process.kill()
                process.wait(timeout=4)
        except BaseException as error:
            cleanup_failure = error
            failure = failure or error
        finally:
            if ACTIVE is not None:
                signal.signal(signal.SIGALRM, prior)
                left = ACTIVE.end - time.monotonic() - 10
                if left > 0:
                    signal.setitimer(signal.ITIMER_REAL, left)
                else:
                    failure = failure or TimeoutError('Inclusive work deadline exhausted during cleanup')
        if path.exists():
            save(path.with_suffix('.json'), {'argv': argv, 'cwd': str(cwd), 'environment': recorded_environment(env),
                'started_utc': stamp, 'elapsed_seconds': time.monotonic() - start,
                'command_work_cap_seconds': cap, 'returncode': None if process is None else process.returncode,
                'exception': None if failure is None else repr(failure), 'cleanup': closed,
                'cleanup_exception': None if cleanup_failure is None else repr(cleanup_failure),
                'log_sha256': sha(path), 'cpu': CPU, 'cpu_exclusive': False})
    if failure is not None:
        raise failure
    return path.read_text()


def capture(args, cwd=ROOT, timeout=30):
    global QUERY
    require(CURRENT_STAGE in STAGES, 'Query requires an explicit stage namespace')
    QUERY += 1
    return command(args, 'query-%s-%04d' % (CURRENT_STAGE, QUERY), cwd, timeout)


def archive_run(args, cwd=None, check=None, stdout=None):
    require(args == ['git', 'archive', REVISION], 'Unexpected inherited subprocess.run')
    archive = WORK / 'materialized-source.tar'
    command(['git', 'archive', REVISION, '--output', archive], 'source-archive', ROOT)
    return subprocess.CompletedProcess(args, 0, stdout=archive.read_bytes())


q.command, q.capture = command, capture
q.subprocess = types.SimpleNamespace(**vars(subprocess))
q.subprocess.run = archive_run


def read_manifest():
    path = HERE / 'campaign-manifest.json'
    require(sha(path) == MANIFEST_SHA, 'Campaign manifest changed')
    value = json.loads(path.read_text())
    identity = value['candidate']['exporter_identity']
    require(value['id'] == 'entry-latch-cache-guard-1' and identity['revision'] == REVISION
            and identity['files_sha256'] == GENERATOR_SHA, 'Current candidate identity differs')
    return value


def production_sources(manifest):
    records = dict(manifest['candidate']['exporter_identity']['files'])
    for name, value in manifest['candidate']['numerical_sources'].items():
        require(name not in records or records[name] == value['sha256'], 'Conflicting registered source')
        records[name] = value['sha256']
    return records


def validate_source(directory, manifest):
    for name, digest in production_sources(manifest).items():
        path = directory / name
        if digest is None:
            require(path.is_symlink() and not path.exists(), 'Expected unresolved historical link: ' + name)
        else:
            require(path.is_file() and sha(path) == digest, 'Registered source mismatch: ' + name)


def source_seal():
    index = json.loads((HERE / 'source-index.json').read_text())
    for name, record in index['files'].items():
        require(sha(HERE / name) == record['sha256'], 'Frozen protocol changed: ' + name)
    manifest = read_manifest()
    validate_source(ROOT, manifest)
    return index, manifest


def frozen():
    source_seal()
    state = json.loads((REPORT / 'source-freeze.json').read_text())
    require(state['source_index_sha256'] == sha(HERE / 'source-index.json')
            and state['campaign_manifest_sha256'] == MANIFEST_SHA, 'Execution freeze differs')
    require(state['protocol'] == {name: sha(HERE / name) for name in PROTOCOL}, 'Copied protocol differs from sealed sources')
    for name, digest in state['protocol'].items():
        require(sha(REPORT / name) == digest, 'Execution protocol changed: ' + name)
    require(sha(ROOT / 'input/mg7_configuration.txt') == state['protected_configuration_sha256'], 'Main configuration changed')
    require(q.installation() == state['installation'], 'Installed native package changed')
    return state


q.frozen = frozen


def plan():
    return {'schema': 'tirust-current-source-cache-lifecycle-plan-v1',
        'campaign': 'entry-latch-cache-guard-1', 'manifest_sha256': MANIFEST_SHA,
        'source_revision': REVISION, 'generator_sha256': GENERATOR_SHA,
        'cpu': CPU, 'cpu_exclusive': False, 'jobs': 1, 'stages': CAPS,
        'commands': [[sys.executable, '-B', str(HERE / 'run.py'), stage, '--execute'] for stage in STAGES],
        'physical_workers': 4, 'physical_calls_per_worker': 19, 'key_calls_per_worker': 28,
        'warmup_maximum_per_prefix': 32, 'native_backend': 9,
        'native_lock': str(ROOT / '.tirust-build/numerical-tests.lock'),
        'performance_eligible': False, 'execution_performed': False}


def freeze():
    index, manifest = source_seal()
    require(not REPORT.exists() and not WORK.exists(), 'Fresh report/work namespace required')
    REPORT.mkdir(parents=True)
    WORK.mkdir(parents=True)
    for name in PROTOCOL:
        shutil.copy2(HERE / name, REPORT / name)
    installed = q.installation()
    for name, digest in manifest['candidate']['native']['files'].items():
        require(sha(q.INSTALL.parent / name) == digest, 'Manifest/native package mismatch: ' + name)
    save(REPORT / 'source-freeze.json', {'schema_version': 1, 'implementation_revision': REVISION,
        'source_index_sha256': sha(HERE / 'source-index.json'), 'campaign_manifest_sha256': MANIFEST_SHA,
        'registered_sources': production_sources(manifest), 'installation': installed,
        'protocol': {name: sha(HERE / name) for name in PROTOCOL},
        'protected_configuration_sha256': sha(ROOT / 'input/mg7_configuration.txt'),
        'cpu': CPU, 'native_execution_authorized': False, 'plan': plan()})


def materialize():
    q.materialize()
    validate_source(MG, read_manifest())


def prepare_build():
    q.prepare_build()
    sub = OUTPUT / 'SubProcesses'
    roles = json.loads((REPORT / 'generation.json').read_text())['roles']
    source = (HERE / 'card_smoke.f.in').read_text()
    for key, value in {'PREFIX_A': roles['101']['prefix'], 'PREFIX_B': roles['202']['prefix'],
                       'CAPACITY': roles['101']['capacity']}.items():
        source = source.replace('@' + key + '@', str(value))
    require(not re.search(r'@[A-Z_]+@', source), 'Unexpanded smoke driver')
    (sub / 'lifecycle_card_smoke.f').write_text(source)
    make = sub / 'lifecycle.mk'
    old = make.read_text()
    flags = ' '.join('-Wl,--wrap=ml_tirust_' + n for n in q.WRAPS)
    extra = '''
lifecycle_card_smoke: lifecycle_card_smoke.o observer_observed.o $(OLP_PROCESS) $(LIBS) mg5_citation.o
\t$(FC) $(FFLAGS) -o $@ lifecycle_card_smoke.o observer_observed.o $(OLP_PROCESS) mg5_citation.o $(LINKLIBS) $(RPATH_LIBS) @WRAP@ -Wl,-Map,lifecycle_card_smoke.map
lifecycle-objects: lifecycle_card_smoke.o
'''.replace('@WRAP@', flags)
    make.write_text(old + extra)
    initial = REPORT / 'generated-source-freeze.json'
    initial.rename(REPORT / 'generated-source-freeze-before-smoke.json')
    for name in ('lifecycle.mk', 'lifecycle_card_smoke.f'):
        shutil.copy2(sub / name, REPORT / ('rendered-final-' + name))
    state = json.loads((REPORT / 'generated-source-freeze-before-smoke.json').read_text())
    state.update(source_inventory=q.inventory(OUTPUT, generated=True),
        rendered={name: sha(sub / name) for name in ('lifecycle_driver.f', 'lifecycle_card_smoke.f',
                                                    'observer.c', 'tirust.h', 'lifecycle.mk')},
        initial_freeze_sha256=sha(REPORT / 'generated-source-freeze-before-smoke.json'))
    save(initial, state)


def interpreter(path):
    raw = Path(path).read_bytes()
    require(raw[:6] == b'\x7fELF\x02\x01', 'Expected ELF64 little endian')
    offset = struct.unpack_from('<Q', raw, 32)[0]
    size, count = struct.unpack_from('<HH', raw, 54)
    found = []
    for index in range(count):
        start = offset + index * size
        if struct.unpack_from('<I', raw, start)[0] == 3:
            pos, length = struct.unpack_from('<Q', raw, start + 8)[0], struct.unpack_from('<Q', raw, start + 32)[0]
            found.append(raw[pos:pos+length].rstrip(b'\0').decode())
    require(len(found) == 1 and Path(found[0]).is_absolute(), 'Unique ELF interpreter required')
    return Path(found[0]).resolve()


def runtime(executable, deadline=None):
    loader = interpreter(executable)
    loader_sha256 = sha(loader)
    raw = capture([loader, '--list', executable], timeout=remaining(30))
    require(sha(loader) == loader_sha256, 'Actual ELF loader changed during dependency query')
    require('not found' not in raw, 'Missing runtime dependency')
    paths = {str(Path(p).resolve()) for p in re.findall(r'(/[^\s()]+)', raw) if Path(p).is_file()}
    paths.add(str(loader))
    require(str((q.INSTALL / 'libmadloop_tirust.so').resolve()) in paths, 'Selected TIRust DSO absent')
    return {'loader': str(loader), 'loader_sha256': loader_sha256, 'loader_listing': raw,
            'files': {p: sha(p) for p in sorted(paths)}}


q.runtime = runtime


def validate_rendered(state):
    required = {'lifecycle_driver.f', 'lifecycle_card_smoke.f', 'observer.c', 'tirust.h', 'lifecycle.mk'}
    require(set(state['rendered']) == required, 'Incomplete rendered-source inventory')
    for name, digest in state['rendered'].items():
        require(sha(OUTPUT / 'SubProcesses' / name) == digest, 'Rendered source changed: ' + name)


def build():
    frozen()
    state = json.loads((REPORT / 'generated-source-freeze.json').read_text())
    sub = OUTPUT / 'SubProcesses'
    q.verify_paths(state['source_inventory'], OUTPUT)
    validate_rendered(state)
    info = command(['make', '-s', '--no-print-directory', '-f', 'lifecycle.mk', 'lifecycle-link-info'], 'link-info', sub)
    fields = {line.split()[0]: shlex.split(line)[1:] for line in info.splitlines() if line.strip()}
    require(not {'-Ofast', '-ffast-math', '-funsafe-math-optimizations', '-ffinite-math-only'} & set(fields['FC'] + fields['FFLAGS'] + fields['LDFLAGS']), 'Unsafe floating flags')
    compilers = {}
    for key in ('FC', 'CC'):
        executable = Path(shutil.which(fields[key][0], path=environment(sub)['PATH'])).resolve()
        compilers[key] = {'argv': fields[key], 'path': str(executable), 'sha256': sha(executable),
                          'version': command(fields[key] + ['--version'], 'compiler-' + key, sub)}
    command(['make', '--always-make', '-j1'], 'build-model-dhelas', OUTPUT / 'Source')
    names = command(['make', '-s', '--no-print-directory', '-f', 'lifecycle.mk', 'lifecycle-list'], 'object-list', sub).split()
    require(len(names) == len(set(names)) and not any('check_sa' in n for n in names), 'Duplicate/program numerical objects')
    command(['make', '--always-make', '-j1', '-f', 'lifecycle.mk', 'lifecycle-objects'], 'build-process-objects', sub)
    objects = q.object_inputs(sub, names + ['lifecycle_driver.o', 'lifecycle_card_smoke.o', 'observer_plain.o', 'observer_observed.o'])
    external = q.external_inputs(sub, info, fields['FC'], ACTIVE.end - 10)
    # Include compiler-selected script search paths before linking, avoiding the
    # historical libgcc_s linker-script omission without an unbound-input waiver.
    search = command(fields['FC'] + ['-print-search-dirs'], 'compiler-search-dirs', sub)
    for line in search.splitlines():
        if line.startswith('libraries: ='):
            for name in line.split('=', 1)[1].split(':'):
                directory = Path(name)
                if directory.is_dir():
                    for path in directory.iterdir():
                        if path.is_file() and (path.suffix in ('.a', '.o') or '.so' in path.name):
                            external[str(path.resolve())] = sha(path)
    wrapper = linker.wrapper_flags(Path(compilers['FC']['path']), environment(sub))
    paths = linker.search_paths(info, search, wrapper['flags'], sub)
    closure = linker.prebind(external, paths, linker.compiler_libraries(info + '\n' + wrapper['flags']))
    external.update(closure['files'])
    external.update({name: item['sha256'] for name, item in wrapper['inputs'].items()})
    save(REPORT / 'linker-prebinding.json', {'wrapper': wrapper, 'closure': closure,
         'search_paths': [str(p) for p in paths], 'completed_before_any_lifecycle_link': True})
    bound = {**external, **objects}
    q.verify_hashes(bound)
    executables = {}
    for key, target in [('plain', 'lifecycle_plain'), ('observed', 'lifecycle_observed'), ('card-smoke', 'lifecycle_card_smoke')]:
        command(['make', '-n', '-f', 'lifecycle.mk', target], 'recipe-' + key, sub)
        command(['make', '-j1', '-f', 'lifecycle.mk', target], 'link-' + key, sub)
        q.verify_hashes(bound)
        map_path = sub / (target + '.map')
        loads = {}
        for spelling in re.findall(r'^LOAD (.+)$', map_path.read_text(), re.M):
            path = Path(spelling)
            if not path.is_absolute():
                path = sub / path
            resolved = str(path.resolve())
            linker.require_prebound_load(path, bound)
            loads[resolved] = bound[resolved]
        shutil.copy2(map_path, REPORT / 'execution' / map_path.name)
        executable = sub / target
        executables[key] = {'path': str(executable), 'sha256': sha(executable),
                            'runtime': runtime(executable), 'link_loads': loads, 'map_sha256': sha(map_path)}
    q.verify_paths(state['source_inventory'], OUTPUT)
    validate_rendered(state)
    q.verify_hashes(bound)
    frozen()
    save(REPORT / 'build.json', {'schema_version': 1, 'implementation_revision': REVISION,
        'source_freeze_sha256': sha(REPORT / 'source-freeze.json'),
        'generated_source_freeze_sha256': sha(REPORT / 'generated-source-freeze.json'),
        'campaign_manifest_sha256': MANIFEST_SHA, 'numerical_objects': objects,
        'all_link_inputs_before_after': bound, 'executables': executables, 'compilers': compilers,
        'linker_prebinding_sha256': sha(REPORT / 'linker-prebinding.json'),
        'fresh_numerical_source_build': True, 'libraries': q.inventory(OUTPUT / 'lib')})


def alternate_card(text):
    block, count, lines = None, 0, []
    for line in text.splitlines():
        head = line.split('#', 1)[0].split()
        if head and head[0].lower() == 'block':
            block = head[1].lower()
        elif head and head[0].lower() == 'decay':
            block = None
        if block == 'mass' and len(head) >= 2 and head[0] == '23':
            line = '  23  9.200000e+01 # MZ lifecycle alternate'
            count += 1
        lines.append(line)
    require(count == 1, 'Unique MZ required')
    result = '\n'.join(lines) + '\n'
    smoke_parser.numeric_prevalidation(result)
    return result


def input_layout(directory, cache):
    resources = OUTPUT / 'SubProcesses/MadLoop5_resources'
    shutil.copytree(resources, directory / 'resources')
    nominal = (OUTPUT / 'Cards/param_card.dat').read_text()
    smoke_parser.numeric_prevalidation(nominal)
    for label, card in [('nominal', nominal), ('alternate', alternate_card(nominal))]:
        (directory / label).mkdir()
        (directory / label / 'param_card.dat').write_text(card)
        shutil.copy2(OUTPUT / 'Cards/ident_card.dat', directory / label / 'ident_card.dat')
    text = q.edit_card((resources / 'MadLoopParams.dat').read_text(), {
        'MLReductionLib': 9, 'CTModeInit': 1, 'CTModeRun': -1, 'MLStabThres': '1.0e-6',
        'TIRustGlobalCache': cache, 'TIRustComputeFinite': '.TRUE.', 'UseQPIntegrandForTIRust': '.TRUE.',
        'LoopPoleMode': 0, 'TIRustCacheMaxBytes': 268435456, 'WriteOutFilters': '.FALSE.'})
    (directory / 'resources/MadLoopParams.dat').write_text(text)


def mapped_files(raw):
    paths = set()
    for line in raw.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            require(not fields[5].endswith(' (deleted)'), 'Deleted live mapping')
            path = Path(fields[5])
            with path.open('rb') as stream:
                if stream.read(4) == b'\x7fELF':
                    paths.add(str(path.resolve()))
    return {name: sha(name) for name in sorted(paths)}


def validate_build():
    frozen()
    state = json.loads((REPORT / 'build.json').read_text())
    require(state['source_freeze_sha256'] == sha(REPORT / 'source-freeze.json')
            and state['generated_source_freeze_sha256'] == sha(REPORT / 'generated-source-freeze.json'), 'Build/source freeze changed')
    require(state['linker_prebinding_sha256'] == sha(REPORT / 'linker-prebinding.json'), 'Pre-link closure evidence changed')
    q.verify_hashes(state['all_link_inputs_before_after'])
    generated = json.loads((REPORT / 'generated-source-freeze.json').read_text())
    q.verify_paths(generated['source_inventory'], OUTPUT)
    validate_rendered(generated)
    q.verify_paths(state['libraries'], OUTPUT / 'lib')
    validate_source(MG, read_manifest())
    return state


def require_complete_console(text):
    require('##W03' not in text, 'Incomplete amplitude warning')


def execute_worker(label, variant, cache, seconds):
    with Deadline(seconds):
        return _execute_worker(label, variant, cache, seconds)


def _execute_worker(label, variant, cache, seconds):
    started = time.monotonic()
    built = validate_build()
    binary = built['executables'][variant]
    executable = Path(binary['path'])
    require(sha(executable) == binary['sha256'], 'Executable changed')
    require(runtime(executable)['files'] == binary['runtime']['files'], 'Runtime changed')
    directory, evidence = WORK / 'runs' / label, REPORT / 'runs' / label
    directory.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)
    input_layout(directory, cache)
    before = q.inventory(directory)
    save(evidence / 'inputs.json', {'label': label, 'variant': variant, 'cache': cache,
        'input_files': before, 'executable': binary, 'build_sha256': sha(REPORT / 'build.json')})
    for name in ('nominal', 'alternate'):
        shutil.copytree(directory / name, evidence / name)
    shutil.copy2(directory / 'resources/MadLoopParams.dat', evidence / 'MadLoopParams.dat')
    failure = None
    try:
        command([executable, str(cache)], 'native-' + label, directory,
                timeout=max(.1, seconds - (time.monotonic() - started) - 10))
    except BaseException as error:
        failure = error
    finally:
        for name in OUTPUTS:
            if (directory / name).is_file():
                shutil.copy2(directory / name, evidence / name)
    after = q.inventory(directory)
    outputs = {name: after.pop(name) for name in OUTPUTS if name in after}
    save(evidence / 'outputs.json', {'declared_outputs': outputs, 'input_files_after': after,
        'inputs_unchanged': before == after, 'execution_exception': None if failure is None else repr(failure)})
    if failure is not None:
        raise failure
    require(before == after and set(outputs) == OUTPUTS, 'Incomplete outputs or mutated runtime inputs')
    expected = {**binary['runtime']['files'], str(executable.resolve()): binary['sha256']}
    for name in ('runtime-maps-open.txt', 'runtime-maps-close.txt'):
        require(mapped_files((evidence / name).read_text()) == expected, 'Actual runtime mappings differ: ' + name)
    console = (REPORT / 'execution' / ('native-' + label + '.txt')).read_text()
    require_complete_console(console)
    validate_build()
    require(sha(executable) == binary['sha256'] and runtime(executable)['files'] == binary['runtime']['files'], 'Post-run runtime changed')
    require(time.monotonic() - started <= seconds, 'Worker inclusive bound exceeded')
    return evidence


def smoke():
    evidence = execute_worker('card-smoke', 'card-smoke', 0, 30)
    result = smoke_parser.check_smoke(evidence / 'records.txt', evidence / 'trace.txt')
    save(REPORT / 'card-smoke-result.json', {'analysis': result, 'build_sha256': sha(REPORT / 'build.json'),
        'evidence': q.inventory(evidence), 'execution_sha256': sha(REPORT / 'execution/native-card-smoke.json')})


def run():
    stage = json.loads((REPORT / 'stage-smoke.json').read_text())
    require(stage['status'] == 'passed' and stage['source_index_sha256'] == sha(HERE / 'source-index.json')
            and stage['campaign_manifest_sha256'] == MANIFEST_SHA, 'Smoke stage did not pass under current protocol')
    execution = json.loads((REPORT / 'execution/native-card-smoke.json').read_text())
    smoke_binary = validate_build()['executables']['card-smoke']
    require(execution['returncode'] == 0 and execution['exception'] is None
            and execution['cleanup_exception'] is None and not execution['cleanup']['survivors']
            and execution['argv'] == ['taskset', '-c', str(CPU), smoke_binary['path'], '0']
            and execution['cwd'] == str(WORK / 'runs/card-smoke'), 'Smoke execution prerequisite failed')
    smoke_outputs = json.loads((REPORT / 'runs/card-smoke/outputs.json').read_text())
    require(smoke_outputs['inputs_unchanged'] and smoke_outputs['execution_exception'] is None
            and set(smoke_outputs['declared_outputs']) == OUTPUTS, 'Smoke inputs/outputs prerequisite failed')
    prior = json.loads((REPORT / 'card-smoke-result.json').read_text())
    require(prior['evidence'] == q.inventory(REPORT / 'runs/card-smoke')
            and prior['build_sha256'] == sha(REPORT / 'build.json')
            and prior['execution_sha256'] == sha(REPORT / 'execution/native-card-smoke.json'), 'Smoke prerequisite changed')
    require(prior['analysis'] == smoke_parser.check_smoke(REPORT / 'runs/card-smoke/records.txt', REPORT / 'runs/card-smoke/trace.txt'), 'Smoke prerequisite parser mismatch')
    for label, variant, cache in RUNS:
        evidence = execute_worker(label, variant, cache, 180)
        analyzer.parse_records(evidence / 'records.txt')
        analyzer.parse_trace(evidence / 'trace.txt')
    result = analyzer.analyze(REPORT / 'runs')
    save(REPORT / 'results.json', result)
    require(result['passed'], 'Lifecycle comparison failed')


def main():
    global REQUESTS, CURRENT_STAGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', choices=STAGES)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Run from repository root')
    if not args.execute:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return
    require(args.stage is not None, '--execute requires one explicit stage')
    CURRENT_STAGE = args.stage
    source_seal()
    if args.stage != 'freeze':
        require(REPORT.exists() and WORK.exists(), 'Explicit freeze stage required')
    # Native lock waiting is scheduling only and starts no child; all actual work
    # has its separate inclusive deadline after acquisition. No CPU reservation.
    with cleanup.SignalRequests() as requests:
        REQUESTS = requests
        lock = None
        try:
            if args.stage in ('smoke', 'run'):
                lock = (ROOT / '.tirust-build/numerical-tests.lock').open('a')
                while True:
                    requests.check()
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(.1)
            marker = REPORT / ('stage-' + args.stage + '.json')
            require(not marker.exists(), 'Stage is append-only; no retry')
            start, stamp = time.monotonic(), now()
            failure = None
            try:
                with Deadline(CAPS[args.stage]):
                    {'freeze': freeze, 'materialize': materialize, 'generate': q.generate,
                     'prepare-build': prepare_build, 'build': build, 'smoke': smoke, 'run': run}[args.stage]()
                    source_seal()
                    frozen()
                    require(time.monotonic() - start <= CAPS[args.stage], 'Inclusive stage cap exceeded')
            except BaseException as error:
                failure = error
            if REPORT.exists():
                save(marker, {'stage': args.stage, 'started_utc': stamp, 'finished_utc': now(),
                    'elapsed_seconds': time.monotonic() - start, 'cap_seconds': CAPS[args.stage],
                    'status': 'passed' if failure is None else 'failed',
                    'exception': None if failure is None else repr(failure), 'source_index_sha256': sha(HERE / 'source-index.json'),
                    'campaign_manifest_sha256': MANIFEST_SHA, 'native_execution': args.stage in ('build', 'smoke', 'run'),
                    'native_physics_evaluations_permitted': args.stage == 'run',
                    'performance_eligible': False})
            if failure is not None:
                raise failure
        finally:
            if lock is not None:
                lock.close()


if __name__ == '__main__':
    main()
