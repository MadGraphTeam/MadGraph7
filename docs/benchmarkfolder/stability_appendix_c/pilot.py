#!/usr/bin/env python3
"""Appendix-C stability pilot. Planning is inert; native stages are explicit.

Only this private controller changes the driver or numerical card. Canonical
exports are read-only donors. A complete setup checkpoint is required by run.
No failed or incomplete request is retried, silently removed, or replaced.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import time
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAMPAIGN = ROOT / 'docs/benchmarkfolder/campaigns/invariant-symmetry-1'
WORK = ROOT / '.tirust-build/stability-appendix-c-pilot-1'
REPORT = WORK / 'evidence'
CASE_IDS = ('qcd_n2_gg_ttx', 'qcd_n3_uux_zgg', 'ew_n2_epem_mupmum', 'qcd_n4_uux_zzgg')
BACKENDS = (1, 6, 7, 8, 9)
REVISION = '2c13246ffa63aa357ad1e29e5e79aff63d88dc7e'
MANIFEST_SHA = 'ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0'
GENERATOR_SHA = 'a90f80545696a648dbcfdfb8de73b1e319364edf2b93d02d4458c7b9656b8b19'
SPEC_SHA = 'f4ecf8d003faf1a570de53d803ca46ed5d12089235786d8a9a590198406cafd7'
HISTORICAL_LDD_HOST_LOADER = {
    'path': '/nix/store/0d8g8n0a11v6f5m2h416ajyxmnkwc3md-glibc-2.42-67/lib64/ld-linux-x86-64.so.2',
    'sha256': 'e655ec5a68127ee03586457e67039ea9deef55abaeae519c1f53ad1cf18ee5b2',
    'binding': 'closed main numerical specification ' + SPEC_SHA + '; historical ldd metadata only',
}
HELPERS = {
    'links': ('docs/benchmarkfolder/experiments/current-cms-replay-supervisor-2/link_closure.py',
              'e6ac52e09b699e8929a567a9c963e7f8e52da6e18b0220f8ef8b1ec5a5ac92a7'),
    'cleanup': ('docs/benchmarkfolder/experiments/current-cms-replay-supervisor-1/ownership/process_cleanup.py',
                '240d1067a7b7098f056f0f367097def8f374d7979c3001d249c8b1f7cb0bbd84'),
}
MAKEFILE = '''include makefile
.PHONY: pilot-build
pilot-build:
\t$(FC) $(FFLAGS) -c check_sa.f -o check_sa.o $(LOOP_INCLUDE)
\t$(FC) $(FFLAGS) -o stability-check check_sa.o $(PROCESS) $(LINKLIBS)
'''
EXEC_TARGET = '''import json, os, sys
payload = json.loads(sys.argv[1])
if os.read(0, 1) != b"G":
    raise SystemExit(125)
os.close(0)
descriptor = os.open(os.devnull, os.O_RDONLY)
if descriptor != 0:
    os.dup2(descriptor, 0)
    os.close(descriptor)
environment = os.environ.copy()
environment.update(payload["environment"])
os.execvpe(payload["argv"][0], payload["argv"], environment)
'''


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')}
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def save(path, value):
    """Exclusive publication; a previous checkpoint is never overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(safe_json(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load_helpers():
    result = {}
    for name, (relative, checksum) in HELPERS.items():
        path = ROOT / relative
        blob = path.read_bytes()
        require(hashlib.sha256(blob).hexdigest() == checksum, 'Pinned helper changed: ' + relative)
        module = types.ModuleType('stability_pilot_' + name)
        module.__file__ = str(path)
        exec(compile(blob, str(path), 'exec'), module.__dict__)
        result[name] = module
    index = read(HERE / 'source-index.json')
    for name in ('points', 'plot_metrics'):
        path = HERE / (name + '.py')
        expected = index['files'][path.name]
        expected = expected['sha256'] if isinstance(expected, dict) else expected
        blob = path.read_bytes()
        require(hashlib.sha256(blob).hexdigest() == expected, 'Sealed local helper changed: ' + path.name)
        module = types.ModuleType('stability_pilot_' + name)
        module.__file__ = str(path)
        exec(compile(blob, str(path), 'exec'), module.__dict__)
        result[name] = module
    return result


def protocol():
    value = read(HERE / 'protocol.json')
    require(value['campaign_manifest_sha256'] == MANIFEST_SHA
            and value['generator_identity_sha256'] == GENERATOR_SHA
            and value['candidate_source_revision'] == REVISION,
            'Unexpected campaign or candidate identity')
    require(tuple(case['id'] for case in value['cases']) == CASE_IDS
            and tuple(value['backends']) == BACKENDS
            and value['point_count_per_process'] == 100,
            'The complete four-case, five-backend, 100-point pilot is required')
    limits = value['limits']
    require(limits['request_count'] == 20 and limits['measured_DP_calls'] == 6000
            and limits['request_wall_seconds'] == 1200
            and limits['request_memory_bytes'] == 4294967296
            and limits['cpu'] == 12 and limits['cpu_is_reserved'] is False
            and limits['automatic_retry'] is False
            and limits['common_lock'] == '.tirust-build/numerical-tests.lock'
            and limits['lock_backends'] == [8, 9] and limits['lock_wait_seconds'] == 3600
            and limits['lock_wait_is_outside_execution_cap'] is True,
            'Native limits differ from the admitted pilot')
    return value


def requests(value):
    return [{'id': case['id'] + '-backend' + str(backend), 'case': case['id'],
             'backend': backend, 'primary_calls_expected': 300,
             'lock_required': backend in (8, 9)}
            for case in value['cases'] for backend in value['backends']]


def verify_seal():
    """An execute stage needs a root-reviewed seal, including the worker."""
    index = read(HERE / 'source-index.json')
    files = index['files']
    required = {'pilot.py', 'test_pilot.py', 'protocol.json', 'stability_worker.f', 'points.py', 'plot_metrics.py'}
    require(required.issubset(files), 'Pilot source seal is incomplete')
    for name, item in files.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe source seal path')
        checksum = item['sha256'] if isinstance(item, dict) else item
        require(digest(HERE / name) == checksum, 'Sealed pilot source changed: ' + name)
    for name, checksum in index.get('inputs', {}).items():
        require(digest(ROOT / name) == checksum, 'Sealed pilot input changed: ' + name)
    for name, checksum in protocol().get('inputs', {}).items():
        require(digest(ROOT / name) == checksum, 'Protocol input changed: ' + name)
    return digest(HERE / 'source-index.json')


def point_array(text, nexternal):
    fields = text.split()
    require(fields and fields[0] == '100' and len(fields) == 1 + 100 * nexternal * 4,
            'Point stream must contain exactly 100 complete physical points')
    values = [float(item.replace('D', 'E').replace('d', 'e')) for item in fields[1:]]
    require(all(math.isfinite(item) for item in values), 'Nonfinite point input')
    return [[values[(point * nexternal + particle) * 4:(point * nexternal + particle + 1) * 4]
             for particle in range(nexternal)] for point in range(100)]


def cut_report(points, pt_min=50., dr_min=.5):
    """Cuts apply to final-state particles, before any amplitude is evaluated."""
    rejected = []
    for index, momenta in enumerate(points, 1):
        final = momenta[2:]
        pt = [math.hypot(p[1], p[2]) for p in final]
        if any(value <= pt_min for value in pt):
            rejected.append({'point': index, 'reason': 'final-state pt cut'})
            continue
        eta = [math.asinh(p[3] / transverse) for p, transverse in zip(final, pt)]
        phi = [math.atan2(p[2], p[1]) for p in final]
        if any(math.hypot(eta[i] - eta[j], math.remainder(phi[i] - phi[j], 2 * math.pi)) <= dr_min
               for i in range(len(final)) for j in range(i)):
            rejected.append({'point': index, 'reason': 'final-state pair delta_R cut'})
    return {'count': len(points), 'passed': len(points) - len(rejected), 'rejected': rejected}


def check_points(value, path, nexternal):
    points = point_array(Path(path).read_text(), nexternal)
    sampling = value['sampling']
    cuts = cut_report(points, sampling['final_state_pt_min_GeV'], sampling['final_state_pair_delta_R_min'])
    require(cuts['passed'] == 100,
            'Registered points fail Appendix-C cuts; freeze a separately declared stream before admission')
    return cuts


def verify_production(value):
    require(digest(CAMPAIGN / 'manifest.json') == MANIFEST_SHA, 'Campaign manifest changed')
    manifest = read(CAMPAIGN / 'manifest.json')
    identity = manifest['candidate']['exporter_identity']
    require(identity['revision'] == REVISION and identity['files_sha256'] == GENERATOR_SHA,
            'Candidate generator differs')
    files = dict(identity['files'])
    files.update({name: item['sha256'] for name, item in manifest['candidate']['numerical_sources'].items()})
    for name, checksum in files.items():
        require(digest(ROOT / name) == checksum, 'Production source differs: ' + name)
    native = manifest['candidate']['native']
    for name, checksum in native['files'].items():
        require(digest(Path(native['prefix']) / name) == checksum, 'Installed production artifact differs: ' + name)
    require(native['files']['lib/libmadloop_tirust.so'] == value['candidate_library_sha256'],
            'Registered TIRust DSO differs')
    spec_path = CAMPAIGN / 'numerical-qualification-1/specification.json'
    require(digest(spec_path) == SPEC_SHA, 'Closed-build numerical specification changed')
    return manifest, read(spec_path)


def donor(value, spec, case_id):
    """Read only one of four explicitly closed exports; EW n4 is forbidden."""
    require(case_id in CASE_IDS, 'Unregistered or potentially active case')
    export = ROOT / '.benchmark-work/tirust/campaigns/invariant-symmetry-1/outputs/candidate' / case_id
    retained = CAMPAIGN / 'inputs/candidate' / case_id
    bound = spec['compiled_inputs']['candidate/' + case_id]
    for filename, key in (('generation.json', 'generation_sha256'), ('compilation.json', 'compilation_sha256')):
        require(digest(export / filename) == digest(retained / filename) == bound[key],
                'Donor build receipt differs: ' + case_id + '/' + filename)
    generated, compiled = read(export / 'generation.json'), read(export / 'compilation.json')
    require(generated['source'] == 'candidate'
            and generated['source_identity']['files_sha256'] == GENERATOR_SHA
            and generated['source_identity']['revision'] == REVISION
            and compiled['full_numerical_rebuild_this_invocation'] is True,
            'Donor is not the registered full current-source build')
    for name, checksum in compiled['sources'].items():
        require(digest(export / name) == checksum, 'Generated scientific source differs: ' + name)
    directories = [path.parent for path in export.glob('SubProcesses/P*/check_sa.f')]
    require(len(directories) == 1, 'Exactly one generated subprocess is required')
    directory = directories[0]
    require(digest(directory / 'check') == compiled['executable_sha256'] == bound['executable_sha256'],
            'Donor executable changed since qualification')
    require(digest(export / 'Cards/param_card.dat') == bound['param_card_sha256']
            and digest(retained / 'param_card.dat') == bound['param_card_sha256'],
            'Physical parameter card differs')
    for section in bound['runtime'].values():
        for name, checksum in section.items():
            require(digest(name) == checksum, 'Registered dependency changed: ' + name)
    numerical = CAMPAIGN / 'experiments/numerical' / ('full100-warmup-' + case_id + '-1')
    result = read(numerical / 'results.json')
    require(result['case']['id'] == case_id
            and [(lane['source'], lane['backend']) for lane in result['results']]
            == [('baseline', 1)] + [('candidate', backend) for backend in BACKENDS]
            and all(lane['status'] == 'complete' for lane in result['results']),
            'Donor case does not have all six closed completed lanes')
    points = CAMPAIGN / 'points' / (case_id + '.dat')
    require(digest(points) == bound['points_sha256']
            and digest(points.with_suffix('.json')) == bound['points_metadata_sha256'], 'Registered points changed')
    metadata = read(points.with_suffix('.json'))
    require(metadata['inputs']['incoming_masses'] + metadata['inputs']['masses'] == compiled['masses'],
            'Saved physical mass inputs differ from the compiled process mass probe')
    require(metadata['inputs']['count'] == 100 and len(compiled['masses'])
            == compiled['worker_render_parameters']['nexternal'], 'Unexpected mass probe dimensions')
    return {'export': str(export), 'subprocess': str(directory.relative_to(export)),
            'compiled': compiled, 'original_points': str(points), 'mass_inputs': metadata['inputs'],
            'physical_card_sha256': bound['param_card_sha256'], 'runtime': bound['runtime'],
            'closed_case_receipt': {'path': str(numerical / 'results.json'), 'sha256': digest(numerical / 'results.json')}}


def link_audit(links, snapshot):
    """Freeze finite directory envelopes, including legitimate directory links.

    This is a present setup snapshot. The original build's used libraries and
    metadata are independently matched to its historical hashes in donor().
    No newly discovered ambient member is described as historically qualified.
    """
    roots = {}; leaves = {}; observations = {}; pending = [snapshot]
    while pending:
        inventory = pending.pop()
        for item in inventory['entries'].values():
            if item['kind'] != 'symlink':
                continue
            target = Path(item['resolved'])
            for hop in item['symlink_chain']:
                if not Path(hop['logical_path']).is_relative_to(Path(snapshot['root'])):
                    observations[hop['logical_path']] = links._describe(Path(hop['logical_path']))
            if target.is_relative_to(Path(snapshot['root'])):
                continue
            if item['resolved_type'] == 'directory':
                if str(target) not in roots:
                    require(len(roots) < 32, 'External directory envelope limit exceeded')
                    roots[str(target)] = links.tree_snapshot(target)
                    require(sum(len(record['entries']) for record in roots.values()) <= 20000,
                            'External graph membership limit exceeded')
                    pending.append(roots[str(target)])
            else:
                leaves[str(target)] = {'all_recorded_bindings_match': True,
                    'observed_sha256': item['sha256'], 'references': [{'sha256': item['sha256'],
                    'meaning': 'actual pre-copy setup freeze; not historical qualification'}]}
    audit = {'schema': 'cms-replay-link-closure-audit-v1',
             'external_directory_envelopes': roots, 'external_directory_alias_target_envelopes': {},
             'historical_leaf_bindings': leaves, 'logical_bound_leaf_observations': observations,
             'transitive_symlink_nodes': {}, 'scope': 'current finite setup closure; not an amplitude qualification'}
    links.verify_external_graph(audit)
    return audit


def environment(cwd):
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith('LD_') or key in ('GLIBC_TUNABLES', 'GCONV_PATH', 'LOCPATH', 'NIX_LD', 'NIX_LD_LIBRARY_PATH'):
            env.pop(key, None)
    for key in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'FFLAGS', 'FCFLAGS', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS',
                'GLOBAL_FLAG', 'MACFLAG', 'STDLIB_FLAG', 'MFLAGS', 'MAKEOVERRIDES', 'MAKELEVEL'):
        env.pop(key, None)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
        env[key] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['LC_ALL'] = 'C'
    env['PWD'] = str(cwd)
    env['MAKEFLAGS'] = '-j1'
    for key in ('XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME', 'MPLCONFIGDIR'):
        env[key] = str(WORK / 'environment' / key.lower())
    return env


@contextmanager
def signal_requests(cleanup):
    """Extend the immutable owner helper's INT/TERM handling to HUP locally."""
    with cleanup.SignalRequests() as requests_guard:
        old = signal.getsignal(signal.SIGHUP)
        signal.signal(signal.SIGHUP, lambda number, frame: requests_guard.requested.append(number))
        try:
            yield requests_guard
        finally:
            signal.signal(signal.SIGHUP, old)


def native_environment(manifest):
    """Set native library search only immediately before target execve.

    The compiler, taskset, and bound Python gate run in a clean environment;
    older native runtimes must not contaminate newer Nix utility processes.
    """
    paths = [ROOT / 'HEPTools/ninja/lib', ROOT / 'HEPTools/collier',
             ROOT / 'HEPTools/ninjars/lib', Path(manifest['candidate']['native']['prefix']) / 'lib']
    return {'LD_LIBRARY_PATH': os.pathsep.join(map(str, paths))}


def command(argv, cwd, evidence, seconds, cleanup, requests_guard, memory=4294967296,
            target_environment=None):
    """Own one bounded process tree; preserve its full merged console and maps."""
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    process = owned = failure = closed = None
    maps = []; seen = set(); useful_saved = False
    def child_limits():
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    target_argv = list(map(str, argv))
    target_environment = dict(target_environment or {})
    require(set(target_environment).issubset({'LD_LIBRARY_PATH'}), 'Unexpected target environment override')
    payload = {'argv': target_argv, 'environment': target_environment}
    argv = ['taskset', '-c', '12', sys.executable, '-I', '-B', '-c', EXEC_TARGET,
            json.dumps(payload, sort_keys=True)]
    application = str(Path(target_argv[0]).resolve()) if Path(target_argv[0]).name == 'stability-check' else None
    try:
        requests_guard.check()
        with (evidence / 'stdout.txt').open('xb') as stream:
            process = subprocess.Popen(argv, cwd=cwd, env=environment(cwd), stdout=stream,
                                       stderr=subprocess.STDOUT, stdin=subprocess.PIPE, start_new_session=True,
                                       preexec_fn=child_limits)
            owned = cleanup.OwnedChild(process)
            # The isolated Python child cannot launch make or a numerical
            # target until its exact process/session identity is owned.
            requests_guard.check()
            process.stdin.write(b'G'); process.stdin.flush(); process.stdin.close()
            while process.poll() is None:
                owned.sample(); requests_guard.check()
                if time.monotonic() - started >= seconds - 10:
                    raise TimeoutError('Native/compile work deadline; cleanup reserve begins')
                try:
                    raw = Path('/proc/%d/maps' % process.pid).read_bytes()
                    checksum = hashlib.sha256(raw).hexdigest()
                    names = [line.split(maxsplit=5)[-1] for line in raw.decode(errors='replace').splitlines()
                             if len(line.split(maxsplit=5)) == 6]
                    useful = application in names and any(Path(name).name == 'libmadloop_tirust.so' for name in names)
                    if checksum not in seen and (len(maps) < 16 or (useful and not useful_saved)):
                        name = 'runtime-maps-%02d.txt' % len(maps)
                        (evidence / name).write_bytes(raw)
                        maps.append({'path': name, 'sha256': checksum}); seen.add(checksum)
                        useful_saved = useful_saved or useful
                except (FileNotFoundError, ProcessLookupError):
                    pass
                try:
                    process.wait(timeout=.1)
                except subprocess.TimeoutExpired:
                    pass
            require(process.returncode == 0, 'Child returned ' + str(process.returncode))
    except BaseException as error:
        failure = {'class': type(error).__name__, 'message': str(error)}
    finally:
        try:
            if process is not None and process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if owned is not None:
                closed = owned.stop(cooperative_seconds=0, terminate_seconds=4, kill_seconds=4)
                require(not closed['survivors'], 'Owned descendant survived cleanup')
            elif process is not None:
                process.kill(); process.wait(timeout=4)
        except BaseException as error:
            failure = failure or {'class': type(error).__name__, 'message': str(error)}
        elapsed = time.monotonic() - started
        if elapsed > seconds:
            failure = failure or {'class': 'TimeoutError', 'message': 'Inclusive wall cap exceeded including terminal cleanup'}
        record = {'argv': argv, 'target_argv': target_argv, 'target_environment': target_environment,
                  'exec_gate_source_sha256': hashlib.sha256(EXEC_TARGET.encode()).hexdigest(),
                  'setup_environment': {key: value for key, value in environment(cwd).items()
                                        if key in ('PATH', 'LC_ALL', 'NIX_CFLAGS_COMPILE', 'NIX_LDFLAGS')
                                        or key.startswith('LD_')},
                  'cwd': str(cwd), 'elapsed_seconds': elapsed,
                  'wall_cap_seconds': seconds, 'memory_bytes': memory, 'cpu': 12, 'cpu_exclusive': False,
                  'process_started': process is not None,
                  'returncode': None if process is None else process.returncode,
                  'error': failure, 'cleanup': closed, 'runtime_map_samples': maps,
                  'runtime_map_capture': 'first sixteen distinct samples plus first useful application-and-DSO sample if needed',
                  'stdout_sha256': digest(evidence / 'stdout.txt') if (evidence / 'stdout.txt').exists() else None}
        save(evidence / 'command.json', record)
    return record


def render(template, parameters):
    for token, key in (('@PREFIX@', 'prefix'), ('@NEXTERNAL@', 'nexternal'), ('@NBORN@', 'nborn')):
        template = template.replace(token, str(parameters[key]))
    require(not re.search(r'@[A-Z_]+@', template), 'Unexpanded worker placeholder')
    return template


def elf_interpreter(path):
    """Read the qualified Linux ELF64 PT_INTERP without launching a utility."""
    with Path(path).open('rb') as stream:
        header = stream.read(64)
        require(len(header) == 64 and header[:6] == b'\x7fELF\x02\x01', 'Expected ELF64 little-endian executable')
        offset = struct.unpack_from('<Q', header, 32)[0]
        size, count = struct.unpack_from('<HH', header, 54)
        require(size >= 56 and 0 < count <= 1024, 'Invalid ELF program header table')
        found = []
        for index in range(count):
            stream.seek(offset + index * size); entry = stream.read(56)
            require(len(entry) == 56, 'Truncated ELF program header')
            if struct.unpack_from('<I', entry)[0] != 3: continue
            position, length = struct.unpack_from('<Q', entry, 8)[0], struct.unpack_from('<Q', entry, 32)[0]
            require(1 < length <= 4096, 'Invalid ELF interpreter length')
            stream.seek(position); name = stream.read(length)
            require(len(name) == length and name.endswith(b'\0'), 'Truncated ELF interpreter')
            found.append(Path(name[:-1].decode()))
    require(len(found) == 1 and found[0].is_absolute(), 'Unique absolute ELF interpreter required')
    return found[0].resolve()


def qualify_loader_closure(executable, donor_executable, historical, listing):
    """Distinguish the target interpreter from a historical host-ldd artifact."""
    loader = elf_interpreter(executable)
    require(loader == elf_interpreter(donor_executable), 'Private driver changed the target interpreter')
    expected = {str(Path(path).resolve()): checksum for path, checksum in historical.items()}
    require(expected.get(str(loader)) == digest(loader), 'Actual target loader is not historically bound')
    require('not found' not in listing.lower(), 'Target loader reported a missing dependency')
    paths = {str(Path(path).resolve()) for path in re.findall(r'(/[^\s()]+)', listing)
             if Path(path).is_file()}
    paths.discard(str(Path(executable).resolve()))
    paths.add(str(loader))
    require(paths.issubset(expected), 'Fresh target closure has an unregistered library')
    excluded = set(expected) - paths
    exclusions = {}
    if excluded:
        host = Path(HISTORICAL_LDD_HOST_LOADER['path']).resolve()
        require(host != loader and excluded == {str(host)}
                and host.name == 'ld-linux-x86-64.so.2'
                and expected[str(host)] == HISTORICAL_LDD_HOST_LOADER['sha256'] == digest(host),
                'Historical missing dependency is not the verified non-target host loader')
        exclusions[str(host)] = {'sha256': expected[str(host)],
            'binding': HISTORICAL_LDD_HOST_LOADER['binding'],
            'reason': 'exact historically bound ldd host loader; distinct from actual target PT_INTERP and current Python'}
    require(any(Path(path).name == 'libmadloop_tirust.so' for path in paths), 'Production TIRust DSO absent from target closure')
    for path in paths:
        require(digest(path) == expected[path], 'Fresh target dependency changed: ' + path)
    return {'loader': str(loader), 'loader_sha256': digest(loader), 'listing': listing,
            'files': {path: expected[path] for path in sorted(paths)},
            'historical_libraries': expected, 'verified_historical_host_loader_exclusions': exclusions}


def logical_fortran(text):
    statements = []
    for line in text.splitlines():
        if not line.strip() or line[0] in 'Cc*!': continue
        require(len(line) >= 6, 'Malformed fixed-format Fortran source')
        body = line[6:].split('!', 1)[0].strip()
        if line[5] not in (' ', '0'):
            require(statements, 'Fortran continuation without statement')
            statements[-1] += ' ' + body
        else:
            statements.append(body)
    return statements


def group_layout(directory, expected_prefix=None):
    """Derive active tensor coefficient extents from canonical generated calls."""
    directory = Path(directory)
    statements = [re.sub(r'\s+', '', item.upper())
                  for item in logical_fortran((directory / 'loop_matrix.f').read_text())]
    matrix = ' '.join(logical_fortran((directory / 'loop_matrix.f').read_text())).upper()
    def integer(name, text):
        matches = {int(item) for item in re.findall(r'\b' + name + r'\s*=\s*(\d+)\b', text)}
        require(len(matches) == 1, 'Ambiguous generated ' + name)
        return matches.pop()
    norders, ngroups = integer('NSQUAREDSO', matrix), integer('NLOOPGROUPS', matrix)
    prefixes = {match.group(1) for statement in statements
                if (match := re.fullmatch(r'COMMON/(\w+)LCOEFS/LOOPCOEFS', statement))}
    require(len(prefixes) == 1, 'Unique actual process prefix/common coefficient layout required')
    prefix = prefixes.pop()
    require(expected_prefix is None or prefix == expected_prefix.upper(), 'Compiled process prefix/common layout differs')
    blocks = {
        'LCOEFS': 'LOOPCOEFS',
        'LOOPRES': 'LOOPRES,S',
        'SO_REQS': 'UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO',
        'SOCHOICE': 'SQSO_TARGET',
        'FPE_IN_REDUCTION': 'FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION',
    }
    declarations = {
        'COMPLEX*16LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)',
        'COMPLEX*16LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)',
        'LOGICALS(NSQUAREDSO,NLOOPGROUPS)',
        'LOGICAL' + blocks['SO_REQS'], 'INTEGERSQSO_TARGET',
        'LOGICALFPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION',
        'DATAFPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION/.FALSE.,.FALSE./',
    }
    declarations.update('DATA' + name + '/.FALSE./' for name in blocks['SO_REQS'].split(','))
    require(declarations.issubset(statements), 'COMMON observation type/extent/DATA initialization is not established')
    for name, members in blocks.items():
        start = 'COMMON/' + prefix + name + '/'
        observed = [statement for statement in statements if statement.startswith(start)]
        require(observed and all(statement == start + members for statement in observed),
                'COMMON member order/layout differs: ' + name)
    compact = re.sub(r'\s+', '', matrix)
    facts = {
        'finite_reduction_enabled': 'PARAMETER(SKIPLOOPEVAL=.FALSE.)',
        'default_all_split_orders': 'DATASQSO_TARGET/-1/',
        'coefficient_initialization': 'DOI=1,NLOOPGROUPSDOJ=0,LOOPMAXCOEFS-1DOK=1,NSQUAREDSOLOOPCOEFS(J,K,I)=(0.0D0,0.0D0)ENDDOENDDOENDDO',
        'result_initialization': 'DOI=1,NLOOPGROUPSDOJ=1,3DOK=1,NSQUAREDSOLOOPRES(J,K,I)=(0.0D0,0.0D0)ENDDOENDDOENDDO',
        'stability_initialization': 'DOI_SO=1,NSQUAREDSODOJ=1,NLOOPGROUPSS(I_SO,J)=.TRUE.ENDDO',
    }
    require(all(fragment in compact for fragment in facts.values()),
            'Canonical initialized Born loop layout prerequisites are not established')
    require(compact.index(facts['coefficient_initialization']) < compact.index(facts['stability_initialization'])
            and compact.index(facts['result_initialization']) < compact.index(facts['stability_initialization']),
            'Loop array initialization must precede initialized stability flags and reduction')
    born_text = ' '.join(logical_fortran((directory / 'nsqso_born.inc').read_text())).upper()
    require(integer('NSQSO_BORN', born_text) > 0 and integer('NBORNAMPS', matrix) > 0,
            'Only qualified Born-process coefficient layouts are supported')
    maximum = integer('LOOPMAXCOEFS', ' '.join(logical_fortran((directory / 'loop_max_coefs.inc').read_text())).upper())
    require(norders > 0 and ngroups > 0 and maximum > 0, 'Unsupported empty loop layout')
    ranks = {}; inputs = {}
    for path in sorted(directory.glob('loop_CT_calls_*.f')):
        inputs[path.name] = digest(path)
        for statement in logical_fortran(path.read_text()):
            if not re.match(r'CALL\s+\w*LOOP_\d+(?:_\d+)*\s*\(', statement, re.I): continue
            match = re.search(r',\s*(\d+)\s*,\s*I_SO\s*,\s*(\d+)\s*\)\s*$', statement, re.I)
            require(match is not None, 'Unsupported nonconstant loop rank/group call')
            rank, group = map(int, match.groups())
            require(1 <= group <= ngroups and 0 <= rank <= 6, 'Unqualified loop group/rank')
            require(group not in ranks or ranks[group] == rank, 'Inconsistent rank for repeated loop group')
            ranks[group] = rank
    require(set(ranks) == set(range(1, ngroups + 1)), 'Incomplete generated loop group mapping')
    counts = [math.comb(ranks[group] + 4, 4) for group in range(1, ngroups + 1)]
    require(max(counts) <= maximum, 'Tensor coefficient extent exceeds generated storage')
    lines = ['      INTEGER NSQUAREDSO,NLOOPGROUPS',
             '      PARAMETER(NSQUAREDSO=%d,NLOOPGROUPS=%d)' % (norders, ngroups),
             '      INTEGER STABILITY_NCOEFS(NLOOPGROUPS)', '      DATA STABILITY_NCOEFS /']
    for start in range(0, len(counts), 12):
        suffix = '/' if start + 12 >= len(counts) else ','
        lines.append('     $ ' + ','.join(map(str, counts[start:start + 12])) + suffix)
    return {'nsquaredso': norders, 'nloopgroups': ngroups, 'loopmaxcoefs': maximum,
            'prefix': prefix, 'common_member_layouts': blocks,
            'common_declarations_and_static_initialization_verified': True,
            'layout_source_hashes': {name: digest(directory / name)
                for name in ('loop_matrix.f', 'loop_max_coefs.inc', 'nsqso_born.inc')},
            'ranks': [ranks[group] for group in range(1, ngroups + 1)], 'ncoefs': counts,
            'mapping_sources': inputs, 'source_prerequisites': {name: True for name in facts},
            'include': '\n'.join(lines) + '\n'}


def private_changes(links, output, context, changed, allowed_additions=()):
    """Allow only declared local products; no new symlink or external write."""
    observed = links.tree_snapshot(output)
    additions = {name: item['kind'] for name, item in observed['entries'].items()
                 if name not in context['expected_worker']['entries']}
    for name, kind in additions.items():
        require(kind == 'regular_file' and (name in allowed_additions
                or ('MadLoop5_resources' in Path(name).parts
                    and Path(name).name.endswith(('HelFilter.dat', 'LoopFilter.dat')))),
                'Unadmitted private addition: ' + name)
        links.assert_private_path(output, output / name)
    links.assert_worker_inputs(output, context, allow_file_changes=changed,
                              allow_new_paths={str(output / name): kind for name, kind in additions.items()})
    return observed


def prepare(value, helpers):
    source_index = verify_seal()
    require(not WORK.exists() and not WORK.is_symlink(), 'Fresh pilot workspace required')
    links = helpers['links']; records = {}
    destination, unused_kind, chain = links._resolve(WORK, missing=True)
    require(not chain and destination == WORK, 'Private work path traverses an ancestor alias')
    WORK.mkdir(parents=True); REPORT.mkdir()
    save(REPORT / 'prepare-started.json', {'source_index_sha256': source_index,
         'protocol_sha256': digest(HERE / 'protocol.json'), 'request_count': 20, 'amplitude_evaluations': 0})
    failure = None
    try:
        manifest, spec = verify_production(value)
        # Admit every donor before point generation, copies, or any child launch.
        donors = {case: donor(value, spec, case) for case in CASE_IDS}
        with signal_requests(helpers['cleanup']) as signals:
            for case, data in donors.items():
                inputs = data['mass_inputs']
                prefix = WORK / 'points' / case
                sample = helpers['points'].freeze_cut_points(prefix, masses=inputs['masses'],
                    incoming_masses=inputs['incoming_masses'], card_sha256=data['physical_card_sha256'],
                    seed=value['sampling']['seeds'][case], count=100, energy=1000.)
                require(check_points(value, prefix.with_suffix('.dat'),
                                     data['compiled']['worker_render_parameters']['nexternal'])['passed'] == 100,
                        'Fresh pilot stream fails the declared cuts')
                data = dict(data, points=str(prefix.with_suffix('.dat')),
                            points_sha256=sample['sha256'], point_metadata_sha256=digest(prefix.with_suffix('.json')))
                original = Path(data['export']); target = WORK / 'masters' / case
                snapshot = links.tree_snapshot(original); audit = link_audit(links, snapshot)
                save(REPORT / 'prepare-inputs' / (case + '.json'), {'donor_snapshot': snapshot,
                     'audit': audit, 'point_sha256': data['points_sha256'], 'donor': data})
                context = links.prepare_worker(original, target, snapshot, audit)
                directory = target / data['subprocess']
                layout = group_layout(directory, data['compiled']['worker_render_parameters']['prefix'])
                for name in ('check_sa.f', 'check_sa.o', 'pilot.mk', 'stability-check', 'stability-layout.inc'):
                    links.assert_private_path(target, directory / name)
                (directory / 'check_sa.f').write_text(render((HERE / 'stability_worker.f').read_text(),
                                                            data['compiled']['worker_render_parameters']))
                (directory / 'pilot.mk').write_text(MAKEFILE)
                (directory / 'stability-layout.inc').write_text(layout['include'])
                compiler = data['compiled']['effective_compiler_flags']['scopes']['process']['recipes']['check_sa.f']['compiler']
                require(len(compiler) == 1 and digest(compiler[0]['path']) == compiler[0]['sha256'],
                        'The recorded GNU Fortran compiler differs')
                shell = shutil.which('bash', path=environment(directory)['PATH'])
                require(shell is not None, 'Pinned environment Bash is missing')
                record = command(['make', '--no-print-directory', '-j1', '-f', 'pilot.mk', 'pilot-build',
                                  'FC=' + compiler[0]['path'], 'SHELL=' + str(Path(shell).resolve())], directory,
                                 REPORT / 'prepare' / case, 600, helpers['cleanup'], signals)
                changed = [directory / 'check_sa.f', directory / 'check_sa.o']
                allowed = [str((directory / name).relative_to(target))
                           for name in ('pilot.mk', 'stability-check', 'stability-layout.inc')]
                private = private_changes(links, target, context, changed, allowed)
                require(record['error'] is None and (directory / 'stability-check').is_file(),
                        'Private driver relink failed; no numerical rebuild fallback')
                executable = directory / 'stability-check'
                loader = elf_interpreter(executable)
                runtime_command = command([loader, '--list', executable], directory,
                    REPORT / 'prepare-loader' / case, 60, helpers['cleanup'], signals,
                    target_environment=native_environment(manifest))
                require(runtime_command['error'] is None, 'Actual PT_INTERP dependency preflight failed')
                runtime = qualify_loader_closure(executable,
                    Path(data['export']) / data['subprocess'] / 'check', data['runtime']['libraries'],
                    (REPORT / 'prepare-loader' / case / 'stdout.txt').read_text())
                private_changes(links, target, context, changed, allowed)
                records[case] = {'donor': data, 'donor_context': context,
                    'master_snapshot': private, 'executable_sha256': digest(directory / 'stability-check'),
                    'command': record, 'driver_sha256': digest(directory / 'check_sa.f'), 'group_layout': layout,
                    'runtime_closure': runtime, 'runtime_command': runtime_command,
                    'native_environment': native_environment(manifest)}
                save(REPORT / 'prepare' / case / 'checkpoint.json', records[case])
        verify_production(value)
    except BaseException as error:
        failure = {'class': type(error).__name__, 'message': str(error)}
    finally:
        save(REPORT / 'prepare-terminal.json', {'status': 'failed' if failure else 'passed',
            'source_index_sha256': source_index, 'protocol_sha256': digest(HERE / 'protocol.json'),
            'error': failure, 'completed_cases': list(records), 'cases': records, 'amplitude_evaluations': 0,
            'requests': [{'id': request['id'], 'case': request['case'], 'backend': request['backend'],
                          'state': 'unexecuted', 'primary_calls_expected': 300} for request in requests(value)]})
    require(failure is None, 'Pilot preparation failed: ' + str(failure))


@contextmanager
def numerical_lock(value, request, signals):
    if not request['lock_required']:
        yield {'required': False, 'wait_seconds': 0.}
        return
    path = ROOT / value['limits']['common_lock']
    started = time.monotonic()
    with path.open('a+') as stream:
        while True:
            signals.check()
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= value['limits']['lock_wait_seconds']:
                    raise TimeoutError('Common numerical lock wait expired before native launch')
                time.sleep(.2)
        try:
            yield {'required': True, 'path': str(path), 'wait_seconds': time.monotonic() - started,
                   'execution_cap_starts_after_lock': True}
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def verify_card_tokens(text, settings):
    """Independent readback of explicitly written Fortran card values."""
    values = {}; current = None
    for line in text.splitlines():
        token = line.split('!', 1)[0].strip()
        if not token: continue
        if token.startswith('#'):
            current = token[1:].strip().lower()
            require(current and current not in values, 'Duplicate or empty numerical card header')
            values[current] = None
        else:
            require(current is not None and values[current] is None, 'Unexpected extra numerical card value')
            values[current] = token
            current = None
    observed = {}
    for name, expected in settings.items():
        token = values.get(name.lower())
        require(token is not None, 'Explicit numerical card value is missing: ' + name)
        if isinstance(expected, bool):
            require(token.upper() in ('.TRUE.', '.FALSE.'), 'Malformed logical card value: ' + name)
            actual = token.upper() == '.TRUE.'
        elif isinstance(expected, int):
            require(re.fullmatch('[+-]?[0-9]+', token) is not None, 'Malformed integer card value: ' + name)
            actual = int(token)
        elif isinstance(expected, float):
            actual = float(token.replace('D', 'E').replace('d', 'e'))
            require(math.isfinite(actual), 'Nonfinite numerical card control')
        else:
            actual = token
        require(actual == expected, 'Numerical card roundtrip differs: ' + name)
        observed[name] = actual
    return observed


def configure_card(links, output, backend):
    sys.path.insert(0, str(ROOT))
    from madgraph.various.banner import MadLoopParam
    path = output / 'Cards/MadLoopParams.dat'
    links.assert_private_path(output, path)
    card = MadLoopParam(str(path))
    settings = {'MLReductionLib': str(backend), 'CTModeRun': 1, 'CTModeInit': 1,
                'LoopPoleMode': -1, 'TIRustComputeFinite': True, 'UseQPIntegrandForTIRust': True,
                'COLLIERComputeUVpoles': True, 'COLLIERComputeIRpoles': True,
                'COLLIERGlobalCache': 0, 'COLLIERUseInternalStabilityTest': False,
                'COLLIERRequiredAccuracy': 1e-13, 'TIRustGlobalCache': 0}
    for key, item in settings.items():
        require(key.lower() in {str(name).lower() for name in card}, 'Missing required card parameter: ' + key)
        card[key] = item
    template = ROOT / 'Template/loop_material/StandAlone/Cards/MadLoopParams.dat'
    destinations = {path} | set((output / 'SubProcesses').rglob('MadLoopParams.dat'))
    for destination in destinations:
        links.assert_private_path(output, destination)
        card.write(str(destination), template=str(template))
    parsed = {str(path.relative_to(output)): verify_card_tokens(path.read_text(), settings) for path in destinations}
    return {'settings': settings, 'readback': parsed,
            'files': {str(path.relative_to(output)): digest(path) for path in destinations}}


def parse_groups(raw, layout):
    """Validate full initialized coefficient coverage without interpreting FP bits.

    Raw hex words are retained in the journal. Summaries keep every call and
    invalid group; integer TRANSFER observations are not an accuracy estimate.
    """
    groups = {}; calls = {}; before = {}; issues = []; total = 0; complete = None; header = None
    for number, line in enumerate(raw.splitlines(), 1):
        if line.startswith('# GROUP_SCHEMA '):
            try:
                require(header is None, 'Duplicate group journal header')
                header = list(map(int, line.split()[2:]))
                require(header == [1, layout['nsquaredso'], layout['nloopgroups']], 'Group journal header/layout differs')
            except ValueError as error:
                issues.append({'line': number, 'raw': line, 'error': str(error)})
            continue
        if line.startswith('# COMPLETE '):
            try: complete = list(map(int, line.split()[2:]))
            except ValueError: issues.append({'line': number, 'error': 'Malformed group COMPLETE'})
            continue
        if not line.strip() or line.lstrip().startswith('#'): continue
        try:
            fields = line.split()
            require(fields[0] in ('B', 'G', 'C'), 'Unknown group journal record')
            if fields[0] == 'B':
                require(len(fields) == 7, 'Before-call record needs six integers')
                phase, call, point, method, dp, qp = map(int, fields[1:])
                key = (phase, call, point, method)
                require(key not in before and key not in calls and key not in groups
                        and dp in (0, 1) and qp in (0, 1), 'Invalid/duplicate/late before-call flags')
                before[key] = (dp, qp)
            elif fields[0] == 'G':
                require(len(fields) == 20, 'Group record needs eleven integers and eight hex words')
                phase, call, point, method, order, group, ncoefs, nz, badcoef, stable, badres = map(int, fields[1:12])
                key = (phase, call, point, method)
                require(key in before, 'Group row precedes its before-call observation')
                require(phase in (0, 1) and call > 0 and 1 <= point <= 100
                        and ((phase == 0 and method == 0) or (phase == 1 and method in (1, 2, 3))),
                        'Invalid group call identity')
                require(1 <= order <= layout['nsquaredso'] and 1 <= group <= layout['nloopgroups']
                        and ncoefs == layout['ncoefs'][group - 1], 'Group coefficient coverage differs from canonical layout')
                require(0 <= nz <= 2 * ncoefs and 0 <= badcoef <= nz and 0 <= badres <= 6,
                        'Invalid coefficient/result bit count')
                require(all(re.fullmatch('[0-9A-Fa-f]{16}', word) for word in fields[12:]),
                        'Expected eight exact 64-bit words')
                active = nz > 0 or badcoef > 0
                require((active and stable in (0, 1)) or (not active and stable == -1 and badres == 0),
                        'Inactive/stability/read-result contract differs')
                require(active or all(int(word, 16) == 0 for word in fields[12:]),
                        'Inactive group must not expose uninitialized result/first-coefficient words')
                words = [int(word, 16) for word in fields[12:]]
                first_nonzero = sum((word & 0x7fffffffffffffff) != 0 for word in words[:2])
                first_nonfinite = sum(((word >> 52) & 0x7ff) == 0x7ff for word in words[:2])
                result_nonfinite = sum(((word >> 52) & 0x7ff) == 0x7ff for word in words[2:])
                require((not active or first_nonzero > 0) and first_nonzero <= nz and first_nonfinite <= badcoef
                        and result_nonfinite == badres, 'Retained binary words disagree with reported component counts')
                table = groups.setdefault(key, {})
                require((order, group) not in table and key not in calls, 'Duplicate or late group row')
                table[(order, group)] = {'order': order, 'group': group, 'active': active,
                                        'stable': stable, 'bad_coefficients': badcoef, 'bad_results': badres}
                total += 1
            else:
                require(len(fields) == 12, 'Coverage record needs eleven integers')
                phase, call, point, method, coverage, nactive, nfailed, ncoefbad, nresbad, fpe_dp, fpe_qp = map(int, fields[1:])
                key = (phase, call, point, method)
                require(key in before and key not in calls and coverage in (0, 1) and fpe_dp in (0, 1) and fpe_qp in (0, 1),
                        'Invalid/duplicate coverage summary')
                table = groups.pop(key, {})
                expected = {(order, group) for order in range(1, layout['nsquaredso'] + 1)
                            for group in range(1, layout['nloopgroups'] + 1)}
                derived = [sum(item['active'] for item in table.values()),
                           sum(item['active'] and item['stable'] == 0 for item in table.values()),
                           sum(item['bad_coefficients'] > 0 for item in table.values()),
                           sum(item['active'] and item['bad_results'] > 0 for item in table.values())]
                require(derived == [nactive, nfailed, ncoefbad, nresbad], 'Coverage summary does not match group observations')
                require((coverage == 1 and set(table) == expected) or (coverage == 0 and not table),
                        'Coverage is incomplete or unavailable call unexpectedly reads groups')
                reasons = []
                if not coverage: reasons.append('Initialized loop group coverage unavailable')
                if nfailed: reasons.append('Active loop group reported numerical failure')
                if ncoefbad or nresbad: reasons.append('Nonfinite tensor coefficient or loop result bits')
                if fpe_dp or fpe_qp or any(before[key]): reasons.append('Sticky or newly set MadLoop floating-point failure latch')
                calls[key] = {'phase': phase, 'call': call, 'point': point, 'method': method,
                    'coverage': bool(coverage), 'active_groups': nactive, 'failed_groups': nfailed,
                    'nonfinite_coefficient_groups': ncoefbad, 'nonfinite_result_groups': nresbad,
                    'fpe_dp': bool(fpe_dp), 'fpe_qp': bool(fpe_qp), 'eligible': not reasons,
                    'fpe_dp_before': bool(before[key][0]), 'fpe_qp_before': bool(before[key][1]),
                    'fpe_dp_became_set': not bool(before[key][0]) and bool(fpe_dp),
                    'fpe_qp_became_set': not bool(before[key][1]) and bool(fpe_qp),
                    'reason': '; '.join(reasons) or None,
                    'invalid_groups': [item for item in table.values() if item['bad_coefficients']
                                       or item['bad_results'] or (item['active'] and item['stable'] == 0)]}
        except (ValueError, OverflowError) as error:
            issues.append({'line': number, 'raw': line, 'error': str(error)})
    if groups: issues.append({'error': 'Group observations lack their closing coverage record', 'calls': list(groups)})
    if header is None: issues.append({'error': 'Missing group schema/layout header'})
    if set(before) != set(calls):
        issues.append({'error': 'Before-call observations lack their closing coverage record',
                       'calls': list(set(before) - set(calls))})
    return {'calls': calls, 'parse_issues': issues, 'group_row_count': total, 'complete_record': complete}


def parse_diagnostics(dimensions, raw, console, group_raw='', layout=None):
    """Accept unchecked forced-DP data only as internal consistency diagnostics.

    No returned host accuracy, positive 110 status, or backend request is a
    checked accuracy certificate. Unexpected QP output remains in the raw rows.
    """
    fields = dimensions.split()
    require(len(fields) == 5, 'Invalid stability dimensions header')
    schema, count, ndim, nso, nborn = map(int, fields)
    require(schema == 1 and count == 100 and ndim >= 1 and 0 <= nso <= ndim and 0 <= nborn <= ndim,
            'Unexpected stability dimensions')
    require(ndim == max(nso, nborn) and (layout is None or nso == layout['nsquaredso']),
            'Stability dimensions differ from frozen initialized loop layout')
    group_data = parse_groups(group_raw, layout) if layout is not None else {
        'calls': {}, 'parse_issues': [{'error': 'Canonical loop group layout unavailable'}],
        'group_row_count': 0, 'complete_record': None}
    active = None; warnings = set(); global_warning = False
    for line in console.splitlines():
        parts = line.split()
        if parts[:1] == ['STABILITY_CALL_BEGIN'] and len(parts) == 6:
            active = tuple(map(int, parts[1:5]))
        elif '##W03' in line:
            if active is None: global_warning = True
            else: warnings.add(active)
        elif parts[:1] == ['STABILITY_CALL_END']:
            active = None
    rows = []; issues = []; identities = set(); complete = None; bootstrap_ready = None
    for number, line in enumerate(raw.splitlines(), 1):
        if line.startswith('# BOOTSTRAP_COMPLETE '):
            try:
                require(bootstrap_ready is None, 'Duplicate bootstrap completion marker')
                marker = list(map(int, line.split()[2:]))
                require(len(marker) == 1 and 1 <= marker[0] <= 16, 'Malformed bootstrap completion count')
                bootstrap_ready = marker[0]
            except ValueError as error:
                issues.append({'line': number, 'raw': line, 'error': str(error)})
            continue
        if line.startswith('# COMPLETE '):
            try: complete = list(map(int, line.split()[2:]))
            except ValueError: issues.append({'line': number, 'error': 'Malformed COMPLETE record'})
            continue
        if not line.strip() or line.lstrip().startswith('#'): continue
        try:
            fields = line.split(); require(len(fields) == 11, 'Expected eleven diagnostic fields')
            phase, call, point, method, order, status = map(int, fields[:6])
            values = [float(item.replace('D', 'E').replace('d', 'e')) for item in fields[6:]]
            identity = (phase, call, point, method, order)
            require(identity not in identities, 'Duplicate diagnostic row')
            identities.add(identity)
            require(phase in (0, 1) and 1 <= point <= 100 and 0 <= order <= ndim,
                    'Diagnostic identity outside expected dimensions')
            require((phase == 0 and method == 0) or (phase == 1 and method in (1, 2, 3)),
                    'Unexpected phase/method')
            reasons = []
            if status != 110: reasons.append('Unexpected unchecked DP status; retained without DP qualification')
            if not all(math.isfinite(item) for item in values): reasons.append('Nonfinite coefficient or accuracy')
            if global_warning or (phase, call, point, method) in warnings: reasons.append('MadLoop ##W03 warning')
            coverage = group_data['calls'].get((phase, call, point, method))
            if coverage is None or group_data['parse_issues']:
                reasons.append('Native initialized loop group coverage unavailable or malformed')
            elif not coverage['eligible']:
                reasons.append(coverage['reason'])
            rows.append({'phase': phase, 'call': call, 'point': point, 'method': method,
                'order': order, 'status': status, 'coefficients': values[:4], 'accuracy': values[4],
                'raw_DP_diagnostic_eligible': phase == 1 and not reasons,
                'loop_group_coverage_verified': coverage is not None and coverage['eligible'] and not group_data['parse_issues'],
                'host_accuracy_qualified': False, 'backend_observation': 'not_available_in_unchecked_retcode',
                'reason': '; '.join(reasons) or None, 'line_number': number})
        except (ValueError, OverflowError) as error:
            issues.append({'line': number, 'raw': line, 'error': str(error)})
    grouped_rows = {}
    for row in rows:
        grouped_rows.setdefault((row['phase'], row['call'], row['point'], row['method']), []).append(row)
    for key, group in grouped_rows.items():
        uniform = len({row['status'] for row in group}) == 1
        complete_orders = [row['order'] for row in group] == list(range(ndim + 1))
        if not uniform or not complete_orders or (key[0] == 1 and not all(row['raw_DP_diagnostic_eligible'] for row in group)):
            for row in group:
                row['raw_DP_diagnostic_eligible'] = False
                row['reason'] = '; '.join(item for item in (row['reason'],
                    'Incomplete, inconsistent, or ineligible split-order rows within this call') if item)
    bootstrap_valid = bootstrap_ready is not None
    if bootstrap_valid:
        boot_groups = [group for key, group in grouped_rows.items() if key[0] == 0]
        bootstrap_valid = (len(boot_groups) == bootstrap_ready
            and [(group[0]['call'], [row['order'] for row in group], {row['status'] for row in group})
                 for group in boot_groups] == [(call, list(range(ndim + 1)), {110 if call == bootstrap_ready else 130})
                                              for call in range(1, bootstrap_ready + 1)])
    if not bootstrap_valid:
        for row in rows:
            if row['phase'] == 1:
                row['raw_DP_diagnostic_eligible'] = False
                row['reason'] = '; '.join(item for item in (row['reason'], 'Forced-DP bootstrap readiness is unqualified') if item)
    primary = [row for row in rows if row['phase'] == 1]
    outcomes = []
    for point in range(1, 101):
        for method in (1, 2, 3):
            selected = [row for row in primary if row['point'] == point and row['method'] == method]
            keys = [row['order'] for row in selected]
            eligible = keys == list(range(ndim + 1)) and all(row['raw_DP_diagnostic_eligible'] for row in selected)
            outcomes.append({'point': point, 'method': method,
                'state': 'diagnostic_available' if eligible else ('rejected' if selected else 'unexecuted'),
                'rows': len(selected), 'expected_rows': ndim + 1})
    structurally_complete = (not issues and bootstrap_valid and complete is not None and len(complete) == 2
        and 1 <= complete[0] <= 16 and complete[1] == 300
        and complete[0] == bootstrap_ready
        and len(primary) == 300 * (ndim + 1)
        and [(row['point'], row['method'], row['order']) for row in primary]
        == [(point, method, order) for point in range(1, 101) for method in (1, 2, 3) for order in range(ndim + 1)]
        and [row['call'] for row in primary]
        == [complete[0] + call for call in range(1, 301) for order in range(ndim + 1)]
        and [(row['phase'], row['call'], row['order']) for row in rows if row['phase'] == 0]
        == [(0, call, order) for call in range(1, complete[0] + 1) for order in range(ndim + 1)])
    call_keys = {(row['phase'], row['call'], row['point'], row['method']) for row in rows}
    groups_complete = (not group_data['parse_issues'] and set(group_data['calls']) == call_keys
                       and group_data['complete_record'] == complete and complete is not None)
    if not groups_complete:
        for row in rows:
            row['raw_DP_diagnostic_eligible'] = False
            row['loop_group_coverage_verified'] = False
        for outcome in outcomes:
            if outcome['state'] == 'diagnostic_available': outcome['state'] = 'rejected'
    return safe_json({'dimensions': {'ndim': ndim, 'nso': nso, 'nborn': nborn, 'points': count},
        'rows': rows, 'primary_call_outcomes': outcomes, 'parse_issues': issues,
        'complete_record': complete, 'structurally_complete': structurally_complete,
        'bootstrap_readiness_verified': bootstrap_valid,
        'loop_groups_structurally_complete': groups_complete,
        'loop_group_calls': list(group_data['calls'].values()), 'loop_group_parse_issues': group_data['parse_issues'],
        'loop_group_row_count': group_data['group_row_count'],
        'host_accuracy_qualified': False, 'warning_calls': sorted(warnings),
        'unattributed_warning': global_warning})


def runtime_binding(evidence, executable, libraries):
    """Authenticate sampled application mappings against the registered closure.

    Samples taken while taskset/loader still starts are retained but excluded.
    Absence of an authenticated application map makes this observation unusable.
    """
    expected = {str(Path(path).resolve()): checksum for path, checksum in libraries.items()}
    executable = str(Path(executable).resolve())
    expected[executable] = digest(executable)
    required = {executable}
    required.update(path for path in expected if Path(path).name == 'libmadloop_tirust.so')
    require(len(required) == 2, 'The production TIRust shared library must be in the registered runtime closure')
    hashed = {}; samples = []; application_seen = False
    for source in sorted(Path(evidence).glob('runtime-maps-*.txt')):
        raw = source.read_text()
        if not any(line.split(maxsplit=5)[-1] == executable for line in raw.splitlines()
                   if len(line.split(maxsplit=5)) == 6):
            continue
        application_seen = True; actual = {}; locale = {}
        for line in raw.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith('/'): continue
            spelling = fields[5]
            require(not spelling.endswith(' (deleted)'), 'Deleted mapped file')
            path = Path(spelling); resolved = str(path.resolve()); info = path.stat()
            require(int(fields[4]) == info.st_ino
                    and tuple(int(item, 16) for item in fields[3].split(':'))
                    == (os.major(info.st_dev), os.minor(info.st_dev)), 'Mapped inode/device differs')
            with path.open('rb') as stream: header = stream.read(4)
            if resolved not in hashed: hashed[resolved] = digest(path)
            if header == b'\x7fELF':
                require(expected.get(resolved) == hashed[resolved], 'Unregistered mapped ELF: ' + resolved)
                actual[resolved] = hashed[resolved]
            else:
                require(spelling.startswith('/nix/store/') and spelling.endswith('/lib/locale/locale-archive')
                        and fields[1] == 'r--p', 'Unclassified application mapping: ' + spelling)
                locale[resolved] = hashed[resolved]
        samples.append({'path': source.name, 'elf_files': actual, 'read_only_locale': locale,
                        'required_application_files_present': required.issubset(actual)})
    require(application_seen and samples, 'No authenticated application mapping was captured')
    require(any(item['required_application_files_present'] and set(item['elf_files']) == set(expected) for item in samples),
            'No authenticated map captured the exact target closure, application, and production TIRust DSO')
    # Every ELF in every captured application map must be registered. The
    # selected backend is still not observable in the unchecked return code.
    return {'verified': True, 'samples': samples, 'backend_observed': False,
            'scope': 'sampled runtime ELF bytes only; no backend or accuracy inference'}


def point_metrics(value, results, spread):
    cases = []
    reasons = {}
    by_id = {item['request']['id']: item for item in results}
    for case in value['cases']:
        lanes = []
        for backend in BACKENDS:
            request_id = case['id'] + '-backend' + str(backend)
            result = by_id.get(request_id, {})
            parsed = result.get('diagnostics', {})
            rows = parsed.get('rows', [])
            values = []; unavailable = []
            bound = (result.get('runtime_binding', {}).get('verified') is True
                     and result.get('readonly_guard_passed') is True)
            for point in range(1, 101):
                selected = [row for row in rows if row['phase'] == 1 and row['point'] == point and row['order'] == 0]
                reason = None
                if not bound:
                    reason = 'Runtime or read-only input binding unavailable/failed'
                elif [row['method'] for row in selected] != [1, 2, 3]:
                    reason = 'Three complete forced-DP probes unavailable'
                elif not all(row['raw_DP_diagnostic_eligible'] for row in selected):
                    reason = '; '.join(dict.fromkeys(row['reason'] for row in selected if row['reason']))
                    reason = reason or 'Raw forced-DP call is ineligible'
                elif any(not isinstance(row['coefficients'][0], (int, float))
                         or row['coefficients'][0] == 0 for row in selected):
                    reason = 'Born normalization is zero or nonfinite'
                if reason:
                    values.append(None); unavailable.append({'point': point, 'reason': reason}); continue
                observables = [row['coefficients'][1] / row['coefficients'][0] for row in selected]
                if not all(math.isfinite(item) for item in observables):
                    values.append(None); unavailable.append({'point': point, 'reason': 'Nonfinite normalized finite observable'})
                    continue
                metric = spread(observables)
                values.append('inf' if math.isinf(metric) else metric)
            lanes.append({'backend': backend, 'values': values})
            reasons[request_id] = unavailable
        cases.append({'id': case['id'], 'label': case['process'], 'point_ids': list(range(1, 101)), 'lanes': lanes})
    return {'schema': 'madloop-stability-point-metrics-v1', 'metric': 'raw-dp-finite-consistency',
        'provenance': {'source_revision': REVISION, 'manifest_sha256': MANIFEST_SHA,
            'protocol_sha256': digest(HERE / 'protocol.json'), 'point_count_per_case': 100,
            'scope': 'unchecked forced-DP internal consistency; not host-qualified accuracy or true error',
            'observable': 'finite/Born; common alpha_s/(2*pi) factor cancels from relative spread and is omitted',
            'backend_observation': 'effective singleton numerical card; unchecked return code has no reducer identity',
            'unavailable_reasons': reasons}, 'cases': cases}


def run(value, helpers):
    source_index = verify_seal()
    checkpoint = read(REPORT / 'prepare-terminal.json')
    require(checkpoint['status'] == 'passed' and checkpoint['source_index_sha256'] == source_index
            and checkpoint['protocol_sha256'] == digest(HERE / 'protocol.json'), 'Passed matching preparation required')
    require(not (REPORT / 'run-started.json').exists(), 'Run stage cannot be retried or resumed in place')
    verify_production(value)
    schedule = requests(value); results = []; failure = None
    save(REPORT / 'run-started.json', {'source_index_sha256': source_index,
                                     'preparation_sha256': digest(REPORT / 'prepare-terminal.json'),
                                     'requests': schedule})
    try:
        with signal_requests(helpers['cleanup']) as signals:
            for request in schedule:
                links = helpers['links']; case = checkpoint['cases'][request['case']]
                master = Path(case['master_snapshot']['root'])
                audit = case['donor_context']['audit']
                output = WORK / 'workers' / request['id']
                evidence = REPORT / 'requests' / request['id']
                evidence.mkdir(parents=True)
                record = {'request': request, 'state': 'unexecuted', 'native_started': False,
                          'host_accuracy_qualified': False, 'primary_calls_expected': 300}
                context = None
                try:
                    signals.check(); verify_production(value)
                    links.verify_tree(Path(case['donor_context']['master']), case['donor_context']['expected_master'])
                    context = links.prepare_worker(master, output, case['master_snapshot'], audit)
                    directory = output / case['donor']['subprocess']
                    card = configure_card(links, output, request['backend'])
                    record['effective_card'] = card
                    physical = directory / 'param_card.dat'
                    links.assert_private_path(output, physical)
                    require(digest(physical) == case['donor']['physical_card_sha256'],
                            'Driver-resolved physical card differs or is shadowed')
                    changed = [output / name for name in card['files']]
                    for name in ('stability-values.dat', 'stability-dimensions.dat', 'stability-groups.dat'):
                        links.assert_private_path(output, directory / name)
                        require(not (directory / name).exists(), 'Inherited diagnostic output would pollute worker')
                    require(digest(directory / 'stability-check') == case['executable_sha256'], 'Private executable changed')
                    require(digest(case['donor']['points']) == case['donor']['points_sha256']
                            and digest(Path(case['donor']['points']).with_suffix('.json'))
                            == case['donor']['point_metadata_sha256'], 'Fresh point stream changed after preparation')
                    with numerical_lock(value, request, signals) as lock:
                        record.update(native_launch_attempted=True, state='running', lock=lock)
                        native = command([directory / 'stability-check', case['donor']['points'], str(request['backend'])],
                                         directory, evidence / 'native', value['limits']['request_wall_seconds'],
                                         helpers['cleanup'], signals,
                                         target_environment=case['native_environment'])
                    record['native'] = native
                    record['native_started'] = native['process_started']
                    record['state'] = 'complete' if native['error'] is None else 'failed'
                except BaseException as error:
                    record.update(state='failed' if record['native_started'] else 'unexecuted',
                                  error={'class': type(error).__name__, 'message': str(error)})
                finally:
                    if context is not None:
                        try:
                            directory = output / case['donor']['subprocess']
                            changed = [output / name for name in record.get('effective_card', {}).get('files', {})]
                            # Canonical donor may have param.log; only the private file may change.
                            changed.extend(path for path in (output / 'SubProcesses').rglob('*')
                                           if path.name == 'param.log' or path.name.endswith(('HelFilter.dat', 'LoopFilter.dat')))
                            allowed = [str((directory / name).relative_to(output))
                                       for name in ('stability-values.dat', 'stability-dimensions.dat', 'stability-groups.dat', 'param.log')]
                            private_changes(links, output, context, changed, allowed)
                            links.verify_tree(Path(case['donor_context']['master']), case['donor_context']['expected_master'])
                            verify_production(value)
                            record['readonly_guard_passed'] = True
                        except BaseException as error:
                            record.update(state='failed', readonly_guard_error={'class': type(error).__name__, 'message': str(error)})
                    if record['native_started']:
                        record['raw_files'] = {}
                        for filename in ('stability-dimensions.dat', 'stability-values.dat', 'stability-groups.dat'):
                            try:
                                path = directory / filename
                                if path.is_file():
                                    blob = path.read_bytes()
                                    with (evidence / filename).open('xb') as stream: stream.write(blob)
                                    record['raw_files'][filename] = {'sha256': hashlib.sha256(blob).hexdigest(), 'size': len(blob)}
                            except BaseException as error:
                                record.setdefault('collection_errors', []).append({'file': filename,
                                    'class': type(error).__name__, 'message': str(error)})
                        try:
                            dimensions = (evidence / 'stability-dimensions.dat').read_text()
                            raw = (evidence / 'stability-values.dat').read_text()
                            console = (evidence / 'native/stdout.txt').read_text(errors='replace')
                            group_raw = (evidence / 'stability-groups.dat').read_text()
                            record['diagnostics'] = parse_diagnostics(dimensions, raw, console, group_raw, case['group_layout'])
                            record['runtime_binding'] = runtime_binding(evidence / 'native',
                                directory / 'stability-check', case['runtime_closure']['files'])
                            require(record['diagnostics']['structurally_complete'], 'Incomplete diagnostic stream')
                            require(record['diagnostics']['loop_groups_structurally_complete'], 'Incomplete native group coverage journal')
                        except BaseException as error:
                            record['state'] = 'failed'
                            record['diagnostic_error'] = {'class': type(error).__name__, 'message': str(error)}
                    if 'diagnostics' not in record:
                        record['primary_call_outcomes'] = [{'point': point, 'method': method, 'state': 'unexecuted'}
                                                          for point in range(1, 101) for method in (1, 2, 3)]
                    save(evidence / 'result.json', record)
                    results.append(record)
                # A native numerical failure is retained; remaining scheduled requests still run once.
                signals.check()
    except BaseException as error:
        failure = {'class': type(error).__name__, 'message': str(error)}
    finally:
        for request in schedule[len(results):]:
            results.append({'request': request, 'state': 'unexecuted', 'native_started': False,
                            'host_accuracy_qualified': False, 'primary_calls_expected': 300,
                            'primary_call_outcomes': [{'point': point, 'method': method, 'state': 'unexecuted'}
                                                     for point in range(1, 101) for method in (1, 2, 3)]})
        save(REPORT / 'run-terminal.json', {'source_index_sha256': source_index,
            'preparation_sha256': digest(REPORT / 'prepare-terminal.json'), 'error': failure,
            'status': 'passed' if failure is None and all(item['state'] == 'complete' for item in results) else 'failed',
            'request_count': 20, 'requests': results, 'host_accuracy_qualified': False,
            'interpretation': 'unchecked forced-DP consistency pilot; not a true-error or runtime acceptance test'})
        dataset = point_metrics(value, results, helpers['plot_metrics'].relative_spread)
        dataset['provenance'].update({
            'closed_run_terminal_sha256': digest(REPORT / 'run-terminal.json'),
            'preparation_checkpoint_sha256': digest(REPORT / 'prepare-terminal.json'),
            'fresh_point_streams': {case: {'sha256': item['donor']['points_sha256'],
                'metadata_sha256': item['donor']['point_metadata_sha256'],
                'physical_card_sha256': item['donor']['physical_card_sha256']}
                for case, item in checkpoint['cases'].items()},
            'closed_request_records': {str(path.relative_to(REPORT)): digest(path)
                for path in sorted((REPORT / 'requests').glob('*/result.json'))}})
        helpers['plot_metrics'].validate_dataset(dataset)
        save(REPORT / 'point-metrics.json', dataset)
    require(failure is None and all(item['state'] == 'complete' for item in results),
            'Pilot retains failed or unexecuted requests')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'prepare', 'run'), nargs='?', default='plan')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    value = protocol()
    if args.stage == 'plan' or not args.execute:
        print(json.dumps({'stage': args.stage, 'execute': False, 'commands_executed': 0,
            'requests': requests(value), 'work_directory': str(WORK),
            'preparation': 'read-only closed-export validation, private driver compilation/link only',
            'sampling_gate': 'four new separate cut-qualified 100-point streams; frozen before any native call',
            'native_calls': 0}, indent=2, sort_keys=True))
        return 0
    verify_seal()
    helpers = load_helpers()
    if args.stage == 'prepare': prepare(value, helpers)
    else: run(value, helpers)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
