"""One pinned tensor replay, with owned cleanup and READY/DONE runtime maps.

CLI: python -B worker.py SPEC SPEC_SHA256. Numerical statuses 0..3 are retained
diagnostic outcomes; process/protocol failure is separate and never retried.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT/'.tirust-build/kinematic-provenance-current-executor-1/source'
EXECUTOR = ROOT/'.tirust-build/kinematic-projection-replay-executor-1'
SOURCE_SHA = '7ea709b48d8de462b754fabfb47dcbd5d21d7847458382f5844190028e8c1bed'
TIMEOUT = 20.0
STDOUT_LIMIT = 1048576
SCHEMA = 'private-kinematic-projection-worker-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def record(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def verify(path, pin):
    require(record(path) == {k: pin[k] for k in ('bytes', 'sha256')},
            'Pinned bytes differ: '+str(path))


def save(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def seal(path, checksum):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == checksum, 'Reviewed index differs: '+str(path))
    for name, pin in json.loads(raw)['files'].items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts, 'Unsafe index member')
        verify(Path(path).parent/p, pin)


def admitted_helpers(proposal_sha):
    seal(HERE/'index.json', proposal_sha)
    seal(SOURCE/'index.json', SOURCE_SHA)
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location('admission', SOURCE/'admission.py')
    admission = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = admission
    spec.loader.exec_module(admission)
    admission.initial_admission()
    cleanup = admission.load('kproj_owned_cleanup', ROOT/
        'docs/benchmarkfolder/experiments/current-cms-replay-supervisor-1/ownership/process_cleanup.py')
    native = admission.load('kproj_existing_maps', SOURCE/'native_job.py')
    return cleanup, native.mapped_files


def input_words(path, request):
    lines = Path(path).read_text().splitlines()
    require(len(lines) >= 2 and lines[0] == 'KPROJ_WORDS_V1', 'Input schema differs')
    count = 41 if request['mode'] == 1 else 82
    require(lines[1].split() == [str(request['point']), str(request['mode']), '4', '2', str(count)],
            'Input identity differs')
    words = lines[2:]
    require(len(words) == count and all(len(w) == 16 and
            all(c in '0123456789ABCDEF' for c in w) for w in words), 'Input words differ')
    return words


class Protocol:
    """Only the fixed replay driver's records; native error text stays verbatim."""
    def __init__(self, request, words):
        self.request, self.words = request, words
        self.phase = 'initial'
        self.fields = {}
        self.events = []
        self.error_open = False
        self.error_seen = False
        self.error_lines = []

    def line(self, line):
        if self.error_open:
            if line == 'ERROR_END':
                self.error_open = False
            else:
                self.error_lines.append(line)
            return None
        if line == 'ERROR_BEGIN':
            require(self.phase == 'evaluating' and not self.error_seen, 'Unexpected error frame')
            self.error_open = self.error_seen = True
            return None
        if line.startswith(('READY', 'DONE', 'CLOSED')):
            if line == 'READY KPROJ_V1':
                require(self.phase == 'initial', 'Repeated/out-of-order READY')
                expected = [str(self.request['point']), str(self.request['mode']), '4', '2', str(len(self.words))]
                require(self.fields.get('INPUT_ID') == expected, 'Driver input identity differs')
                require(self.fields.get('INPUT_WORDS') == self.words, 'Driver input word echo differs')
                require(self.fields.get('OPTIONS') == ['pole=0', 'finite=1', 'check=0', 'cache=0', 'budget=268435456'],
                        'Driver evaluation options differ')
                self.phase = 'evaluating'
                action = 'READY'
            elif line == 'DONE KPROJ_V1':
                require(self.phase == 'evaluating' and self.error_seen, 'Unexpected DONE/error frame missing')
                require(self.fields.get('FORTRAN_STATUS') in ([str(x)] for x in range(4)), 'Invalid native status')
                for name in ('FORTRAN_EVALUATION_CALLS', 'CONTEXT_GENERATION', 'INPUTS_UNCHANGED'):
                    require(self.fields.get(name) == ['1'], 'Driver count/context/input check differs: '+name)
                require(self.fields.get('POST_INPUT_WORDS') == self.words, 'Post-evaluation input words differ')
                require(self.fields.get('ERROR_BUFFER_FULL') == ['0'], 'Native error text missing or truncated')
                require(self.fields.get('COEFFICIENT_BUFFERS_NOT_INSPECTED') == [], 'Coefficient policy record missing')
                self.phase = 'closing'
                action = 'DONE'
            elif line == 'CLOSED KPROJ_V1':
                require(self.phase == 'closing' and self.fields.get('FINALIZE_STATUS') == ['0'],
                        'Close/finalize status differs')
                self.phase = 'closed'
                action = 'CLOSED'
            else:
                raise ValueError('Unexpected protocol marker: '+line)
            self.events.append(action)
            return action
        tokens = line.split()
        if not tokens:
            return None
        before = {'INPUT_ID', 'OPTIONS', 'INPUT_WORDS'}
        measured = {'FORTRAN_STATUS', 'FORTRAN_EVALUATION_CALLS', 'CONTEXT_GENERATION',
                    'POST_INPUT_WORDS', 'INPUTS_UNCHANGED', 'ERROR_BUFFER_FULL',
                    'COEFFICIENT_BUFFERS_NOT_INSPECTED'}
        key = tokens[0]
        if key in before | measured | {'FINALIZE_STATUS'}:
            expected = 'initial' if key in before else 'evaluating' if key in measured else 'closing'
            require(self.phase == expected and key not in self.fields, 'Repeated/out-of-order driver field: '+key)
            self.fields[key] = tokens[1:]
        elif key == 'ERROR_END':
            raise ValueError('Error frame ended without beginning')
        return None


def verify_inputs(spec):
    verify(spec['input_file'], spec['input_pin'])
    verify(spec['executable'], spec['executable_pin'])
    for name, checksum in spec['expected_runtime'].items():
        require(record(name)['sha256'] == checksum, 'Runtime bytes changed: '+name)


def execute(specpath, spec_sha):
    raw = Path(specpath).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == spec_sha, 'Parent-pinned specification differs')
    spec = json.loads(raw)
    require(spec['schema'] == SCHEMA and Path.cwd() == Path(spec['cwd']).resolve(), 'Worker schema/cwd differs')
    cleanup, mapped_files = admitted_helpers(spec['proposal_index_sha256'])
    request = spec['request']
    certificate = json.loads((HERE/'input-certificate.json').read_bytes())
    matches = [r for r in certificate['requests'] if r['id'] == request['id']]
    require(matches == [request], 'Request is not the exact pinned diagnostic request')
    require(spec['input_pin'] == {k: request['input'][k] for k in ('bytes', 'sha256')}, 'Input pin differs from certificate')
    directory, evidence = Path(spec['cwd']).resolve(), Path(spec['evidence']).resolve()
    executable = Path(spec['executable']).resolve()
    require(directory == EXECUTOR/'work/workers'/request['id']
            and executable == EXECUTOR/'work/build/replay'
            and evidence == EXECUTOR/'results/runs'/request['id'], 'Worker paths differ from fixed private executor')
    require(directory.is_dir() and evidence.is_dir(), 'Parent must create private directories')
    require(spec['expected_runtime'].get(str(executable)) == spec['executable_pin']['sha256'],
            'Executable missing from exact expected runtime')
    require(all(Path(p).is_absolute() and str(Path(p).resolve()) == p for p in spec['expected_runtime']),
            'Runtime paths must be absolute and resolved')
    require(not any((evidence/n).exists() or (evidence/n).is_symlink()
            for n in ('result.json', 'stdout.txt', 'maps-READY.txt', 'maps-DONE.txt')),
            'Append-only worker evidence; no retry')
    process = owned = None
    failure = cleanup_failure = None
    close_receipt = None
    captures, protocol, sent = [], None, []
    total = 0
    overflow = False
    postcheck_failures = []
    start = time.monotonic()
    argv = [sys.executable, '-B', str(HERE/'limit_exec.py'), str(executable), str(Path(spec['input_file']).resolve())]
    env = os.environ.copy()
    for name in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT'):
        env.pop(name, None)
    env.update(RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with cleanup.SignalRequests() as signals:
        try:
            verify_inputs(spec)
            protocol = Protocol(request, input_words(spec['input_file'], request))
            signals.check()
            deadline = time.monotonic()+TIMEOUT
            with (evidence/'stdout.txt').open('xb') as output, selectors.DefaultSelector() as poller:
                process = subprocess.Popen(argv, cwd=directory, env=env, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
                owned = cleanup.OwnedChild(process)
                os.set_blocking(process.stdout.fileno(), False)
                poller.register(process.stdout, selectors.EVENT_READ)
                pending = b''
                eof = False

                def tick():
                    owned.sample()
                    signals.check()
                    require(time.monotonic() < deadline, 'Replay twenty-second deadline exhausted')

                def observe(boundary):
                    tick()
                    text = (Path('/proc')/str(process.pid)/'maps').read_text()
                    path = evidence/('maps-'+boundary+'.txt')
                    with path.open('x') as stream:
                        stream.write(text)
                    capture = {'boundary': boundary, 'file': path.name, **record(path), 'verified': False}
                    captures.append(capture)
                    capture.update(mapped_files(text, spec['expected_runtime']))
                    capture['verified'] = True
                    tick()

                while True:
                    tick()
                    if eof and process.poll() is not None:
                        break
                    for key, _ in poller.select(min(.05, max(0., deadline-time.monotonic()))):
                        try:
                            chunk = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            poller.unregister(process.stdout)
                            eof = True
                            require(not pending, 'Unterminated final output record')
                            continue
                        retained = min(len(chunk), max(0, STDOUT_LIMIT-total))
                        output.write(chunk[:retained]);output.flush()
                        total += len(chunk)
                        overflow = total > STDOUT_LIMIT
                        require(not overflow, 'Replay stdout exceeded one MiB')
                        pending += chunk
                        while b'\n' in pending:
                            line, pending = pending.split(b'\n', 1)
                            action = protocol.line(line.decode('utf-8').rstrip('\r'))
                            if action in ('READY', 'DONE'):
                                observe(action)
                                command = b'EVALUATE\n' if action == 'READY' else b'EXIT\n'
                                process.stdin.write(command);process.stdin.flush()
                                sent.append(command.decode().strip())
                require(process.returncode == 0 and protocol.phase == 'closed', 'Native process did not close successfully')
                require(protocol.events == ['READY', 'DONE', 'CLOSED'] and sent == ['EVALUATE', 'EXIT'],
                        'Replay protocol count differs')
        except BaseException as exc:
            failure = repr(exc)
        finally:
            try:
                if owned is not None:
                    close_receipt = owned.stop(cooperative_seconds=0, terminate_seconds=2, kill_seconds=2)
                    require(not close_receipt['survivors'], 'Owned replay survived cleanup')
                    process.wait(timeout=2)
                elif process is not None:
                    process.kill();process.wait(timeout=2)
                    close_receipt = {'fallback_exact_popen_child': True, 'survivors': []}
            except BaseException as exc:
                cleanup_failure = repr(exc)
            # Preserve any already-written tail after a protocol failure. The
            # child has been stopped; this nonblocking drain cannot wait on it.
            try:
                if process is not None and process.stdout is not None:
                    os.set_blocking(process.stdout.fileno(), False)
                    with (evidence/'stdout.txt').open('ab') as output:
                        while True:
                            try:
                                chunk = os.read(process.stdout.fileno(), 65536)
                            except BlockingIOError:
                                break
                            if not chunk:
                                break
                            output.write(chunk[:max(0, STDOUT_LIMIT-min(total, STDOUT_LIMIT))])
                            total += len(chunk)
                            overflow = total > STDOUT_LIMIT
                            if overflow:
                                break
            except BaseException as exc:
                cleanup_failure = cleanup_failure or repr(exc)
            for stream in (getattr(process, 'stdin', None), getattr(process, 'stdout', None)):
                if stream is not None:
                    try:
                        stream.close()
                    except BaseException as exc:
                        cleanup_failure = cleanup_failure or repr(exc)
    try:
        verify_inputs(spec)
    except BaseException as exc:
        postcheck_failures.append(repr(exc))
    fields = {} if protocol is None else protocol.fields
    status_words = fields.get('FORTRAN_STATUS')
    fortran_status = int(status_words[0]) if status_words in ([str(x)] for x in range(4)) else None
    passed = (failure is None and cleanup_failure is None and not postcheck_failures and not overflow
              and process is not None and process.returncode == 0 and close_receipt is not None
              and [c['boundary'] for c in captures if c['verified']] == ['READY', 'DONE'])
    result = {'schema': SCHEMA, 'request': request, 'specification_sha256': spec_sha,
        'orchestration_complete': passed, 'failure': failure, 'cleanup_failure': cleanup_failure,
        'cleanup': close_receipt, 'postcheck_failures': postcheck_failures,
        'native_pid': None if process is None else process.pid,
        'native_returncode': None if process is None else process.returncode,
        'argv': argv, 'elapsed_seconds_including_cleanup': time.monotonic()-start,
        'work_timeout_seconds': TIMEOUT, 'stdout_limit_bytes': STDOUT_LIMIT,
        'stdout_observed_bytes': total, 'stdout_overflow': overflow,
        'stdout': record(evidence/'stdout.txt') if (evidence/'stdout.txt').exists() else None,
        'runtime_maps': captures, 'commands_sent': sent,
        'protocol': [] if protocol is None else protocol.events,
        'driver_fields': fields, 'fortran_status': fortran_status,
        'native_error': None if protocol is None else '\n'.join(protocol.error_lines),
        'coefficient_buffers_read': False, 'independent_accuracy_qualified': False,
        'performance_measurement': False, 'retry_performed': False}
    save(evidence/'result.json', result)
    require(passed, 'Owned tensor replay incomplete; failure evidence retained')
    return result


if __name__ == '__main__':
    require(len(sys.argv) == 3, 'Pinned specification and SHA256 required')
    execute(Path(sys.argv[1]), sys.argv[2])
