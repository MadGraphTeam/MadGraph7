"""One serial execution of the two already-built tensor diagnostics.

--verify reads source/build bindings only. Normal execution needs root's
numerical-slot admission; it never compiles, retries, or changes fixtures.
"""
from decimal import localcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import runpy
import shutil
import signal
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = ROOT / '.tirust-build/generated-rank6-poles-1'
OUT = WORK / 'execution-1'
SUPERVISOR = ROOT / 'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py'
SUPERVISOR_SHA = 'ed25a9fc96e2faf2b545a9bf70a91b34315c01e534a6ebe68d7520ec2171aa42'
LOADER = Path('/nix/store/j193mfi0f921y0kfs8vjc1znnr45ispv-glibc-2.40-66/lib/ld-linux-x86-64.so.2')
LOADER_SHA = '1e08370bba3ee9e4f97bb0500d1f32afb0417babca6ae455c6458b9a3edb86a8'
BRIDGE = ROOT / '.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib/libmadloop_tirust.so'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def interpreter(path):
    """Read PT_INTERP of these fixed little-endian ELF64 executables."""
    with path.open('rb') as stream:
        header = stream.read(64)
        assert header[:6] == b'\x7fELF\x02\x01'
        offset = struct.unpack_from('<Q', header, 32)[0]
        size, count = struct.unpack_from('<HH', header, 54)
        names = []
        for number in range(count):
            stream.seek(offset + number * size)
            entry = stream.read(size)
            if struct.unpack_from('<I', entry)[0] == 3:
                start = struct.unpack_from('<Q', entry, 8)[0]
                length = struct.unpack_from('<Q', entry, 32)[0]
                stream.seek(start)
                names.append(stream.read(length).rstrip(b'\0').decode())
    assert names == [str(LOADER)], names


def verify():
    assert Path.cwd() == ROOT
    assert sha(HERE / 'review-index.json') == '38b30cda616b7ef9d589b8c83a0d4e1eaa8699397d567d97c8517695a80f72fb'
    previous = sys.argv
    try:
        sys.argv = [str(HERE / 'verify.py'), '--sources']
        runpy.run_path(str(HERE / 'verify.py'), run_name='__main__')
    finally:
        sys.argv = previous
    assert sha(HERE / 'build1/archive.json') == 'dc5fcf8ea9c52c2ede4db605e8aaea2a554ed638fafda7190e1e885fc4473d40'
    archive = json.loads((HERE / 'build1/archive.json').read_text())
    for name, expected in archive['files'].items():
        assert sha(HERE / 'build1' / name) == expected, name
    build = json.loads((HERE / 'build1/build.json').read_text())
    assert build['complete'] and build['sources_and_dependencies_unchanged']
    assert len(build['commands']) == 14 and all(row['exit_code'] == 0 for row in build['commands'])
    for name, item in build['artifacts'].items():
        assert sha(WORK / name) == item['sha256'], name
    assert sha(SUPERVISOR) == SUPERVISOR_SHA and sha(LOADER) == LOADER_SHA
    for model in ('ew', 'qcd'):
        interpreter(WORK / model / 'rank6-poles')
    return json.loads((HERE / 'build-plan.json').read_text()), build


def supervisor():
    # Reuse the reviewed ownership implementation verbatim. Only its fixed
    # mapping-affinity expectation and corresponding error text move to CPU12.
    source = SUPERVISOR.read_text()
    assert source.count("affinity!=['11']") == source.count('differs from CPU11.') == 1
    source = source.replace("affinity!=['11']", "affinity!=['12']").replace('differs from CPU11.', 'differs from CPU12.')
    namespace = {'__name__': 'rank6_owned_supervisor', '__file__': str(SUPERVISOR)}
    exec(compile(source, str(SUPERVISOR), 'exec'), namespace)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def worker_environment(spec):
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith('LD_') or key in ('GLIBC_TUNABLES', 'GCONV_PATH'):
            del env[key]
    env.update(LD_LIBRARY_PATH=':'.join(dict.fromkeys(spec['runtime_library_paths'])),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    return env


def closure(text, plan):
    assert 'not found' not in text
    names = re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(0x[0-9a-f]+\)', text, re.M)
    paths = {str(Path(name).resolve()) for name in names}
    assert str(LOADER.resolve()) in paths
    expected = {str(Path(name).resolve()): value for name, value in plan['resolved_library_files'].items()}
    critical = {str(Path(name).resolve()): value for name, value in expected.items()
                if Path(name).name in ('libmadloop_tirust.so', 'libcollier.so', 'libninja.so.0.0.0')}
    assert len(critical) == 3 and set(critical).issubset(paths), paths
    found = {name: sha(name) for name in sorted(paths)}
    for name, value in found.items():
        if name in expected:
            assert value == expected[name], name
        else:
            assert name.startswith('/nix/store/'), name
    assert all(found[name] == value for name, value in critical.items())
    return found


def analyze():
    analyzer = runpy.run_path(str(HERE / 'analyze.py'))
    with localcontext() as context:
        context.prec = 100
        result = analyzer['analyze']([WORK / 'ew', WORK / 'qcd'])
    save(OUT / 'results.json', result)
    return {'status': result['status'], 'open_items': len(result['open_items']),
            'results_sha256': sha(OUT / 'results.json')}


def main():
    plan, build = verify()
    if sys.argv[1:] == ['--verify']:
        print('PASS: source/build/executable/interpreter bindings; no loader or numerical execution')
        return
    assert not sys.argv[1:]
    helper, adapted_sha = supervisor()
    os.sched_setaffinity(0, {12})
    with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Numerical slot busy; no attempt created.')
            raise SystemExit(75)
        for model in ('ew', 'qcd'):
            assert not any((WORK / model / name).exists() for name in plan['models'][model]['runtime_outputs'])
        OUT.mkdir(exist_ok=False)
        record = {'complete': False, 'runner_sha256': sha(__file__), 'review_index_sha256': sha(HERE / 'review-index.json'),
                  'build_sha256': sha(HERE / 'build1/build.json'), 'supervisor_sha256': SUPERVISOR_SHA,
                  'adapted_supervisor_sha256': adapted_sha, 'adaptation': 'mapping affinity 11 -> 12 only',
                  'analyzer_sha256': sha(HERE / 'analyze.py'), 'loader_sha256': LOADER_SHA,
                  'cpu': 12, 'worker_timeout_seconds': 900, 'worker_address_space_bytes': 4294967296,
                  'scope': 'tensor diagnostics, not physical-amplitude or timing qualification', 'workers': {}}
        start = time.monotonic()
        previous = {}
        baseline = {model: {str(p.relative_to(WORK / model)): sha(p) for p in (WORK / model).rglob('*') if p.is_file()}
                    for model in ('ew', 'qcd')}
        save(OUT / 'inputs-before.json', baseline)
        def interrupt(signum, frame):
            raise InterruptedError('Rank-six replay interrupted by signal %d' % signum)
        def checkpoint():
            save(OUT / 'execution.json', record)
        try:
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                previous[signum] = signal.signal(signum, interrupt)
            resource.setrlimit(resource.RLIMIT_AS, (4294967296, 4294967296))
            record['inherited_rlimit_as'] = resource.getrlimit(resource.RLIMIT_AS)
            # Inspect both exact executable closures before either numerical run.
            for model in ('ew', 'qcd'):
                spec = plan['models'][model]
                env = worker_environment(spec)
                executable = WORK / model / 'rank6-poles'
                item = {'executable_sha256': sha(executable), 'environment_overrides': {
                            key: env[key] for key in ('LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS')},
                        'loader': {'argv': [str(LOADER), '--list', str(executable)], 'progress': {}}}
                record['workers'][model] = item
                checkpoint()
                with (OUT / (model + '-loader.log')).open('xb') as log:
                    item['loader']['exit_code'] = helper['run_plain_capped'](
                        item['loader']['argv'], log, cwd=WORK / model, env=env,
                        progress=item['loader']['progress'], seconds=30)
                assert item['loader']['exit_code'] == 0
                item['runtime_sha256'] = closure((OUT / (model + '-loader.log')).read_text(), plan)
                checkpoint()
            for model in ('ew', 'qcd'):
                spec, item = plan['models'][model], record['workers'][model]
                executable = WORK / model / 'rank6-poles'
                assert sha(executable) == item['executable_sha256']
                assert all(sha(name) == value for name, value in item['runtime_sha256'].items())
                item['argv'] = [str(executable)]
                item['progress'] = {}
                checkpoint()
                with (OUT / (model + '-console.log')).open('xb') as log:
                    item['exit_code'], item['mapping'] = helper['run_captured'](
                        item['argv'], log, cwd=WORK / model, env=worker_environment(spec),
                        expected_runtime=item['runtime_sha256'], executable=str(executable), selected_library=str(BRIDGE),
                        maps_path=OUT / (model + '-maps.txt'), progress=item['progress'], seconds=900)
                checkpoint()
                assert item['exit_code'] == 0, model
                assert sha(executable) == item['executable_sha256']
                for name, value in baseline[model].items():
                    assert sha(WORK / model / name) == value, name
                assert all((WORK / model / name).is_file() for name in spec['runtime_outputs'])
            record['analysis'] = analyze()
            verify()
            record['complete'] = True
        except BaseException as error:
            record['failure'] = repr(error)
            if not (OUT / 'results.json').exists():
                try:
                    record['analysis'] = analyze()
                except BaseException as analysis_error:
                    record['analysis_failure'] = repr(analysis_error)
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            record['elapsed_seconds'] = time.monotonic() - start
            record['outputs'] = {}
            for model in ('ew', 'qcd'):
                outputs = {}
                for path in sorted((WORK / model).rglob('*')):
                    if path.is_file() and str(path.relative_to(WORK / model)) not in baseline[model]:
                        name = str(path.relative_to(WORK / model))
                        target = OUT / model / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
                        outputs[name] = {'sha256': sha(target), 'bytes': target.stat().st_size}
                record['outputs'][model] = outputs
            checkpoint()
    print(json.dumps({'complete': record['complete'], 'analysis': record.get('analysis')}))


if __name__ == '__main__':
    main()
