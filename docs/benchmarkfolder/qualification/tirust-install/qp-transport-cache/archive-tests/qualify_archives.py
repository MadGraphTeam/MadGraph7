"""Qualify installed Fortran archives; never recompile the wrapper source."""
import ast
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path.cwd()
WORK = ROOT / '.tirust-build/qp-transport-install'
REPORT = ROOT / 'docs/benchmarkfolder/qualification/tirust-install/qp-transport-cache'
PREFIXES = {
    'old095e': ROOT / '.tirust-build/current-install/mg/HEPTools/tirust',
    'new059b': WORK / 'mg/HEPTools/tirust',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def run(command, cwd, log, timeout=60, native=False):
    entry = {'command': [str(x) for x in command], 'cwd': str(cwd),
             'queued_utc': now(), 'timeout_seconds_after_lock': timeout,
             'native_shared_lock': native, 'log': str(log.relative_to(REPORT))}
    lock = None
    try:
        if native:
            lock = (ROOT / '.tirust-build/numerical-tests.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX)
        entry['started_utc'] = now()
        if Path(command[0]).is_file():
            entry['executable_sha256'] = sha(Path(command[0]))
        with log.open('w') as output:
            result = subprocess.run(command, cwd=cwd, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=timeout,
                                    env={k: v for k, v in os.environ.items() if k not in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT')})
        entry['returncode'] = result.returncode
    except subprocess.TimeoutExpired:
        entry.update(returncode=None, timed_out=True)
    finally:
        if lock is not None:
            lock.close()
    entry.update(finished_utc=now(), log_sha256=sha(log))
    if 'executable_sha256' in entry:
        entry['executable_sha256_after'] = sha(Path(command[0]))
        if entry['executable_sha256_after'] != entry['executable_sha256']:
            raise RuntimeError('Executable changed during invocation')
    return entry


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    output = REPORT / 'archive-tests'
    output.mkdir(exist_ok=False)
    builds = WORK / 'archive-tests'
    builds.mkdir(exist_ok=False)
    fixture = WORK / 'fixtures/test_tirust_transport_cache.py'
    require(sha(fixture) == '80cb1533ba18a3bd9b0bceddda7f551b5ac392c1680d6bb1ec67a675f410a2f0',
            'Committed transport fixture changed')
    constants = {}
    for node in ast.parse(fixture.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('PROVIDER', 'DRIVER'):
                    constants[target.id] = ast.literal_eval(node.value)
    require(set(constants) == {'PROVIDER', 'DRIVER'}, 'Missing fixture source')
    (output / 'provider.c').write_text(constants['PROVIDER'])
    (output / 'driver.f90').write_text(constants['DRIVER'])
    shutil.copy2(fixture, output / fixture.name)
    shutil.copy2(WORK / 'fixtures/real-provider.f90', output / 'real-provider.f90')
    shutil.copy2(WORK / 'fixtures/smoke.f90', output / 'smoke.f90')
    data = {'schema_version': 1, 'state': 'running', 'performance_eligible': False,
            'scope': 'Actual installed Fortran archive and module with independent mock C provider; '
                     'real installed-provider smoke and separately classified bounded extreme fixture.',
            'fixture_sha256': sha(fixture), 'source_sha256': {name: sha(output / name) for name in ('provider.c', 'driver.f90', 'real-provider.f90', 'smoke.f90')}, 'commands': [], 'identities': {}, 'results': {}}
    write(output / 'results.json', data)
    for label, prefix in PREFIXES.items():
        lib = prefix / 'lib'
        directory = builds / label
        directory.mkdir()
        data['identities'][label] = {'prefix': str(prefix),
            'artifacts': {str(path.relative_to(prefix)): sha(path) for path in
                [lib / 'libtirust.a', lib / 'mtirust.mod', lib / 'libmadloop_tirust.so',
                 lib / 'tirust-config.json', prefix / 'include/mtirust.f90']}}
        for name in ('provider.c', 'driver.f90', 'smoke.f90', 'real-provider.f90'):
            shutil.copy2(output / name, directory / name)
        commands = [
            ['cc', '-std=c99', '-O0', '-c', 'provider.c', '-o', 'provider.o'],
            ['gfortran', '-O0', '-fcheck=all', '-frounding-math', '-I' + str(lib),
             'driver.f90', 'provider.o', str(lib / 'libtirust.a'), '-o', 'mock-check'],
        ]
        for index, command in enumerate(commands):
            item = run(command, directory, output / f'{label}-mock-build-{index}.log')
            data['commands'].append(item)
            require(item['returncode'] == 0, 'Installed mock build failed: ' + label)
        modes = ['transport', 'underflow'] + (['clear-failure'] if label == 'new059b' else [])
        for mode in modes:
            item = run([str(directory / 'mock-check'), mode], directory,
                       output / f'{label}-mock-{mode}.log', timeout=30)
            data['commands'].append(item)
            text = (output / f'{label}-mock-{mode}.log').read_text()
            expected = {'transport': 'PASS transport cases=28',
                        'clear-failure': 'PASS clear-failure cases=7',
                        'underflow': 'PASS underflow cases=2'}[mode]
            if label == 'old095e' and mode == 'transport':
                passed = item['returncode'] not in (0, None) and 'FAIL rejected entries must be absent' in text
                classification = 'expected-failure-negative-control'
            else:
                passed = item['returncode'] == 0 and expected in text
                classification = 'passed'
            data['results'][label + '-mock-' + mode] = {'passed': passed, 'classification': classification}
            write(output / 'results.json', data)
            require(passed, 'Installed mock check failed: ' + label + ' ' + mode)
        before = shlex.split((lib / 'tirust-link-before.txt').read_text())
        after = shlex.split((lib / 'tirust-native-libs.txt').read_text())
        for name in ('smoke', 'real-provider'):
            command = ['gfortran', '-O0', '-fcheck=all', '-frounding-math', '-I' + str(lib),
                       name + '.f90'] + before + [str(lib / 'libtirust.a'), '-L' + str(lib),
                       '-Wl,-rpath,' + str(lib)] + after + ['-o', name]
            item = run(command, directory, output / f'{label}-{name}-build.log')
            data['commands'].append(item)
            require(item['returncode'] == 0, 'Installed real-provider link failed: ' + label)
        for executable in ('smoke', 'real-provider'):
            item = run(['ldd', str(directory / executable)], directory, output / f'{label}-{executable}-runtime.log')
            data['commands'].append(item)
            require(item['returncode'] == 0 and str(lib / 'libmadloop_tirust.so') in (output / f'{label}-{executable}-runtime.log').read_text(), 'Runtime DSO resolution mismatch')
        item = run([str(directory / 'smoke'), '--require-poles'], directory,
                   output / f'{label}-smoke.log', timeout=180, native=True)
        data['commands'].append(item)
        text = (output / f'{label}-smoke.log').read_text()
        passed = item['returncode'] == 0 and 'TIRust ABI1: DP, binary128 input, UV/IR poles and cache checks passed' in text
        data['results'][label + '-smoke'] = {'passed': passed}
        write(output / 'results.json', data)
        require(passed, 'Installed real smoke failed: ' + label)
        # Exactly one bounded invocation per archive; it performs one seed and
        # two calls of the same extreme fixture. Refusal is not boundary proof.
        item = run([str(directory / 'real-provider')], directory,
                   output / f'{label}-real-provider.log', timeout=180, native=True)
        data['commands'].append(item)
        text = (output / f'{label}-real-provider.log').read_text()
        classes = [line.split(' ', 1)[1] for line in text.splitlines() if line.startswith('CLASSIFICATION ')]
        data['results'][label + '-real-provider'] = {
            'returncode': item['returncode'], 'timed_out': item.get('timed_out', False),
            'classifications': classes,
            'public_transport_rejection_observed': 'public-transport-rejection' in classes,
            'claim': 'Optional causal diagnostic; provider refusal or timeout is retained without a successful boundary claim.'}
        write(output / 'results.json', data)
        for path, expected in data['identities'][label]['artifacts'].items():
            require(sha(prefix / path) == expected, 'Installed artifact changed: ' + label + ' ' + path)
    for name, expected in data['source_sha256'].items():
        require(sha(output / name) == expected, 'Fixture source changed during qualification')
        for label in PREFIXES:
            require(sha(builds / label / name) == expected, 'Compiled fixture copy differs')
    old = (output / 'old095e-mock-underflow.log').read_bytes()
    new = (output / 'new059b-mock-underflow.log').read_bytes()
    require(old == new, 'Finite-underflow outputs/IEEE flags changed')
    data.update(state='complete', finite_underflow_logs_identical=True,
                finite_underflow_log_sha256=hashlib.sha256(new).hexdigest())
    write(output / 'results.json', data)
    shutil.copy2(Path(__file__), output / 'qualify_archives.py')
    print(json.dumps({'state': data['state'], 'results': data['results']}, indent=2))


if __name__ == '__main__':
    main()
