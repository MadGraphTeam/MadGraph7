#!/usr/bin/env python3
"""Append-only, staged private Fortran experiment. Nothing runs on import."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.tirust-build/fortran-lazy-tirust-clear-2'


def outer_environment():
    result = os.environ.copy()
    for name in list(result):
        if name.startswith('LD_'):
            result.pop(name)
    result.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  RAYON_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def verify_source():
    source = json.loads((HERE / 'source-index.json').read_text())
    for name, value in source['files'].items():
        if sha(HERE / name) != value['sha256']:
            raise ValueError('Proposal source changed: ' + name)
    binding = json.loads((HERE / 'bindings.json').read_text())
    for name, expected in binding['dependencies'].items():
        if sha(Path(name)) != expected:
            raise ValueError('Bound dependency changed: ' + name)
    return binding


def replacement(text, candidate=True):
    delta = json.loads((HERE / 'replacement.json').read_text())
    if text.count(delta['old']) != 1:
        raise ValueError('Expected one unchanged generated clear wrapper')
    return text.replace(delta['old'], delta['new']) if candidate else text


def wrapper(text, candidate):
    value = replacement(text, candidate)
    match = re.search(r'      SUBROUTINE ML5_0_CLEAR_TIRUST_CACHE\(\)\n.*?      END\n', value, re.S)
    if not match:
        raise ValueError('Generated wrapper missing')
    body = match.group()
    return body.replace('ML5_0_CLEAR_TIRUST_CACHE', 'A_CLEAR') + '\n' + body.replace('ML5_0_CLEAR_TIRUST_CACHE', 'B_CLEAR')


def card(backend):
    text = (HERE / 'fixtures/MadLoopParams.dat').read_text()
    text, count = re.subn(r'(?m)(^#MLReductionLib\s*\n)[^\n]+', lambda match: match[1] + str(backend), text)
    if count != 1:
        raise ValueError('Ambiguous backend card')
    return text


def values_equal(left, right):
    # The policy and all input arithmetic are identical; require byte equality.
    if left != right:
        raise ValueError('Numerical/status/accuracy evidence is not byte identical')


def runtime_paths(text):
    return {str(Path(match).resolve()) for match in re.findall(r'(/[^\s()]+)', text)}


def run():
    if Path.cwd() != ROOT:
        raise ValueError('Run from the specified repository directory')
    binding = verify_source()
    lock = (ROOT / '.tirust-build/numerical-tests.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        WORK.mkdir(exist_ok=False)
    except BaseException:
        lock.close()
        raise
    started = time.monotonic()
    save(WORK / 'intent.json', {'source_index_sha256': sha(HERE / 'source-index.json'),
        'binding': binding, 'wall_clock_started_unix': time.time()})
    env = outer_environment()
    providers = WORK / 'original-providers'
    providers.mkdir()
    expected = {str(Path(p).resolve()) for p in binding['runtime']}
    for name in binding['runtime']:
        (providers / Path(name).name).symlink_to(name)
    donor, process = Path(binding['donor']), Path(binding['process'])
    build = WORK / 'build'
    build.mkdir()
    fc = binding['fc']
    install = ROOT / '.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib'
    includes = ['-I' + str(p) for p in (process, donor / 'Source/MODEL', donor / 'Source/DHELAS',
        ROOT / 'HEPTools/ninja/include', ROOT / 'HEPTools/include', ROOT / 'HEPTools/ninjars/lib', install)]
    compile_flags = ['-w', '-fPIC', '-ffixed-line-length-132', *includes]
    objects = binding['objects']
    libraries = ['-L' + str(donor / 'lib'), '-ldhelas', '-lmodel', *shlex.split(binding['link_flags']), '-lstdc++']
    sequence = 0

    def remaining():
        result = binding['global_wall_seconds'] - (time.monotonic() - started)
        if result < 20:
            raise TimeoutError('Global work deadline; cleanup reserve begins')
        return result

    def command(argv, directory, label, trace=False, native=False):
        nonlocal sequence
        remaining()
        sequence += 1
        output = WORK / 'commands' / f'{sequence:03d}-{label}'
        output.mkdir(parents=True)
        local_env = dict(env)
        target = list(map(str, argv))
        if native or trace:
            target = [binding['programs']['python'], str(HERE / 'exec_target.py'), str(providers),
                str(int(trace)), *target]
        argv = [binding['programs']['taskset'], '-c', str(binding['cpu']),
            binding['programs']['prlimit'], '--as=' + str(binding['memory_bytes']), '--', *target]
        save(output / 'command.json', {'argv': argv, 'cwd': str(directory),
            'outer_loader_variables': {k: local_env[k] for k in local_env if k.startswith('LD_')},
            'target_LD_LIBRARY_PATH': str(providers) if native or trace else None, 'trace': trace})
        begin = time.monotonic()
        with (output / 'stdout.txt').open('xb') as stream:
            child = subprocess.Popen(argv, cwd=directory, env=local_env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=min(90, remaining()))
            except BaseException:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(child.pid, sig)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=3)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                child.wait()
                save(output / 'terminal.json', {'returncode': child.returncode,
                    'elapsed_seconds': time.monotonic() - begin, 'interrupted': True})
                raise
        save(output / 'terminal.json', {'returncode': code, 'elapsed_seconds': time.monotonic() - begin})
        if code:
            raise RuntimeError('Command failed: ' + str(output))
        return (output / 'stdout.txt').read_text()

    def link(arm, observed):
        selected = [build / (arm + '-TIRUST_interface.o') if Path(p).name == 'TIRUST_interface.o' else p for p in objects]
        target = build / (arm + ('-observed' if observed else '-plain'))
        extra = [build / 'observer.o', '-Wl,--wrap=ml_tirust_context_create',
            '-Wl,--wrap=ml_tirust_clear_cache', '-Wl,--wrap=ml_tirust_register_plan'] if observed else []
        if arm == 'original' and observed:
            reuse = binding['reuse_artifacts']['original-observed']
            shutil.copyfile(reuse['path'], target)
            if sha(target) != reuse['sha256']:
                raise ValueError('Reused original executable identity mismatch')
            target.chmod(0o755)
            text = Path(binding['reuse_loader_trace']['path']).read_text()
            save(build / 'original-observed-reuse.json', {'artifact': reuse,
                'loader_trace': binding['reuse_loader_trace'], 'new_compilation_or_trace': False})
        else:
            command([fc, *compile_flags, '-o', target, build / 'worker.o', *selected, *extra, *libraries], build, 'link-' + target.name)
            text = command([target], build, 'trace-' + target.name, trace=True)
        if runtime_paths(text) != expected:
            raise ValueError('Loader providers differ: ' + repr(runtime_paths(text) ^ expected))
        return target

    worker_spec = importlib.util.spec_from_file_location('fortran_clear_worker', ROOT / 'tools/tirust_benchmark_worker.py')
    worker_module = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker_module)

    def worker(executable, backend, label):
        remaining()
        directory = WORK / 'workers' / label
        directory.mkdir(parents=True)
        shutil.copyfile(HERE / 'fixtures/points.dat', directory / 'points.dat')
        cards = directory / 'Cards'
        cards.mkdir()
        shutil.copyfile(HERE / 'fixtures/param_card.dat', cards / 'param_card.dat')
        shutil.copyfile(HERE / 'fixtures/ident_card.dat', cards / 'ident_card.dat')
        resource = directory / 'MadLoop5_resources'
        resource.mkdir()
        for path in sorted((donor / 'SubProcesses/MadLoop5_resources').iterdir()):
            if path.name in ('ML5_0_ColorDenomFactors.dat', 'ML5_0_ColorNumFactors.dat', 'ML5_0_HelConfigs.dat'):
                shutil.copyfile(path, resource / path.name)
        (resource / 'MadLoopParams.dat').write_text(card(backend))
        argv = [binding['programs']['taskset'], '-c', str(binding['cpu']), binding['programs']['prlimit'],
            '--as=' + str(binding['memory_bytes']), '--', binding['programs']['python'],
            str(HERE / 'exec_target.py'), str(providers), '0', str(executable), str(directory / 'points.dat')]
        instance = worker_module.Worker(argv, directory, directory / 'stdout.txt', env=env,
            timeout=min(binding['worker_timeout_seconds'], remaining() - 10))
        text = Path(f'/proc/{instance.process.pid}/maps').read_text()
        (directory / 'runtime-maps.txt').write_text(text)
        mapped = {str(Path(line.split()[-1]).resolve()) for line in text.splitlines()
            if line.split()[-1].startswith('/') and '.so' in line.split()[-1]}
        if mapped != expected:
            instance.close(graceful=False)
            raise ValueError('Actual runtime providers differ: ' + repr(mapped ^ expected))
        return instance, directory

    def observation(instance, directory, backend, loops, label):
        remaining()
        instance.run(loops)
        result = worker_module.parse_outputs(directory, backend, 100, loops)
        save(directory / (label + '.json'), result)
        return result, (directory / 'values.dat').read_bytes()

    try:
        # Profile the unchanged implementation before writing/compiling a candidate.
        for name in ('original-TIRUST_interface.o', 'worker.o', 'observer.o'):
            reuse = binding['reuse_artifacts'][name]
            shutil.copyfile(reuse['path'], build / name)
            if sha(build / name) != reuse['sha256']:
                raise ValueError('Reused preparation object identity mismatch: ' + name)
        save(build / 'preparation-reuse.json', {'artifacts': binding['reuse_artifacts'],
            'prior_loader_trace': binding['reuse_loader_trace'], 'new_original_preparation_commands': 0})
        original_observed = link('original', True)
        profile = {}
        observed_values = {}
        for backend in binding['main_backend_ids']:
            instance, directory = worker(original_observed, backend, 'profile-original-' + str(backend))
            with instance:
                item, _ = observation(instance, directory, backend, 100, 'profile')
            counts = json.loads((directory / 'ffi-profile.json').read_text())
            if counts['create_calls'] != 1 or counts['clear_calls'] < 20000:
                raise ValueError('Original profile did not demonstrate the expected lifecycle work')
            if backend == 1 and counts['register_calls']:
                raise ValueError('Unselected backend unexpectedly registered TIRust plans')
            profile[str(backend)] = {'observation': item, 'native_lifecycle': counts}
            observed_values[backend, 'original'] = (directory / 'values.dat').read_bytes()
        save(WORK / 'original-profile.json', profile)

        # This is the first candidate generated-source write, after actual profiling.
        candidate_source = build / 'candidate-TIRUST_interface.f'
        candidate_source.write_text(replacement((process / 'TIRUST_interface.f').read_text()))
        command([fc, *compile_flags, '-c', candidate_source, '-o', build / 'candidate-TIRUST_interface.o'], build, 'compile-candidate')
        plain = {arm: link(arm, False) for arm in ('original', 'candidate')}
        candidate_observed = link('candidate', True)
        for backend in binding['main_backend_ids']:
            instance, directory = worker(candidate_observed, backend, 'profile-candidate-' + str(backend))
            with instance:
                item, value = observation(instance, directory, backend, 100, 'profile')
            counts = json.loads((directory / 'ffi-profile.json').read_text())
            if backend == 1 and (counts['create_calls'] or counts['clear_calls']):
                raise ValueError('Candidate unselected backend still created/cleared TIRust')
            if backend == 9:
                original = profile['9']['native_lifecycle']
                if (counts['create_calls'] != 1 or counts['register_calls'] != original['register_calls']
                    or counts['clear_calls'] - counts['clears_before_first_plan_registration']
                    != original['clear_calls'] - original['clears_before_first_plan_registration']):
                    raise ValueError('Candidate selected-backend active lifecycle changed')
            values_equal((WORK / 'workers' / ('profile-original-' + str(backend)) / 'values.dat').read_bytes(), value)
            observed_values[backend, 'candidate'] = value

        # Native actual-module tests and isolated wrapper timing use the same providers.
        micro = {'0': {'original': [], 'candidate': []}, '1': {'original': [], 'candidate': []}}
        module_libs = ['-L' + str(install), '-ltirust', '-lmadloop_tirust']
        for arm in ('original', 'candidate'):
            (build / (arm + '-wrappers.f')).write_text(wrapper((process / 'TIRUST_interface.f').read_text(), arm == 'candidate'))
            command([fc, *compile_flags, '-c', build / (arm + '-wrappers.f'), '-o', build / (arm + '-wrappers.o')], build, 'compile-wrappers-' + arm)
            for kind in ('lifecycle', 'micro'):
                command([fc, '-I' + str(install), HERE / (kind + '.f90'), build / (arm + '-wrappers.o'),
                    *module_libs, '-o', build / (arm + '-' + kind)], build, 'compile-' + arm + '-' + kind)
            directory = WORK / 'lifecycle' / arm
            directory.mkdir(parents=True)
            text = command([build / (arm + '-lifecycle'), str(int(arm == 'candidate'))], directory, 'lifecycle-' + arm, native=True)
            if 'LIFECYCLE_PASS' not in text:
                raise ValueError('Lifecycle test did not complete')
        for active in (0, 1):
            for pair in range(binding['pairs']):
                for arm in (('original', 'candidate') if pair % 2 == 0 else ('candidate', 'original')):
                    directory = WORK / 'micro' / f'{active}-{pair}-{arm}'
                    directory.mkdir(parents=True)
                    text = command([build / (arm + '-micro'), str(active), str(binding['micro_clear_calls'])], directory, 'micro', native=True)
                    fields = next(line.split() for line in text.splitlines() if line.split()[:1] == ['MICRO'])
                    if len(fields) != 7:
                        raise ValueError('Malformed micro timing')
                    seconds = int(fields[-2]) / int(fields[-1])
                    if seconds <= 0:
                        raise ValueError('Nonpositive micro timing')
                    micro[str(active)][arm].append(seconds)
        save(WORK / 'micro.json', micro)

        full = {}
        for backend in binding['main_backend_ids']:
            with ExitStack() as stack:
                workers = {}
                references = {}
                calibration = {}
                for arm in ('original', 'candidate'):
                    instance, directory = worker(plain[arm], backend, f'full-{backend}-{arm}')
                    stack.enter_context(instance)
                    workers[arm] = instance, directory
                    value, raw = observation(instance, directory, backend, 100, 'calibrate')
                    values_equal(observed_values[backend, arm], raw)
                    references[arm] = raw
                    calibration[arm] = value
                values_equal(references['original'], references['candidate'])
                seconds = min(value['seconds'] for value in calibration.values())
                loops = min(binding['max_loops'], max(1, math.ceil(100 * binding['target_batch_seconds'] / seconds)))
                measurements = {'original': [], 'candidate': []}
                for pair in range(binding['pairs']):
                    for arm in (('original', 'candidate') if pair % 2 == 0 else ('candidate', 'original')):
                        instance, directory = workers[arm]
                        value, raw = observation(instance, directory, backend, loops, 'pair-' + str(pair))
                        values_equal(references[arm], raw)
                        measurements[arm].append(value['seconds'])
                full[str(backend)] = {'loops': loops, 'calibration': calibration,
                    'measurements': measurements, 'statistics': worker_module.paired_statistics(measurements['original'], measurements['candidate'])}
        save(WORK / 'summary.json', {'original_profile': profile, 'micro': micro, 'full_amplitude': full,
            'numerical_byte_parity': True, 'performance_acceptance_eligible': False,
            'source_changed_in_production': False, 'elapsed_seconds': time.monotonic() - started})
        verify_source()
        save(WORK / 'terminal.json', {'returncode': 0, 'elapsed_seconds': time.monotonic() - started})
    except BaseException as error:
        save(WORK / 'terminal.json', {'returncode': 1, 'elapsed_seconds': time.monotonic() - started,
            'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('verify', 'run'))
    args = parser.parse_args()
    if args.operation == 'verify':
        binding = verify_source()
        print(json.dumps({'verified': True, 'dependencies': len(binding['dependencies']), 'native_execution': False}))
    else:
        run()
