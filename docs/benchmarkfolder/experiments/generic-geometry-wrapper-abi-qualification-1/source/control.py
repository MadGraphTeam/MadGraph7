"""One-shot bounded wrapper/mock build or run. Scientific dispatch stays legacy.

Source/controller preparation does not authorize execution by itself. Root
reviews the sealed command/corpus stage before invoking this entry point.
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import time

import analyze
import prepare
import supervise

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / '.tirust-build/generic-geometry-wrapper-abi-qualification-1'
BUILD_FILES = ('aloha_object.f', 'invariant_bounds.f90', 'invariant_bounds_data.inc',
               'environment_token.c', 'wrapper.f', 'MadLoopParams.inc',
               'recording.f90', 'driver.f90')
OBJECTS = ('environment_token.o', 'aloha_object.o', 'invariant_bounds.o',
           'recording.o', 'wrapper.o', 'driver.o')


def identity(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def source_verified(wanted):
    if identity(HERE / 'index.json')['sha256'] != wanted:
        raise ValueError('source index mismatch')
    index = json.loads((HERE / 'index.json').read_bytes())
    for name, pin in index['files'].items():
        path = HERE / name
        if path.is_symlink() or identity(path) != pin:
            raise ValueError('source payload changed: ' + name)
    prepare.verify_source3()
    for name, pin in index['tools'].items():
        if identity(Path(name)) != pin:
            raise ValueError('tool changed: ' + name)
    if identity(Path(index['expected_interpreter']['path'])) != index['expected_interpreter']['identity']:
        raise ValueError('expected interpreter changed')
    return index


def elf_interpreter(path):
    data = path.read_bytes()
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('only ELF64 little-endian fixture is admitted')
    offset = struct.unpack_from('<Q', data, 32)[0]
    size, count = struct.unpack_from('<HH', data, 54)
    found = []
    for i in range(count):
        start = offset + i * size
        if struct.unpack_from('<I', data, start)[0] != 3:
            continue
        place = struct.unpack_from('<Q', data, start + 8)[0]
        length = struct.unpack_from('<Q', data, start + 32)[0]
        found.append(data[place:place + length].rstrip(b'\0').decode('ascii'))
    if len(found) != 1 or not Path(found[0]).is_absolute():
        raise ValueError('missing/ambiguous PT_INTERP')
    return found[0]


def clean_env(out):
    # Preserve tool search, but no LD_*, compiler flags or library overrides.
    env = {k: os.environ[k] for k in ('PATH', 'HOME') if k in os.environ}
    (out / 'tmp').mkdir()
    env.update(LC_ALL='C', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TMPDIR=str(out / 'tmp'), TMP=str(out / 'tmp'), TEMP=str(out / 'tmp'))
    return env


def interrupt(signum, frame):
    raise InterruptedError('controller interrupted: ' + str(signum))


def make_map_adapter(out, fifo, ready_log, interpreter):
    """Adapt the unchanged owned-group supervisor to a reducer-free fixture.

    The FIFO holds the worker before its case calls. The adapter records and
    hashes every observed file-backed ELF before releasing RUN; it makes no
    continuous mapping, kernel-vDSO, or scientific-provider qualification claim.
    """
    released = False
    def capture(pid, expected, executable, selected_library, maps_path, state):
        nonlocal released
        if released:
            raise RuntimeError('duplicate fixture release')
        if expected or selected_library:
            raise ValueError('mock fixture must not claim a scientific provider')
        if not ready_log.exists() or not ready_log.read_bytes().startswith(b'READY 240\n'):
            return None
        raw = Path(f'/proc/{pid}/maps').read_text()
        mapped = {}
        for line in raw.splitlines():
            fields = line.split(None, 5)
            if len(fields) != 6 or not fields[5].startswith('/'):
                continue
            name = fields[5]
            if name.endswith(' (deleted)'):
                raise ValueError('deleted mapped file')
            path = Path(name).resolve()
            with path.open('rb') as stream:
                is_elf = stream.read(4) == b'\x7fELF'
            if is_elf:
                mapped[str(path)] = identity(path)
        if str(Path(executable).resolve()) not in mapped or str(Path(interpreter).resolve()) not in mapped:
            return None
        if not any(Path(name).name.startswith('libgfortran.so') for name in mapped):
            return None
        if any(any(tag in Path(name).name.lower() for tag in ('tirust', 'ninjars', 'collier', 'cuttools'))
               for name in mapped if Path(name).resolve() != Path(executable).resolve()):
            raise ValueError('scientific library unexpectedly mapped by mock fixture')
        try:
            fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno == errno.ENXIO:
                return None
            raise
        try:
            maps_path.write_text(raw)
            if os.write(fd, b'RUN\n') != 4:
                raise RuntimeError('partial FIFO release')
        finally:
            os.close(fd)
        released = True
        state['released_once'] = True
        state['actual_file_backed_elf_count'] = len(mapped)
        return dict(pid=pid, maps_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    observed_file_backed_ELF=mapped, executable=executable,
                    actual_interpreter=interpreter, release='one RUN line after captured READY',
                    scope='single pre-case runtime snapshot; no scientific library or continuous monitor')
    return capture


def stage(action, wanted):
    started = time.monotonic()
    budget, reserve = (90, 15) if action == 'build' else (15, 5)
    deadline = started + budget
    out = WORK / action
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema='generic-wrapper-abi-stage-v1', stage=action, passed=False,
                   source_index_sha256=wanted, budget_seconds=budget,
                   cleanup_finalization_reserve_seconds=reserve, commands=[],
                   scientific_evaluations=0, full_sloop_lifetime_coverage=False)
    old = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        index = source_verified(wanted)
        env = clean_env(out)
        receipt['environment'] = env
        prefix = [index['compiler_paths']['prlimit'], '--as=4294967296',
                  '--fsize=536870912', '--core=0', '--']
        def command(argv, label):
            remaining = deadline - time.monotonic() - reserve
            if remaining <= 0:
                raise TimeoutError('stage work cutoff reached')
            record = dict(argv=argv, cwd=str(out), progress={})
            receipt['commands'].append(record)
            log_path = out / (label + '.log')
            try:
                with log_path.open('wb') as log:
                    code = supervise.run_plain_capped(argv, log, cwd=out, env=env,
                                                      progress=record['progress'], seconds=remaining)
                record['returncode'] = code
                if code != 0:
                    raise RuntimeError('command failed: ' + label)
            finally:
                record['log'] = identity(log_path) if log_path.exists() else None
        if action == 'build':
            for name in BUILD_FILES:
                shutil.copyfile(HERE / name, out / name)
            c = index['compiler_paths']['gcc']; f = index['compiler_paths']['gfortran']
            flags = ['-O2', '-Wall', '-Wextra', '-fcheck=all', '-fno-fast-math', '-fno-align-commons']
            command(prefix + [c, '-O2', '-Wall', '-Wextra', '-std=c11', '-fno-fast-math',
                              '-c', 'environment_token.c', '-o', 'environment_token.o'], '01-c')
            for number, name in enumerate(('aloha_object.f', 'invariant_bounds.f90',
                                           'recording.f90', 'wrapper.f', 'driver.f90'), 2):
                length = '-ffixed-line-length-none' if name.endswith('.f') else '-ffree-line-length-none'
                command(prefix + [f, *flags, length, '-c', name, '-o', str(Path(name).with_suffix('.o'))],
                        f'{number:02d}-' + Path(name).stem)
            command(prefix + [f, *flags, *OBJECTS, '-Wl,-Map,probe.map', '-o', 'probe'], '07-link')
            interpreter = elf_interpreter(out / 'probe')
            if interpreter != index['expected_interpreter']['path']:
                raise ValueError('actual PT_INTERP differs from reviewed interpreter')
            receipt['PT_INTERP'] = dict(path=interpreter, identity=identity(Path(interpreter)))
            receipt['executable'] = identity(out / 'probe')
            receipt['objects'] = {name: identity(out / name) for name in OBJECTS}
            receipt['sources'] = {name: identity(out / name) for name in BUILD_FILES}
        else:
            built = json.loads((WORK / 'build/result.json').read_bytes())
            if built['passed'] is not True or built['source_index_sha256'] != wanted:
                raise ValueError('closed matching build required')
            for name, pin in built['objects'].items():
                if identity(WORK / 'build' / name) != pin:
                    raise ValueError('build object changed')
            probe = WORK / 'build/probe'
            if identity(probe) != built['executable']:
                raise ValueError('built executable changed')
            interpreter = elf_interpreter(probe)
            if interpreter != index['expected_interpreter']['path']:
                raise ValueError('runtime interpreter changed')
            shutil.copyfile(HERE / 'inputs.txt', out / 'inputs.txt')
            fifo = out / 'start.fifo'; os.mkfifo(fifo, 0o600)
            log_path = out / 'native.log'
            capture = make_map_adapter(out, fifo, log_path, interpreter)
            original_capture = supervise.capture_maps
            record = dict(argv=prefix + [str(probe), str(out / 'inputs.txt'), str(fifo)],
                          cwd=str(out), progress={})
            receipt['commands'].append(record)
            try:
                supervise.capture_maps = capture
                remaining = deadline - time.monotonic() - reserve
                if remaining <= 0:
                    raise TimeoutError('run work cutoff reached')
                with log_path.open('wb') as log:
                    code, mapping = supervise.run_captured(record['argv'], log, cwd=out, env=env,
                        expected_runtime={}, executable=str(probe), selected_library='',
                        maps_path=out / 'runtime-maps.txt', progress=record['progress'], seconds=remaining)
                record['returncode'] = code
                if code != 0:
                    raise RuntimeError('native ABI fixture failed')
                for name, pin in mapping['observed_file_backed_ELF'].items():
                    if identity(Path(name)) != pin:
                        raise ValueError('observed runtime file changed after exit')
                mapping['observed_file_backed_ELF_rehashed_after_exit'] = True
                receipt['runtime'] = mapping
            finally:
                supervise.capture_maps = original_capture
                record['log'] = identity(log_path) if log_path.exists() else None
            receipt['qualification'] = analyze.parse(log_path.read_bytes(),
                json.loads((HERE / 'cases.json').read_bytes()), json.loads((HERE / 'expected.json').read_bytes()))
            if identity(out / 'inputs.txt') != identity(HERE / 'inputs.txt') or identity(probe) != built['executable']:
                raise ValueError('runtime input/executable changed')
            receipt['executable'] = built['executable']
        source_verified(wanted)
        if time.monotonic() > deadline:
            raise TimeoutError('aggregate stage budget exceeded during final verification')
        receipt['passed'] = True
    except BaseException as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        dump(out / 'result.json', receipt)
        for sig, handler in old.items():
            signal.signal(sig, handler)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('build', 'run'))
    parser.add_argument('--index-sha256', required=True)
    args = parser.parse_args()
    stage(args.stage, args.index_sha256)
