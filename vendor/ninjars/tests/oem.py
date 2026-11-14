#!/usr/bin/env python3
"""Validate concurrent NinjaRS Fortran use without a personal Symbolica key.

Each process evaluates analytic DP/MP controls, waits with its numerical worker
alive until every peer is ready, and then finalizes and recreates that worker.
Build products, logs and the machine-readable receipt stay under --output.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import threading
import time


HERE = Path(__file__).resolve().parent
MARKER = re.compile(r'^OEM_(READY|ACTIVE|DONE) ([0-9]+)$|^OEM_PASSED$')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    lib, output = args.lib.resolve(), args.output.resolve()
    if output == HERE or HERE in output.parents:
        raise ValueError('--output must be outside the retained test-source directory')
    output.mkdir(parents=True, exist_ok=True)
    source = HERE / 'oem_lifecycle.f90'
    executable = output / 'oem_lifecycle'
    command = shlex.split(args.fc) + ['-O2', '-fcheck=all', str(source), '-I' + str(lib)]
    command += shlex.split((lib / 'ninjars-link-before.txt').read_text())
    command += ['-L' + str(lib), str(lib / 'libninjars.a')]
    command += shlex.split((lib / 'ninjars-native-libs.txt').read_text())
    command += ['-o', str(executable)]
    with (output / 'build.log').open('w') as stream:
        subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT, check=True)

    env = os.environ.copy()
    env.pop('SYMBOLICA_LICENSE', None)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    events = queue.Queue()
    children, readers, records = [], [], []
    start = time.monotonic()
    summary = dict(passed=False, personal_symbolica_license_removed=True,
                   processes=args.processes, cycles=args.cycles,
                   source_sha256=sha(source), library_sha256=sha(lib / 'libninjars.a'),
                   build_command=command, barriers=records)
    config = lib / 'ninjars-config.json'
    if config.is_file():
        summary['installation_sha256'] = sha(config)

    def collect(index, child):
        with (output / ('process-%d.log' % index)).open('w') as stream:
            for line in child.stdout:
                stream.write(line)
                stream.flush()
                events.put((index, line.rstrip('\r\n')))
        events.put((index, None))

    def await_stage(stage, cycle=None):
        expected = 'OEM_' + stage + ('' if cycle is None else ' ' + str(cycle))
        arrived = set()
        while len(arrived) != args.processes:
            remaining = args.timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError('Timed out waiting for ' + expected)
            try:
                index, line = events.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError('Timed out waiting for ' + expected) from exc
            if line is None:
                # A completed peer may exit after its final success marker
                # while the other peers are still finishing.
                if stage == 'PASSED' and index in arrived:
                    continue
                raise RuntimeError('Process %d exited before %s; see its log' % (index, expected))
            if line.startswith(('OEM_FAILED', 'OEM_STATUS')):
                raise RuntimeError('Process %d: %s' % (index, line))
            if not MARKER.fullmatch(line):
                continue
            if line != expected or index in arrived:
                raise RuntimeError('Unexpected marker from process %d: %s (wanted %s)' %
                                   (index, line, expected))
            arrived.add(index)
            records.append(dict(process=index, marker=line, seconds=time.monotonic() - start))

    try:
        for index in range(args.processes):
            child = subprocess.Popen([str(executable), str(args.cycles)], cwd=output,
                                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            children.append(child)
            reader = threading.Thread(target=collect, args=(index, child), daemon=True)
            reader.start()
            readers.append(reader)
        for cycle in range(1, args.cycles + 1):
            for stage in ('READY', 'ACTIVE', 'DONE'):
                await_stage(stage, cycle)
                for child in children:
                    child.stdin.write('GO\n')
                    child.stdin.flush()
        await_stage('PASSED')
        for child in children:
            remaining = max(0.1, args.timeout - (time.monotonic() - start))
            if child.wait(timeout=remaining) != 0:
                raise RuntimeError('A process returned a nonzero exit status after its success marker')
        summary['passed'] = True
    except Exception as exc:
        summary['error'] = str(exc)
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
            with contextlib.suppress(OSError):
                child.stdin.close()
        for reader in readers:
            reader.join(timeout=5)
        for child in children:
            with contextlib.suppress(OSError):
                child.stdout.close()
        summary['returncodes'] = [child.returncode for child in children]
        summary['elapsed_seconds'] = time.monotonic() - start
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ('barriers', 'build_command')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lib', type=Path, required=True, help='installed NinjaRS lib directory')
    parser.add_argument('--output', type=Path, required=True, help='disposable build and evidence directory')
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=600, help='runtime timeout in seconds')
    parser.add_argument('--fc', default=os.environ.get('FC', 'gfortran'))
    args = parser.parse_args()
    if args.processes < 2 or args.cycles < 2 or args.timeout <= 0:
        parser.error('require at least two processes, two cycles and a positive timeout')
    run(args)


if __name__ == '__main__':
    main()
