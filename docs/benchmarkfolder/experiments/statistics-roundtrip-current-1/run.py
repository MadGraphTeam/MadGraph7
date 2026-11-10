#!/usr/bin/env python3
"""One bounded current-source statistics qualification; no numerical reducers.

Run from the repository root after sourcing tools/tirust_env.sh. The selected
module compiles/runs four small production tally/writer fragments with synthetic
return codes. It never calls a MadLoop amplitude or a Rust reduction library.
Existing result paths prevent a second attempt in this evidence namespace.
"""
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
REPORT = ROOT / 'docs/benchmarkfolder/experiments/statistics-roundtrip-current-1'
WORK = ROOT / '.tirust-build/statistics-roundtrip-current-1'
TEST = 'tests/unit_tests/various/test_ninjars_statistics.py'
SOURCES = (
    TEST, 'madgraph/__init__.py', 'madgraph/iolibs/file_writers.py',
    'madgraph/madevent/sum_html.py', 'madgraph/interface/amcatnlo_run_interface.py',
    'Template/LO/Source/dsample.f',
    'madgraph/iolibs/template_files/matrix_loop_induced_madevent.inc',
    'madgraph/iolibs/template_files/matrix_loop_induced_madevent_group.inc',
    'Template/NLO/SubProcesses/BinothLHA.f',
    'Template/NLO/SubProcesses/driver_mintFO.f',
    'Template/NLO/SubProcesses/driver_mintMC.f',
    'tools/tirust_env.sh', 'flake.nix', 'flake.lock',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    with (REPORT / name).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def tool(name, arguments):
    path = shutil.which(name)
    if not path:
        raise RuntimeError('Missing required tool: ' + name)
    result = subprocess.run([path, *arguments], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=10, check=True)
    return {'path': path, 'resolved_path': str(Path(path).resolve()),
            'sha256': sha(path), 'version_argv': [path, *arguments],
            'version_output': result.stdout, 'returncode': result.returncode}


def main():
    if Path.cwd() != ROOT:
        raise RuntimeError('Start from the MadGraph repository root')
    if (REPORT / 'before.json').exists() or WORK.exists():
        raise RuntimeError('Attempt already exists; no automatic retry')
    if 12 not in os.sched_getaffinity(0):
        raise RuntimeError('Requested CPU12 is not available')
    WORK.mkdir()
    temporary = WORK / 'tmp'
    temporary.mkdir()
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(temporary),
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    tools = {'gfortran': tool('gfortran', ['--version']),
             'taskset': tool('taskset', ['--version']),
             'python': tool('python3', ['--version'])}
    modules = {}
    for name in ('pytest', 'unittest'):
        path = importlib.util.find_spec(name).origin
        modules[name] = {'path': path, 'sha256': sha(path)}
    command = [tools['taskset']['path'], '-c', '12', sys.executable,
               '-m', 'pytest', '-v', '-p', 'no:cacheprovider', TEST]
    before = {
        'schema_version': 1, 'runner_sha256': sha(__file__),
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'cwd': str(ROOT), 'argv': command, 'cpu': 12, 'timeout_seconds': 120,
        'environment_setup': 'source tools/tirust_env.sh',
        'environment': {key: environment[key] for key in (
            'PATH', 'TMPDIR', 'PYTHONDONTWRITEBYTECODE',
            'PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'PYTEST_ADDOPTS',
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        'tools': tools, 'python_modules': modules,
        'pytest_version': importlib.metadata.version('pytest'),
        'source_sha256': source_hashes(),
        'scope': 'Current statistics sources: synthetic return-code inputs, four compiled Fortran tally/writer fragment executions, production Python readers. No amplitude, Rust/reducer, timing-acceptance or historical execution claim.',
        'cleanup_policy': 'Own new process group; after120s send TERM, wait2s, then KILL if needed. Never retry a failed or timed-out run.',
    }
    save('before.json', before)
    started = time.monotonic()
    cleanup, timed_out = [], False
    returncode = None
    with (REPORT / 'pytest-output.txt').open('xb') as output:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            returncode = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timed_out = True
            for kind, deadline in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
                try:
                    os.killpg(process.pid, kind)
                    cleanup.append(kind.name)
                except ProcessLookupError:
                    cleanup.append('group already exited')
                try:
                    returncode = process.wait(timeout=deadline)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            # Kill only descendants in this owned session if a failing pytest
            # process left them behind; this never targets another worker.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                cleanup.append('TERM surviving owned process group')
            except ProcessLookupError:
                pass
            else:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.02)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                        cleanup.append('KILL surviving owned process group')
                    except ProcessLookupError:
                        pass
    after = source_hashes()
    after_modules = {name: sha(value['path']) for name, value in modules.items()}
    after_tools = {name: sha(value['path']) for name, value in tools.items()}
    raw = (REPORT / 'pytest-output.txt').read_text()
    passed = (returncode == 0 and not timed_out and not cleanup and after == before['source_sha256']
              and all(after_modules[name] == value['sha256'] for name, value in modules.items())
              and all(after_tools[name] == value['sha256'] for name, value in tools.items())
              and '10 passed' in raw and 'SKIPPED' not in raw)
    save('results.json', {
        'schema_version': 1, 'passed': passed, 'returncode': returncode,
        'timed_out': timed_out, 'elapsed_seconds': time.monotonic() - started,
        'cleanup': cleanup, 'source_sha256_after': after,
        'python_module_sha256_after': after_modules, 'tool_sha256_after': after_tools,
        'before_sha256': sha(REPORT / 'before.json'),
        'raw_output_sha256': sha(REPORT / 'pytest-output.txt'),
        'temporary_entries_after': sorted(str(path.relative_to(WORK)) for path in WORK.rglob('*')),
        'scope': before['scope'],
        'historical92_pass_invocation_recovered': False,
    })
    print(json.dumps({'passed': passed, 'returncode': returncode, 'timed_out': timed_out}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
