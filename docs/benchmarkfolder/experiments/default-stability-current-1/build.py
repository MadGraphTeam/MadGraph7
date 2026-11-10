"""Build the retained synthetic host fixture with current sources; no integrals."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'docs/benchmarkfolder/qualification/tirust-bridge/stability-audit'
BUILD = ROOT / '.tirust-build/default-stability-current-1'
BRIDGE = ROOT / '.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib/libmadloop_tirust.so'
EXPECTED = 'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, OLD / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    assert Path.cwd() == ROOT
    assert digest(BRIDGE) == EXPECTED
    assert not BUILD.exists(), 'A build attempt may not be overwritten.'
    os.sched_setaffinity(0, {14})
    BUILD.mkdir(parents=True)
    sys.dont_write_bytecode = True
    start = time.monotonic()
    record = {'complete': False, 'native_evaluation': False, 'cpu': 14,
              'timeout_seconds': 120, 'bridge_sha256': EXPECTED,
              'launcher_sha256': digest(__file__), 'commands': []}

    def command(args, directory):
        remaining = 120 - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError('Combined compiler deadline')
        with (directory / 'build.log').open('ab') as log:
            log.write((json.dumps(args) + '\n').encode())
            log.flush()
            child = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                output, _ = child.communicate(timeout=remaining)
            except BaseException:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate()
                raise
            log.write(output)
        record['commands'].append({'argv': args, 'exit_code': child.returncode})
        if child.returncode:
            raise RuntimeError(output.decode(errors='replace'))
        return output.decode()

    try:
        adapter = load('current_stability_adapter_builder', 'build_harness.py')
        adapter.BUILD = BUILD / 'adapter'
        adapter.BRIDGE = BRIDGE
        adapter.run = lambda args: command(args, adapter.BUILD)
        adapter.main()
        host = load('current_stability_host_builder', 'build_host_rescue.py')
        host.BUILD = BUILD / 'host'
        host.NATIVE = adapter.BUILD / 'libtirust_stability_audit.so'
        host.GENERATED = ROOT / '.benchmark-work/tirust/campaigns/split-dp-terminal-1/outputs/candidate/qcd_n1_uux_z/SubProcesses/P0_uux_z/loop_matrix.f'
        host.run = lambda args: command(args, host.BUILD)
        sys.argv = [str(OLD / 'build_host_rescue.py')]
        host.main()
        record['builder_sha256'] = {str(OLD / f): digest(OLD / f) for f in
                                   ('build_harness.py', 'build_host_rescue.py')}
        record['build_metadata'] = {name: json.loads((BUILD / name / 'build-metadata.json').read_text())
                                    for name in ('adapter', 'host')}
        assert digest(BRIDGE) == EXPECTED
        record['complete'] = True
    except BaseException as error:
        record['failure'] = repr(error)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        (BUILD / 'build.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'complete': True, 'commands': len(record['commands']),
                      'build_record_sha256': digest(BUILD / 'build.json')}))


if __name__ == '__main__':
    main()
