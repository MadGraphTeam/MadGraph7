#!/usr/bin/env python3
"""One capped observed replay per backend; require exact frozen-prototype parity."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

ROOT = Path.cwd().resolve()
HERE = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer-native'
OUT = HERE / 'attempt1'
WORK = ROOT / '.tirust-build/qcd-loop-kinematics-observer/attempt1'
BORN = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype'
PROC = ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg'
PHYSICAL = ROOT / 'docs/benchmarkfolder/experiments/physical-stress'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p):
    return str(p.relative_to(ROOT))


def rows(text):
    return [line.strip() for line in text.splitlines()
            if line.strip().startswith(('AMPLITUDE_', 'BORN_RESCUE_COUNTS', 'LOOP_NORMALIZATION'))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=int, choices=(8, 9), required=True)
    args = parser.parse_args()
    result = OUT / ('backend-' + str(args.backend))
    assert not result.exists(), 'No automatic native retry or replacement.'
    result.mkdir()
    build = json.loads((OUT / 'build.json').read_text())
    assert build['complete'] and build['protected_inputs_unchanged'] and not build['native_executed']
    checks = dict(build['protected_inputs_before'])
    checks.update(build['artifacts_sha256'])
    checks[rel(OUT / 'build.json')] = sha(OUT / 'build.json')
    checks[rel(HERE / 'run.py')] = sha(HERE / 'run.py')
    old_dir = BORN / 'amplitude-native-1' / ('backend-' + str(args.backend))
    born_index_path = BORN / 'evidence-index.json'
    assert sha(born_index_path) == 'd7ace06aad2ae7850f2bf97cc9d48001ca9d2f4c1a8a02a349f975a97c7c0137'
    checks[rel(born_index_path)] = sha(born_index_path)
    born_index = json.loads(born_index_path.read_text())
    for path in (old_dir / 'execution.json', old_dir / 'prototype.log'):
        entry = born_index['files'][str(path.relative_to(BORN))]
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], str(path)
    old = json.loads((old_dir / 'execution.json').read_text())
    assert old['complete'] and old['protected_inputs_unchanged']
    prior_run = next(run for run in old['runs'] if run['mode'] == 'prototype')
    old_log = old_dir / 'prototype.log'
    assert sha(old_log) == prior_run['log_sha256']
    checks[rel(old_dir / 'execution.json')] = sha(old_dir / 'execution.json')
    checks[rel(old_log)] = sha(old_log)
    card = PHYSICAL / f'runs/qcd_n3_uux_zgg/candidate-{args.backend}-auto/MadLoopParams.dat'
    assert sha(card) == old['protected_inputs_before'][rel(card)], 'Frozen numerical card changed.'
    checks[rel(card)] = sha(card)
    parameters = dict(re.findall(r'^#(\w+)\s*\n([^!#\n]+)', card.read_text(), flags=re.M))
    assert parameters['MLReductionLib'].strip() == str(args.backend)
    for key, value in {'CTModeRun': '-1', 'MLStabThres': '1.000000d-06', 'OSThres': '1.000000d-08',
                       'ImprovePSPoint': '2', 'TIRustComputeFinite': '.true.'}.items():
        assert parameters[key].strip().lower() == value.lower(), (key, parameters[key])
    record = {'schema': 1, 'complete': False, 'backend': args.backend, 'cpu': 11,
              'timeout_per_process_seconds': 180, 'formal_performance_eligible': False,
              'scope': 'One observed ordinary/stress/ordinary replay; exact prototype outputs/flags/420 required.',
              'protected_inputs_before': checks, 'reference_log_sha256': sha(old_log)}
    env = os.environ.copy()
    native = [ROOT / 'HEPTools/tirust/lib', ROOT / 'HEPTools/ninja/lib', ROOT / 'HEPTools/collier']
    env['LD_LIBRARY_PATH'] = ':'.join(map(str, native)) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    with (ROOT / '.tirust-build/numerical-tests.lock').open('a+') as lock:
        record['lock_wait_started_unix'] = time.time()
        fcntl.flock(lock, fcntl.LOCK_EX)
        record['lock_acquired_unix'] = time.time()
        try:
            for name, digest in checks.items():
                assert sha(ROOT / name) == digest, name
            private = WORK / ('backend-' + str(args.backend))
            private.mkdir()
            resources = private / 'MadLoop5_resources'
            resources.mkdir()
            for name in ('ML5_0_ColorDenomFactors.dat', 'ML5_0_ColorNumFactors.dat', 'ML5_0_HelConfigs.dat'):
                source = PROC / 'SubProcesses/MadLoop5_resources' / name
                assert sha(source) == old['protected_inputs_before'][rel(source)]
                checks[rel(source)] = sha(source)
                shutil.copyfile(source, resources / name)
            shutil.copyfile(card, resources / 'MadLoopParams.dat')
            runtime = subprocess.check_output(['ldd', str(WORK / 'amplitude-observed')], cwd=ROOT, env=env, text=True)
            (result / 'ldd.txt').write_text(runtime)
            runtime_files = {}
            for line in runtime.splitlines():
                match = re.search(r'(?:=>\s*)?(/\S+)\s+\(', line)
                if match:
                    path = Path(match[1]).resolve()
                    runtime_files[str(path)] = sha(path)
            actual_dso = next(line.split('=>', 1)[1].split('(')[0].strip() for line in runtime.splitlines()
                              if 'libmadloop_tirust.so =>' in line)
            assert Path(actual_dso).resolve() == (ROOT / 'HEPTools/tirust/lib/libmadloop_tirust.so').resolve()
            assert sha(Path(actual_dso)) == '49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'
            record['runtime_tirust_path'] = actual_dso
            record['runtime_sha256_before'] = runtime_files
            command = ['taskset', '-c', '11', str(WORK / 'amplitude-observed'),
                       str(WORK / 'param_card.dat'), str(WORK / 'points.dat'), str(resources), str(result / 'trace.dat')]
            record['command'] = command
            record['host_load_start'] = os.getloadavg()
            start = time.monotonic()
            try:
                with (result / 'observed.log').open('x') as log:
                    completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, timeout=180)
                record['returncode'] = completed.returncode
                assert completed.returncode == 0, 'Observed process failed.'
            finally:
                record['wall_seconds'] = time.monotonic() - start
                record['host_load_end'] = os.getloadavg()
                record['log_sha256'] = sha(result / 'observed.log')
                record['runtime_sha256_after'] = {name: sha(Path(name)) for name in runtime_files}
                assert record['runtime_sha256_after'] == runtime_files
                record['resource_sha256'] = {rel(p): sha(p) for p in sorted(resources.iterdir()) if p.is_file()}
            output = (result / 'observed.log').read_text()
            observed, expected = rows(output), rows(old_log.read_text())
            record['observed_numeric_rows'] = observed
            record['reference_numeric_rows'] = expected
            record['all_original_rows_exact'] = observed == expected
            assert record['all_original_rows_exact'], 'Observer changed original numeric/status/IEEE output rows.'
            assert 'AMPLITUDE_DIAGNOSTIC_COMPLETE' in observed
            values = [line for line in observed if line.startswith('AMPLITUDE_VALUE')]
            assert len(values) == 6
            stress = [line.split() for line in values if line.split()[1] == '2']
            assert len(stress) == 2 and all(int(fields[4]) == 420 for fields in stress)
            health = [tuple(map(int, line.split()[1:])) for line in output.splitlines() if line.startswith('QO_HEALTH ')]
            assert health == [(1, 0, 0), (2, 0, 0), (3, 0, 0)], health
            record['health'] = health
            trace = result / 'trace.dat'
            record['trace_sha256'] = sha(trace)
            record['trace_bytes'] = trace.stat().st_size
            record['complete'] = True
            record['interpretation_ready'] = False
            record['remaining_gate'] = 'Independent complete trace parsing/metadata/payload validation before interpretation.'
        except BaseException as exc:
            record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            record['protected_inputs_after'] = {name: sha(ROOT / name) for name in checks}
            record['protected_inputs_unchanged'] = record['protected_inputs_after'] == checks
            (result / 'execution.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
            assert record['protected_inputs_unchanged']
    print(f'Backend {args.backend}: exact frozen prototype rows and zero observer health counters; trace analysis pending.')


if __name__ == '__main__':
    main()
