#!/usr/bin/env python3
"""Compare original/variant n1 amplitudes for all five explicit reducers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

from run_common import ROOT, HERE, FORTRAN, campaign, clone, digest, dump, MadLoopParam
from tirust_benchmark_worker import Worker, parse_outputs


def packed_rows(rows):
    return b''.join(struct.pack('!iiiddddd', row['point'], row['order'], row['status'],
                                *row['coefficients'], row['accuracy']) for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--fortran-build-metadata', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--build-metadata', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=9)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name) or args.cpu not in os.sched_getaffinity(0):
        raise ValueError('Invalid name/CPU')
    build = json.loads(args.fortran_build_metadata.read_text())
    if not build['built'] or build['changed_objects'] != ['TIR_interface.o']:
        raise ValueError('Require a single-common-interface build')
    library = args.library.resolve(strict=True)
    if digest(library) != json.loads(args.build_metadata.read_text())['library_sha256']:
        raise ValueError('DSO build identity differs')
    original = ROOT/'.benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z'
    frozen_path = ROOT/'docs/benchmarkfolder/runs/qcd_n1_uux_z/tirust-first-attempt-3/candidate-inputs.json'
    frozen = json.loads(frozen_path.read_text())
    if build['original_executable_sha256'] != frozen['compilation']['executable_sha256']:
        raise ValueError('Build derives from another frozen executable')
    if digest(original/'SubProcesses/P0_uux_z/check') != build['original_executable_sha256']:
        raise ValueError('Original executable changed')
    variant = Path(build['output'])
    for name, expected in build['objects_after'].items():
        if digest(variant/'SubProcesses/P0_uux_z'/name) != expected:
            raise ValueError('Variant object changed: '+name)
    if digest(variant/'SubProcesses/P0_uux_z/check') != build['executable_sha256']:
        raise ValueError('Variant executable changed')
    if digest(variant/'SubProcesses/P0_uux_z/TIR_interface.f') != build['source_sha256']:
        raise ValueError('Variant source changed')
    points = ROOT/'docs/benchmarkfolder/points/qcd_n1_uux_z.dat'
    if digest(points) != frozen['points_sha256']:
        raise ValueError('Physical points changed')
    report = HERE/args.name
    report.mkdir(exist_ok=False)
    work = ROOT/'.tirust-build/n1-profile/fortran-common-allocation'/args.name
    work.mkdir(parents=True, exist_ok=False)
    result = {'scope': __doc__, 'performance_eligible': False,
              'cpu': args.cpu, 'machine': campaign.machine_identity(args.cpu),
              'runner_sha256': digest(__file__), 'points_sha256': digest(points),
              'fortran_build_metadata_sha256': digest(args.fortran_build_metadata),
              'frozen_inputs_sha256': digest(frozen_path),
              'library': {'path': str(library), 'sha256': digest(library)},
              'runs': [], 'comparisons': []}
    shutil.copy2(__file__, report/'runner-used.py')
    shutil.copy2(args.fortran_build_metadata, report/'fortran-build-metadata.json')
    shutil.copy2(args.build_metadata, report/'bridge-build-metadata.json')
    expected_runtime = dict(frozen['runtime']['libraries'])
    old = [p for p in expected_runtime if Path(p).name == 'libmadloop_tirust.so']
    if len(old) != 1:
        raise ValueError('Expected one historical TIRust DSO')
    del expected_runtime[old[0]]
    expected_runtime[str(library)] = digest(library)
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for backend in (1, 6, 7, 8, 9):
            observations = {}
            for label, source in [('original', original), ('variant', variant)]:
                target = report/f'{backend}-{label}'
                target.mkdir()
                output = work/f'{backend}-{label}'
                clone(source, output)
                directory = output/'SubProcesses/P0_uux_z'
                if digest(output/'Cards/param_card.dat') != frozen['param_card_sha256']:
                    raise ValueError('Physical parameter card changed')
                card_path = output/'Cards/MadLoopParams.dat'
                card = MadLoopParam(str(card_path))
                for key, value in frozen['settings'].items():
                    card[key] = value
                card['mlreductionlib'] = str(backend)
                card.write(str(card_path), template=str(card_path))
                for local in (output/'SubProcesses').rglob('MadLoopParams.dat'):
                    if local.is_symlink():
                        local.unlink()
                    shutil.copy2(card_path, local)
                for pattern in ('*HelFilter.dat', '*LoopFilter.dat'):
                    for local in (output/'SubProcesses').rglob(pattern):
                        local.unlink()
                shutil.copy2(card_path, target/'MadLoopParams.dat')
                env = campaign.environment()
                env['LD_LIBRARY_PATH'] = str(library.parent)+os.pathsep+env.get('LD_LIBRARY_PATH', '')
                env.pop('ML_TIRUST_FORTRAN_PROFILE', None)
                env.pop('ML_TIRUST_BRIDGE_PROFILE', None)
                linkage = subprocess.check_output(['ldd', str(directory/'check')], env=env, text=True)
                (target/'ldd.txt').write_text(linkage)
                runtime = {name: digest(name) for name in re.findall(r'(?:=>\s*)?(/[^\s]+)', linkage)
                           if Path(name).is_file()}
                if runtime != expected_runtime:
                    raise ValueError('Loaded dependency identities differ')
                dump(target/'runtime-identity.json', runtime)
                with Worker(['taskset', '-c', str(args.cpu), './check', str(points)],
                            directory, target/'worker.log', env=env, timeout=60.) as worker:
                    worker.run(1)
                    for filename in ('timing.dat', 'values.dat', 'timed_status.dat', 'dimensions.dat'):
                        shutil.copy2(directory/filename, target/filename)
                    observation = parse_outputs(target, [backend], 100, 1)
                observations[label] = observation
                row = {'backend': backend, 'variant': label, 'observation': observation,
                       'executable_sha256': digest(directory/'check'),
                       'driver_sha256': digest(directory/'check_sa.f')}
                result['runs'].append(row)
                dump(target/'observation.json', row)
                dump(report/'results.json', result)
            left, right = (observations[x] for x in ('original', 'variant'))
            comparison = {'backend': backend, 'points': 100,
                          'rows_status_accuracy_bit_identical': packed_rows(left['rows']) == packed_rows(right['rows']),
                          'status_histograms_identical': left['timed_status_counts'] == right['timed_status_counts'],
                          'original_usage': left['backend_precision_counts'],
                          'variant_usage': right['backend_precision_counts']}
            result['comparisons'].append(comparison)
            dump(report/'results.json', result)
            if not comparison['rows_status_accuracy_bit_identical'] or not comparison['status_histograms_identical']:
                raise RuntimeError('Native backend parity failed; original evidence retained')
    result['passed'] = True
    dump(report/'results.json', result)
    print(json.dumps({'passed': True, 'backends': 5, 'original_variant_pairs': 500,
                      'performance_eligible': False}))


if __name__ == '__main__':
    main()
