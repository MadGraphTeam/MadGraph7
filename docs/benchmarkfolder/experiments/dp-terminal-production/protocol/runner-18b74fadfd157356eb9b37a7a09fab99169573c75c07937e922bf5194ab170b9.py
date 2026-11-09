#!/usr/bin/env python3
"""Fresh CLI proof coverage and strict CutTools replay; never a timing gate."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'tools'), str(ROOT)]
import tirust_benchmark as bench

CASES = ('qcd_n1_uux_z', 'qcd_n2_gg_ttx', 'qcd_n3_uux_zgg',
         'ew_n1_udx_wp', 'ew_n2_epem_mupmum')
SOURCE = ('madgraph/loop/loop_exporters.py', 'madgraph/loop/dp_terminal_proof.py',
          'madgraph/various/q_polynomial.py', 'madgraph/iolibs/helas_call_writers.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, action='append')
    parser.add_argument('--build-n1', action='store_true')
    parser.add_argument('--observe-n1', action='store_true')
    parser.add_argument('--cpu', type=int, default=10)
    args = parser.parse_args()
    runner = Path(__file__).read_bytes()
    runner_hash = hashlib.sha256(runner).hexdigest()
    snapshot = HERE/'protocol'/('runner-'+runner_hash+'.py')
    snapshot.parent.mkdir(exist_ok=True)
    snapshot.write_bytes(runner)
    bench.dump(HERE/'protocol'/('invocation-'+runner_hash+'.json'), {
        'argv': sys.argv, 'started_utc': bench.utc_now(), 'runner_sha256': runner_hash,
        'snapshot': str(snapshot.relative_to(HERE)), 'performance_acceptance_eligible': False})
    config = json.loads((ROOT/'docs/benchmarkfolder/cases.json').read_text())
    bench.WORK = ROOT/'.tirust-build/dp-terminal-production'
    bench.REPORT = HERE
    bench.MODEL_ROOT = ROOT/'.benchmark-work/tirust/models'
    model_pin = ROOT/'docs/benchmarkfolder/inputs/loop_qcd_qed_sm_Gmu.json'
    private_pin = HERE/'inputs'/model_pin.name
    private_pin.parent.mkdir(parents=True, exist_ok=True)
    if private_pin.exists() and private_pin.read_bytes() != model_pin.read_bytes():
        raise RuntimeError('Private electroweak model pin changed')
    if not private_pin.exists():
        shutil.copy2(model_pin, private_pin)
    original_run = bench.run
    def observed_run(command, cwd, log, **kwargs):
        if len(command) > 1 and command[1] == str(ROOT/'bin/madgraph'):
            if any(line.startswith('set ') and not line.endswith(' --no_save')
                   for line in Path(command[-1]).read_text().splitlines()):
                raise RuntimeError('Private CLI settings must explicitly suppress config persistence')
            case_id = Path(command[-1]).stem.removeprefix('candidate-')
            command = [command[0], str(HERE/'capture_cli.py'),
                       str(HERE/'cli'/case_id/'proof-decisions.json'), *command[2:]]
        return original_run(command, cwd, log, **kwargs)
    bench.run = observed_run
    source_before = {name: bench.digest(ROOT/name) for name in SOURCE}
    generator_before = bench.source_identity(ROOT, persist=False)['files_sha256']
    configuration = ROOT/'input/mg7_configuration.txt'
    for case_id in args.case or CASES:
        case = next(case for case in config['cases'] if case['id'] == case_id)
        destination = HERE/'cli'/case_id
        destination.mkdir(parents=True, exist_ok=True)
        existing = (bench.output_dir('candidate', case)/'generation.json').exists()
        config_before = bench.digest(configuration)
        try:
            generation = bench.generate('candidate', case)
        finally:
            config_after = bench.digest(configuration)
            if not existing:
                bench.dump(destination/'main-configuration-guard.json', {
                    'path': str(configuration.relative_to(ROOT)), 'before_sha256': config_before,
                    'after_sha256': config_after, 'unchanged': config_before == config_after})
        if config_before != config_after:
            raise RuntimeError('Main configuration changed during private CLI export')
        if generation['source_identity']['files_sha256'] != generator_before:
            raise RuntimeError('Cached/exported source identity differs from current generator bytes')
        if existing:
            coverage = json.loads((destination/'coverage.json').read_text())
            if coverage['production_sources_sha256'] != source_before:
                raise RuntimeError('Cached coverage belongs to another DP production source')
            print(case_id, coverage['counts'], '(retained generation)', flush=True)
            continue
        shutil.copy2(bench.WORK/'logs'/f'generate-candidate-{case_id}.log', destination/'generate.log')
        output = bench.output_dir('candidate', case)
        directories = sorted(output.glob('SubProcesses/P*_*'))
        decisions = json.loads((destination/'proof-decisions.json').read_text())
        selected = {entry['directory']: entry for entry in decisions}
        counts = [{'process': p.name, **{key: selected.get(str(p), {}).get(key, 0)
                    for key in ('proposed', 'applied', 'declined')}} for p in directories]
        bench.dump(destination/'coverage.json', {'case': case, 'counts': counts,
            'generated_sources_sha256': bench.generated_sources(output),
            'production_sources_sha256': source_before,
            'parameter_card_sha256': generation['parameter_card_sha256'],
            'source_only': True, 'performance_acceptance_eligible': False})
        for directory in directories:
            for pattern in ('coef_construction_*.f', 'polynomial.f'):
                for path in directory.glob(pattern):
                    target = destination/'generated'/directory.name/path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
        print(case_id, counts, flush=True)
    source_after = {name: bench.digest(ROOT/name) for name in SOURCE}
    if (source_before != source_after or generator_before !=
            bench.source_identity(ROOT, persist=False)['files_sha256']):
        raise RuntimeError('DP production sources changed during CLI coverage')
    n1 = next(case for case in config['cases'] if case['id'] == CASES[0])
    if args.build_n1:
        bench.build('candidate', n1, jobs=1)
        bench.dump(HERE/'n1-build-object-sha256.json', bench.numerical_objects(bench.output_dir('candidate', n1)))
        from madgraph.loop.dp_terminal_proof import validate_dp_terminal_wavefunctions
        decisions = json.loads((HERE/'cli'/CASES[0]/'proof-decisions.json').read_text())
        proofs = []
        for entry in decisions:
            expected = {int(key): value for key, value in entry['expected'].items()}
            passed = validate_dp_terminal_wavefunctions(entry['directory'], entry['prefix'], expected)
            proofs.append({'directory': entry['directory'], 'passed': passed})
        bench.dump(HERE/'n1-proof-after-worker-build.json', proofs)
        if not proofs or not all(item['passed'] for item in proofs):
            raise RuntimeError('Fresh benchmark worker no longer satisfies the source consumer proof')
    if args.observe_n1:
        reference = ROOT/'docs/benchmarkfolder/experiments/numerical/strict-policy-qcd-n1/baseline-1'
        points = ROOT/'docs/benchmarkfolder/points/qcd_n1_uux_z.dat'
        evidence = HERE/'n1-cuttools-replay'
        reference_inputs = json.loads((reference/'baseline-inputs.json').read_text())
        observation = json.loads((reference/'raw/observation.json').read_text())
        if bench.digest(points) != reference_inputs['points_sha256']:
            raise RuntimeError('Frozen CutTools reference uses different points')
        with ExitStack() as stack:
            worker, settings = bench.start_worker(stack, 'candidate', n1, [1], config,
                args.cpu, evidence, timeout=180., points_path=points)
            if settings['mlstabthres'] != 1.e-6:
                raise RuntimeError('Expected strict 1e-6 host stability checks')
            actual = bench.observe(worker, [1], 1, evidence/'raw', count=100)
        inputs = json.loads((evidence/'candidate-inputs.json').read_text())
        if inputs['param_card_sha256'] != reference_inputs['param_card_sha256']:
            raise RuntimeError('Physical parameter card differs from CutTools reference')
        differences = []
        for left, right in zip(observation['rows'], actual['rows'], strict=True):
            for column, (a, b) in enumerate(zip(left['coefficients'], right['coefficients'], strict=True)):
                if struct.pack('!d', a) != struct.pack('!d', b):
                    differences.append({'point': left['point'], 'order': left['order'], 'column': column,
                                        'reference': a, 'candidate': b})
        bench.dump(evidence/'comparison.json', {
            'reference': str(reference.relative_to(ROOT)),
            'reference_input_sha256': bench.digest(reference/'baseline-inputs.json'),
            'reference_observation_sha256': bench.digest(reference/'raw/observation.json'),
            'candidate_observation_sha256': bench.digest(evidence/'raw/observation.json'),
            'point_count': 100, 'strict_host_threshold': 1e-6,
            'bitwise_equal': not differences, 'bitwise_differences': differences,
            'comparison': bench.compare_rows(observation['rows'], actual['rows'], tolerance=1e-6),
            'performance_acceptance_eligible': False})
        print('CutTools replay comparison retained', flush=True)


if __name__ == '__main__':
    main()
