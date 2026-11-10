#!/usr/bin/env python3
"""Fresh higher-rank export versus the immutable 1x1-only QCD n3 control."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import struct
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'tools'), str(ROOT)]
import tirust_benchmark as bench

CASE = 'qcd_n3_uux_zgg'
SOURCES = ('madgraph/loop/loop_exporters.py', 'madgraph/loop/dp_terminal_proof.py',
           'madgraph/various/q_polynomial.py', 'madgraph/iolibs/helas_call_writers.py')
CONTROL = ROOT/'.tirust-build/dp-terminal-production/outputs/candidate'/CASE
CONTROL_EVIDENCE = ROOT/'docs/benchmarkfolder/experiments/dp-terminal-production'


def source_guard(expected):
    current = bench.source_identity(ROOT, persist=False)['files_sha256']
    if current != expected:
        raise RuntimeError('Generator source changed during qualification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('generate', 'build', 'observe'))
    parser.add_argument('--cpu', type=int, default=10)
    args = parser.parse_args()
    bench.WORK = ROOT/'.tirust-build/dp-higher-rank-production'
    bench.REPORT = HERE
    bench.MODEL_ROOT = ROOT/'.benchmark-work/tirust/models'
    config = json.loads((ROOT/'docs/benchmarkfolder/cases.json').read_text())
    config['madloop_parameters'].update(config['candidate_parameters'])
    config['madloop_parameters']['MLStabThres'] = 1e-6
    case = next(item for item in config['cases'] if item['id'] == CASE)
    stamp = bench.utc_now().replace(':', '').replace('.', '')
    protocol = HERE/'protocol'/stamp
    protocol.mkdir(parents=True)
    for name in ('run_qualification.py', 'capture_cli.py'):
        shutil.copyfile(HERE/name, protocol/name)
    for name in ('tirust_benchmark.py', 'tirust_benchmark_worker.py', 'tirust_benchmark_worker.f',
                 'tirust_benchmark_compiler.py'):
        shutil.copyfile(ROOT/'tools'/name, protocol/name)
    before = bench.source_identity(ROOT, persist=False)['files_sha256']
    bench.dump(protocol/'invocation.json', {'argv': sys.argv, 'started_utc': bench.utc_now(),
        'source_sha256': {name: bench.digest(ROOT/name) for name in SOURCES},
        'generator_sha256': before, 'cpu': args.cpu, 'performance_acceptance_eligible': False})
    if args.phase == 'generate':
        coverage = json.loads((CONTROL_EVIDENCE/'cli'/CASE/'coverage.json').read_text())
        if bench.generated_sources(CONTROL) != coverage['generated_sources_sha256']:
            raise RuntimeError('Immutable 1x1-only control sources changed')
        target = bench.output_dir('control', case)
        if target.exists():
            raise RuntimeError('Control output already exists; do not overwrite retained evidence')
        shutil.copytree(CONTROL, target, symlinks=True)
        bench.dump(HERE/'control-origin.json', {'source': str(CONTROL.relative_to(ROOT)),
            'qualified_control': str(CONTROL_EVIDENCE.relative_to(ROOT)),
            'coverage_sha256': bench.digest(CONTROL_EVIDENCE/'cli'/CASE/'coverage.json'),
            'generated_sources_sha256': coverage['generated_sources_sha256'],
            'generation_sha256': bench.digest(CONTROL/'generation.json'),
            'description': 'Immutable committed 1x1-only generated source; copied for a separate full source build.'})
        original_run = bench.run
        def observed_run(command, cwd, log, **kwargs):
            if len(command) > 1 and command[1] == str(ROOT/'bin/madgraph'):
                if any(line.startswith('set ') and not line.endswith(' --no_save')
                       for line in Path(command[-1]).read_text().splitlines()):
                    raise RuntimeError('All isolated CLI settings must suppress config persistence')
                command = [command[0], str(HERE/'capture_cli.py'),
                           str(HERE/'cli/proof-decisions.json'), *command[2:]]
            return original_run(command, cwd, log, **kwargs)
        bench.run = observed_run
        configuration = ROOT/'input/mg7_configuration.txt'
        old_hash = bench.digest(configuration)
        try:
            generation = bench.generate('candidate', case)
        finally:
            new_hash = bench.digest(configuration)
            bench.dump(HERE/'main-configuration-guard.json', {
                'before_sha256': old_hash, 'after_sha256': new_hash,
                'unchanged': old_hash == new_hash})
        if old_hash != new_hash or generation['source_identity']['files_sha256'] != before:
            raise RuntimeError('CLI configuration/source identity changed')
        shutil.copyfile(bench.WORK/'logs'/f'generate-candidate-{CASE}.log', HERE/'cli/generate.log')
        decisions = json.loads((HERE/'cli/proof-decisions.json').read_text())
        if len(decisions) != 1 or decisions[0]['applied'] != 147 or decisions[0]['declined']:
            raise RuntimeError('Expected all 147 terminal updates to pass the completed source proof')
        bench.dump(HERE/'generation-bindings.json', {'control': coverage['generated_sources_sha256'],
            'candidate': bench.generated_sources(bench.output_dir('candidate', case)),
            'candidate_generator_sha256': before})
    else:
        bindings = json.loads((HERE/'generation-bindings.json').read_text())
        if bindings['candidate_generator_sha256'] != before:
            raise RuntimeError('Candidate no longer belongs to the qualified source bytes')
        if args.phase == 'build':
            from madgraph.loop.dp_terminal_proof import validate_dp_terminal_wavefunctions
            control_decisions = json.loads((CONTROL_EVIDENCE/'cli'/CASE/'proof-decisions.json').read_text())
            candidate_decisions = json.loads((HERE/'cli/proof-decisions.json').read_text())
            proofs = {}
            for name, decisions in (('control', control_decisions), ('candidate', candidate_decisions)):
                if bench.generated_sources(bench.output_dir(name, case)) != bindings[name]:
                    raise RuntimeError('Generated sources changed before compilation: '+name)
                bench.build(name, case, jobs=1)
                expected = {int(key): value for key, value in decisions[0]['expected'].items()}
                proofs[name] = validate_dp_terminal_wavefunctions(bench.subprocess_dir(name, case),
                    decisions[0]['prefix'], expected)
                bench.dump(HERE/(name+'-objects.json'), bench.numerical_objects(bench.output_dir(name, case)))
            bench.dump(HERE/'proof-after-worker-build.json', proofs)
            if not all(proofs.values()):
                raise RuntimeError('Compiled worker source closure did not pass the consumer proof')
        else:
            points = ROOT/'docs/benchmarkfolder/points'/f'{CASE}.dat'
            observations, inputs = {}, {}
            for name in ('control', 'candidate'):
                report = HERE/'cuttools'/name
                with ExitStack() as stack:
                    worker, settings = bench.start_worker(stack, name, case, [1], config,
                        args.cpu, report, timeout=180., points_path=points)
                    if settings['mlstabthres'] != 1e-6:
                        raise RuntimeError('Expected strict 1e-6 host policy')
                    observations[name] = bench.observe(worker, [1], 1, report/'raw', count=100)
                inputs[name] = json.loads((report/(name+'-inputs.json')).read_text())
            if inputs['control']['settings'] != inputs['candidate']['settings']:
                raise RuntimeError('Numerical parameter settings differ')
            for key in ('param_card_sha256', 'points_sha256', 'points_metadata_sha256'):
                if inputs['control'][key] != inputs['candidate'][key]:
                    raise RuntimeError('Numerical input differs: '+key)
            differences = []
            for left, right in zip(observations['control']['rows'], observations['candidate']['rows'], strict=True):
                if (left['point'], left['order']) != (right['point'], right['order']):
                    raise RuntimeError('Observed point/order changed')
                if any(struct.pack('!d', a) != struct.pack('!d', b)
                       for a, b in zip(left['coefficients'], right['coefficients'], strict=True)):
                    differences.append({'control': left, 'candidate': right})
            status_equal = all(observations['control'][key] == observations['candidate'][key]
                               for key in ('status_counts', 'timed_status_counts', 'backend_precision_counts'))
            # Per-point raw statuses must also match, not only their histograms.
            status_equal = status_equal and all(
                all(left[key] == right[key] for key in ('status', 'actual_backend', 'precision'))
                for left, right in zip(observations['control']['rows'], observations['candidate']['rows'], strict=True))
            result = {'point_count': 100, 'bitwise_equal': not differences, 'differences': differences,
                'status_equal': status_equal, 'strict_host_threshold': 1e-6,
                'comparison': bench.compare_rows(observations['control']['rows'], observations['candidate']['rows'], 1e-6),
                'performance_acceptance_eligible': False}
            bench.dump(HERE/'cuttools/comparison.json', result)
            if differences or not status_equal:
                raise RuntimeError('Committed-control and candidate numerical replay differ')
    source_guard(before)
    print(args.phase+' completed with unchanged generator bytes', flush=True)


if __name__ == '__main__':
    main()
