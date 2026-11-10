"""Portable closed-data replay. No compilation, subprocess or foreign calls."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import re
import shlex
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def paired(original, candidate):
    values = [math.log(c) - math.log(o) for o, c in zip(original, candidate)]
    if len(values) != 7 or any(not math.isfinite(x) for x in values):
        raise ValueError('Expected seven finite paired timings')
    mean = statistics.mean(values)
    distance = 2.446912 * statistics.stdev(values) / math.sqrt(7)
    return {'geometric_mean_ratio': math.exp(mean), 'lower_95': math.exp(mean - distance),
        'upper_95': math.exp(mean + distance)}


def verify(root):
    index = load(root / 'index.json')
    for name, item in index['files'].items():
        path = root / name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Payload identity mismatch: ' + name)
    capture = load(root / 'capture.json')
    binding = load(root / 'source-bindings.json')
    source = load(root / 'prior-source-index.json')
    if sha(root / 'prior-source-index.json') != capture['source_index_sha256']:
        raise ValueError('Wrong admitted source index')
    if sha(root / 'source-bindings.json') != source['files']['bindings.json']['sha256']:
        raise ValueError('Wrong admitted binding')
    for name in ('candidate.patch', 'replacement.json', 'lifecycle.f90', 'micro.f90', 'observer.c', 'worker.f'):
        if sha(root / name) != source['files'][name]['sha256']:
            raise ValueError('Admitted fixture mismatch: ' + name)
    for name, item in capture['complete_work_regular_inventory'].items():
        if name not in capture['excluded_build_artifact_identities']:
            if sha(root / 'retained' / name) != item['sha256']:
                raise ValueError('Capture inventory mismatch: ' + name)
    if capture['production_files_modified'] or capture['performance_acceptance_eligible']:
        raise ValueError('Incorrect promotion/precise acceptance claim')
    if capture['native_controller_observed_exit_status'] != 0:
        raise ValueError('Native controller did not complete')
    terminal = load(root / 'retained/terminal.json')
    if terminal['returncode'] != 0 or terminal['elapsed_seconds'] > binding['global_wall_seconds']:
        raise ValueError('Terminal/deadline mismatch')
    delta = load(root / 'replacement.json')
    original = (root / 'original-generated-TIRUST_interface.f').read_text()
    candidate = (root / 'retained/build/candidate-TIRUST_interface.f').read_text()
    if original.count(delta['old']) != 1 or candidate != original.replace(delta['old'], delta['new']):
        raise ValueError('Candidate changed more than the admitted clear procedure')
    receipt = load(root / 'compile-policy-receipt.json')
    metadata = load(root / 'original-campaign-inputs.json')
    if sha(root / 'original-campaign-inputs.json') != binding['metadata_sha256']:
        raise ValueError('Original campaign inputs differ')
    if sha(root / 'original-campaign-build.log') != metadata['compilation']['build_logs']['process']:
        raise ValueError('Original actual build log differs')
    recipe = metadata['compilation']['effective_compiler_flags']['scopes']['process']['recipes']['TIRUST_interface.f']
    if receipt['original_effective_recipe'] != recipe:
        raise ValueError('Original compiler recipe differs')
    compile_command = load(root / 'retained/commands/001-compile-candidate/command.json')
    if receipt['candidate_actual_command'] != compile_command:
        raise ValueError('Candidate compiler receipt differs')
    actual_argv = compile_command['argv'][compile_command['argv'].index('--') + 1:]
    original_lines = [line for line in (root / 'original-campaign-build.log').read_text().splitlines()
        if '-c TIRUST_interface.f ' in line]
    if len(original_lines) != 1 or receipt['original_adapter_compile_line'] != original_lines[0]:
        raise ValueError('Original adapter actual compile command differs')
    normalized = []
    args = iter(actual_argv[1:])
    for token in args:
        if token in ('-o', '-I', '-J', '-isystem', '-iquote'):
            next(args)
        elif token != '-c' and not token.startswith(('-I', '-J')) and not token.endswith(('.f', '.f90', '.o')):
            normalized.append(token)
    if normalized != recipe['flags'] or actual_argv[0] != recipe['compiler'][0]['path']:
        raise ValueError('Candidate compiler/nonpath flags differ from original recipe')
    if normalized != ['-w', '-fPIC', '-ffixed-line-length-132']:
        raise ValueError('Unexpected optimization/bounds/floating-point policy')

    spec = importlib.util.spec_from_file_location('closed_clear_worker', root / 'worker_protocol.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    summary = load(root / 'retained/summary.json')
    expected_paths = {item['resolved_target'] for item in capture['provider_symlinks'].values()}
    rows = 0
    timed_calls = 0
    observations = 0
    for backend in (1, 9):
        observed = {}
        for arm in ('original', 'candidate'):
            directory = root / 'retained/workers' / f'profile-{arm}-{backend}'
            observed[arm] = (directory / 'values.dat').read_bytes()
            profile = load(directory / 'ffi-profile.json')
            if backend == 1 and arm == 'candidate' and any(profile.values()):
                raise ValueError('Unselected candidate created or cleared TIRust')
            if arm == 'original' and (profile['create_calls'] != 1 or profile['clear_calls'] != 20202):
                raise ValueError('Original lifecycle profile differs')
        if observed['original'] != observed['candidate']:
            raise ValueError('Observed numerical parity differs')
        op = load(root / 'retained/workers' / f'profile-original-{backend}' / 'ffi-profile.json')
        cp = load(root / 'retained/workers' / f'profile-candidate-{backend}' / 'ffi-profile.json')
        if backend == 9 and (op['register_calls'] != cp['register_calls'] or
            op['clear_calls'] - op['clears_before_first_plan_registration'] !=
            cp['clear_calls'] - cp['clears_before_first_plan_registration']):
            raise ValueError('Selected active-context lifecycle differs')
        for arm in ('original', 'candidate'):
            for kind in ('profile', 'full'):
                directory = root / 'retained/workers' / f'{kind}-{backend}-{arm}' if kind == 'full' else root / 'retained/workers' / f'profile-{arm}-{backend}'
                if sha(directory / 'points.dat') != source['files']['fixtures/points.dat']['sha256']:
                    raise ValueError('Point stream differs')
                if (directory / 'param_card.dat').exists():
                    raise ValueError('Wrong bare parameter-card fixture reappeared')
                for name in ('param_card.dat', 'ident_card.dat'):
                    if sha(directory / 'Cards' / name) != source['files']['fixtures/' + name]['sha256']:
                        raise ValueError('Model card differs')
                maps = (directory / 'runtime-maps.txt').read_text()
                mapped = {line.split()[-1] for line in maps.splitlines()
                    if line.split()[-1].startswith('/') and '.so' in line.split()[-1]}
                if mapped != expected_paths:
                    raise ValueError('Actual provider maps differ')
                archives = sorted(directory.glob('worker-raw/session-*/*/command.json'))
                if len(archives) != (8 if kind == 'full' else 1):
                    raise ValueError('Wrong persistent command count')
                for record in archives:
                    command = load(record)
                    if command['error'] is not None:
                        raise ValueError('A measurement command failed')
                    observation = worker.parse_outputs(record.parent, backend, 100, command['loops'])
                    if observation['backend_precision_counts'] != {str(backend) + ':DP': 100 * command['loops']}:
                        raise ValueError('Native-DP cohort changed')
                    if (record.parent / 'values.dat').read_bytes() != observed[arm]:
                        raise ValueError('Observed/plain or batch numerical parity differs')
                    rows += len(observation['rows'])
                    timed_calls += 100 * command['loops']
                    observations += 1
                if kind == 'full':
                    recorded = summary['full_amplitude'][str(backend)]
                    series = []
                    for pair in range(7):
                        parsed = worker.parse_outputs(archives[pair + 1].parent, backend, 100, recorded['loops'])
                        seconds = parsed['seconds']
                        if seconds != recorded['measurements'][arm][pair]:
                            raise ValueError('Summary time differs from raw worker ticks')
                        series.append(seconds)
                    for key, value in paired(recorded['measurements']['original'], recorded['measurements']['candidate']).items():
                        if abs(value - recorded['statistics'][key]) > 1e-14:
                            raise ValueError('Paired confidence calculation differs')
    commands = sorted((root / 'retained/commands').glob('*'))
    if len(commands) != 43 or any(load(p / 'terminal.json')['returncode'] for p in commands):
        raise ValueError('Compilation/lifecycle/micro command failure or scope differs')
    for arm in ('original', 'candidate'):
        matches = [p for p in commands if p.name.endswith('lifecycle-' + arm)]
        if len(matches) != 1 or 'LIFECYCLE_PASS' not in (matches[0] / 'stdout.txt').read_text():
            raise ValueError('Actual lifecycle fixture did not pass')
    raw_micro = {'0': {'original': [], 'candidate': []}, '1': {'original': [], 'candidate': []}}
    for path in commands:
        if not path.name.endswith('-micro') or 'compile-' in path.name:
            continue
        command = load(path / 'command.json')
        arm = 'candidate' if 'candidate-micro' in command['argv'][-3] else 'original'
        fields = next(line.split() for line in (path / 'stdout.txt').read_text().splitlines() if line.split()[:1] == ['MICRO'])
        active, repetitions = int(fields[1]), int(fields[2])
        if repetitions != 1000000:
            raise ValueError('Micro repetition count differs')
        if active == 0 and arm == 'candidate' and fields[3:5] != ['-1', '-1']:
            raise ValueError('Inactive micro created a context')
        raw_micro[str(active)][arm].append(int(fields[-2]) / int(fields[-1]))
    if raw_micro != summary['micro']:
        raise ValueError('Micro summary differs from raw clocks')
    return {'verified': True, 'observation_count': observations, 'numerical_rows_replayed': rows,
        'timed_amplitudes_replayed': timed_calls, 'all_native_dp': True, 'byte_parity': True,
        'full_amplitude': {k: paired(v['measurements']['original'], v['measurements']['candidate'])
            for k, v in summary['full_amplitude'].items()},
        'micro': {k: paired(v['original'], v['candidate']) for k, v in raw_micro.items()},
        'performance_acceptance_eligible': False, 'production_promoted': False}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), sort_keys=True))
