#!/usr/bin/env python3
"""Offline analysis of corrected startup; reuse the frozen tensor parser."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if sys.flags.optimize:
    raise RuntimeError('Do not disable the frozen parser assertions with -O')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PARSER_SHA = '5c255423ba0f656df2e1e338beebfab2dee4c7a5f85c2a92ac57d503b143ebb8'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load_parser():
    path = HERE.parent / 'analyze.py'
    need(sha(path) == PARSER_SHA, 'Frozen tensor parser changed')
    spec = importlib.util.spec_from_file_location('frozen_ew_mass_analyzer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory_matches(directory, expected):
    actual = {}
    for path in sorted(directory.rglob('*')):
        need(not path.is_symlink(), 'Symlink in runtime evidence')
        if path.is_file():
            actual[str(path.relative_to(directory))] = sha(path)
    need(actual == expected, 'Runtime inventory mismatch: ' + str(directory))


def good_event(event):
    return event['returncode'] == 0 and event['timed_out'] is False and not event.get('error')


def envelope(run_root):
    prepared = json.loads((HERE / 'prepared.json').read_text())
    build = json.loads((HERE / 'build.json').read_text())
    smoke = json.loads((HERE / 'smoke.json').read_text())
    run = json.loads((HERE / 'run.json').read_text())
    names = ('prepared.json', 'control.py', 'prepare.py', 'check_observed.f', 'check_setpara.f')
    parts = {name: sha(HERE / name) for name in names}
    review = hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    for record in (build, smoke, run):
        need(record['state'] == 'complete' and record['review_hash'] == review, 'Incomplete or mismatched run envelope')
        need(record['review_sources'] == parts, 'Review source mismatch')
        need(record['cpu'] == 8 and record['jobs'] == 1, 'Unexpected CPU/jobs')
        need(record['automatic_retry'] is False and record['performance_eligible'] is False and
             record['host_stability_qualified'] is False, 'Invalid diagnostic scope')
        need(record['build_seconds'] == 180 and record['smoke_seconds'] == 30 and
             record['process_seconds'] == 120 and record['total_seconds'] == 480, 'Bounds changed')
    need(len(build['commands']) == 6 and all(good_event(e) for e in build['commands']), 'Incomplete driver/smoke build')
    need(set(build['outputs']) == set(prepared['build_outputs']), 'Built output list changed')
    need(smoke['build_json_sha256'] == run['build_json_sha256'] == sha(HERE / 'build.json'), 'Build binding changed')
    need(run['smoke_json_sha256'] == sha(HERE / 'smoke.json'), 'Smoke binding changed')
    need(smoke['executable_sha256'] == build['outputs'][prepared['smoke']['executable']], 'Smoke executable mismatch')
    need(all(good_event(smoke[k]) for k in ('loader', 'symbols', 'evaluation')), 'Smoke command failed')
    smoke_dir = run_root / 'card-only'
    inventory_matches(smoke_dir, smoke['files_after'])
    # The controller owns the exact model/card record validator. Its reviewed
    # implementation is reused read-only; there are no build/run calls here.
    spec = importlib.util.spec_from_file_location('ew_mass_attempt2_control', HERE / 'control.py')
    controller = importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    controller.smoke_output(smoke_dir, prepared)
    need(all(run['environment'][k] == '1' for k in
             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
         'Thread environment mismatch')
    need(len(run['cases']) == 4, 'Missing cases')
    need({(c['arm'], c['backend']) for c in run['cases']} ==
         {(a, b) for a in ('original', 'candidate') for b in (8, 9)}, 'Wrong cases')
    loaded = None
    for case in run['cases']:
        name = case['arm'] + '-' + str(case['backend']);directory = run_root / name
        need(case['complete'] is True and not case.get('error'), 'Incomplete case: ' + name)
        need(good_event(case['evaluation']) and good_event(case['loader_command']), 'Failed process: ' + name)
        need(case['executable_sha256'] == build['outputs'][prepared['arms'][case['arm']]['executable']], 'Executable mismatch')
        inventory_matches(directory, case['files_after'])
        need(sha(directory / 'inputs.json') == case['inputs_sha256'], 'Input binding changed')
        inputs = json.loads((directory / 'inputs.json').read_text())
        need(inputs['parameter_path'] == 'MadLoop5_resources/param_card.dat', 'Card path changed')
        for path, expected in inputs['after'].items():
            need(case['files_after'][path] == expected, 'Input modified during evaluation')
        need({'mass-observer.hex', 'observer-run.log', 'param.log', 'stdout.log', 'ldd.txt', 'inputs.json'} <=
             set(case['files_after']), 'Missing diagnostic output')
        lines = [x.strip() for x in (directory / 'observer-run.log').read_text().splitlines() if x.strip()]
        need(lines[0] == 'FIXED_QP_TRANSPORT_ONLY_UNCHECKED' and lines[-1] == 'COMPLETE_DIAGNOSTIC_ONLY', 'Driver incomplete')
        observations = [x.split() for x in lines if x.startswith('OBSERVED ')]
        need(len(observations) == 2 and [int(x[1]) for x in observations] == [1, 2], 'Wrong point observations')
        if loaded is None:
            loaded = case['loaded_libraries']
        need(loaded == case['loaded_libraries'], 'Runtime identity varied between cases')
        for soname, expected in prepared['expected_runtime'].items():
            need(loaded[soname] == {'path': str(ROOT / expected['path']), 'sha256': expected['sha256']}, 'Native DSO changed')
        system = {name: info for name, info in loaded.items() if name not in prepared['expected_runtime']}
        need(system == prepared['prior_system_runtime'], 'System runtime identity changed')
    return {'verified': True, 'review_hash': review, 'build_sha256': sha(HERE / 'build.json'),
            'smoke_sha256': sha(HERE / 'smoke.json'), 'run_sha256': sha(HERE / 'run.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT / '.tirust-build/ew-qp-mass-prototype/observer-native-design/attempt2/runs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();legacy = load_parser()
    result = {'scope': 'Fixed CTModeRun=4 unchecked transport diagnostics only', 'cases': {}, 'comparisons': {}}
    failures = (OSError, ValueError, AssertionError, IndexError, KeyError, TypeError)
    try:
        result['runner_envelope'] = envelope(args.run_root)
    except failures as error:
        result['runner_envelope'] = {'verified': False, 'error': type(error).__name__ + ': ' + str(error)}
    controls = json.loads((HERE.parent / 'historical-controls.json').read_text())['cases'];captures = {}
    for arm in ('original', 'candidate'):
        for backend in (8, 9):
            name = arm + '-' + str(backend)
            try:
                capture = legacy.parse(args.run_root / name / 'mass-observer.hex', arm, backend)
                result['cases'][name] = {'complete': True, 'points': legacy.summarize(capture, backend)}
                if name in controls:
                    result['cases'][name]['historical_control'] = legacy.historical_control(capture, controls[name])
                captures[name] = capture
            except failures as error:
                result['cases'][name] = {'complete': False, 'error': type(error).__name__ + ': ' + str(error)}
    pairs = [('original-' + str(b), 'candidate-' + str(b), 'same_backend') for b in (8, 9)]
    pairs += [(a + '-8', a + '-9', 'cross_backend') for a in ('original', 'candidate')]
    for left, right, kind in pairs:
        if left in captures and right in captures:
            result['comparisons'][left + '_vs_' + right] = legacy.compare_pair(captures[left], captures[right], kind)
    result['transport_interpretation_eligible'] = (result['runner_envelope']['verified'] and len(captures) == 4 and
        all(x['complete'] for x in result['cases'].values()) and len(result['comparisons']) == 4 and
        all(result['cases'][name]['historical_control']['exact_retained_binary64_status_and_available_host_flag_parity'] for name in controls) and
        all(x['nonmass_input_bits_match'] for x in result['comparisons'].values()))
    result['limitations'] = ['Fixed4 status120 is unchecked; automatic-1 is a separate unexecuted phase.',
        'No amplitude speed, threshold physical accuracy or all-five-backend conclusion.',
        'Historical NinjaRS host IEEE/FPE flags were not retained; their parity is unavailable.',
        'Model-only smoke validates card loading and declared mass/scale bits, not all coupling values.']
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2);stream.write('\n')


if __name__ == '__main__':
    main()
