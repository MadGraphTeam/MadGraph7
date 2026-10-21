#!/usr/bin/env python3
"""Correct one analysis import after four completed calls; never rerun physics."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
CONTROLLER_SHA256 = 'd58bd237efa35b6dd400cbebe120f5cfc02c45a5383a7a96759dd885b6710f43'
SPECIFICATION_SHA256 = '221750e50feadf9aac3e5dd834cb269eb18ad4e16bd5fddc42ef1e387c679528'
TERMINAL_SHA256 = '417a348b9b78283124f020e97879a5c4fe92b0d129bfcc9e9513aafe0b1ceeb4'
RESULT = HERE / 'run-v1/result-finalized-v1.json'


def bound(path):
    data = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def load_reader():
    controller = HERE / 'control.py'
    if bound(controller)['sha256'] != CONTROLLER_SHA256:
        raise ValueError('The original measured controller must remain unchanged')
    module_spec = importlib.util.spec_from_file_location('higher_higgs_frozen_reader', controller)
    reader = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(reader)
    # The original source imports both modules. Its single comparison call used
    # the v3 name, although the already-bound pure function lives in v2. Replace
    # only that local module reference; no imported module or file is mutated.
    if hasattr(reader.policy3, 'compare_rows') or not callable(reader.policy.compare_rows):
        raise ValueError('Unexpected comparison API; preserve the reviewed correction')
    reader.policy3 = reader.policy
    return reader


def compare_rows(groups):
    """Expose the exact real comparator for the API regression test."""
    return load_reader().policy3.compare_rows(groups)


def validate():
    if bound(HERE / 'held-specification-v2.json')['sha256'] != SPECIFICATION_SHA256:
        raise ValueError('Changed authorized two-point specification')
    terminal_path = HERE / 'native-terminal-before-finalizer-v1.json'
    if bound(terminal_path)['sha256'] != TERMINAL_SHA256:
        raise ValueError('Changed original native/parent terminal evidence')
    terminal = json.loads(terminal_path.read_text())
    if (terminal['controller']['parent_exit_code'] != 1 or
            terminal['all_four_native_children_reaped_exit_zero_and_absent'] is not True or
            terminal['native_rerun'] is not False):
        raise ValueError('Unexpected original execution outcome')
    reader = load_reader()
    science = reader.validate()
    value = {'schema': 'higher-higgs-fixed-two-point-finalized-v1',
             'frozen_controller': bound(HERE / 'control.py'), 'versioned_finalizer': bound(Path(__file__).resolve()),
             'original_collection_terminal': bound(terminal_path),
             'comparison_implementation': bound(ROOT / 'tools/ninjars_benchmark_policy_v2.py'),
             'reader_correction': 'The single post-collection call resolves to the already-bound policy_v2.compare_rows pure function.',
             'original_parent_exit_code': 1, 'native_calls_repeated': 0,
             'science': science, 'passed': science['passed'], 'performance_measurement': False}
    if RESULT.exists() and json.loads(RESULT.read_text()) != value:
        raise ValueError('Published finalization differs from retained raw proof')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('finalize', 'validate'))
    args = parser.parse_args()
    if args.action == 'finalize' and RESULT.exists():
        raise ValueError('Preserve the existing finalization')
    value = validate()
    if args.action == 'finalize':
        RESULT.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(value, indent=2))
