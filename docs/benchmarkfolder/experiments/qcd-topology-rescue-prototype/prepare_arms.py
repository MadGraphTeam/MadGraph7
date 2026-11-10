#!/usr/bin/env python3
"""Bind four private source arms without compiling, linking or evaluating."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OBSERVER = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer'
NATIVE = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer-native'
BORN = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype'
SOURCE = HERE / 'source-variants/born-c'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size}


def main():
    require(Path.cwd().resolve() == ROOT, 'Run from MadGraph root')
    require(not SOURCE.exists() and not (HERE / 'arms.json').exists(), 'Preserve one-shot preparation')
    require(sha(NATIVE / 'evidence-index.json') == '6a2e48f62ee48ebd2a892da18f797249e686be660225e9d05c66415f98e8f5ac',
            'Frozen native evidence changed')
    helper = HERE / 'born-c-design/derive.py'
    require(sha(helper) == '90f161f78a9a9a3f69ea77d64600c53c6f71d00c8c4b91e988e4f3c9d6182473', 'C derivation changed')
    spec = importlib.util.spec_from_file_location('private_qcd_born_c', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patch, bindings = module.artifacts()
    require(patch == (HERE / 'born-c-design/born-c-handoff.patch').read_text(), 'C patch mismatch')
    require(bindings == json.loads((HERE / 'born-c-design/source-bindings.json').read_text()), 'C bindings mismatch')
    _, derived = module.derive_sources()
    SOURCE.mkdir(parents=True)
    for name, text in derived.items():
        (SOURCE / name).write_text(text)
        require(sha(SOURCE / name) == bindings['derived_sources'][name]['sha256'], name)
    fixed = {
        'observer.f90': record(OBSERVER / 'observer.f90'),
        'observer_context.f': record(OBSERVER / 'observer_context.f'),
        'amplitude_driver.f': record(OBSERVER / 'sources/amplitude_driver.f'),
    }
    arms = {}
    for name, topology, coordinate in (('original-b', False, 'B'), ('topology-b', True, 'B'),
                                       ('original-c', False, 'C'), ('topology-c', True, 'C')):
        sources = dict(fixed)
        adapter = HERE / 'topology-design' if topology else OBSERVER / 'sources'
        for file in ('CT_interface.f', 'TIRUST_interface.f'):
            sources[file] = record(adapter / file)
        sources['loop_matrix.f'] = record(SOURCE / 'loop_matrix.f' if coordinate == 'C'
                                          else OBSERVER / 'sources/loop_matrix.f')
        sources['born_fallback.f'] = record(SOURCE / 'born_fallback.f' if coordinate == 'C'
                                            else BORN / 'born_fallback.f')
        if topology:
            sources['shell_classes.f'] = record(HERE / 'topology-design/shell_classes.f')
        arms[name] = {'born_coordinates': coordinate, 'topology_gate': topology, 'sources': sources,
                      'loop_coordinates': 'Actual existing SET_MP_PS output C, including later host rotations',
                      'born_parameter_policy': 'Unchanged exact promotion of DP parameters',
                      'ordinary_dp_source_path': 'Unchanged original finite-DP branch',
                      'runtime_evaluated': False}
    plan = {'schema': 1, 'kind': 'private source-arm plan only', 'compiler_executed': False,
            'native_executed': False, 'production_modified': False, 'arms': arms,
            'historical_control': record(NATIVE / 'evidence-index.json'),
            'historical_build': record(NATIVE / 'attempt1/build.json'),
            'source_preparation': record(HERE / 'topology-design/prepared.json'),
            'c_derivation': record(helper), 'script': record(Path(__file__)),
            'future_runtime': {'requires_separate_approval': True, 'cpu': 11,
                               'shared_lock': '.tirust-build/numerical-tests.lock',
                               'max_seconds_per_process': 180, 'backends': [9, 8],
                               'point_sequence': ['ordinary', 'stress', 'ordinary'],
                               'automatic_retries': 0, 'formal_performance_eligible': False},
            'interpretation': 'A factorial source design, not authorization to execute all arms or a claim of amplitude accuracy.'}
    with (HERE / 'arms.json').open('x') as stream:
        stream.write(json.dumps(plan, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'arms_prepared': list(arms), 'compiler_executed': False, 'native_executed': False}, sort_keys=True))


if __name__ == '__main__':
    main()
