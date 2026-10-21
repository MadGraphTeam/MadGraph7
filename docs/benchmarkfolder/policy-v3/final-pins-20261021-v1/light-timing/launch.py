#!/usr/bin/env python3
"""Owned launch of only the new approved light timing specification."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import light_pair as light

owner_path = HERE.parent / 'timing-launch/launch.py'
module_spec = importlib.util.spec_from_file_location('light_retained_timing_owner', owner_path)
owner = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(owner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=HERE / 'specification-held-v1.json')
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--execute-numerical', action='store_true')
    args = parser.parse_args()
    if not args.execute_numerical:
        raise ValueError('No actual launch without explicit execution and root quiet handoff')
    spec, _ = light.validate_spec(light.read(args.spec)); light.validate_handoff(args.spec, args.handoff)
    if not os.sched_getaffinity(0).issubset({6, 7, 207}) or 207 not in os.sched_getaffinity(0):
        raise ValueError('Use taskset -c6,7,207; numerical workers alone are pinned207')
    if not args.attempt.resolve().is_relative_to(HERE):
        raise ValueError('Keep retained launch evidence inside the light campaign')
    report, work = light.policy.path(spec['report']), light.policy.path(spec['work'])
    if args.attempt.exists() or report.exists() or work.exists():
        raise ValueError('Preserve existing attempts; no automatic restart or append')
    original_handoff = light.supplement.bound(args.handoff)
    args.attempt.parent.mkdir(parents=True, exist_ok=True)
    retained = args.attempt.parent / (args.attempt.name + '-root-handoff.json')
    with retained.open('xb') as stream:
        stream.write(args.handoff.read_bytes())
    if light.workers.digest(retained) != original_handoff['sha256']:
        raise ValueError('Root handoff changed during retained copying')
    light.validate_handoff(args.spec, retained)
    metadata = {'stage': 'light-six-case-pairs', 'specification': light.supplement.bound(args.spec),
        'root_handoff': original_handoff, 'retained_handoff': light.supplement.bound(retained),
        'candidate_cohort': spec['candidate_cohort'], 'report': spec['report'], 'work': spec['work'],
        'sources': {'launcher': light.supplement.bound(Path(__file__)),
                    'collector': light.supplement.bound(HERE / 'light_pair.py'),
                    'frozen_owned_controller': light.supplement.bound(owner_path)},
        'actual_points': light.POINTS, 'repetitions': light.REPETITIONS,
        'minimum_seconds': light.MINIMUM_SECONDS, 'wall_budget_seconds': light.WALL_BUDGET_SECONDS,
        'sampling_label': spec['sampling_label'], 'old_full8_runtime_gate_required_for_completion': False}
    command = [sys.executable, '-B', str(HERE / 'light_pair.py'), 'collect',
               '--spec', str(args.spec), '--handoff', str(retained), '--execute-numerical']
    return owner.run_owned(command, report, args.attempt, metadata)


if __name__ == '__main__':
    raise SystemExit(main())
