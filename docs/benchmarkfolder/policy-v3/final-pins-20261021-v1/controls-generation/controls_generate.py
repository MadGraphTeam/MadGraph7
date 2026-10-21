#!/usr/bin/env python3
"""Reuse the frozen CLI/build procedure for five existing physical controls."""
import argparse
import concurrent.futures
import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('final_release_generation', HERE / 'regenerate.py')
generation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generation)
ROOT = generation.ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    generation.verify_freeze(freeze)
    report = HERE.parent / 'controls-generation'
    work = ROOT / '.benchmark-work/policy-v3/final-pins-20261021-v1/controls-generation'
    if report.exists() or work.exists():
        raise ValueError('Preserve every earlier controls generation attempt')
    config = json.loads((generation.OLD_REPORT / 'policy-v2/controls.json').read_text())
    expected = {'control-gg-h', 'control-gg-hh', 'control-mixed',
                'control-complex', 'control-forced-qp'}
    if {case['id'] for case in config['cases']} != expected or config['points_per_case'] != 10:
        raise ValueError('The fixed representative control configuration changed')
    generation.policy.validate_config(config, core=False)
    generation.CORRECTION = generation.policy.validate_correction(
        generation.OLD_REPORT / 'policy-v2/model-correction.json', live=True)
    generation.FREEZE = freeze
    report.mkdir(parents=True); work.mkdir(parents=True)
    generation.REPORT = report; generation.WORK = work
    generation.bench.REPORT = report; generation.bench.WORK = work
    source_state = generation.bench.generator_tree_state()
    source_digest = generation.bench.content_digest(source_state)
    original_archive = generation.verify_original_archive()
    originals = {case['id']: generation.bench.generated_source_manifest(
        generation.OLD_WORK / 'outputs/baseline' / case['id']) for case in config['cases']}
    dependencies = generation.dependency_manifest()
    generation.bench.dump(report / 'cases.json', config)
    generation.bench.dump(report / 'executed-freeze.json', freeze)
    generation.bench.dump(report / 'baseline-integrity-before.json', original_archive)
    shutil.copy2(Path(__file__), report / 'controls_generate.py')
    shutil.copy2(HERE / 'regenerate.py', report / 'regenerate.py')
    generation.SUMMARY.update(schema='final-release-control-generation-v1',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_revision=freeze['committed_revision'], generator_tree_state_sha256=source_digest,
        generator_tree_state=source_state, final_freeze=freeze, dependencies=dependencies,
        cases={case['id']: {'status': 'pending', 'case': case} for case in config['cases']},
        amplitude_evaluations=0, initialized_symbolica_workers=0,
        scope='Actual CLI generation and clean subprocess build only; masses path evaluates no loop.')
    generation.bench.dump(report / 'status.json', generation.SUMMARY)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [pool.submit(generation.process, case, config, source_digest, originals)
                for case in config['cases']]
        for job in concurrent.futures.as_completed(jobs):
            job.result()
    generation.verify_freeze(freeze)
    preserved = all(generation.bench.generated_source_manifest(
        generation.OLD_WORK / 'outputs/baseline' / cid) == manifest
        for cid, manifest in originals.items())
    generation.SUMMARY.update(
        complete=all(item['status'] == 'complete' for item in generation.SUMMARY['cases'].values()),
        preserved_existing_inputs=preserved,
        dependencies_unchanged=generation.dependency_manifest() == dependencies,
        original_archive_unchanged=generation.verify_original_archive() == original_archive,
        script_sha256=generation.digest(Path(__file__)),
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    generation.bench.dump(report / 'status.json', generation.SUMMARY)
    if not all(generation.SUMMARY[key] for key in (
            'complete', 'preserved_existing_inputs', 'dependencies_unchanged', 'original_archive_unchanged')):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
