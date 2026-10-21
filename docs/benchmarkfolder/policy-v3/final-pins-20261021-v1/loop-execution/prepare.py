#!/usr/bin/env python3
"""Prepare held loop-induced full-sum inputs without evaluating an amplitude.

The existing supplement collector consumes canonical supplement IDs.  Its
candidate inputs are exact archived copies of the final actual control exports;
their original generation records and CLI case IDs are retained unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'madgraph/__init__.py').is_file())
sys.path.insert(0, str(ROOT / 'tools'))
import ninjars_benchmark as bench
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_policy_v3 as release
import ninjars_benchmark_workers as workers
import ninjars_supplement_v2 as supplement

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
OLD = ROOT / 'docs/benchmarkfolder/policy-v2/final-supplement-generation-v1'
CONTROL = FINAL / 'controls-generation'
WORK = ROOT / '.benchmark-work/policy-v3/final-pins-20261021-v1/loop-execution'
ALIASES = {'supplement-gg-h': 'control-gg-h', 'supplement-gg-hh': 'control-gg-hh'}


def read(path):
    return json.loads(Path(path).read_text())


def exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def redirect_internal_links(original, target):
    """Keep a worker copy independent of every writable original-output link."""
    if original.is_symlink() or target.is_symlink() or not target.is_dir():
        raise ValueError('Archive inputs must be real directories, never top-level symlink aliases')
    changes = []
    for path in sorted(target.rglob('*')):
        if not path.is_symlink():
            continue
        resolved = path.resolve()
        if resolved.is_relative_to(original):
            before = os.readlink(path)
            replacement = target / resolved.relative_to(original)
            path.unlink()
            path.symlink_to(os.path.relpath(replacement, path.parent))
            changes.append({'path': str(path.relative_to(target)), 'before': before,
                            'after': os.readlink(path)})
    return changes


def prepare():
    if os.sched_getaffinity(0) != {6}:
        raise ValueError('Static archive preparation requires the released CPU6 only')
    config = supplement.loop_configuration()
    supplement.validate_configuration(config)
    cohort = release.validate_cohort(FINAL / 'candidate-cohort.json')
    status = read(CONTROL / 'status.json')
    if not all(status.get(k) is True for k in ('complete', 'preserved_existing_inputs',
            'dependencies_unchanged', 'original_archive_unchanged')):
        raise ValueError('The actual final control CLI/build campaign is incomplete')
    if (WORK.exists() or (HERE / 'specification-held-v1.json').exists() or
            (HERE / 'readiness-v1.json').exists()):
        raise ValueError('This immutable loop preparation namespace already exists')
    WORK.mkdir(parents=True)
    ledger = WORK / 'archive-preparation'
    ledger.mkdir()
    targets, source_proofs = {}, {'baseline': {}, 'candidate': {}}
    candidate_root = WORK / 'inputs/candidate'
    candidate_root.mkdir(parents=True)
    old_work = ROOT / '.benchmark-work/policy-v2/final-supplement-generation-v1'
    expected_runtime = cohort['runtime_identity_sha256']
    points = {}
    with pair.owned_preparation(ledger, label='loop-archive'):
        for case in config['cases']:
            cid, actual_id = case['id'], ALIASES[case['id']]
            saved = status['cases'][actual_id]
            if (saved['status'] != 'complete' or
                    {k: saved['case'][k] for k in ('model', 'process', 'nfinal')} !=
                    {k: case[k] for k in ('model', 'process', 'nfinal')}):
                raise ValueError('An alias changes the physical process')
            original = policy.path(saved['output'])
            if original.is_symlink() or not original.is_dir():
                raise ValueError('The actual control export must be a real directory')
            original = original.resolve()
            target = candidate_root / cid
            manifest = bench.generated_source_manifest(original)
            generation, compilation = read(original / 'generation.json'), read(original / 'compilation.json')
            identity = generation['source_identity']
            if (identity['revision'] != cohort['generation_revision'] or
                    identity['tree_state_sha256'] != cohort['generator_identity_sha256'] or
                    identity.get('dirty_patch_sha256') is not None or identity['tree_state']['untracked'] or
                    compilation['source_identity'] != identity or
                    compilation['generated_sources_sha256'] != manifest['files_sha256']):
                raise ValueError('The actual source differs from the final measured cohort')
            pair.validate_origin_records('candidate', original, generation, compilation, manifest)
            runtime = pair.actual_runtime(original, manifest, 8)
            if runtime['identity_sha256'] != expected_runtime:
                raise ValueError('An actual control export has another complete NinjaRS runtime')
            physical = original / 'Cards/param_card.dat'
            old_card = OLD / 'inputs/baseline' / cid / 'param_card.dat'
            if physical.read_bytes() != old_card.read_bytes():
                raise ValueError('Loop original/candidate parameter cards differ')
            point_file = OLD / 'points' / (cid + '.dat')
            metadata = supplement.point_metadata(point_file, saved['masses'], case['nfinal'])
            if metadata['points'] != 100:
                raise ValueError('The retained loop stream must contain exactly100 points')
            points[cid] = supplement.bound(point_file)
            workers.subprocess.run(['cp', '-a', '--reflink=auto', str(original), str(target)], check=True)
            redirects = redirect_internal_links(original, target)
            if (bench.generated_source_manifest(target) != manifest or
                    read(target / 'generation.json') != generation or
                    read(target / 'compilation.json') != compilation or
                    (target / 'Cards/param_card.dat').read_bytes() != physical.read_bytes() or
                    pair.actual_runtime(target, manifest, 8) != runtime):
                raise ValueError('An archived alias changed actual source, metadata, card or runtime')
            pair.validate_origin_records('candidate', target, generation, compilation, manifest)
            # The anchor keeps the real actual-CLI control case, not the benchmark alias.
            command = CONTROL / 'commands' / ('candidate-' + actual_id + '.mg5')
            source_log = ROOT / '.benchmark-work/policy-v3/final-pins-20261021-v1/controls-generation/logs' / ('generate-candidate-' + actual_id + '.log')
            retained_log = HERE / 'logs' / source_log.name
            retained_log.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_log, retained_log)
            anchor = policy.seal({'schema': 'supplement-generation-anchor-v1',
                'role': 'candidate_corrected', 'case': saved['case'],
                'compiled_manifest': manifest, 'generation': generation, 'compilation': compilation,
                'parameter_card_sha256': supplement.digest(physical),
                'cli_command': supplement.bound(command), 'cli_log': supplement.bound(retained_log),
                'baseline_integrity': None, 'physical_case_alias': {'benchmark_case': case,
                    'actual_cli_case': saved['case'], 'original_output': policy.relative(original)},
                'final_generation_status': supplement.bound(CONTROL / 'status.json')})
            anchor_path = HERE / 'source-proofs/candidate' / (cid + '.json')
            exclusive(anchor_path, anchor)
            source_proofs['candidate'][cid] = supplement.bound(anchor_path)
            baseline_anchor = OLD / 'source-proofs/with-cli-v2/baseline' / (cid + '.json')
            policy.verified(read(baseline_anchor), 'supplement-generation-anchor-v1')
            source_proofs['baseline'][cid] = supplement.bound(baseline_anchor)
            baseline = old_work / 'outputs/baseline' / cid
            baseline_generation, baseline_compilation = read(baseline / 'generation.json'), read(baseline / 'compilation.json')
            baseline_manifest = bench.generated_source_manifest(baseline)
            pair.validate_origin_records('baseline', baseline, baseline_generation, baseline_compilation, baseline_manifest)
            targets[cid] = {'benchmark_case': case, 'actual_candidate_cli_case': saved['case'],
                'candidate_original_output': policy.relative(original), 'candidate_archived_output': policy.relative(target),
                'candidate_generation': supplement.bound(target / 'generation.json'),
                'candidate_compilation': supplement.bound(target / 'compilation.json'),
                'candidate_compiled_sources_sha256': manifest['files_sha256'],
                'candidate_runtime': runtime, 'internal_link_redirects': redirects,
                'baseline_output': policy.relative(baseline),
                'baseline_runtime_6': pair.actual_runtime(baseline, baseline_manifest, 6),
                'points': points[cid], 'point_metadata': metadata,
                'physical_card_sha256': supplement.digest(physical)}
    aliases_path = HERE / 'archive-aliases-v1.json'
    exclusive(aliases_path, policy.seal({'schema': 'final-loop-archived-input-aliases-v1',
        'candidate_cohort': supplement.bound(FINAL / 'candidate-cohort.json'), 'targets': targets,
        'generation_metadata_rewritten': False, 'compiler_invocations': 0, 'loop_amplitude_evaluations': 0}))
    config_path = HERE / 'cases.json'
    exclusive(config_path, config)
    spec = {'timeout': 86400., 'configuration': supplement.bound(config_path), 'points': points,
        'role_map': {'baseline': 'original', 'candidate': 'candidate_corrected'},
        'cpu': 207, 'repetitions': 7, 'timing': True, 'threshold': False,
        'run_id': 'final-pins-loop-fullsum-paired-v1',
        'report': policy.relative(HERE / 'runs/loop-timing'),
        'work': policy.relative(WORK / 'runs/loop-timing'),
        'output_roots': {'baseline': policy.relative(old_work / 'outputs/baseline'),
                         'candidate': policy.relative(candidate_root)}, 'source_proofs': source_proofs,
        'candidate_cohort': supplement.bound(FINAL / 'candidate-cohort.json'),
        'archive_aliases': supplement.bound(aliases_path),
        'runtime_qualification': supplement.bound(FINAL / 'representative/representative-proof.json'),
        'pair_gate': {'targets': [['baseline', 6], ['candidate', 8]],
            'ordering': 'Adjacent original6/candidate8; reverse the complete target order on odd repetitions',
            'required_repetitions': 7, 'max_runtime_ratio': 1.05,
            'uncertainty_rule': 'Unresolved confidence bounds do not pass'},
        'additional_fullsum_diagnostics': [['baseline', 1], ['baseline', 7],
            ['candidate', 1], ['candidate', 6], ['candidate', 7]]}
    spec_path = HERE / 'specification-held-v1.json'
    exclusive(spec_path, spec)
    readiness = policy.seal({'schema': 'final-loop-fullsum-held-readiness-v1',
        'status': 'held-no-numerical-launch', 'specification': supplement.bound(spec_path),
        'archive_aliases': supplement.bound(aliases_path),
        'existing_collector': supplement.bound(ROOT / 'tools/ninjars_supplement_v2.py'),
        'ownership_helper': supplement.bound(ROOT / 'tools/ninjars_benchmark_pair_v3.py'),
        'preparation_script': supplement.bound(Path(__file__)),
        'actual_loop_points_per_case': 100, 'actual_timing_pairs': 0,
        'performance_gate_passed': False, 'compiler_invocations': 0, 'loop_amplitude_evaluations': 0,
        'source_generation_invocations': 0,
        'launch_requires': ['Root explicit MP and quiet CPU207 handoff after actual core correctness',
            'Revalidate held source/card/point/runtime hashes and exact archived directory copies',
            'Run unchanged supplement.collect inside frozen pair.owned_preparation',
            'Independently verify actual adjacent6/8 seven-pair order and denominator agreement before CI gate']})
    exclusive(HERE / 'readiness-v1.json', readiness)
    return readiness


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-archives', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(prepare(), indent=2))
