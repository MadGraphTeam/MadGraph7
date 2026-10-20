#!/usr/bin/env python3
"""Read-only verification of a completed fixed21 process experiment."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def parse_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        assert len(fields) == 8
        rows.append(dict(point=int(fields[0]), order=int(fields[1]), status=int(fields[2]),
                         coefficients=[float(x) for x in fields[3:7]], accuracy=float(fields[7])))
    assert [(row['point'], row['order']) for row in rows] == [(p, o) for p in range(1, 101) for o in (0, 1)]
    assert all(math.isfinite(v) for row in rows for v in row['coefficients'] + [row['accuracy']])
    assert all(row['accuracy'] >= 0 for row in rows)
    return rows


def verify(run_id, post_run_assets=False):
    target = HERE / run_id
    result, protocol, preflight = [load(target / name) for name in ('result.json', 'protocol.json', 'preflight.json')]
    prepared = load(HERE / 'preparation.json')
    control_path = HERE.parent / 'process-qcd3-v1/backend-6/result.json'
    controls = load(control_path)
    reference_path = HERE.parent.parent / 'symmetric-factorization-production/dev-qcd3-v1/data/candidate-qcd-3-6-0.json'
    reference = load(reference_path)
    assert result['complete'] and result['all_samples_retained'] and not result['acceptance_eligible']
    assert preflight['passed'] and preflight['cpu'] == protocol['cpu']
    assert preflight['idle'][str(protocol['cpu'])] >= .9 and min(preflight['idle'].values()) >= .8
    assert protocol['pairs'] == 21 and protocol['loops'] == 1 and protocol['points'] == 100 and protocol['backend'] == 6
    assert protocol['all_samples_retained'] and protocol['original_first_on_odd_pairs']
    assert protocol['plan_sha256'] == prepared['plan_sha256'] == sha(HERE / 'PLAN.md')
    assert protocol['preparation_sha256'] == sha(HERE / 'preparation.json')
    assert protocol['runner_sha256'] == sha(target / 'run_pairs.py') == sha(HERE / 'run_pairs.py')
    assert prepared['prerequisite_ninja6_sha256'] == sha(control_path)
    assert protocol['reference_sha256'] == sha(reference_path)
    assert protocol['point_sha256'] == reference['points_sha256'] == sha(ROOT / 'docs/benchmarkfolder/points/qcd-3.dat')
    assert protocol['point_sha256'] == sha(HERE.parent / 'process-qcd3-v1/points.dat')
    assert len(result['measurements']) == 42 and len(result['comparisons']) == 21
    expected_order = [(pair, pos, variant) for pair in range(1, 22)
                      for pos, variant in enumerate(('original', 'factor') if pair % 2 else ('factor', 'original'), 1)]
    assert [(r['pair'], r['position'], r['variant']) for r in result['measurements']] == expected_order
    times = {'original': [], 'factor': []}
    rows_by_pair = {}
    card_hashes = set()
    parameter_hashes = set()
    settings = []
    for record in result['measurements']:
        pair, variant = record['pair'], record['variant']
        folder = target / f'pair-{pair:02d}-{variant}'
        assert record['returncode'] == 0 and record['exact_correctness_stream']
        assert record['command'] == ['taskset', '-c', str(protocol['cpu']), './check',
            str(ROOT / 'docs/benchmarkfolder/points/qcd-3.dat'), '1']
        assert record['cwd'] == prepared['builds'][variant]['cwd']
        assert sha(folder / 'run.log') == record['log_sha256']
        rows = parse_rows(folder / 'values.dat')
        assert sha(folder / 'values.dat') == record['values_sha256']
        assert rows == record['rows'] == controls['measurements']['original' if variant == 'original' else 'factor-trace']['rows']
        assert list(map(int, (folder / 'dimensions.dat').read_text().split())) == record['dimensions'] == [1, 1, 1]
        hist_rows = [tuple(map(int, line.split())) for line in (folder / 'timed_status.dat').read_text().splitlines()]
        assert len({a for a, _ in hist_rows}) == len(hist_rows)
        assert dict(hist_rows) == {216: 98, 328: 2}
        assert {str(k): v for k, v in hist_rows} == record['timed_status_counts']
        assert Counter(r['status'] for r in rows if r['order'] == 0) == Counter({216: 98, 328: 2})
        assert [r['point'] for r in rows if r['order'] == 0 and r['status'] == 328] == [44, 80]
        cold, timed = map(float, (folder / 'timing.dat').read_text().split())
        assert math.isfinite(cold) and math.isfinite(timed) and min(cold, timed) > 0
        assert cold == record['cold_seconds'] and timed == record['timed_seconds']
        assert timed / 100 == record['seconds_per_point']
        times[variant].append(timed / 100)
        rows_by_pair[pair, variant] = rows
        card_hashes.add(sha(folder / 'MadLoopParams.dat'))
        parameter_hashes.add(sha(folder / 'param_card.dat'))
        settings.append(record['shared_numerical_settings'])
    assert len(card_hashes) == len(parameter_hashes) == 1 and all(value == settings[0] for value in settings)
    assert parameter_hashes == {reference['parameter_card_sha256']}
    assert card_hashes == {reference['madloop_card_sha256']}
    assert settings[0] == reference['shared_numerical_settings']
    assert settings[0]['mlstabthres'] == 1e-9 and settings[0]['collierrequiredaccuracy'] == 1e-12
    worst = Fraction(0)
    compared = 0
    for pair in range(1, 22):
        a, b = rows_by_pair[pair, 'original'], rows_by_pair[pair, 'factor']
        for point in range(1, 101):
            ar, br = a[2*(point-1):2*point], b[2*(point-1):2*point]
            norm = max(abs(Fraction(x)) for row in ar + br for x in row['coefficients'])
            for left, right in zip(ar, br, strict=True):
                for x, y in zip(left['coefficients'], right['coefficients'], strict=True):
                    x, y = Fraction(x), Fraction(y)
                    tolerance = Fraction(1, 10**7) * max(abs(x), abs(y)) + Fraction(1, 10**10) * norm
                    delta = abs(x-y)
                    assert delta <= tolerance
                    worst = max(worst, delta / tolerance if tolerance else Fraction(0))
                    compared += 1
    import numpy as np
    original, factor = np.array(times['original']), np.array(times['factor'])
    indices = np.random.default_rng(410083).integers(0, 21, size=(10000, 21))
    ratios = np.median(factor[indices], axis=1) / np.median(original[indices], axis=1)
    recomputed = dict(ratio_of_medians=float(np.median(factor)/np.median(original)),
        ci95=np.quantile(ratios, [.025, .975]).tolist(), pair_ratios=(factor/original).tolist(),
        original_median_seconds_per_point=float(np.median(original)), factor_median_seconds_per_point=float(np.median(factor)),
        bootstrap_seed=410083, bootstrap_resamples=10000)
    assert recomputed == result['statistics']
    activity = [json.loads(line) for line in (target / 'activity.jsonl').read_text().splitlines()]
    assert len(activity) > 2 and all(a['monotonic'] < b['monotonic'] for a, b in zip(activity, activity[1:]))
    post_run = None
    if post_run_assets:
        sys.path.insert(0, str(ROOT / 'tools'))
        import ninjars_benchmark as bench
        import ninjars_benchmark_policy_v2 as policy
        post_run = {}
        for variant, expected in prepared['builds'].items():
            output = ROOT / '.benchmark-work/experiments/symmetric-factorization-dp/process-timing21-v2' / variant
            sub = next(output.glob('SubProcesses/P*_*'))
            actual = dict(executable_sha256=sha(sub / 'check'),
                generated_sources_sha256=bench.generated_source_manifest(output)['files_sha256'],
                libraries=policy.library_hashes(output),
                metadata={name: policy.bound_file(path) for name, path in policy.runtime_metadata_files(output).items()},
                link_inputs={name: sha(output / name) for name in policy.RUNTIME_LINK_INPUTS})
            assert all(actual[key] == expected[key] for key in actual)
            post_run[variant] = actual
    return dict(complete=True, engine_calls=0, pairs=21, invocations=42, compared_components=compared,
        all_rows_statuses_accuracies_equal_bound_controls=True, all_samples_retained=True,
        maximum_tolerance_fraction=float(worst), statistics=recomputed,
        post_run_assets=post_run,
        asset_check_scope='Optional current-workspace post-run rehash against preparation; not a claim of per-invocation asset checks.',
        result_sha256=sha(target / 'result.json'), verifier_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    parser.add_argument('--post-run-assets', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.run_id, args.post_run_assets), indent=2))
