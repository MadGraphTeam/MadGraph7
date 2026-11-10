#!/usr/bin/env python3
"""Reparse retained data only; never compile, load a native library, or run physics.

Default: verify the frozen package and its repository evidence. --check-current
also requires live source hashes to match the source-only capture. --capture is
one-time creation, refusing to overwrite any generated package artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / 'docs/benchmarkfolder/qualification/tirust-bridge/stability-audit'
FIXTURE = ROOT / '../references/TIRust/crates/iregi-rust/tests/fixtures/gate_passed_lanes.json'
MANIFEST = ROOT / 'docs/benchmarkfolder/campaigns/split-dp-terminal-1/manifest.json'
TEMPLATE = ROOT / 'madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc'
COMMONS = ROOT / 'Template/loop_material/StandAlone/SubProcesses/MadLoopCommons.inc'
OLD_DSO = '49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'
CURRENT_DSO = 'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9'
FIXTURE_SHA = '4ebf900f99805d7894a7c484f1755c896307cea97e032942937b88cd3f3459e8'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(path):
    data = path.read_bytes()
    return {'sha256': sha(data), 'bytes': len(data)}


def read(path):
    return json.loads(path.read_text())


def dump(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def excerpt(text, start, end):
    first = text.index(start)
    last = text.index(end, first)
    value = text[first:last]
    return {'start_line': text[:first].count('\n') + 1,
            'end_line': text[:last].count('\n'), 'text': value,
            'sha256': sha(value.encode())}


def accuracy(evaluations):
    """Production COMPUTE_ACCURACY for one active, finite split-order lane."""
    midpoints = [(max(v) + min(v)) / 2.0 for v in zip(*evaluations)]
    acc = avg = 0.0
    for values, midpoint in zip(zip(*evaluations), midpoints):
        spread = max(values) - min(values)
        relative = spread if midpoint == 0.0 else spread / abs(midpoint)
        acc += relative * abs(midpoint)
        avg += midpoint
    if avg != 0.0:
        acc /= abs(avg) / 3.0
    return acc, midpoints


def symmetric(a, b):
    denominator = abs(a) + abs(b)
    return abs(a-b) / denominator if denominator else 0.0


def relative(a, b):
    return abs(a-b) / abs(b) if b else (0.0 if a == 0.0 else None)


def close(a, b):
    assert math.isclose(a, b, rel_tol=2e-14, abs_tol=1e-30), (a, b)


def observations():
    raw = read(OLD / 'release-raw-and-native.json')
    host = read(OLD / 'host-dispatch-release.json')
    adjudication = read(OLD / 'release-adjudication.json')
    assert raw['library_sha256'] == OLD_DSO
    assert host['native_report_sha256'] == identity(OLD / 'release-raw-and-native.json')['sha256']
    assert adjudication['input_sha256'] == host['native_report_sha256']
    assert read(OLD / 'host-dispatch-adjudication.json')['input_sha256'] == identity(OLD / 'host-dispatch-release.json')['sha256']
    rows = []
    expected_keys = {(t, r, p) for t in (1e-3, 1e-6)
                     for r in (0, 2) for p in ('real', 'imaginary')}
    assert len(host['rows']) == 8
    assert {(r['threshold'], r['rotations'], r['projection']) for r in host['rows']} == expected_keys
    for row in host['rows']:
        n = 2 + row['rotations']
        strict = row['threshold'] == 1e-6
        dp_acc, dp_midpoint = accuracy(row['evaluations'][:n])
        close(dp_acc, row['dp_accuracy'])
        active = row['evaluations'][n:] if strict else row['evaluations']
        final_acc, final_midpoint = accuracy(active)
        close(final_acc, row['accuracy'])
        assert final_midpoint == row['result']
        assert row['calls'] == len(row['modes']) == len(row['evaluations'])
        assert row['status'] == 0 and row['backend_statuses'] == [0] * row['calls']
        dp_modes = [1, 2] + [1] * row['rotations']
        assert row['modes'] == dp_modes + ([4, 5] + [4] * row['rotations'] if strict else [])
        assert row['return_code'] == (323 if strict else 219)
        assert (dp_acc >= row['threshold']) == strict
        truth = row['same_input_native_qp_truth']
        derived = {
            'recomputed_dp_accuracy': dp_acc,
            'recomputed_final_accuracy': final_acc,
            'finite_symmetric_error_to_same_input_qp': symmetric(row['result'][0], truth[0]),
            'finite_ordinary_relative_error_to_same_input_qp': relative(row['result'][0], truth[0]),
            'finite_ordinary_relative_error_to_fixture_truth': relative(row['result'][0], row['fixture_truth'][0]),
            'all_coefficients_equal_same_input_qp': row['result'] == truth,
            'passes_independent_finite_symmetric_3e_minus_6': symmetric(row['result'][0], truth[0]) <= 3e-6,
        }
        close(derived['finite_ordinary_relative_error_to_same_input_qp'], row['relative_error_to_same_input_qp'])
        close(derived['finite_ordinary_relative_error_to_fixture_truth'], row['relative_error_to_fixture_truth'])
        assert derived['passes_independent_finite_symmetric_3e_minus_6'] == strict
        if strict:
            assert derived['all_coefficients_equal_same_input_qp']
            assert row['accuracy'] == 0.0
            assert derived['finite_ordinary_relative_error_to_fixture_truth'] < 1.2e-15
        rows.append({'historical_row': row, 'derived': derived})
    cache_checks = []
    for origin, precisions in raw['cases'].items():
        for precision, values in precisions.items():
            if precision not in ('dp', 'qp'):
                continue
            check, off, shared = [values[k] for k in ('reverse_check_bypass', 'reverse_cache_off', 'reverse_shared_cache')]
            assert check['coefficients'] == off['coefficients'] == shared['coefficients']
            assert check['cache_before'] == check['cache_after']
            assert off['cache_before'] == off['cache_after']
            assert values['primary']['coefficients'] == values['repeat_cached']['coefficients']
            cache_checks.append({'origin': origin, 'precision': precision,
                                 'reversal_modes_equal': True, 'bypass_counters_unchanged': True})
    assert len(cache_checks) == 4
    return {
        'schema_version': 1, 'scope': 'read-only reparse of historical synthetic tensor/host-helper execution',
        'historical_bridge_dso_sha256': OLD_DSO,
        'current_campaign_bridge_dso_sha256': CURRENT_DSO,
        'current_campaign_bridge_replayed_for_this_fixture': False,
        'fixture': raw['label'], 'coefficient_order': ['finite', 'single_pole', 'double_pole'],
        'metric_definitions': {
            'symmetric': 'abs(value-truth)/(abs(value)+abs(truth)); both-zero maps to zero',
            'ordinary_relative': 'abs(value-truth)/abs(truth)',
            'host_accuracy': 'range/midpoint consistency, weighted across three Laurent coefficients; see current-compute-accuracy.inc',
            'zero_qp_accuracy': 'identical rounded reported QP evaluations, not zero mathematical error',
        },
        'raw_original_origin_dp': raw['cases']['original_origin']['dp']['primary'],
        'generated_adapter_dp': raw['generated_adapter']['dp']['primary'],
        'generated_adapter_qp': raw['generated_adapter']['qp']['primary'],
        'rows': rows, 'cache_checks': cache_checks,
        'strict_rescue_for_retained_helper': 'closed: all four 1e-6 cases execute QP and return 323',
        'default_policy_robustness': 'retained limitation: all four 1e-3 cases accept correlated DP error as 219',
        'same_fixture_other_backend_default_host_measurements': [],
        'other_backend_reference_only': 'Supplied fixture truth is NinjaRS generic at 40 digits; it is not a current MadLoop NinjaRS DP/default-host comparison.',
        'production_fix_justified_by_this_case_alone': False,
        'full_physical_amplitude_qualification': False,
        'performance_qualification': False,
    }


def capture():
    generated = ['observations.json', 'inputs.json', 'source-bindings.json',
                 'current-compute-accuracy.inc', 'current-source-excerpts.json', 'index.json']
    assert not any((HERE / name).exists() for name in generated), 'Capture refuses overwrites'
    raw = read(OLD / 'release-raw-and-native.json')
    assert identity(FIXTURE)['sha256'] == FIXTURE_SHA == raw['fixture_sha256']
    fixture = read(FIXTURE)
    lane, = [x for x in fixture['lanes'] if x['label'] == raw['label']]
    inputs = {'original_fixture_source': fixture['source'], 'original_fixture_file_sha256': FIXTURE_SHA,
              'original_fixture_lane': lane, 'generated_adapter_input': raw['generated_adapter']['input'],
              'input_policy': raw['generated_adapter']['input_policy'],
              'normalization': raw['generated_adapter']['normalization']}
    template = TEMPLATE.read_text()
    commons = COMMONS.read_text()
    extracts = {
        'host_policy': excerpt(template, 'IF(.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.(CTMODERUN.EQ.-1)) THEN', 'C Finally for the summed result in ANS'),
        'retcode': excerpt(commons, '      INTEGER FUNCTION SET_RET_CODE_U', '      SUBROUTINE DETECT_LOOPLIB'),
        'compute_accuracy': excerpt(template, '      SUBROUTINE %(proc_prefix)scompute_accuracy', '      SUBROUTINE %(proc_prefix)sSET_N_EVALS'),
        'precision_contract': excerpt(template, '      SUBROUTINE %(proc_prefix)sSLOOPMATRIX_THRES', '      IMPLICIT NONE'),
    }
    for key, name in [('host_policy', 'host-production-policy.inc'), ('retcode', 'host-production-retcode.inc')]:
        assert extracts[key]['text'].encode() == (OLD / name).read_bytes()
        extracts[key]['equals_retained_historical_excerpt'] = name
    host = read(OLD / 'host-dispatch-release.json')
    assert identity(TEMPLATE)['sha256'] == host['build_metadata']['source_hashes'][str(TEMPLATE.relative_to(ROOT))]
    historical = set(read(OLD / 'host-dispatch-index.json')['files'])
    historical.update(['README.md', 'release-raw-and-native.json', 'release-adjudication.json',
                       'release-build-metadata.json', 'host-dispatch-index.json', 'host-dispatch-adjudication.json',
                       'build_harness.py', 'harness_driver.f', 'check_stability.py', 'adjudicate.py', 'adjudicate_host.py'])
    for name, info in read(OLD / 'host-dispatch-index.json')['files'].items():
        assert identity(OLD / name) == info
    historical_paths = [OLD / name for name in sorted(historical)]
    historical_paths += [ROOT / 'docs/benchmarkfolder/qualification/benchmark-audit/stability-policy.md', MANIFEST]
    recorded = read(OLD / 'release-build-metadata.json')['input_sha256']
    source_paths = [p for p in recorded if not p.startswith(('.tirust-build/', 'HEPTools/', 'docs/'))]
    source_paths += ['vendor/tirust/src/lib.rs', 'Template/loop_material/StandAlone/SubProcesses/MadLoopCommons.inc',
                     'Template/loop_material/StandAlone/Cards/MadLoopParams.dat']
    extra_history = read(OLD / 'release-adjudication.json')['source_sha256']
    extra_history.update(host['build_metadata']['source_hashes'])
    current = {}
    for name in sorted(set(source_paths)):
        info = identity(ROOT / name)
        historical_sha = recorded.get(name, extra_history.get(name))
        current[name] = {**info, 'historical_sha256': historical_sha,
                         'equals_historical_source': info['sha256'] == historical_sha}
    manifest = read(MANIFEST)
    assert manifest['candidate']['native']['files']['lib/libmadloop_tirust.so'] == CURRENT_DSO
    bindings = {
        'schema_version': 1,
        'historical_repository_files': {str(p.relative_to(ROOT)): identity(p) for p in historical_paths},
        'external_read_only_fixture': {'path_from_repo': str(FIXTURE.relative_to(ROOT)), **identity(FIXTURE)},
        'current_source_capture': current,
        'current_source_live_equality_required_for_portable_reparse': False,
        'compute_accuracy_historical_equivalence': 'Full loop_matrix_standalone.inc bytes equal its recorded historical execution hash; therefore this extracted routine is byte-identical.',
        'current_manifest_native_field': '/candidate/native/files/lib/libmadloop_tirust.so',
        'current_manifest_dso_sha256': CURRENT_DSO,
        'current_dso_same_fixture_execution_claimed': False,
    }
    content = {'observations.json': dump(observations()), 'inputs.json': dump(inputs),
               'source-bindings.json': dump(bindings), 'current-source-excerpts.json': dump(extracts),
               'current-compute-accuracy.inc': extracts['compute_accuracy']['text'].encode()}
    for name, data in content.items():
        (HERE / name).write_bytes(data)
    files = {p.name: identity(p) for p in sorted(HERE.iterdir()) if p.is_file()}
    (HERE / 'index.json').write_bytes(dump({'schema_version': 1,
        'kind': 'read-only-derived evidence; no new compiler/native/test execution',
        'files': files, 'limitations_preserved': True}))


def verify(check_current=False):
    index = read(HERE / 'index.json')
    assert {p.name for p in HERE.iterdir() if p.is_file()} == set(index['files']) | {'index.json'}
    for name, expected in index['files'].items():
        assert identity(HERE / name) == expected, name
    bindings = read(HERE / 'source-bindings.json')
    for name, expected in bindings['historical_repository_files'].items():
        assert identity(ROOT / name) == expected, name
    assert read(HERE / 'observations.json') == observations()
    inputs = read(HERE / 'inputs.json')
    raw = read(OLD / 'release-raw-and-native.json')
    assert inputs['generated_adapter_input'] == raw['generated_adapter']['input']
    assert inputs['original_fixture_lane']['label'] == raw['label']
    assert inputs['original_fixture_file_sha256'] == raw['fixture_sha256'] == FIXTURE_SHA
    if FIXTURE.exists():
        assert identity(FIXTURE)['sha256'] == FIXTURE_SHA
        fixture = read(FIXTURE)
        assert inputs['original_fixture_lane'] in fixture['lanes']
        assert inputs['original_fixture_source'] == fixture['source']
    excerpts = read(HERE / 'current-source-excerpts.json')
    for value in excerpts.values():
        assert sha(value['text'].encode()) == value['sha256']
        if 'equals_retained_historical_excerpt' in value:
            assert value['text'].encode() == (OLD / value['equals_retained_historical_excerpt']).read_bytes()
    assert (HERE / 'current-compute-accuracy.inc').read_text() == excerpts['compute_accuracy']['text']
    if check_current:
        for name, expected in bindings['current_source_capture'].items():
            assert identity(ROOT / name) == {k: expected[k] for k in ('sha256', 'bytes')}, name
    print(json.dumps({'passed': True, 'historical_host_rows': 8,
                      'default_dp_acceptances': 4, 'strict_qp_rescues': 4,
                      'indexed_files': len(index['files']),
                      'checked_current_sources': check_current,
                      'new_native_or_compiler_execution': False}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--check-current', action='store_true')
    args = parser.parse_args()
    if args.capture:
        capture()
    verify(args.check_current)
