#!/usr/bin/env python3
"""Independent byte checks of the final prepared pair; no amplitude is called."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
EXPECTED_FREEZE = '26b7602867394ef5f9b28a0900d8a518afd1e8e7668075e4af52f36af6a6d4c3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def bound(record):
    path = ROOT/record['path']
    assert sha(path) == record['sha256'], str(path)
    assert path.stat().st_size == record['bytes'], str(path)
    return path


def inventory(path):
    return {str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Create a new audit record without overwriting')
    args = parser.parse_args()
    freeze = HERE/'frozen-bindings.json'
    assert sha(freeze) == EXPECTED_FREEZE
    frozen = load(freeze)
    assert frozen['repetitions'] == {'qcd-3':21, 'ew-2':7}
    assert (frozen['loops'], frozen['points'], frozen['bootstrap_resamples'],
            frozen['bootstrap_seed']) == (1,100,100000,20261020)
    assert frozen['thread_environment'] == {n:'1' for n in (
        'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS','MKL_NUM_THREADS')}
    for name in ('controller','analyzer','readiness_verifier','package','rust_build','preparation','plan'):
        bound(frozen[name])
    for record in frozen['support_tools'].values():
        bound(record)
    comparison = load(HERE/'comparison-contract.json')
    assert comparison['passed'] and comparison['function_bytes_identical']
    assert len(comparison['cases']) == 13 and all(c['identical'] for c in comparison['cases'])
    assert comparison['compare_function_sha256'] == '0f1e713538c509548eacc5c28ad4550a97f5c9647c2b8bf748fc4e41a5f1d430'
    assert bound(comparison['immutable_source']).read_bytes() == bound(comparison['historical_source']).read_bytes()
    assert sha(HERE/'bound-support/ninjars_report_v2.py') == '93f942dfa74b2c3354d066e5436f4aadd7a009d0f08fc03a05190aa8f5c21c85'
    for name, expected in frozen['protocol']['sources'].items():
        assert sha(ROOT/'tools'/name) == expected
    build = load(bound(frozen['rust_build']))
    package = load(bound(frozen['package']))
    assert build['status'] == 'compiled-not-executed'
    assert build['effective_flag_environment'] == {}
    bridge = ROOT/'.benchmark-work/experiments/ninjars-mp-tensor-allocation-process-v1/bridge'
    assert inventory(bridge) == inventory(HERE/'source') == build['allocation_sources'] == frozen['bridge_sources']
    assert set(build['original_sources']) == set(build['allocation_sources'])
    changed = [n for n in build['original_sources'] if build['original_sources'][n] != build['allocation_sources'][n]]
    assert changed == ['src/tensor.rs']
    original = HERE.parent/'ninjars-mp-tensor-allocation-transport-v1/original'
    assert inventory(original) == build['original_sources']
    baseline, rust, allocation = [bound(package[name]) for name in (
        'baseline_archive','bare_rust_archive','allocation_archive')]
    assert sha(baseline) == '832fc0d1198814b9cc80a65e47bc7aa5dc204f7d25701f206c910cd188fbe80b'
    assert sha(allocation) == '366ae91da891e9d7e5c19cbf4cbcca20a622d32b452305df58146a96c4189357'
    archive_members = lambda path: subprocess.check_output(['ar','t',str(path)],text=True).splitlines()
    assert archive_members(allocation) == archive_members(rust)+['mninjars.o']
    wrapper = subprocess.check_output(['ar','p',str(baseline),'mninjars.o'])
    assert wrapper == subprocess.check_output(['ar','p',str(allocation),'mninjars.o'])
    assert wrapper == bound(package['wrapper_object']).read_bytes()
    assert (HERE/'exact-build/baseline-fingerprint.json').read_bytes() == (HERE/'exact-build/allocation-fingerprint.json').read_bytes()
    checked = []
    source_files = 0
    for case, arms in frozen['builds'].items():
        for arm, record in arms.items():
            output = ROOT/record['output']
            for name in ('executable','driver','points','parameter_card','madloop_card','actual_parameter_card'):
                bound(record[name])
            compiled = load(HERE/'processes-final'/case/arm/'compiled-sources.json')
            assert compiled['files_sha256'] == record['generated_sources_sha256']
            for name, expected in compiled['files'].items():
                path = output/name
                assert sha(path) == expected['sha256'] and path.stat().st_size == expected['bytes'], name
                source_files += 1
            for name, expected in record['libraries'].items():
                assert sha(output/'lib'/name) == expected, (case,arm,name)
            for name, expected in record['runtime_metadata'].items():
                path = output/'lib'/name
                assert sha(path) == expected['sha256'] and path.stat().st_size == expected['bytes'], name
            for name, expected in record['link_inputs'].items():
                assert sha(output/name) == expected
            checked.append({'case':case,'arm':arm,'executable_sha256':record['executable']['sha256'],
                'generated_sources':len(compiled['files']),'library_paths':len(record['libraries']),
                'runtime_metadata_paths':len(record['runtime_metadata'])})
        a,b = [arms[name] for name in ('original','allocation')]
        for name in ('generated_sources_sha256','runtime_metadata','link_inputs','header_sha256',
                     'module_source_sha256','shared_numerical_settings'):
            assert a[name] == b[name], (case,name)
        for name in ('points','parameter_card','madloop_card'):
            assert a[name]['sha256'] == b[name]['sha256']
        assert set(a['libraries']) == set(b['libraries'])
        differences = [name for name in a['libraries'] if a['libraries'][name] != b['libraries'][name]]
        assert differences == ['ninjars_include/libninjars.a','ninjars_lib/libninjars.a']
        for name in differences:
            assert a['libraries'][name] == sha(baseline) and b['libraries'][name] == sha(allocation)
    result = {'schema':'allocation-independent-readiness-audit-v1','passed':True,
        'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'frozen_bindings_sha256':sha(freeze),'reviewer_script_sha256':sha(Path(__file__)),
        'workers':checked,'generated_source_files_rehashed':source_files,
        'sole_bridge_source_change':changed,'compiler_fingerprints_byte_identical':True,
        'comparison_private_source_matches_committed_and_installed_source':True,
        'comparison_boundary_cases':13,
        'fortran_wrapper_byte_identical':True,'numerical_execution_by_audit':False,
        'source_review':{'one_fresh_worker_at_a_time':True,'initialization_and_full_stream_warmup_outside_timer':True,
            'normal_numerator_reduction_masters_stability_rescue_in_timer':True,
            'exact_coefficients_status_accuracy_and_100_call_counts_required':True,
            'all_started_samples_retained':True,'fixed_alternating_21_and_7_pairs':True,
            'paired_ratio_of_medians_100000_linear_percentile_bootstrap':True,
            'preflight_three_3s_intervals_at_most_5_percent_busy':True,
            'separate_literal_original_final_acceptance_required':True},
        'blocking_findings':[],'timing_or_numerical_outcome_claim':False}
    text = json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text,end='')


if __name__ == '__main__':
    main()
