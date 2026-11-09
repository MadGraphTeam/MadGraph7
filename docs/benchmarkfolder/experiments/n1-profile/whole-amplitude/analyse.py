#!/usr/bin/env python3
"""Summarize a retained, explicitly ineligible full-amplitude diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def distribution(values):
    return {'values':values,'median':statistics.median(values),'minimum':min(values),'maximum':max(values)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    source=args.directory/'results.json'
    data=json.loads(source.read_text())
    result={'performance_eligible':False,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'cpu':data['cpu'],'points_sha256':data['points_sha256'],
            'candidate_library':data['candidate_library'],'build_metadata':data['build_metadata'],
            'original_uses_candidate_library':data.get('original_uses_candidate_library',False),
            'fortran_build_metadata':data.get('fortran_build_metadata'),
            'paired_statistics':data['paired_statistics'],
            'improvement_established_by_diagnostic_interval':data['paired_statistics']['split_over_original']['upper_95']<1.0,
            'wall_microseconds_per_amplitude':{key:distribution([v*1e6 for v in values])
                                                for key,values in data['timings'].items()},
            'all_original_split_bit_identical':data['all_original_split_bit_identical'],
            'maximum_original_split_relative_error':data['maximum_original_split_relative_error'],
            'all_measured_and_calibration_parity_passed':all(r['parity_passed'] and r['calibration_parity_passed']
                                                            for r in data['runs']),
            'all_expected_backend_usage':all(r['expected_backend_usage_passed'] for r in data['runs']),
            'limits':['Diagnostic triads with concurrent ordinary CutTools work and explicit host-load records.',
                      'This CPU differs from prior attribution; do not subtract their absolute stage times.',
                      'Profiling bridge has disabled timer branches in the main comparisons.',
                      'Enabled-timer stages are inclusive and cannot be added to full-amplitude or nested timings.']}
    result['variant_descriptions'] = {
        'baseline':'Frozen NinjaRS-first baseline executable and libraries',
        'original':('Split-workspace DSO with original Fortran adapter' if data.get('fortran_build_metadata')
                    else 'Original TIRust DSO with original Fortran adapter'),
        'split':((json.loads((args.directory/'candidate-fortran-build-metadata.json').read_text()).get(
                   'candidate_description','Same split-workspace DSO with adapter-only -O3 build'))
                 if data.get('fortran_build_metadata') else 'Split-workspace DSO with original Fortran adapter')}
    profiles=[r for r in data['runs'] if r['variant']=='profile-split']
    if profiles:
        points=data['loops']*100
        result['profile']={'stage_raw_microseconds_per_amplitude':{
            str(stage):distribution([sum(x['seconds_per_point'] for x in r['profile']['stages']
                                         if x['stage']==stage)*1e6 for r in profiles]) for stage in range(1,8)},
            'bridge_microseconds_per_amplitude':{
                key:distribution([sum(x[key] for x in r['profile']['bridge']['rows'])/points/1000 for r in profiles])
                for key in ('ffi_total_ns','input_batch_ns','store_attach_ns','reduce_batch_ns','output_pack_ns',
                            'output_copy_ns','workspace_setup_ns','upstream_arguments_ns',
                            'upstream_scalar_masters_ns','upstream_reduction_ns','upstream_total_ns')},
            'all_stage_counts_expected':all(r['expected_stage_counts_passed'] for r in profiles),
            'failed_calls':sum(r['bridge_failed_calls'] for r in profiles),
            'arguments_label':'Upstream geometry/read_lane/Gram/variant/conditioning, not Symbolica expression arguments.'}
    (args.directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_exact':result['all_original_split_bit_identical'],
                      'split_over_original':result['paired_statistics']['split_over_original']},indent=2))


if __name__=='__main__':
    main()
