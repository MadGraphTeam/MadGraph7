#!/usr/bin/env python3
"""Apply the pre-sample root rule to the complete fixed request-screen series."""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

HERE = Path(__file__).resolve().parent
POLICIES = ('Interpreted', 'NativeDenominators', 'FullNative')
ORDERS = ((0,1,2),(1,2,0),(2,0,1),(0,2,1),(2,1,0),(1,0,2),(0,1,2))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    position = (len(values)-1)*fraction
    left = int(position)
    right = min(left+1, len(values)-1)
    return values[left]+(values[right]-values[left])*(position-left)


def paired_ratio(numerator, denominator):
    assert len(numerator) == len(denominator) == 7
    rng = random.Random(20261021)
    ratios = []
    for _ in range(100000):
        indices = [rng.randrange(7) for _ in range(7)]
        ratios.append(statistics.median(numerator[index] for index in indices)
            /statistics.median(denominator[index] for index in indices))
    ratios.sort()
    interval = [percentile(ratios,.025), percentile(ratios,.975)]
    return {'ratio_of_medians': statistics.median(numerator)/statistics.median(denominator),
        'paired_bootstrap_95_interval': interval,
        'demonstrated_request_scope_gain': interval[1] < 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('screening_analysis', type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    analysis_path = args.screening_analysis.resolve(strict=True)
    analysis = json.loads(analysis_path.read_text())
    result = json.loads((run/'result.json').read_text())
    assert result['complete'] and result['returncode'] == 0 and result['stage'] == 'screening'
    assert analysis['complete'] and analysis['all_output_status_controls_exact']
    assert analysis['all_samples_retained'] and analysis['triplets'] == 7 and analysis['samples'] == 21
    assert analysis['result_sha256'] == sha(run/'result.json')
    assert analysis['plan_sha256'] == sha(HERE/'SCREENING-PLAN.md')
    assert analysis['raw_sha256'] == result['files']['raw.jsonl'] == sha(run/'raw.jsonl')
    # Independently reconstruct the exact fixed triplet series rather than
    # allowing an edited analysis file to supply different measurements.
    samples = []
    with (run/'raw.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'sample': samples.append(row)
    expected = [(triplet,position,arm) for triplet,order in enumerate(ORDERS)
        for position,arm in enumerate(order)]
    assert [(row['triplet'],row['position'],row['arm']) for row in samples] == expected
    times = {arm: [int(next(row for row in samples
        if row['triplet'] == triplet and row['arm'] == arm)['elapsed_ns'])/1e9
        for triplet in range(7)] for arm in range(3)}
    assert all(math.isfinite(value) and value > 0 for values in times.values() for value in values)
    assert {POLICIES[arm]:times[arm] for arm in range(3)} == analysis['seconds_per_complete_1392_request_stream']
    comparisons = {POLICIES[arm]:paired_ratio(times[arm], times[0]) for arm in (1,2)}
    assert comparisons == analysis['comparisons']
    incremental = paired_ratio(times[2], times[1])
    denominator_gain = comparisons['NativeDenominators']['demonstrated_request_scope_gain']
    full_gain = comparisons['FullNative']['demonstrated_request_scope_gain']
    if denominator_gain and full_gain:
        if incremental['demonstrated_request_scope_gain']:
            choice,reason = 'FullNative','Both arms gain; full native also has a resolved incremental gain over denominator-only.'
        else:
            choice,reason = 'NativeDenominators','Both arms gain; the incremental full-native gain is unresolved or negative, so prefer denominator-only.'
    elif denominator_gain:
        choice,reason = 'NativeDenominators','Only denominator-only demonstrates a gain over interpreted.'
    elif full_gain:
        choice,reason = 'FullNative','Only full native demonstrates a gain over interpreted.'
    else:
        choice,reason = 'Interpreted','Neither native arm demonstrates a gain over interpreted.'
    print(json.dumps({'complete':True,'engine_calls':0,'compiler_calls':0,
        'analysis_source_sha256':sha(Path(__file__)),
        'selection_policy_sha256':sha(HERE/'ROOT-SELECTION-POLICY.md'),
        'screening_analysis_sha256':sha(analysis_path),'raw_sha256':sha(run/'raw.jsonl'),
        'result_sha256':sha(run/'result.json'),'fixed_triplets':7,'all_21_samples_retained':True,
        'bootstrap':{'resamples':100000,'seed':20261021,'unit':'paired triplet indices',
            'percentiles':'linear2.5%97.5%'},
        'original_comparisons_vs_interpreted':comparisons,
        'additional_predeclared_full_over_denominators':incremental,
        'provisional_request_screen_choice':choice,'reason':reason,
        'production_change_authorized_by_this_analysis':False,
        'conditions_still_required':['Root review of all numerical gates and disclosed raw precision/history differences',
            'Final selected-source ordinary, fixture, replay and256/384 proof',
            'Final generated-process correctness, stability and full MadLoop performance campaign'],
        'full_MadLoop_performance_claim':False,'original_Ninja_or_5_percent_gate_claim':False},indent=2))


if __name__ == '__main__': main()
