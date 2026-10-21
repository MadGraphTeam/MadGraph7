#!/usr/bin/env python3
"""Independent fixed-screen control and predeclared selection reconstruction."""
from collections import Counter
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT/'.benchmark-work/experiments/ninjars-native-float256-v1'
RUN = HERE/'supplement/run-screening'
POLICIES = ('Interpreted','NativeDenominators','FullNative')
ORDERS = ((0,1,2),(1,2,0),(2,0,1),(0,2,1),(2,1,0),(1,0,2),(0,1,2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def ratio(numerator,denominator):
    generator = random.Random(20261021)
    estimates = []
    for _ in range(100000):
        ids = [generator.randrange(7) for _ in range(7)]
        estimates.append(statistics.median(numerator[i] for i in ids)/
                         statistics.median(denominator[i] for i in ids))
    estimates.sort()
    def percentile(q):
        x = (len(estimates)-1)*q;lo,hi = math.floor(x),math.ceil(x)
        return estimates[lo]+(estimates[hi]-estimates[lo])*(x-lo)
    interval = [percentile(.025),percentile(.975)]
    return dict(ratio_of_medians=statistics.median(numerator)/statistics.median(denominator),
        paired_bootstrap_95_interval=interval,demonstrated_request_scope_gain=interval[1] < 1)


def main():
    os.sched_setaffinity(0,{3})
    result = load(RUN/'result.json')
    assert result['complete'] and result['returncode'] == 0 and result['stage'] == 'screening'
    assert not result['production_modified'] and result['controller_affinity'] == [2] and result['cpu'] == 190
    assert result['thread_environment'] == {name:'1' for name in
        ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS','MKL_NUM_THREADS')}
    for name,expected in result['files'].items():assert sha(RUN/name) == expected,name
    assert sha(HERE/'run_supplement.py') == result['runner_sha256']
    source_audit = load(HERE/'screen-controller-independent-audit.json')
    assert source_audit['passed_source_review'] and not source_audit['blocking_findings']
    for name,record in source_audit['source_files'].items():
        path = HERE/name
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'],name
    build_path = HERE/'supplement/build-completed.json'
    assert sha(build_path) == result['build_completed_sha256']
    build = load(build_path);assert build['complete'] and build['original_executable_restored']
    executable = ROOT/build['executables'][0]['path']
    assert sha(executable) == build['executables'][0]['sha256'] == result['executable_sha256']
    assert result['executable_sha256'] == '26d8648fe69fe4fe38fade56352d950aebd0d7f3e7bce23d94a3387f06e20578'
    for name,expected in build['source_hashes'].items():
        assert sha(WORK/'supplement/bridge'/name) == expected,name
    original = load(HERE/'build.json')
    assert sha(HERE/'build.json') == build['original_build_sha256']
    assert sha(ROOT/original['executables'][0]['path']) == build['original_executable_sha256']
    prepared = load(HERE/'prepared.json')
    assert sha(WORK/'actual-mp-requests.jsonl') == result['input_sha256'] == prepared['capture']['uncompressed_sha256']
    for name,record in result['shared_libraries'].items():
        assert str(Path(name).resolve()) == record['resolved'] and sha(Path(record['resolved'])) == record['sha256']
    handoff_path = Path(result['handoff']['path'])
    if not handoff_path.is_absolute():handoff_path = ROOT/handoff_path
    assert sha(handoff_path) == result['handoff']['sha256']
    handoff = load(handoff_path)
    assert handoff['root_authorized'] and handoff['phase'] == 'native-screening'
    assert handoff['executable_sha256'] == result['executable_sha256']
    assert sha(HERE/'SCREENING-PLAN.md') == handoff['screening_plan_sha256']
    assert sha(HERE/'ROOT-SELECTION-POLICY.md') == handoff['selection_policy_sha256']
    assert sha(HERE/'analyze_selection.py') == handoff['selection_analyzer_sha256']
    preflight_path = ROOT/handoff['preflight_path']
    assert sha(preflight_path) == handoff['preflight_sha256']
    preflight = load(preflight_path)
    assert preflight['complete'] and preflight['qualified'] and preflight['cpu'] == handoff['cpu'] == result['cpu']
    assert preflight['interval_count'] == len(preflight['intervals']) == 3
    assert preflight['interval_seconds'] == 3 and preflight['maximum_busy_fraction'] == .05
    for interval in preflight['intervals']:
        deltas = [b-a for a,b in zip(interval['before'],interval['after'])]
        assert deltas == interval['deltas'] and all(v >= 0 for v in deltas)
        assert interval['busy_fraction'] == 1-deltas[3]/sum(deltas) <= .05
    identity = load(RUN/'worker-identity.json')
    assert identity['pid'] == result['pid'] and identity['affinity'] == [190]
    assert identity['executable_sha256'] == result['executable_sha256']
    assert not (Path('/proc')/str(identity['pid'])).exists(), 'Screening worker remains active'
    controls_path = HERE/'run-replay/raw.jsonl'
    assert sha(controls_path) == result['controls_sha256']
    controls = [json.loads(line) for line in controls_path.read_text().splitlines()]
    assert len(controls) == 1393 and controls[-1]['kind'] == 'summary'
    assert [r['sequence'] for r in controls[:-1]] == list(range(1392))
    sample_outputs,warm_outputs = Counter(),Counter()
    samples,warmups = [],[]
    summary = None
    for line in (RUN/'raw.jsonl').read_text().splitlines():
        row = json.loads(line);kind = row['kind']
        if kind in ('sample_output','warmup_output'):
            arm,seq = row['arm'],row['sequence'];observed = row['result'];control = controls[seq]['arms'][arm]
            assert observed['status'] == control['status'] == 0 and observed['passed'] is True
            assert observed['output_bytes'] == 512 and len(observed['output']) == 8
            assert observed['output'] == control['output']
            if kind == 'sample_output':
                assert ORDERS[row['triplet']][row['position']] == arm
                key = (row['triplet'],row['position'],arm)
                assert seq == sample_outputs[key];sample_outputs[key] += 1
            else:
                assert seq == warm_outputs[arm];warm_outputs[arm] += 1
        elif kind in ('sample','warmup'):
            assert row['policy'] == POLICIES[row['arm']] and row['requests'] == 1392 and int(row['elapsed_ns']) > 0
            (samples if kind == 'sample' else warmups).append(row)
        elif kind == 'summary':
            assert summary is None;summary = row
        else:raise AssertionError(kind)
    expected = [(triplet,position,arm) for triplet,order in enumerate(ORDERS) for position,arm in enumerate(order)]
    assert [(r['triplet'],r['position'],r['arm']) for r in samples] == expected
    assert sample_outputs == Counter({key:1392 for key in expected})
    assert list(warm_outputs.items()) == [(0,1392),(1,1392),(2,1392)]
    assert [r['arm'] for r in warmups] == [0,1,2] and all(r['performance_sample'] is False for r in warmups)
    assert summary == result['summary'] and summary['complete'] and summary['input_unchanged']
    assert summary['failed_outputs'] == 0 and summary['samples'] == 21 and summary['triplets'] == 7
    assert summary['warmup_streams'] == 3 and summary['requests_per_sample'] == 1392
    values = {arm:[int(next(r for r in samples if r['triplet'] == triplet and r['arm'] == arm)['elapsed_ns'])/1e9
                   for triplet in range(7)] for arm in range(3)}
    analysis,selection = load(HERE/'screening-analysis.json'),load(HERE/'selection-analysis.json')
    assert analysis['seconds_per_complete_1392_request_stream'] == {POLICIES[a]:v for a,v in values.items()}
    comparisons = {POLICIES[arm]:ratio(values[arm],values[0]) for arm in (1,2)}
    incremental = ratio(values[2],values[1])
    assert comparisons == analysis['comparisons'] == selection['original_comparisons_vs_interpreted']
    assert incremental == selection['additional_predeclared_full_over_denominators']
    denominator_gain = comparisons['NativeDenominators']['demonstrated_request_scope_gain']
    full_gain = comparisons['FullNative']['demonstrated_request_scope_gain']
    if denominator_gain:
        choice = 'FullNative' if full_gain and incremental['demonstrated_request_scope_gain'] else 'NativeDenominators'
    else:choice = 'FullNative' if full_gain else 'Interpreted'
    assert choice == selection['provisional_request_screen_choice'] == 'Interpreted'
    activity = [json.loads(line) for line in (RUN/'activity.jsonl').read_text().splitlines()]
    assert len(activity) == analysis['host_activity_snapshots'] and all(a['monotonic_ns'] < b['monotonic_ns'] for a,b in zip(activity,activity[1:]))
    warm_seconds = {r['policy']:int(r['elapsed_ns'])/1e9 for r in warmups}
    remainder = result['elapsed_seconds']-sum(sum(v) for v in values.values())-sum(warm_seconds.values())
    assert remainder >= 0 and remainder == analysis['untimed_lifecycle_remainder_seconds']
    record = dict(schema='native-fixed-screen-independent-audit-v1',passed=True,
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_affinity=[3],
        reviewer_script_sha256=sha(Path(__file__)),result_sha256=sha(RUN/'result.json'),
        raw_sha256=sha(RUN/'raw.jsonl'),controls_sha256=sha(controls_path),
        input_sha256=result['input_sha256'],executable_sha256=result['executable_sha256'],
        source_files_rehashed=len(build['source_hashes']),measured_streams=21,warmup_streams=3,
        measured_outputs_checked=21*1392,warmup_outputs_checked=3*1392,
        exact_ascii_fields_checked=24*1392*8,comparisons=comparisons,
        predeclared_full_over_denominators=incremental,provisional_choice=choice,
        all_samples_retained=True,source_input_controls_and_libraries_unchanged=True,
        actual_worker_affinity=[190],controller_affinity=[2],preflight_intervals=3,
        host_activity_snapshots=len(activity),untimed_lifecycle_remainder_seconds=remainder,
        numerical_or_compiler_executions_by_audit=0,all_audit_children_exited=True,
        limit='Private MpState::evaluate request scope; no full MadLoop, original Ninja, or final 5-percent acceptance claim.')
    with (HERE/'screening-independent-audit.json').open('x') as stream:
        json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
