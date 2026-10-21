#!/usr/bin/env python3
"""Review the fixed recovery bindings without compiling or evaluating."""
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
OLD = HERE.parent/'ninjars-mp-tensor-allocation-process-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def bound(record):
    path = ROOT/record['path']
    assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], path
    return path


def normalize(value):
    if isinstance(value, str):
        return value.replace('ninjars-mp-tensor-allocation-process-v2',
                             'ninjars-mp-tensor-allocation-process-v1')
    if isinstance(value, list):
        return [normalize(x) for x in value]
    if isinstance(value, dict):
        return {k:normalize(v) for k,v in value.items()}
    return value


def main():
    freeze = HERE/'frozen-bindings.json'
    assert sha(freeze) == '64766a546d751705993e4b074f830d9ff32260c89e4ecfd6f6e4aa9114e2871e'
    assert sha(HERE/'recovery-result.json') == '0abe8d7266a8a69543a6645e5032d2fa190a4701c86a322080a754886ba987e8'
    frozen = load(freeze)
    old = load(OLD/'frozen-bindings.json')
    assert sha(OLD/'frozen-bindings.json') == '26b7602867394ef5f9b28a0900d8a518afd1e8e7668075e4af52f36af6a6d4c3'
    for key in ('repetitions','loops','points','bootstrap_resamples','bootstrap_seed',
                'thread_environment','bridge_sources','protocol','package','rust_build'):
        assert frozen[key] == old[key], key
    for key in ('controller','analyzer','readiness_verifier','package','rust_build','preparation','plan'):
        bound(frozen[key])
    for record in frozen['support_tools'].values():
        bound(record)
    for name in ('run-processes.py','analyze-processes.py','comparison.py',
                 'comparison-contract.json','bound-support/ninjars_report_v2.py'):
        assert (HERE/name).read_bytes() == (OLD/name).read_bytes(), name
    recovery = load(HERE/'recovery-result.json')
    assert recovery['status'] == 'retired-without-resumption'
    assert not recovery['numerical_execution'] and recovery['old_measurements_unchanged']
    assert recovery['old_retention_verified_files'] == 595
    assert all(v in ('absent','Z') for v in recovery['after_states'].values())
    assert recovery['signals'] == [dict(pid=pid, signal='SIGKILL', via='pidfd', never_SIGCONT=True)
                                   for pid in (2896495,3369596)]
    for key in ('controller','pause','old_retention'):
        bound(recovery[key])
    for record in recovery['before']:
        assert record['state'] == 'T'
        assert (record['pid'],record['start_ticks']) in ((2896495,57322835),(3369596,57422045))
    preparation = load(bound(frozen['preparation']))
    assert preparation['complete'] and not preparation['amplitudes_evaluated']
    assert not preparation['compilation_performed']
    assert preparation['builds'] == frozen['builds']
    for key in ('old_frozen_bindings','old_preparation','recovery','new_plan','controller'):
        bound(preparation[key])
    checked, generated, libraries, metadata = [], 0, 0, 0
    # Rehash only paths already listed in the sealed preparation. Do not
    # discover a new inventory or execute an imported campaign controller.
    for case,arms in frozen['builds'].items():
        for arm,record in arms.items():
            prior = old['builds'][case][arm]
            source = dict(record)
            origin = source.pop('fresh_copy_origin')
            assert normalize(source) == prior, (case,arm)
            assert not origin['recompiled']
            bound(origin['build'])
            output = ROOT/record['output']
            for key in ('executable','driver','points','parameter_card','madloop_card','actual_parameter_card'):
                bound(record[key])
            compiled = load(HERE/'processes-final'/case/arm/'compiled-sources.json')
            assert compiled == load(OLD/'processes-final'/case/arm/'compiled-sources.json')
            for name,expected in compiled['files'].items():
                p = output/name
                assert sha(p) == expected['sha256'] and p.stat().st_size == expected['bytes'],p
                generated += 1
            for name,expected in record['libraries'].items():
                assert sha(output/'lib'/name) == expected
                libraries += 1
            for name,expected in record['runtime_metadata'].items():
                p = output/'lib'/name
                assert sha(p) == expected['sha256'] and p.stat().st_size == expected['bytes']
                metadata += 1
            for name,expected in record['link_inputs'].items():
                assert sha(output/name) == expected
            checked.append({'case':case,'arm':arm,'executable_sha256':record['executable']['sha256']})
    result = {'schema':'allocation-independent-recovery-readiness-v2','passed':True,
        'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reviewer_script_sha256':sha(Path(__file__)), 'frozen_bindings_sha256':sha(freeze),
        'recovery_sha256':sha(HERE/'recovery-result.json'), 'workers':checked,
        'generated_source_paths_rehashed':generated,'library_paths_rehashed':libraries,
        'runtime_metadata_paths_rehashed':metadata,
        'controller_analyzer_and_comparison_byte_identical':True,
        'prepared_build_records_identical_except_namespace_and_copy_provenance':True,
        'interrupted_interval_never_resumed_or_accepted':True,
        'historical_samples_reused':False,'new_inventories_created':False,
        'compilers_or_engines_started':0,'blocking_findings':[],
        'limit':'Readiness and retirement evidence only; no timing or numerical outcome claimed.'}
    with (HERE/'independent-recovery-audit.json').open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
