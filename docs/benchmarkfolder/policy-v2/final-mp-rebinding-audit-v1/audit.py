#!/usr/bin/env python3
"""Read-only independent source/raw-output audit of explicit-QP rebinding."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import os
import sys
import traceback

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
BASE = Path(__file__).resolve().parent
REPORT = BASE.parent/'final-mp-rebinding-v1'
ORDINARY = BASE.parent/'final-variant-controls-v1'
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers
from madgraph.various.banner import MadLoopParam

EXPECTED = '8b454519cb6818ef1fafe37378930c276f7e85394a2333ebc9fdb00e75726cff'
HELPER = '93e41b821058e89c04d01f72070406ede09a55cb37021b5995cb27635897ff9c'
NAMES = ('control-rebinding', 'control-rebinding-fresh')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bound(p):
    p = Path(p)
    return {'path': str(p.relative_to(ROOT)), 'sha256': digest(p), 'bytes': p.stat().st_size}


def write(p, d):
    with p.open('x') as f:
        json.dump(d, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def signature(output, manifest):
    return policy.reducer_signature(dict(backend=8, libraries=policy.library_hashes(output),
        compiled_manifest=manifest,
        runtime_metadata={k: policy.bound_file(p) for k, p in policy.runtime_metadata_files(output).items()}))


def parse_raw(directory, expected_status, expected_accuracy):
    rows = []
    for line in (directory/'values.dat').read_text().splitlines():
        f = line.split()
        if len(f) != 8:
            raise RuntimeError('Raw coefficient row has wrong width')
        row = dict(point=int(f[0]), order=int(f[1]), status=int(f[2]),
                   coefficients=[float(x) for x in f[3:7]], accuracy=float(f[7]))
        if (row['status'] != expected_status or len(row['coefficients']) != 4 or
                not all(math.isfinite(x) for x in row['coefficients']+[row['accuracy']])):
            raise RuntimeError('Raw coefficient/status invalid')
        if expected_accuracy is not None and row['accuracy'] != expected_accuracy:
            raise RuntimeError('Explicit-QP accuracy is not the unavailable sentinel')
        if expected_accuracy is None and row['accuracy'] < 0:
            raise RuntimeError('Ordinary DP estimate missing')
        rows.append(row)
    if [(r['point'],r['order']) for r in rows] != [(p,o) for p in range(1,11) for o in (0,1)]:
        raise RuntimeError('Raw physical point/order rows missing or reordered')
    if list(map(int,(directory/'dimensions.dat').read_text().split())) != [1,1,0]:
        raise RuntimeError('HH physical dimensions changed')
    if [list(map(int,line.split())) for line in (directory/'timed_status.dat').read_text().splitlines()] != [[expected_status,10]]:
        raise RuntimeError('Actual evaluation count/status changed')
    return rows


def main():
    if (BASE/'receipt.json').exists():
        raise RuntimeError('Preserve previous audit')
    prepared = policy.verified(policy.read(REPORT/'prepared.json'), 'final-direct-qp-rebinding-v1')
    original = policy.verified(policy.read(policy.check_file(prepared['ordinary_preparation'])), 'final-variant-controls-preparation-v1')
    if digest(REPORT/'execute.py') != prepared['script_sha256']:
        raise RuntimeError('Actual QP controller differs from prepared source')
    if digest(REPORT/'controls_driver.py') != HELPER or digest(ORDINARY/'controls_driver.py') != HELPER:
        raise RuntimeError('Corrected DP/MP helper source changed')
    for phase in ('prepare','run','compare-rotation'):
        if policy.read(ORDINARY/'ledgers'/phase/'result.json')['status'] != 'passed':
            raise RuntimeError('DP parent campaign is not passed')
    config = policy.read(REPORT/'config.json')
    if config['relative_tolerance'] != 1e-7 or config['scale_tolerance'] != 1e-10:
        raise RuntimeError('Physical tolerance changed')
    if prepared['forced_mode'] != dict(CTModeRun=4, UseQPIntegrandForNinjaRS=True, expected_status=120,
            expected_accuracy=-1, stability_measurement=False, automatic_rescue=False):
        raise RuntimeError('Explicit public-QP semantics changed')
    inputs, validation, qp, dp = {}, {}, {}, {}
    for p in (REPORT/'prepared.json',ORDINARY/'prepared.json',REPORT/'config.json',REPORT/'execute.py',
              REPORT/'controls_driver.py',ORDINARY/'controls_driver.py'):
        inputs[str(p.relative_to(ROOT))] = bound(p)
    for name in NAMES:
        p = prepared['variants'][name]
        o = original['variants'][name]
        output = policy.path(p['output'])
        parent = ROOT/'.benchmark-work/policy-v2/final-variant-controls-v1/outputs/candidate'/name
        if bench.generated_source_manifest(output) != p['sources'] or bench.generated_source_manifest(parent) != o['sources']:
            raise RuntimeError('Actual generated source tree changed')
        if (signature(output,p['sources']) != p['runtime'] or
                signature(parent,o['sources']) != o['reducer_signatures']['8'] or
                p['runtime'] != o['reducer_signatures']['8'] or p['runtime']['identity_sha256'] != EXPECTED):
            raise RuntimeError('Actual QP/DP full runtime binding differs')
        if policy.library_hashes(output) != p['libraries'] or p['libraries'] != o['libraries']:
            raise RuntimeError('QP library set differs from passed DP parent')
        ordinary_source = REPORT/'drivers'/name/'ordinary.f'
        forced_source = REPORT/'drivers'/name/'forced-qp.f'
        if (ordinary_source.read_bytes() != policy.check_file(o['driver']).read_bytes() or
                digest(ordinary_source) != p['ordinary_source_sha256'] or digest(forced_source) != p['forced_source_sha256']):
            raise RuntimeError('Driver source copies differ from bound sources')
        text, modified = ordinary_source.read_text(), forced_source.read_text()
        old,new = 'CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)','CALL ML5_0_FORCE_STABILITY_CHECK(.FALSE.)'
        if text.count(old) != 1 or modified != text.replace(old,new):
            raise RuntimeError('QP clone has more than the approved one-line driver change')
        changed = policy.delta(o['sources']['files'],p['sources']['files'])
        if len(changed) != 1 or not next(iter(changed)).endswith('/check_sa.f'):
            raise RuntimeError('QP numerical source changes beyond approved test driver')
        driver_path = next(output.glob('SubProcesses/P*_*/check_sa.f'))
        if driver_path.read_bytes() != forced_source.read_bytes():
            raise RuntimeError('Actual compiled driver source differs')
        executable = next(output.glob('SubProcesses/P*_*/check'))
        dp_executable = next(parent.glob('SubProcesses/P*_*/check'))
        if digest(executable) != p['executable_sha256'] or digest(dp_executable) != o['executable_sha256']:
            raise RuntimeError('Actual diagnostic executable changed')
        for fragment in ('MP__MDL_MT=REAL(MDL_MT,KIND=16)', 'MP__MU_R=CMPLX(REAL(MU_R,KIND=16),0.0_16,KIND=16)', 'CALL COUP()'):
            if fragment not in modified:
                raise RuntimeError('Independent MP model-state update missing')
        finalize = 'CALL NINJARS_FINALIZE()'
        if (finalize in modified) != name.endswith('-fresh'):
            raise RuntimeError('Context-finalization variant differs')
        if modified.count('CALL CONTROL_STATE(') != 4:
            raise RuntimeError('State updater does not cover every driver phase')
        point = policy.check_file(p['points'])
        if point.read_bytes() != policy.check_file(o['points']).read_bytes() or point.read_bytes() != (ROOT/'docs/benchmarkfolder/points'/(name+'.dat')).read_bytes():
            raise RuntimeError('Fixed state-rebinding input stream changed')
        if (REPORT/'points'/(name+'.json')).read_bytes() != (ORDINARY/'points'/(name+'.json')).read_bytes():
            raise RuntimeError('Mass/scale schedule metadata changed')
        physical = output/'Cards/param_card.dat'
        if physical.read_bytes() != (parent/'Cards/param_card.dat').read_bytes() or digest(physical) != p['parameter_card_sha256']:
            raise RuntimeError('Fixed physical card differs')
        qp_card = MadLoopParam(str(REPORT/'inputs/candidate'/name/'MadLoopParams-8.dat'))
        dp_card = MadLoopParam(str(ORDINARY/'inputs/candidate'/name/'MadLoopParams-8.dat'))
        if qp_card['CTModeRun'] != 4 or not qp_card['UseQPIntegrandForNinjaRS'] or str(qp_card['MLReductionLib']) != '8':
            raise RuntimeError('Actual explicit-QP backend card differs')
        differences = {str(k).lower():[dp_card[k],v] for k,v in qp_card.items() if v != dp_card[k]}
        if differences != {'ctmoderun':[-1,4]}:
            raise RuntimeError('Other numerical settings differ from passed DP')
        qp_path = REPORT/'data'/('candidate-'+name+'-8-0.json')
        qp[name] = policy.read(qp_path)
        dp_path = policy.check_file(p['ordinary_record'])
        dp[name] = policy.read(dp_path)
        qp_rows = parse_raw(REPORT/'raw'/name,120,-1)
        dp_rows = parse_raw(ORDINARY/'raw'/name/'8',218,None)
        if qp_rows != qp[name]['rows'] or dp_rows != dp[name]['rows']:
            raise RuntimeError('Raw streams disagree with recorded numerical rows')
        for d in (qp[name],dp[name]):
            if d['points_sha256'] != digest(point) or d['parameter_card_sha256'] != digest(physical):
                raise RuntimeError('Recorded numerical physical inputs differ')
        if qp[name]['executable_sha256'] != p['executable_sha256'] or dp[name]['executable_sha256'] != o['executable_sha256']:
            raise RuntimeError('Raw result executable binding differs')
        if qp[name]['runtime_identity'] != EXPECTED or dp[name].get('measurement_validation_error'):
            raise RuntimeError('Actual result runtime or DP validation failed')
        if (qp[name]['shared_numerical_settings']['ctmoderun'] != 4 or
                not qp[name]['shared_numerical_settings']['selected_backend_qp_integrand'] or
                not qp[name]['direct_forced_qp'] or qp[name]['automatic_rescue'] or
                qp[name]['stability_measurement'] or qp[name]['performance_acceptance_claim']):
            raise RuntimeError('Explicit QP mislabeled as normal stability/performance result')
        for source in (point,qp_path,dp_path,ordinary_source,forced_source,REPORT/'inputs/candidate'/name/'MadLoopParams-8.dat'):
            inputs[str(source.relative_to(ROOT))] = bound(source)
        for prefix in (REPORT/'raw'/name,ORDINARY/'raw'/name/'8'):
            for filename in ('values.dat','timed_status.dat','dimensions.dat'):
                source=prefix/filename;inputs[str(source.relative_to(ROOT))]=bound(source)
        validation[name] = dict(runtime_identity_sha256=EXPECTED, approved_driver_changed_paths=sorted(changed),
            qp_status=120, qp_accuracy=-1, qp_points=10, qp_rows=20, dp_status=218, physical_card_sha256=digest(physical),
            points_sha256=digest(point), qp_executable_sha256=digest(executable), dp_executable_sha256=digest(dp_executable),
            only_card_difference=differences, resets_context_before_each_evaluation=name.endswith('-fresh'))
    comparisons={}
    groups=[('retained-qp-vs-dp',{'QP':qp[NAMES[0]]['rows'],'DP':dp[NAMES[0]]['rows']},'direct-qp-vs-dp-'+NAMES[0]),
            ('finalized-qp-vs-dp',{'QP':qp[NAMES[1]]['rows'],'DP':dp[NAMES[1]]['rows']},'direct-qp-vs-dp-'+NAMES[1]),
            ('retained-vs-finalized-qp',{'retained':qp[NAMES[0]]['rows'],'finalized':qp[NAMES[1]]['rows']},'direct-qp-retained-vs-finalized')]
    for label,rows,original_label in groups:
        agreement=policy.compare_rows(rows)
        scales={}
        for values in rows.values():
            for r in values:scales[r['point']]=max(scales.get(r['point'],0.0),*map(abs,r['coefficients']))
        a,b=rows.values();maximum=dict(fraction_of_tolerance=-1.0);exact=True;count=0
        for x,y in zip(a,b):
            for index,(left,right) in enumerate(zip(x['coefficients'],y['coefficients'])):
                delta=abs(left-right);tol=1e-7*max(abs(left),abs(right))+1e-10*scales[x['point']]
                fraction=delta/tol if tol else 0.0
                if fraction>maximum['fraction_of_tolerance']:
                    maximum=dict(fraction_of_tolerance=fraction,point=x['point'],order=x['order'],coefficient=index,
                                 values=[left,right],difference=delta,tolerance=tol,point_scale=scales[x['point']])
                count+=1;exact &= left==right
        recorded=policy.read(REPORT/'data'/('agreement-'+original_label+'.json'))
        if (recorded['passed']!=agreement['passed'] or recorded['coefficient_comparisons']!=count or
                recorded['maximum_tolerance_fraction']!=maximum['fraction_of_tolerance']):
            raise RuntimeError('Original comparison summary differs from independent recomputation')
        comparisons[label]=dict(agreement=agreement,scalar_comparisons=count,maximum=maximum,coefficient_values_exact=exact)
    receipt=policy.seal(dict(schema='final-mp-rebinding-independent-audit-v1',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),source_only=True,numerical_executions=0,compiler_calls=0,tests_or_worker_reruns=0,
        validation=validation,comparisons=comparisons,total_scalar_comparisons=sum(x['scalar_comparisons'] for x in comparisons.values()),
        passed=all(x['agreement']['passed'] for x in comparisons.values()),bound_inputs=inputs,auditor_source=bound(__file__),
        helper_sha256=HELPER,reader_sources={n:bound(ROOT/'tools'/n) for n in ('ninjars_benchmark.py','ninjars_benchmark_policy_v2.py','ninjars_benchmark_workers.py')},
        automatic_stability_measurement=False,automatic_rescue=False,performance_acceptance_claim=False,
        limitations=['Explicit public CTModeRun4 returns120/accuracy-1; no automated stability or rescue claim.',
                     'One fixed ten-point rebinding stream per context lifecycle; no full campaign/performance acceptance claim.']))
    write(BASE/'receipt.json',receipt)
    print(json.dumps({'passed':receipt['passed'],'identity_sha256':receipt['identity_sha256'],
                      'comparisons':receipt['total_scalar_comparisons'],
                      'maximum_fraction_of_tolerance':max(x['maximum']['fraction_of_tolerance'] for x in comparisons.values())},indent=2))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(BASE/'failure.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        raise
