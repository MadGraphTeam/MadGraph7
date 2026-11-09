#!/usr/bin/env python3
"""Validate the isolated transport delta and retain its unresolved results."""
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[4]
D = Decimal
SLOTS = ('born', 'finite', 'single', 'double')
PI = D('3.141592653589793238462643383279502884197169399375105820974944592307816406286')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def values(path):
    result = {}
    for line in path.read_text().splitlines():
        row = line.split()
        k, so, code = map(int, row[:3])
        assert code == 120 and D(row[7]) == -1
        vals = dict(zip(SLOTS, map(D, row[3:7])))
        assert all(v.is_finite() for v in vals.values())
        if so == 0:
            result[k] = vals
        else:
            assert so == 1 and result[k] == vals
    assert set(result) == {1, 2}
    return result


def relative(a, b):
    scale = max(abs(a), abs(b))
    return D(0) if scale == 0 else abs(a-b)/scale


def analyze():
    spec = importlib.util.spec_from_file_location('ct_transport_runner', REPORT/'run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    prepared = json.loads((REPORT/'prepared.json').read_text())
    compiled = json.loads((REPORT/'compilation.json').read_text())
    executed = json.loads((REPORT/'execution.json').read_text())
    historical_path = runner.DIAGNOSTIC/'diagnostic-compilation.json'
    historical = json.loads(historical_path.read_text())
    assert digest(historical_path) == prepared['historical_compilation_sha256']
    assert compiled['original_inputs_unchanged'] and executed['original_inputs_unchanged']
    assert executed['complete'] and not executed['stability_certified']
    assert not executed['performance_eligible'] and not executed['uses_symbolica_api']
    assert executed['cpu'] == 3
    assert executed['policy'] == prepared['policy'] == {
        'backend':1,'init_mode':1,'run_mode':4,'scalar_cutoff':1e-6,
        'stability_checks':False,'QP_numerator':True}
    runner.verify(prepared)
    assert runner.inventory(runner.WORK) == compiled['objects_archives_after']
    before, after = prepared['objects_archives_before'], compiled['objects_archives_after']
    assert [p for p in before if before[p] != after.get(p)] == ['SubProcesses/P0_epem_ttxa/CT_interface.o']
    assert sorted(after.keys()-before.keys()) == ['SubProcesses/P0_epem_ttxa/cts_cutroutines_qpinput.o']
    assert prepared['external_archives_before'] == compiled['external_archives_after'] == executed['external_archives_after']
    for filename, expected in (('output.log',executed['output_sha256']),
                               ('values.dat',executed['values_sha256']),
                               ('build.log',compiled['build_log_sha256']),
                               ('ldd.txt',executed['ldd_sha256']),
                               ('run.py',prepared['run_script_sha256']),
                               ('selected.dat',prepared['point_sha256'])):
        assert digest(REPORT/filename) == expected
    assert digest(runner.WORK/runner.PROCESS/'check') == compiled['executable_sha256']
    for name, expected in prepared['card_sha256'].items():
        assert digest(REPORT/name) == expected == digest(runner.DIAGNOSTIC/name)
    original = (REPORT/'sources/CT_interface.f.original').read_text()
    patched = (REPORT/'sources/CT_interface.f').read_text()
    assert runner.patch_interface(original) == patched
    dp = original.split('      SUBROUTINE ML5_0_INITCT()\n',1)[0]
    assert patched.startswith(dp)
    assert runner.patch_cuttools((REPORT/'sources/cts_cutroutines_qpinput.f90.original').read_text()) == (REPORT/'sources/cts_cutroutines_qpinput.f90').read_text()
    assert digest(REPORT/'sources/CT_interface.f.original') == historical['sources']['SubProcesses/P0_epem_ttxa/CT_interface.f']
    assert digest(REPORT/'sources/cts_cutroutines_qpinput.f90.original') == prepared['vendor_source_sha256']
    assert 'CALL ML5_0_FORCE_STABILITY_CHECK(.FALSE.)' in (runner.ORIGIN/runner.PROCESS/'check_sa.f').read_text()
    card = (REPORT/'MadLoopParams.dat').read_text()
    for name, expected in {'MLReductionLib':'1','CTModeInit':'1','CTModeRun':'4',
                           'UseQPIntegrandForCutTools':'.true.','CTLoopLibrary':'2'}.items():
        assert re.findall(r'(?mi)^#'+name+r'\s*\n([^\n]+)',card) == [expected]
    log = (REPORT/'output.log').read_text()
    for name, expected in {'MLReductionLib':'1','CTModeInit':'1','CTModeRun':'4',
                           'UseQPIntegrandForCutTools':'T'}.items():
        assert re.findall(r'> '+name+r'\s*=\s*(\S+)',log) == [expected]
    new = values(REPORT/'values.dat')
    old = {backend:values(runner.OLD/f'diagnostics/ew_n3_epem_ttxa/candidate-{backend}-qp-init-dp/raw/values.dat')
           for backend in (1,6,8,9)}
    geometry_path = REPORT.parent/'geometry/summary.json'
    geometry = json.loads(geometry_path.read_text())
    gal = D(geometry['scalar_inputs']['DP_GAL_PROMOTED'][0])
    alpha = gal**2/(4*PI)
    results=[]
    for k in (1,2):
        assert new[k]['born'] == old[1][k]['born']
        prediction = -new[k]['born']*alpha/PI
        results.append({'point':k,'private_QP_denominator_values':new[k],
            'historical_values':{str(b):v[k] for b,v in old.items()},
            'relative_errors_against_historical':{str(b):{s:relative(new[k][s],v[k][s]) for s in SLOTS}
                                                  for b,v in old.items()},
            'independent_double_pole_from_reported_DP_born':prediction,
            'double_pole_relative_residual':relative(new[k]['double'],prediction)})
    warning_ids=list(map(int,re.findall(r'##W03 WARNING Contribution\s+(\d+)\s+is unstable',log)))
    return {'scope':'Unchecked fixed-QP denominator transport diagnostic. No numerical/performance acceptance.',
        'metric':'abs(a-b)/max(abs(a),abs(b)); zero-zero=0. Distinct from the campaign abs(a-b)/abs(a+b) metric.',
        'conclusion':'At original scalar cutoff1e-6, QP denominator transport alone does not resolve the EW threshold discrepancy; retain the changed finite/single-pole failure.',
        'results':results,'whole_worker_unstable_warning_ids':warning_ids,
        'warning_scope':'Includes initialization/warmup/measured work; not attributed to a specific measured call.',
        'independent_pole_limit':'Uses returned DP Born and DP GAL coupling; not an independent QP Born certificate.',
        'original_DP_subroutine_bytes_preserved':len(dp.encode()),
        'generated_dispatch_sites':7,'validation':'PASS: frozen inputs, isolated source/object delta, original cutoff/policy.',
        'geometry_summary_sha256':digest(geometry_path),
        'evidence_sha256':{str(p.relative_to(REPORT)):digest(p) for p in sorted(REPORT.rglob('*'))
                           if p.is_file() and p.name!='summary.json'}}


if __name__ == '__main__':
    with localcontext() as context:
        context.prec=80
        result=analyze()
    (REPORT/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+'\n')
    print(result['validation'])
    for point in result['results']:
        print('point',point['point'],json.dumps(point['private_QP_denominator_values'],default=str))
