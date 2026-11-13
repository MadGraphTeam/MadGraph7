"""Pure consistency check of the one retained FULL endpoint; no evaluations."""
from pathlib import Path
import hashlib,json,math,re
def compare_rows(reference, candidate, tolerance=3e-6):
    """LoopMEComparator criterion, mapped to native ANS(0:3) column order.

ANS columns are Born, finite, simple pole and double pole. The comparator uses
abs(a-b)/abs(a+b), and permits poles small compared with the finite coefficient.
Opposite nonzero inputs have an undefined relative ratio and fail unless the
pole-size exception applies. Retain undefined ratios as null, never JSON inf.
    """
    if [(r["point"], r["order"]) for r in reference] != [(r["point"], r["order"]) for r in candidate]:
        raise RuntimeError("Numerical rows do not describe identical points/orders")
    failures, residuals, worst = [], [], 0.0
    target_passed = True
    for left, right in zip(reference, candidate):
        finite_scale = max(abs(left["coefficients"][1]), abs(right["coefficients"][1]), 1e-99)
        for column, (a, b) in enumerate(zip(left["coefficients"], right["coefficients"])):
            magnitude = max(abs(a), abs(b))
            symmetric = abs(a - b) / abs(a + b) if a + b else 0.0 if not magnitude else None
            raw_relative = abs(a - b) / magnitude if magnitude else 0.0
            pole_fraction = magnitude / finite_scale if column >= 2 else None
            def accepted(threshold):
                return ((symmetric is not None and symmetric < threshold)
                        or (pole_fraction is not None and pole_fraction < threshold))
            target_passed = target_passed and accepted(1e-8)
            residual = {"point": left["point"], "order": left["order"], "column": column,
                        "reference": a, "candidate": b, "relative_error": raw_relative,
                        "symmetric_relative_error": symmetric, "pole_to_finite": pole_fraction,
                        "passed": accepted(tolerance)}
            residuals.append(residual)
            worst = max(worst, raw_relative)
            if not residual["passed"]:
                failures.append(residual)
    return {"passed": not failures, "maximum_relative_error": worst,
            "criterion": "LoopMEComparator: abs(a-b)/abs(a+b)<tolerance, or pole magnitude/finite maximum<tolerance",
            "relative_error_metric": "abs(a-b)/max(abs(a),abs(b)); zero-zero is zero",
            "ordinary_target_tolerance": 1e-8, "ordinary_target_passed": target_passed,
            "tolerance": tolerance, "failures": failures, "residuals": residuals}
CRITERION_SHA256='0a9fffd567bfc98ecf2cff6b8dafb3c333096d990291a666508db73ac8b3b47e'


def endpoint_consistency(packet):
    packet=Path(packet);records={};all_ps=[];normalization=[]
    for backend in (1,6,7,8,9):
        meta=json.loads((packet/'runs'/str(backend)/'profile_timing_metadata.json').read_text())
        stage=next(s for s in meta['stages'] if s['name']=='unpolarized_full')
        raw=stage['raw_outputs']['result.dat']['text'];lines=[line.split() for line in raw.splitlines()]
        ps=[line[1:] for line in lines if line and line[0]=='PS'];assert len(ps)==6 and all(len(p)==4 for p in ps)
        tags={line[0]:line[1:] for line in lines if line and line[0] in ('BORN','FIN','1EPS','2EPS','RETCODE','EXP','ASO2PI','Export_Format')}
        coefficients=[float(tags[k][0].replace('D','E')) for k in ('BORN','FIN','1EPS','2EPS')];assert all(map(math.isfinite,coefficients))
        all_ps.append(ps);norm={k:tags[k] for k in ('EXP','ASO2PI','Export_Format')};normalization.append(norm)
        records[str(backend)]={'PS_words':ps,'coefficients':dict(zip(('BORN','FIN','1EPS','2EPS'),coefficients)),'normalization_words':norm,'retained_returncode_words':tags.get('RETCODE'),'raw_snapshot_sha256':stage['raw_outputs']['result.dat']['sha256'],'raw_output_freshness_verified':stage.get('raw_output_freshness_verified',False)}
    same=all(p==all_ps[0] for p in all_ps) and all(n==normalization[0] for n in normalization)
    comparisons={}
    if same:
        for i,a in enumerate((1,6,7,8,9)):
            for b in (1,6,7,8,9)[i+1:]:
                def row(k):return [{'point':1,'order':0,'coefficients':list(records[str(k)]['coefficients'].values())}]
                comparisons[str(a)+':'+str(b)]=compare_rows(row(a),row(b),3e-6)
    return {'schema':'zggg-stock-profile-retained-endpoint-consistency-v1','scope':'One retained end-of-batch FULL snapshot per configured backend; not all50 timed points and not per-point dispatch/accuracy proof. Native output freshness is not independently instrumented.','all_PS_and_normalization_words_identical':same,'records':records,'comparisons':comparisons,'all_3e6_criterion_passed':same and all(c['passed'] for c in comparisons.values()),'all_1e8_ordinary_target_passed':same and all(c['ordinary_target_passed'] for c in comparisons.values()),'maximum_relative_error':max((c['maximum_relative_error'] for c in comparisons.values()),default=None),'criterion_function_sha256':CRITERION_SHA256}
