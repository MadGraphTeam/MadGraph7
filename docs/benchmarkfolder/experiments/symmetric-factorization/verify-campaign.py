#!/usr/bin/env python3
"""Offline integrity and outcome verification for the completed factor campaign."""
import sys
sys.dont_write_bytecode = True

import hashlib
import json
from pathlib import Path
import random
import statistics

HERE=Path(__file__).resolve().parent


def main():
    seal=json.loads((HERE/"campaign-seal.json").read_text())
    for name,digest in seal["files_sha256"].items():
        path=(HERE/name).resolve()
        assert path.is_relative_to(HERE) and path.is_file(),name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    load=lambda name:json.loads((HERE/name).read_text())
    fallback=load("fallback-safety-v2/results.json")
    assert fallback["complete"] and fallback["fallback_comparisons"]==48000
    assert fallback["trap_comparisons"]==40
    guards=load("true-entry-guards-v1/results.json")
    assert guards["complete"] and guards["comparisons"]==152
    streams=0
    fixtures=0
    outputs=0
    for name in seal["files_sha256"]:
        if name.endswith("/exact-oracle.json"):
            result=load(name)
            assert result["complete"] and result["passed"],name
            assert len(result["cases"])==10
            streams+=1
            fixtures+=len(result["cases"])
            outputs+=sum(case["output_coefficients"] for case in result["cases"])
    assert (streams,fixtures,outputs)==(24,240,273280)
    control=load("qcd3-factor-trace-v1/result.json")
    assert control["complete"] and control["comparison"]["passed"]
    assert control["comparison"]["printed_binary64_identical_coefficients"]==800
    assert control["guard_totals"]=={"fast":6528,"fallback":0}
    trial=load("qcd3-factor-quiet-v1/result.json")
    assert trial["complete"] and not trial["acceptance_eligible"]
    for variant in ("original","factor"):
        assert [r["repetition"] for r in trial["measurements"][variant]]==list(range(7))
        assert all(r["timed_status_counts"]=={"216":98,"328":2}
                   for r in trial["measurements"][variant])
    assert len(trial["comparisons"])==7
    for comparison in trial["comparisons"]:
        assert comparison["passed"] and comparison["all_printed_coefficients_identical"]
        assert comparison["accuracy_or_status_changes"]==[]
        assert comparison["added_qp_rescues"]==0
    original=[r["seconds_per_point"] for r in trial["measurements"]["original"]]
    factor=[r["seconds_per_point"] for r in trial["measurements"]["factor"]]
    ratio=statistics.median(factor)/statistics.median(original)
    rng=random.Random(410083)
    samples=[]
    for _ in range(10000):
        selected=rng.choices(range(7),k=7)
        samples.append(statistics.median(factor[i] for i in selected)/
                       statistics.median(original[i] for i in selected))
    samples.sort()
    interval=[samples[249],samples[9749]]
    stored=trial["paired_statistics"]
    assert stored["runtime_ratio"]==ratio
    assert stored["ratio_confidence_interval_95"]==interval
    assert stored["status"]=="unresolved" and interval[0]<1<interval[1]
    print(json.dumps(dict(passed=True,files=len(seal["files_sha256"]),
        oracle_streams=streams,fixtures=fixtures,complex_outputs=outputs,
        runtime_ratio=ratio,confidence_interval_95=interval,
        outcome="Inconclusive speed improvement; no production promotion.")))


if __name__=="__main__":
    main()
