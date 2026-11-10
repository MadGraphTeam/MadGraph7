#!/usr/bin/env python3
"""Offline counter proof; preserves the executed native protocol unchanged."""
import hashlib
import importlib.util
import json
from pathlib import Path

REPORT=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    sources=REPORT/"protocol-sources"
    freeze=json.loads((sources/"freeze.json").read_text())
    for name,expected in freeze["files"].items():
        assert digest(sources/name)==expected,("frozen protocol changed",name)
    module_spec=importlib.util.spec_from_file_location("frozen_success_parity",sources/"validate.py")
    parser=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(parser)
    specification=json.loads((sources/"specification.json").read_text())
    execution=json.loads((REPORT/"execution.json").read_text())
    actions=[a for a in specification["actions"] if a["allow_refusal"]]
    assert len(actions)==2
    proof={}
    for label in ("old","new"):
        path=REPORT/label/"frames.txt"
        assert digest(path)==execution["workers"][label]["frames_sha256"]
        frames=parser.parse(path.read_text(),specification)
        qualified=parser.qualify(path.read_text(),specification)
        assert qualified==json.loads((REPORT/label/"qualification.json").read_text())
        assert qualified["passed"] and qualified["calls"]==65
        assert all(frame["status"]==0 for frame in frames.values())
        rows=[]
        for action in actions:
            assert (action["precision"],action["cache"],action["control"])==(64,0,1)
            frame=frames[action["id"]]
            assert not any(frame["B"]+frame["A"])
            rows.append(dict(id=action["id"],name=action["name"],native_status=frame["status"],
                aggregate_and_qp_before=frame["B"],aggregate_and_qp_after=frame["A"],
                all_15_counters_zero_before_and_after=True))
        proof[label]=dict(calls=65,successful_calls=65,refused_calls=0,rows=rows,
            frames_sha256=digest(path),retained_qualification_reproduced=True)
    assert (REPORT/"old/frames.txt").read_bytes()==(REPORT/"new/frames.txt").read_bytes()
    return dict(state="complete",passed=True,analysis="Offline analysis only; no native calls or protocol changes",
        analysis_source_sha256=digest(Path(__file__)),frozen_parser_sha256=digest(sources/"validate.py"),
        per_installation=proof,allowed_refusal_branch_exercised=False,
        interpretation="The raw public Fortran API succeeded on both massive-triangle DP calls. Historical generated-adapter refusals are not results of this run.")


if __name__=="__main__":
    result=verify()
    (REPORT/"post-analysis.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("Offline counter proof passed: both massive-triangle DP rows succeeded with all 15 counters zero in each installation.")
