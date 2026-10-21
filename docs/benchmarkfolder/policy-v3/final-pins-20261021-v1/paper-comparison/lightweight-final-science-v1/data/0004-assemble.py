"""Assemble/revalidate retained science records; never launch an executable."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"madgraph/__init__.py").is_file())
sys.path[:0]=[str(ROOT/"tools"),str(ROOT)]
import ninjars_dp_stability as dp

def bound(path):
    path=Path(path).resolve()
    return {"path":str(path.relative_to(ROOT)),"sha256":dp.digest(path),"bytes":path.stat().st_size}

def read(record):
    path=dp._bound(ROOT,record,None)
    return json.loads(path.read_text())

def discrepancy(summary, runs):
    """Failure-visible native-DP diagnostic, separate from the rescued gate."""
    output={"schema":"native-dp-cross-backend-discrepancy-v1","requested_tools":[6,8],
        "stream":"untransformed_native_dp_random_pilot","correctness_gate_replacement":False,
        "tolerance":"abs(a-b) <= 1e-7*max(abs(a),abs(b)) + 1e-10*S",
        "scale":"S=max absolute Born/Laurent coefficient across both tools and all split orders at this point",
        "cases":{}}
    by_case={}
    for run in runs:by_case.setdefault(run["case"],{})[run["tool"]]=run
    for cid,tools in by_case.items():
        if set(tools)!={6,8}:
            output["cases"][cid]={"status":"pending_matched_tools","requested_tools":[6,8]};continue
        a,b=tools[6],tools[8]
        if a["N"]!=b["N"] or a["artifacts"]["points"]["sha256"]!=b["artifacts"]["points"]["sha256"] or a["helicity"]!=b["helicity"]:
            raise ValueError("Cross-backend diagnostic requires identical point/selector streams")
        parsed={tool:dp.parse_raw(dp._bound(ROOT,run["raw"],None),run["N"],tool,run["nso"]) for tool,run in tools.items()}
        physically_qualified=all(summary["cases"][cid]["tools"][str(tool)]["physical_qualified"] for tool in (6,8))
        rows=[]
        for point in range(1,a["N"]+1):
            observations={tool:parsed[tool]["observations"].get((point,1)) for tool in (6,8)}
            failures={str(tool):dp.observation_failure(obs,tool) for tool,obs in observations.items()}
            all_values=[value for obs in observations.values() if obs for row in obs["rows"].values() for value in row["values"][:4]]
            expected_values=2*4*(a["nso"]+1)
            scale=max(map(abs,all_values),default=0.) if len(all_values)==expected_values and all(v is not None for v in all_values) else None
            coefficients=[]
            for order in range(a["nso"]+1):
                for index,label in enumerate(("born","finite","single_pole","double_pole")):
                    values={str(tool):(obs["rows"][order]["values"][index] if obs and order in obs["rows"] else None) for tool,obs in observations.items()}
                    finite=all(value is not None for value in values.values())
                    raw_difference=abs(values["6"]-values["8"]) if finite else None
                    overflow=finite and not math.isfinite(raw_difference)
                    difference=raw_difference if finite and not overflow else None
                    pair_scale=max(map(abs,values.values())) if finite else None
                    tolerance=1e-7*pair_scale+1e-10*scale if finite and scale is not None else None
                    valid=physically_qualified and not any(failures.values()) and tolerance is not None
                    coefficients.append({"order":order,"coefficient":label,"values":values,
                        "absolute_difference":difference,"absolute_difference_overflow":overflow,"pair_scale":pair_scale,
                        "relative_difference":(difference/pair_scale if difference is not None else abs(values["6"]/pair_scale-values["8"]/pair_scale)) if finite and pair_scale else (0. if finite else None),
                        "point_scale_normalized_difference":(difference/scale if difference is not None else abs(values["6"]/scale-values["8"]/scale)) if finite and scale else (0. if finite and scale==0 else None),
                        "tolerance":tolerance,"passes_tolerance":raw_difference<=tolerance if valid else None})
            rows.append({"point":point,"failure_categories":failures,"scale":scale,
                "status_codes":{str(tool):obs["metadata"][1] if obs and obs["metadata"] else None for tool,obs in observations.items()},
                "physical_qualified":physically_qualified,
                "coefficients":coefficients})
        output["cases"][cid]={"N":a["N"],"helicity":a["helicity"],"points":a["artifacts"]["points"],
            "initialization":{str(tool):summary["cases"][cid]["tools"][str(tool)].get("initialization") for tool in (6,8)},
            "point_failures":sum(any(row["failure_categories"].values()) for row in rows),
            "finite_coefficient_tolerance_failures":sum(c["passes_tolerance"] is False for row in rows for c in row["coefficients"]),
            "points_data":rows}
    return output

def assemble(proofs,destination):
    if destination.exists():raise ValueError("Preserve previous assembly output")
    destination.mkdir(parents=True)
    runs=[];bindings=[bound(Path(__file__)),bound(ROOT/"tools/ninjars_dp_stability.py")];cohort=None;cohort_record=None
    for proof_path in proofs:
        proof_binding=bound(proof_path);proof=read(proof_binding);start=read(proof["start"])
        science=read(proof["science_result"]);tool=science["tool"];cid=start["case"]
        if tool not in (6,8):raise ValueError("Only the declared Ninja6/NinjaRS8 study is assembled")
        manifest=read(start["random_manifest"]);prepared=read(start["prepared"])
        if cohort is None:cohort,cohort_record=manifest["release"],start["cohort"]
        if cohort!=manifest["release"] or cohort_record!=start["cohort"]:
            raise ValueError("Science proofs use different explicit cohorts")
        definitions=manifest["cases"][cid];origin=ROOT/prepared["origin"]
        source=next(origin.glob("SubProcesses/P*_*"));output=ROOT/science["output"]
        directory=output/"SubProcesses"/source.name;normal=destination/(cid+"-"+str(tool));normal.mkdir()
        execution={"requested_tool":tool,"helicity":prepared["helicity"]["argument"],
            "exit_code":science["execution"]["exit_code"],"cpu":12,"thread_environment":start["thread_environment"],
            "source_record":proof["science_result"]}
        build={"driver_sha256":science["driver_sha256"],"instrumented_interface_sha256":science["interface_sha256"],
            "executable_sha256":science["executable_sha256"],"source_record":proof["science_result"]}
        dp.dump(normal/"execution.json",execution);dp.dump(normal/"build.json",build)
        artifacts={name:bound(path) for name,path in {"original_interface":source/"CT_interface.f",
            "instrumented_interface":directory/"CT_interface.f","driver":directory/"check_sa.f",
            "executable":directory/"check","parameter_card":output/"Cards/param_card.dat",
            "madloop_card":output/"Cards/MadLoopParams.dat","execution":normal/"execution.json","build":normal/"build.json"}.items()}
        artifacts.update(points=science["points"],stdout=science["execution"]["stdout"],stderr=science["execution"]["stderr"])
        process=definitions["definition"]["process"]
        group="stress" if cid.startswith("stress-") else "loop-induced" if "sqrvirt=" in process else "QCD" if "virt=QCD" in process else "EW"
        run={"case":cid,"process":process,"group":group,"N":science["N"],"nexternal":prepared["nexternal"],
            "nso":prepared["dimensions"][1],"helicity":prepared["helicity"]["argument"],"tool":tool,
            "raw":science["raw"],"artifacts":artifacts,"qualification":{key:proof[key] for key in ("start","plain_result","observed_result","parity_receipt","science_result")},
            "resource_protocol":"dp-resource-custody-v1","measurement_authority":{"kind":"random-dp-v1","manifest":start["random_manifest"],"case":cid}}
        if definitions["origin_authority"] is not None:run["origin_authority"]=definitions["origin_authority"]
        runs.append(run);bindings.append(proof_binding)
    if not runs:raise ValueError("No actual science proofs supplied")
    authority={"schema":dp.RAW_SCHEMA,"stream":"forced_dp_self_consistency","rescue_enabled":False,
        "instrumentation":"per-dispatch-observer-v1","requested_tools":[6,8],"cohort":cohort,"cohort_record":cohort_record,
        "bindings":bindings,"thresholds":[10.**(-16+i/5) for i in range(81)],"runs":runs}
    dp.dump(destination/"manifest.json",authority)
    summary=dp.report_snapshot(destination/"manifest.json",expected={"cohort":cohort})
    dp.dump(destination/"summary.json",summary)
    dp.dump(destination/"cross-backend-discrepancy.json",discrepancy(summary,runs))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof",action="append",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args();assemble([path.resolve() for path in args.proof],args.destination.resolve())
