#!/usr/bin/env python3
"""Pure report derivation for the original closed zggg ten-point timing pilot.

No controller, compiler, native loader, reducer or network is invoked. The
original TIRust timeout is immutable; later repairs need a separate phase.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import types

IDS=(1,6,7,8,9)
MEASURED=(1,6,7,8)
CASE="qcd_n4_uux_zggg"
PROCESS="u u~ > z g g g [virt=QCD]"
TIRUST="10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1"
DSO="85ee0cdf860e5ad4c7a5c26163802673de22404690a10c7c53fb73ebe1ff5bb0"
SUMMARIZER="618e5d74605f6785ccb7348c101e611337db819411ee0de53eddf3d0ce8159c2"
NAMES={1:"CutTools",6:"Ninja",7:"COLLIER",8:"NinjaRS",9:"TIRust"}
HERE=Path(__file__).resolve().parent


def require(value,message):
    if not value:raise ValueError(message)


def encoded(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()


def read(path):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,"Duplicate JSON key");out[k]=v
        return out
    def invalid(value):raise ValueError("Nonfinite JSON: "+value)
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=invalid)


def identity(path):
    raw=Path(path).read_bytes()
    return {"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}


def inside(root,name):
    require(isinstance(name,str) and name and "\\" not in name,"Invalid relative path")
    part=PurePosixPath(name)
    require(not part.is_absolute() and ".." not in part.parts and str(part)==name,"Unsafe relative path")
    path=Path(root)
    for p in part.parts:
        path/=p;require(not path.is_symlink(),"Symlink evidence")
    return path


def packet(root,choice):
    require(set(choice)=={"path","index_name","index_sha256"},"Unexpected packet selector")
    require(choice["index_name"] in ("index.json","evidence-index.json"),"Unsafe packet index")
    folder=inside(root,choice["path"]);index=folder/choice["index_name"]
    require(identity(index)["sha256"]==choice["index_sha256"],"Timing packet seal changed")
    data=read(index)
    require({p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
            ==set(data["files"])|{choice["index_name"]},"Timing packet inventory changed")
    for name,expected in data["files"].items():
        require(identity(inside(folder,name))==expected,"Timing payload changed: "+name)
    return folder


def row_checks(observation,backend):
    rows=observation["rows"]
    require(len(rows)==20 and {(r["point"],r["order"]) for r in rows}
            =={(p,o) for p in range(1,11) for o in (0,1)},"Incomplete ten-point coefficient traversal")
    for r in rows:
        require(r["actual_backend"]==backend and r["precision"] in ("DP","QP")
                and len(r["coefficients"])==4 and all(math.isfinite(x) for x in r["coefficients"]),
                "Unexpected actual backend/precision or coefficient")
    for point in range(1,11):
        pair=[r for r in rows if r["point"]==point]
        require(all(pair[0][k]==pair[1][k] for k in ("coefficients","status","precision","actual_backend")),
                "Overall and sole split-order results are not duplicate rows")
    # Only overall order0 contributes to output-call histograms. Order1 is the
    # duplicate sole split-order result, not a second physical evaluation.
    steady=[r for r in rows if r["order"]==0]
    counts=collections.Counter(str(r["actual_backend"])+":"+r["precision"] for r in steady)
    statuses=collections.Counter(str(r["status"]) for r in steady)
    require(observation["status_counts"]==dict(statuses),"Final traversal status histogram changed")
    return counts


def validate(summary,directory,controller):
    directory=Path(directory)
    spec=read(directory/"specification.json")
    require(summary["schema"]=="madloop-zggg-five-backend-pilot-summary-v1"
            and summary["process_case"]==CASE and summary["case"]["model"]=="loop_sm"
            and summary["case"]["process"]==PROCESS,"Wrong process evidence")
    require(summary["point_count"]==10 and summary["fixture_point_count"]==100
            and summary["frozen_membership_one_based"]==list(range(1,11)),"Changed pilot membership")
    require(summary["tirust_revision"]==TIRUST and summary["tirust_dso_sha256"]==DSO
            and summary["performance_acceptance"] is False and summary["cpu_reservation"] is False,
            "Wrong source/runtime or acceptance scope")
    require(summary["completed_cycle_receipts"]==list(range(1,11))
            and len(summary["cycle_schedule"])==10,"All ten cycles must close before final reporting")
    require(summary["closure"]["shared_unchanged"] is True and not summary["closure"]["cleanup_failures"]
            and summary["global_failure"] is None,"Unclosed shared-input/worker/global state")
    require(identity(controller)["sha256"]==spec["numerical_controller_sha256"],"Controller binding changed")
    require("timeout=min(600.,remaining())" in Path(controller).read_text(),"Original READY budget differs")
    full=read(directory/"points"/(CASE+"-100.json"))
    pilot=read(directory/"points"/(CASE+"-first10.json"))
    require(len(full["points"])==100 and len(pilot["points"])==10 and pilot["points"]==full["points"][:10]
            and pilot["membership_one_based"]==list(range(1,11))
            and full["sha256"]==summary["full_points_sha256"]
            and pilot["sha256"]==summary["pilot_points_sha256"]
            and pilot["full_fixture_sha256"]==full["sha256"],"Frozen 100/first10 fixtures changed")
    for suffix,value in (("-100",full),("-first10",pilot)):
        require(identity(directory/"points"/(CASE+suffix+".dat"))["sha256"]==value["sha256"],
                "Physical point file changed")
    require({p["sqrt_s"] for p in full["points"]}=={1000.0},"Unexpected physical energy")
    require(set(summary["backends"])=={str(i) for i in IDS},"One of five requested rows was removed")
    policies=[]
    for backend in IDS:
        lane=summary["backends"][str(backend)]
        settings=lane["input_binding"]["settings"]
        require(settings["mlreductionlib"]==str(backend),"Requested backend changed")
        require(settings["looppolemode"]==0 and settings["tirustcomputefinite"] is True
                and settings["colliercomputeirpoles"] is True and settings["colliercomputeuvpoles"] is True,
                "Full finite+UV+IR policy changed")
        policies.append({k:v for k,v in settings.items() if k!="mlreductionlib"})
        libs=lane["input_binding"]["runtime"]["libraries"]
        require([v for k,v in libs.items() if Path(k).name=="libmadloop_tirust.so"]==[DSO],
                "Input runtime DSO differs")
        if backend not in MEASURED:continue
        require(lane["timing_available"] is True and lane["sample_count"]==10 and len(lane["samples"])==10
                and lane["numerical_regression_pass"] is True and lane["numerical_check_count"]==13,
                "Measured lane lacks ten qualified cycles and all learning/calibration checks")
        require(sorted(s["cycle"] for s in lane["samples"])==list(range(1,11)),"Duplicate/missing lane cycle")
        for phase in ("before","after"):
            maps=read(directory/"lanes"/str(backend)/("maps-"+phase+".json"))
            require(set(maps["signed_tirust_dso"].values())=={DSO},"Live READY runtime binding differs")
        row_checks(read(directory/"lanes"/str(backend)/"learning.json"),backend)
        for sample in lane["samples"]:
            obs=read(inside(directory,sample["observation_path"]))
            counts=row_checks(obs,backend)
            require(sample["loops"]==1 and sample["evaluations"]==sample["loops"]*10
                    and all(k.startswith(str(backend)+":") for k in sample["backend_precision_counts"]),
                    "Incorrect timed output-call membership")
            require(sample["native_dp_only"]==all(k.endswith(":DP") for k in counts),"DP label differs from rows")
        for item in lane["numerical_checks"]:
            residuals=item["result"]["residuals"]
            require(len(residuals)==80 and {(r["point"],r["order"],r["column"]) for r in residuals}
                    =={(p,o,c) for p in range(1,11) for o in (0,1) for c in range(4)},
                    "Incomplete physical coefficient comparison")
    require(all(p==policies[0] for p in policies),"Common numerical policies differ")
    failed=summary["backends"]["9"]
    require(failed["status"]=="failed-initialization" and failed["timing_available"] is False
            and failed["initialization"] is None and failed["samples"]==[]
            and failed["aggregate_backend_precision_counts"]=={}
            and all(failed[k] is None for k in ("median_seconds_per_point","mad_seconds_per_point",
                "median_ratio_to_8","paired_ratio_to_8_bootstrap_95")),"Original TIRust timeout was overwritten")
    require(failed["failure"]["error_type"]=="ProtocolError"
            and "timeout" in failed["failure"]["error"].lower(),"Missing original initialization timeout")
    return policies[0]



def diagnostic_replay(directory):
    """Recompute the partial point4 corroboration; never creates timing data."""
    directory=Path(directory);record={};inputs={}
    for backend in MEASURED:
        name="lanes/"+str(backend)+"/learning.json";path=inside(directory,name)
        row=next(r for r in read(path)["rows"] if r["point"]==4 and r["order"]==0)
        record[str(backend)]={"finite":row["coefficients"][1],"status":row["status"],
            "precision":row["precision"],"source_phase":"last complete steady warm-up traversal"}
        inputs[name]=identity(path)["sha256"]
    name="lanes/9/failed-startup-raw/warmup.dat";path=inside(directory,name)
    fields=[line.split() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    steady=[f for f in fields if int(f[0])>0 and int(f[2])==0]
    usage=collections.Counter();statuses=collections.Counter()
    for f in steady:
        code=int(f[3]);require(code in (219,323),"Unrecognized partial TIRust accepted status")
        usage["9:"+("DP" if code==219 else "QP")]+=1;statuses[str(code)]+=1
    partial={"phase":"partial steady warm-up","calls":len(steady),"points_one_based":[int(f[1]) for f in steady],
        "backend_precision_counts":dict(usage),"status_counts":dict(statuses),
        "complete_ten_point_traversal":False,"timing_available":False}
    candidates=[f for f in steady if int(f[1])==4]
    require(candidates and len(steady)<10,"Missing point4 or unexpected complete TIRust warmup")
    row=candidates[-1];code=int(row[3])
    record["9"]={"finite":float(row[5]),"status":code,"precision":"DP" if code==219 else "QP","call":int(row[0]),
        "source_phase":"partial steady warm-up trace; no complete READY/timing qualification inferred"}
    inputs[name]=identity(path)["sha256"]
    values=[v["finite"] for key,v in record.items() if key!="1"];ct=record["1"]["finite"]
    derived={"records":record,"tirust_partial_warmup":partial,"input_sha256":inputs,
        "cluster_backends":sorted(int(k) for k in record if k!="1"),
        "cluster_relative_spread":(max(values)-min(values))/max(map(abs,values)),
        "cuttools_raw_relative_residuals":{key:abs(ct-v["finite"])/max(abs(ct),abs(v["finite"])) for key,v in record.items() if key!="1"}}
    retained=read(directory/"point4-corroboration.json")
    require(retained["schema"]=="madloop-zggg-point4-finite-corroboration-v1" and retained["point"]==4
            and retained["order"]==0 and retained["coefficient"]=="finite"
            and retained["new_native_evaluations"]==0 and retained["performance_acceptance"] is False
            and all(retained[key]==value for key,value in derived.items()),"Partial diagnostic differs from pure replay")
    return retained


def format_data(summary,directory,selector,date):
    """Format only already validated, closed observations; preserve raw details."""
    directory=Path(directory)
    backends={};checks={};init={};all_seconds=[];target_misses={}
    for backend in IDS:
        key=str(backend);lane=dict(summary["backends"][key]);ci=lane["paired_ratio_to_8_bootstrap_95"]
        lane["paired_ratio_ci95_to_8"]=(None if ci is None else {"lower":ci["lower_95"],"upper":ci["upper_95"]})
        backends[key]=lane
        init[key]=None if lane["initialization"] is None else lane["initialization"]["cold_seconds"]
        misses=[{"comparison":check["path"],**r} for check in lane["numerical_checks"]
                for r in check["result"]["residuals"] if r["relative_error"]>=1e-8]
        target_misses[key]=misses
        checks[key]={"regression_pass":lane["numerical_regression_pass"],
            "ordinary_target_pass":lane["ordinary_target_passed"],"retained_checks":lane["numerical_check_count"],
            "ordinary_target_misses":misses,"regression_failures":lane["numerical_failures"],
            "status":"unqualified: initialization timeout" if backend==9 else "qualified regression comparison"}
        all_seconds.extend(s["seconds"] for s in lane["samples"])
    diagnostic=diagnostic_replay(directory)
    miss_points=sorted({r["point"] for key in ("6","7","8") for r in target_misses[key]})
    largest=max((r for key in ("6","7","8") for r in target_misses[key]),key=lambda r:r["relative_error"])
    maximum=largest["relative_error"]
    measured=[backends[str(i)] for i in MEASURED]
    native=all(l["native_dp_only"] is True for l in measured)
    partial=diagnostic["tirust_partial_warmup"]
    precision="All four timed lanes have native-DP accepted outputs." if native else "The table preserves all accepted DP/QP mixtures in the four complete timed lanes."
    precision+=" TIRust’s partial warmup has "+str(partial["calls"])+" calls ("+", ".join(k+"="+str(v) for k,v in sorted(partial["backend_precision_counts"].items()))+"); these are neither timed counts nor a complete ten-point traversal."
    case={"case_id":CASE,"model":"loop_sm","process":PROCESS,"nfinal":4,"status":"closed-with-TIRust-initialization-timeout",
        "summary":{"backends":backends,"points_per_traversal":10,"cycle_count":10},
        "initialization_seconds":init,"numerical_validation":{"backends":checks},
        "cohort":{"kind":"ordinary deterministic first-ten-point pilot","sqrt_s_GeV":1000.0},
        "source_packet":selector,"provenance":summary,"point4_diagnostic":diagnostic}
    host=summary["machine"]
    confidence=all(l["paired_ratio_to_8_bootstrap_95"] is not None for l in measured)
    data={"schema":"madloop-zggg-five-backend-timing-report-data-v1","cases":[case],"date":date,
        "source_label":"MG "+summary["mg_revision"][:10]+" · original signed TIRust",
        "machine_note":summary["compiler"]+"; "+host["cpu_model"]+"; affinity CPU "+str(summary["cpu"])+"; one-thread settings; shared, unreserved host. Exact flags/runtime hashes are retained.",
        "pilot_note":"This pilot uses the unselected first ten of a 100-point fixture frozen before amplitudes; no failed point is removed. The main campaign remains separate.",
        "cycle_note":"Ten frozen Latin/reversed cycles skip unavailable TIRust slots; four live positions are near-balanced. Two calibrations fix one full traversal per timed batch.",
        "clock_note":"Fortran SYSTEM_CLOCK wall time is divided by actual points × loops. The minimum whole-cohort batch exceeds the nominal 1.1 s target; no points are thinned.",
        "interval_note":"95% intervals use 10,000 percentile bootstrap resamples of the median matched-cycle ratio (seed 20261112), resampling ten whole cycles; they do not test baseline acceptance.",
        "initialization_note":"TIRust was cleanly terminated at its original 600 s READY budget; its initialization and warmed timings are unavailable.",
        "accuracy_note":"All four completed lanes pass MadLoop’s 3e-6 regression criterion against same-source CutTools DP. Reducers 6/7/8 miss the separate 1e-8 target in finite terms at points "+", ".join(map(str,miss_points))+"; the maximum raw relative residual is "+format(maximum,".5g")+" at point "+str(largest["point"])+". Overall/sole-order duplicate rows are not extra physical points.",
        "failure_note":"TIRust’s incomplete ten-point lane is unqualified. Its point-4 QP finite value clusters with 6/7/8 (spread "+format(diagnostic["cluster_relative_spread"],".4g")+"); this neither supplies a timing nor establishes absolute truth or fault. Every miss and the original timeout remain retained.",
        "precision_note":precision,
        "policy_note":"Cards use full finite+UV+IR (LoopPoleMode=0), explicit COLLIER poles and normal stability/rescue. Four READY lanes have live DSO-map witnesses. TIRust9 has executable/loader/ldd/shared-file binding, with no startup live-map claim.",
        "evidence_label":"Original closed capsule: "+selector["path"]+"; index SHA-256 "+selector["index_sha256"]+". Reproduce with source binding, pure derivation and CSV.",
        "actual_batch_seconds":{"minimum":min(all_seconds),"maximum":max(all_seconds)},
        "performance_acceptance":False,"confidence_intervals":confidence,"new_native_physics_evaluations":0}
    selection={"schema":"madloop-zggg-timing-report-selection-v1","case_ids":[CASE],"closed":True,
        "production_source_promotion":False,"performance_acceptance":False,"confidence_intervals":confidence,
        "new_native_physics_evaluations":0,"tirust_source_revision":TIRUST,"tirust_dso_sha256":DSO,
        "style_sha256":identity(HERE/"style.typ")["sha256"],"packets":[selector]}
    return data,selection


def derive(root):
    binding=read(HERE/"zggg-source-binding.json")
    require(binding["schema"]=="madloop-zggg-timing-report-source-binding-v1","Wrong source binding")
    selector=binding["packet"];folder=packet(Path(root),selector)
    directory=inside(folder,binding["measurements"])
    path=inside(folder,binding["summarizer"])
    require(identity(path)["sha256"]==SUMMARIZER,"Unreviewed summarizer source")
    module=types.ModuleType("closed_zggg_timing_summarizer");module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),"exec"),module.__dict__)
    summary=module.derive(directory)
    require(summary==read(directory/"summary.json"),"Retained summary differs from pure replay")
    validate(summary,directory,inside(folder,binding["controller"]))
    return format_data(summary,directory,selector,binding["date"])


def csv_bytes(data):
    out=io.StringIO(newline="")
    columns=("case_id","backend_id","backend","status","timing_available","fixture_points","timed_points",
        "median_ms_per_point","unscaled_MAD_ms_per_point","median_paired_ratio_to_NinjaRS8","paired_ratio95_lower",
        "paired_ratio95_upper","READY_seconds","accepted_timed_backend_precision_counts","numerical_3e-6_pass",
        "ordinary_1e-8_target_pass","source_packet_index_sha256")
    writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader();case=data["cases"][0]
    for backend in IDS:
        key=str(backend);lane=case["summary"]["backends"][key];ci=lane["paired_ratio_ci95_to_8"]
        number=lambda value: "" if value is None else format(value,".17g")
        writer.writerow({"case_id":CASE,"backend_id":backend,"backend":NAMES[backend],"status":lane["status"],
            "timing_available":lane["timing_available"],"fixture_points":100,"timed_points":10 if lane["timing_available"] else 0,
            "median_ms_per_point":number(None if lane["median_seconds_per_point"] is None else lane["median_seconds_per_point"]*1000),
            "unscaled_MAD_ms_per_point":number(None if lane["mad_seconds_per_point"] is None else lane["mad_seconds_per_point"]*1000),
            "median_paired_ratio_to_NinjaRS8":number(lane["median_ratio_to_8"]),"paired_ratio95_lower":number(None if ci is None else ci["lower"]),
            "paired_ratio95_upper":number(None if ci is None else ci["upper"]),"READY_seconds":number(case["initialization_seconds"][key]),
            "accepted_timed_backend_precision_counts":json.dumps(lane["aggregate_backend_precision_counts"],sort_keys=True),
            "numerical_3e-6_pass":lane["numerical_regression_pass"],"ordinary_1e-8_target_pass":lane["ordinary_target_passed"],
            "source_packet_index_sha256":case["source_packet"]["index_sha256"]})
    return out.getvalue().encode()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    data,selection=derive(args.root.resolve())
    require(not args.output.exists(),"Fresh output directory required");args.output.mkdir(parents=True)
    for name,value in (("zggg-timing-data.json",data),("zggg-timing-selection.json",selection)):
        (args.output/name).write_bytes(encoded(value))
    (args.output/"timing-summary.csv").write_bytes(csv_bytes(data))
    print(encoded({"cases":1,"rows":5,"performance_acceptance":False,"new_native_physics_evaluations":0}).decode(),end="")

if __name__=="__main__":main()
