#!/usr/bin/env python3
"""One copied full QCD3 factor control; unchanged physics/stability policy."""
import sys
sys.dont_write_bytecode = True

import argparse
from collections import Counter
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import process_prototype
import check_fallback as check

ROOT, HERE = check.ROOT, Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(ROOT/"tools")]
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam


def dump(path, value):
    path.write_text(json.dumps(value,indent=2)+"\n")


def rows(path):
    result=[]
    for line in path.read_text().splitlines():
        f=line.split()
        assert len(f)==8
        row=dict(point=int(f[0]),order=int(f[1]),status=int(f[2]),
                 coefficients=list(map(float,f[3:7])),accuracy=float(f[7]))
        assert all(math.isfinite(x) for x in row["coefficients"]+[row["accuracy"]])
        assert row["accuracy"]>=0
        bench.validate_status(row["status"],6)
        result.append(row)
    return result


def compare(left,right,config):
    assert [(r["point"],r["order"]) for r in left] == [
        (r["point"],r["order"]) for r in right]
    scales={}
    for row in left+right:
        scales[row["point"]]=max(scales.get(row["point"],0),
                                *map(abs,row["coefficients"]))
    failures,changes=[],[]
    maximum=0.0
    equal=0
    for a,b in zip(left,right):
        if a["accuracy"]!=b["accuracy"] or a["status"]!=b["status"]:
            changes.append(dict(point=a["point"],order=a["order"],
                original_status=a["status"],factor_status=b["status"],
                original_accuracy=a["accuracy"],factor_accuracy=b["accuracy"]))
        for coefficient,(x,y) in enumerate(zip(a["coefficients"],b["coefficients"])):
            tolerance=config["relative_tolerance"]*max(abs(x),abs(y))
            tolerance+=config["scale_tolerance"]*scales[a["point"]]
            difference=abs(x-y)
            fraction=difference/tolerance if tolerance else (0 if difference==0 else math.inf)
            maximum=max(maximum,fraction)
            equal+=x.hex()==y.hex()
            if difference>tolerance:
                failures.append(dict(point=a["point"],order=a["order"],
                    coefficient=coefficient,original=x,factor=y,
                    tolerance=tolerance,difference=difference))
    counts=lambda data:dict(Counter(str(r["status"]) for r in data if r["order"]==0))
    original_counts,factor_counts=counts(left),counts(right)
    rescues=lambda data:sum(v for k,v in data.items() if int(k)//100==3)
    return dict(passed=not failures,failures=failures,
        maximum_tolerance_fraction=maximum,coefficient_count=len(left)*4,
        printed_binary64_identical_coefficients=equal,
        all_printed_coefficients_identical=equal==len(left)*4,
        accuracy_or_status_changes=changes,original_status_counts=original_counts,
        factor_status_counts=factor_counts,
        added_qp_rescues=rescues(factor_counts)-rescues(original_counts),
        requires_rescue_investigation=rescues(factor_counts)>rescues(original_counts))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--cpu",type=int,default=2)
    args=parser.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    report=HERE/args.run_id
    report.mkdir(exist_ok=False)
    source=ROOT/".benchmark-work/policy-v2/generation-abi3-terminal-fixed/outputs/candidate/qcd-3"
    work=ROOT/".benchmark-work/experiments/symmetric-factorization"/args.run_id
    points=ROOT/"docs/benchmarkfolder/points/qcd-3.dat"
    config=json.loads((ROOT/"docs/benchmarkfolder/cases.json").read_text())
    assert int(points.read_text().splitlines()[0])==config["points_per_case"]==100
    assert config["relative_tolerance"]==1e-7 and config["scale_tolerance"]==1e-10
    policy=dict(relative_tolerance=config["relative_tolerance"],
        scale_tolerance=config["scale_tolerance"],
        formula="abs(a-b) <= 1e-7*max(abs(a),abs(b)) + 1e-10*"
                "max(abs(all four coefficients of both variants at this point))",
        stability={"MLStabThres":1e-9,"COLLIERRequiredAccuracy":1e-10},
        backend=6,points=100,passes=1,
        input_policy="Byte-identical retained physics cards and all100 point stream",
        status_policy="All rows require stable own Ninja DP/QP statuses; preserve "
                      "and report every accuracy/status change. Added rescues "
                      "require investigation; never loosen stability thresholds.",
        acceptance_eligible=False)
    dump(report/"predeclared-policy.json",policy)
    result=dict(complete=False,acceptance_eligible=False,case="qcd-3",
        backend=6,cpu=args.cpu,policy=policy,builds={},measurements={},
        points_sha256=check.sha(points),
        source_manifest=bench.generated_source_manifest(source),
        source_identity=json.loads((source/"generation.json").read_text()),
        scope="Copied full100-point changed-association correctness and guard-count "
              "diagnostic. Concurrent host work; no process speed claim.")
    dump(report/"result.json",result)
    for name in ("run_process.py","process_prototype.py","factor.py","check_fallback.py"):
        (report/name).write_bytes((HERE/name).read_bytes())
    shutil.copy2(points,report/"points.dat")
    inputs=report/"source-inputs"
    inputs.mkdir()
    for name in ("generation.json","compilation.json","compiled-sources.json",
                 "Cards/param_card.dat","Cards/MadLoopParams.dat","Source/make_opts"):
        p=source/name
        if p.exists():
            target=inputs/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        MAKEFLAGS="SHELL="+shutil.which("bash"),
        LD_LIBRARY_PATH=os.pathsep.join(str(bench.BASELINE/"HEPTools"/p)
                                        for p in ("ninja/lib","collier")))
    jobs={}
    try:
      for variant in ("original","factor"):
        output=work/variant
        output.parent.mkdir(parents=True,exist_ok=True)
        assert not output.exists()
        subprocess.run(["cp","-a","--reflink=auto",str(source),str(output)],check=True)
        for p in output.rglob("*"):
            if p.is_symlink() and p.resolve().is_relative_to(source):
                target=output/p.resolve().relative_to(source)
                p.unlink()
                p.symlink_to(os.path.relpath(target,p.parent))
        directory=next(output.glob("SubProcesses/P*_*"))
        jobs[variant]=(output,directory)
        assert (output/"Cards/param_card.dat").read_bytes()==(source/"Cards/param_card.dat").read_bytes()
        card=MadLoopParam(str(output/"Cards/MadLoopParams.dat"))
        for key,value in dict(MLReductionLib="6",CTModeRun=-1,
                             MLStabThres=1e-9,COLLIERRequiredAccuracy=1e-10).items():
            card[key]=value
        for p in [output/"Cards/MadLoopParams.dat",
                  *(output/"SubProcesses").rglob("MadLoopParams.dat")]:
            card.write(str(p),template=str(ROOT/"Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
        shutil.copy2(output/"Cards/MadLoopParams.dat",report/(variant+"-MadLoopParams.dat"))
        changed={}
        if variant=="factor":
            changed,proof=process_prototype.transform(directory,trace=True)
            dump(report/"factor-proof.json",proof)
        patch=[]
        for name,text in changed.items():
            p=directory/name
            old=p.read_text()
            patch.extend(difflib.unified_diff(old.splitlines(True),text.splitlines(True),
                fromfile="original/"+name,tofile="factor/"+name))
            p.write_text(text)
        if patch:
            (report/"factor.patch").write_text("".join(patch))
        start=time.monotonic()
        with (report/(variant+"-build.log")).open("w") as log:
            subprocess.run(["make","-j2","check"],cwd=directory,env=environment,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest=bench.generated_source_manifest(output)
        dump(report/(variant+"-compiled-sources.json"),manifest)
        result["builds"][variant]=dict(seconds=time.monotonic()-start,
            executable_sha256=check.sha(directory/"check"),
            compiled_sources_sha256=manifest["files_sha256"])
        dump(report/"result.json",result)
      assert (report/"original-MadLoopParams.dat").read_bytes()==(
          report/"factor-MadLoopParams.dat").read_bytes()
      for variant,(output,directory) in jobs.items():
        # Match the generated SETPARA search order and reject shadow cards.
        candidates=[directory/"param_card.dat"]+[
            directory.joinpath(*([".."]*level),"Cards/param_card.dat") for level in range(6)]
        actual=next(p for p in candidates if p.is_file())
        assert actual.read_bytes()==(source/"Cards/param_card.dat").read_bytes()
        for pattern in ("*HelFilter.dat","*LoopFilter.dat"):
            for p in (output/"SubProcesses").rglob(pattern):
                p.unlink()
        for name in ("values.dat","timing.dat","timed_status.dat","factor_trace.dat"):
            (directory/name).unlink(missing_ok=True)
        start=time.monotonic()
        with (report/(variant+"-run.log")).open("w") as log:
            ran=subprocess.run(["./check",str(points),"1"],cwd=directory,env=environment,
                stdout=log,stderr=subprocess.STDOUT)
        for name in ("values.dat","timing.dat","timed_status.dat","dimensions.dat","factor_trace.dat"):
            if (directory/name).exists():
                shutil.copy2(directory/name,report/(variant+"-"+name))
        result["measurements"][variant]=dict(returncode=ran.returncode,
                                            wall_seconds=time.monotonic()-start)
        dump(report/"result.json",result)
        ran.check_returncode()
        data=rows(directory/"values.dat")
        dims=list(map(int,(directory/"dimensions.dat").read_text().split()))
        assert len(dims)==3 and dims[0]==max(dims[1:]) and dims[2]<=dims[1]
        assert [(r["point"],r["order"]) for r in data]==[
            (p,o) for p in range(1,101) for o in range(dims[1]+1)]
        timed={}
        for line in (directory/"timed_status.dat").read_text().splitlines():
            code,count=map(int,line.split())
            bench.validate_status(code,6)
            assert count>0 and str(code) not in timed
            timed[str(code)]=count
        assert sum(timed.values())==100
        times=list(map(float,(directory/"timing.dat").read_text().split()))
        assert len(times)==2 and all(math.isfinite(t) and t>0 for t in times)
        result["measurements"][variant].update(rows=data,timed_status_counts=timed,
            seconds_per_point=times[1]/100,cold_initialization_and_warm_pass_seconds=times[0],
            values_sha256=check.sha(directory/"values.dat"),answer_dimensions=dims)
        dump(report/"result.json",result)
        print(variant,"100 points complete",timed,flush=True)
      comparison=compare(result["measurements"]["original"]["rows"],
                         result["measurements"]["factor"]["rows"],config)
      result["comparison"]=comparison
      counts=[]
      for line in (report/"factor-factor_trace.dat").read_text().splitlines():
        index,fast,fallback=map(int,line.split())
        counts.append(dict(index=index,fast=fast,fallback=fallback))
      assert [c["index"] for c in counts]==list(range(1,len(proof["helpers"])+1))
      result["guard_counts"]=counts
      result["guard_totals"]={name:sum(c[name] for c in counts) for name in ("fast","fallback")}
      result["complete"]=True
      dump(report/"result.json",result)
      assert comparison["passed"],comparison["failures"]
      print("comparison passed",comparison["maximum_tolerance_fraction"],
            "guard totals",result["guard_totals"],flush=True)
    except Exception as error:
      result["error"]=repr(error)
      dump(report/"result.json",result)
      raise


if __name__=="__main__":
    main()
