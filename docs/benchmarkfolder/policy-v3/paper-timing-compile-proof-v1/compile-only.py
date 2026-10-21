"""Exactly two root-authorized compile/link checks; never run an executable."""
from pathlib import Path
import datetime
import os
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_paper_timing_v3 as paper

WORK = ROOT / ".benchmark-work/paper-timing-compatibility-v1"
REPORT = ROOT / "docs/benchmarkfolder/policy-v3/paper-timing-compile-proof-v1"
SPEC = ROOT / "docs/benchmarkfolder/policy-v3/paper-timing-software-proof-v1/specification-held-v2.json"
REPORT.mkdir(parents=True, exist_ok=False)
stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
status = {"scope": "compile/link only; no Worker/SETPARA/reducer/MP/executable invocation",
    "pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgrp(), "start_ticks": int(stat[19]),
    "affinity": sorted(os.sched_getaffinity(0)), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "controller": paper.supplement.bound(Path(__file__)), "specification": paper.supplement.bound(SPEC),
    "requested": ["qcd-1", "supplement-gg-hh"], "backend": 8, "completed": [], "status": "incomplete"}
assert status["affinity"] == [6]
paper.bench.dump(REPORT / "status.json", status)
spec = paper.validate_spec(paper.read(SPEC))
parent = paper.read(paper.supplement.checked(spec["parent_pair_spec"]))
config = paper.read(paper.supplement.checked(parent["configuration"]))
try:
    with paper.ownership.owned_runtime(REPORT), paper.workers.isolated_harness(WORK, REPORT):
        for cid in status["requested"]:
            case = spec["cases"][cid]
            job, directory, build = paper.prepare_target(case, 8, config, WORK, REPORT)
            state = paper.target_state(job / "outputs/candidate" / cid, 8,
                                       case["origin"]["reducer_runtime"], WORK / "runtime-assets")
            paper.check_build(case, 8, REPORT, state)
            filters = list((job / "outputs/candidate" / cid / "SubProcesses").rglob("*HelFilter.dat"))
            assert not filters, "No executable may initialize a helicity filter"
            paper.bench.dump(REPORT / "states" / (cid + ".json"), state)
            status["completed"].append({"case": cid, "output": paper.policy.relative(job / "outputs/candidate" / cid),
                "executable_sha256": build["executable_sha256"], "helicity_index": case["helicity_index"],
                "physical_helicity": case["physical_helicity"], "fortran_executable_invoked": False,
                "helicity_filters_created": 0})
            paper.bench.dump(REPORT / "status.json", status)
    status["status"] = "complete"
except BaseException as error:
    status["status"] = "failed"
    status["error"] = str(error)
    raise
finally:
    status["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    paper.bench.dump(REPORT / "status.json", status)
