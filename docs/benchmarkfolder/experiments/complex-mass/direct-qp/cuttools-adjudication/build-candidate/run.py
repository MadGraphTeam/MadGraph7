#!/usr/bin/env python3
"""Isolated CutTools scalar-threshold diagnosis; never primary qualification.

Only the generated INITCT routine is instrumented. The existing private
OneLOop API changes its scalar threshold after CTSINIT; production sources,
libraries, physical points, and numerator construction remain unchanged.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

REPORT = Path(__file__).resolve().parent
DIRECT = REPORT.parent
ROOT = REPORT.parents[5]
PROCESS = Path("SubProcesses/P0_epem_mupmum")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def inventory(work):
    paths = list(work.rglob("*.o")) + list(work.rglob("*.a"))
    return {str(p.relative_to(work)): digest(p) for p in sorted(paths)}


def base(source):
    suffix = "-candidate" if source == "candidate" else ""
    return ROOT / (".tirust-build/complex-stress-direct-qp" + suffix)


def worktree(source):
    return ROOT / (".tirust-build/complex-stress-cuttools-" + source)


def patch(text):
    head = "      SUBROUTINE ML5_0_INITCT()\n"
    decl = ("      USE AVH_OLO_FORCUTTOOLS, ONLY: OLO_ONSHELL,\n"
            "     $ OLO_QP_ONSHELL,OLO_UNIT\n"
            "      REAL*8 CT_PROBE_THRESHOLD\n")
    call = "      CALL CTSINIT(THRS,LOOPLIB,EXT_NUM_FOR_R1)\n"
    body = ("      WRITE(*,*) 'CT_SCALAR_THRESHOLD_BEFORE',OLO_QP_ONSHELL()\n"
            "      OPEN(UNIT=913,FILE='cuttools_onshell.input',STATUS='OLD')\n"
            "      READ(913,*) CT_PROBE_THRESHOLD\n"
            "      CLOSE(913)\n"
            "      CALL OLO_ONSHELL(CT_PROBE_THRESHOLD)\n"
            "      WRITE(*,*) 'CT_SCALAR_THRESHOLD_AFTER',OLO_QP_ONSHELL()\n"
            "      CALL OLO_UNIT(6)\n"
            "      CALL OLO_UNIT(6,'printall')\n")
    if text.count(head) != 1 or text.count(call) != 1:
        raise ValueError("Unexpected INITCT source")
    return text.replace(head, head + decl).replace(call, call + body)


def build(source):
    origin, work = base(source), worktree(source)
    destination = REPORT / ("build-" + source)
    destination.mkdir(exist_ok=False)
    if work.exists():
        raise ValueError("Refusing to overwrite a diagnostic build")
    shutil.copytree(origin, work, symlinks=True)
    proc = work / PROCESS
    original = (origin / PROCESS / "CT_interface.f").read_text()
    modified = patch(original)
    (destination / "CT_interface.original.f").write_text(original)
    (destination / "CT_interface.instrumented.f").write_text(modified)
    (proc / "CT_interface.f").write_text(modified)
    before = inventory(work)
    module = ROOT / "vendor/CutTools/includects"
    command = [shutil.which("gfortran"), "-w", "-fPIC", "-ffixed-line-length-132",
               "-I../../Source/MODEL", "-I../../Source/DHELAS", "-I" + str(module),
               "-c", "CT_interface.f", "-o", "CT_interface.o"]
    definitions = (work / "SubProcesses/MadLoop_makefile_definitions").read_text()
    include = re.findall(r"(?m)^LOOP_INCLUDE = (.*)$", definitions)
    if len(include) != 1 or "$" in include[0]:
        raise ValueError("Unexpected generated include flags")
    command += shlex.split(include[0])
    make = ["make", "SHELL=" + shutil.which("bash"), "-j1", "check"]
    with (destination / "build.log").open("w") as log:
        subprocess.run(command, cwd=proc, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(make, cwd=proc, env=dict(os.environ, PWD=str(proc)),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    after = inventory(work)
    changed = [p for p in before if before[p] != after.get(p)]
    if changed != [str(PROCESS / "CT_interface.o")] or before.keys() != after.keys():
        raise ValueError("Unexpected changes to numerical objects/archives: " + repr(changed))
    shutil.copy2(__file__, destination / "run.py")
    dump(destination / "build.json", {
        "source": source, "origin": str(origin.relative_to(ROOT)),
        "isolated_worktree": str(work.relative_to(ROOT)), "commands": [command, make],
        "objects_archives_before": before, "objects_archives_after": after,
        "changed_objects": changed, "production_sources_changed": False,
        "original_interface_sha256": digest(destination / "CT_interface.original.f"),
        "instrumented_interface_sha256": digest(proc / "CT_interface.f"),
        "driver_sha256": digest(proc / "check_sa.f"),
        "executable_sha256": digest(proc / "check"),
        "physical_card_sha256": digest(work / "Cards/param_card.dat"),
        "module_sha256": digest(module / "avh_olo_forcuttools.mod"),
        "runner_sha256": digest(Path(__file__)),
    })


def run(args):
    work, destination = worktree(args.source), REPORT / args.label
    destination.mkdir(exist_ok=False)
    proc = work / PROCESS
    build_record = json.loads((REPORT / ("build-" + args.source) / "build.json").read_text())
    if inventory(work) != build_record["objects_archives_after"]:
        raise ValueError("Bound numerical objects/archives changed")
    for filename, field in (("check", "executable_sha256"),
                            ("check_sa.f", "driver_sha256"),
                            ("CT_interface.f", "instrumented_interface_sha256")):
        if digest(proc / filename) != build_record[field]:
            raise ValueError("Bound executable or source changed")
    parent_file = DIRECT.parent / "points/cms_ew_n2_epem_mupmum.json"
    parent = json.loads(parent_file.read_text())
    if digest(parent_file.with_suffix(".dat")) != parent["sha256"]:
        raise ValueError("Physical parent stream changed")
    momentum = "".join(" ".join("%.17e" % x for x in p) + "\n"
                       for p in parent["points"][args.point-1]["momenta"])
    card_file = "candidate-MadLoopParams.dat" if args.source == "candidate" else "MadLoopParams.dat"
    card = (DIRECT / card_file).read_text()
    for name, value in (("MLReductionLib", "1"), ("OSThres", "1e-14"),
                        ("CTModeRun", str(args.mode)), ("CTModeInit", str(args.mode))):
        card, count = re.subn(r"(?mi)(^#" + name + r"\s*\n)[^\n]*",
                            lambda m: m[1] + value, card)
        if count != 1:
            raise ValueError("Unexpected numerical card layout")
    (destination / "PS.input").write_text(momentum)
    (destination / "MadLoopParams.dat").write_text(card)
    (destination / "cuttools_onshell.input").write_text("%.17e\n" % args.threshold)
    shutil.copy2(__file__, destination / "run.py")
    for filename in ("PS.input", "cuttools_onshell.input"):
        shutil.copy2(destination / filename, proc / filename)
    (proc / "MadLoopParams.dat").unlink(missing_ok=True)
    (work / "SubProcesses/MadLoop5_resources/MadLoopParams.dat").write_text(card)
    (proc / "result.dat").unlink(missing_ok=True)
    record = {"source": args.source, "backend": 1, "point_index": args.point,
              "parent_points_sha256": parent["sha256"], "mode": args.mode,
              "madloop_onshell_threshold": 1e-14, "cuttools_scalar_threshold": args.threshold,
              "stability_certified": False, "performance_eligible": False,
              "build_manifest_sha256": digest(REPORT / ("build-" + args.source) / "build.json"),
              "physical_card_sha256": digest(work / "Cards/param_card.dat"),
              "numerical_card_sha256": digest(destination / "MadLoopParams.dat"),
              "point_input_sha256": digest(destination / "PS.input"),
              "threshold_input_sha256": digest(destination / "cuttools_onshell.input"),
              "runner_sha256": digest(Path(__file__))}
    started = time.monotonic()
    with (destination / "output.log").open("w") as log:
        try:
            result = subprocess.run([str(proc / "check")], cwd=proc, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.timeout,
                                    env=dict(os.environ, GFORTRAN_UNBUFFERED_ALL="y"))
            record.update(returncode=result.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            record.update(timed_out=True, timeout_seconds=args.timeout)
    record["elapsed_seconds_diagnostic_only"] = time.monotonic() - started
    if (proc / "result.dat").exists():
        shutil.copy2(proc / "result.dat", destination / "result.dat")
        record["result_sha256"] = digest(destination / "result.dat")
    record["output_sha256"] = digest(destination / "output.log")
    if inventory(work) != build_record["objects_archives_after"]:
        raise ValueError("Numerical objects/archives changed during observation")
    dump(destination / "status.json", record)
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "run"))
    parser.add_argument("--source", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--label")
    parser.add_argument("--point", type=int, choices=range(1, 22), default=20)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--mode", type=int, choices=(4, 5, 6), default=4)
    parser.add_argument("--timeout", type=float, default=30.)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from the repository root")
    if args.phase == "build":
        build(args.source)
    else:
        if not args.label or not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
            parser.error("Run needs a safe, previously unused label")
        if not math.isfinite(args.threshold) or not 0 < args.threshold < 1:
            parser.error("Invalid scalar threshold")
        run(args)


if __name__ == "__main__":
    main()
