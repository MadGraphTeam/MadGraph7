#!/usr/bin/env python3
"""Compile exact reviewed host policy slices with explicit dependency stubs.

This tests control flow and state, not Born/reducer numerical implementations.
Run from the repository with the toolchain environment loaded. All compiler and
test processes are pinned to CPU11; binaries/modules stay in ignored scratch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PROTOTYPE = HERE.parent
PREPARATION_SHA = "e4f6235e7c7d3d337c567a0acd248cfa49daaaa4d265f842f4ebf5a54a81602f"
HOST_SHA = "8c09cb06d2d087403ce0603e99dede30f22c4d7f450b5158043a7932047733b0"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def segment(text: str, start: str, end: str, name: str):
    if text.count("\n" + start) != 1:
        raise ValueError(f"non-unique {name} start")
    a = text.index("\n" + start) + 1
    b = text.index("\n" + end, a) + 1
    body = text[a:b]
    return body, {"name": name, "first_line": text[:a].count("\n") + 1,
                  "last_line": text[:b].count("\n"),
                  "sha256": hashlib.sha256(body.encode()).hexdigest()}


def extracted(source: Path):
    text = source.read_text()
    # The first two blocks are contiguous in the real host: rotation handling,
    # startup reads, Born calculation, counter, filter publication and shortcut.
    before, first = segment(text,
        "      NROTATIONS_DP_BU = NROTATIONS_DP\n",
        "      DO I=1,NEXTERNAL\n", "rotation-through-selected-helicity")
    qp, second = segment(text,
        "          IF(FPE_IN_QP_REDUCTION) THEN\n",
        "          IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN\n",
        "current-call-QP-certificate-and-status")
    cleanup, third = segment(text, " 9999 CONTINUE\n",
        "      SUBROUTINE ML5_0_CLEAR_CACHES()\n", "final-cleanup-and-acceptance")
    if not cleanup.rstrip().endswith("END"):
        raise ValueError("host cleanup did not end at the subroutine boundary")
    generated = (
        "C     Source-derived host policy fixture. See run_policy.py for slices.\n"
        "      SUBROUTINE AUDIT_HOST()\n"
        "      USE AUDIT_STATE\n"
        "      USE, INTRINSIC :: IEEE_ARITHMETIC\n"
        "      IMPLICIT NONE\n"
        "      INTEGER I,J,SET_RET_CODE_U\n" + before +
        "C     TEST HOOK: no loop amplitude or native reducer is evaluated.\n"
        "      CALL SIMULATED_LOOP_COMPLETION()\n"
        "      IF (LOOP_MODE.NE.0) THEN\n" + qp +
        "      ENDIF\n" + cleanup)
    return generated, [first, second, third]


def invoke(command, cwd, destination, expected=0):
    proc = subprocess.run(["taskset", "-c", "11", *command], cwd=cwd,
                          text=True, capture_output=True, timeout=60)
    destination.with_suffix(".stdout").write_text(proc.stdout)
    destination.with_suffix(".stderr").write_text(proc.stderr)
    record = {"argv": ["taskset", "-c", "11", *command], "cwd": str(cwd),
              "returncode": proc.returncode, "expected_returncode": expected,
              "stdout_sha256": sha(destination.with_suffix(".stdout")),
              "stderr_sha256": sha(destination.with_suffix(".stderr"))}
    if proc.returncode != expected:
        raise RuntimeError(f"unexpected exit {proc.returncode}: {destination}\n{proc.stdout}\n{proc.stderr}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="qualification-1")
    args = parser.parse_args()
    if not args.label.replace("-", "").isalnum():
        parser.error("label must be alphanumeric with hyphens")
    prep = PROTOTYPE / "preparation.json"
    source = PROTOTYPE / "loop_matrix.f"
    old = PROTOTYPE / "revision-1" / "loop_matrix.f"
    assert sha(prep) == PREPARATION_SHA
    assert sha(source) == HOST_SHA
    original_hashes = {str(p.relative_to(ROOT)): sha(p) for p in [prep, source, old]}
    output = HERE / args.label
    output.mkdir()
    scratch = ROOT / ".tirust-build" / "born-host-edge-fixtures" / args.label
    scratch.mkdir(parents=True, exist_ok=False)
    compiler = shutil.which("gfortran")
    if not compiler:
        raise RuntimeError("load tools/tirust_env.sh first")
    version = subprocess.check_output([compiler, "--version"], text=True)
    (output / "compiler-version.txt").write_text(version)
    sources = {}
    for name, original in [("revision-2", source), ("revision-1-negative", old)]:
        body, slices = extracted(original)
        retained = output / (name + ".f")
        retained.write_text(body)
        sources[name] = {"original": str(original.relative_to(ROOT)),
                         "original_sha256": sha(original), "slices": slices,
                         "retained_source": retained.name, "retained_sha256": sha(retained)}
    results = []
    for optimize in ["-O0", "-O3"]:
        for name in sources:
            work = scratch / (name + optimize)
            work.mkdir()
            exe = work / "host-policy"
            command = [compiler, optimize, "-g", "-fcheck=all", "-fbacktrace",
                       "-ffixed-line-length-132", "-ffree-line-length-none",
                       str(HERE / "policy-state.f90"),
                       str(output / (name + ".f")), str(HERE / "policy-driver.f90"),
                       "-o", str(exe)]
            build = invoke(command, work, output / (name + optimize + "-compile"))
            cases = []
            scenarios = ["startup", "stale", "certificate", "bypass"] if name == "revision-2" else ["startup", "stale"]
            for scenario in scenarios:
                run_dir = work / scenario
                run_dir.mkdir()
                expected = 0 if name == "revision-2" else {"startup": 71, "stale": 72}[scenario]
                result = invoke([str(exe), scenario], run_dir,
                                output / (name + optimize + "-" + scenario), expected)
                stdout = (output / (name + optimize + "-" + scenario + ".stdout")).read_text()
                if expected and {71: "DETECTED_FAILED_BORN_STARTUP_PUBLICATION", 72: "DETECTED_STALE_H3_ACCEPTANCE"}[expected] not in stdout:
                    raise AssertionError("negative control failed for an unintended reason")
                result["scenario"] = scenario
                result["published_filter_files"] = sorted(p.name for p in run_dir.glob("*-filter.dat"))
                cases.append(result)
            results.append({"source": name, "optimization": optimize, "build": build,
                            "executable_sha256": sha(exe), "cases": cases})
    for path, original_hash in original_hashes.items():
        assert sha(ROOT / path) == original_hash, f"source changed during run: {path}"
    record = {"schema": 1, "scope": "Extracted generated host policy with explicit Born/cache/loop stubs; no native reducer or physical amplitude.",
              "production_edited": False, "native_reducer_executed": False,
              "performance_eligible": False, "source_hashes_before_and_after": original_hashes,
              "dimensions": {"nexternal": 5, "ncomb": 48, "nloopgroups": 116, "nsquaredso": 1, "nsqso_born": 1},
              "sources": sources, "compiler": compiler, "runs": results,
              "source_inputs": {p.name: sha(p) for p in [Path(__file__), HERE / "policy-state.f90", HERE / "policy-driver.f90"]},
              "result": "PASS: revision 2 policy scenarios; revision 1 counterexamples detected at both compiler levels."}
    (output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record["result"])


if __name__ == "__main__":
    main()
