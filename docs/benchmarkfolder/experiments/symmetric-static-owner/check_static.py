#!/usr/bin/env python3
"""Exact safety controls for separately versioned static-owner kernels."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import static_owner
import check_shared_general as fixture
from check_shared_traps import DRIVER as TRAP_DRIVER

check = fixture.check
HERE = Path(__file__).resolve().parent
FLAGS = {
    "strict": ["-O3", "-frounding-math", "-fno-automatic", "-fcheck=all",
               "-ffixed-line-length-132"],
    "standalone": ["-O3", "-w", "-fPIC", "-ffixed-line-length-132"],
    "nlo": ["-O3", "-fPIC", "-ffixed-line-length-132", "-fno-automatic"],
}


def instrument(source):
    source = source.replace("  INTEGER I,J,K,L,M\n",
        "  INTEGER I,J,K,L,M\n  INTEGER FAST,FALLBACK\n"
        "  COMMON/SHARED_TRACE/FAST,FALLBACK\n", 1)
    assert source.count("C STATIC_OWNER_FAST_PATH") == 1
    source = source.replace("C STATIC_OWNER_FAST_PATH",
                            "C STATIC_OWNER_FAST_PATH\nFAST=FAST+1", 1)
    return re.sub(r"(?m)^( +)(CALL TEST_(?:UPDATE|TERMINAL)_WL_\d+_1(?:_DENSE)?\()",
                  r"\1FALLBACK=FALLBACK+1\n\1\2", source)


def write_sources(work, rank, vertex, terminal, case, inventory, folded):
    work.mkdir(parents=True, exist_ok=False)
    (work / "coef_specs.inc").write_text(
        "      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n"
        "      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n")
    (work / "loop_max_coefs.inc").write_text(
        "      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=210)\n")
    gen = check.FortranPolynomialRoutines(6, updater_max_rank=2, proc_prefix="TEST_",
        sub_prefix="TEST_", coef_format="complex*32")
    name = "TEST_" + ("TERMINAL" if terminal else "UPDATE") + f"_WL_{rank}_1"
    source = static_owner.generate(gen, rank, vertex, terminal, "_SHARED",
        case=case, inventory_path=inventory, fold_guards=folded)
    full = gen.write_compact_wl_updater(rank, 1)
    if terminal:
        full += "\n\n" + gen.write_compact_wl_updater(rank, 1, terminal=True)
    for filename, text in [
        ("constants.f", gen.write_polynomial_constant_module()),
        ("full.f", full), ("static.f", source),
        ("checked-static.f", instrument(source)),
    ]:
        with check.FortranWriter(str(work / filename)) as writer:
            writer.writelines(text)
    groups = static_owner.shared.groups(vertex, terminal, case=case,
                                        inventory_path=inventory)
    (work / "driver.f90").write_text(fixture.driver(name, groups, terminal))
    trap = TRAP_DRIVER.replace("TEST_UPDATE_WL_2_1_SHARED", name + "_SHARED")
    if terminal:
        trap = trap.replace("b,4,4,out)", "b,4,4,out,.false.)")
    (work / "traps.f90").write_text(trap)
    return groups


def run(command, work, log):
    result = subprocess.run(command, cwd=work, capture_output=True, text=True)
    (work / log).write_text(result.stdout + result.stderr)
    result.check_returncode()
    return result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ranks", default="2,4")
    parser.add_argument("--case", default="qcd-4")
    parser.add_argument("--vertex", default="MP_FFV1L2_1")
    parser.add_argument("--flags", default="standalone,nlo")
    parser.add_argument("--modes", default="folded")
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    report = HERE / args.tag
    workroot = check.ROOT / ".benchmark-work/experiments/symmetric-static-owner" / args.tag
    report.mkdir(exist_ok=False)
    inventory = static_owner.PREVIOUS / "final-inventory.json"
    data = json.loads(inventory.read_text())[args.case]["routines"]
    vertices = ([args.vertex] if args.vertex != "all" else
        [v for v, d in data.items()
         if any(s["in_size"] == s["out_size"] == 4 for s in d["shapes"])])
    for path in (Path(__file__), HERE / "static_owner.py", inventory):
        shutil.copy2(path, report / path.name)
    results = []
    state = dict(complete=False, acceptance_eligible=False, results=results,
        cpu=args.cpu, case=args.case, flags=FLAGS, command=vars(args),
        previous_manifest_sha256=check.sha(static_owner.PREVIOUS / "SHA256SUMS.json"),
        generator_sha256=check.sha(check.ROOT / "madgraph/various/q_polynomial.py"))
    try:
      for mode in args.modes.split(","):
        assert mode in ("folded", "owner-only")
        for flags_name in args.flags.split(","):
          flags = FLAGS[flags_name]
          for rank in (int(r) for r in args.ranks.split(",")):
            for terminal in (False, True):
              for vertex in vertices:
                groups = static_owner.shared.groups(vertex, terminal,
                    case=args.case, inventory_path=inventory)
                if not groups:
                    continue
                key = f"{mode}-{flags_name}-{vertex}-rank{rank}-"
                key += "terminal" if terminal else "full"
                work = workroot / key
                write_sources(work, rank, vertex, terminal, args.case, inventory,
                              mode == "folded")
                started = time.monotonic()
                # Reuse the same exact compiled kernel objects for both drivers.
                command = ["gfortran", *flags, "-c", "constants.f", "full.f",
                           "checked-static.f"]
                run(command, work, "kernel-build.log")
                runs = {}
                for mainfile, exe in (("driver.f90", "check"), ("traps.f90", "traps")):
                    link = ["gfortran", *flags, "constants.o", "full.o",
                            "checked-static.o", mainfile, "-o", exe]
                    run(link, work, exe + "-build.log")
                    output = run([str(work / exe)], work, exe + ".log")
                    runs[exe] = dict(command=link, passed=True, output=output,
                                    executable_sha256=check.sha(work / exe))
                result = dict(key=key, rank=rank, terminal=terminal, vertex=vertex,
                    mode=mode, flags=flags_name, groups=groups, runs=runs,
                    build_command=command, seconds=time.monotonic() - started,
                    sources={p.name: check.sha(p) for p in work.iterdir()
                             if p.suffix in (".f", ".f90", ".inc")})
                results.append(result)
                destination = report / key
                destination.mkdir()
                for path in work.iterdir():
                    if path.suffix in (".f", ".f90", ".inc", ".log"):
                        shutil.copy2(path, destination / path.name)
                (report / "exact.json").write_text(json.dumps(state, indent=2) + "\n")
                print(key, runs["check"]["output"].strip(),
                      runs["traps"]["output"].strip(), flush=True)
    except Exception as exc:
        state["failure"] = repr(exc)
        state["failed_work"] = str(work)
        destination = report / ("failed-" + key)
        destination.mkdir(exist_ok=True)
        for path in work.iterdir():
            if path.suffix in (".f", ".f90", ".inc", ".log"):
                shutil.copy2(path, destination / path.name)
        (report / "exact.json").write_text(json.dumps(state, indent=2) + "\n")
        raise
    state.update(complete=True,
        comparisons=sum(12000 if r["terminal"] else 6000 for r in results),
        trap_controls=5 * len(results),
        scope="Every output/input bit and cumulative IEEE flag versus the "
              "original public/private updater; same retained four-rounding, "
              "signed-zero, cancellation, exceptional/subnormal, empty and "
              "terminal entry-flag controls. Repeated calls cover retained "
              "scratch. Actual fast/fallback counters and trap fallback required. "
              "Kernel correctness only, no process/performance acceptance.")
    (report / "exact.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
