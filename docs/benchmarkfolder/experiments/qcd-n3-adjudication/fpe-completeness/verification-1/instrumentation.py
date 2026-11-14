#!/usr/bin/env python3
"""Read host reduction-failure flags after the retained four-point QP calls.

Only the private protocol driver changes. Fixed QP remains unchecked, and
false failure flags are evidence of reported completeness, not an accuracy test.
"""
from contextlib import ExitStack
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench
import tirust_benchmark_cohorts as cohort
from tirust_benchmark_compiler import capture_effective_flags
from tirust_complex_stress import parse_forced_qp

REPORT = Path(__file__).resolve().parent
PARENT = REPORT.parent
ORIGINAL_REPORT = PARENT / "runs/minimal/candidate-9-init-dp-unchecked"
RESULT = REPORT / "attempt-1"
WORKSPACE = ROOT / ".tirust-build/qcd-n3-fpe-completeness"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def instrument(driver, prefix):
    """Insert only declarations, an output open, and post-call observations."""
    if driver.count("FORCE_STABILITY_CHECK(.FALSE.)") != 1:
        raise ValueError("Expected the retained fixed-QP unchecked driver")
    declaration = "      CHARACTER*16 COMMAND\n"
    if driver.count(declaration) != 1:
        raise ValueError("Unexpected driver declarations")
    driver = driver.replace(declaration, declaration +
        "      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION\n" +
        f"      COMMON/{prefix}FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,\n" +
        "     $ FPE_IN_QP_REDUCTION\n")
    marker = f"      CALL {prefix}FORCE_STABILITY_CHECK(.FALSE.)\n"
    driver = driver.replace(marker, marker +
        "      OPEN(78,FILE='fpe-calls.dat',STATUS='REPLACE')\n")
    call = f"CALL {prefix}SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)"
    if driver.count(call) != 3:
        raise ValueError("Unexpected number of call sites in retained driver")
    details = [(0, "0", "1"), (1, "K", "1+MOD(K-1,NPOINTS)"), (2, "L", "K")]
    rows = []
    count = 0
    for line in driver.splitlines(keepends=True):
        rows.append(line)
        if line.strip() == call:
            phase, iteration, point = details[count]
            rows.append(f"      WRITE(78,'(6I12)') {phase},{iteration},{point},CODE,\n"
                        "     $ MERGE(1,0,FPE_IN_DP_REDUCTION),\n"
                        "     $ MERGE(1,0,FPE_IN_QP_REDUCTION)\n"
                        "      FLUSH(78)\n")
            count += 1
    return "".join(rows)


def summarize_flags(path, count):
    rows = [list(map(int, line.split())) for line in Path(path).read_text().splitlines()]
    if not rows or any(len(row) != 6 or any(flag not in (0, 1) for flag in row[-2:]) for row in rows):
        raise ValueError("Malformed FPE flag evidence")
    if rows[0][:3] != [0, 0, 1]:
        raise ValueError("Missing bootstrap FPE observation")
    warm = [row for row in rows if row[0] == 1]
    if len(warm) < count or any(row[1:3] != [i, 1+(i-1) % count] for i, row in enumerate(warm, 1)):
        raise ValueError("Incomplete/reordered warmup FPE observations")
    final = [row for row in rows if row[0] == 2]
    if [row[:4] for row in final] != [[2, 1, i, 120] for i in range(1, count+1)]:
        raise ValueError("Incomplete/reordered fixed-QP FPE observations")
    if rows != [rows[0]] + warm + final:
        raise ValueError("Unexpected FPE phase ordering")
    return {"calls": rows, "flagged_calls": [row for row in rows if any(row[-2:])],
            "all_observed_flags_false": not any(any(row[-2:]) for row in rows),
            "final_qp_flags_false": not any(any(row[-2:]) for row in final)}


def run():
    RESULT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, RESULT / "run.py")
    try:
        parent = module(PARENT / "run.py", "qcd3_original_adjudication")
        pilot, config, case, parent_points, parent_file, card_sha = parent.inputs()
        old = json.loads((ORIGINAL_REPORT / "result.json").read_text())
        parent.validate_binding(ORIGINAL_REPORT, old, pilot, config, case, parent_points, parent_file, card_sha)
        old_compiled = json.loads((ORIGINAL_REPORT / "diagnostic-compilation.json").read_text())
        old_inputs = json.loads((ORIGINAL_REPORT / "candidate-inputs.json").read_text())
        original = Path(json.loads((ORIGINAL_REPORT / "isolation.json").read_text())["copy"])
        points = PARENT / "points/minimal.dat"
        for name in ("MadLoopParams.dat", "param_card.dat", "result.json", "candidate-inputs.json",
                     "diagnostic-compilation.json", "check_sa.f"):
            shutil.copy2(ORIGINAL_REPORT / name, RESULT / ("original-" + name))
        shutil.copy2(points, RESULT / "points.dat")
        shutil.copy2(points.with_suffix(".json"), RESULT / "points.json")
        for name in cohort.PROTOCOLS:
            shutil.copy2(ROOT / "tools" / name, RESULT / name)
        extra_protocols = {
            "tirust_complex_stress.py": ROOT / "tools/tirust_complex_stress.py",
            "original-adjudication.py": PARENT / "run.py",
            "frozen-worker.py": PARENT / "protocol/tirust_benchmark_worker.py",
        }
        for name, path in extra_protocols.items():
            shutil.copy2(path, RESULT / name)
        specification = {"scope": "Fixed-QP reported tensor completeness; no independent stability or performance certification",
            "performance_eligible": False, "stability_certified": False, "source_indices": [39, 60, 91, 93],
            "original_result_sha256": bench.digest(ORIGINAL_REPORT / "result.json"),
            "points_sha256": bench.digest(points), "parent_points_sha256": bench.digest(parent_file),
            "host_before": bench.host_load_snapshot(), "created_utc": bench.utc_now(),
            "protocols": {name: bench.digest(RESULT / name) for name in (*cohort.PROTOCOLS, "run.py", *extra_protocols)}}
        cohort.exclusive_json(RESULT / "specification.json", specification)
        with ExitStack() as stack:
            lock = stack.enter_context((ROOT / ".tirust-build/numerical-tests.lock").open("a"))
            print("Waiting for numerical-tests lock", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print("Acquired numerical-tests lock", flush=True)
            def ignored(directory, names):
                return [name for name in names if name in ("build-history", "worker-raw")
                        or stat.S_ISFIFO((Path(directory) / name).lstat().st_mode)]
            shutil.copytree(original, WORKSPACE, symlinks=True, ignore=ignored)
            cohort.mutable_paths_are_local(WORKSPACE)
            directory = next(WORKSPACE.glob("SubProcesses/P*_*"))
            for name in ("check_sa.f", "check_sa.o", "check"):
                if not (directory / name).resolve().is_relative_to(WORKSPACE):
                    raise ValueError("Diagnostic build output escapes private directory")
            before = bench.generated_sources(WORKSPACE)
            objects = {str(p.relative_to(WORKSPACE)): bench.digest(p) for p in WORKSPACE.rglob("*.o") if p.name != "check_sa.o"}
            deps = bench.dependency_inputs(directory)
            runtime = bench.runtime_identity(directory)
            env = bench.environment()
            flags = capture_effective_flags(directory, env=env)
            if (before != old_compiled["sources"] or objects != old_compiled["numerical_objects_after"]
                    or bench.digest(directory / "check") != old_compiled["executable_sha256"]
                    or deps != old_compiled["link_inputs_after"]
                    or not bench.same_compiler_policy(flags, old_compiled["effective_compiler_flags"])
                    or any(runtime[k] != old_inputs["runtime"][k] for k in ("libraries", "archives", "metadata"))):
                raise ValueError("Retained QP source/objects/runtime changed before instrumentation")
            prefix = old_inputs["compilation"]["worker_render_parameters"]["prefix"]
            source = (directory / "loop_matrix.f").read_text()
            cohort.render(cohort.TEMPLATE.read_text(), old_inputs["compilation"]["worker_render_parameters"], source)
            driver = instrument((directory / "check_sa.f").read_text(), prefix)
            (directory / "check_sa.f").write_text(driver)
            (RESULT / "check_sa.f").write_text(driver)
            plan = cohort.link_commands(WORKSPACE, directory, env, RESULT)
            cohort.exclusive_json(RESULT / "driver-build-plan.json", plan)
            with (RESULT / "driver-build.log").open("x") as log:
                for command in plan["commands"]:
                    subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
            expected = dict(before, **{str((directory / "check_sa.f").relative_to(WORKSPACE)): bench.digest(directory / "check_sa.f")})
            after_objects = {str(p.relative_to(WORKSPACE)): bench.digest(p) for p in WORKSPACE.rglob("*.o") if p.name != "check_sa.o"}
            if (objects != after_objects or bench.generated_sources(WORKSPACE) != expected
                    or deps != bench.dependency_inputs(directory)
                    or not bench.same_compiler_policy(flags, capture_effective_flags(directory, env=env))):
                raise ValueError("Post-call instrumentation altered numerical implementation")
            cohort.exclusive_json(RESULT / "compilation.json", {"objects_before": objects, "objects_after": after_objects,
                "sources_before": before, "sources_after": expected, "deps_before": deps, "deps_after": bench.dependency_inputs(directory),
                "flags": flags, "executable_sha256": bench.digest(directory / "check"), "runtime_before": runtime})
            card = (ORIGINAL_REPORT / "MadLoopParams.dat").read_bytes()
            for path in {WORKSPACE / "Cards/MadLoopParams.dat", *(WORKSPACE / "SubProcesses").rglob("MadLoopParams.dat")}:
                path.write_bytes(card)
            if bench.digest(WORKSPACE / "Cards/param_card.dat") != card_sha:
                raise ValueError("Physical card changed")
            shutil.copy2(WORKSPACE / "Cards/MadLoopParams.dat", RESULT / "MadLoopParams.dat")
            (directory / "fpe-calls.dat").unlink(missing_ok=True)
            worker_module = module(PARENT / "protocol/tirust_benchmark_worker.py", "fpe_frozen_worker")
            try:
                with worker_module.Worker(["taskset", "-c", "2", "./check", str(points)], directory,
                        RESULT / "worker.log", env=env, timeout=180) as worker:
                    worker.run(1)
            finally:
                cohort.capture(directory, RESULT / "raw", ("fpe-calls.dat", "timing.dat", "values.dat", "timed_status.dat", "dimensions.dat"))
            values = parse_forced_qp(RESULT, 9, 4)
            for row in values["rows"]:
                row["point"] = specification["source_indices"][row["point"]-1]
            same = values["rows"] == old["observation"]["rows"]
            summary = summarize_flags(RESULT / "raw/fpe-calls.dat", 4)
            after = bench.runtime_identity(directory)
            if (any(runtime[k] != after[k] for k in ("libraries", "archives", "metadata"))
                    or bench.generated_sources(WORKSPACE) != expected or deps != bench.dependency_inputs(directory)
                    or bench.digest(points) != specification["points_sha256"]
                    or any(bench.digest(path) != specification["protocols"][name] for name, path in extra_protocols.items())):
                raise ValueError("Inputs/runtime changed during the FPE observation")
            cohort.exclusive_json(RESULT / "result.json", {**specification, **summary, "coefficients_byte_equivalent_rows": same,
                "values_file_byte_identical": (RESULT / "raw/values.dat").read_bytes() == (ORIGINAL_REPORT / "raw/values.dat").read_bytes(),
                "observation": values, "runtime_after": after, "host_after": bench.host_load_snapshot()})
            print(json.dumps({**summary, "same_rows": same}), flush=True)
    except BaseException as error:
        cohort.exclusive_json(RESULT / "failure.json", {"error": repr(error), "performance_eligible": False})
        raise


if __name__ == "__main__":
    run()
