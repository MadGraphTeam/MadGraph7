#!/usr/bin/env python3
"""Isolated automatic-policy cohort diagnostics, never acceptance timings.

Warm the original 100-point stream before freezing and timing its subsets.
The diagnostic driver buffers each call's accepted status, NinjaRS retry
counters, finite-value flag and host DP/QP reduction-failure flags. Numerical
objects and libraries are unchanged. The flags are observed, never reset here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import stat
import statistics
import subprocess

import tirust_benchmark as bench
from tirust_benchmark_compiler import capture_effective_flags
from tirust_benchmark_worker import Worker, parse_outputs, validate_status

ROOT = bench.ROOT
TEMPLATE = ROOT / "tools/tirust_benchmark_cohorts.f"
COHORTS = ("dp", "compensated_dp", "qp", "failed")
# TIRust status 219 is DP transport only. Per-call bridge route deltas split it
# into binary64-only reductions, double-double tier rescues and MPFR tier rescues.
TIRUST_COHORTS = ("dp_native", "dp_dd", "dp_mpfr", "qp", "failed")
ROUTE_SCHEMA = 1
ROUTE_FILES = ("classification-routes.dat", "cohort-routes.dat")
SCOPE = ("Diagnostic full-amplitude timing with per-call trace/finite-check overhead; "
         "not an acceptance measurement. DP denotes binary64 reduction arithmetic "
         "with internal method selection included, not absence of every internal retry. "
         "QP cohort includes failed DP attempts and natural automatic rescue. "
         "DP/QP FPE flags are sampled without reset; any set flag invalidates a "
         "cohort call, but a false flag is not an independent accuracy proof. "
         "Shared-host load is recorded; concurrent jobs can affect these timings.")
EXTRA = ("cohort-calls.dat", "classification-calls.dat", "classification-values.dat",
         "classification-timing.dat", "cohort-selection.dat", "warmup-flags.dat", *ROUTE_FILES)
PROTOCOLS = ("tirust_benchmark_cohorts.py", "tirust_benchmark_cohorts.f",
             "tirust_benchmark.py", "tirust_benchmark_worker.py", "tirust_benchmark_compiler.py")


def cohorts_for(backend):
    return TIRUST_COHORTS if backend == 9 else COHORTS


def route_hooks(prefix, enabled):
    """Fixed-form code for the C@COHORT_ROUTE_*@ anchors of the driver template.

    Each snapshot pair brackets exactly one amplitude call. The deltas cover
    bridge executions; MadLoop clears TIRust numerical caches and its Fortran
    coefficient cache at every new phase-space point, so a point's deltas
    include every reduction its amplitude reused.
    """
    if not enabled:
        return {}
    getter = f"      CALL {prefix}GET_TIRUST_DP_ROUTE_STATISTICS("
    return {
        "DECLARATIONS": ("      INTEGER*8 COHORT_RB(20),COHORT_RA(20)\n"
                         "      INTEGER COHORT_RSB,COHORT_RSA\n"
                         "      INTEGER*8, ALLOCATABLE :: ROUTEDATA(:,:)"),
        "BEFORE": getter + "COHORT_RB,COHORT_RSB)",
        "AFTER": getter + "COHORT_RA,COHORT_RSA)",
        "CLASS_OPEN": "      OPEN(79,FILE='classification-routes.dat',STATUS='REPLACE')",
        "CLASS_WRITE": (getter + "COHORT_RA,COHORT_RSA)\n"
                        "      WRITE(79,'(3I12,20I20)') K,COHORT_RSB,COHORT_RSA,\n"
                        "     $ COHORT_RA-COHORT_RB"),
        "CLASS_CLOSE": "      CLOSE(79)",
        "ALLOCATE": "      ALLOCATE(ROUTEDATA(22,NLOOPS*NSELECT))",
        "TIMED_STORE": ("      ROUTEDATA(1,NCALL)=COHORT_RSB\n"
                        "      ROUTEDATA(2,NCALL)=COHORT_RSA\n"
                        "      ROUTEDATA(3:22,NCALL)=COHORT_RA-COHORT_RB"),
        "TIMED_WRITE": ("      OPEN(73,FILE='cohort-routes.dat',STATUS='REPLACE')\n"
                        "      DO I=1,NCALL\n"
                        "         WRITE(73,'(4I12,20I20)') CALLDATA(1:2,I),\n"
                        "     $        ROUTEDATA(1:2,I),ROUTEDATA(3:22,I)\n"
                        "      ENDDO\n"
                        "      CLOSE(73)"),
        "DEALLOCATE": "      DEALLOCATE(ROUTEDATA)",
    }


def read_routes(path, keys):
    """Per-call route rows: key fields, two snapshot statuses, twenty deltas."""
    width = len(keys[0]) if keys else 0
    rows = []
    for line in Path(path).read_text().splitlines():
        values = list(map(int, line.split()))
        if len(values) != width + 22:
            raise ValueError("Malformed per-call route evidence")
        delta = values[width + 2:]
        if any(x < 0 for x in delta):
            raise ValueError("Route counters decreased within one amplitude call")
        rows.append({"key": tuple(values[:width]), "status_before": values[width],
                     "status_after": values[width + 1], "delta": delta})
    if [row["key"] for row in rows] != [tuple(k) for k in keys]:
        raise ValueError("Missing, reordered or substituted per-call route evidence")
    return rows


def route_category(category, route):
    """Refine a TIRust 'dp' call by its bridge routes; other categories are kept.

    A refusal, unreported reduction error, unavailable snapshot or a call with
    no bridge reduction at all cannot certify the arithmetic: it is 'failed'.
    """
    if category != "dp":
        return category
    delta = route["delta"]
    if route["status_before"] or route["status_after"]:
        return "failed"
    blocks = (delta[:10], delta[10:])
    if any(block[5] or block[6] for block in blocks):
        return "failed"
    if any(block[3] or block[4] for block in blocks):
        return "dp_mpfr"
    if any(block[2] for block in blocks):
        return "dp_dd"
    if any(block[0] or block[1] for block in blocks):
        return "dp_native"
    return "failed"


def apply_routes(calls, routes):
    for call, route in zip(calls, routes):
        call["route_status"] = [route["status_before"], route["status_after"]]
        call["route_delta"] = route["delta"]
        call["category"] = route_category(call["category"], route)
    return calls


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def locations(label):
    """Use explicit campaign paths; keep the historical CLI locations intact."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,100}", label):
        raise ValueError("Use a new simple lowercase report label")
    report = bench.REPORT / "experiments/cohorts" / label
    workspace = (bench.WORK / "cohort-diagnostics" / label if bench.CAMPAIGN_ID is not None
                 else ROOT / ".tirust-build/cohorts" / label)
    return report, workspace


def require_generation_binding(generation, compilation):
    binding = bench.campaign_binding()
    if any(record.get("campaign_binding") != binding for record in (generation, compilation)):
        raise ValueError("Cohort generation/compilation belongs to a different campaign")


def configure_clone(workspace, source, case, chain, config):
    """Only card/filter edits use the private workspace; guards keep the campaign namespace."""
    original = bench.WORK
    try:
        bench.WORK = workspace
        return bench.configure(source, case, chain, config)
    finally:
        bench.WORK = original


def validate_campaign_evidence(report, spec, inputs, compiled, results):
    """Bind diagnostic evidence to the selected, retained campaign manifest."""
    binding = spec.get("campaign_binding")
    if binding != bench.campaign_binding():
        raise ValueError("Select the cohort's exact campaign before validating its evidence")
    require_generation_binding(inputs["generation"], compiled["original"])
    if any(record.get("campaign_binding") != binding for record in (inputs, compiled, results)):
        raise ValueError("Cohort records have inconsistent campaign bindings")
    if binding is None:
        return
    manifest_path = Path(report) / "campaign-manifest.json"
    config_path = Path(report) / "cases.json"
    manifest = json.loads(manifest_path.read_text())
    shared = manifest["shared_inputs"]["files"]
    if (bench.digest(manifest_path) != binding["manifest_sha256"]
            or manifest != bench.CAMPAIGN_MANIFEST
            or bench.digest(config_path) != shared["cases.json"]
            or json.loads(config_path.read_text()) != spec["config"]
            or bench.content_digest(spec["config"]) != manifest["shared_inputs"]["policy_sha256"]
            or inputs["points_sha256"] != shared["points/" + spec["case"]["id"] + ".dat"]
            or inputs["point_metadata_sha256"] != shared["points/" + spec["case"]["id"] + ".json"]):
        raise ValueError("Cohort manifest, policy or point inputs differ from campaign preregistration")
    # Historical validation uses the snapshot and recorded native identities,
    # not today's installed library or exporter checkout.
    bench.campaign_guard("validate-retained", inputs=inputs)


def category(code, retries, backend, valid=True, fpe=(False, False)):
    """Strict steady status plus absolute, immediately sampled retry counters."""
    if (not valid or len(fpe) != 2 or any(fpe)
            or len(retries) != 4 or any(x < 0 for x in retries)
            or retries[1] + retries[2] != retries[0]
            or (backend != 8 and any(retries))):
        return "failed"
    try:
        actual = validate_status(code, backend)
    except ValueError:
        return "failed"
    if actual["hundreds"] == 3 and actual["tens"] == 2:
        return "qp"
    if actual["hundreds"] == 2 and actual["tens"] == 1:
        if retries[0]:
            return "compensated_dp" if retries[1] == retries[0] else "failed"
        return "dp" if retries[3] == 0 else "failed"
    return "failed"


def read_calls(path, backend, indices, loops, flags_required=True):
    result = []
    for line in Path(path).read_text().splitlines():
        values = list(map(int, line.split()))
        if (len(values) != (10 if flags_required else 8)
                or any(value not in (0, 1) for value in values[7:])):
            raise ValueError("Malformed per-call cohort evidence")
        iteration, point, code, *tail = values
        result.append({"iteration": iteration, "point": point, "status": code,
                       "retries": tail[:4], "finite_and_checked": bool(tail[4]),
                       "category": category(code, tail[:4], backend, bool(tail[4]))})
        if flags_required:
            flags = list(map(bool, tail[5:]))
            result[-1].update(fpe_dp=flags[0], fpe_qp=flags[1],
                             category=category(code, tail[:4], backend, bool(tail[4]), flags))
    if [(r["iteration"], r["point"]) for r in result] != [
            (i, p) for i in range(1, loops + 1) for p in indices]:
        raise ValueError("Missing, reordered or substituted per-call point evidence")
    return result


def validate_indices(indices, count):
    if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
            or indices != sorted(set(indices)) or any(i < 1 or i > count for i in indices)):
        raise ValueError("Cohort indices must be unique ordered original point IDs")


def classification(directory, backend, count, parent_sha, flags_required=True):
    calls = read_calls(Path(directory) / "classification-calls.dat", backend,
                       list(range(1, count + 1)), 1, flags_required)
    ndim, nso, nborn = map(int, (Path(directory) / "dimensions.dat").read_text().split())
    if ndim != max(nso, nborn) or min(nso, nborn) < 0:
        raise ValueError("Classification dimensions omit an answer column")
    values = []
    for line in (Path(directory) / "classification-values.dat").read_text().splitlines():
        fields = line.split()
        if len(fields) != 8:
            raise ValueError("Malformed classification numerical row")
        point, order, code = map(int, fields[:3])
        numbers = list(map(float, fields[3:]))
        values.append({"point": point, "order": order, "status": code,
                       "coefficients": [x if math.isfinite(x) else None for x in numbers[:4]],
                       "accuracy": numbers[4] if math.isfinite(numbers[4]) else None})
    if [(r["point"], r["order"]) for r in values] != [(p, j) for p in range(1, count+1) for j in range(ndim+1)]:
        raise ValueError("Classification columns/points are missing or reordered")
    for row in values:
        call = calls[row["point"]-1]
        if row["status"] != call["status"]:
            raise ValueError("Classification values and per-call status disagree")
        if call["finite_and_checked"] and (None in row["coefficients"] or row["accuracy"] is None or row["accuracy"] < 0):
            raise ValueError("Classification finite flag contradicts numerical values")
    if backend == 9:
        routes = read_routes(Path(directory) / "classification-routes.dat", [(p,) for p in range(1, count + 1)])
        apply_routes(calls, routes)
    indices = {name: [r["point"] for r in calls if r["category"] == name] for name in cohorts_for(backend)}
    for group in indices.values():
        validate_indices(group, count)
    result = {"parent_points_sha256": parent_sha, "backend": backend, "calls": calls, "rows": values,
            "indices": indices, "frozen_before_calibration": True,
            "classification_calls_sha256": bench.digest(Path(directory) / "classification-calls.dat"),
            "classification_values_sha256": bench.digest(Path(directory) / "classification-values.dat")}
    if backend == 9:
        result["route_schema"] = ROUTE_SCHEMA
        result["classification_routes_sha256"] = bench.digest(Path(directory) / "classification-routes.dat")
    if flags_required:
        flags_path = Path(directory) / "warmup-flags.dat"
        flags = [list(map(int, line.split())) for line in flags_path.read_text().splitlines()]
        if any(len(row) != 9 or any(x not in (0, 1) for x in row[-2:]) for row in flags):
            raise ValueError("Malformed warm-up FPE evidence")
        warm_rows = [line.split() for line in (Path(directory) / "warmup.dat").read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
        warm_calls = []
        for row in warm_rows:
            call = [int(row[0]), int(row[1]), int(row[3])]
            if not warm_calls or warm_calls[-1] != call:
                warm_calls.append(call)
        if [row[:3] for row in flags] != warm_calls:
            raise ValueError("Warm-up FPE trace differs from numerical calls")
        result["warmup_flags_sha256"] = bench.digest(flags_path)
        result["warmup_flags"] = flags
    return result


def read_observation(directory, backend, indices, loops, expected, flags_required=True):
    """Retain migrations/failures rather than reclassifying a timed subset."""
    calls = read_calls(Path(directory) / "cohort-calls.dat", backend, indices, loops, flags_required)
    route_error = None
    if backend == 9:
        try:
            routes = read_routes(Path(directory) / "cohort-routes.dat",
                                 [(r["iteration"], r["point"]) for r in calls])
        except (OSError, ValueError) as error:
            # Keep the failed raw evidence; unobserved routes cannot certify DP.
            route_error = {"type": type(error).__name__, "message": str(error)}
            routes = [{"status_before": -1, "status_after": -1, "delta": [0] * 20} for _ in calls]
        apply_routes(calls, routes)
    migrations = [row for row in calls if row["category"] != expected]
    numerical_error = None
    try:
        observed = parse_outputs(directory, backend, len(indices), loops)
    except Exception as error:
        observed = None
        numerical_error = {"type": type(error).__name__, "message": str(error)}
    if observed is not None:
        counts = Counter(str(row["status"]) for row in calls)
        if counts != Counter(observed["timed_status_counts"]):
            raise ValueError("Per-call statuses disagree with timed histogram")
        final = {row["point"]: row for row in calls[-len(indices):]}
        for row in observed["rows"]:
            row["point"] = indices[row["point"] - 1]
            if row["status"] != final[row["point"]]["status"]:
                raise ValueError("Final numerical rows disagree with per-call trace")
    result = {"observation": observed, "calls": calls, "migrations": migrations,
              "numerical_error": numerical_error,
              "pure_cohort": not migrations and observed is not None,
              "requested_category": expected, "indices": indices, "loops": loops}
    if backend == 9:
        result["route_error"] = route_error
    return result


def render(template, parameters, numerical_source, routes=False):
    prefix = parameters["prefix"]
    active = "\n".join(line for line in numerical_source.splitlines()
                       if not line or line[0] not in "Cc*!" )
    compact = re.sub(r"\s+|\$", "", active.upper())
    common = (f"COMMON/{prefix}NRS_RETRY_STATS/NRS_RETRY_COUNT,NRS_RETRY_PASS,"
              "NRS_RETRY_FAIL,NRS_RETRY_HARD_SKIP").upper()
    if common not in compact:
        raise ValueError("Generated numerical source lacks the exact NinjaRS retry counter layout")
    declaration = (f"      COMMON/{prefix}NRS_RETRY_STATS/NRS_RETRY_COUNT,\n"
                   "     $ NRS_RETRY_PASS,NRS_RETRY_FAIL,NRS_RETRY_HARD_SKIP")
    text = template.replace("@RETRY_COMMON@", declaration).replace("@RETRY_RESET@", "")
    if "@FPE_COMMON@" in text:
        common = f"COMMON/{prefix}FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION".upper()
        if common not in compact:
            raise ValueError("Generated numerical source lacks the exact reduction FPE flag layout")
        text = text.replace("@FPE_COMMON@",
            f"      COMMON/{prefix}FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,\n"
            "     $ FPE_IN_QP_REDUCTION")
    for name, code in route_hooks(prefix, routes).items():
        anchor = f"C@COHORT_ROUTE_{name}@"
        if anchor not in text:
            raise ValueError(f"Cohort template lacks route anchor {anchor}")
        text = text.replace(anchor, code)
    return bench.render_worker(text, parameters)


def link_commands(output, directory, env, report=None):
    """Only compile check_sa.f and link check; never execute dependency recipes."""
    # The existing numerical objects are source-bound above. Ignore source
    # timestamps when deriving the diagnostic link; do not rebuild them merely
    # because copied/symlinked resources have newer mtimes.
    frozen_objects = sorted(path.name for path in directory.glob("*.o") if path.name != "check_sa.o")
    command = ["make", "--no-print-directory", "--dry-run",
               *("--old-file=" + name for name in frozen_objects), "check"]
    completed = subprocess.run(command, cwd=directory, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if report is not None:
        (report / "driver-build-preflight.log").write_text(completed.stdout + completed.stderr)
    recipes = []
    for line in completed.stdout.splitlines():
        argv = shlex.split(line)
        if not argv:
            continue
        if any(token in {";", "&&", "||", "|", ">", ">>", "<"} for token in argv):
            raise ValueError("Compound build recipe is not a driver-only build")
        if "-c" in argv:
            sources = [x for x in argv if Path(x).suffix.lower() in (".f", ".f90")]
            if (sources != ["check_sa.f"] or
                    ("-o" in argv and argv[argv.index("-o") + 1] != "check_sa.o")):
                raise ValueError("Refusing a numerical-object build in diagnostic clone")
        elif "-o" in argv and argv[argv.index("-o") + 1] == "check":
            pass
        else:
            raise ValueError("Unexpected diagnostic build recipe: " + line)
        recipes.append(argv)
    if len(recipes) != 2 or "-c" not in recipes[0] or "-c" in recipes[1]:
        raise ValueError("Expected exactly one driver compilation and one executable link")
    return {"make_command": command, "expanded_make_output": completed.stdout, "commands": recipes}


def mutable_paths_are_local(output):
    for path in (output / "SubProcesses").rglob("*"):
        if path.is_file() and (path.name.endswith(("HelFilter.dat", "LoopFilter.dat"))
                               or path.name == "MadLoopParams.dat"):
            if not path.resolve().is_relative_to(output.resolve()):
                raise ValueError("Mutable resource escapes isolated diagnostic directory")
    if not (output / "Cards/MadLoopParams.dat").resolve().is_relative_to(output.resolve()):
        raise ValueError("Mutable numerical card escapes diagnostic directory")


def build_clone(original, output, report, env, routes=False, template=None, renderer=None):
    """Copy an export, render a diagnostic driver into check_sa.f and relink only it.

    ``template``/``renderer`` select another driver (default: the cohort driver).
    """
    template = TEMPLATE if template is None else Path(template)
    renderer = render if renderer is None else renderer
    compilation = json.loads((original / "compilation.json").read_text())
    generation = json.loads((original / "generation.json").read_text())
    require_generation_binding(generation, compilation)
    bench.campaign_guard("cohort-build-before", inputs=generation)
    if not compilation.get("numerical_build_from_sources"):
        raise ValueError("Original numerical objects lack source-build evidence")
    if bench.generated_sources(original) != compilation["sources"]:
        raise ValueError("Original generated sources changed")
    def ignored(directory, names):
        return [name for name in names if name in ("build-history", "worker-raw")
                or stat.S_ISFIFO((Path(directory) / name).lstat().st_mode)]
    shutil.copytree(original, output, symlinks=True, ignore=ignored)
    mutable_paths_are_local(output)
    for archive in (output / "lib").glob("*.a"):
        if not archive.is_symlink():
            archive.unlink()
            archive.symlink_to((original / "lib" / archive.name).resolve())
    subdirs = sorted(output.glob("SubProcesses/P*_*"))
    if len(subdirs) != 1:
        raise ValueError("Expected one diagnostic subprocess")
    directory = subdirs[0]
    for name in ("check_sa.f", "check_sa.o", "check"):
        if not (directory / name).resolve().is_relative_to(output.resolve()):
            raise ValueError("Diagnostic build output escapes isolated directory")
    if bench.digest(directory / "check") != compilation["executable_sha256"]:
        raise ValueError("Original executable changed")
    dependencies = bench.dependency_inputs(directory)
    if dependencies != compilation["link_inputs_before"] or dependencies != compilation["link_inputs_after"]:
        raise ValueError("Original link dependencies changed")
    flags = capture_effective_flags(directory, env=env)
    if not bench.same_compiler_policy(flags, compilation["effective_compiler_flags"]):
        raise ValueError("Original effective compiler policy changed")
    objects = {str(path.relative_to(output)): bench.digest(path)
               for path in output.rglob("*.o") if path.name != "check_sa.o"}
    original_sources = bench.generated_sources(output)
    shutil.copy2(directory / "check_sa.f", report / "original-check_sa.f")
    shutil.copy2(directory / "loop_matrix.f", report / "original-loop_matrix.f")
    driver = renderer(template.read_text(), compilation["worker_render_parameters"],
                      (directory / "loop_matrix.f").read_text(), routes)
    (directory / "check_sa.f").write_text(driver)
    (report / "check_sa.f").write_text(driver)
    local_env = dict(env, PWD=str(directory))
    plan = link_commands(output, directory, local_env, report)
    exclusive_json(report / "driver-build-plan.json", plan)
    with (report / "driver-build.log").open("x") as log:
        for command in plan["commands"]:
            subprocess.run(command, cwd=directory, env=local_env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=180)
    after_objects = {str(path.relative_to(output)): bench.digest(path)
                     for path in output.rglob("*.o") if path.name != "check_sa.o"}
    sources = bench.generated_sources(output)
    driver_key = str((directory / "check_sa.f").relative_to(output))
    expected_sources = dict(original_sources, **{driver_key: bench.digest(directory / "check_sa.f")})
    after_flags = capture_effective_flags(directory, env=env)
    if (objects != after_objects or sources != expected_sources
            or dependencies != bench.dependency_inputs(directory)
            or not bench.same_compiler_policy(flags, after_flags)):
        raise ValueError("Driver-only build changed numerical objects, sources, dependencies or flags")
    record = {**bench.campaign_fields(), "original": compilation, "original_directory": str(original),
              "diagnostic_directory": str(output), "sources": sources,
              "numerical_objects_before": objects, "numerical_objects_after": after_objects,
              "link_inputs_before": dependencies, "link_inputs_after": bench.dependency_inputs(directory),
              "effective_compiler_flags": after_flags, "template_sha256": bench.digest(template),
              "driver_sha256": bench.digest(report / "check_sa.f"), "driver_key": driver_key,
              "executable_sha256": bench.digest(directory / "check")}
    exclusive_json(report / "compilation.json", record)
    return directory, record


def capture(directory, destination, names):
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = directory / name
        if path.is_file():
            with (destination / name).open("xb") as stream:
                stream.write(path.read_bytes())


def measure(worker, report, backend, indices, loops, expected, reference):
    report.mkdir(parents=True, exist_ok=False)
    selection = str(len(indices)) + "\n" + "".join(f"{i}\n" for i in indices)
    (report / "cohort-selection.dat").write_text(selection)
    (worker.directory / "cohort-selection.dat").write_text(selection)
    (worker.directory / "cohort-calls.dat").unlink(missing_ok=True)
    (worker.directory / "cohort-routes.dat").unlink(missing_ok=True)
    exclusive_json(report / "request.json", {"backend": backend, "indices": indices,
                   "loops": loops, "category": expected, "utc": bench.utc_now(),
                   "host_before": bench.host_load_snapshot()})
    try:
        worker.run(loops)
    finally:
        capture(worker.directory, report, ("timing.dat", "values.dat", "timed_status.dat",
                                           "dimensions.dat", "cohort-calls.dat", "cohort-routes.dat"))
    result = read_observation(report, backend, indices, loops, expected)
    result["host_after"] = bench.host_load_snapshot()
    if result["observation"] is not None:
        comparison = bench.compare_rows([row for row in reference if row["point"] in indices],
                                        result["observation"]["rows"])
        result["classification_comparison"] = comparison
        result["pure_cohort"] = result["pure_cohort"] and comparison["passed"]
    calls = result.pop("calls")
    result["timed_calls"] = len(calls)
    result["category_counts"] = dict(Counter(row["category"] for row in calls))
    result["retry_count_totals"] = [sum(row["retries"][i] for row in calls) for i in range(4)]
    result["fpe_flag_counts"] = {key: sum(row[key] for row in calls) for key in ("fpe_dp", "fpe_qp")}
    result["call_trace_sha256"] = bench.digest(report / "cohort-calls.dat")
    exclusive_json(report / "result.json", result)
    return result


def run(args):
    bench.campaign_guard("cohort-before")
    cohorts = cohorts_for(args.backend)
    config = bench.load_config()
    case = next(case for case in config["cases"] if case["id"] == args.case)
    original = bench.output_dir(args.source, case)
    report, workspace = locations(args.label)
    report.mkdir(parents=True, exist_ok=False)
    env = bench.environment()
    protocols = (*PROTOCOLS, "tirust_benchmark_campaign.py") if bench.CAMPAIGN_ID is not None else PROTOCOLS
    specification = {**bench.campaign_fields(), "scope": SCOPE, "performance_acceptance_eligible": False, "trace_schema": 2,
                     "case": case, "source": args.source, "backend": args.backend,
                     "repetitions": args.repetitions, "target_seconds": args.target_seconds,
                     "route_schema": ROUTE_SCHEMA if args.backend == 9 else None,
                     "cpu": args.cpu, "created_utc": bench.utc_now(), "config": config,
                     "machine": bench.machine_identity(args.cpu),
                     "protocols": {name: bench.digest(ROOT / "tools" / name) for name in protocols}}
    for name in protocols:
        shutil.copy2(ROOT / "tools" / name, report / name)
    if bench.CAMPAIGN_ID is not None:
        shutil.copy2(bench.REPORT / "manifest.json", report / "campaign-manifest.json")
        shutil.copy2(bench.REPORT / "cases.json", report / "cases.json")
    exclusive_json(report / "specification.json", specification)
    try:
        with ExitStack() as stack:
            if args.backend in (8, 9):
                lock = stack.enter_context((ROOT / ".tirust-build/numerical-tests.lock").open("a"))
                print("Waiting for numerical-tests lock", flush=True)
                fcntl.flock(lock, fcntl.LOCK_EX)
                print("Acquired numerical-tests lock", flush=True)
            output = workspace / "outputs" / args.source / case["id"]
            directory, compilation = build_clone(original, output, report, env, routes=args.backend == 9)
            settings = configure_clone(workspace, args.source, case, [args.backend], config)
            bench.validate_worker_modes(settings)
            points = bench.REPORT / "points" / (case["id"] + ".dat")
            metadata = json.loads(points.with_suffix(".json").read_text())
            if bench.digest(points) != metadata["sha256"] or metadata["inputs"]["count"] != 100:
                raise ValueError("Expected unchanged original 100-point stream")
            generation = json.loads((output / "generation.json").read_text())
            require_generation_binding(generation, compilation["original"])
            card_hash = bench.digest(output / "Cards/param_card.dat")
            if (generation["case"] != case or generation["source"] != args.source
                    or generation["parameter_card_sha256"] != card_hash
                    or metadata["inputs"]["incoming_masses"] + metadata["inputs"]["masses"]
                    != compilation["original"]["masses"]):
                raise ValueError("Physical model/card/mass identity changed")
            cards = [directory / "param_card.dat"] + [directory.joinpath(*([".."] * i), "Cards/param_card.dat") for i in range(6)]
            if bench.digest(next(path for path in cards if path.is_file())) != card_hash:
                raise ValueError("Fortran resolves a different physical card")
            shutil.copy2(points, report / "points.dat")
            shutil.copy2(points.with_suffix(".json"), report / "points.json")
            shutil.copy2(output / "Cards/param_card.dat", report / "param_card.dat")
            shutil.copy2(output / "Cards/MadLoopParams.dat", report / "MadLoopParams.dat")
            runtime = bench.runtime_identity(directory)
            inputs = {**bench.campaign_fields(), "settings": settings, "generation": generation,
                           "points_sha256": metadata["sha256"], "physical_card_sha256": card_hash,
                           "point_metadata_sha256": bench.digest(report / "points.json"),
                           "numerical_card_sha256": bench.digest(report / "MadLoopParams.dat"),
                           "runtime": runtime}
            bench.campaign_guard("cohort-worker", inputs=inputs)
            exclusive_json(report / "inputs.json", inputs)
            for name in EXTRA:
                (directory / name).unlink(missing_ok=True)
            try:
                worker = Worker(["taskset", "-c", str(args.cpu), "./check", str(points)],
                                directory, report / "worker.log", env=env, timeout=args.timeout)
                stack.enter_context(worker)
            finally:
                capture(directory, report / "initialization", ("warmup.dat", "warmup-flags.dat", "dimensions.dat",
                        "classification-calls.dat", "classification-values.dat", "classification-timing.dat",
                        "classification-routes.dat"))
            frozen = classification(report / "initialization", args.backend, 100, metadata["sha256"])
            exclusive_json(report / "cohorts.json", frozen)
            results = {name: {"status": "empty", "indices": frozen["indices"][name]}
                       for name in cohorts if not frozen["indices"][name]}
            results["failed"] = {"status": "not_timed", "indices": frozen["indices"]["failed"]}
            loops = {}
            for name in cohorts[:-1]:
                indices = frozen["indices"][name]
                if not indices:
                    continue
                calibration = measure(worker, report / name / "calibration-0", args.backend, indices, 1, name, frozen["rows"])
                if not calibration["pure_cohort"]:
                    results[name] = {"status": "migration_or_failure", "indices": indices}
                    continue
                seconds = calibration["observation"]["seconds"]
                nloops = max(1, math.ceil(args.target_seconds * 1.2 / seconds))
                if nloops * len(indices) > 10000000:
                    raise ValueError("Diagnostic trace budget exceeded")
                sustained = measure(worker, report / name / "calibration-1", args.backend, indices, nloops, name, frozen["rows"])
                if not sustained["pure_cohort"]:
                    results[name] = {"status": "migration_or_failure", "indices": indices}
                    continue
                loops[name] = max(nloops, math.ceil(nloops * args.target_seconds * 1.1 / sustained["observation"]["seconds"]))
                if loops[name] * len(indices) > 10000000:
                    raise ValueError("Diagnostic trace budget exceeded")
                results[name] = {"status": "measuring", "indices": indices, "batches": []}
            exclusive_json(report / "timing-schedule.json", {"loops": loops, "repetitions": args.repetitions,
                           "order": "alternate forward/reverse cohort order; cohorts frozen before calibration"})
            for repetition in range(args.repetitions):
                names = list(loops) if repetition % 2 == 0 else list(reversed(loops))
                for name in names:
                    result = measure(worker, report / name / f"batch-{repetition:02d}", args.backend,
                                     frozen["indices"][name], loops[name], name, frozen["rows"])
                    results[name]["batches"].append(result)
            for name in loops:
                batches = results[name]["batches"]
                eligible = all(row["pure_cohort"] and row["observation"]["seconds"] >= .9 * args.target_seconds for row in batches)
                values = [row["observation"]["seconds_per_point"] for row in batches if row["observation"]]
                results[name].update(status="diagnostic" if eligible else "migration_failure_or_short_batch",
                                     eligible_diagnostic=eligible,
                                     median_seconds_per_point=statistics.median(values) if eligible else None)
        after = bench.runtime_identity(directory)
        if (any(runtime[key] != after[key] for key in ("libraries", "archives", "metadata"))
                or bench.digest(directory / "check") != compilation["executable_sha256"]
                or bench.generated_sources(output) != compilation["sources"]
                or bench.digest(output / "Cards/param_card.dat") != card_hash
                or bench.digest(output / "Cards/MadLoopParams.dat") != bench.digest(report / "MadLoopParams.dat")
                or bench.digest(points) != metadata["sha256"]):
            raise ValueError("Source/runtime/physical inputs changed during diagnostic")
        bench.campaign_guard("cohort-after", inputs={**inputs, "runtime": after})
        exclusive_json(report / "results.json", {**bench.campaign_fields(), "scope": SCOPE, "performance_acceptance_eligible": False,
                       "cohorts": results, "runtime_after": after,
                       "manifest_sha256": bench.digest(report / "cohorts.json")})
        validate_report(report)
        print(json.dumps({name: {k: v for k, v in value.items() if k != "batches"}
                          for name, value in results.items()}), flush=True)
    except BaseException as error:
        exclusive_json(report / "failure.json", {**bench.campaign_fields(), "error_type": type(error).__name__, "error": str(error),
                       "performance_acceptance_eligible": False})
        raise


def validate_report(report):
    """Recompute every cohort observation from portable retained evidence."""
    report = Path(report)
    if (report / "failure.json").exists():
        return {"status": "failed", "failure": json.loads((report / "failure.json").read_text())}
    spec = json.loads((report / "specification.json").read_text())
    flags_required = spec.get("trace_schema", 1) == 2
    if spec.get("trace_schema", 1) not in (1, 2):
        raise ValueError("Unknown cohort trace schema")
    inputs = json.loads((report / "inputs.json").read_text())
    compiled = json.loads((report / "compilation.json").read_text())
    original = compiled["original"]
    results = json.loads((report / "results.json").read_text())
    manifest = json.loads((report / "cohorts.json").read_text())
    validate_campaign_evidence(report, spec, inputs, compiled, results)
    if spec["performance_acceptance_eligible"] or results["performance_acceptance_eligible"]:
        raise ValueError("Cohort diagnostics cannot qualify acceptance")
    if any(bench.digest(report / name) != checksum for name, checksum in spec["protocols"].items()):
        raise ValueError("Retained cohort protocol changed")
    driver_key = compiled["driver_key"]
    original_loop = str(Path(driver_key).with_name("loop_matrix.f"))
    expected_driver = render((report / "tirust_benchmark_cohorts.f").read_text(),
                             original["worker_render_parameters"],
                             (report / "original-loop_matrix.f").read_text(),
                             spec.get("route_schema") == ROUTE_SCHEMA and spec["backend"] == 9)
    sources = dict(original["sources"], **{driver_key: compiled["driver_sha256"]})
    if (not original.get("numerical_build_from_sources")
            or bench.digest(report / "original-check_sa.f") != original["sources"][driver_key]
            or bench.digest(report / "original-loop_matrix.f") != original["sources"][original_loop]
            or (report / "check_sa.f").read_text() != expected_driver
            or bench.digest(report / "check_sa.f") != compiled["driver_sha256"]
            or bench.digest(report / "tirust_benchmark_cohorts.f") != compiled["template_sha256"]
            or compiled["sources"] != sources
            or compiled["numerical_objects_before"] != compiled["numerical_objects_after"]
            or compiled["link_inputs_before"] != compiled["link_inputs_after"]
            or compiled["link_inputs_before"] != original["link_inputs_before"]
            or compiled["link_inputs_before"] != original["link_inputs_after"]
            or not bench.same_compiler_policy(compiled["effective_compiler_flags"], original["effective_compiler_flags"])):
        raise ValueError("Diagnostic driver-only compilation provenance changed")
    if any(inputs["runtime"][key] != results["runtime_after"][key] for key in ("libraries", "archives", "metadata")):
        raise ValueError("Diagnostic runtime changed")
    metadata = json.loads((report / "points.json").read_text())
    if (bench.digest(report / "points.dat") != inputs["points_sha256"]
            or bench.digest(report / "points.json") != inputs["point_metadata_sha256"]
            or metadata["sha256"] != inputs["points_sha256"]
            or metadata["inputs"]["count"] != 100
            or metadata["inputs"]["incoming_masses"] + metadata["inputs"]["masses"] != original["masses"]
            or bench.digest(report / "param_card.dat") != inputs["physical_card_sha256"]
            or inputs["generation"]["parameter_card_sha256"] != inputs["physical_card_sha256"]
            or inputs["generation"]["case"] != spec["case"]
            or bench.digest(report / "MadLoopParams.dat") != inputs["numerical_card_sha256"]):
        raise ValueError("Diagnostic physical/numerical input binding changed")
    from madgraph.various.banner import MadLoopParam
    card = MadLoopParam(str(report / "MadLoopParams.dat"))
    settings = {str(key).lower(): value for key, value in card.items()}
    expected = dict(spec["config"]["madloop_parameters"])
    if spec["source"] == "candidate":
        expected.update(spec["config"]["candidate_parameters"])
    if (settings != inputs["settings"] or settings["mlreductionlib"] != str(spec["backend"])
            or any(settings[name.lower()] != value for name, value in expected.items())):
        raise ValueError("Diagnostic policy changed")
    bench.validate_worker_modes(settings)
    recomputed = classification(report / "initialization", spec["backend"], 100, inputs["points_sha256"], flags_required)
    if recomputed != manifest or results["manifest_sha256"] != bench.digest(report / "cohorts.json"):
        raise ValueError("Frozen cohort classification changed")
    schedule = json.loads((report / "timing-schedule.json").read_text())
    if schedule["repetitions"] != spec["repetitions"] or spec["repetitions"] < 7:
        raise ValueError("Cohort repetition count changed")
    verified = {}
    for name in cohorts_for(spec["backend"]):
        indices = manifest["indices"][name]
        if results["cohorts"][name]["indices"] != indices:
            raise ValueError("Cohort membership changed after freezing")
        verified[name] = []
        for batch in sorted((report / name).glob("*")):
            if not batch.is_dir():
                continue
            request = json.loads((batch / "request.json").read_text())
            selection = str(len(indices)) + "\n" + "".join(f"{i}\n" for i in indices)
            if (request["indices"] != indices or request["backend"] != spec["backend"]
                    or request["category"] != name or (batch / "cohort-selection.dat").read_text() != selection):
                raise ValueError("Timed selection differs from frozen cohort")
            actual = read_observation(batch, spec["backend"], indices, request["loops"], name, flags_required)
            stored = json.loads((batch / "result.json").read_text())
            calls = actual.pop("calls")
            actual.update(timed_calls=len(calls), category_counts=dict(Counter(row["category"] for row in calls)),
                          retry_count_totals=[sum(row["retries"][i] for row in calls) for i in range(4)],
                          call_trace_sha256=bench.digest(batch / "cohort-calls.dat"), host_after=stored["host_after"])
            if flags_required:
                actual["fpe_flag_counts"] = {key: sum(row[key] for row in calls) for key in ("fpe_dp", "fpe_qp")}
            if actual["observation"] is not None:
                comparison = bench.compare_rows([row for row in manifest["rows"] if row["point"] in indices],
                                                actual["observation"]["rows"])
                actual["classification_comparison"] = comparison
                actual["pure_cohort"] = actual["pure_cohort"] and comparison["passed"]
            if actual != stored:
                raise ValueError("Retained cohort result differs from raw per-call evidence")
            if batch.name.startswith("batch-"):
                if request["loops"] != schedule["loops"][name]:
                    raise ValueError("Timed loop count differs from registered schedule")
                verified[name].append(actual)
        if name in schedule["loops"]:
            batches = verified[name]
            eligible = (len(batches) == spec["repetitions"] and all(
                b["pure_cohort"] and b["observation"]["seconds"] >= .9 * spec["target_seconds"] for b in batches))
            median = statistics.median(b["observation"]["seconds_per_point"] for b in batches) if eligible else None
            summary = results["cohorts"][name]
            if (summary["batches"] != batches or summary["eligible_diagnostic"] != eligible
                    or summary["median_seconds_per_point"] != median):
                raise ValueError("Cohort timing summary does not match retained observations")
    return {"status": "diagnostic", "acceptance_eligible": False,
            "cohorts": {name: {k: v for k, v in value.items() if k != "batches"}
                        for name, value in results["cohorts"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "validate"))
    parser.add_argument("--case")
    parser.add_argument("--backend", type=int, choices=(1, 6, 7, 8, 9))
    parser.add_argument("--source", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--label", required=True)
    parser.add_argument("--campaign", help="Explicit registered source/native/policy campaign ID")
    parser.add_argument("--cpu", type=int, default=min(bench.os.sched_getaffinity(0)))
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--target-seconds", type=float, default=1.)
    parser.add_argument("--timeout", type=float, default=1800.)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error(f"Run from {ROOT}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,100}", args.label):
        parser.error("Use a new simple lowercase report label")
    if args.repetitions < 7 or not math.isfinite(args.target_seconds) or args.target_seconds < 1:
        parser.error("Use at least seven repetitions and a one-second target")
    if args.cpu not in bench.os.sched_getaffinity(0):
        parser.error("CPU outside available affinity")
    if args.campaign:
        from tirust_benchmark_campaign import activate_campaign
        activate_campaign(bench, args.campaign)
    if args.phase == "validate":
        print(json.dumps(validate_report(locations(args.label)[0])))
    else:
        if not args.case or args.backend is None:
            parser.error("run requires --case and --backend")
        run(args)


if __name__ == "__main__":
    main()
