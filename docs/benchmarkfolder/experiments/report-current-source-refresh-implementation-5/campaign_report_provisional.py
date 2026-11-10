"""Opt-in, sealed shared-host evidence. Never discover or execute live work."""
import math
from pathlib import PurePosixPath
import runpy
import statistics

SCHEMA = "madloop-tirust-closed-provisional-report-v1"
RUNTIME_SCHEMA = "madloop-tirust-provisional-full100-v1"
BACKENDS = (1, 6, 7, 8, 9)
UNAVAILABLE = {"unavailable_deadline", "unavailable_or_numerical_failure",
               "numerical_failure", "not_run_budget", "not_completed_budget"}
FORBIDDEN = {"upper_95", "lower_95", "threshold", "acceptance_eligible", "passed"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def safe(name):
    require(isinstance(name, str) and name and "\\" not in name,
            "Invalid closed provisional path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "Unsafe closed provisional path")
    return name


def collect(evidence, selected, report, replay=None, reader=None):
    """Optional replay/reader arguments are synthetic-test seams, never CLI flags.

    The selected index pins a reviewed portable wrapper whose no-argument
    replay_completed() returns exact completed-result dictionaries keyed by ID.
    All package bytes are verified before the frozen wrapper is imported.
    """
    name, checksum = selected["index"], selected["sha256"]
    require(isinstance(checksum, str) and len(checksum) == 64
            and all(c in "0123456789abcdef" for c in checksum), "Missing provisional index pin")
    index = evidence.json(name, checksum)
    require(index.get("schema") == SCHEMA and index.get("closed") is True
            and index.get("performance_acceptance_eligible") is False,
            "Provisional package is not sealed descriptive evidence")
    require(index.get("campaign_binding") == report["campaign_binding"],
            "Provisional campaign differs")
    prefix = str(PurePosixPath(name).parent)
    files = index.get("files")
    require(isinstance(files, dict) and files, "Empty provisional allowlist")
    for relative, identity in files.items():
        raw = evidence.add(prefix + "/" + safe(relative), identity["sha256"])
        require(type(identity.get("bytes")) is int and len(raw) == identity["bytes"],
                "Provisional byte count differs")
    wrapper = safe(index["replay"]["path"])
    require(wrapper in files and index["replay"].get("function") == "replay_completed",
            "Unbound portable replay wrapper")
    retained = index.get("retained_files", files)
    require(isinstance(retained, dict) and retained, "Empty retained provisional inventory")
    for relative in retained:
        safe(relative)
    compact = "retained_files" in index
    if replay is None or (compact and reader is None):
        module = runpy.run_path(str(evidence.path(prefix + "/" + wrapper)))
        if replay is None:
            replay = module["replay_completed"]
        if compact and reader is None:
            if "_load" in module:
                # This reviewed frozen loader verifies every stored and logical
                # byte once. Keep replay_completed itself unchanged: it performs
                # its own independent load and portable protocol reparse below.
                loaded_index, retained_bytes = module["_load"](expected_index_sha256=checksum)
                require(loaded_index == index, "Loaded provisional index differs")
                require(set(retained_bytes) == set(retained), "Loaded retained inventory differs")
                reader = lambda relative: module["parsed"](retained_bytes[relative])
            else:
                reader = module["read_retained_json"]

    def read(relative):
        require(safe(relative) in retained, "Provisional dependency is not indexed: " + relative)
        if compact:
            return reader(relative)
        return evidence.json(prefix + "/" + relative, files[relative]["sha256"])

    paths = index["paths"]
    spec, closure, manifest = (read(paths[k]) for k in ("specification", "closure", "manifest"))
    require(retained[paths["manifest"]]["sha256"] == report["campaign_binding"]["manifest_sha256"],
            "Sealed campaign manifest differs")
    require(spec.get("schema") == RUNTIME_SCHEMA and spec.get("campaign_binding") == report["campaign_binding"]
            and spec.get("pairs") == 3 and spec.get("points") == 100
            and spec.get("point_subset") is None and spec.get("cpu_reservation") is False
            and spec.get("total_cap_seconds") == 1800
            and spec.get("performance_acceptance_eligible") is False,
            "Provisional protocol scope differs")
    require(closure.get("schema") == RUNTIME_SCHEMA and closure.get("status") in ("closed", "closed_budget")
            and closure.get("performance_acceptance_eligible") is False
            and closure.get("precise_acceptance") == "deferred_to_local"
            and closure.get("budget_seconds") == 1800,
            "Provisional controller is not closed")
    require(nonnegative(closure.get("elapsed_seconds")) and type(closure.get("budget_exceeded")) is bool
            and (closure["elapsed_seconds"] <= 1800 or closure["budget_exceeded"]),
            "Invalid or undisclosed provisional elapsed accounting")
    candidate = manifest["candidate"]
    require(candidate["exporter_identity"]["files_sha256"] == report["registered_source_sha256"]
            and candidate["numerical_identity_sha256"] == report["numerical_identity_sha256"]
            and candidate["exporter_identity"]["revision"] == report["registered_exporter_revision"],
            "Provisional candidate is not the frozen report candidate")
    schedule = spec["schedule"]
    cases = {row["id"] for row in report["numerical"]["cases"]}
    require(len(cases) == 11 and len(schedule) == 66
            and len({row["id"] for row in schedule}) == 66, "Incomplete provisional schedule")
    require({row["case"] for row in schedule[:11]} == cases
            and all(row["kind"] == "chain" and row["backend"] is None and row["cap_seconds"] == 120
                    and row["chains"]["candidate"] == [9, *row["chains"]["baseline"]]
                    for row in schedule[:11]), "Provisional chain schedule differs")
    require({(row["case"], row["backend"]) for row in schedule[11:]} ==
            {(case, backend) for case in cases for backend in BACKENDS}
            and all(row["kind"] == "standalone" and row["cap_seconds"] == 180
                    and row["chains"] == {"baseline": [1 if row["backend"] == 9 else row["backend"]],
                                          "candidate": [row["backend"]]} for row in schedule[11:]),
            "Provisional standalone schedule differs")
    rows = closure["rows"]
    require(len(rows) == 66 and [r["comparison"] for r in rows] == schedule,
            "Closed rows omit, duplicate or reorder the schedule")
    require(all(row.get("performance_acceptance_eligible") is False
                and row["status"] in UNAVAILABLE | {"complete"} for row in rows),
            "Unknown provisional completion state")
    bindings = index["comparisons"]
    require(set(bindings) == {row["id"] for row in schedule}, "Comparison allowlist differs")
    completed = {}
    for row in rows:
        key = row["comparison"]["id"]
        binding = bindings[key]
        elapsed = row.get("elapsed_seconds_including_cleanup")
        require(elapsed is None or nonnegative(elapsed), "Invalid comparison elapsed accounting")
        if binding.get("completion") is not None:
            require(read(binding["completion"]) == row, "Closure/completion disagreement")
        else:
            require(row["status"] in {"not_run_budget", "not_completed_budget"},
                    "Attempted comparison lacks completion")
        if row["status"] == "complete":
            require(positive(elapsed) and elapsed <= row["comparison"]["cap_seconds"],
                    "Complete comparison exceeded its all-in cap")
            require(binding.get("completion") is not None, "Complete comparison lacks completion")
            result = read(binding["result"])
            require(result.get("status") == "complete" and result.get("numerical_pass") is True
                    and result.get("performance_acceptance_eligible") is False
                    and result.get("comparison") == row["comparison"], "Complete result scope differs")
            completed[key] = result
    reparsed = replay()
    require(reparsed == completed, "Frozen completed-result replay differs")

    displayed = []
    for row in rows:
        comparison = row["comparison"]
        view = {**comparison, "status": row["status"], "numerical_pass": None,
                "ratio": None, "ratio_range": None, "paired_ratios": None,
                "median_seconds_per_point": None, "batch_seconds": None,
                "initialization_seconds": None, "accepted_usage": None,
                "elapsed_seconds_including_cleanup": row.get("elapsed_seconds_including_cleanup")}
        if row["status"] == "complete":
            result = completed[comparison["id"]]
            stats = result["statistics"]
            require(not FORBIDDEN.intersection(stats) and stats.get("performance_acceptance_eligible") is False
                    and stats.get("paired_repetitions") == 3, "Provisional confidence gate or pair count")
            timings = result["timings"]
            require(set(timings) == {"baseline", "candidate"}
                    and all(len(v) == 3 and all(positive(x) for x in v) for v in timings.values()),
                    "Invalid provisional timing samples")
            ratios = [right / left for left, right in zip(timings["baseline"], timings["candidate"])]
            ratio = math.exp(sum(map(math.log, ratios)) / 3)
            require(stats["paired_ratios"] == ratios and stats["geometric_mean_paired_ratio"] == ratio
                    and stats["ratio_range"] == [min(ratios), max(ratios)]
                    and stats["median_seconds_per_point"] == {k: statistics.median(v) for k, v in timings.items()},
                    "Provisional descriptive statistics differ")
            durations, initialization = {}, {}
            for source in ("baseline", "candidate"):
                ready = read(bindings[comparison["id"]]["ready"][source])
                cold = result["initialization_and_warmup_seconds"][source]
                require(positive(cold) and cold == ready["cold_seconds"], "Initialization timing differs")
                initialization[source] = cold
                observations = result["observations"][source]
                loops = result["calibration_loops"][source]
                require(type(loops) is int and loops > 0 and len(observations) == 3,
                        "Provisional full100 loops differ")
                durations[source] = [value["seconds"] for value in observations]
                require(all(positive(value) for value in durations[source])
                        and [value / (100 * loops) for value in durations[source]] == timings[source],
                        "Provisional batch durations differ")
                usage = result["actual_backend_precision_counts"][source]
                require(all(type(count) is int and count > 0 for count in usage.values())
                        and sum(usage.values()) == 300 * loops, "Accepted usage count differs")
            view.update(numerical_pass=True, ratio=ratio, ratio_range=stats["ratio_range"],
                        paired_ratios=ratios, median_seconds_per_point=stats["median_seconds_per_point"],
                        batch_seconds=durations, initialization_seconds=initialization,
                        accepted_usage=result["actual_backend_precision_counts"])
        displayed.append(view)
    return {"index_path": name, "index_sha256": checksum, "scope": "Descriptive shared-host observations only",
            "performance_acceptance_eligible": False, "precise_acceptance": "deferred to Hua-Sheng",
            "pairs": 3, "points": 100, "rows": displayed,
            "closed_status": closure["status"], "publication_scope": closure.get("publication_scope"),
            "elapsed_seconds": closure.get("elapsed_seconds"), "budget_seconds": 1800,
            "budget_exceeded": closure.get("budget_exceeded"),
            "candidate_source_sha256": report["registered_source_sha256"],
            "candidate_numerical_identity_sha256": report["numerical_identity_sha256"],
            "candidate_exporter_revision": report["registered_exporter_revision"],
            "baseline_revision": manifest["baseline"]["revision"],
            "accepted_usage_scope": "Final accepted amplitudes; internal attempts and IEEE flags unobserved"}
