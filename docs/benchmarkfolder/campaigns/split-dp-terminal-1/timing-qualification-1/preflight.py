"""Read-only admission checks; never compile or evaluate an amplitude."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
CAMPAIGN = "split-dp-terminal-1"
MANIFEST_SHA = "1237c563b16a13e2c98327b494a4e5d0b8dec8eb6c7020d632b9b86839143a31"
PREPARATION_SHA = "91e1274318ce90a5f708d4457e57a46197453f39c1bb06f2fb2ed621041385c0"
CPU = 9
MAIN = ("qcd_n1_uux_z", "qcd_n2_gg_ttx", "qcd_n3_uux_zgg", "qcd_n4_uux_zzgg",
        "ew_n1_udx_wp", "ew_n2_epem_mupmum", "ew_n3_epem_ttxa", "ew_n4_epem_mupmumaa")
CONTROLS = ("mixed_uux_ga", "li_gg_h", "li_gg_hh")
LANES = (("baseline", 1), *(("candidate", b) for b in (1, 6, 7, 8, 9)))
PROTOCOLS = ("tirust_benchmark.py", "tirust_benchmark_campaign.py", "tirust_benchmark_compiler.py",
             "tirust_benchmark_numerical.py", "tirust_benchmark_worker.py",
             "tirust_benchmark_worker.f", "tirust_benchmark_warmup.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def command():
    return [sys.executable, "-B", "tools/tirust_benchmark.py", "campaign",
            "--campaign", CAMPAIGN, "--cpu", str(CPU)]


def freeze_check():
    """Check source hashes before importing the frozen benchmark helpers."""
    frozen = read(HERE / "source-freeze.json")
    for name, expected in frozen["files"].items():
        path = ROOT / name
        require(path.is_file() and sha(path) == expected, "Frozen source changed: " + name)
    require(sha(REPORT / "manifest.json") == MANIFEST_SHA, "Campaign manifest changed")
    require(sha(REPORT / "preparation-1/index.json") == PREPARATION_SHA, "Preparation index changed")
    require(sys.flags.optimize == 0, "Qualification requires Python assertions enabled")
    return frozen


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configurations(bench):
    paths = sorted(set((ROOT / "input").glob("*configuration*.txt")) |
                   set((bench.BASELINE / "input").glob("*configuration*.txt")))
    return {str(path.relative_to(ROOT)): {"sha256": sha(path), "text": path.read_text()}
            for path in paths if path.is_file()}


def verify_no_formal_runs():
    runs = REPORT / "runs"
    require(not runs.exists() or not any(runs.iterdir()),
            "Formal run evidence already exists; no automatic restart, retry or overwrite")


def verify_prerequisites():
    """Reparse all 66 ordinary lanes and bind their unchanged live inputs."""
    frozen = freeze_check()
    sys.path.insert(0, str(ROOT / "tools"))
    import tirust_benchmark as bench
    import tirust_benchmark_campaign as campaign
    from tirust_benchmark_compiler import capture_effective_flags
    from tirust_benchmark_warmup import parse_warmup
    require(ROOT == bench.ROOT, "Run from the repository root")
    campaign.activate_campaign(bench, CAMPAIGN)
    campaign.verify_campaign(bench, stage="formal-timing-admission")
    config = bench.load_config()
    require(tuple(c["id"] for c in config["cases"] if c["role"] == "main") == MAIN
            and tuple(c["id"] for c in config["cases"] if c["role"] == "control") == CONTROLS
            and len(config["cases"]) == 11, "The full eight-main/three-control matrix is required")
    require(config["repetitions"] == 7 and config["points_per_case"] == 100
            and config["minimum_batch_seconds"] == 1.0 and config["performance_upper_ratio"] == 1.05,
            "Registered timing policy changed")
    index = read(REPORT / "preparation-1/index.json")
    for name, record in index["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Invalid preparation path")
        path = REPORT / relative
        require(path.is_file() and not path.is_symlink() and sha(path) == record["sha256"]
                and path.stat().st_size == record["bytes"], "Preparation evidence changed: " + name)
    preparation = load_file("timing_preparation_verifier", REPORT / "preparation-1/verify.py")
    live = preparation.inspect_live()
    require(live == read(REPORT / "preparation-1/results.json"), "The 22 prepared builds changed")
    numeric_dir = REPORT / "numerical-qualification-2"
    numeric = load_file("timing_numerical_supervisor", numeric_dir / "run.py")
    final = read(numeric_dir / "status.json")
    spec = read(numeric_dir / "specification.json")
    expected_schedule = numeric.commands(spec["cpu"])
    require(final.get("status") == "passed" and final.get("failed_cases") == []
            and final.get("finished_utc") and final.get("performance_acceptance_eligible") is False
            and final.get("timing_authorized") is False, "All numerical cases must finish successfully first")
    require(final.get("cleanup_verified") is not False, "Numerical supervisor has unresolved cleanup")
    require(spec["manifest_sha256"] == MANIFEST_SHA and spec["campaign"] == CAMPAIGN
            and spec["supervisor_sha256"] == sha(numeric_dir / "run.py")
            and spec["process_cleanup_sha256"] == sha(numeric_dir / "process_cleanup.py")
            and spec["schedule"] == expected_schedule
            and spec["worker_timeouts"] == bench.CAMPAIGN_MANIFEST["administrative_worker_timeout_seconds"],
            "Numerical supervisor specification differs from its frozen protocol")
    require([v["case"] for v in final["cases"]] == [v["case"] for v in expected_schedule],
            "Numerical final status must cover all eleven cases exactly once")
    protocols = {name: sha(ROOT / "tools" / name) for name in PROTOCOLS}
    require(spec["protocols"] == protocols, "Numerical protocol changed before timing")
    builds = numeric.verify_builds(config)
    require(builds == spec["compiled_inputs"] and len(builds) == 22, "Numerical build identities changed")
    configs = configurations(bench)
    require(configs == read(numeric_dir / "configuration-initial.json")
            == read(numeric_dir / "configuration-final.json")
            == read(REPORT / "preparation-1/configuration-initial.json"), "Configuration drift")
    binding = bench.campaign_binding()
    runtimes, lane_summary = {}, []
    for case in config["cases"]:
        case_id = case["id"]
        scheduled = next(item for item in expected_schedule if item["case"] == case_id)
        state = next(item for item in final["cases"] if item["case"] == case_id)
        require(state["status"] == "passed" and state["returncode"] == 0, "Numerical case failed: " + case_id)
        directory = REPORT / "experiments/numerical" / scheduled["label"]
        require(state["results_path"] == str((directory / "results.json").relative_to(REPORT))
                and state["results_sha256"] == sha(directory / "results.json")
                and state["console_sha256"] == sha(numeric_dir / (case_id + ".log")), "Numerical final bindings differ")
        expected_state = numeric.observed_result(scheduled, 0)
        require(all(state.get(k) == v for k, v in expected_state.items()), "Numerical status differs from results")
        run_spec, results = read(directory / "specification.json"), read(directory / "results.json")
        require(run_spec["campaign_binding"] == binding and run_spec["case"] == case
                and run_spec["measurement_phase"] == "steady-warmup"
                and run_spec["worker_timeout_seconds"] == bench.case_worker_timeout(case)
                and run_spec["configuration_sha256"] == sha(REPORT / "cases.json")
                and run_spec["protocols"] == protocols
                and sha(directory / "campaign-manifest.json") == MANIFEST_SHA
                and sha(directory / "cases.json") == sha(REPORT / "cases.json"), "Numerical case specification differs")
        for name, checksum in protocols.items():
            require(sha(directory / name) == checksum, "Retained numerical parser changed: " + name)
        require(results["case"] == case and results["performance_eligible"] is False
                and [(v["source"], v["backend"]) for v in results["results"]] == list(LANES),
                "Expected six explicit numerical lanes")
        reference = None
        for source, backend in LANES:
            lane = directory / f"{source}-{backend}"
            require(not any(lane.rglob("failure.json")), "Failure marker retained in numerical lane")
            value = read(lane / "result.json")
            require(value == next(v for v in results["results"] if (v["source"], v["backend"]) == (source, backend))
                    and value["status"] == "complete" and value["performance_eligible"] is False,
                    "Lane result is incomplete or differs from case results")
            inputs, after = read(lane / (source + "-inputs.json")), read(lane / "inputs-after.json")
            output, process = bench.output_dir(source, case), bench.subprocess_dir(source, case)
            require(inputs["generation"] == read(output / "generation.json")
                    and inputs["compilation"] == read(output / "compilation.json"), "Lane numerical build changed")
            campaign.verify_campaign(bench, stage="formal-timing-numerical-input", inputs=inputs)
            expected = {k.lower(): v for k, v in config["madloop_parameters"].items()}
            if source == "candidate":
                expected.update({k.lower(): v for k, v in config["candidate_parameters"].items()})
            require(inputs["settings"]["mlreductionlib"] == str(backend)
                    and all(inputs["settings"].get(k) == v for k, v in expected.items())
                    and inputs["worker_timeout_seconds"] == bench.case_worker_timeout(case), "Lane policy changed")
            point = REPORT / "points" / (case_id + ".dat")
            require(Path(inputs["points_path"]).resolve() == point.resolve()
                    and inputs["points_sha256"] == sha(point)
                    and inputs["points_metadata_sha256"] == sha(point.with_suffix(".json")), "Lane point binding changed")
            card = bench.resolved_parameter_card(process)
            require(card is not None and sha(card) == inputs["param_card_sha256"]
                    == sha(output / "Cards/param_card.dat"), "Physical parameter card changed")
            cards = inputs["numerical_cards"]
            require("Cards/MadLoopParams.dat" in cards and len(set(cards.values())) == 1
                    and after["numerical_cards"] == cards
                    and after["points_sha256"] == inputs["points_sha256"]
                    and after["executable_sha256"] == inputs["compilation"]["executable_sha256"],
                    "Post-numerical input binding differs")
            key = source + "/" + case_id
            if key not in runtimes:
                actual = bench.runtime_identity(process)
                runtimes[key] = {k: actual[k] for k in ("libraries", "archives", "metadata")}
                flags = capture_effective_flags(process, env=bench.environment())
                require(bench.same_compiler_policy(flags, inputs["compilation"]["effective_compiler_flags"]),
                        "Effective compiler policy changed")
            require(all(inputs["runtime"][k] == after["runtime"][k] == runtimes[key][k]
                        for k in runtimes[key]), "Actual runtime differs from numerical evidence")
            raw = lane / "raw"
            observation = parse_warmup(raw, [backend], 100, read(raw / "ready.json"))
            require(observation == value["observation"] == read(raw / "observation.json"), "Warmup raw reparse differs")
            if source == "baseline":
                reference = observation
            comparison = bench.compare_rows(reference["rows"], observation["rows"], config["relative_tolerance"])
            require(comparison == value["comparison"] and comparison["passed"] is True, "Numerical agreement failed")
            lane_summary.append({"case": case_id, "source": source, "backend": backend,
                                 "points": 100, "passed": True,
                                 "accepted_backend_precision_counts": observation["backend_precision_counts"]})
    # Pin complete retained input trees after successful raw reparse. These are
    # checked again before launch; no final numerical result is inferred early.
    bound = {}
    for tree in (REPORT / "preparation-1", numeric_dir, REPORT / "experiments/numerical"):
        for path in sorted(tree.rglob("*")):
            if path.is_dir():
                continue
            require(path.is_file() and not path.is_symlink() and path.suffix != ".pyc",
                    "Unexpected numerical evidence artifact: " + str(path))
            bound[str(path.relative_to(ROOT))] = sha(path)
    for name, record in index["files"].items():
        bound[str((REPORT / name).relative_to(ROOT))] = record["sha256"]
    campaign.verify_campaign(bench, stage="formal-timing-admission-after")
    freeze_check()
    return {"campaign": CAMPAIGN, "manifest_sha256": MANIFEST_SHA,
            "source_freeze_sha256": sha(HERE / "source-freeze.json"), "command": command(), "cpu": CPU,
            "prepared_builds": builds, "runtime": runtimes, "configuration": configs,
            "protocols": protocols, "numerical_lanes": lane_summary, "retained_inputs": bound,
            "python": {"path": str(Path(sys.executable).resolve()), "sha256": sha(sys.executable)},
            "ordinary_numerical_qualification_passed": True, "overall_goal_qualified": False}
