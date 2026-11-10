#!/usr/bin/env python3
"""Serialize existing CLI qualification for the nine remaining cases."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
ADAPTER = REPORT / "n1-qualification-1/run.py"
ADAPTER_SHA = "2fc766cab10bf4822a0216f566944ad6e4386225b21633b544e3fed7870a8c49"
ORDER = ["li_gg_h", "qcd_n2_gg_ttx", "ew_n2_epem_mupmum", "mixed_uux_ga", "li_gg_hh",
         "qcd_n3_uux_zgg", "ew_n3_epem_ttxa", "qcd_n4_uux_zzgg", "ew_n4_epem_mupmumaa"]
import hashlib
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if digest(ADAPTER) != ADAPTER_SHA:
    raise ValueError("Reviewed n1 adapter changed")
spec = importlib.util.spec_from_file_location("remaining_n1_support", ADAPTER)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
bench, campaign = adapter.bench, adapter.campaign
read, dump, now = adapter.read, adapter.dump, adapter.now


def verify_seal():
    value = read(HERE / "source-index.json")
    for section, base in (("files", HERE), ("inputs", ROOT)):
        for name, checksum in value[section].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or digest(base / relative) != checksum:
                raise ValueError("Continuation source/input seal changed: " + name)
    if digest(REPORT / "manifest.json") != adapter.MANIFEST_SHA:
        raise ValueError("Registered manifest changed")


def cases():
    mapping = {case["id"]: case for case in read(REPORT / "cases.json")["cases"]}
    return [mapping[name] for name in ORDER]


def schedule(case):
    rank = case["nfinal"]
    generate = 900. if rank <= 2 else 1800. if rank == 3 else 3600.
    build = 1200. if rank <= 2 else 3600. if rank == 3 else 7200.
    numerical = 7200. if rank <= 3 else 28800.
    name = case["id"]
    tool = ["taskset", "-c", "12-15", sys.executable, "-B", "tools/tirust_benchmark.py"]
    tail = ["--campaign", adapter.CAMPAIGN, "--cases", name, "--cpu", "12", "--jobs", "4"]
    return [
        {"id": "reuse-baseline", "cap_seconds": 600., "command": [*tool, "reuse-baseline", *tail]},
        {"id": "generate-candidate", "cap_seconds": generate, "command": [*tool, "generate", "--source", "candidate", *tail]},
        {"id": "build-both", "cap_seconds": build, "command": [*tool, "build", "--source", "both", *tail]},
        {"id": "numerical-" + name, "case": name, "label": "full100-warmup-" + name + "-1",
         "cap_seconds": numerical, "command": [sys.executable, "-B", "tools/tirust_benchmark_numerical.py",
             "--campaign", adapter.CAMPAIGN, "--case", name, "--label", "full100-warmup-" + name + "-1",
             "--warmup-only", "--backends", "1", "6", "7", "8", "9", "--cpu", "12"]},
    ]


def preflight():
    verify_seal()
    # Run the pinned common source/model/native/config checks. Its case-specific
    # n1 assumptions are replaced below by the exact inherited case records.
    adapter.CASES = []
    adapter.verify_seal = verify_seal
    common = adapter.readonly_preflight()
    manifest = read(REPORT / "manifest.json")
    baseline = read(REPORT / "inputs/baseline.json")
    parent = ROOT / ".benchmark-work/tirust/campaigns" / manifest["predecessor"]["id"]
    records = {}
    for case in cases():
        name = case["id"]
        timeout = 14400. if case["nfinal"] == 4 else 3600.
        if manifest["administrative_worker_timeout_seconds"][name] != timeout:
            raise ValueError("Registered worker deadline changed: " + name)
        output = parent / "outputs/baseline" / name
        if any(output.rglob("worker-protocol.fifo")):
            raise ValueError("Predecessor baseline is active: " + name)
        generation = read(output / "generation.json")
        identity = generation["source_identity"]
        if (identity["revision"] != baseline["revision"] or identity["archive_sha256"] != baseline["archive_sha256"]
                or identity["files_sha256"] != bench.content_digest(baseline["files"])):
            raise ValueError("Baseline starting source differs: " + name)
        card = manifest["physical_card_origins"]["cards"][name]["param_card_sha256"]
        if generation["parameter_card_sha256"] != card or digest(output / "Cards/param_card.dat") != card:
            raise ValueError("Inherited physical card differs: " + name)
        points = REPORT / "points" / (name + ".dat")
        metadata = read(points.with_suffix(".json"))
        if metadata["inputs"]["count"] != 100 or digest(points) != metadata["sha256"]:
            raise ValueError("Original full100 point stream differs: " + name)
        records[name] = {"generation_sha256": digest(output / "generation.json"),
            "compilation_sha256": digest(output / "compilation.json"), "param_card_sha256": card,
            "default_chain": bench.parse_chain(generation["default_chain"]),
            "worker_deadline_seconds": timeout, "points_sha256": digest(points)}
    common["baseline_records"] = records
    return common


def declared_cards():
    records = {}
    frozen = read(HERE / "preflight.json")["preflight"]["baseline_records"]
    for name in adapter.CASES:
        expected = frozen[name]
        for source in ("baseline", "candidate"):
            output = bench.WORK / "outputs" / source / name
            generation = read(output / "generation.json")
            if (generation.get("campaign_binding") != bench.campaign_binding()
                    or generation["parameter_card_sha256"] != expected["param_card_sha256"]
                    or digest(output / "Cards/param_card.dat") != expected["param_card_sha256"]
                    or bench.parse_chain(generation["default_chain"]) != expected["default_chain"]):
                raise ValueError("Inherited physical card/default chain changed: " + source + "/" + name)
            records[source + "/" + name] = {"param_card_sha256": expected["param_card_sha256"],
                                            "generation_sha256": digest(output / "generation.json")}
    return records


def seal_case(case, state, preparation):
    """Bound append-only publication separately; never add numerical calls."""
    preparation.STAGE_SECONDS, preparation.CLEANUP_RESERVE = 120., 10.
    with preparation.StageDeadline():
        verify_seal()
        name = case["id"]
        directory = HERE / name
        files = [p for p in directory.rglob("*") if p.is_file() and p.name != "checkpoint.json"]
        for source in ("baseline", "candidate"):
            files += [p for p in (REPORT / "inputs" / source / name).rglob("*") if p.is_file()]
        raw = REPORT / "experiments/numerical" / ("full100-warmup-" + name + "-1")
        files += [p for p in raw.rglob("*") if p.is_file()]
        files += [HERE / "source-index.json", HERE / "specification.json", REPORT / "manifest.json",
                  REPORT / "cases.json", REPORT / "points" / (name + ".dat"), REPORT / "points" / (name + ".json")]
        entries = {}
        for path in sorted(set(files)):
            if path.is_symlink() or "__pycache__" in path.parts or path.suffix == ".pyc":
                raise ValueError("Unexpected artifact in closed checkpoint")
            entries[str(path.relative_to(ROOT))] = {"sha256": digest(path), "bytes": path.stat().st_size}
        result = read(directory / ("numerical-" + name) / "numerical-result.json")
        if state["status"] != "passed" or result["status"] != "passed":
            raise ValueError("Only a complete passed case can receive this checkpoint")
        checkpoint = {"schema": "madloop-tirust-one-case-closure-v1", "case": case,
            "campaign": adapter.CAMPAIGN, "manifest_sha256": adapter.MANIFEST_SHA,
            "source_index_sha256": digest(HERE / "source-index.json"), "files": entries,
            "file_count": len(entries), "total_bytes": sum(v["bytes"] for v in entries.values()),
            "numerical_result": result, "cases": [result], "full_matrix_complete": False,
            "performance_acceptance_eligible": False, "native_calls_after_case": 0}
        dump(directory / "checkpoint.json", checkpoint, True)
        return {"path": str((directory / "checkpoint.json").relative_to(ROOT)),
                "sha256": digest(directory / "checkpoint.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    verify_seal()
    if not args.execute:
        print(json.dumps({"status": "dry-run-no-execution", "campaign": adapter.CAMPAIGN,
            "cases": [{"case": c, "stages": schedule(c)} for c in cases()],
            "preflight": preflight() if args.preflight else None, "points_per_lane": 100,
            "lanes_per_case": 6, "performance_acceptance_eligible": False}, indent=2))
        return
    if (HERE / "specification.json").exists():
        raise ValueError("Append-only continuation already exists; no automatic retry")
    readiness = preflight()
    if readiness != read(HERE / "preflight.json")["preflight"]:
        raise ValueError("Current readiness differs from the frozen source-only preflight")
    support = adapter.load_support()
    campaign.activate_campaign(bench, adapter.CAMPAIGN)
    config = bench.load_config()
    initial = adapter.configurations()
    for case in cases():
        for source in ("baseline", "candidate"):
            if (bench.WORK / "outputs" / source / case["id"]).exists():
                raise ValueError("Selected output already exists; no overwrite")
        label = schedule(case)[-1]["label"]
        if (REPORT / "experiments/numerical" / label).exists():
            raise ValueError("Numerical label already exists; no overwrite")
    specification = {"schema": "madloop-tirust-remaining-cli-supervision-v1", "created_utc": now(),
        "campaign": adapter.CAMPAIGN, "manifest_sha256": adapter.MANIFEST_SHA,
        "source_index_sha256": digest(HERE / "source-index.json"), "readiness": readiness,
        "cases": [{"case": c, "stages": schedule(c)} for c in cases()],
        "source_identity": bench.source_identity(ROOT, persist=False), "cleanup_reserve_seconds": 60.,
        "checkpoint_publication_cap_seconds": 120., "performance_acceptance_eligible": False,
        "timing_authorized": False}
    dump(HERE / "specification.json", specification, True)
    dump(HERE / "configuration-initial.json", initial, True)
    state = {"status": "running", "cases": [], "started_utc": now(), "performance_acceptance_eligible": False}
    adapter.declared_cards = declared_cards
    adapter.verify_seal = verify_seal
    try:
        with support[0].SignalRequests() as requests:
            for case in cases():
                requests.check()
                name = case["id"]
                adapter.CASES, adapter.HERE = [name], HERE / name
                adapter.HERE.mkdir(exist_ok=False)
                child_state = {"case": name, "status": "running", "stages": [], "started_utc": now(),
                               "performance_acceptance_eligible": False, "full_matrix_complete": False}
                state["cases"].append(child_state)
                dump(HERE / "status.json", state)
                try:
                    for item in schedule(case):
                        adapter.run_stage(item, config, initial, bench.environment(), requests, child_state, support)
                    requests.check()
                    child_state.update(status="passed", finished_utc=now())
                    dump(adapter.HERE / "status.json", child_state)
                    checkpoint = seal_case(case, child_state, support[1])
                    # The local closed status is immutable after it enters the index.
                    state["cases"][-1] = {**child_state, "checkpoint": checkpoint}
                    print("CLOSED " + name + " checkpoint=" + checkpoint["sha256"], flush=True)
                except BaseException as error:
                    child_state.update(status="failed", error=repr(error), finished_utc=now())
                    dump(adapter.HERE / "status.json", child_state)
                    raise
                finally:
                    dump(HERE / "status.json", state)
            requests.check()
        state.update(status="passed", finished_utc=now())
    except BaseException as error:
        state.update(status="failed", error=repr(error), finished_utc=now())
        raise
    finally:
        dump(HERE / "status.json", state)


if __name__ == "__main__":
    main()
