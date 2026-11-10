#!/usr/bin/env python3
"""Read-only final-implementation inventory; never register or execute a campaign."""
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench
from tirust_benchmark_campaign import export_environment_binding, _native

PROGRAM = Path(__file__).resolve()
HERE = PROGRAM.parent
PRODUCTION = "059bb9c979194bfda783cad83443223def8df280"
PREFIX = ROOT / ".tirust-build/qp-transport-install/mg/HEPTools/tirust"
PREFLIGHT = bench.LEGACY_REPORT / "experiments/baseline-object-replay/all11-preflight-1"
EXPECTED_PREFLIGHT_INDEX = "f839e57f8d66a84c2060d38d4876717f221e2debefacfb434c46443a9dbd561c"
EXPECTED_NATIVE = {
    "lib/libmadloop_tirust.so": "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9",
    "lib/libtirust.a": "0c86349a4fd0dadd918476731e899529ce1c7b17f5892a4af2c22dd762250c1a",
    "lib/tirust-config.json": "25182cfb31e68f8ff6246717e6819f0ec86232de88742f6e47e73676e6a41af9",
}


def record(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": bench.digest(path)}


def read(path):
    return json.loads(Path(path).read_text())


def collect():
    config = bench.load_config()
    source = bench.source_identity(ROOT, persist=False)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", PRODUCTION, "HEAD", "--", *bench.GENERATOR_PATHS],
        cwd=ROOT, text=True).splitlines()
    if source["dirty"] or changed:
        raise ValueError("Observed exporter differs from the frozen production implementation")
    native = _native(bench, PREFIX)
    for name, expected in EXPECTED_NATIVE.items():
        if native["files"].get(name) != expected:
            raise ValueError(f"Unexpected corrected native artifact: {name}")
    if bench.digest(PREFLIGHT / "index.json") != EXPECTED_PREFLIGHT_INDEX:
        raise ValueError("The independently audited baseline preflight index changed")
    preflight_index = read(PREFLIGHT / "index.json")
    for name, identity in preflight_index["files"].items():
        expected = identity["sha256"] if isinstance(identity, dict) else identity
        if bench.digest(PREFLIGHT / name) != expected:
            raise ValueError(f"Baseline preflight evidence changed: {name}")
    proofs = {item["case"]["id"]: item for item in read(PREFLIGHT / "results.json")["results"]}
    cases = []
    for case in config["cases"]:
        output = bench.LEGACY_WORK / "outputs/baseline" / case["id"]
        generation = read(output / "generation.json")
        compilation = read(output / "compilation.json")
        directories = sorted(output.glob("SubProcesses/P*_*"))
        if len(directories) != 1:
            raise ValueError(f"Expected one compiled subprocess for {case['id']}")
        executable = directories[0] / "check"
        proof = proofs[case["id"]]
        if proof["status"] != "passed" or not proof["original_inputs_unchanged"]:
            raise ValueError(f"Baseline preflight did not pass: {case['id']}")
        source_matches = bench.generated_sources(output) == compilation["sources"]
        executable_matches = bench.digest(executable) == compilation["executable_sha256"]
        if not source_matches or not executable_matches:
            raise ValueError(f"Baseline source/executable changed: {case['id']}")
        if bench.digest(executable) != proof["relinked_executable_sha256"]:
            raise ValueError(f"Baseline executable differs from the audited proof: {case['id']}")
        cases.append({
            "case": case,
            "administrative_worker_timeout_seconds": 14400 if case["nfinal"] == 4 else 3600,
            "generation": record(output / "generation.json"),
            "compilation": record(output / "compilation.json"),
            "baseline_revision": generation["source_identity"]["revision"],
            "materialized_baseline_chain": generation["default_chain"],
            "proposed_candidate_chain": "9|" + generation["default_chain"],
            "historical_export_environment_recorded": "export_environment" in generation,
            "historical_export_environment": generation.get("export_environment"),
            "reuse_environment_binding": export_environment_binding(),
            "source_generation_repeated_on_reuse": False,
            "numerical_build_from_sources": compilation.get("numerical_build_from_sources"),
            "source_inventory_matches": source_matches,
            "executable": record(executable),
            "executable_matches_compilation": executable_matches,
            "effective_compiler_flags_recorded": bool(compilation.get("effective_compiler_flags")),
            "recorded_link_snapshots_equal": bool(compilation.get("link_inputs_before"))
                and compilation["link_inputs_before"] == compilation.get("link_inputs_after"),
            "active_worker_fifo": bool(list(output.rglob("worker-protocol.fifo"))),
            "points": record(bench.LEGACY_REPORT / "points" / (case["id"] + ".dat")),
            "point_metadata": record(bench.LEGACY_REPORT / "points" / (case["id"] + ".json")),
            "untouched_baseline_relink_preflight": record(PREFLIGHT / case["id"] / "result.json"),
            "audited_preflight_original_snapshot": record(PREFLIGHT / case["id"] / "inputs-after.json"),
            "reuse_status": "preflight passed; final reuse repeats current dependency/flag checks, untouched-driver proof, private copy and worker-only relink",
        })
    model = read(bench.LEGACY_REPORT / "inputs/loop_qcd_qed_sm_Gmu.json")
    qualification = read(HERE / "observed-installation.json")
    if qualification["source_revision"] != PRODUCTION or qualification["prefix"] != str(PREFIX):
        raise ValueError("Observed installation does not bind the final implementation/prefix")
    if qualification["metadata_sha256"] != EXPECTED_NATIVE["lib/tirust-config.json"]:
        raise ValueError("Observed installation metadata does not bind the actual package")
    return {
        "schema": "madloop-tirust-final-implementation-readiness-v1",
        "created_utc": bench.utc_now(),
        "scope": "Read-only readiness snapshot; no campaign registration, source compilation, relink or numerical execution",
        "proposed_campaign_id": "split-dp-terminal-1",
        "proposed_reason": "Split primary/check TIRust DP workspaces, guarded DP terminal rank pairs 1x1/2x1/3x1, and invalidate caches after failed Fortran QP transport",
        "registration_authorized": False,
        "coordinator_clearance_pending": True,
        "production_commit": PRODUCTION,
        "observed_source_identity": source,
        "exporter_changes_since_production_commit": changed,
        "export_environment_binding": export_environment_binding(),
        "configuration": record(bench.LEGACY_REPORT / "cases.json"),
        "model_pin": record(bench.LEGACY_REPORT / "inputs/loop_qcd_qed_sm_Gmu.json"),
        "model_archive_sha256": model["archive_sha256"],
        "native": native,
        "observed_installation": record(HERE / "observed-installation.json"),
        "observed_installation_state": qualification["state"],
        "qualification_status": "pending final installer evidence and independent audit; publication/smoke alone do not authorize registration",
        "baseline_preflight_index": record(PREFLIGHT / "index.json"),
        "baseline_preflight_case_count": len(proofs),
        "baseline_preflight_reuse_imported": False,
        "cases": cases,
        "intended_terminal_rank_pairs": [[1, 1], [2, 1], [3, 1]],
        "excluded_unpromoted_proposals": ["Born cancellation fallback", "512-bit bridge",
            "QP scalar-shadow profiles", "Any terminal rank pair other than (1,1), (2,1), (3,1)"],
        "protocol": {
            "saved_points_per_case": config["points_per_case"],
            "paired_repetitions": config["repetitions"],
            "minimum_batch_seconds": config["minimum_batch_seconds"],
            "sustained_calibration_target_seconds": 2.0,
            "performance_upper_95_ratio_limit": config["performance_upper_ratio"],
            "full_point_traversal_required": True,
            "baseline_numerical_backends": [1],
            "candidate_numerical_backends": [1, 6, 7, 8, 9],
            "numerical_cases": len(cases),
            "candidate_fresh_cli_source_builds": len(cases),
            "numerical_preparation_required_before_timing": True,
            "main_chain_gates_before_explicit_regressions": True,
            "main_chain_gate_count": sum(case["role"] == "main" for case in config["cases"]),
            "defer_expensive_regressions_if_any_main_chain_fails": True,
        },
        "scheduling_estimate": record(bench.LEGACY_REPORT / "experiments/scheduling/historical-complete-pilots-2/estimate.json"),
        "inventory_program": record(PROGRAM),
        "protocol_programs": [record(ROOT / "tools" / name) for name in (
            "tirust_benchmark.py", "tirust_benchmark_campaign.py", "tirust_benchmark_compiler.py",
            "tirust_benchmark_relink.py", "tirust_benchmark_numerical.py", "tirust_benchmark_worker.f")],
    }


if __name__ == "__main__":
    if ROOT != bench.ROOT:
        raise SystemExit(f"Run from {bench.ROOT}")
    print(json.dumps(collect(), sort_keys=True, indent=2, allow_nan=False))
