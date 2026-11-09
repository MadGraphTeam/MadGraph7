#!/usr/bin/env python3
"""Read-only readiness inventory; never register, compile or run a campaign."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench


def record(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": bench.digest(path)}


def collect():
    config = bench.load_config()
    prefix = ROOT / ".tirust-build/current-install/mg/HEPTools/tirust"
    source = bench.source_identity(ROOT, persist=False)
    cases = []
    for case in config["cases"]:
        output = bench.LEGACY_WORK / "outputs/baseline" / case["id"]
        generation = json.loads((output / "generation.json").read_text())
        compilation = json.loads((output / "compilation.json").read_text())
        directories = sorted(output.glob("SubProcesses/P*_*"))
        executable = directories[0] / "check"
        cases.append({"case": case, "administrative_worker_timeout_seconds":
                      14400 if case["nfinal"] == 4 else 3600,
                      "generation": record(output / "generation.json"),
                      "compilation": record(output / "compilation.json"),
                      "baseline_revision": generation["source_identity"]["revision"],
                      "materialized_baseline_chain": generation["default_chain"],
                      "proposed_candidate_chain": "9|" + generation["default_chain"],
                      "numerical_build_from_sources": compilation.get("numerical_build_from_sources"),
                      "source_inventory_matches": bench.generated_sources(output) == compilation["sources"],
                      "executable": record(executable),
                      "executable_matches_compilation": bench.digest(executable) == compilation["executable_sha256"],
                      "effective_compiler_flags_recorded": bool(compilation.get("effective_compiler_flags")),
                      "recorded_link_snapshots_equal": bool(compilation.get("link_inputs_before"))
                          and compilation["link_inputs_before"] == compilation.get("link_inputs_after"),
                      "active_worker_fifo": bool(list(output.rglob("worker-protocol.fifo"))),
                      "points": record(bench.LEGACY_REPORT / "points" / (case["id"] + ".dat")),
                      "reuse_status": "pending per-case current dependency/flag checks, untouched deterministic relink, private copy and worker-only relink"})
    names = ["lib/libmadloop_tirust.so", "lib/libtirust.a", "lib/tirust-config.json",
             "include/mtirust.f90", "Cargo.lock"]
    provenance = bench.LEGACY_REPORT / "qualification/tirust-install/current-source"
    installation = json.loads((provenance / "check-installation.json").read_text())
    proof = bench.LEGACY_REPORT / "experiments/baseline-object-replay/qcd-n1"
    model = json.loads((bench.LEGACY_REPORT / "inputs/loop_qcd_qed_sm_Gmu.json").read_text())
    return {"schema": "madloop-tirust-campaign-readiness-plan-v1", "created_utc": bench.utc_now(),
            "scope": "Planning snapshot only; no campaign registration, compilation, relink or numerical execution",
            "proposed_campaign_id": "split-dp-terminal-1", "registration_authorized": False,
            "source_freeze_pending": True, "observed_source_identity": source,
            "production_commit": subprocess.check_output(["git", "rev-parse", "8ae62e25d8"], text=True).strip(),
            "timeout_protocol_commit": subprocess.check_output(["git", "rev-parse", "f2cdf78566"], text=True).strip(),
            "configuration": record(bench.LEGACY_REPORT / "cases.json"),
            "model_pin": record(bench.LEGACY_REPORT / "inputs/loop_qcd_qed_sm_Gmu.json"),
            "model_archive_sha256": model["archive_sha256"],
            "native_prefix": str(prefix), "native_files": [record(prefix / name) for name in names],
            "installation_qualification": record(provenance / "check-installation.json"),
            "installation_source_revision": installation["source_revision"],
            "native_replay": installation["final_native_replay"],
            "existing_qcd_n1_relink_proof": record(proof / "proof.json"),
            "existing_qcd_n1_relink_qualification": record(proof / "qualification.json"),
            "cases": cases,
            "excluded_unpromoted_proposals": ["Born cancellation fallback", "512-bit bridge", "higher-rank kernel alternatives"],
            "protocol": {"saved_points_per_case": config["points_per_case"],
                         "paired_repetitions": config["repetitions"],
                         "minimum_batch_seconds": config["minimum_batch_seconds"],
                         "sustained_calibration_target_seconds": 2.0,
                         "performance_upper_95_ratio_limit": config["performance_upper_ratio"],
                         "full_point_traversal_required": True,
                         "numerical_preparation_required_before_timing": True,
                         "main_chain_gates_before_explicit_regressions": True},
            "inventory_program": record(Path(__file__).resolve())}


if __name__ == "__main__":
    if ROOT != bench.ROOT:
        raise SystemExit(f"Run from {bench.ROOT}")
    print(json.dumps(collect(), sort_keys=True, indent=2, allow_nan=False))
