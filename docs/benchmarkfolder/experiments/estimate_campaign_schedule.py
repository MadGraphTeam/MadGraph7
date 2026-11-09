#!/usr/bin/env python3
"""Schedule estimates from completed historical observations; no new measurements."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench
from tirust_benchmark_worker import parse_outputs

LABELS = {
    "qcd_n1_uux_z": "strict-policy-qcd-n1", "qcd_n2_gg_ttx": "strict-policy-qcd-n2",
    "qcd_n3_uux_zgg": "strict-policy-qcd-n3", "qcd_n4_uux_zzgg": "strict-policy-qcd-n4-warmup-v2",
    "ew_n1_udx_wp": "strict-policy-ew-n1", "ew_n2_epem_mupmum": "strict-policy-ew-n2",
    "ew_n3_epem_ttxa": "strict-policy-ew-n3-warmup-v2", "ew_n4_epem_mupmumaa": "strict-policy-ew-n4-warmup-v2",
    "mixed_uux_ga": "strict-policy-mixed", "li_gg_h": "strict-policy-li-h", "li_gg_hh": "strict-policy-li-hh",
}


def cost(observation, repetitions, target):
    seconds = observation["seconds_per_traversal"]
    loops = max(1, math.ceil(target / seconds))
    repeated = repetitions * loops * seconds
    calibration = seconds + loops * seconds
    warmup = observation["initialization_and_warmup_seconds"]
    return {"timed_full_traversals_per_repetition": loops,
            "timed_seconds": repeated, "calibration_seconds": calibration,
            "initialization_and_warmup_seconds": warmup,
            "total_seconds": repeated + calibration + warmup}


def load_observation(root, label, source, backend, count):
    directory = root / "experiments/numerical" / label / f"{source}-{backend}"
    path = directory / "result.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    if result["status"] != "complete":
        return None
    saved = result["observation"]
    loops, remainder = divmod(sum(saved["timed_status_counts"].values()), count)
    if remainder or not loops:
        raise ValueError("Not a complete fixed-point traversal")
    actual = parse_outputs(directory / "raw", [backend], count, loops)
    if actual != saved:
        raise ValueError("Pilot observation differs from raw evidence")
    inputs = json.loads((directory / (source + "-inputs.json")).read_text())
    if inputs["settings"]["mlstabthres"] != 1e-6:
        raise ValueError("Wrong stability policy for scheduling proxy")
    files = [path, directory / (source + "-inputs.json"), *(directory / "raw").glob("*.dat")]
    return {"source": source, "backend": backend,
            "generation_default_chain": inputs["generation"]["default_chain"],
            "directory": str(directory.relative_to(root)),
            "seconds_per_traversal": saved["seconds"] / loops,
            "initialization_and_warmup_seconds": saved["cold_seconds"],
            "actual_backend_precision_counts": saved["backend_precision_counts"],
            "files": {str(file.relative_to(root)): bench.digest(file) for file in sorted(files)}}


def estimate():
    root = bench.LEGACY_REPORT
    config = bench.load_config()
    repetitions, count = config["repetitions"], config["points_per_case"]
    target = 2 * config["minimum_batch_seconds"]
    cases = []
    for case in config["cases"]:
        available = {(source, backend): load_observation(root, LABELS[case["id"]], source, backend, count)
            for source, backend in [("baseline", 1), *(("candidate", b) for b in config["backends"]) ]}
        pairs = []
        for backend in config["backends"]:
            lanes = []
            for role, observation, assumption in [
                ("baseline", available[("baseline", 1)] if backend in (1, 9) else available[("candidate", backend)],
                 "measured historical baseline1" if backend in (1, 9) else "historical candidate backend used as unmeasured baseline-backend cost proxy"),
                ("candidate", available[("candidate", backend)], "historical c91 candidate used as unmeasured final-candidate cost proxy")]:
                lanes.append({"role": role, "assumption": assumption, "observation": observation,
                              "cost": cost(observation, repetitions, target) if observation else None})
            pairs.append({"label": f"backend-{backend}", "lanes": lanes,
                          "known_cost_subtotal_seconds": sum(lane["cost"]["total_seconds"] for lane in lanes if lane["cost"]),
                          "missing_lanes": [lane["role"] for lane in lanes if lane["cost"] is None]})
        baseline_inputs = available[("baseline", 1)]
        if baseline_inputs is None:
            raise ValueError("Missing source-bound baseline generation record")
        default_chain = bench.parse_chain(baseline_inputs["generation_default_chain"])
        baseline_chain_proxy = available[("candidate", default_chain[0])]
        standalone9 = available[("candidate", 9)]
        chain = {"baseline_chain": default_chain, "candidate_chain": [9, *default_chain],
                 "actual_chain_costs_measured": False,
                 "baseline_proxy": baseline_chain_proxy,
                 "baseline_proxy_cost": cost(baseline_chain_proxy, repetitions, target) if baseline_chain_proxy else None,
                 "candidate_proxy_scenarios": {
                     "cost_of_original_first_backend": cost(baseline_chain_proxy, repetitions, target) if baseline_chain_proxy else None,
                     "cost_of_explicit_tirust": cost(standalone9, repetitions, target) if standalone9 else None},
                 "scenario_scope": "illustrations only, not bounds: the full chain exhausts all DP backends before QP and can differ from either standalone cost"}
        known = sum(pair["known_cost_subtotal_seconds"] for pair in pairs)
        if chain["baseline_proxy_cost"]:
            known += chain["baseline_proxy_cost"]["total_seconds"]
        cases.append({"id": case["id"], "role": case["role"], "pairs": pairs, "chain": chain,
                      "known_cost_subtotal_seconds_excluding_candidate_chain": known,
                      "missing_explicit_lanes": [{"pair": pair["label"], "roles": pair["missing_lanes"]}
                                                 for pair in pairs if pair["missing_lanes"]],
                      "baseline_chain_proxy_missing": baseline_chain_proxy is None})
    totals = {}
    for role in ("main", "control"):
        selected = [case for case in cases if case["role"] == role]
        totals[role] = {"cases": len(selected),
            "known_cost_subtotal_seconds_excluding_candidate_chains": sum(case["known_cost_subtotal_seconds_excluding_candidate_chain"] for case in selected),
            "missing_explicit_lanes": sum(len(pair["missing_lanes"]) for case in selected for pair in case["pairs"]),
            "missing_baseline_chain_proxies": sum(case["baseline_chain_proxy_missing"] for case in selected),
            "candidate_chains_unmeasured": len(selected)}
    return {"schema": "madloop-historical-schedule-estimate-v1", "created_utc": bench.utc_now(),
        "performance_acceptance_eligible": False, "native_measurements_launched": False,
        "points_per_traversal": count, "repetitions": repetitions, "calibration_target_seconds": target,
        "formula": "per lane: cold + one full traversal + sustained calibration traversal_count*s + 7*traversal_count*s; traversal_count=max(1,ceil(2s/s))",
        "scope": "Scheduling estimate from single historical nonquiet pilot batches; not a confidence interval, lower bound, final-candidate measurement or acceptance prediction",
        "exclusions": ["CLI generation/compilation", "external lock queues", "Python input hashing and driver startup/SETPARA", "extra qualification and stress diagnostics", "failures/retries", "unknown heavy lanes", "actual candidate-chain costs"],
        "configuration_sha256": bench.digest(root / "cases.json"), "estimator_sha256": bench.digest(__file__),
        "totals": totals, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if not args.label.replace("-", "").replace("_", "").isalnum():
        parser.error("Use a simple append-only estimate label")
    result = estimate()
    destination = bench.LEGACY_REPORT / "experiments/scheduling" / args.label
    destination.mkdir(parents=True, exist_ok=False)
    bench.dump(destination / "estimate.json", result)
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
