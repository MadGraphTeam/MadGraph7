#!/usr/bin/env python3
"""Read-only proposed higher-rank campaign inventory; no registration or jobs."""
import importlib.util
import json
from pathlib import Path

PROGRAM = Path(__file__).resolve()
PARENT = PROGRAM.parent.parent / "split-dp-terminal-plan-1/inventory.py"
specification = importlib.util.spec_from_file_location("preceding_readiness_inventory", PARENT)
parent = importlib.util.module_from_spec(specification)
specification.loader.exec_module(parent)
from tirust_benchmark_campaign import export_environment_binding


def collect():
    result = parent.collect()
    result["production_basis_commit"] = result.pop("production_commit")
    result["production_commit"] = None
    result["production_commit_status"] = "Pending coordinator source freeze after deterministic full-amplitude qualification"
    result["intended_terminal_rank_pairs"] = [[1, 1], [2, 1], [3, 1]]
    result["proposed_campaign_id"] = "split-dp-terminal-rank3-1"
    result["export_environment_binding"] = export_environment_binding()
    result["excluded_unpromoted_proposals"] = ["Born cancellation fallback", "512-bit bridge",
                                             "Any terminal rank pair other than (1,1), (2,1), (3,1)"]
    result["parent_inventory_program"] = result["inventory_program"]
    result["inventory_program"] = parent.record(PROGRAM)
    for case in result["cases"]:
        path = parent.ROOT / case["generation"]["path"]
        generation = json.loads(path.read_text())
        case["historical_export_environment_recorded"] = "export_environment" in generation
        case["historical_export_environment"] = generation.get("export_environment")
        case["proposed_reuse_environment_binding"] = export_environment_binding()
        case["source_generation_repeated_on_reuse"] = False
    return result


if __name__ == "__main__":
    if Path.cwd() != parent.bench.ROOT:
        raise SystemExit(f"Run from {parent.bench.ROOT}")
    print(json.dumps(collect(), sort_keys=True, indent=2, allow_nan=False))
