#!/usr/bin/env python3
"""Plan by default; generate an append-once current selection after real closure."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def pin(root, name):
    raw = module("campaign_report").Evidence(root).add(name)
    return {"index": name, "sha256": hashlib.sha256(raw).hexdigest()}


def generate(root, checkpoint, current_lifecycle=None):
    """Read explicit closed packages only; never inspect supervisor status."""
    report, refresh = module("campaign_report"), module("campaign_report_refresh")
    evidence = report.Evidence(root)
    manifest = evidence.json("campaigns/invariant-symmetry-1/manifest.json",
        "ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0")
    config = evidence.json("campaigns/invariant-symmetry-1/cases.json", manifest["shared_inputs"]["files"]["cases.json"])
    checkpoint_pin = pin(root, checkpoint)
    closed = module("campaign_report_checkpoint").collect(evidence, checkpoint_pin, manifest, config)
    old = evidence.json("campaigns/split-dp-terminal-1/report-snapshots/closed11-provisional-selection-1.json",
        "2443d39de743c6973b1a1e0d8a222a81345f255264212dbd439dd867d7aa4a3f")
    diagnostics = module("campaign_report_diagnostics")
    guarded_build = module("campaign_report_guarded_build")
    guarded_physical = module("campaign_report_guarded_physical")
    selected = {"schema": refresh.CURRENT_SCHEMA, "campaign": manifest["id"],
        "numerical_checkpoint": checkpoint_pin, "closed_cases": closed["summary"]["case_ids"],
        "documents": {}, "packages": dict(refresh.CURRENT_SUPPORT_PACKAGES), "implementation_packages": {
            **old["implementation_packages"], **diagnostics.default_selection(),
            **guarded_build.default_selection(), **guarded_physical.default_selection()},
        "historical_archive": {"index": refresh.ARCHIVE, "sha256": refresh.ARCHIVE_SHA},
        "local_handoff_path": "campaigns/invariant-symmetry-1/timing-handoff-4/LOCAL_TIMING_TODO.md",
        "snapshot_date": "2026-11-10", "performance_status": "deferred to Hua-Sheng",
        "final_performance_accepted": False, "current_lifecycle": current_lifecycle}
    for name in ("PLAN-AMENDMENT-2026-11-10.md", "PRECISION_STREAM_TODO.md",
                 "experiments/cayley-guarded-adaptive-build-proposal-3/README.md"):
        selected["documents"][name] = hashlib.sha256(evidence.add(name)).hexdigest()
    handoff = "campaigns/invariant-symmetry-1/timing-handoff-4/index.json"
    handoff_index = evidence.json(handoff, "f218a9fa65b15f79d6ea0444d31a5c16517d0cbb6341fe3c7c4cde4c43ab295a")
    selected["documents"][handoff] = hashlib.sha256(evidence.add(handoff)).hexdigest()
    for name, identity in handoff_index["files"].items():
        path = name.removeprefix("docs/benchmarkfolder/")
        raw = evidence.add(path, identity["sha256"])
        refresh.require(len(raw) == identity["bytes"], "Handoff input bytes differ")
        selected["documents"][path] = hashlib.sha256(raw).hexdigest()
    refresh.current_selection(evidence, selected, manifest, config)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DOCS)
    parser.add_argument("--checkpoint", help="Explicit docs-relative sealed index; never discovers a run")
    parser.add_argument("--output", help="New docs-relative selection path, opened exclusively")
    parser.add_argument("--current-lifecycle-index", help="Explicit closed successor2 evidence index (optional)")
    parser.add_argument("--current-lifecycle-sha256", help="Exact SHA-256 of that closed lifecycle index")
    args = parser.parse_args()
    if (args.current_lifecycle_index is None) != (args.current_lifecycle_sha256 is None):
        parser.error("Current lifecycle requires both explicit index and SHA-256")
    if args.checkpoint is None:
        if args.output is not None or args.current_lifecycle_index is not None:
            parser.error("--output requires an explicitly closed --checkpoint")
        print(json.dumps({"status": "source-only-no-ingestion", "native_calls": 0,
            "typst_execution": False, "current_results_claimed": False,
            "requires": "observed terminal eleven-case immutable checkpoint; separate rendering admission"}))
        return
    if args.output is None:
        parser.error("--checkpoint requires --output")
    lifecycle = None if args.current_lifecycle_index is None else {
        "index": args.current_lifecycle_index, "sha256": args.current_lifecycle_sha256}
    selected = generate(args.root, args.checkpoint, lifecycle)
    target = module("campaign_report").Evidence(args.root).path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(selected, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "selection-created", "path": str(target), "native_calls": 0}))


if __name__ == "__main__":
    main()
