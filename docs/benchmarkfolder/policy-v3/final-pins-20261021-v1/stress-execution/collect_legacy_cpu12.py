#!/usr/bin/env python3
"""Separate held legacy-only stress controls on CPU12; no NinjaRS dispatch.

Use the retained collector and workers. Full fresh-export validation precedes
an explicit requested-case subset. Existing four-backend readers stay intact.
"""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess

import collect_final as adapter

raw, policy, supplement = adapter.raw, adapter.policy, adapter.supplement
CPU = 12
EW = "stress-ew-4-uu-4l"
QCD = "stress-qcd-4-gg-ttgg"
ORIGINAL_CONFIGURATION = raw.stress.configuration


def selection(variant):
    if variant == "ordinary":
        return {"cpu": CPU, "cases": [EW], "backends": [1, 6, 7]}
    if variant not in ("rotation", "scale"):
        raise ValueError("Unknown fixed legacy control")
    return {"cpu": CPU, "cases": [QCD + "-" + variant, EW + "-" + variant], "backends": [6]}


def scoped_configuration(variant):
    result = copy.deepcopy(ORIGINAL_CONFIGURATION(variant))
    scope = selection(variant)
    result["cases"] = [case for case in result["cases"] if case["id"] in scope["cases"]]
    if [case["id"] for case in result["cases"]] != scope["cases"]:
        raise ValueError("Requested legacy control is missing from the fixed plan")
    return result


def selected_spec(base, bound_base):
    return policy.seal({**base, "retained_base_specification": bound_base,
        "explicit_legacy_only_scope": selection(base["variant"]),
        "complete_source_configuration": ORIGINAL_CONFIGURATION(base["variant"]),
        "scope_reason": "Missing independent controls; accepted historical QCD ordinary calibration references remain separate",
        "rust_or_symbolica_dispatch_permitted": False,
        "four_backend_stress_dataset_claim": False})


def validate_selection(spec):
    policy.verified(spec, raw.SCHEMA)
    base = policy.verified(adapter.final.read(supplement.checked(
        spec["retained_base_specification"])), raw.SCHEMA)
    if spec != selected_spec(base, spec["retained_base_specification"]):
        raise ValueError("Legacy-only stress scope differs from its immutable base specification")
    scope = selection(spec["variant"])
    if (spec["configuration"] != scoped_configuration(spec["variant"])
            or spec["cpu"] != CPU or 8 in scope["backends"]):
        raise ValueError("Legacy-only CPU/case/backend scope is inconsistent")
    return scope


def stable_orchestration(spec, record=True):
    """Allow unrelated commits while requiring the exact imported/numerical bytes."""
    if (spec["numerical_revision"] != adapter.final.REVISION
            or spec["generation_revision"] != adapter.final.REVISION
            or spec["generator_identity_sha256"] != adapter.final.GENERATOR
            or spec["installed_archive_sha256"] != adapter.final.ARCHIVE):
        raise ValueError("Legacy stress numerical cohort differs from the final pinned source")
    for source in spec["sources"].values():
        supplement.checked(source)
    adapter.numerical_sources(spec["numerical_revision"])
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=adapter.ROOT,
                                     text=True).strip()
    if record:
        path = Path(__file__).parent / "legacy-source-head-observations" / (spec["identity_sha256"] + ".jsonl")
        path.parent.mkdir(exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "specification_orchestration_head": spec["operational_source_state"]["actual_git_head"],
                "observed_orchestration_head": actual,
                "numerical_revision": adapter.final.REVISION,
                "generator_identity_sha256": adapter.final.GENERATOR,
                "exact_GEN_vendor_and_bound_helper_bytes_verified": True}) + "\n")
    return actual


def configure(variant):
    adapter.configure_adapters()
    raw.CPU = CPU
    raw.BACKENDS = tuple(selection(variant)["backends"])
    raw.check_orchestration_head = stable_orchestration
    original_cohort = raw.cohort

    def cohort(filename):
        # Validate both complete CLI exports and the full retained input plan;
        # requested-case selection happens only for later worker dispatch.
        current = raw.stress.configuration
        raw.stress.configuration = ORIGINAL_CONFIGURATION
        try:
            return original_cohort(filename)
        finally:
            raw.stress.configuration = current

    raw.cohort = cohort
    raw.stress.configuration = scoped_configuration
    original_sources = raw.frozen_sources

    def frozen_sources():
        return {**original_sources(), "collect_legacy_cpu12.py": supplement.bound(Path(__file__))}

    raw.frozen_sources = frozen_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("check", help="Read-only source, complete-export and selector validation")
    check.add_argument("--early-result", type=Path, required=True)
    check.add_argument("--variant", choices=tuple(adapter.final.stress.VARIANTS), required=True)
    create = commands.add_parser("write-spec", help="Write a held CPU12 legacy-only specification")
    create.add_argument("--early-result", type=Path, required=True)
    create.add_argument("--variant", choices=tuple(adapter.final.stress.VARIANTS), required=True)
    create.add_argument("--report", type=Path, required=True)
    create.add_argument("--work", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("collect", help="Requires the explicit separate root CPU12 legacy handoff")
    run.add_argument("specification", type=Path)
    run.add_argument("--execute-numerical", action="store_true")
    run.add_argument("--handoff", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "collect":
        spec = adapter.final.read(args.specification)
        scope = validate_selection(spec)
        variant = spec["variant"]
    else:
        variant, scope = args.variant, selection(args.variant)
    configure(variant)
    if args.action == "check":
        _, registration, signature = raw.cohort(args.early_result)
        value = {"status": "legacy-only-scope-and-full-cli-assets-verified", "scope": scope,
                 "generation_revision": registration["generation_revision"],
                 "runtime_identity_sha256": signature["identity_sha256"],
                 "configuration": scoped_configuration(variant),
                 "numerical_evaluations": 0, "compiler_calls": 0,
                 "rust_or_symbolica_dispatch_permitted": False}
    elif args.action == "write-spec":
        requested = args.output
        args.output = requested.with_name(requested.stem + ".retained-base.json")
        raw.write_spec(args)
        base = policy.verified(adapter.final.read(args.output), raw.SCHEMA)
        value = selected_spec(base, supplement.bound(args.output))
        validate_selection(value)
        raw.exclusive_json(requested, value)
        value = {"specification": supplement.bound(requested), "scope": scope,
                 "numerical_evaluations": 0, "compiler_calls": 0}
    else:
        if os.sched_getaffinity(0) != {CPU}:
            raise ValueError("Legacy-only stress controller requires affinity CPU12")
        if any(os.environ.get(name) != "1" for name in adapter.pair.THREADS):
            raise ValueError("Every legacy-only stress evaluation thread setting must be one")
        handoff = adapter.final.read(args.handoff)
        if (handoff.get("legacy_only_scope") != scope
                or handoff.get("rust_or_symbolica_dispatch_permitted") is not False):
            raise ValueError("Root handoff does not authorize exactly this legacy-only scope")
        value = raw.collect(args)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    def interrupted(sig, _frame):
        raise SystemExit("Legacy-only CPU12 stress controller interrupted by signal " + str(sig))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    main()
