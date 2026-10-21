#!/usr/bin/env python3
"""Held raw NinjaRS stress calls and matched rotation/scale Ninja controls.

Ordinary legacy references are audited separately. This file never promotes a
raw run to the four-backend stress dataset or claims a completed core identity.
The all-four extension remains available through collect_final.py.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess

import collect_final as adapter

raw, policy, supplement = adapter.raw, adapter.policy, adapter.supplement


def target_backends(variant):
    if variant not in adapter.final.stress.VARIANTS:
        raise ValueError("Unknown fixed stress variant")
    return (8,) if variant == "ordinary" else (6, 8)


def selected_spec(base, bound_base):
    return policy.seal({**base, "retained_base_specification": bound_base,
        "explicit_target_backends": list(target_backends(base["variant"])),
        "target_policy": "fresh-NinjaRS8; matched-Ninja6-rotation-and-scale-controls",
        "ordinary_legacy_reference_policy": "Separate strict source/point/card/runtime audit; no reference row relabeling",
        "four_backend_stress_dataset_claim": False})


def validate_selection(spec):
    policy.verified(spec, raw.SCHEMA)
    base = policy.verified(adapter.final.read(supplement.checked(
        spec["retained_base_specification"])), raw.SCHEMA)
    if spec != selected_spec(base, spec["retained_base_specification"]):
        raise ValueError("Stress raw backend selection differs from the bound fixed policy")
    return target_backends(spec["variant"])


def stable_orchestration(spec, record=True):
    """Retain numerical/imported bytes while recording unrelated HEAD advances."""
    if (spec["numerical_revision"] != adapter.final.REVISION
            or spec["generation_revision"] != adapter.final.REVISION
            or spec["generator_identity_sha256"] != adapter.final.GENERATOR
            or spec["installed_archive_sha256"] != adapter.final.ARCHIVE):
        raise ValueError("NinjaRS stress numerical cohort differs from the final pinned source")
    for source in spec["sources"].values():
        supplement.checked(source)
    adapter.numerical_sources(spec["numerical_revision"])
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=adapter.ROOT,
                                     text=True).strip()
    if record:
        path = Path(__file__).parent / "nrs-source-head-observations" / (spec["identity_sha256"] + ".jsonl")
        path.parent.mkdir(exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "specification_orchestration_head": spec["operational_source_state"]["actual_git_head"],
                "observed_orchestration_head": actual,
                "numerical_revision": adapter.final.REVISION,
                "generator_identity_sha256": adapter.final.GENERATOR,
                "exact_GEN_vendor_and_bound_helper_bytes_verified": True}) + "\n")
    return actual


def configure():
    adapter.configure_adapters()
    raw.check_orchestration_head = stable_orchestration
    original = raw.frozen_sources

    def frozen_sources():
        return {**original(), "collect_nrs_first.py": supplement.bound(Path(__file__))}

    raw.frozen_sources = frozen_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("write-spec", help="Bind a held explicit subset; no numerical calls")
    create.add_argument("--early-result", type=Path, required=True)
    create.add_argument("--variant", choices=tuple(adapter.final.stress.VARIANTS), required=True)
    create.add_argument("--report", type=Path, required=True)
    create.add_argument("--work", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("collect", help="Requires the later root CPU190 engine handoff")
    run.add_argument("specification", type=Path)
    run.add_argument("--execute-numerical", action="store_true")
    run.add_argument("--handoff", type=Path, required=True)
    args = parser.parse_args()
    configure()
    if args.action == "write-spec":
        requested = args.output
        args.output = requested.with_name(requested.stem + ".retained-base.json")
        raw.write_spec(args)
        base = policy.verified(adapter.final.read(args.output), raw.SCHEMA)
        value = selected_spec(base, supplement.bound(args.output))
        raw.exclusive_json(requested, value)
        value = {"specification": supplement.bound(requested), "numerical_evaluations": 0,
                 "explicit_target_backends": value["explicit_target_backends"],
                 "final_dataset_claim": False}
    else:
        spec = adapter.final.read(args.specification)
        raw.BACKENDS = validate_selection(spec)
        if os.sched_getaffinity(0) != {190}:
            raise ValueError("Actual stress collection requires controller affinity CPU190")
        if any(os.environ.get(name) != "1" for name in adapter.pair.THREADS):
            raise ValueError("Every stress evaluation thread setting must be one")
        value = raw.collect(args)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    def interrupted(sig, _frame):
        raise SystemExit("NinjaRS stress controller interrupted by signal " + str(sig))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    main()
