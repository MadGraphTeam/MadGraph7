#!/usr/bin/env python3
"""Held final-pin stress raw collection; no final core identity is invented.

Reuse the retained collector, numerical workers and fixed input variants. Only
publication/source checks and subprocess ownership adapt to the final release.
Actual collection still requires the explicit root CPU190 handoff record.
"""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_final as final

ROOT, bench, pair = final.ROOT, final.bench, final.pair
policy, supplement = final.policy, final.supplement
OLD_COLLECTOR = ROOT / "docs/benchmarkfolder/policy-v2/final-stress-raw-collection-v1/collect.py"
loader = importlib.util.spec_from_file_location("retained_stress_raw_collector", OLD_COLLECTOR)
raw = importlib.util.module_from_spec(loader)
loader.loader.exec_module(raw)


def numerical_sources(revision):
    """Bind actual GEN/vendor bytes; operational commits are recorded separately."""
    if revision != final.REVISION:
        raise ValueError("Stress numerical source is not the final pinned release")
    paths = (*bench.GENERATOR_PATHS, "vendor")
    for origin in ("HEAD", revision):
        subprocess.run(["git", "diff", "--exit-code", origin, "--", *paths], cwd=ROOT, check=True)
    untracked = set(subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths], cwd=ROOT,
        text=True).splitlines())
    if untracked - {"vendor/StdHEP/fail"}:
        raise ValueError("Uncommitted numerical/generator inputs are present")
    if bench.content_digest(bench.generator_tree_state()) != final.GENERATOR:
        raise ValueError("The actual generator differs from final CLI source")
    names = subprocess.check_output(["git", "ls-files", "-z", "--", *paths], cwd=ROOT,
                                    text=True).split("\0")
    changed = subprocess.check_output(["git", "diff", "--name-only", "-z", revision, "HEAD"],
                                      cwd=ROOT, text=True).split("\0")
    outside = [name for name in changed if name and not any(
        name == path or name.startswith(path + "/") for path in paths)]
    checksum = lambda name: supplement.digest(ROOT / name) if (ROOT / name).is_file() else None
    return {"production": {name: checksum(name) for name in names if name},
            "operational_committed_changes": {name: checksum(name) for name in outside},
            "retained_untracked_diagnostics": {name: checksum(name) for name in untracked}}


@contextmanager
def owned_processes(work, report):
    """Reuse registered ownership, preserving every numerical worker primitive."""
    with pair.owned_preparation(report, label="stress-final"):
        original = bench.run

        def run(command, cwd, log, **kwargs):
            actual = ["make", "-j1", "check"] if command == ["make", "-j4", "check"] else command
            return original(actual, cwd, log, **kwargs)

        bench.run = run
        try:
            yield
        finally:
            bench.run = original


def configure_adapters():
    raw.early_build.runtime = final.runtime
    raw.early_build.sources = numerical_sources
    raw.owned_processes = owned_processes
    original = raw.frozen_sources

    def frozen_sources():
        return {**original(), "collect_final.py": supplement.bound(Path(__file__)),
                "prepare_final.py": supplement.bound(Path(final.__file__)),
                "ninjars_benchmark_pair_v3.py": supplement.bound(Path(pair.__file__)),
                "ninjars_benchmark_references_v2.py": supplement.bound(
                    ROOT / "tools/ninjars_benchmark_references_v2.py")}

    raw.frozen_sources = frozen_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("check", help="Validate actual fresh CLI assets; no engines")
    check.add_argument("--early-result", type=Path, required=True)
    create = commands.add_parser("write-spec", help="Prepare a held fixed-variant specification")
    create.add_argument("--early-result", type=Path, required=True)
    create.add_argument("--variant", choices=tuple(final.stress.VARIANTS), required=True)
    create.add_argument("--report", type=Path, required=True)
    create.add_argument("--work", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("collect", help="Requires the later explicit root engine handoff")
    run.add_argument("specification", type=Path)
    run.add_argument("--execute-numerical", action="store_true")
    run.add_argument("--handoff", type=Path, required=True)
    args = parser.parse_args()
    configure_adapters()
    if args.action == "check":
        result, registration, signature = raw.cohort(args.early_result)
        equality = final.validate_inputs(result)
        value = {"status": "fresh-cli-inputs-verified", "numerical_evaluations": 0,
                 "generation_revision": registration["generation_revision"],
                 "runtime_identity_sha256": signature["identity_sha256"],
                 "fixed_byte_identity_checks": len(equality), "final_dataset_claim": False}
    elif args.action == "write-spec":
        value = raw.write_spec(args)
    else:
        if os.sched_getaffinity(0) != {190}:
            raise ValueError("Actual stress collection requires exclusive controller affinity CPU190")
        if any(os.environ.get(name) != "1" for name in pair.THREADS):
            raise ValueError("Every stress evaluation thread setting must be one")
        value = raw.collect(args)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    def interrupted(sig, _frame):
        raise SystemExit("Final stress controller interrupted by signal " + str(sig))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    main()
