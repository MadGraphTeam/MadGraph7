#!/usr/bin/env python3
"""Copied generated-process comparison; original Ninja6 only."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "madgraph/__init__.py").is_file())
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(HERE)]
import prototype
import static_owner

ORIGINAL_RUNNER = HERE.parent / "expanded-wl-sparsity/run.py"
spec = importlib.util.spec_from_file_location("static_owner_runner", ORIGINAL_RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def modify(directory, variant):
    if variant == "original":
        return {}
    operations = variant.endswith("-counts")
    trace = operations or variant.endswith("-trace")
    mode = variant.removesuffix("-trace").removesuffix("-counts")
    assert mode in ("lazy", "owner-only", "folded")
    files, proof = prototype.transform(directory, mode, trace, operations)
    runner.bench.dump(HERE / directory.parents[2].name / (variant + "-proof.json"), proof)
    return files


if __name__ == "__main__":
    runner.HERE = HERE
    runner.EXPERIMENT = "symmetric-static-owner"
    runner.RUNNER_DEPENDENCIES = [
        Path(__file__), ORIGINAL_RUNNER, HERE / "static_owner.py",
        HERE / "count_products.py",
        static_owner.PREVIOUS / "prototype.py",
        static_owner.PREVIOUS / "shared.py",
        static_owner.PREVIOUS / "gather.py",
        static_owner.PREVIOUS / "inventory.json",
        static_owner.PREVIOUS / "SHA256SUMS.json",
    ]
    runner.bench.WORK = ROOT / ".benchmark-work/policy-v2/generation-abi3-terminal-fixed"
    runner.modify_variant = modify
    runner.main()
    for path in (runner.bench.WORK / "experiments" / runner.EXPERIMENT).glob(
            "*/*/SubProcesses/P*_*/sharing_*.dat"):
        report = HERE / path.parents[3].name
        target = report / (path.parents[2].name + "-" + path.name)
        if report.is_dir() and not target.exists():
            runner.shutil.copy2(path, target)
