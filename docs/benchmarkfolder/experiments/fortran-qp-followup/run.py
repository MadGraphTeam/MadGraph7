#!/usr/bin/env python3
"""Reuse the isolated full-evaluation experiment with the final v2 input export."""
import importlib.util
from pathlib import Path
import re
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(HERE)]
import prototype
from prototype import transform_fixed
# The reused runner imports this name before its modifier is replaced below.
prototype.transform = transform_fixed

ORIGINAL_RUNNER = HERE.parent / "expanded-wl-sparsity/run.py"
spec = importlib.util.spec_from_file_location("qp_followup_runner", ORIGINAL_RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def modify(directory, variant):
    source = (directory / "polynomial.f").read_text()
    from terminal import current_classifier
    source = current_classifier(source)
    if variant == "original":
        return {"polynomial.f": source}
    if variant == "terminal":
        from terminal import transform
        path = directory / "polynomial.f"
        original = path.read_text()
        try:
            path.write_text(source)
            files, proof = transform(directory)
        finally:
            path.write_text(original)
        proof["shared_classifier_preparation"] = {
            "description": "The committed ASSOCIATE storage fix is applied equally "
                           "to baseline and terminal variants before comparison.",
            "before_sha256": runner.hashlib.sha256(original.encode()).hexdigest(),
            "after_sha256": runner.hashlib.sha256(source.encode()).hexdigest()}
        # Keep the dependency proof next to the measurements rather than only
        # in the disposable copied process directory.
        runner.bench.dump(HERE / directory.parents[2].name / "terminal-proof.json",
                          proof)
        return files
    if variant != "classifier":
        raise RuntimeError("Unknown isolated prototype")
    changed, count = transform_fixed(source)
    def dp_routines(text):
        return re.findall(r"(?m)^      SUBROUTINE (?!MP_)\w+\(.*?^      END\s*$",
                          text, flags=re.S | re.M)
    if dp_routines(source) != dp_routines(changed):
        raise RuntimeError("DP source changed")
    print(f"Transformed {count} finite-path QP routines", flush=True)
    return {"polynomial.f": changed}


if __name__ == "__main__":
    runner.HERE = HERE
    runner.EXPERIMENT = "fortran-qp-followup"
    runner.RUNNER_DEPENDENCIES = [
        Path(__file__).resolve(), ORIGINAL_RUNNER, HERE / "terminal.py"]
    runner.bench.WORK = ROOT / ".benchmark-work/policy-v2/generation"
    runner.modify_variant = modify
    runner.main()
