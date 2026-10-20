#!/usr/bin/env python3
"""Use the same isolated full-process experiment protocol for the color cache."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


color = load("color_cache_transform", HERE / "prototype.py")
shared = load("isolated_fortran_experiment", HERE.parent / "expanded-wl-sparsity/run.py")


def modify_variant(directory, variant):
    original = (directory / "polynomial.f").read_text()
    if variant == "original":
        return {"polynomial.f": original}
    assert variant == "color"
    changed, metadata = color.polynomial(original)
    prefix = metadata["prefix"]
    return {
        "polynomial.f": changed,
        "loop_matrix.f": color.loop_matrix((directory / "loop_matrix.f").read_text(), prefix),
        "mp_compute_loop_coefs.f": color.mp_compute(
            (directory / "mp_compute_loop_coefs.f").read_text(), prefix),
    }


if __name__ == "__main__":
    shared.HERE = HERE
    shared.EXPERIMENT = "color-contraction"
    shared.RUNNER_DEPENDENCIES = [Path(__file__)]
    shared.modify_variant = modify_variant
    shared.main()
