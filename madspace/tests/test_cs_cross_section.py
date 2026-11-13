"""The cross sections of the Catani-Seymour NLO driver (madgraph/mg7/cs_driver.py)
on the aMC@NLO outputs of the references of test_fks_cross_section.py
(MADSPACE_FKS_OUTPUTS), with their run and param cards.

For e+ e- > j j at a fixed scale the n-body stream is sigma_LO (1 + C_F
alpha_s / pi) on the same points: the virtual and the insertion operator I
add up to C_F alpha_s / pi times the Born, whatever the scale, which fixes
the finite part of I against the normalization of the virtual of MadLoop.
"""

import functools
import json
import math
import os
import tempfile

import numpy as np
import pytest
from madgraph.mg7.cs_driver import CSRun
from madgraph.mg7.nlo_driver import Runtime

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
needs_outputs = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)
POINTS = int(os.environ.get("MADSPACE_FKS_POINTS", "1000"))
DATA = os.path.join(os.path.dirname(__file__), "test_data", "fks_cross_section")
CF = 4.0 / 3.0


@functools.cache
def reference(name):
    with open(os.path.join(DATA, f"{name}.json")) as f:
        return json.load(f)


@functools.cache
def cs_run(name, alpha=1.0):
    """The finished run of the CS driver on the output of a reference."""
    data = reference(name)
    run = CSRun(
        os.path.join(OUTPUTS, data["output"]),
        alpha=alpha,
        run_card=data["run_card"],
        param_card=data["param_card"],
        run_path=os.path.join(
            tempfile.gettempdir(), "madspace_cs_runs", f"{name}_{alpha}"
        ),
        seed=4711,
        accuracy=0.0,
        grid_points=POINTS,
        grid_iterations=2,
        points=POINTS,
        max_iterations=3,
    )
    run.run()
    return run


@needs_outputs
def test_epem_jj_nbody():
    """At a fixed scale without cuts the n-body stream minus (1 + C_F
    alpha_s / pi) LO is zero on the same points, per subprocess, with the
    alpha_s of the run."""
    data = reference("epem_jj_fixed")
    run = cs_run("epem_jj_fixed")
    scale = float(data["run_card"]["mur_ref_fixed"])
    alpha_s = Runtime(run.couplings.running, run.context)(np.array([scale]))
    k = 1 + CF * float(alpha_s["alpha_s"][0]) / math.pi
    result = run.result()
    for subprocess, entry in result["subprocesses"].items():
        difference, error = run.combination([1.0, -k, 0.0], subprocess)
        assert abs(difference) < 1e-12 * entry["lo"]["sigma"], subprocess
        np.testing.assert_allclose(
            entry["nlo"]["sigma"] / entry["lo"]["sigma"], k, rtol=1e-12
        )
