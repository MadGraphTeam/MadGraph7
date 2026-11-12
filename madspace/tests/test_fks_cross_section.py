"""The cross sections of the FKS integrand of madspace with the Fortran test
provider (fks_integration.py) against those of MG5aMC at LO and NLO
(test_data/fks_cross_section, from make_fks_cross_section_reference.py), per
subprocess, within PULL combined errors. For e+ e- > j j at a fixed scale
without cuts the NLO also has to be sigma_LO (1 + alpha_s / pi) on the same
points.

The provider needs the aMC@NLO outputs of the references, as for
test_fks_fortran_provider.py (MADSPACE_FKS_OUTPUTS). MADSPACE_FKS_POINTS sets
the number of points of an iteration (VEGAS adapts in four, six add up),
POINTS by default, with which all setups take three minutes on one core,
most of it for the virtual of g g > t t~.
"""

import functools
import glob
import json
import math
import os
import tempfile

import numpy as np
import pytest
from fks_integration import Couplings, SubprocessIntegrand
from test_fks_couplings import alpha_s_madfks

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
pytestmark = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)
POINTS = int(os.environ.get("MADSPACE_FKS_POINTS", "2000"))
PULL = 4.0
DATA = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(__file__), "test_data", "fks_cross_section", "*.json"
        )
    )
)
IDS = [os.path.basename(path)[:-5] for path in DATA]


@functools.cache
def reference(path):
    with open(path) as f:
        return json.load(f)


@functools.cache
def cross_sections(path, k=None):
    """The LO and NLO cross sections of madspace per subprocess."""
    data = reference(path)
    run, param = data["run_card"], data["param_card"]
    couplings = Couplings(
        os.path.join(tempfile.gettempdir(), "madspace_fks_cross_section"),
        float(param["sminputs"]["3"]),
        int(run["lpp1"]) != 0,
    )
    results = {}
    for subprocess in data["nlo"]["subprocesses"]:
        directory = os.path.join(OUTPUTS, data["output"], "SubProcesses", subprocess)
        integrand = SubprocessIntegrand(directory, data, couplings)
        try:
            results[subprocess] = integrand.integrate(POINTS, k=k)
        finally:
            integrand.close()
    return results


def pull(ours, theirs):
    return (ours["sigma"] - theirs["sigma"]) / math.hypot(
        ours["error"], theirs["error"]
    )


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_against_mg5amc(path):
    """The LO and NLO cross sections of each subprocess and their sums."""
    data = reference(path)
    ours = cross_sections(path)
    for order in ("lo", "nlo"):
        totals = dict(sigma=0.0, error=0.0)
        for subprocess, theirs in data[order]["subprocesses"].items():
            value = ours[subprocess][order]
            assert abs(pull(value, theirs)) < PULL, (subprocess, order, value, theirs)
            totals["sigma"] += value["sigma"]
            totals["error"] = math.hypot(totals["error"], value["error"])
        assert abs(pull(totals, data[order])) < PULL, (order, totals, data[order])


def test_epem_jj_k_factor():
    """At the fixed scale mu = sqrt(s) without cuts, NLO - (1 + alpha_s /
    pi) LO is zero on the same points, alpha_s that of MadFKS."""
    (path,) = (p for p in DATA if p.endswith("epem_jj_fixed.json"))
    data = reference(path)
    scale = float(data["run_card"]["mur_ref_fixed"])
    k = 1 + alpha_s_madfks(float(data["param_card"]["sminputs"]["3"]), scale) / math.pi
    for subprocess, result in cross_sections(path, k).items():
        difference = result["difference"]
        assert abs(difference["sigma"]) < PULL * difference["error"], subprocess
        assert difference["error"] < 1e-2 * result["lo"]["sigma"]
        np.testing.assert_allclose(
            result["nlo"]["sigma"] / result["lo"]["sigma"], k, rtol=1e-2
        )
