"""The cross sections of the MG7 NLO driver (madgraph/mg7/fks_driver.py) with
the FKS integrand of madspace against those of MG5aMC at LO and NLO
(test_data/fks_cross_section, from make_fks_cross_section_reference.py), per
subprocess, within PULL combined errors. For e+ e- > j j at a fixed scale
without cuts the NLO also has to be sigma_LO (1 + alpha_s / pi) on the same
points.

The driver runs on the aMC@NLO outputs of the references, as for
test_fks_fortran_provider.py (MADSPACE_FKS_OUTPUTS), with the run card of
each reference. MADSPACE_FKS_POINTS sets the number of points per channel of
an iteration (four adapt the grids, six add up), POINTS by default, with
which all setups take a minute on one core, half of it for the virtual of
g g > t t~. With MADSPACE_FKS_ACCURACY=reference the runs go on to the
accuracy of each reference instead (req_acc_FO of its run card), as a
fixed-order run of MG5aMC does, in about 20 minutes on one core. p p > e+
e- j (twelve subprocesses, two and a half minutes there) runs only with
MADSPACE_FKS_SLOW=1.
"""

import functools
import glob
import json
import math
import os
import tempfile

import numpy as np
import pytest
from madgraph.mg7.fks_driver import FKSRun, default_histograms, read_histograms
from madspace_testing import default_context
from test_fks_couplings import alpha_s_madfks

from madspace.fks import fks_histogram_jet_count

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
needs_outputs = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)
POINTS = int(os.environ.get("MADSPACE_FKS_POINTS", "2000"))
TO_ACCURACY = os.environ.get("MADSPACE_FKS_ACCURACY") == "reference"
SLOW = os.environ.get("MADSPACE_FKS_SLOW") == "1"
# the references that run only with MADSPACE_FKS_SLOW=1
SLOW_REFERENCES = {"pp_epemj"}
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
def fks_run(path):
    """The finished run of the driver on the output of a reference, with
    its run card and param card."""
    data = reference(path)
    name = os.path.basename(path)[:-5]
    if name in SLOW_REFERENCES and not SLOW:
        pytest.skip(f"{name} runs only with MADSPACE_FKS_SLOW=1")
    points = {}
    if not TO_ACCURACY:
        points = {
            "accuracy": 0.0,
            "grid_points": POINTS,
            "grid_iterations": 4,
            "points": POINTS,
            "max_iterations": 6,
        }
    run = FKSRun(
        os.path.join(OUTPUTS, data["output"]),
        run_card=data["run_card"],
        param_card=data["param_card"],
        run_path=os.path.join(tempfile.gettempdir(), "madspace_fks_runs", name),
        seed=4711,
        **points,
    )
    run.run()
    return run


def pull(ours, theirs):
    return (ours["sigma"] - theirs["sigma"]) / math.hypot(
        ours["error"], theirs["error"]
    )


@needs_outputs
@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_against_mg5amc(path):
    """The LO and NLO cross sections of each subprocess and their sums."""
    data = reference(path)
    ours = fks_run(path).result()
    for order in ("lo", "nlo"):
        for subprocess, theirs in data[order]["subprocesses"].items():
            value = ours["subprocesses"][subprocess][order]
            assert abs(pull(value, theirs)) < PULL, (subprocess, order, value, theirs)
        assert abs(pull(ours[order], data[order])) < PULL, (order, ours, data[order])


@needs_outputs
@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_info_json(path):
    """The info.json of the run: the NLO cross section and its channels as
    MadBoard reads them, and histograms that add up to the cross section
    with their underflow and overflow. The grids of the run stay out of the
    default context."""
    run = fks_run(path)
    assert not default_context().global_exists("alpha_s_logq2")
    with open(os.path.join(run.run_path, "info.json")) as f:
        info = json.load(f)
    result = run.result()
    assert info["status"] == "done"
    np.testing.assert_allclose(info["process"]["mean"], result["nlo"]["sigma"])
    np.testing.assert_allclose(info["process"]["error"], result["nlo"]["error"])
    channels = info["channels"]
    np.testing.assert_allclose(
        sum(c["mean"] for c in channels), result["nlo"]["sigma"], rtol=1e-12
    )
    assert {c["subprocess"] for c in channels} == set(range(len(run.integrands)))
    assert info["histograms"]
    for histogram in info["histograms"]:
        assert len(histogram["bin_values"]) == histogram["bin_count"] + 2
        np.testing.assert_allclose(
            sum(histogram["bin_values"]), result["nlo"]["sigma"], rtol=1e-10
        )
    assert os.path.isfile(os.path.join(run.run_path, "MADatNLO_madspace.HwU"))
    for statistics in info["fks"]["matrix_elements"].values():
        assert 0 < statistics["virtual"]["points"] <= statistics["born"]["points"]


@needs_outputs
def test_epem_jj_k_factor():
    """At the fixed scale mu = sqrt(s) without cuts, NLO - (1 + alpha_s /
    pi) LO is zero on the same points, alpha_s that of MadFKS."""
    (path,) = (p for p in DATA if p.endswith("epem_jj_fixed.json"))
    data = reference(path)
    scale = float(data["run_card"]["mur_ref_fixed"])
    k = 1 + alpha_s_madfks(float(data["param_card"]["sminputs"]["3"]), scale) / math.pi
    run = fks_run(path)
    for subprocess, result in run.result()["subprocesses"].items():
        difference, error = run.combination([1.0, -k, 0.0], subprocess)
        assert abs(difference) < PULL * error, subprocess
        assert error < 1e-2 * result["lo"]["sigma"]
        np.testing.assert_allclose(
            result["nlo"]["sigma"] / result["lo"]["sigma"], k, rtol=1e-2
        )


def test_histograms_of_run_card(tmp_path):
    """The histograms of a run_card.toml of MG7 by their names over the
    groups of [multiparticles], without the histogram of the event weight,
    and the defaults named the same way."""
    card = tmp_path / "run_card.toml"
    card.write_text(
        "[multiparticles]\n"
        "jet = [21, 1, -1, 2, -2]\n"
        "e = [11, -11]\n"
        "[histograms]\n"
        'order_by = "pt"\n'
        '"jet_1-pt" = { min = 0.0, max = 100.0, bin_count = 10 }\n'
        '"jet-sum-pt" = { min = 0.0, max = 50.0, bin_count = 5 }\n'
        '"e-jet_2-delta_r" = { min = 0.0, max = 5.0, bin_count = 5 }\n'
        '"sqrt_s" = { min = 0.0, max = 1000.0, bin_count = 20 }\n'
        '"weight" = { min = 0.0, max = 5.0, bin_count = 50 }\n'
    )
    histograms = {h["name"]: h for h in read_histograms(card)}
    assert list(histograms) == ["jet_1-pt", "jet-sum-pt", "e-jet_2-delta_r", "sqrt_s"]
    jet = [21, 1, -1, 2, -2]
    assert histograms["jet_1-pt"]["observable"]["select_pids"] == [jet]
    assert histograms["jet_1-pt"]["observable"]["order_indices"] == [1]
    assert histograms["jet-sum-pt"]["observable"]["sum_momenta"]
    assert histograms["jet-sum-pt"]["observable"]["order_observable"] is None
    pair = histograms["e-jet_2-delta_r"]
    assert pair["observable"]["select_pids"] == [[11, -11], jet]
    assert pair["observable"]["order_indices"] == [0, 2]
    assert histograms["sqrt_s"]["observable"]["select_pids"] == []
    # a single jet by its order too, which the real emission needs, as the
    # FKS histograms take it
    configurations = {"nincoming": 2, "born_pdgs": [-1, 1, 11, -11, 21]}
    defaults = default_histograms(configurations, 13000.0, False, False, 4)
    assert [h["name"] for h in defaults] == [
        "e_1-pt",
        "e_2-pt",
        "jet_1-pt",
        "e_1-eta",
        "e_2-eta",
        "jet_1-eta",
        "e_1-e_2-pair_mass",
        "e_1-jet_1-pair_mass",
        "e_2-jet_1-pair_mass",
        "sqrt_s",
    ]
    for histogram in defaults:
        fks_histogram_jet_count(histogram["observable"])
