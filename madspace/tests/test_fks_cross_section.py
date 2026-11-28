"""The cross sections of the MG7 NLO driver (madgraph/mg7/nlo/fks.py) with
the FKS integrand of madspace, through the event generator of madspace,
against those of MG5aMC at LO and NLO
(test_data/fks_cross_section, from make_fks_cross_section_reference.py), per
subprocess, within PULL combined errors. For e+ e- > j j at a fixed scale
without cuts the NLO also has to be sigma_LO (1 + alpha_s / pi) on the same
points. With --ewsudakov the Sudakov approximation of the EW corrections per
sud_mod is that of the histograms of MG5aMC, whose NLO sums both modes with
it (nlo_with_ew_sudakov of the driver).

The driver runs on the aMC@NLO outputs of the references, as for
test_fks_fortran_provider.py (MADSPACE_FKS_OUTPUTS), with the run card of
each reference. MADSPACE_FKS_POINTS sets the number of points per channel of
an iteration (four adapt the grids, six add up), POINTS by default, with
which all setups take a few minutes. With MADSPACE_FKS_ACCURACY=reference
the runs go on to the accuracy of each reference instead (req_acc_FO of its
run card), as a fixed-order run of MG5aMC does. p p > e+ e- j (twelve
subprocesses) runs only with MADSPACE_FKS_SLOW=1.
"""

import functools
import glob
import json
import math
import os
import tempfile

import numpy as np
import pytest
from fks_testing import fks_output
from madgraph.mg7.nlo.fks import FKSRun, default_histograms, read_histograms
from madspace_testing import default_context
from test_fks_couplings import alpha_s_madfks
from test_nlo_pdlabels import TEST_LHAID, write_test_set

from madspace.nlo import histogram_jet_count

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
needs_outputs = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)
POINTS = int(os.environ.get("MADSPACE_FKS_POINTS", "2000"))
TO_ACCURACY = os.environ.get("MADSPACE_FKS_ACCURACY") == "reference"
SLOW = os.environ.get("MADSPACE_FKS_SLOW") == "1"
# the references that run only with MADSPACE_FKS_SLOW=1
SLOW_REFERENCES = {"pp_epemj"}
# the references whose runs take a multiple of POINTS. The LO of p p > a a
# has rare large weights, with which a subprocess came out 4.2 standard
# deviations low in the scalar mode otherwise.
LONGER_RUNS = {"pp_aa": 4}
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
        count = LONGER_RUNS.get(name, 1) * POINTS
        points = {
            "accuracy": 0.0,
            "grid_points": count,
            "grid_iterations": 4,
            "points": count,
            "max_iterations": 6,
        }
    run = FKSRun(
        fks_output(data["output"]),
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
        # MadFKS sums the Sudakov approximation with the NLO
        key = "nlo_with_ew_sudakov" if "ew_sudakov" in data[order] else order
        for subprocess, theirs in data[order]["subprocesses"].items():
            value = ours["subprocesses"][subprocess][key]
            assert abs(pull(value, theirs)) < PULL, (subprocess, order, value, theirs)
        assert abs(pull(ours[key], data[order])) < PULL, (order, ours, data[order])


# the references with the Sudakov approximation of the EW corrections
SUDAKOV = [path for path in DATA if "ew_sudakov" in reference(path)["nlo"]]


@needs_outputs
@pytest.mark.parametrize(
    "path", SUDAKOV, ids=[os.path.basename(path)[:-5] for path in SUDAKOV]
)
def test_ew_sudakov_against_mg5amc(path):
    """The Sudakov approximation of the EW corrections for sud_mod 0 and 1,
    against the histograms of MG5aMC (ibody 10 and 11 of its analysis), and
    the NLO without them is the sum of the channels."""
    data = reference(path)
    run = fks_run(path)
    ours = run.result()
    theirs = data["nlo"]["ew_sudakov"]
    assert [e["sud_mod"] for e in ours["ew_sudakov"]] == [e["sud_mod"] for e in theirs]
    for mine, other in zip(ours["ew_sudakov"], theirs):
        assert mine["sigma"] < 0
        assert abs(pull(mine, other)) < PULL, (mine, other)
    total = ours["nlo"]["sigma"] + sum(e["sigma"] for e in ours["ew_sudakov"])
    np.testing.assert_allclose(ours["nlo_with_ew_sudakov"]["sigma"], total, rtol=1e-12)
    channels = run.channel_status()
    np.testing.assert_allclose(
        sum(c["mean"] for c in channels), ours["nlo"]["sigma"], rtol=1e-12
    )


# the references with the scale variations of reweight_scale
VARIED = [path for path in DATA if "scale_variations" in reference(path)["nlo"]]


@needs_outputs
@pytest.mark.parametrize(
    "path", VARIED, ids=[os.path.basename(path)[:-5] for path in VARIED]
)
def test_scale_variations_against_mg5amc(path):
    """The cross section at each pair of factors of mu_R and mu_F relative to
    the central one, against that of the total rate of the histograms of
    MG5aMC within PULL relative errors of the NLO of both (the ratio of the
    same points has less), and their envelope."""
    data = reference(path)
    ours = fks_run(path).result()
    theirs = data["nlo"]["scale_variations"]
    pairs = [(entry["mu_r"], entry["mu_f"]) for entry in ours["scale_variations"]]
    assert pairs == [(entry["mu_r"], entry["mu_f"]) for entry in theirs]
    our_central = ours["nlo"]["sigma"]
    (their_central,) = [e["sigma"] for e in theirs if (e["mu_r"], e["mu_f"]) == (1, 1)]
    error = math.hypot(
        ours["nlo"]["error"] / our_central, data["nlo"]["error"] / data["nlo"]["sigma"]
    )
    for mine, other in zip(ours["scale_variations"], theirs):
        ratio = mine["sigma"] / our_central
        expected = other["sigma"] / their_central
        assert abs(ratio - expected) < PULL * error * expected, (mine, other)
    up = (max(e["sigma"] for e in theirs) / their_central - 1) * 100
    down = (1 - min(e["sigma"] for e in theirs) / their_central) * 100
    band = ours["scale_uncertainty"]
    assert abs(band["up"] - up) < PULL * error * 100 * (1 + up / 100)
    assert abs(band["down"] - down) < PULL * error * 100


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
    hwu = os.path.join(run.run_path, "MADatNLO_madspace.HwU")
    if not run.histograms:
        # the default histograms do not cover reals that differ in more than
        # their QCD partons (QED corrections), the run goes on without them
        assert not info["histograms"]
        return
    assert info["histograms"]
    for histogram in info["histograms"]:
        assert len(histogram["bin_values"]) == histogram["bin_count"] + 2
        np.testing.assert_allclose(
            sum(histogram["bin_values"]), result["nlo"]["sigma"], rtol=1e-10
        )
    assert os.path.isfile(hwu)
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


@needs_outputs
def test_pdf_variations(tmp_path, monkeypatch):
    """With reweight_PDF the run gives the cross section with every member
    of the set on the same points: that of a member with the central
    densities (those of nn23nlo) is the NLO, that of a member with the
    densities times f the NLO times f^2, and their uncertainty that of
    LHAPDF."""
    (path,) = (p for p in DATA if p.endswith("pp_epem.json"))
    data = reference(path)
    factors = [1.0, 1.1, 0.95]
    directory = write_test_set(str(tmp_path / "lhapdf"), factors)
    monkeypatch.setenv("LHAPDF_DATA_PATH", directory)
    run_card = dict(
        data["run_card"], pdlabel="lhapdf", lhaid=TEST_LHAID, reweight_pdf=True
    )
    # the running of alpha_s of the last member differs, that of the others
    # is the central one
    run = FKSRun(
        fks_output(data["output"]),
        run_card=run_card,
        param_card=data["param_card"],
        run_path=str(tmp_path / "run"),
        seed=4711,
        accuracy=0.0,
        grid_points=500,
        grid_iterations=2,
        points=500,
        max_iterations=2,
    )
    result = run.run()
    members = result["pdf_variations"]
    assert [m["member"] for m in members] == [0, 1, 2]
    nlo = result["nlo"]["sigma"]
    np.testing.assert_allclose(members[0]["sigma"], nlo, rtol=1e-12)
    np.testing.assert_allclose(members[1]["sigma"], 1.21 * nlo, rtol=1e-10)
    # the alpha_s of the last member changes the NLO part a little
    assert members[2]["sigma"] == pytest.approx(0.95**2 * nlo, rel=1e-2)
    band = result["pdf_uncertainty"]
    assert (band["name"], band["error_type"]) == ("TestSet", "hessian")
    np.testing.assert_allclose(band["central"], nlo, rtol=1e-12)
    np.testing.assert_allclose(band["up"], 21.0, rtol=1e-8)
    expected = (1 - members[2]["sigma"] / nlo) * 100
    np.testing.assert_allclose(band["down"], expected, rtol=1e-8)


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
        histogram_jet_count(histogram["observable"])
