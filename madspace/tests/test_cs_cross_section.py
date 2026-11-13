"""The cross sections of the Catani-Seymour NLO driver (madgraph/mg7/cs_driver.py)
on the aMC@NLO outputs of the references of test_fks_cross_section.py
(MADSPACE_FKS_OUTPUTS), with their run and param cards.

For e+ e- > j j at a fixed scale the n-body stream is sigma_LO (1 + C_F
alpha_s / pi) on the same points: the virtual and the insertion operator I
add up to C_F alpha_s / pi times the Born, whatever the scale, which fixes
the finite part of I against the normalization of the virtual of MadLoop.
Without cuts the NLO is sigma_LO (1 + alpha_s / pi), so that the real
stream gives (1 - C_F) alpha_s / pi sigma_LO.

The real stream subtracts the dipoles with the Born blocks of the Fortran
provider: at the points of the reference of test_cs_dipoles.py (test_data/cs)
they have to give the dipoles of pyAmpliCol, and their sum the real matrix
element in the soft and collinear limits. The NLO cross sections are those
of MG5aMC within PULL combined errors and do not depend on alpha.
MADSPACE_FKS_POINTS sets the number of points per channel of an iteration,
POINTS by default, with which the tests take two minutes on one core, most
of it for the 108 channels of the real stream of p p > t t~.
"""

import functools
import json
import math
import os
import tempfile

import numpy as np
import pytest
from madgraph.mg7.cs_driver import STREAMS, CSRun, minkowski
from madgraph.mg7.fks_provider import boost_z, partonic_rapidity
from madgraph.mg7.nlo_driver import Runtime

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
needs_outputs = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)
POINTS = int(os.environ.get("MADSPACE_FKS_POINTS", "1000"))
PULL = 4.0
TESTS = os.path.dirname(__file__)
DATA = os.path.join(TESTS, "test_data", "fks_cross_section")
CF = 4.0 / 3.0


@functools.cache
def reference(name):
    with open(os.path.join(DATA, f"{name}.json")) as f:
        return json.load(f)


@functools.cache
def dipole_reference(name):
    with open(os.path.join(TESTS, "test_data", "cs", f"{name}.json")) as f:
        return json.load(f)


def make_run(name, alpha=1.0, streams=STREAMS, points=POINTS, **options):
    data = reference(name)
    tag = "_".join([name, str(alpha), *streams])
    return CSRun(
        os.path.join(OUTPUTS, data["output"]),
        alpha=alpha,
        streams=streams,
        run_card=data["run_card"],
        param_card=data["param_card"],
        run_path=os.path.join(tempfile.gettempdir(), "madspace_cs_runs", tag),
        seed=4711,
        accuracy=0.0,
        grid_points=points,
        grid_iterations=2,
        points=points,
        max_iterations=3,
        **options,
    )


@functools.cache
def cs_run(name, alpha=1.0, streams=STREAMS):
    """The finished run of the CS driver on the output of a reference."""
    run = make_run(name, alpha, streams)
    run.run()
    return run


def alpha_s_of(run, scale):
    return float(
        Runtime(run.couplings.running, run.context)(np.array([scale]))["alpha_s"][0]
    )


def pull(ours, theirs):
    return (ours["sigma"] - theirs["sigma"]) / math.hypot(
        ours["error"], theirs["error"]
    )


@needs_outputs
def test_epem_jj_nbody():
    """At a fixed scale without cuts the n-body stream minus (1 + C_F
    alpha_s / pi) LO is zero on the same points, per subprocess, with the
    alpha_s of the run."""
    data = reference("epem_jj_fixed")
    run = cs_run("epem_jj_fixed", streams=("nbody",))
    k = 1 + CF * alpha_s_of(run, float(data["run_card"]["mur_ref_fixed"])) / math.pi
    result = run.result()
    for subprocess, entry in result["subprocesses"].items():
        difference, error = run.combination([1.0, -k, 0.0], subprocess)
        assert abs(difference) < 1e-12 * entry["lo"]["sigma"], subprocess
        np.testing.assert_allclose(
            entry["nlo"]["sigma"] / entry["lo"]["sigma"], k, rtol=1e-12
        )


@needs_outputs
@pytest.mark.parametrize("alpha", [1.0, 0.1])
def test_epem_jj_real(alpha):
    """At a fixed scale without cuts the NLO of e+ e- > j j is sigma_LO (1 +
    alpha_s / pi) and, at alpha = 1, its real stream (1 - C_F) alpha_s / pi
    sigma_LO, within PULL errors."""
    data = reference("epem_jj_fixed")
    run = cs_run("epem_jj_fixed", alpha)
    a = alpha_s_of(run, float(data["run_card"]["mur_ref_fixed"])) / math.pi
    result = run.result()
    lo = result["lo"]["sigma"]
    reals = [
        run.combination([1.0, 0.0, 0.0], name)
        for name in result["subprocesses"]
        if " real " in name
    ]
    assert reals
    if alpha == 1.0:
        sigma = sum(value for value, _ in reals)
        error = math.sqrt(sum(e**2 for _, e in reals))
        assert abs(sigma - (1 - CF) * a * lo) < PULL * error, (sigma, error)
    difference, error = run.combination([1.0, -(1 + a), 0.0])
    assert abs(difference) < PULL * error, (difference, error)
    assert error < 1e-3 * lo


@needs_outputs
@pytest.mark.parametrize("name", ["epem_jj", "epem_jj_antikt", "pp_epem", "pp_ttx"])
def test_against_mg5amc(name):
    """The LO and NLO cross sections of the CS driver and those of MG5aMC,
    the LO also per subprocess."""
    data = reference(name)
    ours = cs_run(name).result()
    for subprocess, theirs in data["lo"]["subprocesses"].items():
        value = ours["subprocesses"][subprocess]["lo"]
        assert abs(pull(value, theirs)) < PULL, (subprocess, value, theirs)
    for order in ("lo", "nlo"):
        assert abs(pull(ours[order], data[order])) < PULL, (order, ours, data[order])


@needs_outputs
@pytest.mark.parametrize("name", ["epem_jj", "pp_epem"])
def test_alpha_independence(name):
    """The NLO cross section does not depend on alpha (1 and 0.1 for all
    families), within PULL combined errors."""
    ours = cs_run(name).result()["nlo"]
    other = cs_run(name, 0.1).result()["nlo"]
    assert abs(pull(ours, other)) < PULL, (ours, other)


@needs_outputs
def test_info_json():
    """The info.json of a run with both streams: its channels add up to the
    NLO cross section, and so do its histograms, whose real stream fills the
    real emission and the Borns of the dipoles of each point."""
    run = cs_run("epem_jj")
    with open(os.path.join(run.run_path, "info.json")) as f:
        info = json.load(f)
    result = run.result()
    assert info["status"] == "done"
    np.testing.assert_allclose(info["process"]["mean"], result["nlo"]["sigma"])
    channels = info["channels"]
    np.testing.assert_allclose(
        sum(c["mean"] for c in channels), result["nlo"]["sigma"], rtol=1e-12
    )
    assert any(" real " in c["name"] for c in channels)
    assert info["histograms"]
    for histogram in info["histograms"]:
        assert len(histogram["bin_values"]) == histogram["bin_count"] + 2
        np.testing.assert_allclose(
            sum(histogram["bin_values"]), result["nlo"]["sigma"], rtol=1e-10
        )
    assert set(info["cs"]["matrix_elements"]) == set(run.subprocesses)


@pytest.fixture(scope="module")
def real_stream():
    """The run of p p > t t~ with the real stream alone, set up and not run,
    by its technical cut, closed at the end of the module."""
    runs = {}

    def get(technical_cut):
        if technical_cut not in runs:
            run = make_run("pp_ttx", streams=("real",), technical_cut=technical_cut)
            run.setup()
            runs[technical_cut] = run
        return runs[technical_cut]

    yield get
    for run in runs.values():
        run.close()


def real_integrand(run, pdgs):
    return next(i for i in run.integrands if list(pdgs) in i.flavors)


def reference_weights(integrand, points):
    """The real weight and the weights of the dipoles of the integrand at
    the real emissions of reference points, in channel 0 with a Jacobian of
    one."""
    momenta = np.array([p["momenta"] for p in points])
    x1 = np.array([p["x1"] for p in points])
    x2 = np.array([p["x2"] for p in points])
    return integrand.weights(0, momenta, x1, x2, np.ones(len(points)))


@needs_outputs
def test_dipole_blocks(real_stream):
    """The dipoles of the real stream of p p > t t~ with the blocks of the
    Fortran provider are those of pyAmpliCol at the points of the reference
    of test_cs_dipoles.py, with the spin correlation in the factorized form
    <T_j . T_k> k_t B k_t / B: its Born B(k_t) of the azimuthal part of an
    FKS configuration is the spin tensor of the Born contracted with
    k_t."""
    data = dipole_reference("pp_ttx")
    run = real_stream(1e-6)
    g2 = 4 * math.pi * data["setup"]["alpha_s"]
    reals = {}
    for point in data["points"]:
        reals.setdefault(point["real"], []).append(point)
    checked = 0
    for real, points in reals.items():
        integrand = real_integrand(run, data["reals"][real]["pdgs"])
        dipoles = [d for d in data["cs"] if d["real"] == real]
        assert [(d["i"], d["j"], d["k"]) for d in dipoles] == [
            (d.entry["i"], d.entry["j"], d.entry["k"]) for d in integrand.dipoles
        ]
        momenta = np.array([p["momenta"] for p in points])
        x1 = np.array([p["x1"] for p in points])
        x2 = np.array([p["x2"] for p in points])
        out = integrand.subtraction(momenta, x1, x2)
        for m, dipole in enumerate(integrand.dipoles):
            me = out["born_momenta"][:, m][:, dipole.order]
            y = -partonic_rapidity(me, 2)
            cm = boost_z(me, y)
            k_t = boost_z(out["spin_vector"][:, m], y)
            born = dipole.born.provider.born(
                cm, np.full(len(points), dipole.configuration)
            )
            ours = integrand.dipole_value(
                dipole,
                cm,
                k_t,
                out["color_weight"][:, m],
                out["spin_weight"][:, m],
                born,
            )[:, 0]
            entries = [p["dipoles"][m] for p in points]
            theirs = out["color_weight"][:, m] * np.array(
                [e["color_correlated"] for e in entries]
            )
            if dipole.gluon:
                lab = out["spin_vector"][:, m]
                tensor = np.array([e["born_spin_correlated"] for e in entries])
                ratio = np.array([e["color_correlated"] / e["born"] for e in entries])
                contracted = np.einsum("bm,bmn,bn->b", lab, tensor, lab)
                theirs = theirs + out["spin_weight"][:, m] * ratio * contracted
                checked += 1
            np.testing.assert_allclose(ours, theirs / g2**2, rtol=1e-9, atol=0)
    assert checked


@needs_outputs
def test_real_limits(real_stream):
    """The real stream of p p > t t~ along the rays of the reference of
    test_cs_dipoles.py into the soft and collinear limits of each pair of
    dipoles, without the technical cut: |R - sum D| / R falls by a factor
    3 from lambda = 1e-2 to 1e-3 (0.1 to 1e-2 for the soft limits, where
    the real matrix element of the Fortran loses digits as 1 / xi^2), to
    below 1e-2, with the scales, densities and alpha_s of each dipole those
    of its Born. Their differences to those of the real emission add terms
    linear in lambda, which can cancel those of the dipoles at lambda =
    0.1."""
    data = dipole_reference("pp_ttx")
    run = real_stream(0.0)
    reals = {}
    for point in data["points"]:
        ray = point["ray"]
        if ray["limit"] == "none" or ray["lam"] < (
            1e-2 if ray["limit"] == "soft" else 1e-3
        ):
            continue
        reals.setdefault(point["real"], []).append(point)
    rays = {}
    for real, points in reals.items():
        out = reference_weights(
            real_integrand(run, data["reals"][real]["pdgs"]), points
        )
        ratios = np.abs(out["real_weight"] + out["dipole_weights"].sum(1))
        ratios /= np.abs(out["real_weight"])
        for point, ratio in zip(points, ratios):
            ray = point["ray"]
            key = (real, ray["dipole"], ray["limit"], ray["random"][-1])
            rays.setdefault(key, []).append((ray["lam"], ratio))
    assert len(rays) > 20
    for key, values in rays.items():
        ratios = [r for _, r in sorted(values, reverse=True)]
        assert ratios[-1] < max(ratios[-2] / 3, 1e-6), (key, ratios)
        assert ratios[-1] < 1e-2, (key, ratios)


@needs_outputs
def test_technical_cut(real_stream):
    """An event with 2 p_i . p_j below the technical cut times s_hat for a
    pair of its dipoles has neither the real emission nor the dipoles: the
    soft gluons of 1e-6 GeV of the reference, where the real of the Fortran
    is off, and none of the events away from the limits."""
    data = dipole_reference("pp_ttx")
    run = real_stream(1e-6)
    points = [p for p in data["points"] if p["real"] == 0]
    integrand = real_integrand(run, data["reals"][0]["pdgs"])
    out = reference_weights(integrand, points)
    momenta = np.array([p["momenta"] for p in points])
    pairs = integrand.pairs
    invariants = 2 * np.abs(
        minkowski(momenta[:, pairs[:, 0]], momenta[:, pairs[:, 1]])
    ).min(1)
    s_hat = minkowski(momenta[:, 0] + momenta[:, 1], momenta[:, 0] + momenta[:, 1])
    cut = invariants < 1e-6 * s_hat
    soft = np.array(
        [p["ray"]["limit"] == "soft" and p["ray"]["lam"] == 1e-4 for p in points]
    )
    none = np.array([p["ray"]["limit"] == "none" for p in points])
    assert cut[soft].all() and not cut[none].any()
    assert (out["real_weight"][cut] == 0).all()
    assert (out["dipole_weights"][cut] == 0).all()
    assert (out["real_weight"][none] != 0).all()
