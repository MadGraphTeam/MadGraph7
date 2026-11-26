"""The C++ integrand of the n-body stream of Catani-Seymour
(CSBornChannelIntegrand of madspace) against the driver in Python
(CSBornIntegrand.evaluate of madgraph/mg7/nlo/cs.py) at the same random
numbers and points with the virtual, on the aMC@NLO outputs of the references
of test_cs_cross_section.py (MADSPACE_FKS_OUTPUTS): hadron beams with K and P
(p p > e+ e-, t t~, a a, e+ e- j, w+ with a W at rest), lepton beams (e+ e-
> j j, j j j), dressed leptons (e+ e- > t t~ with ISR), elastic photons (a a
> t t~), decays (t > b w+ with a massive and a massless b), QED corrections
(p p > e+ e- [QED]) and the scale of the user.

The Born momenta agree bit for bit, the Born and the virtual to the
rounding of the frame of the provider, the weights to the rounding of their
parts: the driver takes minus the sum of the other entries of a row of the
colour matrix on its diagonal, the integrand C_m B (ColorMatrix), and the
operators I, K and P take both."""

import contextlib
import os
import tempfile

import numpy as np
import pytest
from fks_testing import fks_output
from madgraph.mg7.nlo.core import Runtime, VirtualTrick
from madgraph.mg7.nlo.cs import CSRun
from test_cs_cross_section import reference
from test_fks_channel_integrand import half_transverse_mass

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
pytestmark = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)

NAMES = [
    "epem_jj",
    "epem_jjj",
    "pp_epem",
    "pp_ttx",
    "pp_aa",
    "pp_wp",
    "pp_epemj",
    "t_bw",
    "t_bw_mb0",
    "t_bwg",
    "epem_ttx_isr",
    "aa_ttx",
    "pp_epem_qed",
]
POINTS = 160
FRACTION = 0.6
# the weights against the sum of the absolute values of the Born, the
# weight and its virtual part
RTOL = 1e-13


@contextlib.contextmanager
def cs_setup(name, **run_card):
    """The n-body stream of the driver on the output of a reference with the
    entries of run_card changed, set up, and closed afterwards with the
    processes of its providers."""
    data = reference(name)
    run = CSRun(
        fks_output(data["output"]),
        run_card={**data["run_card"], **run_card},
        param_card=data["param_card"],
        run_path=os.path.join(tempfile.gettempdir(), "madspace_cs_channel", name),
        seed=4711,
        streams=("nbody",),
    )
    run.setup()
    try:
        yield run
    finally:
        run.close()


def assert_weights(ours, theirs, scale, name):
    ours, theirs = np.asarray(ours), np.asarray(theirs)
    assert ours.shape == theirs.shape, name
    bound = RTOL * scale.reshape((-1,) + (1,) * (theirs.ndim - 1))
    bad = np.abs(ours - theirs) > bound
    assert not bad.any(), (name, ours[bad][:5], theirs[bad][:5])


def compare_with_driver(run):
    """The integrand against the driver on two channels of every subprocess
    of a run."""
    checked = 0
    for integrand in run.integrands:
        rng = np.random.default_rng(29)
        for channel in range(min(integrand.channel_count, 2)):
            function = integrand.channel_integrand(channel, run.context)
            assert function.random_dim() == integrand.channel_dimension(channel)
            assert function.convolves() == integrand.hadrons
            runtime = Runtime(function, run.context)
            x = rng.random((POINTS, integrand.channel_dimension(channel)))
            trick = VirtualTrick(integrand.nlo_count, FRACTION, 0.01)
            trick.ratios = rng.normal(size=integrand.nlo_count)
            seed = int(rng.integers(1 << 30))
            mask = np.random.default_rng(seed).random(POINTS) < FRACTION
            theirs = integrand.evaluate(
                channel, x, None, None, trick, np.random.default_rng(seed)
            )
            ours = runtime(
                x,
                mask.astype(float),
                np.tile(trick.ratios, (POINTS, 1)),
                np.full(POINTS, FRACTION),
            )
            np.testing.assert_array_equal(ours["momenta"], theirs["momenta"])
            np.testing.assert_array_equal(
                ours["evaluated"] != 0, theirs["evaluated"], "evaluated"
            )
            born_scale = np.abs(theirs["born"]).max(axis=1) + 1e-300
            bound = 1e-14 * born_scale[:, None]
            assert (np.abs(ours["born"] - theirs["born"]) <= bound).all()
            np.testing.assert_allclose(
                ours["virtual"], theirs["virtual"], rtol=1e-13, atol=0
            )
            scale = (
                np.abs(theirs["lo"])
                + np.abs(theirs["nlo"])
                + np.abs(theirs["residual"])
                + 1e-300
            )
            assert_weights(ours["cross_section"], theirs["nlo"], scale, "nlo")
            assert_weights(ours["lo_weight"], theirs["lo"], scale, "lo")
            assert_weights(
                ours["virtual_residual"], theirs["residual"], scale, "residual"
            )
            checked += int((np.asarray(ours["cross_section"]) != 0).sum())
    assert checked > 0


@pytest.mark.parametrize("name", NAMES)
def test_against_driver(name):
    with cs_setup(name) as run:
        compare_with_driver(run)


@pytest.mark.parametrize("choice", [0, 10])
def test_user_scale(choice, monkeypatch):
    """The scale of the user (dynamical_scale_choice 0 and 10) on p p > e+ e-,
    with the fixed user_dynamical_scale of the provider replaced by half the
    sum of the transverse masses in the driver and in the library."""
    from madgraph.mg7.nlo.provider import FortranProvider

    monkeypatch.setattr(
        FortranProvider,
        "reference_scales",
        lambda self, momenta, valid=None: half_transverse_mass(momenta, valid),
    )
    with cs_setup("pp_epem", dynamical_scale_choice=choice) as run:
        assert all(integrand.user_scale for integrand in run.integrands)
        compare_with_driver(run)


def test_refusals():
    """The integrand refuses pieces that do not fit the Born: another number
    of powers of g_s than Born orders, convolutions without K and P."""
    from madgraph.mg7.nlo.core import madspace

    ms = madspace()
    captured = {}
    with cs_setup("pp_epem") as run:
        integrand = run.integrands[0]
        original = ms.CSBornChannelIntegrand

        def capture(*args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            return original(*args, **kwargs)

        ms.CSBornChannelIntegrand = capture
        try:
            integrand.channel_integrand(0, run.context)
        finally:
            ms.CSBornChannelIntegrand = original
        args, kwargs = list(captured["args"]), captured["kwargs"]
        for position, value in (
            (11, args[11] + [2.0]),  # born_powers
            (17, None),  # collinear: convolutions without K and P
        ):
            bad = list(args)
            bad[position] = value
            with pytest.raises(ValueError):
                original(*bad, **kwargs)
