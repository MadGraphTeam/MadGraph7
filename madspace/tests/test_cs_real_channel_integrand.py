"""The C++ integrand of the real stream of Catani-Seymour
(CSRealChannelIntegrand of madspace) against the Python driver that came
first (CSRealReference.evaluate of nlo_reference with the combined
channels) at the same random numbers, on the aMC@NLO outputs of the
references of test_cs_cross_section.py (MADSPACE_FKS_OUTPUTS): hadron beams
(p p > e+ e-, t t~, a a, e+ e- j, w+), lepton beams (e+ e- > j j, j j j),
elastic photons (a a > t t~), decays (t > b w+ with a massive and a
massless b), QED corrections (p p > e+ e- [QED]), dressed leptons (e+ e- >
j j and t t~ with ISR) and the scale of the user, on channels of the
dipoles and of the diagrams of the real processes, also with the channels
of the default (pairs) and of the diagrams alone (reals).

The real emissions agree bit for bit, the weights to the rounding of their
parts, whose sums are taken in another order. The spin correlations of
both take the square of the part of k_t transverse to the gluon, for which
they round alike."""

import contextlib
import os
import tempfile

import numpy as np
import pytest
from fks_testing import fks_output
from madgraph.mg7.nlo.core import Runtime
from nlo_reference import CSReferenceRun
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
    "aa_ttx",
    "pp_epem_qed",
    "epem_jj_isr",
    "epem_ttx_isr",
]
POINTS = 200
# real processes and channels per real process of each run (the first ones
# of the dipoles and of the diagrams)
REALS = 3
CHANNELS = 2
# the weights against the sum of the absolute values of their parts
RTOL = 1e-13


@contextlib.contextmanager
def cs_setup(name, channels="combined", **run_card):
    """The real stream of the Python reference (CSReferenceRun) with the
    channels (by default all of them, combined) on the output of a reference
    of MG5aMC with the entries of run_card changed, set up, and closed
    afterwards with the processes of its providers."""
    data = reference(name)
    run = CSReferenceRun(
        fks_output(data["output"]),
        run_card={**data["run_card"], **run_card},
        param_card=data["param_card"],
        run_path=os.path.join(tempfile.gettempdir(), "madspace_cs_real", name),
        seed=4711,
        streams=("real",),
        channels=channels,
    )
    run.setup()
    try:
        yield run
    finally:
        run.close()


def channels_of(integrand):
    """The first channels of the dipoles and of the diagrams."""
    dipoles = [c for c, (m, _) in enumerate(integrand.channels) if m is not None]
    diagrams = [c for c, (m, _) in enumerate(integrand.channels) if m is None]
    return dipoles[:CHANNELS] + diagrams[:CHANNELS]


def compare_with_reference(run):
    """The integrand against the reference on channels of the first real
    processes of a run."""
    checked = 0
    for integrand in run.integrands[:REALS]:
        rng = np.random.default_rng(31)
        for channel in channels_of(integrand):
            function = integrand.channel_integrand(channel, run.context)
            assert function.random_dim() == integrand.channel_dimension(channel)
            assert function.channel_count() == integrand.channel_count
            assert function.dipole_count() == len(integrand.dipoles)
            runtime = Runtime(function, run.context)
            x = rng.random((POINTS, integrand.channel_dimension(channel)))
            theirs = integrand.evaluate(channel, x, None, None, None, rng)
            ours = runtime(x)
            np.testing.assert_array_equal(ours["momenta"], theirs["momenta"])
            parts = np.abs(theirs["real_weight"]) + np.abs(
                theirs["dipole_weights"]
            ).sum(1)
            bound = RTOL * parts + 1e-300
            for a, b in (("cross_section", "nlo"), ("real_weight", "real_weight")):
                bad = np.abs(ours[a] - theirs[b]) > bound
                assert not bad.any(), (a, ours[a][bad][:5], theirs[b][bad][:5])
            bad = np.abs(ours["dipole_weights"] - theirs["dipole_weights"]) > (
                bound[:, None]
            )
            assert not bad.any(), (ours["dipole_weights"][bad][:5],)
            weighted = theirs["dipole_weights"] != 0
            np.testing.assert_array_equal(
                ours["born_momenta"][weighted], theirs["born_momenta"][weighted]
            )
            checked += int((theirs["nlo"] != 0).sum())
    assert checked > 0


@pytest.mark.parametrize("name", NAMES)
def test_against_reference(name):
    with cs_setup(name) as run:
        compare_with_reference(run)


@pytest.mark.parametrize("channels", ["pairs", "reals"])
@pytest.mark.parametrize("name", ["pp_ttx", "epem_jjj"])
def test_channel_sets(name, channels):
    """The channels of the default (the diagrams and one dipole per singular
    pair) and those of the diagrams alone, with no channel of a dipole."""
    with cs_setup(name, channels) as run:
        dipoles = sum(
            m is not None for integrand in run.integrands for m, _ in integrand.channels
        )
        assert (dipoles == 0) == (channels == "reals")
        compare_with_reference(run)


@pytest.mark.parametrize("choice", [0, 10])
def test_user_scale(choice, monkeypatch):
    """The scale of the user (dynamical_scale_choice 0 and 10) on p p > e+ e-,
    with the fixed user_dynamical_scale of the provider replaced by half the
    sum of the transverse masses in the reference and in the libraries."""
    from madgraph.mg7.nlo.provider import FortranProvider

    monkeypatch.setattr(
        FortranProvider,
        "reference_scales",
        lambda self, momenta, valid=None: half_transverse_mass(momenta, valid),
    )
    with cs_setup("pp_epem", dynamical_scale_choice=choice) as run:
        assert all(i.source.user_scale for i in run.integrands)
        compare_with_reference(run)


@pytest.mark.parametrize("name", ["pp_ttx", "epem_ttx_isr"])
def test_channel_weights(name):
    """The weights of the channels of a real process add up to one at the
    events of each channel where it has a density, also for dressed leptons,
    whose inverse maps take 1 - x."""
    with cs_setup(name) as run:
        integrand = run.integrands[0]
        functions = [
            Runtime(integrand.channel_integrand(c, run.context), run.context)
            for c in range(integrand.channel_count)
        ]
        rng = np.random.default_rng(5)
        for c in channels_of(integrand):
            x = rng.random((POINTS, integrand.channel_dimension(c)))
            momenta = functions[c](x)["momenta"]
            out = integrand.forward[c].call([np.ascontiguousarray(x)])
            fractions = [np.array(np.from_dlpack(t)) for t in out[1:-1]]
            valid = np.array(np.from_dlpack(out[-1])) != 0
            densities = integrand.densities(momenta, fractions, valid)
            total = densities.sum(1)
            weights = densities / np.where(total > 0, total, 1.0)[:, None]
            np.testing.assert_allclose(
                weights[valid & (total > 0)].sum(1), 1.0, rtol=1e-14
            )
            np.testing.assert_allclose(
                functions[c](x)["channel_weight"][valid],
                weights[valid, c],
                rtol=1e-14,
            )


def test_refusals():
    """The integrand refuses a channel outside its channels and a real
    library with another number of orders than the powers of g_s."""
    from madgraph.mg7.nlo.core import madspace

    ms = madspace()
    captured = {}
    with cs_setup("pp_epem") as run:
        integrand = run.integrands[0]
        original = ms.CSRealChannelIntegrand

        def capture(*args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            return original(*args, **kwargs)

        ms.CSRealChannelIntegrand = capture
        try:
            integrand.channel_integrand(0, run.context)
        finally:
            ms.CSRealChannelIntegrand = original
        args, kwargs = list(captured["args"]), captured["kwargs"]
        for position, value in (
            (4, integrand.channel_count),  # channel
            (10, args[10] + [2.0]),  # real_powers
        ):
            bad = list(args)
            bad[position] = value
            with pytest.raises(ValueError):
                original(*bad, **kwargs)
