"""The C++ integrand of a channel of the FKS driver (FKSChannelIntegrand of
madspace, with the Fortran provider as UMAMI libraries) against the driver
in Python (SubprocessIntegrand.evaluate of madgraph/mg7/nlo/fks.py) at the
same random numbers, FKS configurations, weights of their choice, points
with the virtual and averages and fraction of the virtual trick.

The kinematics agree bit for bit. The libraries ask the provider for the
Born and the real with one configuration per spin leg and per real matrix
element, the driver with the configuration of each point, which round apart
in the last bits (see test_fks_provider_libraries.py), and the integrand
takes C_m B on the diagonal of the colour matrix where the driver takes
minus the sum of the row (colour conservation). So the weights agree to the
rounding of their parts: per point to RTOL of the sum of the absolute
weights of the event, its counter-events and the n-body part (at most
1.3e-15 of it on these references). The scale and PDF variations agree in
the same way.

The runs are those of test_fks_cross_section.py, set up and not run, on the
aMC@NLO outputs of MADSPACE_FKS_OUTPUTS.
"""

import contextlib
import os
import tempfile

import numpy as np
import pytest
from fks_testing import fks_output
from madgraph.mg7.nlo.core import Runtime, VirtualTrick
from madgraph.mg7.nlo.fks import FKSRun
from test_fks_cross_section import reference

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
pytestmark = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)
DATA = os.path.join(os.path.dirname(__file__), "test_data", "fks_cross_section")
# initial- and final-state gluons, QED corrections, tagged photons, a 2 > 1
# process, decays with a massive and a massless b, dressed leptons with ISR
# and beamstrahlung, elastic photons, a jet against the incoming partons,
# isolated photons, scale and PDF variations
NAMES = [
    "pp_ttx",
    "epem_jjj",
    "pp_epem_qed",
    "pp_aa_tagged",
    "pp_wp",
    "t_bw",
    "t_bw_mb0",
    "t_bwg",
    "epem_ttx_isr",
    "epem_ttx_ilc",
    "aa_ttx",
    "pp_epemj",
    "pp_aa",
    "pp_epem_rw",
]
POINTS = 160
FRACTION = 0.6
RTOL = 1e-14
PARTS = (
    "event_weight",
    "soft_weight",
    "collinear_weight",
    "soft_collinear_weight",
    "nbody_weight",
)


@contextlib.contextmanager
def fks_setup(name, output=None, **run_card):
    """The run of the driver on the output of a reference with the entries
    of run_card changed, or on an output with its own cards, set up, and
    closed afterwards with the processes of its providers."""
    if output is None:
        data = reference(os.path.join(DATA, f"{name}.json"))
        cards = dict(
            run_card={**data["run_card"], **run_card}, param_card=data["param_card"]
        )
        output = data["output"]
    else:
        cards = dict(run_card=run_card or None)
    run = FKSRun(
        fks_output(output),
        run_path=os.path.join(tempfile.gettempdir(), "madspace_fks_channel", name),
        seed=4711,
        **cards,
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
        if integrand.ew_sudakov is not None:
            pytest.skip("no Sudakov approximation yet")
        rng = np.random.default_rng(23)
        for channel in range(min(integrand.channel_count, 2)):
            function = integrand.channel_integrand(channel, run.context)
            runtime = Runtime(function, run.context)
            x = rng.random((POINTS, integrand.dimension))
            fks_index = rng.integers(0, integrand.configuration_count, POINTS)
            fks_index = fks_index.astype(np.int32)
            configuration_weight = rng.uniform(0.5, 4.0, POINTS)
            trick = VirtualTrick(integrand.nlo_count, FRACTION, 0.01)
            trick.ratios = rng.normal(size=integrand.nlo_count)
            seed = int(rng.integers(1 << 30))
            mask = np.random.default_rng(seed).random(POINTS) < FRACTION
            theirs = integrand.evaluate(
                channel,
                x,
                fks_index,
                configuration_weight,
                trick,
                np.random.default_rng(seed),
            )
            ours = runtime(
                x,
                fks_index,
                configuration_weight,
                mask.astype(float),
                np.tile(trick.ratios, (POINTS, 1)),
                np.full(POINTS, FRACTION),
            )
            for kinematics in (
                "momenta",
                "soft_momenta",
                "collinear_momenta",
                "soft_collinear_momenta",
                "born_momenta",
            ):
                np.testing.assert_array_equal(
                    ours[kinematics], theirs["kinematics"][kinematics], kinematics
                )
            np.testing.assert_array_equal(
                ours["evaluated"] != 0, theirs["evaluated"], "evaluated"
            )
            scale = sum(np.abs(theirs["outputs"][part]) for part in PARTS)
            scale = scale + np.abs(theirs["lo"]) + 1e-300
            for part in PARTS:
                assert_weights(ours[part], theirs["outputs"][part], scale, part)
            assert_weights(ours["cross_section"], theirs["nlo"], scale, "nlo")
            assert_weights(ours["lo_weight"], theirs["lo"], scale, "lo")
            assert_weights(
                ours["virtual_residual"], theirs["residual"], scale, "residual"
            )
            outputs = theirs["outputs"]
            for name in ("scale_variations", "pdf_variations"):
                if name in outputs:
                    assert_weights(ours[name], outputs[name], scale, name)
            born_scale = np.abs(theirs["born"]).max(axis=1) + 1e-300
            assert_weights(ours["born"], theirs["born"], born_scale, "born")
            np.testing.assert_array_equal(ours["virtual"], theirs["virtual"])
            checked += int((np.asarray(ours["cross_section"]) != 0).sum())
    assert checked > 0


@pytest.mark.parametrize("name", NAMES)
def test_against_driver(name):
    with fks_setup(name) as run:
        compare_with_driver(run)


def half_transverse_mass(momenta, valid=None):
    """Half the sum of the transverse masses of the final-state particles
    of kinematics (batch, n + 1, 4) of the real process, zero where valid is
    false, a reference scale of the user that depends on the kinematics."""
    momenta = np.asarray(momenta, dtype=float)
    final = momenta[:, 2:]
    mass2 = final[..., 0] ** 2 - (final[..., 1:] ** 2).sum(-1)
    pt2 = final[..., 1] ** 2 + final[..., 2] ** 2
    scale = 0.5 * np.sqrt(np.maximum(mass2 + pt2, 0.0)).sum(-1)
    if valid is not None:
        scale = np.where(np.asarray(valid, dtype=bool), scale, 0.0)
    return scale


@pytest.mark.parametrize("choice", [0, 10])
def test_user_scale(choice):
    """With the scale of the user the integrand calls the reference scale
    (a scale library of the provider through CustomFunction) at the event,
    the soft and the collinear counter-events where the driver does, and the
    Born and the virtual take it at the soft counter-event. The provider of
    the outputs has the fixed scale of user_dynamical_scale, so both take
    half_transverse_mass in its place."""
    with fks_setup("pp_epem", dynamical_scale_choice=choice) as run:
        for integrand in run.integrands:
            assert integrand.user_scale
            integrand.provider.reference_scales = half_transverse_mass
        compare_with_driver(run)


def test_extra_counterterms():
    """g u > u a [QCD QED] (gu_ua_mix): configurations with a quark pair from
    a gluon and from a photon, whose extra counterterm has the Born a u > u a
    (a library of its own), with its own cards."""
    with fks_setup("gu_ua_mix", output="gu_ua_mix") as run:
        assert any(i.provider.orders.get("extra_born") for i in run.integrands)
        compare_with_driver(run)


@pytest.mark.parametrize("name", ["pp_epem", "pp_epemj"])
def test_degenerate_born(name):
    """Where the first random number of the Born comes close to zero, the
    Born of an initial-state emitter, which no cut bounds, has a momentum
    fraction down to 1e-17. Its soft counter-event lies below the lower
    bound on s_hat at xi_min > 0, boosted with 1 / sqrt(1 - xi_min), where
    its momenta can pass the cuts by rounding. The n-body part and the
    virtual take its cuts only where it exists, at xi = 0, and the weights
    are finite: the contribution weight takes s_hat from the product of the
    incoming momenta, which the square of their sum loses, and is zero where
    the cuts leave no luminosity, also where the coefficients of such a Born
    are not finite. (VEGAS had concentrated its grids on these points, with
    not-a-number weights and n-body weights up to 1e42 pb, and the loop
    stopped on their momenta.) In the channels where the first random number
    is the mass of the lepton pair, the Born mapping boosts a pair of
    1e-6 GeV from its rest frame and the Born loses momentum conservation
    (up to 4e-3). The cuts take kinematics that keep it to 1e-8 only, so the
    loop gets Borns that pass its check (1e-6, where MadLoop stops)."""
    with fks_setup(name) as run:
        rng = np.random.default_rng(5)
        below = bad_borns = 0
        for integrand in run.integrands:
            for channel in range(integrand.channel_count):
                runtime = Runtime(
                    integrand.channel_integrand(channel, run.context), run.context
                )
                x = rng.random((POINTS, integrand.dimension))
                x[:, 0] = np.geomspace(1e-15, 1e-6, POINTS)
                fks_index = rng.integers(0, integrand.configuration_count, POINTS)
                fks_index = fks_index.astype(np.int32)
                ours = runtime(
                    x,
                    fks_index,
                    np.full(POINTS, float(integrand.configuration_count)),
                    np.ones(POINTS),
                    np.zeros((POINTS, integrand.nlo_count)),
                    np.ones(POINTS),
                )
                for key, value in ours.items():
                    value = np.asarray(value)
                    if value.dtype.kind == "f":
                        assert np.isfinite(value).all(), (integrand.name, key)
                kinematics = integrand.kinematics[channel](x, fks_index=fks_index)
                soft_below = np.asarray(kinematics["soft_xi"]) > 0
                below += soft_below.sum()
                for key in ("nbody_weight", "evaluated"):
                    value = np.asarray(ours[key])[soft_below]
                    assert (value == 0).all(), (integrand.name, key)
                born = np.asarray(ours["born_momenta"])
                total = born[:, 2:].sum(1) - born[:, :2].sum(1)
                scale = np.abs(born[:, :, 0]).sum(1)[:, None]
                violation = (np.abs(total) / scale).max(1)
                evaluated = np.asarray(ours["evaluated"]) != 0
                assert (violation[evaluated] < 1e-6).all(), integrand.name
                bad_borns += (violation > 1e-6).sum()
        assert below > 0
        if name == "pp_epemj":
            assert bad_borns > 0
