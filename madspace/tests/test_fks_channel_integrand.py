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
def fks_setup(name):
    """The run of the driver on the output of a reference, set up, and
    closed afterwards with the processes of its providers."""
    data = reference(os.path.join(DATA, f"{name}.json"))
    run = FKSRun(
        fks_output(data["output"]),
        run_card=data["run_card"],
        param_card=data["param_card"],
        run_path=os.path.join(tempfile.gettempdir(), "madspace_fks_channel", name),
        seed=4711,
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


@pytest.mark.parametrize("name", NAMES)
def test_against_driver(name):
    with fks_setup(name) as run:
        checked = 0
        for integrand in run.integrands:
            if integrand.user_scale or integrand.ew_sudakov is not None:
                pytest.skip("no scale of the user and no Sudakov approximation yet")
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
