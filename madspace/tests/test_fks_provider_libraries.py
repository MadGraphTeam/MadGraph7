"""The Fortran provider as UMAMI libraries (madgraph/mg7/nlo/libraries.py,
madspace.PythonFunction with the provisional NLO keys of umami.h), called
through MatrixElement, against the blocks that the FKS driver takes from the
provider (fks_blocks) at the kinematics of FKSKinematics: random points of
the first channel of every subprocess with random FKS configurations. The
driver asks the provider for the Born and the real with the configuration of
each point, the libraries with one configuration per spin leg and per real
matrix element (they have no FKS rows), and the provider evaluates in the
frame of the configuration: the Born, its amp2, its colour and spin
correlations and the reals agree to the rounding of that frame, the spin
correlation of a gluon or photon mother with k_perp also to that of the
square of the transverse part of k_perp (one), and the virtual, which takes
no configuration, bit for bit. The colour correlations come as the pairs of
coloured Born legs, and the coupling g_ew = e of the model gives the blocks as
the provider has them, g_ew = 1 without e. The blocks of the Sudakov
approximation of the EW corrections agree bit for bit.

The runs are those of test_fks_cross_section.py, set up and not run, on the
aMC@NLO outputs of MADSPACE_FKS_OUTPUTS.
"""

import contextlib
import os
import tempfile

import numpy as np
import pytest
from fks_testing import fks_output
from madgraph.mg7.nlo.fks import FKSRun
from madgraph.mg7.nlo.libraries import (
    ALPHA_S,
    ProviderLibraries,
    partonic_frame,
    transverse_square,
)
from madgraph.mg7.nlo.provider import boost_z, fks_blocks
from test_fks_cross_section import reference

import madspace as ms

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
pytestmark = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)
DATA = os.path.join(os.path.dirname(__file__), "test_data", "fks_cross_section")
# gluon mothers in the initial and final state (p p > t t~, e+ e- > j j j),
# photon mothers (p p > e+ e- [QED]), a 2 > 1 process and a decay
NAMES = ["pp_ttx", "epem_jjj", "pp_epem_qed", "pp_wp", "t_bw"]
POINTS = 48
# relative to the largest value of a block
RTOL = 2e-15

ME = ms.MatrixElement


@contextlib.contextmanager
def fks_setup(name, own_cards=False):
    """The run of the driver on the output of a reference, or on an output
    with its own cards, set up, and closed afterwards with the processes of
    its providers."""
    if own_cards:
        output, cards = name, {}
    else:
        data = reference(os.path.join(DATA, f"{name}.json"))
        output = data["output"]
        cards = dict(run_card=data["run_card"], param_card=data["param_card"])
    run = FKSRun(
        fks_output(output),
        run_path=os.path.join(tempfile.gettempdir(), "madspace_fks_libraries", name),
        seed=4711,
        **cards,
    )
    run.setup()
    try:
        yield run
    finally:
        run.close()


@pytest.fixture(scope="module")
def context():
    # several threads, so that the lock of the provider serializes the calls
    return ms.Context(device=ms.cpu_device(), thread_count=4)


def kinematics(integrand, rng):
    """FKSKinematics and k_perp of the first channel at random points and
    configurations, and the renormalization scale of the Born."""
    x = rng.random((POINTS, integrand.dimension))
    fks_index = rng.integers(0, integrand.configuration_count, POINTS).astype(np.int32)
    out = integrand.kinematics[0](x, fks_index=fks_index)
    if integrand.resonance and "resonance_momentum" not in out:
        out["resonance_momentum"] = np.zeros((POINTS, 4))
    resonance = [out["resonance_momentum"]] if integrand.resonance else []
    kernels = integrand.kernels(
        out["collinear_momenta"],
        out["p_i_hat"],
        out["born_momenta"],
        *resonance,
        fks_index=fks_index,
    )
    ren_scale = integrand.born_scale(out["born_momenta"])["ren_scale"]
    return out, fks_index, kernels["k_perp"], ren_scale


def call(context, api, inputs, outputs, *values):
    """The outputs of the library through MatrixElement, a list."""
    element = ME(api, inputs, outputs)
    result = ms.FunctionRuntime(element.function(), context)(*values)
    return [result] if len(outputs) == 1 else result


def assert_close(ours, theirs, rtol=0.0):
    ours, theirs = np.asarray(ours), np.asarray(theirs)
    assert ours.shape == theirs.shape
    if rtol == 0:
        np.testing.assert_array_equal(ours, theirs)
    else:
        scale = np.abs(theirs).max() if theirs.size else 0.0
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=rtol * scale)


@pytest.mark.parametrize("name", NAMES)
def test_libraries_against_blocks(name, context):
    with fks_setup(name) as run:
        rng = np.random.default_rng(17)
        for integrand in run.integrands:
            provider = integrand.provider
            if integrand.user_scale:
                pytest.skip("the scale of the user is not a library")
            libraries = ProviderLibraries(provider, integrand.name)
            apis = libraries.add_to(context)
            out, fks_index, k_perp, ren_scale = kinematics(integrand, rng)
            blocks = fks_blocks(provider, out, fks_index, k_perp, ren_scale)
            b = libraries.born_orders

            # the Born at the points with a Born, the spin leg the mother of a
            # gluon or photon configuration
            valid = np.flatnonzero(np.asarray(out["born_weight"]) != 0)
            assert len(valid)
            born = np.asarray(out["born_momenta"])[valid]
            legs = np.full((len(valid), 2), -1, dtype=np.int32)
            for k, c in enumerate(fks_index[valid]):
                configuration = provider.configurations[c]
                i_fks, j_fks = configuration["i_fks"], configuration["j_fks"]
                leg = j_fks - 1 - (j_fks > i_fks)
                if leg in libraries.spin_configuration and configuration["ij_pdg"] in (
                    21,
                    22,
                ):
                    legs[k, 0] = leg
            vectors = np.zeros((len(valid), 2, 4))
            vectors[:, 0] = np.asarray(k_perp)[valid]
            g_ew = np.full(len(valid), libraries.e)
            born_inputs = [
                ME.momenta_in,
                ME.alpha_s_in,
                ME.g_ew_in,
                ME.spin_legs_in,
                ME.spin_vectors_in,
            ]
            # the colour correlations with two coloured legs at least
            born_outputs = [
                ME.squared_orders_out,
                ME.diagram_amp2_out,
                ME.spin_correlated_out,
            ]
            if libraries.has_color:
                born_outputs.append(ME.color_correlated_out)
            orders, amp2, spin, *color = call(
                context,
                apis["born"],
                born_inputs,
                born_outputs,
                born,
                np.full(len(valid), ALPHA_S),
                g_ew,
                legs,
                vectors,
            )
            assert_close(orders, blocks["born"][valid, :b], RTOL)
            assert_close(amp2, blocks["amp2"][valid], RTOL)
            # the pairs i < j of the coloured legs, zero without them
            colored = [
                leg for leg, r in enumerate(libraries.color_representations) if r != 1
            ]
            full = blocks["color_correlated"][valid]
            if libraries.has_color:
                pairs = np.stack(
                    [
                        full[:, :, a, c]
                        for k, a in enumerate(colored)
                        for c in colored[k + 1 :]
                    ],
                    axis=-1,
                )
                assert_close(color[0], pairs, RTOL)
            else:
                assert (full == 0).all()
            spin = np.asarray(spin)
            spin_leg = legs[:, 0] >= 0
            assert (spin[~spin_leg] == 0).all()
            if spin_leg.any():
                mothers = born[spin_leg, legs[spin_leg, 0]]
                transverse = transverse_square(mothers, vectors[spin_leg, 0])
                # the provider gives B / 2 where k_perp has no transverse part
                has = transverse > 0
                assert_close(
                    spin[spin_leg][has],
                    blocks["spin_correlated"][valid][spin_leg][has, :b],
                    RTOL,
                )

            # the reals of the events with a weight, per real matrix element
            events = np.flatnonzero(np.asarray(out["weight"]) != 0)
            real_me = np.array(
                [provider.configurations[c]["real_me"] - 1 for c in fks_index[events]]
            )
            for r in np.unique(real_me):
                points = events[real_me == r]
                (real,) = call(
                    context,
                    apis["reals"][r],
                    [ME.momenta_in, ME.alpha_s_in, ME.g_ew_in],
                    [ME.squared_orders_out],
                    np.asarray(out["momenta"])[points],
                    np.full(len(points), ALPHA_S),
                    np.full(len(points), libraries.e),
                )
                assert_close(real, blocks["matrix_element"][points], RTOL)

            # the virtual at the renormalization scale of the Born
            (loop,) = call(
                context,
                apis["loop"],
                [ME.momenta_in, ME.alpha_s_in, ME.g_ew_in, ME.scale_in],
                [ME.loop_out],
                born,
                np.full(len(valid), ALPHA_S),
                g_ew,
                np.asarray(ren_scale)[valid],
            )
            loop = np.asarray(loop)
            assert loop.shape == (len(valid), libraries.nlo_orders, 3)
            assert_close(loop[..., 0], blocks["virtual"][valid])

            # g_ew = 1 gives the Born without its powers of e
            assert provider.qed_pos >= 0
            powers = np.array([o[provider.qed_pos] for o in provider.orders["born"]])
            (unit,) = call(
                context,
                apis["born"],
                born_inputs,
                [ME.squared_orders_out],
                born,
                np.full(len(valid), ALPHA_S),
                np.ones(len(valid)),
                legs,
                vectors,
            )
            assert_close(np.asarray(unit) * libraries.e**powers, orders, RTOL)


def test_extra_counterterm_library(context):
    """g u > u a [QCD QED]: the library of the Born a u > u a of the extra
    counterterm gives the columns of the Born inputs of the provider for the
    configurations with it, the Born and B_perp at the mother of the
    configuration, at g_ew = e and to the rounding of the transverse
    square of k_perp."""
    with fks_setup("gu_ua_mix", own_cards=True) as run:
        integrand = run.integrands[0]
        provider = integrand.provider
        libraries = ProviderLibraries(provider, integrand.name)
        apis = libraries.add_to(context)
        assert len(apis["extras"]) == 1
        rng = np.random.default_rng(5)
        out, fks_index, k_perp, _ = kinematics(integrand, rng)
        extra = np.asarray(libraries.extra_indices())[fks_index] >= 0
        valid = np.flatnonzero((np.asarray(out["born_weight"]) != 0) & extra)
        assert len(valid) > 4
        born = np.asarray(out["born_momenta"])[valid]
        legs = np.full((len(valid), 2), -1, dtype=np.int32)
        legs[:, 0] = np.asarray(libraries.extra_spin_legs())[fks_index[valid]]
        vectors = np.zeros((len(valid), 2, 4))
        vectors[:, 0] = np.asarray(k_perp)[valid]
        ours = call(
            context,
            apis["extras"][0],
            [
                ME.momenta_in,
                ME.alpha_s_in,
                ME.g_ew_in,
                ME.spin_legs_in,
                ME.spin_vectors_in,
            ],
            [ME.squared_orders_out, ME.spin_correlated_out],
            born,
            np.full(len(valid), ALPHA_S),
            np.full(len(valid), libraries.e),
            legs,
            vectors,
        )
        b = len(provider.orders["born"])
        partonic, y = partonic_frame(born, provider.incoming)
        direction = boost_z(np.asarray(k_perp)[valid][:, None], y)[:, 0]
        plain, perp = provider.born_inputs(partonic, direction, fks_index[valid])
        assert np.abs(plain[:, b:]).max() > 0
        assert_close(ours[0], plain[:, b:], RTOL)
        assert_close(ours[1], perp[:, b:], RTOL)


def test_ew_sudakov_library(context):
    """With --ewsudakov the library of the blocks of the Sudakov
    approximation gives those of the provider per helicity combination at the
    Born, the interferences as real and imaginary parts, at g_ew = e only, and
    without --ewsudakov there is none."""
    with fks_setup("epem_ttx_sdk") as run:
        integrand = run.integrands[0]
        provider = integrand.provider
        libraries = ProviderLibraries(provider, integrand.name)
        apis = libraries.add_to(context)
        info = provider.ew_sudakov_info()
        assert apis["ew_sudakov"].helicity_count() == info["helicity_count"]
        assert apis["ew_sudakov"].ew_sudakov_interference_count() == max(
            info["interference_count"], 1
        )
        rng = np.random.default_rng(9)
        out, _, _, _ = kinematics(integrand, rng)
        valid = np.flatnonzero(np.asarray(out["born_weight"]) != 0)
        born = np.asarray(out["born_momenta"])[valid]
        count = len(valid)
        outputs = [
            ME.ew_sudakov_diagonal_out,
            ME.ew_sudakov_lo2_out,
            ME.ew_sudakov_derivatives_out,
            ME.ew_sudakov_interference_out,
        ]
        inputs = [ME.momenta_in, ME.alpha_s_in, ME.g_ew_in]
        ours = call(
            context,
            apis["ew_sudakov"],
            inputs,
            outputs,
            born,
            np.full(count, ALPHA_S),
            np.full(count, libraries.e),
        )
        partonic, _ = partonic_frame(born, provider.incoming)
        theirs = provider.ew_sudakov(partonic, np.zeros(count, dtype=int))
        assert np.abs(theirs["diagonal"]).max() > 0
        assert_close(ours[0], theirs["diagonal"])
        assert_close(ours[1], theirs["lo2"])
        assert_close(ours[2], theirs["derivatives"])
        t = theirs["interference"].shape[2]
        assert_close(ours[3][:, :, :t, 0], theirs["interference"].real)
        assert_close(ours[3][:, :, :t, 1], theirs["interference"].imag)
        with pytest.raises(RuntimeError, match="g_ew = e"):
            call(
                context,
                apis["ew_sudakov"],
                inputs,
                outputs,
                born,
                np.full(count, ALPHA_S),
                np.ones(count),
            )
    with fks_setup("pp_epem") as run:
        integrand = run.integrands[0]
        libraries = ProviderLibraries(integrand.provider, integrand.name)
        assert libraries.ew_sudakov is None
        assert libraries.add_to(context)["ew_sudakov"] is None


def test_scale_library(context):
    """The reference scale of the user through CustomFunction is that of
    the provider at the kinematics of the real process."""
    with fks_setup("pp_epem") as run:
        integrand = run.integrands[0]
        libraries = ProviderLibraries(integrand.provider, integrand.name)
        apis = libraries.add_to(context)
        rng = np.random.default_rng(6)
        out, _, _, _ = kinematics(integrand, rng)
        momenta = np.asarray(out["momenta"])
        n = integrand.provider.particle_count
        function = ms.CustomFunction(apis["scale"], ms.CustomFunction.scale, n)
        ours = ms.FunctionRuntime(function.function(), context)(momenta)
        theirs = integrand.provider.reference_scales(momenta)
        assert theirs.max() > 0
        for value in ours:
            np.testing.assert_array_equal(value, theirs)


def test_refusals(context):
    """Other couplings and spin legs the provider has no correlation for."""
    with fks_setup("pp_epem_qed") as run:
        integrand = next(i for i in run.integrands if "aa" not in i.name)
        libraries = ProviderLibraries(integrand.provider, integrand.name)
        apis = libraries.add_to(context)
        rng = np.random.default_rng(3)
        out, fks_index, k_perp, _ = kinematics(integrand, rng)
        valid = np.flatnonzero(np.asarray(out["born_weight"]) != 0)[:2]
        born = np.asarray(out["born_momenta"])[valid]
        inputs = [
            ME.momenta_in,
            ME.alpha_s_in,
            ME.g_ew_in,
            ME.spin_legs_in,
            ME.spin_vectors_in,
        ]
        g_ew = np.full(2, libraries.e)
        outputs = [ME.squared_orders_out]
        legs = np.full((len(valid), 2), -1, dtype=np.int32)
        vectors = np.zeros((len(valid), 2, 4))
        with pytest.raises(RuntimeError, match="g_s = 1"):
            call(
                context,
                apis["born"],
                inputs,
                outputs,
                born,
                np.full(2, 0.118),
                g_ew,
                legs,
                vectors,
            )
        # a final-state lepton is the mother of no gluon or photon configuration
        legs[:, 0] = 2
        with pytest.raises(RuntimeError, match="no spin correlation of leg 2"):
            call(
                context,
                apis["born"],
                inputs,
                outputs,
                born,
                np.full(2, ALPHA_S),
                g_ew,
                legs,
                vectors,
            )
        legs[:, 0], legs[:, 1] = -1, 0
        with pytest.raises(RuntimeError, match="one spin leg at most"):
            call(
                context,
                apis["born"],
                inputs,
                outputs,
                born,
                np.full(2, ALPHA_S),
                g_ew,
                legs,
                vectors,
            )


def test_color_representations():
    """The colour representations of the Born legs, all taken as outgoing."""
    with fks_setup("pp_ttx") as run:
        for integrand in run.integrands:
            libraries = ProviderLibraries(integrand.provider, integrand.name)
            pdgs = integrand.provider.fks_configurations["born_pdgs"]
            expected = []
            for leg, pdg in enumerate(pdgs):
                outgoing = -pdg if leg < 2 else pdg
                expected.append(8 if pdg == 21 else 3 if 0 < outgoing <= 6 else -3)
            assert libraries.color_representations == expected
            assert libraries.has_color
