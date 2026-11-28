"""The functions of the fixed-order NLO runs for Python (NLOFunctions of
madgraph/mg7/nlo/functions.py) on the aMC@NLO outputs of the references of
test_fks_cross_section.py (MADSPACE_FKS_OUTPUTS):

- the channel integrands at given random numbers, with their defaults the
  weights of the points without the virtual trick, against the C++
  integrands called with every input, for FKS and both streams of CS;
- the generator integrands, their weights at the first iteration those of
  the channel integrand at the random numbers and configurations they
  draw, with a flow for a MadNIS training;
- the matrix elements of a subprocess against the blocks of the provider
  (fks_blocks), the Born, the real, the loop and the spin correlation;
- the refusals of unknown, missing and doubled inputs and of unknown
  subprocesses.
"""

import os
import tempfile

import numpy as np
import pytest
from fks_testing import fks_output
from madgraph.mg7.nlo.cs import CSBornIntegrand, CSRun
from madgraph.mg7.nlo.fks import FKSRun
from madgraph.mg7.nlo.functions import Function, NLOFunctions
from madgraph.mg7.nlo.provider import boost_z, partonic_rapidity
from test_fks_cross_section import reference

import madspace as ms

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
pytestmark = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)
DATA = os.path.join(os.path.dirname(__file__), "test_data", "fks_cross_section")
POINTS = 200
# the parts of the FKS weight
PARTS = (
    "event_weight",
    "soft_weight",
    "collinear_weight",
    "soft_collinear_weight",
    "nbody_weight",
)


def functions_of(run_class, name, **options):
    data = reference(os.path.join(DATA, f"{name}.json"))
    run = run_class(
        fks_output(data["output"]),
        run_card=data["run_card"],
        param_card=data["param_card"],
        run_path=os.path.join(tempfile.gettempdir(), "madspace_nlo_functions", name),
        seed=4711,
        **options,
    )
    return NLOFunctions(run)


def explicit(function, context, *args):
    names = list(function.return_types().keys())
    values = ms.FunctionRuntime(function.function(), context)(*args)
    return dict(zip(names, [np.asarray(v) for v in values]))


@pytest.mark.parametrize("name", ["pp_ttx", "epem_jjj"])
def test_fks_channel_integrand(name):
    """The FKS channel integrand takes the random numbers and the
    configurations, the weight of the choice one and the virtual on every
    point without the trick by default."""
    with functions_of(FKSRun, name) as nlo:
        rng = np.random.default_rng(3)
        for channel in nlo.channels[:2]:
            weights = nlo.channel_integrand(channel)
            assert weights.inputs[:2] == ["random", "fks_index"]
            x = rng.random((POINTS, channel.dimension))
            k = rng.integers(0, channel.choices, POINTS).astype(np.int32)
            ours = weights(random=x, fks_index=k)
            assert set(weights.outputs) == set(ours)
            orders = weights.function.nlo_order_count()
            theirs = explicit(
                weights.function,
                nlo.context,
                x,
                k,
                np.ones(POINTS),
                np.ones(POINTS),
                np.zeros((POINTS, orders)),
                np.ones(POINTS),
            )
            for key, value in theirs.items():
                np.testing.assert_array_equal(ours[key], value, err_msg=key)
            # by position as well, the configurations of numpy as int64
            again = weights(x, k.astype(np.int64))
            np.testing.assert_array_equal(again["cross_section"], ours["cross_section"])
            assert np.count_nonzero(ours["cross_section"]) > POINTS // 10


def test_cs_channel_integrands():
    """The channel integrands of both streams of CS, the n-body stream with
    the virtual on every point by default."""
    with functions_of(CSRun, "pp_epem") as nlo:
        rng = np.random.default_rng(5)
        streams = set()
        for channel in nlo.channels:
            integrand = nlo.run.channels[channel.number].integrand
            born = isinstance(integrand, CSBornIntegrand)
            if born in streams:
                continue
            streams.add(born)
            weights = nlo.channel_integrand(channel)
            x = rng.random((POINTS, channel.dimension))
            ours = weights(random=x)
            if born:
                orders = len(integrand.nlo_powers)
                args = (x, np.ones(POINTS), np.zeros((POINTS, orders)), np.ones(POINTS))
            else:
                args = (x,)
            theirs = explicit(weights.function, nlo.context, *args)
            for key, value in theirs.items():
                np.testing.assert_array_equal(ours[key], value, err_msg=key)
            assert np.count_nonzero(ours["cross_section"]) > POINTS // 10
        assert streams == {False, True}


def test_generator_integrand():
    """The generator integrand of a channel draws its random numbers and
    configurations, its weights those of the channel integrand there with
    the weight of the choice, and with a flow it gives the outputs of a
    MadNIS training."""
    with functions_of(FKSRun, "pp_ttx") as nlo:
        channel = nlo.channels[0]
        sample = nlo.generator_integrand(channel)
        out = sample(POINTS)
        assert len(out["weight"]) == POINTS
        weights = nlo.channel_integrand(channel)
        fraction = nlo.run.fks_params["VirtualFraction"]
        expected = weights(
            random=out["random"],
            fks_index=out["fks_index"].astype(np.int32),
            configuration_weight=out["configuration_weight"],
            virtual_mask=out["evaluated"],
            virtual_fraction=np.full(POINTS, fraction),
        )
        # at the points the uniform grid maps the random numbers to, which
        # they are to its rounding, with a Jacobian of one: to 1e-7 of the
        # sum of the absolute values of the parts of the weight, which cancel
        parts = sum(np.abs(expected[name]) for name in PARTS)
        assert (np.abs(out["weight"] - expected["cross_section"]) <= 1e-7 * parts).all()
        flow = ms.Flow(input_dim=channel.dimension, prefix="functions_test.flow")
        flow.initialize_globals(nlo.context, 1)
        training = nlo.generator_integrand(channel, flow=flow, madnis_training=True)
        out = training(POINTS)
        assert list(out)[:4] == ["full_weight", "weight", "latent", "adaptive_prob"]
        assert isinstance(training.function, ms.FKSIntegrand)


def test_matrix_elements():
    """The matrix elements of a subprocess at Born and real kinematics of
    the channel integrand are the blocks of the provider there: the Born per
    order and amp2 of its diagrams, the real of each point, the virtual at
    the renormalization scale of the input."""
    with functions_of(FKSRun, "pp_ttx") as nlo:
        channel = nlo.channels[0]
        integrand = nlo.run.channels[0].integrand
        provider = integrand.provider
        weights = nlo.channel_integrand(channel)
        rng = np.random.default_rng(7)
        x = rng.random((POINTS, channel.dimension))
        k = np.zeros(POINTS, dtype=np.int32)
        kinematics = weights(random=x, fks_index=k)
        blocks = nlo.matrix_elements(channel.subprocess)
        assert set(blocks) == {"born", "born_spin", "reals", "loop", "extras"}
        born = kinematics["born_momenta"]
        valid = (kinematics["born"] != 0).any(axis=1)
        born = born[valid]
        cm = boost_z(born, -partonic_rapidity(born, 2))
        per_position, _, amp2 = provider.born(cm, np.zeros(len(born), dtype=int))
        ours = blocks["born"](momenta=born)
        np.testing.assert_allclose(
            ours["squared_orders"], provider.born_orders_of(per_position), rtol=1e-13
        )
        np.testing.assert_allclose(ours["diagram_amp2"], amp2, rtol=1e-13)
        real = kinematics["momenta"][kinematics["event_weight"] != 0]
        cm = boost_z(real, -partonic_rapidity(real, 2))
        r = integrand.configurations["configurations"][0]["real_me"] - 1
        ours = blocks["reals"][r](momenta=real)
        theirs = provider.real_orders(cm, np.zeros(len(real), dtype=int))
        np.testing.assert_allclose(ours["squared_orders"], theirs, rtol=1e-13)
        scale = np.full(8, 173.0)
        ours = blocks["loop"](momenta=born[:8], scale=scale)
        cm = boost_z(born[:8], -partonic_rapidity(born[:8], 2))
        theirs = provider.virtual_orders(cm, scale)
        np.testing.assert_allclose(ours["loop"][..., 0], theirs, rtol=1e-12)


def test_refusals():
    """A Function refuses unknown, missing and doubled inputs, and the
    matrix elements an unknown subprocess."""
    with functions_of(FKSRun, "epem_jj_fixed") as nlo:
        weights = nlo.channel_integrand(0)
        x = np.zeros((2, nlo.channels[0].dimension))
        with pytest.raises(TypeError, match="no input"):
            weights(random=x, fks_index=np.zeros(2, np.int32), momenta=x)
        with pytest.raises(TypeError, match="fks_index missing"):
            weights(random=x)
        with pytest.raises(TypeError, match="given twice"):
            weights(x, random=x)
        with pytest.raises(KeyError, match="no subprocess"):
            nlo.matrix_elements("P0_none")
    assert isinstance(weights, Function)
