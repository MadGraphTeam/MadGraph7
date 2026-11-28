"""The FKS integrand of a channel through the C++ event generator (FKSIntegrand,
NLOIterationState, EventGenerator.integrate):

- the integrand of its first iteration (equally likely configurations, the
  virtual on every point, no ratios) against FKSChannelIntegrand at the
  same random numbers and configurations, bit for bit;
- the adaptation of the state after an iteration against the rules of the
  Python driver that came first (DiscreteChoice and VirtualTrick of
  nlo_reference) on the same points, and the globals it writes;
- the results of the state against the sums of the points;
- a run with the scale of the user against one at the same fixed scale;
- the refusal of the survey and the generation of a fixed-order channel.

The runs are those of test_fks_cross_section.py on the aMC@NLO outputs of
MADSPACE_FKS_OUTPUTS, which runs them through the generator against
MG5aMC.
"""

import contextlib
import math
import os
import tempfile

import numpy as np
import pytest
from fks_testing import fks_output
from madgraph.mg7.nlo.fks import FKSRun
from nlo_reference import DiscreteChoice, VirtualTrick
from test_fks_cross_section import reference

import madspace as ms

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
pytestmark = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)
DATA = os.path.join(os.path.dirname(__file__), "test_data", "fks_cross_section")
POINTS = 600
# the batch size argument of an integrand
BATCH = np.array([POINTS], dtype=np.int64)
SIGMAS = 4.0


def run_of(name, **options):
    data = reference(os.path.join(DATA, f"{name}.json"))
    return data, FKSRun(
        fks_output(data["output"]),
        run_card=data["run_card"],
        param_card=data["param_card"],
        run_path=os.path.join(tempfile.gettempdir(), "madspace_fks_generator", name),
        seed=4711,
        **options,
    )


@contextlib.contextmanager
def fks_setup(name):
    """The run of a reference, set up, and closed afterwards."""
    _, run = run_of(name)
    run.setup()
    try:
        yield run
    finally:
        run.close()


def pull(ours, theirs):
    return (ours["sigma"] - theirs["sigma"]) / math.hypot(
        ours["error"], theirs["error"]
    )


def outputs_by_name(generator, values):
    return dict(zip(generator.return_types().keys(), [np.asarray(v) for v in values]))


@pytest.mark.parametrize("name", ["pp_ttx", "epem_jjj", "pp_epem_qed", "t_bw"])
def test_first_iteration(name):
    """At its first iteration the integrand without a VEGAS grid is
    FKSChannelIntegrand at its random numbers and the configuration it
    draws, with the weight of its choice, the virtual on every point that
    takes it and no ratios."""
    with fks_setup(name) as run:
        context = run.context
        for integrand in run.integrands:
            for channel in range(integrand.channel_count):
                channel_integrand = integrand.channel_integrand(channel, context)
                generator = ms.FKSIntegrand(
                    channel_integrand, None, f"first.{integrand.name}.{channel}"
                )
                generator.initialize_globals(context)
                out = outputs_by_name(
                    generator, ms.FunctionRuntime(generator.function(), context)(BATCH)
                )
                count = integrand.choice_count(channel)
                np.testing.assert_allclose(
                    out["configuration_weight"], count, rtol=1e-14
                )
                assert set(np.unique(out["fks_index"])) <= set(range(count))
                orders = channel_integrand.nlo_order_count()
                expected = ms.FunctionRuntime(channel_integrand.function(), context)(
                    out["random"],
                    out["fks_index"].astype(np.int32),
                    out["configuration_weight"],
                    np.ones(POINTS),
                    np.zeros((POINTS, orders)),
                    np.ones(POINTS),
                )
                expected = outputs_by_name(channel_integrand, expected)
                np.testing.assert_array_equal(out["weight"], expected["cross_section"])
                for key in ("lo_weight", "virtual_residual", "event_weight"):
                    np.testing.assert_array_equal(out[key], expected[key])
                for key in ("virtual", "born", "evaluated", "momenta", "born_momenta"):
                    np.testing.assert_array_equal(out[key], expected[key])


def test_adaptation():
    """After an iteration the state takes the probabilities of DiscreteChoice,
    the ratios and fraction of VirtualTrick on the same points, writes them
    into the globals of the context, and its results are the sums of the
    points of the iterations since its reset, each with its own variance."""
    with fks_setup("pp_ttx") as run:
        context = run.context
        integrand = run.integrands[0]
        generator = integrand.generator_integrand(
            0, context, "adapt", 0.5, 0.005, 50, None, None
        )
        state = ms.NLOIterationState(
            generator.return_types(),
            generator.probability_name(),
            generator.fraction_name(),
            generator.ratio_name(),
            integrand.choice_count(0),
            0.5,
            0.005,
            0.1,
        )
        runtime = ms.FunctionRuntime(generator.function(), context)
        choice = DiscreteChoice(integrand.choice_count(0))
        trick = VirtualTrick(integrand.nlo_count, 0.5, 0.005)
        iterations = []
        for _ in range(2):
            values = runtime(BATCH)
            out = outputs_by_name(generator, values)
            state.add(values)
            state.end_iteration([context])
            choice.add(
                out["fks_index"],
                (out["weight"] - out["nbody_weight"]) / out["configuration_weight"],
            )
            choice.adapt()
            trick.add(out["virtual"], out["born"], out["evaluated"] != 0)
            trick.adapt(
                out["virtual_residual"].std() / math.sqrt(POINTS),
                out["weight"].std() / math.sqrt(POINTS),
            )
            np.testing.assert_allclose(
                state.configuration_probabilities(), choice.probabilities, rtol=1e-12
            )
            np.testing.assert_allclose(state.virtual_ratios(), trick.ratios, rtol=1e-12)
            assert state.virtual_fraction() == pytest.approx(trick.fraction, rel=1e-12)
            probabilities = np.from_dlpack(
                context.get_global(generator.probability_name())
            )
            np.testing.assert_array_equal(
                probabilities.reshape(-1), state.configuration_probabilities()
            )
            fraction = np.from_dlpack(context.get_global(generator.fraction_name()))
            assert fraction.reshape(-1)[0] == state.virtual_fraction()
            iterations.append(np.vstack([out["weight"], out["lo_weight"]]))
        # the sums of both iterations, the variance of each about its mean
        sums = sum(v.sum(1) for v in iterations)
        covariance = sum(
            v @ v.T - np.outer(v.sum(1), v.sum(1)) / v.shape[1] for v in iterations
        )
        count = 2 * POINTS
        assert state.count() == count
        for coefficients in ([1.0], [0.0, 1.0], [1.0, -2.0]):
            c = np.zeros(2)
            c[: len(coefficients)] = coefficients
            mean, error = state.combination(coefficients)
            assert mean == pytest.approx(c @ sums / count, rel=1e-12)
            assert error == pytest.approx(
                math.sqrt(c @ covariance @ c) / count, rel=1e-9
            )
        state.reset()
        assert state.count() == 0
        assert state.combination([1.0]) == (0.0, 0.0)


def test_user_scale():
    """With the scale of the user (dynamical_scale_choice 10, the fixed scale
    muR_ref_fixed of user_dynamical_scale of the output, from its scale
    library on the threads of the event generator) a run agrees with one at
    that fixed scale for mu_R and mu_F."""
    results = []
    data = reference(os.path.join(DATA, "pp_epem.json"))
    for name, scales in (
        ("user", {"dynamical_scale_choice": 10}),
        ("fixed", {"fixed_ren_scale": True, "fixed_fac_scale": True}),
    ):
        run = FKSRun(
            fks_output(data["output"]),
            run_card={**data["run_card"], **scales},
            param_card=data["param_card"],
            run_path=os.path.join(
                tempfile.gettempdir(), "madspace_fks_generator", f"scale_{name}"
            ),
            seed=4711,
            accuracy=0.0,
            grid_points=2000,
            grid_iterations=3,
            points=4000,
            max_iterations=3,
            threads=2,
        )
        results.append(run.run())
    user, fixed = results
    for key in ("lo", "nlo"):
        assert abs(pull(user[key], fixed[key])) < SIGMAS, key


def test_discrete_choice():
    """DiscreteChoice takes probabilities proportional to the mean absolute
    integrand of each option, mixed with the uniform choice by its floor, as
    MC_integer of MadFKS, unseen options with the floor alone."""
    choice = DiscreteChoice(4, floor=0.2)
    choice.add(np.array([0, 0, 1, 1, 2]), np.array([3.0, -1.0, 0.5, -0.5, 6.0]))
    choice.adapt()
    target = np.array([2.0, 0.5, 6.0, 0.0]) / 8.5
    np.testing.assert_allclose(choice.probabilities, 0.8 * target + 0.05, rtol=1e-15)
    # without points the probabilities stay
    choice.adapt()
    np.testing.assert_allclose(choice.probabilities, 0.8 * target + 0.05, rtol=1e-15)


def test_refusals():
    """The survey refuses a fixed-order channel."""
    with fks_setup("epem_jj_fixed") as run:
        integrand = run.integrands[0]
        generator = integrand.generator_integrand(
            0, run.context, "refuse", 1.0, 0.005, 50, None, None
        )
        assert not generator.unweighted()
        channel = ms.ChannelEventGenerator(
            contexts=[run.context],
            integrand=generator,
            event_file="",
            weight_file="",
            config=ms.GeneratorConfig(),
            subprocess_index=0,
            name="refuse",
            histograms=None,
        )
        assert not channel.unweighted()
        event_generator = ms.EventGenerator([run.context], [channel], seed=1)
        with pytest.raises(ValueError, match="fixed order"):
            event_generator.survey()
        with pytest.raises(ValueError, match="fixed order"):
            event_generator.generate()
