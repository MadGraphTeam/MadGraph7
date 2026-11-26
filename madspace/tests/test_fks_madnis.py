"""MadNIS for the FKS integrand of a channel (FKSIntegrand with a Flow and a
DiscreteFlow, FKSIntegrandProbability, MultiMadnisTraining on FKSIntegrand):

- the outputs of the training against FKSChannelIntegrand at the random
  numbers and configurations of the training, and the density of the
  adaptive samplers against FKSIntegrandProbability;
- the state of a channel continuing from the globals of an earlier one, and
  a state that leaves the probabilities of the configurations to a
  DiscreteFlow;
- runs of FKSRun with a MadNIS training, also of a DiscreteFlow, against the
  cross sections of MG5aMC per subprocess.

The runs are those of test_fks_cross_section.py on the aMC@NLO outputs of
MADSPACE_FKS_OUTPUTS.
"""

import json
import os

import numpy as np
import pytest
from test_fks_generator import (
    BATCH,
    POINTS,
    SIGMAS,
    fks_setup,
    outputs_by_name,
    pull,
    run_of,
)

import madspace as ms

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
pytestmark = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)


def probability(integrand, context, latent, fks_index):
    function = integrand.probability().function()
    return np.asarray(ms.FunctionRuntime(function, context)(latent, fks_index))


@pytest.mark.parametrize("name", ["pp_ttx", "epem_jjj"])
def test_training_outputs(name):
    """The training gives the cross section of FKSChannelIntegrand without the
    virtual at its random numbers and configurations, its real part times
    the probability of the configuration and its n-body part over the number
    of configurations, the weight of the point (the cross section over the
    density), and the density of its samplers as FKSIntegrandProbability,
    with the probabilities of the state or of a DiscreteFlow."""
    with fks_setup(name) as run:
        context = run.context
        integrand = run.integrands[0]
        channel_integrand = integrand.channel_integrand(0, context)
        dimension = integrand.channel_dimension(0)
        count = integrand.choice_count(0)
        prefix = f"train.{name}"
        flow = ms.Flow(input_dim=dimension, prefix=f"{prefix}.flow")
        flow.initialize_globals(context, 1)
        discrete = ms.DiscreteFlow(
            option_counts=[count],
            prefix=f"{prefix}.discrete",
            dims_with_prior=[0],
            condition_dim=dimension,
        )
        discrete.initialize_globals(context, 2)
        plain = ms.FKSIntegrand(channel_integrand, flow, prefix, madnis_training=True)
        plain.initialize_globals(context)
        trained = ms.FKSIntegrand(
            channel_integrand,
            flow,
            prefix,
            discrete=discrete,
            madnis_training=True,
        )
        orders = channel_integrand.nlo_order_count()
        channel_runtime = ms.FunctionRuntime(channel_integrand.function(), context)
        for generator, uniform in [(plain, True), (trained, False)]:
            values = ms.FunctionRuntime(generator.function(), context)(BATCH)
            out = outputs_by_name(generator, values)
            assert list(out) == [
                "full_weight",
                "weight",
                "latent",
                "adaptive_prob",
                "fks_index",
            ]
            fks_index = out["fks_index"].astype(np.int32)
            density = probability(generator, context, out["latent"], fks_index)
            np.testing.assert_allclose(density, out["adaptive_prob"], rtol=1e-10)
            # the probability of the configuration: the density over that of
            # the flow, the density with equally likely configurations times
            # their number
            flow_density = count * probability(plain, context, out["latent"], fks_index)
            choice = density / flow_density
            if uniform:
                np.testing.assert_allclose(choice, 1.0 / count, rtol=1e-12)
            else:
                assert np.all(choice > 0) and np.all(choice < 1)
            expected = channel_runtime(
                out["latent"],
                fks_index,
                1.0 / choice,
                np.zeros(POINTS),
                np.zeros((POINTS, orders)),
                np.ones(POINTS),
            )
            expected = outputs_by_name(channel_integrand, expected)
            assert np.all(expected["evaluated"] == 0)
            cross_section, nbody = expected["cross_section"], expected["nbody_weight"]
            np.testing.assert_allclose(
                out["full_weight"],
                cross_section * choice / out["adaptive_prob"],
                rtol=1e-10,
            )
            # the n-body part in equal parts, the real part times p_k
            target = (cross_section - nbody) * choice + nbody / count
            # to the rounding of the difference, which can cancel, and to the
            # agreement of the cross sections (as for the full weight)
            difference = (np.abs(cross_section) + np.abs(nbody)) * choice
            scale = difference + np.abs(nbody) / count
            assert np.all(np.abs(out["weight"] - target) <= 1e-10 * scale + 1e-300)
            assert np.any(nbody != 0)
            assert np.count_nonzero(out["weight"]) > POINTS // 4


def test_training_refusals():
    """The training needs a flow, and a discrete flow has to fit."""
    with fks_setup("pp_ttx") as run:
        context = run.context
        integrand = run.integrands[0]
        channel_integrand = integrand.channel_integrand(0, context)
        dimension = integrand.channel_dimension(0)
        count = integrand.choice_count(0)
        vegas = ms.VegasMapping(dimension, 10, "refuse")
        with pytest.raises(ValueError, match="needs a flow"):
            ms.FKSIntegrand(channel_integrand, vegas, "refuse", madnis_training=True)
        flow = ms.Flow(input_dim=dimension + 1, prefix="refuse.flow")
        with pytest.raises(ValueError, match="dimension"):
            ms.FKSIntegrand(channel_integrand, flow, "refuse")
        for discrete in [
            ms.DiscreteFlow(option_counts=[count + 1], condition_dim=dimension),
            ms.DiscreteFlow(option_counts=[count], condition_dim=dimension),
            ms.DiscreteFlow(option_counts=[count], dims_with_prior=[0]),
        ]:
            with pytest.raises(ValueError, match="discrete flow"):
                ms.FKSIntegrand(channel_integrand, None, "refuse", discrete=discrete)


def test_state_continues():
    """A new state of a channel starts from the globals an earlier state left,
    and a state with a DiscreteFlow keeps the probabilities of the
    configurations."""
    with fks_setup("pp_ttx") as run:
        context = run.context
        integrand = run.integrands[0]
        generator = integrand.generator_integrand(
            0, context, "continue", 0.5, 0.005, 50, None, None
        )
        runtime = ms.FunctionRuntime(generator.function(), context)
        count = integrand.choice_count(0)

        def state(adapt=True):
            return ms.NLOIterationState(
                generator.return_types(),
                generator.probability_name(),
                generator.fraction_name(),
                generator.ratio_name(),
                count,
                0.5,
                0.005,
                0.1,
                adapt,
            )

        first = state()
        first.add(runtime(BATCH))
        first.end_iteration([context])
        probabilities = np.array(first.configuration_probabilities())
        ratios = np.array(first.virtual_ratios())
        assert not np.allclose(probabilities, 1.0 / count)
        assert np.any(ratios != 0)
        # without points the next state writes back what it took
        second = state()
        second.end_iteration([context])
        np.testing.assert_array_equal(
            second.configuration_probabilities(), probabilities
        )
        np.testing.assert_array_equal(second.virtual_ratios(), ratios)
        assert second.virtual_fraction() == first.virtual_fraction()
        third = state()
        third.read_globals(context)
        np.testing.assert_array_equal(
            third.configuration_probabilities(), probabilities
        )
        # with a DiscreteFlow the probabilities stay, the virtual trick adapts
        fixed = state(adapt=False)
        fixed.add(runtime(BATCH))
        fixed.end_iteration([context])
        np.testing.assert_array_equal(
            fixed.configuration_probabilities(), probabilities
        )
        written = np.from_dlpack(context.get_global(generator.probability_name()))
        np.testing.assert_array_equal(written.reshape(-1), probabilities)
        assert not np.array_equal(fixed.virtual_ratios(), ratios)


@pytest.mark.parametrize(
    "name, discrete", [("epem_jj_fixed", False), ("pp_ttx", False), ("pp_ttx", True)]
)
def test_against_mg5amc(name, discrete):
    """Runs with a MadNIS training between the grid and the integration
    iterations, also of a DiscreteFlow of the configurations, agree with
    MG5aMC per subprocess, and their histograms add up to the cross
    section."""
    data, run = run_of(
        name,
        accuracy=0.0,
        grid_points=2000,
        grid_iterations=3,
        points=4000,
        max_iterations=3,
        generator=True,
        threads=2,
        madnis_batches=100,
        madnis_discrete=discrete,
    )
    result = run.run()
    for key in ("lo", "nlo"):
        assert abs(pull(result[key], data[key])) < SIGMAS, (key, result[key])
        for subprocess, theirs in data[key]["subprocesses"].items():
            ours = result["subprocesses"][subprocess][key]
            assert abs(pull(ours, theirs)) < SIGMAS, (subprocess, key, ours, theirs)
    assert run.iterations == 3
    with open(os.path.join(run.run_path, "info.json")) as f:
        info = json.load(f)
    assert info["status"] == "done"
    histograms = info["histograms"]
    assert histograms
    total = sum(histograms[0]["bin_values"])
    assert total == pytest.approx(result["nlo"]["sigma"], rel=1e-9)
