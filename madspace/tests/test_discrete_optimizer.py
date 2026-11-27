"""The optimizers of the adaptive samplers: DiscreteOptimizer takes the running
average of the old probabilities and of the contributions of the categories,
VegasGridOptimizer and DiscreteOptimizer take the outputs of their
histograms from Python, and with GeneratorConfig.adaptive_discrete the
flavour choice of an integrand adapts in a survey without changing its cross
section (e+ e- -> q q~ g with a d and a u flavour of different matrix
elements, on the setup of test_event_stream)."""

import os

import numpy as np
import pytest
from test_event_stream import CM_ENERGY, TOPOLOGY, make_config, make_context

import madspace as ms

pytestmark = pytest.mark.cpu_only

# the events of the updates before the first, see DiscreteOptimizer
START_COUNT = 7000


def probabilities(context, name):
    return np.from_dlpack(context.get_global(name)).reshape(-1).copy()


def test_running_average():
    context = ms.Context(ms.cpu_device(), 1)
    sampler = ms.DiscreteSampler([4], "average")
    sampler.initialize_globals(context)
    (name,) = sampler.prob_names()
    optimizer = ms.DiscreteOptimizer([context], [name])
    # without data the probabilities stay
    optimizer.optimize()
    np.testing.assert_array_equal(probabilities(context, name), np.full(4, 0.25))

    old = np.full(4, 0.25)
    count = 0
    for sums, counts in [
        ([100.0, 200.0, 300.0, 0.0], [100, 100, 100, 0]),
        ([50.0, -400.0, 300.0, 20.0], [50, 100, 60, 10]),
    ]:
        optimizer.add_data([np.array([sums]), np.array([counts], dtype=np.int32)])
        optimizer.optimize()
        sums, counts = np.array(sums), np.array(counts)
        contributions = np.where(
            counts > 0, old * np.abs(sums) / np.maximum(counts, 1), 0
        )
        share = (START_COUNT + count) / (START_COUNT + count + counts.sum())
        expected = old * share + contributions / contributions.sum() * (1 - share)
        new = probabilities(context, name)
        np.testing.assert_allclose(new, expected, rtol=1e-14)
        assert new.sum() == pytest.approx(1.0, rel=1e-14)
        # a category without events keeps a share of its probability
        assert (new > 0).all()
        old, count = new, count + counts.sum()


def test_vegas_optimizer_takes_its_histogram():
    context = ms.Context(ms.cpu_device(), 1)
    vegas = ms.VegasMapping(2, 10, "vegas_python")
    vegas.initialize_globals(context)
    optimizer = ms.VegasGridOptimizer([context], vegas.grid_name(), 1.5)
    rng = np.random.default_rng(1)
    r = rng.random((5000, 2))
    weights = np.exp(-20 * r[:, 0])
    histogram = ms.FunctionRuntime(ms.VegasHistogram(2, 10).function(), context)
    optimizer.add_data(*histogram(r, weights))
    optimizer.optimize()
    grid = np.from_dlpack(context.get_global(vegas.grid_name()))[0]
    # the bins narrow where the weights of the first dimension are large
    edges = grid[0]
    assert edges[0] == 0.0 and edges[-1] == 1.0
    assert (np.diff(edges) > 0).all()
    assert edges[1] < 0.1


def flavor_integrand(alphas_file, context, discrete_flavor):
    """e+ e- -> d d~ g and u u~ g with the matrix elements 1 and 3."""
    pids = [[11, -11, 1, -1, 21], [11, -11, 2, -2, 21]]
    api = context.add_matrix_element(
        ms.PythonFunction(
            lambda momenta, flavor: np.where(flavor == 1, 3.0, 1.0),
            ["momenta", "flavor_index"],
            ["matrix_element"],
            particle_count=5,
        )
    )
    matrix_element = ms.MatrixElement(
        api.index(),
        5,
        [ms.MatrixElement.momenta_in, ms.MatrixElement.flavor_in],
        [ms.MatrixElement.matrix_element_out],
    )
    cuts = ms.Cuts(
        [
            ms.CutItem(ms.Observable(pids[0], ms.Observable.obs_pt, [[i]]), min=20.0)
            for i in (2, 3, 4)
        ]
    )
    mapping = ms.PhaseSpaceMapping(
        TOPOLOGY, CM_ENERGY, permutations=[[0, 1, 2, 3, 4]], leptonic=True, cuts=cuts
    )
    diff_xs = ms.DifferentialCrossSection(
        matrix_element=matrix_element,
        cm_energy=CM_ENERGY,
        running_coupling=None,
        energy_scale=ms.CachedScale(),
    )
    return ms.Integrand(
        mapping,
        [diff_xs],
        discrete_flavor=discrete_flavor,
        pid_options=pids,
        running_coupling=ms.RunningCoupling(ms.AlphaSGrid(alphas_file)),
        energy_scale=ms.EnergyScale(5, 91.188),
        channel_indices=[0],
    )


def test_adaptive_flavor_choice(tmp_path):
    """With adaptive_discrete the survey adapts the flavour probabilities
    towards the shares of the flavours (1/4 and 3/4), without it they stay,
    and the cross section is that of the uniform choice either way."""
    context, alphas_file = make_context(str(tmp_path), 2)
    statuses = []
    for setup in ("uniform", "fixed", "adaptive"):
        sampler = None
        if setup != "uniform":
            sampler = ms.DiscreteSampler([2], f"flavor_{setup}")
            sampler.initialize_globals(context)
        integrand = flavor_integrand(alphas_file, context, sampler)
        config = make_config()
        config.target_count = 2000
        assert not config.adaptive_discrete
        config.adaptive_discrete = setup == "adaptive"
        channel = ms.ChannelEventGenerator(
            contexts=[context],
            integrand=integrand,
            event_file=os.path.join(str(tmp_path), f"events_{setup}.npy"),
            weight_file=os.path.join(str(tmp_path), f"weights_{setup}.npy"),
            config=config,
            subprocess_index=0,
            name=f"flavor_{setup}",
            histograms=None,
        )
        generator = ms.EventGenerator([context], [channel], seed=7, config=config)
        generator.survey()
        generator.generate()
        statuses.append(generator.status())
        if sampler is not None:
            (name,) = sampler.prob_names()
            p = probabilities(context, name)
            if setup == "fixed":
                np.testing.assert_array_equal(p, [0.5, 0.5])
            else:
                assert p.sum() == pytest.approx(1.0, rel=1e-12)
                # moved from (1/2, 1/2) towards (1/4, 3/4)
                assert 0.25 < p[0] < 0.45
    uniform = statuses[0]
    for status in statuses[1:]:
        error = np.hypot(uniform.error, status.error)
        assert abs(uniform.mean - status.mean) < 5 * error
