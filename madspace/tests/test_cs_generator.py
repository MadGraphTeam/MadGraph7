"""The Catani-Seymour integrands of a channel through the C++ event generator
(CSIntegrand, NLOIterationState, EventGenerator.integrate) and with MadNIS
(CSIntegrand with a Flow, CSIntegrandProbability, MultiMadnisTraining):

- the integrand of the first iteration without a VEGAS grid against
  CSBornChannelIntegrand (the virtual on the points that draw it, no
  ratios) and CSRealChannelIntegrand at the same random numbers, bit for
  bit;
- the adaptation of the virtual trick of the n-body stream after an
  iteration against VirtualTrick of nlo/core.py on the same points, the
  globals it writes, and the results of the states of both streams against
  the sums of the points;
- the outputs of the training against the channel integrands without the
  virtual, and the density of the flow against CSIntegrandProbability;
- the histograms of each stream (CSObservableHistograms) against those of
  the driver in Python (CSHistograms) at the outputs of the integrand;
- runs of CSRun through the generator, also with the channels of the real
  diagrams alone, of dressed leptons and with a MadNIS training, against the
  cross sections of MG5aMC, the LO also per subprocess, with histograms that
  add up to the cross section;
- the refusals of a discrete flow, of a training without a flow, of an
  adaptive mapping of another dimension or with a condition, of fractions
  of the virtual outside (0, 1].

The runs are those of test_cs_cross_section.py on the aMC@NLO outputs of
MADSPACE_FKS_OUTPUTS.
"""

import contextlib
import json
import math
import os
import tempfile

import numpy as np
import pytest
from fks_testing import fks_output
from madgraph.mg7.nlo.core import VirtualTrick
from madgraph.mg7.nlo.cs import CSBornIntegrand, CSRun
from test_cs_cross_section import reference

import madspace as ms

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
pytestmark = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)
POINTS = 600
# the batch size argument of an integrand
BATCH = np.array([POINTS], dtype=np.int64)
SIGMAS = 4.0
FRACTION = 0.5
# the real processes of a run whose channels the tests take
REALS = 2


def run_of(name, tag=None, **options):
    data = reference(name)
    return data, CSRun(
        fks_output(data["output"]),
        run_card=data["run_card"],
        param_card=data["param_card"],
        run_path=os.path.join(
            tempfile.gettempdir(), "madspace_cs_generator", tag or name
        ),
        seed=4711,
        **options,
    )


@contextlib.contextmanager
def cs_setup(name, **options):
    """The run of a reference, set up, and closed afterwards with the
    processes of its providers."""
    _, run = run_of(name, **options)
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


def nbody(integrand):
    return isinstance(integrand, CSBornIntegrand)


def channels_of(run):
    """The first two channels of every Born and, of the first real
    processes, the first channels of a dipole and of a diagram."""
    reals = 0
    for integrand in run.integrands:
        if nbody(integrand):
            for channel in range(min(integrand.channel_count, 2)):
                yield integrand, channel
            continue
        reals += 1
        if reals > REALS:
            continue
        kinds = [m is None for m, _ in integrand.channels]
        for kind in (False, True):
            if kind in kinds:
                yield integrand, kinds.index(kind)


def expected_outputs(channel_integrand, context, random, born, mask=None):
    """The channel integrand at the random numbers, for a Born with the
    virtual where mask is not zero (none by default), no ratios and the
    fraction FRACTION."""
    runtime = ms.FunctionRuntime(channel_integrand.function(), context)
    if not born:
        return outputs_by_name(channel_integrand, runtime(random))
    count = len(random)
    orders = channel_integrand.nlo_order_count()
    values = runtime(
        random,
        np.zeros(count) if mask is None else mask,
        np.zeros((count, orders)),
        np.full(count, FRACTION),
    )
    return outputs_by_name(channel_integrand, values)


def probability(integrand, context, latent):
    function = integrand.probability().function()
    return np.asarray(ms.FunctionRuntime(function, context)(latent))


@pytest.mark.parametrize("name", ["pp_ttx", "epem_jjj", "pp_epem_qed", "t_bw"])
def test_first_iteration(name):
    """At its first iteration the integrand without a VEGAS grid is the
    channel integrand at its random numbers, for a Born with the virtual on
    the points that draw it and no ratios."""
    with cs_setup(name) as run:
        context = run.context
        streams = set()
        for integrand, channel in channels_of(run):
            born = nbody(integrand)
            streams.add(born)
            channel_integrand = integrand.channel_integrand(channel, context)
            generator = ms.CSIntegrand(
                channel_integrand,
                None,
                f"first.{integrand.name}.{channel}",
                virtual_fraction=FRACTION,
            )
            generator.initialize_globals(context)
            assert generator.nbody() == born
            assert generator.random_dim() == integrand.channel_dimension(channel)
            out = outputs_by_name(
                generator, ms.FunctionRuntime(generator.function(), context)(BATCH)
            )
            if born:
                mask = (out["evaluated"] != 0).astype(float)
                assert 0 < mask.sum() < POINTS
                keys = ("lo_weight", "virtual_residual", "virtual", "born")
                keys += ("evaluated", "momenta")
            else:
                mask = None
                keys = ("real_weight", "dipole_weights", "momenta")
            expected = expected_outputs(
                channel_integrand, context, out["random"], born, mask
            )
            np.testing.assert_array_equal(out["weight"], expected["cross_section"])
            for key in keys:
                np.testing.assert_array_equal(out[key], expected[key])
        assert streams == {False, True}


def test_adaptation():
    """After an iteration the state of the n-body stream takes the ratios
    and the fraction of VirtualTrick on the same points and writes them into
    the globals of the context, that of the real stream adapts nothing, and
    the results of both are the sums of the points of the iterations since
    their reset, each with its own variance."""
    with cs_setup("pp_ttx") as run:
        context = run.context
        born = next(i for i in run.integrands if nbody(i))
        real = next(i for i in run.integrands if not nbody(i))
        for k, integrand in enumerate((born, real)):
            generator = integrand.generator_integrand(
                0, context, f"adapt{k}", FRACTION, 0.005, 50, None, None
            )
            state = ms.NLOIterationState(
                generator.return_types(),
                "",
                generator.fraction_name() if generator.nbody() else "",
                generator.ratio_name() if generator.nbody() else "",
                1,
                FRACTION,
                0.005,
                0.1,
            )
            assert state.has_virtual() == generator.nbody()
            assert not state.has_choice()
            assert list(state.configuration_probabilities()) == []
            runtime = ms.FunctionRuntime(generator.function(), context)
            trick = VirtualTrick(integrand.nlo_count, FRACTION, 0.005)
            names = ("weight", "lo_weight") if generator.nbody() else ("weight",)
            iterations = []
            for _ in range(2):
                values = runtime(BATCH)
                out = outputs_by_name(generator, values)
                state.add(values)
                state.end_iteration([context])
                iterations.append(np.vstack([out[n] for n in names]))
                if not generator.nbody():
                    assert state.virtual_fraction() == FRACTION
                    assert list(state.virtual_ratios()) == []
                    continue
                trick.add(out["virtual"], out["born"], out["evaluated"] != 0)
                trick.adapt(
                    out["virtual_residual"].std() / math.sqrt(POINTS),
                    out["weight"].std() / math.sqrt(POINTS),
                )
                np.testing.assert_allclose(
                    state.virtual_ratios(), trick.ratios, rtol=1e-12
                )
                assert state.virtual_fraction() == pytest.approx(
                    trick.fraction, rel=1e-12
                )
                fraction = np.from_dlpack(context.get_global(generator.fraction_name()))
                assert fraction.reshape(-1)[0] == state.virtual_fraction()
                ratios = np.from_dlpack(context.get_global(generator.ratio_name()))
                np.testing.assert_array_equal(
                    ratios.reshape(-1), state.virtual_ratios()
                )
            assert state.value_count() == (3 if generator.nbody() else 1)
            # the sums of both iterations, the variance of each about its mean
            sums = sum(v.sum(1) for v in iterations)
            covariance = sum(
                v @ v.T - np.outer(v.sum(1), v.sum(1)) / v.shape[1] for v in iterations
            )
            count = 2 * POINTS
            assert state.count() == count
            combinations = ([1.0], [0.0, 1.0], [1.0, -2.0])
            if not generator.nbody():
                combinations = ([1.0], [-2.0])
            for coefficients in combinations:
                c = np.zeros(len(names))
                c[: len(coefficients)] = coefficients
                mean, error = state.combination(coefficients)
                assert mean == pytest.approx(c @ sums / count, rel=1e-12)
                assert error == pytest.approx(
                    math.sqrt(c @ covariance @ c) / count, rel=1e-9
                )
            state.reset()
            assert state.count() == 0


@pytest.mark.parametrize("name", ["pp_ttx", "epem_jjj"])
def test_training_outputs(name):
    """The training gives the cross section of the channel integrand without
    the virtual at its random numbers, for a Born with the ratios of the
    globals, and the density of its flow as CSIntegrandProbability."""
    with cs_setup(name) as run:
        context = run.context
        streams = set()
        for integrand, channel in channels_of(run):
            born = nbody(integrand)
            streams.add(born)
            channel_integrand = integrand.channel_integrand(channel, context)
            dimension = integrand.channel_dimension(channel)
            prefix = f"train.{integrand.name}.{channel}"
            flow = ms.Flow(input_dim=dimension, prefix=f"{prefix}.flow")
            flow.initialize_globals(context, 1)
            generator = ms.CSIntegrand(
                channel_integrand,
                flow,
                prefix,
                virtual_fraction=FRACTION,
                madnis_training=True,
            )
            generator.initialize_globals(context)
            out = outputs_by_name(
                generator, ms.FunctionRuntime(generator.function(), context)(BATCH)
            )
            assert list(out) == ["full_weight", "weight", "latent", "adaptive_prob"]
            density = probability(generator, context, out["latent"])
            np.testing.assert_allclose(density, out["adaptive_prob"], rtol=1e-10)
            np.testing.assert_allclose(
                out["weight"] / out["adaptive_prob"], out["full_weight"], rtol=1e-12
            )
            expected = expected_outputs(channel_integrand, context, out["latent"], born)
            if born:
                assert np.all(expected["evaluated"] == 0)
            np.testing.assert_array_equal(out["weight"], expected["cross_section"])
            assert np.count_nonzero(out["weight"]) > POINTS // 10
        assert streams == {False, True}


@pytest.mark.parametrize("name", ["pp_ttx", "epem_jjj", "t_bwg"])
def test_histograms(name):
    """The histograms of a channel of each stream through the generator
    (CSObservableHistograms) are those of the driver in Python (CSHistograms)
    at the outputs of the integrand, the real emission and the Borns of its
    dipoles in the same bins, to the rounding of the bins, and the integrand
    of the other stream refuses them."""
    with cs_setup(name) as run:
        context = run.context
        histograms, jets = run.histograms, run.jets()
        assert histograms
        streams = {}
        for integrand, channel in channels_of(run):
            observables = integrand.generator_histograms(histograms, jets)
            streams[nbody(integrand)] = (integrand, channel, observables)
            generator = ms.CSIntegrand(
                integrand.channel_integrand(channel, context),
                None,
                f"histograms.{integrand.name}.{channel}",
                histograms=observables,
            )
            generator.initialize_globals(context)
            out = outputs_by_name(
                generator, ms.FunctionRuntime(generator.function(), context)(BATCH)
            )
            names = list(observables.arg_types().keys())
            ours = ms.FunctionRuntime(observables.function(), context)(
                *[out[name] for name in names]
            )
            runtime = integrand.histogram_runtime(histograms, jets)
            ones = np.ones(POINTS)
            if nbody(integrand):
                result = dict(nlo=out["weight"], momenta=out["momenta"])
                theirs = integrand.histogram_sums(runtime, result, ones, None)
            else:
                assert out["born_momenta"].shape[1] == len(integrand.dipoles)
                theirs = runtime(out, ones)
            assert len(ours) == len(theirs) == 2 * len(histograms)
            # the graph of the generator fuses the observables of the
            # kinematics otherwise, so that a bin can differ by its rounding
            for a, b in zip(ours, theirs):
                a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-14 * np.abs(b).max())
        assert set(streams) == {False, True}
        for born in (False, True):
            integrand, channel, _ = streams[born]
            _, _, other = streams[not born]
            with pytest.raises(ValueError, match="histograms take"):
                ms.CSIntegrand(
                    integrand.channel_integrand(channel, context),
                    None,
                    "refuse",
                    histograms=other,
                )


def test_refusals():
    """CSIntegrand refuses a training without a flow, an adaptive mapping of
    another dimension or with a condition and fractions of the virtual
    outside (0, 1], and the CS integrands a discrete flow."""
    with cs_setup("pp_ttx", streams=("nbody",)) as run:
        context = run.context
        integrand = run.integrands[0]
        channel_integrand = integrand.channel_integrand(0, context)
        dimension = integrand.channel_dimension(0)
        vegas = ms.VegasMapping(dimension, 10, "refuse")
        with pytest.raises(ValueError, match="needs a flow"):
            ms.CSIntegrand(channel_integrand, vegas, "refuse", madnis_training=True)
        for flow, match in [
            (ms.Flow(input_dim=dimension + 1, prefix="refuse.flow"), "dimension"),
            (
                ms.Flow(input_dim=dimension, condition_dim=2, prefix="refuse.flow"),
                "condition",
            ),
        ]:
            with pytest.raises(ValueError, match=match):
                ms.CSIntegrand(channel_integrand, flow, "refuse")
        for fraction, minimum in [(0.0, 0.005), (1.5, 0.005), (0.5, 0.0)]:
            with pytest.raises(ValueError, match="fractions"):
                ms.CSIntegrand(
                    channel_integrand,
                    None,
                    "refuse",
                    virtual_fraction=fraction,
                    min_virtual_fraction=minimum,
                )
        discrete = ms.DiscreteFlow(option_counts=[2], condition_dim=dimension)
        with pytest.raises(ValueError, match="discrete"):
            integrand.generator_integrand(
                0, context, "refuse", 1.0, 0.005, 10, None, None, None, discrete
            )


@pytest.mark.parametrize(
    "name, options",
    [
        ("epem_jj_fixed", {}),
        ("pp_epem", {}),
        ("t_bw", {}),
        ("pp_ttx", {"channels": "reals"}),
        ("epem_ttx_isr", {}),
        ("epem_jj_fixed", {"madnis_batches": 100}),
        ("pp_epem", {"madnis_batches": 100}),
    ],
)
def test_against_mg5amc(name, options):
    """Runs through the event generator, with the channels of the real
    diagrams alone, of dressed leptons and with a MadNIS training, agree with
    MG5aMC, the LO also
    per subprocess, and their histograms add up to the cross section."""
    tag = "_".join([name, *(f"{k}{v}" for k, v in options.items())])
    data, run = run_of(
        name,
        tag,
        accuracy=0.0,
        grid_points=2000,
        grid_iterations=3,
        points=4000,
        max_iterations=3,
        generator=True,
        threads=2,
        **options,
    )
    result = run.run()
    for key in ("lo", "nlo"):
        assert abs(pull(result[key], data[key])) < SIGMAS, (key, result[key])
    # the real streams belong to the real processes, not to the subprocesses
    for subprocess, theirs in data["lo"]["subprocesses"].items():
        ours = result["subprocesses"][subprocess]["lo"]
        assert abs(pull(ours, theirs)) < SIGMAS, (subprocess, ours, theirs)
    assert run.iterations == 3
    with open(os.path.join(run.run_path, "info.json")) as f:
        info = json.load(f)
    assert info["status"] == "done"
    histograms = info["histograms"]
    assert histograms
    total = sum(histograms[0]["bin_values"])
    assert total == pytest.approx(result["nlo"]["sigma"], rel=1e-9)
