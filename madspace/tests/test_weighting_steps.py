"""Re- and unweighting steps after the first unweighting (WeightingStep).

The integrand uses a deformed "surrogate" matrix element, and a later step
re- or unweights the events with the "true" one. The results have to agree with
a run that uses the true matrix element in the integrand, and must not depend
on the number of threads. Uses the e+ e- -> d d~ g setup of test_event_stream
with PythonFunction matrix elements.
"""

import os

import numpy as np
import pytest
from test_event_stream import (
    CM_ENERGY,
    PERMUTATIONS,
    PIDS,
    TOPOLOGY,
    make_config,
    make_context,
)
from test_integrand_me_outputs import python_matrix_element

import madspace as ms


def true_me(momenta):
    return 1.0 + 3.0 * (momenta[:, 2, 0] / CM_ENERGY) ** 2


def surrogate_me(momenta):
    return true_me(momenta) * (1.0 + 0.3 * np.sin(momenta[:, 3, 1] / 50.0))


def make_integrand(alphas_file, matrix_element, pt_min=20.0):
    mapping = ms.PhaseSpaceMapping(
        TOPOLOGY,
        CM_ENERGY,
        permutations=PERMUTATIONS,
        leptonic=True,
        cuts=ms.Cuts(
            [
                ms.CutItem(ms.Observable(PIDS, ms.Observable.obs_pt, [[i]]), min=pt_min)
                for i in (2, 3, 4)
            ]
        ),
    )
    diff_xs = ms.DifferentialCrossSection(
        matrix_element=matrix_element,
        cm_energy=CM_ENERGY,
        running_coupling=None,
        energy_scale=ms.CachedScale(),
        return_matrix_element=True,
    )
    return ms.Integrand(
        mapping,
        [diff_xs],
        running_coupling=ms.RunningCoupling(ms.AlphaSGrid(alphas_file)),
        energy_scale=ms.EnergyScale(len(PIDS), 91.188),
        channel_indices=[0],
    )


def generate(
    run_dir,
    integrand_me,
    step_modes=(),
    step_me=true_me,
    target_count=5000,
    threads=2,
    pt_mins=(20.0,),
    grouped=(0,),
    seed=7,
    step_outputs=("matrix_element",),
    override_outputs=(),
    integrand_outputs=("matrix_element",),
):
    """Survey and generate, returns the status, the combined events and the
    channels."""
    os.makedirs(run_dir, exist_ok=True)
    context, alphas_file = make_context(run_dir, threads)
    me0 = python_matrix_element(context, integrand_me, integrand_outputs)
    me1 = python_matrix_element(context, step_me, step_outputs)
    config = make_config()
    config.target_count = target_count
    config.cpu_batch_size = 500
    channels = [
        ms.ChannelEventGenerator(
            contexts=[context],
            integrand=make_integrand(alphas_file, me0, pt_min),
            event_file=os.path.join(run_dir, f"events.{i}.npy"),
            weight_file=os.path.join(run_dir, f"weights.{i}.npy"),
            config=config,
            subprocess_index=0,
            name=str(i),
            histograms=None,
        )
        for i, pt_min in enumerate(pt_mins)
    ]
    steps = [
        ms.WeightingStep([me1], mode=mode, override_outputs=list(override_outputs))
        for mode in step_modes
    ]
    groups = [ms.WeightingGroup(list(grouped), steps)] if steps else []
    generator = ms.EventGenerator(
        [context], channels, seed=seed, config=config, groups=groups
    )
    generator.survey()
    generator.generate()
    output_file = os.path.join(run_dir, "combined.npy")
    generator.combine_to_compact_npy(output_file)
    return generator.status(), np.load(output_file), channels


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """The true matrix element in the integrand."""
    return generate(
        str(tmp_path_factory.mktemp("reference")), true_me, target_count=20000
    )


def check_compatible(status, events, reference):
    ref_status, ref_events, _ = reference
    error = np.hypot(status.error, ref_status.error)
    assert abs(status.mean - ref_status.mean) < 4 * error
    # the events are normalized to the cross section
    assert np.mean(events["weight"]) == pytest.approx(status.mean_abs, rel=1e-9)
    # mean energy of the first quark, weighted
    energies = [
        np.sum(ev["weight"] * ev["part3_energy"]) / np.sum(ev["weight"])
        for ev in (events, ref_events)
    ]
    assert energies[0] == pytest.approx(energies[1], rel=0.01)


def test_unweight_step(tmp_path, reference):
    status, events, channels = generate(
        str(tmp_path), surrogate_me, ["unweight"], target_count=20000
    )
    assert len(events) == 20000
    check_compatible(status, events, reference)
    # unweighted: apart from the overweights, all events have the same weight
    weights = events["weight"]
    assert np.mean(np.isclose(weights, np.min(weights))) > 0.98
    assert channels[0].step_max_weight(1) > 0


def test_reweight_step(tmp_path, reference):
    status, events, _ = generate(
        str(tmp_path), surrogate_me, ["reweight"], target_count=20000
    )
    check_compatible(status, events, reference)
    # the ratios of the matrix elements stay in the weights
    weights = events["weight"]
    assert np.std(weights) / np.mean(weights) > 0.1


def test_reweight_then_unweight(tmp_path, reference):
    """The unweighting step absorbs the ratio of the reweighting step: the
    second step has the same matrix element, so it only unweights."""
    status, events, _ = generate(
        str(tmp_path), surrogate_me, ["reweight", "unweight"], target_count=20000
    )
    check_compatible(status, events, reference)
    weights = events["weight"]
    assert np.mean(np.isclose(weights, np.min(weights))) > 0.98


@pytest.mark.parametrize("step_modes", [["unweight"], ["reweight", "unweight"]])
def test_reproducible(tmp_path, step_modes):
    """Two grouped channels and one without steps, which share the batches
    differently with different thread counts."""
    results = [
        generate(
            str(tmp_path / str(threads)),
            surrogate_me,
            step_modes,
            threads=threads,
            pt_mins=(20.0, 40.0, 60.0),
            grouped=(0, 2),
        )
        for threads in (1, 4)
    ]
    (status_a, events_a, _), (status_b, events_b, _) = results
    assert len(events_a) == 5000
    assert np.array_equal(events_a, events_b)
    assert status_a.mean == status_b.mean


def test_index_outputs(tmp_path):
    """A step fills in indices the integrand matrix element does not return,
    and replaces the others only with override_outputs."""

    def with_helicity(momenta):
        return true_me(momenta), np.full(len(momenta), 1, np.int32)

    outputs = ("matrix_element", "helicity_index")
    _, events, _ = generate(
        str(tmp_path / "fill"),
        surrogate_me,
        ["unweight"],
        step_me=with_helicity,
        step_outputs=outputs,
    )
    assert (events["helicity_index"] == 1).all()
    assert (events["color_index"] == -1).all()

    def surrogate_with_helicity(momenta):
        return surrogate_me(momenta), np.zeros(len(momenta), np.int32)

    for override, helicity in [((), 0), ((ms.MatrixElement.helicity_index_out,), 1)]:
        _, events, _ = generate(
            str(tmp_path / f"override{helicity}"),
            surrogate_with_helicity,
            ["unweight"],
            step_me=with_helicity,
            step_outputs=outputs,
            integrand_outputs=outputs,
            override_outputs=override,
        )
        assert (events["helicity_index"] == helicity).all()


def test_invalid_groups(tmp_path):
    context, alphas_file = make_context(str(tmp_path), 1)
    me = python_matrix_element(context, true_me)
    config = make_config()

    def channel(return_matrix_element=True):
        integrand = make_integrand(alphas_file, me)
        if not return_matrix_element:
            diff_xs = ms.DifferentialCrossSection(
                matrix_element=me,
                cm_energy=CM_ENERGY,
                running_coupling=None,
                energy_scale=ms.CachedScale(),
            )
            integrand = ms.Integrand(
                ms.PhaseSpaceMapping(
                    TOPOLOGY, CM_ENERGY, permutations=PERMUTATIONS, leptonic=True
                ),
                [diff_xs],
                running_coupling=ms.RunningCoupling(ms.AlphaSGrid(alphas_file)),
                energy_scale=ms.EnergyScale(len(PIDS), 91.188),
                channel_indices=[0],
            )
        return ms.ChannelEventGenerator(
            [context], integrand, "", "", config, 0, "c", None
        )

    step = ms.WeightingStep([me])
    with pytest.raises(ValueError, match="matrix element"):
        ms.EventGenerator(
            [context], [channel(False)], seed=1, groups=[ms.WeightingGroup([0], [step])]
        )
    with pytest.raises(ValueError, match="one weighting group"):
        ms.EventGenerator(
            [context],
            [channel()],
            seed=1,
            groups=[ms.WeightingGroup([0], [step]), ms.WeightingGroup([0], [step])],
        )
    with pytest.raises(ValueError, match="sampled internally"):
        sampled = ms.MatrixElement(
            me.matrix_element_index(),
            len(PIDS),
            [ms.MatrixElement.momenta_in, ms.MatrixElement.random_helicity_in],
            [ms.MatrixElement.matrix_element_out],
            1,
            True,
        )
        ms.EventGenerator(
            [context],
            [channel()],
            seed=1,
            groups=[ms.WeightingGroup([0], [ms.WeightingStep([sampled])])],
        )
