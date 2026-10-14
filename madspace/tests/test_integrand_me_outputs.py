"""Integrand with matrix elements that only return |M|^2: the channel weights,
helicity, color and diagram index are optional outputs of the matrix element.

Uses the e+ e- -> d d~ g setup of test_event_stream with PythonFunction matrix
elements, as a stand-in for surrogates that only predict |M|^2.
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

import madspace as ms


def python_matrix_element(context, function, outputs=("matrix_element",)):
    api = context.add_matrix_element(
        ms.PythonFunction(
            function, ["momenta"], list(outputs), particle_count=len(PIDS)
        )
    )
    output_enums = {
        "matrix_element": ms.MatrixElement.matrix_element_out,
        "color_index": ms.MatrixElement.color_index_out,
        "helicity_index": ms.MatrixElement.helicity_index_out,
        "diagram_index": ms.MatrixElement.diagram_index_out,
        "matrix_element_ratio": ms.MatrixElement.matrix_element_ratio_out,
    }
    return ms.MatrixElement(
        api.index(),
        len(PIDS),
        [ms.MatrixElement.momenta_in],
        [output_enums[output] for output in outputs],
    )


def flat_matrix_element():
    return ms.MatrixElement(
        0xBADCAFE,
        len(PIDS),
        ms.Integrand.matrix_element_inputs,
        ms.Integrand.matrix_element_outputs,
        1,
        True,
    )


def make_integrand(alphas_file, matrix_element, return_matrix_element=False, **kwargs):
    mapping = ms.PhaseSpaceMapping(
        TOPOLOGY,
        CM_ENERGY,
        permutations=PERMUTATIONS,
        leptonic=True,
        cuts=ms.Cuts(
            [
                ms.CutItem(ms.Observable(PIDS, ms.Observable.obs_pt, [[i]]), min=20.0)
                for i in (2, 3, 4)
            ]
        ),
    )
    diff_xs = ms.DifferentialCrossSection(
        matrix_element=matrix_element,
        cm_energy=CM_ENERGY,
        running_coupling=None,
        energy_scale=ms.CachedScale(),
        return_matrix_element=return_matrix_element,
    )
    return ms.Integrand(
        mapping,
        [diff_xs],
        running_coupling=ms.RunningCoupling(ms.AlphaSGrid(alphas_file)),
        energy_scale=ms.EnergyScale(len(PIDS), 91.188),
        channel_indices=[0],
        **kwargs,
    )


def generate(run_dir, context, integrand, name):
    config = make_config()
    config.target_count = 2000
    channel = ms.ChannelEventGenerator(
        contexts=[context],
        integrand=integrand,
        event_file=os.path.join(run_dir, f"events.{name}.npy"),
        weight_file=os.path.join(run_dir, f"weights.{name}.npy"),
        config=config,
        subprocess_index=0,
        name=name,
        histograms=None,
    )
    generator = ms.EventGenerator([context], [channel], seed=7, config=config)
    generator.survey()
    generator.generate()
    output_file = os.path.join(run_dir, f"combined.{name}.npy")
    generator.combine_to_compact_npy(output_file)
    return generator.status(), np.load(output_file)


def test_only_matrix_element_output(tmp_path):
    """Without the index outputs the events carry -1, and |M|^2 = 1 reproduces
    the integral of the built-in flat matrix element."""
    context, alphas_file = make_context(str(tmp_path), 2)
    matrix_element = python_matrix_element(
        context, lambda momenta: np.ones(len(momenta))
    )
    status, events = generate(
        str(tmp_path), context, make_integrand(alphas_file, matrix_element), "py"
    )
    flat_status, flat_events = generate(
        str(tmp_path),
        context,
        make_integrand(alphas_file, flat_matrix_element()),
        "flat",
    )
    assert len(events) == len(flat_events) == 2000
    for field in ("helicity_index", "color_index", "diagram_index"):
        assert (events[field] == -1).all()
        assert (flat_events[field] >= 0).all()
    error = np.hypot(status.error, flat_status.error)
    assert abs(status.mean - flat_status.mean) < 5 * error


def test_provided_index_output(tmp_path):
    """An index the matrix element returns is written to the events."""
    context, alphas_file = make_context(str(tmp_path), 1)
    matrix_element = python_matrix_element(
        context,
        lambda momenta: (np.ones(len(momenta)), np.full(len(momenta), 1, np.int32)),
        ("matrix_element", "helicity_index"),
    )
    _, events = generate(
        str(tmp_path), context, make_integrand(alphas_file, matrix_element), "hel"
    )
    assert (events["helicity_index"] == 1).all()
    assert (events["color_index"] == -1).all()


def test_raw_matrix_element_output(tmp_path):
    """Returning the raw |M|^2 adds an integrand output without changing the
    generated events."""
    context, alphas_file = make_context(str(tmp_path), 1)
    matrix_element = python_matrix_element(
        context, lambda momenta: 1.0 + momenta[:, 2, 0] / CM_ENERGY
    )
    results = [
        generate(
            str(tmp_path),
            context,
            make_integrand(alphas_file, matrix_element, raw),
            f"raw{raw}",
        )
        for raw in (False, True)
    ]
    (status_a, events_a), (status_b, events_b) = results
    assert status_a.mean == status_b.mean
    assert np.array_equal(events_a, events_b)


def test_diagram_channel_weights_need_amp2(tmp_path):
    context, alphas_file = make_context(str(tmp_path), 1)
    matrix_element = python_matrix_element(
        context, lambda momenta: np.ones(len(momenta))
    )
    with pytest.raises(ValueError, match="diagram_amp2"):
        make_integrand(
            alphas_file,
            matrix_element,
            first_chan_weight_remap=[[0, 1]],
            first_remapped_chan_count=2,
        )
