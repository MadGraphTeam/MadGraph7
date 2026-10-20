# SPDX-License-Identifier: 0BSD
"""A complex loop-ket Ward control that detects missing rational completion."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.loop_nightly


class _LongitudinalKernel:
    """Replace only the selected ket source, keeping the original Born bra."""

    def __init__(self, executor, dag, bare, batch, leg):
        self.executor = executor
        names = [p.get_name() for p in dag.parameters]
        self.replacements = {
            names.index(symbol.get_name()): component
            for component, symbol in enumerate(bare.external_states[leg][1])
            if symbol.get_name() in names
        }
        self.momentum = batch.bound.take_columns(
            [batch.positions[p.get_name()] for p in bare.momenta[leg]]
        )

    def evaluate_batch(self, values):
        from pyamplicol.loops.native import load_loop_native

        native = load_loop_native()

        columns = list(range(values.columns))
        for column, component in self.replacements.items():
            columns[column] = values.columns + component
        return self.executor.evaluate_batch(
            native.NativeBatch.concat_columns([values, self.momentum]).take_columns(
                columns
            )
        )


def _complex_ket_projectors(program):
    """Expose the complex weighted ket; taking 2Re could hide a Ward failure."""
    from pyamplicol.loops.native import load_loop_native

    native = load_loop_native()
    from symbolica import E, S

    from pyamplicol.loops.physical import _saved

    poles = [
        [S(f"ward::family{i}_pole{p}") for p in range(3)]
        for i in range(len(program.bare.families))
    ]
    loop = native.CoefficientEvaluator(
        _saved(
            [
                E("1i") * sum((row[p] for row in poles), E("0")) / (16 * E("pi") ** 2)
                for p in range(3)
            ],
            [p for row in poles for p in row],
        ),
        arithmetic="double",
    )
    parameters, ultraviolet, rational = [], [E("0")] * 3, [E("0")] * 3
    for helicity in range(len(program.bare.helicities)):
        born = [
            S(f"ward::born{helicity}_{i}") for i in range(len(program.bare.born_colors))
        ]
        parameters.extend(born)
        for index, current in enumerate(program.currents):
            amplitudes = [
                S(f"ward::ct{helicity}_{index}_{i}") for i in range(current.term_count)
            ]
            parameters.extend(amplitudes)
            value = sum(
                (
                    left.conj() * weight * amplitude
                    for left, row in zip(born, current.color_weights, strict=True)
                    for weight, amplitude in zip(row, amplitudes, strict=True)
                ),
                E("0"),
            )
            target = rational if current.insertion.rule.kind == "R2" else ultraviolet
            target[-current.epsilon_power] += value
        parameters.extend(
            S(f"ward::field{helicity}_{i}") for i in range(len(program.field_currents))
        )
    # These field terms multiply the longitudinal Born ket. The test checks its
    # vanishing independently before using zero in this complex projection.
    counterterms = native.CoefficientEvaluator(
        _saved([*ultraviolet, *rational, E("0"), E("0"), E("0")], parameters),
        arithmetic="double",
    )
    return loop, counterterms


@pytest.mark.parametrize(
    "r2_strategy",
    ["model", pytest.param("direct", marks=pytest.mark.deferred_direct_r2)],
)
def test_complete_gluon_loop_ket_ward_cancels_rational_remainder(
    loop_sm_ufo, loop_native, monkeypatch, r2_strategy
):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native Ward control requires licensed Symbolica")
    native = loop_native
    source_path = str(loop_sm_ufo)
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.physical_counterterms import (
        gluon_scattering_model_virtual_program,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    source = build_loop_model(load_raw_loop_model(source_path))
    if r2_strategy == "direct":
        # No effective rational kernels are available to this generated branch.
        # The lossless raw sidecar remains source evidence; only retained kernel
        # entries participate in counterterm preparation and matching.
        source = replace(
            source,
            kernels=tuple(entry for entry in source.kernels if entry.kind != "R2"),
        )
    program = gluon_scattering_model_virtual_program(
        source,
        arithmetic="double",
        reduction_engine="fast",
        coefficient_engine="jit",
        r2_strategy=r2_strategy,
    )
    if r2_strategy == "direct":
        assert all(i.rule.kind != "R2" for i in program.matched_insertions)
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    points = np.asarray(
        json.loads(fixture.read_text(), parse_int=str, parse_float=str)["points"],
        dtype=str,
    )
    scales, model = ["1000000"] * len(points), ["0.118", "173"]
    bare = program.bare
    physical = bare.evaluate_native(points, scales, model_values=model)
    batch = physical.batch
    loop_executors = bare.loop_executors
    ct_executors = [current.executor for current in program.currents]
    loop_projection, ct_projection = _complex_ket_projectors(program)
    monkeypatch.setattr(bare, "postprocess", loop_projection)
    monkeypatch.setattr(program, "projector", ct_projection)
    for leg in range(4):
        longitudinal_born = native.NativeBatch.concat_columns(
            [
                _LongitudinalKernel(executor, dag, bare, batch, leg).evaluate_batch(
                    batch.bound.take_columns(
                        [batch.positions[p.get_name()] for p in dag.parameters]
                    )
                )
                for executor, dag in zip(
                    bare.born_executors, bare.born_dags, strict=True
                )
            ]
        )
        born_ward = bare.born_projector.evaluate_batch(
            longitudinal_born.reshape_rows(
                batch.size,
                len(bare.helicities) * longitudinal_born.columns,
                batch.point_ids,
            )
        )
        np.testing.assert_allclose(
            np.asarray(born_ward.to_decimal(), dtype=float), 0, atol=2e-8
        )
        monkeypatch.setattr(
            bare,
            "loop_executors",
            [
                _LongitudinalKernel(executor, recurrence.dag, bare, batch, leg)
                for executor, recurrence in zip(
                    loop_executors, bare.recurrences, strict=True
                )
            ],
        )
        for current, executor in zip(program.currents, ct_executors, strict=True):
            monkeypatch.setattr(
                current,
                "executor",
                _LongitudinalKernel(executor, current.dag, bare, batch, leg),
            )
        result = program.evaluate_decimal(points, scales, model_values=model)
        np.testing.assert_array_equal(
            result.born, np.asarray(physical.born.to_decimal(), dtype=object)
        )
        np.testing.assert_allclose(result.virtual.astype(float), 0, atol=2e-9)
        if leg < 2:
            # The control is sensitive to omitting R2: each incomplete ket is
            # sizable although their complete complex sum satisfies the identity.
            assert np.max(np.abs(result.bare.astype(float))) > 10
            assert np.max(np.abs(result.r2.astype(float))) > 10
