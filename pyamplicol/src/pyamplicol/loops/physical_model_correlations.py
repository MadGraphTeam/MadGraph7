# SPDX-License-Identifier: 0BSD
"""Complete directed colour for existing validated full-colour QCD controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .color_bilinears import SUNColorSpace
from .color_connections import directed_color_bilinears
from .kernels import require_finite_batch
from .physical import _saved
from .physical_correlations import (
    DirectedBareResult,
    NativeDirectedBareEvaluation,
    NativeDirectedBareProgram,
    _physical_color_ports,
    _rename_physical_color_tensors,
)


@dataclass(frozen=True, slots=True)
class NativeDirectedModelEvaluation(NativeDirectedBareEvaluation):
    bare: Any
    uv: Any
    r2: Any
    uv_local: Any
    uv_external_fields: Any


@dataclass(frozen=True, slots=True, kw_only=True)
class DirectedModelResult(DirectedBareResult):
    """Complex component arrays on point/history/order/pole/decimal-pair axes."""

    bare: Any
    uv: Any
    r2: Any
    uv_local: Any
    uv_external_fields: Any
    convention: str = "MadLoop-Cepsilon-directed-model-UV-R2-fields"

    def __post_init__(self):
        DirectedBareResult.__post_init__(self)
        for name in ("bare", "uv", "r2", "uv_local", "uv_external_fields"):
            getattr(self, name).setflags(write=False)


class NativeDirectedModelProgram(NativeDirectedBareProgram):
    """Reuse ordinary currents for both complex model-complete cross terms.

    The model's validated effective rules already contain their loop measure.
    Their forward term and conjugated swapped term use separate colour kernels
    and exact Born/loop pair masks. External fields keep the source delta-Z/2
    and the insertion's specific Born graph. No source-model completeness beyond
    the admitted QCD control, precision rescue or flavour mapping is added.
    """

    def __init__(self, model, histories, *, spin_layout=None):
        from symbolica import E, S

        from .physical_counterterms import NativeModelVirtualProgram

        if not isinstance(model, NativeModelVirtualProgram):
            raise TypeError("directed completion requires a NativeModelVirtualProgram")
        self.model = model
        self._current_ids = tuple(id(current) for current in model.currents)
        self._field_ids = tuple(id(current) for current in model.field_currents)
        super().__init__(model.bare, histories, spin_layout=spin_layout)
        bare = self.bare
        _, born_colors, _ = _physical_color_ports(bare)
        nc = S("loop_directed_model::Nc", is_real=True)
        space = SUNColorSpace(nc)
        color_cache = {}

        def matrices(colors, key):
            if key not in color_cache:
                values = []
                for bra, ket in self.histories:
                    kernels = directed_color_bilinears(
                        space, born_colors, colors, bra, ket
                    )
                    forward = kernels.born_loop
                    swapped = tuple(
                        tuple(
                            kernels.loop_born[j][i].conj() for j in range(len(colors))
                        )
                        for i in range(len(born_colors))
                    )
                    values.extend(
                        tuple(
                            tuple(
                                (entry.replace(nc, 3) / bare.initial_average).together()
                                for entry in row
                            )
                            for row in matrix
                        )
                        for matrix in (forward, swapped)
                    )
                color_cache[key] = tuple(values)
            return color_cache[key]

        current_weights = tuple(
            matrices(
                _rename_physical_color_tensors(current.color_tensors),
                ("ct", id(current.color_tensors)),
            )
            for current in model.currents
        )
        field_weights = tuple(
            matrices(
                tuple(born_colors[index] for index in current.born_indices),
                ("field", current.born_indices),
            )
            for current in model.field_currents
        )
        sectors = len(bare.order_layout.virtual)
        width = len(self.histories) * sectors * 3
        # Components are local UV, R2, external fields; each retains two terms.
        cross_terms = [[E("0")] * width for _ in range(6)]
        parameters = []
        for helicity in range(len(self.helicities)):
            born = tuple(
                S(f"loop_directed_model::born_{helicity}_{index}")
                for index in range(len(born_colors))
            )
            parameters.extend(born)
            for index, (current, weights) in enumerate(
                zip(model.currents, current_weights, strict=True)
            ):
                amplitudes = tuple(
                    S(f"loop_directed_model::ct_{helicity}_{index}_{term}")
                    for term in range(current.term_count)
                )
                parameters.extend(amplitudes)
                component = 1 if current.insertion.rule.kind == "R2" else 0
                self._accumulate(
                    cross_terms,
                    component,
                    born,
                    amplitudes,
                    weights,
                    current.insertion.graph.order,
                    current.epsilon_power,
                    sectors,
                )
            for index, (current, weights) in enumerate(
                zip(model.field_currents, field_weights, strict=True)
            ):
                factor = S(f"loop_directed_model::field_{helicity}_{index}")
                parameters.append(factor)
                amplitudes = tuple(
                    factor * born[right] for right in current.born_indices
                )
                self._accumulate(
                    cross_terms,
                    2,
                    born,
                    amplitudes,
                    weights,
                    current.insertion.order,
                    current.epsilon_power,
                    sectors,
                )
        self.counterterm_projector = bare.backend.CoefficientEvaluator(
            _saved([value for row in cross_terms for value in row], parameters),
            **bare.options,
        )
        # Input rows are bare first/second then the six effective-rule blocks.
        inputs = tuple(S(f"loop_directed_model::sum_{i}") for i in range(8 * width))
        first = [
            sum((inputs[k * width + i] for k in (0, 2, 4, 6)), E("0"))
            for i in range(width)
        ]
        second = [
            sum((inputs[k * width + i] for k in (1, 3, 5, 7)), E("0"))
            for i in range(width)
        ]
        local, rational, field = (
            [inputs[k * width + i] + inputs[(k + 1) * width + i] for i in range(width)]
            for k in (2, 4, 6)
        )
        self.assemble_model = bare.backend.CoefficientEvaluator(
            _saved(
                [
                    *[a + b for a, b in zip(first, second, strict=True)],
                    *[a + b for a, b in zip(local, field, strict=True)],
                    *rational,
                    *local,
                    *field,
                    *first,
                    *second,
                ],
                inputs,
            ),
            **bare.options,
        )

    def _accumulate(
        self, outputs, component, born, amplitudes, weights, order, power, sectors
    ):
        from symbolica import E

        for history in range(len(self.histories)):
            for sector, pairs in enumerate(self.model.sector_pairs):
                for direction in (0, 1):
                    value = sum(
                        (
                            left.conj() * weight * amplitude
                            for left, left_order, row in zip(
                                born,
                                self.bare.born_term_orders,
                                weights[2 * history + direction],
                                strict=True,
                            )
                            if (left_order, order) in pairs
                            for weight, amplitude in zip(row, amplitudes, strict=True)
                        ),
                        E("0"),
                    )
                    index = 3 * (history * sectors + sector) - power
                    outputs[2 * component + direction][index] += (
                        value if direction == 0 else value.conj()
                    )

    def _check_scope(self):
        model = self.model
        if (
            model.r2_strategy != "model"
            or model.color_accuracy != "full"
            or model.color_projection is not None
            or getattr(model.bare, "_formal_color_projected", False)
        ):
            raise NotImplementedError(
                "directed completion requires fixed full colour and model R2"
            )
        if model.bare.order_layout.correction_names != ("QCD",):
            raise NotImplementedError(
                "directed model completion is limited to QCD controls"
            )
        if (
            tuple(id(current) for current in model.currents) != self._current_ids
            or tuple(id(current) for current in model.field_currents) != self._field_ids
        ):
            raise ValueError(
                "model current catalogue changed after directed preparation"
            )
        self._check_metric_scope()

    def evaluate_native(
        self, momenta, scales_squared, *, model_values, spin_sources=None
    ):
        self._check_scope()
        currents = self._prepare_currents_native(
            momenta,
            scales_squared,
            model_values=model_values,
            spin_sources=spin_sources,
        )
        reduced = super().evaluate_currents_native(currents)
        batch, backend = currents.batch, self.bare.backend
        amplitudes = self.model.evaluate_counterterm_currents_native(batch)
        inputs = backend.NativeBatch.concat_columns([currents.born, *amplitudes])
        counterterms = self.counterterm_projector.evaluate_batch(
            inputs.reshape_rows(
                batch.size, len(self.helicities) * inputs.columns, batch.point_ids
            )
        )
        require_finite_batch(counterterms, "directed counterterm cross terms")
        complete = self.assemble_model.evaluate_batch(
            backend.NativeBatch.concat_columns(
                [reduced.first, reduced.second, counterterms]
            )
        )
        require_finite_batch(complete, "complete directed virtual assembly")
        width = len(self.histories) * len(self.bare.order_layout.virtual) * 3
        values = [
            complete.take_columns(range(k * width, (k + 1) * width)) for k in range(7)
        ]
        return NativeDirectedModelEvaluation(
            batch=batch,
            born=reduced.born,
            virtual=values[0],
            first=values[5],
            second=values[6],
            statuses=reduced.statuses,
            bare=reduced.virtual,
            uv=values[1],
            r2=values[2],
            uv_local=values[3],
            uv_external_fields=values[4],
        )

    def evaluate_decimal(
        self, momenta, scales_squared, *, model_values, spin_sources=None
    ):
        result = self.evaluate_native(
            momenta,
            scales_squared,
            model_values=model_values,
            spin_sources=spin_sources,
        )
        size, histories = result.batch.size, len(self.histories)

        def values(buffer, *axes):
            return np.asarray(buffer.to_decimal(), dtype=object).reshape(
                size, histories, *axes, 2
            )

        return DirectedModelResult(
            born=values(result.born, len(self.bare.order_layout.born)),
            **{
                name: values(
                    getattr(result, name), len(self.bare.order_layout.virtual), 3
                )
                for name in (
                    "virtual",
                    "bare",
                    "uv",
                    "r2",
                    "uv_local",
                    "uv_external_fields",
                    "first",
                    "second",
                )
            },
            orders=self.bare.order_layout,
            histories=self.histories,
            channels=self.channels,
            statuses=result.statuses,
            arithmetic=self.bare.arithmetic,
            precision=self.bare.precision,
            spin_legs=() if self.spin_layout is None else self.spin_layout.legs,
        )
