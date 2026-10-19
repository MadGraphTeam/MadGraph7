# SPDX-License-Identifier: 0BSD
"""Complete directed colour for existing validated full-colour QCD controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .kernels import require_finite_batch
from .physical_correlations import (
    DirectedBareResult,
    NativeDirectedBareEvaluation,
    NativeDirectedBareProgram,
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


def require_directed_model_scope(model, histories=()):
    """One current producer admission boundary, including cold preparation.

    QCD is the validated completion domain today. EW correlation coverage is a
    pending extension, not a claim that electroweak histories are incompatible.
    """
    if model.r2_strategy != "model":
        raise NotImplementedError("directed completion requires model R2")
    if model.bare.order_layout.correction_names != ("QCD",):
        raise NotImplementedError(
            "directed model completion is currently validated for QCD corrections; "
            "EW correlation coverage is a pending extension"
        )
    from .physical_correlations import directed_color_convention

    if (model.color_projection is not None) != bool(
        getattr(model.bare, "_formal_color_projected", False)
    ) or model.color_accuracy != getattr(model.bare, "color_accuracy", "full"):
        raise ValueError("directed model and bare colour projection owners differ")
    directed_color_convention(model.bare, histories)


class NativeDirectedModelProgram(NativeDirectedBareProgram):
    """Reuse ordinary currents for both complex model-complete cross terms.

    The model's validated effective rules already contain their loop measure.
    Their forward term and conjugated swapped term use separate colour kernels
    and exact Born/loop pair masks. External fields keep the source delta-Z/2
    and the insertion's specific Born graph. No source-model completeness beyond
    the admitted QCD control, precision rescue or flavour mapping is added.
    """

    def __init__(self, model, histories, *, spin_layout=None, _prepared=None):
        from .physical_counterterms import NativeModelVirtualProgram
        from .physical_source import NativeSourceModelProgram
        from .prepared_correlations import PreparedDirectedModel, prepare_directed_model

        if isinstance(model, NativeSourceModelProgram):
            model = model.virtual
        if not isinstance(model, NativeModelVirtualProgram):
            raise TypeError("directed completion requires a physical model program")
        if _prepared is not None and not isinstance(_prepared, PreparedDirectedModel):
            raise TypeError("expected immutable directed model preparation")
        if _prepared is not None:
            _prepared.require_owner(model)
        self.model = model
        self._current_ids = tuple(id(current) for current in model.currents)
        self._field_ids = tuple(id(current) for current in model.field_currents)
        super().__init__(
            model.bare,
            histories,
            spin_layout=spin_layout,
            _prepared=None if _prepared is None else _prepared.bare,
        )
        prepared = _prepared or prepare_directed_model(model, self.prepared)
        prepared.require_owner(model)
        self.prepared = prepared
        self.counterterm_projector = self.bare.backend.CoefficientEvaluator(
            prepared.counterterm_state, **self.bare.options
        )
        self.assemble_model = self.bare.backend.CoefficientEvaluator(
            prepared.assembly_state, **self.bare.options
        )

    def _check_scope(self):
        model = self.model
        require_directed_model_scope(model, self.histories)
        if (
            tuple(id(current) for current in model.currents) != self._current_ids
            or tuple(id(current) for current in model.field_currents) != self._field_ids
        ):
            raise ValueError(
                "model current catalogue changed after directed preparation"
            )
        self._check_metric_scope()

    def evaluate_native(
        self,
        momenta,
        scales_squared,
        *,
        model_values=None,
        spin_sources=None,
        point_ids=None,
    ):
        self._check_scope()
        currents = self._prepare_currents_native(
            momenta,
            scales_squared,
            model_values=model_values,
            spin_sources=spin_sources,
            point_ids=point_ids,
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
        self,
        momenta,
        scales_squared,
        *,
        model_values=None,
        spin_sources=None,
        point_ids=None,
    ):
        result = self.evaluate_native(
            momenta,
            scales_squared,
            model_values=model_values,
            spin_sources=spin_sources,
            point_ids=point_ids,
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
