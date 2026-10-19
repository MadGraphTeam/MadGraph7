# SPDX-License-Identifier: 0BSD
"""Literal gluon source plans using the existing tree correlation contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pyamplicol.correlators import CorrelatorConfig
from pyamplicol.runtime.correlated_exact import _prepare_spin_vectors

from .kernels import require_finite_batch
from .physical import _saved


@dataclass(frozen=True, slots=True)
class SpinSourceLayout:
    """One immutable joint source class with compact spectator helicities."""

    legs: tuple[int, ...]
    helicities: tuple[tuple[int, ...], ...]


class NativeSpinSourcePlan:
    """Validate exact source rows and reuse cold QCD SourceIR crossing phases.

    Public source labels are one-based, as in tree correlations. Declared keys
    must form one joint class. Every supplied vector has four literal complex
    components; neither outgoing epsilon_bar slots nor initial-state crossing
    cause conjugation, momentum replacement, normalization or projection.
    """

    def __init__(self, bare, spin_correlations):
        from symbolica import E, S

        from pyamplicol.models import BuiltinSMModel
        from pyamplicol.models._physics_ir import CrossingIR

        declarations = CorrelatorConfig(spin_correlations=spin_correlations)
        self.bare, self.backend = bare, bare.backend
        self.declarations = declarations
        edges = {
            edge.external_index + 1: edge
            for edge in bare.process.born[0].native.external_edges
        }
        # This bounded bridge uses the existing model-owned gluon SourceIR.
        # It does not assign these conventions to arbitrary external vectors.
        source = BuiltinSMModel()._source_ir(21)
        if source.wavefunction_family != "vector" or source.component_dimension != 4:
            raise ValueError("QCD source contract lacks four vector components")
        self.source_ir = source
        phase_evaluators = {}
        phase_states = {}
        components = tuple(
            S(f"loop_spin_source::component_{index}") for index in range(4)
        )
        for leg in declarations.spin_legs:
            if (
                leg not in edges
                or edges[leg].particle_pdg != 21
                or bare.state_kinds[leg - 1] not in {"epsilon", "epsilon_bar"}
                or edges[leg].particle.mass != 0
            ):
                raise NotImplementedError(
                    "spin source bridge requires a massless external gluon"
                )
            crossing = (
                source.crossing
                if bare.state_kinds[leg - 1] == "epsilon"
                else CrossingIR.identity()
            )
            # SourceIR's exact QCD phase is (1,0); converting its literal
            # metadata does not touch any supplied numerical source component.
            phase = E(str(crossing.phase[0])) + E("1i") * E(str(crossing.phase[1]))
            phase_states[leg] = _saved(
                [phase * value for value in components], components
            )
            phase_evaluators[leg] = bare.backend.CoefficientEvaluator(
                phase_states[leg],
                **bare.options,
            )
        self._phase_evaluators = MappingProxyType(phase_evaluators)
        self.phase_states = MappingProxyType(phase_states)
        layouts = {}
        for legs in ((), *declarations.spin_correlations):
            representatives = {}
            for helicities in bare.helicities:
                spectator = tuple(
                    value
                    for index, value in enumerate(helicities, 1)
                    if index not in legs
                )
                representatives.setdefault(spectator, helicities)
            layouts[legs] = SpinSourceLayout(legs, tuple(representatives.values()))
        self.layouts = MappingProxyType(layouts)

    def layout(self, vectors=None):
        vectors = {} if vectors is None else vectors
        if not isinstance(vectors, Mapping):
            raise TypeError("spin vectors must map public leg labels to source rows")
        if any(type(leg) is not int for leg in vectors):
            raise ValueError("spin source labels must be one-based integer legs")
        legs = tuple(sorted(vectors))
        if legs not in self.layouts:
            raise ValueError("this joint spin-correlation class was not declared")
        return self.layouts[legs]

    def prepare_native(self, vectors, point_ids):
        """Apply checked SourceIR phases without intermediate decimal transport."""
        layout = self.layout(vectors)
        point_ids = list(point_ids)
        output = {}
        for leg in layout.legs:
            values = vectors[leg]
            if not isinstance(values, self.backend.NativeBatch):
                raise TypeError("native spin vectors require NativeBatch values")
            if values.columns != 4 or values.rows not in (1, len(point_ids)):
                raise ValueError(
                    "spin vectors require one row or one four-component row per point"
                )
            if values.rows != 1 and values.row_ids != point_ids:
                raise ValueError(
                    "spin vector row identities differ from physical points"
                )
            require_finite_batch(values, "spin source components")
            if values.rows == 1:
                values = values.take_rows([0] * len(point_ids), row_ids=point_ids)
            # The dependency evaluator validates carrier and precision, including
            # empty batches, and multiplies the ordinary source phase once.
            values = self._phase_evaluators[leg].evaluate_batch(values)
            require_finite_batch(values, "crossed spin source components")
            output[leg] = values
        return output

    def prepare_decimal(self, vectors, point_ids):
        """Use tree input parsing; exact decimals cross only at native ingress."""
        layout = self.layout(vectors)
        point_ids = list(point_ids)
        parsed = _prepare_spin_vectors(
            vectors,
            allowed_legs=self.declarations.spin_legs,
            point_count=len(point_ids),
        )
        native = {
            leg: self.backend.NativeBatch.from_decimal(
                [
                    tuple(str(part) for part in value)
                    for point in parsed
                    for value in point[leg]
                ],
                len(point_ids),
                4,
                row_ids=point_ids,
                **self.bare.options,
            )
            for leg in layout.legs
        }
        return self.prepare_native(native, point_ids)


class NativeSpinSourceProgram:
    """Complete QCD model-rule views with literal gluon and directed colour sources.

    A joint class compiles an immutable spectator-helicity layout. The underlying
    model's ordinary helicities and initial-state average remain unchanged. The
    same supplied vectors enter Born, loop, local CT and external-field currents.
    This is an internal validated-control bridge, not public process dispatch.
    """

    def __init__(self, model, histories, *, spin_correlations):
        from .physical_model_correlations import NativeDirectedModelProgram

        self.model, self.bare = model, model.bare
        self.sources = NativeSpinSourcePlan(self.bare, spin_correlations)
        self.histories = tuple(histories)
        self._views = {(): NativeDirectedModelProgram(model, self.histories)}

    def _view(self, vectors):
        from .physical_model_correlations import NativeDirectedModelProgram

        layout = self.sources.layout(vectors)
        if layout.legs not in self._views:
            self._views[layout.legs] = NativeDirectedModelProgram(
                self.model, self.histories, spin_layout=layout
            )
        return self._views[layout.legs]

    def evaluate_native(
        self, momenta, scales_squared, *, model_values, spin_vectors=None
    ):
        vectors = self.sources.prepare_native(spin_vectors, range(len(momenta)))
        view = self._view(vectors)
        return view.evaluate_native(
            momenta,
            scales_squared,
            model_values=model_values,
            spin_sources=vectors,
        )

    def evaluate_decimal(
        self, momenta, scales_squared, *, model_values, spin_vectors=None
    ):
        vectors = self.sources.prepare_decimal(spin_vectors, range(len(momenta)))
        view = self._view(vectors)
        return view.evaluate_decimal(
            momenta,
            scales_squared,
            model_values=model_values,
            spin_sources=vectors,
        )


def replace_vector_sources(
    states, sources, *, point_rows, lane_ids, state_layout, backend
):
    """Gather native source columns into compact physical spectator rows."""
    if not sources:
        return states
    if any(
        type(leg) is not int or not 1 <= leg <= len(state_layout.legs)
        for leg in sources
    ):
        raise ValueError("spin source label is outside the physical vector slots")
    columns = []
    for leg in state_layout.legs:
        if leg.index + 1 in sources:
            if leg.kind not in {"epsilon", "epsilon_bar"}:
                raise ValueError("spin overrides require vector source slots")
            columns.append(
                sources[leg.index + 1].take_rows(point_rows, row_ids=lane_ids)
            )
        else:
            columns.append(
                states.take_columns(range(leg.offset, leg.offset + leg.components))
            )
    return backend.NativeBatch.concat_columns(columns)
