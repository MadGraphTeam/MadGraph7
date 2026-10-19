# SPDX-License-Identifier: 0BSD
"""Literal vector sources using the model-owned physical source contract."""

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
    owner: object
    contracts: tuple
    phase_states: tuple[bytes, ...]

    def require_owner(self, bare):
        if self.owner is not bare.prepared:
            raise ValueError("spin source layout belongs to a different bare plan")


class NativeSpinSourcePlan:
    """Validate exact source rows and reuse model-owned SourceIR crossing phases.

    Public source labels are one-based, as in tree correlations. Declared keys
    must form one joint class. Every supplied vector has four literal complex
    components; neither outgoing epsilon_bar slots nor initial-state crossing
    cause conjugation, momentum replacement, normalization or projection.
    """

    def __init__(self, bare, spin_correlations, *, source_contracts=None):
        from symbolica import E, S

        from pyamplicol.models._physics_ir import CrossingIR, SourceIR

        declarations = CorrelatorConfig(spin_correlations=spin_correlations)
        self.bare, self.backend = bare, bare.backend
        self.declarations = declarations
        edges = {
            edge.external_index + 1: edge
            for edge in bare.process.born[0].native.external_edges
        }
        contracts = (
            bare.source_contracts if source_contracts is None else source_contracts
        )
        source_irs = {}
        phase_evaluators = {}
        phase_states = {}
        components = tuple(
            S(f"loop_spin_source::component_{index}") for index in range(4)
        )
        for leg in declarations.spin_legs:
            if (
                leg not in edges
                or bare.state_kinds[leg - 1] not in {"epsilon", "epsilon_bar"}
            ):
                raise NotImplementedError(
                    "spin source bridge requires an external vector"
                )
            pdg = edges[leg].particle_pdg
            particle = edges[leg].particle
            orientation = (
                "self-conjugate"
                if particle.is_self_antiparticle
                else "antiparticle" if particle.is_antiparticle else "particle"
            )
            helicities = bare.state_layout.legs[leg - 1].helicities
            expected_kind = (
                "epsilon" if bare.state_layout.legs[leg - 1].incoming else "epsilon_bar"
            )
            if bare.state_kinds[leg - 1] != expected_kind:
                raise ValueError(
                    "spin source orientation differs from physical state port"
                )
            source = contracts.get(pdg)
            if not isinstance(source, SourceIR):
                raise ValueError(
                    "spin sources require explicit model-owned SourceIR contracts"
                )
            if (
                source.identity.pdg_label != pdg
                or source.identity.anti_pdg_label != particle.antiparticle.pdg_code
                or source.identity.orientation != orientation
                or source.identity.self_conjugate != particle.is_self_antiparticle
                or source.statistics != "boson"
                or source.wavefunction_family != "vector"
                or source.component_dimension != 4
                or source.basis != "lorentz-vector"
                or len(source.states) != len(helicities)
                or {state.helicity for state in source.states} != set(helicities)
                or any(
                    state.chirality != 0 or state.spin_state != state.helicity
                    for state in source.states
                )
            ):
                raise ValueError(
                    "source contract differs from the physical vector ports"
                )
            source_irs[leg] = source
            crossing = (
                source.crossing
                if bare.state_kinds[leg - 1] == "epsilon"
                else CrossingIR.identity()
            )
            # Compile only the source's declared phase; numerical model defaults
            # and supplied vector values never enter this structural preparation.
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
        self.source_irs = MappingProxyType(source_irs)
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
            layouts[legs] = SpinSourceLayout(
                legs,
                tuple(representatives.values()),
                bare.prepared,
                tuple(source_irs[leg] for leg in legs),
                tuple(phase_states[leg] for leg in legs),
            )
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
    """Complete QCD model-rule views with literal vector and directed colour sources.

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
