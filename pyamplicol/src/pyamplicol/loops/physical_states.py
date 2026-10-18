# SPDX-License-Identifier: 0BSD
"""Immutable external-state packing over dependency-owned wavefunctions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod


@dataclass(frozen=True, slots=True)
class ExternalStateLeg:
    index: int
    kind: str
    incoming: bool
    massless: bool
    offset: int

    @property
    def components(self):
        return 1 if self.kind == "scalar" else 4

    @property
    def helicities(self):
        if self.kind == "scalar":
            return (0,)
        if self.kind in {"epsilon", "epsilon_bar"} and not self.massless:
            return (-1, 0, 1)
        return (-1, 1)


@dataclass(frozen=True, slots=True)
class ExternalStateLayout:
    """Leg order, component offsets and physical helicities are fixed at build time.

    Masslessness is structural: an exact fixed zero may resolve a named mass,
    while an adjustable mass keeps its massive state space. A runtime update to
    zero does not silently change the helicity catalogue. Native longitudinal
    states at zero spatial momentum retain the dependency's explicit error.
    """

    legs: tuple[ExternalStateLeg, ...]

    @classmethod
    def from_graph(cls, graph, state_kinds, model_expressions):
        edges = sorted(
            graph.native.external_edges, key=lambda edge: edge.external_index
        )
        if len(edges) != len(state_kinds) or [
            edge.external_index for edge in edges
        ] != list(range(len(edges))):
            raise ValueError("external state layout differs from native process legs")
        legs, offset = [], 0
        for edge, kind in zip(edges, state_kinds, strict=True):
            expected = {
                1: {"scalar"},
                2: {"u", "v", "u_bar", "v_bar"},
                3: {"epsilon", "epsilon_bar"},
            }
            if kind not in expected.get(edge.particle.spin, set()):
                raise NotImplementedError(
                    "physical states require compatible scalar, Dirac or vector species"
                )
            if edge.particle.spin == 2 and edge.particle.is_self_antiparticle:
                raise NotImplementedError(
                    "Majorana external states are outside the tree model envelope"
                )
            mass = edge.particle.mass
            for symbol in mass.get_all_symbols(False):
                if symbol.get_name() in model_expressions:
                    mass = mass.replace(symbol, model_expressions[symbol.get_name()])
            leg = ExternalStateLeg(
                edge.external_index,
                kind,
                edge.external_state == "incoming",
                mass == 0,
                offset,
            )
            legs.append(leg)
            offset += leg.components
        return cls(tuple(legs))

    @property
    def components(self):
        return sum(leg.components for leg in self.legs)

    @property
    def helicities(self):
        return tuple(product(*(leg.helicities for leg in self.legs)))

    @property
    def initial_spin_average(self):
        return prod(len(leg.helicities) for leg in self.legs if leg.incoming)

    def state_positions(self, external_states):
        """Scalar unit wavefunctions have no tensor ports or input symbols."""
        seen, positions = set(), {}
        for index, symbols in external_states:
            if index in seen or not 0 <= index < len(self.legs):
                raise ValueError("duplicate or invalid external state port")
            seen.add(index)
            leg = self.legs[index]
            if leg.kind == "scalar" or len(symbols) != leg.components:
                raise ValueError(
                    "external tensor ports disagree with state component layout"
                )
            positions.update(
                {
                    symbol.get_name(): leg.offset + component
                    for component, symbol in enumerate(symbols)
                }
            )
        if seen != {leg.index for leg in self.legs if leg.kind != "scalar"}:
            raise ValueError("missing external spinor or vector tensor ports")
        return positions

    def prepare_native(self, backend, momenta, helicities, *, partial=False):
        """At most two native bulk calls: width-one and width-four state blocks."""
        count = len(self.legs)
        if momenta.columns != 4 * count:
            raise ValueError("external state momenta do not match leg layout")
        if not helicities or momenta.rows % len(helicities):
            raise ValueError("external state rows do not match helicity layout")
        if any(
            len(row) != count
            or any(
                value not in leg.helicities
                for value, leg in zip(row, self.legs, strict=True)
            )
            for row in helicities
        ):
            raise ValueError(
                "external state helicity is outside its declared state space"
            )
        groups, statuses = [], []
        for width in sorted({leg.components for leg in self.legs}):
            indices = tuple(leg.index for leg in self.legs if leg.components == width)
            columns = [4 * leg + component for leg in indices for component in range(4)]
            selected = (
                momenta if len(indices) == count else momenta.take_columns(columns)
            )
            ids = [lane * count + leg for lane in momenta.row_ids for leg in indices]
            inputs = selected.reshape_rows(momenta.rows * len(indices), 4, ids)
            kinds = [self.legs[leg].kind for _ in momenta.row_ids for leg in indices]
            states_helicities = [
                helicities[row % len(helicities)][leg]
                for row in range(momenta.rows)
                for leg in indices
            ]
            if partial:
                values, group_statuses = backend.wavefunctions_partial_batch(
                    inputs, states_helicities, kinds, components=width
                )
                statuses.extend(group_statuses)
            else:
                values = backend.wavefunctions_batch(
                    inputs, states_helicities, kinds, components=width
                )
            groups.append((indices, width, values))
        return NativeStateBlocks(self, backend, tuple(groups), tuple(statuses))


@dataclass(frozen=True, slots=True)
class NativeStateBlocks:
    layout: ExternalStateLayout
    backend: object
    groups: tuple
    statuses: tuple

    def assemble(self, lane_ids):
        """Pack successful whole lanes, retaining the caller's original IDs."""
        lane_ids = list(lane_ids)
        count = len(self.layout.legs)
        blocks, positions, offset = [], {}, 0
        for indices, width, values in self.groups:
            wanted = [lane * count + leg for lane in lane_ids for leg in indices]
            if values.row_ids != wanted:
                available = {row_id: row for row, row_id in enumerate(values.row_ids)}
                if any(row_id not in available for row_id in wanted):
                    raise ValueError(
                        "cannot assemble an unsuccessful external state lane"
                    )
                values = values.take_rows([available[row_id] for row_id in wanted])
            blocks.append(
                values.reshape_rows(len(lane_ids), len(indices) * width, lane_ids)
            )
            for index, leg in enumerate(indices):
                positions[leg] = tuple(
                    range(offset + index * width, offset + (index + 1) * width)
                )
            offset += len(indices) * width
        if len(blocks) == 1:
            return blocks[0]
        joined = self.backend.NativeBatch.concat_columns(blocks)
        return joined.take_columns(
            [column for leg in self.layout.legs for column in positions[leg.index]]
        )
