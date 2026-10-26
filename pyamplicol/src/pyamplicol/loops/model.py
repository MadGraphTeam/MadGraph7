# SPDX-License-Identifier: 0BSD
"""Lossless loop-model scalar and local covariant-kernel preparation.

This is not a renormalization-scheme or physics completeness certificate. UV
insertion and R2 loop-group matching remain process-generation responsibilities.
In particular, preserving an SU(3) UFO does not establish its continuation in Nc.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .model_color import prepare_color_expression
from .model_expressions import (
    ScalarDefinition,
    lower_scalar_definitions,
    parse_exact_ufo,
)
from .orders import canonical_order
from .scalar_frontend import (
    ScalarExpression,
    ScalarProgram,
    SourceScalarDefinition,
    require_scalar_value,
)


@dataclass(frozen=True, slots=True)
class KernelBasisEntry:
    """One original color x Lorentz x loop-group entry, never a summed vertex."""

    collection: str
    vertex_id: int
    vertex_name: str
    kind: str
    color_index: int
    lorentz_index: int
    loop_group_index: int | None
    particle_ids: tuple[int, ...]
    particle_names: tuple[str, ...]
    color: str
    lorentz_id: int
    lorentz_name: str
    lorentz: str
    # Alternatives within the selected UFO group retain their original species
    # sequence and multiplicity. They are not summed into an inferred Nf.
    loop_particles: tuple[tuple[int, ...], ...] | None
    coupling: ScalarDefinition | SourceScalarDefinition
    order: tuple[tuple[str, int], ...]
    scalar_program: ScalarProgram | None = field(default=None, repr=False)

    def scalar_expression(self, power: int | None = None) -> ScalarExpression:
        """Retain the source diagnostic when selecting a Laurent coefficient."""
        from symbolica import E

        value = self.coupling.value
        return ScalarExpression(
            value.expression if power is None else value.coefficient(power),
            self.coupling.diagnostic
            if isinstance(self.coupling, SourceScalarDefinition)
            else E("0"),
        )

    def scalar_value(self, power: int | None = None):
        return require_scalar_value(
            self.scalar_expression(power), f"kernel {self.key}", self.scalar_program
        )

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.collection,
            self.vertex_id,
            self.color_index,
            self.lorentz_index,
            self.loop_group_index,
        )


@dataclass(frozen=True, slots=True)
class NativeLocalKernel:
    """Dependency-owned semantic tensors in the same local half-edge frame.

    All momenta are incoming, with the native q_edge labels. ``leg_half_edges``
    maps the UFO leg sequence to native half-edge IDs; callers rebind these with
    native tensor/graph APIs. No momentum routing or contraction is done here.
    Tensor axes retain their native order, which can differ from the leg order.
    Inspect each tensor's ``axes`` or use native ``permute_axes`` before accessing
    components by a chosen leg ordering.
    """

    color: Any = field(repr=False)
    lorentz: Any = field(repr=False)
    leg_half_edges: tuple[int, ...]
    particle_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoopModelIR:
    name: str
    raw: Any = field(repr=False)
    scalars: Any = field(repr=False)
    kernels: tuple[KernelBasisEntry, ...]
    # Retain helper definitions without evaluating arbitrary Python bodies.
    functions: tuple[Any, ...]
    scalar_program: ScalarProgram | None = field(default=None, repr=False)
    stage: str = field(default="exact-local-kernel-preparation", init=False)
    scheme_validated: bool = field(default=False, init=False)
    completeness_validated: bool = field(default=False, init=False)
    arbitrary_precision_physics_validated: bool = field(default=False, init=False)
    formal_nc_continuation_validated: bool = field(default=False, init=False)
    eligibility: str = field(default="requires-process-loop-group-matching", init=False)

    def counterterms(self, kind: str) -> tuple[KernelBasisEntry, ...]:
        if kind not in {"UV", "R2"}:
            raise ValueError("counterterm family must be UV or R2")
        return tuple(
            entry
            for entry in self.kernels
            if (entry.kind.startswith("UV") if kind == "UV" else entry.kind == "R2")
        )


def _reference(raw: Any, value: Any, collection: str) -> Any:
    if value.get("kind") != "reference" or value.get("collection") != collection:
        raise ValueError(f"expected a raw {collection} object reference")
    record = raw.collections[collection][value["id"]]
    if record["name"] != value["name"]:
        raise ValueError("raw UFO reference name/identity mismatch")
    return record


def build_loop_model(
    raw: Any, *, scalar_program: ScalarProgram | None = None
) -> LoopModelIR:
    """Prepare exact unsummed kernels, optionally retaining a paired scalar owner.

    An explicitly supplied program must belong to this exact raw object. Its
    definitions and library remain intact; selecting coefficients never converts
    diagnostic-bearing source records into legacy value-only definitions.
    """
    if scalar_program is None:
        scalars = lower_scalar_definitions(raw)
    else:
        if scalar_program.library.raw is not raw:
            raise ValueError(
                "scalar program belongs to a different raw source instance"
            )
        scalars = scalar_program.definitions
        expected = {
            (collection, record["id"]): record["name"]
            for collection in ("parameters", "counterterm_parameters", "couplings")
            for record in raw.collections[collection]
        }
        if set(scalars) != set(expected) or any(
            not isinstance(scalars[key], SourceScalarDefinition)
            or (scalars[key].collection, scalars[key].id, scalars[key].name)
            != (*key, name)
            for key, name in expected.items()
        ):
            raise ValueError("scalar program declarations differ from the raw source")
    entries = []
    for collection in ("vertices", "counterterm_vertices"):
        for vertex in raw.collections[collection]:
            fields = vertex["fields"]
            kind = "tree" if collection == "vertices" else fields["type"]
            if kind != "tree" and kind != "R2" and not kind.startswith("UV"):
                raise NotImplementedError(f"unsupported counterterm type {kind!r}")
            particles = tuple(
                _reference(raw, value, "particles") for value in fields["particles"]
            )
            for key, coupling_ref in fields["couplings"]["entries"]:
                indices = key["items"]
                expected = 2 if kind == "tree" else 3
                if len(indices) != expected or any(
                    isinstance(index, bool) or not isinstance(index, int) or index < 0
                    for index in indices
                ):
                    raise ValueError(f"invalid basis indices for {vertex['name']}")
                color, lorentz = indices[:2]
                group = indices[2] if expected == 3 else None
                loop_particles = None
                if group is not None:
                    groups = fields["loop_particles"]
                    if not groups:
                        # UFO/MadLoop: an empty outer list gives one unrestricted
                        # alternative for each declared coupling group. The raw
                        # empty list remains untouched in the source sidecar.
                        loop_particles = ((),)
                    elif group >= len(groups):
                        raise ValueError(f"absent loop group in {vertex['name']}")
                    else:
                        loop_particles = tuple(
                            tuple(
                                _reference(raw, item, "particles")["id"]
                                for item in alternative
                            )
                            for alternative in groups[group]
                        )
                coupling_record = _reference(raw, coupling_ref, "couplings")
                lorentz_record = _reference(
                    raw, fields["lorentz"][lorentz], "lorentz_structures"
                )
                entries.append(
                    KernelBasisEntry(
                        collection,
                        vertex["id"],
                        vertex["name"],
                        kind,
                        color,
                        lorentz,
                        group,
                        tuple(p["id"] for p in particles),
                        tuple(p["name"] for p in particles),
                        fields["color"][color],
                        lorentz_record["id"],
                        lorentz_record["name"],
                        lorentz_record["fields"]["structure"],
                        loop_particles,
                        scalars[("couplings", coupling_record["id"])],
                        canonical_order(coupling_record["fields"]["order"]["entries"]),
                        scalar_program,
                    )
                )
    return LoopModelIR(
        raw.name,
        raw,
        scalars,
        tuple(entries),
        raw.collections["functions"],
        scalar_program,
    )


class CovariantKernelCatalogue:
    """Lower single local components using public HepKit Feynman-rule APIs.

    The structural model supplies species/propagator metadata only. Its cached
    numeric parameter/coupling values are never used for a kernel coefficient.
    Color and Lorentz structures come from the exact raw sidecar. Public native
    one-vertex diagrams provide index localization, color tensors and momenta;
    no private FeynKit mapping rules are duplicated here.
    """

    def __init__(self, model: LoopModelIR, structural_model: Any):
        self.model = model
        self._template = (
            deepcopy(dict(structural_model))
            if isinstance(structural_model, Mapping)
            else json.loads(structural_model.to_json())
        )
        self._cache: dict[tuple[Any, ...], tuple[Any, tuple[int, ...]]] = {}
        species = {
            particle["name"]: particle for particle in self._template["particles"]
        }
        for record in model.raw.collections["particles"]:
            particle = species.get(record["name"])
            if particle is None or any(
                particle[field] != record["fields"][field]
                for field in ("pdg_code", "spin", "color")
            ):
                raise ValueError(f"structural model species mismatch: {record['name']}")

    def _component(
        self, entry: KernelBasisEntry, domain: str
    ) -> tuple[Any, tuple[int, ...]]:
        from symbolica.community import hepkit as hep
        from ufo_model_loader.symbolica_processing import wrap_indices

        source = entry.color if domain == "color" else entry.lorentz
        key = (entry.particle_names, domain, source)
        # JSON is the native public model interchange. Make an isolated local
        # component without changing the supplied model or its scalar defaults.
        payload = dict(self._template)
        spins = [
            self.model.raw.collections["particles"][index]["fields"]["spin"]
            for index in entry.particle_ids
        ]
        declared_spins = self.model.raw.collections["lorentz_structures"][
            entry.lorentz_id
        ]["fields"]["spins"]
        # Older UFOs label a ghost's Lorentz representation as scalar spin 1.
        # Native models require the particle's -1 marker; this changes no tensor.
        if tuple(map(abs, spins)) != tuple(map(abs, declared_spins)):
            raise NotImplementedError(
                f"inconsistent Lorentz spin signature at {entry.vertex_name}"
            )
        if key in self._cache:
            return self._cache[key]
        payload["vertex_rules"] = [
            {
                "name": "LOOP_LOCAL",
                "particles": list(entry.particle_names),
                "color_structures": [
                    prepare_color_expression(
                        source,
                        [
                            self.model.raw.collections["particles"][index]["fields"][
                                "color"
                            ]
                            for index in entry.particle_ids
                        ],
                    ).to_canonical_string()
                    if domain == "color"
                    else "1"
                ],
                "lorentz_structures": ["LOOP_LOCAL_TENSOR"],
                "couplings": [["LOOP_LOCAL_ONE"]],
            }
        ]
        payload["lorentz_structures"] = [
            {
                "name": "LOOP_LOCAL_TENSOR",
                "spins": spins,
                "structure": wrap_indices(parse_exact_ufo(source)).to_canonical_string()
                if domain == "lorentz"
                else "1",
            }
        ]
        payload["couplings"] = [
            {"name": "LOOP_LOCAL_ONE", "expression": "1", "orders": [], "value": None}
        ]
        native_model = hep.Model.from_json(json.dumps(payload))
        dot = 'digraph local { ext [style=invis]; v [int_id="LOOP_LOCAL"]; '
        dot += (
            " ".join(
                f"ext -> v [particle={json.dumps(name)}];"
                for name in entry.particle_names
            )
            + " }"
        )
        graph = hep.FeynmanDiagram.from_dot(native_model, dot).apply_feynman_rules()
        value = native_model.expand_couplings(graph.vertices[0].numerator_expression())
        # DOT matching owns the UFO-leg assignment. All edges entered in the
        # same order, including repeated species; expose that native frame.
        half_edges = tuple(half.id for half in graph.half_edges)
        result = value, half_edges
        self._cache[key] = result
        return result

    def lower(self, entry: KernelBasisEntry) -> NativeLocalKernel:
        if not any(entry is candidate for candidate in self.model.kernels):
            raise ValueError("kernel entry does not belong to this loop model")
        color, color_half_edges = self._component(entry, "color")
        lorentz, lorentz_half_edges = self._component(entry, "lorentz")
        if color_half_edges != lorentz_half_edges:
            raise ValueError("native component half-edge frames disagree")
        return NativeLocalKernel(color, lorentz, color_half_edges, entry.particle_names)
