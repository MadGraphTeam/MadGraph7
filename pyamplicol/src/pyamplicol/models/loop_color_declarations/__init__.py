# SPDX-License-Identifier: 0BSD
"""Shipped formal-N_c continuations of loop models (M-QCD MQ-NC).

LC and NLC grade the counterterm (R2 and UV) and external-field couplings of
a loop model by their N_c dependence (design §6.7). A UFO states those
couplings at N_c = 3 only, so the dependence is model input: a
``LoopColorDeclaration`` (``models.loop_color_declaration``). pyAmpliCol
ships that input for the loop models it was written for, here as data files
(``loop_sm.json``: MG5_aMC 3.7.0 ``models/loop_sm``), so that LC and NLC work
on such a model without a hand-written declaration.

A shipped data set is keyed per vertex by a **structural digest** of the
vertex as the source writes it (``vertex_digest``): its counterterm type,
its particles (spin, colour, charge, mass and width parameters, never their
names or PDG codes), colour structures, Lorentz structures, its coupling
entries with each coupling's expression and orders, its loop-particle
groups, and the definitions of every internal, counterterm and helper
parameter those expressions read (transitively; external parameters by
their LHA address and type, never by a value). Vertex names, record
positions and the model's name and path never enter it, and neither does a
restriction: a restriction lives in the source's metadata (card values and
the zero parameters it fixes), never in its declarations, so every
restriction of a source has the same digests and resolves the subset of
vertices its couplings keep (``ct_sites.restricted_to_zero`` removes the
others). External-field (wave-function) factors are keyed the same way
(``field_digest``: the particle, its correction, its loop group and the
factor's Laurent coefficients). A vertex whose content differs in any of
these (a different coupling, colour structure, loop content or parameter
definition) has another digest and is not covered: an LC or NLC request
that reaches it is refused with the listed key ``colour-declaration-coverage``
(``loops.v2.refusals``), and full colour is unaffected.

Each data entry states the colour factor of every coupling of its vertex,
taken from the literature (``reference``: textbook UV renormalisation; the
general-N_c R2 of Draggiotis et al. 0903.0356 for QCD vertices and of
Garzelli et al. 0910.3130 for vertices with electroweak legs and QCD
loops). The continuation itself is the source's own coupling with the
declared identities (C_A = N_c, C_F = (N_c^2-1)/(2 N_c), T_F = 1/2,
N_col = N_c), which the model preparation proves equal to the source at
N_c = 3 (``loops.source_color``); ``tests/unit/test_loop_v2_nc_declaration.py``
proves every entry's coupling equal to its literature colour factor times an
N_c-independent rest, so no N_c = 3 number hides in a coupling. Vertices
whose fixed-SU(3) colour tensors no identity continues (the four-gluon R2)
are covered by a whole-rule recipe instead (``WholeRuleColorRecipe``), whose
general-N_c tensors ``loops.qcd_r2_color`` builds and checks against the
source at N_c = 3.

``resolve_shipped_declaration`` turns the shipped data into the ordinary
``LoopColorDeclaration`` of one source (its kernel and field keys at that
source's positions), which ``resolve_loop_color_declaration`` returns when
the source carries no declaration of its own and states the data's loop
scheme (its model metadata's loop declaration, else its shipped
``[loop.model]`` profile): a model that merely shares some vertex digests
gets none. Like every declaration it is declared, not certified.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

DATA_SCHEMA = "pyamplicol-loop-colour-data/1"
DIGEST_SCHEMA = "pyamplicol-ct-vertex-digest/1"
SHIPPED_DATA = ("loop_sm.json",)
# The colour factors a data entry may declare per coupling, as functions of
# the formal N_c (the literature's normalisation T_F = 1/2).
COLOUR_FACTORS = ("Nc", "CA", "CF", "TF")

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ---------------------------------------------------------------- canonical


def _plain(value: Any) -> Any:
    """A raw source record as plain JSON data (mappings as dicts, tuples as
    lists)."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _entries(value: Any) -> list[tuple[Any, Any]]:
    """The (key, value) pairs of a raw ``{"kind": "mapping"}`` record."""

    if not isinstance(value, Mapping) or value.get("kind") != "mapping":
        raise ValueError("expected a raw mapping record")
    return [(key, item) for key, item in value["entries"]]


def _key_items(key: Any) -> tuple[int, ...]:
    if not isinstance(key, Mapping) or key.get("kind") != "tuple":
        raise ValueError("expected a raw tuple key")
    return tuple(int(item) for item in key["items"])


def _value_text(value: Any) -> Any:
    """The structural content of a parameter or coupling value record."""

    if not isinstance(value, Mapping):
        return _plain(value)
    kind = value.get("kind")
    if kind == "expression":
        return {"expression": str(value["expression"])}
    if kind == "laurent":
        return {"laurent": [[int(p), str(t)] for p, t in value["terms"]]}
    return _plain(value)


class _Definitions:
    """The source's named definitions an expression can read, by name:
    internal parameters (their value expression), external parameters (LHA
    address and type, never a value), counterterm parameters (their Laurent
    coefficients) and helper functions (arguments and body)."""

    def __init__(self, raw: Any) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        for record in raw.collections["parameters"]:
            fields = record["fields"]
            if fields.get("nature") == "external":
                block = fields.get("lhablock")
                entry = {
                    "kind": "external",
                    "type": fields.get("type"),
                    "lhablock": None if block is None else str(block).upper(),
                    "lhacode": _plain(fields.get("lhacode")),
                }
            else:
                entry = {
                    "kind": "internal",
                    "type": fields.get("type"),
                    "value": _value_text(fields.get("value")),
                }
            self.records[str(record["name"])] = entry
        for record in raw.collections["counterterm_parameters"]:
            fields = record["fields"]
            self.records[str(record["name"])] = {
                "kind": "counterterm",
                "type": fields.get("type"),
                "value": _value_text(fields.get("value")),
            }
        for record in raw.collections["functions"]:
            fields = record["fields"]
            self.records[str(record["name"])] = {
                "kind": "function",
                "arguments": _plain(fields.get("arguments")),
                "expr": str(fields.get("expr")),
            }

    def closure(self, texts: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Every definition the expressions ``texts`` read, transitively."""

        found: dict[str, dict[str, Any]] = {}
        pending = list(texts)
        while pending:
            text = pending.pop()
            for name in _IDENTIFIER.findall(text):
                if name in found or name not in self.records:
                    continue
                record = self.records[name]
                found[name] = record
                pending.extend(_expression_texts(record))
        return dict(sorted(found.items()))


def _expression_texts(record: Mapping[str, Any]) -> list[str]:
    """The expression texts of a definition record (what it reads)."""

    if record["kind"] == "function":
        return [record["expr"]]
    value = record.get("value")
    if not isinstance(value, Mapping):
        return []
    if "expression" in value:
        return [str(value["expression"])]
    return [str(text) for _power, text in value.get("laurent", ())]


def _particle(raw: Any, index: int) -> dict[str, Any]:
    """A particle by its structure (spin, colour, charge, mass and width
    parameters, self-conjugacy, ghost number, Goldstone flag), never its
    name or PDG code: two particles of one structure are interchangeable in
    a vertex's N_c continuation."""

    fields = raw.collections["particles"][index]["fields"]
    charge = fields.get("charge_exact")
    return {
        "spin": int(fields["spin"]),
        "color": int(fields["color"]),
        "charge": _plain(charge if charge is not None else fields.get("charge")),
        "mass": str(fields["mass"]["name"]),
        "width": str(fields["width"]["name"]),
        "selfconjugate": bool(fields.get("selfconjugate")),
        "ghost": int(fields.get("GhostNumber", 0) or 0),
        "goldstone": bool(fields.get("goldstoneboson", False)),
    }


def _by_pdg(raw: Any) -> dict[int, int]:
    return {
        int(record["fields"]["pdg_code"]): index
        for index, record in enumerate(raw.collections["particles"])
    }


def _digest(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- digests


def vertex_record(raw: Any, vertex: Mapping[str, Any]) -> dict[str, Any]:
    """The structural record of one counterterm vertex that
    ``vertex_digest`` hashes (module docstring)."""

    definitions = _Definitions(raw)
    fields = vertex["fields"]
    couplings = []
    texts = []
    for key, reference in _entries(fields["couplings"]):
        record = raw.collections["couplings"][reference["id"]]
        if record["name"] != reference["name"]:
            raise ValueError("raw UFO coupling reference name/identity mismatch")
        value = _value_text(record["fields"]["value"])
        texts.extend(_expression_texts({"kind": "coupling", "value": value}))
        couplings.append(
            {
                "indices": list(_key_items(key)),
                "value": value,
                "order": sorted(
                    [str(name), int(power)]
                    for name, power in record["fields"]["order"]["entries"]
                ),
            }
        )
    couplings.sort(key=lambda item: item["indices"])
    particles = [_particle(raw, ref["id"]) for ref in fields["particles"]]
    for particle in particles:
        texts.extend((particle["mass"], particle["width"]))
    lorentz = []
    for reference in fields["lorentz"]:
        record = raw.collections["lorentz_structures"][reference["id"]]
        lorentz.append(
            {
                "spins": _plain(record["fields"]["spins"]),
                "structure": str(record["fields"]["structure"]),
            }
        )
    loops = [
        [[_particle(raw, item["id"]) for item in alternative] for alternative in group]
        for group in fields.get("loop_particles") or ()
    ]
    return {
        "schema": DIGEST_SCHEMA,
        "type": str(fields["type"]),
        "particles": particles,
        "color": [str(item) for item in fields["color"]],
        "lorentz": lorentz,
        "couplings": couplings,
        "loop_particles": loops,
        "definitions": definitions.closure(texts),
    }


def vertex_digest(raw: Any, vertex: Mapping[str, Any]) -> str:
    """The structural digest of a counterterm vertex (module docstring)."""

    return _digest(vertex_record(raw, vertex))


@dataclass(frozen=True, slots=True)
class FieldFactor:
    """One external-field factor of a source: the declaration's field key
    ``(particle_id, correction, loop_group_index)``, its source label and its
    structural digest."""

    key: tuple[int, str, int]
    label: str
    digest: str


def field_factors(raw: Any) -> tuple[FieldFactor, ...]:
    """The one-loop external-field factors of ``raw`` (the particles'
    ``counterterm`` entries with one unit coupling-order power, as
    ``loops.wavefunction.prepare_external_field_rules`` reads them), each
    with its structural digest: the particle, the correction, the loop
    group's species and the factor's Laurent coefficients with the
    definitions they read."""

    definitions = _Definitions(raw)
    orders = tuple(str(record["name"]) for record in raw.collections["coupling_orders"])
    by_pdg = _by_pdg(raw)
    factors = []
    for index, particle in enumerate(raw.collections["particles"]):
        fields = particle["fields"]
        values = fields.get("counterterm")
        if values is None:
            continue
        groups = fields.get("loop_particles") or ()
        for key, value in _entries(values):
            items = _key_items(key)
            powers, group = items[:-1], items[-1]
            if sum(powers) != 1 or len(powers) != len(orders):
                continue
            correction = orders[powers.index(1)]
            terms = [[int(p), str(t)] for p, t in _entries(value)]
            species = [
                [_particle(raw, by_pdg[int(pdg)]) for pdg in alternative]
                for alternative in groups[group]
            ]
            record = {
                "schema": DIGEST_SCHEMA,
                "particle": _particle(raw, index),
                "correction": correction,
                "loop_group": species,
                "terms": terms,
                "definitions": definitions.closure(text for _p, text in terms),
            }
            factors.append(
                FieldFactor(
                    (index, correction, group),
                    f"{particle['name']} {correction} group {group}",
                    _digest(record),
                )
            )
    return tuple(factors)


def field_digest(raw: Any, key: tuple[int, str, int]) -> str:
    """The structural digest of the external-field factor ``key``."""

    for factor in field_factors(raw):
        if factor.key == tuple(key):
            return factor.digest
    raise KeyError(f"no external-field factor {key!r}")


# ---------------------------------------------------------------------- data


@dataclass(frozen=True, slots=True)
class VertexColourData:
    """The N_c data of one counterterm vertex: ``continuation`` is
    ``"identities"`` (the source coupling with the declared identities;
    ``factors`` maps each coupling entry, ``"colour,lorentz,group"`` indices
    of the vertex, to its literature colour factor) or ``"whole-rule"``
    (``recipes``: one ``WholeRuleColorRecipe`` input per loop group)."""

    digest: str
    label: str
    kind: str
    continuation: str
    factors: Mapping[str, str]
    recipes: tuple[Mapping[str, Any], ...]
    reference: str


@dataclass(frozen=True, slots=True)
class FieldColourData:
    digest: str
    label: str
    factor: str
    reference: str


@dataclass(frozen=True, slots=True)
class ShippedColourData:
    """One shipped data file (``loop_sm.json``)."""

    name: str
    source: str
    scheme: Mapping[str, str]
    formal_symbol: str
    parameter_identities: Mapping[str, str]
    scheme_parameter_values: Mapping[str, str]
    vertices: Mapping[str, VertexColourData]
    fields: Mapping[str, FieldColourData]

    @classmethod
    def from_dict(cls, name: str, payload: Mapping[str, Any]) -> ShippedColourData:
        if payload.get("schema") != DATA_SCHEMA:
            raise ValueError(f"shipped colour data {name} has an unknown schema")
        vertices = {}
        for entry in payload["vertices"]:
            item = VertexColourData(
                str(entry["digest"]),
                str(entry["label"]),
                str(entry["kind"]),
                str(entry["continuation"]),
                dict(entry.get("factors", {})),
                tuple(dict(recipe) for recipe in entry.get("recipes", ())),
                str(entry["reference"]),
            )
            if item.continuation not in ("identities", "whole-rule"):
                raise ValueError(f"unknown continuation {item.continuation!r}")
            if item.continuation == "identities" and (
                not item.factors
                or not set(item.factors.values()) <= set(COLOUR_FACTORS)
            ):
                raise ValueError(f"{item.label}: invalid colour factors")
            if item.continuation == "whole-rule" and not item.recipes:
                raise ValueError(f"{item.label}: a whole rule needs its recipes")
            if item.digest in vertices:
                # Vertices of one structure share their entry (the builder
                # writes it once).
                raise ValueError(f"duplicate shipped vertex digest {item.digest}")
            vertices[item.digest] = item
        fields = {}
        for entry in payload["fields"]:
            item = FieldColourData(
                str(entry["digest"]),
                str(entry["label"]),
                str(entry["factor"]),
                str(entry["reference"]),
            )
            if item.factor not in COLOUR_FACTORS:
                raise ValueError(f"{item.label}: invalid colour factor")
            fields[item.digest] = item
        return cls(
            name,
            str(payload["source"]),
            dict(payload["scheme"]),
            str(payload["formal_symbol"]),
            dict(payload["parameter_identities"]),
            dict(payload["scheme_parameter_values"]),
            vertices,
            fields,
        )


@cache
def shipped_colour_data() -> tuple[ShippedColourData, ...]:
    """Every shipped data file, parsed once."""

    package = resources.files(__name__)
    return tuple(
        ShippedColourData.from_dict(
            name, json.loads(package.joinpath(name).read_text(encoding="utf-8"))
        )
        for name in SHIPPED_DATA
    )


@dataclass(frozen=True, slots=True)
class ColourCoverage:
    """How a shipped data set covers one source: the covered and uncovered
    counterterm vertices (record positions in ``counterterm_vertices``) and
    field factors (declaration field keys)."""

    data: ShippedColourData
    vertices: Mapping[int, VertexColourData]
    uncovered_vertices: tuple[int, ...]
    fields: Mapping[tuple[int, str, int], FieldColourData]
    uncovered_fields: tuple[tuple[int, str, int], ...]


def colour_coverage(raw: Any) -> Iterator[ColourCoverage]:
    """The coverage of ``raw`` by each shipped data set that covers at least
    one of its counterterm vertices."""

    vertices = raw.collections.get("counterterm_vertices", ())
    if not vertices:
        return
    digests = [vertex_digest(raw, vertex) for vertex in vertices]
    factors = field_factors(raw)
    for data in shipped_colour_data():
        covered = {
            index: data.vertices[digest]
            for index, digest in enumerate(digests)
            if digest in data.vertices
        }
        if not covered:
            continue
        fields = {
            factor.key: data.fields[factor.digest]
            for factor in factors
            if factor.digest in data.fields
        }
        yield ColourCoverage(
            data,
            covered,
            tuple(i for i in range(len(digests)) if i not in covered),
            fields,
            tuple(f.key for f in factors if f.key not in fields),
        )


def _declaration(raw: Any, coverage: ColourCoverage) -> LoopColorDeclaration:
    from pyamplicol.models.loop_color_declaration import (
        LoopColorDeclaration,
        WholeRuleColorRecipe,
    )
    from pyamplicol.models.loop_declaration import LoopSchemeDeclaration

    kernel_keys = []
    recipes = []
    for index, data in sorted(coverage.vertices.items()):
        vertex = raw.collections["counterterm_vertices"][index]
        if data.continuation == "whole-rule":
            recipes.extend(
                WholeRuleColorRecipe(
                    "counterterm_vertices",
                    int(vertex["id"]),
                    int(recipe["loop_group_index"]),
                    tuple(tuple(item) for item in recipe["order"]),
                    str(recipe["gauge_coupling"]),
                    recipe["contribution"],
                )
                for recipe in data.recipes
            )
            continue
        for key, _reference in _entries(vertex["fields"]["couplings"]):
            colour, lorentz, group = _key_items(key)
            kernel_keys.append(
                ("counterterm_vertices", int(vertex["id"]), colour, lorentz, group)
            )
    data = coverage.data
    return LoopColorDeclaration(
        scheme=LoopSchemeDeclaration(**data.scheme),
        formal_symbol=data.formal_symbol,
        parameter_identities=dict(data.parameter_identities),
        kernel_keys=tuple(kernel_keys),
        field_keys=tuple(sorted(coverage.fields)),
        scheme_parameter_values=dict(data.scheme_parameter_values),
        whole_rules=tuple(recipes),
    )


def _source_scheme(raw: Any) -> Any:
    """The loop scheme ``raw`` states: its model metadata's loop declaration
    (the public route attaches the effective one there), else its shipped
    ``[loop.model]`` profile (``loops.shipped_profiles``), else None."""

    from pyamplicol.models.loop_declaration import (
        LOOP_DECLARATION_KEY,
        LoopModelDeclaration,
    )

    payload = raw.metadata.get(LOOP_DECLARATION_KEY)
    if payload is not None:
        return LoopModelDeclaration.from_dict(payload).scheme
    from pyamplicol.loops.shipped_profiles import shipped_loop_profile

    profile = shipped_loop_profile(raw)
    return None if profile is None else profile.declaration.scheme


def resolve_shipped_declaration(raw: Any) -> LoopColorDeclaration | None:
    """The shipped declaration of ``raw`` (module docstring): the
    ``LoopColorDeclaration`` of the counterterm vertices and field factors a
    shipped data set covers, at ``raw``'s positions, or None when no data set
    covers any of its counterterm vertices, ``raw`` lacks a parameter the
    data's identities or scheme values name, or ``raw``'s own loop scheme
    (``_source_scheme``) is not the data's: another model sharing a few
    vertex digests with loop_sm (e.g. loop_qcd_qed_sm, 3 of its 499) gets
    none, so LC and NLC refuse it for want of colour data."""

    from pyamplicol.models.loop_declaration import LoopSchemeDeclaration

    scheme = None
    for coverage in colour_coverage(raw):
        data = coverage.data
        names = {str(record["name"]) for record in raw.collections["parameters"]}
        if not set(data.parameter_identities) | set(data.scheme_parameter_values) <= (
            names
        ):
            continue
        if scheme is None:
            scheme = _source_scheme(raw)
        if scheme == LoopSchemeDeclaration(**data.scheme):
            return _declaration(raw, coverage)
    return None


__all__ = [
    "COLOUR_FACTORS",
    "DATA_SCHEMA",
    "ColourCoverage",
    "FieldColourData",
    "FieldFactor",
    "ShippedColourData",
    "VertexColourData",
    "colour_coverage",
    "field_digest",
    "field_factors",
    "resolve_shipped_declaration",
    "shipped_colour_data",
    "vertex_digest",
    "vertex_record",
]
