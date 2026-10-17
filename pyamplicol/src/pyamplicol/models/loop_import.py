# SPDX-License-Identifier: 0BSD
"""Unlowered NLO UFO data captured before tree-model normalization.

This is an import boundary, not a loop evaluator or a completeness certificate.
In particular, a scalar expression may reference Laurent-valued CT parameters;
it must not be interpreted as an epsilon-zero coefficient before substitution.
Expressions remain UFO strings. No series algebra or numerical evaluation is
performed here.

Python numeric literals have already been evaluated at this boundary. Exact
source recovery still requires matching parameter/coupling declarations to
their imported objects and retaining ``ast.get_source_segment`` for their
``value`` arguments. Computed or dynamically reassigned values also require an
exact lowering policy; source text alone does not certify their evaluation.
"""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

RAW_LOOP_MODEL_KIND = "pyamplicol-raw-loop-model"
RAW_LOOP_MODEL_SCHEMA_VERSION = 1

_COLLECTIONS = {
    "parameters": "all_parameters",
    "counterterm_parameters": "all_CTparameters",
    "particles": "all_particles",
    "couplings": "all_couplings",
    "vertices": "all_vertices",
    "counterterm_vertices": "all_CTvertices",
    "lorentz_structures": "all_lorentz",
    "coupling_orders": "all_orders",
    "functions": "all_functions",
    "propagators": "all_propagators",
    "form_factors": "all_form_factors",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RawLoopModel:
    """Immutable source records, independent of the restricted tree-model IR.

    Every record has ``id``, ``name``, ``ufo_type`` and ``fields``. Named UFO objects in
    fields become typed references. Maps with non-string keys (including CT
    coupling index tuples) are represented by ordered key/value entries. UFO
    expression strings and all explicit Laurent powers are retained verbatim.
    Numeric literals already evaluated by Python retain their runtime value;
    their source's lost decimal digits cannot be reconstructed by this importer.
    """

    name: str
    collections: Mapping[str, tuple[Mapping[str, object], ...]]
    metadata: Mapping[str, object]
    capabilities: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.name or set(self.collections) != set(_COLLECTIONS):
            raise ValueError(
                "raw loop model requires a name and all source collections"
            )
        for collection, records in self.collections.items():
            for index, record in enumerate(records):
                if set(record) != {"id", "name", "ufo_type", "fields"}:
                    raise ValueError(f"invalid raw UFO record in {collection}")
                name = record["name"]
                if not isinstance(name, str) or not name or record["id"] != index:
                    raise ValueError(f"invalid raw UFO name or ID in {collection}")
                if not isinstance(record["ufo_type"], str) or not isinstance(
                    record["fields"], Mapping
                ):
                    raise ValueError(f"invalid raw UFO fields for {collection}.{name}")
        object.__setattr__(self, "collections", _freeze(self.collections))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        object.__setattr__(self, "capabilities", _freeze(self.capabilities))

    def record(self, collection: str, name: str) -> Mapping[str, object]:
        matches = [
            record for record in self.collections[collection] if record["name"] == name
        ]
        if not matches:
            raise KeyError(f"unknown raw UFO record {collection}.{name}")
        if len(matches) != 1:
            raise ValueError(
                f"ambiguous raw UFO name {collection}.{name}; use its record ID"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": RAW_LOOP_MODEL_KIND,
            "schema_version": RAW_LOOP_MODEL_SCHEMA_VERSION,
            "name": self.name,
            "collections": _plain(self.collections),
            "metadata": _plain(self.metadata),
            "capabilities": _plain(self.capabilities),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RawLoopModel:
        if (
            payload.get("kind") != RAW_LOOP_MODEL_KIND
            or payload.get("schema_version") != RAW_LOOP_MODEL_SCHEMA_VERSION
        ):
            raise ValueError("unsupported raw loop model schema; regenerate the model")
        if set(payload) != {
            "kind",
            "schema_version",
            "name",
            "collections",
            "metadata",
            "capabilities",
        }:
            raise ValueError("invalid raw loop model fields")
        for field in ("collections", "metadata", "capabilities"):
            if not isinstance(payload[field], Mapping):
                raise ValueError(f"raw loop model {field} must be an object")
        if not isinstance(payload["name"], str):
            raise ValueError("raw loop model name must be a string")
        return cls(
            payload["name"],
            payload["collections"],
            payload["metadata"],
            payload["capabilities"],
        )


def _expression(value: object, context: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an expression or number, not a boolean")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if (
        isinstance(value, complex)
        and math.isfinite(value.real)
        and math.isfinite(value.imag)
    ):
        return f"complex({value.real!r},{value.imag!r})"
    if isinstance(value, Decimal) and value.is_finite():
        return str(value)
    if isinstance(value, Fraction):
        return f"({value.numerator}/{value.denominator})"
    raise ValueError(
        f"unsupported UFO expression value in {context}: {type(value).__name__}"
    )


def _expression_value(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {"kind": "expression", "expression": _expression(value, context)}
    terms = []
    for power, coefficient in value.items():
        if isinstance(power, bool) or not isinstance(power, int):
            raise ValueError(f"{context} has a non-integer epsilon power: {power!r}")
        terms.append([power, _expression(coefficient, context)])
    return {"kind": "laurent", "terms": terms}


def _encode(
    value: object, references: Mapping[int, tuple[str, int, str]], context: str
) -> object:
    reference = references.get(id(value))
    if reference is not None:
        return {
            "kind": "reference",
            "collection": reference[0],
            "id": reference[1],
            "name": reference[2],
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (complex, Decimal, Fraction)):
        return {"kind": "number", "expression": _expression(value, context)}
    if isinstance(value, type):
        return {"kind": "type", "name": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, Mapping):
        # Use entries even for string keys: a UFO mapping cannot then collide
        # with one of this schema's tagged references or expression records.
        return {
            "kind": "mapping",
            "entries": [
                [_encode(key, references, context), _encode(item, references, context)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_encode(item, references, context) for item in value],
        }
    if isinstance(value, list):
        return [_encode(item, references, context) for item in value]
    raise ValueError(f"unsupported raw UFO value in {context}: {type(value).__name__}")


def _objects(module: ModuleType, registry: str) -> list[Any]:
    values = getattr(module, registry, None)
    if values is None:
        values = getattr(getattr(module, "object_library", None), registry, ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"UFO registry {registry} must be a sequence")
    result: list[Any] = []
    seen: set[int] = set()
    for value in values:
        name = getattr(value, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(f"UFO registry {registry} contains an unnamed object")
        if id(value) in seen:
            continue
        seen.add(id(value))
        result.append(value)
    return result


def capture_raw_loop_model(
    module: ModuleType,
    *,
    name: str | None = None,
    restriction: Mapping[str, object] | None = None,
) -> RawLoopModel:
    """Snapshot an imported UFO before any tree normalization or restriction."""
    objects = {
        key: _objects(module, registry) for key, registry in _COLLECTIONS.items()
    }
    # Some UFOs register CTVertex objects in both registries, or only all_vertices.
    ct_vertices = objects["counterterm_vertices"]
    ct_ids = {id(vertex) for vertex in ct_vertices}
    for vertex in objects["vertices"]:
        if type(vertex).__name__ == "CTVertex" and id(vertex) not in ct_ids:
            ct_vertices.append(vertex)
            ct_ids.add(id(vertex))
    objects["vertices"] = [v for v in objects["vertices"] if id(v) not in ct_ids]
    ct_parameter_ids = {
        id(parameter) for parameter in objects["counterterm_parameters"]
    }
    objects["parameters"] = [
        p for p in objects["parameters"] if id(p) not in ct_parameter_ids
    ]
    # A UFO may attach a custom propagator without exporting all_propagators.
    propagator_ids = {id(prop) for prop in objects["propagators"]}
    for particle in objects["particles"]:
        prop = getattr(particle, "propagator", None)
        if (
            prop is not None
            and not isinstance(prop, str)
            and id(prop) not in propagator_ids
        ):
            objects["propagators"].append(prop)
            propagator_ids.add(id(prop))
    references = {
        id(value): (collection, index, str(value.name))
        for collection, values in objects.items()
        for index, value in enumerate(values)
    }
    collections: dict[str, tuple[Mapping[str, object], ...]] = {}
    for collection, values in objects.items():
        records = []
        for index, value in enumerate(values):
            context = f"{collection}.{value.name}"
            fields = {
                key: (
                    _expression_value(item, f"{context}.value")
                    if key == "value"
                    and collection
                    in {"parameters", "counterterm_parameters", "couplings"}
                    else _encode(item, references, f"{context}.{key}")
                )
                for key, item in vars(value).items()
            }
            records.append(
                {
                    "id": index,
                    "name": value.name,
                    "ufo_type": type(value).__name__,
                    "fields": fields,
                }
            )
        collections[collection] = tuple(records)

    # Preserve explicitly declared conventions; never infer a scheme from a
    # model name, a parameter named lhv, or the presence of gamma-five tensors.
    metadata = {
        key: _encode(value, references, f"model.{key}")
        for key, value in vars(module).items()
        if not key.startswith("_")
        and not key.startswith("all_")
        and not isinstance(value, ModuleType)
        and not callable(value)
        and type(value).__module__ != "__future__"
    }
    metadata["restriction"] = dict(
        restriction or {"applied": False, "name": None, "card": None}
    )
    particles = objects["particles"]
    ct_types = sorted(
        {str(getattr(vertex, "type", "unknown")) for vertex in ct_vertices}
    )
    capabilities: dict[str, object] = {
        "stage": "raw-import",
        "counterterm_vertex_types": ct_types,
        "has_uv_vertices": any(kind.startswith("UV") for kind in ct_types),
        "has_r2_vertices": "R2" in ct_types,
        "has_laurent_parameters": any(
            isinstance(getattr(item, "value", None), Mapping)
            for key in ("parameters", "counterterm_parameters")
            for item in objects[key]
        ),
        "has_laurent_couplings": any(
            isinstance(getattr(item, "value", None), Mapping)
            for item in objects["couplings"]
        ),
        "ghost_particles": [
            str(p.name) for p in particles if getattr(p, "spin", None) == -1
        ],
        "goldstone_particles": [
            str(p.name)
            for p in particles
            if bool(getattr(p, "GoldstoneBoson", getattr(p, "goldstoneboson", False)))
        ],
        "perturbative_orders": [
            str(order.name)
            for order in objects["coupling_orders"]
            if getattr(order, "perturbative_expansion", 0) > 0
        ],
        "uv_complete": None,
        "model_r2_complete": None,
        "dimensional_kernels_validated": False,
        "complex_mass_kernels_validated": False,
        "loop_runtime_supported": False,
        "expression_strings_preserved": True,
        "numeric_literal_source_preserved": False,
        "arbitrary_precision_inputs_validated": False,
    }
    return RawLoopModel(name or module.__name__, collections, metadata, capabilities)


@contextmanager
def imported_ufo_module(source: Path) -> Iterator[ModuleType]:
    """Use the loader's import convention while isolating same-name UFOs.

    The caller owns the existing model-load lock/environment scope. Keep this
    context open through tree normalization so both readers see one import.
    """
    source = source.resolve()
    if not (source / "__init__.py").is_file():
        raise ValueError(f"UFO model directory has no __init__.py: {source}")
    names = {source.name, *(path.stem for path in source.glob("*.py"))}

    def owned(name: str) -> bool:
        return name.split(".", 1)[0] in names

    saved = {name: module for name, module in sys.modules.items() if owned(name)}
    previous_path = sys.path[:]
    for name in saved:
        del sys.modules[name]
    sys.path[:0] = [str(source), str(source.parent)]
    try:
        yield importlib.import_module(source.name)
    finally:
        for name in tuple(sys.modules):
            if owned(name):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = previous_path


def load_raw_loop_model(
    source: str | Path, *, restriction: str = "default"
) -> RawLoopModel:
    """Read NLO UFO records without requiring tree compilation or loop kernels."""
    from .loading import _loader_restriction_name, _sanitized_model_environment

    path = Path(source).expanduser().resolve()
    selected = _loader_restriction_name(path, restriction)
    with _sanitized_model_environment(), imported_ufo_module(path) as module:
        return capture_raw_loop_model(
            module, name=path.name, restriction=raw_restriction(path, selected)
        )


def raw_restriction(source: Path, restriction: str | None) -> dict[str, object]:
    """Keep the restriction card for later loop lowering; do not apply it here."""
    card = (
        None
        if restriction in {"full", "none"}
        else source / f"restrict_{restriction or 'default'}.dat"
    )
    if card is not None and restriction is not None and not card.is_file():
        raise ValueError(f"UFO restriction card does not exist: {card}")
    return {
        "applied": False,
        "name": None
        if card is None or not card.is_file()
        else restriction or "default",
        "card": None
        if card is None or not card.is_file()
        else card.read_text(encoding="utf-8"),
    }
