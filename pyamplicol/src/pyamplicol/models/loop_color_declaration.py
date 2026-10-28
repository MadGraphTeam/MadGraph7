# SPDX-License-Identifier: 0BSD
"""Explicit SU(Nc) source-model data carried by the existing model bundle.

These records declare a continuation; source tensor/scalar conformance and
selected-process coverage remain mandatory when the owned model is prepared.
No tensor implementation or compiled numerical state is serialized here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias, overload

from .loop_declaration import LoopSchemeDeclaration

if TYPE_CHECKING:
    from .loading import CompiledModel
    from .loop_import import RawLoopModel

LOOP_COLOR_DECLARATION_KEY = "pyamplicol_loop_color"
KernelKey: TypeAlias = tuple[str, int, int, int, int | None]
FieldKey: TypeAlias = tuple[int, str, int]
CouplingOrder: TypeAlias = tuple[tuple[str, int], ...]


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("colour declaration requires a sequence")
    return value


def _index(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("source colour indices must be nonnegative integers")
    return value


def _name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source colour names/expressions must be nonempty strings")
    return value


def _collection(value: object) -> str:
    if value not in ("vertices", "counterterm_vertices"):
        raise ValueError("source colour key has an unknown vertex collection")
    return str(value)


def _kernel_key(value: object) -> KernelKey:
    row = _sequence(value)
    if len(row) != 5:
        raise ValueError("source colour kernel key requires five entries")
    return (
        _collection(row[0]),
        _index(row[1]),
        _index(row[2]),
        _index(row[3]),
        None if row[4] is None else _index(row[4]),
    )


def _field_key(value: object) -> FieldKey:
    row = _sequence(value)
    if len(row) != 3:
        raise ValueError("source colour field key requires three entries")
    return _index(row[0]), _name(row[1]), _index(row[2])


def _order(value: object) -> CouplingOrder:
    rows = tuple(_sequence(row) for row in _sequence(value))
    if any(len(row) != 2 for row in rows):
        raise ValueError("source colour coupling orders require name/power pairs")
    result = tuple(sorted((_name(row[0]), _index(row[1])) for row in rows))
    if len({name for name, _ in result}) != len(result):
        raise ValueError("duplicate source colour coupling order")
    return result


@dataclass(frozen=True, slots=True)
class WholeRuleColorRecipe:
    """Input to the existing independently checked whole-rule constructor."""

    collection: str
    vertex_id: int
    loop_group_index: int
    order: CouplingOrder
    gauge_coupling: str
    contribution: Literal["gauge", "fundamental-dirac"]
    kind: Literal["qcd-four-gluon-r2-hv"] = field(
        default="qcd-four-gluon-r2-hv", init=False
    )

    def __post_init__(self) -> None:
        _collection(self.collection)
        _index(self.vertex_id)
        _index(self.loop_group_index)
        _name(self.gauge_coupling)
        object.__setattr__(self, "order", _order(self.order))
        if self.contribution not in ("gauge", "fundamental-dirac"):
            raise ValueError("unknown whole-rule colour contribution")

    @property
    def key(self) -> tuple[str, int, int, CouplingOrder]:
        return self.collection, self.vertex_id, self.loop_group_index, self.order

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "collection": self.collection,
            "vertex_id": self.vertex_id,
            "loop_group_index": self.loop_group_index,
            "order": [list(item) for item in self.order],
            "gauge_coupling": self.gauge_coupling,
            "contribution": self.contribution,
        }

    @classmethod
    def from_dict(cls, value: object) -> WholeRuleColorRecipe:
        if (
            not isinstance(value, Mapping)
            or set(value) != set(cls.__dataclass_fields__)
            or value["kind"] != "qcd-four-gluon-r2-hv"
        ):
            raise ValueError("invalid whole-rule colour recipe")
        return cls(**{key: item for key, item in value.items() if key != "kind"})


@dataclass(frozen=True, slots=True)
class LoopColorDeclaration:
    """Exact explicit model input, never a model-name capability certificate.

    Formula strings use the existing exact UFO scalar syntax. ``formal_symbol``
    names the sole formal Nc placeholder and must differ from source parameter
    names. Scheme ties are exact decimal strings/complex pairs, matching the
    existing fixed-parameter interface; they never round through binary64.
    """

    scheme: LoopSchemeDeclaration
    formal_symbol: str
    parameter_identities: Mapping[str, str]
    kernel_keys: tuple[KernelKey, ...]
    field_keys: tuple[FieldKey, ...]
    scheme_parameter_values: Mapping[str, str | tuple[str, str]] = field(
        default_factory=dict
    )
    whole_rules: tuple[WholeRuleColorRecipe, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scheme, LoopSchemeDeclaration):
            raise TypeError("source colour declaration requires an explicit scheme")
        if (
            not isinstance(self.formal_symbol, str)
            or not self.formal_symbol.isidentifier()
        ):
            raise ValueError("formal Nc placeholder must be a scalar identifier")
        if not isinstance(self.parameter_identities, Mapping):
            raise TypeError("source colour identities require a mapping")
        identities = {
            _name(name): _name(value)
            for name, value in self.parameter_identities.items()
        }
        object.__setattr__(self, "parameter_identities", MappingProxyType(identities))
        keys = tuple(_kernel_key(row) for row in self.kernel_keys)
        fields = tuple(_field_key(row) for row in self.field_keys)
        if len(set(keys)) != len(keys) or len(set(fields)) != len(fields):
            raise ValueError("duplicate source colour kernel/field key")
        object.__setattr__(self, "kernel_keys", keys)
        object.__setattr__(self, "field_keys", fields)
        if not isinstance(self.scheme_parameter_values, Mapping):
            raise TypeError("source colour scheme values require a mapping")
        values: dict[str, str | tuple[str, str]] = {}
        for name, value in self.scheme_parameter_values.items():
            _name(name)
            if isinstance(value, str):
                values[name] = value
            else:
                pair = _sequence(value)
                if len(pair) != 2 or not all(isinstance(part, str) for part in pair):
                    raise TypeError(
                        "scheme values require exact decimal strings or pairs"
                    )
                values[name] = (str(pair[0]), str(pair[1]))
        object.__setattr__(self, "scheme_parameter_values", MappingProxyType(values))
        rules = tuple(self.whole_rules)
        if any(not isinstance(rule, WholeRuleColorRecipe) for rule in rules):
            raise TypeError("source colour whole rules require typed recipes")
        if len({rule.key for rule in rules}) != len(rules):
            raise ValueError("duplicate whole-rule colour recipe")
        object.__setattr__(self, "whole_rules", rules)

    def to_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return {
            "scheme": asdict(self.scheme),
            "formal_symbol": self.formal_symbol,
            "parameter_identities": dict(self.parameter_identities),
            "kernel_keys": [list(key) for key in self.kernel_keys],
            "field_keys": [list(key) for key in self.field_keys],
            "scheme_parameter_values": dict(self.scheme_parameter_values),
            "whole_rules": [rule.to_dict() for rule in self.whole_rules],
        }

    @classmethod
    def from_dict(cls, value: object) -> LoopColorDeclaration:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("invalid source colour declaration fields")
        return cls(
            scheme=LoopSchemeDeclaration.from_dict(value["scheme"]),
            formal_symbol=value["formal_symbol"],
            parameter_identities=value["parameter_identities"],
            kernel_keys=value["kernel_keys"],
            field_keys=value["field_keys"],
            scheme_parameter_values=value["scheme_parameter_values"],
            whole_rules=tuple(
                WholeRuleColorRecipe.from_dict(rule)
                for rule in _sequence(value["whole_rules"])
            ),
        )


def source_loop_color_declaration(raw: RawLoopModel) -> LoopColorDeclaration | None:
    """The declaration the source carries itself (its metadata), or None."""
    from .loop_import import _declaration_data

    payload = raw.metadata.get(LOOP_COLOR_DECLARATION_KEY)
    return (
        None
        if payload is None
        else LoopColorDeclaration.from_dict(_declaration_data(payload))
    )


def resolve_loop_color_declaration(raw: RawLoopModel) -> LoopColorDeclaration | None:
    """The source's own declaration (``source_loop_color_declaration``),
    otherwise the shipped one of its counterterm vertices
    (``models.loop_color_declarations``, keyed by structural vertex digests:
    e.g. MG5_aMC's loop_sm under every restriction), otherwise None; absence
    leaves full colour unchanged."""
    own = source_loop_color_declaration(raw)
    if own is not None:
        return own
    from .loop_color_declarations import resolve_shipped_declaration

    return resolve_shipped_declaration(raw)


@overload
def with_loop_color_declaration(
    model: CompiledModel, declaration: LoopColorDeclaration
) -> CompiledModel: ...


@overload
def with_loop_color_declaration(
    model: RawLoopModel, declaration: LoopColorDeclaration
) -> RawLoopModel: ...


def with_loop_color_declaration(
    model: RawLoopModel | CompiledModel, declaration: LoopColorDeclaration
) -> RawLoopModel | CompiledModel:
    """Attach explicit colour data through the ordinary model bundle writer."""
    from .loop_import import RawLoopModel

    if not isinstance(declaration, LoopColorDeclaration):
        raise TypeError("expected a LoopColorDeclaration")
    if isinstance(model, RawLoopModel):
        return replace(
            model,
            metadata={
                **model.metadata,
                LOOP_COLOR_DECLARATION_KEY: declaration.to_dict(),
            },
        )
    from .loading import CompiledModel

    if not isinstance(model, CompiledModel) or model.raw_loop_model is None:
        raise ValueError(
            "source colour declarations require an existing raw NLO sidecar"
        )
    return replace(
        model,
        raw_loop_model=with_loop_color_declaration(model.raw_loop_model, declaration),
        _serialized_path=None,
        _prepared_bundle=None,
    )
