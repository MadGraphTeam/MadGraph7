# SPDX-License-Identifier: 0BSD
"""Explicit source preparation inputs; no physics-capability certification."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from pyamplicol.config.models import LoopPreparationConfig

    from .loading import CompiledModel
    from .loop_import import RawLoopModel

LOOP_PREPARATION_KEY = "pyamplicol_loop_preparation"
ExactParameter = str | tuple[str, str]
_ModelT = TypeVar("_ModelT", bound="RawLoopModel | CompiledModel")


def _pdg(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value):
        return int(value)
    raise TypeError("covariant multiplet keys must be signed integer PDGs")


def _fixed(values: Mapping[str, ExactParameter]) -> dict[str, tuple[str, str]]:
    if not isinstance(values, Mapping) or any(
        not isinstance(name, str) or not name.strip() for name in values
    ):
        raise TypeError("fixed parameters require nonempty source parameter names")
    if not values:
        return {}
    from pyamplicol.loops.scalar_bindings import _normalize_parameter_values

    # Source names and real/complex types are checked by the actual model later.
    # Reuse its exact lexical/numerical validation without inventing a scalar API.
    normalize = cast(
        Callable[
            [Mapping[str, ExactParameter], tuple[str, ...], tuple[str, ...]],
            dict[str, tuple[str, str]],
        ],
        _normalize_parameter_values,
    )
    return normalize(values, tuple(values), ("complex",) * len(values))


@dataclass(frozen=True, slots=True)
class LoopPreparation:
    compatibility: str | None = None
    covariant_multiplets: Mapping[int | str, tuple[int, ...]] = field(
        default_factory=dict
    )
    fixed_parameters: Mapping[str, ExactParameter] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.compatibility is not None and (
            not isinstance(self.compatibility, str) or not self.compatibility.strip()
        ):
            raise TypeError("loop preparation compatibility must be a nonempty string")
        if not isinstance(self.covariant_multiplets, Mapping):
            raise TypeError("covariant multiplets must be a mapping")
        multiplets = {}
        for key, partners in self.covariant_multiplets.items():
            pdg = _pdg(key)
            if pdg in multiplets:
                raise ValueError("duplicate covariant multiplet PDG identity")
            if (
                not isinstance(partners, Sequence)
                or isinstance(partners, (str, bytes))
                or any(type(partner) is not int for partner in partners)
            ):
                raise TypeError("covariant multiplet members must be integer PDGs")
            multiplets[pdg] = tuple(partners)
        object.__setattr__(self, "covariant_multiplets", MappingProxyType(multiplets))
        object.__setattr__(
            self, "fixed_parameters", MappingProxyType(_fixed(self.fixed_parameters))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "compatibility": self.compatibility,
            "covariant_multiplets": {
                str(pdg): list(partners)
                for pdg, partners in self.covariant_multiplets.items()
            },
            "fixed_parameters": {
                name: list(value) for name, value in self.fixed_parameters.items()
            },
        }

    @classmethod
    def from_dict(cls, value: object) -> LoopPreparation:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError(
                "loop preparation requires compatibility, covariant_multiplets "
                "and fixed_parameters"
            )
        return cls(**value)

    def merged_fixed_parameters(
        self, additional: Mapping[str, ExactParameter] | None = None
    ) -> dict[str, tuple[str, str]]:
        """Add exact colour-scheme ties without overriding source preparation."""
        extra = _fixed({} if additional is None else additional)
        fixed = cast(Mapping[str, tuple[str, str]], self.fixed_parameters)
        for name in fixed.keys() & extra.keys():
            if tuple(map(Decimal, fixed[name])) != tuple(map(Decimal, extra[name])):
                raise ValueError(
                    f"fixed parameter {name!r} contradicts loop preparation"
                )
        return {**fixed, **extra}


def resolve_loop_preparation(
    raw: RawLoopModel, configured: LoopPreparationConfig | None = None
) -> LoopPreparation:
    supplied = None if configured is None else configured.to_preparation()
    if supplied is not None:
        return supplied
    value = raw.metadata.get(LOOP_PREPARATION_KEY)
    return LoopPreparation() if value is None else LoopPreparation.from_dict(value)


def with_loop_preparation(model: _ModelT, preparation: LoopPreparation) -> _ModelT:
    from .loading import CompiledModel
    from .loop_import import RawLoopModel

    if not isinstance(preparation, LoopPreparation):
        raise TypeError("expected a LoopPreparation")
    if isinstance(model, RawLoopModel):
        return cast(
            _ModelT,
            replace(
                model,
                metadata={
                    **model.metadata,
                    LOOP_PREPARATION_KEY: preparation.to_dict(),
                },
            ),
        )
    if not isinstance(model, CompiledModel) or model.raw_loop_model is None:
        raise ValueError("loop preparation requires an existing raw NLO sidecar")
    return cast(
        _ModelT,
        replace(
            model,
            raw_loop_model=with_loop_preparation(model.raw_loop_model, preparation),
            _serialized_path=None,
            _prepared_bundle=None,
        ),
    )


def require_loop_model_source(model: CompiledModel) -> None:
    """Keep shared exclusions; the source frontend owns its scalar language."""
    deferred = {"unknown-functions", "function-arity"}
    blocking = tuple(
        issue
        for issue in model.issues
        if issue.severity == "error" and issue.code not in deferred
    )
    if blocking:
        raise ValueError(
            f"model {model.name!r} is not supported for loop preparation: "
            + "; ".join(f"{issue.code}: {issue.message}" for issue in blocking)
        )


__all__ = ["LoopPreparation", "resolve_loop_preparation", "with_loop_preparation"]
