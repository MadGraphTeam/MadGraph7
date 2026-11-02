# SPDX-License-Identifier: 0BSD
"""Explicit NLO model declarations, distinct from executable capability checks.

The declaration lives in the existing raw-model metadata. It neither certifies
physics nor changes tensor, graph, precision or reduction capability checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from pyamplicol.config.models import LoopModelConfig

    from .loading import CompiledModel
    from .loop_import import RawLoopModel

LOOP_DECLARATION_KEY = "pyamplicol_loop"


@dataclass(frozen=True, slots=True)
class LoopSchemeDeclaration:
    dimensional_regularization: str
    external_states: str
    gamma5: str
    renormalization: str
    mass_scheme: str
    finite_restoration: str
    model_convention: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"loop scheme {name} must be explicitly supplied")

    @classmethod
    def from_dict(cls, value: object) -> LoopSchemeDeclaration:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("loop model declaration requires all seven scheme fields")
        return cls(**value)


def _corrections(value: object) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
        or any(not isinstance(name, str) or not name.strip() for name in value)
    ):
        raise ValueError(
            "declared loop_coupling_orders must be a nonempty string sequence"
        )
    return tuple(sorted(set(value)))


@dataclass(frozen=True, slots=True)
class LoopModelDeclaration:
    """A provider/user claim covering the joint correction set and its subsets.

    A joint QCD/QED declaration includes mixed loops. Separate declarations are
    never combined to manufacture joint coverage. Completeness is a declaration,
    not a certificate inferred from the presence of counterterm vertices.
    """

    scheme: LoopSchemeDeclaration
    loop_coupling_orders: tuple[str, ...]
    uv_complete: bool
    model_r2_complete: bool
    status: Literal["declared-not-certified"] = field(
        default="declared-not-certified", init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.scheme, LoopSchemeDeclaration):
            raise TypeError("loop model declaration requires a typed scheme")
        object.__setattr__(
            self, "loop_coupling_orders", _corrections(self.loop_coupling_orders)
        )
        for name in ("uv_complete", "model_r2_complete"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"loop model declaration {name} must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> LoopModelDeclaration:
        required = {
            "scheme",
            "loop_coupling_orders",
            "uv_complete",
            "model_r2_complete",
        }
        if not isinstance(value, Mapping) or set(value) not in (
            required,
            required | {"status"},
        ):
            raise ValueError("invalid loop model declaration fields")
        if value.get("status", "declared-not-certified") != "declared-not-certified":
            raise ValueError(
                "loop model declarations cannot claim physics certification"
            )
        return cls(
            scheme=LoopSchemeDeclaration.from_dict(value["scheme"]),
            loop_coupling_orders=_corrections(value["loop_coupling_orders"]),
            uv_complete=value["uv_complete"],
            model_r2_complete=value["model_r2_complete"],
        )

    def require_declared_completion(self, *, strategy: str = "model") -> None:
        if strategy not in {"model", "direct"}:
            raise ValueError(
                "completion requires a resolved model or direct R2 strategy"
            )
        missing = []
        if not self.uv_complete:
            missing.append("complete UV counterterms")
        if strategy == "model" and not self.model_r2_complete:
            missing.append("complete model R2 rules")
        if missing:
            from pyamplicol.loops.v2.refusals import LoopRequestUnsupportedValue

            raise LoopRequestUnsupportedValue(
                "loop-model-completion",
                "loop model does not declare " + " and ".join(missing),
            )


def resolve_loop_model_declaration(
    raw: RawLoopModel,
    configured: LoopModelConfig | None = None,
    *,
    selected_corrections: Sequence[str],
) -> LoopModelDeclaration:
    """Use an explicit whole config declaration, otherwise the source
    declaration, otherwise the shipped profile of the exact source
    (``loops.shipped_profiles.shipped_loop_profile``).

    Partial config declarations fail; they are never completed from unrelated
    source fields. The chosen declaration covers the selected joint correction
    set only when every requested correction belongs to its declared set.
    """
    declaration = None if configured is None else configured.to_declaration()
    if declaration is None:
        payload = raw.metadata.get(LOOP_DECLARATION_KEY)
        if payload is not None:
            declaration = LoopModelDeclaration.from_dict(payload)
        else:
            from pyamplicol.loops.shipped_profiles import shipped_loop_profile

            # A source pyAmpliCol ships the declaration of (by the digest of
            # its declarations, ``loops.shipped_profiles``), e.g. MG5_aMC's loop_sm.
            profile = shipped_loop_profile(raw)
            if profile is None:
                from pyamplicol.loops.v2.refusals import LoopRequestUnsupportedValue

                raise LoopRequestUnsupportedValue(
                    "loop-model-declaration",
                    "loop model has no explicit scheme/completeness declaration; "
                    "supply [loop.model] and [loop.model.scheme] or model metadata "
                    + LOOP_DECLARATION_KEY,
                )
            declaration = profile.declaration
    missing = set(_corrections(selected_corrections)) - set(
        declaration.loop_coupling_orders
    )
    if missing:
        from pyamplicol.loops.v2.refusals import LoopRequestUnsupportedValue

        raise LoopRequestUnsupportedValue(
            "loop-model-corrections",
            "loop model declaration does not cover selected corrections: "
            + ", ".join(sorted(missing))
            + f" (it declares {', '.join(declaration.loop_coupling_orders)})",
        )
    return declaration


@overload
def with_loop_model_declaration(
    model: RawLoopModel, declaration: LoopModelDeclaration
) -> RawLoopModel: ...


@overload
def with_loop_model_declaration(
    model: CompiledModel, declaration: LoopModelDeclaration
) -> CompiledModel: ...


def with_loop_model_declaration(
    model: RawLoopModel | CompiledModel, declaration: LoopModelDeclaration
) -> RawLoopModel | CompiledModel:
    """Return a model carrying this declaration in its existing source sidecar.

    Raw capability claims remain unchanged. A compiled copy drops its cached
    serialization so ordinary model/artifact writers persist the new metadata.
    """
    from .loop_import import RawLoopModel

    if not isinstance(declaration, LoopModelDeclaration):
        raise TypeError("expected a LoopModelDeclaration")
    if isinstance(model, RawLoopModel):
        return replace(
            model,
            metadata={**model.metadata, LOOP_DECLARATION_KEY: declaration.to_dict()},
        )
    from .loading import CompiledModel

    if not isinstance(model, CompiledModel) or model.raw_loop_model is None:
        raise ValueError(
            "a loop model declaration requires an existing raw NLO sidecar"
        )
    return replace(
        model,
        raw_loop_model=with_loop_model_declaration(model.raw_loop_model, declaration),
        _serialized_path=None,
        _prepared_bundle=None,
    )
