# SPDX-License-Identifier: 0BSD
"""Loop declarations over the loader's lossless UFO source records.

The upstream loader owns source capture, exact literal observation, import
isolation and serialization. This module keeps the established consumer names
and validates pyAmpliCol's provider declaration without duplicating that work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from ufo_model_loader.raw import (
    RAW_UFO_MODEL_KIND as RAW_LOOP_MODEL_KIND,
)
from ufo_model_loader.raw import (
    RAW_UFO_MODEL_SCHEMA_VERSION as RAW_LOOP_MODEL_SCHEMA_VERSION,
)
from ufo_model_loader.raw import (
    RawUFOModel as RawLoopModel,
)
from ufo_model_loader.raw import (
    capture_raw_ufo_model,
    load_raw_ufo_model,
)
from ufo_model_loader.raw import (
    imported_ufo_module as imported_ufo_module,
)
from ufo_model_loader.raw import (
    raw_restriction as raw_restriction,
)

__all__ = [
    "RAW_LOOP_MODEL_KIND",
    "RAW_LOOP_MODEL_SCHEMA_VERSION",
    "RawLoopModel",
    "adapt_raw_ufo_model",
    "capture_raw_loop_model",
    "imported_ufo_module",
    "load_raw_loop_model",
    "raw_loop_model_from_dict",
    "raw_restriction",
]


def _declaration_data(value: object) -> object:
    """Unwrap only the provider's plain-data declaration from raw UFO mappings."""
    if isinstance(value, Mapping):
        if value.get("kind") == "tuple":
            if set(value) != {"kind", "items"}:
                raise ValueError("invalid loop declaration sequence")
            return [_declaration_data(item) for item in value["items"]]
        if value.get("kind") == "mapping":
            if set(value) != {"kind", "entries"}:
                raise ValueError("invalid loop declaration mapping")
            result = {}
            for entry in value["entries"]:
                if (
                    not isinstance(entry, (list, tuple))
                    or len(entry) != 2
                    or not isinstance(entry[0], str)
                    or entry[0] in result
                ):
                    raise ValueError("invalid loop declaration mapping entry")
                result[entry[0]] = _declaration_data(entry[1])
            return result
        return {key: _declaration_data(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_declaration_data(item) for item in value]
    return value


def adapt_raw_ufo_model(raw: RawLoopModel) -> RawLoopModel:
    """Validate the provider claim locally; source records remain unchanged."""
    from .loop_declaration import LOOP_DECLARATION_KEY, LoopModelDeclaration

    if not isinstance(raw, RawLoopModel):
        raise TypeError("expected lossless UFO source records from ufo_model_loader")
    if LOOP_DECLARATION_KEY not in raw.metadata:
        return raw
    declaration = LoopModelDeclaration.from_dict(
        _declaration_data(raw.metadata[LOOP_DECLARATION_KEY])
    )
    return replace(
        raw,
        metadata={**raw.metadata, LOOP_DECLARATION_KEY: declaration.to_dict()},
    )


def raw_loop_model_from_dict(payload: Mapping[str, Any]) -> RawLoopModel:
    """Read the existing compiled-model field using the authoritative loader.

    The earlier local payload has identical records and version 1 semantics;
    translating its kind is sufficient for retained development artifacts.
    """
    if payload.get("kind") == "pyamplicol-raw-loop-model":
        payload = {**payload, "kind": RAW_LOOP_MODEL_KIND}
    return adapt_raw_ufo_model(RawLoopModel.from_dict(payload))


def capture_raw_loop_model(
    module: ModuleType,
    *,
    name: str | None = None,
    restriction: Mapping[str, object] | None = None,
    value_overrides: Mapping[tuple[str, int], object] | None = None,
) -> RawLoopModel:
    return adapt_raw_ufo_model(
        capture_raw_ufo_model(
            module,
            name=name,
            restriction=restriction,
            value_overrides=value_overrides,
        )
    )


def load_raw_loop_model(
    source: str | Path,
    *,
    restriction: str = "default",
    value_overrides: Mapping[tuple[str, int], object] | None = None,
) -> RawLoopModel:
    from .loading import _loader_restriction_name, _sanitized_model_environment

    path = Path(source).expanduser().resolve()
    selected = _loader_restriction_name(path, restriction)
    with _sanitized_model_environment():
        try:
            raw = load_raw_ufo_model(
                path, restriction=selected, value_overrides=value_overrides
            )
        except Exception as error:
            from .source_refusals import refuse_python2_ufo

            refuse_python2_ufo(path, error)
            raise
        return adapt_raw_ufo_model(raw)
