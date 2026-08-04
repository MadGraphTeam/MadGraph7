# SPDX-License-Identifier: 0BSD
"""Validated process selection exported by the native runtime.

Rusticol is the sole authority for resolving explicit aliases and inferred
process permutations.  Python exact execution consumes that resolved state;
it must not try to reproduce the selector rules from the artifact manifest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from pyamplicol.api.errors import ArtifactError, CompatibilityError


@dataclass(frozen=True, slots=True)
class NativeProcessSelection:
    """One representative process and its representative-to-public mapping."""

    process: Mapping[str, object]
    representative_process_id: str
    representative_process_key: str
    external_permutation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NativePhysicsAxes:
    """Aligned representative/public selector identifiers from Rusticol."""

    public_physics: dict[str, object]
    helicity_ids: Mapping[str, str]
    color_ids: Mapping[str, str]


def exact_runtime_state_payload(native_runtime: Any) -> Mapping[str, object]:
    """Return Rusticol's private exact-runtime state as a JSON object."""

    try:
        payload = json.loads(native_runtime._exact_runtime_state_json())
    except AttributeError as exc:
        raise CompatibilityError(
            "the installed Rusticol extension is too old for high-precision "
            "evaluation"
        ) from exc
    except (TypeError, json.JSONDecodeError) as exc:
        raise ArtifactError("Rusticol returned invalid exact-runtime state") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError("Rusticol exact-runtime state is not an object")
    return cast(Mapping[str, object], payload)


def native_process_selection(
    native_runtime: Any,
    processes: Sequence[Mapping[str, object]],
) -> NativeProcessSelection:
    """Authenticate Rusticol's resolved representative and public permutation."""

    payload = exact_runtime_state_payload(native_runtime)
    representative_id = payload.get("representative_process_id")
    representative_key = payload.get("representative_process_key")
    if not isinstance(representative_id, str) or not representative_id:
        raise ArtifactError(
            "Rusticol exact-runtime representative process ID is invalid"
        )
    if not isinstance(representative_key, str) or not representative_key:
        raise ArtifactError(
            "Rusticol exact-runtime representative process key is invalid"
        )

    matches = tuple(
        process for process in processes if process.get("id") == representative_id
    )
    if len(matches) != 1:
        raise ArtifactError(
            "Rusticol exact-runtime representative process "
            f"{representative_id!r} is absent or repeated in its artifact"
        )
    process = matches[0]
    raw_pdgs = process.get("external_pdgs")
    if isinstance(raw_pdgs, str | bytes) or not isinstance(raw_pdgs, Sequence):
        raise ArtifactError(
            f"representative process {representative_id!r} has invalid external PDGs"
        )
    external_count = len(raw_pdgs)
    raw_permutation = payload.get("external_permutation")
    if isinstance(raw_permutation, str | bytes) or not isinstance(
        raw_permutation, Sequence
    ):
        raise ArtifactError("Rusticol exact-runtime external permutation is invalid")
    permutation = tuple(raw_permutation)
    if (
        len(permutation) != external_count
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in permutation
        )
        or set(permutation) != set(range(external_count))
    ):
        raise ArtifactError("Rusticol exact-runtime external permutation is invalid")

    return NativeProcessSelection(
        process=process,
        representative_process_id=representative_id,
        representative_process_key=representative_key,
        external_permutation=cast(tuple[int, ...], permutation),
    )


def native_physics_axes(
    native_runtime: Any,
    representative_physics: Mapping[str, object],
) -> NativePhysicsAxes:
    """Load Rusticol's public physics and authenticate aligned selector axes.

    Rusticol preserves axis order while rewriting process-order-dependent IDs,
    vectors, and LC words.  Exact executors use these maps only to translate
    representative execution overrides onto that already-validated public
    payload.
    """

    try:
        public = json.loads(native_runtime.physics_json())
    except AttributeError as exc:
        raise CompatibilityError(
            "the installed Rusticol extension does not expose public physics"
        ) from exc
    except (TypeError, json.JSONDecodeError) as exc:
        raise ArtifactError("Rusticol returned invalid public physics") from exc
    if not isinstance(public, dict):
        raise ArtifactError("Rusticol public physics is not an object")
    if public.get("color_accuracy") != representative_physics.get("color_accuracy"):
        raise ArtifactError(
            "Rusticol public physics changes the representative color accuracy"
        )
    return NativePhysicsAxes(
        public_physics=cast(dict[str, object], public),
        helicity_ids=_aligned_axis_ids(
            representative_physics,
            public,
            "helicities",
            "helicity",
        ),
        color_ids=_aligned_axis_ids(
            representative_physics,
            public,
            "color_components",
            "color component",
        ),
    )


def remap_reduction(
    reduction: Mapping[str, object],
    axes: NativePhysicsAxes,
) -> dict[str, object]:
    """Translate representative reduction selectors to public IDs."""

    groups = reduction.get("groups")
    if isinstance(groups, str | bytes) or not isinstance(groups, Sequence):
        raise ArtifactError("physics reduction groups are invalid")
    remapped = dict(reduction)
    remapped["groups"] = [
        remap_reduction_group(group, axes, index=index)
        for index, group in enumerate(groups)
    ]
    return remapped


def remap_reduction_group(
    value: object,
    axes: NativePhysicsAxes,
    *,
    index: int,
) -> dict[str, object]:
    """Translate one representative reduction group to public IDs."""

    if not isinstance(value, Mapping):
        raise ArtifactError(f"physics reduction group {index} is not an object")
    group = dict(value)
    group["representative_helicity_id"] = _remapped_id(
        value.get("representative_helicity_id"),
        axes.helicity_ids,
        f"reduction group {index} representative helicity",
    )
    group["representative_color_id"] = _remapped_id(
        value.get("representative_color_id"),
        axes.color_ids,
        f"reduction group {index} representative color",
    )
    group["physical_helicity_ids"] = _remapped_ids(
        value.get("physical_helicity_ids"),
        axes.helicity_ids,
        f"reduction group {index} physical helicities",
    )
    group["physical_color_ids"] = _remapped_ids(
        value.get("physical_color_ids"),
        axes.color_ids,
        f"reduction group {index} physical colors",
    )
    return group


def representative_vector_to_public(
    values: Sequence[int], permutation: tuple[int, ...]
) -> tuple[int, ...]:
    """Apply a representative-index to public-index external permutation."""

    vector = tuple(values)
    if len(vector) != len(permutation) or set(permutation) != set(range(len(vector))):
        raise ArtifactError("external vector permutation is invalid")
    public = [0] * len(vector)
    for representative_index, public_index in enumerate(permutation):
        public[public_index] = vector[representative_index]
    return tuple(public)


def _aligned_axis_ids(
    representative: Mapping[str, object],
    public: Mapping[str, object],
    key: str,
    description: str,
) -> Mapping[str, str]:
    representative_axis = representative.get(key)
    public_axis = public.get(key)
    if (
        isinstance(representative_axis, str | bytes)
        or not isinstance(representative_axis, Sequence)
        or isinstance(public_axis, str | bytes)
        or not isinstance(public_axis, Sequence)
        or len(representative_axis) != len(public_axis)
    ):
        raise ArtifactError(f"Rusticol public {description} axis is inconsistent")
    result: dict[str, str] = {}
    public_ids: set[str] = set()
    for index, (representative_record, public_record) in enumerate(
        zip(representative_axis, public_axis, strict=True)
    ):
        if not isinstance(representative_record, Mapping) or not isinstance(
            public_record, Mapping
        ):
            raise ArtifactError(f"{description} axis entry {index} is not an object")
        representative_id = representative_record.get("id")
        public_id = public_record.get("id")
        if (
            not isinstance(representative_id, str)
            or not representative_id
            or not isinstance(public_id, str)
            or not public_id
            or representative_id in result
            or public_id in public_ids
        ):
            raise ArtifactError(f"{description} axis IDs are invalid or repeated")
        result[representative_id] = public_id
        public_ids.add(public_id)
    return result


def _remapped_id(value: object, identifiers: Mapping[str, str], context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{context} is invalid")
    try:
        return identifiers[value]
    except KeyError as exc:
        raise ArtifactError(f"{context} {value!r} is absent from its axis") from exc


def _remapped_ids(
    value: object,
    identifiers: Mapping[str, str],
    context: str,
) -> list[str]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} are invalid")
    return [
        _remapped_id(identifier, identifiers, f"{context} entry {index}")
        for index, identifier in enumerate(value)
    ]


__all__ = [
    "NativePhysicsAxes",
    "NativeProcessSelection",
    "exact_runtime_state_payload",
    "native_physics_axes",
    "native_process_selection",
    "remap_reduction",
    "remap_reduction_group",
    "representative_vector_to_public",
]
