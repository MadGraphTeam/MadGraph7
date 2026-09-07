# SPDX-License-Identifier: 0BSD
"""ID-selected direct Born correlations, isolated from ordinary runtime calls."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.api.protocols import Momenta
from pyamplicol.artifacts import load_manifest
from pyamplicol.artifacts.security import confined_path
from pyamplicol.color.connections import COLOR_CONNECTION_CONVENTION
from pyamplicol.correlators import CorrelatedValue, CorrelatorConfig
from pyamplicol.runtime._native_selection import native_process_selection
from pyamplicol.runtime.correlated_exact import (
    CoherentAmplitudeBatch,
    CorrelatedExactExecutor,
    _prepare_spin_vectors,
)
from pyamplicol.runtime.symbolica_exact import _complex_mul, _working_precision


@dataclass(frozen=True)
class _MatrixEntry:
    left: int
    right: int
    real: Fraction
    imag: Fraction


@dataclass(frozen=True)
class _Matrix:
    id: str
    sectors: tuple[int, ...]
    entries: tuple[_MatrixEntry, ...]


def _exact_integer(value: object) -> int:
    """Read the writer's integer strings without accepting lossy coercions."""
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value):
        return int(value)
    raise ValueError("invalid exact matrix integer")


def _matrix(record: object) -> _Matrix:
    try:
        if not isinstance(record, Mapping):
            raise ValueError("matrix is not an object")
        if (
            record["convention"] != COLOR_CONNECTION_CONVENTION
            or record["color_accuracy"] != "full"
            or record["storage"] != "sparse-directed"
            or record["includes_color_factor"] is not True
        ):
            raise ValueError("unsupported matrix convention")
        identifier = record["id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("missing correlation ID")
        sectors = tuple(record["sector_ids"])
        if (
            not sectors
            or any(type(item) is not int or item < 0 for item in sectors)
            or len(set(sectors)) != len(sectors)
        ):
            raise ValueError("invalid colour sector IDs")
        entries = []
        pairs: set[tuple[int, int]] = set()
        for item in record["entries"]:
            left, right = item["left_sector_id"], item["right_sector_id"]
            if (
                type(left) is not int
                or type(right) is not int
                or left not in sectors
                or right not in sectors
                or (left, right) in pairs
            ):
                raise ValueError("invalid or repeated matrix entry")
            pairs.add((left, right))
            real, imag = item["weight"]["real"], item["weight"]["imag"]
            if any(
                isinstance(parts, (str, bytes))
                or not isinstance(parts, Sequence)
                or len(parts) != 2
                for parts in (real, imag)
            ):
                raise ValueError("invalid exact matrix coefficient")
            entries.append(
                _MatrixEntry(
                    left,
                    right,
                    Fraction(_exact_integer(real[0]), _exact_integer(real[1])),
                    Fraction(_exact_integer(imag[0]), _exact_integer(imag[1])),
                )
            )
        return _Matrix(identifier, sectors, tuple(entries))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ArtifactError(f"invalid correlated colour matrix: {exc}") from exc


def _contract(
    batch: CoherentAmplitudeBatch, matrix: _Matrix, *, precision: int
) -> tuple[CorrelatedValue, ...]:
    """Sum spectator helicities incoherently, but retain all colour phases."""
    groups: dict[str, dict[int, int]] = {}
    for index, group in enumerate(batch.groups):
        sectors = groups.setdefault(group.helicity_id, {})
        if group.color_sector_id in sectors:
            raise ArtifactError("repeated coherent helicity/colour group")
        sectors[group.color_sector_id] = index
    expected = set(matrix.sectors)
    if any(set(indices) != expected for indices in groups.values()):
        raise ArtifactError("correlated amplitudes do not cover the colour matrix")
    results = []
    with localcontext() as context:
        context.prec = _working_precision(precision)
        context.rounding = ROUND_HALF_EVEN
        entries = tuple(
            (
                entry.left,
                entry.right,
                (
                    Decimal(entry.real.numerator) / Decimal(entry.real.denominator),
                    Decimal(entry.imag.numerator) / Decimal(entry.imag.denominator),
                ),
            )
            for entry in matrix.entries
        )
        for point in batch.values:
            if len(point) != len(batch.groups):
                raise ArtifactError(
                    "correlated amplitude point has an invalid group count"
                )
            real, imag = Decimal(0), Decimal(0)
            for indices in groups.values():
                for left, right, weight in entries:
                    a, b = point[indices[left]], point[indices[right]]
                    term = _complex_mul(_complex_mul((a[0], -a[1]), weight), b)
                    real += term[0]
                    imag += term[1]
            results.append(
                (real * batch.normalization_factor, imag * batch.normalization_factor)
            )
    with localcontext() as context:
        context.prec = precision
        context.rounding = ROUND_HALF_EVEN
        return tuple(CorrelatedValue(+real, +imag) for real, imag in results)


class CorrelatorEvaluator:
    """Small stateful facade; source vectors belong only to this runtime."""

    def __init__(self, backend: Any) -> None:
        artifact = getattr(backend, "_artifact_path", None)
        native = getattr(backend, "_runtime", None)
        if not isinstance(artifact, Path) or native is None:
            raise CompatibilityError("this runtime does not support Born correlations")
        manifest = load_manifest(artifact)
        extension = manifest.extensions.get("correlators")
        if not isinstance(extension, Mapping):
            raise CompatibilityError(
                "correlations must be declared at generation with CorrelatorConfig"
            )
        try:
            if (
                type(extension["schema_version"]) is not int
                or extension["schema_version"] != 1
            ):
                raise ValueError("unsupported correlator schema version")
            path = extension["path"]
            if not isinstance(path, str):
                raise ValueError("missing correlator catalogue path")
            payload = json.loads(confined_path(artifact, path).read_text())
            if (
                type(payload["schema_version"]) is not int
                or payload["schema_version"] != 1
                or payload["complete_source_basis"] is not True
            ):
                raise ValueError("correlations require a complete source basis")
            declarations = CorrelatorConfig.from_json_dict(payload["declarations"])
            selection = native_process_selection(native, manifest.processes)
            process_id = selection.representative_process_id
            process = payload["processes"][process_id]
            if (
                any(type(leg) is not int for leg in process["spin_legs"])
                or tuple(process["spin_legs"]) != declarations.spin_legs
            ):
                raise ValueError("inconsistent declared spin legs")
            matrices = tuple(_matrix(item) for item in process["matrices"])
            coherent_groups = process["coherent_groups"]
            self._matrices = {matrix.id: matrix for matrix in matrices}
            if len(self._matrices) != len(matrices) or set(self._matrices) != {
                request.id for request in declarations.color_requests
            }:
                raise ValueError("inconsistent colour correlation IDs")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ArtifactError(f"invalid correlator catalogue: {exc}") from exc
        self._declarations = declarations
        self._vectors: dict[int, object] = {}
        self._executor = CorrelatedExactExecutor(
            artifact,
            process_id,
            native,
            spin_correlated_legs=declarations.spin_legs,
            coherent_groups=coherent_groups,
        )

    def set_spin_correlation_vectors(
        self, vectors: Mapping[int, object] | None
    ) -> None:
        if vectors is None:
            vectors = {}
        if not isinstance(vectors, Mapping):
            raise EvaluationError("spin vectors must map public leg labels to vectors")
        legs = tuple(vectors)
        if any(type(leg) is not int for leg in legs):
            raise EvaluationError("spin-vector legs must be positive integer labels")
        if legs and tuple(sorted(legs)) not in self._declarations.spin_correlations:
            raise EvaluationError("this joint spin-correlation class was not declared")
        frozen: dict[int, object] = {}
        for leg, value in vectors.items():
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise EvaluationError("spin vectors must be four-vectors or a batch")
            vector = tuple(
                tuple(item)
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes))
                else item
                for item in value
            )
            if not vector:
                raise EvaluationError(
                    "a spin-vector batch must contain at least one point"
                )
            broadcast = len(vector) == 4 and all(
                not isinstance(item, tuple) for item in vector
            )
            # Validate before replacing the old state; a failed setter is atomic.
            _prepare_spin_vectors(
                {leg: vector},
                allowed_legs=self._declarations.spin_legs,
                point_count=1 if broadcast else len(vector),
            )
            frozen[leg] = vector
        self._vectors = frozen

    def clear(self) -> None:
        """Release warmed evaluators, preserving current numerical spin inputs."""
        self._executor.clear()

    def evaluate(
        self,
        momenta: Momenta,
        *,
        color_correlation: str = "born",
        helicities: Sequence[str] | None = None,
        precision: int = 16,
    ) -> tuple[CorrelatedValue, ...]:
        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision < 1
        ):
            raise EvaluationError(
                "precision must be a positive number of decimal digits"
            )
        if (
            not isinstance(color_correlation, str)
            or color_correlation not in self._matrices
        ):
            raise EvaluationError(
                f"unknown colour correlation ID {color_correlation!r}"
            )
        batch = self._executor.coherent_amplitudes(
            momenta,
            spin_vectors=self._vectors,
            helicities=helicities,
            precision=_working_precision(precision),
        )
        return _contract(batch, self._matrices[color_correlation], precision=precision)
