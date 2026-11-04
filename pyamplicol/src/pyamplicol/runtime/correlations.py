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
from typing import Any, cast

from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.api.protocols import Momenta
from pyamplicol.artifacts import load_manifest
from pyamplicol.artifacts.security import confined_path
from pyamplicol.color.connections import COLOR_CONNECTION_CONVENTION
from pyamplicol.correlators import (
    CorrelatedRequest,
    CorrelatedValue,
    CorrelatorConfig,
    GradedCorrelatedEvaluation,
)
from pyamplicol.runtime._double_double import correlated_precision
from pyamplicol.runtime._native_selection import native_process_selection
from pyamplicol.runtime.correlated_exact import (
    CoherentAmplitudeBatch,
    CoherentAmplitudeGroup,
    CorrelatedDoubleDoubleExecutor,
    CorrelatedExactExecutor,
    _copy_spin_vectors,
    _prepare_spin_vectors,
)
from pyamplicol.runtime.symbolica_exact import _complex_mul, _working_precision


@dataclass(frozen=True)
class _MatrixEntry:
    left: int
    right: int
    real: Fraction
    imag: Fraction
    # N_c power p of this term of a graded matrix (strict or graded colour).
    nc_power: int | None = None


@dataclass(frozen=True)
class _Matrix:
    id: str
    sectors: tuple[int, ...]
    entries: tuple[_MatrixEntry, ...]
    # Reported N_c degrees (decreasing) of a graded matrix; a term of power p
    # between groups of grades g_l, g_r has the degree p + g_l + g_r.
    degrees: tuple[int, ...] | None = None


@dataclass(frozen=True)
class CorrelatorCatalogue:
    """Validated selected-process metadata, without an amplitude executor."""

    artifact: Path
    process_id: str
    native_runtime: Any
    declarations: CorrelatorConfig
    matrices: tuple[_Matrix, ...]
    coherent_groups: Sequence[Mapping[str, object]]
    # Exact factor of a process-group member evaluated by its representative.
    factor: Fraction = Fraction(1)


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
            or record["color_accuracy"]
            not in {"lc", "nlc", "full", "strict-lc", "strict-nlc"}
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
        expansion = record.get("nc_expansion")
        degrees: tuple[int, ...] | None = None
        if expansion is not None:
            degrees = tuple(expansion["degrees"])
            if (
                not degrees
                or any(type(item) is not int for item in degrees)
                or list(degrees) != sorted(set(degrees), reverse=True)
            ):
                raise ValueError("invalid N_c degrees")
        elif record["color_accuracy"] in {"strict-lc", "strict-nlc"}:
            raise ValueError("strict colour matrix has no N_c expansion")
        entries = []
        pairs: set[tuple[int, int, int | None]] = set()
        for item in record["entries"]:
            left, right = item["left_sector_id"], item["right_sector_id"]
            power = item.get("nc_power")
            if (
                type(left) is not int
                or type(right) is not int
                or left not in sectors
                or right not in sectors
                or (left, right, power) in pairs
                or (power is None) != (degrees is None)
                or (power is not None and type(power) is not int)
            ):
                raise ValueError("invalid or repeated matrix entry")
            pairs.add((left, right, power))
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
                    power,
                )
            )
        return _Matrix(identifier, sectors, tuple(entries), degrees)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ArtifactError(f"invalid correlated colour matrix: {exc}") from exc


@dataclass(frozen=True)
class _ContractionPlan:
    group_count: int
    terms: tuple[tuple[int, int, tuple[Decimal, Decimal]], ...]
    # The N_c-degree slot of every term of a graded matrix (``degrees``).
    term_slots: tuple[int, ...] | None = None
    degrees: tuple[int, ...] | None = None

    def evaluate(
        self, batch: CoherentAmplitudeBatch, *, precision: int
    ) -> tuple[CorrelatedValue, ...]:
        return tuple(
            point[-1] for point in self.evaluate_graded(batch, precision=precision)
        )

    def evaluate_graded(
        self, batch: CoherentAmplitudeBatch, *, precision: int
    ) -> tuple[tuple[CorrelatedValue, ...], ...]:
        """Per point: the value of each N_c degree of a graded matrix, then
        the total (only the total for an ungraded matrix)."""

        slot_count = 0 if self.degrees is None else len(self.degrees)
        results = []
        with localcontext() as context:
            context.prec = _working_precision(precision)
            context.rounding = ROUND_HALF_EVEN
            for point in batch.values:
                if len(point) != self.group_count:
                    raise ArtifactError(
                        "correlated amplitude point has an invalid group count"
                    )
                scalar = type(batch.normalization_factor)
                real, imag = scalar(0), scalar(0)
                slots = [[scalar(0), scalar(0)] for _ in range(slot_count)]
                for index, (left, right, weight) in enumerate(self.terms):
                    a, b = point[left], point[right]
                    # Keep the single-ID multiplication/accumulation order.
                    term = _complex_mul(_complex_mul((a[0], -a[1]), weight), b)
                    real += term[0]
                    imag += term[1]
                    if self.term_slots is not None:
                        slot = slots[self.term_slots[index]]
                        slot[0] += term[0]
                        slot[1] += term[1]
                results.append(
                    tuple(
                        (
                            part_re * batch.normalization_factor,
                            part_im * batch.normalization_factor,
                        )
                        for part_re, part_im in (*slots, (real, imag))
                    )
                )
        with localcontext() as context:
            context.prec = precision
            context.rounding = ROUND_HALF_EVEN
            return tuple(
                tuple(
                    CorrelatedValue(+Decimal(real), +Decimal(imag))
                    for real, imag in point
                )
                for point in results
            )


def _prepare_contraction(
    amplitude_groups: tuple[CoherentAmplitudeGroup, ...],
    matrix: _Matrix,
    *,
    precision: int,
    scalar: type[Decimal] = Decimal,
) -> _ContractionPlan:
    """Prepare exact weights and indices once, without numerical amplitude state.

    A graded matrix (strict or graded colour) contracts every N_c grade of
    each sector's amplitude: its term of power p between groups of grades
    g_l and g_r enters the degree p + g_l + g_r, if the matrix reports it.
    """
    if matrix.degrees is not None:
        return _prepare_graded_contraction(
            amplitude_groups, matrix, precision=precision, scalar=scalar
        )
    groups: dict[str, dict[int, int]] = {}
    for index, group in enumerate(amplitude_groups):
        sectors = groups.setdefault(group.helicity_id, {})
        if group.color_sector_id in sectors or group.nc_grade != 0:
            raise ArtifactError("repeated coherent helicity/colour group")
        sectors[group.color_sector_id] = index
    expected = set(matrix.sectors)
    if any(set(indices) != expected for indices in groups.values()):
        raise ArtifactError("correlated amplitudes do not cover the colour matrix")
    with localcontext() as context:
        context.prec = _working_precision(precision)
        context.rounding = ROUND_HALF_EVEN
        entries = tuple(
            (
                entry.left,
                entry.right,
                (
                    scalar(entry.real.numerator) / scalar(entry.real.denominator),
                    scalar(entry.imag.numerator) / scalar(entry.imag.denominator),
                ),
            )
            for entry in matrix.entries
        )
    return _ContractionPlan(
        len(amplitude_groups),
        tuple(
            (indices[left], indices[right], weight)
            for indices in groups.values()
            for left, right, weight in entries
        ),
    )


def _prepare_graded_contraction(
    amplitude_groups: tuple[CoherentAmplitudeGroup, ...],
    matrix: _Matrix,
    *,
    precision: int,
    scalar: type[Decimal],
) -> _ContractionPlan:
    degrees = matrix.degrees
    assert degrees is not None
    slot_by_degree = {degree: slot for slot, degree in enumerate(degrees)}
    groups: dict[str, dict[int, list[tuple[int, int]]]] = {}
    for index, group in enumerate(amplitude_groups):
        sectors = groups.setdefault(group.helicity_id, {})
        graded = sectors.setdefault(group.color_sector_id, [])
        if any(grade == group.nc_grade for grade, _ in graded):
            raise ArtifactError("repeated coherent helicity/colour group")
        graded.append((group.nc_grade, index))
    expected = set(matrix.sectors)
    if any(set(indices) != expected for indices in groups.values()):
        raise ArtifactError("correlated amplitudes do not cover the colour matrix")
    terms: list[tuple[int, int, tuple[Decimal, Decimal]]] = []
    slots: list[int] = []
    with localcontext() as context:
        context.prec = _working_precision(precision)
        context.rounding = ROUND_HALF_EVEN
        for sectors in groups.values():
            for entry in matrix.entries:
                weight = (
                    scalar(entry.real.numerator) / scalar(entry.real.denominator),
                    scalar(entry.imag.numerator) / scalar(entry.imag.denominator),
                )
                for left_grade, left in sectors[entry.left]:
                    for right_grade, right in sectors[entry.right]:
                        slot = slot_by_degree.get(
                            int(cast(int, entry.nc_power)) + left_grade + right_grade
                        )
                        if slot is None:
                            continue
                        terms.append((left, right, weight))
                        slots.append(slot)
    return _ContractionPlan(len(amplitude_groups), tuple(terms), tuple(slots), degrees)


def _contract(
    batch: CoherentAmplitudeBatch, matrix: _Matrix, *, precision: int
) -> tuple[CorrelatedValue, ...]:
    """Sum spectator helicities incoherently, but retain all colour phases."""
    return _prepare_contraction(
        batch.groups,
        matrix,
        precision=precision,
        scalar=type(batch.normalization_factor),
    ).evaluate(batch, precision=precision)


def load_correlator_catalogue(backend: Any) -> CorrelatorCatalogue:
    """Read and validate the correlation sidecar once for a runtime selection."""
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
        selection = native_process_selection(native, manifest.processes)
        identity = selection.external_permutation == tuple(
            range(len(selection.external_permutation))
        )
        relation = selection.process_group
        if relation is not None and (relation.conjugated or not identity):
            raise CompatibilityError(
                "correlated evaluation of process-group member "
                f"{relation.member_process_id!r} through its representative "
                f"{relation.representative_process_key!r} is available only for "
                "an identity leg order without charge conjugation; generate the "
                "process explicitly"
            )
        if not identity:
            raise CompatibilityError(
                "correlated catalogues and evaluation do not support "
                "non-identity process aliases"
            )
        process_id = selection.representative_process_id
        process = payload["processes"][process_id]
        declarations = CorrelatorConfig.from_json_dict(process["declarations"])
        requests = declarations.color_requests
        if (
            any(type(leg) is not int for leg in process["spin_legs"])
            or tuple(process["spin_legs"]) != declarations.spin_legs
        ):
            raise ValueError("inconsistent declared spin legs")
        matrices = tuple(_matrix(item) for item in process["matrices"])
        identifiers = {matrix.id for matrix in matrices}
        if len(identifiers) != len(matrices) or identifiers != {
            request.id for request in requests
        }:
            raise ValueError("inconsistent colour correlation IDs")
        coherent_groups = cast(
            Sequence[Mapping[str, object]], process["coherent_groups"]
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ArtifactError(f"invalid correlator catalogue: {exc}") from exc
    return CorrelatorCatalogue(
        artifact,
        process_id,
        native,
        declarations,
        matrices,
        coherent_groups,
        Fraction(1) if relation is None else relation.factor,
    )


class CorrelatorEvaluator:
    """Small stateful facade; source vectors belong only to this runtime."""

    def __init__(
        self, backend: Any, *, catalogue: CorrelatorCatalogue | None = None
    ) -> None:
        if catalogue is None:
            catalogue = load_correlator_catalogue(backend)
        self._matrices = {matrix.id: matrix for matrix in catalogue.matrices}
        self._catalogue = catalogue
        self._double_double_executor: CorrelatedDoubleDoubleExecutor | None = None
        self._declarations = catalogue.declarations
        self._vectors: dict[int, object] = {}
        self._executor = CorrelatedExactExecutor(
            catalogue.artifact,
            catalogue.process_id,
            catalogue.native_runtime,
            spin_correlated_legs=catalogue.declarations.spin_legs,
            coherent_groups=catalogue.coherent_groups,
        )

    def _select_executor(self, arithmetic: str) -> CorrelatedExactExecutor:
        if arithmetic != "double-double":
            return self._executor
        if self._double_double_executor is None:
            catalogue = self._catalogue
            self._double_double_executor = CorrelatedDoubleDoubleExecutor(
                catalogue.artifact,
                catalogue.process_id,
                catalogue.native_runtime,
                spin_correlated_legs=catalogue.declarations.spin_legs,
                coherent_groups=catalogue.coherent_groups,
            )
        return self._double_double_executor

    def set_spin_correlation_vectors(
        self, vectors: Mapping[int, object] | None
    ) -> None:
        self._vectors = self._copy_spin_vectors(vectors)

    def _copy_spin_vectors(
        self, vectors: Mapping[int, object] | None
    ) -> dict[int, object]:
        return _copy_spin_vectors(vectors, self._declarations.spin_correlations)

    def clear(self) -> None:
        """Release warmed evaluators, preserving current numerical spin inputs."""
        self._executor.clear()
        if self._double_double_executor is not None:
            self._double_double_executor.clear()

    def evaluate(
        self,
        momenta: Momenta,
        *,
        color_correlation: str = "born",
        helicities: Sequence[str] | None = None,
        precision: int = 16,
        arithmetic: str = "arbitrary",
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
        amplitude_precision = correlated_precision(precision, arithmetic)
        batch = self._select_executor(arithmetic).coherent_amplitudes(
            momenta,
            spin_vectors=self._vectors,
            helicities=helicities,
            precision=amplitude_precision,
        )
        return self._scaled(
            _contract(batch, self._matrices[color_correlation], precision=precision),
            precision,
        )

    def evaluate_graded(
        self,
        momenta: Momenta,
        *,
        color_correlation: str = "born",
        helicities: Sequence[str] | None = None,
        precision: int = 16,
        arithmetic: str = "arbitrary",
    ) -> GradedCorrelatedEvaluation:
        """``evaluate`` split by the N_c degrees of a graded matrix."""

        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision < 1
        ):
            raise EvaluationError(
                "precision must be a positive number of decimal digits"
            )
        matrix = self._matrices.get(color_correlation)
        if not isinstance(color_correlation, str) or matrix is None:
            raise EvaluationError(
                f"unknown colour correlation ID {color_correlation!r}"
            )
        if matrix.degrees is None:
            raise CompatibilityError(
                "graded correlations need an artifact generated with "
                "color.accuracy='strict-lc' or 'strict-nlc', or with "
                "color.graded and full colour"
            )
        amplitude_precision = correlated_precision(precision, arithmetic)
        batch = self._select_executor(arithmetic).coherent_amplitudes(
            momenta,
            spin_vectors=self._vectors,
            helicities=helicities,
            precision=amplitude_precision,
        )
        points = _prepare_contraction(
            batch.groups,
            matrix,
            precision=precision,
            scalar=type(batch.normalization_factor),
        ).evaluate_graded(batch, precision=precision)
        columns = [
            self._scaled(tuple(point[index] for point in points), precision)
            for index in range(len(matrix.degrees) + 1)
        ]
        return GradedCorrelatedEvaluation(
            degrees=matrix.degrees,
            values=tuple(
                tuple(column[point] for column in columns[:-1])
                for point in range(len(points))
            ),
            total=columns[-1],
        )

    def _scaled(
        self, values: tuple[CorrelatedValue, ...], precision: int
    ) -> tuple[CorrelatedValue, ...]:
        """Apply a process-group member's exact factor with one rounding."""

        factor = self._catalogue.factor
        if factor == 1:
            return values
        with localcontext() as context:
            context.rounding = ROUND_HALF_EVEN
            scaled = []
            for value in values:
                parts = []
                for part in (value.real, value.imag):
                    context.prec = max(precision, len(part.as_tuple().digits)) + len(
                        str(factor.numerator)
                    )
                    product = part * factor.numerator
                    context.prec = precision
                    parts.append(product / factor.denominator)
                scaled.append(CorrelatedValue(parts[0], parts[1]))
        return tuple(scaled)

    def evaluate_many(
        self,
        momenta: Momenta,
        requests: Mapping[str, CorrelatedRequest],
        *,
        helicities: Sequence[str] | None = None,
        precision: int = 16,
        arithmetic: str = "arbitrary",
    ) -> dict[str, tuple[CorrelatedValue, ...]]:
        """Share amplitudes and exact unchanged stages within one labelled batch."""
        if type(precision) is not int or precision < 1:
            raise EvaluationError(
                "precision must be a positive number of decimal digits"
            )
        amplitude_precision = correlated_precision(precision, arithmetic)
        if not isinstance(requests, Mapping):
            raise EvaluationError(
                "correlated requests must map labels to CorrelatedRequest"
            )
        snapshot = dict(self._vectors)
        assignments: list[dict[int, object]] = []
        assignment_ids: dict[tuple[object, ...], int] = {}
        routes: list[tuple[str, str, int]] = []
        for label, request in requests.items():
            if not isinstance(label, str) or not label:
                raise EvaluationError(
                    "correlated request labels must be nonempty strings"
                )
            if not isinstance(request, CorrelatedRequest):
                raise EvaluationError(
                    "correlated requests must contain CorrelatedRequest records"
                )
            identifier = request.color_correlation
            if identifier not in self._matrices:
                raise EvaluationError(f"unknown colour correlation ID {identifier!r}")
            vectors = self._copy_spin_vectors(
                snapshot if request.spin_vectors is None else request.spin_vectors
            )
            prepared = _prepare_spin_vectors(
                vectors,
                allowed_legs=self._declarations.spin_legs,
                point_count=len(momenta),
            )
            # Normalize broadcast/per-point shapes, without a float conversion.
            # Decimal tuples conservatively preserve signed zero and precision.
            key: tuple[object, ...] = tuple(
                tuple(
                    (
                        leg,
                        tuple(
                            (real.as_tuple(), imag.as_tuple()) for real, imag in wave
                        ),
                    )
                    for leg, wave in sorted(point.items())
                )
                for point in prepared
            )
            if key not in assignment_ids:
                assignment_ids[key] = len(assignments)
                assignments.append(vectors)
            routes.append((label, identifier, assignment_ids[key]))
        if not routes:
            return {}
        by_assignment: dict[int, dict[str, list[str]]] = {}
        for label, identifier, assignment in routes:
            by_assignment.setdefault(assignment, {}).setdefault(identifier, []).append(
                label
            )
        results: dict[str, list[CorrelatedValue]] = {
            label: [] for label, _, _ in routes
        }
        plans: dict[tuple[int, str], _ContractionPlan] = {}
        for (
            point_index,
            assignment,
            batch,
        ) in self._select_executor(arithmetic).iter_coherent_amplitudes_many(
            momenta,
            spin_vector_sets=assignments,
            helicities=helicities,
            precision=amplitude_precision,
        ):
            for identifier, labels in by_assignment[assignment].items():
                plan_key = (assignment, identifier)
                if plan_key not in plans:
                    plans[plan_key] = _prepare_contraction(
                        batch.groups,
                        self._matrices[identifier],
                        precision=precision,
                        scalar=type(batch.normalization_factor),
                    )
                value = plans[plan_key].evaluate(batch, precision=precision)[0]
                for label in labels:
                    if len(results[label]) != point_index:
                        raise ArtifactError(
                            "correlated batch returned inconsistent point ordering"
                        )
                    results[label].append(value)
        if any(len(values) != len(momenta) for values in results.values()):
            raise ArtifactError("correlated batch did not cover all requested points")
        return {
            label: self._scaled(tuple(values), precision)
            for label, values in results.items()
        }
