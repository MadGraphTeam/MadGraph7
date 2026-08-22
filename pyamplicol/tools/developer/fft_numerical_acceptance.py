# SPDX-License-Identifier: 0BSD
"""Opt-in process-table numerical acceptance for FFT colour contraction.

This gate is deliberately separate from the ordinary numerical-acceptance
campaign.  It constructs its process surface from the performance catalog,
discovers one nonzero helicity with a cached direct oracle, and compares direct
and symmetric-group-FFT colour contraction without ever generating an artifact
per process.  Recurrence artifacts specialize that tuple at generation time;
on-the-fly artifacts retain complete helicity coverage and apply the identical
recorded tuple through the runtime selector API.

The real campaign is intentionally entered only through
``just fft-numerical-acceptance``.  ``--dry-run`` authenticates the frozen
MadGraph fixture and prints the complete planned comparison surface without
loading the native runtime or writing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tomllib
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Generic, Literal, TypeVar, cast

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from tools.developer.numerical_acceptance import (  # noqa: E402
    DEFAULT_FIXTURE,
    POINT_SEED,
    RELATIVE_TOLERANCE,
    current_model_identity,
    prepare_ufo_sm_model,
    validation_momenta,
)
from tools.developer.reference_capture.common import canonical_decimal  # noqa: E402
from tools.performance_report.catalog import PROCESS_FAMILIES  # noqa: E402
from tools.performance_report.models import (  # noqa: E402
    Accuracy,
    open_quark_line_count,
)
from tools.performance_report.selector_policy import (  # noqa: E402
    _preferred_helicities,
    fixed_selector_helicity,
    selector_helicity_id,
)

GATE_KIND = "pyamplicol-fft-numerical-acceptance"
GATE_SCHEMA_VERSION = 1
SELECTED_MAX_N_FINAL = 5
TOTAL_MAX_N_FINAL = 4
SELECTED_CASE_COUNT = 47
TOTAL_CASE_COUNT = 33

MethodName = Literal["direct", "symmetric-group-fft"]
ModelName = Literal["built-in-sm", "ufo-sm"]
ModeName = Literal["recurrence", "on-the-fly"]
AccuracyName = Literal["full", "nlc"]
SelectionExecution = Literal[
    "generation-specialized",
    "runtime-query-complete-coverage",
]

METHODS: tuple[MethodName, ...] = ("direct", "symmetric-group-fft")
MODELS: tuple[ModelName, ...] = ("built-in-sm", "ufo-sm")
MODES: tuple[ModeName, ...] = ("recurrence", "on-the-fly")
ACCURACIES: tuple[AccuracyName, ...] = ("full", "nlc")

_PURE_ADJOINT_REFLECTION_CASE_ID = "catalog:gg_gluons:n2"
_OPEN_LINE_REFLECTION_CASE_ID = "catalog:gg_tt_jets:n2"
# Query-family closures retain one amplitude destination per requested colour;
# the certified reflection fold is visible in the shared current finalizations.
_REFLECTION_CENSUS_FIELDS = (
    "union_unique_current_count",
    "union_contribution_rows",
    "union_finalization_rows",
)
_EXPECTED_PURE_ADJOINT_REFLECTION_CENSUS = {
    "direct": (22, 36, 6),
    "symmetric-group-fft": (13, 18, 3),
}


class FFTAcceptanceError(RuntimeError):
    """The FFT acceptance contract could not be executed or authenticated."""


class FFTAcceptanceMismatch(AssertionError):
    """At least one strict numerical or nonzero assertion failed."""


@dataclass(frozen=True, slots=True)
class CatalogCase:
    case_id: str
    family_id: int
    family_key: str
    n_final: int
    process: str

    @property
    def artifact_name(self) -> str:
        return self.case_id.replace(":", "_").replace("-", "_")


def _symmetric_group_degree(case: CatalogCase) -> int:
    """Return the nontrivial group degree implied by one catalog process."""

    initial, separator, final = case.process.partition(">")
    if not separator:
        raise FFTAcceptanceError(
            f"catalog case {case.case_id} has no initial/final separator"
        )
    tokens = (*initial.split(), *final.split())
    adjoint_count = sum(token == "g" for token in tokens)
    try:
        open_line_count = open_quark_line_count(case.process)
    except ValueError as error:
        raise FFTAcceptanceError(
            f"catalog case {case.case_id} has invalid open-line content"
        ) from error
    return adjoint_count if open_line_count else max(adjoint_count - 1, 0)


def _expected_inspection_method(
    request_method: MethodName,
    case: CatalogCase,
) -> str | None:
    if request_method == "direct":
        return None
    return "symmetric-group-fourier" if _symmetric_group_degree(case) >= 2 else None


def catalog_cases(max_n_final: int = SELECTED_MAX_N_FINAL) -> tuple[CatalogCase, ...]:
    """Derive the contracted-colour process surface from the live catalog."""

    if max_n_final < 1:
        raise ValueError("max_n_final must be positive")
    cases: list[CatalogCase] = []
    for n_final in range(1, max_n_final + 1):
        for family in sorted(PROCESS_FAMILIES, key=lambda item: item.identifier):
            process = family.process(n_final)
            if process is None or n_final > family.maximum_n(Accuracy.FULL):
                continue
            cases.append(
                CatalogCase(
                    case_id=f"catalog:{family.key}:n{n_final}",
                    family_id=family.identifier,
                    family_key=family.key,
                    n_final=n_final,
                    process=process,
                )
            )
    result = tuple(cases)
    expected = {
        TOTAL_MAX_N_FINAL: TOTAL_CASE_COUNT,
        SELECTED_MAX_N_FINAL: SELECTED_CASE_COUNT,
    }.get(max_n_final)
    if expected is not None and len(result) != expected:
        raise FFTAcceptanceError(
            f"the performance catalog through n_final={max_n_final} must contain "
            f"{expected} contracted-colour cases; observed {len(result)}"
        )
    return result


@dataclass(frozen=True, slots=True)
class FrozenFullCase:
    case: CatalogCase
    momenta: tuple[tuple[Decimal, ...], ...]
    expected_full: Decimal


@dataclass(frozen=True, slots=True)
class FrozenFullEvidence:
    """The immutable FullColour subset of the tracked MadGraph capture."""

    path: Path
    madgraph_version: str
    model_source_sha256: str
    driver_sha256: str
    cases: tuple[FrozenFullCase, ...]

    def case(self, case_id: str) -> FrozenFullCase:
        for case in self.cases:
            if case.case.case_id == case_id:
                return case
        raise KeyError(case_id)


def _mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FFTAcceptanceError(f"{where} must be an object")
    return cast(Mapping[str, object], value)


def _reflection_census_triplet(
    census: Mapping[str, object],
    *,
    label: str,
) -> tuple[int, int, int]:
    if (
        census.get("basis") != "shared-query-family-union-v1"
        or census.get("scope") != "active-family-union"
    ):
        raise FFTAcceptanceError(
            f"{label} has the wrong active-family census identity: "
            f"basis={census.get('basis')!r}, scope={census.get('scope')!r}"
        )
    values: list[int] = []
    for field in _REFLECTION_CENSUS_FIELDS:
        value = census.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FFTAcceptanceError(
                f"{label} has invalid {field}: {value!r}"
            )
        values.append(value)
    return cast(tuple[int, int, int], tuple(values))


def validate_on_the_fly_reflection_census(
    *,
    pure_adjoint_direct: Mapping[str, object],
    pure_adjoint_fft: Mapping[str, object],
    open_line_direct: Mapping[str, object],
    open_line_fft: Mapping[str, object],
) -> dict[str, object]:
    """Lock the demonstrated n=2 production reflection boundary."""

    pure = {
        "direct": _reflection_census_triplet(
            pure_adjoint_direct,
            label=f"{_PURE_ADJOINT_REFLECTION_CASE_ID} direct",
        ),
        "symmetric-group-fft": _reflection_census_triplet(
            pure_adjoint_fft,
            label=f"{_PURE_ADJOINT_REFLECTION_CASE_ID} symmetric-group-fft",
        ),
    }
    for method, expected in _EXPECTED_PURE_ADJOINT_REFLECTION_CENSUS.items():
        observed = pure[method]
        if observed != expected:
            raise FFTAcceptanceError(
                f"{_PURE_ADJOINT_REFLECTION_CASE_ID} {method} active-family census "
                f"{_REFLECTION_CENSUS_FIELDS!r} is {observed!r}; expected {expected!r}"
            )

    open_direct = dict(open_line_direct)
    open_fft = dict(open_line_fft)
    open_triplets = {
        "direct": _reflection_census_triplet(
            open_direct,
            label=f"{_OPEN_LINE_REFLECTION_CASE_ID} direct",
        ),
        "symmetric-group-fft": _reflection_census_triplet(
            open_fft,
            label=f"{_OPEN_LINE_REFLECTION_CASE_ID} symmetric-group-fft",
        ),
    }
    if open_direct != open_fft:
        differences = {
            field: {
                "direct": open_direct.get(field),
                "symmetric-group-fft": open_fft.get(field),
            }
            for field in sorted(open_direct.keys() | open_fft.keys())
            if open_direct.get(field) != open_fft.get(field)
        }
        raise FFTAcceptanceError(
            f"{_OPEN_LINE_REFLECTION_CASE_ID} has a nontrivial FFT workspace but "
            "an open color line, so its direct and FFT active-family censuses "
            f"must be identical; differences={differences!r}"
        )

    return {
        "basis": "selected-helicity-production-active-family-census-v1",
        "source": "Runtime.inspect().on_the_fly_state.active_family_union_census",
        "pure_adjoint": {
            "case_id": _PURE_ADJOINT_REFLECTION_CASE_ID,
            "fields": list(_REFLECTION_CENSUS_FIELDS),
            "direct": list(pure["direct"]),
            "symmetric_group_fft": list(pure["symmetric-group-fft"]),
        },
        "open_line": {
            "case_id": _OPEN_LINE_REFLECTION_CASE_ID,
            "direct_equals_symmetric_group_fft": True,
            "fields": list(_REFLECTION_CENSUS_FIELDS),
            "shared": list(open_triplets["direct"]),
        },
    }


def _sequence(value: object, where: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise FFTAcceptanceError(f"{where} must be an array")
    return cast(Sequence[object], value)


def _canonical_fixture_decimal(value: object, where: str) -> Decimal:
    if not isinstance(value, str):
        raise FFTAcceptanceError(f"{where} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except ArithmeticError as error:
        raise FFTAcceptanceError(f"{where} is not a decimal") from error
    if not parsed.is_finite() or canonical_decimal(parsed) != value:
        raise FFTAcceptanceError(f"{where} must be a canonical finite decimal")
    return parsed


def _duplicate_rejecting_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FFTAcceptanceError(f"frozen fixture repeats JSON key {key!r}")
        result[key] = value
    return result


def load_frozen_full_evidence(path: Path = DEFAULT_FIXTURE) -> FrozenFullEvidence:
    """Authenticate and load only the immutable independent FullColour evidence.

    The capture records the exact MadGraph driver digest used to produce it.
    Replaying immutable values must not require that today's capture adapter has
    the same digest: doing so would turn harmless adapter maintenance into a
    demand to rerun the explicitly frozen independent oracle.
    """

    source = path.expanduser().resolve(strict=True)
    try:
        payload = json.loads(
            source.read_text(encoding="ascii"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FFTAcceptanceError(f"frozen fixture has non-finite token {token!r}")
            ),
        )
    except FFTAcceptanceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FFTAcceptanceError(
            f"cannot read frozen fixture {source}: {error}"
        ) from error
    root = _mapping(payload, "frozen fixture")
    if (
        root.get("kind") != "pyamplicol-ufo-sm-numerical-acceptance"
        or root.get("schema_version") != 1
        or root.get("catalog_max_n_final") != TOTAL_MAX_N_FINAL
    ):
        raise FFTAcceptanceError("frozen fixture has the wrong identity or bounds")
    point_policy = _mapping(root.get("point_policy"), "point_policy")
    if dict(point_policy) != {
        "generator": "generic_validation_point",
        "seed": POINT_SEED,
        "stored_components": "canonical-binary64-decimal",
    }:
        raise FFTAcceptanceError(
            "frozen fixture does not use the seed-101 point policy"
        )
    comparison = _mapping(root.get("comparison"), "comparison")
    if (
        _canonical_fixture_decimal(
            comparison.get("relative_tolerance"), "comparison.relative_tolerance"
        )
        != RELATIVE_TOLERANCE
        or comparison.get("absolute_tolerance") is not None
    ):
        raise FFTAcceptanceError(
            "frozen fixture does not use strict relative-only tolerance 1e-10"
        )
    model = _mapping(root.get("model"), "model")
    if dict(model) != current_model_identity().as_payload():
        raise FFTAcceptanceError(
            "frozen fixture model identity differs from the packaged UFO-SM"
        )
    references = _mapping(root.get("references"), "references")
    full_reference = _mapping(references.get("full"), "references.full")
    if (
        full_reference.get("kind") != "madgraph-standalone-ufo-sm"
        or full_reference.get("precision") != "binary64"
        or full_reference.get("command_protocol")
        != "generate-output-standalone-launch-force"
        or full_reference.get("external_parameters_sha256")
        != model.get("external_parameters_sha256")
    ):
        raise FFTAcceptanceError("frozen FullColour reference identity is invalid")
    madgraph_version = full_reference.get("madgraph_version")
    model_source_sha256 = full_reference.get("model_source_sha256")
    driver_sha256 = full_reference.get("driver_sha256")
    if not isinstance(madgraph_version, str) or not madgraph_version:
        raise FFTAcceptanceError("frozen MadGraph version is invalid")
    for label, digest in (
        ("model source", model_source_sha256),
        ("driver", driver_sha256),
    ):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise FFTAcceptanceError(f"frozen MadGraph {label} digest is invalid")

    specs = catalog_cases(TOTAL_MAX_N_FINAL)
    raw_cases = _sequence(root.get("catalog_cases"), "catalog_cases")
    if len(raw_cases) != len(specs):
        raise FFTAcceptanceError(
            f"frozen fixture must contain {len(specs)} catalog cases"
        )
    cases: list[FrozenFullCase] = []
    for spec, raw_case in zip(specs, raw_cases, strict=True):
        record = _mapping(raw_case, f"frozen case {spec.case_id}")
        observed_identity = (
            record.get("id"),
            record.get("family_id"),
            record.get("family_key"),
            record.get("n_final"),
            record.get("process"),
        )
        expected_identity = (
            spec.case_id,
            spec.family_id,
            spec.family_key,
            spec.n_final,
            spec.process,
        )
        if observed_identity != expected_identity:
            raise FFTAcceptanceError(
                f"frozen case {spec.case_id} differs from the live catalog"
            )
        raw_momenta = _sequence(record.get("momenta"), f"{spec.case_id}.momenta")
        momenta = tuple(
            tuple(
                _canonical_fixture_decimal(
                    component,
                    f"{spec.case_id}.momenta[{row_index}][{component_index}]",
                )
                for component_index, component in enumerate(
                    _sequence(row, f"{spec.case_id}.momenta[{row_index}]")
                )
            )
            for row_index, row in enumerate(raw_momenta)
        )
        if len(momenta) != spec.n_final + 2 or any(
            len(momentum) != 4 for momentum in momenta
        ):
            raise FFTAcceptanceError(
                f"frozen case {spec.case_id} has malformed momentum dimensions"
            )
        expected = _mapping(record.get("expected"), f"{spec.case_id}.expected")
        expected_full = _canonical_fixture_decimal(
            expected.get("full"), f"{spec.case_id}.expected.full"
        )
        if expected_full < 0:
            raise FFTAcceptanceError(
                f"frozen case {spec.case_id} has a negative FullColour value"
            )
        cases.append(FrozenFullCase(spec, momenta, expected_full))
    return FrozenFullEvidence(
        source,
        madgraph_version,
        cast(str, model_source_sha256),
        cast(str, driver_sha256),
        tuple(cases),
    )


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    authority: Literal["direct-vs-fft", "frozen-madgraph"]
    model: ModelName
    mode: ModeName
    accuracy: AccuracyName
    helicity_scope: Literal["selected", "total"]
    case_id: str

    def as_payload(self) -> dict[str, object]:
        return {
            "authority": self.authority,
            "model": self.model,
            "mode": self.mode,
            "accuracy": self.accuracy,
            "helicity_scope": self.helicity_scope,
            "case_id": self.case_id,
            "selection_execution": (
                None
                if self.helicity_scope == "total"
                else selection_execution(self.mode)
            ),
        }


def selection_execution(mode: ModeName) -> SelectionExecution:
    """Describe how one recorded helicity is applied in an execution mode."""

    if mode == "recurrence":
        return "generation-specialized"
    return "runtime-query-complete-coverage"


def comparison_specs() -> tuple[ComparisonSpec, ...]:
    """Return the complete required parity and independent-oracle matrix."""

    selected = catalog_cases(SELECTED_MAX_N_FINAL)
    totals = catalog_cases(TOTAL_MAX_N_FINAL)
    specs: list[ComparisonSpec] = []
    for model in MODELS:
        for mode in MODES:
            for accuracy in ACCURACIES:
                specs.extend(
                    ComparisonSpec(
                        "direct-vs-fft",
                        model,
                        mode,
                        accuracy,
                        "selected",
                        case.case_id,
                    )
                    for case in selected
                )
            specs.extend(
                ComparisonSpec(
                    "direct-vs-fft",
                    model,
                    mode,
                    "full",
                    "total",
                    case.case_id,
                )
                for case in totals
            )
    for mode in MODES:
        specs.extend(
            ComparisonSpec(
                "frozen-madgraph",
                "ufo-sm",
                mode,
                "full",
                "total",
                case.case_id,
            )
            for case in totals
        )
    return tuple(specs)


def _fixture_case_ids(fixture: FrozenFullEvidence) -> tuple[str, ...]:
    return tuple(case.case.case_id for case in fixture.cases)


def dry_run_payload(
    fixture_path: Path = DEFAULT_FIXTURE,
) -> dict[str, object]:
    """Authenticate all static inputs and describe the non-mutating campaign."""

    fixture = load_frozen_full_evidence(fixture_path)
    selected = catalog_cases(SELECTED_MAX_N_FINAL)
    totals = catalog_cases(TOTAL_MAX_N_FINAL)
    expected_fixture_ids = tuple(case.case_id for case in totals)
    if _fixture_case_ids(fixture) != expected_fixture_ids:
        raise FFTAcceptanceError(
            "the frozen UFO-SM fixture no longer matches the live n_final<=4 catalog"
        )
    specs = comparison_specs()
    direct_fft = sum(spec.authority == "direct-vs-fft" for spec in specs)
    frozen = sum(spec.authority == "frozen-madgraph" for spec in specs)
    return {
        "kind": GATE_KIND,
        "schema_version": GATE_SCHEMA_VERSION,
        "dry_run": True,
        "point_policy": {
            "generator": "generic_validation_point",
            "seed": POINT_SEED,
        },
        "comparison_policy": {
            "relative_tolerance": canonical_decimal(RELATIVE_TOLERANCE),
            "absolute_tolerance": None,
        },
        "catalog": {
            "source": "tools/performance_report/catalog.py:PROCESS_FAMILIES",
            "selected_max_n_final": SELECTED_MAX_N_FINAL,
            "selected_case_count_per_model": len(selected),
            "total_max_n_final": TOTAL_MAX_N_FINAL,
            "total_case_count_per_model": len(totals),
            "cases": [
                {
                    "id": case.case_id,
                    "family_id": case.family_id,
                    "family_key": case.family_key,
                    "n_final": case.n_final,
                    "process": case.process,
                }
                for case in selected
            ],
        },
        "process_set_cache_policy": {
            "base_key_axes": [
                "method",
                "model",
                "mode",
                "accuracy",
                "helicity_scope",
            ],
            "selected_partition": (
                "recurrence only: canonical external-helicity-domain signature "
                "plus exact discovered tuple"
            ),
            "selected_execution": {
                "recurrence": "generation-specialized",
                "on-the-fly": "runtime-query-complete-coverage",
            },
            "on_the_fly_complete_coverage": (
                "one ProcessSet per method/model/accuracy, shared by selected "
                "runtime queries and FullColour totals"
            ),
            "discovery": (
                "grouped direct recurrence preferred-selector probes; at most "
                "one all-helicity fallback ProcessSet per model/accuracy, only "
                "for observed zero rows"
            ),
            "per_process_generation": False,
        },
        "frozen_authority": {
            "fixture": str(fixture_path.resolve(strict=True)),
            "model": "ufo-sm",
            "accuracy": "full",
            "max_n_final": TOTAL_MAX_N_FINAL,
            "case_count_per_mode": len(fixture.cases),
            "madgraph_version": fixture.madgraph_version,
            "captured_model_source_sha256": fixture.model_source_sha256,
            "captured_driver_sha256": fixture.driver_sha256,
            "madgraph_rerun": False,
        },
        "comparison_counts": {
            "direct_vs_fft": direct_fft,
            "frozen_madgraph": frozen,
            "total": len(specs),
        },
        "comparisons": [spec.as_payload() for spec in specs],
    }


def _decimal_component(value: object, where: str) -> tuple[Decimal, Decimal]:
    if isinstance(value, bool):
        raise FFTAcceptanceError(f"{where} must be numeric")
    if isinstance(value, Decimal):
        real, imaginary = value, Decimal(0)
    elif isinstance(value, int):
        real, imaginary = Decimal(value), Decimal(0)
    else:
        try:
            number = complex(value)  # type: ignore[arg-type]
        except (OverflowError, TypeError, ValueError) as error:
            raise FFTAcceptanceError(f"{where} must be numeric") from error
        if not math.isfinite(number.real) or not math.isfinite(number.imag):
            raise FFTAcceptanceError(f"{where} must be finite")
        real = Decimal(canonical_decimal(number.real))
        imaginary = Decimal(canonical_decimal(number.imag))
    if not real.is_finite() or not imaginary.is_finite():
        raise FFTAcceptanceError(f"{where} must be finite")
    return real, imaginary


@dataclass(frozen=True, slots=True)
class StrictRelativeComparison:
    candidate_real: Decimal
    candidate_imaginary: Decimal
    reference_real: Decimal
    reference_imaginary: Decimal
    difference: Decimal
    scale: Decimal
    bound: Decimal
    passed: bool

    def as_payload(self) -> dict[str, object]:
        return {
            "candidate": [
                canonical_decimal(self.candidate_real),
                canonical_decimal(self.candidate_imaginary),
            ],
            "reference": [
                canonical_decimal(self.reference_real),
                canonical_decimal(self.reference_imaginary),
            ],
            "difference": canonical_decimal(self.difference),
            "scale": canonical_decimal(self.scale),
            "bound": canonical_decimal(self.bound),
            "passed": self.passed,
        }


def strict_relative_compare(
    candidate: object,
    reference: object,
    *,
    tolerance: Decimal = RELATIVE_TOLERANCE,
) -> StrictRelativeComparison:
    """Apply the established scale-relative comparison with no absolute floor."""

    candidate_real, candidate_imaginary = _decimal_component(
        candidate, "candidate matrix element"
    )
    reference_real, reference_imaginary = _decimal_component(
        reference, "reference matrix element"
    )
    if tolerance < 0 or not tolerance.is_finite():
        raise FFTAcceptanceError("relative tolerance must be finite and non-negative")
    precision = max(
        80,
        *(
            len(value.as_tuple().digits) + 20
            for value in (
                candidate_real,
                candidate_imaginary,
                reference_real,
                reference_imaginary,
            )
        ),
    )
    with localcontext() as context:
        context.prec = precision
        candidate_magnitude = (
            candidate_real * candidate_real + candidate_imaginary * candidate_imaginary
        ).sqrt()
        reference_magnitude = (
            reference_real * reference_real + reference_imaginary * reference_imaginary
        ).sqrt()
        difference = (
            (candidate_real - reference_real) ** 2
            + (candidate_imaginary - reference_imaginary) ** 2
        ).sqrt()
        scale = max(candidate_magnitude, reference_magnitude)
        bound = tolerance * scale
    passed = difference == 0 if scale == 0 else difference <= bound
    return StrictRelativeComparison(
        candidate_real=candidate_real,
        candidate_imaginary=candidate_imaginary,
        reference_real=reference_real,
        reference_imaginary=reference_imaginary,
        difference=difference,
        scale=scale,
        bound=bound,
        passed=passed,
    )


@dataclass(frozen=True, slots=True)
class HelicityObservation:
    helicity_id: str
    values: tuple[int, ...]
    structural_zero: bool
    matrix_element: object


@dataclass(frozen=True, slots=True)
class ChosenHelicity:
    observation: HelicityObservation
    domains: tuple[tuple[int, ...], ...]
    source: Literal["preferred-structural-selector", "largest-nonzero-fallback"]

    @property
    def signature(self) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
        return self.domains, self.observation.values


def choose_nonzero_helicity(
    observations: Sequence[HelicityObservation],
    *,
    preferred: Sequence[int],
) -> ChosenHelicity:
    """Choose a stable nonzero row, preferring the report selector policy."""

    rows = tuple(observations)
    if not rows:
        raise FFTAcceptanceError("helicity discovery returned no physical rows")
    width = len(rows[0].values)
    if width == 0 or any(len(row.values) != width for row in rows):
        raise FFTAcceptanceError("helicity discovery rows have inconsistent width")
    if len({row.helicity_id for row in rows}) != len(rows):
        raise FFTAcceptanceError("helicity discovery rows have duplicate IDs")
    domains = tuple(
        tuple(sorted({row.values[index] for row in rows})) for index in range(width)
    )
    nonzero = tuple(
        row
        for row in rows
        if not row.structural_zero
        and strict_relative_compare(row.matrix_element, 0).scale > 0
    )
    preferred_tuple = tuple(int(value) for value in preferred)
    preferred_rows = tuple(row for row in nonzero if row.values == preferred_tuple)
    if len(preferred_rows) == 1:
        return ChosenHelicity(
            preferred_rows[0], domains, "preferred-structural-selector"
        )
    if not nonzero:
        raise FFTAcceptanceError(
            "direct all-helicity oracle found no nonzero physical helicity"
        )
    ordered = sorted(nonzero, key=lambda row: (row.values, row.helicity_id))
    selected = max(
        ordered,
        key=lambda row: strict_relative_compare(row.matrix_element, 0).scale,
    )
    return ChosenHelicity(selected, domains, "largest-nonzero-fallback")


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    model: ModelName
    accuracy: AccuracyName
    case: CatalogCase
    helicity_id: str
    values: tuple[int, ...]
    domains: tuple[tuple[int, ...], ...]
    source: str
    discovery_value: object

    @property
    def signature(self) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
        return self.domains, self.values

    @property
    def source_mapping(self) -> tuple[tuple[int, int], ...]:
        return tuple(enumerate(self.values, start=1))

    def as_payload(self) -> dict[str, object]:
        value_real, value_imaginary = _decimal_component(
            self.discovery_value, "discovery matrix element"
        )
        return {
            "model": self.model,
            "accuracy": self.accuracy,
            "case_id": self.case.case_id,
            "process": self.case.process,
            "n_final": self.case.n_final,
            "helicity_id": self.helicity_id,
            "source_helicities": [list(item) for item in self.source_mapping],
            "external_helicity_domains": [list(domain) for domain in self.domains],
            "selection_source": self.source,
            "discovery_oracle": {
                "method": "direct",
                "mode": "recurrence",
                "value": [
                    canonical_decimal(value_real),
                    canonical_decimal(value_imaginary),
                ],
                "nonzero": strict_relative_compare(self.discovery_value, 0).scale > 0,
            },
        }


def _signature_digest(
    signature: tuple[tuple[tuple[int, ...], ...], tuple[int, ...]],
) -> str:
    domains, values = signature
    encoded = json.dumps(
        {"domains": domains, "values": values},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class SelectedGroup:
    signature: tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]
    cases: tuple[CatalogCase, ...]

    @property
    def scope(self) -> str:
        return f"selected-{_signature_digest(self.signature)}"

    @property
    def source_mapping(self) -> tuple[tuple[int, int], ...]:
        return tuple(enumerate(self.signature[1], start=1))


def group_selected_cases(
    cases: Sequence[CatalogCase],
    selections: Mapping[str, SelectionRecord],
) -> tuple[SelectedGroup, ...]:
    """Partition a selected scope by domain and exact discovered tuple."""

    grouped: OrderedDict[
        tuple[tuple[tuple[int, ...], ...], tuple[int, ...]], list[CatalogCase]
    ] = OrderedDict()
    for case in cases:
        try:
            selection = selections[case.case_id]
        except KeyError as error:
            raise FFTAcceptanceError(
                f"missing discovered helicity for {case.case_id}"
            ) from error
        if selection.case != case:
            raise FFTAcceptanceError(
                f"discovered helicity identity differs for {case.case_id}"
            )
        grouped.setdefault(selection.signature, []).append(case)
    return tuple(
        SelectedGroup(signature, tuple(group_cases))
        for signature, group_cases in grouped.items()
    )


@dataclass(frozen=True, slots=True)
class ArtifactKey:
    method: MethodName
    model: ModelName
    mode: ModeName
    accuracy: AccuracyName
    helicity_scope: str

    @property
    def slug(self) -> str:
        method = "fft" if self.method == "symmetric-group-fft" else "direct"
        mode = self.mode.replace("-", "_")
        return "-".join((method, self.model, mode, self.accuracy, self.helicity_scope))

    def as_payload(self) -> dict[str, str]:
        return {
            "method": self.method,
            "model": self.model,
            "mode": self.mode,
            "accuracy": self.accuracy,
            "helicity_scope": self.helicity_scope,
        }


@dataclass(frozen=True, slots=True)
class GroupRequest:
    key: ArtifactKey
    cases: tuple[CatalogCase, ...]
    selected_source_helicities: tuple[tuple[int, int], ...] | None = None
    external_helicity_domains: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("a ProcessSet group must not be empty")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("a ProcessSet group contains duplicate cases")
        selected = self.selected_source_helicities
        domains = self.external_helicity_domains
        if (selected is None) != (domains is None):
            raise ValueError("selected source tuples and domains must be paired")
        if selected is not None:
            if self.key.mode == "on-the-fly":
                raise ValueError(
                    "on-the-fly ProcessSets must retain complete helicity coverage"
                )
            if tuple(label for label, _value in selected) != tuple(
                range(1, len(selected) + 1)
            ):
                raise ValueError("selected source labels must be dense and one-based")
            if domains is None or len(domains) != len(selected):
                raise ValueError("selected source tuple and domain widths differ")
            for (label, value), domain in zip(selected, domains, strict=True):
                if value not in domain:
                    raise ValueError(
                        f"selected helicity {value} for label {label} is outside "
                        f"its domain {domain!r}"
                    )

    def as_payload(self) -> dict[str, object]:
        return {
            "boundary": self.key.as_payload(),
            "helicity_generation_contract": (
                "complete-coverage"
                if self.selected_source_helicities is None
                else "generation-specialized"
            ),
            "case_count": len(self.cases),
            "case_ids": [case.case_id for case in self.cases],
            "selected_source_helicities": (
                None
                if self.selected_source_helicities is None
                else [list(item) for item in self.selected_source_helicities]
            ),
            "external_helicity_domains": (
                None
                if self.external_helicity_domains is None
                else [list(item) for item in self.external_helicity_domains]
            ),
        }


def otf_complete_group_request(
    *,
    method: MethodName,
    model: ModelName,
    accuracy: AccuracyName,
    cases: Sequence[CatalogCase],
) -> GroupRequest:
    """Build the single reusable OTF helicity-coverage boundary for one lane."""

    return GroupRequest(
        ArtifactKey(
            method,
            model,
            "on-the-fly",
            accuracy,
            "complete-helicity-runtime-query-and-total",
        ),
        tuple(cases),
    )


RecordT = TypeVar("RecordT")


class ProcessSetCache(Generic[RecordT]):
    """Memoize one builder call per exact method/model/mode/scope boundary."""

    def __init__(self, builder: Callable[[GroupRequest], RecordT]) -> None:
        self._builder = builder
        self._requests: dict[ArtifactKey, GroupRequest] = {}
        self._records: dict[ArtifactKey, RecordT] = {}
        self._generation_order: list[ArtifactKey] = []

    def get(self, request: GroupRequest) -> RecordT:
        previous = self._requests.get(request.key)
        if previous is not None:
            if previous != request:
                raise FFTAcceptanceError(
                    "a ProcessSet cache boundary was reused with a different case "
                    f"or helicity scope: {request.key.slug}"
                )
            return self._records[request.key]
        record = self._builder(request)
        self._requests[request.key] = request
        self._records[request.key] = record
        self._generation_order.append(request.key)
        return record

    @property
    def generation_count(self) -> int:
        return len(self._generation_order)

    @property
    def entries(self) -> tuple[tuple[GroupRequest, RecordT], ...]:
        return tuple(
            (self._requests[key], self._records[key]) for key in self._generation_order
        )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    request: GroupRequest
    path: Path


def _run_config(request: GroupRequest) -> object:
    from pyamplicol.config import (
        Action,
        ColorAccuracy,
        ColorConfig,
        ColorContraction,
        EvaluatorConfig,
        EvaluatorExecutionMode,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        GenerationRelationDiscoveryConfig,
        GenerationValidationConfig,
        JITConfig,
        ProcessConfig,
        RelationDiscoveryMode,
        RunConfig,
    )

    selected = request.selected_source_helicities or ()
    return RunConfig(
        action=Action.GENERATE,
        process=ProcessConfig(
            selected_source_helicities={str(label): value for label, value in selected}
        ),
        color=ColorConfig(
            accuracy=ColorAccuracy(request.key.accuracy),
            contraction=ColorContraction(request.key.method),
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
            relation_discovery=GenerationRelationDiscoveryConfig(
                mode=RelationDiscoveryMode.OFF
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=EvaluatorExecutionMode(request.key.mode),
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


class NativeArtifactBuilder:
    """Generate and authenticate one grouped native ProcessSet artifact."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        models: Mapping[ModelName, object | None],
    ) -> None:
        self.artifact_root = artifact_root
        self.models = models
        self.artifact_root.mkdir(parents=True, exist_ok=False)

    def __call__(self, request: GroupRequest) -> ArtifactRecord:
        from pyamplicol import Generator, ProcessSet

        path = self.artifact_root / request.key.slug
        process_set = ProcessSet.from_expressions(
            (case.process for case in request.cases),
            names=tuple(case.artifact_name for case in request.cases),
        )
        Generator(_run_config(request)).generate(
            process_set,
            path,
            model=self.models[request.key.model],  # type: ignore[arg-type]
        )
        self._validate(path, request)
        return ArtifactRecord(request=request, path=path)

    @staticmethod
    def _validate(path: Path, request: GroupRequest) -> None:
        from pyamplicol.artifacts import inspect_artifact

        try:
            effective = tomllib.loads(
                (path / "config" / "effective.toml").read_text(encoding="utf-8")
            )
            observed_lane = (
                str(cast(Mapping[str, object], effective["color"])["accuracy"]),
                str(cast(Mapping[str, object], effective["color"])["contraction"]),
                str(
                    cast(Mapping[str, object], effective["evaluator"])["execution_mode"]
                ),
            )
        except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
            raise FFTAcceptanceError(
                f"artifact {request.key.slug} has no valid effective config"
            ) from error
        expected_lane = (
            request.key.accuracy,
            request.key.method,
            request.key.mode,
        )
        if observed_lane != expected_lane:
            raise FFTAcceptanceError(
                f"artifact {request.key.slug} effective lane is {observed_lane!r}; "
                f"expected {expected_lane!r}"
            )

        inspection = inspect_artifact(path)
        cases_by_name = {case.artifact_name: case for case in request.cases}
        public_names: set[str] = set()
        for process in inspection.processes:
            process_names = (process.id, *(alias.id for alias in process.aliases))
            public_names.update(process_names)
            unknown_names = tuple(
                name for name in process_names if name not in cases_by_name
            )
            if unknown_names:
                raise FFTAcceptanceError(
                    f"artifact {request.key.slug} contains unknown public process "
                    f"names {unknown_names!r}"
                )
            if process.color_accuracy != request.key.accuracy:
                raise FFTAcceptanceError(
                    f"artifact {request.key.slug} contains the wrong color accuracy"
                )
            if process.execution_mode != request.key.mode:
                raise FFTAcceptanceError(
                    f"artifact {request.key.slug} contains the wrong execution mode"
                )
            expected_methods = {
                _expected_inspection_method(request.key.method, cases_by_name[name])
                for name in process_names
            }
            if len(expected_methods) != 1:
                raise FFTAcceptanceError(
                    f"artifact {request.key.slug} aliases processes with different "
                    "symmetric-group eligibility"
                )
            expected_method = expected_methods.pop()
            method = process.recurrence_color_contraction_method
            if method != expected_method:
                raise FFTAcceptanceError(
                    f"artifact {request.key.slug} carries inspection method "
                    f"{method!r}; expected {expected_method!r} for public "
                    f"processes {process_names!r}"
                )
            if request.selected_source_helicities is None:
                if process.selected_source_helicities:
                    raise FFTAcceptanceError(
                        f"all-helicity artifact {request.key.slug} is specialized"
                    )
                if (
                    process.helicity_coverage != "complete"
                    or "helicity" in process.generation_specialized_axes
                ):
                    raise FFTAcceptanceError(
                        f"all-helicity artifact {request.key.slug} does not retain "
                        "complete reusable helicity coverage"
                    )
            elif (
                process.helicity_coverage != "selected"
                or "helicity" not in process.generation_specialized_axes
            ):
                raise FFTAcceptanceError(
                    f"selected artifact {request.key.slug} is not "
                    "generation-specialized"
                )
        expected_names = {case.artifact_name for case in request.cases}
        if public_names != expected_names:
            raise FFTAcceptanceError(
                f"artifact {request.key.slug} public process names differ: "
                f"missing={sorted(expected_names - public_names)!r}, "
                f"extra={sorted(public_names - expected_names)!r}"
            )


def _runtime_points(case: CatalogCase) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return (
        tuple(
            tuple(float(component) for component in momentum)
            for momentum in validation_momenta(case.process)
        ),
    )


def _preferred_structural_selection(
    case: CatalogCase,
    *,
    model: ModelName,
    accuracy: AccuracyName,
) -> SelectionRecord:
    """Build the existing deterministic selector without all-helicity generation."""

    from pyamplicol.models.builtin.process_ir import build_process_ir

    process = build_process_ir(case.process, color_accuracy=accuracy)
    pdgs = tuple(int(leg.pdg) for leg in process.legs if leg.pdg is not None)
    if len(pdgs) != case.n_final + 2:
        raise FFTAcceptanceError(
            f"{case.case_id} structural selector has the wrong external width"
        )
    values = fixed_selector_helicity(pdgs)
    domains = tuple(_preferred_helicities(pdg) for pdg in pdgs)
    if any(value not in domain for value, domain in zip(values, domains, strict=True)):
        raise FFTAcceptanceError(
            f"{case.case_id} structural selector is outside its physical domain"
        )
    return SelectionRecord(
        model=model,
        accuracy=accuracy,
        case=case,
        helicity_id=selector_helicity_id(values),
        values=values,
        domains=domains,
        source="preferred-structural-selector",
        discovery_value=0,
    )


def _validate_runtime_lane(runtime: object, key: ArtifactKey) -> None:
    if getattr(runtime, "execution_mode", None) != key.mode:
        raise FFTAcceptanceError(
            f"runtime {key.slug} loaded with the wrong execution mode"
        )
    physics = getattr(runtime, "physics", None)
    if getattr(physics, "color_accuracy", None) != key.accuracy:
        raise FFTAcceptanceError(f"runtime {key.slug} loaded with the wrong accuracy")


def _discover_case_selection(
    record: ArtifactRecord,
    case: CatalogCase,
) -> SelectionRecord:
    from pyamplicol import Runtime

    runtime = Runtime.load(record.path, process=case.artifact_name)
    try:
        _validate_runtime_lane(runtime, record.request.key)
        resolved = runtime.evaluate_resolved(_runtime_points(case))
        physics = runtime.physics
        helicities = tuple(physics.helicities)
        if resolved.helicity_ids != tuple(item.id for item in helicities):
            raise FFTAcceptanceError(
                f"{case.case_id} discovery resolved helicity order is inconsistent"
            )
        if resolved.color_accuracy != record.request.key.accuracy:
            raise FFTAcceptanceError(
                f"{case.case_id} discovery resolved the wrong color accuracy"
            )
        if len(resolved.values) != 1 or any(
            len(color_values) != 1 for color_values in resolved.values[0]
        ):
            raise FFTAcceptanceError(
                f"{case.case_id} discovery did not return one contracted color value"
            )
        particles = tuple(physics.external_particles)
        preferred = fixed_selector_helicity(
            tuple(int(particle.pdg_id) for particle in particles)
        )
        chosen = choose_nonzero_helicity(
            tuple(
                HelicityObservation(
                    helicity_id=helicity.helicity_id
                    if hasattr(helicity, "helicity_id")
                    else helicity.id,
                    values=tuple(int(value) for value in helicity.values),
                    structural_zero=bool(helicity.structural_zero),
                    matrix_element=resolved.values[0][index][0],
                )
                for index, helicity in enumerate(helicities)
            ),
            preferred=preferred,
        )
        return SelectionRecord(
            model=record.request.key.model,
            accuracy=record.request.key.accuracy,
            case=case,
            helicity_id=chosen.observation.helicity_id,
            values=chosen.observation.values,
            domains=chosen.domains,
            source=chosen.source,
            discovery_value=chosen.observation.matrix_element,
        )
    finally:
        runtime.clear()


def _evaluate_scalar(
    record: ArtifactRecord,
    case: CatalogCase,
    *,
    selection: SelectionRecord | None,
    active_family_census: dict[str, object] | None = None,
) -> object:
    from pyamplicol import Runtime

    runtime = Runtime.load(record.path, process=case.artifact_name)
    try:
        _validate_runtime_lane(runtime, record.request.key)
        selected_helicity_ids: tuple[str, ...] | None = None
        if selection is not None:
            if selection.case != case:
                raise FFTAcceptanceError(
                    f"{case.case_id}/{record.request.key.slug} received a selection "
                    "for a different process"
                )
            if (
                selection.model != record.request.key.model
                or selection.accuracy != record.request.key.accuracy
            ):
                raise FFTAcceptanceError(
                    f"{case.case_id}/{record.request.key.slug} received a selection "
                    "from a different model or accuracy lane"
                )
            physical = tuple(runtime.physics.helicities)
            observed = tuple(
                tuple(int(value) for value in item.values) for item in physical
            )
            if record.request.key.mode == "recurrence":
                if observed != (selection.values,):
                    raise FFTAcceptanceError(
                        f"{case.case_id}/{record.request.key.slug} exposes selected "
                        f"helicities {observed!r}; expected {(selection.values,)!r}"
                    )
                if runtime.physics.helicity_coverage != "selected":
                    raise FFTAcceptanceError(
                        f"{case.case_id}/{record.request.key.slug} is not "
                        "generation-selected"
                    )
            else:
                if (
                    record.request.selected_source_helicities is not None
                    or runtime.physics.helicity_coverage != "complete"
                ):
                    raise FFTAcceptanceError(
                        f"{case.case_id}/{record.request.key.slug} is not a "
                        "complete-coverage on-the-fly runtime"
                    )
                matches = tuple(
                    item for item in physical if tuple(item.values) == selection.values
                )
                if len(matches) != 1:
                    raise FFTAcceptanceError(
                        f"{case.case_id}/{record.request.key.slug} exposes "
                        f"{len(matches)} rows for recorded tuple {selection.values!r}"
                    )
                observed_id = str(matches[0].id)
                if observed_id != selection.helicity_id:
                    raise FFTAcceptanceError(
                        f"{case.case_id}/{record.request.key.slug} maps recorded "
                        f"tuple to {observed_id!r}; expected {selection.helicity_id!r}"
                    )
                selected_helicity_ids = (observed_id,)
        values = runtime.evaluate(
            _runtime_points(case),
            helicities=selected_helicity_ids,
        )
        if len(values) != 1:
            raise FFTAcceptanceError(
                f"{case.case_id}/{record.request.key.slug} returned {len(values)} "
                "matrix elements for one point"
            )
        if active_family_census is not None:
            if record.request.key.mode != "on-the-fly":
                raise FFTAcceptanceError(
                    "active-family census capture requires an on-the-fly runtime"
                )
            if active_family_census:
                raise FFTAcceptanceError(
                    "active-family census capture destination must start empty"
                )
            inspect = getattr(runtime, "inspect", None)
            if not callable(inspect):
                raise FFTAcceptanceError(
                    f"{case.case_id}/{record.request.key.slug} has no runtime "
                    "inspection"
                )
            inspection = _mapping(
                inspect(),
                f"{case.case_id}/{record.request.key.slug} runtime inspection",
            )
            state = _mapping(
                inspection.get("on_the_fly_state"),
                f"{case.case_id}/{record.request.key.slug} on-the-fly state",
            )
            active = _mapping(
                state.get("active_family_union_census"),
                f"{case.case_id}/{record.request.key.slug} active-family census",
            )
            active_family_census.update(active)
        return values[0]
    finally:
        runtime.clear()


class NativeAcceptanceHarness:
    """Run the complete opt-in matrix with grouped ProcessSet generation."""

    def __init__(
        self,
        run_root: Path,
        *,
        fixture: FrozenFullEvidence,
        fixture_path: Path,
    ) -> None:
        self.run_root = run_root
        self.fixture = fixture
        self.fixture_path = fixture_path
        self.selected_cases = catalog_cases(SELECTED_MAX_N_FINAL)
        self.total_cases = catalog_cases(TOTAL_MAX_N_FINAL)
        self.comparisons: list[dict[str, object]] = []
        self.selections: dict[tuple[ModelName, AccuracyName, str], SelectionRecord] = {}
        self._values: dict[
            tuple[Path, str, tuple[str, tuple[int, ...]] | None], object
        ] = {}
        self._reflection_censuses: dict[
            tuple[str, MethodName], dict[str, object]
        ] = {}

        for frozen in fixture.cases:
            if frozen.momenta != validation_momenta(frozen.case.process):
                raise FFTAcceptanceError(
                    f"frozen case {frozen.case.case_id} differs from its current "
                    "generic_validation_point(seed=101)"
                )

        models: dict[ModelName, object | None] = {
            "built-in-sm": None,
            "ufo-sm": prepare_ufo_sm_model(run_root / "prepared-ufo-sm"),
        }
        builder = NativeArtifactBuilder(run_root / "artifacts", models=models)
        self.cache: ProcessSetCache[ArtifactRecord] = ProcessSetCache(builder)

    def _discover_selections(
        self,
    ) -> tuple[
        dict[tuple[ModelName, AccuracyName, str], ArtifactRecord],
        frozenset[tuple[ModelName, AccuracyName]],
    ]:
        """Probe the cheap structural tuple and expand only observed zeros."""

        probe_records: dict[tuple[ModelName, AccuracyName, str], ArtifactRecord] = {}
        fallback_scopes: set[tuple[ModelName, AccuracyName]] = set()
        for model in MODELS:
            for accuracy in ACCURACIES:
                preferred = {
                    case.case_id: _preferred_structural_selection(
                        case,
                        model=model,
                        accuracy=accuracy,
                    )
                    for case in self.selected_cases
                }
                zero_cases: list[CatalogCase] = []
                for group in group_selected_cases(self.selected_cases, preferred):
                    request = GroupRequest(
                        ArtifactKey(
                            "direct",
                            model,
                            "recurrence",
                            accuracy,
                            f"preferred-probe-{_signature_digest(group.signature)}",
                        ),
                        group.cases,
                        selected_source_helicities=group.source_mapping,
                        external_helicity_domains=group.signature[0],
                    )
                    record = self.cache.get(request)
                    for case in group.cases:
                        candidate = preferred[case.case_id]
                        value = self._value(record, case, selection=candidate)
                        probe_records[(model, accuracy, case.case_id)] = record
                        if strict_relative_compare(value, 0).scale == 0:
                            zero_cases.append(case)
                            continue
                        self.selections[(model, accuracy, case.case_id)] = (
                            SelectionRecord(
                                model=model,
                                accuracy=accuracy,
                                case=case,
                                helicity_id=candidate.helicity_id,
                                values=candidate.values,
                                domains=candidate.domains,
                                source="preferred-structural-selector",
                                discovery_value=value,
                            )
                        )
                if not zero_cases:
                    continue
                fallback_scopes.add((model, accuracy))
                fallback_request = GroupRequest(
                    ArtifactKey(
                        "direct",
                        model,
                        "recurrence",
                        accuracy,
                        "all-helicity-fallback-discovery",
                    ),
                    tuple(zero_cases),
                )
                fallback_record = self.cache.get(fallback_request)
                for case in zero_cases:
                    selection = _discover_case_selection(fallback_record, case)
                    if selection.values == preferred[case.case_id].values:
                        raise FFTAcceptanceError(
                            f"{case.case_id} fallback discovery retained the zero "
                            "preferred tuple"
                        )
                    self.selections[(model, accuracy, case.case_id)] = selection
        return probe_records, frozenset(fallback_scopes)

    def _selected_artifacts(
        self,
        probe_records: Mapping[tuple[ModelName, AccuracyName, str], ArtifactRecord],
        fallback_scopes: frozenset[tuple[ModelName, AccuracyName]],
    ) -> dict[
        tuple[ModelName, ModeName, AccuracyName, MethodName, str], ArtifactRecord
    ]:
        records: dict[
            tuple[ModelName, ModeName, AccuracyName, MethodName, str], ArtifactRecord
        ] = {}
        for model in MODELS:
            for accuracy in ACCURACIES:
                selections = {
                    case.case_id: self.selections[(model, accuracy, case.case_id)]
                    for case in self.selected_cases
                }
                groups = group_selected_cases(self.selected_cases, selections)
                for method in METHODS:
                    for group in groups:
                        if (
                            method == "direct"
                            and (model, accuracy) not in fallback_scopes
                        ):
                            for case in group.cases:
                                records[
                                    (
                                        model,
                                        "recurrence",
                                        accuracy,
                                        method,
                                        case.case_id,
                                    )
                                ] = probe_records[(model, accuracy, case.case_id)]
                            continue
                        request = GroupRequest(
                            ArtifactKey(
                                method,
                                model,
                                "recurrence",
                                accuracy,
                                group.scope,
                            ),
                            group.cases,
                            selected_source_helicities=group.source_mapping,
                            external_helicity_domains=group.signature[0],
                        )
                        record = self.cache.get(request)
                        for case in group.cases:
                            records[
                                (
                                    model,
                                    "recurrence",
                                    accuracy,
                                    method,
                                    case.case_id,
                                )
                            ] = record

                    otf_record = self.cache.get(
                        otf_complete_group_request(
                            method=method,
                            model=model,
                            accuracy=accuracy,
                            cases=self.selected_cases,
                        )
                    )
                    for case in self.selected_cases:
                        records[
                            (
                                model,
                                "on-the-fly",
                                accuracy,
                                method,
                                case.case_id,
                            )
                        ] = otf_record
        return records

    def _total_artifacts(
        self,
        selected: Mapping[
            tuple[ModelName, ModeName, AccuracyName, MethodName, str],
            ArtifactRecord,
        ],
    ) -> dict[tuple[ModelName, ModeName, MethodName], ArtifactRecord]:
        records: dict[tuple[ModelName, ModeName, MethodName], ArtifactRecord] = {}
        for model in MODELS:
            for mode in MODES:
                for method in METHODS:
                    if mode == "on-the-fly":
                        otf_records = tuple(
                            selected[(model, mode, "full", method, case.case_id)]
                            for case in self.selected_cases
                        )
                        if len({record.path for record in otf_records}) != 1:
                            raise FFTAcceptanceError(
                                "on-the-fly selected queries do not share one "
                                "complete-coverage ProcessSet"
                            )
                        record = otf_records[0]
                        if (
                            record.request.selected_source_helicities is not None
                            or not {case.case_id for case in self.total_cases}
                            <= {case.case_id for case in record.request.cases}
                        ):
                            raise FFTAcceptanceError(
                                "on-the-fly FullColour totals cannot reuse the "
                                "complete selected-query ProcessSet"
                            )
                        records[(model, mode, method)] = record
                        continue
                    request = GroupRequest(
                        ArtifactKey(
                            method,
                            model,
                            mode,
                            "full",
                            "all-helicity-total",
                        ),
                        self.total_cases,
                    )
                    records[(model, mode, method)] = self.cache.get(request)
        return records

    def _value(
        self,
        record: ArtifactRecord,
        case: CatalogCase,
        *,
        selection: SelectionRecord | None,
    ) -> object:
        selection_key = (
            None if selection is None else (selection.helicity_id, selection.values)
        )
        key = (record.path, case.case_id, selection_key)
        existing = self._values.get(key)
        if existing is not None:
            return existing
        artifact_key = record.request.key
        census_key = (
            (case.case_id, artifact_key.method)
            if (
                selection is not None
                and case.case_id
                in {
                    _PURE_ADJOINT_REFLECTION_CASE_ID,
                    _OPEN_LINE_REFLECTION_CASE_ID,
                }
                and artifact_key.model == "built-in-sm"
                and artifact_key.mode == "on-the-fly"
                and artifact_key.accuracy == "full"
                and artifact_key.helicity_scope
                == "complete-helicity-runtime-query-and-total"
                and record.request.selected_source_helicities is None
            )
            else None
        )
        active_family_census: dict[str, object] | None = (
            {} if census_key is not None else None
        )
        if active_family_census is None:
            value = _evaluate_scalar(record, case, selection=selection)
        else:
            value = _evaluate_scalar(
                record,
                case,
                selection=selection,
                active_family_census=active_family_census,
            )
        if census_key is not None:
            if active_family_census is None:
                raise FFTAcceptanceError(
                    f"production census capture was not initialized for {census_key!r}"
                )
            previous = self._reflection_censuses.setdefault(
                census_key,
                active_family_census,
            )
            if previous != active_family_census:
                raise FFTAcceptanceError(
                    f"production active-family census changed for {census_key!r}"
                )
        self._values[key] = value
        return value

    def _validated_reflection_census(self) -> dict[str, object]:
        def required(case_id: str, method: MethodName) -> Mapping[str, object]:
            try:
                return self._reflection_censuses[(case_id, method)]
            except KeyError as error:
                raise FFTAcceptanceError(
                    "the 574-comparison campaign did not evaluate the required "
                    f"built-in-SM/full/on-the-fly selected census {case_id}/{method}"
                ) from error

        return validate_on_the_fly_reflection_census(
            pure_adjoint_direct=required(
                _PURE_ADJOINT_REFLECTION_CASE_ID,
                "direct",
            ),
            pure_adjoint_fft=required(
                _PURE_ADJOINT_REFLECTION_CASE_ID,
                "symmetric-group-fft",
            ),
            open_line_direct=required(
                _OPEN_LINE_REFLECTION_CASE_ID,
                "direct",
            ),
            open_line_fft=required(
                _OPEN_LINE_REFLECTION_CASE_ID,
                "symmetric-group-fft",
            ),
        )

    def _record_direct_fft(
        self,
        *,
        model: ModelName,
        mode: ModeName,
        accuracy: AccuracyName,
        scope: Literal["selected", "total"],
        case: CatalogCase,
        direct_record: ArtifactRecord,
        fft_record: ArtifactRecord,
        selection: SelectionRecord | None,
    ) -> None:
        direct = self._value(direct_record, case, selection=selection)
        fft = self._value(fft_record, case, selection=selection)
        comparison = strict_relative_compare(fft, direct)
        direct_nonzero = (
            strict_relative_compare(direct, 0).scale > 0
            if selection is not None
            else None
        )
        self.comparisons.append(
            {
                "authority": "direct-vs-fft",
                "model": model,
                "mode": mode,
                "accuracy": accuracy,
                "helicity_scope": scope,
                "case_id": case.case_id,
                "n_final": case.n_final,
                "selection": (
                    None
                    if selection is None
                    else {
                        "helicity_id": selection.helicity_id,
                        "source_helicities": [
                            list(item) for item in selection.source_mapping
                        ],
                        "execution": selection_execution(mode),
                        "artifact_helicity_coverage": (
                            "selected" if mode == "recurrence" else "complete"
                        ),
                        "direct_nonzero": direct_nonzero,
                    }
                ),
                "comparison": comparison.as_payload(),
                "passed": comparison.passed and direct_nonzero is not False,
            }
        )

    def _record_frozen(
        self,
        *,
        mode: ModeName,
        case: CatalogCase,
        fft_record: ArtifactRecord,
    ) -> None:
        frozen = self.fixture.case(case.case_id)
        candidate = self._value(fft_record, case, selection=None)
        comparison = strict_relative_compare(candidate, frozen.expected_full)
        self.comparisons.append(
            {
                "authority": "frozen-madgraph",
                "model": "ufo-sm",
                "mode": mode,
                "accuracy": "full",
                "helicity_scope": "total",
                "case_id": case.case_id,
                "n_final": case.n_final,
                "fixture": str(self.fixture_path),
                "comparison": comparison.as_payload(),
                "passed": comparison.passed,
            }
        )

    def run(self) -> dict[str, object]:
        probe_records, fallback_scopes = self._discover_selections()
        selected = self._selected_artifacts(probe_records, fallback_scopes)
        totals = self._total_artifacts(selected)

        for model in MODELS:
            for mode in MODES:
                for accuracy in ACCURACIES:
                    for case in self.selected_cases:
                        selection = self.selections[(model, accuracy, case.case_id)]
                        self._record_direct_fft(
                            model=model,
                            mode=mode,
                            accuracy=accuracy,
                            scope="selected",
                            case=case,
                            direct_record=selected[
                                (model, mode, accuracy, "direct", case.case_id)
                            ],
                            fft_record=selected[
                                (
                                    model,
                                    mode,
                                    accuracy,
                                    "symmetric-group-fft",
                                    case.case_id,
                                )
                            ],
                            selection=selection,
                        )
                for case in self.total_cases:
                    self._record_direct_fft(
                        model=model,
                        mode=mode,
                        accuracy="full",
                        scope="total",
                        case=case,
                        direct_record=totals[(model, mode, "direct")],
                        fft_record=totals[(model, mode, "symmetric-group-fft")],
                        selection=None,
                    )

        for mode in MODES:
            for case in self.total_cases:
                self._record_frozen(
                    mode=mode,
                    case=case,
                    fft_record=totals[("ufo-sm", mode, "symmetric-group-fft")],
                )

        expected = comparison_specs()
        if len(self.comparisons) != len(expected):
            raise FFTAcceptanceError(
                f"campaign produced {len(self.comparisons)} comparisons; "
                f"expected {len(expected)}"
            )
        reflection_census = self._validated_reflection_census()
        accepted = all(bool(item["passed"]) for item in self.comparisons)
        cache_payload: list[dict[str, object]] = []
        for request, record in self.cache.entries:
            entry = request.as_payload()
            entry["artifact"] = str(record.path.relative_to(self.run_root))
            entry["generation_count"] = 1
            cache_payload.append(entry)
        return {
            "kind": GATE_KIND,
            "schema_version": GATE_SCHEMA_VERSION,
            "accepted": accepted,
            "point_policy": {
                "generator": "generic_validation_point",
                "seed": POINT_SEED,
            },
            "comparison_policy": {
                "relative_tolerance": canonical_decimal(RELATIVE_TOLERANCE),
                "absolute_tolerance": None,
            },
            "on_the_fly_reflection_census": reflection_census,
            "catalog": {
                "source": "tools/performance_report/catalog.py:PROCESS_FAMILIES",
                "selected_max_n_final": SELECTED_MAX_N_FINAL,
                "selected_case_count_per_model": len(self.selected_cases),
                "total_max_n_final": TOTAL_MAX_N_FINAL,
                "total_case_count_per_model": len(self.total_cases),
            },
            "frozen_authority": {
                "fixture": str(self.fixture_path),
                "madgraph_version": self.fixture.madgraph_version,
                "captured_model_source_sha256": (self.fixture.model_source_sha256),
                "captured_driver_sha256": self.fixture.driver_sha256,
                "madgraph_rerun": False,
            },
            "process_set_cache": {
                "generation_count": self.cache.generation_count,
                "per_process_generation": False,
                "selected_execution": {
                    "recurrence": "generation-specialized",
                    "on-the-fly": "runtime-query-complete-coverage",
                },
                "on_the_fly_total_reuses_complete_selected_query_artifact": True,
                "entries": cache_payload,
            },
            "selections": [
                self.selections[key].as_payload() for key in sorted(self.selections)
            ],
            "comparisons": self.comparisons,
            "comparison_counts": {
                "direct_vs_fft": sum(
                    item["authority"] == "direct-vs-fft" for item in self.comparisons
                ),
                "frozen_madgraph": sum(
                    item["authority"] == "frozen-madgraph" for item in self.comparisons
                ),
                "total": len(self.comparisons),
            },
        }


def _fresh_run_root(output_root: Path) -> Path:
    resolved = output_root.expanduser().resolve(strict=False)
    if not resolved.is_relative_to(ROOT):
        raise FFTAcceptanceError("acceptance output must remain inside the workspace")
    resolved.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = resolved / f"run-{stamp}-{uuid.uuid4().hex[:10]}"
    run_root.mkdir()
    return run_root


def _write_report(payload: Mapping[str, object], path: Path) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(encoded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="tracked UFO-SM seed-101 numerical fixture",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".artifacts" / "fft-numerical-acceptance",
        help="workspace-local parent for one fresh campaign directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="authenticate inputs and print the comparison plan without writes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fixture_path = args.fixture.expanduser().resolve(strict=True)
    if args.dry_run:
        print(
            json.dumps(
                dry_run_payload(fixture_path),
                ensure_ascii=True,
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    fixture = load_frozen_full_evidence(fixture_path)
    if _fixture_case_ids(fixture) != tuple(
        case.case_id for case in catalog_cases(TOTAL_MAX_N_FINAL)
    ):
        raise FFTAcceptanceError(
            "the frozen UFO-SM fixture no longer matches the live n_final<=4 catalog"
        )
    run_root = _fresh_run_root(args.output_root)
    payload = NativeAcceptanceHarness(
        run_root,
        fixture=fixture,
        fixture_path=fixture_path,
    ).run()
    report = run_root / "report.json"
    _write_report(payload, report)
    print(report)
    if not payload["accepted"]:
        raise FFTAcceptanceMismatch(
            f"FFT numerical acceptance failed; complete report: {report}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
