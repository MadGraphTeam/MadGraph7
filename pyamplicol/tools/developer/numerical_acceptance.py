# SPDX-License-Identifier: 0BSD
"""Durable UFO-SM numerical acceptance capture and replay.

The fixture written by this module deliberately contains one deterministic
point and one fully summed matrix element per process.  Full-colour values are
independent MadGraph standalone results.  LC and NLC values are high-precision
recurrence regression references captured only after the full-colour gate has
passed.

Normal unit tests exercise the wire contract with synthetic values.  Real
generation is entered either through :class:`NumericalAcceptanceHarness` or
the ``capture`` command and is intentionally not part of the default test run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Literal, cast

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from tools.developer.reference_capture.common import canonical_decimal  # noqa: E402
from tools.performance_report.catalog import PROCESS_FAMILIES  # noqa: E402

MODEL_ROOT = ROOT / "src" / "pyamplicol" / "assets" / "models" / "json" / "sm"
MODEL_SOURCE = MODEL_ROOT / "sm.json"
MODEL_RESTRICTION = MODEL_ROOT / "restrict_default.json"
DEFAULT_FIXTURE = (
    ROOT / "tests" / "fixtures" / "numerical_acceptance" / "ufo-sm-seed101-v1.json"
)

FIXTURE_KIND = "pyamplicol-ufo-sm-numerical-acceptance"
FIXTURE_SCHEMA_VERSION = 1
CATALOG_MAX_N_FINAL = 4
CATALOG_CASE_COUNT = 33
POINT_SEED = 101
RELATIVE_TOLERANCE = Decimal("0.0000000001")
FULL_CANDIDATE_PRECISION_DIGITS = 200
NATIVE_PRECISION_DIGITS = 16

AccuracyName = Literal["full", "lc", "nlc"]
ModeName = Literal["recurrence", "eager", "compiled", "on-the-fly"]
LCFlowLayoutName = Literal["topology-replay", "all-flow-union"]
ArtifactGroup = Literal["catalog", "extra"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class NumericalAcceptanceError(RuntimeError):
    """The acceptance fixture or capture contract is invalid."""


class NumericalAcceptanceMismatch(AssertionError):
    """A matrix element failed the strict scale-conditioned comparison."""


@dataclass(frozen=True, slots=True)
class AcceptanceCaseSpec:
    """One catalog or deliberately separate full-colour stress process."""

    case_id: str
    process: str
    n_final: int
    family_id: int | None = None
    family_key: str | None = None

    @property
    def is_catalog(self) -> bool:
        return self.family_id is not None

    @property
    def artifact_name(self) -> str:
        """Return a stable public ProcessSet name accepted by pyAmpliCol."""

        return self.case_id.replace(":", "_").replace("-", "_")


def catalog_cases() -> tuple[AcceptanceCaseSpec, ...]:
    """Derive the complete 33-process n=1..4 catalog surface."""

    cases: list[AcceptanceCaseSpec] = []
    for n_final in range(1, CATALOG_MAX_N_FINAL + 1):
        for family in sorted(PROCESS_FAMILIES, key=lambda item: item.identifier):
            process = family.process(n_final)
            if process is None:
                continue
            cases.append(
                AcceptanceCaseSpec(
                    case_id=f"catalog:{family.key}:n{n_final}",
                    process=process,
                    n_final=n_final,
                    family_id=family.identifier,
                    family_key=family.key,
                )
            )
    result = tuple(cases)
    if len(result) != CATALOG_CASE_COUNT:
        raise NumericalAcceptanceError(
            "the n=1..4 performance catalog no longer contains exactly "
            f"{CATALOG_CASE_COUNT} processes (observed {len(result)})"
        )
    return result


EXTRA_FULL_COLOUR_CASES = (
    AcceptanceCaseSpec(
        case_id="extra:identical-u-four-lines",
        process="u u~ > u u~ u u~ u u~",
        n_final=6,
    ),
    AcceptanceCaseSpec(
        case_id="extra:identical-electron-three-pair",
        process="e+ e- > e+ e- e+ e- e+ e-",
        n_final=6,
    ),
)


@dataclass(frozen=True, slots=True)
class AcceptanceLane:
    """One native generation/evaluation lane in the acceptance matrix."""

    accuracy: AccuracyName
    mode: ModeName
    jit_optimization_level: int
    lc_flow_layout: LCFlowLayoutName = "topology-replay"

    def __post_init__(self) -> None:
        if self.lc_flow_layout == "all-flow-union" and self.accuracy != "lc":
            raise ValueError("all-flow-union acceptance is available only for LC")
        if self.lc_flow_layout == "all-flow-union" and self.mode == "on-the-fly":
            raise ValueError(
                "on-the-fly acceptance uses its compact query-local LC artifact"
            )

    @property
    def lane_id(self) -> str:
        if self.lc_flow_layout == "all-flow-union":
            return f"lc-all-flow-union-{self.mode}"
        return f"{self.accuracy}-{self.mode}"

    @property
    def includes_extra_cases(self) -> bool:
        return self.accuracy == "full"

    @property
    def evaluation_precisions(self) -> tuple[int, ...]:
        """Return the supported acceptance precisions for this lane."""

        if self.accuracy == "full" and self.mode != "on-the-fly":
            return (NATIVE_PRECISION_DIGITS, FULL_CANDIDATE_PRECISION_DIGITS)
        return (NATIVE_PRECISION_DIGITS,)

    def evaluation_precisions_for(
        self,
        group: ArtifactGroup,
    ) -> tuple[int, ...]:
        """Return the minimal precision replay for one artifact group."""

        if group == "extra":
            return (NATIVE_PRECISION_DIGITS,)
        return self.evaluation_precisions


ACCEPTANCE_LANES = (
    AcceptanceLane("full", "recurrence", 2),
    AcceptanceLane("full", "eager", 2),
    AcceptanceLane("full", "compiled", 3),
    AcceptanceLane("full", "on-the-fly", 2),
    AcceptanceLane("lc", "recurrence", 2),
    AcceptanceLane("lc", "eager", 2),
    AcceptanceLane("lc", "compiled", 3),
    AcceptanceLane("lc", "on-the-fly", 2),
    AcceptanceLane("lc", "recurrence", 2, "all-flow-union"),
    AcceptanceLane("lc", "eager", 2, "all-flow-union"),
    AcceptanceLane("lc", "compiled", 3, "all-flow-union"),
    AcceptanceLane("nlc", "recurrence", 2),
    AcceptanceLane("nlc", "eager", 2),
    AcceptanceLane("nlc", "compiled", 3),
    AcceptanceLane("nlc", "on-the-fly", 2),
)
FULL_LANES = tuple(lane for lane in ACCEPTANCE_LANES if lane.accuracy == "full")
LC_ALL_FLOW_UNION_LANES = tuple(
    lane for lane in ACCEPTANCE_LANES if lane.lc_flow_layout == "all-flow-union"
)


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    source_sha256: str
    restriction_sha256: str
    external_parameters_sha256: str

    def as_payload(self) -> dict[str, object]:
        return {
            "kind": "ufo-sm-json-default-restriction",
            "source_sha256": self.source_sha256,
            "restriction_sha256": self.restriction_sha256,
            "external_parameters_sha256": self.external_parameters_sha256,
        }


@dataclass(frozen=True, slots=True)
class MadGraphReferenceIdentity:
    madgraph_version: str
    model_source_sha256: str
    driver_sha256: str
    external_parameters_sha256: str

    def as_payload(self) -> dict[str, object]:
        return {
            "kind": "madgraph-standalone-ufo-sm",
            "precision": "binary64",
            "command_protocol": "generate-output-standalone-launch-force",
            "madgraph_version": self.madgraph_version,
            "model_source_sha256": self.model_source_sha256,
            "driver_sha256": self.driver_sha256,
            "external_parameters_sha256": self.external_parameters_sha256,
        }


@dataclass(frozen=True, slots=True)
class AcceptanceCase:
    spec: AcceptanceCaseSpec
    momenta: tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]
    expected: tuple[tuple[AccuracyName, Decimal], ...]

    def expected_for(self, accuracy: AccuracyName) -> Decimal:
        for key, value in self.expected:
            if key == accuracy:
                return value
        raise NumericalAcceptanceError(
            f"case {self.spec.case_id!r} has no {accuracy} reference"
        )

    def runtime_momenta(self) -> tuple[tuple[tuple[float, ...], ...], ...]:
        return (
            tuple(
                tuple(float(component) for component in momentum)
                for momentum in self.momenta
            ),
        )


@dataclass(frozen=True, slots=True)
class AcceptanceFixture:
    model: ModelIdentity
    full_reference: MadGraphReferenceIdentity
    captured_source_revision: str
    catalog: tuple[AcceptanceCase, ...]
    extra_full_colour: tuple[AcceptanceCase, ...]

    def case(self, case_id: str) -> AcceptanceCase:
        for case in (*self.catalog, *self.extra_full_colour):
            if case.spec.case_id == case_id:
                return case
        raise KeyError(case_id)


@dataclass(frozen=True, slots=True)
class RelativeComparison:
    actual_real: Decimal
    actual_imaginary: Decimal
    expected: Decimal
    difference: Decimal
    scale: Decimal
    bound: Decimal
    passed: bool


@dataclass(frozen=True, slots=True)
class MadGraphIngest:
    values: Mapping[str, Decimal]
    identity: MadGraphReferenceIdentity


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_model_identity() -> ModelIdentity:
    """Return the material identity shared by pyAmpliCol and MadGraph."""

    try:
        model = json.loads(MODEL_SOURCE.read_text(encoding="utf-8"))
        restriction = json.loads(MODEL_RESTRICTION.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NumericalAcceptanceError(
            "cannot read the packaged UFO-SM model"
        ) from error
    if not isinstance(model, Mapping) or not isinstance(restriction, Mapping):
        raise NumericalAcceptanceError("the packaged UFO-SM model is malformed")
    parameters = model.get("parameters")
    if not isinstance(parameters, list):
        raise NumericalAcceptanceError("the packaged UFO-SM model has no parameters")
    externals: dict[str, list[float]] = {}
    for raw_parameter in parameters:
        if not isinstance(raw_parameter, Mapping):
            raise NumericalAcceptanceError("the UFO-SM parameter list is malformed")
        if raw_parameter.get("nature") != "external":
            continue
        name = raw_parameter.get("name")
        restricted = restriction.get(name) if isinstance(name, str) else None
        if (
            not isinstance(name, str)
            or not isinstance(restricted, list)
            or len(restricted) != 2
            or isinstance(restricted[0], bool)
            or not isinstance(restricted[0], (int, float))
            or isinstance(restricted[1], bool)
            or not isinstance(restricted[1], (int, float))
            or float(restricted[1]) != 0.0
        ):
            raise NumericalAcceptanceError(
                "the UFO-SM default restriction does not bind every external "
                "parameter to a real value"
            )
        if name in externals:
            raise NumericalAcceptanceError(f"duplicate UFO-SM external {name!r}")
        externals[name] = [float(restricted[0]), 0.0]
    if not externals or set(externals) != set(restriction):
        raise NumericalAcceptanceError(
            "the UFO-SM default restriction and external parameter set differ"
        )
    canonical = json.dumps(
        externals,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return ModelIdentity(
        source_sha256=_sha256_file(MODEL_SOURCE),
        restriction_sha256=_sha256_file(MODEL_RESTRICTION),
        external_parameters_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def validation_momenta(
    process: str,
) -> tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]:
    """Materialize the exact decimal spelling of the seed-101 binary64 point."""

    from pyamplicol.models.builtin.validation import generic_validation_point

    point = generic_validation_point(process, seed=POINT_SEED)
    rows: list[tuple[Decimal, Decimal, Decimal, Decimal]] = []
    for particle in point:
        values = tuple(
            Decimal(canonical_decimal(float(component)))
            for component in particle.momentum
        )
        if len(values) != 4:
            raise NumericalAcceptanceError(
                f"validation point for {process!r} has a malformed momentum"
            )
        rows.append(cast(tuple[Decimal, Decimal, Decimal, Decimal], values))
    return tuple(rows)


def _momenta_payload(
    momenta: tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...],
) -> list[list[str]]:
    return [
        [canonical_decimal(component) for component in momentum] for momentum in momenta
    ]


def _decimal_from_number(value: object, where: str) -> Decimal:
    if isinstance(value, bool):
        raise NumericalAcceptanceError(f"{where} must be a finite real number")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise NumericalAcceptanceError(f"{where} must be finite")
        result = Decimal(canonical_decimal(value))
    elif isinstance(value, str):
        try:
            result = Decimal(value)
        except InvalidOperation as error:
            raise NumericalAcceptanceError(f"{where} is not a decimal") from error
    else:
        raise NumericalAcceptanceError(f"{where} must be a finite real number")
    if not result.is_finite():
        raise NumericalAcceptanceError(f"{where} must be finite")
    return result


def _complex_decimal(value: object) -> tuple[Decimal, Decimal]:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise NumericalAcceptanceError("matrix element is not finite")
        return value, Decimal(0)
    if isinstance(value, bool):
        raise NumericalAcceptanceError("matrix element must be numeric")
    try:
        number = complex(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise NumericalAcceptanceError("matrix element must be numeric") from error
    if not math.isfinite(number.real) or not math.isfinite(number.imag):
        raise NumericalAcceptanceError("matrix element is not finite")
    return (
        Decimal(canonical_decimal(number.real)),
        Decimal(canonical_decimal(number.imag)),
    )


def compare_relative(
    actual: object,
    expected: Decimal,
    *,
    tolerance: Decimal = RELATIVE_TOLERANCE,
) -> RelativeComparison:
    """Compare a real reference without an absolute-tolerance escape hatch."""

    expected = _decimal_from_number(expected, "expected matrix element")
    tolerance = _decimal_from_number(tolerance, "relative tolerance")
    if tolerance < 0:
        raise NumericalAcceptanceError("relative tolerance must be non-negative")
    actual_real, actual_imaginary = _complex_decimal(actual)
    precision = max(
        80,
        len(actual_real.as_tuple().digits) + 20,
        len(actual_imaginary.as_tuple().digits) + 20,
        len(expected.as_tuple().digits) + 20,
    )
    with localcontext() as context:
        context.prec = precision
        actual_magnitude = (
            actual_real * actual_real + actual_imaginary * actual_imaginary
        ).sqrt()
        delta_real = actual_real - expected
        difference = (
            delta_real * delta_real + actual_imaginary * actual_imaginary
        ).sqrt()
        scale = max(actual_magnitude, abs(expected))
        bound = tolerance * scale
    passed = difference == 0 if scale == 0 else difference <= bound
    return RelativeComparison(
        actual_real=actual_real,
        actual_imaginary=actual_imaginary,
        expected=expected,
        difference=difference,
        scale=scale,
        bound=bound,
        passed=passed,
    )


def assert_relative_match(
    actual: object,
    expected: Decimal,
    *,
    context: str,
    tolerance: Decimal = RELATIVE_TOLERANCE,
) -> None:
    comparison = compare_relative(actual, expected, tolerance=tolerance)
    if comparison.passed:
        return
    raise NumericalAcceptanceMismatch(
        f"{context}: actual={comparison.actual_real}"
        f"{comparison.actual_imaginary:+}i, expected={comparison.expected}, "
        f"difference={comparison.difference}, scale={comparison.scale}, "
        f"strict relative bound={comparison.bound}"
    )


def _mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NumericalAcceptanceError(f"{where} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, where: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise NumericalAcceptanceError(f"{where} must be an array")
    return cast(Sequence[object], value)


def _exact_keys(
    value: Mapping[str, object], expected: Iterable[str], where: str
) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        raise NumericalAcceptanceError(
            f"{where} fields differ: missing={sorted(expected_set - observed)}, "
            f"extra={sorted(observed - expected_set)}"
        )


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise NumericalAcceptanceError(f"{where} must be a non-empty string")
    return value


def _integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NumericalAcceptanceError(f"{where} must be an integer")
    return value


def _sha256(value: object, where: str) -> str:
    text = _string(value, where)
    if _SHA256_RE.fullmatch(text) is None:
        raise NumericalAcceptanceError(f"{where} must be a lowercase SHA-256")
    return text


def _revision(value: object, where: str) -> str:
    text = _string(value, where)
    if _REVISION_RE.fullmatch(text) is None or text == "0" * 40:
        raise NumericalAcceptanceError(f"{where} must be a real lowercase Git SHA")
    return text


def _canonical_decimal(value: object, where: str) -> Decimal:
    if not isinstance(value, str):
        raise NumericalAcceptanceError(
            f"{where} must be a canonical fixed-point decimal string"
        )
    try:
        parsed = Decimal(value)
        normalized = canonical_decimal(parsed)
    except (InvalidOperation, RuntimeError) as error:
        raise NumericalAcceptanceError(f"{where} is not a finite decimal") from error
    if not parsed.is_finite() or normalized != value:
        raise NumericalAcceptanceError(
            f"{where} must be a canonical fixed-point decimal string"
        )
    return parsed


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NumericalAcceptanceError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                NumericalAcceptanceError(f"non-finite JSON token {token!r}")
            ),
        )
    except NumericalAcceptanceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NumericalAcceptanceError(
            f"cannot read JSON document {path}: {error}"
        ) from error


def _parse_model(value: object) -> ModelIdentity:
    record = _mapping(value, "model")
    _exact_keys(
        record,
        {
            "kind",
            "source_sha256",
            "restriction_sha256",
            "external_parameters_sha256",
        },
        "model",
    )
    if record.get("kind") != "ufo-sm-json-default-restriction":
        raise NumericalAcceptanceError("model.kind is unsupported")
    return ModelIdentity(
        source_sha256=_sha256(record["source_sha256"], "model.source_sha256"),
        restriction_sha256=_sha256(
            record["restriction_sha256"], "model.restriction_sha256"
        ),
        external_parameters_sha256=_sha256(
            record["external_parameters_sha256"],
            "model.external_parameters_sha256",
        ),
    )


def _parse_full_reference(value: object) -> MadGraphReferenceIdentity:
    from tools.performance_report.madgraph import MADGRAPH_DRIVER_SOURCE_SHA256

    record = _mapping(value, "references.full")
    _exact_keys(
        record,
        {
            "kind",
            "precision",
            "command_protocol",
            "madgraph_version",
            "model_source_sha256",
            "driver_sha256",
            "external_parameters_sha256",
        },
        "references.full",
    )
    fixed = {
        "kind": "madgraph-standalone-ufo-sm",
        "precision": "binary64",
        "command_protocol": "generate-output-standalone-launch-force",
    }
    for key, expected in fixed.items():
        if record.get(key) != expected:
            raise NumericalAcceptanceError(
                f"references.full.{key} must equal {expected!r}"
            )
    driver_sha256 = _sha256(record["driver_sha256"], "references.full.driver_sha256")
    if driver_sha256 != MADGRAPH_DRIVER_SOURCE_SHA256:
        raise NumericalAcceptanceError(
            "references.full.driver_sha256 differs from the canonical adapter driver"
        )
    return MadGraphReferenceIdentity(
        madgraph_version=_string(
            record["madgraph_version"], "references.full.madgraph_version"
        ),
        model_source_sha256=_sha256(
            record["model_source_sha256"],
            "references.full.model_source_sha256",
        ),
        driver_sha256=driver_sha256,
        external_parameters_sha256=_sha256(
            record["external_parameters_sha256"],
            "references.full.external_parameters_sha256",
        ),
    )


def _parse_regression_reference(value: object, accuracy: str) -> str:
    where = f"references.{accuracy}"
    record = _mapping(value, where)
    _exact_keys(
        record,
        {"kind", "precision_digits", "captured_source_revision"},
        where,
    )
    if record.get("kind") != "pyamplicol-recurrence-regression":
        raise NumericalAcceptanceError(f"{where}.kind is unsupported")
    if _integer(record.get("precision_digits"), f"{where}.precision_digits") != (
        FULL_CANDIDATE_PRECISION_DIGITS
    ):
        raise NumericalAcceptanceError(
            f"{where}.precision_digits must be {FULL_CANDIDATE_PRECISION_DIGITS}"
        )
    return _revision(
        record.get("captured_source_revision"),
        f"{where}.captured_source_revision",
    )


def _parse_momenta(
    value: object, where: str
) -> tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]:
    rows: list[tuple[Decimal, Decimal, Decimal, Decimal]] = []
    for row_index, raw_row in enumerate(_sequence(value, where)):
        components = _sequence(raw_row, f"{where}[{row_index}]")
        if len(components) != 4:
            raise NumericalAcceptanceError(
                f"{where}[{row_index}] must contain four components"
            )
        parsed = tuple(
            _canonical_decimal(component, f"{where}[{row_index}][{index}]")
            for index, component in enumerate(components)
        )
        rows.append(cast(tuple[Decimal, Decimal, Decimal, Decimal], parsed))
    if not rows:
        raise NumericalAcceptanceError(f"{where} must not be empty")
    return tuple(rows)


def _parse_case(
    raw: object,
    spec: AcceptanceCaseSpec,
    *,
    catalog: bool,
) -> AcceptanceCase:
    where = f"case {spec.case_id}"
    record = _mapping(raw, where)
    catalog_fields = {
        "id",
        "family_id",
        "family_key",
        "n_final",
        "process",
        "momenta",
        "expected",
    }
    extra_fields = {"id", "process", "momenta", "expected"}
    _exact_keys(record, catalog_fields if catalog else extra_fields, where)
    if record.get("id") != spec.case_id or record.get("process") != spec.process:
        raise NumericalAcceptanceError(f"{where} identity differs from the catalog")
    if catalog and (
        _integer(record.get("family_id"), f"{where}.family_id") != spec.family_id
        or record.get("family_key") != spec.family_key
        or _integer(record.get("n_final"), f"{where}.n_final") != spec.n_final
    ):
        raise NumericalAcceptanceError(f"{where} metadata differs from the catalog")
    momenta = _parse_momenta(record.get("momenta"), f"{where}.momenta")
    expected_momenta = validation_momenta(spec.process)
    if momenta != expected_momenta:
        raise NumericalAcceptanceError(
            f"{where}.momenta differ from generic_validation_point(seed={POINT_SEED})"
        )
    expected_record = _mapping(record.get("expected"), f"{where}.expected")
    expected_accuracies: tuple[AccuracyName, ...] = (
        ("full", "lc", "nlc") if catalog else ("full",)
    )
    _exact_keys(expected_record, expected_accuracies, f"{where}.expected")
    expected: list[tuple[AccuracyName, Decimal]] = []
    for accuracy in expected_accuracies:
        parsed = _canonical_decimal(
            expected_record.get(accuracy), f"{where}.expected.{accuracy}"
        )
        if parsed < 0:
            raise NumericalAcceptanceError(
                f"{where}.expected.{accuracy} must be non-negative"
            )
        expected.append((accuracy, parsed))
    return AcceptanceCase(spec=spec, momenta=momenta, expected=tuple(expected))


def parse_acceptance_fixture(
    payload: object,
    *,
    validate_current_model: bool = True,
) -> AcceptanceFixture:
    """Strictly parse and semantically bind a numerical acceptance payload."""

    root = _mapping(payload, "fixture")
    _exact_keys(
        root,
        {
            "kind",
            "schema_version",
            "model",
            "point_policy",
            "comparison",
            "references",
            "catalog_max_n_final",
            "catalog_cases",
            "extra_full_colour_cases",
        },
        "fixture",
    )
    if root.get("kind") != FIXTURE_KIND:
        raise NumericalAcceptanceError("fixture.kind is unsupported")
    if _integer(root.get("schema_version"), "fixture.schema_version") != (
        FIXTURE_SCHEMA_VERSION
    ):
        raise NumericalAcceptanceError("fixture.schema_version is unsupported")
    if _integer(root.get("catalog_max_n_final"), "catalog_max_n_final") != (
        CATALOG_MAX_N_FINAL
    ):
        raise NumericalAcceptanceError(
            f"catalog_max_n_final must be {CATALOG_MAX_N_FINAL}"
        )

    model = _parse_model(root.get("model"))
    if validate_current_model and model != current_model_identity():
        raise NumericalAcceptanceError(
            "fixture UFO-SM model identity differs from the current packaged model"
        )

    point_policy = _mapping(root.get("point_policy"), "point_policy")
    _exact_keys(
        point_policy, {"generator", "seed", "stored_components"}, "point_policy"
    )
    expected_point_policy = {
        "generator": "generic_validation_point",
        "seed": POINT_SEED,
        "stored_components": "canonical-binary64-decimal",
    }
    if dict(point_policy) != expected_point_policy:
        raise NumericalAcceptanceError(
            f"point_policy must equal {expected_point_policy!r}"
        )

    comparison = _mapping(root.get("comparison"), "comparison")
    _exact_keys(
        comparison,
        {
            "relative_tolerance",
            "absolute_tolerance",
            "full_candidate_precision_digits",
            "native_precision",
        },
        "comparison",
    )
    if (
        _canonical_decimal(
            comparison.get("relative_tolerance"), "comparison.relative_tolerance"
        )
        != RELATIVE_TOLERANCE
        or comparison.get("absolute_tolerance") is not None
        or _integer(
            comparison.get("full_candidate_precision_digits"),
            "comparison.full_candidate_precision_digits",
        )
        != FULL_CANDIDATE_PRECISION_DIGITS
        or comparison.get("native_precision") != "binary64"
    ):
        raise NumericalAcceptanceError(
            "comparison must use strict relative-only 1e-10, p200, and binary64"
        )

    references = _mapping(root.get("references"), "references")
    _exact_keys(references, {"full", "lc", "nlc"}, "references")
    full_reference = _parse_full_reference(references.get("full"))
    lc_revision = _parse_regression_reference(references.get("lc"), "lc")
    nlc_revision = _parse_regression_reference(references.get("nlc"), "nlc")
    if lc_revision != nlc_revision:
        raise NumericalAcceptanceError(
            "LC and NLC regression references must come from one source revision"
        )
    if full_reference.external_parameters_sha256 != model.external_parameters_sha256:
        raise NumericalAcceptanceError(
            "MadGraph and pyAmpliCol external-parameter identities differ"
        )

    catalog_specs = catalog_cases()
    raw_catalog = _sequence(root.get("catalog_cases"), "catalog_cases")
    raw_extra = _sequence(
        root.get("extra_full_colour_cases"), "extra_full_colour_cases"
    )
    if len(raw_catalog) != len(catalog_specs):
        raise NumericalAcceptanceError(
            f"catalog_cases must contain exactly {len(catalog_specs)} records"
        )
    if len(raw_extra) != len(EXTRA_FULL_COLOUR_CASES):
        raise NumericalAcceptanceError(
            "extra_full_colour_cases must contain exactly the two declared "
            "non-catalog benchmarks"
        )
    catalog = tuple(
        _parse_case(raw, spec, catalog=True)
        for raw, spec in zip(raw_catalog, catalog_specs, strict=True)
    )
    extra = tuple(
        _parse_case(raw, spec, catalog=False)
        for raw, spec in zip(raw_extra, EXTRA_FULL_COLOUR_CASES, strict=True)
    )
    ids = tuple(case.spec.case_id for case in (*catalog, *extra))
    if len(ids) != len(set(ids)):
        raise NumericalAcceptanceError("fixture case IDs are not unique")
    return AcceptanceFixture(
        model=model,
        full_reference=full_reference,
        captured_source_revision=lc_revision,
        catalog=catalog,
        extra_full_colour=extra,
    )


def load_acceptance_fixture(path: Path = DEFAULT_FIXTURE) -> AcceptanceFixture:
    """Load the tracked acceptance fixture with full semantic validation."""

    return parse_acceptance_fixture(_read_json(path))


def _serialized_value(value: object, where: str) -> str:
    decimal = _decimal_from_number(value, where)
    if decimal < 0:
        raise NumericalAcceptanceError(f"{where} must be non-negative")
    return canonical_decimal(decimal)


def build_fixture_payload(
    *,
    full_values: Mapping[str, object],
    lc_values: Mapping[str, object],
    nlc_values: Mapping[str, object],
    full_reference: MadGraphReferenceIdentity,
    captured_source_revision: str,
) -> dict[str, object]:
    """Build, then strictly re-parse, a complete fixture payload.

    Callers must provide every value; this function never substitutes or
    invents a numerical result.
    """

    catalog_specs = catalog_cases()
    catalog_ids = {case.case_id for case in catalog_specs}
    extra_ids = {case.case_id for case in EXTRA_FULL_COLOUR_CASES}
    if set(full_values) != catalog_ids | extra_ids:
        raise NumericalAcceptanceError(
            "full_values must cover exactly all catalog and extra cases"
        )
    if set(lc_values) != catalog_ids or set(nlc_values) != catalog_ids:
        raise NumericalAcceptanceError(
            "LC and NLC values must each cover exactly the 33 catalog cases"
        )
    revision = _revision(captured_source_revision, "captured_source_revision")

    catalog_payload: list[dict[str, object]] = []
    for spec in catalog_specs:
        catalog_payload.append(
            {
                "id": spec.case_id,
                "family_id": spec.family_id,
                "family_key": spec.family_key,
                "n_final": spec.n_final,
                "process": spec.process,
                "momenta": _momenta_payload(validation_momenta(spec.process)),
                "expected": {
                    "full": _serialized_value(
                        full_values[spec.case_id], f"full_values[{spec.case_id}]"
                    ),
                    "lc": _serialized_value(
                        lc_values[spec.case_id], f"lc_values[{spec.case_id}]"
                    ),
                    "nlc": _serialized_value(
                        nlc_values[spec.case_id], f"nlc_values[{spec.case_id}]"
                    ),
                },
            }
        )
    extra_payload = [
        {
            "id": spec.case_id,
            "process": spec.process,
            "momenta": _momenta_payload(validation_momenta(spec.process)),
            "expected": {
                "full": _serialized_value(
                    full_values[spec.case_id], f"full_values[{spec.case_id}]"
                )
            },
        }
        for spec in EXTRA_FULL_COLOUR_CASES
    ]
    regression_reference = {
        "kind": "pyamplicol-recurrence-regression",
        "precision_digits": FULL_CANDIDATE_PRECISION_DIGITS,
        "captured_source_revision": revision,
    }
    payload: dict[str, object] = {
        "kind": FIXTURE_KIND,
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "model": current_model_identity().as_payload(),
        "point_policy": {
            "generator": "generic_validation_point",
            "seed": POINT_SEED,
            "stored_components": "canonical-binary64-decimal",
        },
        "comparison": {
            "relative_tolerance": canonical_decimal(RELATIVE_TOLERANCE),
            "absolute_tolerance": None,
            "full_candidate_precision_digits": FULL_CANDIDATE_PRECISION_DIGITS,
            "native_precision": "binary64",
        },
        "references": {
            "full": full_reference.as_payload(),
            "lc": dict(regression_reference),
            "nlc": dict(regression_reference),
        },
        "catalog_max_n_final": CATALOG_MAX_N_FINAL,
        "catalog_cases": catalog_payload,
        "extra_full_colour_cases": extra_payload,
    }
    parse_acceptance_fixture(payload)
    return payload


def write_acceptance_fixture(payload: object, path: Path) -> AcceptanceFixture:
    """Validate and exclusively create a fixture; never overwrite one."""

    fixture = parse_acceptance_fixture(payload)
    destination = path.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    try:
        with destination.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(encoded)
    except FileExistsError as error:
        raise NumericalAcceptanceError(
            f"refusing to overwrite existing fixture {destination}"
        ) from error
    return fixture


def lane_run_config(lane: AcceptanceLane) -> object:
    """Return the lean, deterministic generation config for one lane."""

    from pyamplicol.config import (
        Action,
        ColorAccuracy,
        ColorConfig,
        EvaluatorConfig,
        EvaluatorExecutionMode,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        GenerationRelationDiscoveryConfig,
        GenerationValidationConfig,
        JITConfig,
        LCFlowLayout,
        RelationDiscoveryMode,
        RunConfig,
    )

    return RunConfig(
        action=Action.GENERATE,
        color=ColorConfig(
            accuracy=ColorAccuracy(lane.accuracy),
            lc_flow_layout=LCFlowLayout(lane.lc_flow_layout),
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
            execution_mode=EvaluatorExecutionMode(lane.mode),
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=lane.jit_optimization_level),
        ),
    )


def lane_case_specs(
    lane: AcceptanceLane,
    *,
    group: ArtifactGroup = "catalog",
) -> tuple[AcceptanceCaseSpec, ...]:
    """Return the static process order without pulling n=6 into catalog builds."""

    if group == "catalog":
        return catalog_cases()
    if group == "extra" and lane.includes_extra_cases:
        return EXTRA_FULL_COLOUR_CASES
    raise NumericalAcceptanceError(
        f"{lane.lane_id} has no {group!r} acceptance artifact"
    )


def lane_process_set(
    lane: AcceptanceLane,
    *,
    group: ArtifactGroup = "catalog",
) -> object:
    """Return one ordered ProcessSet for a lane's catalog or extra group."""

    from pyamplicol import ProcessSet

    cases = lane_case_specs(lane, group=group)
    return ProcessSet.from_expressions(
        (case.process for case in cases),
        names=tuple(case.artifact_name for case in cases),
    )


def prepare_ufo_sm_model(work_root: Path) -> object:
    """Prepare one recurrence JIT-O2 UFO-SM bundle reusable by all lanes."""

    from pyamplicol import ModelSource

    root = work_root.expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    recurrence_lane = AcceptanceLane("full", "recurrence", 2)
    evaluator = lane_run_config(recurrence_lane).evaluator  # type: ignore[attr-defined]
    model = ModelSource.from_path(
        MODEL_SOURCE,
        restriction=MODEL_RESTRICTION,
    ).compile(
        cache_dir=root / "model-cache",
        use_cache=True,
        prepared_output=root / "ufo-sm-jit-o2.pyamplicol-model",
        evaluator=evaluator,
    )
    if not getattr(model, "is_prepared", False):
        raise NumericalAcceptanceError("UFO-SM preparation returned no prepared model")
    return model


def generate_lane_artifact(
    lane: AcceptanceLane,
    *,
    artifact_path: Path,
    model: object,
    group: ArtifactGroup = "catalog",
) -> Path:
    """Generate exactly one ProcessSet artifact for one lane/group pair."""

    from pyamplicol import Generator

    destination = artifact_path.expanduser().resolve(strict=False)
    Generator(lane_run_config(lane)).generate(
        lane_process_set(lane, group=group),
        destination,
        model=model,  # type: ignore[arg-type]
    )
    _validate_artifact_lane(destination, lane)
    return destination


def _validate_artifact_lane(
    artifact: Path,
    lane: AcceptanceLane,
) -> None:
    """Bind a generated lane to its exact effective mode, accuracy, and layout."""

    import tomllib

    effective_path = artifact / "config" / "effective.toml"
    try:
        effective = tomllib.loads(effective_path.read_text(encoding="utf-8"))
        color = effective["color"]
        evaluator = effective["evaluator"]
        observed = (
            str(color["accuracy"]),
            str(evaluator["execution_mode"]),
            str(color["lc_flow_layout"]),
        )
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise NumericalAcceptanceError(
            f"{lane.lane_id} has no valid effective lane configuration"
        ) from exc
    expected = (lane.accuracy, lane.mode, lane.lc_flow_layout)
    if observed != expected:
        raise NumericalAcceptanceError(
            f"{lane.lane_id} effective lane is {observed!r}; expected {expected!r}"
        )


def _validate_runtime_lane(
    runtime: object,
    lane: AcceptanceLane,
    *,
    context: str,
) -> None:
    observed_mode = getattr(runtime, "execution_mode", None)
    physics = getattr(runtime, "physics", None)
    observed_accuracy = getattr(physics, "color_accuracy", None)
    if observed_mode != lane.mode or observed_accuracy != lane.accuracy:
        raise NumericalAcceptanceError(
            f"{context} loaded as mode={observed_mode!r}, "
            f"color_accuracy={observed_accuracy!r}; expected "
            f"{lane.mode!r}/{lane.accuracy!r}"
        )


class NumericalAcceptanceHarness:
    """Generate each lane once and fail fast through its ordered processes."""

    def __init__(
        self,
        fixture: AcceptanceFixture,
        *,
        artifact_root: Path,
        model: object,
    ) -> None:
        self.fixture = fixture
        self.artifact_root = artifact_root.expanduser().resolve(strict=False)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.model = model
        self._artifacts: dict[tuple[str, ArtifactGroup], Path] = {}

    @classmethod
    def prepare(
        cls,
        fixture_path: Path,
        *,
        work_root: Path,
    ) -> NumericalAcceptanceHarness:
        root = work_root.expanduser().resolve(strict=False)
        fixture = load_acceptance_fixture(fixture_path)
        model = prepare_ufo_sm_model(root / "prepared-model")
        return cls(fixture, artifact_root=root / "artifacts", model=model)

    def _artifact(self, lane: AcceptanceLane, *, group: ArtifactGroup) -> Path:
        key = (lane.lane_id, group)
        existing = self._artifacts.get(key)
        if existing is not None:
            return existing
        artifact_name = lane.lane_id if group == "catalog" else f"{lane.lane_id}-extra"
        artifact = generate_lane_artifact(
            lane,
            artifact_path=self.artifact_root / artifact_name,
            model=self.model,
            group=group,
        )
        self._artifacts[key] = artifact
        return artifact

    def _assert_case(self, case: AcceptanceCase, lane: AcceptanceLane) -> None:
        from pyamplicol import Runtime

        group: ArtifactGroup = "catalog" if case.spec.is_catalog else "extra"
        runtime = Runtime.load(
            self._artifact(lane, group=group),
            process=case.spec.artifact_name,
        )
        try:
            _validate_runtime_lane(runtime, lane, context=lane.lane_id)
            expected = case.expected_for(lane.accuracy)
            for precision in lane.evaluation_precisions_for(group):
                actual = runtime.evaluate(
                    case.runtime_momenta(),
                    precision=precision,
                )[0]
                assert_relative_match(
                    actual,
                    expected,
                    context=f"{case.spec.case_id}/{lane.lane_id}/p{precision}",
                )
        finally:
            runtime.clear()

    def assert_catalog_lane(self, lane: AcceptanceLane) -> None:
        for case in self.fixture.catalog:
            self._assert_case(case, lane)

    def assert_catalog_case(self, case_id: str) -> None:
        case = self.fixture.case(case_id)
        if not case.spec.is_catalog:
            raise NumericalAcceptanceError(
                "assert_catalog_case requires a catalog case"
            )
        for lane in ACCEPTANCE_LANES:
            self._assert_case(case, lane)

    def assert_full_extra(self, case_id: str) -> None:
        case = self.fixture.case(case_id)
        if case.spec.is_catalog:
            raise NumericalAcceptanceError(
                "assert_full_extra requires a deliberately non-catalog case"
            )
        for lane in FULL_LANES:
            self._assert_case(case, lane)


def _normalized_report_momenta(
    value: object,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    points: list[tuple[tuple[float, ...], ...]] = []
    for raw_point in _sequence(value, "MadGraph provenance.report_momenta"):
        rows: list[tuple[float, ...]] = []
        for raw_row in _sequence(raw_point, "MadGraph report point"):
            components = _sequence(raw_row, "MadGraph report momentum")
            if len(components) != 4:
                raise NumericalAcceptanceError(
                    "MadGraph report momentum must have four components"
                )
            row = tuple(float(component) for component in components)
            if any(not math.isfinite(component) for component in row):
                raise NumericalAcceptanceError("MadGraph report momentum is not finite")
            rows.append(row)
        points.append(tuple(rows))
    return tuple(points)


def _madgraph_measurement(
    measurement_value: object,
    spec: AcceptanceCaseSpec,
) -> tuple[Decimal, MadGraphReferenceIdentity]:
    from tools.performance_report.madgraph import (
        MADGRAPH_DRIVER_SOURCE_SHA256,
        madgraph_command_card,
    )
    from tools.performance_report.runner import point_digest

    measurement = _mapping(measurement_value, f"MadGraph measurement {spec.case_id}")
    if measurement.get("status") != "ok":
        raise NumericalAcceptanceError(
            f"MadGraph measurement {spec.case_id} is not successful"
        )
    matrix_element = _decimal_from_number(
        measurement.get("matrix_element"),
        f"MadGraph measurement {spec.case_id}.matrix_element",
    )
    if matrix_element < 0:
        raise NumericalAcceptanceError(
            f"MadGraph measurement {spec.case_id} is negative"
        )
    expected_point = (
        tuple(
            tuple(float(component) for component in row)
            for row in validation_momenta(spec.process)
        ),
    )
    validation = _mapping(measurement.get("validation"), "MadGraph validation")
    if (
        validation.get("status") != "ok"
        or validation.get("method") != "independent-madgraph-tree-level-oracle"
        or validation.get("point_digest") != point_digest(expected_point)
    ):
        raise NumericalAcceptanceError(
            f"MadGraph measurement {spec.case_id} has invalid point evidence"
        )
    provenance = _mapping(measurement.get("provenance"), "MadGraph provenance")
    model = _mapping(provenance.get("model"), "MadGraph provenance.model")
    exact_card = _mapping(
        provenance.get("exact_param_card"),
        "MadGraph provenance.exact_param_card",
    )
    restriction_card = _mapping(
        provenance.get("default_restriction"),
        "MadGraph provenance.default_restriction",
    )
    external_sha = current_model_identity().external_parameters_sha256
    expected_command_card = madgraph_command_card(spec.process)
    expected_command_sha = hashlib.sha256(
        expected_command_card.encode("utf-8")
    ).hexdigest()
    driver_sha256 = _sha256(
        provenance.get("driver_sha256"), "MadGraph provenance.driver_sha256"
    )
    if (
        provenance.get("method") != "madgraph-standalone-custom-fortran-driver"
        or provenance.get("command_card") != expected_command_card
        or provenance.get("command_card_sha256") != expected_command_sha
        or _normalized_report_momenta(provenance.get("report_momenta"))
        != expected_point
        or model.get("name") != "sm"
        or driver_sha256 != MADGRAPH_DRIVER_SOURCE_SHA256
        or exact_card.get("binary64_exact_match") is not True
        or restriction_card.get("binary64_exact_match") is not True
        or exact_card.get("external_parameters_sha256") != external_sha
        or restriction_card.get("external_parameters_sha256") != external_sha
        or exact_card.get("format") != "%.14e"
        or restriction_card.get("format") != "%.14e"
    ):
        raise NumericalAcceptanceError(
            f"MadGraph measurement {spec.case_id} violates the standalone/model "
            "authority contract"
        )
    for label, card in (
        ("exact_param_card", exact_card),
        ("default_restriction", restriction_card),
    ):
        count = card.get("external_parameter_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise NumericalAcceptanceError(
                f"MadGraph measurement {spec.case_id} has invalid {label} coverage"
            )
    identity = MadGraphReferenceIdentity(
        madgraph_version=_string(
            provenance.get("version"), "MadGraph provenance.version"
        ),
        model_source_sha256=_sha256(
            model.get("source_sha256"), "MadGraph provenance.model.source_sha256"
        ),
        driver_sha256=driver_sha256,
        external_parameters_sha256=external_sha,
    )
    return matrix_element, identity


def _one_madgraph_identity(
    identities: Iterable[MadGraphReferenceIdentity],
) -> MadGraphReferenceIdentity:
    unique = set(identities)
    if len(unique) != 1:
        raise NumericalAcceptanceError(
            "MadGraph records do not share one version/model/driver identity"
        )
    return next(iter(unique))


def ingest_authenticated_madgraph_cache(path: Path) -> MadGraphIngest:
    """Load catalog values only after existing campaign-cache validation."""

    from tools.performance_report.cache import validate_cache
    from tools.performance_report.catalog import REPORT_CATALOG

    payload = _read_json(path)
    cache = _mapping(payload, "MadGraph cache")
    if cache.get("dataset_id") != "reference_madgraph_full":
        raise NumericalAcceptanceError(
            "MadGraph cache must be the reference_madgraph_full dataset"
        )
    entries = _sequence(cache.get("entries"), "MadGraph cache.entries")
    expected_cells = {
        cell.cell_id: cell
        for cell in REPORT_CATALOG.reference_cells()
        if cell.dataset_id == "reference_madgraph_full"
    }
    observed_cell_ids: list[str] = []
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"MadGraph cache.entries[{index}]")
        cell_id = entry.get("cell_id")
        if not isinstance(cell_id, str) or cell_id not in expected_cells:
            raise NumericalAcceptanceError(
                f"MadGraph cache contains unknown cell {cell_id!r}"
            )
        observed_cell_ids.append(cell_id)
    if set(observed_cell_ids) != set(expected_cells):
        raise NumericalAcceptanceError(
            "MadGraph cache does not cover the complete authenticated "
            "reference_madgraph_full dataset"
        )
    try:
        validate_cache(
            payload,
            expected_cells=expected_cells.values(),
            catalog=REPORT_CATALOG,
        )
    except (TypeError, ValueError) as error:
        raise NumericalAcceptanceError(
            f"MadGraph campaign cache authentication failed: {error}"
        ) from error
    entries_by_process = {
        (entry.get("process_key"), entry.get("n_final")): entry
        for entry in (_mapping(raw, "MadGraph cache entry") for raw in entries)
    }
    values: dict[str, Decimal] = {}
    identities: list[MadGraphReferenceIdentity] = []
    for spec in catalog_cases():
        entry = entries_by_process.get((spec.family_key, spec.n_final))
        if entry is None or entry.get("process") != spec.process:
            raise NumericalAcceptanceError(
                f"MadGraph cache lacks catalog record {spec.case_id}"
            )
        value, identity = _madgraph_measurement(entry.get("measurement"), spec)
        values[spec.case_id] = value
        identities.append(identity)
    return MadGraphIngest(values=values, identity=_one_madgraph_identity(identities))


def _gate_point(path: Path, spec: AcceptanceCaseSpec) -> None:
    raw_rows = _sequence(_read_json(path), f"gate point {path}")
    observed: list[tuple[float, ...]] = []
    for index, raw_row in enumerate(raw_rows):
        components = _sequence(raw_row, f"gate point {path}[{index}]")
        if len(components) != 4:
            raise NumericalAcceptanceError(
                f"gate point {path} contains a malformed momentum"
            )
        try:
            row = tuple(float(component) for component in components)
        except (TypeError, ValueError, OverflowError) as error:
            raise NumericalAcceptanceError(
                f"gate point {path} contains a non-numeric momentum"
            ) from error
        if any(not math.isfinite(component) for component in row):
            raise NumericalAcceptanceError(f"gate point {path} is not finite")
        observed.append(row)
    expected = tuple(
        tuple(float(component) for component in row)
        for row in validation_momenta(spec.process)
    )
    if tuple(observed) != expected:
        raise NumericalAcceptanceError(
            f"gate point {path} differs from {spec.case_id} seed-{POINT_SEED}"
        )


def ingest_authenticated_madgraph_wave_root(root: Path) -> MadGraphIngest:
    """Reuse the complete n=1..4 MadGraph authority prefix from a wave."""

    gate_root = root.expanduser().resolve(strict=True)
    manifest = _mapping(_read_json(gate_root / "manifest.json"), "wave manifest")
    if (
        manifest.get("kind") != "pyamplicol-madgraph-ufo-sm-full-p200-wave-gate-v1"
        or manifest.get("status") not in {"running", "ok"}
        or _integer(manifest.get("seed"), "wave seed") != POINT_SEED
        or _integer(manifest.get("max_n"), "wave max_n") < CATALOG_MAX_N_FINAL
        or manifest.get("ordering") != "n-final-then-family-id-then-mode"
    ):
        raise NumericalAcceptanceError(
            "wave root cannot supply the ordered n=1..4 MadGraph authority prefix"
        )

    specs = catalog_cases()
    plan = _sequence(manifest.get("plan"), "wave plan")
    if len(plan) < len(specs):
        raise NumericalAcceptanceError("wave plan does not contain all 33 cases")
    for ordinal, (raw_plan, spec) in enumerate(
        zip(plan[: len(specs)], specs, strict=True)
    ):
        record = _mapping(raw_plan, f"wave plan[{ordinal}]")
        if (
            record.get("cell_ordinal") != ordinal
            or record.get("n_final") != spec.n_final
            or record.get("family_id") != spec.family_id
            or record.get("process_key") != spec.family_key
            or record.get("process") != spec.process
        ):
            raise NumericalAcceptanceError(
                f"wave plan[{ordinal}] differs from {spec.case_id}"
            )

    values: dict[str, Decimal] = {}
    identities: list[MadGraphReferenceIdentity] = []
    for spec in specs:
        case_root = gate_root / f"n{spec.n_final}" / f"family-{spec.family_id:02d}"
        _gate_point(case_root / "point-seed-101.json", spec)
        value, identity = _madgraph_measurement(
            _read_json(case_root / "madgraph-measurement.json"),
            spec,
        )
        authority = _mapping(
            _read_json(case_root / "madgraph-authority.json"),
            f"wave authority {spec.case_id}",
        )
        if (
            authority.get("process") != spec.process
            or authority.get("n_final") != spec.n_final
            or authority.get("seed") != POINT_SEED
            or _decimal_from_number(
                authority.get("madgraph_value"),
                f"wave authority {spec.case_id}.madgraph_value",
            )
            != value
        ):
            raise NumericalAcceptanceError(
                f"wave authority for {spec.case_id} is inconsistent"
            )
        values[spec.case_id] = value
        identities.append(identity)
    return MadGraphIngest(values=values, identity=_one_madgraph_identity(identities))


def ingest_authenticated_extra_madgraph_root(root: Path) -> MadGraphIngest:
    """Reuse the two MadGraph authority records from the successful n=6 gate."""

    gate_root = root.expanduser().resolve(strict=True)
    manifest = _mapping(_read_json(gate_root / "manifest.json"), "stress manifest")
    process_keys = ("uu_four_identical_lines", "ee_four_identical_lines")
    expected_process_order = [
        {"key": key, "process": spec.process, "n_final": spec.n_final}
        for key, spec in zip(process_keys, EXTRA_FULL_COLOUR_CASES, strict=True)
    ]
    if (
        manifest.get("kind")
        != "pyamplicol-authenticated-noncatalog-n6-madgraph-stress-v1"
        or manifest.get("status") != "ok"
        or manifest.get("stage") != "complete"
        or _integer(manifest.get("seed"), "stress seed") != POINT_SEED
        or manifest.get("process_order") != expected_process_order
        or manifest.get("catalog_policy")
        != "direct synthetic CellSpec; REPORT_CATALOG not modified"
    ):
        raise NumericalAcceptanceError(
            "extra root cannot supply the authenticated n=6 MadGraph authority"
        )

    required_checks = (
        "status",
        "exact_seed_101_point",
        "exact_command_card",
        "custom_fortran_method",
        "custom_driver_digest",
        "exact_generated_process",
    )
    values: dict[str, Decimal] = {}
    identities: list[MadGraphReferenceIdentity] = []
    for case_ordinal, (key, spec) in enumerate(
        zip(process_keys, EXTRA_FULL_COLOUR_CASES, strict=True)
    ):
        case_root = gate_root / f"case-{case_ordinal + 1:02d}-{key}"
        _gate_point(case_root / "point-seed-101.json", spec)
        value, identity = _madgraph_measurement(
            _read_json(case_root / "madgraph-measurement.json"),
            spec,
        )
        authority = _mapping(
            _read_json(case_root / "madgraph-authority-verification.json"),
            f"stress authority {spec.case_id}",
        )
        checks = _mapping(
            authority.get("checks"), f"stress authority {spec.case_id}.checks"
        )
        expected_point = (
            tuple(
                tuple(float(component) for component in row)
                for row in validation_momenta(spec.process)
            ),
        )
        if (
            authority.get("process") != spec.process
            or authority.get("seed") != POINT_SEED
            or _normalized_report_momenta((authority.get("point"),)) != expected_point
            or _decimal_from_number(
                authority.get("matrix_element_binary64"),
                f"stress authority {spec.case_id}.matrix_element_binary64",
            )
            != value
            or any(checks.get(name) is not True for name in required_checks)
        ):
            raise NumericalAcceptanceError(
                f"stress MadGraph authority for {spec.case_id} is inconsistent"
            )
        values[spec.case_id] = value
        identities.append(identity)
    return MadGraphIngest(values=values, identity=_one_madgraph_identity(identities))


def capture_extra_madgraph_values(
    *,
    installation: Path,
    artifact_root: Path,
) -> MadGraphIngest:
    """Capture the two non-catalog MadGraph benchmarks with the shared adapter."""

    from tools.performance_report.madgraph import (
        MadGraphMeasurementAdapter,
        MadGraphSettings,
    )
    from tools.performance_report.models import (
        Accuracy,
        CellSpec,
        ExecutionMode,
        MeasurementSpec,
        ModelKey,
        Workload,
    )

    settings = MadGraphSettings(
        installation=installation,
        json_model_path=MODEL_SOURCE,
        restriction_json_path=MODEL_RESTRICTION,
    )
    adapter = MadGraphMeasurementAdapter()
    values: dict[str, Decimal] = {}
    identities: list[MadGraphReferenceIdentity] = []
    for spec in EXTRA_FULL_COLOUR_CASES:
        cell = CellSpec(
            dataset_id="numerical_acceptance_extra_full",
            process=spec.process,
            n_final=spec.n_final,
            process_key=spec.case_id,
            measurement=MeasurementSpec(
                ExecutionMode.MADGRAPH,
                ModelKey.UFO_SM,
                Accuracy.FULL,
                "fortran",
                None,
            ),
            workload=Workload.CONTRACTED,
        )
        measurement = adapter.measure(
            cell,
            artifact_path=artifact_root / spec.artifact_name,
            settings=settings,
        )
        value, identity = _madgraph_measurement(measurement, spec)
        values[spec.case_id] = value
        identities.append(identity)
    return MadGraphIngest(values=values, identity=_one_madgraph_identity(identities))


@dataclass(frozen=True, slots=True)
class _RuntimeBuildIdentity:
    source_revision: str
    native_build_inputs_sha256: str
    version: str


def _current_runtime_identity() -> _RuntimeBuildIdentity:
    """Authenticate the installed runtime without consulting the dirty checkout."""

    try:
        import pyamplicol
        from pyamplicol._internal.versions import active_native_source_identity

        expected_prefix = (ROOT / ".venv").resolve(strict=True)
        observed_prefix = Path(sys.prefix).resolve(strict=True)
        package_raw = getattr(pyamplicol, "__file__", None)
        if not isinstance(package_raw, str):
            raise NumericalAcceptanceError(
                "the active pyamplicol package has no filesystem origin"
            )
        package_path = Path(package_raw).resolve(strict=True)
        if (
            observed_prefix != expected_prefix
            or not package_path.is_relative_to(expected_prefix)
            or "site-packages" not in package_path.relative_to(expected_prefix).parts
        ):
            raise NumericalAcceptanceError(
                "capture must import pyamplicol from this repository's .venv "
                "site-packages"
            )
        revision, native_digest = active_native_source_identity()
        return _RuntimeBuildIdentity(
            source_revision=_revision(revision, "active runtime source revision"),
            native_build_inputs_sha256=_sha256(
                native_digest, "active runtime native build inputs"
            ),
            version=_string(
                importlib.metadata.version("pyamplicol"),
                "active runtime package version",
            ),
        )
    except NumericalAcceptanceError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise NumericalAcceptanceError(
            f"could not authenticate the active pyamplicol runtime: {error}"
        ) from error


def _capture_recurrence_references(
    *,
    model: object,
    artifact_root: Path,
    full_values: Mapping[str, Decimal],
) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
    """Recheck current recurrence p200/p16 and capture LC/NLC references."""

    from pyamplicol import Runtime

    artifact_root.mkdir(parents=True, exist_ok=True)
    captured: dict[str, dict[str, Decimal]] = {"lc": {}, "nlc": {}}

    for group in cast(tuple[ArtifactGroup, ArtifactGroup], ("catalog", "extra")):
        lane = AcceptanceLane("full", "recurrence", 2)
        artifact = generate_lane_artifact(
            lane,
            artifact_path=artifact_root / f"capture-{lane.lane_id}-{group}",
            model=model,
            group=group,
        )
        for spec in lane_case_specs(lane, group=group):
            momenta = (
                tuple(
                    tuple(float(component) for component in row)
                    for row in validation_momenta(spec.process)
                ),
            )
            runtime = Runtime.load(artifact, process=spec.artifact_name)
            try:
                _validate_runtime_lane(
                    runtime,
                    lane,
                    context=f"capture/{spec.case_id}",
                )
                precise = runtime.evaluate(
                    momenta, precision=FULL_CANDIDATE_PRECISION_DIGITS
                )[0]
                assert_relative_match(
                    precise,
                    full_values[spec.case_id],
                    context=f"capture/{spec.case_id}/full-recurrence/p200",
                )
                native = runtime.evaluate(momenta, precision=NATIVE_PRECISION_DIGITS)[0]
                assert_relative_match(
                    native,
                    full_values[spec.case_id],
                    context=f"capture/{spec.case_id}/full-recurrence/p16",
                )
            finally:
                runtime.clear()

    for accuracy in cast(tuple[AccuracyName, AccuracyName], ("lc", "nlc")):
        lane = AcceptanceLane(accuracy, "recurrence", 2)
        artifact = generate_lane_artifact(
            lane,
            artifact_path=artifact_root / f"capture-{lane.lane_id}",
            model=model,
        )
        for spec in lane_case_specs(lane):
            momenta = (
                tuple(
                    tuple(float(component) for component in row)
                    for row in validation_momenta(spec.process)
                ),
            )
            runtime = Runtime.load(artifact, process=spec.artifact_name)
            try:
                _validate_runtime_lane(
                    runtime,
                    lane,
                    context=f"capture/{spec.case_id}",
                )
                precise = runtime.evaluate(
                    momenta, precision=FULL_CANDIDATE_PRECISION_DIGITS
                )[0]
                precise_decimal = _decimal_from_number(
                    precise, f"captured {spec.case_id}/{accuracy}/p200"
                )
                native = runtime.evaluate(momenta, precision=NATIVE_PRECISION_DIGITS)[0]
                assert_relative_match(
                    native,
                    precise_decimal,
                    context=f"capture/{spec.case_id}/{accuracy}/p16",
                )
                captured[accuracy][spec.case_id] = precise_decimal
            finally:
                runtime.clear()
    return captured["lc"], captured["nlc"]


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def capture_acceptance_fixture(
    *,
    madgraph_cache: Path | None = None,
    madgraph_wave_root: Path | None = None,
    madgraph_installation: Path | None = None,
    extra_madgraph_root: Path | None = None,
    work_root: Path,
    output: Path,
) -> AcceptanceFixture:
    """Capture a fixture, refusing to publish until all strict gates pass."""

    if (madgraph_cache is None) == (madgraph_wave_root is None):
        raise NumericalAcceptanceError(
            "provide exactly one of madgraph_cache or madgraph_wave_root"
        )
    if extra_madgraph_root is None and madgraph_installation is None:
        raise NumericalAcceptanceError(
            "madgraph_installation is required when extra_madgraph_root is absent"
        )
    destination = output.expanduser().resolve(strict=False)
    root = work_root.expanduser().resolve(strict=False)
    if destination.exists():
        raise NumericalAcceptanceError(
            f"refusing to overwrite existing fixture {destination}"
        )
    if root.exists():
        raise NumericalAcceptanceError(f"capture work root already exists: {root}")
    if _paths_overlap(destination, root):
        raise NumericalAcceptanceError(
            "fixture output must be outside the disposable capture work root"
        )
    for source in (
        madgraph_cache,
        madgraph_wave_root,
        extra_madgraph_root,
        madgraph_installation,
    ):
        if source is None:
            continue
        resolved_source = source.expanduser().resolve(strict=False)
        if _paths_overlap(destination, resolved_source) or _paths_overlap(
            root, resolved_source
        ):
            raise NumericalAcceptanceError(
                "fixture output and work root must not overlap a MadGraph input"
            )

    runtime_identity = _current_runtime_identity()
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise NumericalAcceptanceError(
            f"capture work root already exists: {root}"
        ) from error

    catalog_madgraph = (
        ingest_authenticated_madgraph_cache(madgraph_cache)
        if madgraph_cache is not None
        else ingest_authenticated_madgraph_wave_root(cast(Path, madgraph_wave_root))
    )
    if extra_madgraph_root is not None:
        extra_madgraph = ingest_authenticated_extra_madgraph_root(extra_madgraph_root)
    else:
        extra_madgraph = capture_extra_madgraph_values(
            installation=cast(Path, madgraph_installation),
            artifact_root=root / "madgraph-extra",
        )
    if catalog_madgraph.identity != extra_madgraph.identity:
        raise NumericalAcceptanceError(
            "catalog and extra MadGraph records have different authority identities"
        )
    full_values = {**catalog_madgraph.values, **extra_madgraph.values}
    model = prepare_ufo_sm_model(root / "prepared-model")
    lc_values, nlc_values = _capture_recurrence_references(
        model=model,
        artifact_root=root / "pyamplicol",
        full_values=full_values,
    )
    payload = build_fixture_payload(
        full_values=full_values,
        lc_values=lc_values,
        nlc_values=nlc_values,
        full_reference=catalog_madgraph.identity,
        captured_source_revision=runtime_identity.source_revision,
    )
    return write_acceptance_fixture(payload, destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate an existing fixture")
    validate.add_argument("fixture", type=Path)
    capture = subparsers.add_parser(
        "capture",
        help="capture strict full-colour controls and p200 regression references",
    )
    catalog_source = capture.add_mutually_exclusive_group(required=True)
    catalog_source.add_argument("--madgraph-cache", type=Path)
    catalog_source.add_argument("--madgraph-wave-root", type=Path)
    capture.add_argument("--madgraph-installation", type=Path)
    capture.add_argument(
        "--extra-madgraph-root",
        type=Path,
        help="reuse a successful authenticated non-catalog n=6 stress gate",
    )
    capture.add_argument("--work-root", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        fixture = load_acceptance_fixture(arguments.fixture)
        print(
            f"validated {len(fixture.catalog)} catalog and "
            f"{len(fixture.extra_full_colour)} extra cases"
        )
        return 0
    if arguments.command == "capture":
        fixture = capture_acceptance_fixture(
            madgraph_cache=arguments.madgraph_cache,
            madgraph_wave_root=arguments.madgraph_wave_root,
            madgraph_installation=arguments.madgraph_installation,
            extra_madgraph_root=arguments.extra_madgraph_root,
            work_root=arguments.work_root,
            output=arguments.output,
        )
        print(
            f"captured {len(fixture.catalog)} catalog and "
            f"{len(fixture.extra_full_colour)} extra cases to "
            f"{arguments.output}"
        )
        return 0
    raise NumericalAcceptanceError(f"unsupported command {arguments.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCEPTANCE_LANES",
    "CATALOG_CASE_COUNT",
    "CATALOG_MAX_N_FINAL",
    "DEFAULT_FIXTURE",
    "EXTRA_FULL_COLOUR_CASES",
    "FULL_CANDIDATE_PRECISION_DIGITS",
    "FULL_LANES",
    "LC_ALL_FLOW_UNION_LANES",
    "NATIVE_PRECISION_DIGITS",
    "POINT_SEED",
    "RELATIVE_TOLERANCE",
    "AcceptanceCase",
    "AcceptanceCaseSpec",
    "AcceptanceFixture",
    "AcceptanceLane",
    "MadGraphReferenceIdentity",
    "NumericalAcceptanceError",
    "NumericalAcceptanceHarness",
    "NumericalAcceptanceMismatch",
    "build_fixture_payload",
    "capture_acceptance_fixture",
    "catalog_cases",
    "compare_relative",
    "current_model_identity",
    "ingest_authenticated_extra_madgraph_root",
    "ingest_authenticated_madgraph_cache",
    "ingest_authenticated_madgraph_wave_root",
    "lane_case_specs",
    "lane_process_set",
    "lane_run_config",
    "load_acceptance_fixture",
    "main",
    "parse_acceptance_fixture",
    "prepare_ufo_sm_model",
    "validation_momenta",
    "write_acceptance_fixture",
]
