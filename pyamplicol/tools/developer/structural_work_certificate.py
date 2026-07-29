#!/usr/bin/env python3
"""Certify pure-gluon full-color work against an AmpliCol replay.

The legacy generated library materializes one reduced current/interaction
module and replays one module for every retained color ordering.  pyAmpliCol
instead fuses those color orderings into one recurrence/DAG/eager schedule.
Comparing pyAmpliCol with only one legacy module is therefore misleading.  This
tool computes the exact legacy dynamic replay work and checks that the final
pyAmpliCol schedule does not evaluate more currents or interactions.

For recurrence artifacts it additionally bounds construction-only inflation.
Those peaks are intentionally kept separate from the final runtime work: they
diagnose cold-generation regressions without pretending that discarded
construction candidates execute at every phase-space point.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class StructuralWorkError(RuntimeError):
    """The artifacts cannot produce a valid structural-work certificate."""


_AFTER_FILTER = re.compile(
    r"Total number of currents, vertices and amplitudes after filter"
    r"\s+(\d+)\s+(\d+)\s+(\d+)"
)
_VAL_C = re.compile(r"dimension\s*\(\s*1\s*:\s*\d+\s*,\s*(\d+)\s*\)\s*::\s*val_c\b")
_INT_C = re.compile(r"dimension\s*\(\s*1\s*:\s*\d+\s*,\s*(\d+)\s*\)\s*::\s*int_c\b")
_PEAK_CONTRIBUTION_COUNT_SEMANTICS = "resident-pending-contributions-v1"


@dataclass(frozen=True)
class LegacyReplayWork:
    module_current_count: int
    module_interaction_count: int
    retained_color_ordering_count: int
    replay_current_count: int
    replay_interaction_count: int
    generated_module_count: int


@dataclass(frozen=True)
class CandidateFinalWork:
    mode: str
    current_count: int
    interaction_count: int
    current_ratio_to_legacy_replay: float
    interaction_ratio_to_legacy_replay: float


@dataclass(frozen=True)
class FinalWorkComparison:
    current_delta_count: int
    interaction_delta_count: int
    current_savings_fraction: float
    interaction_savings_fraction: float
    classification: str


@dataclass(frozen=True)
class RecurrenceConstructionWork:
    peak_current_count: int
    peak_contribution_count: int
    peak_to_final_current_ratio: float
    peak_to_final_contribution_ratio: float
    peak_current_ratio_to_legacy_replay: float


@dataclass(frozen=True)
class StructuralWorkCertificate:
    schema: str
    process: str
    external_pdg_order: tuple[int, ...]
    legacy: LegacyReplayWork
    candidate: CandidateFinalWork
    final_work_comparison: FinalWorkComparison
    recurrence_construction: RecurrenceConstructionWork | None
    limits: dict[str, float]
    status: str


@dataclass(frozen=True)
class AdjacentMultiplicityCensus:
    schema: str
    mode: str
    lower_n_final: int
    higher_n_final: int
    legacy_current_growth: float
    candidate_current_growth: float
    normalized_current_growth: float
    legacy_interaction_growth: float
    candidate_interaction_growth: float
    normalized_interaction_growth: float
    normalized_peak_current_growth: float | None
    limits: dict[str, float]
    status: str


@dataclass(frozen=True)
class StaticTemplateMaterializationCensus:
    schema: str
    mode: str
    legacy_generated_module_count: int
    legacy_static_current_count: int
    legacy_static_interaction_count: int
    legacy_replay_multiplicity_per_module: int
    candidate_final_current_count: int
    candidate_final_interaction_count: int
    candidate_final_current_ratio: float
    candidate_final_interaction_ratio: float
    candidate_peak_current_count: int | None
    candidate_peak_interaction_count: int | None
    candidate_peak_current_ratio: float | None
    candidate_peak_interaction_ratio: float | None
    limits: dict[str, float]
    violations: tuple[str, ...]
    status: str


MAX_UNPROVEN_FINAL_TO_LEGACY = 1.05
MAX_ADJACENT_FINAL_GROWTH = 1.05
MAX_ADJACENT_PEAK_GROWTH = 1.25
MAX_STATIC_TEMPLATE_MATERIALIZATION = 1.05


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise StructuralWorkError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise StructuralWorkError(f"JSON root is not an object: {path}")
    return payload


def _single_file(root: Path, name: str) -> Path:
    files = sorted(root.rglob(name))
    if len(files) != 1:
        raise StructuralWorkError(
            f"{root} must contain exactly one {name}; found {len(files)}"
        )
    return files[0]


def legacy_replay_work(legacy_artifact: Path) -> LegacyReplayWork:
    probe = _single_file(legacy_artifact, "amplicol_color_library_probe.output")
    match = _AFTER_FILTER.search(probe.read_text())
    if match is None:
        raise StructuralWorkError(
            "legacy probe lacks authenticated after-filter counts"
        )
    _, _, retained_amplitudes = (int(value) for value in match.groups())
    if retained_amplitudes <= 0:
        raise StructuralWorkError("legacy probe has no retained color orderings")

    modules = sorted(legacy_artifact.rglob("amp*_lib.f03"))
    if not modules:
        raise StructuralWorkError("legacy artifact has no generated library modules")
    dimensions: set[tuple[int, int]] = set()
    for module in modules:
        text = module.read_text()
        current = _VAL_C.search(text)
        interaction = _INT_C.search(text)
        if current is None or interaction is None:
            raise StructuralWorkError(
                f"legacy module lacks val_c/int_c dimensions: {module}"
            )
        dimensions.add((int(current.group(1)), int(interaction.group(1))))
    if len(dimensions) != 1:
        raise StructuralWorkError(
            "legacy pure-gluon modules do not share one replay shape"
        )
    module_currents, module_interactions = dimensions.pop()
    return LegacyReplayWork(
        module_current_count=module_currents,
        module_interaction_count=module_interactions,
        retained_color_ordering_count=retained_amplitudes,
        replay_current_count=module_currents * retained_amplitudes,
        replay_interaction_count=module_interactions * retained_amplitudes,
        generated_module_count=len(modules),
    )


def _positive_integer(payload: dict[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise StructuralWorkError(f"{label} {key} must be a positive integer")
    return value


def candidate_final_work(
    execution: dict[str, Any],
    legacy: LegacyReplayWork,
) -> tuple[CandidateFinalWork, RecurrenceConstructionWork | None]:
    kind = execution.get("kind")
    construction = None
    if kind == "pyamplicol-runtime-recurrence-execution":
        try:
            inspection = execution["plan"]["inspection_summary"]
            schedule = inspection["schedule"]
        except (KeyError, TypeError) as error:
            raise StructuralWorkError(
                "recurrence execution lacks inspection schedule"
            ) from error
        mode = "recurrence"
        currents = _positive_integer(schedule, "current_count", "recurrence schedule")
        interactions = _positive_integer(
            schedule, "contribution_count", "recurrence schedule"
        )
        construction_payload = inspection.get("construction")
        if not isinstance(construction_payload, dict):
            raise StructuralWorkError(
                "recurrence execution lacks construction diagnostics"
            )
        if (
            construction_payload.get("peak_contribution_count_semantics")
            != _PEAK_CONTRIBUTION_COUNT_SEMANTICS
        ):
            raise StructuralWorkError(
                "recurrence construction lacks authenticated resident "
                "peak-contribution semantics"
            )
        peak_currents = _positive_integer(
            construction_payload,
            "peak_current_count",
            "recurrence construction",
        )
        peak_contributions = _positive_integer(
            construction_payload,
            "peak_contribution_count",
            "recurrence construction",
        )
        construction = RecurrenceConstructionWork(
            peak_current_count=peak_currents,
            peak_contribution_count=peak_contributions,
            peak_to_final_current_ratio=peak_currents / currents,
            peak_to_final_contribution_ratio=peak_contributions / interactions,
            peak_current_ratio_to_legacy_replay=(
                peak_currents / legacy.replay_current_count
            ),
        )
    elif kind == "pyamplicol-runtime-execution":
        summary = execution.get("helicity_sum_execution", {}).get("dag_summary")
        if not isinstance(summary, dict):
            raise StructuralWorkError(
                "compiled execution lacks helicity-sum DAG summary"
            )
        mode = "compiled"
        currents = _positive_integer(summary, "current_count", "compiled DAG")
        interactions = _positive_integer(summary, "interaction_count", "compiled DAG")
    elif kind == "pyamplicol-runtime-eager-execution":
        try:
            summary = execution["plan"]["inspection_summary"]
        except (KeyError, TypeError) as error:
            raise StructuralWorkError(
                "eager execution lacks inspection summary"
            ) from error
        mode = "eager"
        currents = _positive_integer(summary, "current_count", "eager plan")
        # Attachments are the eager representation of DAG interaction fan-in.
        interactions = _positive_integer(summary, "attachment_count", "eager plan")
    else:
        raise StructuralWorkError(f"unsupported candidate execution kind: {kind!r}")

    return (
        CandidateFinalWork(
            mode=mode,
            current_count=currents,
            interaction_count=interactions,
            current_ratio_to_legacy_replay=currents / legacy.replay_current_count,
            interaction_ratio_to_legacy_replay=(
                interactions / legacy.replay_interaction_count
            ),
        ),
        construction,
    )


def _final_work_comparison(
    legacy: LegacyReplayWork,
    candidate: CandidateFinalWork,
) -> FinalWorkComparison:
    current_delta = candidate.current_count - legacy.replay_current_count
    interaction_delta = candidate.interaction_count - legacy.replay_interaction_count
    classification = (
        "proven-recycling"
        if current_delta <= 0 and interaction_delta <= 0
        else "within-structural-tolerance"
    )
    return FinalWorkComparison(
        current_delta_count=current_delta,
        interaction_delta_count=interaction_delta,
        current_savings_fraction=-current_delta / legacy.replay_current_count,
        interaction_savings_fraction=(
            -interaction_delta / legacy.replay_interaction_count
        ),
        classification=classification,
    )


def certify(
    legacy_artifact: Path,
    candidate_artifact: Path,
    *,
    max_final_to_legacy: float = MAX_UNPROVEN_FINAL_TO_LEGACY,
    max_peak_current_to_legacy: float = 1.5,
    max_peak_to_final_current: float = 4.0,
    max_peak_to_final_contribution: float = 6.0,
) -> StructuralWorkCertificate:
    if (
        not math.isfinite(max_final_to_legacy)
        or max_final_to_legacy <= 0.0
        or max_final_to_legacy > MAX_UNPROVEN_FINAL_TO_LEGACY
    ):
        raise StructuralWorkError(
            "unreviewed final-work budget must be positive and no greater "
            f"than {MAX_UNPROVEN_FINAL_TO_LEGACY:.2f}x legacy replay"
        )
    legacy = legacy_replay_work(legacy_artifact)
    execution_path = _single_file(candidate_artifact, "execution.json")
    execution = _read_json(execution_path)
    external = execution.get("external_pdg_order")
    if (
        not isinstance(external, list)
        or len(external) < 4
        or any(pdg != 21 for pdg in external)
        or execution.get("color_accuracy") != "full"
    ):
        raise StructuralWorkError(
            "certificate currently requires a full-color pure-gluon process"
        )
    candidate, construction = candidate_final_work(execution, legacy)
    limits = {
        "max_final_to_legacy_replay": max_final_to_legacy,
        "max_peak_current_to_legacy_replay": max_peak_current_to_legacy,
        "max_peak_to_final_current": max_peak_to_final_current,
        "max_peak_to_final_contribution": max_peak_to_final_contribution,
    }
    failures = []
    if candidate.current_ratio_to_legacy_replay > max_final_to_legacy:
        failures.append("final current work exceeds legacy replay")
    if candidate.interaction_ratio_to_legacy_replay > max_final_to_legacy:
        failures.append("final interaction work exceeds legacy replay")
    if construction is not None:
        if (
            construction.peak_current_ratio_to_legacy_replay
            > max_peak_current_to_legacy
        ):
            failures.append("peak construction currents exceed legacy replay budget")
        if construction.peak_to_final_current_ratio > max_peak_to_final_current:
            failures.append("peak/final construction current ratio exceeds budget")
        if (
            construction.peak_to_final_contribution_ratio
            > max_peak_to_final_contribution
        ):
            failures.append("peak/final construction contribution ratio exceeds budget")
    if failures:
        raise StructuralWorkError("; ".join(failures))

    process = execution.get("process")
    if not isinstance(process, str) or not process:
        process = execution_path.parent.name
    return StructuralWorkCertificate(
        schema="pyamplicol-structural-work-certificate-v1",
        process=process,
        external_pdg_order=tuple(external),
        legacy=legacy,
        candidate=candidate,
        final_work_comparison=_final_work_comparison(legacy, candidate),
        recurrence_construction=construction,
        limits=limits,
        status="ok",
    )


def adjacent_multiplicity_census(
    lower: StructuralWorkCertificate,
    higher: StructuralWorkCertificate,
    *,
    max_normalized_final_growth: float = MAX_ADJACENT_FINAL_GROWTH,
    max_normalized_peak_growth: float = MAX_ADJACENT_PEAK_GROWTH,
) -> AdjacentMultiplicityCensus:
    lower_n_final = len(lower.external_pdg_order) - 2
    higher_n_final = len(higher.external_pdg_order) - 2
    if higher_n_final != lower_n_final + 1:
        raise StructuralWorkError(
            "structural-work census requires adjacent final-state multiplicities"
        )
    if lower.candidate.mode != higher.candidate.mode:
        raise StructuralWorkError(
            "structural-work census requires one candidate generation mode"
        )
    for value, label in [
        (max_normalized_final_growth, "adjacent final-work growth"),
        (max_normalized_peak_growth, "adjacent construction-peak growth"),
    ]:
        if not math.isfinite(value) or value <= 0.0:
            raise StructuralWorkError(f"{label} limit must be positive and finite")

    legacy_current_growth = (
        higher.legacy.replay_current_count / lower.legacy.replay_current_count
    )
    candidate_current_growth = (
        higher.candidate.current_count / lower.candidate.current_count
    )
    legacy_interaction_growth = (
        higher.legacy.replay_interaction_count / lower.legacy.replay_interaction_count
    )
    candidate_interaction_growth = (
        higher.candidate.interaction_count / lower.candidate.interaction_count
    )
    normalized_current_growth = candidate_current_growth / legacy_current_growth
    normalized_interaction_growth = (
        candidate_interaction_growth / legacy_interaction_growth
    )
    normalized_peak_current_growth = None
    if (
        lower.recurrence_construction is not None
        and higher.recurrence_construction is not None
    ):
        peak_growth = (
            higher.recurrence_construction.peak_current_count
            / lower.recurrence_construction.peak_current_count
        )
        normalized_peak_current_growth = peak_growth / legacy_current_growth

    failures = []
    if normalized_current_growth > max_normalized_final_growth:
        failures.append(
            "adjacent final-current growth exceeds legacy-normalized budget"
        )
    if normalized_interaction_growth > max_normalized_final_growth:
        failures.append(
            "adjacent final-interaction growth exceeds legacy-normalized budget"
        )
    if (
        normalized_peak_current_growth is not None
        and normalized_peak_current_growth > max_normalized_peak_growth
    ):
        failures.append(
            "adjacent construction-peak growth exceeds legacy-normalized budget"
        )
    if failures:
        raise StructuralWorkError("; ".join(failures))

    return AdjacentMultiplicityCensus(
        schema="pyamplicol-adjacent-structural-work-census-v1",
        mode=lower.candidate.mode,
        lower_n_final=lower_n_final,
        higher_n_final=higher_n_final,
        legacy_current_growth=legacy_current_growth,
        candidate_current_growth=candidate_current_growth,
        normalized_current_growth=normalized_current_growth,
        legacy_interaction_growth=legacy_interaction_growth,
        candidate_interaction_growth=candidate_interaction_growth,
        normalized_interaction_growth=normalized_interaction_growth,
        normalized_peak_current_growth=normalized_peak_current_growth,
        limits={
            "max_normalized_final_growth": max_normalized_final_growth,
            "max_normalized_peak_growth": max_normalized_peak_growth,
        },
        status="ok",
    )


def static_template_materialization_census(
    certificate: StructuralWorkCertificate,
    *,
    max_materialization_to_legacy_static: float = (MAX_STATIC_TEMPLATE_MATERIALIZATION),
) -> StaticTemplateMaterializationCensus:
    """Compare artifact materialization with generated legacy templates.

    Legacy dynamic replay is the correct runtime-work comparator, but it can
    hide a severe cold-generation expansion.  This independent census treats
    the generated module inventory as the static baseline and reports both the
    final artifact and recurrence-construction peak.
    """

    limit = float(max_materialization_to_legacy_static)
    if not math.isfinite(limit) or limit <= 0.0:
        raise StructuralWorkError(
            "static-template materialization limit must be positive and finite"
        )
    legacy = certificate.legacy
    if (
        legacy.generated_module_count <= 0
        or legacy.retained_color_ordering_count % legacy.generated_module_count
    ):
        raise StructuralWorkError(
            "legacy color replay count is not an integer multiple of its "
            "generated module inventory"
        )
    static_currents = legacy.module_current_count * legacy.generated_module_count
    static_interactions = (
        legacy.module_interaction_count * legacy.generated_module_count
    )
    candidate = certificate.candidate
    final_current_ratio = candidate.current_count / static_currents
    final_interaction_ratio = candidate.interaction_count / static_interactions
    construction = certificate.recurrence_construction
    peak_currents = None if construction is None else construction.peak_current_count
    peak_interactions = (
        None if construction is None else construction.peak_contribution_count
    )
    peak_current_ratio = (
        None if peak_currents is None else peak_currents / static_currents
    )
    peak_interaction_ratio = (
        None if peak_interactions is None else peak_interactions / static_interactions
    )
    violations = []
    if final_current_ratio > limit:
        violations.append("final currents exceed legacy static-template budget")
    if final_interaction_ratio > limit:
        violations.append("final interactions exceed legacy static-template budget")
    if peak_current_ratio is not None and peak_current_ratio > limit:
        violations.append("peak currents exceed legacy static-template budget")
    if peak_interaction_ratio is not None and peak_interaction_ratio > limit:
        violations.append("peak interactions exceed legacy static-template budget")
    return StaticTemplateMaterializationCensus(
        schema="pyamplicol-static-template-materialization-census-v1",
        mode=candidate.mode,
        legacy_generated_module_count=legacy.generated_module_count,
        legacy_static_current_count=static_currents,
        legacy_static_interaction_count=static_interactions,
        legacy_replay_multiplicity_per_module=(
            legacy.retained_color_ordering_count // legacy.generated_module_count
        ),
        candidate_final_current_count=candidate.current_count,
        candidate_final_interaction_count=candidate.interaction_count,
        candidate_final_current_ratio=final_current_ratio,
        candidate_final_interaction_ratio=final_interaction_ratio,
        candidate_peak_current_count=peak_currents,
        candidate_peak_interaction_count=peak_interactions,
        candidate_peak_current_ratio=peak_current_ratio,
        candidate_peak_interaction_ratio=peak_interaction_ratio,
        limits={"max_materialization_to_legacy_static": limit},
        violations=tuple(violations),
        status="ok" if not violations else "exceeds-budget",
    )


def require_static_template_materialization_parity(
    certificate: StructuralWorkCertificate,
    *,
    max_materialization_to_legacy_static: float = (MAX_STATIC_TEMPLATE_MATERIALIZATION),
) -> StaticTemplateMaterializationCensus:
    """Fail closed unless final and peak materialization have static parity."""

    census = static_template_materialization_census(
        certificate,
        max_materialization_to_legacy_static=(max_materialization_to_legacy_static),
    )
    if census.violations:
        raise StructuralWorkError("; ".join(census.violations))
    return census


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_artifact", type=Path)
    parser.add_argument("candidate_artifact", type=Path)
    parser.add_argument(
        "--max-final-to-legacy",
        type=float,
        default=MAX_UNPROVEN_FINAL_TO_LEGACY,
    )
    parser.add_argument("--max-peak-current-to-legacy", type=float, default=1.5)
    parser.add_argument("--max-peak-to-final-current", type=float, default=4.0)
    parser.add_argument("--max-peak-to-final-contribution", type=float, default=6.0)
    parser.add_argument(
        "--require-static-template-parity",
        action="store_true",
        help=(
            "also reject final/peak materialization above 1.05x the generated "
            "legacy template inventory"
        ),
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    certificate = certify(
        arguments.legacy_artifact,
        arguments.candidate_artifact,
        max_final_to_legacy=arguments.max_final_to_legacy,
        max_peak_current_to_legacy=arguments.max_peak_current_to_legacy,
        max_peak_to_final_current=arguments.max_peak_to_final_current,
        max_peak_to_final_contribution=(arguments.max_peak_to_final_contribution),
    )
    if arguments.require_static_template_parity:
        require_static_template_materialization_parity(certificate)
    print(json.dumps(asdict(certificate), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
