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
class RecurrenceConstructionWork:
    peak_current_count: int
    peak_contribution_attempt_count: int
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
    recurrence_construction: RecurrenceConstructionWork | None
    limits: dict[str, float]
    status: str


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
            peak_contribution_attempt_count=peak_contributions,
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
        interactions = _positive_integer(
            summary, "interaction_count", "compiled DAG"
        )
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
        interactions = _positive_integer(
            summary, "attachment_count", "eager plan"
        )
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


def certify(
    legacy_artifact: Path,
    candidate_artifact: Path,
    *,
    max_final_to_legacy: float = 1.0,
    max_peak_current_to_legacy: float = 1.5,
    max_peak_to_final_current: float = 4.0,
    max_peak_to_final_contribution: float = 6.0,
) -> StructuralWorkCertificate:
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
            failures.append("peak/final contribution-attempt ratio exceeds budget")
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
        recurrence_construction=construction,
        limits=limits,
        status="ok",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_artifact", type=Path)
    parser.add_argument("candidate_artifact", type=Path)
    parser.add_argument("--max-final-to-legacy", type=float, default=1.0)
    parser.add_argument("--max-peak-current-to-legacy", type=float, default=1.5)
    parser.add_argument("--max-peak-to-final-current", type=float, default=4.0)
    parser.add_argument("--max-peak-to-final-contribution", type=float, default=6.0)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    certificate = certify(
        arguments.legacy_artifact,
        arguments.candidate_artifact,
        max_final_to_legacy=arguments.max_final_to_legacy,
        max_peak_current_to_legacy=arguments.max_peak_current_to_legacy,
        max_peak_to_final_current=arguments.max_peak_to_final_current,
        max_peak_to_final_contribution=(
            arguments.max_peak_to_final_contribution
        ),
    )
    print(json.dumps(asdict(certificate), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
