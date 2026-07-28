from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.developer.structural_work_certificate import (
    StructuralWorkError,
    adjacent_multiplicity_census,
    certify,
)


def _legacy(
    root: Path,
    *,
    module_currents: int = 406,
    module_interactions: int = 2440,
    retained_colors: int = 720,
) -> Path:
    library = root / "contracted-generated-library" / "Library"
    library.mkdir(parents=True)
    probe = (
        root
        / "contracted-generated-library"
        / "amplicol_color_library_probe.output"
    )
    probe.write_text(
        "Total number of currents, vertices and amplitudes after filter"
        f" 1597 4260 {retained_colors}\n"
    )
    for index in range(2):
        (library / f"amp{index + 1}_1_lib.f03").write_text(
            f"complex(kind=8),dimension(1:6,{module_currents}) :: val_c\n"
            f"complex(kind=8),dimension(1:6,{module_interactions}) :: int_c\n"
        )
    return root


def _candidate(
    root: Path,
    payload: dict[str, object],
    *,
    external_count: int = 7,
) -> Path:
    process = root / "processes" / "g_g_to_g_g_g_g_g"
    process.mkdir(parents=True)
    common = {
        "external_pdg_order": [21] * external_count,
        "color_accuracy": "full",
        "process": "g_g_to_g_g_g_g_g",
    }
    (process / "execution.json").write_text(json.dumps(common | payload))
    return root


@pytest.mark.parametrize(
    "payload,mode",
    [
        (
            {
                "kind": "pyamplicol-runtime-recurrence-execution",
                "plan": {
                    "inspection_summary": {
                        "schedule": {
                            "current_count": 101_942,
                            "contribution_count": 955_368,
                        },
                        "construction": {
                            "peak_current_count": 372_422,
                            "peak_contribution_count": 4_868_016,
                        },
                    }
                },
            },
            "recurrence",
        ),
        (
            {
                "kind": "pyamplicol-runtime-execution",
                "helicity_sum_execution": {
                    "dag_summary": {
                        "current_count": 90_000,
                        "interaction_count": 900_000,
                    }
                },
            },
            "compiled",
        ),
        (
            {
                "kind": "pyamplicol-runtime-eager-execution",
                "plan": {
                    "inspection_summary": {
                        "current_count": 90_000,
                        "attachment_count": 900_000,
                    }
                },
            },
            "eager",
        ),
    ],
)
def test_certifies_all_generation_modes(
    tmp_path: Path,
    payload: dict[str, object],
    mode: str,
) -> None:
    certificate = certify(
        _legacy(tmp_path / "legacy"),
        _candidate(tmp_path / "candidate", payload),
    )
    assert certificate.status == "ok"
    assert certificate.candidate.mode == mode
    assert certificate.legacy.replay_current_count == 406 * 720
    assert certificate.legacy.replay_interaction_count == 2440 * 720
    assert (
        certificate.final_work_comparison.classification
        == "proven-recycling"
    )
    assert certificate.final_work_comparison.current_savings_fraction > 0.0


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kind": "pyamplicol-runtime-recurrence-execution",
            "plan": {
                "inspection_summary": {
                    "schedule": {
                        "current_count": int(406 * 720 * 1.05) + 1,
                        "contribution_count": 2440 * 720,
                    },
                    "construction": {
                        "peak_current_count": 406 * 720 + 1,
                        "peak_contribution_count": 2440 * 720,
                    },
                }
            },
        },
        {
            "kind": "pyamplicol-runtime-execution",
            "helicity_sum_execution": {
                "dag_summary": {
                    "current_count": int(406 * 720 * 1.05) + 1,
                    "interaction_count": 2440 * 720,
                }
            },
        },
        {
            "kind": "pyamplicol-runtime-eager-execution",
            "plan": {
                "inspection_summary": {
                    "current_count": int(406 * 720 * 1.05) + 1,
                    "attachment_count": 2440 * 720,
                }
            },
        },
    ],
)
def test_rejects_final_work_above_legacy_replay(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    with pytest.raises(
        StructuralWorkError,
        match="final current work exceeds legacy replay",
    ):
        certify(
            _legacy(tmp_path / "legacy"),
            _candidate(tmp_path / "candidate", payload),
        )


def test_accepts_only_bounded_unproven_structural_tolerance(
    tmp_path: Path,
) -> None:
    replay_currents = 406 * 720
    certificate = certify(
        _legacy(tmp_path / "legacy"),
        _candidate(
            tmp_path / "candidate",
            {
                "kind": "pyamplicol-runtime-execution",
                "helicity_sum_execution": {
                    "dag_summary": {
                        "current_count": replay_currents + 1,
                        "interaction_count": 2440 * 720,
                    }
                },
            },
        ),
    )
    assert (
        certificate.final_work_comparison.classification
        == "within-structural-tolerance"
    )
    with pytest.raises(StructuralWorkError, match="unreviewed final-work budget"):
        certify(
            _legacy(tmp_path / "legacy-too-wide"),
            _candidate(
                tmp_path / "candidate-too-wide",
                {
                    "kind": "pyamplicol-runtime-execution",
                    "helicity_sum_execution": {
                        "dag_summary": {
                            "current_count": 1,
                            "interaction_count": 1,
                        }
                    },
                },
            ),
            max_final_to_legacy=1.051,
        )


def test_adjacent_multiplicity_census_normalizes_candidate_growth(
    tmp_path: Path,
) -> None:
    lower = certify(
        _legacy(
            tmp_path / "legacy-n4",
            module_currents=188,
            module_interactions=816,
            retained_colors=120,
        ),
        _candidate(
            tmp_path / "candidate-n4",
            {
                "kind": "pyamplicol-runtime-recurrence-execution",
                "plan": {
                    "inspection_summary": {
                        "schedule": {
                            "current_count": 8_372,
                            "contribution_count": 57_720,
                        },
                        "construction": {
                            "peak_current_count": 26_052,
                            "peak_contribution_count": 255_600,
                        },
                    }
                },
            },
            external_count=6,
        ),
    )
    higher = certify(
        _legacy(tmp_path / "legacy-n5"),
        _candidate(
            tmp_path / "candidate-n5",
            {
                "kind": "pyamplicol-runtime-recurrence-execution",
                "plan": {
                    "inspection_summary": {
                        "schedule": {
                            "current_count": 101_942,
                            "contribution_count": 955_368,
                        },
                        "construction": {
                            "peak_current_count": 372_422,
                            "peak_contribution_count": 4_868_016,
                        },
                    }
                },
            },
        ),
    )
    census = adjacent_multiplicity_census(lower, higher)
    assert census.status == "ok"
    assert census.mode == "recurrence"
    assert (census.lower_n_final, census.higher_n_final) == (4, 5)
    assert census.normalized_current_growth < 1.0
    assert census.normalized_interaction_growth < 1.0
    assert census.normalized_peak_current_growth is not None
    assert census.normalized_peak_current_growth < 1.25


def test_rejects_recurrence_construction_inflation(tmp_path: Path) -> None:
    payload = {
        "kind": "pyamplicol-runtime-recurrence-execution",
        "plan": {
            "inspection_summary": {
                "schedule": {
                    "current_count": 10,
                    "contribution_count": 10,
                },
                "construction": {
                    "peak_current_count": 100,
                    "peak_contribution_count": 100,
                },
            }
        },
    }
    with pytest.raises(
        StructuralWorkError,
        match="peak/final construction current ratio exceeds budget",
    ):
        certify(
            _legacy(tmp_path / "legacy"),
            _candidate(tmp_path / "candidate", payload),
        )


def test_rejects_nonuniform_legacy_modules(tmp_path: Path) -> None:
    legacy = _legacy(tmp_path / "legacy")
    module = legacy / "contracted-generated-library" / "Library" / "amp2_1_lib.f03"
    module.write_text(
        "complex(kind=8),dimension(1:6,407) :: val_c\n"
        "complex(kind=8),dimension(1:6,2440) :: int_c\n"
    )
    candidate = _candidate(
        tmp_path / "candidate",
        {
            "kind": "pyamplicol-runtime-eager-execution",
            "plan": {
                "inspection_summary": {
                    "current_count": 1,
                    "attachment_count": 1,
                }
            },
        },
    )
    with pytest.raises(StructuralWorkError, match="do not share one replay shape"):
        certify(legacy, candidate)
