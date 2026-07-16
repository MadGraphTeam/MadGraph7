# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

from pyamplicol.artifacts import (
    ArtifactAliasInspection,
    ArtifactDependencyInspection,
    ArtifactInspection,
    ArtifactProcessInspection,
)
from pyamplicol.cli.main import write_result
from pyamplicol.reporting import render_summary


@dataclass(frozen=True)
class _Result:
    status: str
    generated_processes: int
    adjustments: tuple[str, ...]


def test_human_structured_results_use_aligned_prettytable() -> None:
    rendered = render_summary(_Result("complete", 2, ()), color=False)
    assert rendered is not None
    assert "field" in rendered
    assert "generated processes" in rendered
    assert "complete" in rendered


def test_json_result_never_contains_table_or_color_sequences() -> None:
    stream = io.StringIO()
    write_result(_Result("complete", 2, ()), format="json", stream=stream, color=True)
    assert json.loads(stream.getvalue()) == {
        "adjustments": [],
        "generated_processes": 2,
        "status": "complete",
    }
    assert "\x1b[" not in stream.getvalue()


def _artifact_inspection() -> ArtifactInspection:
    alias = ArtifactAliasInspection(
        id="ddbar_zg_alias",
        expression="d d~ > g z",
        representative_id="ddbar_zg",
        external_pdgs=(1, -1, 21, 23),
    )
    process = ArtifactProcessInspection(
        id="ddbar_zg",
        expression="d d~ > z g",
        color_accuracy="lc",
        external_pdgs=(1, -1, 23, 21),
        default=True,
        physical_helicities=24,
        computed_helicities=12,
        physical_color_components=1,
        computed_color_components=1,
        helicity_coverage="complete",
        color_coverage="complete",
        aliases=(alias,),
    )
    return ArtifactInspection(
        kind="pyamplicol-artifact-inspection",
        path=Path("/tmp/artifact"),
        artifact_kind="pyamplicol-process-set",
        artifact_id="a" * 64,
        created_utc="2026-07-16T00:00:00Z",
        producer_version="0.1.0",
        target="x86_64-unknown-linux-gnu",
        cpu_features=(),
        model_name="sm",
        model_source="ufo-json",
        model_restriction="default",
        default_process_id="ddbar_zg",
        runtime_engine="rusticol",
        runtime_version="0.1.0",
        runtime_capabilities=("symjit.application.complex-f64.v1",),
        payload_count=12,
        payload_size_bytes=2048,
        integrity="verified",
        processes=(process,),
        dependencies=(
            ArtifactDependencyInspection(
                name="symbolica",
                version="2.1.0",
                license="Symbolica Software License Agreement",
                source="https://symbolica.io/",
            ),
        ),
    )


def test_artifact_inspection_uses_process_and_alias_tables() -> None:
    rendered = render_summary(_artifact_inspection(), color=False)

    assert rendered is not None
    assert "Artifact" in rendered
    assert "Processes" in rendered
    assert "Aliases" in rendered
    assert "Dependencies" in rendered
    assert "ddbar_zg" in rendered
    assert "d d~ > z g" in rendered
    assert "24 (12 eval.)" in rendered


def test_artifact_inspection_color_is_optional() -> None:
    rendered = render_summary(_artifact_inspection(), color=True)

    assert rendered is not None
    assert "\x1b[" in rendered
