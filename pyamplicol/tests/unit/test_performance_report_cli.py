# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

from tools.performance_report.cli import _parser, main


def test_table_filler_defaults_to_five_seconds_per_cell() -> None:
    populate = _parser().parse_args(("populate",))
    assert populate.target_runtime == 5.0

    worker = _parser().parse_args(
        (
            "_worker",
            "--cell-id",
            "cell",
            "--attempt-root",
            "attempt",
            "--result-json",
            "result.json",
        )
    )
    assert worker.target_runtime == 5.0


def test_report_profile_is_a_global_architecture_scope() -> None:
    parsed = _parser().parse_args(
        ("--report-profile", "macbook_M3", "populate")
    )
    assert parsed.report_profile == "macbook_M3"


def test_reset_and_validate_cli_use_new_service(tmp_path: Path, capsys) -> None:
    repo = tmp_path / "repo"
    (repo / "docs/results").mkdir(parents=True)

    assert main(("--repo-root", str(repo), "reset")) == 0
    reset_output = capsys.readouterr().out
    assert "docs/result_matrix_recurrence_builtin_sm_lc_table.tex" in reset_output

    assert main(("--repo-root", str(repo), "validate")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["table_count"] == 17
    assert payload["cache_count"] > 12

    assert main(("--repo-root", str(repo), "audit")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cache_render_match"]


def test_populate_dry_run_supports_exact_filters_and_dependencies(
    tmp_path: Path,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    (repo / "docs/results").mkdir(parents=True)
    main(("--repo-root", str(repo), "reset"))
    capsys.readouterr()

    assert (
        main(
            (
                "--repo-root",
                str(repo),
                "populate",
                "--dataset",
                "matrix_compiled_builtin_sm_lc",
                "--process-key",
                "dd_z_jets",
                "--n-final",
                "1",
                "--workload",
                "selected-flow",
                "--dry-run",
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["requested"] == 1
    assert payload["scheduled"] == 3
    assert [cell["rank"] for cell in payload["cells"]] == [0, 1, 2]
