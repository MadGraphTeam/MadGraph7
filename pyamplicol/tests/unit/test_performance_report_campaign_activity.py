# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.performance_report.campaign_activity import (
    CampaignActivityError,
    blocking_coordination_files,
    blocking_process_lines,
    campaign_activity,
    require_campaign_idle,
)


def test_publisher_render_processes_are_not_campaign_workers(tmp_path: Path) -> None:
    entrypoint = tmp_path / "result_tables.py"
    process_output = "\n".join(
        (
            (
                "94622 Python /usr/bin/python "
                f"{entrypoint} publish-snapshot --watch"
            ),
            "94623 latexmk /usr/bin/latexmk -pdf pyAmpliCol.tex",
            f"94624 Python /usr/bin/python {entrypoint} populate --workers 1",
            "94625 rustc /usr/bin/rustc --crate-name pyamplicol",
        )
    )

    assert blocking_process_lines(
        process_output,
        entrypoints=(entrypoint,),
    ) == (
        f"94624 Python /usr/bin/python {entrypoint} populate --workers 1",
        "94625 rustc /usr/bin/rustc --crate-name pyamplicol",
    )


def test_only_regular_publisher_private_coordination_files_are_ignored(
    tmp_path: Path,
) -> None:
    coordination_root = tmp_path / "coordination"
    publication = coordination_root / "publication"
    named = coordination_root / "named"
    lsof_output = "\n".join(
        (
            "p94622",
            "cPython",
            "f3u",
            "tREG",
            f"n{publication / 'daemon.guard'}",
            "f4r",
            "tREG",
            f"n{publication / 'daemon.json'}",
            "f5r",
            "tREG",
            f"n{publication / 'snapshot-staging' / 'current.json'}",
            "f6r",
            "tDIR",
            f"n{publication}",
            "p94624",
            "cPython",
            "f3u",
            "tREG",
            f"n{named / 'report-writer-deadbeef.lock'}",
            "f4u",
            "tREG",
            f"n{named / 'cell-example-deadbeef.lock'}",
        )
    )

    blocked = blocking_coordination_files(
        lsof_output,
        coordination_root=coordination_root,
    )

    assert [record.path for record in blocked] == [
        publication,
        named / "report-writer-deadbeef.lock",
        named / "cell-example-deadbeef.lock",
    ]


def test_idle_census_ignores_publisher_but_rejects_report_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordination_root = tmp_path / "coordination"
    publication = coordination_root / "publication"
    named = coordination_root / "named"
    responses = iter(
        (
            subprocess.CompletedProcess(
                args=("ps",),
                returncode=0,
                stdout=(
                    "94622 Python /usr/bin/python result_tables.py "
                    "publish-snapshot --watch\n"
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=("lsof",),
                returncode=0,
                stdout="\n".join(
                    (
                        "p94622",
                        "cPython",
                        "f3u",
                        "tREG",
                        f"n{publication / 'daemon.guard'}",
                    )
                ),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(
        "tools.performance_report.campaign_activity.subprocess.run",
        lambda *_args, **_kwargs: next(responses),
    )

    assert campaign_activity(coordination_root=coordination_root).idle

    responses = iter(
        (
            subprocess.CompletedProcess(
                args=("ps",),
                returncode=0,
                stdout="",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=("lsof",),
                returncode=0,
                stdout="\n".join(
                    (
                        "p94624",
                        "cPython",
                        "f3u",
                        "tREG",
                        f"n{named / 'report-writer-deadbeef.lock'}",
                    )
                ),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(
        "tools.performance_report.campaign_activity.subprocess.run",
        lambda *_args, **_kwargs: next(responses),
    )

    with pytest.raises(CampaignActivityError, match="report-writer"):
        require_campaign_idle(coordination_root=coordination_root)
