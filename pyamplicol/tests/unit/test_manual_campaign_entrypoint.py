# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import tools.performance_report.manual_campaign as manual_campaign

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = (
    ROOT / "docs/performance_reports/macbook_M3_manual/steer_performance_campaign.py"
)


def _entrypoint_namespace() -> dict[str, object]:
    return runpy.run_path(
        str(ENTRYPOINT),
        run_name="manual_campaign_entrypoint_test",
    )


def test_campaign_profile_is_derived_from_copied_directory_name() -> None:
    derive = cast(
        Callable[[Path, Path], str],
        _entrypoint_namespace()["_embedded_profile"],
    )
    copied = (
        ROOT
        / "docs/performance_reports/another_macbook_campaign"
        / "steer_performance_campaign.py"
    )

    assert derive(copied, ROOT) == "another_macbook_campaign"


def test_campaign_entrypoint_rejects_nested_non_profile_directory() -> None:
    derive = cast(
        Callable[[Path, Path], str],
        _entrypoint_namespace()["_embedded_profile"],
    )
    nested = (
        ROOT
        / "docs/performance_reports/campaign/archive"
        / "steer_performance_campaign.py"
    )

    with pytest.raises(RuntimeError, match="directly inside"):
        derive(nested, ROOT)


def test_campaign_entrypoint_passes_copied_profile_to_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _entrypoint_namespace()
    copied = (
        ROOT
        / "docs/performance_reports/independent_campaign"
        / "steer_performance_campaign.py"
    )
    entrypoint_main = cast(Callable[[], int], namespace["main"])
    entrypoint_main.__globals__["__file__"] = str(copied)
    entrypoint_main.__globals__["_reexecute_with_repository_python"] = lambda _root: (
        None
    )
    observed: dict[str, object] = {}

    def fake_campaign_main(
        arguments: list[str],
        *,
        repo_root: Path,
        profile: str,
    ) -> int:
        observed.update(
            arguments=arguments,
            repo_root=repo_root,
            profile=profile,
        )
        return 17

    monkeypatch.setattr(manual_campaign, "main", fake_campaign_main)
    monkeypatch.setattr(sys, "argv", [str(copied), "inspect"])

    assert entrypoint_main() == 17
    assert observed == {
        "arguments": ["inspect"],
        "repo_root": ROOT,
        "profile": "independent_campaign",
    }
