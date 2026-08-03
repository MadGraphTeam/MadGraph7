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
    ROOT / "src/pyamplicol/_profiling_campaign/steer_performance_campaign.py"
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
    entrypoint_main.__globals__["_reexecute_with_repository_python"] = (
        lambda _root, _entrypoint: None
    )
    observed: dict[str, object] = {}

    def fake_campaign_main(
        arguments: list[str],
        *,
        repo_root: Path,
        profile: str,
        docs_dir: Path,
    ) -> int:
        observed.update(
            arguments=arguments,
            repo_root=repo_root,
            profile=profile,
            docs_dir=docs_dir,
        )
        return 17

    monkeypatch.setattr(manual_campaign, "main", fake_campaign_main)
    monkeypatch.setattr(sys, "argv", [str(copied), "inspect"])

    assert entrypoint_main() == 17
    assert observed == {
        "arguments": ["inspect"],
        "repo_root": ROOT,
        "profile": "independent_campaign",
        "docs_dir": copied.parent,
    }


def test_installed_campaign_entrypoint_passes_its_directory_to_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _entrypoint_namespace()
    copied = tmp_path / "independent_campaign" / "steer_performance_campaign.py"
    copied.parent.mkdir()
    entrypoint_main = cast(Callable[[], int], namespace["main"])
    entrypoint_main.__globals__["__file__"] = str(copied)
    observed: dict[str, object] = {}

    class InstalledController:
        @staticmethod
        def main(
            arguments: list[str],
            *,
            repo_root: Path,
            profile: str,
            docs_dir: Path,
            installed: bool,
        ) -> int:
            observed.update(
                arguments=arguments,
                repo_root=repo_root,
                profile=profile,
                docs_dir=docs_dir,
                installed=installed,
            )
            return 19

    entrypoint_main.__globals__["import_module"] = lambda _name: InstalledController
    monkeypatch.setattr(sys, "argv", [str(copied), "inspect"])

    assert entrypoint_main() == 19
    assert observed == {
        "arguments": ["inspect"],
        "repo_root": copied.parent,
        "profile": "independent_campaign",
        "docs_dir": copied.parent,
        "installed": True,
    }


def test_source_launcher_preserves_symlinked_campaign_path_for_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _entrypoint_namespace()
    fake_repo = tmp_path / "checkout"
    (fake_repo / "tools/performance_report").mkdir(parents=True)
    (fake_repo / "src/pyamplicol").mkdir(parents=True)
    profile_parent = fake_repo / "docs/performance_reports"
    profile_parent.mkdir(parents=True)
    actual = tmp_path / "actual-campaign"
    actual.mkdir()
    linked = profile_parent / "linked-campaign"
    linked.symlink_to(actual, target_is_directory=True)
    copied = linked / "steer_performance_campaign.py"
    entrypoint_main = cast(Callable[[], int], namespace["main"])
    entrypoint_main.__globals__["__file__"] = str(copied)
    entrypoint_main.__globals__["_reexecute_with_repository_python"] = (
        lambda _root, _entrypoint: None
    )

    def rejecting_campaign_main(
        _arguments: list[str],
        *,
        repo_root: Path,
        profile: str,
        docs_dir: Path,
    ) -> int:
        assert profile == "linked-campaign"
        assert docs_dir == linked
        with pytest.raises(
            manual_campaign.ManualCampaignError,
            match="symbolic link",
        ):
            manual_campaign._campaign_report_paths(repo_root, docs_dir)
        return 23

    monkeypatch.setattr(manual_campaign, "main", rejecting_campaign_main)
    monkeypatch.setattr(sys, "argv", [str(copied), "inspect"])

    assert entrypoint_main() == 23
