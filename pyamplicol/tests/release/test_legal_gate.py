# SPDX-License-Identifier: 0BSD

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_release_workflow_blocks_artifacts_on_the_strict_legal_gate() -> None:
    workflow = (WORKFLOWS / "release-artifacts.yml").read_text(encoding="utf-8")
    assert "legal-release-gate:" in workflow
    assert "name: Native dependency legal gate" in workflow
    assert "python tools/release/check_legal_inventory.py --mode release" in workflow
    assert "retained-sdist:\n    needs: legal-release-gate" in workflow
    assert workflow.index("Enforce native dependency legal gate") < workflow.index(
        "Build the one retained sdist"
    )


def test_publisher_requires_the_source_runs_strict_legal_gate() -> None:
    workflow = (WORKFLOWS / "publish-pypi.yml").read_text(encoding="utf-8")
    verification = workflow.index("jobs?per_page=100")
    download = workflow.index("Download the previously validated bundle")
    assert verification < download
    assert 'job["name"] == "Native dependency legal gate"' in workflow
    assert 'legal_gate["conclusion"] == "success"' in workflow
    assert 'step["name"] == "Enforce native dependency legal gate"' in workflow
    assert 'enforcement_steps[0]["conclusion"] == "success"' in workflow


def test_just_release_targets_force_the_strict_gate_and_release_mode() -> None:
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    artifact_target, publish_target = justfile.split("publish-dry-run:", maxsplit=1)
    artifact_target = artifact_target.rsplit("release-artifacts:", maxsplit=1)[1]
    strict_gate = (
        "PYAMPLICOL_BUILD_MODE=release {{python}} "
        "tools/release/check_legal_inventory.py --mode release"
    )
    assert strict_gate in artifact_target
    assert strict_gate in publish_target
    assert (
        "PYAMPLICOL_BUILD_MODE=release {{python}} "
        "tools/release/build_release_artifacts.py" in artifact_target
    )
    assert (
        "PYAMPLICOL_BUILD_MODE=release {{python}} "
        "tools/release/publish_dry_run.py" in publish_target
    )


def test_candidate_workflow_labels_artifacts_non_publishable() -> None:
    workflow = (WORKFLOWS / "candidate.yml").read_text(encoding="utf-8").lower()
    assert workflow.count("non-publishable candidate") >= 2
    assert "id-token: write" not in workflow
