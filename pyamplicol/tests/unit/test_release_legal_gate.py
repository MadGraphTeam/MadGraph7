# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))

import build_release_artifacts  # noqa: E402
import publish_dry_run  # noqa: E402
from _common import ReleaseError  # noqa: E402


@pytest.mark.parametrize(
    "release_module",
    [build_release_artifacts, publish_dry_run],
    ids=("artifact-build", "publish-dry-run"),
)
@pytest.mark.parametrize("mode", ["release", "candidate"])
def test_release_entry_points_invoke_the_normative_legal_checker(
    release_module,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, env, **_kwargs):
        observed.update(
            command=[os.fspath(part) for part in command],
            cwd=cwd,
            env=env,
        )

    monkeypatch.setattr(release_module, "run", fake_run)
    release_module._check_legal_gate(mode)

    assert observed["command"] == [
        sys.executable,
        os.fspath(release_module.LEGAL_GATE),
        "--mode",
        mode,
    ]
    assert release_module.LEGAL_GATE.name == "check_legal_inventory.py"
    assert observed["cwd"] == ROOT
    environment = observed["env"]
    assert isinstance(environment, dict)
    assert environment["PYAMPLICOL_BUILD_MODE"] == mode
    output = capsys.readouterr().out
    assert ("NON-PUBLISHABLE CANDIDATE" in output) is (mode == "candidate")


def test_strict_legal_failure_prevents_artifact_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "release"

    def closed_gate(mode: str) -> None:
        assert mode == "release"
        raise ReleaseError("static LGPL compliance unresolved")

    def unexpected(*_args, **_kwargs) -> None:
        pytest.fail("artifact work proceeded after the legal gate failed")

    monkeypatch.setattr(build_release_artifacts, "_check_legal_gate", closed_gate)
    monkeypatch.setattr(build_release_artifacts, "check_dependency_gate", unexpected)
    monkeypatch.setattr(build_release_artifacts, "require_clean_checkout", unexpected)
    monkeypatch.setattr(build_release_artifacts, "_build", unexpected)

    with pytest.raises(ReleaseError, match="static LGPL compliance unresolved"):
        build_release_artifacts.build_release_artifacts(
            output,
            mode="release",
            python=Path(sys.executable),
            allow_dirty_candidate=False,
            sdist_only=False,
            retained_sdist_path=None,
        )
    assert not output.exists()


def test_candidate_artifact_listing_remains_explicitly_non_publishable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "pyamplicol-candidate.whl"
    observed: dict[str, object] = {}

    def fake_build(output, **kwargs):
        observed.update(output=output, **kwargs)
        return [artifact]

    monkeypatch.delenv("PYAMPLICOL_BUILD_MODE", raising=False)
    monkeypatch.setattr(build_release_artifacts, "build_release_artifacts", fake_build)

    assert build_release_artifacts.main(["--candidate"]) == 0
    assert observed["mode"] == "candidate"
    output = capsys.readouterr().out
    assert "Retained NON-PUBLISHABLE candidate artifacts:" in output
    assert f"  {artifact}" in output


def test_strict_legal_failure_prevents_publish_dry_run_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def closed_gate(mode: str) -> None:
        assert mode == "release"
        raise ReleaseError("static LGPL compliance unresolved")

    def unexpected(*_args, **_kwargs) -> None:
        pytest.fail("publish dry-run work proceeded after the legal gate failed")

    monkeypatch.delenv("PYAMPLICOL_BUILD_MODE", raising=False)
    monkeypatch.setattr(publish_dry_run, "_check_legal_gate", closed_gate)
    monkeypatch.setattr(publish_dry_run, "check_dependency_gate", unexpected)
    monkeypatch.setattr(publish_dry_run, "verify_manifest", unexpected)
    monkeypatch.setattr(publish_dry_run, "collect_unique_artifacts", unexpected)

    with pytest.raises(ReleaseError, match="static LGPL compliance unresolved"):
        publish_dry_run.main(
            ["--artifact-dir", str(tmp_path), "--no-build", "--skip-twine-check"]
        )
