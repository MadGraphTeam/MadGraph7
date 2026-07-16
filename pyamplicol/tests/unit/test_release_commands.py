# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))

import build_from_sdist  # noqa: E402
import build_release_artifacts  # noqa: E402
import install_wheel  # noqa: E402
import publish_dry_run  # noqa: E402
import test_deployment as release_deployment  # noqa: E402
from _common import ReleaseError  # noqa: E402


def _dependency_wheel(
    directory: Path,
    distribution: str,
    version: str,
    tag: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    normalized = distribution.replace("-", "_")
    wheel = directory / f"{normalized}-{version}-{tag}.whl"
    dist_info = f"{normalized}-{version}.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.4\nName: {distribution}\nVersion: {version}\n\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            f"Wheel-Version: 1.0\nTag: {tag}\n\n",
        )
    return wheel


def test_select_compatible_abi3_wheel_uses_target_tag_order(tmp_path: Path) -> None:
    mac = tmp_path / "pyamplicol-0.1.0-cp311-abi3-macosx_11_0_arm64.whl"
    linux = tmp_path / "pyamplicol-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl"
    mac.touch()
    linux.touch()
    selected = install_wheel.select_compatible_wheel(
        [linux, mac],
        [
            "cp314-cp314-macosx_15_0_arm64",
            "cp311-abi3-macosx_11_0_arm64",
            "py3-none-any",
        ],
    )
    assert selected == mac.resolve()


@pytest.mark.parametrize(
    ("mode", "candidate_flag"),
    [("candidate", True), ("release", False)],
)
def test_install_builds_through_release_artifact_tool_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    candidate_flag: bool,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, env, **_kwargs):
        observed.update(command=list(map(os.fspath, command)), cwd=cwd, env=env)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("PYTHONPATH", "/parent/source")
    monkeypatch.setattr(install_wheel, "run", fake_run)
    install_wheel.build_if_missing(mode, tmp_path)
    command = observed["command"]
    assert isinstance(command, list)
    assert command[1].endswith("build_release_artifacts.py")
    assert ("--candidate" in command) is candidate_flag
    assert observed["cwd"] == ROOT
    environment = observed["env"]
    assert isinstance(environment, dict)
    assert "PYTHONPATH" not in environment


def test_deployment_path_guard_allows_only_the_isolated_sandbox(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    sandbox = checkout / "PYPI_DEPLOYMENT_TEST" / "release-1"
    site_packages = sandbox / "venv" / "lib" / "site-packages"
    source = checkout / "src"
    environment = {
        **os.environ,
        "PYAMPLICOL_FORBIDDEN_ROOT": str(checkout),
        "PYAMPLICOL_DEPLOYMENT_SANDBOX": str(sandbox),
    }

    def guarded_path(path: Path) -> subprocess.CompletedProcess[str]:
        script = (
            "import os\nfrom pathlib import Path\nimport sys\n"
            f"sys.path[:] = [{str(path)!r}]\n"
            + release_deployment._PATH_ISOLATION_SMOKE
        )
        return subprocess.run(
            [sys.executable, "-I", "-c", script],
            env=environment,
            capture_output=True,
            text=True,
        )

    assert guarded_path(site_packages).returncode == 0
    assert guarded_path(source).returncode != 0


def test_candidate_deployment_installs_patched_wheels_by_exact_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    symbolica = _dependency_wheel(
        wheelhouse / "symbolica",
        "symbolica",
        "2.1.0",
        "cp311-abi3-test_platform",
    )
    loader = _dependency_wheel(
        wheelhouse / "loader",
        "ufo-model-loader",
        "0.1.7",
        "py3-none-any",
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append([os.fspath(item) for item in command])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(release_deployment, "run", fake_run)
    virtual_env = tmp_path / "venv"
    installation = release_deployment._install_dependencies(
        Path(sys.executable),
        virtual_env=virtual_env,
        mode="candidate",
        wheelhouses=[wheelhouse],
        supported_tags=["cp311-abi3-test_platform", "py3-none-any"],
    )

    assert installation.local_wheels == {
        "symbolica": symbolica.resolve(),
        "ufo-model-loader": loader.resolve(),
    }
    assert len(commands) == 1
    command = commands[0]
    assert "--require-hashes" in command
    assert "--find-links" not in command
    assert "--index-url" in command
    requirements = (tmp_path / "locked-requirements.txt").read_text(encoding="utf-8")
    assert symbolica.resolve().as_uri() in requirements
    assert loader.resolve().as_uri() in requirements
    assert hashlib.sha256(symbolica.read_bytes()).hexdigest() in requirements
    assert hashlib.sha256(loader.read_bytes()).hexdigest() in requirements
    assert "numpy==2.4.2" in requirements
    assert "python-utils==4.0.0" in requirements
    assert "typing-extensions==4.16.0" in requirements
    assert "wcwidth==0.8.2" in requirements


def test_candidate_deployment_rejects_ambiguous_local_patched_wheels(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        _dependency_wheel(
            directory,
            "symbolica",
            "2.1.0",
            "cp311-abi3-test_platform",
        )
    _dependency_wheel(first, "ufo-model-loader", "0.1.7", "py3-none-any")
    with pytest.raises(ReleaseError, match="expected one compatible local symbolica"):
        release_deployment._candidate_dependency_wheels(
            {"symbolica": "2.1.0", "ufo-model-loader": "0.1.7"},
            [first, second],
            ["cp311-abi3-test_platform", "py3-none-any"],
        )


def test_candidate_deployment_rejects_symlinked_wheelhouse(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    linked = tmp_path / "linked-wheelhouse"
    linked.symlink_to(wheelhouse, target_is_directory=True)
    with pytest.raises(ReleaseError, match="wheelhouse may not be a symlink"):
        release_deployment._candidate_dependency_wheels(
            {"symbolica": "2.1.0", "ufo-model-loader": "0.1.7"},
            [linked],
            ["py3-none-any"],
        )


def test_sdist_build_uses_clean_external_source_and_replaces_direct_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_wheel = tmp_path / "pyamplicol-0.1.0-cp311-abi3-test.whl"
    source_wheel.write_bytes(b"source wheel")
    sdist = tmp_path / "pyamplicol-0.1.0.tar.gz"
    sdist.write_bytes(b"sdist")
    scratch = tmp_path / "external"
    extracted = scratch / "unpacked" / "pyamplicol-0.1.0"
    extracted.mkdir(parents=True)
    observed: dict[str, object] = {}

    @contextlib.contextmanager
    def fake_temporary(_prefix: str):
        scratch.mkdir(exist_ok=True)
        yield scratch

    def fake_run(command, *, cwd, env, **_kwargs):
        observed["cwd"] = cwd
        observed["env"] = env
        output = Path(command[command.index("--outdir") + 1])
        output.mkdir(exist_ok=True)
        shutil.copy2(source_wheel, output / source_wheel.name)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("PYTHONPATH", "/parent/source")
    monkeypatch.setattr(build_from_sdist, "audit_sdist", lambda *_a, **_k: None)
    monkeypatch.setattr(build_from_sdist, "audit_wheel", lambda *_a, **_k: None)
    monkeypatch.setattr(build_from_sdist, "compare_wheels", lambda *_a: None)
    monkeypatch.setattr(
        build_from_sdist, "external_temporary_directory", fake_temporary
    )
    monkeypatch.setattr(build_from_sdist, "safe_extract_sdist", lambda *_a: extracted)
    monkeypatch.setattr(build_from_sdist, "run", fake_run)
    output = tmp_path / "retained"
    rebuilt = build_from_sdist.build_wheel_from_sdist(
        sdist,
        source_wheel,
        output,
        mode="release",
        python=Path(sys.executable),
    )
    assert rebuilt.read_bytes() == b"source wheel"
    assert observed["cwd"] == extracted
    assert "PYTHONPATH" not in observed["env"]


def test_release_gate_failure_prevents_artifact_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = False

    monkeypatch.setattr(
        build_release_artifacts, "require_clean_checkout", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        build_release_artifacts, "_check_legal_gate", lambda _mode: None
    )

    def closed_gate(*_args, **_kwargs):
        raise ReleaseError("closed by check_dependencies")

    def unexpected_build(*_args, **_kwargs):
        nonlocal built
        built = True

    monkeypatch.setattr(build_release_artifacts, "check_dependency_gate", closed_gate)
    monkeypatch.setattr(build_release_artifacts, "_build", unexpected_build)
    with pytest.raises(ReleaseError, match="check_dependencies"):
        build_release_artifacts.build_release_artifacts(
            tmp_path / "output",
            mode="release",
            python=Path(sys.executable),
            allow_dirty_candidate=False,
            sdist_only=False,
            retained_sdist_path=None,
        )
    assert built is False


def test_publish_dry_run_prints_but_never_executes_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wheel = tmp_path / "pyamplicol-0.1.0-cp311-abi3-test.whl"
    sdist = tmp_path / "pyamplicol-0.1.0.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    payload = {
        "artifacts": [
            {"filename": wheel.name, "sha256": "a" * 64},
            {"filename": sdist.name, "sha256": "b" * 64},
        ]
    }
    commands: list[list[str]] = []

    monkeypatch.delenv("PYAMPLICOL_BUILD_MODE", raising=False)
    monkeypatch.setattr(publish_dry_run, "_check_legal_gate", lambda _mode: None)
    monkeypatch.setattr(
        publish_dry_run, "check_dependency_gate", lambda *_a, **_k: None
    )
    monkeypatch.setattr(publish_dry_run, "verify_manifest", lambda *_a, **_k: payload)
    monkeypatch.setattr(
        publish_dry_run,
        "collect_unique_artifacts",
        lambda *_a: [wheel, sdist],
    )
    monkeypatch.setattr(publish_dry_run, "audit_wheel", lambda *_a, **_k: None)
    monkeypatch.setattr(publish_dry_run, "audit_sdist", lambda *_a, **_k: None)

    def fake_run(command, **_kwargs):
        commands.append([os.fspath(item) for item in command])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(publish_dry_run, "run", fake_run)
    assert (
        publish_dry_run.main(
            ["--artifact-dir", str(tmp_path), "--no-build", "--skip-twine-check"]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "twine upload" in output
    assert commands == []


def test_candidate_dry_run_accepts_wheel_only_and_withholds_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wheel = tmp_path / "pyamplicol-candidate-cp311-abi3-test.whl"
    wheel.write_bytes(b"candidate wheel")
    payload = {"artifacts": [{"filename": wheel.name, "sha256": "c" * 64}]}
    monkeypatch.delenv("PYAMPLICOL_BUILD_MODE", raising=False)
    monkeypatch.setattr(publish_dry_run, "_check_legal_gate", lambda _mode: None)
    monkeypatch.setattr(
        publish_dry_run, "check_dependency_gate", lambda *_a, **_k: None
    )
    monkeypatch.setattr(publish_dry_run, "verify_manifest", lambda *_a, **_k: payload)
    monkeypatch.setattr(
        publish_dry_run, "collect_unique_artifacts", lambda *_a: [wheel]
    )
    monkeypatch.setattr(publish_dry_run, "audit_wheel", lambda *_a, **_k: None)
    assert (
        publish_dry_run.main(
            [
                "--candidate",
                "--artifact-dir",
                str(tmp_path),
                "--no-build",
                "--skip-twine-check",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "non-publishable" in output
    assert "twine upload" not in output
