# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.resources
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pyamplicol import Generator, ProcessSet
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    JITConfig,
    RunConfig,
)

_PARAMETER_ID = "normalization.alpha_s_me_check"
_MIXED_PROCESSES = (
    "d d~ > z g",
    "d d~ > z g g",
)
_MIXED_NAMES = ("ddbar_zg", "ddbar_zgg")
_METADATA_KEYS = (
    "process",
    "process_key",
    "color_accuracy",
    "external_particles",
    "helicities",
    "colors",
    "shape",
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _source_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_paths = [str((_PROJECT_ROOT / "src").resolve())]
    source_paths.extend(
        str(Path(entry).resolve())
        for entry in sys.path
        if entry and str(Path(entry).resolve()) not in source_paths
    )
    environment["PYTHONPATH"] = os.pathsep.join(source_paths)
    return environment


@dataclass(frozen=True, slots=True)
class _NativeTools:
    make: str
    cxx: str
    fortran: str
    rustc: str
    rusticol_config: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _BuiltBundle:
    artifact: Path
    process_ids: tuple[str, ...]
    python_driver: Path
    cpp_driver: Path
    fortran_driver: Path
    rust_driver: Path


def _required_native_tests() -> bool:
    return os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1"


def _unavailable(reason: str) -> None:
    if _required_native_tests():
        pytest.fail(reason)
    pytest.skip(reason)


def _tool_from_environment(variable: str, default: str) -> str | None:
    configured = os.environ.get(variable)
    if configured:
        command = configured.split()[0]
        located = shutil.which(command)
        return located or (command if Path(command).is_file() else None)
    return shutil.which(default)


def _rusticol_config() -> tuple[str, ...] | None:
    configured = os.environ.get("RUSTICOL_CONFIG")
    if configured:
        command = shlex.split(configured)
        if command:
            executable = shutil.which(command[0])
            if executable or Path(command[0]).expanduser().is_file():
                return (executable or str(Path(command[0]).expanduser()), *command[1:])
    sibling = Path(sys.executable).parent / "rusticol-config"
    if sibling.is_file():
        return (str(sibling),)
    discovered = shutil.which("rusticol-config")
    if discovered:
        return (discovered,)
    if importlib.util.find_spec("pyamplicol._sdk.config") is not None:
        return (sys.executable, "-m", "pyamplicol._sdk.config")
    return None


@pytest.fixture(scope="module")
def native_tools() -> _NativeTools:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    discovered = {
        "make": shutil.which("make"),
        "cxx": _tool_from_environment("CXX", "c++"),
        "fortran": _tool_from_environment("FC", "gfortran"),
        "rustc": _tool_from_environment("RUSTC", "rustc"),
        "rusticol_config": _rusticol_config(),
    }
    missing = tuple(name for name, value in discovered.items() if value is None)
    if missing:
        _unavailable("cross-language tests require " + ", ".join(missing))
    tools = _NativeTools(**discovered)  # type: ignore[arg-type]
    checked = subprocess.run(
        [*tools.rusticol_config, "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if checked.returncode != 0:
        _unavailable(
            "the installed Rusticol SDK is unavailable: " + checked.stderr.strip()
        )
    return tools


def _generation_config(accuracy: str) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy),
        generation=GenerationConfig(workers=1, emit_api_bundle=True),
        evaluator=EvaluatorConfig(
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=1),
        ),
    )


def _manifest_process_ids(artifact: Path) -> tuple[str, ...]:
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    return tuple(str(process["id"]) for process in manifest["processes"])


def _assert_complete_runtime_coverage(
    artifact: Path, process_ids: tuple[str, ...], accuracy: str
) -> None:
    for process_id in process_ids:
        physics = json.loads(
            (artifact / "processes" / process_id / "physics.json").read_text(
                encoding="utf-8"
            )
        )
        assert physics["color_accuracy"] == accuracy
        assert physics["coverage"]["helicities"] == "complete"
        expected_color_coverage = "complete" if accuracy == "lc" else "contracted"
        assert physics["coverage"]["color"] == expected_color_coverage
        assert physics["selectors"]["helicity"] is True
        assert physics["selectors"]["color_flow"] is (accuracy == "lc")
        assert physics["selectors"]["contracted_color"] is False


def _assert_one_root_bundle(artifact: Path, process_ids: tuple[str, ...]) -> None:
    api_directories = tuple(path for path in artifact.rglob("API") if path.is_dir())
    assert api_directories == (artifact / "API",)
    for process_id in process_ids:
        assert not (artifact / "processes" / process_id / "API").exists()

    rows = {
        line.split("\t", 1)[0]
        for line in (artifact / "API/validation_points.dat")
        .read_text(encoding="ascii")
        .splitlines()[1:]
        if line and not line.startswith("#")
    }
    assert rows == set(process_ids)

    rust_source = (artifact / "API/rust/check_standalone.rs").read_text(
        encoding="utf-8"
    )
    assert "Runtime::load" in rust_source
    assert "unsafe" not in rust_source
    assert 'extern "C"' not in rust_source
    assert "rusticol_runtime_" not in rust_source


def _build_bundle(
    artifact: Path,
    expressions: tuple[str, ...],
    names: tuple[str, ...],
    accuracy: str,
    tools: _NativeTools,
) -> _BuiltBundle:
    processes = ProcessSet.from_expressions(expressions, names=names)
    Generator(_generation_config(accuracy)).generate(processes, artifact)
    process_ids = _manifest_process_ids(artifact)
    assert process_ids == names
    _assert_complete_runtime_coverage(artifact, process_ids, accuracy)
    _assert_one_root_bundle(artifact, process_ids)

    environment = _source_environment()
    environment.update(
        {
            "CXX": tools.cxx,
            "FC": tools.fortran,
            "RUSTC": tools.rustc,
            "RUSTICOL_CONFIG": shlex.join(tools.rusticol_config),
        }
    )
    for language in ("cpp", "fortran", "rust"):
        make_arguments = [
            tools.make,
            "-C",
            str(artifact / "API" / language),
        ]
        if language == "rust":
            make_arguments.append(f"RUSTC={tools.rustc}")
        make_arguments.append("all")
        completed = subprocess.run(
            make_arguments,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, (
            f"{language} API driver failed to build\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    return _BuiltBundle(
        artifact=artifact,
        process_ids=process_ids,
        python_driver=artifact / "API/python/check_standalone.py",
        cpp_driver=(
            artifact.parent
            / ".pyamplicol-api-build"
            / artifact.name
            / "cpp/check_standalone"
        ),
        fortran_driver=(
            artifact.parent
            / ".pyamplicol-api-build"
            / artifact.name
            / "fortran/check_standalone"
        ),
        rust_driver=(
            artifact.parent
            / ".pyamplicol-api-build"
            / artifact.name
            / "rust/check_standalone"
        ),
    )


def test_safe_rust_demo_compiles_and_runs_with_installed_or_source_sdk(
    native_tools: _NativeTools,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [*native_tools.rusticol_config, "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    sdk = json.loads(completed.stdout)
    package = importlib.resources.files("pyamplicol")
    if not isinstance(package, os.PathLike):
        _unavailable("the Rust SDK integration test requires an unpacked installation")
    package_path = Path(os.fspath(package))
    artifact = package_path / "assets" / "selftest" / sdk["target"] / "artifact"
    source = package_path / "assets" / "api_templates" / "rust" / "check_standalone.rs"
    if not artifact.is_dir() or not source.is_file():
        _unavailable("the target-specific Rust SDK self-test fixture is unavailable")

    binary = tmp_path / "check_standalone_rust"
    environment = os.environ.copy()
    environment["RUSTICOL_RUST_SOURCE"] = str(sdk["rust_source"])
    compiled = subprocess.run(
        [
            native_tools.rustc,
            "--edition=2021",
            "-Dwarnings",
            "-C",
            "opt-level=2",
            str(source),
            "-o",
            str(binary),
            *map(str, sdk["rust_flags"]),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, (
        f"safe Rust API driver failed to compile\n"
        f"stdout:\n{compiled.stdout}\nstderr:\n{compiled.stderr}"
    )

    expected = json.loads(
        (artifact.parent / "expected.json").read_text(encoding="utf-8")
    )
    process_id = str(expected["process_id"])

    def run(*arguments: str) -> dict[str, Any]:
        result = subprocess.run(
            [str(binary), "--json", "--process", process_id, *arguments],
            cwd=artifact,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    baseline = run()
    model_card = tmp_path / "model-parameters.json"
    model_card.write_text(
        json.dumps({"normalization.alpha_ew": [0.008, 0.0]}) + "\n",
        encoding="utf-8",
    )
    card = run("--model-parameters", str(model_card))
    direct = run("--set-parameter", "normalization.alpha_ew", "0.008", "0")

    for payload in (baseline, card, direct):
        assert payload["language"] == "rust"
        assert payload["available"] is True
        assert payload["process_key"] == process_id
        assert payload["resolved_sum"] == pytest.approx(
            payload["compatibility_total"], rel=1.0e-12, abs=1.0e-15
        )
        assert sum(payload["values"]) == pytest.approx(
            payload["compatibility_total"][0], rel=1.0e-12, abs=1.0e-15
        )
    assert card["values"] == pytest.approx(direct["values"])
    assert card["compatibility_total"] != pytest.approx(
        baseline["compatibility_total"], rel=1.0e-10
    )


@pytest.fixture(scope="module")
def generated_bundles(
    tmp_path_factory: pytest.TempPathFactory,
    native_tools: _NativeTools,
) -> dict[str, _BuiltBundle]:
    root = tmp_path_factory.mktemp("multilanguage-api")
    return {
        "mixed": _build_bundle(
            root / "mixed-lc",
            _MIXED_PROCESSES,
            _MIXED_NAMES,
            "lc",
            native_tools,
        ),
        "single": _build_bundle(
            root / "single-lc",
            (_MIXED_PROCESSES[0],),
            (_MIXED_NAMES[0],),
            "lc",
            native_tools,
        ),
        "nlc": _build_bundle(
            root / "single-nlc",
            (_MIXED_PROCESSES[0],),
            (_MIXED_NAMES[0],),
            "nlc",
            native_tools,
        ),
        "full": _build_bundle(
            root / "single-full",
            (_MIXED_PROCESSES[0],),
            (_MIXED_NAMES[0],),
            "full",
            native_tools,
        ),
    }


def _driver_command(bundle: _BuiltBundle, language: str) -> list[str]:
    if language == "python":
        return [sys.executable, str(bundle.python_driver)]
    if language == "cpp":
        return [str(bundle.cpp_driver)]
    if language == "fortran":
        return [str(bundle.fortran_driver)]
    if language == "rust":
        return [str(bundle.rust_driver)]
    raise AssertionError(language)


def _run_driver(
    bundle: _BuiltBundle,
    language: str,
    *,
    process: str | None,
    model_card: Path | None = None,
    override: float | None = None,
) -> dict[str, Any]:
    command = [*_driver_command(bundle, language), "--json"]
    if process is not None:
        command.extend(("--process", process))
    if model_card is not None:
        command.extend(("--model-parameters", str(model_card)))
    if override is not None:
        command.extend(("--set-parameter", _PARAMETER_ID, f"{override:.17g}", "0"))
    completed = subprocess.run(
        command,
        cwd=bundle.artifact,
        env=_source_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"{language} API driver failed\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        pytest.fail(
            f"{language} driver did not emit one JSON object: {error}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    assert payload["language"] == language
    assert payload["available"] is True
    assert payload["precision"] == 16
    return payload


def _assert_numeric_sequence(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    assert actual == pytest.approx(expected, rel=1.0e-12, abs=1.0e-15)


def _assert_language_payloads(payloads: dict[str, dict[str, Any]]) -> None:
    reference = payloads["python"]
    for language, payload in payloads.items():
        for key in _METADATA_KEYS:
            assert payload[key] == reference[key], (language, key)
        _assert_numeric_sequence(payload["values"], reference["values"])
        _assert_numeric_sequence(payload["resolved_sum"], reference["resolved_sum"])
        _assert_numeric_sequence(
            payload["compatibility_total"], reference["compatibility_total"]
        )
        _assert_numeric_sequence(
            payload["resolved_sum"], payload["compatibility_total"]
        )
        assert len(payload["values"]) == (
            payload["shape"][0] * payload["shape"][1] * payload["shape"][2]
        )


def _all_languages(
    bundle: _BuiltBundle,
    *,
    process: str | None,
    model_card: Path | None = None,
    override: float | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        language: _run_driver(
            bundle,
            language,
            process=process,
            model_card=model_card,
            override=override,
        )
        for language in ("python", "cpp", "fortran", "rust")
    }


def test_mixed_lc_bundle_agrees_for_all_processes_and_parameter_updates(
    generated_bundles: dict[str, _BuiltBundle],
    tmp_path: Path,
) -> None:
    bundle = generated_bundles["mixed"]
    model_card = tmp_path / "model-parameters.json"
    model_card.write_text(
        json.dumps({_PARAMETER_ID: [0.109, 0.0]}) + "\n",
        encoding="utf-8",
    )
    assert "alpha_s" not in model_card.read_text(encoding="utf-8").replace(
        _PARAMETER_ID, ""
    )

    for process_id in bundle.process_ids:
        baseline = _all_languages(bundle, process=process_id)
        card = _all_languages(bundle, process=process_id, model_card=model_card)
        direct = _all_languages(bundle, process=process_id, override=0.127)
        for payloads in (baseline, card, direct):
            _assert_language_payloads(payloads)

        baseline_total = baseline["python"]["compatibility_total"][0]
        card_total = card["python"]["compatibility_total"][0]
        direct_total = direct["python"]["compatibility_total"][0]
        assert card_total != pytest.approx(baseline_total, rel=1.0e-10)
        assert direct_total != pytest.approx(baseline_total, rel=1.0e-10)


def test_single_process_bundle_uses_default_process_without_selector(
    generated_bundles: dict[str, _BuiltBundle],
) -> None:
    bundle = generated_bundles["single"]
    payloads = _all_languages(bundle, process=None)
    _assert_language_payloads(payloads)
    assert payloads["python"]["process_key"] == bundle.process_ids[0]


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_native_contracted_color_is_one_component_per_helicity(
    generated_bundles: dict[str, _BuiltBundle],
    accuracy: str,
) -> None:
    bundle = generated_bundles[accuracy]
    payloads = _all_languages(bundle, process=bundle.process_ids[0])
    _assert_language_payloads(payloads)
    payload = payloads["python"]
    assert payload["color_accuracy"] == accuracy
    assert payload["shape"][2] == 1
    assert len(payload["colors"]) == 1
    assert payload["colors"][0]["kind"] == "contracted-color"
    assert payload["colors"][0]["word"] == []
