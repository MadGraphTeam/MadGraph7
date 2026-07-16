# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Literal, cast

import pytest

import pyamplicol.api.services as api_services
import pyamplicol.generation.service as generation_service
import pyamplicol.licensing as licensing
import pyamplicol.models.loading as model_loading
from pyamplicol import (
    CompiledModel,
    Generator,
    ModelSource,
    ProcessSet,
    generate,
)
from pyamplicol.api.results import GenerationResult
from pyamplicol.cli import run_cli
from pyamplicol.config import (
    Action,
    ConfigResolution,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    ModelConfig,
    RunConfig,
)
from pyamplicol.licensing import SymbolicaLicenseState


def _snapshot(root: Path) -> tuple[tuple[str, int, int, int, bytes | None], ...]:
    paths = (root, *sorted(root.rglob("*")))
    result: list[tuple[str, int, int, int, bytes | None]] = []
    for path in paths:
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        payload = path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None
        result.append(
            (
                relative,
                metadata.st_mode,
                metadata.st_size,
                metadata.st_mtime_ns,
                payload,
            )
        )
    return tuple(result)


def _forbidden(name: str):
    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"dry-run called forbidden operation: {name}")

    return fail


def _restricted_license(**_kwargs: object) -> SymbolicaLicenseState:
    return SymbolicaLicenseState(licensed=False, restricted=True)


def test_model_compilation_returns_the_canonical_public_model() -> None:
    from pyamplicol.models.loading import CompiledModel as LoadingCompiledModel

    compiled = ModelSource.built_in_sm().compile(use_cache=False)

    assert CompiledModel is LoadingCompiledModel
    assert isinstance(compiled, CompiledModel)
    assert compiled.name == "built-in-sm"
    assert compiled.schema_version == model_loading.COMPILED_MODEL_SCHEMA_VERSION
    assert compiled.model_compiler_version == model_loading.MODEL_COMPILER_VERSION


def test_injected_license_plan_avoids_symbolica_and_model_compilers() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "import pyamplicol.licensing as licensing",
                    "from pyamplicol import Generator",
                    "licensing.detect_symbolica_license = lambda **kwargs: "
                    "licensing.SymbolicaLicenseState(False, True)",
                    "assert 'symbolica' not in sys.modules",
                    "Generator().plan('d d~ > z')",
                    "assert 'symbolica' not in sys.modules",
                    "assert not any(name.startswith("
                    "'pyamplicol.models.compiler') for name in sys.modules)",
                )
            ),
        ),
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


class _CapturingGenerationBackend:
    def __init__(
        self,
        resolution: ConfigResolution,
        captured: list[CompiledModel],
    ) -> None:
        self._resolution = resolution
        self._captured = captured

    def generate(
        self,
        processes: ProcessSet,
        output: Path,
        *,
        model: ModelSource | CompiledModel | None = None,
        mode: str = "error",
    ) -> GenerationResult:
        assert isinstance(model, CompiledModel)
        self._captured.append(model)
        return GenerationResult(
            output=output,
            processes=processes,
            mode=cast(Literal["error", "append", "replace"], mode),
        )


def test_generator_and_generate_accept_the_compiled_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = ModelSource.built_in_sm().compile(use_cache=False)
    captured: list[CompiledModel] = []

    def factory(config: object, _progress: object) -> _CapturingGenerationBackend:
        assert isinstance(config, ConfigResolution)
        return _CapturingGenerationBackend(config, captured)

    monkeypatch.setattr(api_services, "_generator_factory", factory)
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)

    first = Generator(GenerationConfig(workers=1)).generate(
        "d d~ > z",
        tmp_path / "first",
        model=compiled,
    )
    second = generate(
        "d d~ > z",
        tmp_path / "second",
        model=compiled,
        config=GenerationConfig(workers=1),
    )

    assert captured == [compiled, compiled]
    assert first.output == (tmp_path / "first").resolve()
    assert second.output == (tmp_path / "second").resolve()
    assert not first.output.exists()
    assert not second.output.exists()


def test_generator_plan_is_strictly_non_writing_and_does_not_compile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = ModelSource.built_in_sm().compile(use_cache=False)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    marker = sandbox / "marker.txt"
    marker.write_text("unchanged\n", encoding="utf-8")
    output = sandbox / "output" / "artifact"
    cache = sandbox / "cache" / "models"
    config = RunConfig(
        action=Action.GENERATE,
        model=ModelConfig(cache_dir=cache),
        generation=GenerationConfig(output=output, workers=4),
        evaluator=EvaluatorConfig(optimization=EvaluatorOptimizationConfig(cores=8)),
    )
    before = _snapshot(sandbox)

    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)
    monkeypatch.setattr(
        model_loading,
        "compile_model_source",
        _forbidden("model compilation"),
    )
    monkeypatch.setattr(
        generation_service.GenerationBackend,
        "_resolve_model",
        _forbidden("generation model resolution"),
    )
    monkeypatch.setattr(
        generation_service.GenerationBackend,
        "_compile_concrete_process",
        _forbidden("process DAG compilation"),
    )
    monkeypatch.setattr(
        generation_service,
        "TemporaryDirectory",
        _forbidden("temporary directory creation"),
    )
    monkeypatch.setattr(
        generation_service,
        "write_schema_v3_artifact",
        _forbidden("artifact writing"),
    )

    plan = Generator(config).plan("p p > z", model=compiled)

    assert _snapshot(sandbox) == before
    assert not output.exists()
    assert not cache.exists()
    assert len(plan.concrete_processes) == 5
    assert plan.requested_settings.generation.workers == 4
    assert plan.effective_settings.generation.workers == 1
    assert plan.requested_settings.evaluator.optimization.cores == 8
    assert plan.effective_settings.evaluator.optimization.cores == 1
    assert [adjustment.path for adjustment in plan.adjustments] == [
        "generation.workers",
        "evaluator.optimization.cores",
    ]


def test_cli_generate_dry_run_preserves_the_filesystem_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "marker.txt").write_text("unchanged\n", encoding="utf-8")
    output = sandbox / "output" / "artifact"
    cache = sandbox / "cache" / "models"
    before = _snapshot(sandbox)
    stdout = io.StringIO()
    stderr = io.StringIO()

    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)
    monkeypatch.setattr(
        model_loading,
        "compile_model_source",
        _forbidden("model compilation"),
    )
    monkeypatch.setattr(
        generation_service.GenerationBackend,
        "_compile_concrete_process",
        _forbidden("process DAG compilation"),
    )
    monkeypatch.setattr(
        generation_service,
        "TemporaryDirectory",
        _forbidden("temporary directory creation"),
    )
    monkeypatch.setattr(
        generation_service,
        "write_schema_v3_artifact",
        _forbidden("artifact writing"),
    )

    status = run_cli(
        (
            "generate",
            "d d~ > z",
            str(output),
            "--dry-run",
            "--workers",
            "4",
            "--cores",
            "8",
            "--format",
            "json",
            "--progress",
            "off",
            "--set",
            f"model.cache_dir={cache}",
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0, stderr.getvalue()
    assert stderr.getvalue() == ""
    assert _snapshot(sandbox) == before
    assert not output.exists()
    assert not cache.exists()
    payload = json.loads(stdout.getvalue())
    assert payload["requested_settings"]["generation"]["workers"] == 4
    assert payload["effective_settings"]["generation"]["workers"] == 1
    assert payload["requested_settings"]["evaluator"]["optimization"]["cores"] == 8
    assert payload["effective_settings"]["evaluator"]["optimization"]["cores"] == 1
    assert [item["reason"] for item in payload["adjustments"]] == [
        "Symbolica restricted mode permits one generation instance",
        "Symbolica restricted mode permits one Symbolica core",
    ]


def test_plan_reads_an_existing_external_model_cache_without_modifying_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "pyamplicol"
        / "assets"
        / "models"
        / "json"
        / "scalars"
        / "scalars.json"
    )
    source = ModelSource.from_path(source_path)
    sandbox = tmp_path / "sandbox"
    cache = sandbox / "cache" / "models"
    output = sandbox / "output" / "artifact"
    source.compile(cache_dir=cache)
    before = _snapshot(sandbox)
    config = RunConfig(
        action=Action.GENERATE,
        model=ModelConfig(cache_dir=cache),
        generation=GenerationConfig(output=output),
    )

    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)
    monkeypatch.setattr(
        model_loading,
        "compile_model_source",
        _forbidden("model compilation"),
    )
    monkeypatch.setattr(
        generation_service.GenerationBackend,
        "_compile_concrete_process",
        _forbidden("process DAG compilation"),
    )
    monkeypatch.setattr(
        generation_service,
        "write_schema_v3_artifact",
        _forbidden("artifact writing"),
    )

    plan = Generator(config).plan(
        "scalar_0 scalar_0 > scalar_0 scalar_0",
        model=source,
    )

    assert _snapshot(sandbox) == before
    assert not output.exists()
    assert plan.estimated_coverage["model_kind"] == "json"
    assert len(plan.concrete_processes) == 1


def test_plan_uses_licensed_concrete_process_resource_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        licensing,
        "detect_symbolica_license",
        lambda **_kwargs: SymbolicaLicenseState(licensed=True, restricted=False),
    )
    monkeypatch.setattr(licensing, "_cpu_budget", lambda: 12)
    monkeypatch.setattr(
        model_loading,
        "compile_model_source",
        _forbidden("model compilation"),
    )

    plan = Generator(RunConfig(action=Action.GENERATE)).plan("p p > z")

    assert len(plan.concrete_processes) == 5
    assert plan.requested_settings.generation.workers == "auto"
    assert plan.effective_settings.generation.workers == 5
    assert plan.requested_settings.evaluator.optimization.cores == "auto"
    assert plan.effective_settings.evaluator.optimization.cores == 2
    assert [adjustment.path for adjustment in plan.adjustments] == [
        "generation.workers",
        "evaluator.optimization.cores",
    ]
    assert all(adjustment.reason for adjustment in plan.adjustments)
