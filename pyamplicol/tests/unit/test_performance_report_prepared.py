# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.performance_report.artifacts import ArtifactStore
from tools.performance_report.models import ModelKey
from tools.performance_report.prepared import (
    PUBLIC_MODEL_COMPILE_COMMAND_PATH,
    PreparedModelError,
    _PublicCliModelCompiler,
    ensure_prepared_model,
    ensure_report_prepared_model,
    prepared_identity,
    validate_prepared_record,
)


class Source:
    def __init__(self) -> None:
        self.calls = 0

    def compile(self, *, prepared_output: Path, **_kwargs: object) -> object:
        self.calls += 1
        prepared_output.write_bytes(b"prepared-bundle")
        return object()


def _identity() -> dict[str, object]:
    return prepared_identity(
        model=ModelKey.UFO_SM,
        backend="jit",
        jit_optimization_level=2,
        source_digest="a" * 64,
        producer_revision="revision",
    )


def test_prepared_bundle_is_built_once_and_reused(tmp_path: Path) -> None:
    store = ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "locks",
    )
    source = Source()
    bundle = tmp_path / "prepared" / "ufo.pyamplicol-model"

    with ensure_prepared_model(
        store=store,
        bundle_path=bundle,
        source=source,
        evaluator=object(),
        identity=_identity(),
        model_cache_dir=tmp_path / "model-cache",
    ) as (path, reused):
        assert path == bundle
        assert not reused

    with ensure_prepared_model(
        store=store,
        bundle_path=bundle,
        source=source,
        evaluator=object(),
        identity=_identity(),
        model_cache_dir=tmp_path / "model-cache",
    ) as (_path, reused):
        assert reused

    assert source.calls == 1
    record = validate_prepared_record(bundle, expected_identity=_identity())
    assert record["bundle_size"] == len(b"prepared-bundle")


def test_report_prepared_model_requires_explicit_revision(tmp_path: Path) -> None:
    store = ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "locks",
    )

    with pytest.raises(PreparedModelError, match="explicit producer revision"):
        ensure_report_prepared_model(
            store=store,
            repo_root=tmp_path,
            worker_cores=1,
            model=ModelKey.UFO_SM,
            producer_revision=None,
        )


def test_prepared_bundle_digest_or_identity_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "locks",
    )
    source = Source()
    bundle = tmp_path / "prepared" / "ufo.pyamplicol-model"
    with ensure_prepared_model(
        store=store,
        bundle_path=bundle,
        source=source,
        evaluator=object(),
        identity=_identity(),
        model_cache_dir=tmp_path / "model-cache",
    ):
        pass

    bundle.write_bytes(b"tampered")
    with pytest.raises(PreparedModelError, match=r"digest|size"):
        validate_prepared_record(bundle, expected_identity=_identity())

    record_path = bundle.with_suffix(bundle.suffix + ".report.json")
    record = json.loads(record_path.read_text(encoding="ascii"))
    record["identity"]["producer_revision"] = "other"
    record_path.write_text(json.dumps(record), encoding="ascii")
    with pytest.raises(PreparedModelError, match="identity"):
        validate_prepared_record(bundle, expected_identity=_identity())


def test_report_model_compiler_uses_public_parse_resolve_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.config import (
        EvaluatorBackend,
        EvaluatorConfig,
        EvaluatorExecutionMode,
        EvaluatorOptimizationConfig,
        JITConfig,
    )

    output = tmp_path / "prepared.pyamplicol-model"
    evaluator = EvaluatorConfig(
        backend=EvaluatorBackend.JIT,
        execution_mode=EvaluatorExecutionMode.RECURRENCE,
        optimization=EvaluatorOptimizationConfig(cores=3),
        jit=JITConfig(optimization_level=2),
    )
    observed: dict[str, object] = {}

    def fake_dispatch(config, services, progress, *, dry_run):
        observed.update(
            {
                "config": config,
                "services": services,
                "progress": progress,
                "dry_run": dry_run,
            }
        )
        output.write_bytes(b"prepared")
        return {"output": str(output)}

    monkeypatch.setattr("pyamplicol.cli.handlers.dispatch", fake_dispatch)
    compiler = _PublicCliModelCompiler("built-in-sm")
    compiler.compile(
        cache_dir=tmp_path / "cache",
        evaluator=evaluator,
        prepared_output=output,
    )

    config = observed["config"]
    assert config.action == "model-compile"
    assert config.model.source == "built-in-sm"
    assert config.model.cache_dir == tmp_path / "cache"
    assert config.evaluator == evaluator
    assert observed["dry_run"] is False
    assert compiler.command_path == PUBLIC_MODEL_COMPILE_COMMAND_PATH
