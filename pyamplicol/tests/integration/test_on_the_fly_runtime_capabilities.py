# SPDX-License-Identifier: 0BSD
"""Small genuine-artifact regression for on-the-fly capability loading."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, ProcessRequest, Runtime
from pyamplicol._internal.versions import (
    ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
    ON_THE_FLY_RUNTIME_CAPABILITY,
)
from pyamplicol.artifacts import load_manifest
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    Action,
    ColorAccuracy,
    ColorConfig,
    EvaluatorConfig,
    EvaluatorExecutionMode,
    GenerationConfig,
    GenerationValidationConfig,
    RunConfig,
)

_PROCESS = "d d~ > z"
_PROCESS_ID = "d_dbar_to_z"
_CAPABILITIES = (
    ON_THE_FLY_RUNTIME_CAPABILITY,
    ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
)


def _unavailable(reason: str) -> None:
    if os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _require_native_on_the_fly() -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    rusticol = importlib.import_module("pyamplicol._rusticol")
    if not hasattr(rusticol, "_build_on_the_fly_process_seeds_v1"):
        _unavailable("the installed Rusticol extension lacks on-the-fly generation")


def _configuration() -> RunConfig:
    return RunConfig(
        action=Action.GENERATE,
        color=ColorConfig(accuracy=ColorAccuracy.LC),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=EvaluatorExecutionMode.ON_THE_FLY,
        ),
    )


def test_fresh_builtin_on_the_fly_artifact_loads_with_its_declared_capabilities(
    tmp_path: Path,
) -> None:
    _require_native_on_the_fly()
    artifact = tmp_path / "on-the-fly-d-dbar-to-z"
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(_configuration()).generate(
            ProcessRequest.parse(_PROCESS, name=_PROCESS_ID),
            artifact,
            model=ModelSource.from_path(prepared_model),
        )

    manifest = load_manifest(artifact)
    assert manifest.runtime["required_runtime_capabilities"] == _CAPABILITIES
    assert manifest.processes[0]["required_runtime_capabilities"] == _CAPABILITIES

    runtime = Runtime.load(artifact, process=_PROCESS_ID)
    assert runtime.execution_mode == "on-the-fly"
    assert runtime.physics.process_id == _PROCESS_ID
    assert runtime.physics.process == _PROCESS
