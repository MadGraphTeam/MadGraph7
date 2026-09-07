# SPDX-License-Identifier: 0BSD
"""One generated correlator example, evaluated independently through every SDK."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from pyamplicol import (
    ColorCorrelator,
    CorrelatedRequest,
    CorrelatorConfig,
    EmitGluon,
    Generator,
    ModelSource,
    ProcessSet,
    Runtime,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    JITConfig,
    RunConfig,
)
from tests.integration.test_runtime_selector_api_parity import (
    _compile_probe,
    _NativeSdk,
    _source_environment,
    _unavailable,
)
from tests.integration.test_runtime_selector_api_parity import (
    native_sdk as _native_sdk_fixture,
)

native_sdk = _native_sdk_fixture

_ROOT = Path(__file__).resolve().parents[2]
_POINTS = (
    (
        (400, 0, 0, 400),
        (400, 0, 0, -400),
        (300, 300, 0, 0),
        (250, -150, 200, 0),
        (250, -150, -200, 0),
    ),
    (
        (400, 0, 0, 400),
        (400, 0, 0, -400),
        (300, 0, 300, 0),
        (250, 200, -150, 0),
        (250, -200, -150, 0),
    ),
)


@pytest.fixture(scope="module", params=("lc", "nlc", "full"))
def correlated_output(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
):
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    artifact = tmp_path_factory.mktemp(f"correlated-sdk-{request.param}") / "process"
    alpha = (EmitGluon(2, -3), EmitGluon(-3, -1), EmitGluon(-3, -2))
    beta = (EmitGluon(1, -1), EmitGluon(3, -2), EmitGluon(4, -3))
    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator("cascade-interference", alpha, beta),),
        spin_correlations=((5,),),
    )
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=request.param),
        generation=GenerationConfig(workers=1),
        evaluator=EvaluatorConfig(
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=1),
        ),
    )
    Generator(config).generate(
        ProcessSet.from_expressions(("d d~ > d d~ g",), names=("dd_ddg",)),
        artifact,
        model=ModelSource.built_in_sm(),
        correlators=declarations,
    )
    runtime = Runtime.load(artifact, process="dd_ddg")
    oracle = runtime.evaluate_correlated_many(
        _POINTS,
        {
            "born": CorrelatedRequest("born", {}),
            "interference": CorrelatedRequest("cascade-interference", {}),
            "spin": CorrelatedRequest("cascade-interference", {5: (0, 0, 0, 1)}),
        },
        precision=50,
    )
    return artifact, tuple(
        tuple(complex(value) for value in series) for series in oracle.values()
    )


@pytest.fixture(scope="module", params=("c", "cpp", "fortran", "rust"))
def correlated_driver(
    request: pytest.FixtureRequest,
    native_sdk: _NativeSdk,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    extension = {"c": "c", "cpp": "cpp", "fortran": "f90", "rust": "rs"}[request.param]
    source = (_ROOT / "examples" / "native" / f"correlated.{extension}").read_text()
    return _compile_probe(
        request.param,
        source,
        native_sdk,
        tmp_path_factory.mktemp(f"correlated-sdk-{request.param}"),
    )


def test_correlated_sdk_matches_python_oracle(
    correlated_driver: Path, correlated_output
) -> None:
    artifact, oracle = correlated_output
    result = subprocess.run(
        [str(correlated_driver), str(artifact), "dd_ddg"],
        capture_output=True,
        text=True,
        env=_source_environment(),
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    measured = {}
    for line in result.stdout.splitlines():
        if line.startswith("VALUE "):
            _, request, point, real, imaginary = line.split()
            key = (int(request), int(point))
            assert key not in measured
            measured[key] = complex(float(real), float(imaginary))
    assert set(measured) == {
        (request, point) for request in range(3) for point in range(2)
    }
    for request, series in enumerate(oracle):
        for point, expected in enumerate(series):
            assert measured[request, point] == pytest.approx(
                expected, rel=2e-10, abs=1e-15
            )
