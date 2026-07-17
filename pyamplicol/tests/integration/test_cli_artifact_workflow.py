# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pyamplicol import (
    Generator,
    ProcessAlias,
    ProcessRequest,
    ProcessSet,
    Runtime,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Selection:
    selector: str
    process_id: str
    expression: str
    momenta: tuple[tuple[float, float, float, float], ...]


def _native_extension_available() -> bool:
    return importlib.util.find_spec("pyamplicol._rusticol") is not None


def _base_momenta() -> tuple[tuple[float, float, float, float], ...]:
    sqrt_s = 500.0
    z_mass = 91.188
    outgoing_momentum = (sqrt_s * sqrt_s - z_mass * z_mass) / (2.0 * sqrt_s)
    z_energy = (sqrt_s * sqrt_s + z_mass * z_mass) / (2.0 * sqrt_s)
    return (
        (250.0, 0.0, 0.0, 250.0),
        (250.0, 0.0, 0.0, -250.0),
        (z_energy, outgoing_momentum, 0.0, 0.0),
        (outgoing_momentum, -outgoing_momentum, 0.0, 0.0),
    )


@pytest.fixture(scope="module")
def lc_process_set_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not _native_extension_available():
        pytest.skip("the Rusticol extension has not been built")

    artifact = tmp_path_factory.mktemp("cli-artifact-workflow") / "artifact"
    request = ProcessRequest.parse("d d~ > z g", name="ddbar_zg")
    processes = ProcessSet(
        requests=(request,),
        aliases=(
            ProcessAlias(
                name="ddbar_gz",
                process_name=request.name,
                particle_permutation=(0, 1, 3, 2),
            ),
        ),
    )
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
        generation=GenerationConfig(
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(jit=JITConfig(optimization_level=1)),
    )

    Generator(config).generate(processes, artifact)
    return artifact


def _run_json_cli(*arguments: str, timeout: float = 30.0) -> Any:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["SYMBOLICA_HIDE_BANNER"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pyamplicol", *arguments],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            "CLI JSON stdout contains non-JSON output: "
            f"{completed.stdout!r} ({exc})"
        )


def _write_momenta(
    path: Path,
    momenta: tuple[tuple[float, float, float, float], ...],
) -> None:
    path.write_text(json.dumps([momenta]) + "\n", encoding="utf-8")


def _complex_values(payload: Any) -> tuple[complex, ...]:
    assert isinstance(payload, list)
    return tuple(
        complex(float(entry["real"]), float(entry["imag"])) for entry in payload
    )


def test_cli_inspect_evaluate_and_profile_select_processes_by_public_identity(
    lc_process_set_artifact: Path,
    tmp_path: Path,
) -> None:
    base = _base_momenta()
    alias = (base[0], base[1], base[3], base[2])
    selections = (
        _Selection("ddbar_zg", "ddbar_zg", "d d~ > z g", base),
        _Selection("d d~ > z g", "ddbar_zg", "d d~ > z g", base),
        _Selection("d d~ > g z", "ddbar_gz", "d d~ > g z", alias),
    )

    for index, selection in enumerate(selections):
        runtime = Runtime.load(
            lc_process_set_artifact,
            process=selection.selector,
        )
        points = (selection.momenta,)
        expected = runtime.evaluate(points)

        inspection = _run_json_cli(
            "inspect",
            str(lc_process_set_artifact),
            "--process",
            selection.selector,
            "--format",
            "json",
            "--progress",
            "off",
            "--color",
            "never",
            "--log-level",
            "error",
        )
        assert inspection["process_id"] == selection.process_id
        assert inspection["process"] == selection.expression
        assert inspection["color_accuracy"] == "lc"
        assert [
            particle["name"] for particle in inspection["external_particles"]
        ] == selection.expression.replace(">", " ").split()

        momenta_path = tmp_path / f"momenta-{index}.json"
        _write_momenta(momenta_path, selection.momenta)
        evaluated = _run_json_cli(
            "evaluate",
            str(lc_process_set_artifact),
            "--process",
            selection.selector,
            "--momenta",
            str(momenta_path),
            "--format",
            "json",
            "--progress",
            "off",
            "--color",
            "never",
            "--log-level",
            "error",
        )
        assert _complex_values(evaluated) == pytest.approx(expected, rel=1.0e-13)

        profile = _run_json_cli(
            "profile",
            str(lc_process_set_artifact),
            "--process",
            selection.selector,
            "--momenta",
            str(momenta_path),
            "--target-runtime",
            "0.001",
            "--batch-size",
            "2",
            "--warmup-runs",
            "0",
            "--minimum-samples",
            "2",
            "--format",
            "json",
            "--progress",
            "off",
            "--color",
            "never",
            "--log-level",
            "error",
        )
        assert profile["process_id"] == runtime.physics.process_id
        assert profile["process_expression"] == runtime.physics.process
        assert profile["environment"]["wall_time_source"] == (
            "runtime_core_repeated_wall_time"
        )
        assert profile["effective_config"]["target_runtime"] == pytest.approx(0.001)
        assert profile["effective_config"]["batch_size"] == 2
        assert profile["sample_count"] >= 2
        assert profile["wall_time_per_point"] > 0.0
