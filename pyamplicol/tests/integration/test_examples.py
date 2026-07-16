# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pyamplicol.cli import parse_cli, run_cli
from pyamplicol.config import ProcessEntry, RunConfig
from pyamplicol.reporting import ProgressSink

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
RUN_CARDS = tuple(sorted(EXAMPLES.glob("*.toml")))


class _RecordingServices:
    def __init__(self) -> None:
        self.config: RunConfig | None = None

    def _record(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        self.config = config
        return {"action": config.action, "schema_version": config.schema_version}

    generate = _record
    plan = _record
    evaluate = _record
    benchmark = _record
    inspect = _record
    model_inspect = _record
    model_compile = _record
    model_processes = _record


@pytest.mark.parametrize("card", RUN_CARDS, ids=lambda path: path.stem)
def test_cards_cross_cli_resolution_and_typed_dispatch_without_running_work(
    card: Path,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    services = _RecordingServices()
    status = run_cli(
        (str(card), "--progress", "off", "--format", "json"),
        services=services,
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0, stderr.getvalue()
    assert services.config is not None
    assert json.loads(stdout.getvalue()) == {
        "action": services.config.action,
        "schema_version": 1,
    }
    assert stderr.getvalue() == ""


def test_copied_card_rebases_relative_paths_to_user_directory(tmp_path: Path) -> None:
    source = EXAMPLES / "evaluate_total.toml"
    copied = tmp_path / source.name
    shutil.copy2(source, copied)

    config = parse_cli((str(copied),)).resolve().effective
    assert config.evaluation.artifact == (tmp_path / "artifacts/builtin_sm_lc")
    assert config.evaluation.momenta == (tmp_path / "data/ddbar_zg_momenta.json")
    assert config.evaluation.model_parameters == (
        tmp_path / "data/model_parameters.json"
    )


def test_direct_cli_example_resolves_process_set_and_ordered_override(
    tmp_path: Path,
) -> None:
    invocation = parse_cli(
        (
            "generate",
            "d d~ > z g",
            str(tmp_path / "mixed"),
            "--process",
            "d d~ > z g g",
            "--name",
            "ddbar_zg",
            "--name",
            "ddbar_zgg",
            "--color-accuracy",
            "nlc",
            "--workers",
            "4",
            "--set",
            "generation.workers=2",
        )
    )
    config = invocation.resolve().effective
    assert config.process.entries == (
        ProcessEntry("d d~ > z g", "ddbar_zg"),
        ProcessEntry("d d~ > z g g", "ddbar_zgg"),
    )
    assert config.color.accuracy == "nlc"
    assert config.generation.workers == 2


@pytest.mark.parametrize(
    "script",
    tuple(sorted((EXAMPLES / "python").glob("*.py"))),
    ids=lambda path: path.stem,
)
def test_python_examples_offer_help_without_native_or_generation_imports(
    script: Path,
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert "schema-v3 artifact writing" not in completed.stderr
    assert "no pyAmpliCol backend" not in completed.stderr


def test_packaged_model_helper_materializes_external_card_inputs(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    script = EXAMPLES / "python/copy_packaged_models.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == tmp_path / "models"

    expected = (
        tmp_path / "models/json/sm/sm.json",
        tmp_path / "models/json/scalars/scalars.json",
        tmp_path / "models/json/scalar_gravity/scalar_gravity.json",
        tmp_path / "models/ufo/sm/vertices.py",
    )
    assert all(path.is_file() for path in expected)

    copied_card = tmp_path / "external_json_sm.toml"
    shutil.copy2(EXAMPLES / copied_card.name, copied_card)
    config = parse_cli((str(copied_card),)).resolve().effective
    assert Path(config.model.source).is_file()
