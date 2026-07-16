# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from pyamplicol.cli import run_cli
from pyamplicol.cli.handlers import _process_set
from pyamplicol.config import (
    ConfigurationError,
    ProcessConfig,
    ProcessEntry,
    RunConfig,
)
from pyamplicol.reporting import ProgressSink


class _Services:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.config: RunConfig | None = None

    def generate(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        if self.fail:
            raise ConfigurationError("generation rejected")
        self.config = config
        return {
            "action": config.action,
            "workers": config.generation.workers,
            "requests": tuple(entry.expression for entry in config.process.entries),
        }

    def evaluate(self, config: RunConfig, progress: ProgressSink) -> object:
        raise AssertionError((config, progress))

    benchmark = evaluate
    inspect = evaluate
    model_inspect = evaluate
    model_compile = evaluate
    model_processes = evaluate


def test_typed_config_entries_preserve_complete_process_set_behavior() -> None:
    config = RunConfig(
        action="generate",
        process=ProcessConfig(
            entries=(
                ProcessEntry("d d~ > z g", "ddbar_zg"),
                ProcessEntry("u u~ > z g"),
            )
        ),
    )

    processes = _process_set(config)

    assert tuple(request.expression for request in processes.requests) == (
        "d d~ > z g",
        "u u~ > z g",
    )
    assert processes.requests[0].name == "ddbar_zg"
    assert processes.requests[1].name == "u_ubar_to_z_g"


def test_cli_dispatches_protocol_and_keeps_json_on_stdout(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    services = _Services()
    sys.modules.pop("symbolica", None)
    status = run_cli(
        (
            "generate",
            "d d~ > z g",
            str(tmp_path / "artifact"),
            "--workers",
            "3",
            "--format",
            "json",
            "--progress",
            "off",
        ),
        services=services,
        stdout=stdout,
        stderr=stderr,
    )
    assert status == 0
    assert json.loads(stdout.getvalue()) == {
        "action": "generate",
        "requests": ["d d~ > z g"],
        "workers": 3,
    }
    assert stderr.getvalue() == ""
    assert services.config is not None
    assert "symbolica" not in sys.modules


def test_cli_failures_write_only_diagnostics_to_stderr(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    status = run_cli(
        (
            "generate",
            "d d~ > z g",
            str(tmp_path / "artifact"),
            "--format",
            "json",
        ),
        services=_Services(fail=True),
        stdout=stdout,
        stderr=stderr,
    )
    assert status == 2
    assert stdout.getvalue() == ""
    assert "generation rejected" in stderr.getvalue()
