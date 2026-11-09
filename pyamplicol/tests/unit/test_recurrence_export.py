# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.api import Runtime
from pyamplicol.api.errors import CompatibilityError
from pyamplicol.cli import parse_cli, run_cli
from pyamplicol.cli.recurrence_export import RecurrenceExportInvocation
from pyamplicol.runtime.backend import RusticolRuntimeBackend


def _export(process: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_id": "a" * 64,
        "rusticol_version": "1.0.1",
        "execution_semantics": "batched-f64",
        "parameters": [{"name": "aS", "value": {"re": 0.12, "im": 0.0}}],
        "programs": [{"process_key": process, "colour_id": "flow:0"}],
        "kernels": [{"sha256": "b" * 64, "application": [1, 2, 3]}],
    }


def test_public_export_uses_native_prepared_program_without_physics(
    tmp_path: Path,
) -> None:
    calls = []

    def export(*, color_flows: tuple[str, ...] | None) -> str:
        calls.append(color_flows)
        return json.dumps(_export("p0"))

    def unexpected_evaluation(*args: object, **kwargs: object) -> None:
        pytest.fail("export must not evaluate momenta")

    native = SimpleNamespace(
        metadata_json=lambda: json.dumps({"execution_mode": "recurrence"}),
        export_recurrence_json=export,
        evaluate=unexpected_evaluation,
    )
    runtime = Runtime(RusticolRuntimeBackend(native, SimpleNamespace(), tmp_path))
    assert runtime.export_recurrence(color_flows=["flow:0"]) == _export("p0")
    assert calls == [("flow:0",)]
    with pytest.raises(ValueError, match="unique"):
        runtime.export_recurrence(color_flows=["flow:0", "flow:0"])
    assert calls == [("flow:0",)]
    del native.export_recurrence_json
    with pytest.raises(CompatibilityError, match="no recurrence export binding"):
        runtime.export_recurrence()


def test_cli_exports_all_processes_aliases_and_group_members(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pyamplicol.artifacts
    import pyamplicol.artifacts.process_groups

    monkeypatch.setattr(
        pyamplicol.artifacts,
        "load_manifest",
        lambda artifact: SimpleNamespace(
            processes=(
                {"id": "p0", "aliases": ({"id": "alias"},)},
                {"id": "p1", "aliases": ()},
            )
        ),
    )
    monkeypatch.setattr(
        pyamplicol.artifacts.process_groups,
        "read_process_groups",
        lambda artifact: SimpleNamespace(
            members=lambda: iter(
                [SimpleNamespace(process_id="p0"), SimpleNamespace(process_id="member")]
            )
        ),
    )
    calls = []

    def load(artifact: Path, *, process: str, model_parameters: object) -> object:
        calls.append((artifact, process, model_parameters))

        def export(*, color_flows: object) -> dict[str, object]:
            result = _export(process)
            result["parameters"].append(
                {"name": f"mass_{process}", "value": {"re": 1.0, "im": 0.0}}
            )
            return result

        return SimpleNamespace(export_recurrence=export)

    monkeypatch.setattr(Runtime, "load", load)
    parameters = tmp_path / "parameters.json"
    parameters.write_text('{"aS": 0.12}', encoding="utf-8")
    destination = tmp_path / "capture.json"
    output, errors = io.StringIO(), io.StringIO()
    assert (
        run_cli(
            [
                "export-recurrence",
                str(tmp_path),
                "-o",
                str(destination),
                "--model-parameters",
                str(parameters),
                "--json",
            ],
            stdout=output,
            stderr=errors,
        )
        == 0
    )
    result = json.loads(destination.read_text(encoding="utf-8"))
    assert [program["process_key"] for program in result["programs"]] == [
        "p0",
        "alias",
        "p1",
        "member",
    ]
    assert len(result["kernels"]) == 1
    assert [parameter["name"] for parameter in result["parameters"]] == [
        "aS",
        "mass_alias",
        "mass_member",
        "mass_p0",
        "mass_p1",
    ]
    assert calls == [
        (tmp_path, process, {"aS": 0.12}) for process in ("p0", "alias", "p1", "member")
    ]
    assert json.loads(output.getvalue())["status"] == "complete"
    assert not errors.getvalue()


def test_cli_selection_and_failure_preserve_existing_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def load(artifact: Path, *, process: str, model_parameters: object) -> object:
        def export(*, color_flows: tuple[str, ...] | None) -> dict[str, object]:
            calls.append((process, color_flows))
            if process == "missing":
                raise ValueError("unknown process 'missing'")
            return _export(process)

        return SimpleNamespace(export_recurrence=export)

    monkeypatch.setattr(Runtime, "load", load)
    destination = tmp_path / "capture.json"
    destination.write_text("existing capture\n", encoding="utf-8")
    arguments = [
        "export-recurrence",
        str(tmp_path),
        "-o",
        str(destination),
        "--process",
        "p0",
        "--process",
        "missing",
        "--color-flow",
        "flow:0",
    ]
    invocation = parse_cli(arguments)
    assert isinstance(invocation, RecurrenceExportInvocation)
    errors = io.StringIO()
    assert run_cli(arguments, stdout=io.StringIO(), stderr=errors) == 2
    assert "unknown process 'missing'" in errors.getvalue()
    assert destination.read_text(encoding="utf-8") == "existing capture\n"
    assert calls == [("p0", ("flow:0",)), ("missing", ("flow:0",))]


def test_cli_rejects_non_mapping_parameters_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_load(*args: object, **kwargs: object) -> None:
        pytest.fail("runtime must not be loaded for invalid parameters")

    monkeypatch.setattr(Runtime, "load", unexpected_load)
    parameters = tmp_path / "parameters.json"
    parameters.write_text("[]", encoding="utf-8")
    errors = io.StringIO()
    assert (
        run_cli(
            [
                "export-recurrence",
                str(tmp_path),
                "-o",
                str(tmp_path / "capture.json"),
                "--model-parameters",
                str(parameters),
            ],
            stdout=io.StringIO(),
            stderr=errors,
        )
        == 2
    )
    assert "must contain a JSON object" in errors.getvalue()
