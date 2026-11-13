# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.api import ResolvedEvaluation, Runtime
from pyamplicol.api.errors import CompatibilityError
from pyamplicol.cli import parse_cli, run_cli
from pyamplicol.cli.recurrence_export import RecurrenceExportInvocation
from pyamplicol.runtime.backend import RusticolRuntimeBackend


def _export(process: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "artifact_id": "a" * 64,
        "rusticol_version": "1.0.1",
        "execution_semantics": "batched-f64",
        "parameters": [{"name": "aS", "value": {"re": 0.12, "im": 0.0}}],
        "programs": [
            {
                "process_key": process,
                "process": "g g > g g",
                "representative_process_key": "p0"
                if process in {"alias", "member"}
                else process,
                "representative_colour_id": "flow:0",
                "colour_id": "flow:0",
                "crossing": None,
                "group_factor": {"numerator": 1, "denominator": 1},
                "caller_legs": [],
                "helicity_outputs": [{"id": "h0", "helicities": []}],
                "body": [{"kind": "clear_amplitude", "amplitude": 0}],
                "runs": [{"sources": [], "reduction": []}],
                "kernel_bindings": [],
            }
        ],
        "kernels": [{"sha256": "b" * 64, "application_base64": "AQID"}],
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
    assert result["kernels"][0]["application_base64"] == "AQID"
    for member in (result["programs"][1], result["programs"][3]):
        assert member["program_reference"] == {
            "process_key": "p0",
            "colour_id": "flow:0",
        }
        assert "body" not in member and "runs" not in member
    assert destination.read_text(encoding="utf-8").count("\n") == 1
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


@pytest.mark.parametrize("indent", (None, 0, 2))
def test_cli_compact_and_debug_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, indent: int | None
) -> None:
    monkeypatch.setattr(
        Runtime,
        "load",
        lambda *args, **kwargs: SimpleNamespace(
            export_recurrence=lambda **options: _export("p0")
        ),
    )
    destination = tmp_path / "capture.json"
    arguments = [
        "export-recurrence",
        str(tmp_path),
        "-o",
        str(destination),
        "--process",
        "p0",
        "--json",
    ]
    if indent is not None:
        arguments.extend(("--indent", str(indent)))
    assert run_cli(arguments, stdout=io.StringIO(), stderr=io.StringIO()) == 0
    content = destination.read_text(encoding="utf-8")
    if indent is None:
        assert content == json.dumps(_export("p0"), separators=(",", ":")) + "\n"
    else:
        assert content == json.dumps(_export("p0"), indent=indent) + "\n"
    with pytest.raises(SystemExit):
        parse_cli([*arguments, "--indent", "-1"])


def test_cli_member_only_selection_includes_one_representative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def load(artifact: Path, *, process: str, **kwargs: object) -> object:
        def export(**options: object) -> dict[str, object]:
            calls.append((process, options))
            return _export(process)

        return SimpleNamespace(export_recurrence=export)

    monkeypatch.setattr(Runtime, "load", load)
    destination = tmp_path / "capture.json"
    assert (
        run_cli(
            [
                "export-recurrence",
                str(tmp_path),
                "-o",
                str(destination),
                "--process",
                "member",
                "--process",
                "alias",
                "--json",
            ],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        == 0
    )
    capture = json.loads(destination.read_text(encoding="utf-8"))
    programs = capture["programs"]
    assert [program["process_key"] for program in programs] == ["member", "alias", "p0"]
    assert calls == [
        ("member", {"color_flows": None}),
        ("alias", {"color_flows": None}),
        ("p0", {"color_flows": ("flow:0",)}),
    ]
    assert programs[0]["program_reference"] == programs[1]["program_reference"]
    assert "body" not in programs[0] and "body" not in programs[1]
    assert programs[2]["body"] == _export("p0")["programs"][0]["body"]
    assert programs[2]["runs"][0]["sources"] == []
    assert len(capture["kernels"]) == 1


def test_cli_refuses_two_leg_orders_of_one_stored_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def load(artifact: Path, *, process: str, **kwargs: object) -> object:
        def export(**options: object) -> dict[str, object]:
            result = _export("p0")
            if process == "g g > g g, legs swapped":
                # A leg-permuted selection resolves to the same stored process.
                program = result["programs"][0]  # type: ignore[index]
                program["crossing"] = [
                    {"program_leg": 0, "caller_leg": 1, "sign": 1.0},
                    {"program_leg": 1, "caller_leg": 0, "sign": 1.0},
                ]
            return result

        return SimpleNamespace(export_recurrence=export)

    monkeypatch.setattr(Runtime, "load", load)
    destination = tmp_path / "capture.json"

    def export(*processes: str) -> tuple[int, str]:
        arguments = ["export-recurrence", str(tmp_path), "-o", str(destination)]
        for process in processes:
            arguments.extend(("--process", process))
        errors = io.StringIO()
        status = run_cli(arguments, stdout=io.StringIO(), stderr=errors)
        return status, errors.getvalue()

    status, errors = export("p0", "g g > g g, legs swapped")
    assert status == 2
    assert "in different leg orders" in errors
    assert not destination.exists()
    # Two names for the same leg order select one program.
    status, errors = export("p0", "g g > g g")
    assert (status, errors) == (0, "")
    programs = json.loads(destination.read_text(encoding="utf-8"))["programs"]
    assert [program["process_key"] for program in programs] == ["p0"]


@pytest.mark.parametrize(
    ("selector", "process_key", "by_process"),
    (
        ("member", "member", False),
        ("member", "member", True),
        ("u u~ > g g", "p0", True),
        ("u u~ > g g", "member", True),
    ),
)
def test_cli_reference_inputs_follow_resolved_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selector: str,
    process_key: str,
    by_process: bool,
) -> None:
    points = [[[10.0, -0.0, 0.0, 10.0], [10.0, 0.0, 0.0, -10.0]]]
    crossing = (
        [
            {"program_leg": 0, "caller_leg": 1, "sign": 1.0},
            {"program_leg": 1, "caller_leg": 0, "sign": 1.0},
        ]
        if selector != process_key
        else None
    )
    reference = tmp_path / "points.json"
    reference.write_text(
        json.dumps({process_key: points} if by_process else points), encoding="utf-8"
    )
    exports = []
    evaluations = []

    def load(artifact: Path, *, process: str, **kwargs: object) -> object:
        def export(**options: object) -> dict[str, object]:
            exports.append((process, options))
            result = _export(process_key if process == selector else process)
            if process == selector:
                result["programs"][0]["crossing"] = crossing
            return result

        def evaluate(batch: object, **options: object) -> ResolvedEvaluation:
            evaluations.append((process, batch, options))
            return ResolvedEvaluation(
                values=(((2.5,),),) * 2,
                helicity_ids=("h0",),
                color_ids=("flow:0",),
            )

        return SimpleNamespace(export_recurrence=export, evaluate_resolved=evaluate)

    monkeypatch.setattr(Runtime, "load", load)
    destination = tmp_path / "capture.json"
    errors = io.StringIO()
    assert (
        run_cli(
            [
                "export-recurrence",
                str(tmp_path),
                "-o",
                str(destination),
                "--process",
                selector,
                "--reference-points",
                str(reference),
                "--json",
            ],
            stdout=io.StringIO(),
            stderr=errors,
        )
        == 0
    )
    programs = json.loads(destination.read_text(encoding="utf-8"))["programs"]
    program = programs[0]
    assert program["process_key"] == process_key
    assert program["crossing"] == crossing
    assert program["reference_values"] == {
        "momenta": points,
        "helicity_ids": ["h0"],
        "values": [[2.5]],
    }
    assert all("reference_values" not in support for support in programs[1:])
    expected_exports = [(selector, {"color_flows": None})]
    if process_key == "member":
        expected_exports.append(("p0", {"color_flows": ("flow:0",)}))
    assert exports == expected_exports
    assert evaluations == [
        (selector, points * 2, {"color_flows": ["flow:0"], "precision": 16})
    ]
    assert not errors.getvalue()


@pytest.mark.parametrize("batch", (None, {}, "invalid", 7, []))
def test_cli_rejects_invalid_mapped_reference_batch_before_replacing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, batch: object
) -> None:
    calls = []

    def load(artifact: Path, *, process: str, **kwargs: object) -> object:
        def export(**options: object) -> dict[str, object]:
            calls.append(options)
            return _export("p0")

        def evaluate(*args: object, **kwargs: object) -> None:
            pytest.fail("invalid reference input must not be evaluated")

        return SimpleNamespace(export_recurrence=export, evaluate_resolved=evaluate)

    monkeypatch.setattr(Runtime, "load", load)
    reference = tmp_path / "points.json"
    reference.write_text(json.dumps({"p0": batch}), encoding="utf-8")
    destination = tmp_path / "capture.json"
    destination.write_text("existing capture\n", encoding="utf-8")
    errors = io.StringIO()
    assert (
        run_cli(
            [
                "export-recurrence",
                str(tmp_path),
                "-o",
                str(destination),
                "--process",
                "p0",
                "--reference-points",
                str(reference),
                "--json",
            ],
            stdout=io.StringIO(),
            stderr=errors,
        )
        == 2
    )
    assert "reference" in errors.getvalue()
    assert destination.read_text(encoding="utf-8") == "existing capture\n"
    assert calls == [{"color_flows": None}]


def test_public_reference_values_follow_program_axes_and_batched_semantics() -> None:
    from copy import deepcopy

    capture = _export("p0")
    program = capture["programs"][0]
    program["helicity_outputs"] = [
        {"id": "h1", "helicities": []},
        {"id": "h0", "helicities": []},
    ]
    second = {**program, "colour_id": "flow:1"}
    capture["programs"].append(second)
    calls = []

    def evaluate(points: object, **options: object) -> ResolvedEvaluation:
        calls.append((points, options))
        return ResolvedEvaluation(
            values=(((1.0, 2.0), (3.0, 4.0)),) * 2,
            helicity_ids=("h0", "h1"),
            color_ids=("flow:1", "flow:0"),
        )

    backend = SimpleNamespace(
        export_recurrence=lambda **options: deepcopy(capture),
        evaluate_resolved=evaluate,
        physics=None,
        evaluate=None,
        set_model_parameters=None,
        clear=None,
        mute_warnings=None,
        unmute_warnings=None,
    )
    runtime = Runtime(backend)
    assert runtime.export_recurrence() == capture
    assert not calls
    points = [[[10, -0.0, 0, 10]]]
    exported = runtime.export_recurrence(reference_momenta=points)
    references = [program["reference_values"] for program in exported["programs"]]
    assert references[0] == {
        "momenta": points,
        "helicity_ids": ["h1", "h0"],
        "values": [[4.0, 2.0]],
    }
    assert references[1]["values"] == [[3.0, 1.0]]
    assert calls == [
        (
            points * 2,
            {
                "helicities": None,
                "color_flows": ("flow:0", "flow:1"),
                "precision": 16,
            },
        )
    ]
    assert "reference_values" not in capture["programs"][0]


@pytest.mark.parametrize("conflict", ("schema_version", "parameters"))
def test_cli_rejects_inconsistent_exports_before_replacing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, conflict: str
) -> None:
    def load(artifact: Path, *, process: str, **kwargs: object) -> object:
        result = _export(process)
        if process == "p1":
            if conflict == "schema_version":
                result["schema_version"] = 3
            else:
                result["parameters"][0]["value"]["re"] = 0.2
        return SimpleNamespace(export_recurrence=lambda **options: result)

    monkeypatch.setattr(Runtime, "load", load)
    destination = tmp_path / "capture.json"
    destination.write_text("previous capture\n", encoding="utf-8")
    errors = io.StringIO()
    assert (
        run_cli(
            [
                "export-recurrence",
                str(tmp_path),
                "-o",
                str(destination),
                "--process",
                "p0",
                "--process",
                "p1",
                "--json",
            ],
            stdout=io.StringIO(),
            stderr=errors,
        )
        == 2
    )
    assert "differs for 'p1'" in errors.getvalue()
    assert destination.read_text(encoding="utf-8") == "previous capture\n"


def test_cli_help_says_colour_flow_ids_are_per_process() -> None:
    from pyamplicol.cli.recurrence_export import add_export_options

    parser = argparse.ArgumentParser(prog="export-recurrence")
    add_export_options(parser)
    text = " ".join(parser.format_help().split())
    assert "Flow IDs are per process" in text
    assert "combine this with --process" in text
