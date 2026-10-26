# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

_GATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "developer"
    / "otf_contracted_parity_gate.py"
)
_GATE_SPEC = importlib.util.spec_from_file_location(
    "test_otf_contracted_parity_gate_module",
    _GATE_PATH,
)
if _GATE_SPEC is None or _GATE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot load the OTF contracted parity gate")
gate = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(gate)


def test_authority_and_compact_preflights_precede_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    prepared = tmp_path / "prepared-model"
    prepared.write_bytes(b"prepared")
    point = tmp_path / "point.json"
    point.write_text(json.dumps([[1.0, 0.0, 0.0, 1.0]]), encoding="ascii")
    recurrence = tmp_path / "recurrence.json"
    recurrence.write_text("{}\n", encoding="ascii")
    artifact = tmp_path / "artifact"
    output = tmp_path / "result.json"

    class FakeCompiledModelSource:
        def compile(self) -> object:
            events.append("model-compile")
            return object()

    class FakeModelSource:
        @staticmethod
        def from_path(_path: Path) -> FakeCompiledModelSource:
            return FakeCompiledModelSource()

    class FakeProcessSet:
        @staticmethod
        def from_expressions(*_args: object, **_kwargs: object) -> object:
            return object()

    class FakeGenerator:
        def __init__(self, _config: object) -> None:
            pass

        def generate(self, *_args: object, **_kwargs: object) -> None:
            events.append("generation")
            artifact.mkdir()

    class FakeRuntimeInstance:
        execution_mode = "on-the-fly"

        @property
        def physics(self) -> object:
            raise AssertionError("ordering gate must not materialize dense physics")

        def inspect(self) -> dict[str, object]:
            events.append("compact-inspection")
            return {
                "runtime_metadata": {
                    "execution_mode": "on-the-fly",
                    "color_accuracy": "nlc",
                }
            }

        def evaluate(self, *_args: object, **_kwargs: object) -> tuple[complex, ...]:
            events.append("evaluation")
            return (1.0 + 0.0j,)

        def clear(self) -> None:
            pass

    class FakeRuntime:
        @staticmethod
        def load(*_args: object, **_kwargs: object) -> FakeRuntimeInstance:
            events.append("runtime-load")
            return FakeRuntimeInstance()

    package = ModuleType("pyamplicol")
    package.__file__ = str(tmp_path / "pyamplicol" / "__init__.py")
    package.Generator = FakeGenerator  # type: ignore[attr-defined]
    package.ModelSource = FakeModelSource  # type: ignore[attr-defined]
    package.ProcessSet = FakeProcessSet  # type: ignore[attr-defined]
    package.Runtime = FakeRuntime  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyamplicol", package)
    monkeypatch.setattr(gate.importlib.metadata, "version", lambda _name: "test")
    monkeypatch.setattr(gate, "_run_config", lambda *_args: object())

    def candidate_identity(*_args: object) -> dict[str, object]:
        events.append("candidate-authentication")
        return {
            "version": "test",
            "source_revision": None,
            "native_build_inputs_sha256": "a" * 64,
        }

    def recurrence_target(*_args: object, **_kwargs: object) -> tuple[Decimal, dict]:
        events.append("authority-authentication")
        return Decimal(1), {"kind": "mock-recurrence-authority"}

    def artifact_preflight(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("artifact-preflight")
        return {
            "producer": {"version": "test"},
            "effective_config": {"evaluator_optimization_cores": 4},
            "runtime_options": {"query_construction_threads": 4},
        }

    monkeypatch.setattr(gate, "_candidate_identity", candidate_identity)
    monkeypatch.setattr(gate, "_recurrence_target", recurrence_target)
    monkeypatch.setattr(gate, "_authenticate_artifact", artifact_preflight)

    assert (
        gate.main(
            (
                "--output",
                str(output),
                "--artifact",
                str(artifact),
                "--prepared-model",
                str(prepared),
                "--process",
                "g g > g",
                "--process-name",
                "ordering_gate",
                "--n-final",
                "1",
                "--family-id",
                "1",
                "--point",
                str(point),
                "--accuracy",
                "nlc",
                "--mode",
                "on-the-fly",
                "--precision",
                "16",
                "--query-construction-cores",
                "4",
                "--recurrence-authority",
                str(recurrence),
            )
        )
        == 0
    )
    assert events == [
        "candidate-authentication",
        "authority-authentication",
        "model-compile",
        "generation",
        "artifact-preflight",
        "runtime-load",
        "compact-inspection",
        "evaluation",
    ]
