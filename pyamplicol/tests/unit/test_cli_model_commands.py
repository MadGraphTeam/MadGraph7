# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

import pytest

from pyamplicol.cli import DefaultCliServices, parse_cli
from pyamplicol.models.contracts import CompiledModelIR, CompiledParticleRecord
from pyamplicol.reporting import NullProgressSink


class _CompiledModel:
    ir = CompiledModelIR(
        name="toy",
        orders=(),
        parameters=(),
        particles=(
            CompiledParticleRecord(
                "d",
                "d~",
                1,
                2,
                3,
                "ZERO",
                "ZERO",
                -1.0 / 3.0,
                0,
                True,
                False,
                None,
            ),
            CompiledParticleRecord(
                "d~",
                "d",
                -1,
                2,
                -3,
                "ZERO",
                "ZERO",
                1.0 / 3.0,
                0,
                True,
                False,
                None,
            ),
            CompiledParticleRecord(
                "z",
                "z",
                23,
                3,
                1,
                "MZ",
                "WZ",
                0.0,
                0,
                True,
                False,
                None,
            ),
        ),
        couplings=(),
        propagators=(),
        vertex_terms=(),
        oriented_kernels=(),
    )

    def __init__(self) -> None:
        self.name = "toy"
        self.supported = True
        self.source = {"kind": "built-in-sm"}
        self.producer = {"pyamplicol": "test"}
        self.capabilities = {"particle_count": 3}
        self.issues: tuple[object, ...] = ()
        self.conversion_seconds = 0.25
        self.phase_timings = {"total": 0.25}
        self.parameter_defaults = {"mass": (1.0, 0.0)}

    def write(self, path: Path) -> Path:
        return path.with_name(path.name + ".pyAmplicol-model.json")


def test_default_model_compile_calls_existing_compiler_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.models import loading

    calls: list[dict[str, object]] = []

    def compile_model_source(source: str, **options: object) -> _CompiledModel:
        calls.append({"source": source, **options})
        return _CompiledModel()

    monkeypatch.setattr(loading, "compile_model_source", compile_model_source)
    output = tmp_path / "toy"
    invocation = parse_cli(("model", "compile", "built-in-sm", str(output)))
    result = DefaultCliServices().model_compile(
        invocation.resolve().effective,
        NullProgressSink(),
    )

    assert calls == [
        {
            "source": "built-in-sm",
            "restriction": "default",
            "simplify": True,
            "cache_dir": None,
            "use_cache": True,
            "require_supported": True,
        }
    ]
    assert result == {
        "model": "toy",
        "supported": True,
        "output": str(output) + ".pyAmplicol-model.json",
        "conversion_seconds": 0.25,
        "phase_timings": {"total": 0.25},
        "source": {"kind": "built-in-sm"},
        "capabilities": {"particle_count": 3},
    }


def test_default_model_processes_uses_compiled_model_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.models import loading

    monkeypatch.setattr(
        loading,
        "compile_model_source",
        lambda *args, **kwargs: _CompiledModel(),
    )
    invocation = parse_cli(("model", "processes", "d d~ > z"))
    result = DefaultCliServices().model_processes(
        invocation.resolve().effective,
        NullProgressSink(),
    )

    assert isinstance(result, dict)
    assert result["model"] == "toy"
    assert result["n_entries"] == 1
    entry = result["entries"][0]
    assert entry["process"] == "d d~ > z"
    assert entry["ir"]["outgoing_pdgs"] == [-1, 1, 23]
