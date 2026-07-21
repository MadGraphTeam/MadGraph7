# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import json
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from pyamplicol._internal.versions import (
    COMPILED_RUNTIME_SELECTORS_CAPABILITY,
    SYMJIT_F64_RUNTIME_CAPABILITY,
)
from pyamplicol.api import (
    ArtifactError,
    ColorFlow,
    ContractedColorComponent,
    ExternalParticle,
    HelicityConfiguration,
    ModelParameter,
    ProcessPhysics,
    ResolvedEvaluation,
    Runtime,
    RuntimeBackend,
)
from pyamplicol.api.errors import CompatibilityError, EvaluationError
from pyamplicol.artifacts import ArtifactManifest


def _native_physics(accuracy: str = "lc") -> SimpleNamespace:
    particles = [
        SimpleNamespace(
            index=index,
            label=index + 1,
            name=name,
            pdg_id=pdg,
            state="incoming" if index < 2 else "outgoing",
            momentum_slot=index,
        )
        for index, (name, pdg) in enumerate((("u", 2), ("u~", -2), ("g", 21)))
    ]
    helicities = [
        SimpleNamespace(
            id="h0",
            index=0,
            values=[-1, 1, 1],
            computed=True,
            structural_zero=False,
            representative_id="h0",
            coefficient=2.0,
        )
    ]
    flows = []
    contracted = []
    if accuracy == "lc":
        flows.append(
            SimpleNamespace(
                id="c0",
                index=0,
                word=[1, 2, 3],
                computed=True,
                representative_id="c0",
                coefficient=1.0,
            )
        )
    else:
        contracted.append(
            SimpleNamespace(id="contracted", index=0, description="summed color")
        )
    return SimpleNamespace(
        process_id="uux_g",
        process="u u~ > g",
        color_accuracy=accuracy,
        helicity_coverage="complete",
        color_coverage="complete" if accuracy == "lc" else "contracted",
        color_kind="physical-lc-flows" if accuracy == "lc" else "contracted-color",
        structural_zero_helicity_count=0,
        external_particles=particles,
        helicities=helicities,
        color_flows=flows,
        contracted_color_components=contracted,
        reduction=SimpleNamespace(
            kind="lc-diagonal" if accuracy == "lc" else "contracted-color",
            groups=[
                SimpleNamespace(
                    id="g0",
                    representative_helicity_id="h0",
                    representative_color_id=(
                        "c0" if accuracy == "lc" else "contracted"
                    ),
                    physical_helicity_ids=["h0"],
                    physical_color_ids=["c0" if accuracy == "lc" else "contracted"],
                )
            ],
        ),
        model_parameters=[
            SimpleNamespace(
                name="aS",
                kind="coupling",
                default_real=0.118,
                default_imaginary=0.0,
                mutable=True,
            )
        ],
        selector_capabilities=["helicity"]
        + (["color_flow"] if accuracy == "lc" else []),
    )


class _NativeArtifactError(Exception):
    pass


class _NativeRuntime:
    physics_value = _native_physics()
    load_arguments: tuple[object, ...] | None = None
    execution_mode = "compiled"
    last_evaluate_options: dict[str, object] | None = None
    last_benchmark_options: dict[str, object] | None = None
    last_profile_options: dict[str, object] | None = None

    def __init__(self) -> None:
        self.parameter_updates: list[dict[str, complex | float | int]] = []
        self.muted = False

    @classmethod
    def load(cls, artifact: Path, **kwargs: object) -> _NativeRuntime:
        cls.load_arguments = (artifact, kwargs)
        return cls()

    @property
    def physics(self) -> SimpleNamespace:
        return self.physics_value

    def metadata_json(self) -> str:
        return json.dumps(
            {"execution_mode": self.execution_mode, "process_key": "uux_g"}
        )

    def evaluate(self, _momenta: object, **kwargs: object) -> list[object]:
        type(self).last_evaluate_options = dict(kwargs)
        return [Decimal("1.25")] if kwargs["precision"] == 32 else [2.0]

    def evaluate_resolved(self, _momenta: object, **kwargs: object) -> SimpleNamespace:
        accuracy = self.physics_value.color_accuracy
        color_id = "c0" if accuracy == "lc" else "contracted"
        scalar: object = Decimal("1.25") if kwargs["precision"] == 32 else 2.0
        return SimpleNamespace(
            values=[[[scalar]]],
            helicity_ids=["h0"],
            color_ids=[color_id],
            color_accuracy=accuracy,
        )

    def _benchmark_f64_wall_time(
        self, _momenta: object, _repetitions: int, **kwargs: object
    ) -> float:
        type(self).last_benchmark_options = dict(kwargs)
        return 0.25

    def profile(self, _momenta: object, **kwargs: object) -> dict[str, object]:
        type(self).last_profile_options = dict(kwargs)
        return {"wall_time_s": 0.25}

    def profile_repeated(
        self, _momenta: object, _repetitions: int, **kwargs: object
    ) -> dict[str, object]:
        type(self).last_profile_options = dict(kwargs)
        return {"wall_time_s": 0.5}

    def set_model_parameters(self, mapping: dict[str, complex | float | int]) -> None:
        self.parameter_updates.append(mapping)

    def mute_warnings(self) -> None:
        self.muted = True

    def unmute_warnings(self) -> None:
        self.muted = False

    def take_warnings(self) -> list[str]:
        return ["native warning"]


class _PreSelectorNativeRuntime(_NativeRuntime):
    @classmethod
    def load(cls, artifact: Path, **kwargs: object) -> _PreSelectorNativeRuntime:
        cls.load_arguments = (artifact, kwargs)
        return cls()

    def evaluate(
        self,
        _momenta: object,
        *,
        helicities: object = None,
        color_flows: object = None,
        precision: int = 16,
    ) -> list[object]:
        type(self).last_evaluate_options = {
            "helicities": helicities,
            "color_flows": color_flows,
            "precision": precision,
        }
        return [2.0]


class _ExactExecutor:
    def __init__(self, _artifact: Path, _process_id: str, _runtime: object) -> None:
        pass

    def evaluate_resolved(
        self, _momenta: object, **_kwargs: object
    ) -> ResolvedEvaluation:
        accuracy = _NativeRuntime.physics_value.color_accuracy
        color_id = "c0" if accuracy == "lc" else "contracted"
        return ResolvedEvaluation(
            values=(((Decimal("1.25"),),),),
            helicity_ids=("h0",),
            color_ids=(color_id,),
            color_accuracy=accuracy,
        )


def _install_native(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("pyamplicol._rusticol")
    module.Runtime = _NativeRuntime  # type: ignore[attr-defined]
    module.ArtifactError = _NativeArtifactError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _selector_manifest(
    tmp_path: Path,
    *,
    capabilities: tuple[str, ...],
) -> ArtifactManifest:
    physics_path = Path("processes/uux_g/physics.json")
    destination = tmp_path / physics_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "extensions": {
                    "runtime_selectors": {
                        "axes": {
                            "helicity": {
                                "runtime_contract": "complete-reusable"
                            },
                            "color_flow": {
                                "runtime_contract": "complete-reusable"
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifact.json").write_text("{}", encoding="utf-8")
    process = {
        "id": "uux_g",
        "physics_path": physics_path.as_posix(),
        "required_runtime_capabilities": capabilities,
        "aliases": (),
    }
    return ArtifactManifest(
        root=tmp_path,
        kind="pyamplicol-process",
        artifact_id="0" * 64,
        created_utc="2026-07-21T00:00:00Z",
        producer={},
        model={},
        configuration={},
        processes=(process,),
        default_process_id="uux_g",
        runtime={"required_runtime_capabilities": capabilities},
        payloads=(),
        dependencies=(),
        extensions={},
    )


def test_runtime_discovery_does_not_import_native_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module
    imports: list[str] = []

    def tracked_import(name: str, package: str | None = None) -> Any:
        imports.append(name)
        return real_import(name, package)

    monkeypatch.delitem(sys.modules, "pyamplicol._rusticol", raising=False)
    monkeypatch.setattr(importlib, "import_module", tracked_import)
    runtime = real_import("pyamplicol.runtime")

    assert callable(runtime.load_runtime_backend)
    assert "pyamplicol._rusticol" not in imports


def test_adapter_maps_typed_metadata_totals_and_runtime_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    monkeypatch.setattr(
        "pyamplicol.runtime.symbolica_exact.SymbolicaExactExecutor",
        _ExactExecutor,
    )
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("lc")
    backend = load_runtime_backend(
        tmp_path,
        process="uux_g",
        model_parameters={"aS": 0.12},
        mute_warnings=True,
    )

    assert isinstance(backend, RuntimeBackend)
    assert isinstance(backend.physics, ProcessPhysics)
    assert isinstance(backend.physics.external_particles[0], ExternalParticle)
    assert isinstance(backend.physics.helicities[0], HelicityConfiguration)
    assert isinstance(backend.physics.color_flows[0], ColorFlow)
    assert isinstance(backend.physics.model_parameters[0], ModelParameter)
    assert backend.physics.external_particles[0].name == "u"
    assert backend.physics.external_particles[0].pdg_id == 2
    assert backend.physics.external_particles[0].state == "incoming"
    assert backend.physics.selector_capabilities == ("helicity", "color_flow")
    assert backend.evaluate([], precision=16) == (2.0 + 0.0j,)
    assert backend.evaluate([], precision=32) == (Decimal("1.25"),)

    resolved = backend.evaluate_resolved([], precision=32)
    assert isinstance(resolved, ResolvedEvaluation)
    assert resolved.shape == (1, 1, 1)
    assert resolved.total() == (Decimal("1.25"),)

    backend.set_model_parameters({"aS": 0.13})
    backend.mute_warnings()
    assert backend._runtime.parameter_updates == [{"aS": 0.13}]
    assert backend._runtime.muted is True
    backend.unmute_warnings()
    assert backend._runtime.muted is False
    assert backend.take_warnings() == ("native warning",)

    path, options = _NativeRuntime.load_arguments or (None, {})
    assert path == tmp_path.resolve()
    assert options == {
        "process": "uux_g",
        "model_parameters": {"aS": 0.12},
        "mute_warnings": True,
    }

    public = Runtime.load(tmp_path, process="uux_g")
    assert isinstance(public.physics, ProcessPhysics)
    assert public.evaluate([], precision=32) == (Decimal("1.25"),)


def test_adapter_routes_eager_high_precision_to_eager_exact_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    monkeypatch.setattr(
        "pyamplicol.runtime.eager_exact.EagerExactExecutor",
        _ExactExecutor,
    )
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.execution_mode = "eager"
    try:
        backend = load_runtime_backend(tmp_path, process="uux_g")
        assert backend.evaluate([], precision=32) == (Decimal("1.25"),)
    finally:
        _NativeRuntime.execution_mode = "compiled"


def test_adapter_accepts_one_based_color_flow_ordinals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("lc")
    _NativeRuntime.last_evaluate_options = None
    backend = load_runtime_backend(tmp_path, process="uux_g")

    backend.evaluate([], color_flows=("1",))

    assert _NativeRuntime.last_evaluate_options is not None
    assert _NativeRuntime.last_evaluate_options["color_flows"] == ("c0",)
    with pytest.raises(EvaluationError, match=r"choose 1\.\.1"):
        backend.evaluate([], color_flows=("2",))


def test_adapter_resolves_per_point_selector_ids_to_native_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("lc")
    _NativeRuntime.last_evaluate_options = None
    backend = load_runtime_backend(tmp_path, process="uux_g")

    backend.evaluate(
        [(), ()],
        helicity_by_point=("h0", "h0"),
        color_flow_by_point=("1", "c0"),
    )

    assert _NativeRuntime.last_evaluate_options is not None
    assert _NativeRuntime.last_evaluate_options["helicity_by_point"] == (0, 0)
    assert _NativeRuntime.last_evaluate_options["color_flow_by_point"] == (0, 0)
    with pytest.raises(EvaluationError, match=r"helicity_by_point\[1\]"):
        backend.evaluate(
            [(), ()],
            helicity_by_point=("h0", "missing"),
        )


def test_adapter_requires_selector_capability_for_reusable_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_native(monkeypatch)
    import pyamplicol.runtime.backend as backend_module

    manifest = _selector_manifest(
        tmp_path,
        capabilities=(SYMJIT_F64_RUNTIME_CAPABILITY,),
    )
    manifest_options: list[dict[str, object]] = []

    def load_adapter_manifest(
        _path: Path,
        **options: object,
    ) -> ArtifactManifest:
        manifest_options.append(options)
        return manifest

    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        load_adapter_manifest,
    )

    with pytest.raises(
        CompatibilityError,
        match=r"declares reusable runtime selectors.*does not require",
    ):
        backend_module.load_runtime_backend(tmp_path, process="uux_g")
    assert manifest_options == [{"verify_payloads": False}]


def test_adapter_checks_native_selector_callable_against_capability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _install_native(monkeypatch)
    module.Runtime = _PreSelectorNativeRuntime  # type: ignore[attr-defined]
    import pyamplicol.runtime.backend as backend_module

    capabilities = (
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        SYMJIT_F64_RUNTIME_CAPABILITY,
    )
    manifest = _selector_manifest(tmp_path, capabilities=capabilities)
    monkeypatch.setattr(
        backend_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )

    with pytest.raises(
        CompatibilityError,
        match="installed native runtime does not accept per-point selectors",
    ):
        backend_module.load_runtime_backend(tmp_path, process="uux_g")


def test_adapter_does_not_pass_selector_keywords_when_axes_are_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _install_native(monkeypatch)
    module.Runtime = _PreSelectorNativeRuntime  # type: ignore[attr-defined]
    from pyamplicol.runtime import load_runtime_backend

    _PreSelectorNativeRuntime.last_evaluate_options = None
    backend = load_runtime_backend(tmp_path, process="uux_g")

    assert backend.evaluate(
        [],
        helicities=(),
        color_flows=(),
        helicity_by_point=(),
        color_flow_by_point=(),
    ) == (2.0 + 0.0j,)
    assert _PreSelectorNativeRuntime.last_evaluate_options == {
        "helicities": None,
        "color_flows": None,
        "precision": 16,
    }

    with pytest.raises(CompatibilityError, match="does not support per-point"):
        backend.evaluate([()], helicity_by_point=("h0",))


def test_native_wall_timer_resolves_per_point_selectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("lc")
    backend = load_runtime_backend(tmp_path, process="uux_g")
    elapsed = backend._benchmark_f64_wall_time(
        ((), ()),
        3,
        helicity_by_point=("h0", "h0"),
        color_flow_by_point=("1", "c0"),
    )

    assert elapsed == 0.25
    assert _NativeRuntime.last_benchmark_options == {
        "helicities": None,
        "color_flows": None,
        "precision": 16,
        "helicity_by_point": (0, 0),
        "color_flow_by_point": (0, 0),
    }


def test_native_profile_resolves_per_point_selectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("lc")
    backend = load_runtime_backend(tmp_path, process="uux_g")
    payload = backend.profile_repeated(
        ((), ()),
        3,
        helicity_by_point=("h0", "h0"),
        color_flow_by_point=("1", "c0"),
    )

    assert payload == {"wall_time_s": 0.5}
    assert _NativeRuntime.last_profile_options == {
        "helicities": None,
        "color_flows": None,
        "precision": 16,
        "include_values": False,
        "helicity_by_point": (0, 0),
        "color_flow_by_point": (0, 0),
    }


def test_adapter_maps_contracted_color_and_native_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_native(monkeypatch)
    from pyamplicol.runtime import load_runtime_backend

    _NativeRuntime.physics_value = _native_physics("full")
    backend = load_runtime_backend(
        tmp_path,
        process=None,
        model_parameters=None,
        mute_warnings=False,
    )

    assert backend.physics.color_flows == ()
    assert isinstance(
        backend.physics.contracted_color_components[0],
        ContractedColorComponent,
    )
    assert backend.evaluate_resolved([], precision=16).color_ids == ("contracted",)
    assert backend.physics.selector_capabilities == ("helicity",)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise _NativeArtifactError("broken artifact")

    backend._runtime.evaluate = fail
    with pytest.raises(ArtifactError, match="broken artifact"):
        backend.evaluate([])
