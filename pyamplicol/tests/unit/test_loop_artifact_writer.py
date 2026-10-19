# SPDX-License-Identifier: 0BSD
"""Ordinary transactional artifacts carry source-owned, graph-free virtual plans."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol.api.requests import ModelSource
from pyamplicol.artifacts import load_manifest
from pyamplicol.config import (
    Action,
    EvaluatorConfig,
    GenerationConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.artifact_writer import (
    _GenerationConfigProvenance,
    write_schema_v3_artifact,
)
from pyamplicol.generation.loop_artifact import (
    LoopExecutionMetadata,
    LoopProcessArtifact,
    LoopSchemeDeclaration,
)
from pyamplicol.generation.validation import ValidationPointRecord
from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
from pyamplicol.loops.physical_orders import BornSquaredSector, PhysicalOrderLayout
from pyamplicol.models.loading import compile_model_source


@pytest.fixture(autouse=True)
def target_metadata(monkeypatch):
    # The experimental native consumer is built separately from the wheel SDK.
    # No platform detection or package installation is under test here.
    monkeypatch.setattr(
        "pyamplicol.generation.artifact_writer.active_native_build_inputs_sha256",
        lambda: "a" * 64,
    )
    monkeypatch.setattr(
        "pyamplicol.generation.artifact_writer._target_metadata",
        lambda _config: ({"triple": "x86_64-unknown-linux-gnu", "cpu_features": []}, 1),
    )


def source_process(path):
    backend = pytest.importorskip("_loop_reduction")
    data = json.loads(backend.PortableLoopPlan(str(path)).metadata_json())

    def order(value):
        return tuple((name, power) for name, power in value)

    axes = data["order_layout"]
    orders = PhysicalOrderLayout(
        tuple(
            BornSquaredSector(
                order(s["order"]), tuple((order(a), order(b)) for a, b in s["pairs"])
            )
            for s in axes["born"]
        ),
        tuple(
            VirtualOrderSector(
                order(s["order"]),
                tuple(
                    BornLoopPair(order(p["born"]), order(p["loop"])) for p in s["pairs"]
                ),
            )
            for s in axes["virtual"]
        ),
        tuple(axes["correction_names"]),
    )
    # These are fixture declarations, not inferences from the model's name.
    scheme = LoopSchemeDeclaration(
        dimensional_regularization="HV",
        external_states="physical-four-dimensional",
        gamma5="source-UFO-four-dimensional-prescription",
        renormalization="five-flavour-decoupling",
        mass_scheme="real-on-shell",
        finite_restoration="source-UFO-local-rules",
        model_convention="loop_sm lhv=1 supplied QCD rules",
    )
    metadata = LoopExecutionMetadata(
        scheme,
        orders,
        "model",
        ("model",),
        ("double", "double-double", "arbitrary"),
        True,
    )
    return LoopProcessArtifact(
        "gg",
        "g g > g g",
        tuple(data["external_pdgs"]),
        path,
        metadata,
        ValidationPointRecord(
            "gg", "g g > g g", 0, error="not regenerated for artifact transport test"
        ),
        {},
    )


@pytest.fixture(scope="module")
def retained_source_process():
    path = os.environ.get("PYAMPLICOL_NATIVE_SOURCE_PACBIN")
    if not path:
        pytest.skip(
            "retained source PACBIN required; do not regenerate numerical physics "
            "for transport tests"
        )
    return source_process(Path(path))


def write_artifact(destination, processes, *, mode="error", staged_validation=None):
    configuration = _GenerationConfigProvenance.from_config(
        RunConfig(
            action=Action.GENERATE,
            generation=GenerationConfig(emit_api_bundle=False),
            process=ProcessConfig(loop_coupling_orders=("QCD",)),
            evaluator=EvaluatorConfig(execution_mode="loop_recurrence"),
        )
    )
    return write_schema_v3_artifact(
        destination,
        mode=mode,
        source=ModelSource(kind="built-in-sm"),
        compiled_model=compile_model_source("built-in-sm", use_cache=False),
        configuration=configuration,
        processes=processes,
        timings={"total": 0.0},
        staged_validation=staged_validation,
    )


def test_source_artifact_roundtrip_and_append(retained_source_process, tmp_path):
    destination = tmp_path / "source-loop"
    write_artifact(destination, (retained_source_process,))
    manifest = load_manifest(destination)
    assert manifest.processes[0]["required_runtime_capabilities"] == (
        "rusticol.loop-runtime.v3",
    )
    physics = json.loads((destination / "processes/gg/physics.json").read_text())
    assert physics["metadata"]["epsilon_powers"] == [0, -1, -2]
    assert physics["metadata"]["scheme_status"] == "declared-not-certified"
    assert (
        destination / "processes/gg/loop-runtime.pacbin"
    ).read_bytes() == retained_source_process.loop_runtime_path.read_bytes()
    second = replace(
        retained_source_process,
        process_id="gg_second",
        validation_point=replace(
            retained_source_process.validation_point, process_id="gg_second"
        ),
    )
    write_artifact(destination, (second,), mode="append")
    assert len(load_manifest(destination).processes) == 2
    changed = replace(
        second,
        metadata=replace(
            second.metadata,
            scheme=replace(second.metadata.scheme, dimensional_regularization="FDH"),
        ),
    )
    with pytest.raises(ValueError, match="same declared scheme"):
        write_artifact(destination, (changed,), mode="append")
    assert len(load_manifest(destination).processes) == 2


def test_source_artifact_rejects_metadata_mismatch_transactionally(
    retained_source_process, tmp_path
):
    wrong = replace(
        retained_source_process,
        metadata=replace(
            retained_source_process.metadata,
            orders=replace(
                retained_source_process.metadata.orders, correction_names=("QED",)
            ),
        ),
    )
    destination = tmp_path / "bad"
    with pytest.raises(ValueError, match="order axes"):
        write_artifact(destination, (wrong,))
    assert not destination.exists()
    with pytest.raises(ValueError, match="model-supplied R2"):
        replace(
            retained_source_process.metadata,
            r2_strategy="direct",
            available_r2_strategies=("direct",),
        )
    with pytest.raises(ValueError, match="explicitly supplied"):
        replace(retained_source_process.metadata.scheme, gamma5="")


def test_failed_staged_validation_preserves_existing_artifact(
    retained_source_process, tmp_path
):
    destination = tmp_path / "transaction"
    write_artifact(destination, (retained_source_process,))
    before = {
        p.relative_to(destination): p.read_bytes()
        for p in destination.rglob("*")
        if p.is_file()
    }
    seen = []

    def reject(staged):
        manifest = load_manifest(staged)
        assert manifest.processes[0]["id"] == "gg"
        assert staged != destination
        seen.append(staged)
        raise ValueError("deliberate saved numerical mismatch")

    with pytest.raises(ValueError, match="deliberate saved numerical mismatch"):
        write_artifact(
            destination,
            (retained_source_process,),
            mode="replace",
            staged_validation=reject,
        )
    assert len(seen) == 1
    assert not seen[0].exists()
    assert {
        p.relative_to(destination): p.read_bytes()
        for p in destination.rglob("*")
        if p.is_file()
    } == before


def test_changed_single_pole_fails_transactional_numerical_validation(
    retained_source_process, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from pyamplicol.api import GenerationError, Runtime
    from pyamplicol.api.virtual_results import VirtualEvaluation, VirtualLaneDiagnostic
    from pyamplicol.config import GenerationValidationConfig, R2Strategy
    from pyamplicol.generation.loop_service import _validate_artifact, _ValidationValues
    from pyamplicol.loops.strategy import R2Selection

    process = retained_source_process
    destination = tmp_path / "numerical-transaction"
    write_artifact(destination, (process,))
    previous_manifest = (destination / "artifact.json").read_bytes()
    orders = process.metadata.orders
    reference = _ValidationValues(
        ((),), (1.0,), (5.0,) * len(orders.born), (1.0, 2.0, 3.0) * len(orders.virtual)
    )
    changed = VirtualEvaluation(
        (reference.born,),
        (((1.0, 2.25, 3.0),) * len(orders.virtual),),
        tuple(sector.order for sector in orders.born),
        orders.virtual,
        (VirtualLaneDiagnostic("success", "double", 53, False),),
        R2Selection(R2Strategy.AUTO, R2Strategy.MODEL, "only complete strategy"),
    )

    def load(staged, *, process):
        return SimpleNamespace(
            execution_mode="loop_recurrence",
            artifact_id=load_manifest(staged).artifact_id,
            representative_process_key=process,
            evaluate=lambda *args, **kwargs: changed,
        )

    monkeypatch.setattr(Runtime, "load", load)
    backend = SimpleNamespace(
        _generation_config=SimpleNamespace(validation=GenerationValidationConfig())
    )
    context = SimpleNamespace(
        expanded=(SimpleNamespace(request=SimpleNamespace(name=process.process_id)),)
    )
    with pytest.raises(GenerationError, match="virtual coefficient 1"):
        write_artifact(
            destination,
            (process,),
            mode="replace",
            staged_validation=lambda staged: _validate_artifact(
                backend, staged, context, (process,), {process.process_id: reference}
            ),
        )
    assert (destination / "artifact.json").read_bytes() == previous_manifest
