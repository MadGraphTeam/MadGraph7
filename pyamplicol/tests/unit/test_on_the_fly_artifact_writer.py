# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import pyamplicol.generation.artifact_writer as artifact_writer
import pyamplicol.generation.service as service_module
from pyamplicol.api.errors import ArtifactError, GenerationError
from pyamplicol.api.requests import ModelSource, ProcessRequest, ProcessSet
from pyamplicol.api.services import Generator
from pyamplicol.artifacts import inspect_artifact, load_manifest
from pyamplicol.artifacts import inspection as artifact_inspection
from pyamplicol.config import (
    Action,
    EvaluatorConfig,
    GenerationConfig,
    GenerationValidationConfig,
    RunConfig,
)
from pyamplicol.generation.artifact_writer import (
    OnTheFlyProcessArtifact,
    _GenerationConfigProvenance,
    write_schema_v3_artifact,
)
from pyamplicol.generation.evaluator_container import (
    PacbinMemberKind,
    PacbinMemberSource,
    PacbinReader,
    write_pacbin_atomic,
)
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.generation.recurrence_physics import (
    build_on_the_fly_runtime_metadata,
)
from pyamplicol.generation.validation import ValidationPointRecord
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.loading import load_compiled_model
from pyamplicol.models.prepared import load_prepared_model_bundle
from pyamplicol.models.prepared_target import canonical_architecture

ROOT = Path(__file__).resolve().parents[2]
_PROCESS_ID = "d_dbar_to_z"
_SEED_MEMBER = "on-the-fly/process-seed-v1.bin"


def _seed_identity(*, process_digest: str = "1" * 64) -> dict[str, object]:
    return {
        "abi": "pyamplicol-on-the-fly-process-seed-identity-v1",
        "process_digest": process_digest,
        "compiled_model_digest": "2" * 64,
        "recurrence_template_catalog_digest": "3" * 64,
        "prepared_kernel_pack_digest": "4" * 64,
        "recurrence_direct_template_catalog_digest": "5" * 64,
        "semantic_digest": "6" * 64,
        "external_permutation": [0, 1, 2],
        "external_sources": [
            {
                "source_slot": slot,
                "public_label": slot + 1,
                "is_initial": slot < 2,
                "states": [
                    {
                        "state_index": 0,
                        "public_helicity": 0,
                        "prepared_mass_parameter_slot": None,
                    }
                ],
            }
            for slot in range(3)
        ],
    }


def _seed_identity_from_source_projection(
    source_projection: dict[str, object],
    *,
    process_digest: str | None = None,
) -> dict[str, object]:
    projected_digest = source_projection["process_digest"]
    permutation = source_projection["external_permutation"]
    sources = source_projection["external_sources"]
    assert isinstance(projected_digest, str)
    assert isinstance(permutation, list)
    assert isinstance(sources, list)
    identity = _seed_identity(
        process_digest=(projected_digest if process_digest is None else process_digest)
    )
    identity["external_permutation"] = list(permutation)
    compact_sources: list[dict[str, object]] = []
    for source in sources:
        assert isinstance(source, dict)
        states = source["states"]
        assert isinstance(states, list)
        compact_sources.append(
            {
                "source_slot": source["source_slot"],
                "public_label": source["public_label"],
                "is_initial": source["is_initial"],
                "states": [
                    {
                        "state_index": state["state_index"],
                        "public_helicity": state["public_helicity"],
                        "prepared_mass_parameter_slot": None,
                    }
                    for state in states
                    if isinstance(state, dict)
                ],
            }
        )
    identity["external_sources"] = compact_sources
    return identity


def _prepared_model_path() -> Path:
    architecture = canonical_architecture()
    return (
        ROOT
        / "src"
        / "pyamplicol"
        / "assets"
        / "prepared_models"
        / f"built-in-sm-jit-o2-{architecture}.pyamplicol-model"
    )


def _configuration() -> _GenerationConfigProvenance:
    return _GenerationConfigProvenance.from_config(
        RunConfig(
            action=Action.GENERATE,
            generation=GenerationConfig(
                emit_api_bundle=False,
                validation=GenerationValidationConfig(
                    enabled=False,
                    post_build_validation=False,
                ),
            ),
            evaluator=EvaluatorConfig(execution_mode="on-the-fly"),
        )
    )


def _write_seed_container(path: Path, payload: bytes = b"compact-process-seed"):
    return write_pacbin_atomic(
        path,
        (
            PacbinMemberSource(
                _SEED_MEMBER,
                PacbinMemberKind.ON_THE_FLY_PROCESS_SEED,
                source=io.BytesIO(payload),
            ),
        ),
    )


def _process_artifact(
    tmp_path: Path,
    *,
    runtime_name: str = "runtime.pacbin",
) -> OnTheFlyProcessArtifact:
    runtime_path = tmp_path / runtime_name
    runtime_index = _write_seed_container(runtime_path)
    bundle = load_prepared_model_bundle(_prepared_model_path())
    return OnTheFlyProcessArtifact(
        process_id=_PROCESS_ID,
        expression="d d~ > z",
        color_accuracy="lc",
        external_pdgs=(1, -1, 23),
        aliases=(),
        physics={
            "schema_version": 1,
            "kind": artifact_writer.ON_THE_FLY_PUBLIC_METADATA_KIND,
            "process_id": _PROCESS_ID,
            "process": "d d~ > z",
            "color_accuracy": "lc",
            "external_particles": [
                {
                    "index": index,
                    "label": index + 1,
                    "particle": particle,
                    "pdg": pdg,
                    "role": "initial" if index < 2 else "final",
                    "momentum_slot": index,
                    "momentum_components": ["E", "px", "py", "pz"],
                }
                for index, (particle, pdg) in enumerate(
                    (("d", 1), ("d~", -1), ("z", 23))
                )
            ],
            "model_parameters": [],
        },
        runtime_path=runtime_path,
        runtime_size_bytes=runtime_index.file_size,
        runtime_sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        runtime_member_count=1,
        runtime_unpacked_size_bytes=sum(
            member.length for member in runtime_index.members
        ),
        runtime_index_sha256=runtime_index.index_sha256,
        referenced_kernel_ids=frozenset(
            kernel.kernel_id for kernel in bundle.kernel_pack.kernels
        ),
        runtime_metadata={
            "runtime_parameters": [],
            "prepared_parameter_defaults": [],
            "parameter_projection": [],
            "external_legs": [],
            "particle_masses": [],
            "normalization": {},
            "process_seed_identity": _seed_identity(),
        },
        selector_policy={
            "color_coverage": "complete",
            "reference_color_word": None,
            "trace_reflections_folded": False,
            "selector_census": {
                "physical_helicity_count": 1,
                "physical_color_flow_count": 1,
            },
        },
        point_tile_size=128,
        validation_point=ValidationPointRecord(
            process_id=_PROCESS_ID,
            process="d d~ > z",
            seed=7,
            error="not sampled in writer test",
        ),
        generation_filters={"on_the_fly": {"selector_coverage": "complete"}},
    )


def _prepare_writer_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    source_revision = "a" * 40
    native_inputs = "b" * 64
    monkeypatch.setattr(
        artifact_writer,
        "active_source_revision",
        lambda: source_revision,
    )
    monkeypatch.setattr(
        artifact_writer,
        "active_native_source_identity",
        lambda: (source_revision, native_inputs),
    )
    monkeypatch.setattr(
        artifact_writer,
        "_target_metadata",
        lambda _config: (
            {"triple": "aarch64-apple-darwin", "cpu_features": []},
            1,
        ),
    )
    monkeypatch.setattr(
        artifact_writer,
        "_derive_eager_direct_descriptor",
        lambda source, **_widths: b"direct-table:" + source,
    )


def _payload_inventory(root: Path) -> tuple[tuple[object, ...], ...]:
    manifest = load_manifest(root)
    return tuple(
        sorted(
            (
                record.path,
                record.role,
                record.process_id,
            )
            for record in manifest.payloads
        )
    )


def test_on_the_fly_writer_publishes_one_seed_and_compact_metadata_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_writer_identity(monkeypatch)
    source_path = _prepared_model_path()
    compiled = load_compiled_model(source_path)
    first_process = _process_artifact(tmp_path, runtime_name="first.pacbin")
    second_process = _process_artifact(tmp_path, runtime_name="second.pacbin")
    first = tmp_path / "first-artifact"
    second = tmp_path / "second-artifact"

    for output, process in ((first, first_process), (second, second_process)):
        write_schema_v3_artifact(
            output,
            mode="error",
            source=ModelSource.from_path(source_path),
            compiled_model=compiled,
            configuration=_configuration(),
            processes=(process,),
            timings={"total": 0.1},
            api_bundle_hook=None,
        )

    runtime_relative = f"processes/{_PROCESS_ID}/on-the-fly-runtime.pacbin"
    for output in (first, second):
        manifest = load_manifest(output)
        structural_proof_path = f"processes/{_PROCESS_ID}/structural-source-proof.json"
        assert [
            (record.path, record.role)
            for record in manifest.payloads
            if record.path == structural_proof_path
            or record.role == artifact_writer.STRUCTURAL_SOURCE_PROOF_ROLE
        ] == []
        assert not (output / structural_proof_path).exists()
        runtime = output / runtime_relative
        with PacbinReader.open(runtime, verify_payloads=True) as reader:
            assert [
                (member.logical_path, member.kind) for member in reader.members
            ] == [(_SEED_MEMBER, PacbinMemberKind.ON_THE_FLY_PROCESS_SEED)]
        physics = json.loads(
            (output / f"processes/{_PROCESS_ID}/physics.json").read_text(
                encoding="utf-8"
            )
        )
        assert physics["kind"] == artifact_writer.ON_THE_FLY_PUBLIC_METADATA_KIND
        assert len(physics["external_particles"]) == 3
        assert not {
            "color_components",
            "helicities",
            "lc_topology_replay",
            "recurrence",
            "stages",
        }.intersection(physics)
        execution = json.loads(
            (output / f"processes/{_PROCESS_ID}/execution.json").read_text(
                encoding="utf-8"
            )
        )
        assert execution["kind"] == artifact_writer.ON_THE_FLY_RUNTIME_KIND
        assert execution["selector_policy"]["selector_census"] == {
            "physical_helicity_count": 1,
            "physical_color_flow_count": 1,
        }
        assert execution["runtime_container"] == {
            "kind": artifact_writer.ON_THE_FLY_RUNTIME_CONTAINER_KIND,
            "path": "on-the-fly-runtime.pacbin",
            "schema_version": 1,
            "seed_member_path": _SEED_MEMBER,
            "storage_abi": "pacbin-v1",
        }
        inspected = inspect_artifact(output).processes[0]
        assert inspected.execution_mode == "on-the-fly"
        assert inspected.physical_helicities == inspected.computed_helicities == 1
        assert inspected.physical_color_components == 1
        assert inspected.computed_color_components == 1
        assert inspected.helicity_coverage == inspected.color_coverage == "complete"
        assert inspected.generation_specialized_axes == ()

    assert isinstance(execution, dict)
    process_record = manifest.processes[0]
    capabilities = process_record["required_runtime_capabilities"]
    assert isinstance(capabilities, tuple)
    for index, mutation in enumerate(({"key": "wrong-process"}, {"invented": True})):
        mutated_execution = {**execution, **mutation}
        mutated_path = tmp_path / f"mutated-execution-{index}.json"
        mutated_path.write_text(json.dumps(mutated_execution), encoding="utf-8")
        with pytest.raises(ArtifactError, match=r"outer process|root fields"):
            artifact_inspection._execution_inspection(
                manifest,
                mutated_path,
                capabilities,
                process=process_record,
            )

    assert (first / runtime_relative).read_bytes() == (
        second / runtime_relative
    ).read_bytes()
    assert _payload_inventory(first) == _payload_inventory(second)


@pytest.mark.parametrize(
    ("roles", "expected_color_flows"),
    (
        (("singlet",) * 6, 1),
        (("adjoint",) * 4, 6),
        (
            (
                "fundamental",
                "antifundamental",
                "fundamental",
                "antifundamental",
                "adjoint",
                "adjoint",
            ),
            12,
        ),
        (
            (
                "fundamental",
                "antifundamental",
                "fundamental",
                "antifundamental",
                "fundamental",
                "antifundamental",
            ),
            6,
        ),
    ),
)
def test_on_the_fly_selector_census_matches_compact_adapter_formulas(
    roles: tuple[str, ...],
    expected_color_flows: int,
) -> None:
    projection = SimpleNamespace(
        external_sources=tuple(
            SimpleNamespace(source_states=(object(), object())) for _ in roles
        )
    )
    labels = tuple(range(1, len(roles) + 1))
    process = SimpleNamespace(
        legs=tuple(SimpleNamespace(color_role=role) for role in roles),
        fundamental_labels=tuple(
            label
            for label, role in zip(labels, roles, strict=True)
            if role == "fundamental"
        ),
        antifundamental_labels=tuple(
            label
            for label, role in zip(labels, roles, strict=True)
            if role == "antifundamental"
        ),
        adjoint_labels=tuple(
            label
            for label, role in zip(labels, roles, strict=True)
            if role == "adjoint"
        ),
    )

    census = service_module._on_the_fly_selector_census_v1(
        projection,  # type: ignore[arg-type]
        process,  # type: ignore[arg-type]
    )
    assert census["physical_helicity_count"] == 2 ** len(roles)
    assert census["physical_color_flow_count"] == expected_color_flows


def test_on_the_fly_writer_rejects_changed_or_noncanonical_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_writer_identity(monkeypatch)
    source_path = _prepared_model_path()
    compiled = load_compiled_model(source_path)
    changed = _process_artifact(tmp_path, runtime_name="changed.pacbin")
    changed.runtime_path.write_bytes(changed.runtime_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="payload changed"):
        write_schema_v3_artifact(
            tmp_path / "changed-artifact",
            mode="error",
            source=ModelSource.from_path(source_path),
            compiled_model=compiled,
            configuration=_configuration(),
            processes=(changed,),
            timings={},
            api_bundle_hook=None,
        )

    extra_path = tmp_path / "extra.pacbin"
    extra_index = write_pacbin_atomic(
        extra_path,
        (
            PacbinMemberSource(
                _SEED_MEMBER,
                PacbinMemberKind.ON_THE_FLY_PROCESS_SEED,
                source=io.BytesIO(b"seed"),
            ),
            PacbinMemberSource(
                "on-the-fly/extra.bin",
                PacbinMemberKind.ON_THE_FLY_PROCESS_SEED,
                source=io.BytesIO(b"extra"),
            ),
        ),
    )
    noncanonical = replace(
        changed,
        runtime_path=extra_path,
        runtime_size_bytes=extra_index.file_size,
        runtime_sha256=hashlib.sha256(extra_path.read_bytes()).hexdigest(),
        runtime_member_count=len(extra_index.members),
        runtime_unpacked_size_bytes=sum(
            member.length for member in extra_index.members
        ),
        runtime_index_sha256=extra_index.index_sha256,
    )
    with pytest.raises(ValueError, match="exactly one canonical process seed"):
        write_schema_v3_artifact(
            tmp_path / "extra-artifact",
            mode="error",
            source=ModelSource.from_path(source_path),
            compiled_model=compiled,
            configuration=_configuration(),
            processes=(noncanonical,),
            timings={},
            api_bundle_hook=None,
        )

    with pytest.raises(ValueError, match="currently supports LC only"):
        write_schema_v3_artifact(
            tmp_path / "nlc-artifact",
            mode="error",
            source=ModelSource.from_path(source_path),
            compiled_model=compiled,
            configuration=_configuration(),
            processes=(replace(noncanonical, color_accuracy="nlc"),),
            timings={},
            api_bundle_hook=None,
        )

    valid = _process_artifact(tmp_path, runtime_name="valid-census.pacbin")
    for index, selector_policy in enumerate(
        (
            {
                "color_coverage": "complete",
                "reference_color_word": None,
                "trace_reflections_folded": False,
            },
            {
                **valid.selector_policy,
                "selector_census": {
                    "physical_helicity_count": 0,
                    "physical_color_flow_count": 1,
                },
            },
        )
    ):
        with pytest.raises(ValueError, match=r"selector census|selector policy"):
            write_schema_v3_artifact(
                tmp_path / f"invalid-census-{index}",
                mode="error",
                source=ModelSource.from_path(source_path),
                compiled_model=compiled,
                configuration=_configuration(),
                processes=(replace(valid, selector_policy=selector_policy),),
                timings={},
                api_bundle_hook=None,
            )


@pytest.mark.parametrize(
    "selection",
    (
        service_module._ProcessSelection(max_color_sectors=1),
        service_module._ProcessSelection(selected_color_sector_ids=frozenset({0})),
        service_module._ProcessSelection(selected_source_helicities={1: -1}),
    ),
)
def test_on_the_fly_construction_rejects_generation_time_selector_trimming(
    tmp_path: Path,
    selection: service_module._ProcessSelection,
) -> None:
    backend = service_module.GenerationBackend(
        _configuration().effective,
        None,
        process_selection=selection,
    )
    process = build_process_ir("d d~ > z")
    expanded = service_module._ExpandedProcess(
        ProcessRequest.parse(process.process, name=_PROCESS_ID),
        process,
    )

    with pytest.raises(
        GenerationError,
        match="retain complete runtime selector coverage",
    ):
        backend._project_on_the_fly_process(
            expanded,
            BuiltinSMModel(),
            None,  # type: ignore[arg-type]
            index=0,
        )


def test_on_the_fly_construction_never_enters_materialized_process_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = service_module.GenerationBackend(_configuration().effective, None)
    source_path = _prepared_model_path()
    resolved = backend._resolve_model(ModelSource.from_path(source_path))
    model = resolved.model
    assert model is not None
    model_inputs = backend._recurrence_model_inputs(resolved)
    process = build_process_ir("d d~ > z")
    expanded = service_module._ExpandedProcess(
        ProcessRequest.parse(process.process, name=_PROCESS_ID),
        process,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("on-the-fly construction materialized a process lane")

    for name in (
        "build_color_plan",
        "compile_generic_dag",
        "_invoke_rust_recurrence_lowering_v2",
        "run_generic_dag_numerical_current_warmup",
        "run_recurrence_numerical_current_warmup",
    ):
        monkeypatch.setattr(service_module, name, forbidden)
    monkeypatch.setattr(backend, "_prepare_process_construction", forbidden)
    projected = backend._project_on_the_fly_process(
        expanded,
        model,
        model_inputs,
        index=0,
    )
    seed_identity = _seed_identity_from_source_projection(
        projected.projection.seed.to_json_dict()
    )
    wrong_seed_identity = _seed_identity_from_source_projection(
        projected.projection.seed.to_json_dict(),
        process_digest="f" * 64,
    )
    assert wrong_seed_identity["process_digest"] != (
        projected.projection.seed.process_digest
    )
    monkeypatch.setattr(
        service_module,
        "_invoke_rust_on_the_fly_seed_inspector_v1",
        lambda _payload: wrong_seed_identity,
    )
    with pytest.raises(GenerationError, match="process digest disagrees"):
        backend._construct_on_the_fly_artifact(
            projected,
            b"native-compact-seed",
            model,
            model_inputs,
            tmp_path,
            phase=PhaseHandle("test", None, 1),
        )
    assert not (tmp_path / "on-the-fly-runtimes").exists()

    monkeypatch.setattr(
        service_module,
        "_invoke_rust_on_the_fly_seed_inspector_v1",
        lambda _payload: seed_identity,
    )
    generated = backend._construct_on_the_fly_artifact(
        projected,
        b"native-compact-seed",
        model,
        model_inputs,
        tmp_path,
        phase=PhaseHandle("test", None, 1),
    )

    assert generated.artifact.process_id == _PROCESS_ID
    assert generated.artifact.runtime_path.parent == (
        tmp_path / "on-the-fly-runtimes" / _PROCESS_ID
    )
    source_projection = projected.projection.seed.to_json_dict()
    assert not {"dag", "color_plan", "recurrence", "direct_plan"}.intersection(
        source_projection
    )
    with PacbinReader.open(generated.artifact.runtime_path) as reader:
        member = reader.members[0]
        assert (member.logical_path, member.kind) == (
            _SEED_MEMBER,
            PacbinMemberKind.ON_THE_FLY_PROCESS_SEED,
        )
        assert reader.read_member(_SEED_MEMBER, length=member.length) == (
            b"native-compact-seed"
        )
    assert set(generated.artifact.runtime_metadata) == {
        "runtime_parameters",
        "prepared_parameter_defaults",
        "parameter_projection",
        "external_legs",
        "particle_masses",
        "normalization",
        "process_seed_identity",
    }


def test_multi_process_on_the_fly_writer_uses_one_ordered_native_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_writer_identity(monkeypatch)
    calls: list[dict[str, object]] = []

    def batch_builder(
        ordered_sources: object,
        recurrence_json: object,
        direct_json: object,
        pack_digest: object,
        *,
        process_identities: object = (),
    ) -> tuple[bytes, ...]:
        sources = tuple(ordered_sources)  # type: ignore[arg-type]
        identities = tuple(process_identities)  # type: ignore[arg-type]
        calls.append(
            {
                "sources": sources,
                "recurrence_json": recurrence_json,
                "direct_json": direct_json,
                "pack_digest": pack_digest,
                "identities": identities,
            }
        )
        return tuple(
            f"native-compact-seed-{index}".encode() for index in range(len(sources))
        )

    monkeypatch.setattr(
        service_module,
        "_invoke_rust_on_the_fly_seed_batch_builder_v1",
        batch_builder,
    )

    def inspect_seed(payload: bytes) -> dict[str, object]:
        index = int(payload.rsplit(b"-", maxsplit=1)[1])
        assert calls
        ordered_sources = calls[0]["sources"]
        assert isinstance(ordered_sources, tuple)
        source_projection = json.loads(ordered_sources[index])
        assert isinstance(source_projection, dict)
        return _seed_identity_from_source_projection(source_projection)

    monkeypatch.setattr(
        service_module,
        "_invoke_rust_on_the_fly_seed_inspector_v1",
        inspect_seed,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("on-the-fly generation materialized a process lane")

    for name in (
        "build_color_plan",
        "compile_generic_dag",
        "_invoke_rust_recurrence_lowering_v2",
        "run_generic_dag_numerical_current_warmup",
        "run_recurrence_numerical_current_warmup",
    ):
        monkeypatch.setattr(service_module, name, forbidden)
    monkeypatch.setattr(
        service_module.GenerationBackend,
        "_prepare_process_construction",
        forbidden,
    )

    process_ids = ("d_dbar_to_z", "u_ubar_to_z")
    processes = ProcessSet.from_expressions(
        ("d d~ > z", "u u~ > z"),
        names=process_ids,
    )
    artifact = tmp_path / "multi-process-artifact"
    Generator(_configuration().effective).generate(
        processes,
        artifact,
        model=ModelSource.from_path(_prepared_model_path()),
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["identities"] == process_ids
    sources = call["sources"]
    assert isinstance(sources, tuple) and len(sources) == 2
    decoded_sources = tuple(json.loads(source) for source in sources)
    assert [source["schema_version"] for source in decoded_sources] == [1, 1]
    assert decoded_sources[0]["process_digest"] != decoded_sources[1]["process_digest"]
    for source in decoded_sources:
        assert not {"dag", "color_plan", "recurrence", "direct_plan"}.intersection(
            source
        )

    for index, process_id in enumerate(process_ids):
        runtime = artifact / "processes" / process_id / "on-the-fly-runtime.pacbin"
        with PacbinReader.open(runtime, verify_payloads=True) as reader:
            member = reader.members[0]
            assert reader.read_member(_SEED_MEMBER, length=member.length) == (
                f"native-compact-seed-{index}".encode()
            )


def test_on_the_fly_runtime_metadata_reuses_recurrence_support_contract() -> None:
    backend = service_module.GenerationBackend(_configuration().effective, None)
    resolved = backend._resolve_model(ModelSource.from_path(_prepared_model_path()))
    model = resolved.model
    assert model is not None
    inputs = backend._recurrence_model_inputs(resolved)
    process = build_process_ir("d d~ > z")
    projection = service_module.project_on_the_fly_process_seed_v1(
        process,
        inputs.catalog,
        model,
        coupling_order_policy="minimal",
        coupling_order_limits={},
    )

    metadata = build_on_the_fly_runtime_metadata(
        projection.seed.external_sources,
        projection.seed.parameter_projection,
        inputs.catalog,
        model,
        projection.runtime_normalization,
    )

    assert metadata["external_legs"] == [
        {
            "source_slot": leg.source_slot,
            "public_label": leg.public_label,
            "physical_pdg": leg.physical_pdg,
            "outgoing_pdg": leg.outgoing_pdg,
            "is_initial": leg.is_initial,
        }
        for leg in projection.seed.external_sources
    ]
    assert metadata["parameter_projection"] == [
        {
            "runtime_slot": row.runtime_slot,
            "runtime_name": row.runtime_name,
            "parameter_template_id": row.parameter_template_id,
            "prepared_parameter_id": row.prepared_parameter_id,
            "component": row.component,
        }
        for row in projection.seed.parameter_projection
    ]
    assert metadata["normalization"] == projection.runtime_normalization
    assert "source_templates" not in metadata
    assert "public_color_flows" not in metadata


def test_on_the_fly_singleton_seed_bridge_wraps_plural_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def binding(*args: object) -> list[bytes]:
        calls.append(args)
        return [b"stable-seed-bytes"]

    module = SimpleNamespace(_build_on_the_fly_process_seeds_v1=binding)
    monkeypatch.setattr(service_module.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(service_module, "verify_native_module", lambda _module: None)

    result = service_module._invoke_rust_on_the_fly_seed_builder_v1(
        b"source",
        b"recurrence",
        b"direct",
        "a" * 64,
    )

    assert result == b"stable-seed-bytes"
    assert calls == [((b"source",), b"recurrence", b"direct", "a" * 64)]


def test_on_the_fly_native_seed_inspector_returns_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _seed_identity()
    module = SimpleNamespace(
        _inspect_on_the_fly_process_seed_v1=lambda payload: (
            json.dumps(identity) if payload == b"seed" else "{}"
        )
    )
    monkeypatch.setattr(service_module.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(service_module, "verify_native_module", lambda _module: None)

    assert service_module._invoke_rust_on_the_fly_seed_inspector_v1(b"seed") == (
        identity
    )


def test_on_the_fly_native_seed_identity_is_bound_to_source_projection() -> None:
    projection_json: dict[str, object] = {
        "process_digest": "a" * 64,
        "external_permutation": [0, 1],
        "external_sources": [
            {
                "source_slot": 0,
                "public_label": 1,
                "is_initial": True,
                "states": [{"state_index": 0, "public_helicity": -1}],
            },
            {
                "source_slot": 1,
                "public_label": 2,
                "is_initial": False,
                "states": [{"state_index": 0, "public_helicity": 1}],
            },
        ],
    }
    projection = SimpleNamespace(
        process_digest="a" * 64,
        external_permutation=(0, 1),
        external_sources=(
            SimpleNamespace(
                source_slot=0,
                public_label=1,
                is_initial=True,
                source_states=(SimpleNamespace(state_index=0, public_helicity=-1),),
            ),
            SimpleNamespace(
                source_slot=1,
                public_label=2,
                is_initial=False,
                source_states=(SimpleNamespace(state_index=0, public_helicity=1),),
            ),
        ),
    )
    identity = _seed_identity_from_source_projection(projection_json)
    service_module._bind_on_the_fly_process_seed_identity_v1(
        identity,
        projection,  # type: ignore[arg-type]
    )

    wrong_digest = {**identity, "process_digest": "f" * 64}
    wrong_permutation = {**identity, "external_permutation": [1, 0]}
    candidates: list[tuple[dict[str, object], str]] = [
        (wrong_digest, "process digest disagrees"),
        (wrong_permutation, "external permutation disagrees"),
    ]
    for field, replacement in (
        ("source_slot", 7),
        ("public_label", 7),
        ("is_initial", False),
    ):
        candidate = copy.deepcopy(identity)
        sources = candidate["external_sources"]
        assert isinstance(sources, list) and isinstance(sources[0], dict)
        sources[0][field] = replacement
        candidates.append((candidate, "external source domain disagrees"))
    for field, replacement in (
        ("state_index", 7),
        ("public_helicity", 7),
    ):
        candidate = copy.deepcopy(identity)
        sources = candidate["external_sources"]
        assert isinstance(sources, list) and isinstance(sources[0], dict)
        states = sources[0]["states"]
        assert isinstance(states, list) and isinstance(states[0], dict)
        states[0][field] = replacement
        candidates.append((candidate, "external source domain disagrees"))

    for candidate, message in candidates:
        with pytest.raises(GenerationError, match=message):
            service_module._bind_on_the_fly_process_seed_identity_v1(
                candidate,
                projection,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    ("binding", "message"),
    (
        (None, "does not provide the private"),
        (lambda _payload: b"not-json-text", "inspection failed"),
        (lambda _payload: "not-json", "inspection failed"),
        (
            lambda _payload: json.dumps(
                {**_seed_identity(), "abi": "stale-seed-identity-v0"}
            ),
            "unsupported ABI",
        ),
        (
            lambda _payload: json.dumps({**_seed_identity(), "invented": True}),
            "wrong fields",
        ),
    ),
)
def test_on_the_fly_native_seed_inspector_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    binding: object,
    message: str,
) -> None:
    module = SimpleNamespace()
    if binding is not None:
        module._inspect_on_the_fly_process_seed_v1 = binding
    monkeypatch.setattr(service_module.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(service_module, "verify_native_module", lambda _module: None)

    with pytest.raises(GenerationError, match=message):
        service_module._invoke_rust_on_the_fly_seed_inspector_v1(b"seed")


@pytest.mark.parametrize(
    ("binding", "message"),
    (
        (None, "does not provide the private"),
        (lambda *_args: [], "wrong process count"),
        (lambda *_args: [b""], "empty payload for process index 0"),
        (lambda *_args: ["not-bytes"], "empty payload for process index 0"),
        (
            lambda *_args: (_ for _ in ()).throw(RuntimeError("native failure")),
            "process-seed construction failed: native failure",
        ),
    ),
)
def test_on_the_fly_native_seed_bridge_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    binding: object,
    message: str,
) -> None:
    module = SimpleNamespace()
    if binding is not None:
        module._build_on_the_fly_process_seeds_v1 = binding
    monkeypatch.setattr(service_module.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(service_module, "verify_native_module", lambda _module: None)

    with pytest.raises(GenerationError, match=message):
        service_module._invoke_rust_on_the_fly_seed_builder_v1(
            b"{}",
            b"{}",
            b"{}",
            "a" * 64,
        )


def test_on_the_fly_native_seed_bridge_preserves_indexed_process_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def binding(*_args: object) -> object:
        raise RuntimeError(
            "on-the-fly process seed at index 1 "
            '(process digest "abc") failed: stale source'
        )

    module = SimpleNamespace(_build_on_the_fly_process_seeds_v1=binding)
    monkeypatch.setattr(service_module.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(service_module, "verify_native_module", lambda _module: None)

    with pytest.raises(
        GenerationError,
        match=r"process index 1 'u_ubar_to_z'.*process digest.*stale source",
    ):
        service_module._invoke_rust_on_the_fly_seed_batch_builder_v1(
            (b"{}", b"{}"),
            b"{}",
            b"{}",
            "a" * 64,
            process_identities=("d_dbar_to_z", "u_ubar_to_z"),
        )
