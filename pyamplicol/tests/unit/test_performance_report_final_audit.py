# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.performance_report.final_audit as final_audit_module
import tools.performance_report.runner as report_runner
from tools.performance_report.agreements import (
    DIRECT_AGREEMENT_FIELD,
    INDEPENDENT_AUTHORITY_ABI,
    INDEPENDENT_AUTHORITY_FIELD,
    LC_COMMON_COMPONENT_ABI,
    LC_COMMON_COMPONENT_FIELD,
    LC_LEGACY_PYAMPLICOL_COMPONENT,
    Z_RECURRENCE_CROSS_MODE,
    agreement_edges,
    attach_direct_agreements,
    independent_numerical_authorities,
)
from tools.performance_report.arena_profile import (
    ARENA_PHASE_TIMING_SCOPE,
    ARENA_PROFILE_BOUNDARY,
    EMPTY_ARENA_PHASE_VECTOR_FIELDS,
    ZERO_ARENA_COUNTER_FIELDS,
    ZERO_ARENA_PHASE_TIME_FIELDS,
    ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS,
    build_arena_profile_evidence,
    digest_arena_profile_value,
)
from tools.performance_report.cache import (
    build_reset_caches,
    digest_json,
    empty_measurement,
    reset_entry,
)
from tools.performance_report.campaign_policy import PolicyMeasurementState
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.final_audit import (
    ArtifactEvidence,
    FinalAuditError,
    _active_runtime_snapshot,
    _artifact_reference,
    _audit_compiled_execution,
    _audit_direct_agreements,
    _audit_eager_execution,
    _audit_effective_config_mapping,
    _audit_measurement,
    _audit_model_source,
    _audit_on_the_fly_execution,
    _audit_pdf,
    _audit_pointwise,
    _audit_recurrence_execution,
    _audit_recurrence_source_pack,
    _audit_replayed_direct_agreements,
    _audit_runtime_identity,
    _audit_runtime_profile_warmup_evidence,
    _audit_tex_table_reachability,
    _audit_unavailable_execution_timing,
    _authenticated_effective_config,
    _catalog_static_na_projection_errors,
    _count_contract_matches_with_optional_categories,
    _ensure_exact_cli_python,
    _legacy_consumer_coverage,
    _python_package_tree_identity,
    _real_nonnegative,
    _replay_count_contract_matches_direct_edges,
    _runtime_for_measurement_source,
    _runtime_namespace_paths,
    _shared_artifact_contract,
    _validation_baseline_for_audit,
    audit_artifact,
    audit_final_report,
)
from tools.performance_report.measurement_lineage import (
    CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
)
from tools.performance_report.models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    MeasurementSpec,
    ModelKey,
    Workload,
)
from tools.performance_report.publication import portable_publication_value
from tools.performance_report.render import VisibleCompleteness
from tools.performance_report.service import ReportPaths, ReportService

_REVISION = "a" * 40
_LEGACY_REVISION = "dcb3efbcbe89c51969818cddf48113fba19ad763"
_ARTIFACT_ID = "b" * 64
_CAPABILITY = "rusticol.recurrence-direct-arena.complex-f64.v1"
_COLOR_CAPABILITY = "rusticol.recurrence-color.lc.v1"


def test_full_final_audit_distinguishes_declared_and_measurable_edge_totals() -> None:
    declared_ids = {cell.cell_id for cell in REPORT_CATALOG.measurement_cells()}
    measurable_ids = {
        cell.cell_id
        for cell in REPORT_CATALOG.measurement_cells()
        if REPORT_CATALOG.static_na_reason(cell) is None
    }
    declared_edges = tuple(
        edge
        for edge in agreement_edges()
        if edge.candidate.cell_id in declared_ids
        and edge.baseline.cell_id in declared_ids
    )
    measurable_edges = tuple(
        edge
        for edge in declared_edges
        if edge.candidate.cell_id in measurable_ids
        and edge.baseline.cell_id in measurable_ids
    )

    assert len(declared_edges) == 1737
    assert (
        len(declared_edges) == final_audit_module._EXPECTED_FULL_DIRECT_AGREEMENT_COUNT
    )
    assert len(measurable_edges) == 1689
    assert sum(
        final_audit_module._EXPECTED_FULL_DIRECT_AGREEMENT_COUNTS.values()
    ) == len(measurable_edges)
    assert Counter(edge.kind for edge in measurable_edges) == (
        final_audit_module._EXPECTED_FULL_DIRECT_AGREEMENT_COUNTS
    )
    static_reasons = Counter(
        REPORT_CATALOG.static_na_reason(cell)
        for cell in REPORT_CATALOG.measurement_cells()
        if REPORT_CATALOG.static_na_reason(cell) is not None
    )
    assert static_reasons == {
        "on-the-fly-color-accuracy-not-supported-v1": 100,
        "native-backend-generation-cap-n6-v1": 24,
        "original-amplicol-open-quark-line-limit": 10,
    }

    n4_cells = tuple(
        cell for cell in REPORT_CATALOG.measurement_cells() if cell.n_final <= 4
    )
    n4_ids = {cell.cell_id for cell in n4_cells}
    n4_measurable_ids = {
        cell.cell_id
        for cell in n4_cells
        if REPORT_CATALOG.static_na_reason(cell) is None
    }
    n4_edges = tuple(
        edge
        for edge in agreement_edges()
        if edge.candidate.cell_id in n4_measurable_ids
        and edge.baseline.cell_id in n4_measurable_ids
    )
    assert len(n4_ids) == final_audit_module._EXPECTED_N4_CELL_COUNT == 894
    assert Counter(edge.kind for edge in n4_edges) == (
        final_audit_module._EXPECTED_N4_DIRECT_AGREEMENT_COUNTS
    )
    _empty_edges, empty_counts = _audit_direct_agreements(
        (),
        {},
        catalog=REPORT_CATALOG,
    )
    assert set(empty_counts) == set(
        final_audit_module._EXPECTED_FULL_DIRECT_AGREEMENT_COUNTS
    )


@pytest.mark.parametrize(
    ("expected_edges", "expected_replays"),
    (
        (
            final_audit_module._EXPECTED_FULL_DIRECT_AGREEMENT_COUNTS,
            final_audit_module._EXPECTED_FULL_DIRECT_REPLAY_COUNTS,
        ),
        (
            final_audit_module._EXPECTED_N4_DIRECT_AGREEMENT_COUNTS,
            final_audit_module._EXPECTED_N4_DIRECT_REPLAY_COUNTS,
        ),
    ),
)
def test_final_audit_count_contract_bounds_only_optional_authority_edges(
    expected_edges: Mapping[str, int],
    expected_replays: Mapping[str, int],
) -> None:
    without_optional = dict(expected_edges)
    without_optional[Z_RECURRENCE_CROSS_MODE] = 0
    without_optional[LC_LEGACY_PYAMPLICOL_COMPONENT] = 0
    assert _count_contract_matches_with_optional_categories(
        without_optional,
        expected_edges,
        optional_categories={
            Z_RECURRENCE_CROSS_MODE,
            LC_LEGACY_PYAMPLICOL_COMPONENT,
        },
    )
    expected_without_optional_replays = dict(expected_replays)
    expected_without_optional_replays[
        final_audit_module._FULLY_REPLAYED_PYAMPLICOL
    ] -= expected_edges[Z_RECURRENCE_CROSS_MODE]
    expected_without_optional_replays[
        final_audit_module._REPLAYED_PYAMPLICOL_AUTHENTICATED_LEGACY
    ] = 0
    assert _replay_count_contract_matches_direct_edges(
        expected_without_optional_replays,
        without_optional,
        expected_replays,
    )

    optional_categories = {
        Z_RECURRENCE_CROSS_MODE,
        LC_LEGACY_PYAMPLICOL_COMPONENT,
    }
    for required_category in set(expected_edges) - optional_categories:
        missing_required = dict(expected_edges)
        missing_required[required_category] -= 1
        assert not _count_contract_matches_with_optional_categories(
            missing_required,
            expected_edges,
            optional_categories=optional_categories,
        )

    for optional_category in optional_categories:
        too_many_optional = dict(expected_edges)
        too_many_optional[optional_category] += 1
        assert not _count_contract_matches_with_optional_categories(
            too_many_optional,
            expected_edges,
            optional_categories=optional_categories,
        )

    missing_required_replay = dict(expected_without_optional_replays)
    missing_required_replay[final_audit_module._FULLY_REPLAYED_PYAMPLICOL] -= 1
    assert not _replay_count_contract_matches_direct_edges(
        missing_required_replay,
        without_optional,
        expected_replays,
    )

    missing_available_z_replay = dict(expected_replays)
    missing_available_z_replay[final_audit_module._FULLY_REPLAYED_PYAMPLICOL] -= 1
    assert not _replay_count_contract_matches_direct_edges(
        missing_available_z_replay,
        expected_edges,
        expected_replays,
    )

    missing_available_legacy_replay = dict(expected_replays)
    missing_available_legacy_replay[
        final_audit_module._REPLAYED_PYAMPLICOL_AUTHENTICATED_LEGACY
    ] -= 1
    assert not _replay_count_contract_matches_direct_edges(
        missing_available_legacy_replay,
        expected_edges,
        expected_replays,
    )


def _fake_latexmk(tmp_path: Path, log: str) -> Path:
    executable = tmp_path / "fake-latexmk"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "assert not Path('campaign_artifacts').exists()\n"
        "Path('pyAmpliCol.pdf').write_bytes(b'%PDF-1.4\\n%%EOF\\n')\n"
        f"Path('pyAmpliCol.log').write_text({log!r}, encoding='utf-8')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _cell(
    mode: ExecutionMode,
    *,
    backend: str = "jit",
    optimization_level: int | None = 3,
) -> CellSpec:
    return CellSpec(
        dataset_id=f"test_{mode.value}",
        process="d d~ > z",
        n_final=1,
        process_key="dd_z",
        measurement=MeasurementSpec(
            mode,
            ModelKey.BUILTIN_SM,
            Accuracy.LC,
            backend,
            optimization_level,
        ),
        workload=Workload.SELECTED_FLOW,
    )


def test_final_pdf_audit_rejects_successful_latex_with_overfull_box(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    service = ReportService(ReportPaths.from_repo(repo))
    service.paths.docs_dir.mkdir(parents=True, exist_ok=True)
    (service.paths.docs_dir / "campaign_artifacts").mkdir()
    (service.paths.docs_dir / "campaign_artifacts/sentinel").write_text(
        "private\n",
        encoding="ascii",
    )
    (service.paths.docs_dir / "pyAmpliCol.tex").write_text(
        (
            "\\documentclass{article}\\begin{document}"
            "\\input{result_test_table.tex}\\end{document}\n"
        ),
        encoding="ascii",
    )
    (service.paths.docs_dir / "result_test_table.tex").write_text(
        "bad\n",
        encoding="ascii",
    )
    (service.paths.docs_dir / "pyAmpliCol.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    executable = _fake_latexmk(
        tmp_path,
        "Overfull \\vbox (1.0pt too high)\n",
    )
    monkeypatch.setattr(
        "tools.performance_report.final_audit.shutil.which",
        lambda _name: str(executable),
    )
    monkeypatch.setattr(ReportService, "load_caches", lambda _self: {})
    monkeypatch.setattr(
        "tools.performance_report.final_audit.render_all_tables",
        lambda *_args, **_kwargs: {"result_test_table.tex": "bad\n"},
    )

    with pytest.raises(FinalAuditError, match="overfull"):
        _audit_pdf(service)


def _recurrence_binding_digest(binding: dict[str, object]) -> str:
    if binding["kind"] != "rusticol-intrinsic":
        payload = dict(binding)
        payload.pop("payload_digest", None)
        return digest_json(payload)
    fields = (
        "abi",
        "contribution_parent_permutation",
        "kind",
        "runtime_template",
    )
    return digest_json({field: binding.get(field) for field in fields})


def _refresh_recurrence_source_catalog(
    execution: dict[str, object],
    pack: dict[str, object],
    *,
    refresh_binding_digest: bool = True,
) -> None:
    catalog = pack["recurrence_direct_template"]
    assert isinstance(catalog, dict)
    templates = catalog["templates"]
    assert isinstance(templates, list)
    for template in templates:
        assert isinstance(template, dict)
        binding = template["payload_binding"]
        assert isinstance(binding, dict)
        if refresh_binding_digest:
            binding["payload_digest"] = _recurrence_binding_digest(binding)
        semantic = dict(template)
        semantic.pop("semantic_digest", None)
        template["semantic_digest"] = digest_json(semantic)
    semantic_catalog = dict(catalog)
    semantic_catalog.pop("catalog_digest", None)
    catalog["catalog_digest"] = digest_json(semantic_catalog)
    execution["direct_template_catalog_digest"] = catalog["catalog_digest"]
    plan = execution["plan"]
    assert isinstance(plan, dict)
    plan["direct_template_catalog_digest"] = catalog["catalog_digest"]


def _canonical_symjit_evaluator(
    application_path: str,
    plane_path: str,
    *,
    optimization_level: int,
    input_len: int = 1,
    output_len: int = 1,
) -> dict[str, object]:
    return {
        "kind": "symjit-application-evaluator",
        "runtime_capability": "symjit.application.complex-f64.v1",
        "input_len": input_len,
        "output_len": output_len,
        "application_abi": "symjit-application-storage-v3",
        "application_path": application_path,
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": optimization_level,
        "word_bits": 64,
        "endianness": "little",
        "required_defuns": [],
        "evaluator_state_path": f"{application_path}.state",
        "evaluator_state_runtime_capability": (
            "symbolica.legacy-jit-container.complex-f64.v1"
        ),
        "plane_application": {
            "application_path": plane_path,
            "application_abi": "pyamplicol-symjit-plane-application-v2",
            "storage_abi": "symjit-application-storage-v3",
            "element_layout": "split-complex-plane-major",
            "descriptor_order": "inputs-re-im-then-outputs-re-im",
            "input_complex_count": input_len,
            "output_complex_count": output_len,
            "input_plane_count": 2 * input_len,
            "output_plane_count": 2 * output_len,
            "compiler_type": "native",
            "translation_mode": "symbolica-structured-instructions",
            "optimization_level": optimization_level,
            "simd": True,
            "complex": True,
            "fast_math": True,
            "fast_complex": False,
            "compression": False,
            "threading": False,
            "direct_arena": True,
            "source_digest": "1" * 64,
            "target": {"word_bits": 64, "endianness": "little"},
        },
    }


def _recurrence_source_fixture(
    source_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    ordinary_source_path = "kernels/000000/application-0.symjit"
    source_path = "kernels/000000/application-0.plane.symjit"
    binding: dict[str, object] = {
        "abi": "pyamplicol-recurrence-plane-binding-v2",
        "contribution_parent_permutation": [0, 1],
        "destination_operation": "finalize-in-place",
        "direct_application_abi": "pyamplicol-symjit-plane-application-v2",
        "exact_factor_scalar_slots": [0, 1],
        "input_plane_count": 1,
        "input_plane_projections": [{"kind": "destination-current"}],
        "intrinsic_contract_digest": None,
        "kind": "prepared-direct-call",
        "output_alias_inputs": [0],
        "parameter_bindings": [{"index": 0, "kind": "plane"}],
        "payload_digest": "",
        "payload_paths": [source_path],
        "prepared_kernel_id": 0,
        "prepared_template_semantic_digest": "d" * 64,
        "role": "finalization",
        "runtime_template": None,
        "scalar_input_count": 2,
        "scalar_projections": [
            {"imaginary": False, "kind": "exact-factor"},
            {"imaginary": True, "kind": "exact-factor"},
        ],
        "source_application_abi": "pyamplicol-symjit-plane-application-v2",
        "source_application_path": source_path,
        "source_application_sha256": source_sha256,
        "state_plane_indices": [],
    }
    template: dict[str, object] = {
        "abi": "pyamplicol-recurrence-direct-template-v1",
        "alignment_bytes": 64,
        "backend": "jit",
        "coupling_slot_count": 0,
        "destination_aliasing": True,
        "destination_component_count": 1,
        "destination_operation": "finalize-in-place",
        "direct_executor_id": 0,
        "evaluator_binding_id": 0,
        "evaluator_resolver_key": "prepared-kernel-0",
        "exact_expression_digest": "e" * 64,
        "momentum_operand_count": 0,
        "optimization_level": 2,
        "parameter_slot_count": 0,
        "parent_arity": 1,
        "parent_component_counts": [1],
        "payload_binding": binding,
        "portable": True,
        "role": "finalization",
        "semantic_digest": "",
        "semantic_template_ids": ["prepared-finalization"],
        "simd_axis": "points-contiguous",
        "target_triple": "symjit-storage-v3-portable",
        "template_id": "template-0",
    }
    prepared_digest = "a" * 64
    catalog: dict[str, object] = {
        "abi": "pyamplicol-recurrence-direct-template-v1",
        "backend": "jit",
        "backend_abi": "rusticol.recurrence-direct-backend.v1",
        "canonicalization_abi": "pyamplicol-canonical-json-v1",
        "catalog_digest": "",
        "compiled_model_digest": "b" * 64,
        "optimization_level": 2,
        "optimization_settings_digest": "c" * 64,
        "portable": True,
        "prepared_kernel_contract_digest": "4" * 64,
        "prepared_kernel_pack_digest": prepared_digest,
        "prepared_kernel_payload_digest": "5" * 64,
        "recurrence_template_catalog_digest": "6" * 64,
        "target_triple": "symjit-storage-v3-portable",
        "templates": [template],
    }
    plan: dict[str, object] = {
        "builder_input_abi": "pyamplicol-recurrence-builder-input-v2",
        "recurrence_plan_abi": "pyamplicol-recurrence-plan-v2",
        "runtime_layout_abi": "pyamplicol-recurrence-runtime-layout-v2",
        "direct_template_abi": "pyamplicol-recurrence-direct-template-v1",
        "direct_backend_abi": "rusticol.recurrence-direct-backend.v1",
        "prepared_kernel_pack_digest": prepared_digest,
        "direct_template_catalog_digest": "",
        "inspection_summary": {
            "direct_arena": {
                "packed_input_bytes": 0,
                "packed_output_bytes": 0,
                "scatter_bytes": 0,
                "row_group_count": 4,
            }
        },
    }
    execution: dict[str, object] = {
        "kind": "pyamplicol-runtime-recurrence-execution",
        "builder_input_abi": "pyamplicol-recurrence-builder-input-v2",
        "recurrence_plan_abi": "pyamplicol-recurrence-plan-v2",
        "runtime_layout_abi": "pyamplicol-recurrence-runtime-layout-v2",
        "direct_template_abi": "pyamplicol-recurrence-direct-template-v1",
        "direct_backend_abi": "rusticol.recurrence-direct-backend.v1",
        "prepared_kernel_pack_digest": prepared_digest,
        "direct_template_catalog_digest": "",
        "kernel_pack": {
            "manifest_path": "model/eager-kernel-pack.json",
            "payload_root": "model/eager-kernels",
        },
        "required_runtime_capabilities": [_COLOR_CAPABILITY, _CAPABILITY],
        "plan": plan,
    }
    pack: dict[str, object] = {
        "backend": "jit",
        "optimization_settings": {
            "backend": "jit",
            "jit_optimization_level": 2,
        },
        "recurrence_direct_template": catalog,
        "kernel_variants": [],
        "kernels": [
            {
                "kernel_id": 0,
                "f64_evaluator_manifest": _canonical_symjit_evaluator(
                    ordinary_source_path,
                    source_path,
                    optimization_level=2,
                ),
            }
        ],
    }
    _refresh_recurrence_source_catalog(execution, pack)
    return execution, pack


def _legacy_cell() -> CellSpec:
    return CellSpec(
        dataset_id="reference_amplicol_lc",
        process="d d~ > z",
        n_final=1,
        process_key="dd_z",
        measurement=MeasurementSpec(
            ExecutionMode.AMPLICOL,
            None,
            Accuracy.LC,
            "fortran",
            None,
        ),
        workload=Workload.SELECTED_FLOW,
    )


def test_catalog_static_na_projection_requires_reset_cache_and_no_current() -> None:
    cell = replace(
        _legacy_cell(),
        dataset_id="reference_amplicol_static_na",
        process="d d~ > u u~ s s~ c c~",
        n_final=6,
        process_key="dd_4q_lines",
    )
    canonical = reset_entry(cell)

    assert (
        _catalog_static_na_projection_errors(
            (cell,),
            {cell.cell_id: canonical},
            load_current=lambda _cell_id: None,
        )
        == ()
    )

    tampered = deepcopy(canonical)
    tampered["measurement"] = {
        **empty_measurement(),
        "status": "unsupported",
    }
    assert _catalog_static_na_projection_errors(
        (cell,),
        {cell.cell_id: tampered},
        load_current=lambda _cell_id: None,
    ) == (
        f"{cell.cell_id}: catalog static N/A cache entry differs "
        "from the canonical reset entry",
    )
    assert _catalog_static_na_projection_errors(
        (cell,),
        {},
        load_current=lambda _cell_id: None,
    ) == (f"{cell.cell_id}: cache entry is missing",)
    assert _catalog_static_na_projection_errors(
        (cell,),
        {cell.cell_id: canonical},
        load_current=lambda _cell_id: object(),
    ) == (f"{cell.cell_id}: catalog static N/A cell has a published current",)


def _compiled_stage(
    *,
    optimization_level: int = 3,
    backend: str = "jit",
) -> dict[str, object]:
    source_application_abi = (
        "pyamplicol-symjit-plane-application-v2"
        if backend == "jit"
        else "pyamplicol-native-compiled-direct-application-v1"
    )
    application_abi = (
        "pyamplicol-compiled-plane-kernel-v2"
        if backend == "jit"
        else "pyamplicol-native-compiled-direct-application-v1"
    )
    evaluator = (
        {
            "kind": "symjit-application-evaluator",
            "runtime_capability": "symjit.application.complex-f64.v1",
            "input_len": 1,
            "output_len": 1,
            "application_abi": "symjit-application-storage-v3",
            "application_path": "evaluators/stage.symjit",
            "element_layout": "complex-f64",
            "batch_layout": "row-major",
            "compiler_type": "native",
            "translation_mode": "indirect",
            "optimization_level": optimization_level,
            "word_bits": 64,
            "endianness": "little",
            "required_defuns": [],
            "evaluator_state_path": "evaluators/stage.symjit.state",
            "evaluator_state_runtime_capability": (
                "symbolica.legacy-jit-container.complex-f64.v1"
            ),
            "plane_application": {
                "application_path": "evaluators/stage.plane.symjit",
                "application_abi": "pyamplicol-symjit-plane-application-v2",
                "storage_abi": "symjit-application-storage-v3",
                "element_layout": "split-complex-plane-major",
                "descriptor_order": "inputs-re-im-then-outputs-re-im",
                "input_complex_count": 1,
                "output_complex_count": 1,
                "input_plane_count": 2,
                "output_plane_count": 2,
                "compiler_type": "native",
                "translation_mode": "symbolica-structured-instructions",
                "optimization_level": optimization_level,
                "simd": True,
                "complex": True,
                "fast_math": True,
                "fast_complex": False,
                "compression": False,
                "threading": False,
                "direct_arena": True,
                "source_digest": "1" * 64,
                "target": {"word_bits": 64, "endianness": "little"},
            },
        }
        if backend == "jit"
        else {
            "kind": "compiled-complex-evaluator",
            "runtime_capability": (f"symbolica.compiled-{backend}.complex-f64.v1"),
        }
    )
    return {
        "parameter_count": 1,
        "output_length": 1,
        "evaluator": evaluator,
        "compiled_plane_arena": {
            "schema_version": 1,
            "kind": "compiled-plane-arena-stage",
            "application_abi": application_abi,
            "source_application_abi": source_application_abi,
            "element_layout": "split-complex-component-major",
            "output_operation": "overwrite",
            "output_factor": "identity",
            "input_output_aliasing": "forbidden",
            "output_output_aliasing": "forbidden",
            "input_bindings": [{"parameter_index": 0}],
            "output_bindings": [{"output_index": 0}],
            "leaves": [
                {
                    "application_path": (
                        "evaluators/stage.plane.symjit"
                        if backend == "jit"
                        else "evaluators/stage.direct"
                    ),
                    "source_application_abi": source_application_abi,
                    "optimization_level": optimization_level,
                    "direct_codegen_optimization_level": optimization_level,
                }
            ],
        },
    }


def test_compiled_arena_audits_fused_stages_and_parameter_exclusion() -> None:
    execution = {
        "kind": "pyamplicol-runtime-execution",
        "compiled": {
            "stage_evaluators": {
                "stages": [_compiled_stage()],
                "amplitude_stage": _compiled_stage(),
            },
            "model_parameter_evaluator": {
                "evaluator": {"application_abi": "symjit-application-storage-v3"}
            },
        },
    }
    assert _audit_compiled_execution(
        execution,
        _cell(ExecutionMode.COMPILED),
    ) == (2, 2)
    jit_o1 = deepcopy(execution)
    for stage in (
        *jit_o1["compiled"]["stage_evaluators"]["stages"],  # type: ignore[index]
        jit_o1["compiled"]["stage_evaluators"]["amplitude_stage"],  # type: ignore[index]
    ):
        stage["compiled_plane_arena"]["leaves"][0]["optimization_level"] = 1
        stage["compiled_plane_arena"]["leaves"][0][
            "direct_codegen_optimization_level"
        ] = 1
        stage["evaluator"]["optimization_level"] = 1
        stage["evaluator"]["plane_application"]["optimization_level"] = 1
    assert _audit_compiled_execution(
        jit_o1,
        _cell(ExecutionMode.COMPILED, optimization_level=1),
    ) == (2, 2)

    invalid_direct_codegen = deepcopy(execution)
    invalid_direct_codegen["compiled"]["stage_evaluators"]["stages"][0][  # type: ignore[index]
        "compiled_plane_arena"
    ]["leaves"][0]["direct_codegen_optimization_level"] = 2
    with pytest.raises(
        FinalAuditError,
        match=r"leaf\.direct_codegen_optimization_level",
    ):
        _audit_compiled_execution(
            invalid_direct_codegen,
            _cell(ExecutionMode.COMPILED),
        )

    execution["compiled"]["model_parameter_evaluator"] = {  # type: ignore[index]
        "compiled_plane_arena": {}
    }
    with pytest.raises(FinalAuditError, match="model-parameter"):
        _audit_compiled_execution(execution, _cell(ExecutionMode.COMPILED))

    execution["compiled"]["model_parameter_evaluator"] = {  # type: ignore[index]
        "nested": {"native_direct_application": {}}
    }
    with pytest.raises(FinalAuditError, match="DirectApplication metadata"):
        _audit_compiled_execution(execution, _cell(ExecutionMode.COMPILED))

    execution["compiled"]["model_parameter_evaluator"] = {  # type: ignore[index]
        "nested": {
            "application_abi": "pyamplicol-native-compiled-direct-application-v1"
        }
    }
    with pytest.raises(FinalAuditError, match="direct application ABI"):
        _audit_compiled_execution(execution, _cell(ExecutionMode.COMPILED))


@pytest.mark.parametrize(
    ("backend", "capability"),
    [
        ("jit", "symjit.application.complex-f64.v1"),
        ("cpp", "symbolica.compiled-cpp.complex-f64.v1"),
        ("asm", "symbolica.compiled-asm.complex-f64.v1"),
    ],
)
def test_compiled_arena_authenticates_source_runtime_capability(
    backend: str,
    capability: str,
) -> None:
    stage = _compiled_stage(backend=backend)
    execution = {
        "kind": "pyamplicol-runtime-execution",
        "compiled": {
            "stage_evaluators": {
                "stages": [],
                "amplitude_stage": stage,
            },
        },
    }
    cell = _cell(
        ExecutionMode.COMPILED,
        backend=backend,
        optimization_level=3 if backend == "jit" else None,
    )
    assert _audit_compiled_execution(execution, cell) == (1, 1)
    assert stage["evaluator"]["runtime_capability"] == capability  # type: ignore[index]

    stage["evaluator"]["runtime_capability"] = "wrong.capability"  # type: ignore[index]
    with pytest.raises(FinalAuditError, match="runtime_capability"):
        _audit_compiled_execution(execution, cell)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("backend", "native"),
        ("backend", None),
        ("compiler_type", "interpreter"),
    ),
)
def test_compiled_arena_rejects_contradictory_symjit_identity(
    field: str,
    value: object,
) -> None:
    stage = _compiled_stage()
    stage["evaluator"][field] = value  # type: ignore[index]
    execution = {
        "kind": "pyamplicol-runtime-execution",
        "compiled": {
            "stage_evaluators": {
                "stages": [],
                "amplitude_stage": stage,
            },
        },
    }

    with pytest.raises(FinalAuditError, match=field):
        _audit_compiled_execution(execution, _cell(ExecutionMode.COMPILED))


def test_compiled_arena_accepts_fallback_free_portable_symjit_leaf() -> None:
    stage = _compiled_stage()
    evaluator = stage["evaluator"]
    assert isinstance(evaluator, dict)
    evaluator["evaluator_state_path"] = None
    evaluator["evaluator_state_runtime_capability"] = None
    execution = {
        "kind": "pyamplicol-runtime-execution",
        "compiled": {
            "stage_evaluators": {
                "stages": [],
                "amplitude_stage": stage,
            },
        },
    }

    assert _audit_compiled_execution(
        execution,
        _cell(ExecutionMode.COMPILED),
    ) == (1, 1)


@pytest.mark.parametrize(
    ("state_path", "capability"),
    (
        (None, "symbolica.legacy-jit-container.complex-f64.v1"),
        ("evaluators/stage.symjit.state", None),
    ),
)
def test_compiled_arena_rejects_partial_symjit_fallback_pair(
    state_path: str | None,
    capability: str | None,
) -> None:
    stage = _compiled_stage()
    evaluator = stage["evaluator"]
    assert isinstance(evaluator, dict)
    evaluator["evaluator_state_path"] = state_path
    evaluator["evaluator_state_runtime_capability"] = capability
    execution = {
        "kind": "pyamplicol-runtime-execution",
        "compiled": {
            "stage_evaluators": {
                "stages": [],
                "amplitude_stage": stage,
            },
        },
    }

    with pytest.raises(FinalAuditError, match="evaluator_state_fallback"):
        _audit_compiled_execution(execution, _cell(ExecutionMode.COMPILED))


@pytest.mark.parametrize(
    "target",
    (
        {"word_bits": 64.0, "endianness": "little"},
        {
            "word_bits": 64,
            "endianness": "little",
            "cpu_features": ["z", "a"],
        },
        {"word_bits": 64, "endianness": "little", "unknown": True},
    ),
)
def test_compiled_arena_rejects_noncanonical_symjit_target(
    target: dict[str, object],
) -> None:
    stage = _compiled_stage()
    evaluator = stage["evaluator"]
    assert isinstance(evaluator, dict)
    plane = evaluator["plane_application"]
    assert isinstance(plane, dict)
    plane["target"] = target
    execution = {
        "kind": "pyamplicol-runtime-execution",
        "compiled": {
            "stage_evaluators": {
                "stages": [],
                "amplitude_stage": stage,
            },
        },
    }

    with pytest.raises(FinalAuditError, match=r"plane_application\.target"):
        _audit_compiled_execution(execution, _cell(ExecutionMode.COMPILED))


def test_final_replay_matrix_conversion_rejects_sign_and_complex_drift() -> None:
    assert _real_nonnegative(2.0 + 0.0j, "matrix") == 2.0
    assert _real_nonnegative(-1.0e-16 + 0.0j, "matrix") == 0.0
    with pytest.raises(FinalAuditError, match="materially negative"):
        _real_nonnegative(-1.0e-3 + 0.0j, "matrix")
    with pytest.raises(FinalAuditError, match="imaginary part"):
        _real_nonnegative(1.0 + 1.0e-3j, "matrix")


def test_shared_artifact_contract_binds_the_exact_physics_cell() -> None:
    cell = _cell(ExecutionMode.RECURRENCE, optimization_level=2)
    dataset_alias = replace(cell, dataset_id="another_dataset")
    different_process = replace(
        cell,
        process="u u~ > z",
        process_key="uu_z",
    )

    assert _shared_artifact_contract(dataset_alias) == _shared_artifact_contract(cell)
    assert _shared_artifact_contract(different_process) != _shared_artifact_contract(
        cell
    )

    on_the_fly = _cell(ExecutionMode.ON_THE_FLY, optimization_level=2)
    all_flow = replace(on_the_fly, workload=Workload.ALL_FLOW)
    assert _shared_artifact_contract(on_the_fly) == _shared_artifact_contract(all_flow)


def test_native_otf_seed_inspector_wraps_native_decode_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeDecodeError(Exception):
        pass

    module = SimpleNamespace(
        _inspect_on_the_fly_process_seed_v1=lambda _payload: (_ for _ in ()).throw(
            NativeDecodeError("malformed native seed")
        )
    )
    monkeypatch.setattr(
        final_audit_module.importlib,
        "import_module",
        lambda _name: module,
    )

    with pytest.raises(FinalAuditError, match="cannot be decoded") as error:
        final_audit_module._native_on_the_fly_process_seed_identity(b"bad-seed")
    assert isinstance(error.value.__cause__, NativeDecodeError)


def test_otf_artifact_audit_accepts_only_compact_seed_and_prepared_jit_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util
    from types import ModuleType

    # Load the pure-Python PACBIN codec without importing the intentionally
    # stale shared-tree native runtime used by this publication-only test.
    package = ModuleType("pyamplicol")
    package.__path__ = []  # type: ignore[attr-defined]
    generation = ModuleType("pyamplicol.generation")
    generation.__path__ = []  # type: ignore[attr-defined]
    codec_name = "pyamplicol.generation.evaluator_container"
    codec_path = (
        Path(__file__).resolve().parents[2]
        / "src/pyamplicol/generation/evaluator_container.py"
    )
    spec = importlib.util.spec_from_file_location(codec_name, codec_path)
    assert spec is not None and spec.loader is not None
    codec = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "pyamplicol", package)
    monkeypatch.setitem(sys.modules, "pyamplicol.generation", generation)
    monkeypatch.setitem(sys.modules, codec_name, codec)
    spec.loader.exec_module(codec)
    PacbinMemberKind = codec.PacbinMemberKind
    PacbinMemberSource = codec.PacbinMemberSource
    write_pacbin_atomic = codec.write_pacbin_atomic

    process_id = "otf_dd_z"
    pack_relative = "model/eager-kernel-pack.json"
    runtime_relative = f"processes/{process_id}/on-the-fly-runtime.pacbin"
    seed_identity = {
        "abi": "pyamplicol-on-the-fly-process-seed-identity-v1",
        "process_digest": "1" * 64,
        "compiled_model_digest": "2" * 64,
        "recurrence_template_catalog_digest": "3" * 64,
        "prepared_kernel_pack_digest": "4" * 64,
        "recurrence_direct_template_catalog_digest": "5" * 64,
        "semantic_digest": "6" * 64,
        "external_permutation": [0, 1, 2],
        "external_sources": [
            {
                "source_slot": 0,
                "public_label": 1,
                "is_initial": True,
                "states": [
                    {
                        "state_index": 0,
                        "public_helicity": -1,
                        "prepared_mass_parameter_slot": None,
                    }
                ],
            },
            {
                "source_slot": 1,
                "public_label": 2,
                "is_initial": True,
                "states": [
                    {
                        "state_index": 0,
                        "public_helicity": 1,
                        "prepared_mass_parameter_slot": None,
                    }
                ],
            },
            {
                "source_slot": 2,
                "public_label": 3,
                "is_initial": False,
                "states": [
                    {
                        "state_index": 0,
                        "public_helicity": 0,
                        "prepared_mass_parameter_slot": 6,
                    }
                ],
            },
        ],
    }
    pack = {
        "backend": "jit",
        "eager_kernel_abi": "pyamplicol-eager-kernel-v1",
        "dependency_abis": {
            "symjit_application": "symjit-application-storage-v3",
            "symjit_plane_application": "pyamplicol-symjit-plane-application-v2",
        },
        "recurrence_template": {
            "header": {
                "compiled_model_digest": "2" * 64,
                "catalog_digest": "3" * 64,
                "prepared_kernel_pack_digest": "4" * 64,
            }
        },
        "recurrence_direct_template": {
            "catalog_digest": "5" * 64,
            "compiled_model_digest": "2" * 64,
            "recurrence_template_catalog_digest": "3" * 64,
            "prepared_kernel_pack_digest": "4" * 64,
        },
        "optimization_settings": {
            "backend": "jit",
            "jit_optimization_level": 2,
        },
        "kernels": [
            {
                "contract_kind": "vertex",
                "f64_evaluator_manifest": _canonical_symjit_evaluator(
                    "kernels/000000/application.symjit",
                    "kernels/000000/application.plane.symjit",
                    optimization_level=2,
                ),
            }
        ],
        "kernel_variants": [],
    }
    pack_path = tmp_path / pack_relative
    pack_path.parent.mkdir(parents=True)
    pack_data = json.dumps(pack, sort_keys=True).encode("ascii")
    pack_path.write_bytes(pack_data)
    runtime_path = tmp_path / runtime_relative
    runtime_path.parent.mkdir(parents=True)
    write_pacbin_atomic(
        runtime_path,
        (
            PacbinMemberSource(
                "on-the-fly/process-seed-v1.bin",
                PacbinMemberKind.ON_THE_FLY_PROCESS_SEED,
                io.BytesIO(b"compact-process-seed"),
            ),
        ),
    )
    runtime_data = runtime_path.read_bytes()
    prepared_identity = {
        "kind": "pyamplicol-prepared-kernel-pack-identity",
        "schema_version": 1,
        "abi": "pyamplicol-prepared-kernel-pack-identity-v2",
        "backend": "jit",
        "eager_kernel_abi": "pyamplicol-eager-kernel-v1",
        "identity_sha256": "7" * 64,
        "kernel_count": 1,
    }

    def payload(
        relative: str,
        data: bytes,
        *,
        role: str,
        media_type: str,
        owner: str | None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            path=relative,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            role=role,
            media_type=media_type,
            process_id=owner,
            executable=False,
        )

    manifest = SimpleNamespace(
        payloads=(
            payload(
                pack_relative,
                pack_data,
                role="evaluator-manifest",
                media_type="application/json",
                owner=None,
            ),
            payload(
                runtime_relative,
                runtime_data,
                role="evaluator-state",
                media_type="application/octet-stream",
                owner=process_id,
            ),
        ),
        extensions={"eager_prepared_pack": prepared_identity},
    )
    execution = {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-on-the-fly-execution",
        "required_runtime_capabilities": [
            "rusticol.on-the-fly.complex-f64.v1",
            "rusticol.on-the-fly.lc-color.v1",
        ],
        "process": "d d~ > z",
        "key": process_id,
        "color_accuracy": "lc",
        "external_pdg_order": [1, -1, 23],
        "kernel_pack": {
            "manifest_path": pack_relative,
            "payload_root": "model/eager-kernels",
        },
        "runtime_options": {
            "point_tile_size": 128,
            "query_construction_threads": 4,
        },
        "selector_policy": {
            "color_coverage": "complete",
            "reference_color_word": None,
            "trace_reflections_folded": False,
            "selector_census": {
                "physical_helicity_count": 1,
                "physical_color_flow_count": 1,
            },
        },
        "runtime_metadata": {
            "runtime_parameters": [],
            "parameter_projection": [
                {
                    "runtime_name": "particle.23.mass",
                    "component": 0,
                    "prepared_parameter_id": 6,
                }
            ],
            "external_legs": [
                {
                    "source_slot": 0,
                    "public_label": 1,
                    "outgoing_pdg": -1,
                    "is_initial": True,
                },
                {
                    "source_slot": 1,
                    "public_label": 2,
                    "outgoing_pdg": 1,
                    "is_initial": True,
                },
                {
                    "source_slot": 2,
                    "public_label": 3,
                    "outgoing_pdg": 23,
                    "is_initial": False,
                },
            ],
            "process_seed_identity": deepcopy(seed_identity),
        },
        "runtime_container": {
            "kind": "pyamplicol-on-the-fly-runtime-container",
            "schema_version": 1,
            "storage_abi": "pacbin-v1",
            "path": "on-the-fly-runtime.pacbin",
            "seed_member_path": "on-the-fly/process-seed-v1.bin",
        },
    }
    cell = _cell(ExecutionMode.ON_THE_FLY, optimization_level=2)
    effective_config = {"evaluator": {"optimization": {"cores": 4}}}
    monkeypatch.setattr(
        final_audit_module,
        "_expected_eager_prepared_pack_identity",
        lambda _cell: dict(prepared_identity),
    )
    monkeypatch.setattr(
        final_audit_module,
        "_native_on_the_fly_process_seed_identity",
        lambda payload: (
            deepcopy(seed_identity)
            if payload == b"compact-process-seed"
            else (_ for _ in ()).throw(FinalAuditError("undecodable compact seed"))
        ),
    )

    assert (
        _audit_on_the_fly_execution(
            tmp_path,
            manifest,
            execution,
            cell,
            process_id=process_id,
            execution_manifest_path=f"processes/{process_id}/execution.json",
            effective_config=effective_config,
        )
        == 1
    )

    mismatched_threads = deepcopy(execution)
    mismatched_threads["runtime_options"]["query_construction_threads"] = 3  # type: ignore[index]
    with pytest.raises(FinalAuditError, match="effective evaluator cores"):
        _audit_on_the_fly_execution(
            tmp_path,
            manifest,
            mismatched_threads,
            cell,
            process_id=process_id,
            execution_manifest_path=f"processes/{process_id}/execution.json",
            effective_config=effective_config,
        )

    for mutation in ("missing", "zero", "extra", "seed-mismatch"):
        changed_execution = deepcopy(execution)
        census = changed_execution["selector_policy"]["selector_census"]
        assert isinstance(census, dict)
        if mutation == "missing":
            del census["physical_color_flow_count"]
        elif mutation == "zero":
            census["physical_helicity_count"] = 0
        elif mutation == "extra":
            census["invented"] = 1
        else:
            census["physical_helicity_count"] = 2
        with pytest.raises(FinalAuditError, match=r"selector (?:helicity )?census"):
            _audit_on_the_fly_execution(
                tmp_path,
                manifest,
                changed_execution,
                cell,
                process_id=process_id,
                execution_manifest_path=f"processes/{process_id}/execution.json",
            )

    for mutation in ("digest", "missing-abi", "wrong-abi", "extra-field"):
        changed = dict(prepared_identity)
        if mutation == "digest":
            changed["identity_sha256"] = "8" * 64
        elif mutation == "missing-abi":
            del changed["abi"]
        elif mutation == "wrong-abi":
            changed["abi"] = "pyamplicol-prepared-kernel-pack-identity-v1"
        else:
            changed["invented"] = True
        changed_manifest = SimpleNamespace(
            payloads=manifest.payloads,
            extensions={"eager_prepared_pack": changed},
        )
        with pytest.raises(FinalAuditError, match="prepared-pack identity"):
            _audit_on_the_fly_execution(
                tmp_path,
                changed_manifest,
                execution,
                cell,
                process_id=process_id,
                execution_manifest_path=f"processes/{process_id}/execution.json",
            )

    for field, replacement in (
        ("process_digest", "7" * 64),
        ("semantic_digest", "8" * 64),
        ("prepared_kernel_pack_digest", "9" * 64),
    ):
        changed_native = deepcopy(seed_identity)
        changed_native[field] = replacement
        monkeypatch.setattr(
            final_audit_module,
            "_native_on_the_fly_process_seed_identity",
            lambda _payload, identity=changed_native: deepcopy(identity),
        )
        with pytest.raises(FinalAuditError, match="does not identify"):
            _audit_on_the_fly_execution(
                tmp_path,
                manifest,
                execution,
                cell,
                process_id=process_id,
                execution_manifest_path=f"processes/{process_id}/execution.json",
            )

    monkeypatch.setattr(
        final_audit_module,
        "_native_on_the_fly_process_seed_identity",
        lambda _payload: deepcopy(seed_identity),
    )
    changed_state = deepcopy(execution)
    changed_state["runtime_metadata"]["process_seed_identity"][  # type: ignore[index]
        "external_sources"
    ][2]["states"][0]["public_helicity"] = 1
    with pytest.raises(FinalAuditError, match="does not identify"):
        _audit_on_the_fly_execution(
            tmp_path,
            manifest,
            changed_state,
            cell,
            process_id=process_id,
            execution_manifest_path=f"processes/{process_id}/execution.json",
        )

    changed_anchor = deepcopy(execution)
    changed_anchor["runtime_metadata"]["external_legs"][2][  # type: ignore[index]
        "public_label"
    ] = 9
    with pytest.raises(FinalAuditError, match="source identity disagrees"):
        _audit_on_the_fly_execution(
            tmp_path,
            manifest,
            changed_anchor,
            cell,
            process_id=process_id,
            execution_manifest_path=f"processes/{process_id}/execution.json",
        )

    changed_mass_slot = deepcopy(execution)
    changed_mass_slot["runtime_metadata"]["parameter_projection"][0][  # type: ignore[index]
        "prepared_parameter_id"
    ] = 7
    with pytest.raises(FinalAuditError, match="mass slot differs"):
        _audit_on_the_fly_execution(
            tmp_path,
            manifest,
            changed_mass_slot,
            cell,
            process_id=process_id,
            execution_manifest_path=f"processes/{process_id}/execution.json",
        )

    monkeypatch.setattr(
        final_audit_module,
        "_native_on_the_fly_process_seed_identity",
        lambda _payload: (_ for _ in ()).throw(
            FinalAuditError("undecodable compact seed")
        ),
    )
    with pytest.raises(FinalAuditError, match="undecodable compact seed"):
        _audit_on_the_fly_execution(
            tmp_path,
            manifest,
            execution,
            cell,
            process_id=process_id,
            execution_manifest_path=f"processes/{process_id}/execution.json",
        )

    invented_seed_contract = deepcopy(execution)
    invented_seed_contract["runtime_container"]["seed_member_path"] = (  # type: ignore[index]
        "pyamplicol-on-the-fly-process-seed-v1"
    )
    with pytest.raises(FinalAuditError, match="runtime container contract"):
        _audit_on_the_fly_execution(
            tmp_path,
            manifest,
            invented_seed_contract,
            cell,
            process_id=process_id,
            execution_manifest_path=f"processes/{process_id}/execution.json",
        )

    recurrence_plan = deepcopy(execution)
    recurrence_plan["plan"] = {"materialized": True}
    with pytest.raises(FinalAuditError, match="compact contract"):
        _audit_on_the_fly_execution(
            tmp_path,
            manifest,
            recurrence_plan,
            cell,
            process_id=process_id,
            execution_manifest_path=f"processes/{process_id}/execution.json",
        )


def test_otf_effective_config_contract() -> None:
    cell = replace(
        _cell(ExecutionMode.ON_THE_FLY, optimization_level=2),
        workload=Workload.ALL_FLOW,
    )
    effective = {
        "evaluator": {
            "execution_mode": "on-the-fly",
            "backend": "jit",
            "jit": {"optimization_level": 2},
        },
        "color": {"accuracy": "lc", "lc_flow_layout": "topology-replay"},
        "generation": {
            "relation_discovery": {"mode": "off"},
            "validation": {
                "enabled": False,
                "samples": 10,
                "seed": 12345,
                "relative_tolerance": 1.0e-12,
                "absolute_tolerance": 1.0e-300,
                "post_build_validation": False,
            },
        },
    }

    _audit_effective_config_mapping(cell, effective, context="otf")

    wrong_layout = deepcopy(effective)
    wrong_layout["color"]["lc_flow_layout"] = "all-flow-union"  # type: ignore[index]
    with pytest.raises(FinalAuditError, match=r"color\.lc_flow_layout"):
        _audit_effective_config_mapping(cell, wrong_layout, context="otf")


@pytest.mark.parametrize("workload", (Workload.SELECTED_FLOW, Workload.ALL_FLOW))
def test_otf_artifact_audit_distinguishes_requested_and_physical_layout(
    tmp_path: Path,
    monkeypatch,
    workload: Workload,
) -> None:
    cell = replace(
        _cell(ExecutionMode.ON_THE_FLY, optimization_level=2),
        workload=workload,
    )
    process_id = "d_dbar_to_z"
    effective = {
        "evaluator": {
            "execution_mode": "on-the-fly",
            "backend": "jit",
            "jit": {"optimization_level": 2},
        },
        "color": {"accuracy": "lc", "lc_flow_layout": "topology-replay"},
        "generation": {
            "relation_discovery": {"mode": "off"},
            "validation": {
                "enabled": False,
                "samples": 10,
                "seed": 12345,
                "relative_tolerance": 1.0e-12,
                "absolute_tolerance": 1.0e-300,
                "post_build_validation": False,
            },
        },
    }
    capability = final_audit_module._ARENA_CAPABILITY[ExecutionMode.ON_THE_FLY]
    process = {
        "id": process_id,
        "expression": cell.process,
        "color_accuracy": cell.measurement.accuracy.value,
        "required_runtime_capabilities": (capability,),
    }
    manifest = SimpleNamespace(
        artifact_id=_ARTIFACT_ID,
        processes=(process,),
        runtime={"engine_version": "runtime-v1"},
    )
    inspected = SimpleNamespace(
        id=process_id,
        execution_mode=ExecutionMode.ON_THE_FLY.value,
        generation_specialized_axes=(),
        selected_source_helicities=(),
        selected_color_sector_ids=(),
        lc_flow_layout="compact/query-local",
    )
    inspection = SimpleNamespace(integrity="verified", processes=(inspected,))

    monkeypatch.setattr(
        "pyamplicol.artifacts.inspect_artifact", lambda _artifact: inspection
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.load_manifest", lambda *_args, **_kwargs: manifest
    )
    monkeypatch.setattr(
        final_audit_module,
        "_authenticated_effective_config",
        lambda _artifact, _manifest: effective,
    )
    monkeypatch.setattr(
        final_audit_module, "_audit_model_source", lambda _cell, _effective: None
    )
    monkeypatch.setattr(
        final_audit_module,
        "_find_process_execution",
        lambda _artifact, _manifest, _process_id: (
            {},
            f"processes/{process_id}/execution.json",
            "1" * 64,
            (capability,),
        ),
    )
    monkeypatch.setattr(
        final_audit_module,
        "_audit_on_the_fly_execution",
        lambda *_args, **_kwargs: 1,
    )

    evidence = audit_artifact(cell, tmp_path, process_id)
    assert evidence.execution_mode == ExecutionMode.ON_THE_FLY.value
    assert evidence.arena_record_count == 1

    inspected.lc_flow_layout = "topology-replay"
    with pytest.raises(FinalAuditError, match="physics/execution contract"):
        audit_artifact(cell, tmp_path, process_id)


def test_effective_toml_is_payload_authenticated_and_model_bound(
    tmp_path: Path,
) -> None:
    relative = "config/effective.toml"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    data = b'[model]\nsource = "built-in-sm"\n'
    path.write_bytes(data)
    manifest = SimpleNamespace(
        configuration={"effective_path": relative},
        model={"source_kind": "built-in-sm"},
        payloads=(
            SimpleNamespace(
                path=relative,
                role="configuration-effective",
                media_type="application/toml",
                process_id=None,
                executable=False,
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            ),
        ),
    )
    effective = _authenticated_effective_config(tmp_path, manifest)
    _audit_model_source(_cell(ExecutionMode.COMPILED), effective)

    path.write_bytes(b'[model]\nsource = "built-in-xx"\n')
    with pytest.raises(FinalAuditError, match="artifact manifest"):
        _authenticated_effective_config(tmp_path, manifest)


def test_eager_and_recurrence_arena_abis_are_audited(tmp_path: Path) -> None:
    pack_path = tmp_path / "model" / "eager-kernel-pack.json"
    pack_path.parent.mkdir(parents=True)
    vertex_evaluator = _canonical_symjit_evaluator(
        "kernels/000000/application.symjit",
        "kernels/000000/application.plane.symjit",
        optimization_level=2,
    )
    vertex_evaluator["direct_table"] = {
        "capability": "eager-direct-arena-v1",
        "descriptor_abi": "pyamplicol-eager-plane-table-descriptor-v1",
        "binding_abi": "pyamplicol-eager-plane-table-binding-v2",
        "source_application_abi": "pyamplicol-symjit-plane-application-v2",
    }
    pack_payload = {
        "backend": "jit",
        "optimization_settings": {
            "backend": "jit",
            "jit_optimization_level": 2,
        },
        "recurrence_direct_template": {
            "abi": "pyamplicol-recurrence-direct-template-v1",
            "backend": "jit",
            "optimization_level": 2,
            "portable": True,
            "templates": [{"backend": "jit", "optimization_level": 2}],
        },
        "kernel_variants": [],
        "kernels": [
            {
                "contract_kind": "vertex",
                "f64_evaluator_manifest": vertex_evaluator,
            },
            {
                "contract_kind": "model-parameter",
                "f64_evaluator_manifest": {
                    "kind": "symjit-application-evaluator",
                    "runtime_capability": "symjit.application.complex-f64.v1",
                    "optimization_level": 2,
                    "application_abi": "symjit-application-storage-v3",
                },
            },
        ],
    }
    payload_record = SimpleNamespace(
        path="model/eager-kernel-pack.json",
        role="evaluator-manifest",
        media_type="application/json",
        process_id=None,
        executable=False,
        size_bytes=0,
        sha256="",
    )

    def write_pack(value: object) -> None:
        data = json.dumps(value).encode("utf-8")
        pack_path.write_bytes(data)
        payload_record.size_bytes = len(data)
        payload_record.sha256 = hashlib.sha256(data).hexdigest()

    write_pack(pack_payload)
    manifest = SimpleNamespace(
        payloads=(payload_record,),
        extensions={
            "eager_prepared_pack": {
                "kind": "pyamplicol-prepared-kernel-pack-identity",
                "schema_version": 1,
                "backend": "jit",
                "eager_kernel_abi": "pyamplicol-eager-kernel-v1",
                "identity_sha256": "c" * 64,
                "kernel_count": 2,
            }
        },
    )
    eager = {
        "kind": "pyamplicol-runtime-eager-execution",
        "eager_plan_abi": "pyamplicol-eager-plan-v3",
        "plan": {
            "eager_plan_abi": "pyamplicol-eager-plan-v3",
            "lowering_input_abi": "pyamplicol-eager-lowering-input-v1",
            "runtime_layout_abi": "pyamplicol-eager-runtime-layout-v1",
        },
        "kernel_pack": {"manifest_path": "model/eager-kernel-pack.json"},
    }
    assert (
        _audit_eager_execution(
            tmp_path,
            manifest,
            eager,
            _cell(
                ExecutionMode.EAGER,
                optimization_level=2,
            ),
        )
        == 1
    )
    wrong_capability = deepcopy(pack_payload)
    wrong_capability["kernels"][0]["f64_evaluator_manifest"][  # type: ignore[index]
        "runtime_capability"
    ] = "symbolica.compiled-cpp.complex-f64.v1"
    write_pack(wrong_capability)
    with pytest.raises(FinalAuditError, match="runtime_capability"):
        _audit_eager_execution(
            tmp_path,
            manifest,
            eager,
            _cell(ExecutionMode.EAGER, optimization_level=2),
        )
    for nested_corruption in (
        {"nested": [{"direct_table": None}]},
        {"nested": {"capability": "eager-direct-arena-v1"}},
        {"nested": {"descriptor_abi": "pyamplicol-eager-plane-table-descriptor-v1"}},
        {"nested": {"binding_abi": "pyamplicol-eager-plane-table-binding-v2"}},
        {"nested": {"application_abi": "pyamplicol-eager-native-direct-table-v1"}},
    ):
        corrupted = deepcopy(pack_payload)
        corrupted["kernels"][1]["f64_evaluator_manifest"].update(  # type: ignore[index]
            nested_corruption
        )
        write_pack(corrupted)
        with pytest.raises(FinalAuditError, match="model-parameter"):
            _audit_eager_execution(
                tmp_path,
                manifest,
                eager,
                _cell(ExecutionMode.EAGER, optimization_level=2),
            )
    corrupted = deepcopy(pack_payload)
    corrupted["kernels"][1]["nested_kernel_metadata"] = {  # type: ignore[index]
        "direct_table": {}
    }
    write_pack(corrupted)
    with pytest.raises(FinalAuditError, match="model-parameter"):
        _audit_eager_execution(
            tmp_path,
            manifest,
            eager,
            _cell(ExecutionMode.EAGER, optimization_level=2),
        )
    source_relative = "model/eager-kernels/kernels/000000/application-0.plane.symjit"
    source_data = b"authenticated-symjit-application"
    source_sha256 = hashlib.sha256(source_data).hexdigest()
    recurrence, recurrence_pack = _recurrence_source_fixture(source_sha256)
    write_pack(recurrence_pack)
    source_path = tmp_path / source_relative
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source_data)
    source_record = SimpleNamespace(
        path=source_relative,
        role="evaluator-state",
        media_type="application/octet-stream",
        process_id=None,
        executable=False,
        size_bytes=len(source_data),
        sha256=source_sha256,
    )
    manifest.payloads = (*manifest.payloads, source_record)
    assert (
        _audit_recurrence_execution(
            recurrence,
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        )
        == 4
    )
    source_identity = _audit_recurrence_source_pack(
        tmp_path,
        manifest,
        recurrence,
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        execution_manifest_path="processes/d_dbar_to_z/execution.json",
        execution_manifest_sha256="1" * 64,
    )
    assert source_identity == {
        "kind": "authenticated-recurrence-direct-template-source-v1",
        "optimization_level": 2,
        "direct_template_count": 1,
        "prepared_direct_template_count": 1,
        "source_evaluator_leaf_count": 1,
        "source_application_abi": "pyamplicol-symjit-plane-application-v2",
        "direct_application_abi": "pyamplicol-symjit-plane-application-v2",
        "prepared_kernel_pack_digest": "a" * 64,
        "direct_template_catalog_digest": recurrence["direct_template_catalog_digest"],
        "execution_manifest_path": "processes/d_dbar_to_z/execution.json",
        "execution_manifest_sha256": "1" * 64,
        "kernel_pack_path": "model/eager-kernel-pack.json",
        "kernel_pack_sha256": payload_record.sha256,
    }
    wrong_o2 = deepcopy(recurrence_pack)
    wrong_o2["optimization_settings"]["jit_optimization_level"] = 1  # type: ignore[index]
    write_pack(wrong_o2)
    with pytest.raises(FinalAuditError, match="does not prove JIT O2"):
        _audit_recurrence_source_pack(
            tmp_path,
            manifest,
            recurrence,
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
            execution_manifest_path="processes/d_dbar_to_z/execution.json",
            execution_manifest_sha256="1" * 64,
        )
    write_pack(pack_payload)
    recurrence["plan"]["inspection_summary"]["direct_arena"][  # type: ignore[index]
        "packed_input_bytes"
    ] = 8
    with pytest.raises(FinalAuditError, match="packed_input_bytes"):
        _audit_recurrence_execution(
            recurrence,
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("execution_template_abi", "direct_template_abi"),
        ("plan_template_abi", "direct_template_abi"),
        ("execution_backend_abi", "direct_backend_abi"),
        ("plan_backend_abi", "direct_backend_abi"),
        ("plan_prepared_digest", "source digests"),
        ("plan_catalog_digest", "source digests"),
        ("pack_prepared_digest", "execution_digest_links"),
        ("execution_catalog_link", "execution_digest_links"),
        ("catalog_backend_abi", "does not prove JIT O2"),
        ("catalog_canonicalization_abi", "does not prove JIT O2"),
        ("catalog_digest", "catalog_digest"),
        ("template_portable", "template\\[0\\] contract"),
        ("template_abi", "template\\[0\\] contract"),
        ("binding_abi", "payload-binding contract"),
        ("source_application_abi", "source application"),
        ("direct_application_abi", "source application"),
        ("payload_digest", "payload-binding contract"),
        ("source_leaf_runtime_capability", "runtime_capability"),
        ("source_leaf_application_abi", "application_abi"),
        ("source_leaf_optimization", "optimization_level"),
        ("source_leaf_plane_abi", "plane_application.application_abi"),
        ("source_leaf_plane_digest", "plane_application.source_digest"),
        (
            "source_leaf_plane_optimization",
            "plane_application.optimization_level",
        ),
        ("source_leaf_path", "payload is missing"),
        ("source_payload_digest_link", "not bound to its prepared kernel"),
    ),
)
def test_final_audit_recurrence_source_pack_rejects_every_broken_link(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    pack_relative = "model/eager-kernel-pack.json"
    source_relative = "model/eager-kernels/kernels/000000/application-0.plane.symjit"
    source_data = b"authenticated-symjit-application"
    source_sha256 = hashlib.sha256(source_data).hexdigest()
    execution, pack = _recurrence_source_fixture(source_sha256)
    plan = execution["plan"]
    assert isinstance(plan, dict)
    catalog = pack["recurrence_direct_template"]
    assert isinstance(catalog, dict)
    templates = catalog["templates"]
    assert isinstance(templates, list)
    template = templates[0]
    assert isinstance(template, dict)
    binding = template["payload_binding"]
    assert isinstance(binding, dict)
    kernels = pack["kernels"]
    assert isinstance(kernels, list)
    kernel = kernels[0]
    assert isinstance(kernel, dict)
    source_leaf = kernel["f64_evaluator_manifest"]
    assert isinstance(source_leaf, dict)
    refresh_catalog: bool | None = None

    if corruption == "execution_template_abi":
        execution["direct_template_abi"] = "wrong"
    elif corruption == "plan_template_abi":
        plan["direct_template_abi"] = "wrong"
    elif corruption == "execution_backend_abi":
        execution["direct_backend_abi"] = "wrong"
    elif corruption == "plan_backend_abi":
        plan["direct_backend_abi"] = "wrong"
    elif corruption == "plan_prepared_digest":
        plan["prepared_kernel_pack_digest"] = "0" * 64
    elif corruption == "plan_catalog_digest":
        plan["direct_template_catalog_digest"] = "0" * 64
    elif corruption == "pack_prepared_digest":
        catalog["prepared_kernel_pack_digest"] = "0" * 64
        refresh_catalog = True
    elif corruption == "execution_catalog_link":
        execution["direct_template_catalog_digest"] = "0" * 64
        plan["direct_template_catalog_digest"] = "0" * 64
    elif corruption == "catalog_backend_abi":
        catalog["backend_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "catalog_canonicalization_abi":
        catalog["canonicalization_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "catalog_digest":
        catalog["compiled_model_digest"] = "0" * 64
    elif corruption == "template_portable":
        template["portable"] = False
        refresh_catalog = True
    elif corruption == "template_abi":
        template["abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "binding_abi":
        binding["abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "source_application_abi":
        binding["source_application_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "direct_application_abi":
        binding["direct_application_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "payload_digest":
        binding["payload_digest"] = "0" * 64
        refresh_catalog = False
    elif corruption == "source_leaf_runtime_capability":
        source_leaf["runtime_capability"] = "wrong"
    elif corruption == "source_leaf_application_abi":
        source_leaf["application_abi"] = "wrong"
    elif corruption == "source_leaf_optimization":
        source_leaf["optimization_level"] = 1
    elif corruption == "source_leaf_plane_abi":
        source_leaf["plane_application"]["application_abi"] = "wrong"
    elif corruption == "source_leaf_plane_digest":
        source_leaf["plane_application"]["source_digest"] = "wrong"
    elif corruption == "source_leaf_plane_optimization":
        source_leaf["plane_application"]["optimization_level"] = 1
    elif corruption == "source_leaf_path":
        source_leaf["plane_application"]["application_path"] = (
            "kernels/000000/missing.plane.symjit"
        )
    elif corruption == "source_payload_digest_link":
        binding["source_application_sha256"] = "0" * 64
        refresh_catalog = True
    else:
        pytest.fail(f"unknown corruption {corruption}")

    if refresh_catalog is not None:
        _refresh_recurrence_source_catalog(
            execution,
            pack,
            refresh_binding_digest=refresh_catalog,
        )
    pack_path = tmp_path / pack_relative
    pack_path.parent.mkdir(parents=True)
    pack_data = json.dumps(pack, sort_keys=True).encode("ascii")
    pack_path.write_bytes(pack_data)
    source_path = tmp_path / source_relative
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source_data)
    manifest = SimpleNamespace(
        payloads=(
            SimpleNamespace(
                path=pack_relative,
                role="evaluator-manifest",
                media_type="application/json",
                process_id=None,
                executable=False,
                size_bytes=len(pack_data),
                sha256=hashlib.sha256(pack_data).hexdigest(),
            ),
            SimpleNamespace(
                path=source_relative,
                role="evaluator-state",
                media_type="application/octet-stream",
                process_id=None,
                executable=False,
                size_bytes=len(source_data),
                sha256=source_sha256,
            ),
        ),
        extensions={},
    )
    with pytest.raises(FinalAuditError, match=message):
        _audit_recurrence_source_pack(
            tmp_path,
            manifest,
            execution,
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
            execution_manifest_path="processes/d_dbar_to_z/execution.json",
            execution_manifest_sha256="1" * 64,
        )


def test_pointwise_audit_recomputes_arithmetic_and_status() -> None:
    record = {
        "status": "ok",
        "candidate": 1.0 + 1.0e-13,
        "baseline": 1.0,
        "absolute_difference": abs((1.0 + 1.0e-13) - 1.0),
        "relative_difference": abs((1.0 + 1.0e-13) - 1.0),
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance": 1.0e-15,
    }
    _audit_pointwise(
        record,
        context="pointwise",
        expected_candidate=1.0 + 1.0e-13,
        expected_baseline=1.0,
        expected_relative_tolerance=1.0e-12,
    )
    record["relative_difference"] = 0.0
    with pytest.raises(FinalAuditError, match="relative difference"):
        _audit_pointwise(
            record,
            context="pointwise",
            expected_candidate=1.0 + 1.0e-13,
            expected_baseline=1.0,
            expected_relative_tolerance=1.0e-12,
        )


def test_pointwise_audit_authenticates_conditioned_point_binding() -> None:
    from tools.performance_report.runner import pointwise_validation

    selector_identity = {"cell_id": "candidate", "value_kind": "total"}
    record = pointwise_validation(
        1.0 + 1.0e-13,
        1.0,
        comparison_binding={
            "point_digest": "a" * 64,
            "selector_component_identity": selector_identity,
            "selector_component_sha256": hashlib.sha256(
                json.dumps(
                    selector_identity,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
            ).hexdigest(),
            "candidate_source_sha256": "b" * 64,
            "baseline_source_sha256": "c" * 64,
        },
    )
    _audit_pointwise(
        record,
        context="conditioned",
        expected_candidate=1.0 + 1.0e-13,
        expected_baseline=1.0,
        expected_relative_tolerance=1.0e-12,
        expected_point_digest="a" * 64,
        expected_candidate_source_sha256="b" * 64,
        expected_baseline_source_sha256="c" * 64,
        expected_selector_identity=selector_identity,
    )
    with pytest.raises(FinalAuditError, match="different point"):
        _audit_pointwise(
            record,
            context="conditioned",
            expected_candidate=1.0 + 1.0e-13,
            expected_baseline=1.0,
            expected_relative_tolerance=1.0e-12,
            expected_point_digest="b" * 64,
        )
    with pytest.raises(FinalAuditError, match="candidate scale source"):
        _audit_pointwise(
            record,
            context="conditioned",
            expected_candidate=1.0 + 1.0e-13,
            expected_baseline=1.0,
            expected_relative_tolerance=1.0e-12,
            expected_point_digest="a" * 64,
            expected_candidate_source_sha256="d" * 64,
        )


def test_python_package_tree_identity_is_canonical_and_ignores_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools.performance_report import runtime_evidence

    cache_prefix = tmp_path / "absent-cache-prefix"
    monkeypatch.setattr(
        runtime_evidence,
        "_isolated_startup_flags",
        lambda: (True, True, True),
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(cache_prefix))
    package = tmp_path / "pyamplicol"
    (package / "submodule").mkdir(parents=True)
    (package / "__pycache__").mkdir()
    (package / "api.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "submodule" / "data.bin").write_bytes(b"\x00\x01")
    (package / "__pycache__" / "api.pyc").write_bytes(b"cache one")

    first = _python_package_tree_identity(package)
    assert first["kind"] == "pyamplicol-python-package-tree-v2"
    assert first["roots"] == [str(package)]
    assert first["file_count"] == 2
    assert first["total_bytes"] == 12

    (package / "__pycache__" / "api.pyc").write_bytes(b"changed cache")
    assert _python_package_tree_identity(package) == first

    (package / "api.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = _python_package_tree_identity(package)
    assert changed["sha256"] != first["sha256"]


def test_runtime_namespace_uses_only_first_native_candidate(tmp_path: Path) -> None:
    from importlib.machinery import EXTENSION_SUFFIXES

    checkout = tmp_path / "checkout"
    source_package = checkout / "src" / "pyamplicol"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_bytes(b"")
    first_site = tmp_path / "first-site"
    second_site = tmp_path / "second-site"
    first_package = first_site / "pyamplicol"
    second_package = second_site / "pyamplicol"
    first_package.mkdir(parents=True)
    second_package.mkdir(parents=True)
    suffix = EXTENSION_SUFFIXES[0]
    first_native = first_package / f"_rusticol{suffix}"
    second_native = second_package / f"_rusticol{suffix}"
    first_native.write_bytes(b"first")
    second_native.write_bytes(b"stale")

    roots, native = _runtime_namespace_paths(
        checkout,
        search_path=(str(first_site), str(second_site)),
    )

    assert roots == (source_package.resolve(), first_package.resolve())
    assert native == first_native.resolve()
    assert second_package.resolve() not in roots


def test_runtime_namespace_rejects_ambiguous_first_native_root(
    tmp_path: Path,
) -> None:
    from importlib.machinery import EXTENSION_SUFFIXES

    checkout = tmp_path / "checkout"
    (checkout / "src" / "pyamplicol").mkdir(parents=True)
    site = tmp_path / "site"
    package = site / "pyamplicol"
    package.mkdir(parents=True)
    suffix = EXTENSION_SUFFIXES[0]
    (package / f"_rusticol{suffix}").write_bytes(b"one")
    (package / f"_rusticol_duplicate{suffix}").write_bytes(b"two")

    with pytest.raises(FinalAuditError, match="ambiguous"):
        _runtime_namespace_paths(checkout, search_path=(str(site),))


def test_direct_final_audit_cli_fails_closed_when_not_already_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools.performance_report.runtime_evidence import RuntimeEvidenceError

    checkout = tmp_path / "checkout"
    monkeypatch.setattr(
        "tools.performance_report.final_audit.source_only_bytecode_policy",
        lambda: (_ for _ in ()).throw(RuntimeEvidenceError("not isolated")),
    )
    raw = ("--repo-root", str(checkout), "--expected-source-revision", _REVISION)
    with pytest.raises(
        FinalAuditError,
        match=(r"src/pyamplicol/_profiling_campaign/result_tables\.py final-audit"),
    ):
        _ensure_exact_cli_python(checkout, raw)


def test_master_tex_dependency_graph_reaches_every_canonical_table(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    sections = docs / "sections"
    tables = docs / "tables"
    sections.mkdir(parents=True)
    tables.mkdir()
    master = docs / "pyAmpliCol.tex"
    section = sections / "performance.tex"
    first = tables / "result_first_table.tex"
    second = tables / "result_second_table.tex"
    unreachable = tables / "result_unreachable_table.tex"
    for table in (first, second, unreachable):
        table.write_text(f"% {table.name}\n", encoding="utf-8")
    master.write_text(
        "\\input{sections/performance}\n"
        "% \\\\input{tables/result_unreachable_table.tex}\n",
        encoding="utf-8",
    )
    section.write_text(
        "\\input{../tables/result_first_table.tex}\n"
        "\\include{../tables/result_second_table}\n",
        encoding="utf-8",
    )

    identity = _audit_tex_table_reachability(master, (first, second))
    assert identity["reachable_table_source_count"] == 2
    assert identity["reachable_tex_source_count"] == 4

    with pytest.raises(FinalAuditError, match="not reachable"):
        _audit_tex_table_reachability(master, (first, second, unreachable))


def test_active_runtime_snapshot_binds_exact_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyamplicol
    from pyamplicol._internal import versions

    native_path = tmp_path / "_rusticol.so"
    native_path.write_bytes(b"native")
    native_version = pyamplicol.__version__.replace(".dev", "-dev.")
    native = SimpleNamespace(
        __file__=str(native_path),
        package_version=lambda: native_version,
        native_build_inputs_sha256=lambda: "e" * 64,
        target_info=lambda: SimpleNamespace(
            triple="aarch64-apple-darwin",
            cpu_features=(),
        ),
    )
    package_root = Path(str(pyamplicol.__file__)).parent.resolve()
    package_tree = {
        "kind": "pyamplicol-python-package-tree-v2",
        "root": str(package_root),
        "roots": [str(package_root)],
        "file_count": 1,
        "total_bytes": 1,
        "sha256": "9" * 64,
        "member_set_stable": True,
        "namespace_bound_to_root_fd": True,
        "bytecode_policy": _bytecode_policy(),
    }
    native_identity = {
        "path": str(native_path),
        "size": len(b"native"),
        "sha256": hashlib.sha256(b"native").hexdigest(),
    }
    preimport_identity = {
        "kind": "pyamplicol-preimport-runtime-identity-v1",
        "python_package_tree": package_tree,
        "native_extension": native_identity,
    }
    monkeypatch.setattr(
        "tools.performance_report.final_audit._prepare_exact_pyamplicol_namespace",
        lambda _checkout: (
            pyamplicol,
            (package_root,),
            native_path,
            preimport_identity,
        ),
    )
    monkeypatch.setattr(
        "tools.performance_report.final_audit._python_package_tree_identity",
        lambda roots: (
            package_tree
            if tuple(roots) == (package_root,)
            else pytest.fail(f"unexpected package roots: {roots}")
        ),
    )
    monkeypatch.setattr(
        "tools.performance_report.final_audit.loaded_pyamplicol_origin_policy",
        lambda roots, **kwargs: (
            _loaded_origin_policy()
            if (
                tuple(roots) == (package_root,)
                and kwargs["native_extension"] == native_path
                and kwargs["expected_package_identity"] == package_tree
                and kwargs["expected_native_identity"] == native_identity
            )
            else pytest.fail("unexpected loaded-origin evidence request")
        ),
    )
    monkeypatch.setattr(
        "tools.performance_report.final_audit.importlib.import_module",
        lambda name: native if name == "pyamplicol._rusticol" else None,
    )
    monkeypatch.setattr(versions, "verify_native_module", lambda _native: None)
    monkeypatch.setattr(
        versions,
        "_active_build_info",
        lambda: {
            "schema_version": 1,
            "version": pyamplicol.__version__,
            "candidate_fingerprint": "candidate",
            "source_revision": _REVISION,
            "source_checkout": str(tmp_path),
            "native_build_inputs_sha256": "e" * 64,
            "publishable": False,
        },
    )

    snapshot = _active_runtime_snapshot(
        _REVISION,
        expected_checkout=tmp_path,
    )
    assert snapshot["native_build_inputs_sha256"] == "e" * 64
    assert snapshot["python_package_tree"]["kind"] == (  # type: ignore[index]
        "pyamplicol-python-package-tree-v2"
    )
    assert snapshot["native_extension"]["package_version"] == native_version  # type: ignore[index]
    assert snapshot["loaded_module_origin_policy"] == _loaded_origin_policy()

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(FinalAuditError, match="not bound"):
        _active_runtime_snapshot(_REVISION, expected_checkout=other)


@dataclass(frozen=True)
class _Flow:
    id: str
    word: tuple[int, ...]


@dataclass(frozen=True)
class _Helicity:
    id: str
    values: tuple[int, ...]


@dataclass(frozen=True)
class _Particle:
    label: int


class _Resolved:
    values = (((1.0 + 0.0j,),),)
    helicity_ids = ("h:-1,+1,-1",)
    color_ids = ("flow:2,1",)

    def total(self) -> tuple[complex, ...]:
        return (1.0 + 0.0j,)


class _Runtime:
    artifact_id = _ARTIFACT_ID
    execution_mode = "recurrence"

    def __init__(self) -> None:
        self.physics = SimpleNamespace(
            color_accuracy="lc",
            selector_capabilities=("helicity", "color_flow"),
            external_particles=(_Particle(1), _Particle(2), _Particle(3)),
            helicities=(_Helicity("h:-1,+1,-1", (-1, 1, -1)),),
            color_flows=(_Flow("flow:2,1", (2, 1)),),
        )

    def evaluate(
        self,
        _points: object,
        **_kwargs: object,
    ) -> tuple[complex, ...]:
        return (1.0 + 0.0j,)

    def evaluate_resolved(
        self,
        _points: object,
        **_kwargs: object,
    ) -> _Resolved:
        return _Resolved()


class _Catalog:
    def __init__(self, baseline: CellSpec, candidate: CellSpec) -> None:
        self.baseline = baseline
        self.candidate = candidate

    def measurement_cells(self) -> tuple[CellSpec, ...]:
        return (self.baseline, self.candidate)

    def baseline_cell(self, cell: CellSpec) -> CellSpec | None:
        return self.baseline if cell == self.candidate else None

    def validation_baseline_cell(self, cell: CellSpec) -> CellSpec | None:
        return self.baseline_cell(cell)

    def static_na_reason(self, _cell: CellSpec) -> None:
        return None

    def legacy_reference_available(self, _cell: CellSpec) -> bool:
        return True


def _selector() -> dict[str, object]:
    return {
        "selected_color_flow_ids": ["flow:2,1"],
        "selected_color_words": [[2, 1]],
        "all_flow_helicity_ids": ["h:-1,+1,-1"],
        "all_flow_source_helicities": {"1": -1, "2": 1, "3": -1},
        "point_digest": (
            "20d4353b25501bdf6c980930babe948e7c1d0364a3169d1ef5dc2a9da427e05c"
        ),
    }


def _lc_common_component(cell_id: str) -> dict[str, object]:
    selector = _selector()
    return {
        "abi": LC_COMMON_COMPONENT_ABI,
        "cell_id": cell_id,
        "value": 1.0,
        "point_digest": selector["point_digest"],
        "helicity_ids": selector["all_flow_helicity_ids"],
        "color_flow_ids": selector["selected_color_flow_ids"],
    }


def test_otf_v2_replay_uses_compact_selector_context_without_dense_physics(
    tmp_path: Path,
) -> None:
    cell = _cell(ExecutionMode.ON_THE_FLY, optimization_level=2)
    points = final_audit_module.shared_validation_points(cell.process)
    raw_selector = _selector()
    raw_selector["point_digest"] = report_runner.point_digest(points)
    contract = report_runner.SelectorContract.from_mapping(raw_selector)

    class Backend:
        def __init__(self) -> None:
            self.context_requests: list[tuple[str, ...]] = []
            self.selector_requests: list[tuple[str, tuple[str, ...]]] = []

        def _on_the_fly_benchmark_context(
            self,
            requested: tuple[str, ...],
        ) -> dict[str, object]:
            self.context_requests.append(requested)
            return {
                "process_id": "d_dbar_to_z",
                "process_expression": cell.process,
                "color_accuracy": "lc",
                "helicity_count": 1,
                "color_count": 1,
                "selected_color_ids": list(requested),
            }

        def _point_selector_indices(
            self,
            values: tuple[str, ...],
            name: str,
        ) -> tuple[int, ...]:
            self.selector_requests.append((name, values))
            return (0,)

    class Runtime:
        artifact_id = _ARTIFACT_ID
        execution_mode = "on-the-fly"

        def __init__(self) -> None:
            self._backend = Backend()
            self.dense_physics_access_count = 0

        @property
        def physics(self) -> object:
            self.dense_physics_access_count += 1
            raise AssertionError("OTF replay opened dense physics")

        def evaluate(self, _points: object, **_kwargs: object) -> tuple[complex, ...]:
            return (1.0 + 0.0j,)

        def evaluate_resolved(self, _points: object, **_kwargs: object) -> _Resolved:
            return _Resolved()

    runtime = Runtime()
    resolved_sum = report_runner.resolved_sum_validation(
        runtime,
        points,
        cell=cell,
        selector_contract=contract,
    )
    measurement = {
        "matrix_element": 1.0,
        "selector_contract": raw_selector,
        "validation": {
            "resolved_sum": resolved_sum,
            LC_COMMON_COMPONENT_FIELD: {
                "abi": LC_COMMON_COMPONENT_ABI,
                "cell_id": cell.cell_id,
                "value": 1.0,
                "point_digest": contract.point_digest,
                "helicity_ids": list(contract.all_flow_helicity_ids),
                "color_flow_ids": list(contract.selected_color_flow_ids),
            },
        },
        "provenance": {"runtime_identity": {"loaded_execution_mode": "on-the-fly"}},
    }
    reference = final_audit_module._ArtifactReference(
        cell=cell,
        measurement=measurement,
        path=tmp_path,
        process_id="d_dbar_to_z",
    )
    evidence = ArtifactEvidence(
        artifact_id=_ARTIFACT_ID,
        process_id="d_dbar_to_z",
        runtime_version="test",
        runtime_capabilities=(),
        execution_manifest_path="execution.json",
        execution_manifest_sha256="1" * 64,
        execution_mode="on-the-fly",
        arena_record_count=1,
        direct_leaf_count=0,
        effective_config={},
        source_jit_identity=None,
    )

    replay = final_audit_module._replay_cell(runtime, reference, evidence)

    assert replay.matrix_element == 1.0
    assert replay.resolved_maximum_absolute == 0.0
    assert replay.resolved_maximum_relative == 0.0
    assert runtime.dense_physics_access_count == 0
    assert runtime._backend.context_requests == [(), ()]
    assert runtime._backend.selector_requests == [
        ("color_flow_by_point", ("flow:2,1",)),
        ("helicity_by_point", ("h:-1,+1,-1",)),
    ]


def _source_provenance() -> dict[str, object]:
    tree = "c" * 40
    return {
        "report_source_identity_schema": "pyamplicol-report-source-v1",
        "report_source_revision": _REVISION,
        "report_source_tree": tree,
        "report_measured_source_revision": _REVISION,
        "report_measured_source_tree": tree,
        "report_source_clean": True,
    }


def _baseline_measurement(
    cell_id: str = "reference-amplicol-lc-n1-dd-z-selected-flow",
) -> dict[str, object]:
    result = empty_measurement()
    result.update(
        {
            "status": "ok",
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 1.0e-6,
            "execution_seconds_per_point": 1.0e-6,
            "matrix_element": 1.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 1.0e-9,
            "relative_standard_error": 1.0e-3,
            "artifact": {"path": "/legacy", "process_row": "row"},
            "selector_contract": _selector(),
            "validation": {
                "status": "ok",
                "method": "independent-original-amplicol-oracle",
                "point_digest": _selector()["point_digest"],
                DIRECT_AGREEMENT_FIELD: [],
                LC_COMMON_COMPONENT_FIELD: _lc_common_component(cell_id),
            },
            "resources": {"available": True},
            "provenance": {
                **_source_provenance(),
                "revision": _LEGACY_REVISION,
                "runtime_profile": {
                    "measurement": {
                        "target_runtime_seconds": 5.0,
                        "achieved_runtime_seconds": 5.0,
                        "target_runtime_achieved": True,
                        "chunk_count": 5,
                        "interrupted": False,
                    }
                },
            },
            "failure": None,
        }
    )
    return result


def _bytecode_policy() -> dict[str, object]:
    return {
        "kind": "pyamplicol-source-only-bytecode-policy-v1",
        "dont_write_bytecode": True,
        "external_pycache_prefix": True,
        "external_pycache_prefix_absent": True,
        "package_local_bytecode_eligible": False,
        "isolated_startup": True,
        "site_initialization": False,
        "python_environment_ignored_at_startup": True,
    }


def _loaded_origin_policy() -> dict[str, object]:
    observations = [
        {
            "module": "pyamplicol",
            "kind": "package-member",
            "root_index": 0,
            "path": "__init__.py",
            "size": 1,
            "sha256": "1" * 64,
        }
    ]
    return {
        "kind": "pyamplicol-loaded-module-origin-policy-v1",
        "all_loaded_origins_authenticated": True,
        "native_image_origin_bound": True,
        "loaded_bytecode_eligible": False,
        "observed_module_count": len(observations),
        "observations": observations,
        "observations_sha256": digest_json(observations),
    }


def _runtime_identity_provenance(
    identity: Mapping[str, object],
    *,
    postflight_policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    stable_identity = deepcopy(dict(identity))
    stable_policy = stable_identity.get("loaded_module_origin_policy")
    assert isinstance(stable_policy, dict)
    for field in (
        "observed_module_count",
        "observations",
        "observations_sha256",
    ):
        stable_policy.pop(field)
    stable_digest = digest_json(stable_identity)
    return {
        "runtime_identity": dict(identity),
        "runtime_identity_sha256": digest_json(identity),
        "runtime_identity_stable_sha256": stable_digest,
        "runtime_identity_postflight_stable_sha256": stable_digest,
        "runtime_identity_postflight_loaded_module_origin_policy": deepcopy(
            identity["loaded_module_origin_policy"]
            if postflight_policy is None
            else postflight_policy
        ),
        "runtime_identity_postflight_match": True,
    }


def _active_runtime() -> dict[str, object]:
    candidate = {
        "schema_version": 1,
        "version": "0.1.0.dev0+candidate.test",
        "candidate_fingerprint": "test",
        "source_revision": _REVISION,
        "source_checkout": "/repo",
        "native_build_inputs_sha256": "e" * 64,
        "publishable": False,
    }
    return {
        "package_version": candidate["version"],
        "python_package_tree": {
            "kind": "pyamplicol-python-package-tree-v2",
            "root": "/runtime/pyamplicol",
            "roots": [
                "/runtime/pyamplicol",
                "/candidate/site-packages/pyamplicol",
            ],
            "file_count": 10,
            "total_bytes": 1000,
            "sha256": "9" * 64,
            "member_set_stable": True,
            "namespace_bound_to_root_fd": True,
            "bytecode_policy": _bytecode_policy(),
        },
        "loaded_module_origin_policy": _loaded_origin_policy(),
        "candidate_build_identity": candidate,
        "candidate_build_identity_sha256": digest_json(candidate),
        "native_build_inputs_sha256": "e" * 64,
        "native_extension": {
            "path": "/runtime/_rusticol.so",
            "sha256": "f" * 64,
            "package_version": candidate["version"],
        },
        "native_target": {
            "triple": "aarch64-apple-darwin",
            "cpu_features": [],
        },
    }


def _candidate_measurement(artifact: Path) -> dict[str, object]:
    active = _active_runtime()
    capabilities = [_COLOR_CAPABILITY, _CAPABILITY]
    identity = {
        "kind": "pyamplicol-report-runtime-identity-v1",
        "artifact_id": _ARTIFACT_ID,
        "loaded_artifact_id": _ARTIFACT_ID,
        "artifact_identity_match": True,
        "process_id": "d_dbar_to_z",
        "execution_mode": "recurrence",
        "loaded_execution_mode": "recurrence",
        "backend": "jit",
        "required_arena_capability": _CAPABILITY,
        "expected_evaluator_abi": "pyamplicol-recurrence-runtime-layout-v2",
        "expected_source_evaluator_abi": ("pyamplicol-symjit-plane-application-v2"),
        "expected_source_evaluator_runtime_capability": (
            "symjit.application.complex-f64.v1"
        ),
        "source_jit_optimization_level": 2,
        "source_jit_identity": {
            "kind": "authenticated-recurrence-direct-template-source-v1",
            "optimization_level": 2,
            "direct_template_count": 4,
            "prepared_direct_template_count": 3,
            "source_evaluator_leaf_count": 5,
            "source_application_abi": ("pyamplicol-symjit-plane-application-v2"),
            "direct_application_abi": "pyamplicol-symjit-plane-application-v2",
            "prepared_kernel_pack_digest": "7" * 64,
            "direct_template_catalog_digest": "8" * 64,
            "execution_manifest_path": "processes/d_dbar_to_z/execution.json",
            "execution_manifest_sha256": "1" * 64,
            "kernel_pack_path": "model/eager-kernel-pack.json",
            "kernel_pack_sha256": "2" * 64,
        },
        "process_required_runtime_capabilities": capabilities,
        "package_version": active["package_version"],
        "python_package_tree": active["python_package_tree"],
        "loaded_module_origin_policy": active["loaded_module_origin_policy"],
        "artifact_runtime_version": active["package_version"],
        "source_revision": _REVISION,
        "candidate_build_identity": active["candidate_build_identity"],
        "candidate_build_identity_sha256": active["candidate_build_identity_sha256"],
        "native_build_inputs_sha256": active["native_build_inputs_sha256"],
        "native_extension": active["native_extension"],
        "native_target": active["native_target"],
    }
    result = empty_measurement()
    result.update(
        {
            "status": "ok",
            "generation_seconds": 2.0,
            "wall_seconds_per_point": 2.0e-6,
            "execution_seconds_per_point": 1.5e-6,
            "matrix_element": 1.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 1.0e-8,
            "relative_standard_error": 0.01,
            "artifact": {
                "path": str(artifact),
                "process_id": "d_dbar_to_z",
                "policy": "generated",
            },
            "selector_contract": _selector(),
            "validation": {
                "status": "ok",
                DIRECT_AGREEMENT_FIELD: [],
                LC_COMMON_COMPONENT_FIELD: _lc_common_component(
                    "matrix-recurrence-test-n1-dd-z-selected-flow"
                ),
                "resolved_sum": {
                    "status": "ok",
                    "maximum_absolute_difference": 0.0,
                    "maximum_relative_difference": 0.0,
                    "relative_tolerance": 1.0e-12,
                    "absolute_tolerance": 1.0e-15,
                },
                "pointwise": {
                    "status": "ok",
                    "candidate": 1.0,
                    "baseline": 1.0,
                    "absolute_difference": 0.0,
                    "relative_difference": 0.0,
                    "relative_tolerance": 1.0e-8,
                    "absolute_tolerance": 1.0e-15,
                },
            },
            "resources": {"available": True},
            "provenance": {
                **_source_provenance(),
                "source_revision": _REVISION,
                "runtime_profile": {
                    "target_runtime_seconds": 5.0,
                    "achieved_runtime_seconds": 5.0,
                    "target_runtime_achieved": True,
                    "completed_sample_count": 5,
                    "planned_sample_count": 5,
                    "repetitions_per_sample": 128,
                    "measured_point_count": 640,
                    "interrupted": False,
                },
                "effective_config": {
                    "evaluator": {
                        "execution_mode": "recurrence",
                        "backend": "jit",
                        "jit": {"optimization_level": 2},
                    },
                    "color": {
                        "accuracy": "lc",
                        "lc_flow_layout": "topology-replay",
                    },
                    "generation": {
                        "validation": {
                            "enabled": True,
                            "samples": 10,
                            "seed": 12345,
                            "relative_tolerance": 1.0e-12,
                            "absolute_tolerance": 1.0e-300,
                            "post_build_validation": True,
                        }
                    },
                },
                **_runtime_identity_provenance(identity),
            },
            "failure": None,
        }
    )
    return result


def test_class_c_runtime_projection_authenticates_relinked_endpoint(
    tmp_path: Path,
) -> None:
    active = _active_runtime()
    measurement = _candidate_measurement(tmp_path)
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    identity = deepcopy(provenance["runtime_identity"])
    assert isinstance(identity, dict)
    ancestor_revision = "b" * 40
    candidate = identity["candidate_build_identity"]
    package_tree = identity["python_package_tree"]
    native_extension = identity["native_extension"]
    assert isinstance(candidate, dict)
    assert isinstance(package_tree, dict)
    assert isinstance(native_extension, dict)
    candidate["source_revision"] = ancestor_revision
    candidate["source_checkout"] = "/ancestor/repo"
    identity["source_revision"] = ancestor_revision
    identity["candidate_build_identity_sha256"] = digest_json(candidate)
    package_tree["root"] = "/ancestor/runtime/pyamplicol"
    package_tree["roots"] = [
        "/ancestor/runtime/pyamplicol",
        "/ancestor/site-packages/pyamplicol",
    ]
    package_tree["sha256"] = "8" * 64
    native_extension["path"] = "/ancestor/runtime/_rusticol.so"
    native_extension["sha256"] = "7" * 64
    ancestor_environment = {
        "python_package_tree_sha256": package_tree["sha256"],
        "pyamplicol": identity["package_version"],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "native_build_inputs_sha256": identity["native_build_inputs_sha256"],
        "native_extension_sha256": native_extension["sha256"],
        "native_target": identity["native_target"]["triple"],
        "native_cpu_features": "baseline",
    }
    lineage = SimpleNamespace(
        ancestor_revision=ancestor_revision,
        descendant_revision=_REVISION,
        environment_for_source=lambda revision: (
            ancestor_environment if revision == ancestor_revision else None
        ),
    )
    retained_provenance = _runtime_identity_provenance(identity)

    projected = _runtime_for_measurement_source(
        active,
        retained_provenance,
        source_revision=ancestor_revision,
        measurement_lineage=lineage,
    )

    assert projected["native_extension"] == native_extension
    assert projected["native_extension"] != active["native_extension"]
    _audit_runtime_identity(
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        retained_provenance,
        expected_source_revision=ancestor_revision,
        active_runtime=projected,
        artifact=None,
    )

    tampered_environment = dict(ancestor_environment)
    tampered_environment["native_extension_sha256"] = "6" * 64
    tampered_lineage = SimpleNamespace(
        ancestor_revision=ancestor_revision,
        descendant_revision=_REVISION,
        environment_for_source=lambda revision: (
            tampered_environment if revision == ancestor_revision else None
        ),
    )
    with pytest.raises(
        FinalAuditError,
        match="retained ancestor runtime identity differs",
    ):
        _runtime_for_measurement_source(
            active,
            retained_provenance,
            source_revision=ancestor_revision,
            measurement_lineage=tampered_lineage,
        )


def test_class_c_runtime_projection_allows_only_pinned_summary_native_transition(
    tmp_path: Path,
) -> None:
    ancestor_revision = "be11d8304fdc04893dc0e23e9619be848126e3bc"
    descendant_revision = "2594d8b520b802f71d60bd646f73ebaa5547927a"
    ancestor_digest = "23b9637d5d3fba0947d78cf688df18799b0c9ee5b3bcbfa6a2963a1f1a21f870"
    descendant_digest = (
        "96e1ff79a007aaf67a0900dd6d67327ee00f6bd2cca002589b879aa3a734de08"
    )
    active = _active_runtime()
    active["native_build_inputs_sha256"] = descendant_digest
    measurement = _candidate_measurement(tmp_path)
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    identity = deepcopy(provenance["runtime_identity"])
    assert isinstance(identity, dict)
    candidate = identity["candidate_build_identity"]
    package_tree = identity["python_package_tree"]
    native_extension = identity["native_extension"]
    assert isinstance(candidate, dict)
    assert isinstance(package_tree, dict)
    assert isinstance(native_extension, dict)
    candidate["source_revision"] = ancestor_revision
    candidate["source_checkout"] = "/ancestor/repo"
    candidate["native_build_inputs_sha256"] = ancestor_digest
    identity["source_revision"] = ancestor_revision
    identity["native_build_inputs_sha256"] = ancestor_digest
    identity["candidate_build_identity_sha256"] = digest_json(candidate)
    package_tree["sha256"] = "8" * 64
    native_extension["sha256"] = "7" * 64
    ancestor_environment = {
        "python_package_tree_sha256": package_tree["sha256"],
        "pyamplicol": identity["package_version"],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "native_build_inputs_sha256": ancestor_digest,
        "native_extension_sha256": native_extension["sha256"],
        "native_target": identity["native_target"]["triple"],
        "native_cpu_features": "baseline",
    }
    lineage = SimpleNamespace(
        impact=CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
        ancestor_revision=ancestor_revision,
        descendant_revision=descendant_revision,
        environment_for_source=lambda revision: (
            ancestor_environment if revision == ancestor_revision else None
        ),
    )
    retained_provenance = _runtime_identity_provenance(identity)

    projected = _runtime_for_measurement_source(
        active,
        retained_provenance,
        source_revision=ancestor_revision,
        measurement_lineage=lineage,
    )
    assert projected["native_build_inputs_sha256"] == ancestor_digest
    assert projected["candidate_build_identity"] == candidate
    _audit_runtime_identity(
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        retained_provenance,
        expected_source_revision=ancestor_revision,
        active_runtime=projected,
        artifact=None,
    )

    changed_active = dict(active)
    changed_active["native_build_inputs_sha256"] = "0" * 64
    with pytest.raises(
        FinalAuditError,
        match="Class-C invariant native identity",
    ):
        _runtime_for_measurement_source(
            changed_active,
            retained_provenance,
            source_revision=ancestor_revision,
            measurement_lineage=lineage,
        )


def test_portable_artifact_locator_resolves_only_within_profile_root(
    tmp_path: Path,
) -> None:
    paths = ReportPaths.from_repo(
        tmp_path,
        artifact_root=tmp_path / "profile-artifacts",
    )
    artifact = paths.artifact_root / "cells/example/artifact"
    artifact.mkdir(parents=True)
    measurement = {
        "artifact": {
            "path": ("${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/cells/example/artifact"),
            "process_id": "d_dbar_to_z",
        }
    }

    reference = _artifact_reference(
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        measurement,
        report_paths=paths,
    )

    assert reference.path == artifact
    escaped = deepcopy(measurement)
    escaped["artifact"]["path"] = (  # type: ignore[index]
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/../outside"
    )
    with pytest.raises(FinalAuditError, match="not canonical"):
        _artifact_reference(
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
            escaped,
            report_paths=paths,
        )


def _otf_warmup_runtime_state(*, retained: bool) -> dict[str, object]:
    counts = {field: 0 for field in report_runner.OTF_RUNTIME_STATE_COUNT_FIELDS}
    active: dict[str, object] | None = None
    if retained:
        counts.update(
            {
                field: 1
                for field in (
                    *report_runner.OTF_RUNTIME_STATE_RETAINED_BASE_POSITIVE_FIELDS,
                    *report_runner.OTF_RUNTIME_STATE_RETAINED_EXECUTABLE_FIELDS,
                )
            }
        )
        active = {
            "basis": "shared-query-family-union-v1",
            "scope": "active-family-union",
            **{field: 1 for field in report_runner.OTF_ACTIVE_FAMILY_COUNT_FIELDS},
        }
    return {
        "kind": report_runner.OTF_RUNTIME_STATE_CENSUS_KIND,
        "process_id": "otf_warmup_process",
        "family_cache_policy": report_runner.OTF_RUNTIME_STATE_FAMILY_CACHE_POLICY,
        "family_cache_limit": report_runner.OTF_RUNTIME_STATE_FAMILY_CACHE_LIMIT,
        **counts,
        "active_family_union_census": active,
    }


def _otf_warmup_profile() -> dict[str, object]:
    return {
        "cold_warmup_elapsed_seconds": 4.767252833,
        "cold_warmup_run_count": 1,
        "cold_warmup_batch_size": 128,
        "cold_warmup_point_count": 128,
        "cold_warmup_timer_source": report_runner.WARMUP_TIMER_SOURCE,
        "cold_warmup_timing_scope": report_runner.OTF_COLD_WARMUP_TIMING_SCOPE,
        "cold_warmup_runtime_freshness": (
            report_runner.OTF_COLD_WARMUP_RUNTIME_FRESHNESS
        ),
        "cold_warmup_runtime_state_evidence": (
            report_runner.OTF_COLD_WARMUP_RUNTIME_STATE_EVIDENCE
        ),
        "cold_warmup_runtime_state_before": _otf_warmup_runtime_state(retained=False),
        "cold_warmup_runtime_state_after": _otf_warmup_runtime_state(retained=True),
        "cold_warmup_runtime_cold_before_first_evaluation": True,
        "cold_warmup_runtime_retained_before_first_evaluation": False,
        "cold_warmup_runtime_retained_after_first_evaluation": True,
        "cold_warmup_ratio_eligible": False,
        "cold_warmup_acceptance_eligible": False,
        "warmup_elapsed_seconds": 0.021,
        "warmup_configured_run_count": 2,
        "warmup_batch_size": 128,
        "warmup_point_count": 256,
        "warmup_run_outer_wall_seconds": [0.010, 0.011],
        "first_warmup_run_outer_wall_seconds": 0.010,
        "warmup_timer_source": report_runner.WARMUP_TIMER_SOURCE,
        "warmup_timing_scope": report_runner.CONVENTIONAL_WARMUP_TIMING_SCOPE,
    }


def _otf_warmup_provenance(
    profile: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "runtime_profile": _otf_warmup_profile() if profile is None else profile,
        "runtime_load_included_in_cold_warmup": False,
        "generation_timer_excludes_model_preparation": True,
        "effective_config": {"benchmark": {"batch_size": 128, "warmup_runs": 2}},
    }


def test_final_audit_accepts_bounded_otf_warmup_evidence() -> None:
    _audit_runtime_profile_warmup_evidence(
        _cell(ExecutionMode.ON_THE_FLY, optimization_level=2),
        _otf_warmup_provenance(),
        context="otf.measurement",
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("cold_warmup_run_count", 2, "exactly one run"),
        ("cold_warmup_batch_size", 64, "full benchmark batch"),
        (
            "cold_warmup_runtime_freshness",
            "fresh",
            "authenticated runtime state",
        ),
        (
            "cold_warmup_runtime_state_evidence",
            "self-reported",
            "state evidence is unsupported",
        ),
        (
            "cold_warmup_runtime_cold_before_first_evaluation",
            False,
            "booleans do not match",
        ),
        ("cold_warmup_acceptance_eligible", True, "ineligible"),
        ("warmup_configured_run_count", 1, "effective benchmark configuration"),
    ),
)
def test_final_audit_rejects_invalid_otf_warmup_evidence(
    field: str,
    replacement: object,
    message: str,
) -> None:
    profile = _otf_warmup_profile()
    profile[field] = replacement

    with pytest.raises(FinalAuditError, match=message):
        _audit_runtime_profile_warmup_evidence(
            _cell(ExecutionMode.ON_THE_FLY, optimization_level=2),
            _otf_warmup_provenance(profile),
            context="otf.measurement",
        )


def test_final_audit_rejects_missing_conventional_warmup_evidence() -> None:
    profile = _otf_warmup_profile()
    del profile["warmup_run_outer_wall_seconds"]

    with pytest.raises(FinalAuditError, match="incomplete conventional warm-up"):
        _audit_runtime_profile_warmup_evidence(
            _cell(ExecutionMode.ON_THE_FLY, optimization_level=2),
            _otf_warmup_provenance(profile),
            context="otf.measurement",
        )


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("runtime_load_included_in_cold_warmup", True, "must be false"),
        (
            "generation_timer_excludes_model_preparation",
            False,
            "must be true",
        ),
    ),
)
def test_final_audit_rejects_contradictory_otf_phase_scope_flags(
    field: str,
    invalid: bool,
    message: str,
) -> None:
    provenance = _otf_warmup_provenance()
    provenance[field] = invalid

    with pytest.raises(FinalAuditError, match=message):
        _audit_runtime_profile_warmup_evidence(
            _cell(ExecutionMode.ON_THE_FLY, optimization_level=2),
            provenance,
            context="otf.measurement",
        )


def test_final_audit_binds_otf_warmups_to_effective_benchmark_config() -> None:
    coordinated_batch = _otf_warmup_profile()
    coordinated_batch["cold_warmup_batch_size"] = 64
    coordinated_batch["cold_warmup_point_count"] = 64
    coordinated_batch["warmup_batch_size"] = 64
    coordinated_batch["warmup_point_count"] = 128
    with pytest.raises(FinalAuditError, match="effective benchmark configuration"):
        _audit_runtime_profile_warmup_evidence(
            _cell(ExecutionMode.ON_THE_FLY, optimization_level=2),
            _otf_warmup_provenance(coordinated_batch),
            context="otf.measurement",
        )

    coordinated_runs = _otf_warmup_profile()
    coordinated_runs["warmup_configured_run_count"] = 1
    coordinated_runs["warmup_point_count"] = 128
    coordinated_runs["warmup_run_outer_wall_seconds"] = [0.010]
    coordinated_runs["first_warmup_run_outer_wall_seconds"] = 0.010
    coordinated_runs["warmup_elapsed_seconds"] = 0.010
    with pytest.raises(FinalAuditError, match="effective benchmark configuration"):
        _audit_runtime_profile_warmup_evidence(
            _cell(ExecutionMode.ON_THE_FLY, optimization_level=2),
            _otf_warmup_provenance(coordinated_runs),
            context="otf.measurement",
        )


def test_final_audit_rejects_coordinated_nonpublication_otf_benchmark() -> None:
    coordinated_batch = _otf_warmup_profile()
    coordinated_batch["cold_warmup_batch_size"] = 64
    coordinated_batch["cold_warmup_point_count"] = 64
    coordinated_batch["warmup_batch_size"] = 64
    coordinated_batch["warmup_point_count"] = 128
    batch_provenance = _otf_warmup_provenance(coordinated_batch)
    batch_provenance["effective_config"]["benchmark"]["batch_size"] = 64
    with pytest.raises(
        FinalAuditError,
        match=r"batch_size must equal the final report contract 128",
    ):
        _audit_runtime_profile_warmup_evidence(
            _cell(ExecutionMode.ON_THE_FLY, optimization_level=2),
            batch_provenance,
            context="otf.measurement",
        )

    coordinated_runs = _otf_warmup_profile()
    coordinated_runs["warmup_configured_run_count"] = 1
    coordinated_runs["warmup_point_count"] = 128
    coordinated_runs["warmup_run_outer_wall_seconds"] = [0.010]
    coordinated_runs["first_warmup_run_outer_wall_seconds"] = 0.010
    coordinated_runs["warmup_elapsed_seconds"] = 0.010
    run_provenance = _otf_warmup_provenance(coordinated_runs)
    run_provenance["effective_config"]["benchmark"]["warmup_runs"] = 1
    with pytest.raises(
        FinalAuditError,
        match=r"warmup_runs must equal the final report contract 2",
    ):
        _audit_runtime_profile_warmup_evidence(
            _cell(ExecutionMode.ON_THE_FLY, optimization_level=2),
            run_provenance,
            context="otf.measurement",
        )


def test_legacy_measurement_requires_pinned_revision_and_physical_point_digest() -> (
    None
):
    cell = _legacy_cell()
    measurement = _baseline_measurement()
    assert (
        _audit_measurement(
            cell,
            measurement,
            baseline=None,
            expected_source_revision=_REVISION,
            expected_legacy_revision=_LEGACY_REVISION,
            active_runtime=None,
        )
        is None
    )

    wrong_revision = deepcopy(measurement)
    wrong_revision["provenance"]["revision"] = "d" * 40  # type: ignore[index]
    with pytest.raises(FinalAuditError, match="pinned original-AmpliCol revision"):
        _audit_measurement(
            cell,
            wrong_revision,
            baseline=None,
            expected_source_revision=_REVISION,
            expected_legacy_revision=_LEGACY_REVISION,
            active_runtime=None,
        )

    wrong_points = deepcopy(measurement)
    wrong_points["validation"]["point_digest"] = "0" * 64  # type: ignore[index]
    with pytest.raises(FinalAuditError, match="physical process"):
        _audit_measurement(
            cell,
            wrong_points,
            baseline=None,
            expected_source_revision=_REVISION,
            expected_legacy_revision=_LEGACY_REVISION,
            active_runtime=None,
        )


@pytest.mark.parametrize("execution_seconds", [0.0, -1.0, float("inf")])
def test_candidate_measurement_requires_positive_finite_execution_time(
    tmp_path: Path,
    execution_seconds: float,
) -> None:
    measurement = _candidate_measurement(tmp_path)
    measurement["execution_seconds_per_point"] = execution_seconds
    with pytest.raises(
        FinalAuditError,
        match="execution_seconds_per_point",
    ):
        _audit_measurement(
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
            measurement,
            baseline=_baseline_measurement(),
            expected_source_revision=_REVISION,
            expected_legacy_revision=_LEGACY_REVISION,
            active_runtime=_active_runtime(),
        )


def test_recurrence_without_legacy_oracle_requires_high_precision_validation(
    tmp_path: Path,
) -> None:
    cell = replace(
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        dataset_id="matrix_recurrence_builtin_sm_lc",
        process="d d~ > u u~ s s~ c c~",
        n_final=6,
        process_key="dd_4q_lines",
    )
    measurement = _candidate_measurement(tmp_path)
    validation = measurement["validation"]
    assert isinstance(validation, dict)
    validation.pop("pointwise")
    validation["high_precision"] = {
        "status": "ok",
        "candidate": 1.0,
        "baseline": 1.0,
        "absolute_difference": 0.0,
        "relative_difference": 0.0,
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance": 1.0e-15,
    }

    reference = _audit_measurement(
        cell,
        measurement,
        baseline=None,
        expected_source_revision=_REVISION,
        expected_legacy_revision=_LEGACY_REVISION,
        active_runtime=_active_runtime(),
    )

    assert reference is not None
    assert reference.cell == cell

    validation.pop("high_precision")
    with pytest.raises(FinalAuditError, match="high_precision"):
        _audit_measurement(
            cell,
            measurement,
            baseline=None,
            expected_source_revision=_REVISION,
            expected_legacy_revision=_LEGACY_REVISION,
            active_runtime=_active_runtime(),
        )


@pytest.mark.parametrize(
    "baseline_state",
    tuple(
        state
        for state in PolicyMeasurementState
        if state is not PolicyMeasurementState.SUCCESS
    ),
)
def test_final_audit_omits_only_non_successful_optional_legacy_baseline(
    tmp_path: Path,
    baseline_state: PolicyMeasurementState,
) -> None:
    baseline = _legacy_cell()
    candidate = replace(
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        dataset_id="matrix_recurrence_builtin_sm_lc",
    )
    catalog = _Catalog(baseline, candidate)
    baseline_measurement = _baseline_measurement()
    candidate_measurement = _candidate_measurement(tmp_path)

    assert (
        _validation_baseline_for_audit(
            candidate,
            {
                baseline.cell_id: baseline_measurement,
                candidate.cell_id: candidate_measurement,
            },
            {baseline.cell_id: baseline_state},
            catalog=catalog,  # type: ignore[arg-type]
        )
        is None
    )
    assert (
        _validation_baseline_for_audit(
            candidate,
            {candidate.cell_id: candidate_measurement},
            {},
            catalog=catalog,  # type: ignore[arg-type]
        )
        is None
    )


def test_final_audit_consumes_successful_optional_legacy_baseline(
    tmp_path: Path,
) -> None:
    baseline = _legacy_cell()
    candidate = replace(
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        dataset_id="matrix_recurrence_builtin_sm_lc",
    )
    catalog = _Catalog(baseline, candidate)
    baseline_measurement = _baseline_measurement()
    candidate_measurement = _candidate_measurement(tmp_path)

    assert (
        _validation_baseline_for_audit(
            candidate,
            {
                baseline.cell_id: baseline_measurement,
                candidate.cell_id: candidate_measurement,
            },
            {baseline.cell_id: PolicyMeasurementState.SUCCESS},
            catalog=catalog,  # type: ignore[arg-type]
        )
        is baseline_measurement
    )
    with pytest.raises(FinalAuditError, match="successful optional baseline"):
        _validation_baseline_for_audit(
            candidate,
            {candidate.cell_id: candidate_measurement},
            {baseline.cell_id: PolicyMeasurementState.SUCCESS},
            catalog=catalog,  # type: ignore[arg-type]
        )


def test_optional_endpoint_transition_preserves_stored_authority_route(
    tmp_path: Path,
) -> None:
    baseline = _legacy_cell()
    candidate = replace(
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        dataset_id="matrix_recurrence_builtin_sm_lc",
    )
    catalog = _Catalog(baseline, candidate)
    baseline_measurement = _baseline_measurement()
    pointwise_measurement = _candidate_measurement(tmp_path)

    assert (
        _validation_baseline_for_audit(
            candidate,
            {
                baseline.cell_id: baseline_measurement,
                candidate.cell_id: pointwise_measurement,
            },
            {baseline.cell_id: PolicyMeasurementState.GENERATION_LIMIT},
            catalog=catalog,  # type: ignore[arg-type]
        )
        is None
    )
    assert (
        _audit_measurement(
            candidate,
            pointwise_measurement,
            baseline=None,
            expected_source_revision=_REVISION,
            expected_legacy_revision=_LEGACY_REVISION,
            active_runtime=_active_runtime(),
        )
        is not None
    )
    high_precision_measurement = deepcopy(pointwise_measurement)
    validation = high_precision_measurement["validation"]
    assert isinstance(validation, dict)
    pointwise = validation.pop("pointwise")
    assert isinstance(pointwise, dict)
    pointwise["relative_tolerance"] = final_audit_module.RELATIVE_TOLERANCE
    validation["high_precision"] = pointwise
    assert (
        _validation_baseline_for_audit(
            candidate,
            {
                baseline.cell_id: baseline_measurement,
                candidate.cell_id: high_precision_measurement,
            },
            {baseline.cell_id: PolicyMeasurementState.SUCCESS},
            catalog=catalog,  # type: ignore[arg-type]
        )
        is None
    )
    assert (
        _audit_measurement(
            candidate,
            high_precision_measurement,
            baseline=None,
            expected_source_revision=_REVISION,
            expected_legacy_revision=_LEGACY_REVISION,
            active_runtime=_active_runtime(),
        )
        is not None
    )
    assert (
        _legacy_consumer_coverage(
            (baseline, candidate),
            {
                baseline.cell_id: baseline_measurement,
                candidate.cell_id: high_precision_measurement,
            },
            {
                baseline.cell_id: PolicyMeasurementState.SUCCESS,
                candidate.cell_id: PolicyMeasurementState.SUCCESS,
            },
            catalog=catalog,  # type: ignore[arg-type]
        )
        == ()
    )


def test_stored_optional_legacy_agreement_survives_missing_current_endpoint(
    tmp_path: Path,
) -> None:
    legacy_selected = _legacy_cell()
    baseline = replace(legacy_selected, workload=Workload.ALL_FLOW)
    candidate = replace(
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        dataset_id="matrix_recurrence_builtin_sm_lc",
        workload=Workload.ALL_FLOW,
    )
    selected = replace(candidate, workload=Workload.SELECTED_FLOW)

    class DirectCatalog(_Catalog):
        def measurement_cells(self) -> tuple[CellSpec, ...]:
            return (legacy_selected, self.baseline, selected, self.candidate)

    catalog = DirectCatalog(baseline, candidate)
    baseline_measurement = _baseline_measurement(baseline.cell_id)
    selected_measurement = _candidate_measurement(tmp_path / "selected")
    selected_validation = selected_measurement["validation"]
    assert isinstance(selected_validation, dict)
    selected_validation[LC_COMMON_COMPONENT_FIELD] = _lc_common_component(
        selected.cell_id
    )
    candidate_measurement = _candidate_measurement(tmp_path)
    validation = candidate_measurement["validation"]
    assert isinstance(validation, dict)
    validation[LC_COMMON_COMPONENT_FIELD] = _lc_common_component(candidate.cell_id)
    attach_direct_agreements(
        candidate,
        candidate_measurement,
        {
            baseline.cell_id: baseline_measurement,
            selected.cell_id: selected_measurement,
        },
        catalog=catalog,  # type: ignore[arg-type]
    )

    edges, counts = _audit_direct_agreements(
        (selected, candidate),
        {
            selected.cell_id: selected_measurement,
            candidate.cell_id: candidate_measurement,
        },
        catalog=catalog,  # type: ignore[arg-type]
    )

    assert len(edges) == 2
    assert counts[LC_LEGACY_PYAMPLICOL_COMPONENT] == 1
    replay_counts = _audit_replayed_direct_agreements(
        edges,
        {
            selected.cell_id: selected_measurement,
            candidate.cell_id: candidate_measurement,
        },
        {
            selected.cell_id: final_audit_module._ReplayObservation(
                matrix_element=1.0,
                resolved_maximum_absolute=0.0,
                resolved_maximum_relative=0.0,
                lc_common_component=1.0,
            ),
            candidate.cell_id: final_audit_module._ReplayObservation(
                matrix_element=1.0,
                resolved_maximum_absolute=0.0,
                resolved_maximum_relative=0.0,
                lc_common_component=1.0,
            ),
        },
    )
    assert (
        replay_counts[final_audit_module._REPLAYED_PYAMPLICOL_AUTHENTICATED_LEGACY] == 1
    )


@pytest.mark.parametrize(
    "execution_mode",
    (ExecutionMode.COMPILED, ExecutionMode.EAGER),
)
def test_final_audit_treats_terminal_recurrence_authority_as_unavailable(
    tmp_path: Path,
    execution_mode: ExecutionMode,
) -> None:
    recurrence = _cell(ExecutionMode.RECURRENCE, optimization_level=2)
    candidate = _cell(execution_mode, optimization_level=2)
    catalog = _Catalog(recurrence, candidate)
    candidate_measurement = _candidate_measurement(tmp_path)

    with pytest.raises(
        FinalAuditError,
        match="no successful authenticated numerical authority",
    ):
        _validation_baseline_for_audit(
            candidate,
            {
                recurrence.cell_id: {},
                candidate.cell_id: candidate_measurement,
            },
            {recurrence.cell_id: PolicyMeasurementState.GENERATION_LIMIT},
            catalog=catalog,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "execution_mode",
    (ExecutionMode.COMPILED, ExecutionMode.EAGER),
)
def test_final_audit_consumes_durable_selected_independent_authority(
    tmp_path: Path,
    execution_mode: ExecutionMode,
) -> None:
    candidate = REPORT_CATALOG.cell(
        f"matrix-{execution_mode.value}-builtin-sm-lc-n1-dd-z-jets-selected-flow"
    )
    authorities = independent_numerical_authorities(candidate)
    recurrence, legacy = authorities
    candidate_measurement = _candidate_measurement(tmp_path / "candidate")
    validation = candidate_measurement["validation"]
    assert isinstance(validation, dict)
    pointwise = validation["pointwise"]
    assert isinstance(pointwise, dict)
    pointwise["relative_tolerance"] = final_audit_module.RELATIVE_TOLERANCE
    validation[INDEPENDENT_AUTHORITY_FIELD] = {
        "abi": INDEPENDENT_AUTHORITY_ABI,
        "expected_cell_ids": [authority.cell_id for authority in authorities],
        "selected_cell_id": legacy.cell_id,
        "status": "verified",
        "reason": "independent-authority-agreement",
        "same_artifact_diagnostics_are_authority": False,
    }
    legacy_measurement = _baseline_measurement(legacy.cell_id)

    assert (
        _validation_baseline_for_audit(
            candidate,
            {
                candidate.cell_id: candidate_measurement,
                legacy.cell_id: legacy_measurement,
            },
            {
                recurrence.cell_id: PolicyMeasurementState.GENERATION_LIMIT,
                legacy.cell_id: PolicyMeasurementState.SUCCESS,
            },
            catalog=REPORT_CATALOG,
        )
        is legacy_measurement
    )

    with pytest.raises(FinalAuditError, match="higher-priority authority"):
        _validation_baseline_for_audit(
            candidate,
            {
                candidate.cell_id: candidate_measurement,
                recurrence.cell_id: _candidate_measurement(tmp_path / "recurrence"),
                legacy.cell_id: legacy_measurement,
            },
            {
                recurrence.cell_id: PolicyMeasurementState.SUCCESS,
                legacy.cell_id: PolicyMeasurementState.SUCCESS,
            },
            catalog=REPORT_CATALOG,
        )


def _arena_profile_evidence(
    *,
    execution_mode: ExecutionMode = ExecutionMode.COMPILED,
) -> dict[str, object]:
    raw_profile = {
        "execution_mode": execution_mode.value,
        "profile_boundary": ARENA_PROFILE_BOUNDARY,
        "borrowed_flat_input": True,
        "preallocated_output": True,
        "phase_timing_scope": ARENA_PHASE_TIMING_SCOPE,
        "evaluator_timing_available": False,
        "points": 128,
        "wall_time_s": 128 * 1.1e-6,
        "orchestration_time_s": 128 * 1.1e-6,
        **{field: 0 for field in ZERO_ARENA_COUNTER_FIELDS},
        **{field: 0.0 for field in ZERO_ARENA_PHASE_TIME_FIELDS},
        **{field: [] for field in EMPTY_ARENA_PHASE_VECTOR_FIELDS},
        **{field: 0 for field in ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS},
    }
    if execution_mode is ExecutionMode.COMPILED:
        raw_profile.update(
            {
                "compiled_direct_arena_engine_count": 1,
                "compiled_direct_arena_call_count": 128,
                "evaluator_backend_call_count": 128,
            }
        )
    return build_arena_profile_evidence(
        [raw_profile] * 5,
        execution_mode=execution_mode.value,
        repetitions_per_profile=1,
        batch_size=128,
    )


def _arena_unavailable_execution_timing(
    *,
    execution_mode: ExecutionMode = ExecutionMode.COMPILED,
    arena_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    evidence = (
        _arena_profile_evidence(execution_mode=execution_mode)
        if arena_evidence is None
        else arena_evidence
    )
    return {
        "abi": "pyamplicol-report-arena-execution-timing-v2",
        "status": "unavailable",
        "ratio_eligible": False,
        "raw_seconds_per_point": None,
        "sample_count": 5,
        "native_profile_points_per_sample": 128,
        "repetitions_per_sample": 1,
        "batch_size": 128,
        "sample_contract": "paired_unprofiled_headline_profiled_attribution_v1",
        "profile_protocol": "arena",
        "profile_sample_pass": "runtime._profile_arena_repeated",
        "profile_boundary": (
            "warmed-direct-arena-borrowed-input-preallocated-output-v1"
        ),
        "borrowed_flat_input": True,
        "preallocated_output": True,
        "phase_timing_scope": "coarse-arena-boundary-only-v1",
        "evaluator_timing_available": False,
        "paired_with_headline": True,
        "identical_batch": True,
        "identical_repetitions": True,
        "execution_mode": execution_mode.value,
        "warmed_boundary_wall_seconds_per_point": 1.1e-6,
        "arena_profile_evidence_sha256": digest_arena_profile_value(evidence),
    }


def _arena_unavailable_provenance(
    *,
    execution_mode: ExecutionMode = ExecutionMode.COMPILED,
) -> dict[str, object]:
    evidence = _arena_profile_evidence(execution_mode=execution_mode)
    return {
        "arena_profile_evidence": evidence,
        "execution_timing": _arena_unavailable_execution_timing(
            execution_mode=execution_mode,
            arena_evidence=evidence,
        ),
    }


@pytest.mark.parametrize(
    "cell",
    (
        _cell(ExecutionMode.EAGER, optimization_level=2),
        _cell(ExecutionMode.COMPILED, optimization_level=1),
        _cell(ExecutionMode.COMPILED, backend="cpp", optimization_level=None),
    ),
)
def test_final_audit_accepts_authenticated_arena_unavailable_timing(
    cell: CellSpec,
) -> None:
    provenance = _arena_unavailable_provenance(
        execution_mode=cell.measurement.execution_mode,
    )
    _audit_unavailable_execution_timing(
        cell,
        provenance,
        measurement_sample_count=5,
        context="candidate",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("abi", "wrong"),
        ("status", "measured"),
        ("ratio_eligible", True),
        ("raw_seconds_per_point", 0.0),
        ("sample_count", 4),
        ("sample_count", True),
        ("native_profile_points_per_sample", 0),
        ("native_profile_points_per_sample", None),
        ("repetitions_per_sample", 2),
        ("batch_size", 64),
        ("sample_contract", "separate_native_profile_diagnostic_v1"),
        ("profile_protocol", "frozen-pre-arena"),
        ("profile_sample_pass", "runtime.profile_repeated"),
        ("profile_boundary", "materialized-native-profile-v1"),
        ("borrowed_flat_input", False),
        ("preallocated_output", False),
        ("phase_timing_scope", "profiled-evaluator-phases-v1"),
        ("evaluator_timing_available", True),
        ("paired_with_headline", False),
        ("identical_batch", False),
        ("identical_repetitions", False),
        ("execution_mode", "eager"),
        ("warmed_boundary_wall_seconds_per_point", 0.0),
        ("warmed_boundary_wall_seconds_per_point", float("nan")),
        ("arena_profile_evidence_sha256", "0" * 64),
    ),
)
def test_final_audit_rejects_tampered_arena_unavailable_timing(
    field: str,
    value: object,
) -> None:
    provenance = _arena_unavailable_provenance()
    timing = provenance["execution_timing"]
    assert isinstance(timing, dict)
    timing[field] = value
    with pytest.raises(FinalAuditError, match="unavailable-attribution record"):
        _audit_unavailable_execution_timing(
            _cell(ExecutionMode.COMPILED, optimization_level=3),
            provenance,
            measurement_sample_count=5,
            context="candidate",
        )


def test_final_audit_binds_arena_unavailable_shape_and_sample_count() -> None:
    provenance = _arena_unavailable_provenance()
    timing = provenance["execution_timing"]
    assert isinstance(timing, dict)
    timing["unexpected"] = True
    with pytest.raises(FinalAuditError, match="fields do not match"):
        _audit_unavailable_execution_timing(
            _cell(ExecutionMode.COMPILED, optimization_level=3),
            provenance,
            measurement_sample_count=5,
            context="candidate",
        )

    with pytest.raises(FinalAuditError, match="unavailable-attribution record"):
        _audit_unavailable_execution_timing(
            _cell(ExecutionMode.COMPILED, optimization_level=3),
            _arena_unavailable_provenance(),
            measurement_sample_count=6,
            context="candidate",
        )


def test_runtime_identity_audit_distinguishes_source_and_direct_codegen_levels(
    tmp_path: Path,
) -> None:
    measurement = _candidate_measurement(tmp_path)
    raw_provenance = measurement["provenance"]
    assert isinstance(raw_provenance, dict)
    raw_identity = raw_provenance["runtime_identity"]
    assert isinstance(raw_identity, dict)

    def provenance(identity: dict[str, object]) -> dict[str, object]:
        return _runtime_identity_provenance(identity)

    recurrence = _cell(ExecutionMode.RECURRENCE, optimization_level=2)
    _audit_runtime_identity(
        recurrence,
        provenance(raw_identity),
        expected_source_revision=_REVISION,
        active_runtime=_active_runtime(),
        artifact=None,
    )

    missing_source = deepcopy(raw_identity)
    del missing_source["source_jit_optimization_level"]
    with pytest.raises(FinalAuditError, match="source_jit_optimization_level"):
        _audit_runtime_identity(
            recurrence,
            provenance(missing_source),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )

    missing_source_identity = deepcopy(raw_identity)
    del missing_source_identity["source_jit_identity"]
    with pytest.raises(FinalAuditError, match="source_jit_identity"):
        _audit_runtime_identity(
            recurrence,
            provenance(missing_source_identity),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )

    wrong_source_capability = deepcopy(raw_identity)
    wrong_source_capability["expected_source_evaluator_runtime_capability"] = (
        "symbolica.compiled-cpp.complex-f64.v1"
    )
    with pytest.raises(
        FinalAuditError,
        match="expected_source_evaluator_runtime_capability",
    ):
        _audit_runtime_identity(
            recurrence,
            provenance(wrong_source_capability),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )

    missing_loaded_mode = deepcopy(raw_identity)
    del missing_loaded_mode["loaded_execution_mode"]
    with pytest.raises(FinalAuditError, match="loaded_execution_mode"):
        _audit_runtime_identity(
            recurrence,
            provenance(missing_loaded_mode),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )

    spurious_direct_codegen = deepcopy(raw_identity)
    spurious_direct_codegen["direct_codegen_optimization_level"] = 3
    with pytest.raises(FinalAuditError, match="direct_codegen_optimization_level"):
        _audit_runtime_identity(
            recurrence,
            provenance(spurious_direct_codegen),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )

    compiled_identity = deepcopy(raw_identity)
    del compiled_identity["source_jit_identity"]
    compiled_identity.update(
        {
            "execution_mode": "compiled",
            "loaded_execution_mode": "compiled",
            "required_arena_capability": "compiled-plane-arena-v1",
            "expected_evaluator_abi": "pyamplicol-compiled-plane-kernel-v2",
            "expected_source_evaluator_abi": ("pyamplicol-symjit-plane-application-v2"),
            "source_jit_optimization_level": 1,
            "direct_codegen_optimization_level": 1,
            "direct_codegen_identity": {
                "kind": "authenticated-compiled-plane-arena-direct-codegen-v1",
                "optimization_level": 1,
                "source_optimization_level": 1,
                "leaf_count": 2,
                "execution_manifest_path": "execution.json",
                "execution_manifest_sha256": "8" * 64,
            },
        }
    )
    compiled = _cell(ExecutionMode.COMPILED, optimization_level=1)
    _audit_runtime_identity(
        compiled,
        provenance(compiled_identity),
        expected_source_revision=_REVISION,
        active_runtime=_active_runtime(),
        artifact=None,
    )
    compiled_effective = {"evaluator": {"execution_mode": "compiled"}}
    compiled_evidence = ArtifactEvidence(
        artifact_id=_ARTIFACT_ID,
        process_id="d_dbar_to_z",
        runtime_version=str(_active_runtime()["package_version"]),
        runtime_capabilities=(_COLOR_CAPABILITY, _CAPABILITY),
        execution_manifest_path="execution.json",
        execution_manifest_sha256="8" * 64,
        execution_mode="compiled",
        arena_record_count=2,
        direct_leaf_count=2,
        effective_config=compiled_effective,
        source_jit_identity=None,
    )
    compiled_provenance = provenance(compiled_identity)
    compiled_provenance["effective_config"] = compiled_effective
    _audit_runtime_identity(
        compiled,
        compiled_provenance,
        expected_source_revision=_REVISION,
        active_runtime=_active_runtime(),
        artifact=compiled_evidence,
    )

    wrong_leaf_count = deepcopy(compiled_identity)
    wrong_leaf_count["direct_codegen_identity"]["leaf_count"] = 3  # type: ignore[index]
    wrong_leaf_provenance = provenance(wrong_leaf_count)
    wrong_leaf_provenance["effective_config"] = compiled_effective
    with pytest.raises(FinalAuditError, match=r"direct_codegen_identity\.leaf_count"):
        _audit_runtime_identity(
            compiled,
            wrong_leaf_provenance,
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=compiled_evidence,
        )

    wrong_path = deepcopy(compiled_identity)
    wrong_path["direct_codegen_identity"]["execution_manifest_path"] = (  # type: ignore[index]
        "other.json"
    )
    wrong_path_provenance = provenance(wrong_path)
    wrong_path_provenance["effective_config"] = compiled_effective
    with pytest.raises(
        FinalAuditError,
        match=r"direct_codegen_identity\.execution_manifest_path",
    ):
        _audit_runtime_identity(
            compiled,
            wrong_path_provenance,
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=compiled_evidence,
        )

    wrong_effective_provenance = provenance(compiled_identity)
    wrong_effective_provenance["effective_config"] = {
        "evaluator": {"execution_mode": "eager"}
    }
    with pytest.raises(FinalAuditError, match="effective_config"):
        _audit_runtime_identity(
            compiled,
            wrong_effective_provenance,
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=compiled_evidence,
        )

    missing_direct_codegen = deepcopy(compiled_identity)
    del missing_direct_codegen["direct_codegen_optimization_level"]
    with pytest.raises(FinalAuditError, match="direct_codegen_optimization_level"):
        _audit_runtime_identity(
            compiled,
            provenance(missing_direct_codegen),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )


def test_runtime_identity_audit_accepts_real_otf_contract_and_rejects_seed_alias(
    tmp_path: Path,
) -> None:
    measurement = _candidate_measurement(tmp_path)
    raw_provenance = measurement["provenance"]
    assert isinstance(raw_provenance, dict)
    raw_identity = raw_provenance["runtime_identity"]
    assert isinstance(raw_identity, dict)
    identity = deepcopy(raw_identity)
    identity.pop("source_jit_identity")
    identity.update(
        {
            "execution_mode": "on-the-fly",
            "loaded_execution_mode": "on-the-fly",
            "required_arena_capability": "rusticol.on-the-fly.complex-f64.v1",
            "expected_evaluator_abi": "pyamplicol-runtime-on-the-fly-execution",
            "expected_source_evaluator_abi": ("pyamplicol-symjit-plane-application-v2"),
            "expected_source_evaluator_runtime_capability": (
                "symjit.application.complex-f64.v1"
            ),
            "source_jit_optimization_level": 2,
            "process_required_runtime_capabilities": [
                "rusticol.on-the-fly.complex-f64.v1",
                "rusticol.on-the-fly.lc-color.v1",
            ],
        }
    )
    cell = _cell(ExecutionMode.ON_THE_FLY, optimization_level=2)

    _audit_runtime_identity(
        cell,
        _runtime_identity_provenance(identity),
        expected_source_revision=_REVISION,
        active_runtime=_active_runtime(),
        artifact=None,
    )

    invented_seed_alias = deepcopy(identity)
    invented_seed_alias["expected_source_evaluator_abi"] = (
        "pyamplicol-on-the-fly-process-seed-v1"
    )
    with pytest.raises(FinalAuditError, match="expected_source_evaluator_abi"):
        _audit_runtime_identity(
            cell,
            _runtime_identity_provenance(invented_seed_alias),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )


def test_final_audit_rejects_tampered_raw_arena_profile_evidence() -> None:
    provenance = _arena_unavailable_provenance()
    evidence = provenance["arena_profile_evidence"]
    assert isinstance(evidence, dict)
    raw_profiles = evidence["raw_profiles"]
    assert isinstance(raw_profiles, list)
    assert isinstance(raw_profiles[0], dict)
    raw_profiles[0]["native_input_pack_bytes"] = 1

    with pytest.raises(
        FinalAuditError,
        match="not an authenticated Arena",
    ):
        _audit_unavailable_execution_timing(
            _cell(ExecutionMode.COMPILED, optimization_level=1),
            provenance,
            measurement_sample_count=5,
            context="candidate",
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "runtime_identity_stable_sha256",
            "0" * 64,
            "runtime_identity_stable_sha256",
        ),
        (
            "runtime_identity_postflight_stable_sha256",
            "0" * 64,
            "postflight stable SHA-256",
        ),
        (
            "runtime_identity_postflight_match",
            False,
            "runtime_identity_postflight_match",
        ),
        (
            "runtime_identity_postflight_loaded_module_origin_policy",
            {},
            "loaded-origin policy",
        ),
    ],
)
def test_runtime_identity_audit_requires_per_cell_postflight_evidence(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    measurement = _candidate_measurement(tmp_path)
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    provenance[field] = value
    with pytest.raises(FinalAuditError, match=match):
        _audit_runtime_identity(
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
            provenance,
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )


def test_runtime_identity_audit_accepts_only_monotonic_postflight_origins(
    tmp_path: Path,
) -> None:
    measurement = _candidate_measurement(tmp_path)
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    raw_postflight = provenance[
        "runtime_identity_postflight_loaded_module_origin_policy"
    ]
    assert isinstance(raw_postflight, dict)
    observations = raw_postflight["observations"]
    assert isinstance(observations, list)
    observations.append(
        {
            "module": "pyamplicol.runtime",
            "kind": "package-member",
            "root_index": 0,
            "path": "runtime/__init__.py",
            "size": 2,
            "sha256": "2" * 64,
        }
    )
    raw_postflight["observed_module_count"] = len(observations)
    raw_postflight["observations_sha256"] = digest_json(observations)
    _audit_runtime_identity(
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        provenance,
        expected_source_revision=_REVISION,
        active_runtime=_active_runtime(),
        artifact=None,
    )

    observations[0] = {
        "module": "pyamplicol.replaced",
        "kind": "package-member",
        "root_index": 0,
        "path": "replaced.py",
        "size": 3,
        "sha256": "3" * 64,
    }
    raw_postflight["observations_sha256"] = digest_json(observations)
    with pytest.raises(FinalAuditError, match="lost a loaded-module origin"):
        _audit_runtime_identity(
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
            provenance,
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )


def test_final_audit_authenticates_cache_store_and_replays_unique_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    (dependencies / "contributor-lock.toml").write_text(
        f'[legacy_amplicol]\nrevision = "{_LEGACY_REVISION}"\n',
        encoding="utf-8",
    )
    baseline = _legacy_cell()
    candidate = CellSpec(
        dataset_id="matrix_recurrence_test",
        process="d d~ > z",
        n_final=1,
        process_key="dd_z",
        measurement=MeasurementSpec(
            ExecutionMode.RECURRENCE,
            ModelKey.BUILTIN_SM,
            Accuracy.LC,
            "jit",
            2,
        ),
        workload=Workload.SELECTED_FLOW,
    )
    catalog = _Catalog(baseline, candidate)
    paths = ReportPaths.from_repo(
        tmp_path,
        artifact_root=tmp_path / "artifacts",
        coordination_root=tmp_path / "locks",
    )
    service = ReportService(paths, catalog=catalog)  # type: ignore[arg-type]
    artifact = paths.artifact_root / "synthetic-artifact"
    artifact.mkdir()
    measurements = {
        baseline.cell_id: _baseline_measurement(),
        candidate.cell_id: _candidate_measurement(artifact),
    }
    for cell in catalog.measurement_cells():
        service.store.new_attempt(cell.cell_id, ArtifactPolicy.REGENERATE).publish(
            measurements[cell.cell_id]
        )
    caches = build_reset_caches(catalog)  # type: ignore[arg-type]
    for payload in caches.values():
        for entry in payload["entries"]:
            entry["measurement"] = portable_publication_value(
                measurements[entry["cell_id"]],
                paths,
            )
    paths.results_dir.mkdir(parents=True)
    for name, payload in caches.items():
        (paths.results_dir / name).write_text(
            json.dumps(payload, sort_keys=True),
            encoding="ascii",
        )
    lock_state = {"held": False, "entries": 0}

    @contextmanager
    def named_lock(name: str):
        assert name == "report-writer"
        assert lock_state["held"] is False
        lock_state["held"] = True
        lock_state["entries"] += 1
        try:
            yield
        finally:
            lock_state["held"] = False

    def assert_locked() -> None:
        assert lock_state["held"] is True

    def render_auditor() -> dict[str, object]:
        assert_locked()
        return {"cache_render_match": True, "table_count": 1}

    monkeypatch.setattr(service.store, "named_lock", named_lock)
    monkeypatch.setattr(service, "audit", render_auditor)

    def artifact_auditor(
        _cell: CellSpec,
        path: Path,
        process_id: str,
    ) -> ArtifactEvidence:
        assert_locked()
        assert path == artifact
        candidate_provenance = measurements[candidate.cell_id]["provenance"]
        assert isinstance(candidate_provenance, dict)
        effective_config = candidate_provenance["effective_config"]
        assert isinstance(effective_config, dict)
        runtime_identity = candidate_provenance["runtime_identity"]
        assert isinstance(runtime_identity, dict)
        source_jit_identity = runtime_identity["source_jit_identity"]
        assert isinstance(source_jit_identity, dict)
        return ArtifactEvidence(
            artifact_id=_ARTIFACT_ID,
            process_id=process_id,
            runtime_version=str(_active_runtime()["package_version"]),
            runtime_capabilities=(_COLOR_CAPABILITY, _CAPABILITY),
            execution_manifest_path="processes/d_dbar_to_z/execution.json",
            execution_manifest_sha256="1" * 64,
            execution_mode="recurrence",
            arena_record_count=4,
            direct_leaf_count=0,
            effective_config=effective_config,
            source_jit_identity=source_jit_identity,
        )

    def runtime_loader(_path: Path, _process: str) -> _Runtime:
        assert_locked()
        return _Runtime()

    def source_auditor(_root: Path, _revision: str) -> None:
        assert_locked()

    def runtime_auditor(_revision: str, _root: Path) -> dict[str, object]:
        assert_locked()
        return _active_runtime()

    def pdf_auditor(_service: ReportService) -> dict[str, object]:
        assert_locked()
        return {
            "status": "ok",
            "published_sha256": "2" * 64,
        }

    result = audit_final_report(
        tmp_path,
        expected_source_revision=_REVISION,
        max_n_final=1,
        expected_cell_count=2,
        replay=True,
        catalog=catalog,  # type: ignore[arg-type]
        service=service,
        active_runtime=_active_runtime(),
        artifact_auditor=artifact_auditor,
        runtime_loader=runtime_loader,
        source_auditor=source_auditor,
        runtime_auditor=runtime_auditor,
        pdf_auditor=pdf_auditor,
    )

    assert lock_state == {"held": False, "entries": 1}
    assert result["status"] == "incomplete"
    assert result["schema_version"] == 5
    assert result["measurement_source_revision"] == _REVISION
    assert result["publication_revision"] == _REVISION
    assert result["publication_lineage"]["relationship"] == "same-commit"
    assert result["authenticated_current_count"] == 2
    assert result["declared_cell_count"] == 2
    assert result["measurable_cell_count"] == 2
    assert result["catalog_static_na_cell_count"] == 0
    assert result["catalog_static_na_reason_counts"] == {}
    assert result["portable_publication_projection_count"] == 2
    assert result["publication_cache_role"] == ("portable-projection-of-current-result")
    assert result["cryptographic_audit_source"] == "immutable-current-result"
    assert result["numerically_evidenced_cell_count"] == 2
    assert result["legacy_fresh_oracle_count"] == 1
    assert result["legacy_oracles_with_inbound_agreement"] == 1
    assert result["legacy_pointwise_agreement_edge_count"] == 1
    assert result["direct_agreement_edge_count"] == 0
    assert result["direct_agreement_edge_counts"] == {
        "builtin-ufo-recurrence": 0,
        "z-recurrence-cross-mode": 0,
        "lc-cross-layout-component": 0,
        "lc-legacy-pyamplicol-component": 0,
    }
    assert result["replayed_direct_agreement_edge_count"] == 0
    assert set(result["direct_agreement_replay_category_counts"].values()) == {0}
    assert result["unique_artifact_count"] == 1
    assert result["canonical_publication_scope"] is False
    assert result["final_gate_complete"] is False
    assert result["audit_scope"] == "diagnostic-incomplete"
    assert result["pyamplicol_replay_count"] == 1
    assert result["replayed_measurement_count"] == 1
    assert result["visible_completeness"] == {
        "kind": "pyamplicol-report-visible-completeness",
        "schema_version": 2,
        "status": "ok",
        "maximum_n_final": 1,
        "declared_measurement_cell_count": 2,
        "required_measurement_count": 2,
        "rendered_required_measurement_count": 2,
        "structurally_not_applicable_display_slot_count": 0,
        "not_exposed_display_slot_count": 0,
        "catalog_static_na_cell_count": 0,
        "rendered_catalog_static_na_cell_count": 0,
        "applicable_na_display_slot_count": 0,
        "applicable_na_display_slots": [],
        "missing_rendered_cell_count": 0,
        "missing_rendered_cell_ids": [],
        "contract_errors": [],
    }

    def wrong_runtime_loader(_path: Path, _process: str) -> _Runtime:
        assert_locked()
        runtime = _Runtime()
        runtime.execution_mode = "compiled"
        return runtime

    with pytest.raises(FinalAuditError, match="loaded execution mode"):
        audit_final_report(
            tmp_path,
            expected_source_revision=_REVISION,
            max_n_final=1,
            expected_cell_count=2,
            replay=True,
            catalog=catalog,  # type: ignore[arg-type]
            service=service,
            active_runtime=_active_runtime(),
            artifact_auditor=artifact_auditor,
            runtime_loader=wrong_runtime_loader,
            source_auditor=source_auditor,
            runtime_auditor=runtime_auditor,
            pdf_auditor=pdf_auditor,
        )
    assert lock_state == {"held": False, "entries": 2}

    structural = audit_final_report(
        tmp_path,
        expected_source_revision=_REVISION,
        max_n_final=1,
        expected_cell_count=2,
        replay=False,
        catalog=catalog,  # type: ignore[arg-type]
        service=service,
        active_runtime=_active_runtime(),
        artifact_auditor=artifact_auditor,
        source_auditor=source_auditor,
        runtime_auditor=runtime_auditor,
        pdf_auditor=pdf_auditor,
    )
    assert structural["status"] == "incomplete"
    assert structural["final_gate_complete"] is False
    assert structural["audit_scope"] == "diagnostic-incomplete"
    assert structural["pyamplicol_replay_count"] == 0

    no_pdf = audit_final_report(
        tmp_path,
        expected_source_revision=_REVISION,
        max_n_final=1,
        expected_cell_count=2,
        replay=True,
        catalog=catalog,  # type: ignore[arg-type]
        service=service,
        active_runtime=_active_runtime(),
        artifact_auditor=artifact_auditor,
        runtime_loader=runtime_loader,
        source_auditor=source_auditor,
        runtime_auditor=runtime_auditor,
        verify_pdf=False,
    )
    assert no_pdf["status"] == "incomplete"
    assert no_pdf["final_gate_complete"] is False
    assert no_pdf["pdf_audit"] == {"status": "incomplete", "skipped": True}
    assert lock_state == {"held": False, "entries": 4}

    with monkeypatch.context() as local_patch:
        local_patch.setattr(
            final_audit_module,
            "summarize_visible_completeness",
            lambda *_args, **_kwargs: VisibleCompleteness(
                maximum_n_final=1,
                required_measurement_count=2,
                rendered_required_measurement_count=2,
                structurally_not_applicable_display_slot_count=0,
                not_exposed_display_slot_count=0,
                applicable_na_display_slots=("synthetic applicable N/A",),
                missing_rendered_cell_ids=(),
                contract_errors=(),
            ),
        )
        with pytest.raises(
            FinalAuditError,
            match=r"visible-completeness.*applicable_NA=1",
        ):
            audit_final_report(
                tmp_path,
                expected_source_revision=_REVISION,
                max_n_final=1,
                expected_cell_count=2,
                replay=False,
                catalog=catalog,  # type: ignore[arg-type]
                service=service,
                active_runtime=_active_runtime(),
                source_auditor=source_auditor,
                runtime_auditor=runtime_auditor,
                verify_pdf=False,
            )
    assert lock_state == {"held": False, "entries": 5}
