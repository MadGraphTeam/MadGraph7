# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path

import pytest

from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.legacy_structure import (
    LEGACY_PROOF_SCHEMA,
    LEGACY_SCOPE_REASON,
    InstrumentationRecord,
    LegacyStructuralProofError,
    _instrument_direct_source,
    _instrument_library_source,
    _parse_direct,
    _parse_module,
    emit_legacy_scope_unavailable_proof,
    emit_legacy_structural_proof,
    instrument_legacy_structural_probes,
)
from tools.performance_report.models import Accuracy, Workload

_MODULE = """\
module amp1_1_lib
  contains
  subroutine evaluate_amp1_1(p,amps)
    complex(kind=8),dimension(1),intent(out) :: amps
    complex(kind=8),dimension(1:6,3) :: val_c
    complex(kind=8),dimension(1:6,2) :: int_c
  end subroutine evaluate_amp1_1
  subroutine compute_external_currents(pp,val_c)
    call ext_quark(pp(0,1),-1,1,val_c(1,1))
    call ext_gluon(pp(0,2),-1,1,val_c(1,2))
  end subroutine compute_external_currents
  subroutine vertex_2(pp,val_c,int_c)
    integer,parameter,dimension(2) :: int1=[1,2]
  end subroutine vertex_2
  subroutine combine_currents_2(pp,val_c,int_c)
    integer,parameter,dimension(0:2,1) :: int1=reshape([3,1,2], shape=[3,1])
  end subroutine combine_currents_2
  subroutine compute_amps(amps,val_c)
    complex(kind=8),dimension(1),intent(out) :: amps
    amps(1)=sum(val_c(1:2,3)*val_c(1:2,1))
  end subroutine compute_amps
end module amp1_1_lib
"""


_DIRECT_OUTPUT = """\
color_accuracy lc
AMPICOL_COLOR_PROBE_CURRENTS 3
AMPICOL_COLOR_PROBE_VERTICES 2
AMPICOL_COLOR_PROBE_AMPLITUDES 1
AMPICOL_STRUCTURAL_CURRENT 1 1 1 0 1
AMPICOL_STRUCTURAL_CURRENT 2 1 21 0 2
AMPICOL_STRUCTURAL_CURRENT 3 0 1 0 3
AMPICOL_STRUCTURAL_ATTACHMENT 3 1 1 1
AMPICOL_STRUCTURAL_ATTACHMENT 3 2 2 -1
AMPICOL_STRUCTURAL_KERNEL 1 2 0 1 2 1.0D+00 0.0D+00
AMPICOL_STRUCTURAL_KERNEL_SINGLET 1 0
AMPICOL_STRUCTURAL_KERNEL 2 3 0 2 1 2.0D+00 0.0D+00
AMPICOL_STRUCTURAL_KERNEL_SINGLET 2 0
AMPICOL_STRUCTURAL_DESTINATION 1 3 1
"""

_DIRECT_SOURCE = """\
  logical :: print_matrix, fixed_helicity
  print_matrix = .false.
  fixed_helicity = .false.
  call get_environment_variable('AMPICOL_COLOR_PROBE_MATRIX',env_value)
  if (trim(adjustl(env_value)).eq.'1') print_matrix = .true.
  env_value = ''
  call print_recursion_counts()
  subroutine parse_color_accuracy()
"""

_LIBRARY_SOURCE = """\
  logical :: print_matrix
  print_matrix = .false.
  call get_environment_variable('AMPICOL_COLOR_PROBE_MATRIX',env_value)
  if (trim(adjustl(env_value)).eq.'1') print_matrix = .true.

  argc = command_argument_count()
  call build_row_to_integral()
  integer function colour_order_match_sign(jgroup,jint,row,pass,leg_map)
"""


def _instrument_probe_sources_in_process(
    repository: str,
    artifact_path: str,
    entered: object,
    release: object | None,
    outcomes: object,
    *,
    interrupt: bool,
) -> None:
    try:
        with instrument_legacy_structural_probes(
            Path(repository),
            Path(artifact_path),
        ):
            entered.set()
            if release is not None and not release.wait(timeout=5):
                raise TimeoutError("test did not release instrumented checkout")
            if interrupt:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        outcomes.put("interrupted")
    except BaseException as error:
        outcomes.put(f"error:{type(error).__name__}:{error}")
    else:
        outcomes.put("ok")


def _reference(*, workload: Workload, accuracy: Accuracy = Accuracy.LC) -> object:
    return next(
        cell
        for cell in REPORT_CATALOG.reference_cells()
        if cell.workload is workload and cell.measurement.accuracy is accuracy
    )


def _instrumentation(artifact: Path) -> InstrumentationRecord:
    evidence = artifact / "legacy-structural-evidence"
    direct = evidence / "instrumented-amplicol_color_probe.f03"
    library = evidence / "instrumented-amplicol_color_library_probe.f03"
    direct.parent.mkdir(parents=True, exist_ok=True)
    direct.write_text("direct diagnostic source\n")
    library.write_text("library diagnostic source\n")
    return InstrumentationRecord(
        abi="pyamplicol-legacy-probe-structural-diagnostics-v1",
        sources=(
            {
                "path": "amplicol_color_probe.f03",
                "original_sha256": "1" * 64,
                "instrumented_sha256": "2" * 64,
                "instrumented_evidence_path": direct.relative_to(
                    artifact
                ).as_posix(),
            },
            {
                "path": "amplicol_color_library_probe.f03",
                "original_sha256": "3" * 64,
                "instrumented_sha256": "4" * 64,
                "instrumented_evidence_path": library.relative_to(
                    artifact
                ).as_posix(),
            },
        ),
        evidence_paths=(direct, library),
    )


def _base_artifact(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "processes.txt").write_text("process\n")
    (root / "legacy.log").write_text("legacy commands\n")


def _assert_evidence_inventory(artifact: Path, proof: dict[str, object]) -> None:
    evidence = proof["evidence_files"]
    assert isinstance(evidence, list)
    assert evidence
    for record in evidence:
        assert isinstance(record, dict)
        path = artifact / str(record["path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_generated_module_preserves_exact_object_maps(tmp_path: Path) -> None:
    path = tmp_path / "amp1_1_lib.f03"
    path.write_text(_MODULE)
    module = _parse_module(path)
    assert module.source_current_ids == (1, 2)
    assert module.produced_current_ids == (3,)
    assert module.kernel_term_ids == (1, 2)
    assert module.combine_routes == ((3, (1, 2)),)
    assert module.amplitude_destinations == ((1, 3, 1),)
    assert module.counts == {
        "source_current_count": 2,
        "produced_current_count": 1,
        "kernel_evaluation_count": 2,
        "attachment_count": 2,
        "amplitude_destination_count": 1,
    }


def test_direct_probe_requires_dense_maps_and_exact_accuracy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "direct.stdout"
    path.write_text(_DIRECT_OUTPUT)
    structure = _parse_direct(path, "lc")
    assert structure.counts == {
        "source_current_count": 2,
        "produced_current_count": 1,
        "kernel_evaluation_count": 2,
        "attachment_count": 2,
        "amplitude_destination_count": 1,
    }
    with pytest.raises(LegacyStructuralProofError, match="expected full"):
        _parse_direct(path, "full")


def test_probe_instrumentation_has_real_histogram_emitter() -> None:
    patched_direct = _instrument_direct_source(_DIRECT_SOURCE)
    patched_library = _instrument_library_source(_LIBRARY_SOURCE)
    assert "AMPICOL_STRUCTURAL_ATTACHMENT" in patched_direct
    assert "AMPICOL_STRUCTURAL_DESTINATION" in patched_direct
    assert "AMPICOL_COLOR_PROBE_LIBRARY_CALLS" in patched_library
    assert (
        "write (99,'(a,3(1x,i0))') 'AMPICOL_COLOR_PROBE_LIBRARY_CALLS'"
        in patched_library
    )
    assert "AMPICOL_COLOR_PROBE_LIBRARY_ROW" in patched_library


def test_probe_instrumentation_serializes_processes_and_restores_after_interrupt(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "legacy"
    repository.mkdir()
    direct = repository / "amplicol_color_probe.f03"
    library = repository / "amplicol_color_library_probe.f03"
    direct.write_text(_DIRECT_SOURCE, encoding="utf-8")
    library.write_text(_LIBRARY_SOURCE, encoding="utf-8")
    unrelated_repository = tmp_path / "unrelated-legacy"
    unrelated_repository.mkdir()
    (unrelated_repository / direct.name).write_text(_DIRECT_SOURCE, encoding="utf-8")
    (unrelated_repository / library.name).write_text(_LIBRARY_SOURCE, encoding="utf-8")

    context = multiprocessing.get_context("fork")
    first_entered = context.Event()
    release_first = context.Event()
    second_entered = context.Event()
    unrelated_entered = context.Event()
    outcomes = context.Queue()
    first = context.Process(
        target=_instrument_probe_sources_in_process,
        args=(
            str(repository),
            str(tmp_path / "first-artifact"),
            first_entered,
            release_first,
            outcomes,
        ),
        kwargs={"interrupt": True},
    )
    second = context.Process(
        target=_instrument_probe_sources_in_process,
        args=(
            str(repository),
            str(tmp_path / "second-artifact"),
            second_entered,
            None,
            outcomes,
        ),
        kwargs={"interrupt": False},
    )
    unrelated = context.Process(
        target=_instrument_probe_sources_in_process,
        args=(
            str(unrelated_repository),
            str(tmp_path / "unrelated-artifact"),
            unrelated_entered,
            None,
            outcomes,
        ),
        kwargs={"interrupt": False},
    )

    first.start()
    assert first_entered.wait(timeout=3)
    unrelated.start()
    assert unrelated_entered.wait(timeout=3)
    unrelated.join(timeout=3)
    assert unrelated.exitcode == 0
    second.start()
    assert not second_entered.wait(timeout=0.2)
    assert second.is_alive()
    release_first.set()
    assert second_entered.wait(timeout=3)
    first.join(timeout=3)
    second.join(timeout=3)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert sorted(outcomes.get(timeout=1) for _ in range(3)) == [
        "interrupted",
        "ok",
        "ok",
    ]
    assert direct.read_text(encoding="utf-8") == _DIRECT_SOURCE
    assert library.read_text(encoding="utf-8") == _LIBRARY_SOURCE
    assert (
        tmp_path
        / "first-artifact"
        / "legacy-structural-evidence"
        / "instrumented-amplicol_color_probe.f03"
    ).is_file()
    assert (
        tmp_path
        / "second-artifact"
        / "legacy-structural-evidence"
        / "instrumented-amplicol_color_probe.f03"
    ).is_file()


def test_selected_flow_sidecar_is_atomic_and_producer_compatible(
    tmp_path: Path,
) -> None:
    cell = _reference(workload=Workload.SELECTED_FLOW)
    artifact = tmp_path / "artifact"
    _base_artifact(artifact)
    generated = artifact / "selected-flow-generated-library"
    library = generated / "Library"
    library.mkdir(parents=True)
    (library / "amp1_1_lib.f03").write_text(_MODULE)
    benchmark = generated / "amplicol_library_benchmark"
    benchmark.write_bytes(b"benchmark")
    repository = tmp_path / "legacy"
    repository.mkdir()
    proof_path = emit_legacy_structural_proof(
        cell,
        artifact_path=artifact,
        process_row="group:1:integral:1",
        source_revision="a" * 40,
        repository=repository,
        instrumentation=_instrumentation(artifact),
    )
    proof = json.loads(proof_path.read_text())
    assert proof["schema"] == LEGACY_PROOF_SCHEMA
    assert proof["scope"] == "available"
    assert proof["active"]["source_current_count"] == 2
    assert proof["static"] == proof["active"]
    assert proof["object_mapping"]["status"] == "exact"
    assert proof["row_multiplicity"]["call_count"] == 1
    _assert_evidence_inventory(artifact, proof)
    assert {item["path"] for item in proof["evidence_files"]} >= {
        "processes.txt",
        "legacy.log",
        "selected-flow-generated-library/Library/amp1_1_lib.f03",
        "legacy-structural-evidence/legacy-structural-index.json",
        "legacy-structural-evidence/source-contract.json",
    }
    assert not list(artifact.glob(".legacy-structural-proof.json.*"))


def test_direct_sidecar_separates_accuracy_and_maps(tmp_path: Path) -> None:
    cell = _reference(workload=Workload.ALL_FLOW)
    artifact = tmp_path / "artifact"
    _base_artifact(artifact)
    evidence = artifact / "legacy-structural-evidence"
    evidence.mkdir(parents=True)
    (evidence / "direct-structural-probe.stdout").write_text(_DIRECT_OUTPUT)
    repository = tmp_path / "legacy"
    repository.mkdir()
    (repository / "amplicol_color_probe").write_bytes(b"probe")
    proof_path = emit_legacy_structural_proof(
        cell,
        artifact_path=artifact,
        process_row="group:1:integral:1",
        source_revision="b" * 40,
        repository=repository,
        instrumentation=_instrumentation(artifact),
    )
    proof = json.loads(proof_path.read_text())
    assert proof["accuracy"] == "lc"
    assert proof["workload"] == Workload.ALL_FLOW.value
    assert proof["active"]["kernel_evaluation_count"] == 2
    assert proof["object_mapping"]["combine_route_map_sha256"]
    assert proof["object_mapping"]["amplitude_destination_map_sha256"]
    _assert_evidence_inventory(artifact, proof)


def test_scope_unavailable_has_exact_reason_without_fake_counts(
    tmp_path: Path,
) -> None:
    cell = next(
        cell
        for cell in REPORT_CATALOG.reference_cells()
        if not REPORT_CATALOG.legacy_reference_available(cell)
    )
    path = emit_legacy_scope_unavailable_proof(
        cell,
        artifact_path=tmp_path,
        source_revision="c" * 40,
        maximum_open_quark_lines=3,
        observed_open_quark_lines=4,
    )
    proof = json.loads(path.read_text())
    assert proof["scope"] == "unavailable"
    assert proof["reason"] == LEGACY_SCOPE_REASON
    assert "active" not in proof
    assert "static" not in proof
    assert proof["evidence_files"][0]["sha256"]
    _assert_evidence_inventory(tmp_path, proof)
