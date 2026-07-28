from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.generation.structural_source_proof import (
    ROLE as STRUCTURAL_SOURCE_PROOF_ROLE,
)
from pyamplicol.generation.structural_source_proof import (
    build_generation_structural_proof,
)
from tools.developer.final_source_numerical_truth import (
    REFERENCE_SCHEMA,
    TruthProducerError,
    build_witness,
    comparison_sha256,
    validate_witness_payload,
)
from tools.performance_report.models import Accuracy, ExecutionMode, Workload

REVISION = "a" * 40


@dataclass
class _Resolved:
    values: tuple[tuple[tuple[complex | Decimal, ...], ...], ...]
    helicity_ids: tuple[str, ...]
    color_ids: tuple[str, ...]
    color_accuracy: str


class _Runtime:
    def __init__(self, artifact_id: str, mode: str, accuracy: str) -> None:
        self.artifact_id = artifact_id
        self.execution_mode = mode
        self._accuracy = accuracy

    def evaluate_resolved(
        self,
        _momenta: object,
        *,
        helicities: object,
        color_flows: object,
        precision: int,
    ) -> _Resolved:
        selected_helicities = ("h:0",) if helicities else ("h:0", "h:1")
        selected_colors = ("c:0",)
        value: complex | Decimal = 1 + 0j if precision == 16 else Decimal("1")
        return _Resolved(
            values=(
                tuple(
                    tuple(value for _color in selected_colors)
                    for _helicity in selected_helicities
                ),
            ),
            helicity_ids=selected_helicities,
            color_ids=selected_colors,
            color_accuracy=self._accuracy,
        )


def _write_artifact(
    root: Path,
    *,
    artifact_id: str,
    mode: str,
    accuracy: str,
    reference: bool,
) -> None:
    process_id = "process"
    execution_path = "processes/process/execution.json"
    validation_path = "processes/process/validation-momenta.json"
    execution_payload = {
        "schema_version": 3,
        "kind": mode,
        "key": process_id,
        "process": "d d~ > z",
        "color_accuracy": accuracy,
        "external_pdg_order": [1, -1, 23],
        "source_count": 3,
        "current_count": 5,
        "interaction_count": 7,
        "amplitude_root_count": 1,
        "semantic": "reference" if reference else "candidate",
    }
    execution = json.dumps(execution_payload, sort_keys=True).encode()
    validation = json.dumps(
        {
            "available": True,
            "points": [
                [
                    {"pdg": 1, "momentum": ["5", "0", "0", "5"]},
                    {"pdg": -1, "momentum": ["5", "0", "0", "-5"]},
                    {"pdg": 23, "momentum": ["10", "0", "0", "0"]},
                ]
            ],
        },
        sort_keys=True,
    ).encode()
    (root / "processes" / "process").mkdir(parents=True)
    (root / execution_path).write_bytes(execution)
    (root / validation_path).write_bytes(validation)
    payloads: list[dict[str, object]] = [
        {
            "path": execution_path,
            "role": "evaluator-manifest",
            "process_id": process_id,
            "sha256": hashlib.sha256(execution).hexdigest(),
        },
        {
            "path": validation_path,
            "role": "validation-momenta",
            "process_id": process_id,
            "sha256": hashlib.sha256(validation).hexdigest(),
        },
    ]
    native_inputs = "b" * 64
    structural_path = "processes/process/structural-source-proof.json"
    structural = build_generation_structural_proof(
        artifact_root=root,
        process_id=process_id,
        source_revision=REVISION,
        native_build_inputs_sha256=native_inputs,
        execution_path=execution_path,
        execution_sha256=hashlib.sha256(execution).hexdigest(),
        execution=execution_payload,
        evaluator_container_path=None,
        evaluator_container_index_sha256=None,
    )
    structural_bytes = json.dumps(structural, sort_keys=True).encode()
    (root / structural_path).write_bytes(structural_bytes)
    payloads.append(
        {
            "path": structural_path,
            "role": STRUCTURAL_SOURCE_PROOF_ROLE,
            "process_id": process_id,
            "sha256": hashlib.sha256(structural_bytes).hexdigest(),
        }
    )
    if reference:
        contract_path = "independent-reference-semantics.json"
        contract = json.dumps(
            {
                "schema": REFERENCE_SCHEMA,
                "status": "materialized",
                "source_revision": REVISION,
                "artifact_id": artifact_id,
                "process_id": process_id,
                "semantics": "pre-optimization-reference",
                "semantic_payloads": [
                    {
                        "path": execution_path,
                        "sha256": hashlib.sha256(execution).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        ).encode()
        (root / contract_path).write_bytes(contract)
        payloads.append(
            {
                "path": contract_path,
                "role": "independent-reference-semantics",
                "sha256": hashlib.sha256(contract).hexdigest(),
            }
        )
    (root / "artifact.json").write_text(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "producer": {
                    "git_revision": REVISION,
                    "native_build_inputs_sha256": native_inputs,
                },
                "default_process_id": process_id,
                "processes": [
                    {
                        "id": process_id,
                        "color_accuracy": accuracy,
                    }
                ],
                "payloads": payloads,
            }
        )
    )


@pytest.mark.parametrize(
    ("mode", "accuracy", "workload", "helicities", "colors"),
    [
        (
            ExecutionMode.RECURRENCE,
            Accuracy.LC,
            Workload.SELECTED_FLOW,
            ("h:0",),
            ("c:0",),
        ),
        (ExecutionMode.COMPILED, Accuracy.NLC, Workload.CONTRACTED, (), ()),
        (ExecutionMode.EAGER, Accuracy.FULL, Workload.CONTRACTED, (), ()),
    ],
)
def test_builds_cross_mode_independent_precision80_witness(
    tmp_path: Path,
    mode: ExecutionMode,
    accuracy: Accuracy,
    workload: Workload,
    helicities: tuple[str, ...],
    colors: tuple[str, ...],
) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate_id = hashlib.sha256(f"candidate-{mode.value}".encode()).hexdigest()
    reference_id = hashlib.sha256(f"reference-{mode.value}".encode()).hexdigest()
    _write_artifact(
        candidate,
        artifact_id=candidate_id,
        mode=mode.value,
        accuracy=accuracy.value,
        reference=False,
    )
    _write_artifact(
        reference,
        artifact_id=reference_id,
        mode=mode.value,
        accuracy=accuracy.value,
        reference=True,
    )
    runtimes = {
        candidate.resolve(): _Runtime(candidate_id, mode.value, accuracy.value),
        reference.resolve(): _Runtime(reference_id, mode.value, accuracy.value),
    }

    witness = build_witness(
        candidate,
        reference,
        reference_contract_path="independent-reference-semantics.json",
        cell_id=f"cell-{mode.value}-{accuracy.value}",
        source_revision=REVISION,
        mode=mode,
        accuracy=accuracy,
        workload=workload,
        helicity_ids=helicities,
        color_ids=colors,
        runtime_loader=lambda path, **_kwargs: runtimes[Path(path).resolve()],
    )

    assert witness["status"] == "ok"
    assert witness["precision_decimal_digits"] == 80
    assert witness["agreement"]["passes"] is True
    assert validate_witness_payload(witness) == witness


def test_rejects_self_reference_and_digest_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact_id = hashlib.sha256(b"same").hexdigest()
    _write_artifact(
        artifact,
        artifact_id=artifact_id,
        mode=ExecutionMode.COMPILED.value,
        accuracy=Accuracy.LC.value,
        reference=True,
    )
    runtime = _Runtime(artifact_id, ExecutionMode.COMPILED.value, Accuracy.LC.value)
    with pytest.raises(TruthProducerError, match="one content identity"):
        build_witness(
            artifact,
            artifact,
            reference_contract_path="independent-reference-semantics.json",
            cell_id="cell",
            source_revision=REVISION,
            mode=ExecutionMode.COMPILED,
            accuracy=Accuracy.LC,
            workload=Workload.ALL_FLOW,
            runtime_loader=lambda _path, **_kwargs: runtime,
        )


def test_validator_recomputes_values_and_comparison_sha(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate_id = hashlib.sha256(b"candidate").hexdigest()
    reference_id = hashlib.sha256(b"reference").hexdigest()
    for root, identifier, is_reference in (
        (candidate, candidate_id, False),
        (reference, reference_id, True),
    ):
        _write_artifact(
            root,
            artifact_id=identifier,
            mode=ExecutionMode.COMPILED.value,
            accuracy=Accuracy.LC.value,
            reference=is_reference,
        )
    runtimes = {
        candidate.resolve(): _Runtime(
            candidate_id,
            ExecutionMode.COMPILED.value,
            Accuracy.LC.value,
        ),
        reference.resolve(): _Runtime(
            reference_id,
            ExecutionMode.COMPILED.value,
            Accuracy.LC.value,
        ),
    }
    witness = build_witness(
        candidate,
        reference,
        reference_contract_path="independent-reference-semantics.json",
        cell_id="cell",
        source_revision=REVISION,
        mode=ExecutionMode.COMPILED,
        accuracy=Accuracy.LC,
        workload=Workload.ALL_FLOW,
        runtime_loader=lambda path, **_kwargs: runtimes[Path(path).resolve()],
    )
    witness["reference_exact"]["values"][0][0][0] = "2"
    witness["reference_exact"]["values_sha256"] = "0" * 64
    witness["comparison_sha256"] = comparison_sha256(witness)
    with pytest.raises(TruthProducerError, match="values digest mismatch"):
        validate_witness_payload(witness)
