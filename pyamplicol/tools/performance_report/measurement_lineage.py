# SPDX-License-Identifier: 0BSD
"""Fail-closed executable-source lineage for architecture report profiles.

The ordinary report contract deliberately accepts measurements from one exact
source revision only.  A Class-C bridge is the narrow exception used when a
bounded physics correction must replace only the directly affected cells in a
frozen architecture campaign.  It never rewrites measurement provenance:
every retained ancestor current is authorized by its immutable attempt and
manifest digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .agreements import incoming_agreement_edges
from .artifacts import ATTEMPT_SCHEMA, ArtifactStore, CurrentRecord
from .cache import _validate_runtime_identity_postflight, validate_measurement
from .catalog import REPORT_CATALOG, ReportCatalog
from .models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from .source_identity import require_eligible_report_source

MEASUREMENT_LINEAGE_SCHEMA = "pyamplicol-performance-measurement-lineage-v1"
MEASUREMENT_LINEAGE_WRAPPER_SCHEMA = (
    "pyamplicol-performance-measurement-lineage-envelope-v1"
)
MEASUREMENT_LINEAGE_FILENAME = "measurement_lineage.json"
CLASS_C_HZZ_IMPACT = "hzz-orientation-v1"
CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT = "recurrence-summary-cap-v1"
_CLASS_C_IMPACTS = frozenset(
    {
        CLASS_C_HZZ_IMPACT,
        CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40,64}")
_ATTEMPT_KEYS = {
    "schema",
    "cell_id",
    "attempt_id",
    "status",
    "artifact_policy",
    "based_on",
    "result_path",
    "artifacts",
    "error",
}
_PROFILE_ROOT = PurePosixPath("docs/performance_reports")
_HZZ_CONTRACT_IDS = frozenset(
    {
        "pyamplicol.models.builtin.model.BuiltinSMModel:18",
        "pyamplicol.models.builtin.model.BuiltinSMModel:19",
    }
)

# The executable descendant is deliberately specific.  Any additional path
# requires a reviewed source change to this allowlist and therefore cannot be
# smuggled into a bridge manifest at run time.
_HZZ_ALLOWED_PATHS = frozenset(
    {
        "src/pyamplicol/models/builtin/lowering.py",
        "src/pyamplicol/assets/prepared_models/"
        "built-in-sm-jit-o2-aarch64.pyamplicol-model",
        "src/pyamplicol/assets/prepared_models/"
        "built-in-sm-jit-o2-aarch64.metadata.json",
        "src/pyamplicol/assets/prepared_models/"
        "built-in-sm-jit-o2-x86_64.pyamplicol-model",
        "src/pyamplicol/assets/prepared_models/"
        "built-in-sm-jit-o2-x86_64.metadata.json",
        "release_assets/prepared_models/"
        "built-in-sm-jit-o2-aarch64.pyamplicol-model",
        "release_assets/prepared_models/"
        "built-in-sm-jit-o2-aarch64.metadata.json",
        "release_assets/prepared_models/"
        "built-in-sm-jit-o2-x86_64.pyamplicol-model",
        "release_assets/prepared_models/"
        "built-in-sm-jit-o2-x86_64.metadata.json",
        "tools/performance_report/measurement_lineage.py",
        "tools/performance_report/source_identity.py",
        "tools/performance_report/workspace.py",
        "tools/performance_report/cli.py",
        "tools/performance_report/scheduler.py",
        "tools/performance_report/service.py",
        "tools/performance_report/render.py",
        "tools/performance_report/validation_summary.py",
        "tools/performance_report/campaign_policy.py",
        "tools/performance_report/final_audit.py",
        "docs/arxiv/result_tables.py",
        "docs/performance_reports/macbook_M3/result_tables.py",
        "docs/performance_reports/x86_EPYC/result_tables.py",
        "docs/performance_reports/macbook_M3/TABLE_FILLING.md",
        "docs/performance_reports/x86_EPYC/TABLE_FILLING.md",
        "tests/unit/test_performance_report_measurement_lineage.py",
        "tests/unit/test_performance_report_measurement_lineage_adversarial.py",
        "tests/unit/test_performance_report_source_identity.py",
        "tests/unit/test_performance_report_workspace.py",
        "tests/unit/test_performance_report_cli.py",
        "tests/unit/test_three_mode_report_scheduler.py",
        "tests/unit/test_three_mode_report_service.py",
        "tests/unit/test_three_mode_report_render.py",
        "tests/unit/test_performance_report_campaign_policy.py",
        "tests/unit/test_performance_report_final_audit.py",
        "tests/unit/test_model_builtin.py",
        "tests/unit/test_packaged_prepared_model.py",
        "tests/unit/test_recurrence_catalog_builder.py",
    }
)

_HZZ_REQUIRED_FEATURE_PATHS = frozenset(
    {
        "src/pyamplicol/models/builtin/lowering.py",
        "src/pyamplicol/assets/prepared_models/"
        "built-in-sm-jit-o2-aarch64.pyamplicol-model",
        "src/pyamplicol/assets/prepared_models/"
        "built-in-sm-jit-o2-aarch64.metadata.json",
        "src/pyamplicol/assets/prepared_models/"
        "built-in-sm-jit-o2-x86_64.pyamplicol-model",
        "src/pyamplicol/assets/prepared_models/"
        "built-in-sm-jit-o2-x86_64.metadata.json",
        "release_assets/prepared_models/"
        "built-in-sm-jit-o2-aarch64.pyamplicol-model",
        "release_assets/prepared_models/"
        "built-in-sm-jit-o2-aarch64.metadata.json",
        "release_assets/prepared_models/"
        "built-in-sm-jit-o2-x86_64.pyamplicol-model",
        "release_assets/prepared_models/"
        "built-in-sm-jit-o2-x86_64.metadata.json",
        "tests/unit/test_model_builtin.py",
        "tests/unit/test_packaged_prepared_model.py",
        "tests/unit/test_recurrence_catalog_builder.py",
    }
)

_RECURRENCE_SUMMARY_CAP_REPORT_FILES = (
    "pyAmpliCol.tex",
    "result_matrix_best_builtin_sm_full_table.tex",
    "result_matrix_best_builtin_sm_lc_table.tex",
    "result_matrix_best_builtin_sm_nlc_table.tex",
    "result_matrix_compiled_builtin_sm_full_table.tex",
    "result_matrix_compiled_builtin_sm_lc_table.tex",
    "result_matrix_compiled_builtin_sm_nlc_table.tex",
    "result_matrix_eager_builtin_sm_full_table.tex",
    "result_matrix_eager_builtin_sm_lc_table.tex",
    "result_matrix_eager_builtin_sm_nlc_table.tex",
    "result_matrix_recurrence_builtin_sm_full_table.tex",
    "result_matrix_recurrence_builtin_sm_lc_table.tex",
    "result_matrix_recurrence_builtin_sm_nlc_table.tex",
    "result_matrix_recurrence_ufo_sm_full_table.tex",
    "result_matrix_recurrence_ufo_sm_lc_table.tex",
    "result_matrix_recurrence_ufo_sm_nlc_table.tex",
    "result_scalar_contact_table.tex",
    "result_scalar_gravity_table.tex",
    "result_tables.py",
    "result_z_builtin_sm_table.tex",
    "result_z_external_sm_table.tex",
    "section_three_mode_performance_tables.tex",
)
_RECURRENCE_SUMMARY_CAP_ALLOWED_PATHS = (
    frozenset(
        {
            "docs/performance_reports/macbook_M3/TABLE_FILLING.md",
            "docs/performance_reports/x86_EPYC/TABLE_FILLING.md",
            "rust/crates/rusticol-core/src/engine/recurrence_manifest.rs",
            "src/pyamplicol/generation/artifact_writer.py",
            "tests/unit/test_performance_report_artifacts.py",
            "tests/unit/test_performance_report_cli.py",
            "tests/unit/test_performance_report_measurement_lineage.py",
            "tests/unit/test_performance_report_measurement_lineage_adversarial.py",
            "tests/unit/test_performance_report_publisher.py",
            "tests/unit/test_recurrence_direct_artifact_metadata.py",
            "tests/unit/test_three_mode_report_legacy.py",
            "tests/unit/test_three_mode_report_render.py",
            "tests/unit/test_three_mode_report_scheduler.py",
            "tools/performance_report/artifacts.py",
            "tools/performance_report/campaign_policy.py",
            "tools/performance_report/cli.py",
            "tools/performance_report/legacy.py",
            "tools/performance_report/measurement_lineage.py",
            "tools/performance_report/publisher.py",
            "tools/performance_report/render.py",
            "tools/performance_report/scheduler.py",
            "tools/performance_report/workspace.py",
        }
    )
    | frozenset(
        f"{root}/{filename}"
        for root in (
            "docs/arxiv",
            "docs/performance_reports/macbook_M3",
            "docs/performance_reports/x86_EPYC",
        )
        for filename in _RECURRENCE_SUMMARY_CAP_REPORT_FILES
    )
)
_RECURRENCE_SUMMARY_CAP_REQUIRED_PATHS = _RECURRENCE_SUMMARY_CAP_ALLOWED_PATHS

_RECURRENCE_SUMMARY_CAP_PREDECESSOR_REVISION = (
    "6536ef131e18e7ff873eb1f4db1f08631155f3a9"
)
_RECURRENCE_SUMMARY_CAP_ANCESTOR_REVISION = (
    "be11d8304fdc04893dc0e23e9619be848126e3bc"
)
_RECURRENCE_SUMMARY_CAP_DESCENDANT_REVISION = (
    "2594d8b520b802f71d60bd646f73ebaa5547927a"
)
_RECURRENCE_SUMMARY_CAP_ANCESTOR_NATIVE_INPUTS_SHA256 = (
    "23b9637d5d3fba0947d78cf688df18799b0c9ee5b3bcbfa6a2963a1f1a21f870"
)
_RECURRENCE_SUMMARY_CAP_DESCENDANT_NATIVE_INPUTS_SHA256 = (
    "96e1ff79a007aaf67a0900dd6d67327ee00f6bd2cca002589b879aa3a734de08"
)
_RECURRENCE_SUMMARY_CAP_PROFILE = "x86_EPYC"
_RECURRENCE_SUMMARY_CAP_FAILURE_BYTES = {
    "matrix-recurrence-builtin-sm-lc-n7-gg-gluons-selected-flow": 4_270_140,
    "matrix-recurrence-builtin-sm-lc-n8-dd-tt-jets-selected-flow": 1_083_926,
    "matrix-recurrence-builtin-sm-lc-n9-dd-z-jets-selected-flow": 4_449_888,
    "matrix-recurrence-builtin-sm-lc-n9-ud-w-jets-selected-flow": 4_449_912,
}
_RECURRENCE_SUMMARY_CAP_AGREEMENT_IDS = frozenset(
    {
        "matrix-recurrence-builtin-sm-lc-n7-gg-gluons-all-flow",
        "matrix-recurrence-builtin-sm-lc-n8-dd-tt-jets-all-flow",
        "matrix-recurrence-builtin-sm-lc-n9-dd-z-jets-all-flow",
        "matrix-recurrence-builtin-sm-lc-n9-ud-w-jets-all-flow",
        "matrix-recurrence-ufo-sm-lc-n7-gg-gluons-all-flow",
        "matrix-recurrence-ufo-sm-lc-n7-gg-gluons-selected-flow",
        "matrix-recurrence-ufo-sm-lc-n8-dd-tt-jets-all-flow",
        "matrix-recurrence-ufo-sm-lc-n8-dd-tt-jets-selected-flow",
        "matrix-recurrence-ufo-sm-lc-n9-dd-z-jets-all-flow",
        "matrix-recurrence-ufo-sm-lc-n9-dd-z-jets-selected-flow",
        "matrix-recurrence-ufo-sm-lc-n9-ud-w-jets-all-flow",
        "matrix-recurrence-ufo-sm-lc-n9-ud-w-jets-selected-flow",
    }
)
_SIGNED_ZERO_HELICITY_REFERENCE_IDS = frozenset(
    {
        f"reference-amplicol-lc-n{n_final}-{process}-{workload}"
        for n_final in range(4, 8)
        for process in ("dd-epemzh-jets", "dd-ttzh-jets")
        for workload in ("selected-flow", "all-flow")
    }
)
_SIGNED_ZERO_HELICITY_SELECTED_RECURRENCE_IDS = frozenset(
    {
        f"matrix-recurrence-builtin-sm-lc-n{n_final}-{process}-selected-flow"
        for n_final in range(4, 8)
        for process in ("dd-epemzh-jets", "dd-ttzh-jets")
    }
)
_SIGNED_ZERO_HELICITY_IMPACTED_IDS = (
    _SIGNED_ZERO_HELICITY_REFERENCE_IDS
    | _SIGNED_ZERO_HELICITY_SELECTED_RECURRENCE_IDS
)
_SIGNED_ZERO_HELICITY_AGREEMENT_IDS = frozenset(
    {
        f"matrix-recurrence-{model}-lc-n{n_final}-{process}-{workload}"
        for model in ("builtin-sm", "ufo-sm")
        for n_final in range(4, 8)
        for process in ("dd-epemzh-jets", "dd-ttzh-jets")
        for workload in (
            ("all-flow",)
            if model == "builtin-sm"
            else ("selected-flow", "all-flow")
        )
    }
) | frozenset(
    {
        f"matrix-{mode}-builtin-sm-lc-n{n_final}-{process}-all-flow"
        for mode in ("compiled", "eager")
        for n_final in range(4, 8)
        for process in ("dd-epemzh-jets", "dd-ttzh-jets")
    }
)

_ENVIRONMENT_KEYS = {
    "schema",
    "profile",
    "status",
    "source_revision",
    "platform",
    "machine",
    "processor",
    "python",
    "python_implementation",
    "pyamplicol",
    "numpy",
    "native_target",
    "native_cpu_features",
    "native_build_inputs_sha256",
    "native_extension_sha256",
    "python_package_tree_sha256",
    "candidate_fingerprint",
}

# A fresh native relink is authenticated by its exact build-input digest,
# candidate fingerprint, target, and CPU feature set.  The raw extension digest
# is intentionally recorded in each endpoint environment but is not invariant:
# Mach-O UUIDs, temporary build paths, and signatures make that output
# byte-nondeterministic even when every native input is unchanged.
_ENVIRONMENT_INVARIANT_FIELDS = (
    "schema",
    "profile",
    "platform",
    "machine",
    "processor",
    "python",
    "python_implementation",
    "status",
    "pyamplicol",
    "numpy",
    "native_target",
    "native_cpu_features",
    "native_build_inputs_sha256",
    "candidate_fingerprint",
)

_CURRENT_PIN_KEYS = {
    "cell_id",
    "attempt_id",
    "manifest_sha256",
    "current_locator",
    "current_pointer_sha256",
    "source_revision",
    "source_tree",
    "result_sha256",
}

_REACHABILITY_RECORD_KEYS = {
    "cell_id",
    "attempt_id",
    "manifest_sha256",
    "artifact_locator",
    "artifact_owner",
    "process_id",
    "execution_manifest_path",
    "execution_manifest_sha256",
    "schedule_path",
    "schedule_sha256",
    "schedule_index_sha256",
    "schedule_member_sha256",
    "process_binding_path",
    "process_binding_sha256",
    "kernel_pack_path",
    "kernel_pack_sha256",
    "semantic_catalog_sha256",
    "direct_catalog_sha256",
    "active_executor_ids",
    "matched_template_ids",
    "matched_contract_ids",
}

_SUMMARY_CAP_FAILURE_RECORD_KEYS = {
    "cell_id",
    "attempt_id",
    "manifest_sha256",
    "result_locator",
    "result_sha256",
    "summary_bytes",
    "failure_kind",
    "failure_message",
}
_SUMMARY_CAP_EXCLUDED_RECORD_KEYS = {
    "cell_id",
    "attempt_id",
    "manifest_sha256",
    "result_locator",
    "result_sha256",
    "current_locator",
    "current_pointer_sha256",
    "status",
    "failure_sha256",
}
_SUMMARY_CAP_PREDECESSOR_KEYS = {
    "payload_sha256",
    "profile",
    "impact",
    "ancestor_revision",
    "ancestor_tree",
    "descendant_revision",
    "descendant_tree",
    "ancestor_environment",
    "ancestor_environment_sha256",
    "descendant_environment_sha256",
    "retained_currents",
    "retained_currents_sha256",
}

_ARTIFACT_OWNER_KEYS = {
    "relation",
    "artifact_locator",
    "consumer_cell_id",
    "consumer_attempt_id",
    "consumer_manifest_locator",
    "consumer_manifest_sha256",
    "consumer_result_locator",
    "consumer_result_sha256",
    "consumer_source_revision",
    "consumer_source_tree",
    "consumer_runtime_identity_sha256",
    "consumer_runtime_identity_stable_sha256",
    "owner_cell_id",
    "owner_attempt_id",
    "owner_current_locator",
    "owner_current_sha256",
    "owner_manifest_locator",
    "owner_manifest_sha256",
    "owner_result_locator",
    "owner_result_sha256",
    "owner_artifacts_sha256",
    "owner_source_revision",
    "owner_source_tree",
    "owner_runtime_identity_sha256",
    "owner_runtime_identity_stable_sha256",
}

_ATTEMPT_INVENTORY_KEYS = {
    "cell_id",
    "attempt_id",
    "manifest_locator",
    "manifest_sha256",
    "status",
    "error_sha256",
    "result_path",
    "result_sha256",
    "artifact_count",
    "artifacts_sha256",
}

_PAYLOAD_KEYS = {
    "schema",
    "state",
    "class",
    "impact",
    "profile",
    "ancestor_revision",
    "ancestor_tree",
    "descendant_revision",
    "descendant_tree",
    "git_diff",
    "git_diff_sha256",
    "ancestor_environment",
    "ancestor_environment_sha256",
    "descendant_environment",
    "descendant_environment_sha256",
    "runtime_invariant_fields",
    "impacted_cells",
    "agreement_closure_cells",
    "impact_certificate_sha256",
    "catalog_sha256",
    "agreement_graph_sha256",
    "workspace_manifest",
    "workspace_manifest_sha256",
    "campaign_policy_sha256",
    "reachability_certificate",
    "reachability_certificate_sha256",
    "retained_currents",
    "invalidated_currents",
    "recompare_currents",
    "attempt_inventory",
    "no_attempt_cells",
    "current_snapshot_sha256",
}

_CANONICAL_CATALOG_SHA256: str | None = None
_CANONICAL_AGREEMENT_SHA256: str | None = None


class MeasurementLineageError(RuntimeError):
    """A Class-C measurement-lineage record is absent or inconsistent."""


def _canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise MeasurementLineageError(
            f"measurement lineage is not canonical JSON: {error}"
        ) from error
    return f"{encoded}\n".encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise MeasurementLineageError(
                f"measurement-lineage JSON contains duplicate key {key!r}"
            )
        value[key] = item
    return value


def _read_canonical_json(path: Path, *, context: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise MeasurementLineageError(f"{context} is not a regular file: {path}")
    try:
        data = path.read_bytes()
        raw = json.loads(
            data.decode("ascii"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MeasurementLineageError(
            f"cannot read {context} {path}: {error}"
        ) from error
    if not isinstance(raw, dict) or data != _canonical_bytes(raw):
        raise MeasurementLineageError(
            f"{context} has no valid canonical digest/JSON encoding: {path}"
        )
    return raw


def _write_envelope(path: Path, payload: Mapping[str, object]) -> None:
    envelope = {
        "schema": MEASUREMENT_LINEAGE_WRAPPER_SCHEMA,
        "payload": dict(payload),
        "payload_sha256": _digest(payload),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_bytes(envelope))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _restore_file(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
    else:
        _atomic_write_bytes(path, previous)


def _safe_locator(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not Path(value).is_absolute()
        and all(
            part not in {"", ".", ".."}
            for part in Path(value).parts
        )
    )


def _validate_artifact_owner(
    record: Mapping[str, object],
) -> None:
    raw = record.get("artifact_owner")
    if not isinstance(raw, Mapping) or set(raw) != _ARTIFACT_OWNER_KEYS:
        raise MeasurementLineageError(
            "measurement-lineage recurrence artifact-owner record is malformed"
        )
    relation = raw.get("relation")
    try:
        consumer_attempt = (
            str(uuid.UUID(str(raw["consumer_attempt_id"])))
            if isinstance(raw.get("consumer_attempt_id"), str)
            else None
        )
        owner_attempt = (
            str(uuid.UUID(str(raw["owner_attempt_id"])))
            if isinstance(raw.get("owner_attempt_id"), str)
            else None
        )
    except ValueError:
        consumer_attempt = None
        owner_attempt = None
    consumer_cell = raw.get("consumer_cell_id")
    owner_cell = raw.get("owner_cell_id")
    locators = (
        raw.get("artifact_locator"),
        raw.get("consumer_manifest_locator"),
        raw.get("consumer_result_locator"),
        raw.get("owner_manifest_locator"),
        raw.get("owner_result_locator"),
    )
    digests = (
        "consumer_manifest_sha256",
        "consumer_result_sha256",
        "consumer_runtime_identity_sha256",
        "consumer_runtime_identity_stable_sha256",
        "owner_manifest_sha256",
        "owner_result_sha256",
        "owner_artifacts_sha256",
        "owner_runtime_identity_sha256",
        "owner_runtime_identity_stable_sha256",
    )
    revisions = (
        "consumer_source_revision",
        "consumer_source_tree",
        "owner_source_revision",
        "owner_source_tree",
    )
    if (
        relation not in {"consumer-attempt", "equivalent-matrix-peer"}
        or not isinstance(consumer_cell, str)
        or not consumer_cell
        or not isinstance(owner_cell, str)
        or not owner_cell
        or consumer_attempt != raw.get("consumer_attempt_id")
        or owner_attempt != raw.get("owner_attempt_id")
        or consumer_cell != record.get("cell_id")
        or consumer_attempt != record.get("attempt_id")
        or raw.get("consumer_manifest_sha256")
        != record.get("manifest_sha256")
        or raw.get("artifact_locator") != record.get("artifact_locator")
        or not all(_safe_locator(value) for value in locators)
        or any(
            not isinstance(raw.get(field), str)
            or _SHA256_RE.fullmatch(str(raw[field])) is None
            for field in digests
        )
        or any(
            not isinstance(raw.get(field), str)
            or _GIT_SHA_RE.fullmatch(str(raw[field])) is None
            for field in revisions
        )
        or raw.get("consumer_source_revision")
        != raw.get("owner_source_revision")
        or raw.get("consumer_source_tree") != raw.get("owner_source_tree")
        or raw.get("consumer_runtime_identity_stable_sha256")
        != raw.get("owner_runtime_identity_stable_sha256")
    ):
        raise MeasurementLineageError(
            "measurement-lineage recurrence artifact-owner identity is invalid"
        )

    consumer_manifest = PurePosixPath(str(raw["consumer_manifest_locator"]))
    consumer_result = PurePosixPath(str(raw["consumer_result_locator"]))
    owner_manifest = PurePosixPath(str(raw["owner_manifest_locator"]))
    owner_result = PurePosixPath(str(raw["owner_result_locator"]))
    artifact = PurePosixPath(str(raw["artifact_locator"]))
    owner_current = raw.get("owner_current_locator")
    owner_current_sha256 = raw.get("owner_current_sha256")
    if (
        not consumer_manifest.as_posix().endswith(
            f"/attempts/{consumer_attempt}/manifest.json"
        )
        or consumer_result.parent != consumer_manifest.parent
        or not owner_manifest.as_posix().endswith(
            f"/attempts/{owner_attempt}/manifest.json"
        )
        or owner_result.parent != owner_manifest.parent
        or artifact != owner_manifest.parent / "artifact"
    ):
        raise MeasurementLineageError(
            "measurement-lineage recurrence artifact-owner locators are inconsistent"
        )
    if relation == "consumer-attempt":
        if (
            owner_cell != consumer_cell
            or owner_attempt != consumer_attempt
            or owner_current is not None
            or owner_current_sha256 is not None
            or owner_manifest != consumer_manifest
            or owner_result != consumer_result
            or raw.get("owner_manifest_sha256")
            != raw.get("consumer_manifest_sha256")
            or raw.get("owner_result_sha256")
            != raw.get("consumer_result_sha256")
        ):
            raise MeasurementLineageError(
                "measurement-lineage direct artifact owner is inconsistent"
            )
    elif (
        owner_cell == consumer_cell
        or not _safe_locator(owner_current)
        or not str(owner_current).endswith("/current.json")
        or not isinstance(owner_current_sha256, str)
        or _SHA256_RE.fullmatch(owner_current_sha256) is None
        or PurePosixPath(str(owner_current))
        != owner_manifest.parents[2] / "current.json"
    ):
        raise MeasurementLineageError(
            "measurement-lineage matrix-peer artifact owner is inconsistent"
        )


def _summary_cap_failure_message(summary_bytes: int) -> str:
    return (
        "Rust recurrence execution summary must be smaller than 1 MiB; "
        f"received {summary_bytes} bytes"
    )


def _validate_summary_cap_predecessor(
    predecessor: object,
) -> None:
    if (
        not isinstance(predecessor, Mapping)
        or set(predecessor) != _SUMMARY_CAP_PREDECESSOR_KEYS
        or predecessor.get("impact") != CLASS_C_HZZ_IMPACT
        or predecessor.get("profile") != _RECURRENCE_SUMMARY_CAP_PROFILE
        or predecessor.get("ancestor_revision")
        != _RECURRENCE_SUMMARY_CAP_PREDECESSOR_REVISION
        or predecessor.get("descendant_revision")
        != _RECURRENCE_SUMMARY_CAP_ANCESTOR_REVISION
        or not isinstance(predecessor.get("payload_sha256"), str)
        or _SHA256_RE.fullmatch(str(predecessor["payload_sha256"])) is None
        or any(
            not isinstance(predecessor.get(field), str)
            or _GIT_SHA_RE.fullmatch(str(predecessor[field])) is None
            for field in (
                "ancestor_revision",
                "ancestor_tree",
                "descendant_revision",
                "descendant_tree",
            )
        )
        or not isinstance(predecessor.get("ancestor_environment"), Mapping)
        or set(predecessor["ancestor_environment"]) != _ENVIRONMENT_KEYS
        or predecessor.get("ancestor_environment_sha256")
        != _digest(predecessor["ancestor_environment"])
        or not isinstance(
            predecessor.get("descendant_environment_sha256"),
            str,
        )
        or _SHA256_RE.fullmatch(
            str(predecessor["descendant_environment_sha256"])
        )
        is None
        or not isinstance(predecessor.get("retained_currents"), list)
        or predecessor.get("retained_currents_sha256")
        != _digest(predecessor["retained_currents"])
    ):
        raise MeasurementLineageError(
            "measurement-lineage summary-cap predecessor is malformed"
        )
    retained = _validated_pins(predecessor, "retained_currents")
    if any(
        pin["source_revision"] != predecessor["ancestor_revision"]
        or pin["source_tree"] != predecessor["ancestor_tree"]
        for pin in retained
    ):
        raise MeasurementLineageError(
            "measurement-lineage summary-cap predecessor pins another source"
        )


def _validate_summary_cap_certificate(
    certificate: Mapping[str, object],
) -> None:
    expected_ids = sorted(_RECURRENCE_SUMMARY_CAP_FAILURE_BYTES)
    records = certificate.get("records")
    excluded = certificate.get("excluded_non_success_currents")
    predecessor = certificate.get("predecessor")
    invalidated = certificate.get("invalidated_generation_error_current_ids")
    inspected = certificate.get("inspected_current_count")
    successful = certificate.get("successful_current_count")
    excluded_count = certificate.get("excluded_non_success_current_count")
    if (
        set(certificate)
        != {
            "algorithm",
            "predecessor",
            "target_summary_bytes",
            "inspected_current_count",
            "successful_current_count",
            "excluded_non_success_current_count",
            "records",
            "excluded_non_success_currents",
            "invalidated_generation_error_current_ids",
            "sha256",
        }
        or certificate.get("algorithm")
        != "authenticated-recurrence-summary-cap-failure-census-v1"
        or certificate.get("target_summary_bytes")
        != _RECURRENCE_SUMMARY_CAP_FAILURE_BYTES
        or not isinstance(inspected, int)
        or isinstance(inspected, bool)
        or not isinstance(successful, int)
        or isinstance(successful, bool)
        or not isinstance(excluded_count, int)
        or isinstance(excluded_count, bool)
        or inspected != successful + excluded_count + len(expected_ids)
        or successful < 0
        or excluded_count < 0
        or not isinstance(records, list)
        or not isinstance(excluded, list)
        or len(excluded) != excluded_count
        or not isinstance(invalidated, list)
        or invalidated != expected_ids
        or len(records) != len(expected_ids)
    ):
        raise MeasurementLineageError(
            "measurement-lineage recurrence summary-cap census is malformed"
        )
    _validate_summary_cap_predecessor(predecessor)
    record_ids: list[str] = []
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != _SUMMARY_CAP_FAILURE_RECORD_KEYS
        ):
            raise MeasurementLineageError(
                "measurement-lineage recurrence summary-cap record is malformed"
            )
        cell_id = record.get("cell_id")
        attempt_id = record.get("attempt_id")
        try:
            canonical_attempt = (
                str(uuid.UUID(attempt_id))
                if isinstance(attempt_id, str)
                else None
            )
        except ValueError:
            canonical_attempt = None
        result_locator = record.get("result_locator")
        summary_bytes = (
            _RECURRENCE_SUMMARY_CAP_FAILURE_BYTES.get(cell_id)
            if isinstance(cell_id, str)
            else None
        )
        if (
            summary_bytes is None
            or canonical_attempt != attempt_id
            or not _safe_locator(result_locator)
            or not str(result_locator).endswith(
                f"/attempts/{attempt_id}/result.json"
            )
            or not isinstance(record.get("manifest_sha256"), str)
            or _SHA256_RE.fullmatch(str(record["manifest_sha256"])) is None
            or not isinstance(record.get("result_sha256"), str)
            or _SHA256_RE.fullmatch(str(record["result_sha256"])) is None
            or record.get("summary_bytes") != summary_bytes
            or record.get("failure_kind") != "GenerationError"
            or record.get("failure_message")
            != _summary_cap_failure_message(summary_bytes)
        ):
            raise MeasurementLineageError(
                "measurement-lineage recurrence summary-cap record is invalid"
            )
        record_ids.append(cell_id)
    if record_ids != expected_ids:
        raise MeasurementLineageError(
            "measurement-lineage recurrence summary-cap records do not match "
            "the exact failure census"
        )
    excluded_ids: list[str] = []
    for record in excluded:
        if (
            not isinstance(record, dict)
            or set(record) != _SUMMARY_CAP_EXCLUDED_RECORD_KEYS
        ):
            raise MeasurementLineageError(
                "measurement-lineage excluded terminal current is malformed"
            )
        cell_id = record.get("cell_id")
        attempt_id = record.get("attempt_id")
        try:
            canonical_attempt = (
                str(uuid.UUID(attempt_id))
                if isinstance(attempt_id, str)
                else None
            )
        except ValueError:
            canonical_attempt = None
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or cell_id in _RECURRENCE_SUMMARY_CAP_FAILURE_BYTES
            or canonical_attempt != attempt_id
            or not _safe_locator(record.get("result_locator"))
            or not str(record["result_locator"]).endswith(
                f"/attempts/{attempt_id}/result.json"
            )
            or not _safe_locator(record.get("current_locator"))
            or not str(record["current_locator"]).endswith("/current.json")
            or record.get("status") == ResultStatus.OK.value
            or record.get("status")
            not in {status.value for status in ResultStatus}
            or any(
                not isinstance(record.get(field), str)
                or _SHA256_RE.fullmatch(str(record[field])) is None
                for field in (
                    "manifest_sha256",
                    "result_sha256",
                    "current_pointer_sha256",
                    "failure_sha256",
                )
            )
        ):
            raise MeasurementLineageError(
                "measurement-lineage excluded terminal current is invalid"
            )
        excluded_ids.append(cell_id)
    if excluded_ids != sorted(set(excluded_ids)):
        raise MeasurementLineageError(
            "measurement-lineage excluded terminal currents are not a unique census"
        )


def _validate_payload_shape(payload: Mapping[str, object]) -> None:
    if set(payload) != _PAYLOAD_KEYS:
        raise MeasurementLineageError(
            "measurement-lineage payload has an unsupported schema shape"
        )
    if (
        payload.get("schema") != MEASUREMENT_LINEAGE_SCHEMA
        or payload.get("state") not in {"pending", "finalized"}
        or payload.get("class") != "C"
        or payload.get("impact") not in _CLASS_C_IMPACTS
        or not isinstance(payload.get("profile"), str)
        or not payload["profile"]
        or (
            payload.get("impact")
            == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
            and (
                payload.get("profile") != _RECURRENCE_SUMMARY_CAP_PROFILE
                or payload.get("ancestor_revision")
                != _RECURRENCE_SUMMARY_CAP_ANCESTOR_REVISION
                or payload.get("descendant_revision")
                != _RECURRENCE_SUMMARY_CAP_DESCENDANT_REVISION
            )
        )
        or payload.get("runtime_invariant_fields")
        != list(_ENVIRONMENT_INVARIANT_FIELDS)
    ):
        raise MeasurementLineageError(
            "measurement-lineage class, impact, profile, or invariants are invalid"
        )
    for field in (
        "ancestor_revision",
        "ancestor_tree",
        "descendant_revision",
        "descendant_tree",
    ):
        if (
            not isinstance(payload.get(field), str)
            or _GIT_SHA_RE.fullmatch(str(payload[field])) is None
        ):
            raise MeasurementLineageError(
                f"measurement-lineage {field} is not a full Git identity"
            )
    for field in (
        "git_diff_sha256",
        "ancestor_environment_sha256",
        "impact_certificate_sha256",
        "catalog_sha256",
        "agreement_graph_sha256",
        "workspace_manifest_sha256",
        "campaign_policy_sha256",
        "reachability_certificate_sha256",
        "current_snapshot_sha256",
    ):
        if (
            not isinstance(payload.get(field), str)
            or _SHA256_RE.fullmatch(str(payload[field])) is None
        ):
            raise MeasurementLineageError(
                f"measurement-lineage {field} is not a SHA-256 digest"
            )
    ancestor_environment = payload.get("ancestor_environment")
    if (
        not isinstance(ancestor_environment, Mapping)
        or set(ancestor_environment) != _ENVIRONMENT_KEYS
        or payload.get("ancestor_environment_sha256")
        != _digest(ancestor_environment)
    ):
        raise MeasurementLineageError(
            "measurement-lineage ancestor environment is invalid"
        )
    descendant_environment = payload.get("descendant_environment")
    descendant_digest = payload.get("descendant_environment_sha256")
    if payload["state"] == "pending":
        if descendant_environment is not None or descendant_digest is not None:
            raise MeasurementLineageError(
                "pending measurement lineage carries a descendant environment"
            )
    elif (
        not isinstance(descendant_environment, Mapping)
        or set(descendant_environment) != _ENVIRONMENT_KEYS
        or not isinstance(descendant_digest, str)
        or descendant_digest != _digest(descendant_environment)
    ):
        raise MeasurementLineageError(
            "finalized measurement-lineage descendant environment is invalid"
        )
    workspace = payload.get("workspace_manifest")
    if (
        not isinstance(workspace, Mapping)
        or set(workspace)
        != {
            "path",
            "mode",
            "object",
            "sha256",
            "campaign_policy_sha256",
        }
        or workspace.get("mode") != "100644"
        or workspace.get("campaign_policy_sha256")
        != payload.get("campaign_policy_sha256")
        or payload.get("workspace_manifest_sha256") != _digest(workspace)
    ):
        raise MeasurementLineageError(
            "measurement-lineage workspace manifest identity is invalid"
        )
    for field in (
        "git_diff",
        "impacted_cells",
        "agreement_closure_cells",
        "attempt_inventory",
        "no_attempt_cells",
        "retained_currents",
        "invalidated_currents",
        "recompare_currents",
    ):
        if not isinstance(payload.get(field), list):
            raise MeasurementLineageError(
                f"measurement-lineage {field} must be an array"
            )
    no_attempts = payload["no_attempt_cells"]
    if no_attempts != sorted(set(no_attempts)) or not all(
        isinstance(value, str) and value for value in no_attempts
    ):
        raise MeasurementLineageError(
            "measurement-lineage no-attempt inventory is not sorted and unique"
        )
    inventory = payload["attempt_inventory"]
    inventory_keys: list[tuple[str, str]] = []
    for raw in inventory:
        if not isinstance(raw, dict) or set(raw) != _ATTEMPT_INVENTORY_KEYS:
            raise MeasurementLineageError(
                "measurement-lineage attempt inventory row is malformed"
            )
        cell_id = raw.get("cell_id")
        attempt_id = raw.get("attempt_id")
        try:
            canonical_attempt_id = (
                str(uuid.UUID(attempt_id))
                if isinstance(attempt_id, str)
                else None
            )
        except ValueError:
            canonical_attempt_id = None
        locator = raw.get("manifest_locator")
        result_path = raw.get("result_path")
        status = raw.get("status")
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or canonical_attempt_id != attempt_id
            or not isinstance(locator, str)
            or Path(locator).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(locator).parts)
            or not locator.endswith(f"/attempts/{attempt_id}/manifest.json")
            or status not in {"ok", "failed", "interrupted"}
            or (
                status == "ok"
                and (
                    not isinstance(result_path, str)
                    or not result_path
                    or Path(result_path).is_absolute()
                    or any(
                        part in {"", ".", ".."}
                        for part in Path(result_path).parts
                    )
                )
            )
            or (status != "ok" and result_path is not None)
            or not isinstance(raw.get("manifest_sha256"), str)
            or _SHA256_RE.fullmatch(str(raw["manifest_sha256"])) is None
            or not isinstance(raw.get("error_sha256"), str)
            or _SHA256_RE.fullmatch(str(raw["error_sha256"])) is None
            or (
                raw.get("result_sha256") is not None
                and (
                    not isinstance(raw.get("result_sha256"), str)
                    or _SHA256_RE.fullmatch(str(raw["result_sha256"])) is None
                )
            )
            or isinstance(raw.get("artifact_count"), bool)
            or not isinstance(raw.get("artifact_count"), int)
            or int(raw["artifact_count"]) < 0
            or not isinstance(raw.get("artifacts_sha256"), str)
            or _SHA256_RE.fullmatch(str(raw["artifacts_sha256"])) is None
        ):
            raise MeasurementLineageError(
                "measurement-lineage attempt inventory identity is invalid"
            )
        inventory_keys.append((cell_id, attempt_id))
    if inventory_keys != sorted(set(inventory_keys)):
        raise MeasurementLineageError(
            "measurement-lineage attempt inventory is not sorted and unique"
        )
    reachability = payload.get("reachability_certificate")
    if (
        not isinstance(reachability, Mapping)
        or not isinstance(reachability.get("sha256"), str)
        or reachability.get("sha256")
        != _digest(
            {
                key: value
                for key, value in reachability.items()
                if key != "sha256"
            }
        )
        or payload.get("reachability_certificate_sha256")
        != _digest(reachability)
    ):
        raise MeasurementLineageError(
            "measurement-lineage recurrence reachability certificate is invalid"
        )
    assert isinstance(reachability, Mapping)
    if payload.get("impact") == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT:
        _validate_summary_cap_certificate(reachability)
        return
    if (
        reachability.get("algorithm")
        != "authenticated-recurrence-active-executor-reachability-v1"
        or reachability.get("target_contract_ids") != sorted(_HZZ_CONTRACT_IDS)
    ):
        raise MeasurementLineageError(
            "measurement-lineage recurrence reachability certificate is invalid"
        )
    records = reachability.get("records")
    reached_ids = reachability.get("reached_cell_ids")
    existing_target_ids = reachability.get("existing_catalog_target_ids")
    invalidated_failure_ids = reachability.get(
        "invalidated_validation_failed_current_ids"
    )
    if (
        not isinstance(records, list)
        or not isinstance(reached_ids, list)
        or not isinstance(existing_target_ids, list)
        or not isinstance(invalidated_failure_ids, list)
        or reachability.get("inspected_current_count") != len(records)
    ):
        raise MeasurementLineageError(
            "measurement-lineage recurrence reachability inventory is malformed"
        )
    record_ids: list[str] = []
    derived_reached: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != _REACHABILITY_RECORD_KEYS:
            raise MeasurementLineageError(
                "measurement-lineage recurrence reachability record is malformed"
            )
        _validate_artifact_owner(record)
        cell_id = record.get("cell_id")
        attempt_id = record.get("attempt_id")
        try:
            canonical_attempt = (
                str(uuid.UUID(attempt_id))
                if isinstance(attempt_id, str)
                else None
            )
        except ValueError:
            canonical_attempt = None
        safe_paths = (
            record.get("artifact_locator"),
            record.get("execution_manifest_path"),
            record.get("schedule_path"),
            record.get("process_binding_path"),
            record.get("kernel_pack_path"),
        )
        digest_fields = (
            "manifest_sha256",
            "execution_manifest_sha256",
            "schedule_sha256",
            "schedule_index_sha256",
            "schedule_member_sha256",
            "process_binding_sha256",
            "kernel_pack_sha256",
            "semantic_catalog_sha256",
            "direct_catalog_sha256",
        )
        executor_ids = record.get("active_executor_ids")
        templates = record.get("matched_template_ids")
        contracts = record.get("matched_contract_ids")
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or canonical_attempt != attempt_id
            or not isinstance(record.get("process_id"), str)
            or not record["process_id"]
            or any(
                not isinstance(value, str)
                or not value
                or Path(value).is_absolute()
                or any(
                    part in {"", ".", ".."} for part in Path(value).parts
                )
                for value in safe_paths
            )
            or any(
                not isinstance(record.get(field), str)
                or _SHA256_RE.fullmatch(str(record[field])) is None
                for field in digest_fields
            )
            or not isinstance(executor_ids, list)
            or executor_ids != sorted(set(executor_ids))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in executor_ids
            )
            or not isinstance(templates, list)
            or templates != sorted(set(templates))
            or any(not isinstance(value, str) or not value for value in templates)
            or not isinstance(contracts, list)
            or contracts != sorted(set(contracts))
            or any(value not in _HZZ_CONTRACT_IDS for value in contracts)
        ):
            raise MeasurementLineageError(
                "measurement-lineage recurrence reachability record is invalid"
            )
        record_ids.append(cell_id)
        if contracts:
            derived_reached.append(cell_id)
    if (
        record_ids != sorted(set(record_ids))
        or reached_ids != sorted(derived_reached)
        or existing_target_ids != sorted(set(existing_target_ids))
        or invalidated_failure_ids
        != sorted(set(invalidated_failure_ids))
        or any(
            not isinstance(value, str) or value not in set(record_ids)
            for value in existing_target_ids
        )
        or any(
            not isinstance(value, str) or not value
            for value in invalidated_failure_ids
        )
        or set(invalidated_failure_ids) & set(record_ids)
    ):
        raise MeasurementLineageError(
            "measurement-lineage recurrence reachability sets are inconsistent"
        )


def _load_envelope(path: Path, *, expected_state: str | None = None) -> dict[str, Any]:
    raw = _read_canonical_json(path, context="measurement-lineage envelope")
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema", "payload", "payload_sha256"}
        or raw.get("schema") != MEASUREMENT_LINEAGE_WRAPPER_SCHEMA
        or not isinstance(raw.get("payload"), dict)
        or not isinstance(raw.get("payload_sha256"), str)
        or raw["payload_sha256"] != _digest(raw["payload"])
    ):
        raise MeasurementLineageError(
            "measurement-lineage envelope or canonical digest is invalid"
        )
    payload = dict(raw["payload"])
    if payload.get("schema") != MEASUREMENT_LINEAGE_SCHEMA:
        raise MeasurementLineageError("measurement-lineage schema is incompatible")
    _validate_payload_shape(payload)
    if expected_state is not None and payload.get("state") != expected_state:
        raise MeasurementLineageError(
            f"measurement lineage is not in state {expected_state!r}"
        )
    return payload


def _git(
    repo_root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repo_root,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise MeasurementLineageError(f"cannot inspect Git lineage: {error}") from error
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise MeasurementLineageError(
            f"git {' '.join(arguments)} failed: "
            f"{detail or f'exit {completed.returncode}'}"
        )
    return completed


def _git_commit(repo_root: Path, revision: str) -> str:
    value = (
        _git(repo_root, "rev-parse", "--verify", f"{revision}^{{commit}}")
        .stdout.decode("ascii")
        .strip()
    )
    if _GIT_SHA_RE.fullmatch(value) is None:
        raise MeasurementLineageError("Git returned an invalid commit identity")
    return value


def _git_tree(repo_root: Path, revision: str) -> str:
    value = (
        _git(repo_root, "rev-parse", "--verify", f"{revision}^{{tree}}")
        .stdout.decode("ascii")
        .strip()
    )
    if _GIT_SHA_RE.fullmatch(value) is None:
        raise MeasurementLineageError("Git returned an invalid tree identity")
    return value


def _tree_member(
    repo_root: Path,
    revision: str,
    path: str,
) -> dict[str, str] | None:
    raw = _git(repo_root, "ls-tree", "-z", revision, "--", path).stdout
    if not raw:
        return None
    record = raw.split(b"\0", 1)[0]
    header, separator, recorded_path = record.partition(b"\t")
    fields = header.split()
    if (
        not separator
        or os.fsdecode(recorded_path) != path
        or len(fields) != 3
        or fields[1] != b"blob"
    ):
        raise MeasurementLineageError(f"Git tree metadata is malformed for {path!r}")
    return {
        "mode": fields[0].decode("ascii"),
        "object": fields[2].decode("ascii"),
    }


def _blob_sha256(repo_root: Path, revision: str, path: str) -> str | None:
    member = _tree_member(repo_root, revision, path)
    if member is None:
        return None
    return hashlib.sha256(
        _git(repo_root, "show", f"{revision}:{path}").stdout
    ).hexdigest()


def _diff_records(
    repo_root: Path,
    *,
    ancestor: str,
    descendant: str,
    impact: str,
) -> tuple[dict[str, object], ...]:
    if impact == CLASS_C_HZZ_IMPACT:
        allowed_paths = _HZZ_ALLOWED_PATHS
        required_paths = _HZZ_REQUIRED_FEATURE_PATHS
        impact_label = "HZZ"
    elif impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT:
        allowed_paths = _RECURRENCE_SUMMARY_CAP_ALLOWED_PATHS
        required_paths = _RECURRENCE_SUMMARY_CAP_REQUIRED_PATHS
        impact_label = "recurrence-summary-cap"
    else:
        raise MeasurementLineageError(f"unsupported Class-C impact {impact!r}")
    ancestry = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        ancestor,
        descendant,
        check=False,
    )
    if ancestry.returncode != 0:
        raise MeasurementLineageError(
            "Class-C descendant is not a direct Git descendant of its ancestor"
        )
    tokens = [
        os.fsdecode(value)
        for value in _git(
            repo_root,
            "diff",
            "--name-status",
            "-z",
            "--no-renames",
            f"{ancestor}..{descendant}",
            "--",
        ).stdout.split(b"\0")
        if value
    ]
    if len(tokens) % 2:
        raise MeasurementLineageError("Git returned a malformed Class-C diff")
    records: list[dict[str, object]] = []
    for offset in range(0, len(tokens), 2):
        status, path = tokens[offset : offset + 2]
        if status not in {"A", "M"}:
            raise MeasurementLineageError(
                f"Class-C diff contains unsupported status {status!r} for {path!r}"
            )
        if path not in allowed_paths:
            raise MeasurementLineageError(
                f"Class-C {impact_label} descendant changes disallowed path {path!r}"
            )
        old = _tree_member(repo_root, ancestor, path)
        new = _tree_member(repo_root, descendant, path)
        if new is None or new["mode"] != "100644":
            raise MeasurementLineageError(
                f"Class-C descendant member is not a regular 0644 file: {path!r}"
            )
        if old is not None and old["mode"] != "100644":
            raise MeasurementLineageError(
                f"Class-C ancestor member is not a regular 0644 file: {path!r}"
            )
        records.append(
            {
                "status": status,
                "path": path,
                "ancestor_blob": None if old is None else old["object"],
                "ancestor_sha256": _blob_sha256(repo_root, ancestor, path),
                "descendant_blob": new["object"],
                "descendant_sha256": _blob_sha256(repo_root, descendant, path),
                "mode": new["mode"],
            }
        )
    if not records:
        raise MeasurementLineageError("Class-C descendant has an empty Git diff")
    changed = {str(record["path"]) for record in records}
    if not required_paths.issubset(changed):
        missing = sorted(required_paths - changed)
        raise MeasurementLineageError(
            f"{impact_label} Class-C descendant lacks reviewed members: "
            + ", ".join(missing)
        )
    return tuple(records)


def _cell_record(cell: CellSpec) -> dict[str, object]:
    return {
        "cell_id": cell.cell_id,
        "dataset_id": cell.dataset_id,
        "process": cell.process,
        "process_key": cell.process_key,
        "n_final": cell.n_final,
        "mode": cell.measurement.execution_mode.value,
        "model": (
            None if cell.measurement.model is None else cell.measurement.model.value
        ),
        "accuracy": cell.measurement.accuracy.value,
        "backend": cell.measurement.backend,
        "jit_optimization_level": cell.measurement.jit_optimization_level,
        "workload": cell.workload.value,
        "variant": cell.variant,
    }


def hzz_impacted_cells(
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[CellSpec, ...]:
    """Return the exact built-in recurrence cells affected by the HZZ fix."""

    cells = tuple(
        sorted(
            (
                cell
                for cell in catalog.measurement_cells()
                if (
                    cell.measurement.execution_mode is ExecutionMode.RECURRENCE
                    and cell.measurement.model is ModelKey.BUILTIN_SM
                    and cell.process_key == "dd_zzz_jets"
                    and cell.n_final >= 3
                )
            ),
            key=lambda item: item.cell_id,
        )
    )
    if catalog is REPORT_CATALOG and len(cells) != 20:
        raise MeasurementLineageError(
            f"canonical HZZ impact census changed from 20 to {len(cells)} cells"
        )
    return cells


def hzz_agreement_closure(
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[CellSpec, ...]:
    """Return every direct-agreement consumer of an affected HZZ cell."""

    impacted = hzz_impacted_cells(catalog=catalog)
    impacted_ids = {cell.cell_id for cell in impacted}
    impacted_keys = {
        (
            cell.process,
            cell.process_key,
            cell.n_final,
            cell.measurement.accuracy,
            cell.workload,
        )
        for cell in impacted
    }
    peers = tuple(
        sorted(
            (
                cell
                for cell in catalog.measurement_cells()
                if (
                    cell.measurement.execution_mode is ExecutionMode.RECURRENCE
                    and cell.measurement.model is ModelKey.UFO_SM
                    and (
                        cell.process,
                        cell.process_key,
                        cell.n_final,
                        cell.measurement.accuracy,
                        cell.workload,
                    )
                    in impacted_keys
                    and any(
                        edge.baseline.cell_id in impacted_ids
                        for edge in incoming_agreement_edges(
                            cell,
                            catalog=catalog,
                        )
                    )
                )
            ),
            key=lambda item: item.cell_id,
        )
    )
    if catalog is REPORT_CATALOG and (
        len(peers) != 20
        or any(
            cell.measurement.execution_mode is not ExecutionMode.RECURRENCE
            or cell.measurement.model is not ModelKey.UFO_SM
            or cell.process_key != "dd_zzz_jets"
            for cell in peers
        )
    ):
        raise MeasurementLineageError(
            "canonical HZZ direct-agreement closure is no longer the 20 UFO "
            "recurrence peers"
        )
    return peers


def recurrence_summary_cap_impacted_cells(
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[CellSpec, ...]:
    """Return the four currents blocked by the old recurrence summary cap."""

    by_id = {cell.cell_id: cell for cell in catalog.measurement_cells()}
    missing = sorted(set(_RECURRENCE_SUMMARY_CAP_FAILURE_BYTES) - set(by_id))
    if missing:
        raise MeasurementLineageError(
            "recurrence summary-cap targets are absent from the catalog: "
            + ", ".join(missing)
        )
    cells = tuple(
        by_id[cell_id]
        for cell_id in sorted(_RECURRENCE_SUMMARY_CAP_FAILURE_BYTES)
    )
    if any(
        cell.measurement.execution_mode is not ExecutionMode.RECURRENCE
        or cell.measurement.model is not ModelKey.BUILTIN_SM
        or cell.measurement.accuracy is not Accuracy.LC
        or cell.workload is not Workload.SELECTED_FLOW
        for cell in cells
    ):
        raise MeasurementLineageError(
            "recurrence summary-cap target semantics changed in the catalog"
        )
    return cells


def recurrence_summary_cap_agreement_closure(
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[CellSpec, ...]:
    """Return the exact-equivalent and dependent peers of the four targets."""

    closure = _agreement_consumer_closure(
        {
            cell.cell_id
            for cell in recurrence_summary_cap_impacted_cells(catalog=catalog)
        },
        catalog=catalog,
    )
    if catalog is REPORT_CATALOG and {
        cell.cell_id for cell in closure
    } != _RECURRENCE_SUMMARY_CAP_AGREEMENT_IDS:
        raise MeasurementLineageError(
            "canonical recurrence summary-cap agreement closure changed from "
            "its reviewed 12-cell census"
        )
    return closure


def signed_zero_helicity_impacted_cells(
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[CellSpec, ...]:
    """Return the 24 cells directly blocked by legacy unsigned zero."""

    by_id = {cell.cell_id: cell for cell in catalog.measurement_cells()}
    missing = sorted(_SIGNED_ZERO_HELICITY_IMPACTED_IDS - set(by_id))
    if missing:
        raise MeasurementLineageError(
            "signed-zero helicity targets are absent from the catalog: "
            + ", ".join(missing)
        )
    cells = tuple(
        by_id[cell_id]
        for cell_id in sorted(_SIGNED_ZERO_HELICITY_IMPACTED_IDS)
    )
    if any(
        cell.measurement.accuracy is not Accuracy.LC
        or cell.workload not in {Workload.SELECTED_FLOW, Workload.ALL_FLOW}
        or cell.process_key not in {"dd_epemzh_jets", "dd_ttzh_jets"}
        or cell.n_final not in range(4, 8)
        or (
            cell.measurement.execution_mode is ExecutionMode.AMPLICOL
            and cell.measurement.model is not None
        )
        or (
            cell.measurement.execution_mode is ExecutionMode.RECURRENCE
            and (
                cell.measurement.model is not ModelKey.BUILTIN_SM
                or cell.workload is not Workload.SELECTED_FLOW
            )
        )
        or cell.measurement.execution_mode
        not in {ExecutionMode.AMPLICOL, ExecutionMode.RECURRENCE}
        for cell in cells
    ):
        raise MeasurementLineageError(
            "signed-zero helicity target semantics changed in the catalog"
        )
    return cells


def signed_zero_helicity_agreement_closure(
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[CellSpec, ...]:
    """Return the exact 40-cell dependent closure of the signed-zero targets."""

    closure = _agreement_consumer_closure(
        {
            cell.cell_id
            for cell in signed_zero_helicity_impacted_cells(catalog=catalog)
        },
        catalog=catalog,
    )
    if catalog is REPORT_CATALOG and {
        cell.cell_id for cell in closure
    } != _SIGNED_ZERO_HELICITY_AGREEMENT_IDS:
        raise MeasurementLineageError(
            "canonical signed-zero agreement closure changed from its reviewed "
            "40-cell census"
        )
    return closure


def _agreement_consumer_closure(
    affected_cell_ids: set[str],
    *,
    catalog: ReportCatalog,
) -> tuple[CellSpec, ...]:
    """Return the transitive incoming-agreement closure of ``affected_cell_ids``."""

    affected = set(affected_cell_ids)
    consumers: dict[str, CellSpec] = {}
    changed = True
    while changed:
        changed = False
        for cell in catalog.measurement_cells():
            if cell.cell_id in affected:
                continue
            if any(
                edge.baseline.cell_id in affected
                for edge in incoming_agreement_edges(cell, catalog=catalog)
            ):
                affected.add(cell.cell_id)
                consumers[cell.cell_id] = cell
                changed = True
    return tuple(
        sorted(consumers.values(), key=lambda item: item.cell_id)
    )


def _impact_and_agreement_cells(
    impact: str,
    reachability: Mapping[str, object],
    *,
    catalog: ReportCatalog,
) -> tuple[tuple[CellSpec, ...], tuple[CellSpec, ...]]:
    if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT:
        impacted = {
            cell.cell_id: cell
            for cell in (
                *recurrence_summary_cap_impacted_cells(catalog=catalog),
                *signed_zero_helicity_impacted_cells(catalog=catalog),
            )
        }
        return (
            tuple(sorted(impacted.values(), key=lambda item: item.cell_id)),
            _agreement_consumer_closure(set(impacted), catalog=catalog),
        )
    if impact != CLASS_C_HZZ_IMPACT:
        raise MeasurementLineageError(f"unsupported Class-C impact {impact!r}")
    by_id = {cell.cell_id: cell for cell in catalog.measurement_cells()}
    raw_reached = reachability.get("reached_cell_ids")
    if not isinstance(raw_reached, list) or any(
        not isinstance(value, str) or value not in by_id for value in raw_reached
    ):
        raise MeasurementLineageError(
            "reachability certificate has unknown reached cells"
        )
    affected_by_id = {
        cell.cell_id: cell for cell in hzz_impacted_cells(catalog=catalog)
    }
    affected_by_id.update({cell_id: by_id[cell_id] for cell_id in raw_reached})
    affected = tuple(
        sorted(affected_by_id.values(), key=lambda item: item.cell_id)
    )
    closure = _agreement_consumer_closure(
        set(affected_by_id),
        catalog=catalog,
    )
    return affected, closure


def _catalog_digest(catalog: ReportCatalog) -> str:
    global _CANONICAL_CATALOG_SHA256
    if catalog is REPORT_CATALOG and _CANONICAL_CATALOG_SHA256 is not None:
        return _CANONICAL_CATALOG_SHA256
    result = _digest([_cell_record(cell) for cell in catalog.measurement_cells()])
    if catalog is REPORT_CATALOG:
        _CANONICAL_CATALOG_SHA256 = result
    return result


def _agreement_digest(catalog: ReportCatalog) -> str:
    global _CANONICAL_AGREEMENT_SHA256
    if catalog is REPORT_CATALOG and _CANONICAL_AGREEMENT_SHA256 is not None:
        return _CANONICAL_AGREEMENT_SHA256
    edges: list[dict[str, str]] = []
    for cell in catalog.measurement_cells():
        for edge in incoming_agreement_edges(cell, catalog=catalog):
            edges.append(
                {
                    "candidate": cell.cell_id,
                    "baseline": edge.baseline.cell_id,
                    "kind": edge.kind,
                }
            )
    result = _digest(sorted(edges, key=lambda item: tuple(item.values())))
    if catalog is REPORT_CATALOG:
        _CANONICAL_AGREEMENT_SHA256 = result
    return result


def _environment(path: Path, *, expected_profile: str) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MeasurementLineageError(
            f"cannot read profile environment {path}: {error}"
        ) from error
    if (
        not isinstance(raw, dict)
        or set(raw) != _ENVIRONMENT_KEYS
        or raw.get("profile") != expected_profile
        or raw.get("status") != "authenticated"
        or not all(isinstance(value, str) for value in raw.values())
    ):
        raise MeasurementLineageError(
            "profile environment is not an authenticated string mapping"
        )
    return dict(raw)


def _profile_name(repo_root: Path, docs_dir: Path) -> str:
    try:
        relative = docs_dir.resolve(strict=False).relative_to(
            (repo_root / _PROFILE_ROOT).resolve(strict=False)
        )
    except ValueError as error:
        raise MeasurementLineageError(
            "Class-C bridges require an architecture-specific report profile"
        ) from error
    if len(relative.parts) != 1 or relative.parts[0] in {".", ".."}:
        raise MeasurementLineageError(
            "Class-C report profile path is not one safe component"
        )
    return relative.parts[0]


def measurement_lineage_path(repo_root: Path, docs_dir: Path) -> Path:
    _profile_name(repo_root, docs_dir)
    return docs_dir / MEASUREMENT_LINEAGE_FILENAME


def class_c_pending_path(
    store: ArtifactStore,
    *,
    ancestor_revision: str,
    descendant_revision: str,
) -> Path:
    """Return the only permitted pending-bridge locator for an A→D pair."""

    if (
        _GIT_SHA_RE.fullmatch(ancestor_revision) is None
        or _GIT_SHA_RE.fullmatch(descendant_revision) is None
    ):
        raise MeasurementLineageError(
            "pending Class-C bridge identities must be full Git object IDs"
        )
    return (
        store.artifact_root
        / "source-bridges"
        / f"{ancestor_revision}-{descendant_revision}.pending.json"
    )


def _validated_pending_path(store: ArtifactStore, path: Path) -> Path:
    candidate = path.expanduser().absolute()
    root = (store.artifact_root / "source-bridges").absolute()
    if (
        candidate.parent != root
        or root.is_symlink()
        or not root.is_dir()
        or candidate.is_symlink()
        or not candidate.is_file()
    ):
        raise MeasurementLineageError(
            "pending Class-C bridge is not a regular canonical artifact-root member"
        )
    try:
        candidate.resolve(strict=True).relative_to(
            store.artifact_root.resolve(strict=True)
        )
    except (OSError, ValueError) as error:
        raise MeasurementLineageError(
            "pending Class-C bridge escapes the profile artifact root"
        ) from error
    return candidate


def _current_pin(
    record: CurrentRecord,
    *,
    source_epoch_fallback: tuple[str, str] | None = None,
) -> dict[str, object]:
    provenance = record.result.get("provenance")
    if not isinstance(provenance, Mapping) and source_epoch_fallback is None:
        raise MeasurementLineageError(
            f"current {record.cell_id!r} has no source provenance"
        )
    if isinstance(provenance, Mapping):
        revision = provenance.get("report_source_revision")
        tree = provenance.get("report_source_tree")
        valid = (
            isinstance(revision, str)
            and _GIT_SHA_RE.fullmatch(revision) is not None
            and isinstance(tree, str)
            and _GIT_SHA_RE.fullmatch(tree) is not None
            and provenance.get("report_measured_source_revision") == revision
            and provenance.get("report_measured_source_tree") == tree
        )
    else:
        revision, tree = source_epoch_fallback  # type: ignore[misc]
        valid = (
            _GIT_SHA_RE.fullmatch(revision) is not None
            and _GIT_SHA_RE.fullmatch(tree) is not None
        )
    if not valid:
        raise MeasurementLineageError(
            f"current {record.cell_id!r} has invalid source provenance"
        )
    pointer = _current_pointer_identity(record)
    return {
        "cell_id": record.cell_id,
        "attempt_id": record.attempt_id,
        "manifest_sha256": record.manifest_sha256,
        **pointer,
        "source_revision": revision,
        "source_tree": tree,
        "result_sha256": _file_digest(record.result_path),
    }


def _current_pointer_identity(record: CurrentRecord) -> dict[str, str]:
    current_path = record.manifest_path.parent.parent.parent / "current.json"
    return {
        "current_locator": current_path.relative_to(
            record.manifest_path.parents[4]
        ).as_posix(),
        "current_pointer_sha256": _file_digest(current_path),
    }


def _artifact_canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise MeasurementLineageError(
            f"attempt manifest is not canonical JSON: {error}"
        ) from error
    return f"{encoded}\n".encode()


def _read_attempt_manifest(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise MeasurementLineageError(f"attempt has no regular manifest: {path.parent}")
    try:
        data = path.read_bytes()
        raw = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MeasurementLineageError(
            f"cannot read attempt manifest {path}: {error}"
        ) from error
    if not isinstance(raw, dict) or data != _artifact_canonical_bytes(raw):
        raise MeasurementLineageError(
            f"attempt manifest is not canonical JSON: {path}"
        )
    return raw


def _attempt_member(
    attempt_root: Path,
    relative: object,
    *,
    context: str,
) -> tuple[str, Path]:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(relative).parts)
    ):
        raise MeasurementLineageError(f"{context} is not a safe relative path")
    candidate = attempt_root.joinpath(*Path(relative).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(attempt_root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise MeasurementLineageError(
            f"{context} is unavailable or escapes its attempt"
        ) from error
    if candidate.is_symlink() or not resolved.is_file():
        raise MeasurementLineageError(f"{context} is not a regular file")
    return Path(relative).as_posix(), resolved


def _validated_attempt_inventory_row(
    store: ArtifactStore,
    path: Path,
    *,
    expected_cell_id: str,
) -> dict[str, object]:
    raw = _read_attempt_manifest(path)
    attempt_id = path.parent.name
    try:
        parsed = uuid.UUID(attempt_id)
    except ValueError as error:
        raise MeasurementLineageError(
            f"attempt directory is not a UUID: {path.parent}"
        ) from error
    if (
        str(parsed) != attempt_id
        or set(raw) != _ATTEMPT_KEYS
        or raw.get("schema") != ATTEMPT_SCHEMA
        or raw.get("cell_id") != expected_cell_id
        or raw.get("attempt_id") != attempt_id
    ):
        raise MeasurementLineageError(
            f"attempt manifest identity or schema is invalid: {path}"
        )
    try:
        ArtifactPolicy(raw.get("artifact_policy"))
    except (TypeError, ValueError) as error:
        raise MeasurementLineageError(
            f"attempt manifest has an invalid artifact policy: {path}"
        ) from error
    based_on = raw.get("based_on")
    if based_on is not None:
        if (
            not isinstance(based_on, dict)
            or set(based_on) != {"attempt_id", "manifest_sha256"}
            or not isinstance(based_on.get("attempt_id"), str)
            or not isinstance(based_on.get("manifest_sha256"), str)
            or _SHA256_RE.fullmatch(str(based_on["manifest_sha256"])) is None
        ):
            raise MeasurementLineageError(
                f"attempt manifest has an invalid based_on identity: {path}"
            )
        try:
            if str(uuid.UUID(str(based_on["attempt_id"]))) != based_on["attempt_id"]:
                raise ValueError
        except ValueError as error:
            raise MeasurementLineageError(
                f"attempt manifest has an invalid based_on UUID: {path}"
            ) from error

    status = raw.get("status")
    error = raw.get("error")
    result_path = raw.get("result_path")
    if status == "ok":
        if error is not None or result_path is None:
            raise MeasurementLineageError(
                f"successful attempt has inconsistent result/error fields: {path}"
            )
    elif status in {"failed", "interrupted"}:
        if not isinstance(error, str) or not error or result_path is not None:
            raise MeasurementLineageError(
                f"unsuccessful attempt has inconsistent result/error fields: {path}"
            )
    else:
        raise MeasurementLineageError(
            f"attempt manifest has an unsupported status: {path}"
        )

    raw_artifacts = raw.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise MeasurementLineageError(
            f"attempt manifest artifacts are not an array: {path}"
        )
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_artifacts):
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise MeasurementLineageError(
                f"attempt artifact {index} has an invalid schema: {path}"
            )
        relative, member = _attempt_member(
            path.parent,
            item.get("path"),
            context=f"attempt artifact {index}",
        )
        size = item.get("size")
        sha256 = item.get("sha256")
        if (
            relative in seen
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or member.stat().st_size != size
            or not isinstance(sha256, str)
            or _SHA256_RE.fullmatch(sha256) is None
            or _file_digest(member) != sha256
        ):
            raise MeasurementLineageError(
                f"attempt artifact {index} differs from its manifest: {path}"
            )
        seen.add(relative)
        records.append({"path": relative, "size": size, "sha256": sha256})
    if records != sorted(records, key=lambda item: str(item["path"])):
        raise MeasurementLineageError(
            f"attempt artifact records are not canonically sorted: {path}"
        )
    if status == "ok" and (not records or result_path not in seen):
        raise MeasurementLineageError(
            f"successful attempt result is not an authenticated artifact: {path}"
        )
    observed_files: set[str] = set()
    for member in path.parent.rglob("*"):
        if member.is_symlink():
            raise MeasurementLineageError(
                f"attempt contains a symbolic link: {member}"
            )
        if member.is_file():
            observed_files.add(member.relative_to(path.parent).as_posix())
    if observed_files != seen | {"manifest.json"}:
        raise MeasurementLineageError(
            f"attempt contains untracked or missing artifact members: {path}"
        )
    result_sha256: str | None = None
    if result_path is not None:
        _relative, result_file = _attempt_member(
            path.parent,
            result_path,
            context="attempt result",
        )
        result_sha256 = _file_digest(result_file)
    return {
        "cell_id": expected_cell_id,
        "attempt_id": attempt_id,
        "manifest_locator": path.relative_to(store.artifact_root).as_posix(),
        "manifest_sha256": _file_digest(path),
        "status": status,
        "error_sha256": _digest(error),
        "result_path": result_path,
        "result_sha256": result_sha256,
        "artifact_count": len(records),
        "artifacts_sha256": _digest(records),
    }


def _attempt_inventory(
    store: ArtifactStore,
    *,
    catalog: ReportCatalog,
) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
    """Snapshot every catalog attempt, including failed and orphan attempts."""

    catalog_cells = tuple(catalog.measurement_cells())
    catalog_ids = {cell.cell_id for cell in catalog_cells}
    roots = {store._cell_root(cell_id): cell_id for cell_id in catalog_ids}
    if store.cells_root.is_symlink() or not store.cells_root.is_dir():
        raise MeasurementLineageError(
            f"artifact cells root is not a regular directory: {store.cells_root}"
        )
    for path in store.cells_root.iterdir():
        if path not in roots:
            raise MeasurementLineageError(
                f"artifact store contains an unknown catalog cell directory: {path}"
            )
        if path.is_symlink() or not path.is_dir():
            raise MeasurementLineageError(
                f"artifact cell root is not a regular directory: {path}"
            )

    inventory: list[dict[str, object]] = []
    no_attempts: list[str] = []
    for cell_id in sorted(catalog_ids):
        cell_root = store._cell_root(cell_id)  # same-package authenticated layout
        attempts_root = cell_root / "attempts"
        if attempts_root.exists() and (
            attempts_root.is_symlink() or not attempts_root.is_dir()
        ):
            raise MeasurementLineageError(
                f"attempt root is not a regular directory: {attempts_root}"
            )
        attempt_roots = (
            ()
            if not attempts_root.exists()
            else tuple(sorted(attempts_root.iterdir()))
        )
        manifests: list[Path] = []
        for attempt_root in attempt_roots:
            if attempt_root.is_symlink() or not attempt_root.is_dir():
                raise MeasurementLineageError(
                    f"attempt entry is not a regular directory: {attempt_root}"
                )
            manifest = attempt_root / "manifest.json"
            if manifest.is_symlink() or not manifest.is_file():
                raise MeasurementLineageError(
                    f"attempt has no regular manifest: {attempt_root}"
                )
            manifests.append(manifest)
        if not manifests:
            if store.load_current(cell_id, missing_ok=True) is not None:
                raise MeasurementLineageError(
                    f"cell {cell_id!r} has a current pointer but no attempt manifest"
                )
            no_attempts.append(cell_id)
            continue
        for path in manifests:
            inventory.append(
                _validated_attempt_inventory_row(
                    store,
                    path,
                    expected_cell_id=cell_id,
                )
            )
    return (
        tuple(
            sorted(
                inventory,
                key=lambda item: (item["cell_id"], item["attempt_id"]),
            )
        ),
        tuple(no_attempts),
    )


RecurrenceReachabilityInspector = Callable[
    [CurrentRecord, CellSpec],
    Mapping[str, object],
]


def _runtime_identity_digest(value: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise MeasurementLineageError(
            f"runtime identity is not canonical JSON: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _runtime_identity_evidence(
    record: CurrentRecord,
) -> tuple[str, str]:
    provenance = record.result.get("provenance")
    identity = (
        provenance.get("runtime_identity")
        if isinstance(provenance, Mapping)
        else None
    )
    if not isinstance(identity, Mapping):
        raise MeasurementLineageError(
            f"{record.cell_id}: current has no digest-covered runtime identity"
        )
    identity_digest = _runtime_identity_digest(identity)
    stable = dict(identity)
    policy = stable.get("loaded_module_origin_policy")
    if isinstance(policy, Mapping):
        stable_policy = dict(policy)
        for field in (
            "observed_module_count",
            "observations",
            "observations_sha256",
        ):
            stable_policy.pop(field, None)
        stable["loaded_module_origin_policy"] = stable_policy
    stable_digest = _runtime_identity_digest(stable)
    if (
        provenance.get("runtime_identity_sha256") != identity_digest
        or provenance.get("runtime_identity_stable_sha256") != stable_digest
        or provenance.get("runtime_identity_postflight_stable_sha256")
        != stable_digest
        or provenance.get("runtime_identity_postflight_match") is not True
    ):
        raise MeasurementLineageError(
            f"{record.cell_id}: runtime identity digest or postflight binding changed"
        )
    return identity_digest, stable_digest


def _source_epoch(record: CurrentRecord) -> tuple[str, str]:
    provenance = record.result.get("provenance")
    revision = (
        provenance.get("report_source_revision")
        if isinstance(provenance, Mapping)
        else None
    )
    tree = (
        provenance.get("report_source_tree")
        if isinstance(provenance, Mapping)
        else None
    )
    if (
        not isinstance(revision, str)
        or _GIT_SHA_RE.fullmatch(revision) is None
        or not isinstance(tree, str)
        or _GIT_SHA_RE.fullmatch(tree) is None
        or provenance.get("report_measured_source_revision") != revision
        or provenance.get("report_measured_source_tree") != tree
    ):
        raise MeasurementLineageError(
            f"{record.cell_id}: current has no exact measured source epoch"
        )
    return revision, tree


def _attempt_owner_evidence(
    store: ArtifactStore,
    record: CurrentRecord,
) -> tuple[dict[str, object], str]:
    try:
        record.manifest_path.resolve(strict=True).relative_to(
            store.artifact_root.resolve(strict=True)
        )
    except (OSError, ValueError) as error:
        raise MeasurementLineageError(
            f"{record.cell_id}: attempt is outside this profile artifact store"
        ) from error
    inventory = _validated_attempt_inventory_row(
        store,
        record.manifest_path,
        expected_cell_id=record.cell_id,
    )
    if (
        inventory.get("status") != "ok"
        or inventory.get("attempt_id") != record.attempt_id
        or inventory.get("manifest_sha256") != record.manifest_sha256
        or inventory.get("result_sha256") != _file_digest(record.result_path)
    ):
        raise MeasurementLineageError(
            f"{record.cell_id}: attempt/result identity differs from its manifest"
        )
    result_locator = record.result_path.relative_to(
        store.artifact_root
    ).as_posix()
    return inventory, result_locator


def _resolve_recurrence_artifact_owner(
    store: ArtifactStore,
    catalog: ReportCatalog,
    record: CurrentRecord,
    cell: CellSpec,
    artifact_root: Path,
    process_id: str,
) -> dict[str, object]:
    """Authenticate a direct artifact or one bounded matrix-peer reuse."""

    consumer_inventory, consumer_result_locator = _attempt_owner_evidence(
        store,
        record,
    )
    consumer_revision, consumer_tree = _source_epoch(record)
    consumer_runtime, consumer_runtime_stable = _runtime_identity_evidence(record)
    store_root = store.artifact_root.resolve(strict=True)
    try:
        artifact_locator = artifact_root.relative_to(store_root).as_posix()
    except ValueError as error:
        raise MeasurementLineageError(
            f"{cell.cell_id}: recurrence artifact is outside this profile store"
        ) from error

    consumer_attempt = record.manifest_path.parent.resolve(strict=True)
    try:
        artifact_root.relative_to(consumer_attempt)
    except ValueError:
        relation = "equivalent-matrix-peer"
    else:
        try:
            expected_consumer_artifact = (
                consumer_attempt / "artifact"
            ).resolve(strict=True)
        except OSError as error:
            raise MeasurementLineageError(
                f"{cell.cell_id}: direct recurrence artifact is unavailable"
            ) from error
        if artifact_root != expected_consumer_artifact:
            raise MeasurementLineageError(
                f"{cell.cell_id}: direct recurrence artifact is not the canonical "
                "attempt/artifact directory"
            )
        relation = "consumer-attempt"

    owner = record
    owner_current_locator: str | None = None
    owner_current_sha256: str | None = None
    if relation == "equivalent-matrix-peer":
        peers = tuple(
            peer
            for peer in catalog.equivalent_cells(cell)
            if peer.dataset_id.startswith("matrix_")
        )
        if len(peers) != 1:
            raise MeasurementLineageError(
                f"{cell.cell_id}: recurrence artifact owner matrix peer is "
                f"missing or ambiguous"
            )
        try:
            loaded = store.load_current(peers[0].cell_id, missing_ok=True)
        except Exception as error:
            raise MeasurementLineageError(
                f"{cell.cell_id}: recurrence artifact owner current is invalid"
            ) from error
        if loaded is None:
            raise MeasurementLineageError(
                f"{cell.cell_id}: recurrence artifact owner current is missing"
            )
        owner = loaded
        owner_artifact = owner.result.get("artifact")
        owner_raw_path = (
            owner_artifact.get("path")
            if isinstance(owner_artifact, Mapping)
            else None
        )
        owner_process_id = (
            owner_artifact.get("process_id")
            if isinstance(owner_artifact, Mapping)
            else None
        )
        if (
            not isinstance(owner_raw_path, str)
            or not owner_raw_path
            or "${" in owner_raw_path
            or owner_process_id != process_id
        ):
            raise MeasurementLineageError(
                f"{cell.cell_id}: recurrence artifact owner result is malformed"
            )
        try:
            owner_artifact_root = Path(owner_raw_path).expanduser().resolve(
                strict=True
            )
            expected_owner_artifact = (
                owner.manifest_path.parent.resolve(strict=True) / "artifact"
            ).resolve(strict=True)
        except (OSError, ValueError) as error:
            raise MeasurementLineageError(
                f"{cell.cell_id}: recurrence artifact owner chains to another attempt"
            ) from error
        if (
            Path(owner_raw_path).expanduser().is_symlink()
            or not owner_artifact_root.is_dir()
            or owner_artifact_root != expected_owner_artifact
            or owner_artifact_root != artifact_root
        ):
            raise MeasurementLineageError(
                f"{cell.cell_id}: recurrence artifact is not owned by its matrix peer"
            )
        owner_pin = _current_pin(owner)
        owner_current_locator = str(owner_pin["current_locator"])
        owner_current_sha256 = str(owner_pin["current_pointer_sha256"])

    owner_inventory, owner_result_locator = _attempt_owner_evidence(store, owner)
    owner_revision, owner_tree = _source_epoch(owner)
    owner_runtime, owner_runtime_stable = _runtime_identity_evidence(owner)
    if (
        (owner_revision, owner_tree) != (consumer_revision, consumer_tree)
        or owner_runtime_stable != consumer_runtime_stable
    ):
        raise MeasurementLineageError(
            f"{cell.cell_id}: recurrence artifact owner belongs to another "
            "source/runtime epoch"
        )

    return {
        "relation": relation,
        "artifact_locator": artifact_locator,
        "consumer_cell_id": record.cell_id,
        "consumer_attempt_id": record.attempt_id,
        "consumer_manifest_locator": consumer_inventory["manifest_locator"],
        "consumer_manifest_sha256": record.manifest_sha256,
        "consumer_result_locator": consumer_result_locator,
        "consumer_result_sha256": consumer_inventory["result_sha256"],
        "consumer_source_revision": consumer_revision,
        "consumer_source_tree": consumer_tree,
        "consumer_runtime_identity_sha256": consumer_runtime,
        "consumer_runtime_identity_stable_sha256": consumer_runtime_stable,
        "owner_cell_id": owner.cell_id,
        "owner_attempt_id": owner.attempt_id,
        "owner_current_locator": owner_current_locator,
        "owner_current_sha256": owner_current_sha256,
        "owner_manifest_locator": owner_inventory["manifest_locator"],
        "owner_manifest_sha256": owner.manifest_sha256,
        "owner_result_locator": owner_result_locator,
        "owner_result_sha256": owner_inventory["result_sha256"],
        "owner_artifacts_sha256": owner_inventory["artifacts_sha256"],
        "owner_source_revision": owner_revision,
        "owner_source_tree": owner_tree,
        "owner_runtime_identity_sha256": owner_runtime,
        "owner_runtime_identity_stable_sha256": owner_runtime_stable,
    }


def _artifact_member(root: Path, relative: object, *, context: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(relative).parts)
    ):
        raise MeasurementLineageError(f"{context} is not a safe artifact member")
    candidate = root.joinpath(*Path(relative).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise MeasurementLineageError(
            f"{context} is unavailable or escapes its artifact"
        ) from error
    if candidate.is_symlink() or not resolved.is_file():
        raise MeasurementLineageError(f"{context} is not a regular file")
    return resolved


def _validate_numerical_validation_failure(
    record: CurrentRecord,
    cell: CellSpec,
) -> None:
    """Validate one completed pointwise failure selected for replacement."""

    validation = record.result.get("validation")
    failure = record.result.get("failure")
    artifact = record.result.get("artifact")
    provenance = record.result.get("provenance")
    try:
        validate_measurement(record.result, expected_cell=cell)
    except ValueError as error:
        raise MeasurementLineageError(
            f"{record.cell_id}: closure failure measurement schema is invalid"
        ) from error
    if (
        record.result.get("status") != ResultStatus.VALIDATION_FAILED.value
        or not isinstance(validation, Mapping)
        or validation.get("status")
        != ResultStatus.VALIDATION_FAILED.value
        or not isinstance(failure, Mapping)
        or set(failure) != {"kind", "message"}
        or failure.get("kind") != "MeasurementValidationError"
        or failure.get("message")
        != "candidate or same-artifact numerical validation failed"
        or not isinstance(artifact, Mapping)
        or set(artifact) != {"path", "process_id", "policy"}
        or not isinstance(artifact.get("path"), str)
        or not artifact["path"]
        or not isinstance(artifact.get("process_id"), str)
        or not artifact["process_id"]
        or artifact.get("policy") != "generated"
        or not isinstance(provenance, Mapping)
    ):
        raise MeasurementLineageError(
            f"{record.cell_id}: closure current is not an authenticated "
            "numerical-validation failure"
        )

    pointwise = validation.get("pointwise")
    resolved_sum = validation.get("resolved_sum")
    if (
        not isinstance(pointwise, Mapping)
        or set(pointwise)
        != {
            "status",
            "candidate",
            "baseline",
            "absolute_difference",
            "relative_difference",
            "relative_tolerance",
            "absolute_tolerance",
        }
        or not isinstance(resolved_sum, Mapping)
        or set(resolved_sum)
        != {
            "status",
            "maximum_absolute_difference",
            "maximum_relative_difference",
            "relative_tolerance",
            "absolute_tolerance",
        }
    ):
        raise MeasurementLineageError(
            f"{record.cell_id}: closure failure lacks complete pointwise/resolved "
            "validation evidence"
        )
    raw_numbers = (
        pointwise["candidate"],
        pointwise["baseline"],
        pointwise["absolute_difference"],
        pointwise["relative_difference"],
        pointwise["relative_tolerance"],
        pointwise["absolute_tolerance"],
        resolved_sum["maximum_absolute_difference"],
        resolved_sum["maximum_relative_difference"],
        resolved_sum["relative_tolerance"],
        resolved_sum["absolute_tolerance"],
    )
    matrix_element = record.result.get("matrix_element")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in raw_numbers
    ) or (
        isinstance(matrix_element, bool)
        or not isinstance(matrix_element, (int, float))
        or not math.isfinite(float(matrix_element))
        or float(matrix_element) < 0.0
    ):
        raise MeasurementLineageError(
            f"{record.cell_id}: closure validation evidence is not finite and "
            "nonnegative"
        )
    (
        candidate,
        baseline,
        absolute,
        relative,
        relative_tolerance,
        absolute_tolerance,
        maximum_absolute,
        maximum_relative,
        resolved_relative_tolerance,
        resolved_absolute_tolerance,
    ) = (float(value) for value in raw_numbers)
    recomputed_absolute = abs(candidate - baseline)
    recomputed_relative = recomputed_absolute / max(abs(baseline), 1.0e-300)
    if (
        float(matrix_element) != candidate
        or absolute != recomputed_absolute
        or relative != recomputed_relative
        or (
            recomputed_absolute <= absolute_tolerance
            or recomputed_relative <= relative_tolerance
        )
        or pointwise.get("status") != ResultStatus.VALIDATION_FAILED.value
        or (
            maximum_absolute > resolved_absolute_tolerance
            and maximum_relative > resolved_relative_tolerance
        )
        or resolved_sum.get("status") != ResultStatus.OK.value
    ):
        raise MeasurementLineageError(
            f"{record.cell_id}: closure validation evidence is numerically "
            "inconsistent"
        )

    try:
        artifact_root = Path(str(artifact["path"])).expanduser().resolve(strict=True)
        expected_artifact_root = (
            record.manifest_path.parent / "artifact"
        ).resolve(strict=True)
    except OSError as error:
        raise MeasurementLineageError(
            f"{record.cell_id}: closure failure artifact is unavailable"
        ) from error
    if (
        Path(str(artifact["path"])).expanduser().is_symlink()
        or not artifact_root.is_dir()
        or artifact_root != expected_artifact_root
    ):
        raise MeasurementLineageError(
            f"{record.cell_id}: closure failure artifact is not its canonical "
            "attempt artifact"
        )
    try:
        _validate_runtime_identity_postflight(provenance, validation)
    except ValueError as error:
        raise MeasurementLineageError(
            f"{record.cell_id}: closure failure runtime identity is invalid"
        ) from error
    _source_epoch(record)
    _runtime_identity_evidence(record)


def _authenticate_replaceable_validation_failure(
    store: ArtifactStore,
    record: CurrentRecord,
    cell: CellSpec,
) -> None:
    """Authenticate one immutable current selected for replacement."""

    _validate_numerical_validation_failure(record, cell)
    _current_pin(record)
    _attempt_owner_evidence(store, record)


def _default_recurrence_reachability(
    record: CurrentRecord,
    cell: CellSpec,
    *,
    store: ArtifactStore,
    catalog: ReportCatalog,
) -> Mapping[str, object]:
    """Authenticate one recurrence schedule and expose active model contracts."""

    if (
        cell.measurement.execution_mode is not ExecutionMode.RECURRENCE
        or cell.measurement.model is not ModelKey.BUILTIN_SM
    ):
        raise MeasurementLineageError(
            "recurrence reachability received a non-built-in recurrence cell"
        )
    artifact = record.result.get("artifact")
    if not isinstance(artifact, Mapping):
        raise MeasurementLineageError(
            f"{cell.cell_id}: recurrence result has no artifact record"
        )
    raw_path = artifact.get("path")
    process_id = artifact.get("process_id")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or "${" in raw_path
        or not isinstance(process_id, str)
        or not process_id
    ):
        raise MeasurementLineageError(
            f"{cell.cell_id}: recurrence artifact locator is invalid"
        )
    try:
        raw_artifact_root = Path(raw_path).expanduser()
        artifact_root = raw_artifact_root.resolve(strict=True)
    except OSError as error:
        raise MeasurementLineageError(
            f"{cell.cell_id}: recurrence artifact is unavailable"
        ) from error
    if raw_artifact_root.is_symlink() or not artifact_root.is_dir():
        raise MeasurementLineageError(
            f"{cell.cell_id}: recurrence artifact is not a regular directory"
        )
    artifact_owner = _resolve_recurrence_artifact_owner(
        store,
        catalog,
        record,
        cell,
        artifact_root,
        process_id,
    )

    try:
        from pyamplicol.models.prepared import (
            EAGER_KERNEL_ABI,
            PreparedKernelPack,
        )
        from pyamplicol.runtime.recurrence_exact._plan_v2 import (
            DIRECT_NONE_U32,
            _load_recurrence_exact_sections_v1,
        )

        from .final_audit import audit_artifact

        evidence = audit_artifact(cell, artifact_root, process_id)
        execution_path = _artifact_member(
            artifact_root,
            evidence.execution_manifest_path,
            context=f"{cell.cell_id} execution manifest",
        )
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if not isinstance(execution, dict):
            raise MeasurementLineageError(
                "recurrence execution manifest is not an object"
            )
        plan = execution.get("plan")
        kernel_pack = execution.get("kernel_pack")
        if not isinstance(plan, Mapping) or not isinstance(kernel_pack, Mapping):
            raise MeasurementLineageError(
                "recurrence execution lacks plan or kernel-pack identity"
            )
        runtime_schedule = plan.get("runtime_schedule")
        process_binding = plan.get("process_binding")
        if not isinstance(runtime_schedule, Mapping) or not isinstance(
            process_binding, Mapping
        ):
            raise MeasurementLineageError(
                "recurrence execution lacks schedule or process binding"
            )
        schedule_path = _artifact_member(
            artifact_root,
            runtime_schedule.get("path"),
            context=f"{cell.cell_id} recurrence schedule",
        )
        binding_path = _artifact_member(
            artifact_root
            / "processes"
            / process_id,
            process_binding.get("path"),
            context=f"{cell.cell_id} recurrence process binding",
        )
        if (
            _file_digest(schedule_path) != runtime_schedule.get("sha256")
            or schedule_path.stat().st_size != runtime_schedule.get("size_bytes")
            or _file_digest(binding_path) != process_binding.get("sha256")
            or binding_path.stat().st_size != process_binding.get("size_bytes")
        ):
            raise MeasurementLineageError(
                "recurrence schedule or process-binding bytes differ from execution"
            )
        pack_path = _artifact_member(
            artifact_root,
            kernel_pack.get("manifest_path"),
            context=f"{cell.cell_id} recurrence kernel pack",
        )
        raw_pack = json.loads(pack_path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw_pack, dict)
            or raw_pack.pop("eager_kernel_abi", None) != EAGER_KERNEL_ABI
        ):
            raise MeasurementLineageError(
                "recurrence prepared pack has an incompatible eager-kernel ABI"
            )
        prepared_pack = PreparedKernelPack.from_dict(raw_pack)
        direct_catalog = prepared_pack.recurrence_direct_template_catalog
        semantic_catalog = prepared_pack.recurrence_template_catalog
        if direct_catalog is None or semantic_catalog is None:
            raise MeasurementLineageError(
                "recurrence prepared pack lacks authenticated direct semantics"
            )
        sections = _load_recurrence_exact_sections_v1(
            artifact_root,
            process_id,
        )
        active_executor_ids = {
            row.executor_id
            for row in sections.row_groups
            if row.executor_id != DIRECT_NONE_U32
        }
        selected_dispatch_ids = {
            row.dispatch_variant_id
            for row in sections.resolved_source_selections
        }
        for dispatch_id in selected_dispatch_ids:
            try:
                active_executor_ids.add(
                    sections.source_dispatch_variants[dispatch_id].executor_id
                )
            except IndexError as error:
                raise MeasurementLineageError(
                    "recurrence source dispatch selects an absent variant"
                ) from error
        if not active_executor_ids:
            raise MeasurementLineageError(
                "recurrence schedule has no active direct executors"
            )
        if max(active_executor_ids) >= len(direct_catalog.templates):
            raise MeasurementLineageError(
                "recurrence schedule selects an unknown direct executor"
            )
        transition_by_id = {
            transition.template_id: transition
            for transition in semantic_catalog.transitions
        }
        matched_contracts: set[str] = set()
        matched_templates: set[str] = set()
        executor_by_id = {
            executor.executor_id: executor for executor in sections.executors
        }
        for executor_id in active_executor_ids:
            template = direct_catalog.templates[executor_id]
            if template.direct_executor_id != executor_id:
                raise MeasurementLineageError(
                    "recurrence direct executor catalog is not dense"
                )
            executor = executor_by_id.get(executor_id)
            binding = template.payload_binding
            if (
                executor is None
                or executor.role != template.role
                or executor.destination_operation != template.destination_operation
                or executor.parent_component_counts
                != template.parent_component_counts
                or executor.destination_component_count
                != template.destination_component_count
                or executor.momentum_operand_count
                != template.momentum_operand_count
                or executor.prepared_kernel_id != binding.prepared_kernel_id
                or executor.runtime_template != binding.runtime_template
            ):
                raise MeasurementLineageError(
                    "active recurrence executor differs from its authenticated "
                    f"direct template: {executor_id}"
                )
            for template_id in template.semantic_template_ids:
                transition = transition_by_id.get(template_id)
                if (
                    transition is not None
                    and transition.equivalence_class in _HZZ_CONTRACT_IDS
                ):
                    matched_contracts.add(transition.equivalence_class)
                    matched_templates.add(template_id)
    except MeasurementLineageError:
        raise
    except Exception as error:
        raise MeasurementLineageError(
            f"{cell.cell_id}: cannot authenticate recurrence reachability: {error}"
        ) from error

    return {
        "cell_id": cell.cell_id,
        "attempt_id": record.attempt_id,
        "manifest_sha256": record.manifest_sha256,
        "artifact_locator": artifact_root.relative_to(
            record.manifest_path.parents[4]
        ).as_posix(),
        "artifact_owner": artifact_owner,
        "process_id": process_id,
        "execution_manifest_path": evidence.execution_manifest_path,
        "execution_manifest_sha256": evidence.execution_manifest_sha256,
        "schedule_path": schedule_path.relative_to(artifact_root).as_posix(),
        "schedule_sha256": _file_digest(schedule_path),
        "schedule_index_sha256": runtime_schedule.get("index_sha256"),
        "schedule_member_sha256": runtime_schedule.get("sha256"),
        "process_binding_path": binding_path.relative_to(artifact_root).as_posix(),
        "process_binding_sha256": _file_digest(binding_path),
        "kernel_pack_path": pack_path.relative_to(artifact_root).as_posix(),
        "kernel_pack_sha256": _file_digest(pack_path),
        "semantic_catalog_sha256": semantic_catalog.catalog_digest,
        "direct_catalog_sha256": direct_catalog.catalog_digest,
        "active_executor_ids": sorted(active_executor_ids),
        "matched_template_ids": sorted(matched_templates),
        "matched_contract_ids": sorted(matched_contracts),
    }


def _hzz_reachability_certificate(
    store: ArtifactStore,
    *,
    catalog: ReportCatalog,
    inspector: RecurrenceReachabilityInspector | None,
) -> dict[str, object]:
    by_id = {cell.cell_id: cell for cell in catalog.measurement_cells()}
    target_ids = {cell.cell_id for cell in hzz_impacted_cells(catalog=catalog)}
    replacement_ids = target_ids | {
        cell.cell_id for cell in hzz_agreement_closure(catalog=catalog)
    }
    inspect = (
        inspector
        if inspector is not None
        else lambda record, cell: _default_recurrence_reachability(
            record,
            cell,
            store=store,
            catalog=catalog,
        )
    )
    records: list[dict[str, object]] = []
    reached_ids: set[str] = set()
    existing_target_ids: set[str] = set()
    invalidated_failure_ids: set[str] = set()
    for current in store.recover_current_records():
        cell = by_id.get(current.cell_id)
        if cell is None:
            raise MeasurementLineageError(
                f"artifact store current is absent from the report catalog: "
                f"{current.cell_id}"
            )
        status = current.result.get("status")
        if status != ResultStatus.OK.value:
            if current.cell_id not in replacement_ids:
                raise MeasurementLineageError(
                    "current outside the exact HZZ replacement closure is not "
                    f"successful: {current.cell_id}"
                )
            _authenticate_replaceable_validation_failure(store, current, cell)
            invalidated_failure_ids.add(current.cell_id)
            continue
        if (
            cell.measurement.execution_mode is not ExecutionMode.RECURRENCE
            or cell.measurement.model is not ModelKey.BUILTIN_SM
        ):
            continue
        raw = dict(inspect(current, cell))
        matched = raw.get("matched_contract_ids")
        if (
            not isinstance(matched, list)
            or matched != sorted(set(matched))
            or any(value not in _HZZ_CONTRACT_IDS for value in matched)
        ):
            raise MeasurementLineageError(
                f"recurrence reachability record is malformed: {current.cell_id}"
            )
        if matched:
            reached_ids.add(current.cell_id)
        if current.cell_id in target_ids:
            existing_target_ids.add(current.cell_id)
            if not matched:
                raise MeasurementLineageError(
                    "catalog HZZ target is not structurally reachable through its "
                    f"stored schedule: {current.cell_id}"
                )
        records.append(raw)
    certificate = {
        "algorithm": "authenticated-recurrence-active-executor-reachability-v1",
        "target_contract_ids": sorted(_HZZ_CONTRACT_IDS),
        "inspected_current_count": len(records),
        "records": sorted(records, key=lambda item: str(item.get("cell_id"))),
        "reached_cell_ids": sorted(reached_ids),
        "existing_catalog_target_ids": sorted(existing_target_ids),
        "invalidated_validation_failed_current_ids": sorted(
            invalidated_failure_ids
        ),
    }
    certificate["sha256"] = _digest(certificate)
    return certificate


def _validate_recurrence_summary_cap_failure(
    store: ArtifactStore,
    record: CurrentRecord,
    cell: CellSpec,
    *,
    expected_bytes: int,
) -> dict[str, object]:
    """Authenticate one exact source-less summary-cap failure attempt."""

    try:
        validate_measurement(record.result, expected_cell=cell)
    except ValueError as error:
        raise MeasurementLineageError(
            f"{record.cell_id}: recurrence summary-cap failure schema is invalid"
        ) from error
    failure = record.result.get("failure")
    expected_message = _summary_cap_failure_message(expected_bytes)
    if (
        _RECURRENCE_SUMMARY_CAP_FAILURE_BYTES.get(record.cell_id)
        != expected_bytes
        or cell.cell_id != record.cell_id
        or record.result.get("status") != ResultStatus.ERROR.value
        or not isinstance(failure, Mapping)
        or set(failure) != {"kind", "message"}
        or failure.get("kind") != "GenerationError"
        or failure.get("message") != expected_message
    ):
        raise MeasurementLineageError(
            f"{record.cell_id}: current is not the authenticated "
            "recurrence summary-cap GenerationError"
        )
    inventory, result_locator = _attempt_owner_evidence(store, record)
    return {
        "cell_id": record.cell_id,
        "attempt_id": record.attempt_id,
        "manifest_sha256": record.manifest_sha256,
        "result_locator": result_locator,
        "result_sha256": inventory["result_sha256"],
        "summary_bytes": expected_bytes,
        "failure_kind": "GenerationError",
        "failure_message": expected_message,
    }


def _summary_cap_excluded_current_record(
    store: ArtifactStore,
    record: CurrentRecord,
    cell: CellSpec,
) -> dict[str, object]:
    try:
        validate_measurement(record.result, expected_cell=cell)
    except ValueError as error:
        raise MeasurementLineageError(
            f"{record.cell_id}: excluded terminal current schema is invalid"
        ) from error
    status = record.result.get("status")
    if (
        status == ResultStatus.OK.value
        or status not in {value.value for value in ResultStatus}
        or record.result.get("failure") is None
    ):
        raise MeasurementLineageError(
            f"{record.cell_id}: excluded current is not a terminal failure"
        )
    inventory, result_locator = _attempt_owner_evidence(store, record)
    return {
        "cell_id": record.cell_id,
        "attempt_id": record.attempt_id,
        "manifest_sha256": record.manifest_sha256,
        "result_locator": result_locator,
        "result_sha256": inventory["result_sha256"],
        **_current_pointer_identity(record),
        "status": status,
        "failure_sha256": _digest(record.result["failure"]),
    }


def _summary_cap_predecessor_record(
    lineage: MeasurementLineage,
) -> dict[str, object]:
    payload = lineage.payload
    ancestor_environment = payload.get("ancestor_environment")
    descendant_environment = payload.get("descendant_environment")
    retained = payload.get("retained_currents")
    if (
        payload.get("state") != "finalized"
        or payload.get("impact") != CLASS_C_HZZ_IMPACT
        or not isinstance(ancestor_environment, Mapping)
        or not isinstance(descendant_environment, Mapping)
        or not isinstance(retained, list)
    ):
        raise MeasurementLineageError(
            "recurrence summary-cap continuity requires one finalized HZZ "
            "predecessor"
        )
    record = {
        "payload_sha256": _digest(payload),
        "profile": payload["profile"],
        "impact": payload["impact"],
        "ancestor_revision": lineage.ancestor_revision,
        "ancestor_tree": lineage.ancestor_tree,
        "descendant_revision": lineage.descendant_revision,
        "descendant_tree": lineage.descendant_tree,
        "ancestor_environment": dict(ancestor_environment),
        "ancestor_environment_sha256": _digest(ancestor_environment),
        "descendant_environment_sha256": _digest(descendant_environment),
        "retained_currents": list(retained),
        "retained_currents_sha256": _digest(retained),
    }
    _validate_summary_cap_predecessor(record)
    return record


def _recurrence_summary_cap_failure_certificate(
    store: ArtifactStore,
    *,
    catalog: ReportCatalog,
    predecessor_lineage: MeasurementLineage,
) -> dict[str, object]:
    by_id = {cell.cell_id: cell for cell in catalog.measurement_cells()}
    target_ids = set(_RECURRENCE_SUMMARY_CAP_FAILURE_BYTES)
    records: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    inspected_count = 0
    successful_count = 0
    for current in store.recover_current_records():
        inspected_count += 1
        cell = by_id.get(current.cell_id)
        if cell is None:
            raise MeasurementLineageError(
                "artifact store current is absent from the report catalog: "
                f"{current.cell_id}"
            )
        if current.cell_id in target_ids:
            records.append(
                _validate_recurrence_summary_cap_failure(
                    store,
                    current,
                    cell,
                    expected_bytes=_RECURRENCE_SUMMARY_CAP_FAILURE_BYTES[
                        current.cell_id
                    ],
                )
            )
        elif current.result.get("status") != ResultStatus.OK.value:
            excluded.append(
                _summary_cap_excluded_current_record(
                    store,
                    current,
                    cell,
                )
            )
        else:
            successful_count += 1
    observed_ids = {str(record["cell_id"]) for record in records}
    if observed_ids != target_ids or len(records) != len(target_ids):
        missing = sorted(target_ids - observed_ids)
        raise MeasurementLineageError(
            "artifact store does not contain the exact recurrence summary-cap "
            "failure census"
            + (": " + ", ".join(missing) if missing else "")
        )
    certificate = {
        "algorithm": "authenticated-recurrence-summary-cap-failure-census-v1",
        "predecessor": _summary_cap_predecessor_record(predecessor_lineage),
        "target_summary_bytes": dict(_RECURRENCE_SUMMARY_CAP_FAILURE_BYTES),
        "inspected_current_count": inspected_count,
        "successful_current_count": successful_count,
        "excluded_non_success_current_count": len(excluded),
        "records": sorted(records, key=lambda item: str(item["cell_id"])),
        "excluded_non_success_currents": sorted(
            excluded,
            key=lambda item: str(item["cell_id"]),
        ),
        "invalidated_generation_error_current_ids": sorted(target_ids),
    }
    certificate["sha256"] = _digest(certificate)
    return certificate


def _reachability_certificate(
    impact: str,
    store: ArtifactStore,
    *,
    catalog: ReportCatalog,
    inspector: RecurrenceReachabilityInspector | None,
    predecessor_lineage: MeasurementLineage | None = None,
) -> dict[str, object]:
    if impact == CLASS_C_HZZ_IMPACT:
        return _hzz_reachability_certificate(
            store,
            catalog=catalog,
            inspector=inspector,
        )
    if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT:
        if predecessor_lineage is None:
            raise MeasurementLineageError(
                "recurrence summary-cap continuity has no audited predecessor"
            )
        return _recurrence_summary_cap_failure_certificate(
            store,
            catalog=catalog,
            predecessor_lineage=predecessor_lineage,
        )
    raise MeasurementLineageError(f"unsupported Class-C impact {impact!r}")


def _certificate_failure_ids(
    impact: str,
    certificate: Mapping[str, object],
) -> frozenset[str]:
    if impact == CLASS_C_HZZ_IMPACT:
        field = "invalidated_validation_failed_current_ids"
    elif impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT:
        field = "invalidated_generation_error_current_ids"
    else:
        raise MeasurementLineageError(f"unsupported Class-C impact {impact!r}")
    raw = certificate.get(field)
    if not isinstance(raw, list) or any(
        not isinstance(value, str) for value in raw
    ):
        raise MeasurementLineageError(
            "measurement-lineage invalidated failure census is malformed"
        )
    return frozenset(raw)


def _summary_cap_inherited_pins(
    certificate: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    predecessor = certificate.get("predecessor")
    if not isinstance(predecessor, Mapping):
        raise MeasurementLineageError(
            "recurrence summary-cap predecessor is absent"
        )
    return _validated_pins(predecessor, "retained_currents")


def _summary_cap_excluded_ids(
    certificate: Mapping[str, object],
) -> frozenset[str]:
    raw = certificate.get("excluded_non_success_currents")
    if not isinstance(raw, list):
        raise MeasurementLineageError(
            "recurrence summary-cap excluded census is absent"
        )
    return frozenset(
        str(record["cell_id"])
        for record in raw
        if isinstance(record, Mapping)
    )


def _snapshot_currents(
    store: ArtifactStore,
    *,
    ancestor_revision: str,
    ancestor_tree: str,
    catalog: ReportCatalog,
    invalidated_cell_ids: Sequence[str],
    recompare_cell_ids: Sequence[str],
    source_less_failure_ids: Sequence[str] = (),
    inherited_current_pins: Sequence[Mapping[str, object]] = (),
    excluded_current_ids: Sequence[str] = (),
) -> dict[str, object]:
    impacted_ids = set(invalidated_cell_ids)
    recompare_ids = set(recompare_cell_ids) - impacted_ids
    source_less_ids = set(source_less_failure_ids)
    inherited_by_cell = {
        str(pin["cell_id"]): pin for pin in inherited_current_pins
    }
    excluded_ids = set(excluded_current_ids)
    if not source_less_ids <= impacted_ids:
        raise MeasurementLineageError(
            "source-less failure pins escape the exact invalidated census"
        )
    retained: list[dict[str, object]] = []
    invalidated: list[dict[str, object]] = []
    recompare: list[dict[str, object]] = []
    observed_excluded: set[str] = set()
    for record in store.recover_current_records():
        if record.cell_id in excluded_ids:
            observed_excluded.add(record.cell_id)
            continue
        pin = _current_pin(
            record,
            source_epoch_fallback=(
                (ancestor_revision, ancestor_tree)
                if record.cell_id in source_less_ids
                else None
            ),
        )
        if (
            pin["source_revision"] != ancestor_revision
            or pin["source_tree"] != ancestor_tree
        ) and dict(inherited_by_cell.get(record.cell_id, {})) != pin:
            raise MeasurementLineageError(
                f"frozen current {record.cell_id!r} is not authorized by the "
                "exact ancestor or its audited predecessor"
            )
        if record.cell_id in impacted_ids:
            invalidated.append(pin)
        elif record.cell_id in recompare_ids:
            recompare.append(pin)
        else:
            retained.append(pin)
    if observed_excluded != excluded_ids:
        raise MeasurementLineageError(
            "excluded terminal current census changed during snapshot"
        )
    attempt_inventory, no_attempts = _attempt_inventory(store, catalog=catalog)
    return {
        "retained_currents": sorted(retained, key=lambda item: item["cell_id"]),
        "invalidated_currents": sorted(
            invalidated, key=lambda item: item["cell_id"]
        ),
        "recompare_currents": sorted(recompare, key=lambda item: item["cell_id"]),
        "attempt_inventory": list(attempt_inventory),
        "no_attempt_cells": list(no_attempts),
    }


def _workspace_manifest_identity(
    repo_root: Path,
    docs_dir: Path,
    *,
    ancestor_revision: str,
    descendant_revision: str,
) -> dict[str, object]:
    path = docs_dir / "report-workspace.json"
    try:
        relative = path.resolve(strict=True).relative_to(repo_root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise MeasurementLineageError(
            "report workspace manifest is unavailable or outside the repository"
        ) from error
    if path.is_symlink() or not path.is_file():
        raise MeasurementLineageError(
            "report workspace manifest is not a regular file"
        )
    logical = relative.as_posix()
    ancestor_member = _tree_member(repo_root, ancestor_revision, logical)
    descendant_member = _tree_member(repo_root, descendant_revision, logical)
    if (
        ancestor_member is None
        or descendant_member is None
        or ancestor_member != descendant_member
        or ancestor_member["mode"] != "100644"
    ):
        raise MeasurementLineageError(
            "Class-C bridge requires one unchanged regular report-workspace blob"
        )
    blob = _git(repo_root, "show", f"{ancestor_revision}:{logical}").stdout
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(blob).digest():
        raise MeasurementLineageError(
            "working report workspace manifest differs from its committed blob"
        )
    try:
        raw = json.loads(blob.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MeasurementLineageError(
            f"cannot read report workspace policy: {error}"
        ) from error
    if not isinstance(raw, dict) or not isinstance(raw.get("campaign_policy"), dict):
        raise MeasurementLineageError("report workspace campaign policy is malformed")
    return {
        "path": logical,
        "mode": ancestor_member["mode"],
        "object": ancestor_member["object"],
        "sha256": hashlib.sha256(blob).hexdigest(),
        "campaign_policy_sha256": _digest(raw["campaign_policy"]),
    }


def _prepare_class_c_bridge_locked(
    repo_root: Path,
    docs_dir: Path,
    store: ArtifactStore,
    *,
    ancestor_revision: str,
    descendant_revision: str,
    impact: str,
    catalog: ReportCatalog = REPORT_CATALOG,
    reachability_inspector: RecurrenceReachabilityInspector | None = None,
) -> dict[str, object]:
    """Snapshot a frozen ancestor campaign before runtime authentication changes."""

    root = repo_root.expanduser().resolve(strict=False)
    profile = _profile_name(root, docs_dir)
    ancestor = _git_commit(root, ancestor_revision)
    descendant = _git_commit(root, descendant_revision)
    source = require_eligible_report_source(root)
    if source.revision != descendant:
        raise MeasurementLineageError(
            "Class-C bridge preparation requires checkout HEAD at the descendant"
        )
    ancestor_tree = _git_tree(root, ancestor)
    descendant_tree = _git_tree(root, descendant)
    environment = _environment(
        docs_dir / "report_environment.json",
        expected_profile=profile,
    )
    if environment.get("source_revision") != ancestor:
        raise MeasurementLineageError(
            "profile environment is not authenticated for the Class-C ancestor"
        )
    diff = _diff_records(
        root,
        ancestor=ancestor,
        descendant=descendant,
        impact=impact,
    )
    workspace_manifest = _workspace_manifest_identity(
        root,
        docs_dir,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
    )
    predecessor_lineage = (
        load_and_audit_measurement_lineage(
            root,
            docs_dir,
            store,
            expected_active_source_revision=ancestor,
            catalog=catalog,
            reachability_inspector=reachability_inspector,
        )
        if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
        else None
    )
    if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT and (
        predecessor_lineage is None
        or predecessor_lineage.descendant_revision != ancestor
        or predecessor_lineage.descendant_tree != ancestor_tree
        or predecessor_lineage.payload.get("profile") != profile
        or predecessor_lineage.payload.get("descendant_environment")
        != environment
    ):
        raise MeasurementLineageError(
            "recurrence summary-cap continuity requires the audited HZZ bridge "
            "ending at its exact measured ancestor"
        )
    reachability = _reachability_certificate(
        impact,
        store,
        catalog=catalog,
        inspector=reachability_inspector,
        predecessor_lineage=predecessor_lineage,
    )
    impacted_cells, agreement_cells = _impact_and_agreement_cells(
        impact,
        reachability,
        catalog=catalog,
    )
    impacted_ids = tuple(cell.cell_id for cell in impacted_cells)
    agreement_ids = tuple(cell.cell_id for cell in agreement_cells)
    snapshot = _snapshot_currents(
        store,
        ancestor_revision=ancestor,
        ancestor_tree=ancestor_tree,
        catalog=catalog,
        invalidated_cell_ids=impacted_ids,
        recompare_cell_ids=agreement_ids,
        source_less_failure_ids=(
            _certificate_failure_ids(impact, reachability)
            if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
            else ()
        ),
        inherited_current_pins=(
            _summary_cap_inherited_pins(reachability)
            if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
            else ()
        ),
        excluded_current_ids=(
            _summary_cap_excluded_ids(reachability)
            if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
            else ()
        ),
    )
    impacted = [_cell_record(cell) for cell in impacted_cells]
    closure = [_cell_record(cell) for cell in agreement_cells]
    payload: dict[str, object] = {
        "schema": MEASUREMENT_LINEAGE_SCHEMA,
        "state": "pending",
        "class": "C",
        "impact": impact,
        "profile": profile,
        "ancestor_revision": ancestor,
        "ancestor_tree": ancestor_tree,
        "descendant_revision": descendant,
        "descendant_tree": descendant_tree,
        "git_diff": list(diff),
        "git_diff_sha256": _digest(diff),
        "ancestor_environment": environment,
        "ancestor_environment_sha256": _digest(environment),
        "descendant_environment": None,
        "descendant_environment_sha256": None,
        "runtime_invariant_fields": list(_ENVIRONMENT_INVARIANT_FIELDS),
        "impacted_cells": impacted,
        "agreement_closure_cells": closure,
        "impact_certificate_sha256": _digest(
            {"impacted": impacted, "agreement_closure": closure}
        ),
        "catalog_sha256": _catalog_digest(catalog),
        "agreement_graph_sha256": _agreement_digest(catalog),
        "workspace_manifest": workspace_manifest,
        "workspace_manifest_sha256": _digest(workspace_manifest),
        "campaign_policy_sha256": workspace_manifest[
            "campaign_policy_sha256"
        ],
        "reachability_certificate": reachability,
        "reachability_certificate_sha256": _digest(reachability),
        **snapshot,
    }
    payload["current_snapshot_sha256"] = _digest(snapshot)
    output = class_c_pending_path(
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
    )
    expected_root = store.artifact_root / "source-bridges"
    expected_root.mkdir(parents=True, exist_ok=True)
    if expected_root.is_symlink() or not expected_root.is_dir():
        raise MeasurementLineageError(
            "pending Class-C bridge root is not a regular directory"
        )
    try:
        output.parent.resolve(strict=True).relative_to(
            store.artifact_root.resolve(strict=True)
        )
    except ValueError as error:
        raise MeasurementLineageError(
            "pending Class-C bridge must stay under the profile artifact root"
        ) from error
    if output.exists() or output.is_symlink():
        raise MeasurementLineageError(
            f"pending Class-C bridge already exists: {output}"
        )
    _write_envelope(output, payload)
    return payload


def prepare_class_c_bridge(
    repo_root: Path,
    docs_dir: Path,
    store: ArtifactStore,
    *,
    ancestor_revision: str,
    descendant_revision: str,
    impact: str,
    catalog: ReportCatalog = REPORT_CATALOG,
    reachability_inspector: RecurrenceReachabilityInspector | None = None,
) -> dict[str, object]:
    """Snapshot a frozen ancestor campaign under the profile writer locks."""

    with (
        store.named_lock("report-writer"),
        store.named_lock("measurement-lineage"),
    ):
        return _prepare_class_c_bridge_locked(
            repo_root,
            docs_dir,
            store,
            ancestor_revision=ancestor_revision,
            descendant_revision=descendant_revision,
            impact=impact,
            catalog=catalog,
            reachability_inspector=reachability_inspector,
        )


def _environment_invariants(environment: Mapping[str, object]) -> dict[str, object]:
    try:
        return {field: environment[field] for field in _ENVIRONMENT_INVARIANT_FIELDS}
    except KeyError as error:
        raise MeasurementLineageError(
            f"profile environment lacks invariant field {error.args[0]!r}"
        ) from error


def _is_authorized_native_inputs_transition(
    *,
    impact: str,
    ancestor_revision: str,
    descendant_revision: str,
    ancestor_digest: object,
    descendant_digest: object,
) -> bool:
    """Recognize the one reviewed Class-C native-input transition.

    Ordinary Class-C bridges remain native-input invariant.  The recurrence
    summary-cap bridge is different: its reviewed source closure intentionally
    changes the native recurrence executor, so both endpoint revisions and
    both independently computed input digests are pinned here.
    """

    return (
        impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
        and ancestor_revision == _RECURRENCE_SUMMARY_CAP_ANCESTOR_REVISION
        and descendant_revision == _RECURRENCE_SUMMARY_CAP_DESCENDANT_REVISION
        and ancestor_digest
        == _RECURRENCE_SUMMARY_CAP_ANCESTOR_NATIVE_INPUTS_SHA256
        and descendant_digest
        == _RECURRENCE_SUMMARY_CAP_DESCENDANT_NATIVE_INPUTS_SHA256
    )


def _require_environment_transition(
    *,
    impact: str,
    ancestor_revision: str,
    descendant_revision: str,
    ancestor_environment: Mapping[str, object],
    descendant_environment: Mapping[str, object],
) -> None:
    """Require exact invariants, with one digest-pinned native exception."""

    ancestor = _environment_invariants(ancestor_environment)
    descendant = _environment_invariants(descendant_environment)
    if impact != CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT:
        if ancestor != descendant:
            raise MeasurementLineageError(
                "Class-C descendant changes dependency/native/host runtime identity"
            )
        return

    ancestor_native = ancestor.pop("native_build_inputs_sha256")
    descendant_native = descendant.pop("native_build_inputs_sha256")
    if (
        ancestor != descendant
        or not _is_authorized_native_inputs_transition(
            impact=impact,
            ancestor_revision=ancestor_revision,
            descendant_revision=descendant_revision,
            ancestor_digest=ancestor_native,
            descendant_digest=descendant_native,
        )
    ):
        raise MeasurementLineageError(
            "Class-C descendant changes dependency/native/host runtime identity "
            "outside the exact recurrence-summary-cap transition"
        )


def _finalize_class_c_bridge_locked(
    repo_root: Path,
    docs_dir: Path,
    store: ArtifactStore,
    *,
    pending_path: Path,
    expected_active_source_revision: str,
    runtime_auditor: Callable[[str, Path], Mapping[str, object]] | None = None,
    catalog: ReportCatalog = REPORT_CATALOG,
    reachability_inspector: RecurrenceReachabilityInspector | None = None,
) -> dict[str, object]:
    """Authenticate the descendant runtime and publish the tracked lineage."""

    root = repo_root.expanduser().resolve(strict=False)
    profile = _profile_name(root, docs_dir)
    pending_input = _validated_pending_path(store, pending_path)
    payload = _load_envelope(pending_input, expected_state="pending")
    active = _git_commit(root, expected_active_source_revision)
    source = require_eligible_report_source(root)
    if (
        payload.get("profile") != profile
        or payload.get("descendant_revision") != active
        or payload.get("descendant_tree") != source.tree
        or source.revision != active
    ):
        raise MeasurementLineageError(
            "pending Class-C bridge does not identify the active descendant"
        )
    canonical_pending = class_c_pending_path(
        store,
        ancestor_revision=str(payload["ancestor_revision"]),
        descendant_revision=active,
    )
    if (
        pending_input != canonical_pending.absolute()
    ):
        raise MeasurementLineageError(
            "pending Class-C bridge does not use its canonical A-D locator"
        )
    expected_diff = _diff_records(
        root,
        ancestor=str(payload["ancestor_revision"]),
        descendant=active,
        impact=str(payload["impact"]),
    )
    workspace_manifest = _workspace_manifest_identity(
        root,
        docs_dir,
        ancestor_revision=str(payload["ancestor_revision"]),
        descendant_revision=active,
    )
    impact = str(payload["impact"])
    predecessor_lineage = (
        load_and_audit_measurement_lineage(
            root,
            docs_dir,
            store,
            expected_active_source_revision=str(payload["ancestor_revision"]),
            catalog=catalog,
            reachability_inspector=reachability_inspector,
        )
        if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
        else None
    )
    if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT and (
        predecessor_lineage is None
        or predecessor_lineage.descendant_revision
        != payload["ancestor_revision"]
        or predecessor_lineage.descendant_tree != payload["ancestor_tree"]
        or predecessor_lineage.payload.get("profile") != profile
        or predecessor_lineage.payload.get("descendant_environment")
        != payload.get("ancestor_environment")
    ):
        raise MeasurementLineageError(
            "recurrence summary-cap predecessor changed after preparation"
        )
    reachability = _reachability_certificate(
        impact,
        store,
        catalog=catalog,
        inspector=reachability_inspector,
        predecessor_lineage=predecessor_lineage,
    )
    if (
        payload.get("git_diff") != list(expected_diff)
        or payload.get("git_diff_sha256") != _digest(expected_diff)
        or payload.get("catalog_sha256") != _catalog_digest(catalog)
        or payload.get("agreement_graph_sha256") != _agreement_digest(catalog)
        or payload.get("workspace_manifest") != workspace_manifest
        or payload.get("workspace_manifest_sha256")
        != _digest(workspace_manifest)
        or payload.get("campaign_policy_sha256")
        != workspace_manifest["campaign_policy_sha256"]
        or payload.get("reachability_certificate") != reachability
        or payload.get("reachability_certificate_sha256")
        != _digest(reachability)
    ):
        raise MeasurementLineageError(
            "pending Class-C source, catalog, agreement, or policy certificate changed"
        )
    snapshot = _snapshot_currents(
        store,
        ancestor_revision=str(payload["ancestor_revision"]),
        ancestor_tree=str(payload["ancestor_tree"]),
        catalog=catalog,
        invalidated_cell_ids=tuple(
            cell.cell_id
            for cell in _impact_and_agreement_cells(
                impact,
                reachability,
                catalog=catalog,
            )[0]
        ),
        recompare_cell_ids=tuple(
            cell.cell_id
            for cell in _impact_and_agreement_cells(
                impact,
                reachability,
                catalog=catalog,
            )[1]
        ),
        source_less_failure_ids=(
            _certificate_failure_ids(impact, reachability)
            if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
            else ()
        ),
        inherited_current_pins=(
            _summary_cap_inherited_pins(reachability)
            if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
            else ()
        ),
        excluded_current_ids=(
            _summary_cap_excluded_ids(reachability)
            if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
            else ()
        ),
    )
    if payload.get("current_snapshot_sha256") != _digest(snapshot):
        raise MeasurementLineageError(
            "current or attempt state changed after Class-C preparation"
        )
    if runtime_auditor is None:
        from .final_audit import _audit_active_runtime

        runtime_auditor = _audit_active_runtime
    try:
        raw_runtime = runtime_auditor(active, root)
    except Exception as error:
        raise MeasurementLineageError(
            f"cannot authenticate descendant runtime: {error}"
        ) from error
    from .workspace import _authenticated_environment_payload

    descendant_environment = _authenticated_environment_payload(
        profile,
        expected_source_revision=active,
        active_runtime=raw_runtime,
    )
    ancestor_environment = payload.get("ancestor_environment")
    if not isinstance(ancestor_environment, Mapping):
        raise MeasurementLineageError("pending ancestor environment is malformed")
    _require_environment_transition(
        impact=impact,
        ancestor_revision=str(payload["ancestor_revision"]),
        descendant_revision=active,
        ancestor_environment=ancestor_environment,
        descendant_environment=descendant_environment,
    )
    if (
        ancestor_environment.get("python_package_tree_sha256")
        == descendant_environment.get("python_package_tree_sha256")
    ):
        raise MeasurementLineageError(
            "Class-C executable descendant did not change the Python package tree"
        )
    finalized = dict(payload)
    finalized.update(
        {
            "state": "finalized",
            "descendant_environment": descendant_environment,
            "descendant_environment_sha256": _digest(descendant_environment),
        }
    )
    target = measurement_lineage_path(root, docs_dir)
    json_path = docs_dir / "report_environment.json"
    tex_path = docs_dir / "report_environment.tex"
    previous_json = json_path.read_bytes() if json_path.exists() else None
    previous_tex = tex_path.read_bytes() if tex_path.exists() else None
    previous_lineage = target.read_bytes() if target.exists() else None
    from .workspace import refresh_profile_environment

    try:
        refreshed = refresh_profile_environment(
            root,
            profile,
            expected_source_revision=active,
            runtime_auditor=lambda _revision, _root: raw_runtime,
            _skip_workspace_validation=True,
        )
        if refreshed != descendant_environment:
            raise MeasurementLineageError(
                "written descendant environment differs from its "
                "authenticated snapshot"
            )
        _write_envelope(target, finalized)
        audit_measurement_lineage(
            root,
            docs_dir,
            store,
            expected_active_source_revision=active,
            catalog=catalog,
            reachability_inspector=reachability_inspector,
        )
        final_snapshot = _snapshot_currents(
            store,
            ancestor_revision=str(payload["ancestor_revision"]),
            ancestor_tree=str(payload["ancestor_tree"]),
            catalog=catalog,
            invalidated_cell_ids=tuple(
                cell.cell_id
                for cell in _impact_and_agreement_cells(
                    impact,
                    reachability,
                    catalog=catalog,
                )[0]
            ),
            recompare_cell_ids=tuple(
                cell.cell_id
                for cell in _impact_and_agreement_cells(
                    impact,
                    reachability,
                    catalog=catalog,
                )[1]
            ),
            source_less_failure_ids=(
                _certificate_failure_ids(impact, reachability)
                if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
                else ()
            ),
            inherited_current_pins=(
                _summary_cap_inherited_pins(reachability)
                if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
                else ()
            ),
            excluded_current_ids=(
                _summary_cap_excluded_ids(reachability)
                if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
                else ()
            ),
        )
        if payload.get("current_snapshot_sha256") != _digest(final_snapshot):
            raise MeasurementLineageError(
                "current or attempt state raced Class-C finalization"
            )
    except BaseException:
        _restore_file(json_path, previous_json)
        _restore_file(tex_path, previous_tex)
        _restore_file(target, previous_lineage)
        raise
    return finalized


def finalize_class_c_bridge(
    repo_root: Path,
    docs_dir: Path,
    store: ArtifactStore,
    *,
    pending_path: Path,
    expected_active_source_revision: str,
    runtime_auditor: Callable[[str, Path], Mapping[str, object]] | None = None,
    catalog: ReportCatalog = REPORT_CATALOG,
    reachability_inspector: RecurrenceReachabilityInspector | None = None,
) -> dict[str, object]:
    """Authenticate and publish a Class-C bridge under the profile writer locks."""

    with (
        store.named_lock("report-writer"),
        store.named_lock("measurement-lineage"),
    ):
        return _finalize_class_c_bridge_locked(
            repo_root,
            docs_dir,
            store,
            pending_path=pending_path,
            expected_active_source_revision=expected_active_source_revision,
            runtime_auditor=runtime_auditor,
            catalog=catalog,
            reachability_inspector=reachability_inspector,
        )


@dataclass(frozen=True, slots=True)
class MeasurementLineage:
    """Validated authorization map for one finalized Class-C bridge."""

    payload: Mapping[str, object]
    retained_by_cell: Mapping[str, Mapping[str, object]]
    invalidated_cell_ids: frozenset[str]
    recompare_cell_ids: frozenset[str]
    required_descendant_cell_ids: frozenset[str]
    inherited_environments_by_revision: Mapping[str, Mapping[str, object]]

    @property
    def ancestor_revision(self) -> str:
        return str(self.payload["ancestor_revision"])

    @property
    def ancestor_tree(self) -> str:
        return str(self.payload["ancestor_tree"])

    @property
    def descendant_revision(self) -> str:
        return str(self.payload["descendant_revision"])

    @property
    def descendant_tree(self) -> str:
        return str(self.payload["descendant_tree"])

    @property
    def impact(self) -> str:
        return str(self.payload["impact"])

    def source_for_current(
        self,
        record: CurrentRecord,
        *,
        active_revision: str,
        active_tree: str,
    ) -> tuple[str, str] | None:
        """Return the exact source authorized for ``record``, or reject it."""

        provenance = record.result.get("provenance")
        if not isinstance(provenance, Mapping):
            return None
        revision = provenance.get("report_source_revision")
        tree = provenance.get("report_source_tree")
        if (
            revision == active_revision
            and tree == active_tree
            and provenance.get("report_measured_source_revision") == active_revision
            and provenance.get("report_measured_source_tree") == active_tree
        ):
            return active_revision, active_tree
        if record.cell_id in self.required_descendant_cell_ids:
            return None
        pin = self.retained_by_cell.get(record.cell_id)
        if pin is None:
            return None
        if (
            record.attempt_id != pin.get("attempt_id")
            or record.manifest_sha256 != pin.get("manifest_sha256")
            or _file_digest(
                record.manifest_path.parent.parent.parent / "current.json"
            )
            != pin.get("current_pointer_sha256")
            or _file_digest(record.result_path) != pin.get("result_sha256")
            or revision != pin.get("source_revision")
            or tree != pin.get("source_tree")
            or provenance.get("report_measured_source_revision") != revision
            or provenance.get("report_measured_source_tree") != tree
        ):
            return None
        return str(revision), str(tree)

    def environment_for_source(self, revision: str) -> Mapping[str, object] | None:
        if revision == self.ancestor_revision:
            value = self.payload.get("ancestor_environment")
        elif revision == self.descendant_revision:
            value = self.payload.get("descendant_environment")
        else:
            return self.inherited_environments_by_revision.get(revision)
        return value if isinstance(value, Mapping) else None


def _validated_pins(
    payload: Mapping[str, object],
    field: str,
) -> tuple[dict[str, object], ...]:
    raw = payload.get(field)
    if not isinstance(raw, list):
        raise MeasurementLineageError(f"measurement lineage {field} must be an array")
    pins: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, dict) or set(value) != {
            "cell_id",
            "attempt_id",
            "manifest_sha256",
            "current_locator",
            "current_pointer_sha256",
            "source_revision",
            "source_tree",
            "result_sha256",
        }:
            raise MeasurementLineageError(
                f"measurement lineage {field} contains an invalid current pin"
            )
        cell_id = value.get("cell_id")
        try:
            canonical_attempt_id = (
                str(uuid.UUID(str(value["attempt_id"])))
                if isinstance(value.get("attempt_id"), str)
                else None
            )
        except ValueError:
            canonical_attempt_id = None
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or cell_id in seen
            or canonical_attempt_id != value.get("attempt_id")
            or not isinstance(value.get("manifest_sha256"), str)
            or _SHA256_RE.fullmatch(str(value["manifest_sha256"])) is None
            or not isinstance(value.get("current_locator"), str)
            or not str(value["current_locator"]).endswith("/current.json")
            or Path(str(value["current_locator"])).is_absolute()
            or any(
                part in {"", ".", ".."}
                for part in Path(str(value["current_locator"])).parts
            )
            or not isinstance(value.get("current_pointer_sha256"), str)
            or _SHA256_RE.fullmatch(str(value["current_pointer_sha256"])) is None
            or not isinstance(value.get("result_sha256"), str)
            or _SHA256_RE.fullmatch(str(value["result_sha256"])) is None
            or not isinstance(value.get("source_revision"), str)
            or _GIT_SHA_RE.fullmatch(str(value["source_revision"])) is None
            or not isinstance(value.get("source_tree"), str)
            or _GIT_SHA_RE.fullmatch(str(value["source_tree"])) is None
        ):
            raise MeasurementLineageError(
                f"measurement lineage {field} current pin is malformed"
            )
        seen.add(cell_id)
        pins.append(dict(value))
    if [str(pin["cell_id"]) for pin in pins] != sorted(seen):
        raise MeasurementLineageError(
            f"measurement lineage {field} is not sorted by cell identity"
        )
    return tuple(pins)


def load_measurement_lineage(
    repo_root: Path,
    docs_dir: Path,
    *,
    expected_active_revision: str,
    expected_active_tree: str,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> MeasurementLineage | None:
    path = measurement_lineage_path(repo_root, docs_dir)
    if not path.exists():
        return None
    payload = _load_envelope(path, expected_state="finalized")
    profile = _profile_name(repo_root, docs_dir)
    if (
        payload.get("profile") != profile
        or payload.get("class") != "C"
        or payload.get("impact") not in _CLASS_C_IMPACTS
        or payload.get("descendant_revision") != expected_active_revision
        or payload.get("descendant_tree") != expected_active_tree
    ):
        raise MeasurementLineageError(
            "finalized measurement lineage does not identify the active profile/source"
        )
    retained = _validated_pins(payload, "retained_currents")
    invalidated = _validated_pins(payload, "invalidated_currents")
    recompare = _validated_pins(payload, "recompare_currents")
    impact = str(payload["impact"])
    groups = [
        {str(pin["cell_id"]) for pin in values}
        for values in (retained, invalidated, recompare)
    ]
    if any(groups[left] & groups[right] for left in range(3) for right in range(left)):
        raise MeasurementLineageError(
            "measurement-lineage current authorization groups overlap"
        )
    all_pins = (*retained, *invalidated, *recompare)
    raw_certificate = payload.get("reachability_certificate")
    inherited_by_cell = (
        {
            str(pin["cell_id"]): pin
            for pin in _summary_cap_inherited_pins(raw_certificate)
        }
        if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
        and isinstance(raw_certificate, Mapping)
        else {}
    )
    if any(
        (
            pin["source_revision"] != payload["ancestor_revision"]
            or pin["source_tree"] != payload["ancestor_tree"]
        )
        and dict(inherited_by_cell.get(str(pin["cell_id"]), {})) != pin
        for pin in all_pins
    ):
        raise MeasurementLineageError(
            "measurement-lineage ancestor pin identifies another source"
        )
    inventory_by_key = {
        (str(row["cell_id"]), str(row["attempt_id"])): row
        for row in payload["attempt_inventory"]  # type: ignore[union-attr]
        if isinstance(row, Mapping)
    }
    for pin in all_pins:
        key = (str(pin["cell_id"]), str(pin["attempt_id"]))
        row = inventory_by_key.get(key)
        manifest_locator = (
            PurePosixPath(str(row["manifest_locator"]))
            if row is not None
            else None
        )
        canonical_current = (
            (manifest_locator.parents[2] / "current.json").as_posix()
            if manifest_locator is not None
            and len(manifest_locator.parents) >= 3
            else None
        )
        if (
            row is None
            or row.get("status") != "ok"
            or row.get("manifest_sha256") != pin["manifest_sha256"]
            or row.get("result_sha256") != pin["result_sha256"]
            or pin.get("current_locator") != canonical_current
        ):
            raise MeasurementLineageError(
                "measurement-lineage current pin is not bound to exactly one "
                "successful attempt-inventory row"
            )
    reachability = payload["reachability_certificate"]
    assert isinstance(reachability, Mapping)
    if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT:
        predecessor = reachability.get("predecessor")
        if (
            not isinstance(predecessor, Mapping)
            or predecessor.get("profile") != payload.get("profile")
            or predecessor.get("descendant_revision")
            != payload.get("ancestor_revision")
            or predecessor.get("descendant_tree") != payload.get("ancestor_tree")
            or predecessor.get("descendant_environment_sha256")
            != payload.get("ancestor_environment_sha256")
            or _environment_invariants(
                predecessor["ancestor_environment"]
            )
            != _environment_invariants(payload["ancestor_environment"])
        ):
            raise MeasurementLineageError(
                "measurement-lineage summary-cap predecessor does not end at "
                "the exact active ancestor"
            )
    reachability_record_ids = {
        str(record["cell_id"])
        for record in reachability["records"]  # type: ignore[index]
        if isinstance(record, Mapping)
    }
    catalog_by_id = {
        cell.cell_id: cell
        for cell in catalog.measurement_cells()
    }
    if impact == CLASS_C_HZZ_IMPACT:
        for record in reachability["records"]:  # type: ignore[index]
            if not isinstance(record, Mapping):
                continue
            owner = record.get("artifact_owner")
            if not isinstance(owner, Mapping):
                raise MeasurementLineageError(
                    "measurement-lineage artifact owner is absent"
                )
            consumer = catalog_by_id.get(str(owner.get("consumer_cell_id")))
            owner_cell = catalog_by_id.get(str(owner.get("owner_cell_id")))
            if consumer is None or owner_cell is None:
                raise MeasurementLineageError(
                    "measurement-lineage artifact owner is absent from the catalog"
                )
            relation = owner.get("relation")
            if relation == "equivalent-matrix-peer" and (
                not owner_cell.dataset_id.startswith("matrix_")
                or owner_cell not in catalog.equivalent_cells(consumer)
            ):
                raise MeasurementLineageError(
                    "measurement-lineage artifact owner is not the catalog matrix peer"
                )
            if relation == "consumer-attempt" and owner_cell != consumer:
                raise MeasurementLineageError(
                    "measurement-lineage direct artifact owner changed catalog cell"
                )
        static_target_ids = {
            cell.cell_id for cell in hzz_impacted_cells(catalog=catalog)
        }
        if reachability.get("existing_catalog_target_ids") != sorted(
            static_target_ids & reachability_record_ids
        ):
            raise MeasurementLineageError(
                "measurement-lineage existing HZZ target census is inconsistent"
            )
    impacted_cells, agreement_cells = _impact_and_agreement_cells(
        impact,
        reachability,
        catalog=catalog,
    )
    impacted = [_cell_record(cell) for cell in impacted_cells]
    closure = [_cell_record(cell) for cell in agreement_cells]
    if (
        payload.get("impacted_cells") != impacted
        or payload.get("agreement_closure_cells") != closure
        or payload.get("impact_certificate_sha256")
        != _digest({"impacted": impacted, "agreement_closure": closure})
    ):
        raise MeasurementLineageError(
            "measurement-lineage impact/agreement closure is invalid"
        )
    target_ids = {cell.cell_id for cell in impacted_cells}
    closure_ids = {cell.cell_id for cell in agreement_cells}
    failure_ids = set(_certificate_failure_ids(impact, reachability))
    if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT:
        failure_record_by_cell = {
            str(record["cell_id"]): record
            for record in reachability["records"]  # type: ignore[index]
            if isinstance(record, Mapping)
        }
        invalidated_pin_by_cell = {
            str(pin["cell_id"]): pin for pin in invalidated
        }
        if any(
            (record := failure_record_by_cell.get(cell_id)) is None
            or (pin := invalidated_pin_by_cell.get(cell_id)) is None
            or record.get("attempt_id") != pin.get("attempt_id")
            or record.get("manifest_sha256") != pin.get("manifest_sha256")
            or record.get("result_sha256") != pin.get("result_sha256")
            for cell_id in failure_ids
        ):
            raise MeasurementLineageError(
                "measurement-lineage summary-cap failure records do not match "
                "their invalidated current pins"
            )
    current_ids = set().union(*groups)
    expected_invalidated = current_ids & target_ids
    expected_recompare = (current_ids & closure_ids) - expected_invalidated
    if (
        not failure_ids <= expected_invalidated | expected_recompare
        or failure_ids & groups[0]
        or (
            impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
            and (
                len(current_ids)
                != int(reachability["successful_current_count"])
                + len(failure_ids)
                or int(reachability["inspected_current_count"])
                != len(current_ids)
                + len(_summary_cap_excluded_ids(reachability))
            )
        )
        or groups[1] != expected_invalidated
        or groups[2] != expected_recompare
        or groups[0] != current_ids - expected_invalidated - expected_recompare
    ):
        raise MeasurementLineageError(
            "measurement-lineage current groups do not match impact reachability"
        )
    snapshot = {
        field: payload[field]
        for field in (
            "retained_currents",
            "invalidated_currents",
            "recompare_currents",
            "attempt_inventory",
            "no_attempt_cells",
        )
    }
    if payload.get("current_snapshot_sha256") != _digest(snapshot):
        raise MeasurementLineageError(
            "measurement-lineage current snapshot digest is invalid"
        )
    catalog_ids = {cell.cell_id for cell in catalog.measurement_cells()}
    attempted_ids = {
        str(row["cell_id"])
        for row in payload["attempt_inventory"]  # type: ignore[union-attr]
        if isinstance(row, Mapping)
    }
    no_attempt_ids = set(payload["no_attempt_cells"])  # type: ignore[arg-type]
    if (
        attempted_ids & no_attempt_ids
        or attempted_ids | no_attempt_ids != catalog_ids
    ):
        raise MeasurementLineageError(
            "measurement-lineage attempt/no-attempt inventory does not partition "
            "the report catalog"
        )
    return MeasurementLineage(
        payload=payload,
        retained_by_cell={str(pin["cell_id"]): pin for pin in retained},
        invalidated_cell_ids=frozenset(groups[1]),
        recompare_cell_ids=frozenset(groups[2]),
        required_descendant_cell_ids=frozenset(target_ids | closure_ids),
        inherited_environments_by_revision=(
            {
                str(reachability["predecessor"]["ancestor_revision"]): dict(
                    reachability["predecessor"]["ancestor_environment"]
                )
            }
            if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
            else {}
        ),
    )


def audit_measurement_lineage(
    repo_root: Path,
    docs_dir: Path,
    store: ArtifactStore,
    *,
    expected_active_source_revision: str,
    catalog: ReportCatalog = REPORT_CATALOG,
    reachability_inspector: RecurrenceReachabilityInspector | None = None,
    _loaded_lineage: MeasurementLineage | None = None,
) -> dict[str, object]:
    """Audit source, environment, pins, and impact closure without mutation."""

    root = repo_root.expanduser().resolve(strict=False)
    active = _git_commit(root, expected_active_source_revision)
    active_tree = _git_tree(root, active)
    lineage = _loaded_lineage or load_measurement_lineage(
        root,
        docs_dir,
        expected_active_revision=active,
        expected_active_tree=active_tree,
        catalog=catalog,
    )
    if lineage is None:
        raise MeasurementLineageError("profile has no finalized measurement lineage")
    payload = lineage.payload
    expected_diff = _diff_records(
        root,
        ancestor=lineage.ancestor_revision,
        descendant=lineage.descendant_revision,
        impact=str(payload["impact"]),
    )
    reachability = payload["reachability_certificate"]
    assert isinstance(reachability, Mapping)
    impact = str(payload["impact"])
    impacted_cells, agreement_cells = _impact_and_agreement_cells(
        impact,
        reachability,
        catalog=catalog,
    )
    impacted = [_cell_record(cell) for cell in impacted_cells]
    closure = [_cell_record(cell) for cell in agreement_cells]
    workspace_manifest = _workspace_manifest_identity(
        root,
        docs_dir,
        ancestor_revision=lineage.ancestor_revision,
        descendant_revision=lineage.descendant_revision,
    )
    if (
        payload.get("git_diff") != list(expected_diff)
        or payload.get("git_diff_sha256") != _digest(expected_diff)
        or payload.get("catalog_sha256") != _catalog_digest(catalog)
        or payload.get("agreement_graph_sha256") != _agreement_digest(catalog)
        or payload.get("workspace_manifest") != workspace_manifest
        or payload.get("workspace_manifest_sha256")
        != _digest(workspace_manifest)
        or payload.get("campaign_policy_sha256")
        != workspace_manifest["campaign_policy_sha256"]
        or payload.get("impacted_cells") != impacted
        or payload.get("agreement_closure_cells") != closure
        or payload.get("impact_certificate_sha256")
        != _digest({"impacted": impacted, "agreement_closure": closure})
    ):
        raise MeasurementLineageError(
            "measurement-lineage source, policy, catalog, or impact certificate changed"
        )
    profile = _profile_name(root, docs_dir)
    descendant_environment = _environment(
        docs_dir / "report_environment.json",
        expected_profile=profile,
    )
    if (
        payload.get("descendant_environment") != descendant_environment
        or payload.get("descendant_environment_sha256")
        != _digest(descendant_environment)
    ):
        raise MeasurementLineageError(
            "active profile environment differs from the finalized lineage"
        )
    ancestor_environment = payload.get("ancestor_environment")
    if not isinstance(ancestor_environment, Mapping):
        raise MeasurementLineageError(
            "Class-C runtime invariant proof no longer holds"
        )
    try:
        _require_environment_transition(
            impact=impact,
            ancestor_revision=lineage.ancestor_revision,
            descendant_revision=lineage.descendant_revision,
            ancestor_environment=ancestor_environment,
            descendant_environment=descendant_environment,
        )
    except MeasurementLineageError as error:
        raise MeasurementLineageError(
            "Class-C runtime invariant proof no longer holds"
        ) from error
    missing_retained: list[str] = []
    for cell_id, pin in lineage.retained_by_cell.items():
        current = store.load_current(cell_id, missing_ok=True)
        if (
            current is None
            or lineage.source_for_current(
                current,
                active_revision=active,
                active_tree=active_tree,
            )
            != (pin.get("source_revision"), pin.get("source_tree"))
        ):
            missing_retained.append(cell_id)
    if missing_retained:
        raise MeasurementLineageError(
            "retained ancestor currents changed or disappeared: "
            + ", ".join(missing_retained[:8])
        )
    missing_history: list[str] = []
    inventory_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    for raw in payload["attempt_inventory"]:  # type: ignore[union-attr]
        assert isinstance(raw, Mapping)
        locator = raw.get("manifest_locator")
        if not isinstance(locator, str):
            missing_history.append("<invalid locator>")
            continue
        try:
            manifest_path = _artifact_member(
                store.artifact_root,
                locator,
                context="historical attempt manifest",
            )
        except MeasurementLineageError:
            missing_history.append(locator)
            continue
        try:
            observed = _validated_attempt_inventory_row(
                store,
                manifest_path,
                expected_cell_id=str(raw["cell_id"]),
            )
        except MeasurementLineageError:
            missing_history.append(locator)
            continue
        if observed != dict(raw):
            missing_history.append(locator)
            continue
        inventory_by_key[(str(raw["cell_id"]), str(raw["attempt_id"]))] = raw
    if missing_history:
        raise MeasurementLineageError(
            "immutable historical attempt evidence changed or disappeared: "
            + ", ".join(missing_history[:8])
        )
    certificate = payload["reachability_certificate"]
    assert isinstance(certificate, Mapping)
    by_cell = {cell.cell_id: cell for cell in catalog.measurement_cells()}
    failure_ids = set(_certificate_failure_ids(impact, certificate))
    historical_status_errors: list[str] = []
    for field in (
        "retained_currents",
        "invalidated_currents",
        "recompare_currents",
    ):
        for pin in payload[field]:  # type: ignore[index]
            assert isinstance(pin, Mapping)
            cell_id = str(pin["cell_id"])
            key = (cell_id, str(pin["attempt_id"]))
            inventory = inventory_by_key.get(key)
            if inventory is None:
                historical_status_errors.append(cell_id)
                continue
            try:
                manifest_path = _artifact_member(
                    store.artifact_root,
                    inventory["manifest_locator"],
                    context="historical pinned attempt manifest",
                )
                historical = store._validate_attempt_manifest(
                    manifest_path,
                    expected_cell_id=cell_id,
                    expected_attempt_id=key[1],
                    expected_digest=str(inventory["manifest_sha256"]),
                )
                if cell_id in failure_ids:
                    cell = by_cell.get(cell_id)
                    if cell is None:
                        raise MeasurementLineageError(
                            f"{cell_id}: historical failure is absent from the catalog"
                        )
                    if impact == CLASS_C_HZZ_IMPACT:
                        _validate_numerical_validation_failure(historical, cell)
                    else:
                        _validate_recurrence_summary_cap_failure(
                            store,
                            historical,
                            cell,
                            expected_bytes=(
                                _RECURRENCE_SUMMARY_CAP_FAILURE_BYTES[cell_id]
                            ),
                        )
                elif historical.result.get("status") != ResultStatus.OK.value:
                    raise MeasurementLineageError(
                        f"{cell_id}: non-successful current was selected for "
                        "retention or replacement without failure evidence"
                    )
            except Exception:
                historical_status_errors.append(cell_id)
    if historical_status_errors:
        raise MeasurementLineageError(
            "historical current status or failure evidence changed: "
            + ", ".join(historical_status_errors[:8])
        )
    inspect = (
        reachability_inspector
        if reachability_inspector is not None
        else lambda record, cell: _default_recurrence_reachability(
            record,
            cell,
            store=store,
            catalog=catalog,
        )
    )
    changed_reachability: list[str] = []
    for raw_record in certificate["records"]:  # type: ignore[index]
        assert isinstance(raw_record, Mapping)
        key = (
            str(raw_record["cell_id"]),
            str(raw_record["attempt_id"]),
        )
        inventory = inventory_by_key.get(key)
        cell = by_cell.get(key[0])
        if inventory is None or cell is None:
            changed_reachability.append(key[0])
            continue
        manifest_path = _artifact_member(
            store.artifact_root,
            inventory["manifest_locator"],
            context="reachability attempt manifest",
        )
        try:
            historical = store._validate_attempt_manifest(
                manifest_path,
                expected_cell_id=key[0],
                expected_attempt_id=key[1],
                expected_digest=str(inventory["manifest_sha256"]),
            )
            if impact == CLASS_C_HZZ_IMPACT:
                observed_record = dict(inspect(historical, cell))
            else:
                observed_record = _validate_recurrence_summary_cap_failure(
                    store,
                    historical,
                    cell,
                    expected_bytes=_RECURRENCE_SUMMARY_CAP_FAILURE_BYTES[key[0]],
                )
        except Exception:
            changed_reachability.append(key[0])
            continue
        if observed_record != dict(raw_record):
            changed_reachability.append(key[0])
    if changed_reachability:
        raise MeasurementLineageError(
            "authenticated recurrence reachability changed or cannot be replayed: "
            + ", ".join(changed_reachability[:8])
        )
    excluded_by_cell: dict[str, Mapping[str, object]] = {}
    if impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT:
        excluded_errors: list[str] = []
        for raw_record in certificate[
            "excluded_non_success_currents"
        ]:  # type: ignore[index]
            assert isinstance(raw_record, Mapping)
            cell_id = str(raw_record["cell_id"])
            excluded_by_cell[cell_id] = raw_record
            key = (cell_id, str(raw_record["attempt_id"]))
            inventory = inventory_by_key.get(key)
            cell = by_cell.get(cell_id)
            if inventory is None or cell is None:
                excluded_errors.append(cell_id)
                continue
            try:
                manifest_path = _artifact_member(
                    store.artifact_root,
                    inventory["manifest_locator"],
                    context="excluded terminal attempt manifest",
                )
                historical = store._validate_attempt_manifest(
                    manifest_path,
                    expected_cell_id=cell_id,
                    expected_attempt_id=key[1],
                    expected_digest=str(inventory["manifest_sha256"]),
                )
                observed = _summary_cap_excluded_current_record(
                    store,
                    historical,
                    cell,
                )
            except Exception:
                excluded_errors.append(cell_id)
                continue
            historical_fields = (
                set(_SUMMARY_CAP_EXCLUDED_RECORD_KEYS)
                - {"current_locator", "current_pointer_sha256"}
            )
            if any(
                observed[field] != raw_record[field]
                for field in historical_fields
            ) or (
                cell_id not in lineage.required_descendant_cell_ids
                and observed != dict(raw_record)
            ):
                excluded_errors.append(cell_id)
        if excluded_errors:
            raise MeasurementLineageError(
                "excluded terminal current evidence changed: "
                + ", ".join(excluded_errors[:8])
            )
    stale_targets: list[str] = []
    invalidated_pin_by_cell = {
        str(pin["cell_id"]): pin
        for pin in payload["invalidated_currents"]  # type: ignore[index]
        if isinstance(pin, Mapping)
    }
    historical_target_pin_by_cell = {
        str(pin["cell_id"]): pin
        for field in ("invalidated_currents", "recompare_currents")
        for pin in payload[field]  # type: ignore[index]
        if isinstance(pin, Mapping)
    }
    for cell_id in sorted(
        lineage.required_descendant_cell_ids
    ):
        current = store.load_current(cell_id, missing_ok=True)
        if current is None:
            continue
        provenance = current.result.get("provenance")
        if not isinstance(provenance, Mapping):
            pin = invalidated_pin_by_cell.get(cell_id)
            excluded = excluded_by_cell.get(cell_id)
            if (
                impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
                and cell_id in failure_ids
                and pin is not None
                and current.attempt_id == pin.get("attempt_id")
                and current.manifest_sha256 == pin.get("manifest_sha256")
                and _file_digest(current.result_path) == pin.get("result_sha256")
                and _file_digest(
                    current.manifest_path.parent.parent.parent / "current.json"
                )
                == pin.get("current_pointer_sha256")
            ):
                continue
            if (
                excluded is not None
                and current.attempt_id == excluded.get("attempt_id")
                and current.manifest_sha256 == excluded.get("manifest_sha256")
                and _file_digest(current.result_path)
                == excluded.get("result_sha256")
                and _file_digest(
                    current.manifest_path.parent.parent.parent / "current.json"
                )
                == excluded.get("current_pointer_sha256")
            ):
                continue
            stale_targets.append(cell_id)
            continue
        historical_pin = historical_target_pin_by_cell.get(cell_id)
        if (
            impact == CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT
            and historical_pin is not None
            and provenance.get("report_source_revision")
            == historical_pin.get("source_revision")
            and provenance.get("report_source_tree")
            == historical_pin.get("source_tree")
            and provenance.get("report_measured_source_revision")
            == historical_pin.get("source_revision")
            and provenance.get("report_measured_source_tree")
            == historical_pin.get("source_tree")
            and current.attempt_id == historical_pin.get("attempt_id")
            and current.manifest_sha256
            == historical_pin.get("manifest_sha256")
            and _file_digest(current.result_path)
            == historical_pin.get("result_sha256")
            and _file_digest(
                current.manifest_path.parent.parent.parent / "current.json"
            )
            == historical_pin.get("current_pointer_sha256")
        ):
            continue
        if (
            provenance.get("report_source_revision") == lineage.ancestor_revision
            and provenance.get("report_source_tree") == lineage.ancestor_tree
            and provenance.get("report_measured_source_revision")
            == lineage.ancestor_revision
            and provenance.get("report_measured_source_tree")
            == lineage.ancestor_tree
        ):
            # This historical pointer remains immutable evidence, but the
            # scheduler and service must not authorize it as fresh.
            continue
        if (
            provenance.get("report_source_revision") != active
            or provenance.get("report_source_tree") != active_tree
            or provenance.get("report_measured_source_revision") != active
            or provenance.get("report_measured_source_tree") != active_tree
        ):
            stale_targets.append(cell_id)
    if stale_targets:
        raise MeasurementLineageError(
            "Class-C target current has an unauthorized source: "
            + ", ".join(stale_targets[:8])
        )
    return {
        "schema": MEASUREMENT_LINEAGE_SCHEMA,
        "profile": profile,
        "impact": payload["impact"],
        "ancestor_revision": lineage.ancestor_revision,
        "descendant_revision": lineage.descendant_revision,
        "retained_current_count": len(lineage.retained_by_cell),
        "invalidated_current_count": len(lineage.invalidated_cell_ids),
        "recompare_current_count": len(lineage.recompare_cell_ids),
        "impacted_cell_count": len(impacted),
        "agreement_closure_cell_count": len(closure),
        "runtime_invariants_match": True,
    }


def load_and_audit_measurement_lineage(
    repo_root: Path,
    docs_dir: Path,
    store: ArtifactStore,
    *,
    expected_active_source_revision: str,
    catalog: ReportCatalog = REPORT_CATALOG,
    reachability_inspector: RecurrenceReachabilityInspector | None = None,
) -> MeasurementLineage | None:
    """Load one lineage object once and fully authenticate it under its lock."""

    root = repo_root.expanduser().resolve(strict=False)
    active = _git_commit(root, expected_active_source_revision)
    active_tree = _git_tree(root, active)
    with store.named_lock("measurement-lineage"):
        lineage = load_measurement_lineage(
            root,
            docs_dir,
            expected_active_revision=active,
            expected_active_tree=active_tree,
            catalog=catalog,
        )
        if lineage is None:
            stale: list[str] = []
            for current in store.recover_current_records():
                provenance = current.result.get("provenance")
                if not isinstance(provenance, Mapping):
                    continue
                measured_revision = provenance.get("report_measured_source_revision")
                measured_tree = provenance.get("report_measured_source_tree")
                if measured_revision is not None and (
                    provenance.get("report_source_revision") != active
                    or measured_revision != active
                    or provenance.get("report_source_tree") != active_tree
                    or measured_tree != active_tree
                ):
                    stale.append(current.cell_id)
            if stale:
                raise MeasurementLineageError(
                    "profile has mixed/ancestor measurements but no authenticated "
                    "measurement lineage: "
                    + ", ".join(sorted(stale)[:8])
                )
            return None
        audit_measurement_lineage(
            root,
            docs_dir,
            store,
            expected_active_source_revision=active,
            catalog=catalog,
            reachability_inspector=reachability_inspector,
            _loaded_lineage=lineage,
        )
        if (
            _load_envelope(
                measurement_lineage_path(root, docs_dir),
                expected_state="finalized",
            )
            != dict(lineage.payload)
        ):
            raise MeasurementLineageError(
                "measurement lineage changed during authentication"
            )
        return lineage


__all__ = [
    "CLASS_C_HZZ_IMPACT",
    "CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT",
    "MEASUREMENT_LINEAGE_FILENAME",
    "MEASUREMENT_LINEAGE_SCHEMA",
    "MeasurementLineage",
    "MeasurementLineageError",
    "audit_measurement_lineage",
    "finalize_class_c_bridge",
    "hzz_agreement_closure",
    "hzz_impacted_cells",
    "load_and_audit_measurement_lineage",
    "load_measurement_lineage",
    "measurement_lineage_path",
    "prepare_class_c_bridge",
    "recurrence_summary_cap_agreement_closure",
    "recurrence_summary_cap_impacted_cells",
    "signed_zero_helicity_agreement_closure",
    "signed_zero_helicity_impacted_cells",
]
