# SPDX-License-Identifier: 0BSD
"""Auditable campaign epochs and host-local original-AmpliCol seeds.

This module deliberately keeps old measurement provenance immutable.  A new
campaign may authorize a digest-pinned, same-profile original-AmpliCol current,
but it never rewrites that current to impersonate the new report source.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore, CurrentRecord
from .cache import validate_measurement
from .campaign_policy import CampaignPolicy, validate_policy_measurement
from .catalog import REPORT_CATALOG, ReportCatalog
from .models import Accuracy, CellSpec, ExecutionMode, ResultStatus, Workload

SEED_FILENAME = "original_amplicol_seed.json"
CAMPAIGN_MARKER_FILENAME = "campaign-epoch.json"
RESET_JOURNAL_FILENAME = "campaign-reset-journal.json"
BASELINE_GATE_FILENAME = "baseline-gate.json"
SEED_SCHEMA = "pyamplicol-original-amplicol-seed-v1"
CAMPAIGN_MARKER_SCHEMA = "pyamplicol-performance-campaign-epoch-v1"
RESET_JOURNAL_SCHEMA = "pyamplicol-performance-campaign-reset-v1"
BASELINE_GATE_SCHEMA = "pyamplicol-performance-baseline-gate-v1"
EXPECTED_CATALOG_CELL_COUNT = 2162
EXPECTED_AMPLICOL_CELL_COUNT = 314
EXPECTED_NON_AMPLICOL_CELL_COUNT = 1848
_SUPPORTED_SEED_CATALOG_CARDINALITIES = frozenset(
    {
        (1646, 284, 1362),
        (1666, 288, 1378),
        (1706, 296, 1410),
        (1796, 314, 1482),
        (
            EXPECTED_CATALOG_CELL_COUNT,
            EXPECTED_AMPLICOL_CELL_COUNT,
            EXPECTED_NON_AMPLICOL_CELL_COUNT,
        ),
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_PROFILE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_LEGACY_METHOD = "original-amplicol-generated-library"
_LEGACY_VALIDATION_METHOD = "independent-original-amplicol-oracle"
_COMPILER_FIELDS = frozenset(
    {"identity", "version", "flags", "target", "executable_sha256"}
)
_HASHED_ARCHIVE_BASENAMES = frozenset(
    {
        "current.json",
        "manifest.json",
        "result.json",
        "worker-result.json",
        "report-workspace.json",
        "report_environment.json",
        "measurement_lineage.json",
        "pyAmpliCol.pdf",
    }
)


class CampaignResetError(RuntimeError):
    """The requested campaign epoch transition is not auditable."""


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignResetError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CampaignResetError(f"{label} must be a JSON object: {path}")
    return payload


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CampaignResetError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_revision(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _GIT_REVISION_RE.fullmatch(value) is None:
        raise CampaignResetError(f"{label} is not a full Git revision")
    return value


def _current_pointer_path(record: CurrentRecord) -> Path:
    return record.manifest_path.parent.parent.parent / "current.json"


def _legacy_contract(
    record: CurrentRecord,
    *,
    requires_selector: bool,
    expected_legacy_revision: str | None = None,
    expected_point_digest: str | None = None,
) -> dict[str, object]:
    provenance = record.result.get("provenance")
    validation = record.result.get("validation")
    selector = record.result.get("selector_contract")
    resources = record.result.get("resources")
    if not isinstance(provenance, Mapping):
        raise CampaignResetError(f"{record.cell_id}: provenance is missing")
    if provenance.get("method") != _LEGACY_METHOD:
        raise CampaignResetError(
            f"{record.cell_id}: result is not original generated-library AmpliCol"
        )
    legacy_revision = _require_revision(
        provenance.get("revision"),
        label=f"{record.cell_id}.provenance.revision",
    )
    if (
        expected_legacy_revision is not None
        and legacy_revision != expected_legacy_revision
    ):
        raise CampaignResetError(
            f"{record.cell_id}: original-AmpliCol revision is not "
            "contributor-lock pinned"
        )
    source_revision = _require_revision(
        provenance.get("report_source_revision"),
        label=f"{record.cell_id}.provenance.report_source_revision",
    )
    source_tree = _require_revision(
        provenance.get("report_source_tree"),
        label=f"{record.cell_id}.provenance.report_source_tree",
    )
    if (
        provenance.get("report_measured_source_revision") != source_revision
        or provenance.get("report_measured_source_tree") != source_tree
        or provenance.get("report_source_clean") is not True
    ):
        raise CampaignResetError(
            f"{record.cell_id}: report source provenance is not internally exact"
        )
    compiler = provenance.get("compiler")
    if not isinstance(compiler, Mapping) or set(compiler) != _COMPILER_FIELDS:
        raise CampaignResetError(f"{record.cell_id}: compiler identity is missing")
    if (
        not isinstance(compiler.get("identity"), str)
        or not compiler["identity"]
        or not isinstance(compiler.get("version"), str)
        or not compiler["version"]
        or not isinstance(compiler.get("target"), str)
        or not compiler["target"]
        or not isinstance(compiler.get("flags"), list)
        or not all(isinstance(flag, str) for flag in compiler["flags"])
    ):
        raise CampaignResetError(
            f"{record.cell_id}: compiler identity/version/flags/target are malformed"
        )
    _require_sha256(
        compiler.get("executable_sha256"),
        label=f"{record.cell_id}.compiler.executable_sha256",
    )
    if (
        not isinstance(validation, Mapping)
        or validation.get("status") != "ok"
        or validation.get("method") != _LEGACY_VALIDATION_METHOD
    ):
        raise CampaignResetError(
            f"{record.cell_id}: numerical validation is not successful"
        )
    validation_point_digest = _require_sha256(
        validation.get("point_digest"),
        label=f"{record.cell_id}.validation.point_digest",
    )
    if requires_selector:
        if not isinstance(selector, Mapping):
            raise CampaignResetError(
                f"{record.cell_id}: LC selector contract is missing"
            )
        from .agreements import (
            LC_COMMON_COMPONENT_FIELD,
            validate_lc_common_component,
        )
        from .runner import SelectorContract

        try:
            typed_selector = SelectorContract.from_mapping(selector)
            validate_lc_common_component(
                validation.get(LC_COMMON_COMPONENT_FIELD),
                expected_cell_id=record.cell_id,
                selector_contract=selector,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise CampaignResetError(
                f"{record.cell_id}: LC selector evidence is invalid"
            ) from error
        if typed_selector.point_digest != validation_point_digest:
            raise CampaignResetError(
                f"{record.cell_id}: LC selector and validation points differ"
            )
        if (
            expected_point_digest is not None
            and typed_selector.point_digest != expected_point_digest
        ):
            raise CampaignResetError(
                f"{record.cell_id}: LC selector point predates the active "
                "deterministic point corpus"
            )
        common_component = validation[LC_COMMON_COMPONENT_FIELD]
        assert isinstance(common_component, Mapping)
        if float(common_component["value"]) <= 0.0:
            raise CampaignResetError(
                f"{record.cell_id}: LC selector common component is structural zero"
            )
        selector_digest: str | None = sha256_payload(dict(selector))
    elif selector is not None:
        raise CampaignResetError(
            f"{record.cell_id}: contracted AmpliCol result has a selector contract"
        )
    else:
        selector_digest = None
    if not isinstance(resources, Mapping):
        raise CampaignResetError(f"{record.cell_id}: resource evidence is missing")
    target = provenance.get("target_runtime_seconds")
    achieved = (
        provenance.get("runtime_profile", {})
        if isinstance(provenance.get("runtime_profile"), Mapping)
        else {}
    )
    measurement = (
        achieved.get("measurement")
        if isinstance(achieved, Mapping)
        else None
    )
    if (
        not isinstance(target, (int, float))
        or isinstance(target, bool)
        or not math.isfinite(float(target))
        or float(target) <= 0.0
        or not isinstance(measurement, Mapping)
        or measurement.get("target_runtime_achieved") is not True
    ):
        raise CampaignResetError(
            f"{record.cell_id}: finite target-runtime evidence is missing"
        )
    return {
        "legacy_revision": legacy_revision,
        "report_source_revision": source_revision,
        "report_source_tree": source_tree,
        "compiler_sha256": sha256_payload(dict(compiler)),
        "selector_contract_sha256": selector_digest,
        "validation_sha256": sha256_payload(dict(validation)),
        "resource_evidence_sha256": sha256_payload(dict(resources)),
        "target_runtime_seconds": float(target),
        "workload_specific_generation": (
            provenance.get("generation_timing_is_workload_specific") is True
        ),
        "row_selection_policy": provenance.get("row_selection_policy"),
        "selector_color_word_policy": provenance.get(
            "selector_color_word_policy"
        ),
    }


def _pin_for_record(
    record: CurrentRecord,
    *,
    profile: str,
    catalog: ReportCatalog,
    policy: CampaignPolicy,
    expected_legacy_revision: str,
) -> dict[str, object]:
    cell = catalog.cell(record.cell_id)
    if cell.measurement.execution_mode is not ExecutionMode.AMPLICOL:
        raise CampaignResetError(f"{record.cell_id}: seed cell is not AmpliCol")
    if record.result.get("status") != ResultStatus.OK.value:
        raise CampaignResetError(f"{record.cell_id}: seed result is not ok")
    validate_measurement(record.result, expected_cell=cell)
    expected_point_digest: str | None = None
    if cell.measurement.accuracy is Accuracy.LC:
        from .measurement import shared_validation_points
        from .runner import point_digest

        expected_point_digest = point_digest(shared_validation_points(cell.process))
    contract = _legacy_contract(
        record,
        requires_selector=(
            cell.measurement.accuracy is Accuracy.LC
        ),
        expected_legacy_revision=expected_legacy_revision,
        expected_point_digest=expected_point_digest,
    )
    validate_policy_measurement(
        policy,
        profile,
        cell,
        record.result,
        expected_source_revision=str(contract["report_source_revision"]),
        expected_source_tree=str(contract["report_source_tree"]),
    )
    pointer = _current_pointer_path(record)
    return {
        "cell_id": record.cell_id,
        "attempt_id": record.attempt_id,
        "current_pointer_sha256": sha256_path(pointer),
        "manifest_sha256": record.manifest_sha256,
        "result_sha256": sha256_path(record.result_path),
        "artifact_count": len(record.artifacts),
        "artifact_manifest_sha256": sha256_payload(
            [
                {
                    "path": artifact.relative_path,
                    "sha256": artifact.sha256,
                    "size": artifact.size,
                }
                for artifact in record.artifacts
            ]
        ),
        "contract": contract,
    }


def _lc_seed_family_rejections(
    *,
    cells: Mapping[str, CellSpec],
    records_by_cell: Mapping[str, CurrentRecord],
    admitted_cell_ids: set[str],
) -> dict[str, str]:
    """Reject LC seeds unless both layouts authenticate one exact selector.

    A selected-flow seed is the selector authority for a subsequently measured
    all-flow row.  Reusing only one historical layout, or two layouts carrying
    different contracts, can therefore make a fresh dependent compare values
    evaluated at different physical components.  Reject the complete pair so
    the current adapter derives both rows from the same deterministic point and
    selector; never rewrite historical numerical evidence.
    """

    families: dict[tuple[object, ...], dict[Workload, str]] = {}
    for cell_id, cell in cells.items():
        if (
            cell.measurement.execution_mode is not ExecutionMode.AMPLICOL
            or cell.measurement.accuracy is not Accuracy.LC
        ):
            continue
        key = (
            cell.dataset_id,
            cell.process,
            cell.process_key,
            cell.n_final,
            cell.measurement,
            cell.variant,
        )
        families.setdefault(key, {})[cell.workload] = cell_id

    rejected: dict[str, str] = {}
    for family in families.values():
        selected_id = family.get(Workload.SELECTED_FLOW)
        all_flow_id = family.get(Workload.ALL_FLOW)
        if selected_id is None or all_flow_id is None:
            continue
        admitted = {
            cell_id
            for cell_id in (selected_id, all_flow_id)
            if cell_id in admitted_cell_ids
        }
        if not admitted:
            continue
        if admitted != {selected_id, all_flow_id}:
            reason = (
                "LC seed family is incomplete; selected/all-flow rows must be "
                "remeasured under one selector authority"
            )
            rejected.update((cell_id, reason) for cell_id in admitted)
            continue
        selected = records_by_cell[selected_id].result.get("selector_contract")
        all_flow = records_by_cell[all_flow_id].result.get("selector_contract")
        if selected != all_flow:
            reason = (
                "LC selected/all-flow seed contracts differ; both layouts must "
                "be remeasured under one selector authority"
            )
            rejected[selected_id] = reason
            rejected[all_flow_id] = reason
    return rejected


def build_seed_manifest(
    *,
    profile: str,
    store: ArtifactStore,
    policy: CampaignPolicy,
    final_source_revision: str,
    final_source_tree: str,
    expected_legacy_revision: str,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> dict[str, object]:
    """Return the fail-closed same-profile AmpliCol seed manifest."""

    if _PROFILE_RE.fullmatch(profile) is None:
        raise CampaignResetError(f"invalid report profile {profile!r}")
    _require_revision(final_source_revision, label="final_source_revision")
    _require_revision(final_source_tree, label="final_source_tree")
    _require_revision(
        expected_legacy_revision,
        label="expected_legacy_revision",
    )
    pins_by_cell: dict[str, dict[str, object]] = {}
    records_by_cell: dict[str, CurrentRecord] = {}
    rejected_by_cell: dict[str, str] = {}
    cells = {cell.cell_id: cell for cell in catalog.measurement_cells()}
    for record in store.recover_current_records():
        cell = cells.get(record.cell_id)
        if (
            cell is None
            or cell.measurement.execution_mode is not ExecutionMode.AMPLICOL
        ):
            continue
        records_by_cell[record.cell_id] = record
        try:
            pins_by_cell[record.cell_id] = _pin_for_record(
                record,
                profile=profile,
                catalog=catalog,
                policy=policy,
                expected_legacy_revision=expected_legacy_revision,
            )
        except (CampaignResetError, KeyError, ValueError) as error:
            rejected_by_cell[record.cell_id] = str(error)
    family_rejections = _lc_seed_family_rejections(
        cells=cells,
        records_by_cell=records_by_cell,
        admitted_cell_ids=set(pins_by_cell),
    )
    for cell_id, reason in family_rejections.items():
        pins_by_cell.pop(cell_id, None)
        rejected_by_cell[cell_id] = reason
    pins = sorted(pins_by_cell.values(), key=lambda item: str(item["cell_id"]))
    rejected = [
        {"cell_id": cell_id, "reason": reason}
        for cell_id, reason in sorted(rejected_by_cell.items())
    ]
    amplicol_count = sum(
        cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        for cell in cells.values()
    )
    if (
        len(cells) != EXPECTED_CATALOG_CELL_COUNT
        or amplicol_count != EXPECTED_AMPLICOL_CELL_COUNT
        or len(cells) - amplicol_count != EXPECTED_NON_AMPLICOL_CELL_COUNT
    ):
        raise CampaignResetError(
            "campaign reset catalog cardinality differs from "
            f"{EXPECTED_CATALOG_CELL_COUNT}/{EXPECTED_AMPLICOL_CELL_COUNT}/"
            f"{EXPECTED_NON_AMPLICOL_CELL_COUNT}"
        )
    payload: dict[str, object] = {
        "schema": SEED_SCHEMA,
        "profile": profile,
        "final_source_revision": final_source_revision,
        "final_source_tree": final_source_tree,
        "expected_legacy_revision": expected_legacy_revision,
        "catalog_cell_count": len(cells),
        "amplicol_catalog_cell_count": amplicol_count,
        "seed_count": len(pins),
        "pins": pins,
        "rejected": rejected,
    }
    payload["seed_manifest_sha256"] = sha256_payload(payload)
    return payload


@dataclass(frozen=True, slots=True)
class OriginalAmplicolSeed:
    """Digest-pinned authorization for host-local inherited baselines."""

    profile: str
    payload: Mapping[str, object]
    pins_by_cell: Mapping[str, Mapping[str, object]]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        profile: str,
        store: ArtifactStore,
        catalog: ReportCatalog = REPORT_CATALOG,
        expected_final_source_revision: str | None = None,
        expected_final_source_tree: str | None = None,
        expected_manifest_sha256: str | None = None,
    ) -> OriginalAmplicolSeed:
        raw = _read_object(path, label="original AmpliCol seed")
        digest = raw.pop("seed_manifest_sha256", None)
        if raw.get("schema") != SEED_SCHEMA or raw.get("profile") != profile:
            raise CampaignResetError("original AmpliCol seed identity differs")
        if _require_sha256(digest, label="seed_manifest_sha256") != sha256_payload(
            raw
        ):
            raise CampaignResetError("original AmpliCol seed digest differs")
        if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
            raise CampaignResetError(
                "original AmpliCol seed is not pinned by the campaign marker"
            )
        final_revision = _require_revision(
            raw.get("final_source_revision"),
            label="seed.final_source_revision",
        )
        final_tree = _require_revision(
            raw.get("final_source_tree"),
            label="seed.final_source_tree",
        )
        if (
            expected_final_source_revision is not None
            and final_revision != expected_final_source_revision
        ):
            raise CampaignResetError(
                "original AmpliCol seed final source revision differs"
            )
        if (
            expected_final_source_tree is not None
            and final_tree != expected_final_source_tree
        ):
            raise CampaignResetError("original AmpliCol seed final source tree differs")
        catalog_cells = catalog.measurement_cells()
        expected_amplicol_count = sum(
            cell.measurement.execution_mode is ExecutionMode.AMPLICOL
            for cell in catalog_cells
        )
        seed_catalog_count = raw.get("catalog_cell_count")
        seed_amplicol_count = raw.get("amplicol_catalog_cell_count")
        seed_non_amplicol_count = (
            seed_catalog_count - seed_amplicol_count
            if isinstance(seed_catalog_count, int)
            and not isinstance(seed_catalog_count, bool)
            and isinstance(seed_amplicol_count, int)
            and not isinstance(seed_amplicol_count, bool)
            else None
        )
        if (
            len(catalog_cells) != EXPECTED_CATALOG_CELL_COUNT
            or expected_amplicol_count != EXPECTED_AMPLICOL_CELL_COUNT
            or len(catalog_cells) - expected_amplicol_count
            != EXPECTED_NON_AMPLICOL_CELL_COUNT
            or (
                seed_catalog_count,
                seed_amplicol_count,
                seed_non_amplicol_count,
            )
            not in _SUPPORTED_SEED_CATALOG_CARDINALITIES
            or seed_catalog_count > len(catalog_cells)
            or seed_amplicol_count > expected_amplicol_count
        ):
            raise CampaignResetError(
                "original AmpliCol seed catalog coverage differs"
            )
        pins = raw.get("pins")
        if not isinstance(pins, list) or raw.get("seed_count") != len(pins):
            raise CampaignResetError("original AmpliCol seed pin count differs")
        by_cell: dict[str, Mapping[str, object]] = {}
        for item in pins:
            if not isinstance(item, Mapping):
                raise CampaignResetError("original AmpliCol seed pin is malformed")
            cell_id = item.get("cell_id")
            if not isinstance(cell_id, str) or cell_id in by_cell:
                raise CampaignResetError("original AmpliCol seed cell is duplicated")
            cell = catalog.cell(cell_id)
            if cell.measurement.execution_mode is not ExecutionMode.AMPLICOL:
                raise CampaignResetError(f"{cell_id}: seed pin is not AmpliCol")
            record = store.load_current(cell_id)
            if record is None or not cls._matches(record, item):
                raise CampaignResetError(f"{cell_id}: seeded current digest differs")
            by_cell[cell_id] = item
        complete = dict(raw)
        complete["seed_manifest_sha256"] = digest
        return cls(profile=profile, payload=complete, pins_by_cell=by_cell)

    @staticmethod
    def _matches(record: CurrentRecord, pin: Mapping[str, object]) -> bool:
        contract = pin.get("contract")
        try:
            requires_selector = (
                REPORT_CATALOG.cell(record.cell_id).measurement.accuracy.value == "lc"
            )
        except KeyError:
            return False
        try:
            observed = _legacy_contract(
                record,
                requires_selector=requires_selector,
                expected_legacy_revision=(
                    str(contract.get("legacy_revision"))
                    if isinstance(contract, Mapping)
                    else None
                ),
            )
        except CampaignResetError:
            return False
        return (
            record.result.get("status") == ResultStatus.OK.value
            and record.attempt_id == pin.get("attempt_id")
            and record.manifest_sha256 == pin.get("manifest_sha256")
            and sha256_path(_current_pointer_path(record))
            == pin.get("current_pointer_sha256")
            and sha256_path(record.result_path) == pin.get("result_sha256")
            and len(record.artifacts) == pin.get("artifact_count")
            and sha256_payload(
                [
                    {
                        "path": artifact.relative_path,
                        "sha256": artifact.sha256,
                        "size": artifact.size,
                    }
                    for artifact in record.artifacts
                ]
            )
            == pin.get("artifact_manifest_sha256")
            and isinstance(contract, Mapping)
            and dict(contract) == observed
        )

    @property
    def digest(self) -> str:
        return str(self.payload["seed_manifest_sha256"])

    def source_for_current(
        self,
        record: CurrentRecord,
        *,
        active_revision: str,
        active_tree: str,
    ) -> tuple[str, str] | None:
        if (
            active_revision != self.payload.get("final_source_revision")
            or active_tree != self.payload.get("final_source_tree")
        ):
            return None
        provenance = record.result.get("provenance")
        if isinstance(provenance, Mapping) and (
            provenance.get("report_source_revision") == active_revision
            and provenance.get("report_source_tree") == active_tree
            and provenance.get("report_measured_source_revision")
            == active_revision
            and provenance.get("report_measured_source_tree") == active_tree
        ):
            return active_revision, active_tree
        pin = self.pins_by_cell.get(record.cell_id)
        if pin is None or not self._matches(record, pin):
            return None
        contract = pin.get("contract")
        assert isinstance(contract, Mapping)
        return (
            str(contract["report_source_revision"]),
            str(contract["report_source_tree"]),
        )


def load_seed_if_present(
    *,
    profile: str,
    store: ArtifactStore,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> OriginalAmplicolSeed | None:
    path = store.artifact_root / SEED_FILENAME
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise CampaignResetError(f"seed manifest is not a regular file: {path}")
    marker = _load_campaign_marker(
        store.artifact_root,
        profile=profile,
        require_ready=True,
    )
    return OriginalAmplicolSeed.load(
        path,
        profile=profile,
        store=store,
        catalog=catalog,
        expected_final_source_revision=str(marker["source_revision"]),
        expected_final_source_tree=str(marker["source_tree"]),
        expected_manifest_sha256=str(marker["seed_manifest_sha256"]),
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            getattr(errno, "EOPNOTSUPP", errno.EPERM),
        }:
            raise
        shutil.copy2(source, destination)


def stage_seed_store(
    *,
    source_store: ArtifactStore,
    destination_root: Path,
    destination_lock_root: Path,
    seed_manifest: Mapping[str, object],
) -> ArtifactStore:
    """Create a fresh store containing only digest-pinned seed attempts."""

    if destination_root.exists() or destination_lock_root.exists():
        raise CampaignResetError("fresh campaign staging roots already exist")
    destination = ArtifactStore(
        artifact_root=destination_root,
        lock_root=destination_lock_root,
    )
    pins = seed_manifest.get("pins")
    if not isinstance(pins, list):
        raise CampaignResetError("seed manifest pins are malformed")
    records = {
        record.cell_id: record for record in source_store.recover_current_records()
    }
    for pin in pins:
        if not isinstance(pin, Mapping) or not isinstance(pin.get("cell_id"), str):
            raise CampaignResetError("seed manifest pin is malformed")
        record = records.get(str(pin["cell_id"]))
        if record is None or not OriginalAmplicolSeed._matches(record, pin):
            raise CampaignResetError(f"{pin['cell_id']}: source seed changed")
        destination_cell = destination._cell_root(record.cell_id)
        source_cell = _current_pointer_path(record).parent
        destination_cell.mkdir(parents=True, exist_ok=True)
        _link_or_copy(
            source_cell / "current.json",
            destination_cell / "current.json",
        )
        source_attempt = record.manifest_path.parent
        destination_attempt = (
            destination_cell / "attempts" / record.attempt_id
        )
        destination_attempt.mkdir(parents=True, exist_ok=True)
        authenticated_files = {
            record.manifest_path,
            *(artifact.path for artifact in record.artifacts),
        }
        for source in sorted(authenticated_files):
            if source.is_symlink() or not source.is_file():
                raise CampaignResetError(
                    f"seed contains an unauthenticated file type: {source}"
                )
            relative = source.relative_to(source_attempt)
            _link_or_copy(source, destination_attempt / relative)
    complete = dict(seed_manifest)
    _atomic_write_json(destination_root / SEED_FILENAME, complete)
    OriginalAmplicolSeed.load(
        destination_root / SEED_FILENAME,
        profile=str(seed_manifest["profile"]),
        store=destination,
        expected_final_source_revision=str(
            seed_manifest["final_source_revision"]
        ),
        expected_final_source_tree=str(seed_manifest["final_source_tree"]),
        expected_manifest_sha256=str(seed_manifest["seed_manifest_sha256"]),
    )
    return destination


def lightweight_archive_inventory(roots: Sequence[Path]) -> dict[str, object]:
    """Inventory an archive without re-reading manifest-authenticated binaries."""

    entries: list[dict[str, object]] = []
    for root in roots:
        resolved = root.expanduser().resolve(strict=False)
        if not resolved.exists():
            continue
        for path in sorted((resolved, *resolved.rglob("*"))):
            stat = path.lstat()
            entry: dict[str, object] = {
                "root": os.fspath(resolved),
                "path": (
                    "."
                    if path == resolved
                    else path.relative_to(resolved).as_posix()
                ),
                "type": (
                    "symlink"
                    if path.is_symlink()
                    else "directory"
                    if path.is_dir()
                    else "file"
                    if path.is_file()
                    else "special"
                ),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            if path.is_file() and (
                path.name in _HASHED_ARCHIVE_BASENAMES
                or path.suffix in {".json", ".pdf"}
            ):
                entry["sha256"] = sha256_path(path)
            entries.append(entry)
    payload: dict[str, object] = {
        "schema": "pyamplicol-performance-archive-inventory-v1",
        "entries": entries,
    }
    payload["inventory_sha256"] = sha256_payload(payload)
    return payload


def campaign_marker(
    *,
    campaign_id: str,
    profile: str,
    source_revision: str,
    source_tree: str,
    policy_sha256: str,
    seed_manifest_sha256: str,
    archive_manifest_sha256: str,
) -> dict[str, object]:
    if not campaign_id or "/" in campaign_id or "\x00" in campaign_id:
        raise CampaignResetError("campaign_id is not a safe component")
    payload: dict[str, object] = {
        "schema": CAMPAIGN_MARKER_SCHEMA,
        "state": "PREPARED",
        "campaign_id": campaign_id,
        "profile": profile,
        "source_revision": _require_revision(
            source_revision, label="source_revision"
        ),
        "source_tree": _require_revision(source_tree, label="source_tree"),
        "policy_sha256": _require_sha256(
            policy_sha256, label="policy_sha256"
        ),
        "seed_manifest_sha256": _require_sha256(
            seed_manifest_sha256, label="seed_manifest_sha256"
        ),
        "archive_manifest_sha256": _require_sha256(
            archive_manifest_sha256, label="archive_manifest_sha256"
        ),
        "baseline_gate_sha256": None,
        "prepared_marker_sha256": None,
    }
    payload["marker_sha256"] = sha256_payload(payload)
    return payload


def _load_campaign_marker(
    artifact_root: Path,
    *,
    profile: str,
    require_ready: bool,
) -> dict[str, object]:
    path = artifact_root / CAMPAIGN_MARKER_FILENAME
    marker = _read_object(path, label="campaign marker")
    digest = marker.pop("marker_sha256", None)
    if (
        marker.get("schema") != CAMPAIGN_MARKER_SCHEMA
        or marker.get("profile") != profile
        or marker.get("state") not in {"PREPARED", "READY"}
        or _require_sha256(digest, label="marker_sha256")
        != sha256_payload(marker)
    ):
        raise CampaignResetError("active campaign marker differs")
    if require_ready:
        if marker.get("state") != "READY":
            raise CampaignResetError("campaign baseline gate is not READY")
        gate_digest = _require_sha256(
            marker.get("baseline_gate_sha256"),
            label="baseline_gate_sha256",
        )
        prepared_digest = _require_sha256(
            marker.get("prepared_marker_sha256"),
            label="prepared_marker_sha256",
        )
        gate = _read_object(
            artifact_root / BASELINE_GATE_FILENAME,
            label="campaign baseline gate",
        )
        observed_gate_digest = gate.pop("baseline_gate_sha256", None)
        if (
            _require_sha256(
                observed_gate_digest,
                label="baseline_gate.baseline_gate_sha256",
            )
            != sha256_payload(gate)
            or observed_gate_digest != gate_digest
            or gate.get("prepared_marker_sha256") != prepared_digest
            or gate.get("campaign_id") != marker.get("campaign_id")
            or gate.get("profile") != profile
            or gate.get("source_revision") != marker.get("source_revision")
            or gate.get("source_tree") != marker.get("source_tree")
        ):
            raise CampaignResetError("campaign baseline gate differs")
    marker["marker_sha256"] = digest
    return marker


def mark_campaign_ready(
    artifact_root: Path,
    *,
    baseline_gate_sha256: str,
) -> dict[str, object]:
    raw_marker = _read_object(
        artifact_root / CAMPAIGN_MARKER_FILENAME,
        label="campaign marker",
    )
    profile = raw_marker.get("profile")
    if not isinstance(profile, str) or not profile:
        raise CampaignResetError("campaign marker profile is malformed")
    marker = _load_campaign_marker(
        artifact_root,
        profile=profile,
        require_ready=False,
    )
    if marker.get("state") != "PREPARED":
        raise CampaignResetError("campaign marker is not PREPARED")
    prepared_digest = str(marker.pop("marker_sha256"))
    marker["state"] = "READY"
    marker["baseline_gate_sha256"] = _require_sha256(
        baseline_gate_sha256,
        label="baseline_gate_sha256",
    )
    marker["prepared_marker_sha256"] = prepared_digest
    marker["marker_sha256"] = sha256_payload(marker)
    _atomic_write_json(artifact_root / CAMPAIGN_MARKER_FILENAME, marker)
    return marker


def assert_campaign_marker(
    artifact_root: Path,
    *,
    campaign_id: str,
    profile: str,
    source_revision: str,
    source_tree: str,
    expected_marker_sha256: str | None = None,
    require_ready: bool = True,
) -> dict[str, object]:
    marker = _load_campaign_marker(
        artifact_root,
        profile=profile,
        require_ready=require_ready,
    )
    digest = marker["marker_sha256"]
    if (
        marker.get("campaign_id") != campaign_id
        or marker.get("profile") != profile
        or marker.get("source_revision") != source_revision
        or marker.get("source_tree") != source_tree
    ):
        raise CampaignResetError("active campaign marker differs")
    if expected_marker_sha256 is not None and digest != expected_marker_sha256:
        raise CampaignResetError("active campaign marker digest differs")
    return marker


def temporary_staging_root(parent: Path, *, campaign_id: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".campaign-reset-{campaign_id}.",
            dir=parent,
        )
    )


@dataclass(frozen=True, slots=True)
class ResetTransactionPaths:
    """Explicit paths participating in one same-filesystem epoch transition."""

    source_publication: Path
    source_artifact_root: Path
    source_coordination_root: Path
    destination_publication: Path
    destination_artifact_root: Path
    destination_coordination_root: Path
    archive_root: Path
    staging_root: Path
    guard_path: Path

    @property
    def journal_path(self) -> Path:
        return self.staging_root / RESET_JOURNAL_FILENAME


def _transaction_payload(
    *,
    state: str,
    profile: str,
    campaign_id: str,
    archive_id: str,
    paths: ResetTransactionPaths,
    archive_manifest_sha256: str,
    seed_manifest_sha256: str,
    marker_sha256: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": RESET_JOURNAL_SCHEMA,
        "state": state,
        "profile": profile,
        "campaign_id": campaign_id,
        "archive_id": archive_id,
        "paths": {
            field: os.fspath(getattr(paths, field))
            for field in (
                "source_publication",
                "source_artifact_root",
                "source_coordination_root",
                "destination_publication",
                "destination_artifact_root",
                "destination_coordination_root",
                "archive_root",
                "staging_root",
                "guard_path",
            )
        },
        "archive_manifest_sha256": archive_manifest_sha256,
        "seed_manifest_sha256": seed_manifest_sha256,
        "marker_sha256": marker_sha256,
    }
    payload["journal_sha256"] = sha256_payload(payload)
    return payload


def write_prepared_journal(
    *,
    profile: str,
    campaign_id: str,
    archive_id: str,
    paths: ResetTransactionPaths,
    archive_manifest_sha256: str,
    seed_manifest_sha256: str,
    marker_sha256: str,
) -> dict[str, object]:
    payload = _transaction_payload(
        state="PREPARED",
        profile=profile,
        campaign_id=campaign_id,
        archive_id=archive_id,
        paths=paths,
        archive_manifest_sha256=archive_manifest_sha256,
        seed_manifest_sha256=seed_manifest_sha256,
        marker_sha256=marker_sha256,
    )
    _atomic_write_json(paths.journal_path, payload)
    return payload


def _validated_journal(path: Path) -> dict[str, Any]:
    payload = _read_object(path, label="campaign reset journal")
    digest = payload.pop("journal_sha256", None)
    if (
        payload.get("schema") != RESET_JOURNAL_SCHEMA
        or payload.get("state") not in {"PREPARED", "COMMITTING", "COMMITTED"}
        or _require_sha256(digest, label="journal_sha256")
        != sha256_payload(payload)
    ):
        raise CampaignResetError("campaign reset journal differs")
    payload["journal_sha256"] = digest
    return payload


def _paths_from_journal(payload: Mapping[str, object]) -> ResetTransactionPaths:
    raw = payload.get("paths")
    if not isinstance(raw, Mapping):
        raise CampaignResetError("campaign reset journal paths are malformed")
    names = (
        "source_publication",
        "source_artifact_root",
        "source_coordination_root",
        "destination_publication",
        "destination_artifact_root",
        "destination_coordination_root",
        "archive_root",
        "staging_root",
        "guard_path",
    )
    if set(raw) != set(names):
        raise CampaignResetError("campaign reset journal path set differs")
    resolved: dict[str, Path] = {}
    for name in names:
        value = raw.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise CampaignResetError(f"campaign reset path {name} is not absolute")
        resolved[name] = Path(value).resolve(strict=False)
    return ResetTransactionPaths(**resolved)


def _rename_forward(source: Path, destination: Path, *, label: str) -> None:
    if destination.exists():
        if source.exists():
            raise CampaignResetError(
                f"{label}: both source and destination exist"
            )
        return
    if not source.exists():
        raise CampaignResetError(f"{label}: source is missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    _fsync_directory(destination.parent)


def _filesystem_device(path: Path) -> int:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise CampaignResetError(
                f"cannot resolve filesystem device for {path}"
            )
        candidate = parent
    return candidate.stat().st_dev


def _require_transaction_filesystem(paths: ResetTransactionPaths) -> None:
    devices = {
        _filesystem_device(path)
        for path in (
            paths.source_publication,
            paths.source_artifact_root,
            paths.source_coordination_root,
            paths.destination_publication,
            paths.destination_artifact_root,
            paths.destination_coordination_root,
            paths.archive_root,
            paths.staging_root,
            paths.guard_path,
        )
    }
    if len(devices) != 1:
        raise CampaignResetError(
            "campaign reset journal crosses filesystem boundaries"
        )


def _install_publication_tree(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise CampaignResetError("staged publication is not a regular directory")
    destination.mkdir(parents=True, exist_ok=True)
    for staged in sorted(source.rglob("*")):
        relative = staged.relative_to(source)
        target = destination / relative
        if staged.is_symlink():
            raise CampaignResetError(
                f"staged publication contains a symlink: {staged}"
            )
        if staged.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not staged.is_file():
            raise CampaignResetError(
                f"staged publication contains a special file: {staged}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4()}.tmp")
        shutil.copy2(staged, temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)


def commit_or_recover_reset(journal_path: Path) -> dict[str, object]:
    """Finish a PREPARED reset transaction; repeated calls are idempotent."""

    journal = _validated_journal(journal_path)
    paths = _paths_from_journal(journal)
    _require_transaction_filesystem(paths)
    paths.guard_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.guard_path.open("a+b") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
        try:
            journal = _validated_journal(journal_path)
            if journal["state"] == "COMMITTED":
                return journal
            committing = dict(journal)
            committing["state"] = "COMMITTING"
            committing.pop("journal_sha256", None)
            committing["journal_sha256"] = sha256_payload(committing)
            _atomic_write_json(journal_path, committing)

            archive_artifacts = paths.archive_root / "artifacts"
            archive_coordination = paths.archive_root / "coordination"
            archive_publication = paths.archive_root / "publication"
            staged_artifacts = paths.staging_root / "fresh-artifacts"
            staged_coordination = paths.staging_root / "fresh-coordination"
            staged_publication = paths.staging_root / "fresh-publication"
            staged_archive_publication = (
                paths.staging_root / "archive-publication"
            )
            staged_archive_manifest = (
                paths.staging_root / "archive-manifest.json"
            )
            _rename_forward(
                staged_archive_publication,
                archive_publication,
                label="archive publication",
            )
            _rename_forward(
                staged_archive_manifest,
                paths.archive_root / "archive-manifest.json",
                label="archive manifest",
            )
            _rename_forward(
                paths.source_artifact_root,
                archive_artifacts,
                label="archive artifact root",
            )
            _rename_forward(
                paths.source_coordination_root,
                archive_coordination,
                label="archive coordination root",
            )
            _rename_forward(
                staged_artifacts,
                paths.destination_artifact_root,
                label="install fresh artifact root",
            )
            _rename_forward(
                staged_coordination,
                paths.destination_coordination_root,
                label="install fresh coordination root",
            )
            _install_publication_tree(
                staged_publication,
                paths.destination_publication,
            )
            committed = dict(committing)
            committed["state"] = "COMMITTED"
            committed.pop("journal_sha256", None)
            committed["journal_sha256"] = sha256_payload(committed)
            archive_journal = paths.archive_root / RESET_JOURNAL_FILENAME
            _atomic_write_json(archive_journal, committed)
            _atomic_write_json(journal_path, committed)
            return committed
        finally:
            fcntl.flock(guard.fileno(), fcntl.LOCK_UN)


__all__ = [
    "BASELINE_GATE_FILENAME",
    "CAMPAIGN_MARKER_FILENAME",
    "CAMPAIGN_MARKER_SCHEMA",
    "EXPECTED_AMPLICOL_CELL_COUNT",
    "EXPECTED_CATALOG_CELL_COUNT",
    "EXPECTED_NON_AMPLICOL_CELL_COUNT",
    "RESET_JOURNAL_FILENAME",
    "SEED_FILENAME",
    "SEED_SCHEMA",
    "CampaignResetError",
    "OriginalAmplicolSeed",
    "ResetTransactionPaths",
    "assert_campaign_marker",
    "build_seed_manifest",
    "campaign_marker",
    "canonical_json_bytes",
    "commit_or_recover_reset",
    "lightweight_archive_inventory",
    "load_seed_if_present",
    "mark_campaign_ready",
    "sha256_path",
    "sha256_payload",
    "stage_seed_store",
    "temporary_staging_root",
    "write_prepared_journal",
]
