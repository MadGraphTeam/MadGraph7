#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Fail-closed semantic census comparison for recurrence artifacts.

This developer-only tool deliberately reuses pyAmpliCol's authenticated
artifact inventory, prepared-kernel-pack decoders, and native recurrence exact
section loader.  It does not decode ``recurrence-runtime.pacbin`` or
``recurrence-binding.bin`` independently.

The persisted exact sections enumerate current support/state projections,
runtime interactions, closures, selector axes, and layout.  Some pre-lowering
``CurrentCoreKey`` fields—notably dynamic LC color-state and local helicity
identities—are intentionally absent from the runtime ABI.  Those fields are
therefore authenticated transitively by the process/native semantic digests
and the exact selector/catalog projections; this tool never claims to
reconstruct them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Any

from pyamplicol.models.prepared import EAGER_KERNEL_ABI, PreparedKernelPack
from pyamplicol.models.recurrence_direct_template import (
    RECURRENCE_DIRECT_IDENTITY_FINALIZER,
)
from pyamplicol.runtime.recurrence_exact._execution import _role_index
from pyamplicol.runtime.recurrence_exact._plan_v2 import (
    DIRECT_NONE_U32,
    _load_recurrence_exact_sections_v1,
    _RecurrenceExactSectionsV1,
)
from tools.developer.recurrence_artifact_compare import (
    ArtifactSnapshot,
    _json_object,
    _load_artifact,
)
from tools.developer.recurrence_artifact_compare import (
    ComparisonError as ArtifactComparisonError,
)

CENSUS_KIND = "pyamplicol-recurrence-semantic-census"
CENSUS_SCHEMA_VERSION = 1
COMPARISON_KIND = "pyamplicol-recurrence-semantic-census-comparison"
COMPARISON_SCHEMA_VERSION = 1
MAX_REPORTED_DIFFERENCES = 256
_UNASSIGNED_ROW = object()

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EXECUTION_PATH_RE = re.compile(r"processes/([^/]+)/execution[.]json")
_REQUIRED_DOMAINS = frozenset(
    {
        "digests",
        "currents",
        "sources",
        "contribution_multisets",
        "closures",
        "selectors",
        "normalization_and_parameters",
        "semantic_catalog_bindings",
        "runtime_layout",
    }
)
_CATALOG_SECTION_NAMES = (
    "parameters",
    "current_states",
    "sources",
    "quantum_flows",
    "transitions",
    "propagators",
    "closures",
    "color_contractions",
    "symmetry_proofs",
    "runtime_helicity_contracts",
    "evaluator_bindings",
)
_STORAGE_AUTHENTICATION_DIGEST_FIELDS = frozenset(
    {
        "runtime_container_index_sha256",
        "runtime_container_sha256",
        "runtime_plan_member_sha256",
        "process_binding_sha256",
    }
)


class SemanticCensusError(RuntimeError):
    """Raised when an input cannot produce an authenticated exact census."""


@dataclass(frozen=True, slots=True)
class ExecutorSemanticBinding:
    """Authenticated semantic context for one exact runtime executor."""

    executor_id: int
    role: str
    semantic_digest: str
    semantic_template_ids: tuple[str, ...]
    direct_template: Mapping[str, object]
    referenced_semantic_records: tuple[Mapping[str, object], ...]


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SemanticCensusError(f"{context} must be an object with string keys")
    return value


def _sequence(value: object, context: str) -> Sequence[Any]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise SemanticCensusError(f"{context} must be an array")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SemanticCensusError(f"{context} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticCensusError(f"{context} must be a non-negative integer")
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise SemanticCensusError(
            "semantic census value is not canonical JSON"
        ) from error


def _domain_census(name: str, records: Iterable[object]) -> dict[str, object]:
    digest = hashlib.sha256()
    domain = name.encode("ascii")
    digest.update(len(domain).to_bytes(8, "little"))
    digest.update(domain)
    _count = 0
    for _count, record in enumerate(records, start=1):
        encoded = _canonical_bytes(record)
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return {
        "record_count": _count,
        "sha256": digest.hexdigest(),
    }


def _exact_factor(
    sections: _RecurrenceExactSectionsV1,
    factor_id: int,
    context: str,
) -> tuple[int, int, int, int]:
    if factor_id < 0 or factor_id >= len(sections.exact_factors):
        raise SemanticCensusError(f"{context} references an absent exact factor")
    return astuple(sections.exact_factors[factor_id])


def _momentum(
    sections: _RecurrenceExactSectionsV1,
    form_id: int,
    context: str,
) -> tuple[tuple[int, int], ...]:
    if form_id < 0 or form_id >= len(sections.momentum_forms):
        raise SemanticCensusError(f"{context} references an absent momentum form")
    form = sections.momentum_forms[form_id]
    stop = form.term_start + form.term_count
    if form.term_start < 0 or stop > len(sections.momentum_terms):
        raise SemanticCensusError(f"{context} momentum form is out of bounds")
    terms = tuple(
        (term.source_slot, term.coefficient)
        for term in sections.momentum_terms[form.term_start : stop]
    )
    if any(
        slot >= sections.external_source_count or coefficient == 0
        for slot, coefficient in terms
    ) or tuple(slot for slot, _ in terms) != tuple(sorted({slot for slot, _ in terms})):
        raise SemanticCensusError(f"{context} momentum form is not canonical")
    return terms


def _executor_rows(
    sections: _RecurrenceExactSectionsV1,
    role: str,
    row_count: int,
    bindings: Mapping[int, ExecutorSemanticBinding],
    *,
    allow_unbound: bool = False,
) -> tuple[ExecutorSemanticBinding | None, ...]:
    expected_role = _role_index(role)
    result: list[ExecutorSemanticBinding | object | None] = [
        _UNASSIGNED_ROW
    ] * row_count
    for group_index, group in enumerate(sections.row_groups):
        if group.role != expected_role:
            continue
        binding = (
            None
            if allow_unbound and group.executor_id == DIRECT_NONE_U32
            else bindings.get(group.executor_id)
        )
        if binding is None and not (
            allow_unbound and group.executor_id == DIRECT_NONE_U32
        ):
            raise SemanticCensusError(
                f"{role} row group {group_index} has no matching semantic executor"
            )
        if binding is not None and binding.role != role:
            raise SemanticCensusError(
                f"{role} row group {group_index} has no matching semantic executor"
            )
        stop = group.row_start + group.row_count
        if group.row_start < 0 or stop > row_count:
            raise SemanticCensusError(
                f"{role} row group {group_index} is out of bounds"
            )
        for row_id in range(group.row_start, stop):
            if result[row_id] is not _UNASSIGNED_ROW:
                raise SemanticCensusError(
                    f"{role} runtime row {row_id} belongs to multiple row groups"
                )
            result[row_id] = binding
    if any(binding is _UNASSIGNED_ROW for binding in result):
        missing = next(
            index for index, binding in enumerate(result) if binding is _UNASSIGNED_ROW
        )
        raise SemanticCensusError(
            f"{role} runtime row {missing} has no exact row-group owner"
        )
    return tuple(
        binding if isinstance(binding, ExecutorSemanticBinding) else None
        for binding in result
    )


def _executor_reference(binding: ExecutorSemanticBinding) -> dict[str, object]:
    return {
        "direct_executor_id": binding.executor_id,
        "direct_semantic_digest": binding.semantic_digest,
        "semantic_template_ids": list(binding.semantic_template_ids),
    }


def _current_records(
    sections: _RecurrenceExactSectionsV1,
    state_templates: Sequence[Mapping[str, object]],
) -> Iterable[dict[str, object]]:
    for current in sections.currents:
        if current.state_template_id >= len(state_templates):
            raise SemanticCensusError(
                f"current {current.semantic_id} references an absent state template"
            )
        state = state_templates[current.state_template_id]
        state_identity = {
            "template_id": state.get("template_id"),
            "semantic_digest": state.get("semantic_digest"),
        }
        if not isinstance(state_identity["template_id"], str):
            raise SemanticCensusError(
                f"current {current.semantic_id} state template has no identity"
            )
        _sha256(
            state_identity["semantic_digest"],
            f"current {current.semantic_id} state semantic digest",
        )
        momentum = _momentum(
            sections,
            current.momentum_form_id,
            f"current {current.semantic_id}",
        )
        source: object = None
        if current.source_row != DIRECT_NONE_U32:
            if current.source_row >= len(sections.sources):
                raise SemanticCensusError(
                    f"current {current.semantic_id} references an absent source row"
                )
            source_row = sections.sources[current.source_row]
            source = {
                "row_id": current.source_row,
                "source_slot": source_row.source_slot,
                "spin_state_class": source_row.spin_state_class,
                "source_template_or_dispatch_domain": (
                    source_row.source_template_or_dispatch_domain
                ),
            }
        yield {
            "semantic_id": current.semantic_id,
            "node_kind": current.node_kind,
            "state_template_id": current.state_template_id,
            "state_template": state_identity,
            "component_count": current.component_count,
            "momentum": momentum,
            "support_source_slots": [slot for slot, _ in momentum],
            "stage": current.stage,
            # The runtime ABI persists selector domains, not the discarded
            # dynamic color/local-helicity CurrentCoreKey fields.
            "color_selector_domain_id": current.selector_domain_id,
            "helicity_selector_domain_id": current.selector_domain_id,
            "source": source,
            "finalization_row": (
                None
                if current.finalization_row == DIRECT_NONE_U32
                else current.finalization_row
            ),
        }


def _source_records(
    sections: _RecurrenceExactSectionsV1,
    bindings: Mapping[int, ExecutorSemanticBinding],
    source_metadata: Mapping[str, object],
) -> Iterable[dict[str, object]]:
    yield {"runtime_source_metadata": source_metadata}
    executor_rows = _executor_rows(
        sections,
        "source",
        len(sections.sources),
        bindings,
        allow_unbound=True,
    )
    for row_id, (source, binding) in enumerate(
        zip(sections.sources, executor_rows, strict=True)
    ):
        yield {
            "row_id": row_id,
            "source_slot": source.source_slot,
            "destination_component_base": source.destination_base,
            "momentum": _momentum(
                sections, source.momentum_form_id, f"source row {row_id}"
            ),
            "source_template_or_dispatch_domain": (
                source.source_template_or_dispatch_domain
            ),
            "spin_state_class": source.spin_state_class,
            "exact_factor": _exact_factor(
                sections, source.exact_factor_id, f"source row {row_id}"
            ),
            "selector_domain_id": source.selector_domain_id,
            "executor": (None if binding is None else _executor_reference(binding)),
        }


def _contribution_records(
    sections: _RecurrenceExactSectionsV1,
    bindings: Mapping[int, ExecutorSemanticBinding],
) -> Iterable[dict[str, object]]:
    executor_rows = _executor_rows(
        sections,
        "contribution",
        len(sections.contributions),
        bindings,
    )
    for row_id, (row, binding) in enumerate(
        zip(sections.contributions, executor_rows, strict=True)
    ):
        parent1 = (
            None
            if row.parent1_base == DIRECT_NONE_U32
            else {
                "component_base": row.parent1_base,
                "momentum": _momentum(
                    sections,
                    row.parent1_momentum,
                    f"contribution {row_id} parent 1",
                ),
            }
        )
        yield {
            # The native constructor emits this table in canonical key order,
            # so hashing persisted order is also a canonical multiset census,
            # independent of transient candidate-enumeration order.
            "canonical_row_id": row_id,
            "parents": [
                {
                    "component_base": row.parent0_base,
                    "momentum": _momentum(
                        sections,
                        row.parent0_momentum,
                        f"contribution {row_id} parent 0",
                    ),
                },
                parent1,
            ],
            "destination_component_base": row.destination_base,
            "exact_factor": _exact_factor(
                sections, row.exact_factor_id, f"contribution {row_id}"
            ),
            "selector_domain_id": row.selector_domain_id,
            "flags": row.flags,
            "executor": _executor_reference(binding),
        }


def _closure_records(
    sections: _RecurrenceExactSectionsV1,
    bindings: Mapping[int, ExecutorSemanticBinding],
) -> Iterable[dict[str, object]]:
    executor_rows = _executor_rows(
        sections, "closure", len(sections.closures), bindings
    )
    for row_id, (row, binding) in enumerate(
        zip(sections.closures, executor_rows, strict=True)
    ):
        parent1 = (
            None
            if row.parent1_base == DIRECT_NONE_U32
            else {
                "component_base": row.parent1_base,
                "momentum": _momentum(
                    sections,
                    row.parent1_momentum,
                    f"closure {row_id} parent 1",
                ),
            }
        )
        yield {
            "canonical_row_id": row_id,
            "parents": [
                {
                    "component_base": row.parent0_base,
                    "momentum": _momentum(
                        sections,
                        row.parent0_momentum,
                        f"closure {row_id} parent 0",
                    ),
                },
                parent1,
            ],
            "amplitude_destination_id": row.amplitude_destination_id,
            "exact_factor": _exact_factor(
                sections, row.exact_factor_id, f"closure {row_id}"
            ),
            "component_factor_start": row.component_factor_start,
            "component_count": row.component_count,
            "selector_domain_id": row.selector_domain_id,
            "flags": row.flags,
            "executor": _executor_reference(binding),
        }


def _selector_records(
    sections: _RecurrenceExactSectionsV1,
    selector_metadata: Mapping[str, object],
) -> Iterable[dict[str, object]]:
    yield {"runtime_selector_metadata": selector_metadata}
    tables = (
        ("replay_targets", sections.replay_targets),
        ("source_permutations", sections.source_permutations),
        ("replay_momentum_signs", sections.replay_momentum_signs),
        ("replay_helicity_map", sections.replay_helicity_map),
        ("amplitude_destinations", sections.amplitude_destinations),
        ("resolved_helicities", sections.resolved_helicities),
        ("source_state_assignments", sections.source_state_assignments),
        ("source_dispatch_variants", sections.source_dispatch_variants),
        ("source_embeddings", sections.source_embeddings),
        ("source_projections", sections.source_projections),
        ("resolved_source_selections", sections.resolved_source_selections),
        ("public_helicities", sections.public_helicities),
        ("public_flow_ids", sections.public_flow_ids),
    )
    for name, rows in tables:
        yield {"table": name, "row_count": len(rows)}
        for row_id, row in enumerate(rows):
            yield {
                "table": name,
                "row_id": row_id,
                "row": (astuple(row) if hasattr(row, "__dataclass_fields__") else row),
            }


def _semantic_catalog_records(
    bindings: Mapping[int, ExecutorSemanticBinding],
    sections: _RecurrenceExactSectionsV1,
) -> Iterable[dict[str, object]]:
    used_executor_ids = sorted(
        {
            group.executor_id
            for group in sections.row_groups
            if group.executor_id != DIRECT_NONE_U32
        }
    )
    for executor_id in used_executor_ids:
        binding = bindings.get(executor_id)
        if binding is None:
            raise SemanticCensusError(
                f"runtime schedule references absent direct executor {executor_id}"
            )
        yield {
            "direct_executor_id": executor_id,
            "direct_template": binding.direct_template,
            "referenced_semantic_records": list(binding.referenced_semantic_records),
        }


def _runtime_layout_records(
    sections: _RecurrenceExactSectionsV1,
) -> Iterable[dict[str, object]]:
    yield {
        "counts": {
            "current_arena_components": sections.current_arena_components,
            "amplitude_destination_count": sections.amplitude_destination_count,
            "parameter_value_count": sections.parameter_value_count,
            "external_source_count": sections.external_source_count,
        }
    }
    for name, rows in (
        ("currents", sections.currents),
        ("sources", sections.sources),
        ("contributions", sections.contributions),
        ("finalizations", sections.finalizations),
        ("closures", sections.closures),
        ("row_groups", sections.row_groups),
        ("momentum_forms", sections.momentum_forms),
        ("momentum_terms", sections.momentum_terms),
        ("exact_factors", sections.exact_factors),
        ("executors", sections.executors),
    ):
        yield {"table": name, "row_count": len(rows)}
        for row_id, row in enumerate(rows):
            yield {"table": name, "row_id": row_id, "row": astuple(row)}


def build_semantic_census(
    *,
    sections: _RecurrenceExactSectionsV1,
    process_metadata: Mapping[str, object],
    executor_bindings: Mapping[int, ExecutorSemanticBinding],
    state_templates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build one timing-free census from already authenticated typed inputs."""

    raw_digests = _mapping(process_metadata.get("digests"), "process metadata digests")
    digests = {
        key: value
        for key, value in raw_digests.items()
        if key not in _STORAGE_AUTHENTICATION_DIGEST_FIELDS
    }
    normalization = _mapping(
        process_metadata.get("normalization_and_parameters"),
        "process normalization/parameters",
    )
    selector_metadata = _mapping(
        process_metadata.get("selectors"), "process selector metadata"
    )
    source_metadata = _mapping(
        process_metadata.get("sources"), "process source metadata"
    )
    if digests.get("process_semantic_digest") != sections.semantic_digest:
        raise SemanticCensusError(
            "process metadata semantic digest does not match exact sections"
        )
    if digests.get("runtime_layout_digest") != sections.runtime_layout_digest:
        raise SemanticCensusError(
            "process metadata runtime-layout digest does not match exact sections"
        )

    domains = {
        "digests": _domain_census("digests", (digests,)),
        "currents": _domain_census(
            "currents", _current_records(sections, state_templates)
        ),
        "sources": _domain_census(
            "sources",
            _source_records(sections, executor_bindings, source_metadata),
        ),
        "contribution_multisets": _domain_census(
            "contribution_multisets",
            _contribution_records(sections, executor_bindings),
        ),
        "closures": _domain_census(
            "closures", _closure_records(sections, executor_bindings)
        ),
        "selectors": _domain_census(
            "selectors", _selector_records(sections, selector_metadata)
        ),
        "normalization_and_parameters": _domain_census(
            "normalization_and_parameters", (normalization,)
        ),
        "semantic_catalog_bindings": _domain_census(
            "semantic_catalog_bindings",
            _semantic_catalog_records(executor_bindings, sections),
        ),
        "runtime_layout": _domain_census(
            "runtime_layout", _runtime_layout_records(sections)
        ),
    }
    if set(domains) != _REQUIRED_DOMAINS:  # pragma: no cover - construction invariant
        raise SemanticCensusError("semantic census domain coverage is incomplete")
    content = {
        "kind": CENSUS_KIND,
        "schema_version": CENSUS_SCHEMA_VERSION,
        "process_id": sections.process_id,
        "strategy": sections.strategy,
        "coverage": {
            "exact_sections_decoder": (
                "pyamplicol.runtime.recurrence_exact._plan_v2."
                "_load_recurrence_exact_sections_v1"
            ),
            "prepared_catalog_decoder": (
                "pyamplicol.models.prepared.PreparedKernelPack.from_dict"
            ),
            "directly_enumerated": [
                "current semantic IDs, node kinds, state templates, momentum support",
                "source rows and spin-state classes",
                "canonical contribution rows, signs, and runtime endpoints",
                "closures and amplitude destinations",
                "flow/helicity/source-dispatch selector axes",
                "normalization, parameter projection, and runtime layout",
            ],
            "transitively_authenticated_not_reconstructed": [
                "pre-lowering dynamic LC color-state IDs",
                "pre-lowering local current-helicity identities",
                "discarded construction-only flavour/quantum/coupling key fields",
            ],
            "canonical_contribution_policy": (
                "persisted-native-canonical-emission-is-the-multiset-order-v1"
            ),
        },
        "domains": domains,
    }
    content["census_sha256"] = hashlib.sha256(_canonical_bytes(content)).hexdigest()
    return content


def _semantic_record_index(
    catalog_payload: Mapping[str, object],
) -> tuple[dict[str, Mapping[str, object]], tuple[Mapping[str, object], ...]]:
    by_id: dict[str, Mapping[str, object]] = {}
    state_templates: tuple[Mapping[str, object], ...] = ()
    for section_name in _CATALOG_SECTION_NAMES:
        rows = _sequence(
            catalog_payload.get(section_name),
            f"recurrence catalog {section_name}",
        )
        parsed_rows: list[Mapping[str, object]] = []
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"recurrence catalog {section_name}[{index}]")
            identity = row.get("template_id", row.get("resolver_key"))
            if not isinstance(identity, str) or not identity:
                raise SemanticCensusError(
                    f"recurrence catalog {section_name}[{index}] lacks an identity"
                )
            if identity in by_id:
                raise SemanticCensusError(
                    f"recurrence catalog repeats semantic identity {identity!r}"
                )
            by_id[identity] = row
            parsed_rows.append(row)
        if section_name == "current_states":
            state_templates = tuple(parsed_rows)
    if not state_templates:
        raise SemanticCensusError("recurrence catalog has no current-state templates")
    return by_id, state_templates


def _referenced_record_closure(
    roots: Iterable[str],
    records: Mapping[str, Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    pending = list(roots)
    visited: set[str] = set()

    def references(value: object) -> Iterable[str]:
        if isinstance(value, str):
            if value in records:
                yield value
        elif isinstance(value, Mapping):
            for child in value.values():
                yield from references(child)
        elif isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            for child in value:
                yield from references(child)

    while pending:
        identity = pending.pop()
        if identity in visited:
            continue
        record = records.get(identity)
        if record is None:
            raise SemanticCensusError(
                f"direct executor references absent semantic record {identity!r}"
            )
        visited.add(identity)
        pending.extend(
            reference for reference in references(record) if reference not in visited
        )
    return tuple(records[identity] for identity in sorted(visited))


def _executor_bindings(
    pack: PreparedKernelPack,
    sections: _RecurrenceExactSectionsV1,
) -> tuple[
    dict[int, ExecutorSemanticBinding],
    tuple[Mapping[str, object], ...],
]:
    semantic_catalog = pack.recurrence_template_catalog
    direct_catalog = pack.recurrence_direct_template_catalog
    if semantic_catalog is None or direct_catalog is None:
        raise SemanticCensusError(
            "prepared kernel pack lacks authenticated recurrence catalogs"
        )
    semantic_payload = semantic_catalog.to_dict()
    record_by_id, state_templates = _semantic_record_index(semantic_payload)
    bindings: dict[int, ExecutorSemanticBinding] = {}
    for template in direct_catalog.templates:
        roots = template.semantic_template_ids
        if template.evaluator_resolver_key != RECURRENCE_DIRECT_IDENTITY_FINALIZER:
            roots = (*roots, template.evaluator_resolver_key)
        binding = ExecutorSemanticBinding(
            executor_id=template.direct_executor_id,
            role=template.role,
            semantic_digest=template.semantic_digest,
            semantic_template_ids=template.semantic_template_ids,
            direct_template=template.to_dict(),
            referenced_semantic_records=_referenced_record_closure(roots, record_by_id),
        )
        if binding.executor_id in bindings:
            raise SemanticCensusError(
                f"direct catalog repeats executor {binding.executor_id}"
            )
        bindings[binding.executor_id] = binding

    if len(sections.executors) != len(bindings):
        raise SemanticCensusError(
            "exact executor inventory differs from the authenticated direct catalog"
        )
    for executor in sections.executors:
        binding = bindings.get(executor.executor_id)
        if binding is None:
            raise SemanticCensusError(
                f"exact sections reference absent executor {executor.executor_id}"
            )
        template = binding.direct_template
        expected = (
            binding.role,
            template.get("destination_operation"),
            tuple(_sequence(template.get("parent_component_counts"), "parent counts")),
            template.get("destination_component_count"),
            template.get("momentum_operand_count"),
            _mapping(template.get("payload_binding"), "direct payload binding").get(
                "prepared_kernel_id"
            ),
            _mapping(template.get("payload_binding"), "direct payload binding").get(
                "runtime_template"
            ),
        )
        actual = (
            executor.role,
            executor.destination_operation,
            executor.parent_component_counts,
            executor.destination_component_count,
            executor.momentum_operand_count,
            executor.prepared_kernel_id,
            executor.runtime_template,
        )
        if actual != expected:
            raise SemanticCensusError(
                "exact executor "
                f"{executor.executor_id} differs from its direct template"
            )
    return bindings, state_templates


def _schedule_summary_counts(
    sections: _RecurrenceExactSectionsV1,
) -> dict[str, int]:
    return {
        "amplitude_destination_count": len(sections.amplitude_destinations),
        "closure_term_count": len(sections.closures),
        "contribution_count": len(sections.contributions),
        "current_count": len(sections.currents),
        "exact_factor_count": len(sections.exact_factors),
        "finalization_count": len(sections.finalizations),
        "replay_target_count": len(sections.replay_targets),
        "resolved_helicity_count": len(sections.resolved_helicities),
        "source_row_count": len(sections.sources),
    }


def _process_metadata(
    *,
    snapshot: ArtifactSnapshot,
    process_id: str,
    execution: Mapping[str, object],
    sections: _RecurrenceExactSectionsV1,
    pack: PreparedKernelPack,
) -> dict[str, object]:
    if execution.get("key") != process_id:
        raise SemanticCensusError("execution key does not match its process path")
    plan = _mapping(execution.get("plan"), "recurrence execution plan")
    inspection = _mapping(
        plan.get("inspection_summary"), "recurrence inspection summary"
    )
    binding = _mapping(plan.get("process_binding"), "recurrence process binding")
    runtime_schedule = _mapping(
        plan.get("runtime_schedule"), "recurrence runtime schedule"
    )
    runtime_metadata = _mapping(
        execution.get("runtime_metadata"), "recurrence runtime metadata"
    )
    schedule_summary = _mapping(
        inspection.get("schedule"), "recurrence schedule summary"
    )
    expected_counts = _schedule_summary_counts(sections)
    for name, expected in expected_counts.items():
        if schedule_summary.get(name) != expected:
            raise SemanticCensusError(
                f"inspection schedule count {name} differs from exact sections"
            )

    direct_catalog = pack.recurrence_direct_template_catalog
    semantic_catalog = pack.recurrence_template_catalog
    assert direct_catalog is not None and semantic_catalog is not None
    schedule_digest = _sha256(
        inspection.get("schedule_digest"), "inspection schedule digest"
    )
    schedule_path = runtime_schedule.get("path")
    if not isinstance(schedule_path, str) or schedule_path not in snapshot.payloads:
        raise SemanticCensusError(
            "runtime schedule path is absent from the authenticated artifact"
        )
    schedule_payload = snapshot.payloads[schedule_path]
    if (
        _sha256(
            runtime_schedule.get("sha256"),
            "runtime container digest",
        )
        != schedule_payload.sha256
        or _nonnegative_int(
            runtime_schedule.get("size_bytes"),
            "runtime container size",
        )
        != schedule_payload.size_bytes
    ):
        raise SemanticCensusError(
            "runtime schedule metadata does not authenticate its payload"
        )
    path_parts = Path(schedule_path).parts
    if len(path_parts) < 3 or path_parts[-2] != schedule_digest:
        raise SemanticCensusError(
            "runtime schedule path does not embed the inspection schedule digest"
        )

    semantic_digest = _sha256(
        binding.get("process_semantic_digest"), "process semantic digest"
    )
    runtime_layout_digest = _sha256(
        inspection.get("runtime_layout_digest"), "runtime-layout digest"
    )
    _sha256(
        runtime_schedule.get("index_sha256"),
        "runtime container index digest",
    )
    _sha256(
        _mapping(
            inspection.get("runtime_container_member"),
            "runtime container member",
        ).get("sha256"),
        "runtime plan member digest",
    )
    binding_path = binding.get("path")
    if not isinstance(binding_path, str) or not binding_path:
        raise SemanticCensusError("process binding path is invalid")
    binding_payload = snapshot.payloads.get(f"processes/{process_id}/{binding_path}")
    if binding_payload is None or (
        _sha256(binding.get("sha256"), "process binding digest")
        != binding_payload.sha256
        or _nonnegative_int(binding.get("size_bytes"), "process binding size")
        != binding_payload.size_bytes
    ):
        raise SemanticCensusError(
            "process binding metadata does not authenticate its payload"
        )
    equalities = (
        (sections.process_id, process_id, "exact process identity"),
        (sections.strategy, inspection.get("lc_flow_layout"), "recurrence strategy"),
        (sections.semantic_digest, semantic_digest, "exact semantic digest"),
        (
            plan.get("builder_input_sha256"),
            semantic_digest,
            "builder-input semantic digest",
        ),
        (
            inspection.get("semantic_digest"),
            semantic_digest,
            "inspection semantic digest",
        ),
        (
            sections.runtime_layout_digest,
            runtime_layout_digest,
            "exact runtime-layout digest",
        ),
        (
            binding.get("schedule_digest"),
            schedule_digest,
            "process-binding schedule digest",
        ),
        (
            execution.get("direct_template_catalog_digest"),
            direct_catalog.catalog_digest,
            "execution direct-template digest",
        ),
        (
            plan.get("direct_template_catalog_digest"),
            direct_catalog.catalog_digest,
            "plan direct-template digest",
        ),
        (
            execution.get("prepared_kernel_pack_digest"),
            direct_catalog.prepared_kernel_pack_digest,
            "execution prepared-pack digest",
        ),
        (
            plan.get("prepared_kernel_pack_digest"),
            direct_catalog.prepared_kernel_pack_digest,
            "plan prepared-pack digest",
        ),
        (
            direct_catalog.recurrence_template_catalog_digest,
            semantic_catalog.header.catalog_digest,
            "semantic/direct recurrence catalog digest",
        ),
    )
    for actual, expected, context in equalities:
        if actual != expected:
            raise SemanticCensusError(f"{context} is inconsistent")

    digests = {
        "process_semantic_digest": semantic_digest,
        "native_schedule_semantic_digest": _sha256(
            binding.get("native_schedule_semantic_digest"),
            "native schedule semantic digest",
        ),
        "runtime_layout_digest": runtime_layout_digest,
        "schedule_digest": schedule_digest,
        "builder_input_sha256": _sha256(
            plan.get("builder_input_sha256"), "builder input digest"
        ),
        "direct_template_catalog_digest": direct_catalog.catalog_digest,
        "recurrence_template_catalog_digest": (semantic_catalog.header.catalog_digest),
        "prepared_kernel_pack_digest": direct_catalog.prepared_kernel_pack_digest,
        "process_binding_remap": binding.get("remap"),
        "process_support_words": binding.get("process_support_words"),
        "abis": {
            "builder_input": execution.get("builder_input_abi"),
            "direct_backend": execution.get("direct_backend_abi"),
            "direct_template": execution.get("direct_template_abi"),
            "recurrence_plan": execution.get("recurrence_plan_abi"),
            "runtime_layout": execution.get("runtime_layout_abi"),
            "process_binding": binding.get("abi"),
        },
    }
    return {
        "digests": digests,
        "normalization_and_parameters": {
            "normalization": runtime_metadata.get("normalization"),
            "parameter_projection": runtime_metadata.get("parameter_projection"),
            "prepared_parameter_defaults": runtime_metadata.get(
                "prepared_parameter_defaults"
            ),
            "runtime_parameters": runtime_metadata.get("runtime_parameters"),
            "parameter_value_count": sections.parameter_value_count,
        },
        "selectors": {
            "color_accuracy": execution.get("color_accuracy"),
            "color_contraction": runtime_metadata.get("color_contraction"),
            "public_color_flows": runtime_metadata.get("public_color_flows"),
            "runtime_options": execution.get("runtime_options"),
        },
        "sources": {
            "process": execution.get("process"),
            "external_pdg_order": execution.get("external_pdg_order"),
            "external_legs": runtime_metadata.get("external_legs"),
            "particle_masses": runtime_metadata.get("particle_masses"),
            "source_templates": runtime_metadata.get("source_templates"),
        },
    }


def _load_pack(snapshot: ArtifactSnapshot) -> PreparedKernelPack:
    relative = "model/eager-kernel-pack.json"
    payload = snapshot.payloads.get(relative)
    if payload is None:
        raise SemanticCensusError(
            "recurrence artifact lacks the prepared kernel-pack manifest"
        )
    raw = _json_object(payload.path, description="prepared kernel pack")
    kernel_abi = raw.pop("eager_kernel_abi", None)
    if kernel_abi != EAGER_KERNEL_ABI:
        raise SemanticCensusError(
            f"prepared kernel pack has unsupported eager ABI {kernel_abi!r}"
        )
    try:
        return PreparedKernelPack.from_dict(raw)
    except ValueError as error:
        raise SemanticCensusError(
            "prepared kernel pack failed authoritative decoding"
        ) from error


def _artifact_censuses(snapshot: ArtifactSnapshot) -> dict[str, dict[str, object]]:
    pack = _load_pack(snapshot)
    result: dict[str, dict[str, object]] = {}
    for execution_path in sorted(snapshot.executions):
        match = _EXECUTION_PATH_RE.fullmatch(execution_path)
        if match is None:  # pragma: no cover - authenticated by artifact loader
            raise SemanticCensusError(
                f"unexpected recurrence execution path {execution_path!r}"
            )
        process_id = match.group(1)
        try:
            sections = _load_recurrence_exact_sections_v1(snapshot.root, process_id)
        except Exception as error:
            raise SemanticCensusError(
                f"native exact-section loading failed for process {process_id!r}"
            ) from error
        bindings, state_templates = _executor_bindings(pack, sections)
        execution = snapshot.executions[execution_path]
        metadata = _process_metadata(
            snapshot=snapshot,
            process_id=process_id,
            execution=execution,
            sections=sections,
            pack=pack,
        )
        result[process_id] = build_semantic_census(
            sections=sections,
            process_metadata=metadata,
            executor_bindings=bindings,
            state_templates=state_templates,
        )
    if not result:
        raise SemanticCensusError("recurrence artifact has no process census")
    return result


def compare_census_sets(
    baseline: Mapping[str, Mapping[str, object]],
    candidate: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Compare two already-built process-census sets without an allowlist."""

    differences: list[dict[str, object]] = []
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    for process_id in sorted(baseline_ids - candidate_ids):
        differences.append(
            {
                "kind": "process-inventory",
                "process_id": process_id,
                "baseline": "present",
                "candidate": "missing",
            }
        )
    for process_id in sorted(candidate_ids - baseline_ids):
        differences.append(
            {
                "kind": "process-inventory",
                "process_id": process_id,
                "baseline": "missing",
                "candidate": "present",
            }
        )
    process_reports: list[dict[str, object]] = []
    for process_id in sorted(baseline_ids & candidate_ids):
        left = baseline[process_id]
        right = candidate[process_id]
        for field in ("kind", "schema_version", "process_id", "strategy", "coverage"):
            if left.get(field) != right.get(field):
                differences.append(
                    {
                        "kind": "census-contract",
                        "process_id": process_id,
                        "field": field,
                        "baseline": left.get(field),
                        "candidate": right.get(field),
                    }
                )
        left_domains = _mapping(left.get("domains"), "baseline census domains")
        right_domains = _mapping(right.get("domains"), "candidate census domains")
        if (
            set(left_domains) != _REQUIRED_DOMAINS
            or set(right_domains) != _REQUIRED_DOMAINS
        ):
            raise SemanticCensusError(
                f"process {process_id!r} has incomplete census domains"
            )
        matched: list[str] = []
        changed: list[str] = []
        for domain in sorted(_REQUIRED_DOMAINS):
            left_domain = _mapping(
                left_domains[domain], f"baseline {process_id} {domain}"
            )
            right_domain = _mapping(
                right_domains[domain], f"candidate {process_id} {domain}"
            )
            if left_domain == right_domain:
                matched.append(domain)
                continue
            changed.append(domain)
            differences.append(
                {
                    "kind": "semantic-domain",
                    "process_id": process_id,
                    "domain": domain,
                    "baseline": dict(left_domain),
                    "candidate": dict(right_domain),
                }
            )
        process_reports.append(
            {
                "process_id": process_id,
                "strategy": left.get("strategy"),
                "matched_domains": matched,
                "changed_domains": changed,
                "baseline_census_sha256": left.get("census_sha256"),
                "candidate_census_sha256": right.get("census_sha256"),
                "passes": not changed
                and all(
                    difference.get("process_id") != process_id
                    for difference in differences
                ),
            }
        )
    difference_count = len(differences)
    return {
        "kind": COMPARISON_KIND,
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "passes": difference_count == 0,
        "policy": {
            "allowlisted_differences": [],
            "domain_set": sorted(_REQUIRED_DOMAINS),
            "unknown_or_missing_domains": "reject",
            "nonpersisted_pre_lowering_keys": (
                "authenticate-transitively-never-reconstruct-v1"
            ),
        },
        "processes": process_reports,
        "difference_count": difference_count,
        "differences_truncated": difference_count > MAX_REPORTED_DIFFERENCES,
        "differences": differences[:MAX_REPORTED_DIFFERENCES],
    }


def compare_artifact_censuses(
    baseline_artifact: Path,
    candidate_artifact: Path,
) -> dict[str, object]:
    """Authenticate, decode, census, and compare two recurrence artifacts."""

    try:
        baseline_snapshot = _load_artifact(baseline_artifact, label="baseline")
        candidate_snapshot = _load_artifact(candidate_artifact, label="candidate")
    except ArtifactComparisonError as error:
        raise SemanticCensusError(str(error)) from error
    baseline = _artifact_censuses(baseline_snapshot)
    candidate = _artifact_censuses(candidate_snapshot)
    report = compare_census_sets(baseline, candidate)
    report["baseline"] = {
        "path": str(baseline_snapshot.root),
        "artifact_id": baseline_snapshot.manifest["artifact_id"],
        "process_count": len(baseline),
    }
    report["candidate"] = {
        "path": str(candidate_snapshot.root),
        "artifact_id": candidate_snapshot.manifest["artifact_id"],
        "process_count": len(candidate),
    }
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline", type=Path, required=True)
    result.add_argument("--candidate", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        report = compare_artifact_censuses(arguments.baseline, arguments.candidate)
    except SemanticCensusError as error:
        print(f"recurrence-semantic-census: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
