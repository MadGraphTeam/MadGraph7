# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501 - uninterrupted SHA-256 goldens are easier to audit.
"""Semantic plan-v2 goldens for the Rust eager-lowering transition.

The snapshots deliberately avoid hashing JSON text or packed table bytes.  They
project the Python lowerer's execution semantics into stable records so that a
future plan-v3 Rust lowerer can be compared without reproducing Python object
layout, selector-domain numbering, or serialization details.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import TypeAlias, cast

import pytest

import pyamplicol.generation.service as service_module
from pyamplicol.api import ProcessRequest
from pyamplicol.config import ColorConfig, EvaluatorConfig, RunConfig
from pyamplicol.generation.eager_lowering import (
    EagerExecutionTables,
    PreparedCatalogEagerKernelResolver,
    lower_fused_eager_execution,
)
from pyamplicol.generation.eager_tables import (
    EAGER_PLAN_ABI,
    EAGER_RUNTIME_CAPABILITY,
    EagerSelectorDomainIdRow,
)
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.prepared_catalog import (
    PreparedKernelCatalog,
    build_prepared_kernel_catalog,
)

_ComplexFraction: TypeAlias = tuple[Fraction, Fraction]

_LEGACY_PLAN_ABI = "pyamplicol-eager-plan-v2"
_LEGACY_RUNTIME_CAPABILITY = "rusticol.eager-dag.complex-f64.v1"


@dataclass(frozen=True, slots=True)
class _GoldenCase:
    name: str
    schema: Mapping[str, object]
    tables: EagerExecutionTables
    catalog: PreparedKernelCatalog


def _canonical(value: object) -> object:
    """Make semantic values JSON-canonical while preserving exact f64 bits."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"f64_hex": value.hex()}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical(item) for item in value]
    raise TypeError(f"unsupported semantic-golden value {type(value).__name__}")


def _digest(value: object) -> str:
    payload = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"{field} must be a mapping")
    return cast(Mapping[str, object], value)


def _records(value: object, field: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AssertionError(f"{field} must be a sequence")
    return tuple(_mapping(item, f"{field}[]") for item in value)


def _build_case(
    *,
    name: str,
    expression: str,
    accuracy: str,
    lc_flow_layout: str,
    catalog: PreparedKernelCatalog,
) -> _GoldenCase:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(
        RunConfig(
            action="generate",
            color=ColorConfig(
                accuracy=accuracy,
                lc_flow_layout=lc_flow_layout,
            ),
            evaluator=EvaluatorConfig(execution_mode="eager"),
        ),
        None,
    )
    process_ir = build_process_ir(expression, color_accuracy=accuracy)
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(
            expanded=service_module._ExpandedProcess(
                request=ProcessRequest.parse(expression, name=name),
                process_ir=process_ir,
            ),
            dag=dag,
            coverage=coverage,
        ),
        model,
        index=0,
        phase=PhaseHandle("semantic-golden", None, 1),
    )
    schema, tables = lower_fused_eager_execution(
        dag=prepared.dag,
        model=model,
        resolver=PreparedCatalogEagerKernelResolver(
            prepared.dag,
            catalog.resolver_manifest(),
        ),
        process_id=name,
    )
    return _GoldenCase(name=name, schema=schema, tables=tables, catalog=catalog)


@pytest.fixture(scope="module")
def golden_cases() -> tuple[_GoldenCase, ...]:
    catalog = build_prepared_kernel_catalog(BuiltinSMModel())
    return (
        _build_case(
            name="lc-topology-replay",
            expression="d d~ > z g g",
            accuracy="lc",
            lc_flow_layout="topology-replay",
            catalog=catalog,
        ),
        _build_case(
            name="lc-all-flow-union",
            expression="d d~ > z g g",
            accuracy="lc",
            lc_flow_layout="all-flow-union",
            catalog=catalog,
        ),
        _build_case(
            name="nlc-contracted",
            expression="g g > g g",
            accuracy="nlc",
            lc_flow_layout="topology-replay",
            catalog=catalog,
        ),
        _build_case(
            name="full-contracted",
            expression="g g > g g",
            accuracy="full",
            lc_flow_layout="topology-replay",
            catalog=catalog,
        ),
    )


def _layout_semantics(schema: Mapping[str, object]) -> object:
    current_storage = _mapping(schema["current_storage"], "current_storage")
    value_storage = _mapping(schema["value_storage"], "value_storage")
    stages = _records(schema["stages"], "stages")
    return {
        "parameter_layout": schema["parameter_layout"],
        "model_parameters": schema["model_parameters"],
        "current_component_count": current_storage["component_count"],
        "current_slots": sorted(
            _records(current_storage["current_slots"], "current_slots"),
            key=lambda row: int(row["current_id"]),
        ),
        "value_component_count": value_storage["component_count"],
        "value_slots": sorted(
            _records(value_storage["value_slots"], "value_slots"),
            key=lambda row: int(row["value_slot_id"]),
        ),
        "source_fill": schema["source_fill"],
        "momentum_slots": sorted(
            _records(schema["momentum_slots"], "momentum_slots"),
            key=lambda row: int(row["momentum_slot_id"]),
        ),
        "stages": [
            {
                key: stage[key]
                for key in (
                    "stage_index",
                    "stage_kind",
                    "subset_size",
                    "input_current_ids",
                    "output_current_ids",
                    "input_value_slot_ids",
                    "output_value_slot_ids",
                    "input_momentum_slot_ids",
                )
            }
            for stage in stages
        ],
    }


def _table_semantics(tables: EagerExecutionTables) -> object:
    couplings = tuple(asdict(row) for row in tables.couplings)

    def coupling(row_id: int) -> object:
        if row_id >= len(couplings):
            return {"missing": True}
        return couplings[row_id]

    return {
        "process_key": tables.process_key,
        "couplings": couplings,
        "stages": [
            {
                "stage_index": stage.stage_index,
                "subset_size": stage.subset_size,
                "invocations": [
                    {
                        **asdict(row),
                        "coupling": coupling(row.coupling_slot_id),
                    }
                    for row in stage.invocations
                ],
                "attachments": [asdict(row) for row in stage.attachments],
                "finalizations": [asdict(row) for row in stage.finalizations],
            }
            for stage in tables.stages
        ],
        "closures": [
            {
                **asdict(row),
                "coupling": coupling(row.coupling_slot_id),
            }
            for row in tables.closures
        ],
    }


def _selector_semantics(
    schema: Mapping[str, object], tables: EagerExecutionTables
) -> object:
    selector = tables.selector_closures
    if selector is None:
        return {"enabled": False}
    memberships = tuple(
        tuple(
            row.coherent_group_id
            for row in selector.domain_group_ids[
                domain.member_start : domain.member_start + domain.member_count
            ]
        )
        for domain in selector.domains
    )

    def resolve(rows: Sequence[EagerSelectorDomainIdRow]) -> list[tuple[int, ...]]:
        return [memberships[row.domain_id] for row in rows]

    physics = _mapping(schema["physics"], "physics")
    extensions = _mapping(physics["extensions"], "physics.extensions")
    return {
        "enabled": True,
        "domain_memberships": sorted(set(memberships)),
        "stages": [
            {
                "stage_index": stage.stage_index,
                "invocations": resolve(stage.invocation_domains),
                "attachments": resolve(stage.attachment_domains),
                "unpropagated_finalizations": resolve(
                    stage.unpropagated_finalization_domains
                ),
                "propagated_finalizations": resolve(
                    stage.propagated_finalization_domains
                ),
            }
            for stage in selector.stages
        ],
        "closures": resolve(selector.closure_domains),
        "public_selectors": physics["selectors"],
        "runtime_selectors": extensions.get("runtime_selectors"),
        "lc_topology_replay": schema.get("lc_topology_replay"),
        "helicity_recurrence": schema.get("helicity_recurrence"),
    }


def _reduction_semantics(schema: Mapping[str, object]) -> object:
    physics = _mapping(schema["physics"], "physics")
    amplitude = _mapping(schema["amplitude_stage"], "amplitude_stage")
    return {
        "roots": amplitude["roots"],
        "coherent_groups": amplitude["coherent_groups"],
        "final_reduction": amplitude["final_reduction"],
        "color_contraction": amplitude["color_contraction"],
        "physics_reduction": physics["reduction"],
        "color_components": physics["color_components"],
        "helicities": physics["helicities"],
        "normalization": schema["normalization"],
    }


def _logical_color_contraction_entries(
    contraction: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    entries = _records(contraction["entries"], "color_contraction.entries")
    repeated_raw = contraction.get("repeated_block")
    if repeated_raw is None:
        return entries
    assert not entries
    repeated = _mapping(repeated_raw, "color_contraction.repeated_block")
    component_count = int(repeated["component_count"])
    component_group_ids = tuple(
        int(value) for value in cast(Sequence[object], repeated["component_group_ids"])
    )
    templates = _records(
        repeated["entries"], "color_contraction.repeated_block.entries"
    )
    return tuple(
        {
            "left_group_id": component_group_ids[
                int(entry["left_group_index"]) * component_count + component_index
            ],
            "right_group_id": component_group_ids[
                int(entry["right_group_index"]) * component_count + component_index
            ],
            "weight": entry["weight"],
            "symmetry_factor": entry["symmetry_factor"],
        }
        for component_index in range(component_count)
        for entry in templates
    )


def _exact_semantics(case: _GoldenCase) -> object:
    referenced = case.tables.referenced_kernel_ids
    kernels = [case.catalog.by_id[kernel_id] for kernel_id in sorted(referenced)]
    amplitude = _mapping(case.schema["amplitude_stage"], "amplitude_stage")
    roots = _records(amplitude["roots"], "amplitude_stage.roots")
    return {
        "kernels": [
            {
                "kernel_id": kernel.kernel_id,
                "contract_kind": kernel.contract_kind,
                "canonical_signature": kernel.canonical_signature,
                "exact_expressions": kernel.exact_expressions,
                "inputs": [item.to_dict() for item in kernel.inputs],
                "output_layout": kernel.output_layout,
                "proof_classes": kernel.proof_classes,
            }
            for kernel in kernels
        ],
        "root_exact_factors": [
            {
                "output_index": root["output_index"],
                "color_weight": root["color_weight"],
                "coupling": root["coupling"],
                "helicity_weight": root["helicity_weight"],
                "all_sector_weight": root["all_sector_weight"],
                "contraction_coefficients": _mapping(
                    root["contraction_ir"], "root.contraction_ir"
                )["coefficients"],
            }
            for root in roots
        ],
    }


def _fraction(value: object) -> Fraction:
    return Fraction.from_float(float(value))


def _complex_mul(left: _ComplexFraction, right: _ComplexFraction) -> _ComplexFraction:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _fraction_record(value: Fraction) -> tuple[str, str]:
    return str(value.numerator), str(value.denominator)


def _complex_record(value: _ComplexFraction) -> object:
    return _fraction_record(value[0]), _fraction_record(value[1])


def _resolved_probe(
    schema: Mapping[str, object], tables: EagerExecutionTables
) -> object:
    """Evaluate reduction semantics with deterministic exact rational amplitudes.

    The probe is deliberately backend-independent.  Its synthetic closure
    values exercise closure factors, coherent grouping, diagonal LC weights,
    and sparse NLC/full contractions exactly, without compiling an evaluator.
    """

    amplitude = _mapping(schema["amplitude_stage"], "amplitude_stage")
    roots = _records(amplitude["roots"], "amplitude_stage.roots")
    root_by_output = {int(root["output_index"]): root for root in roots}
    group_amplitudes: dict[int, _ComplexFraction] = defaultdict(
        lambda: (Fraction(0), Fraction(0))
    )
    for closure in tables.closures:
        output = closure.amplitude_index
        root = root_by_output[output]
        seed = (
            Fraction((output * 17) % 23 + 1, 23),
            Fraction((output * 29) % 31 - 15, 31),
        )
        factor = (_fraction(closure.factor_real), _fraction(closure.factor_imag))
        value = _complex_mul(seed, factor)
        group_id = int(root["coherent_group_id"])
        previous = group_amplitudes[group_id]
        group_amplitudes[group_id] = (
            previous[0] + value[0],
            previous[1] + value[1],
        )

    physics = _mapping(schema["physics"], "physics")
    reduction = _mapping(physics["reduction"], "physics.reduction")
    reduction_groups = _records(reduction["groups"], "physics.reduction.groups")
    component_by_group: dict[int, str] = {}
    for record in reduction_groups:
        group_id = int(str(record["id"]).rsplit(":", maxsplit=1)[-1])
        component_by_group[group_id] = (
            f"{record['representative_color_id']}|"
            f"{record['representative_helicity_id']}"
        )

    resolved: dict[str, Fraction] = defaultdict(Fraction)
    contraction = amplitude["color_contraction"]
    if contraction is None:
        groups = _records(amplitude["coherent_groups"], "coherent_groups")
        for group in groups:
            group_id = int(group["group_id"])
            value = group_amplitudes[group_id]
            resolved[component_by_group[group_id]] += _fraction(
                group["all_sector_weight"]
            ) * (value[0] * value[0] + value[1] * value[1])
    else:
        entries = _logical_color_contraction_entries(
            _mapping(contraction, "color_contraction")
        )
        for entry in entries:
            left_id = int(entry["left_group_id"])
            right_id = int(entry["right_group_id"])
            left = group_amplitudes[left_id]
            right = group_amplitudes[right_id]
            product = (
                left[0] * right[0] + left[1] * right[1],
                left[1] * right[0] - left[0] * right[1],
            )
            weight = cast(Sequence[object], entry["weight"])
            contribution = _fraction(entry["symmetry_factor"]) * (
                _fraction(weight[0]) * product[0] - _fraction(weight[1]) * product[1]
            )
            # Sparse colour contractions only connect coherent groups for the
            # same physical helicity. Aggregate by the public resolved axis,
            # not by upper-triangular entry orientation.
            component = component_by_group[left_id]
            assert component == component_by_group[right_id]
            resolved[component] += contribution

    return {
        "coherent_amplitudes": [
            (group_id, _complex_record(value))
            for group_id, value in sorted(group_amplitudes.items())
        ],
        "resolved_group_totals": [
            (component, _fraction_record(value))
            for component, value in sorted(resolved.items())
        ],
        "total": _fraction_record(sum(resolved.values(), Fraction())),
    }


def _diagnostic_counts(case: _GoldenCase) -> dict[str, int]:
    schema = case.schema
    current_storage = _mapping(schema["current_storage"], "current_storage")
    value_storage = _mapping(schema["value_storage"], "value_storage")
    amplitude = _mapping(schema["amplitude_stage"], "amplitude_stage")
    physics = _mapping(schema["physics"], "physics")
    selector = case.tables.selector_closures
    contraction = amplitude["color_contraction"]
    contraction_entries = (
        ()
        if contraction is None
        else _logical_color_contraction_entries(
            _mapping(contraction, "color_contraction")
        )
    )
    return {
        "current_slots": len(_records(current_storage["current_slots"], "currents")),
        "value_slots": len(_records(value_storage["value_slots"], "values")),
        "momentum_slots": len(_records(schema["momentum_slots"], "momenta")),
        "stages": len(case.tables.stages),
        "invocations": case.tables.invocation_count,
        "attachments": case.tables.attachment_count,
        "finalizations": sum(len(stage.finalizations) for stage in case.tables.stages),
        "closures": len(case.tables.closures),
        "couplings": len(case.tables.couplings),
        "selector_domains": 0 if selector is None else len(selector.domains),
        "selector_memberships": (
            0 if selector is None else len(selector.domain_group_ids)
        ),
        "reduction_groups": len(
            _records(
                _mapping(physics["reduction"], "physics.reduction")["groups"],
                "physics.reduction.groups",
            )
        ),
        "color_contraction_entries": len(contraction_entries),
        "amplitude_roots": len(_records(amplitude["roots"], "roots")),
        "referenced_exact_kernels": len(case.tables.referenced_kernel_ids),
    }


def _snapshot(case: _GoldenCase) -> dict[str, object]:
    sections = {
        "layout": _layout_semantics(case.schema),
        "tables": _table_semantics(case.tables),
        "selectors": _selector_semantics(case.schema, case.tables),
        "reductions": _reduction_semantics(case.schema),
        "exact": _exact_semantics(case),
        "resolved": _resolved_probe(case.schema, case.tables),
    }
    return {
        "counts": _diagnostic_counts(case),
        "digests": {name: _digest(value) for name, value in sections.items()},
        "semantic_sha256": _digest(sections),
    }


# Filled from the audited plan-v2 lowerer at source 55bfedc.  These records are
# intentionally small: counts diagnose structural drift and section digests
# identify its semantic owner without checking Python serialization bytes.
_EXPECTED: dict[str, dict[str, object]] = {
    "lc-topology-replay": {
        "counts": {
            "amplitude_roots": 24,
            "attachments": 126,
            "closures": 24,
            "color_contraction_entries": 0,
            "couplings": 2,
            "current_slots": 69,
            "finalizations": 58,
            "invocations": 126,
            "momentum_slots": 11,
            "reduction_groups": 24,
            "referenced_exact_kernels": 6,
            "selector_domains": 59,
            "selector_memberships": 144,
            "stages": 3,
            "value_slots": 69,
        },
        "digests": {
            "exact": "31b66d618e241e7556bf64d2e7666e3c0093638130f6d3eabdfcb717aeab7fb0",
            "layout": "d7315911d47d8765b6927f60daceaeaf6253b10467eeb6750d972040494ed232",
            "reductions": "56631bb2966cc4e1c93acc745458f76e18c7f9dde39d3eaf2687a39456933b52",
            "resolved": "4347816ea9472a7c77bd4638259933ac67a562a9fa7478795233da6dc43cab7b",
            "selectors": "9dcb69de3d4a90f321803eeab3c9adbdec2daa2bd47ba3389b682e83aafd78f3",
            "tables": "24b944da5afd681ba4e91d2b0120570ef3ea13d25f60861a0f356cbc70e971cc",
        },
        "semantic_sha256": "1b4f917d2431921443359986b385b74889dc7e693e4d64d0a88a44f932d1827d",
    },
    "lc-all-flow-union": {
        "counts": {
            "amplitude_roots": 48,
            "attachments": 242,
            "closures": 48,
            "color_contraction_entries": 0,
            "couplings": 2,
            "current_slots": 117,
            "finalizations": 106,
            "invocations": 210,
            "momentum_slots": 13,
            "reduction_groups": 48,
            "referenced_exact_kernels": 6,
            "selector_domains": 139,
            "selector_memberships": 384,
            "stages": 3,
            "value_slots": 117,
        },
        "digests": {
            "exact": "3f00e8d0ab95958f7b396544f861304affd5dd6c9606cb8790d597fe6fe1fbf9",
            "layout": "a853478261d8e35a684e32b7d62a65bf110b30b7947038edaf5a20f4d3a5ba31",
            "reductions": "3702ceb56c38b667fba9186ce8fe5d0517879f999091b5c934de28d8f3d5d40e",
            "resolved": "f038d7b438a5b59baabc439fb712dbe3fd455f433f20e9da7490e574acc22927",
            "selectors": "3ef749f39eac6664cdb93980aad31cb5cd9af43827c3c79fd24b7dc090631a52",
            "tables": "ebef820462620188bfb615af18d5e673ceab75985a382744df70c1a17504eec3",
        },
        "semantic_sha256": "745c4c88ee5ea0282ba31256de84fa339292c4cd120fd4b60a9a4c466c7cb035",
    },
    "nlc-contracted": {
        "counts": {
            "amplitude_roots": 18,
            "attachments": 120,
            "closures": 18,
            "color_contraction_entries": 63,
            "couplings": 1,
            "current_slots": 73,
            "finalizations": 66,
            "invocations": 84,
            "momentum_slots": 13,
            "reduction_groups": 18,
            "referenced_exact_kernels": 4,
            "selector_domains": 34,
            "selector_memberships": 54,
            "stages": 2,
            "value_slots": 73,
        },
        "digests": {
            "exact": "0596dde559b322fd5341142d288d5dbc8e33fb9610d31fe7d9f9be375fd3d6f0",
            "layout": "2500a3dc283495b7c86405a25ce5de90841d026f27a2a38ee0644b98739696df",
            "reductions": "513d22dc4c4cb819e3e2b15ee87fbdb5e50cefaa33c10a7b1b9457a95cc46c16",
            "resolved": "8992b3b811e142e32f5ea1b1bdf1101bb91ccf8b353db7f6be48626473532ffb",
            "selectors": "a24b39783b6fdacd1579979cd7b9978351f8cb402d983828d179d0bbda836a19",
            "tables": "fbed8d4586abbc1ec8cd6656c310611e8337992574d66221b267fb616e3aeb85",
        },
        "semantic_sha256": "09e7c8ed5d5dce6c5cb16df33a1a95bf204c70718f4ec575c2478ed3b47fb049",
    },
    "full-contracted": {
        "counts": {
            "amplitude_roots": 18,
            "attachments": 120,
            "closures": 18,
            "color_contraction_entries": 63,
            "couplings": 1,
            "current_slots": 73,
            "finalizations": 66,
            "invocations": 84,
            "momentum_slots": 13,
            "reduction_groups": 18,
            "referenced_exact_kernels": 4,
            "selector_domains": 34,
            "selector_memberships": 54,
            "stages": 2,
            "value_slots": 73,
        },
        "digests": {
            "exact": "0596dde559b322fd5341142d288d5dbc8e33fb9610d31fe7d9f9be375fd3d6f0",
            "layout": "b68828f7c57130df4fd916691cb30df937a4f8f6009a69b133bbe58b923aa8dd",
            "reductions": "aee94cd71574e24ee90fef2f8070e3fded7b4f61ec02789d1d6cd60602e3350a",
            "resolved": "c5608596ba42ddfdbad9e0a5c27db395519b32704386767f4b25100b4375a30c",
            "selectors": "9ca1b90505098cc6f19d87eba29baa552da47173ff2658f359e27de838bc1ca2",
            "tables": "7b7c4f181b6525e9356cdb5648bc80085e890018c778d3a55262fd00d0b4faf3",
        },
        "semantic_sha256": "8d17b7de733096b91c859909c5ed08410249357915a6593b7ad13b98b3ba84e8",
    },
}


def test_plan_v2_abi_and_capability_are_the_explicit_legacy_contract() -> None:
    assert EAGER_PLAN_ABI == _LEGACY_PLAN_ABI
    assert EAGER_RUNTIME_CAPABILITY == _LEGACY_RUNTIME_CAPABILITY


def test_plan_v2_semantic_goldens(golden_cases: tuple[_GoldenCase, ...]) -> None:
    actual = {case.name: _snapshot(case) for case in golden_cases}
    assert actual == _EXPECTED
