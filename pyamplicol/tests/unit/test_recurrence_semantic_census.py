# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
from dataclasses import astuple, replace

import pytest

from pyamplicol.runtime.recurrence_exact._plan_v2 import (
    DIRECT_NONE_U32,
    RECURRENCE_EXACT_SECTIONS_ABI,
    RECURRENCE_RUNTIME_LAYOUT_V2_ABI,
    _parse_exact_sections,
)
from tools.developer import recurrence_semantic_census as census


def _sections():
    raw = {
        "abi": RECURRENCE_EXACT_SECTIONS_ABI,
        "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_V2_ABI,
        "process_id": "synthetic",
        "strategy": "topology-replay",
        "semantic_digest": "1" * 64,
        "runtime_layout_digest": "2" * 64,
        "counts": [3, 1, 0, 2],
        "currents": [
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, DIRECT_NONE_U32],
            [1, 0, 1, 1, 1, 1, 0, 0, 0, 3, 1, DIRECT_NONE_U32],
            [2, 1, 2, 2, 1, 2, 1, 0, 1, 3, DIRECT_NONE_U32, 0],
        ],
        "sources": [
            [0, 0, 0, 0, -1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
        ],
        "contributions": [[0, 1, 0, 1, 2, 0, 0, 0]],
        "finalizations": [[2, 1, 2, 0, 0, 0]],
        "closures": [[2, DIRECT_NONE_U32, 2, DIRECT_NONE_U32, 0, 0, 0, 1, 0, 0]],
        "row_groups": [
            [0, 0, 0, 0, 0, 2],
            [1, 1, 1, 1, 0, 1],
            [1, 2, 2, 2, 0, 1],
            [2, 3, 3, 3, 0, 1],
        ],
        "momentum_forms": [[0, 1], [1, 1], [2, 2]],
        "momentum_terms": [[0, 1], [1, 1], [0, 1], [1, 1]],
        "replay_targets": [[0, 0, 0, 2, 0, 1, 0, 1, 0]],
        "source_permutations": [0, 1],
        "replay_momentum_signs": [1, 1],
        "replay_helicity_map": [0],
        "amplitude_destinations": [[0, 0, 0, 0, 1, 0]],
        "resolved_helicities": [[0, 0, 0, 0, 2, 0, 2, 0]],
        "source_state_assignments": [[0, 0], [1, 1]],
        "source_dispatch_variants": [],
        "source_embeddings": [],
        "source_projections": [],
        "resolved_source_selections": [],
        "public_helicities": [-1, 1],
        "exact_factors": [["1", "1", "0", "1"]],
        "public_flow_ids": [0],
        "executors": [
            [0, "source", "initialize", [], 1, 1, None, "source"],
            [1, "contribution", "add", [1, 1], 1, 2, 10, None],
            [2, "finalization", "finalize-in-place", [1], 1, 1, 11, None],
            [3, "closure", "closure-add", [1], 1, 1, 12, None],
        ],
    }
    return _parse_exact_sections(raw, "synthetic")


def _binding(
    executor_id: int,
    role: str,
    semantic_id: str,
    *,
    records: tuple[dict[str, object], ...] = (),
) -> census.ExecutorSemanticBinding:
    return census.ExecutorSemanticBinding(
        executor_id=executor_id,
        role=role,
        semantic_digest=f"{executor_id + 3:x}" * 64,
        semantic_template_ids=(semantic_id,),
        direct_template={
            "direct_executor_id": executor_id,
            "role": role,
            "semantic_digest": f"{executor_id + 3:x}" * 64,
            "semantic_template_ids": [semantic_id],
        },
        referenced_semantic_records=records
        or (
            {
                "record_kind": role,
                "semantic_digest": f"{executor_id + 7:x}" * 64,
                "template_id": semantic_id,
            },
        ),
    )


def _inputs():
    sections = _sections()
    transition = {
        "binding_coupling": {
            "real_numerator": "1",
            "real_denominator": "1",
            "imag_numerator": "0",
            "imag_denominator": "1",
        },
        "color_contraction_template_id": "color:test",
        "quantum_flow_template_id": "quantum:test",
        "record_kind": "transition",
        "semantic_digest": "8" * 64,
        "template_id": "transition:test",
    }
    color = {
        "record_kind": "color-contraction",
        "semantic_digest": "9" * 64,
        "template_id": "color:test",
        "transition_witnesses": [{"proof_digest": "a" * 64}],
    }
    quantum = {
        "record_kind": "quantum-flow",
        "semantic_digest": "b" * 64,
        "template_id": "quantum:test",
    }
    bindings = {
        0: _binding(0, "source", "source:test"),
        1: _binding(
            1,
            "contribution",
            "transition:test",
            records=(transition, color, quantum),
        ),
        2: _binding(2, "finalization", "propagator:test"),
        3: _binding(3, "closure", "closure:test"),
    }
    metadata = {
        "digests": {
            "process_semantic_digest": sections.semantic_digest,
            "runtime_layout_digest": sections.runtime_layout_digest,
            "schedule_digest": "c" * 64,
            "native_schedule_semantic_digest": "d" * 64,
        },
        "normalization_and_parameters": {
            "normalization": {"average_factor": 36, "color_factor": 9},
            "runtime_parameters": [{"name": "alpha_s", "default": 0.118}],
        },
        "selectors": {
            "color_accuracy": "lc",
            "public_color_flows": [{"public_id": "flow:test"}],
        },
        "sources": {
            "external_pdg_order": [1, -1],
            "source_templates": [{"source_template_id": 0}],
        },
    }
    states = (
        {"semantic_digest": "e" * 64, "template_id": "state:left"},
        {"semantic_digest": "f" * 64, "template_id": "state:right"},
        {"semantic_digest": "0" * 64, "template_id": "state:result"},
    )
    return sections, metadata, bindings, states


def _build(
    sections=None,
    metadata=None,
    bindings=None,
    states=None,
):
    base_sections, base_metadata, base_bindings, base_states = _inputs()
    return census.build_semantic_census(
        sections=sections or base_sections,
        process_metadata=metadata or base_metadata,
        executor_bindings=bindings or base_bindings,
        state_templates=states or base_states,
    )


def _changed_domains(left, right) -> set[str]:
    report = census.compare_census_sets({"synthetic": left}, {"synthetic": right})
    assert report["passes"] is False
    return {
        difference["domain"]
        for difference in report["differences"]
        if difference["kind"] == "semantic-domain"
    }


def test_identical_semantic_censuses_pass_with_complete_domain_set() -> None:
    built = _build()

    report = census.compare_census_sets(
        {"synthetic": built}, {"synthetic": copy.deepcopy(built)}
    )

    assert report["passes"] is True
    assert set(built["domains"]) == census._REQUIRED_DOMAINS
    assert report["difference_count"] == 0


def test_current_identity_state_support_color_and_helicity_projection_fail_closed() -> (
    None
):
    sections, metadata, bindings, states = _inputs()
    currents = list(sections.currents)
    currents[2] = replace(
        currents[2],
        state_template_id=1,
        selector_domain_id=7,
        momentum_form_id=0,
    )
    changed = replace(sections, currents=tuple(currents))

    domains = _changed_domains(
        _build(sections, metadata, bindings, states),
        _build(changed, metadata, bindings, states),
    )

    assert {"currents", "runtime_layout"} <= domains


def test_source_semantics_fail_closed() -> None:
    sections, metadata, bindings, states = _inputs()
    sources = list(sections.sources)
    sources[0] = replace(sources[0], spin_state_class=1)

    domains = _changed_domains(
        _build(sections, metadata, bindings, states),
        _build(replace(sections, sources=tuple(sources)), metadata, bindings, states),
    )

    assert {"sources", "runtime_layout"} <= domains


def test_union_source_rows_may_be_owned_without_a_direct_executor() -> None:
    sections, metadata, bindings, states = _inputs()
    groups = list(sections.row_groups)
    groups[0] = replace(groups[0], executor_id=DIRECT_NONE_U32)
    union_sections = replace(
        sections,
        strategy="all-flow-union",
        row_groups=tuple(groups),
    )

    built = _build(union_sections, metadata, bindings, states)
    report = census.compare_census_sets(
        {"synthetic": built},
        {"synthetic": copy.deepcopy(built)},
    )

    assert report["passes"] is True
    assert built["domains"]["sources"]["record_count"] == 3


def test_interaction_endpoints_sign_and_contribution_multiset_fail_closed() -> None:
    sections, metadata, bindings, states = _inputs()
    factors = (
        *sections.exact_factors,
        replace(sections.exact_factors[0], real_numerator=-1),
    )
    contributions = list(sections.contributions)
    contributions[0] = replace(
        contributions[0],
        parent0_base=1,
        exact_factor_id=1,
    )
    changed = replace(
        sections,
        exact_factors=factors,
        contributions=tuple(contributions),
    )

    domains = _changed_domains(
        _build(sections, metadata, bindings, states),
        _build(changed, metadata, bindings, states),
    )

    assert {"contribution_multisets", "runtime_layout"} <= domains


@pytest.mark.parametrize(
    ("record_index", "mutation"),
    (
        (0, ("template_id", "transition:changed")),
        (0, ("binding_coupling", {"real_numerator": "2"})),
        (1, ("transition_witnesses", [{"proof_digest": "0" * 64}])),
    ),
    ids=("transition-identity", "coupling", "color-witness"),
)
def test_transition_coupling_and_witness_catalogs_fail_closed(
    record_index: int,
    mutation: tuple[str, object],
) -> None:
    sections, metadata, bindings, states = _inputs()
    changed_bindings = dict(bindings)
    original = bindings[1]
    records = [
        copy.deepcopy(dict(record)) for record in original.referenced_semantic_records
    ]
    key, value = mutation
    if key == "binding_coupling":
        coupling = dict(records[record_index][key])
        coupling.update(value)
        records[record_index][key] = coupling
    else:
        records[record_index][key] = value
    changed_bindings[1] = replace(original, referenced_semantic_records=tuple(records))

    domains = _changed_domains(
        _build(sections, metadata, bindings, states),
        _build(sections, metadata, changed_bindings, states),
    )

    assert "semantic_catalog_bindings" in domains


def test_closure_endpoints_and_selector_axes_fail_closed() -> None:
    sections, metadata, bindings, states = _inputs()
    closures = list(sections.closures)
    closures[0] = replace(closures[0], amplitude_destination_id=3)
    changed_closure = replace(sections, closures=tuple(closures))
    changed_selector = replace(sections, public_flow_ids=(4,))

    assert "closures" in _changed_domains(
        _build(sections, metadata, bindings, states),
        _build(changed_closure, metadata, bindings, states),
    )
    assert "selectors" in _changed_domains(
        _build(sections, metadata, bindings, states),
        _build(changed_selector, metadata, bindings, states),
    )


def test_normalization_digests_and_layout_each_fail_closed() -> None:
    sections, metadata, bindings, states = _inputs()
    changed_metadata = copy.deepcopy(metadata)
    changed_metadata["normalization_and_parameters"]["normalization"][
        "average_factor"
    ] = 18
    assert "normalization_and_parameters" in _changed_domains(
        _build(sections, metadata, bindings, states),
        _build(sections, changed_metadata, bindings, states),
    )

    changed_digest_sections = replace(sections, semantic_digest="a" * 64)
    changed_digest_metadata = copy.deepcopy(metadata)
    changed_digest_metadata["digests"]["process_semantic_digest"] = "a" * 64
    assert "digests" in _changed_domains(
        _build(sections, metadata, bindings, states),
        _build(
            changed_digest_sections,
            changed_digest_metadata,
            bindings,
            states,
        ),
    )

    currents = list(sections.currents)
    currents[2] = replace(currents[2], component_base=8)
    assert "runtime_layout" in _changed_domains(
        _build(sections, metadata, bindings, states),
        _build(
            replace(sections, currents=tuple(currents)),
            metadata,
            bindings,
            states,
        ),
    )


def test_authenticated_storage_hashes_do_not_define_semantic_equivalence() -> None:
    sections, metadata, bindings, states = _inputs()
    baseline_metadata = copy.deepcopy(metadata)
    candidate_metadata = copy.deepcopy(metadata)
    for index, field in enumerate(
        sorted(census._STORAGE_AUTHENTICATION_DIGEST_FIELDS),
        start=1,
    ):
        baseline_metadata["digests"][field] = f"{index:x}" * 64
        candidate_metadata["digests"][field] = f"{index + 4:x}" * 64

    baseline = _build(sections, baseline_metadata, bindings, states)
    candidate = _build(sections, candidate_metadata, bindings, states)

    assert baseline == candidate
    assert (
        census.compare_census_sets(
            {"synthetic": baseline},
            {"synthetic": candidate},
        )["passes"]
        is True
    )


def test_missing_domain_and_inconsistent_transitive_digest_are_rejected() -> None:
    built = _build()
    incomplete = copy.deepcopy(built)
    incomplete["domains"].pop("closures")
    with pytest.raises(census.SemanticCensusError, match="incomplete census domains"):
        census.compare_census_sets({"synthetic": built}, {"synthetic": incomplete})

    sections, metadata, bindings, states = _inputs()
    metadata["digests"]["process_semantic_digest"] = "a" * 64
    with pytest.raises(
        census.SemanticCensusError, match="does not match exact sections"
    ):
        _build(sections, metadata, bindings, states)


def test_exact_section_fixture_uses_authoritative_parser() -> None:
    sections = _sections()

    assert sections.currents[2].semantic_id == 2
    assert astuple(sections.contributions[0]) == (0, 1, 0, 1, 2, 0, 0, 0)
