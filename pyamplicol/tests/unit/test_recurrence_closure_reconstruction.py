# SPDX-License-Identifier: 0BSD
"""Adversarial contracts for exact recurrence closure reconstruction."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

import pyamplicol.models.recurrence_template as recurrence_template
from pyamplicol import _rusticol
from pyamplicol.models.recurrence_template import (
    ExactComplexRationalV1,
    LCColorTransitionWitnessV1,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _factor(real: int, imag: int = 0) -> ExactComplexRationalV1:
    return ExactComplexRationalV1(real, 1, imag, 1)


def _reconstruction_rule_type() -> type[object]:
    """Return the missing model-generic rule type once production provides it.

    The rule must be catalog-owned rather than keyed by an SM/process name. Its
    semantic payload must include a rule kind, topology predicate, ordered source
    and parent mapping, exact complex factor, positive multiplicity, and proof
    digest. Rust then materializes matching process-specific closure terms.
    """

    return vars(recurrence_template)["ClosureReconstructionRuleV1"]


def _assert_rule_contract(rule_type: type[object]) -> None:
    names = {field.name for field in dataclasses.fields(rule_type)}
    assert {
        "rule_kind",
        "topology_contract",
        "source_slot_permutation",
        "parent_permutation",
        "exact_factor",
        "multiplicity",
        "proof_digest",
    } <= names


def _materialize_terms(
    topology: dict[str, object],
    rules: tuple[object, ...],
) -> tuple[dict[str, object], ...]:
    """Invoke the missing bounded Rust closure-reconstruction inspector.

    Production may expose this only as a private test/inspection binding. It must
    run the same matching and exact-term materialization used by the builder and
    return term IDs, rule kinds, permutations, factors, multiplicities, and proof
    digests without constructing a full process DAG.
    """

    inspect = vars(_rusticol)["_inspect_recurrence_closure_reconstruction_v1"]
    return tuple(inspect(topology, rules))


def test_reflected_pure_gluon_witness_preserves_exact_phase() -> None:
    """The existing primitive witness retains its exact reflection phase."""

    witness = LCColorTransitionWitnessV1(
        input_shape_kinds=("adjoint-segment", "adjoint-segment"),
        input_permutation=(1, 0),
        reverse_parent_mask=0b11,
        component_operation="close",
        result_component_kind=None,
        result_component_role="none",
        result_shape_kind=None,
        exact_factor=_factor(-1),
        proof_digest=_sha256("pure-gluon-reflection"),
    )

    payload = witness.to_dict()
    assert payload["input_permutation"] == [1, 0]
    assert payload["reverse_parent_mask"] == 0b11
    assert payload["exact_factor"] == _factor(-1).to_dict()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the recurrence catalog has no model-generic closure reconstruction rule "
        "for distinct three-open-line partner terms"
    ),
)
def test_three_line_partner_closures_keep_sign_and_multiplicity() -> None:
    """Partner closures sharing one public flow must remain separate terms."""

    rule_type = _reconstruction_rule_type()
    _assert_rule_contract(rule_type)
    direct = rule_type(
        rule_kind="three-open-line-direct",
        topology_contract="three-ordered-open-lines-v1",
        source_slot_permutation=(0, 1, 2, 3, 4, 5),
        parent_permutation=(0, 1),
        exact_factor=_factor(1),
        multiplicity=1,
        proof_digest=_sha256("three-line-direct"),
    )
    partner = rule_type(
        rule_kind="three-open-line-partner",
        topology_contract="three-ordered-open-lines-v1",
        source_slot_permutation=(0, 3, 2, 1, 4, 5),
        parent_permutation=(1, 0),
        exact_factor=_factor(1),
        multiplicity=1,
        proof_digest=_sha256("three-line-partner"),
    )
    terms = _materialize_terms(
        {
            "kind": "three-open-lines",
            "ordered_open_strings": ((0, 1), (2, 3), (4, 5)),
        },
        (direct, partner),
    )
    assert len(terms) == 2
    assert len({term["term_id"] for term in terms}) == 2
    assert tuple(term["rule_kind"] for term in terms) == (
        "three-open-line-direct",
        "three-open-line-partner",
    )
    assert tuple(term["exact_factor"] for term in terms) == (
        _factor(1).to_dict(),
        _factor(1).to_dict(),
    )
    assert tuple(term["multiplicity"] for term in terms) == (1, 1)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the recurrence catalog cannot materialize a reflected pure-gluon closure "
        "as a distinct signed closure term"
    ),
)
def test_pure_gluon_reflected_closure_is_a_distinct_exact_term() -> None:
    """Reflection metadata on a color witness is insufficient by itself."""

    rule_type = _reconstruction_rule_type()
    _assert_rule_contract(rule_type)
    reflected = rule_type(
        rule_kind="pure-gluon-reflection",
        topology_contract="single-trace-reflection-v1",
        source_slot_permutation=(0, 3, 2, 1),
        parent_permutation=(1, 0),
        exact_factor=_factor(-1),
        multiplicity=1,
        proof_digest=_sha256("gluon-reflected-closure"),
    )
    terms = _materialize_terms(
        {
            "kind": "single-trace",
            "trace_word": (0, 1, 2, 3),
            "requires_reflection": True,
        },
        (reflected,),
    )
    assert len(terms) == 1
    assert terms[0]["rule_kind"] == "pure-gluon-reflection"
    assert terms[0]["exact_factor"] == _factor(-1).to_dict()
    assert terms[0]["source_slot_permutation"] == (0, 3, 2, 1)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the recurrence catalog has no exact same-flavour exchange/reconstruction "
        "term contract, so such terms cannot yet be kept distinct"
    ),
)
def test_same_flavour_exchange_terms_are_not_collapsed() -> None:
    """Direct and exchanged terms need distinct proofs even with equal parents."""

    rule_type = _reconstruction_rule_type()
    _assert_rule_contract(rule_type)
    direct = rule_type(
        rule_kind="same-flavour-direct",
        topology_contract="identical-fermion-lines-v1",
        source_slot_permutation=(0, 1, 2, 3),
        parent_permutation=(0, 1),
        exact_factor=_factor(1),
        multiplicity=1,
        proof_digest=_sha256("same-flavour-direct"),
    )
    exchange = rule_type(
        rule_kind="same-flavour-exchange",
        topology_contract="identical-fermion-lines-v1",
        source_slot_permutation=(0, 3, 2, 1),
        parent_permutation=(1, 0),
        exact_factor=_factor(-1),
        multiplicity=1,
        proof_digest=_sha256("same-flavour-exchange"),
    )
    terms = _materialize_terms(
        {
            "kind": "two-open-lines",
            "ordered_open_strings": ((0, 1), (2, 3)),
            "same_flavour_classes": ((0, 2),),
        },
        (direct, exchange),
    )
    assert len(terms) == 2
    assert len({term["proof_digest"] for term in terms}) == 2
    assert tuple(term["exact_factor"] for term in terms) == (
        _factor(1).to_dict(),
        _factor(-1).to_dict(),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "there is no Rust closure-support preflight accepting process topology "
        "and model-generic reconstruction rules"
    ),
)
def test_unsupported_closure_semantics_fail_before_schedule_construction() -> None:
    """Missing reconstruction semantics must reject, never silently degrade.

    Required production contract: the Rust closure-reconstruction inspector and
    builder receive the ordered process color topology plus authenticated generic
    rule catalog. They raise before recurrence state construction when any
    required direct, partner, reflection, or exchange term lacks an exact rule.
    """

    synthetic_topology = {
        "kind": "three-open-lines",
        "ordered_open_strings": ((0, 1), (2, 3), (4, 5)),
        "same_flavour_classes": ((0, 2),),
        "requires_partner_closure": True,
    }
    with pytest.raises(
        ValueError,
        match="unsupported closure reconstruction semantics",
    ):
        _materialize_terms(synthetic_topology, ())
