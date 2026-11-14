# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.loops.orders import (
    BornLoopPair,
    LoopContribution,
    LoopEligibility,
    compile_order_plan,
)

pytestmark = pytest.mark.loop_fast

B20 = (("QCD", 2),)
B02 = (("QED", 2),)
L40 = (("QCD", 4),)
L22 = (("QCD", 2), ("QED", 2))
L04 = (("QED", 4),)
QCD = LoopEligibility((frozenset({"QCD"}),), frozenset({"QCD"}))
QED = LoopEligibility((frozenset({"QED"}),), frozenset({"QED"}))
EITHER = LoopEligibility((frozenset({"QCD", "QED"}),), frozenset({"QCD", "QED"}))


def test_mixed_orders_preserve_both_cross_interferences_and_union_graphs() -> None:
    # The same graph may be returned by both correction selections. Distinct
    # physical graph classes also share an absolute order and must both survive.
    contributions = (
        LoopContribution("qcd40", L40, QCD),
        LoopContribution("shared22", L22, EITHER),
        LoopContribution("shared22", L22, EITHER),
        LoopContribution("other22", L22, QED),
        LoopContribution("qed04", L04, QED),
    )
    plan = compile_order_plan(
        selected_corrections=("QED", "QCD", "QCD"),
        perturbative_orders={"QCD": 1, "QED": 1},
        born_orders=(B20, B02),
        contributions=contributions,
        allowed_pairs=(
            BornLoopPair(born, loop) for born in (B20, B02) for loop in (L40, L22, L04)
        ),
    )
    sector = plan.sector((("QED", 2), ("QCD", 4)))
    assert set(sector.pairs) == {BornLoopPair(B20, L22), BornLoopPair(B02, L40)}
    assert len(plan.contributions) == 4
    shared = next(c for c in plan.contributions if c.id == "shared22")
    assert shared.eligibility == EITHER
    assert len(plan.sectors) == 4
    with pytest.raises(ValueError, match="absent from the catalogue"):
        plan.sector((("QCD", 8),))


def test_equal_absolute_order_does_not_imply_correction_eligibility() -> None:
    plan = compile_order_plan(
        selected_corrections=("QCD",),
        perturbative_orders={"QCD": 1, "QED": 1},
        born_orders=(B02,),
        contributions=(
            LoopContribution("qcd", L22, QCD),
            LoopContribution("qed", L22, QED),
            LoopContribution("evanescent-qcd", L22, QCD, "evanescent"),
        ),
        allowed_pairs=(BornLoopPair(B02, L22),),
    )
    assert {c.id for c in plan.contributions} == {"qcd", "evanescent-qcd"}


def test_joint_selection_includes_mixed_box_excluded_by_singletons() -> None:
    # Quarks can carry either correction, a gluon only QCD, a photon only QED.
    mixed = LoopEligibility(
        (frozenset({"QCD", "QED"}), frozenset({"QCD"}), frozenset({"QED"})),
        frozenset({"QCD", "QED"}),
    )
    assert not mixed.admits(frozenset({"QCD"}))
    assert not mixed.admits(frozenset({"QED"}))
    assert mixed.admits(frozenset({"QCD", "QED"}))
    plan = compile_order_plan(
        selected_corrections=("QCD", "QED"),
        perturbative_orders={"QCD": 1, "QED": 1},
        born_orders=(B20,),
        contributions=(LoopContribution("mixed-box", L22, mixed),),
        allowed_pairs=(BornLoopPair(B20, L22),),
    )
    assert plan.contributions[0].id == "mixed-box"


@pytest.mark.parametrize(
    ("names", "capabilities", "message"),
    [
        ((), {"QCD": 1}, "at least one"),
        (("XYZ",), {"QCD": 1}, "unknown"),
        (("QCD",), {"QCD": 0}, "does not support"),
    ],
)
def test_invalid_correction_selection(names, capabilities, message) -> None:
    with pytest.raises(ValueError, match=message):
        compile_order_plan(
            selected_corrections=names,
            perturbative_orders=capabilities,
            born_orders=(B20,),
            contributions=(),
            allowed_pairs=(),
        )


def test_structural_born_catalogue_distinguishes_loop_induced_processes() -> None:
    with pytest.raises(ValueError, match="loop-induced"):
        compile_order_plan(
            selected_corrections=("QCD",),
            perturbative_orders={"QCD": 1},
            born_orders=(),
            contributions=(LoopContribution("g", L40, QCD),),
            allowed_pairs=(),
        )


def test_unselected_pair_is_an_error_and_duplicate_identity_cannot_hide_a_graph() -> (
    None
):
    args = dict(
        selected_corrections=("QCD",),
        perturbative_orders={"QCD": 1},
        born_orders=(B20,),
        contributions=(LoopContribution("g", L40, QCD),),
    )
    with pytest.raises(ValueError, match="unselected order"):
        compile_order_plan(**args, allowed_pairs=(BornLoopPair(B20, L22),))
    args["contributions"] += (LoopContribution("g", L22, QCD),)
    with pytest.raises(ValueError, match="conflicting"):
        compile_order_plan(**args, allowed_pairs=(BornLoopPair(B20, L40),))
