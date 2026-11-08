# SPDX-License-Identifier: 0BSD
"""TIR-only artifacts: loop families beyond NinjaRS's rank <= N+2 (m-lit
LIT-EFT; Q24 = (b) as amended by the USER 2026-11-06 ~14:55: an open feature
whose results carry NO correctness claim, since no correct R2 exists for
rank > N+2 numerators in the models we have; a run-only check).

* the rank admission takes the backend: NinjaRS's rank classes and TIRust's
  capability query (``family_rank_refusal``: the plan loader's admission,
  ``bounds::admission_refusal``); a family beyond N+2 makes the artifact
  TIR-only (``tir_only``), whose physics declaration records ``reductions =
  ["tirust"]`` and any other's nothing (its bytes unchanged);
* on a TIR-only artifact ``auto`` resolves to TIRust for both process types
  at every carrier with its reason, an explicit NinjaRS is refused typed and
  no NinjaRS fallback exists (``reduction.resolve``'s admitted set);
* nightly, on HC_NLO_X0 (``tools/loop_v2_eft_execution.py``'s rows):
  g g > x0 x0 x0 (three x0 G G insertions; its pentagons at rank N+3) is
  generated TIR-only, evaluates through TIRust in binary64, double-double and
  arbitrary precision and on every rescue level, refuses NinjaRS typed (call,
  rescue level, cross-backend check), is served by the SDKs' native runtime
  on TIRust bitwise, and its families NinjaRS reduces agree lane by lane;
  g g > x0 x0 (two insertions, rank N+2 at most) is no TIR-only artifact and
  the two backends agree on it within L6.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "loop_v2_eft_execution", ROOT / "tools" / "loop_v2_eft_execution.py"
)
assert _spec is not None and _spec.loader is not None
eft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eft)


def _metadata(**extra):
    from pyamplicol.generation.loop_artifact import LoopV2ExecutionMetadata
    from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
    from pyamplicol.models.loop_declaration import LoopSchemeDeclaration

    scheme = LoopSchemeDeclaration(
        "HV", "physical", "none", "on-shell", "real", "model", "test"
    )
    born, loop = (("QCD", 2),), (("QCD", 4),)
    return LoopV2ExecutionMetadata(
        scheme=scheme,
        born_orders=(born,),
        sectors=(VirtualOrderSector((("QCD", 6),), (BornLoopPair(born, loop),)),),
        families=1,
        buckets=((5, 8, 1),),
        color_accuracy="full",
        color_degrees=None,
        color_contraction="auto",
        **extra,
    )


@pytest.mark.loop_fast
def test_reductions_are_declared_for_tir_only_processes_only():
    from pyamplicol.generation.loop_artifact import _metadata_record

    record = _metadata_record(_metadata())
    assert "reductions" not in record
    assert _metadata_record(_metadata(reductions=("tirust",)))["reductions"] == [
        "tirust"
    ]
    with pytest.raises(ValueError, match="TIR-only"):
        _metadata(reductions=("ninjars",))


@pytest.mark.loop_fast
def test_the_rank_admission_takes_the_backend(loop_native):
    from pyamplicol.loops.v2.families import family_rank_refusal, tir_only

    assert not tir_only([(4, 6, 3), (5, 7, 1), (1, 0, 2)])
    assert tir_only([(4, 6, 3), (5, 8, 1)])
    assert family_rank_refusal(4, 6) is None
    if loop_native.tirust_available():
        # TIRust has no rank <= N+2 limit: its capability query decides.
        assert family_rank_refusal(5, 8) is None
        assert family_rank_refusal(2, 9) is None
        refused = family_rank_refusal(1, 1_000_000)
        assert "NinjaRS" in refused and "TIRust cannot reduce" in refused
    else:
        assert "no TIRust" in family_rank_refusal(5, 8)


@pytest.mark.loop_fast
def test_auto_takes_tirust_on_a_tir_only_artifact_at_every_carrier():
    from pyamplicol.loops.v2.reduction import (
        BACKENDS,
        CARRIERS,
        ReductionUnavailable,
        Route,
        resolve,
    )

    class Native:
        @staticmethod
        def reduction_backends():
            return json.dumps({name: list(CARRIERS) for name in BACKENDS})

    route = Route(
        "a route whose auto is NinjaRS", frozenset(BACKENDS), frozenset({"ninjars"})
    )
    only = frozenset({"tirust"})
    for kind in ("virtual", "loop-induced"):
        for carrier in CARRIERS:
            found = resolve(
                "auto",
                contribution=kind,
                carrier=carrier,
                native=Native,
                route=route,
                admitted=only,
            )
            assert (found.requested, found.effective) == ("auto", "tirust")
            assert found.reason == eft.ONLY
            with pytest.raises(ReductionUnavailable, match="admits"):
                resolve(
                    "ninjars",
                    contribution=kind,
                    carrier=carrier,
                    native=Native,
                    route=route,
                    admitted=only,
                )
        # Every backend admitted: the route's auto set (NinjaRS) as before.
        found = resolve(
            "auto", contribution=kind, carrier="double", native=Native, route=route
        )
        assert found.effective == "ninjars"


# ----------------------------------------------------------------- nightly


@pytest.fixture(scope="module")
def eft_artifacts(hc_nlo_x0_ufo, loop_native, tmp_path_factory):
    if not loop_native.tirust_available():
        pytest.fail("the TIR-only rows need a native extension with TIRust")
    work = tmp_path_factory.mktemp("eft")
    return {key: eft.generate(work, key, hc_nlo_x0_ufo) for key in eft.ROWS}


@pytest.mark.loop_nightly
def test_three_x0_insertions_make_a_tir_only_artifact_served_by_tirust(eft_artifacts):
    artifact = eft_artifacts["x0x0x0"]
    loaded = eft.runtime(artifact)
    assert max(r - n for n, r, _count in loaded.inspect()["buckets"]) == 3
    assert loaded.inspect()["reductions"] == ["tirust"]
    points, scales = eft.points("x0x0x0")
    found = eft.tir_only_checks(artifact, loaded, points, scales)
    assert found["passed"], json.dumps(found, indent=1, default=str)[:4000]


@pytest.mark.loop_nightly
def test_two_x0_insertions_stay_within_n_plus_two_and_the_backends_agree(eft_artifacts):
    loaded = eft.runtime(eft_artifacts["x0x0"])
    described = loaded.inspect()
    assert max(r - n for n, r, _count in described["buckets"]) == 2
    assert "reductions" not in described
    points, scales = eft.points("x0x0")
    found = eft.public_agreement(loaded, points, scales)
    assert found["passed"], json.dumps(found, indent=1)[:4000]
