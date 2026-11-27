# SPDX-License-Identifier: 0BSD
"""Every shipped loop-model restriction card loads through the production path.

Separate from ``test_loop_model_restrictions`` because the exact zero
simplification of the electroweak models takes minutes (``loop_physics``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyamplicol.loops.model_restrictions import prepare_loop_restrictions
from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

pytestmark = pytest.mark.loop_physics


def _conflicting_addresses(text):
    """Addresses a card assigns twice with different values (MadGraph rejects)."""
    seen, conflicts, block = {}, set(), None
    for line in text.splitlines():
        tokens = line.split("#", 1)[0].split()
        if not tokens:
            continue
        if tokens[0].lower() in {"block", "decay"}:
            block = tokens[1].upper() if tokens[0].lower() == "block" else "DECAY"
            if tokens[0].lower() == "block":
                continue
            tokens = tokens[1:]
        address = (block, tuple(tokens[:-1]))
        if seen.setdefault(address, float(tokens[-1])) != float(tokens[-1]):
            conflicts.add(address)
    return conflicts


@pytest.mark.parametrize(
    "model", ["loop_sm_ufo", "loop_qcd_qed_sm_ufo", "loop_gmu_ufo"]
)
def test_every_shipped_loop_model_restriction_card_loads(request, model):
    """loop_sm (incl. -ckm), loop_qcd_qed_sm and loop_qcd_qed_sm_Gmu cards."""
    path = request.getfixturevalue(model)
    cards = sorted(Path(path).glob("restrict_*.dat"))
    assert cards
    base = load_raw_loop_model(path, restriction="full").to_dict()
    addresses = {
        (str(fields["lhablock"]).upper(), tuple(fields["lhacode"]))
        for fields in (r["fields"] for r in base["collections"]["parameters"])
        if fields.get("nature") == "external" and fields.get("lhablock")
    }
    loaded = 0
    for card in cards:
        name = card.stem.removeprefix("restrict_")
        text = card.read_text(encoding="utf-8")
        payload = dict(base, metadata=dict(base["metadata"]))
        payload["metadata"]["restriction"] = {
            "applied": False,
            "name": name,
            "card": text,
        }
        raw = RawLoopModel.from_dict(payload)
        if _conflicting_addresses(text):
            # MadGraph 3.7.0 rejects such a card too (loop_qcd_qed_sm_Gmu's
            # restrict_parallel_test.dat assigns SMINPUTS 1 twice).
            with pytest.raises(Exception, match="already define"):
                prepare_loop_restrictions(raw)
            continue
        # The production path, including exact zero simplification (its scalar
        # lowering takes about 20 s per card for the EW models).
        prepared = prepare_loop_restrictions(raw)
        restriction = prepared.derived.metadata["restriction"]
        assert restriction["applied"] and restriction["name"] == name
        assert restriction["simplify"]
        assert set(restriction["fixed_parameters"]) == set(prepared.fixed_parameters)
        # Only entries without a model parameter are ignored, e.g. DECAY 15 in
        # models that declare no tau width.
        for block, code, _value in restriction["ignored_entries"]:
            assert (block, tuple(code)) not in addresses
        loaded += 1
    assert loaded
