# SPDX-License-Identifier: 0BSD
"""Explicit SDK selector tables from a compact on-the-fly source seed.

This does not construct recurrence schedules or discover amplitude relations.
The expanded tables are SDK payloads only; the runtime seed remains compact.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..generation.on_the_fly_seed import OnTheFlyProcessSeedProjectionV1
    from ..processes.ir import CanonicalProcessIR


def build_umami_on_the_fly_physics(
    public_metadata: Mapping[str, Any],
    process: CanonicalProcessIR,
    seed: OnTheFlyProcessSeedProjectionV1,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    """Enumerate available axes, conservatively without additional proofs."""
    from ..color import build_color_plan

    sources = {source.public_label: source for source in seed.external_sources}
    domains = [
        tuple(
            sorted(state.public_helicity for state in sources[leg.label].source_states)
        )
        for leg in process.legs
    ]
    helicities = []
    for index, values in enumerate(product(*domains)):
        identifier = "h:" + ",".join(f"{value:+d}" for value in values)
        helicities.append(
            {
                "id": identifier,
                "index": index,
                "values": list(values),
                "computed": True,
                "representative_id": identifier,
                "coefficient": 1.0,
                "structural_zero": False,
            }
        )
    sectors: tuple[Mapping[str, Any], ...] = ()
    if process.color_accuracy == "lc":
        plan = build_color_plan(process, color_accuracy="lc")
        sectors = tuple(sector.to_json_dict() for sector in plan.sectors)
        colors = []
        for index, sector in enumerate(plan.sectors):
            word = tuple(sector.word_labels or sector.color_words[0])
            identifier = "flow:" + (
                ",".join(str(label) for label in word) if word else "singlet"
            )
            colors.append(
                {
                    "kind": "lc-flow",
                    "id": identifier,
                    "index": index,
                    "word": list(word),
                    "computed": True,
                    "representative_id": identifier,
                    "coefficient": 1.0,
                }
            )
    else:
        colors = [{"kind": "contracted-color", "id": "color:contracted", "index": 0}]
    return (
        {
            **public_metadata,
            "kind": "pyamplicol-resolved-physics",
            "helicities": helicities,
            "color_components": colors,
            "coverage": {
                "helicities": "complete",
                "color": "complete" if process.color_accuracy == "lc" else "contracted",
            },
            "extensions": {},
        },
        sectors,
    )
