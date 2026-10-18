# SPDX-License-Identifier: 0BSD
"""Internal numerical-failure metadata and native point compaction.

A successful native reduction is not an accuracy certificate. These helpers only
isolate failed lanes; they neither estimate errors nor construct missing values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class NumericalLaneFailure:
    point_id: int
    stage: str
    detail: str
    order_index: int | None = None
    family_index: int | None = None


class LaneFailures:
    def __init__(self):
        self.entries = []
        self.point_ids = set()

    def record(self, point_id, stage, detail, *, order_index=None, family_index=None):
        self.entries.append(
            NumericalLaneFailure(point_id, stage, detail, order_index, family_index)
        )
        self.point_ids.add(point_id)

    def require_finite(self, values, stage, *, rows_per_point=1):
        for lane in values.nonfinite_rows():
            self.record(lane // rows_per_point, stage, f"nonfinite native lane {lane}")

    def successful_indices(self, point_ids):
        return [i for i, point in enumerate(point_ids) if point not in self.point_ids]


def select_physical_batch(batch, indices):
    """Gather whole physical points while retaining their original row identities."""
    h = batch.helicity_count
    lanes = [point * h + helicity for point in indices for helicity in range(h)]
    return replace(
        batch,
        size=len(indices),
        rows=len(lanes),
        point_ids=[batch.point_ids[i] for i in indices],
        lane_ids=[batch.lane_ids[i] for i in lanes],
        bound=batch.bound.take_rows(lanes),
        scales_squared=batch.scales_squared.take_rows(indices),
    )


def select_bare_currents(currents, indices):
    """Gather built-in-model currents; source-owned pointwise models are gated."""
    batch = currents.batch
    if batch.model_snapshot is not None:
        raise NotImplementedError("partial source-model currents are not certified")
    h = batch.helicity_count
    lanes = [point * h + helicity for point in indices for helicity in range(h)]
    width = len(currents.momenta_pairs) // batch.size if batch.size else 0
    return replace(
        currents,
        batch=select_physical_batch(batch, indices),
        born=currents.born.take_rows(lanes),
        loops=tuple(value.take_rows(lanes) for value in currents.loops),
        momenta_pairs=[
            pair
            for point in indices
            for pair in currents.momenta_pairs[point * width : (point + 1) * width]
        ],
    )


def require_partial_domain(bare):
    if (
        bare.model_state is not None
        or bare.dimensional_numerators
        or getattr(bare, "color_accuracy", "full") != "full"
        or getattr(bare, "_formal_color_projected", False)
    ):
        raise NotImplementedError(
            "partial native evaluation currently requires built-in full-colour "
            "four-dimensional model-rule controls"
        )
