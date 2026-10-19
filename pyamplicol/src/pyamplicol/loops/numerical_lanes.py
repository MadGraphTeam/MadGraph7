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
    """Gather currents and pointwise source values without changing the snapshot."""
    batch = currents.batch
    h = batch.helicity_count
    lanes = [point * h + helicity for point in indices for helicity in range(h)]
    width = len(currents.momenta_pairs) // batch.size if batch.size else 0
    return replace(
        currents,
        batch=select_physical_batch(batch, indices),
        born=currents.born.take_rows(lanes),
        loops=tuple(value.take_rows(lanes) for value in currents.loops),
        model=(
            currents.model.take_rows(indices)
            if currents.model.rows == batch.size
            else currents.model
        ),
        momenta_pairs=[
            pair
            for point in indices
            for pair in currents.momenta_pairs[point * width : (point + 1) * width]
        ],
    )


def require_partial_domain(bare):
    if bare.reduction_engine == "fast-tensor":
        raise NotImplementedError(
            "partial fast-tensor coefficient-head evaluation is not certified"
        )
    if getattr(bare, "color_accuracy", "full") != "full" or getattr(
        bare, "_formal_color_projected", False
    ):
        raise NotImplementedError(
            "partial native evaluation currently requires full-colour "
            "certified internal controls"
        )
    if bare.dimensional_numerators and (
        bare.model_state is not None
        or not bare.order_layout.is_single
        or tuple(bare.process.orders.correction_names) != ("QCD",)
        or bare.state_kinds != ("epsilon", "epsilon", "epsilon_bar", "epsilon_bar")
        or any(
            edge.particle.pdg_code != 21
            for edge in bare.process.born[0].native.external_edges
        )
        or any(recurrence.kernel_provider is None for recurrence in bare.recurrences)
    ):
        raise NotImplementedError(
            "partial dimensional evaluation requires the certified "
            "QCD four-gluon control"
        )

    if bare.model_state is not None:
        raw = bare.model_state.plan.program.library.raw
        mass_names = {
            particle["fields"]["mass"]["name"]
            for particle in raw.collections["particles"]
        }
        if "complex_mass_derivation" in raw.metadata or any(
            raw.record("parameters", name)["fields"].get("type") != "real"
            for name in mass_names
        ):
            raise NotImplementedError(
                "partial source complex-mass evaluation is not certified"
            )
        if (
            bare.vertex_spinor_frame != "native-physical"
            or bare.model_scale_view is None
        ):
            raise NotImplementedError(
                "partial source evaluation requires the native physical frame "
                "and a shared source scale owner"
            )


def select_bare_evaluation(native, indices):
    """Remove complete points from an already reduced sparse bare result."""
    h = native.batch.helicity_count
    lanes = [point * h + helicity for point in indices for helicity in range(h)]
    return replace(
        native,
        batch=select_physical_batch(native.batch, indices),
        born_amplitudes=native.born_amplitudes.take_rows(lanes),
        born=native.born.take_rows(indices),
        virtual=native.virtual.take_rows(indices),
        rational=None
        if native.rational is None
        else native.rational.take_rows(indices),
        statuses=tuple(native.statuses[index] for index in indices),
    )
