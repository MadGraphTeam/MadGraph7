from __future__ import annotations

import pytest

from pyamplicol.color.plan import (
    build_color_plan,
    color_topology_replay_partitions,
)
from pyamplicol.models.builtin.process_ir import build_process_ir


@pytest.mark.parametrize(
    ("n_final", "sector_count", "partition_count", "partition_size", "representatives"),
    [
        (4, 120, 5, 24, (0, 24, 30, 32, 33)),
        (5, 720, 6, 120, (0, 120, 144, 150, 152, 153)),
        (6, 5_040, 7, 720, (0, 720, 840, 864, 870, 872, 873)),
    ],
)
def test_full_color_pure_gluon_partitions_match_crossing_templates(
    n_final: int,
    sector_count: int,
    partition_count: int,
    partition_size: int,
    representatives: tuple[int, ...],
) -> None:
    process = build_process_ir(
        f"g g > {' '.join(['g'] * n_final)}",
        color_accuracy="full",
    )
    color_plan = build_color_plan(process, color_accuracy="full")
    partitions = color_topology_replay_partitions(color_plan)

    assert color_plan.sector_count == sector_count
    assert len(partitions) == partition_count
    assert tuple(
        partition.representative_sector_id for partition in partitions
    ) == representatives
    assert {
        len(partition.active_sector_ids) for partition in partitions
    } == {partition_size}
    assert {
        sector_id
        for partition in partitions
        for sector_id in partition.active_sector_ids
    } == set(range(sector_count))

    initial_labels = {leg.label for leg in process.initial_legs}
    for partition in partitions:
        for permutation in partition.label_permutations:
            mapping = dict(permutation)
            assert {mapping[label] for label in initial_labels} == initial_labels


def test_existing_lc_runtime_wrapper_remains_lc_only() -> None:
    process = build_process_ir("g g > g g g g g", color_accuracy="full")
    color_plan = build_color_plan(process, color_accuracy="full")

    from pyamplicol.color.plan import lc_topology_replay_partitions

    assert lc_topology_replay_partitions(color_plan) == ()
