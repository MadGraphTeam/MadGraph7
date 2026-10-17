# SPDX-License-Identifier: 0BSD
"""Logical display accounting for publication-facing report tables."""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import REPORT_CATALOG, ReportCatalog
from .models import ExecutionMode, ModelKey, Workload


@dataclass(frozen=True, slots=True)
class DisplayAccounting:
    """Counts that distinguish measured cells from intentional table markers."""

    maximum_n_final: int
    declared_measurement_cell_count: int
    required_measurement_count: int
    catalog_static_na_cell_count: int
    structurally_not_applicable_display_slot_count: int
    not_exposed_display_slot_count: int


def report_display_accounting(
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
    max_n_final: int = 9,
) -> DisplayAccounting:
    """Return canonical logical-slot counts through ``max_n_final``.

    A declared measurement is one catalog ``CellSpec``. Required measurements
    exclude only catalog-authenticated static N/A cells that no supported
    producer can evaluate. A structurally inapplicable slot is one matrix
    process/multiplicity position for which no process exists. A not-exposed
    slot is one original-AmpliCol execution submetric in a dedicated Z-ladder
    row; those rows expose generation and wall timing, but not a separately
    defined native-execution boundary.
    """

    if max_n_final < 1:
        raise ValueError("max_n_final must be positive")
    declared_cells = tuple(
        cell
        for cell in catalog.measurement_cells()
        if cell.n_final <= max_n_final
    )
    static_na = sum(
        catalog.static_na_reason(cell) is not None for cell in declared_cells
    )
    required = len(declared_cells) - static_na
    matrix_datasets = getattr(catalog, "matrix_datasets", ())
    process_families = getattr(catalog, "process_families", ())
    structural = 0
    for dataset in matrix_datasets:
        for family in process_families:
            for n_final in dataset.multiplicities:
                if n_final > max_n_final:
                    continue
                if family.process(n_final) is None or n_final > family.maximum_n(
                    dataset.candidate.accuracy
                ):
                    structural += 1

    variants = getattr(catalog, "z_variants", ())
    reference_variants = sum(
        variant.execution_mode is ExecutionMode.AMPLICOL for variant in variants
    )
    model_count = sum(
        model in getattr(catalog, "models", {})
        for model in (ModelKey.BUILTIN_SM, ModelKey.UFO_SM)
    )
    z_multiplicity_count = sum(n_final <= max_n_final for n_final in range(1, 10))
    not_exposed = (
        reference_variants
        * model_count
        * z_multiplicity_count
        * len((Workload.SELECTED_FLOW, Workload.ALL_FLOW))
    )
    return DisplayAccounting(
        maximum_n_final=max_n_final,
        declared_measurement_cell_count=len(declared_cells),
        required_measurement_count=required,
        catalog_static_na_cell_count=static_na,
        structurally_not_applicable_display_slot_count=structural,
        not_exposed_display_slot_count=not_exposed,
    )


__all__ = ["DisplayAccounting", "report_display_accounting"]
