# SPDX-License-Identifier: 0BSD
"""Canonical report process, mode, dataset, and cell catalog."""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    LEGACY_AMPLICOL_MAX_OPEN_QUARK_LINES,
    Accuracy,
    CellSpec,
    ExecutionMode,
    MatrixDataset,
    MeasurementSpec,
    ModelKey,
    ModelSpec,
    ProcessFamily,
    ScalarDataset,
    Workload,
    ZVariant,
    open_quark_line_count,
)

BUILTIN_SM = ModelSpec(
    ModelKey.BUILTIN_SM,
    "built-in-sm",
    "Built-in Standard Model",
    "built-in-sm",
)
UFO_SM = ModelSpec(ModelKey.UFO_SM, "external-sm", "UFO-SM", "json")
SCALAR_CONTACT = ModelSpec(
    ModelKey.SCALAR_CONTACT,
    "scalar-contact",
    "Massless scalar contact model",
    "json",
)
SCALAR_GRAVITY = ModelSpec(
    ModelKey.SCALAR_GRAVITY,
    "scalar-gravity",
    "Scalar-gravity model",
    "json",
)

MODELS = {
    model.key: model
    for model in (BUILTIN_SM, UFO_SM, SCALAR_CONTACT, SCALAR_GRAVITY)
}

PROCESS_FAMILIES = (
    ProcessFamily(1, "dd_z_jets", r"$d\bar d\to Z+(n-1)g$", ("d", "d~"), ("z",), 9),
    ProcessFamily(
        2,
        "ud_w_jets",
        r"$u\bar d\to W^++(n-1)g$",
        ("u", "d~"),
        ("w+",),
        9,
        include_cc=True,
    ),
    ProcessFamily(
        3,
        "dd_epem_jets",
        r"$d\bar d\to e^+e^-+(n-2)g$",
        ("d", "d~"),
        ("e+", "e-"),
        9,
    ),
    ProcessFamily(
        4,
        "ud_epve_jets",
        r"$u\bar d\to e^+\nu_e+(n-2)g$",
        ("u", "d~"),
        ("e+", "ve"),
        9,
        include_cc=True,
    ),
    ProcessFamily(
        5,
        "dd_zz_jets",
        r"$d\bar d\to ZZ+(n-2)g$",
        ("d", "d~"),
        ("z", "z"),
        9,
    ),
    ProcessFamily(
        6,
        "gg_tt_jets",
        r"$gg\to t\bar t+(n-2)g$",
        ("g", "g"),
        ("t", "t~"),
        8,
    ),
    ProcessFamily(
        7,
        "dd_tt_jets",
        r"$d\bar d\to t\bar t+(n-2)g$",
        ("d", "d~"),
        ("t", "t~"),
        9,
    ),
    ProcessFamily(
        8,
        "gg_gluons",
        r"$gg\to gg+(n-2)g$",
        ("g", "g"),
        ("g", "g"),
        8,
    ),
    ProcessFamily(
        9,
        "dd_zzz_jets",
        r"$d\bar d\to ZZZ+(n-3)g$",
        ("d", "d~"),
        ("z", "z", "z"),
        9,
    ),
    ProcessFamily(
        10,
        "dd_epemzh_jets",
        r"$d\bar d\to e^+e^-ZH+(n-4)g$",
        ("d", "d~"),
        ("e+", "e-", "z", "h"),
        9,
    ),
    ProcessFamily(
        11,
        "dd_ttzh_jets",
        r"$d\bar d\to t\bar t ZH+(n-4)g$",
        ("d", "d~"),
        ("t", "t~", "z", "h"),
        9,
    ),
    ProcessFamily(
        12,
        "dd_4l_jets",
        r"$d\bar d\to e^+e^-e^+e^-+(n-4)g$",
        ("d", "d~"),
        ("e+", "e-", "e+", "e-"),
        9,
    ),
    ProcessFamily(
        13,
        "dd_3q_lines",
        r"$d\bar d\to u\bar u\,s\bar s+(n-4)g$",
        ("d", "d~"),
        ("u", "u~", "s", "s~"),
        8,
        maximum_contracted_n=6,
        include_3qqbar=True,
    ),
    ProcessFamily(
        14,
        "dd_4q_lines",
        r"$d\bar d\to u\bar u\,s\bar s\,c\bar c+(n-6)g$",
        ("d", "d~"),
        ("u", "u~", "s", "s~", "c", "c~"),
        8,
        maximum_contracted_n=6,
        include_3qqbar=True,
    ),
)


def matrix_multiplicities(accuracy: Accuracy) -> tuple[int, ...]:
    """Return publication multiplicities for one matrix accuracy.

    LC retains the full report ladder.  Contracted NLC/full-colour tables
    expose ``n=6`` so multi-quark families can declare candidate coverage
    independently of the historical AmpliCol oracle.
    """

    return tuple(range(1, 10 if accuracy is Accuracy.LC else 7))


def _measurement(
    mode: ExecutionMode,
    model: ModelKey | None,
    accuracy: Accuracy,
) -> MeasurementSpec:
    if mode is ExecutionMode.AMPLICOL:
        return MeasurementSpec(mode, None, accuracy, "fortran", None)
    if mode is ExecutionMode.COMPILED:
        return MeasurementSpec(mode, model, accuracy, "jit", 3)
    return MeasurementSpec(mode, model, accuracy, "jit", 2)


def _matrix_dataset(
    *,
    mode: ExecutionMode,
    model: ModelKey,
    accuracy: Accuracy,
    baseline_mode: ExecutionMode,
) -> MatrixDataset:
    model_label = "Built-in SM" if model is ModelKey.BUILTIN_SM else "UFO-SM"
    accuracy_label = {
        Accuracy.LC: "LC",
        Accuracy.NLC: "NLC",
        Accuracy.FULL: "full-colour",
    }[accuracy]
    mode_label = {
        ExecutionMode.RECURRENCE: "recurrence",
        ExecutionMode.COMPILED: "compiled JIT O3",
        ExecutionMode.EAGER: "eager-DAG JIT O2",
    }[mode]
    baseline_label = (
        "AmpliCol"
        if baseline_mode is ExecutionMode.AMPLICOL
        else "recurrence JIT O2"
    )
    stem = f"matrix_{mode.value}_{model.value}_{accuracy.value}"
    return MatrixDataset(
        dataset_id=stem,
        cache_name=f"{stem}.json",
        table_name=f"result_{stem}_table.tex",
        title=f"{model_label} {mode_label} versus {baseline_label} {accuracy_label}",
        candidate=_measurement(mode, model, accuracy),
        baseline=_measurement(baseline_mode, ModelKey.BUILTIN_SM, accuracy),
        multiplicities=matrix_multiplicities(accuracy),
    )


MATRIX_DATASETS = tuple(
    
        _matrix_dataset(
            mode=ExecutionMode.RECURRENCE,
            model=model,
            accuracy=accuracy,
            baseline_mode=ExecutionMode.AMPLICOL,
        )
        for model in (ModelKey.BUILTIN_SM, ModelKey.UFO_SM)
        for accuracy in Accuracy
    
) + tuple(
    _matrix_dataset(
        mode=mode,
        model=ModelKey.BUILTIN_SM,
        accuracy=accuracy,
        baseline_mode=ExecutionMode.RECURRENCE,
    )
    for mode in (ExecutionMode.COMPILED, ExecutionMode.EAGER)
    for accuracy in Accuracy
)

SCALAR_DATASETS = (
    ScalarDataset(
        dataset_id="scalar_contact",
        cache_name="scalar_contact.json",
        table_name="result_scalar_contact_table.tex",
        title="Scalar-contact ladder",
        model=ModelKey.SCALAR_CONTACT,
        process_template="scalar_0 scalar_0 > n*scalar_0",
        final_particle="scalar_0",
        multiplicities=tuple(range(2, 9)),
        measurement=_measurement(
            ExecutionMode.COMPILED,
            ModelKey.SCALAR_CONTACT,
            Accuracy.FULL,
        ),
    ),
    ScalarDataset(
        dataset_id="scalar_gravity",
        cache_name="scalar_gravity.json",
        table_name="result_scalar_gravity_table.tex",
        title="Scalar-gravity ladder",
        model=ModelKey.SCALAR_GRAVITY,
        process_template="scalar_0 scalar_0 > n*graviton",
        final_particle="graviton",
        multiplicities=tuple(range(2, 5)),
        measurement=_measurement(
            ExecutionMode.COMPILED,
            ModelKey.SCALAR_GRAVITY,
            Accuracy.FULL,
        ),
    ),
)

Z_VARIANTS = (
    ZVariant("reference", "Independent reference", ExecutionMode.AMPLICOL, "fortran"),
    ZVariant("jit_o1", "JIT level 1", ExecutionMode.COMPILED, "jit", 1),
    ZVariant("asm_o3", "ASM O3", ExecutionMode.COMPILED, "asm"),
    ZVariant(
        "cpp_o3",
        "C++ O3",
        ExecutionMode.COMPILED,
        "cpp",
        cpp_optimization="O3",
    ),
    ZVariant("jit_o3", "JIT level 3", ExecutionMode.COMPILED, "jit", 3),
    ZVariant(
        "eager_jit_o2",
        "eager-DAG JIT O2",
        ExecutionMode.EAGER,
        "jit",
        2,
    ),
    ZVariant(
        "recurrence_jit_o2",
        "recurrence JIT O2",
        ExecutionMode.RECURRENCE,
        "jit",
        2,
    ),
)

STATIC_NA_ORIGINAL_AMPLICOL_OPEN_QUARK_LINE_LIMIT = (
    "original-amplicol-open-quark-line-limit"
)


def z_dataset_id(model: ModelKey) -> str:
    if model is ModelKey.BUILTIN_SM:
        return "z_builtin_sm"
    if model is ModelKey.UFO_SM:
        return "z_external_sm"
    raise ValueError("Z ladders support only built-in SM and UFO-SM")


@dataclass(frozen=True, slots=True)
class ReportCatalog:
    models: dict[ModelKey, ModelSpec]
    process_families: tuple[ProcessFamily, ...]
    matrix_datasets: tuple[MatrixDataset, ...]
    scalar_datasets: tuple[ScalarDataset, ...]
    z_variants: tuple[ZVariant, ...]

    def dataset(self, dataset_id: str) -> MatrixDataset:
        matches = [
            dataset
            for dataset in self.matrix_datasets
            if dataset.dataset_id == dataset_id
        ]
        if not matches:
            raise KeyError(f"unknown matrix dataset {dataset_id!r}")
        return matches[0]

    def matrix_cells(self) -> tuple[CellSpec, ...]:
        cells: list[CellSpec] = []
        for dataset in self.matrix_datasets:
            for family in self.process_families:
                for n_final in dataset.multiplicities:
                    process = family.process(n_final)
                    if process is None or n_final > family.maximum_n(
                        dataset.candidate.accuracy
                    ):
                        continue
                    workloads = (
                        (Workload.SELECTED_FLOW, Workload.ALL_FLOW)
                        if dataset.candidate.accuracy is Accuracy.LC
                        else (Workload.CONTRACTED,)
                    )
                    for workload in workloads:
                        cells.append(
                            CellSpec(
                                dataset_id=dataset.dataset_id,
                                process=process,
                                n_final=n_final,
                                process_key=family.key,
                                measurement=dataset.candidate,
                                workload=workload,
                            )
                        )
        return tuple(cells)

    def reference_cells(self) -> tuple[CellSpec, ...]:
        cells: list[CellSpec] = []
        for accuracy in Accuracy:
            multiplicities = matrix_multiplicities(accuracy)
            for family in self.process_families:
                for n_final in multiplicities:
                    process = family.process(n_final)
                    if process is None or n_final > family.maximum_n(accuracy):
                        continue
                    workloads = (
                        (Workload.SELECTED_FLOW, Workload.ALL_FLOW)
                        if accuracy is Accuracy.LC
                        else (Workload.CONTRACTED,)
                    )
                    for workload in workloads:
                        cells.append(
                            CellSpec(
                                dataset_id=f"reference_amplicol_{accuracy.value}",
                                process=process,
                                n_final=n_final,
                                process_key=family.key,
                                measurement=_measurement(
                                    ExecutionMode.AMPLICOL,
                                    None,
                                    accuracy,
                                ),
                                workload=workload,
                            )
                        )
        return tuple(cells)

    def scalar_cells(self) -> tuple[CellSpec, ...]:
        return tuple(
            CellSpec(
                dataset_id=dataset.dataset_id,
                process=dataset.process(n_final),
                n_final=n_final,
                process_key=dataset.dataset_id,
                measurement=dataset.measurement,
                workload=Workload.CONTRACTED,
            )
            for dataset in self.scalar_datasets
            for n_final in dataset.multiplicities
        )

    def z_cells(self) -> tuple[CellSpec, ...]:
        cells: list[CellSpec] = []
        for model in (ModelKey.BUILTIN_SM, ModelKey.UFO_SM):
            for n_final in range(1, 10):
                process = PROCESS_FAMILIES[0].process(n_final)
                assert process is not None
                for variant in self.z_variants:
                    if variant.execution_mode is ExecutionMode.AMPLICOL:
                        continue
                    measurement = MeasurementSpec(
                        variant.execution_mode,
                        model,
                        Accuracy.LC,
                        variant.backend,
                        variant.jit_optimization_level,
                    )
                    for workload in (
                        Workload.SELECTED_FLOW,
                        Workload.ALL_FLOW,
                    ):
                        cells.append(
                            CellSpec(
                                dataset_id=z_dataset_id(model),
                                process=process,
                                n_final=n_final,
                                process_key="dd_z_jets",
                                measurement=measurement,
                                workload=workload,
                                variant=variant.key,
                            )
                        )
        return tuple(cells)

    def measurement_cells(self) -> tuple[CellSpec, ...]:
        return (
            *self.reference_cells(),
            *self.matrix_cells(),
            *self.scalar_cells(),
            *self.z_cells(),
        )

    def cell(self, cell_id: str) -> CellSpec:
        matches = [
            cell for cell in self.measurement_cells() if cell.cell_id == cell_id
        ]
        if len(matches) != 1:
            qualifier = "unknown" if not matches else "ambiguous"
            raise KeyError(f"{qualifier} report cell {cell_id!r}")
        return matches[0]

    def equivalent_cells(self, cell: CellSpec) -> tuple[CellSpec, ...]:
        """Return cells that can reuse exactly the same generated artifact."""

        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL:
            return ()
        identity = (
            cell.process,
            cell.n_final,
            cell.process_key,
            cell.measurement,
            cell.workload,
        )
        return tuple(
            sorted(
                (
                    candidate
                    for candidate in self.measurement_cells()
                    if candidate.cell_id != cell.cell_id
                    and (
                        candidate.process,
                        candidate.n_final,
                        candidate.process_key,
                        candidate.measurement,
                        candidate.workload,
                    )
                    == identity
                ),
                key=lambda candidate: candidate.cell_id,
            )
        )

    def baseline_cell(self, cell: CellSpec) -> CellSpec | None:
        """Return the display baseline for ``cell``.

        This retains the original-AmpliCol row even when the historical
        implementation cannot evaluate the process.  Use
        :meth:`validation_baseline_cell` for measurement dependencies.
        """

        if (
            cell.measurement.execution_mode is ExecutionMode.AMPLICOL
            or cell.dataset_id in {
                dataset.dataset_id for dataset in self.scalar_datasets
            }
        ):
            return None
        if cell.dataset_id.startswith("z_"):
            baseline_mode = ExecutionMode.AMPLICOL
            dataset_id = f"reference_amplicol_{cell.measurement.accuracy.value}"
        else:
            dataset = self.dataset(cell.dataset_id)
            baseline_mode = dataset.baseline.execution_mode
            dataset_id = (
                f"reference_amplicol_{cell.measurement.accuracy.value}"
                if baseline_mode is ExecutionMode.AMPLICOL
                else (
                    "matrix_recurrence_builtin_sm_"
                    f"{cell.measurement.accuracy.value}"
                )
            )
        return next(
            candidate
            for candidate in self.measurement_cells()
            if candidate.dataset_id == dataset_id
            and candidate.process_key == cell.process_key
            and candidate.n_final == cell.n_final
            and candidate.workload is cell.workload
            and candidate.measurement.execution_mode is baseline_mode
        )

    def legacy_reference_available(self, cell: CellSpec) -> bool:
        """Whether original AmpliCol can evaluate ``cell``'s concrete process."""

        return (
            open_quark_line_count(cell.process)
            <= LEGACY_AMPLICOL_MAX_OPEN_QUARK_LINES
        )

    def static_na_reason(self, cell: CellSpec) -> str | None:
        """Return the catalog-authenticated reason ``cell`` is not measurable.

        Static N/A cells remain declared catalog rows so publication layout and
        descriptor identities stay stable.  They are excluded from campaign
        planning and authenticated-current completeness because the historical
        oracle cannot represent the concrete process at all.
        """

        if (
            cell.measurement.execution_mode is ExecutionMode.AMPLICOL
            and not self.legacy_reference_available(cell)
        ):
            return STATIC_NA_ORIGINAL_AMPLICOL_OPEN_QUARK_LINE_LIMIT
        return None

    def validation_baseline_cell(self, cell: CellSpec) -> CellSpec | None:
        """Return the executable numerical baseline for ``cell``.

        Original AmpliCol remains visible in the report for processes beyond
        its three-open-line scope, but it must not censor pyAmpliCol's
        otherwise model-generic recurrence measurements.
        """

        baseline = self.baseline_cell(cell)
        if (
            baseline is not None
            and baseline.measurement.execution_mode is ExecutionMode.AMPLICOL
            and not self.legacy_reference_available(cell)
        ):
            return None
        return baseline


REPORT_CATALOG = ReportCatalog(
    MODELS,
    PROCESS_FAMILIES,
    MATRIX_DATASETS,
    SCALAR_DATASETS,
    Z_VARIANTS,
)

__all__ = [
    "BUILTIN_SM",
    "MATRIX_DATASETS",
    "MODELS",
    "PROCESS_FAMILIES",
    "REPORT_CATALOG",
    "SCALAR_CONTACT",
    "SCALAR_DATASETS",
    "STATIC_NA_ORIGINAL_AMPLICOL_OPEN_QUARK_LINE_LIMIT",
    "SCALAR_GRAVITY",
    "UFO_SM",
    "Z_VARIANTS",
    "ReportCatalog",
    "matrix_multiplicities",
    "z_dataset_id",
]
