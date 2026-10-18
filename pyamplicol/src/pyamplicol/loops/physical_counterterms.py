# SPDX-License-Identifier: 0BSD
"""Native model-CT tree assembly over a shared physical Born evaluation.

Only source bases with explicit ordinary-kernel correspondence or independently
validated local tensors are admitted. CT scalars come from exact model Laurent
series. This boundary does not certify external wavefunction completion or a
general model's renormalization scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .exact import require_exact_numbers
from .graph_basis import build_color_tree_recurrence
from .kernels import NativeCoefficientDAG, require_finite_batch
from .physical import _saved
from .physical_orders import OrderedModelVirtualResult, SummedModelValues
from .physical_process import _color_bilinear, _external_color


@dataclass(frozen=True, slots=True)
class ModelVirtualResult:
    born: Any
    virtual: Any
    bare: Any
    uv: Any
    r2: Any
    uv_local: Any
    uv_external_fields: Any
    statuses: Any
    arithmetic: str
    precision: int | None
    reduction_engine: str
    coefficient_engine: str
    color_accuracy: str
    born_color_degree: int | None
    virtual_color_degree: int | None
    r2_strategy: str = "model"
    convention: str = "MadLoop-Cepsilon-virtual-assembled"


@dataclass(slots=True)
class _CountertermCurrent:
    insertion: Any
    epsilon_power: int
    dag: Any
    executor: Any
    coefficient_evaluator: Any
    coefficient_names: tuple[str, ...]
    color_weights: tuple[Any, ...]
    coefficient_expressions: tuple[Any, ...]
    term_count: int
    input_columns: tuple[int, ...]
    color_grade: int | None = None


@dataclass(slots=True)
class _ExternalFieldCurrent:
    insertion: Any
    epsilon_power: int
    born_indices: tuple[int, ...]
    coefficient_evaluator: Any
    coefficient_expression: Any
    color_weights: tuple[Any, ...]
    color_grade: int | None = None


def _require_laurent_domain(value, context):
    from symbolica import E

    if any(
        coefficient != 0 and power not in (E("-2"), E("-1"), E("0"))
        for power, coefficient in value.series(0)
    ):
        raise NotImplementedError(
            f"{context} has unsupported Laurent powers outside -2..0"
        )


class NativeModelVirtualProgram:
    """Combine bare loops and source-derived CT trees without repeating states.

    ``source_bindings`` translates source parameter symbols into the bare
    program's global model inputs and ``bare.scale_squared_symbol``. The latter
    binds each point's renormalization scale; model inputs remain global.
    No normalized binary64 model values are read. Unknown source parameters
    fail during evaluator construction. Internal ``direct`` selection excludes
    every model R2 insertion and consumes the bare program's closed dimensional
    remainder; local UV and external-field terms retain the source convention.
    """

    def __init__(
        self,
        bare,
        insertions,
        *,
        source_bindings,
        external_fields=(),
        formal_nc=None,
        color_accuracy="full",
        r2_strategy="model",
    ):
        from symbolica import E, Replacement, S

        from pyamplicol.config.models import ColorAccuracy

        from .color_bilinears import SUNColorSpace, external_color_tensor
        from .physical_color import project_physical_color, split_color_grades

        if formal_nc is not None or r2_strategy == "direct":
            bare.require_single_order("formal or direct model counterterm assembly")
        self.bare = bare
        if r2_strategy not in {"model", "direct"}:
            raise ValueError("internal R2 strategy must be model or direct")
        if r2_strategy == "direct" and not bare.dimensional_numerators:
            raise ValueError("direct R2 requires complete dimensional numerators")
        self.r2_strategy = r2_strategy
        self.color_accuracy = ColorAccuracy(color_accuracy).value
        self.color_projection = None
        if getattr(bare, "_model_color_metrics_bound", False) and (
            formal_nc is not None or getattr(bare, "_formal_color_projected", False)
        ):
            raise ValueError(
                "bare color metrics are already bound to a model program; "
                "construct a fresh bare program for each formal color projection"
            )
        if formal_nc is None and self.color_accuracy != "full":
            raise ValueError(
                "projected model colors require an explicit SU(Nc) continuation"
            )
        space = None if formal_nc is None else SUNColorSpace(formal_nc)
        self.source_bindings = tuple(source_bindings)
        backend, options = bare.backend, bare.options
        replacements = []
        for source, target in self.source_bindings:
            require_exact_numbers(target)
            replacements.append(Replacement(source, target))
        sectors = bare.process.orders.sectors
        self.sector_pairs = tuple(
            frozenset((pair.born, pair.loop) for pair in sector.pairs)
            for sector in sectors
        )
        allowed = {pair.loop for sector in sectors for pair in sector.pairs}
        corrections = frozenset(bare.process.orders.correction_names)
        admitted_loops = {graph.id for graph in bare.loop_graphs}
        self.currents = []
        self.matched_insertions = tuple(insertions)
        self.insertions = tuple(
            insertion
            for insertion in self.matched_insertions
            if r2_strategy == "model" or insertion.rule.kind != "R2"
        )
        for insertion in self.insertions:
            rule = insertion.rule
            if rule.perturbation_type and rule.perturbation_type not in corrections:
                raise ValueError("counterterm correction is outside the selected set")
            if rule.kind == "UVtree":
                # Source-defined UVtree interactions are eligible without a loop
                # witness; an unmatched ordinary CT must never take this branch.
                if not corrections.intersection(name for name, _ in rule.order):
                    raise ValueError("UVtree has no selected source correction order")
            elif not admitted_loops.intersection(insertion.witness_loop_ids):
                raise ValueError("counterterm has no admitted loop graph witness")
            if not insertion.rule.has_validated_tensor_basis:
                raise NotImplementedError(
                    "counterterm lacks a calibrated tensor correspondence "
                    "or validated primitive"
                )
            if insertion.graph.order not in allowed:
                raise ValueError(
                    "counterterm order is absent from the allowed Born/loop pairings"
                )
            for _, entry in insertion.rule.couplings:
                _require_laurent_domain(
                    entry.coupling.value, f"counterterm {entry.vertex_name}"
                )
            recurrence = build_color_tree_recurrence(
                insertion.graph,
                variables=bare.recurrences[0].dag.momentum_variables,
                momenta=bare.momenta,
            )
            dag = recurrence.dag
            if any(
                len(dag.nodes[term.node].output_labels) != 1
                or any(dag.nodes[term.node].output_labels[0].powers)
                for term in recurrence.terms
            ):
                raise ValueError("CT color term must have one scalar coefficient")
            if space is None:
                colors = tuple(
                    _external_color(insertion.graph, term.color)
                    for term in recurrence.terms
                )
                weights = tuple(
                    tuple(
                        _color_bilinear(born, color, bare.initial_average)
                        for color in colors
                    )
                    for born in bare.born_colors
                )
            else:
                weights = space.rectangular(
                    bare.born_color_tensors,
                    tuple(
                        external_color_tensor(insertion.graph, term.color)
                        for term in recurrence.terms
                    ),
                )
            executor = NativeCoefficientDAG(
                dag,
                output_nodes=recurrence.final_nodes,
                backend=backend,
                coefficient_engine=bare.coefficient_engine,
                kernel_cache=bare.coefficient_kernel_cache,
                **options,
            )
            for power in (0, -1, -2):
                substitutions = insertion.coefficient_substitutions(power)
                expressions = [
                    value.replace_multiple(replacements) for _, value in substitutions
                ]
                for expression in expressions:
                    require_exact_numbers(expression)
                if all(expression == 0 for expression in expressions):
                    continue
                coefficient_names = tuple(
                    symbol.get_name() for symbol, _ in substitutions
                )
                positions = dict(bare.bound_positions)
                positions.update(
                    {
                        name: bare.bound_columns + i
                        for i, name in enumerate(coefficient_names)
                    }
                )
                grades = (
                    ((None, tuple(expressions)),)
                    if space is None
                    else split_color_grades(expressions, formal_nc)
                )
                if (
                    space is not None
                    and len(recurrence.terms) > 1
                    and (len(grades) != 1 or any(value == 0 for value in grades[0][1]))
                ):
                    raise NotImplementedError(
                        "mixed-grade multi-term counterterms require an exact "
                        "coupling-to-term incidence proof"
                    )
                for degree, coefficients in grades:
                    self.currents.append(
                        _CountertermCurrent(
                            insertion,
                            power,
                            dag,
                            executor,
                            None,
                            coefficient_names,
                            weights
                            if degree is None
                            else tuple(
                                tuple(formal_nc**degree * value for value in row)
                                for row in weights
                            ),
                            coefficients,
                            len(recurrence.terms),
                            tuple(positions[p.get_name()] for p in executor.parameters),
                            degree,
                        )
                    )
        self.external_fields = tuple(external_fields)
        self.field_currents = []
        born_indices = {
            graph.id: tuple(
                term
                for term, source in enumerate(bare.born_term_graphs)
                if source == index
            )
            for index, graph in enumerate(bare.process.born)
        }
        for insertion in self.external_fields:
            if (
                insertion.rule.correction not in corrections
                or insertion.order not in allowed
                or insertion.born.id not in born_indices
            ):
                raise ValueError(
                    "external field is outside the admitted Born/loop catalogue"
                )
            _require_laurent_domain(insertion.rule.coefficient, "external field")
            for power in (0, -1, -2):
                value = insertion.coefficient(power).replace_multiple(replacements)
                weights = tuple(
                    tuple(row[index] for index in born_indices[insertion.born.id])
                    for row in bare.born_metric
                )
                if space is not None:
                    weights = space.rectangular(
                        bare.born_color_tensors,
                        tuple(
                            bare.born_color_tensors[index]
                            for index in born_indices[insertion.born.id]
                        ),
                    )
                require_exact_numbers(value)
                if value == 0:
                    continue
                grades = (
                    ((None, (value,)),)
                    if space is None
                    else split_color_grades((value,), formal_nc)
                )
                for degree, (coefficient,) in grades:
                    self.field_currents.append(
                        _ExternalFieldCurrent(
                            insertion,
                            power,
                            born_indices[insertion.born.id],
                            None,
                            coefficient,
                            weights
                            if degree is None
                            else tuple(
                                tuple(formal_nc**degree * item for item in row)
                                for row in weights
                            ),
                            degree,
                        )
                    )
        if space is not None:
            rows = [
                [value]
                for current in (*self.currents, *self.field_currents)
                for row in current.color_weights
                for value in row
            ]
            self.color_projection = project_physical_color(
                bare, formal_nc, rows, self.color_accuracy
            )
            values = iter(self.color_projection.additional_rows)
            for current in (*self.currents, *self.field_currents):
                current.color_weights = tuple(
                    tuple(next(values) / bare.initial_average for _ in row)
                    for row in current.color_weights
                )
        for current in self.currents:
            current.coefficient_evaluator = backend.CoefficientEvaluator(
                _saved(
                    current.coefficient_expressions,
                    (*bare.model_inputs, bare.scale_squared_symbol),
                ),
                **options,
            )
        for current in self.field_currents:
            current.coefficient_evaluator = backend.CoefficientEvaluator(
                _saved(
                    [current.coefficient_expression],
                    (*bare.model_inputs, bare.scale_squared_symbol),
                ),
                **options,
            )
        width = 3 * len(sectors)
        parameters, ultraviolet, rational, fields = (
            [],
            [E("0")] * width,
            [E("0")] * width,
            [E("0")] * width,
        )
        for helicity in range(len(bare.helicities)):
            born = [
                S(f"loop_counterterm::born{helicity}_{i}")
                for i in range(len(bare.born_colors))
            ]
            parameters.extend(born)
            for index, current in enumerate(self.currents):
                amplitudes = [
                    S(f"loop_counterterm::amplitude{helicity}_{index}_{term}")
                    for term in range(current.term_count)
                ]
                parameters.extend(amplitudes)
                # Keep complex interference until closure. Source rules already
                # include the loop factor; pair masks only select Born bra terms.
                target = (
                    rational if current.insertion.rule.kind == "R2" else ultraviolet
                )
                for sector, pairs in enumerate(self.sector_pairs):
                    value = sum(
                        (
                            left.conj() * weight * amplitude
                            for left, order, row in zip(
                                born,
                                bare.born_term_orders,
                                current.color_weights,
                                strict=True,
                            )
                            if (order, current.insertion.graph.order) in pairs
                            for weight, amplitude in zip(row, amplitudes, strict=True)
                        ),
                        E("0"),
                    )
                    target[3 * sector - current.epsilon_power] += value + value.conj()
            for index, current in enumerate(self.field_currents):
                factor = S(f"loop_counterterm::field{helicity}_{index}")
                parameters.append(factor)
                # The source includes delta-Z/2. Right-hand terms belong to
                # the insertion's exact Born graph; only the left order is masked.
                for sector, pairs in enumerate(self.sector_pairs):
                    value = sum(
                        (
                            left.conj() * weight * factor * born[right]
                            for left, order, row in zip(
                                born,
                                bare.born_term_orders,
                                current.color_weights,
                                strict=True,
                            )
                            if (order, current.insertion.order) in pairs
                            for right, weight in zip(
                                current.born_indices, row, strict=True
                            )
                        ),
                        E("0"),
                    )
                    fields[3 * sector - current.epsilon_power] += value + value.conj()
        self.projector = backend.CoefficientEvaluator(
            _saved([*ultraviolet, *rational, *fields], parameters), **options
        )
        poles = [S(f"loop_counterterm::pole{i}") for i in range(4 * width)]
        self.assemble = backend.CoefficientEvaluator(
            _saved(
                [
                    *[
                        sum((poles[i + k * width] for k in range(4)), E("0"))
                        for i in range(width)
                    ],
                    *[poles[i + width] + poles[i + 3 * width] for i in range(width)],
                ],
                poles,
            ),
            **options,
        )
        # Sum every returned component in the selected carrier, before strings.
        born_count = len(bare.order_layout.born)
        sum_parameters = [
            S(f"loop_counterterm::sum{i}") for i in range(born_count + 6 * width)
        ]
        self.sum_order_values = backend.CoefficientEvaluator(
            _saved(
                [
                    sum(sum_parameters[:born_count], E("0")),
                    *[
                        sum(
                            (
                                sum_parameters[
                                    born_count + component * width + 3 * sector + pole
                                ]
                                for sector in range(len(sectors))
                            ),
                            E("0"),
                        )
                        for component in range(6)
                        for pole in range(3)
                    ],
                ],
                sum_parameters,
            ),
            **options,
        )
        # Dense wrappers can share immutable full-color metrics. A formal
        # projection must own its metrics, since its CT weights use the same P.
        bare._model_color_metrics_bound = True
        bare._formal_color_projected = space is not None

    def evaluate_ordered_decimal(self, momenta, scales_squared, *, model_values):
        native = self.bare.evaluate_ordered_native(
            momenta, scales_squared, model_values=model_values
        )
        batch, backend = native.batch, self.bare.backend
        coefficient_inputs = backend.NativeBatch.concat_columns(
            [
                batch.model_arguments.take_rows(
                    [0] * batch.size, row_ids=batch.point_ids
                ),
                batch.scales_squared,
            ]
        )
        amplitudes = []
        for current in self.currents:
            coefficients = current.coefficient_evaluator.evaluate_batch(
                coefficient_inputs
            )
            require_finite_batch(coefficients, "counterterm scalar coefficients")
            coefficients = coefficients.take_rows(
                np.repeat(np.arange(batch.size), len(self.bare.helicities)).tolist(),
                row_ids=batch.lane_ids,
            )
            combined = backend.NativeBatch.concat_columns([batch.bound, coefficients])
            amplitude = current.executor.evaluate_batch(
                combined.take_columns(current.input_columns)
            )
            require_finite_batch(
                amplitude,
                "counterterm current",
                rows_per_point=len(self.bare.helicities),
            )
            amplitudes.append(amplitude)
        for current in self.field_currents:
            coefficients = current.coefficient_evaluator.evaluate_batch(
                coefficient_inputs
            )
            require_finite_batch(coefficients, "external-field scalar coefficients")
            amplitudes.append(
                coefficients.take_rows(
                    np.repeat(
                        np.arange(batch.size), len(self.bare.helicities)
                    ).tolist(),
                    row_ids=batch.lane_ids,
                )
            )
        inputs = backend.NativeBatch.concat_columns(
            [native.born_amplitudes, *amplitudes]
        )
        counterterms = self.projector.evaluate_batch(
            inputs.reshape_rows(
                batch.size, len(self.bare.helicities) * inputs.columns, batch.point_ids
            )
        )
        require_finite_batch(counterterms, "counterterm projection")
        if self.r2_strategy == "direct":
            if native.rational is None:
                raise ValueError("dimensional numerator evaluation omitted R2")
            counterterms = backend.NativeBatch.concat_columns(
                [
                    counterterms.take_columns([0, 1, 2]),
                    native.rational,
                    counterterms.take_columns([6, 7, 8]),
                ]
            )
        assembled = self.assemble.evaluate_batch(
            backend.NativeBatch.concat_columns([native.virtual, counterterms])
        )
        require_finite_batch(assembled, "complete virtual assembly")
        sectors = len(self.bare.order_layout.virtual)
        width = 3 * sectors
        component_batches = [
            assembled.take_columns(range(width)),
            native.virtual,
            assembled.take_columns(range(width, 2 * width)),
            counterterms.take_columns(range(width, 2 * width)),
            counterterms.take_columns(range(width)),
            counterterms.take_columns(range(2 * width, 3 * width)),
        ]
        totals = self.sum_order_values.evaluate_batch(
            backend.NativeBatch.concat_columns([native.born, *component_batches])
        )
        require_finite_batch(totals, "sum over completed perturbative orders")

        def values(buffer, *shape):
            return np.asarray(buffer.to_decimal(), dtype=object).reshape(
                batch.size, *shape, 2
            )

        summed = values(totals, 19)
        return OrderedModelVirtualResult(
            values(native.born, len(self.bare.order_layout.born)),
            *[values(buffer, sectors, 3) for buffer in component_batches],
            self.bare.order_layout,
            self.bare.order_channels,
            native.statuses,
            self.bare.arithmetic,
            self.bare.precision,
            self.bare.reduction_engine,
            self.bare.coefficient_engine,
            self.color_accuracy,
            None
            if self.color_projection is None
            else self.color_projection.born_degree,
            None
            if self.color_projection is None
            else self.color_projection.virtual_degree,
            SummedModelValues(
                summed[:, 0], *[summed[:, 1 + 3 * i : 4 + 3 * i] for i in range(6)]
            ),
            self.r2_strategy,
        )

    def evaluate_decimal(self, momenta, scales_squared, *, model_values):
        """Single-order adapter; summing distinct physical sectors is explicit."""
        self.bare.require_single_order("single-sector model virtual adapter")
        result = self.evaluate_ordered_decimal(
            momenta, scales_squared, model_values=model_values
        )
        return ModelVirtualResult(
            result.born[:, 0],
            *[
                getattr(result, name)[:, 0]
                for name in (
                    "virtual",
                    "bare",
                    "uv",
                    "r2",
                    "uv_local",
                    "uv_external_fields",
                )
            ],
            result.statuses,
            result.arithmetic,
            result.precision,
            result.reduction_engine,
            result.coefficient_engine,
            result.color_accuracy,
            result.born_color_degree,
            result.virtual_color_degree,
            result.r2_strategy,
        )


def drell_yan_model_virtual_program(
    source, *, color_accuracy="full", formal_color=False, **options
):
    """Photon-only massless QCD DY control using the source model's R2 rule.

    Its exact source-vector basis is tied explicitly to the native ordinary
    representative calibrated by the physical Drell--Yan control.
    """
    from symbolica import E, S
    from symbolica.community import hepkit as h

    from .counterterms import match_counterterms, prepare_counterterm_model
    from .graphs import build_loop_process
    from .model_correspondence import OrdinaryBasisRepresentative
    from .physical_process import NativeBareProcessProgram

    model = h.Model.qcd_qed()
    ordinary = [
        entry
        for entry in source.kernels
        if entry.kind == "tree"
        and tuple(
            source.raw.collections["particles"][i]["fields"]["pdg_code"]
            for i in entry.particle_ids
        )
        == (-2, 2, 22)
    ]
    if len(ordinary) != 1:
        raise NotImplementedError(
            "DY control requires one source ordinary quark-photon basis"
        )
    prepared = prepare_counterterm_model(
        source,
        model,
        ordinary_basis_representatives=[
            OrdinaryBasisRepresentative(
                ordinary[0].key, "V_132", 0, 0, "native-qcd-qed-vector-current-control"
            )
        ],
    )
    process = build_loop_process(
        prepared.process(["u", "u~"], ["e-", "e+"]),
        ordinary_model=prepared.ordinary_model,
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
    )
    matched = match_counterterms(
        prepared,
        prepared.process(["u", "u~"], ["e-", "e+"], counterterms=True),
        process,
        selected_corrections=["QCD"],
    )
    if len(matched.insertions) != 1 or matched.insertions[0].rule.kind != "R2":
        raise NotImplementedError(
            "DY control requires the one matched model R2 insertion"
        )
    alpha_s, inverse_alpha = S(
        "loop_model_input::alpha_s", "loop_model_input::inverse_alpha"
    )
    nc = (
        S("loop_physical_color::Nc", is_real=True)
        if formal_color or color_accuracy != "full"
        else None
    )
    strong, electromagnetic = (
        (4 * E("pi") * alpha_s) ** E("1/2"),
        (4 * E("pi") / inverse_alpha) ** E("1/2"),
    )
    substitutions = [
        (S("UFO::G"), strong),
        (S("UFO::ee"), electromagnetic),
        (S("UFO::Ncol"), E("3") if nc is None else nc),
        (S("UFO::lhv"), E("1")),
    ]
    expressions = {}
    for name in ("GC_2", "GC_3", "GC_11"):
        coupling, value = model.coupling(name), model.coupling(name).expression
        for symbol, expression in substitutions:
            value = value.replace(symbol, expression)
        expressions[coupling.symbol.get_name()] = value
    bare = NativeBareProcessProgram(
        process,
        state_kinds=("u", "v_bar", "u_bar", "v"),
        initial_average=E("36"),
        model_expressions=expressions,
        model_inputs=(alpha_s, inverse_alpha),
        **options,
    )
    return NativeModelVirtualProgram(
        bare,
        matched.insertions,
        source_bindings=substitutions,
        formal_nc=nc,
        color_accuracy=color_accuracy,
    )


def distinct_flavour_model_virtual_program(source, **options):
    """Six-flavour massless u ubar -> d dbar with source UV/R2 rules.

    The calibrated vector-current bases and independently checked gluon
    two-point primitives bound this control. Global model inputs are alpha_s
    and top mass; bottom/charm masses are zero. Scale remains per point.
    """
    return _qcd_model_virtual_program(source, ("d", "d~"), **options)


def top_pair_model_virtual_program(source, **options):
    """Top-pair source UV/R2 and external-field factors on supplied on-shell data."""
    return _qcd_model_virtual_program(source, ("t", "t~"), **options)


def gluon_scattering_model_virtual_program(
    source, *, color_accuracy="full", formal_color=False, r2_strategy="model", **options
):
    """Bounded gg -> gg with source UV and model or direct R2.

    The direct diagnostic covers the selected vectorlike fermion, massless
    gauge and ghost cycles in the same HV convention. It does not enable
    production direct eligibility or a general model continuation. Formal
    full/LC/NLC colour is supported with direct R2 and the controlled UV grades.
    """
    if (formal_color or color_accuracy != "full") and r2_strategy != "direct":
        raise NotImplementedError(
            "gluon model counterterms lack a certified SU(Nc) continuation"
        )
    return _qcd_model_virtual_program(
        source,
        ("g", "g"),
        gluon_only=True,
        color_accuracy=color_accuracy,
        formal_color=formal_color,
        r2_strategy=r2_strategy,
        **options,
    )


def quark_gluon_model_virtual_program(
    source, *, color_accuracy="full", formal_color=False, **options
):
    """Fixed-SU(3) u ubar -> gg with the explicitly controlled quark slash."""
    if formal_color or color_accuracy != "full":
        raise NotImplementedError(
            "multi-term virtual color projection is not supported"
        )
    return _qcd_model_virtual_program(source, ("g", "g"), **options)


def _qcd_model_virtual_program(
    source,
    outgoing,
    *,
    color_accuracy="full",
    formal_color=False,
    gluon_only=False,
    r2_strategy="model",
    **options,
):
    from symbolica import E, S
    from symbolica.community import hepkit as h

    from .counterterms import match_counterterms, prepare_counterterm_model
    from .graphs import build_loop_process
    from .model_correspondence import (
        CountertermPrimitiveValidation,
        OrdinaryBasisRepresentative,
    )
    from .physical_process import NativeBareProcessProgram
    from .wavefunction import match_external_field_counterterms

    model = h.Model.qcd_qed()
    if r2_strategy not in {"model", "direct"}:
        raise ValueError("internal R2 strategy must be model or direct")
    if r2_strategy == "direct" and not gluon_only:
        raise NotImplementedError("direct physical control currently requires gg -> gg")
    incoming = ("g", "g") if gluon_only else ("u", "u~")
    gluon_final = tuple(outgoing) == ("g", "g")
    if gluon_only:
        state_kinds, average = (
            ("epsilon", "epsilon", "epsilon_bar", "epsilon_bar"),
            "512",
        )
    elif gluon_final:
        state_kinds, average = ("u", "v_bar", "epsilon_bar", "epsilon_bar"), "72"
    else:
        state_kinds, average = ("u", "v_bar", "u_bar", "v"), "36"
    if gluon_only:
        from .model_gluons import prepare_gluon_counterterm_basis

        basis = prepare_gluon_counterterm_basis(source, model)
        prepared = prepare_counterterm_model(
            source,
            model,
            entries=basis.entries,
            ordinary_basis_representatives=basis.ordinary_representatives,
            primitive_validations=basis.primitive_validations,
        )
    elif gluon_final:
        from .model_qcd import prepare_qcd_counterterm_basis

        basis = prepare_qcd_counterterm_basis(source, model, quark_pdgs=(2,))
        prepared = prepare_counterterm_model(
            source,
            model,
            entries=basis.entries,
            ordinary_basis_representatives=basis.ordinary_representatives,
            primitive_validations=basis.primitive_validations,
            tensor_representatives=basis.tensor_representatives,
        )
    else:
        representatives, validations = [], []
        # Exact expressions tested against the independent ALOHA component oracle.
        gluon_lorentz = {"P(-1,1)*P(-1,1)*Metric(1,2)", "P(1,1)*P(2,1)", "Metric(1,2)"}
        for entry in source.kernels:
            particles = [
                source.raw.collections["particles"][i]["fields"]
                for i in entry.particle_ids
            ]
            pdgs = tuple(p["pdg_code"] for p in particles)
            ordinary_rules = {
                (-1, 1, 21): "V_74",
                (-2, 2, 21): "V_135",
                (-6, 6, 21): "V_137",
            }
            if entry.kind == "tree" and pdgs in ordinary_rules:
                representatives.append(
                    OrdinaryBasisRepresentative(
                        entry.key,
                        ordinary_rules[pdgs],
                        0,
                        0,
                        "native-qcd-qed-vector-current-control",
                    )
                )
            if (
                entry.kind != "tree"
                and pdgs == (21, 21)
                and entry.color == "Tr(1,2)"
                and entry.lorentz in gluon_lorentz
            ):
                validations.append(
                    CountertermPrimitiveValidation(
                        entry.key,
                        entry.color,
                        entry.lorentz,
                        (21, 21),
                        (3, 3),
                        (8, 8),
                        "local-gluon-two-point-aloha-control",
                    )
                )
        prepared = prepare_counterterm_model(
            source,
            model,
            ordinary_basis_representatives=representatives,
            primitive_validations=validations,
        )
    process = build_loop_process(
        prepared.process(incoming, outgoing),
        ordinary_model=model,
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
    )
    matched = match_counterterms(
        prepared,
        prepared.process(incoming, outgoing, counterterms=True),
        process,
        selected_corrections=["QCD"],
    )
    alpha_s, top_mass = S("loop_model_input::alpha_s", "loop_model_input::top_mass")
    nc = (
        S("loop_physical_color::Nc", is_real=True)
        if formal_color or color_accuracy != "full"
        else None
    )
    strong = (4 * E("pi") * alpha_s) ** E("1/2")
    expressions = {"UFO::MB": E("0"), "UFO::MC": E("0"), "UFO::MT": top_mass}
    for name in ("GC_10", "GC_11", "GC_12"):
        coupling = model.coupling(name)
        expressions[coupling.symbol.get_name()] = coupling.expression.replace(
            S("UFO::G"), strong
        )
    bare = NativeBareProcessProgram(
        process,
        state_kinds=state_kinds,
        initial_average=E(average),
        model_expressions=expressions,
        model_inputs=(alpha_s, top_mass),
        dimensional_numerators=r2_strategy == "direct",
        **options,
    )
    substitutions = [
        (S("UFO::G"), strong),
        (S("UFO::MT"), top_mass),
        (S("UFO::MB"), E("0")),
        (S("UFO::MC"), E("0")),
        (S("UFO::Ncol"), E("3") if nc is None else nc),
        (S("UFO::CA"), E("3") if nc is None else nc),
        (S("UFO::CF"), E("4/3") if nc is None else (nc**2 - 1) / (2 * nc)),
        (S("UFO::TF"), E("1/2")),
        (S("UFO::lhv"), E("1")),
        (S("UFO::MU_R"), bare.scale_squared_symbol ** E("1/2")),
    ]
    fields = match_external_field_counterterms(
        source,
        process,
        selected_corrections=["QCD"],
        allowed_amplitude_orders={
            pair.loop for pair in process.orders.sectors[0].pairs
        },
    )
    return NativeModelVirtualProgram(
        bare,
        matched.insertions,
        source_bindings=substitutions,
        external_fields=fields.insertions,
        formal_nc=nc,
        color_accuracy=color_accuracy,
        r2_strategy=r2_strategy,
    )
