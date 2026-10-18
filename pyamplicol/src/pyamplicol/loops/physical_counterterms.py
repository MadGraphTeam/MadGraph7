# SPDX-License-Identifier: 0BSD
"""Native model-CT tree assembly over a shared physical Born evaluation.

Only source bases with explicit ordinary-kernel correspondence are admitted by
this initial path. CT scalar coefficients come from the model's exact Laurent
series. This boundary does not certify external wavefunction completion or a
general model's renormalization scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .exact import require_exact_numbers
from .kernels import NativeCoefficientDAG, require_finite_batch
from .physical import _born_dag, _saved
from .physical_process import _color_bilinear, _external_color


@dataclass(frozen=True, slots=True)
class ModelVirtualResult:
    born: Any
    virtual: Any
    bare: Any
    uv: Any
    r2: Any
    statuses: Any
    arithmetic: str
    precision: int | None
    reduction_engine: str
    coefficient_engine: str
    convention: str = "MadLoop-Cepsilon-model-counterterms-assembled"


@dataclass(slots=True)
class _CountertermCurrent:
    insertion: Any
    epsilon_power: int
    dag: Any
    executor: Any
    coefficient_evaluator: Any
    coefficient_names: tuple[str, ...]
    color_weights: tuple[Any, ...]


class NativeModelVirtualProgram:
    """Combine bare loops and source-derived CT trees without repeating states.

    ``source_bindings`` translates source parameter symbols into the bare
    program's global model inputs and ``bare.scale_squared_symbol``. The latter
    binds each point's renormalization scale; model inputs remain global.
    No normalized binary64 model values are read. Unknown source parameters
    fail during evaluator construction.
    """

    def __init__(self, bare, insertions, *, source_bindings):
        from symbolica import E, Replacement, S

        self.bare = bare
        self.source_bindings = tuple(source_bindings)
        backend, options = bare.backend, bare.options
        replacements = []
        for source, target in self.source_bindings:
            require_exact_numbers(target)
            replacements.append(Replacement(source, target))
        allowed = {pair.loop for pair in bare.process.orders.sectors[0].pairs}
        self.currents = []
        self.insertions = tuple(insertions)
        for insertion in self.insertions:
            if not insertion.rule.has_ordinary_basis_correspondence:
                raise NotImplementedError(
                    "counterterm lacks a calibrated tensor correspondence"
                )
            if insertion.graph.order not in allowed:
                raise ValueError(
                    "counterterm order is absent from the allowed Born/loop pairings"
                )
            for _, entry in insertion.rule.couplings:
                if any(
                    coefficient != 0 and power not in (E("-2"), E("-1"), E("0"))
                    for power, coefficient in entry.coupling.value.series(0)
                ):
                    raise NotImplementedError(
                        f"counterterm {entry.vertex_name} has unsupported "
                        "Laurent powers outside -2..0"
                    )
            dag, color = _born_dag(
                insertion.graph, bare.recurrences[0].dag.variables, bare.momenta
            )
            color = _external_color(insertion.graph, color)
            weights = tuple(
                _color_bilinear(born, color, bare.initial_average)
                for born in bare.born_colors
            )
            executor = NativeCoefficientDAG(
                dag,
                output_nodes=(len(dag.nodes) - 1,),
                backend=backend,
                coefficient_engine=bare.coefficient_engine,
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
                self.currents.append(
                    _CountertermCurrent(
                        insertion,
                        power,
                        dag,
                        executor,
                        backend.CoefficientEvaluator(
                            _saved(
                                expressions,
                                (*bare.model_inputs, bare.scale_squared_symbol),
                            ),
                            **options,
                        ),
                        tuple(symbol.get_name() for symbol, _ in substitutions),
                        weights,
                    )
                )
        parameters, ultraviolet, rational = [], [E("0")] * 3, [E("0")] * 3
        for helicity in range(len(bare.helicities)):
            born = [
                S(f"loop_counterterm::born{helicity}_{i}")
                for i in range(len(bare.born_dags))
            ]
            parameters.extend(born)
            for index, current in enumerate(self.currents):
                amplitude = S(f"loop_counterterm::amplitude{helicity}_{index}")
                parameters.append(amplitude)
                # CT Feynman rules already contain their loop factors. Unlike
                # a reduced bare numerator, this tree gets no i/(16*pi^2).
                value = sum(
                    (
                        left.conj() * weight * amplitude
                        for left, weight in zip(
                            born, current.color_weights, strict=True
                        )
                    ),
                    E("0"),
                )
                target = (
                    rational if current.insertion.rule.kind == "R2" else ultraviolet
                )
                target[-current.epsilon_power] += value + value.conj()
        self.projector = backend.CoefficientEvaluator(
            _saved([*ultraviolet, *rational], parameters), **options
        )
        poles = [S(f"loop_counterterm::pole{i}") for i in range(9)]
        self.assemble = backend.CoefficientEvaluator(
            _saved([poles[i] + poles[i + 3] + poles[i + 6] for i in range(3)], poles),
            **options,
        )

    def evaluate_decimal(self, momenta, scales_squared, *, model_values):
        native = self.bare.evaluate_native(
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
            positions = dict(batch.positions)
            positions.update(
                {
                    name: batch.bound.columns + i
                    for i, name in enumerate(current.coefficient_names)
                }
            )
            amplitude = current.executor.evaluate_batch(
                combined.take_columns(
                    [positions[p.get_name()] for p in current.dag.parameters]
                )
            )
            require_finite_batch(
                amplitude,
                "counterterm current",
                rows_per_point=len(self.bare.helicities),
            )
            amplitudes.append(amplitude)
        inputs = backend.NativeBatch.concat_columns(
            [native.born_amplitudes, *amplitudes]
        )
        counterterms = self.projector.evaluate_batch(
            inputs.reshape_rows(
                batch.size, len(self.bare.helicities) * inputs.columns, batch.point_ids
            )
        )
        require_finite_batch(counterterms, "counterterm projection")
        assembled = self.assemble.evaluate_batch(
            backend.NativeBatch.concat_columns([native.virtual, counterterms])
        )
        require_finite_batch(assembled, "complete virtual assembly")
        born = np.asarray(native.born.to_decimal(), dtype=object).reshape(batch.size, 2)
        bare = np.asarray(native.virtual.to_decimal(), dtype=object).reshape(
            batch.size, 3, 2
        )
        ct = np.asarray(counterterms.to_decimal(), dtype=object).reshape(
            batch.size, 6, 2
        )
        virtual = np.asarray(assembled.to_decimal(), dtype=object).reshape(
            batch.size, 3, 2
        )
        return ModelVirtualResult(
            born,
            virtual,
            bare,
            ct[:, :3],
            ct[:, 3:],
            native.statuses,
            self.bare.arithmetic,
            self.bare.precision,
            self.bare.reduction_engine,
            self.bare.coefficient_engine,
        )


def drell_yan_model_virtual_program(source, **options):
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
    strong, electromagnetic = (
        (4 * E("pi") * alpha_s) ** E("1/2"),
        (4 * E("pi") / inverse_alpha) ** E("1/2"),
    )
    substitutions = [
        (S("UFO::G"), strong),
        (S("UFO::ee"), electromagnetic),
        (S("UFO::Ncol"), E("3")),
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
        bare, matched.insertions, source_bindings=substitutions
    )
