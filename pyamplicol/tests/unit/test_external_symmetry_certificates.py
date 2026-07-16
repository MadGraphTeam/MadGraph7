# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol.generation.dag_algorithms import (
    infer_minimal_coupling_order_limits,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.models import BuiltinSMModel, CompiledUFOModel, compile_model_source
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.external_symmetries import (
    derive_external_symmetry_certificates,
)
from pyamplicol.processes.model import build_model_process_ir

MODEL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)

_EXTERNAL_SM_TOPOLOGY_LADDER = (
    "d d~ > z",
    "u d~ > w+",
    "d d~ > z g",
    "d d~ > e- e+",
    "u d~ > e+ ve",
    "d d~ > z z",
    "d d~ > u u~",
    "d d~ > d d~",
    "g g > g g",
    "g g > t t~",
    "d d~ > t t~",
    "d d~ > z g g",
    "g g > t t~ g",
    "g g > g g g",
    "d d~ > z z z",
    "d d~ > e+ e- z h",
    "d d~ > t t~ z h",
    "d d~ > e+ e- e+ e-",
    "d d~ > u u~ s s~",
)


@pytest.fixture(scope="module")
def external_sm():
    compiled = compile_model_source(
        MODEL_ROOT / "sm.json",
        restriction=str((MODEL_ROOT / "restrict_default.json").resolve()),
        use_cache=False,
    )
    return compiled, CompiledUFOModel(compiled)


def test_external_sm_symmetries_are_proven_from_compiled_tensors(external_sm) -> None:
    compiled, model = external_sm
    certificates = model._symmetry_certificates

    assert certificates.yang_mills_adjoint_names == frozenset({"g"})
    assert len(certificates.yang_mills_kernel_kinds) == 4
    assert certificates.yang_mills_kernel_kinds <= certificates.parity_kernel_kinds
    reflection_phases = dict(certificates.adjoint_current_reflection_phases)
    assert reflection_phases
    assert set(reflection_phases.values()) == {(-1.0, 0.0)}

    pure_adjoint = build_model_process_ir("g g > g g", compiled.ir)
    qcd_vertices = tuple(
        vertex
        for vertex in model.vertices
        if vertex.kind in certificates.parity_kernel_kinds
    )
    yang_mills_vertices = tuple(
        vertex
        for vertex in model.vertices
        if vertex.kind in certificates.yang_mills_kernel_kinds
    )

    assert model.global_helicity_flip_equivalence_is_proven(qcd_vertices)
    assert model.pure_massless_adjoint_helicity_zero_rule_is_proven(
        pure_adjoint,
        yang_mills_vertices,
    )
    assert model.lc_trace_reflection_equivalence_is_proven(pure_adjoint)


@pytest.mark.parametrize(
    ("process", "topology", "forbidden_order"),
    (
        ("g g > t t~", (36, 44, 32), (2, 1)),
        ("d d~ > z g g", (117, 242, 48), (5, 4)),
    ),
)
def test_external_sm_recovers_builtin_lc_current_reuse(
    external_sm,
    process: str,
    topology: tuple[int, int, int],
    forbidden_order: tuple[int, int],
) -> None:
    compiled, external_model = external_sm
    builtin_dag = compile_generic_dag(
        build_process_ir(process),
        model=BuiltinSMModel(),
    )
    external_dag = compile_generic_dag(
        build_model_process_ir(process, compiled.ir),
        model=external_model,
    )

    for dag in (builtin_dag, external_dag):
        assert (
            len(dag.currents),
            len(dag.interactions),
            len(dag.amplitude_roots),
        ) == topology
        assert all(
            current.index.ordered_external_labels != forbidden_order
            for current in dag.currents
        )


@pytest.mark.parametrize("process", _EXTERNAL_SM_TOPOLOGY_LADDER)
def test_external_sm_matches_builtin_production_dag_topology(
    external_sm,
    process: str,
) -> None:
    """External SM must recover every built-in production DAG reduction."""

    compiled, external_model = external_sm
    builtin_model = BuiltinSMModel()
    builtin_process = build_process_ir(process)
    external_process = build_model_process_ir(process, compiled.ir)
    builtin_limits = infer_minimal_coupling_order_limits(
        builtin_process,
        model=builtin_model,
    )
    external_limits = infer_minimal_coupling_order_limits(
        external_process,
        model=external_model,
    )

    assert external_limits == builtin_limits

    builtin_dag = compile_generic_dag(
        builtin_process,
        model=builtin_model,
        max_coupling_orders=builtin_limits,
    )
    external_dag = compile_generic_dag(
        external_process,
        model=external_model,
        max_coupling_orders=external_limits,
    )

    def topology(dag) -> tuple[int, int, int, int]:
        return (
            len(dag.currents),
            len(dag.interactions),
            len(dag.amplitude_roots),
            len(dag.color_plan.sectors),
        )

    assert topology(external_dag) == topology(builtin_dag)
    # The external compiler may prove additional exact kernel equivalences,
    # but it may never lose a reuse relation available to the built-in model.
    assert (
        external_dag.interaction_evaluation_count
        <= builtin_dag.interaction_evaluation_count
    )


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_lc_current_reflection_reuse_does_not_prune_contracted_color_modes(
    external_sm,
    accuracy: str,
) -> None:
    compiled, external_model = external_sm
    cases = (
        (
            build_process_ir("d d~ > z g", color_accuracy=accuracy),
            BuiltinSMModel(),
        ),
        (
            build_model_process_ir(
                "d d~ > z g",
                compiled.ir,
                color_accuracy=accuracy,
            ),
            external_model,
        ),
    )

    for process, model in cases:
        limits = infer_minimal_coupling_order_limits(process, model=model)
        dag = compile_generic_dag(
            process,
            model=model,
            max_coupling_orders=limits,
        )
        assert (
            len(dag.currents),
            len(dag.interactions),
            len(dag.amplitude_roots),
        ) == (31, 34, 12)


def test_deformed_adjoint_kernel_disables_local_current_reuse(external_sm) -> None:
    compiled, model = external_sm
    reflection_kinds = dict(
        model._symmetry_certificates.adjoint_current_reflection_phases
    )
    target = next(
        kernel
        for kernel in compiled.ir.oriented_kernels
        if kernel.kind in reflection_kinds
    )
    deformed_kernels = tuple(
        replace(
            kernel,
            component_expressions=("1", *kernel.component_expressions[1:]),
        )
        if kernel.kind == target.kind
        else kernel
        for kernel in compiled.ir.oriented_kernels
    )
    certificates = derive_external_symmetry_certificates(
        replace(compiled.ir, oriented_kernels=deformed_kernels)
    )

    assert certificates.adjoint_current_reflection_phases == ()


def test_chiral_gauge_current_does_not_receive_parity_certificate(external_sm) -> None:
    compiled, _model = external_sm
    vectorlike = next(term for term in compiled.ir.vertex_terms if term.id == 75)
    chiral = next(
        term
        for term in compiled.ir.vertex_terms
        if "projm" in term.lorentz_expression.casefold()
    )
    deformed_terms = tuple(
        replace(vectorlike, lorentz_expression=chiral.lorentz_expression)
        if term.id == vectorlike.id
        else term
        for term in compiled.ir.vertex_terms
    )
    certificates = derive_external_symmetry_certificates(
        replace(compiled.ir, vertex_terms=deformed_terms)
    )
    affected_kinds = {
        kernel.kind
        for kernel in compiled.ir.oriented_kernels
        if vectorlike.id in kernel.term_ids
    }

    assert affected_kinds
    assert not (affected_kinds & certificates.parity_kernel_kinds)


def test_deformed_quartic_coupling_disables_yang_mills_theorems(external_sm) -> None:
    compiled, _model = external_sm
    quartic_terms = {
        term.coupling
        for term in compiled.ir.vertex_terms
        if term.id in {36, 37, 38}
    }
    assert len(quartic_terms) == 1
    quartic_name = next(iter(quartic_terms))
    deformed_couplings = tuple(
        replace(coupling, expression=f"2*({coupling.expression})")
        if coupling.name == quartic_name
        else coupling
        for coupling in compiled.ir.couplings
    )
    certificates = derive_external_symmetry_certificates(
        replace(compiled.ir, couplings=deformed_couplings)
    )

    assert certificates.yang_mills_adjoint_names == frozenset()
    assert certificates.yang_mills_kernel_kinds == frozenset()
