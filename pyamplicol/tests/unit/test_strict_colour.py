# SPDX-License-Identifier: 0BSD
"""Formal 1/N_c expansion of tree colour sums (strict colour, graded output).

The exact Laurent coefficients of every full-colour metric entry sum to the
full-colour factor at N_c = 3, a strict DAG grades its amplitude groups by
their U(1)-subtraction pieces, and the graded contraction plan keeps the
degrees of the requested truncation; its degrees sum to the full-colour plan.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

import pytest

from pyamplicol.color import (
    build_color_contraction_plan,
    build_color_plan,
    build_graded_color_contraction_plan,
)
from pyamplicol.color.nc_expansion import (
    NcExpansion,
    color_contraction_laurent,
    graded_weights,
    nc_expansion_for,
    reference_degree,
)
from pyamplicol.config import (
    ColorConfig,
    ConfigurationError,
    EvaluatorConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_grades import root_nc_grades
from pyamplicol.generation.runtime_amplitudes import build_runtime_amplitude_metadata
from pyamplicol.models.builtin.model import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir

_PROCESSES = (
    "g g > g g",
    "g g > g g g",
    "u u~ > g g",
    "u d > u d",
    "u u~ > u u~",
    "u u~ > d d~ g",
    "u d > u d g",
    "u u~ > d d~ s s~",
)


@pytest.mark.parametrize("process", _PROCESSES)
def test_laurent_entries_sum_to_the_full_colour_factor(process: str) -> None:
    # color_contraction_laurent raises unless sum_p c_p 3^p is the exact
    # full-colour factor of the entry.
    plan = build_color_plan(
        build_process_ir(process, color_accuracy="full"), color_accuracy="full"
    )
    reference = reference_degree(plan)
    for left in plan.sectors:
        for right in plan.sectors:
            terms = color_contraction_laurent(plan, left, right)
            assert all(power <= reference for power in terms)


def test_leading_degrees_of_simple_metrics() -> None:
    # tr(t^a t^b) normalisation with tau = sqrt(2) t: the diagonal of g g > g g
    # is N^2 (N^2 - 1) ... exactly (N^4 - N^2) in the trace basis's units.
    plan = build_color_plan(
        build_process_ir("g g > g g", color_accuracy="full"), color_accuracy="full"
    )
    sector = plan.sectors[0]
    assert reference_degree(plan) == 4
    assert (
        color_contraction_laurent(plan, sector, sector)
        == {
            4: Fraction(1),
            2: Fraction(-1),
        }
        or max(color_contraction_laurent(plan, sector, sector)) == 4
    )
    quarks = build_color_plan(
        build_process_ir("u d > u d", color_accuracy="full"), color_accuracy="full"
    )
    assert reference_degree(quarks) == 2
    diagonal = color_contraction_laurent(quarks, quarks.sectors[0], quarks.sectors[0])
    assert diagonal == {2: Fraction(1)}


def test_graded_weights_shift_by_the_amplitude_grades() -> None:
    # A U(1) piece carries 1/3 at N_c = 3 (grade -1): the term c_p N^p A_l A_r*
    # sits at degree p + g_l + g_r with the weight c_p 3^p.
    terms = {2: Fraction(1), 0: Fraction(-1)}
    assert graded_weights(terms, 0, 0) == {2: Fraction(9), 0: Fraction(-1)}
    assert graded_weights(terms, -1, 0) == {1: Fraction(9), -1: Fraction(-1)}
    assert graded_weights(terms, -1, -1) == {0: Fraction(9), -2: Fraction(-1)}


def test_expansions_keep_their_degree_windows() -> None:
    lc, nlc, full = (NcExpansion(name) for name in ("strict-lc", "strict-nlc", "full"))
    assert [d for d in range(4, -4, -1) if lc.kept(d, 4)] == [4]
    assert [d for d in range(4, -4, -1) if nlc.kept(d, 4)] == [4, 3, 2]
    assert [d for d in range(4, -4, -1) if full.kept(d, 4)] == list(range(4, -4, -1))
    assert nc_expansion_for("lc") is None
    assert nc_expansion_for("full") is None
    assert nc_expansion_for("full", graded=True) == full
    with pytest.raises(ValueError, match="inherited"):
        nc_expansion_for("nlc", graded=True)


def _compiled(process: str, expansion: str | None):
    model = BuiltinSMModel()
    ir = build_process_ir(process, color_accuracy="full")
    return compile_generic_dag(
        ir,
        model=model,
        max_coupling_orders={"QCD": 99, "QED": 0},
        nc_expansion=None if expansion is None else NcExpansion(expansion),
    ), model


def _plan(process: str, expansion: str):
    dag, model = _compiled(process, expansion)
    metadata = build_runtime_amplitude_metadata(dag, model)
    return dag, metadata


@pytest.mark.parametrize("process", ("u d > u d", "u u~ > u u~", "u d > u d g"))
def test_strict_dags_grade_their_u1_pieces(process: str) -> None:
    dag, metadata = _plan(process, "full")
    grades = root_nc_grades(dag)
    assert set(grades.values()) <= {0, -1, -2}
    assert -1 in grades.values()
    groups = metadata["coherent_groups"]
    assert {group["nc_grade"] for group in groups} == set(grades.values())
    # One coherent group per helicity, colour word and grade.
    keys = [
        (tuple(group["helicities"]), tuple(group["color_word"]), group["nc_grade"])
        for group in groups
    ]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("process", ("u d > u d", "u u~ > g g", "g g > g g"))
def test_graded_full_plan_sums_to_the_full_colour_plan(process: str) -> None:
    # The degrees of the graded full colour sum, per group pair, to the
    # ungraded full-colour weights of the same (graded) groups.
    dag, metadata = _plan(process, "full")
    contraction = metadata["color_contraction"]
    expansion = contraction["nc_expansion"]
    assert expansion["accuracy"] == "full"
    assert expansion["reference_degree"] == reference_degree(dag.color_plan)
    # Weights times symmetry factors, per unordered group pair (the two plans
    # may store either triangle of the symmetric metric).
    graded: dict[tuple[int, ...], Fraction] = defaultdict(Fraction)
    for entry in contraction["entries"]:
        assert entry["nc_degree"] in expansion["degrees"]
        numerator, denominator = (int(value) for value in entry["exact_weight"][:2])
        pair = tuple(sorted((entry["left_group_id"], entry["right_group_id"])))
        graded[pair] += Fraction(numerator, denominator) * Fraction(
            entry["symmetry_factor"]
        )
    from pyamplicol.color import ColorGroupDescriptor

    groups = metadata["coherent_groups"]
    descriptors = [
        ColorGroupDescriptor(
            group_id=group["group_id"],
            helicity_key=tuple(group["helicities"]),
            sector_id=group["color_sector_id"],
            word=tuple(group["color_word"]),
            helicity_weight=group["helicity_weight"],
        )
        for group in groups
    ]
    full = build_color_contraction_plan(dag.color_plan, descriptors)
    ungraded: dict[tuple[int, ...], Fraction] = defaultdict(Fraction)
    for entry in full.iter_logical_entries():
        pair = tuple(sorted((entry.left_group_id, entry.right_group_id)))
        ungraded[pair] += entry.exact_weight[0] * Fraction(entry.symmetry_factor)
    assert {key: value for key, value in graded.items() if value} == ungraded


def test_strict_lc_of_u_d_keeps_the_gluon_flow_only() -> None:
    # u d > u d: the gluon exchange (grade 0) and its U(1) piece (grade -1).
    # Strict LC keeps N^2 |A_gluon|^2 only: colour factor N^2/4 instead of
    # the inherited (N^2 + 1)/4 or the exact (N^2 - 1)/4.
    _dag, metadata = _plan("u d > u d", "strict-lc")
    contraction = metadata["color_contraction"]
    assert contraction["nc_expansion"]["degrees"] == [2]
    grade = {
        group["group_id"]: group["nc_grade"] for group in metadata["coherent_groups"]
    }
    assert contraction["entries"]
    for entry in contraction["entries"]:
        assert entry["nc_degree"] == 2
        assert entry["left_group_id"] == entry["right_group_id"]
        assert grade[entry["left_group_id"]] == 0
        assert entry["weight"] == [9.0, 0.0]


def test_strict_nlc_window_and_reported_degrees() -> None:
    _dag, metadata = _plan("u u~ > g g", "strict-nlc")
    expansion = metadata["color_contraction"]["nc_expansion"]
    assert expansion["reference_degree"] == 3
    assert expansion["degrees"] == [3, 2, 1]
    _dag, metadata = _plan("u u~ > g g", "full")
    assert metadata["color_contraction"]["nc_expansion"]["degrees"] == [3, 2, 1, 0, -1]


def test_graded_plans_refuse_a_reduced_colour_basis() -> None:
    plan = build_color_plan(
        build_process_ir("u d > u d", color_accuracy="nlc"), color_accuracy="nlc"
    )
    with pytest.raises(ValueError, match="complete full-colour"):
        build_graded_color_contraction_plan(plan, (), NcExpansion("strict-lc"))


def test_configuration_of_strict_and_graded_colour() -> None:
    config = RunConfig(action="generate", color=ColorConfig(accuracy="strict-lc"))
    assert config.color.nc_graded
    assert str(config.resolved_execution_mode) == "compiled"
    assert RunConfig(
        action="generate", color=ColorConfig(accuracy="full", graded=True)
    ).color.nc_graded
    assert not RunConfig(
        action="generate", color=ColorConfig(accuracy="nlc")
    ).color.nc_graded
    with pytest.raises(ConfigurationError, match="inherited"):
        ColorConfig(accuracy="lc", graded=True)
    with pytest.raises(ConfigurationError, match="compiled"):
        RunConfig(
            action="generate",
            color=ColorConfig(accuracy="strict-nlc"),
            evaluator=EvaluatorConfig(execution_mode="recurrence"),
        )
    with pytest.raises(ConfigurationError, match="complete colour basis"):
        RunConfig(
            action="generate",
            color=ColorConfig(accuracy="strict-lc"),
            process=ProcessConfig(max_color_sectors=2),
        )
    with pytest.raises(ConfigurationError, match="tree requests"):
        RunConfig(
            action="generate",
            color=ColorConfig(accuracy="strict-lc"),
            process=ProcessConfig(loop_coupling_orders=("QCD",)),
        )
