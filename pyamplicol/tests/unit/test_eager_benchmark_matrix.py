# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import argparse

import pytest

from tools.developer import eager_benchmark_matrix as matrix


def test_smoke_suite_selects_the_two_bounded_cases() -> None:
    selected = matrix._selected_cases("smoke", ())

    assert [case.key for case in selected] == ["dd_z_3g", "dd_3q_1g"]
    assert all(case.smoke for case in selected)


def test_explicit_case_selection_preserves_request_order() -> None:
    selected = matrix._selected_cases("milestone", ("dd_tt_3g", "dd_z_3g"))

    assert [case.key for case in selected] == ["dd_tt_3g", "dd_z_3g"]


def test_unknown_case_fails_closed() -> None:
    with pytest.raises(matrix.MatrixError, match="unknown process case"):
        matrix._selected_cases("smoke", ("not-a-case",))


def test_lc_workloads_choose_computed_nonzero_selectors() -> None:
    physics = {
        "color_components": [
            {"id": "flow:zero", "computed": False},
            {"id": "flow:chosen", "computed": True},
        ],
        "helicities": [
            {"id": "h:zero", "computed": True, "structural_zero": True},
            {"id": "h:chosen", "computed": True, "structural_zero": False},
        ],
    }

    assert matrix._workloads("lc", physics) == (
        {
            "name": "single-flow-helicity-sum",
            "selectors": {"color_flow": "flow:chosen"},
        },
        {
            "name": "all-flow-single-helicity",
            "selectors": {"helicity": "h:chosen"},
        },
    )


def test_contracted_color_has_one_summed_workload() -> None:
    assert matrix._workloads("nlc", {}) == ({"name": "summed", "selectors": {}},)
    assert matrix._workloads("full", {}) == ({"name": "summed", "selectors": {}},)


def test_generation_command_keeps_compiled_and_eager_modes_explicit() -> None:
    command = matrix._generation_command(
        matrix.Path("python"),
        process="d d~ > z g",
        artifact=matrix.Path("artifact"),
        model=matrix.Path("model.pyamplicol-model"),
        color="full",
        execution_mode="eager",
    )

    assert command[0] == "python"
    assert command[command.index("--execution-mode") + 1] == "eager"
    assert command[command.index("--color-accuracy") + 1] == "full"
    assert command[command.index("--jit-optimization-level") + 1] == "3"
    assert "--no-post-build-validation" in command


def test_profile_command_encodes_both_lc_selector_axes() -> None:
    command = matrix._profile_command(
        matrix.Path("python"),
        artifact=matrix.Path("artifact"),
        process_id="process",
        batch_size=1024,
        target_runtime=5.0,
        minimum_samples=5,
        selectors={"color_flow": "flow:1,2", "helicity": "h:-1,+1"},
    )

    assert command[command.index("--batch-size") + 1] == "1024"
    assert command[command.index("--color-flow") + 1] == "flow:1,2"
    assert command[command.index("--helicity") + 1] == "h:-1,+1"


def test_ufo_model_requires_source_and_pack(tmp_path: matrix.Path) -> None:
    arguments = argparse.Namespace(
        models=("ufo-sm",),
        builtin_pack=tmp_path / "builtin.pyamplicol-model",
        ufo_source=None,
        ufo_pack=None,
    )

    with pytest.raises(matrix.MatrixError, match="requires both"):
        matrix._model_specs(arguments)


def test_relative_difference_handles_zero_values() -> None:
    assert matrix._relative_difference(0j, 0j) == 0.0
    assert matrix._relative_difference(1 + 0j, 1 + 1.0e-13j) < 1.1e-13


def test_generation_gates_require_each_non_lc_cell_and_geometric_mean() -> None:
    records = [
        {
            "case": {"key": "process"},
            "model": "built-in",
            "color": color,
            "compiled_over_eager_generation": ratio,
        }
        for color, ratio in (("lc", 2.0), ("nlc", 20.0), ("full", 30.0))
    ]

    assert matrix._generation_gates(records) == {
        "nlc_full_each_at_least_10x": True,
        "lc_no_generation_regression": True,
        "per_process_geometric_mean_at_least_10x": True,
    }


def test_generation_gate_rejects_partial_or_slow_matrix() -> None:
    records = [
        {
            "case": {"key": "process"},
            "model": "built-in",
            "color": "nlc",
            "compiled_over_eager_generation": 9.99,
        }
    ]

    assert matrix._generation_gates(records) == {
        "nlc_full_each_at_least_10x": False,
        "lc_no_generation_regression": False,
        "per_process_geometric_mean_at_least_10x": False,
    }
