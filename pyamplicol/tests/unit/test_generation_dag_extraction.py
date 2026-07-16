# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.api import Generator, ProcessAlias, ProcessRequest, ProcessSet
from pyamplicol.api.errors import GenerationError
from pyamplicol.generation.dag_algorithms import infer_minimal_coupling_order_limits
from pyamplicol.generation.dag_types import ColorState, CurrentIndex
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def _index(*, chirality: int = 1, ordered: tuple[int, ...] = (1, 2)) -> CurrentIndex:
    return CurrentIndex(
        particle_id=1,
        external_mask=3,
        external_labels=(1, 2),
        ordered_external_labels=ordered,
        helicity_ancestry=3,
        chirality=chirality,
        spin_state=chirality,
        flavour_flow=(1,),
        quantum_number_flow=(("electric_charge", "-1/3"),),
        color_state=ColorState(
            accuracy="lc",
            sector_id=0,
            line_groups=(0,),
            basis_key=(1, 2),
        ),
        momentum_mask=3,
        coupling_orders=(("qed", 1),),
    )


def test_generation_current_identity_keeps_every_physics_field() -> None:
    reference = _index()
    assert reference == _index()
    assert reference != _index(chirality=-1)
    assert reference != _index(ordered=(2, 1))
    payload = reference.to_json_dict()
    assert payload["ordered_external_labels"] == [1, 2]
    assert payload["coupling_orders"] == [["QED", 1]]
    assert payload["color_state"]["basis_key"] == [1, 2]


def test_generation_plan_defers_dag_compilation() -> None:
    plan = Generator().plan("d d~ > z")
    process = plan.estimated_coverage["processes"][0]

    assert process["key"] == "d_dbar_to_z"
    assert process["dag_compilation_deferred"] is True
    assert "source_count" not in process
    assert plan.concrete_processes[0].expression == "d d~ > z"


def test_generation_plan_uses_production_alias_validation() -> None:
    request = ProcessRequest.parse("d d~ > z g", name="base")
    valid = ProcessSet(
        (request,),
        aliases=(
            ProcessAlias(
                name="permuted",
                process_name="base",
                particle_permutation=(0, 1, 3, 2),
            ),
        ),
    )

    plan = Generator().plan(valid)

    assert plan.estimated_coverage["alias_count"] == 1
    assert plan.concrete_processes == (request,)

    invalid = ProcessSet(
        (request,),
        aliases=(
            ProcessAlias(
                name="bad",
                process_name="base",
                particle_permutation=(0, 1, 2),
            ),
        ),
    )
    with pytest.raises(GenerationError, match="permutation has length"):
        Generator().plan(invalid)


def test_minimal_coupling_limits_zero_nonminimal_model_orders() -> None:
    limits = infer_minimal_coupling_order_limits(
        build_process_ir("d d~ > u u~"),
        model=BuiltinSMModel(),
    )

    assert limits == {"QCD": 2, "QED": 0}
