# SPDX-License-Identifier: 0BSD
"""Non-native checks for the certified DDM generation scope and seed boundary."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

from pyamplicol.api.errors import GenerationError
from pyamplicol.config import ColorConfig, EvaluatorConfig, ProcessConfig, RunConfig
from pyamplicol.generation.on_the_fly_seed import project_on_the_fly_process_seed_v1
from pyamplicol.generation.service import GenerationBackend
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.contracts import CompiledOrientedKernel
from pyamplicol.models.external_symmetries import _yang_mills_sector_is_closed
from tests.unit.test_on_the_fly_seed_projection import (
    _catalog,
    _gluon_process,
    _NormalizationModel,
)


class _UncertifiedAdjointModel(BuiltinSMModel):
    shared_single_trace_color_basis_is_proven = (
        Model.shared_single_trace_color_basis_is_proven
    )


def _backend(lane: str, basis: str = "adjoint", accuracy: str = "full"):
    return GenerationBackend(
        RunConfig(
            action="generate",
            process=ProcessConfig(
                coupling_order_policy="explicit",
                selected_source_helicities={"1": -1},
            ),
            color=ColorConfig(
                accuracy=accuracy,
                contraction="symmetric-group-fft",
                fft_basis=basis,
            ),
            evaluator=EvaluatorConfig(execution_mode=lane),
        ),
        None,
    )


@pytest.mark.parametrize("lane", ("recurrence", "on-the-fly"))
@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_builtin_certified_gluons_use_ddm_in_planning_and_construction(lane, accuracy):
    backend = _backend(lane, accuracy=accuracy)
    model = BuiltinSMModel()
    process = build_process_ir("g g > g g g", color_accuracy=accuracy)
    assert backend._fft_color_basis(process, model) == "adjoint"
    assert (
        backend._plan_concrete_process(process, model=model)["color_sector_count"] == 6
    )
    prepared = backend._prepare_process_construction(process, model)
    assert prepared.complete_color_plan.basis == "adjoint"
    assert prepared.restricted_color_plan.basis == "adjoint"
    assert prepared.complete_color_plan.sector_count == 6
    assert not prepared.complete_color_plan.trace_reflections_folded


@pytest.mark.parametrize("lane", ("recurrence", "on-the-fly"))
def test_adjoint_scope_requires_model_proof_not_external_adjoint_roles(lane):
    process = build_process_ir("g g > g g", color_accuracy="full")
    backend = _backend(lane)
    for model in (None, _UncertifiedAdjointModel()):
        with pytest.raises(GenerationError, match="certified pure Yang--Mills"):
            backend._fft_color_basis(process, model)


@pytest.mark.parametrize("process", ("d d~ > g g", "g g > g g z"))
def test_adjoint_scope_rejects_non_gluon_external_domains(process):
    with pytest.raises(GenerationError, match="certified pure Yang--Mills"):
        _backend("recurrence")._fft_color_basis(
            build_process_ir(process, color_accuracy="full"), BuiltinSMModel()
        )


def test_trace_scope_does_not_require_the_new_adjoint_certificate():
    assert (
        _backend("recurrence", basis="trace")._fft_color_basis(
            build_process_ir("d d~ > g g", color_accuracy="full"), None
        )
        == "trace"
    )


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_adjoint_otf_projection_changes_only_the_separate_colour_plan(accuracy):
    process = replace(_gluon_process(5), color_accuracy=accuracy)
    projections = {
        basis: project_on_the_fly_process_seed_v1(
            process,
            _catalog(),
            cast(Model, _NormalizationModel()),
            coupling_order_policy="minimal",
            coupling_order_limits={"QCD": 3},
            color_basis=basis,
        )
        for basis in ("trace", "adjoint")
    }
    trace = projections["trace"]
    adjoint = projections["adjoint"]
    assert trace.color_plan is not None and adjoint.color_plan is not None
    assert trace.color_plan.sector_count == 24
    assert adjoint.color_plan.sector_count == 6
    assert adjoint.color_plan.basis == "adjoint"
    assert adjoint.seed.to_json_bytes() == trace.seed.to_json_bytes()
    assert adjoint.runtime_normalization == trace.runtime_normalization
    assert all(len(leg.source_states) == 2 for leg in adjoint.seed.external_sources)


def test_adjoint_otf_projection_canonicalizes_a_valid_cyclic_reference():
    projection = project_on_the_fly_process_seed_v1(
        replace(_gluon_process(4), color_accuracy="full"),
        _catalog(),
        cast(Model, _NormalizationModel()),
        coupling_order_policy="minimal",
        coupling_order_limits={},
        color_basis="adjoint",
        reference_color_order=(3, 2, 4, 1),
    )
    assert projection.color_plan is not None
    assert projection.color_plan.sectors[0].trace_labels == (1, 3, 2, 4)


def test_existing_yang_mills_certificate_excludes_reachable_singlet_exchange():
    # This closure check is part of the existing model certificate. Merely
    # recognizing ggg and gggg is insufficient when gg can also produce H.
    def kernels(*rows):
        return cast(
            tuple[CompiledOrientedKernel, ...],
            tuple(
                SimpleNamespace(kind=kind, particles=particles)
                for kind, particles in rows
            ),
        )

    standard = kernels(
        (1, ("g", "g", "g")), (2, ("g", "g", "aux")), (3, ("aux", "g", "g"))
    )
    certified = frozenset({1, 2, 3})
    assert _yang_mills_sector_is_closed(
        "g", standard, yang_mills_kernel_kinds=certified
    )
    assert not _yang_mills_sector_is_closed(
        "g",
        (*standard, *kernels((4, ("g", "g", "H")))),
        yang_mills_kernel_kinds=certified,
    )
    # A quark pair cannot be produced from purely gluonic tree currents.
    assert _yang_mills_sector_is_closed(
        "g",
        (*standard, *kernels((4, ("q", "q~", "g")))),
        yang_mills_kernel_kinds=certified,
    )
