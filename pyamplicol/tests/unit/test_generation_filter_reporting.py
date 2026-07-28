# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

import pyamplicol.generation.service as service_module
from pyamplicol.api import ProcessRequest
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    RunConfig,
)
from pyamplicol.generation.helicity_replay import (
    HELICITY_RECURRENCE_CONTRACT_VERSION,
)
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def test_generation_reports_structural_reduction_and_helicity_recurrence() -> None:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(GenerationConfig(), None)
    process_ir = build_process_ir("d d~ > z", color_accuracy="lc")
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    expanded = service_module._ExpandedProcess(
        request=ProcessRequest.parse("d d~ > z", name="ddbar_z"),
        process_ir=process_ir,
        aliases=(
            {
                "id": "ddbar_z_alias",
                "expression": "d d~ > z",
                "external_pdgs": [1, -1, 23],
                "external_permutation": [0, 1, 2],
            },
        ),
    )

    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(expanded, dag, coverage),
        model,
        index=0,
        phase=PhaseHandle("test", None, 1),
    )

    assert set(prepared.filters) == {
        "dynamic_color_projection",
        "structural_helicity_reduction",
        "helicity_recurrence",
        "lc_flow_layout",
    }
    dynamic_projection = prepared.filters["dynamic_color_projection"]
    assert isinstance(dynamic_projection, dict)
    assert dynamic_projection["equality_check_status"].startswith("passed-")
    structural = prepared.filters["structural_helicity_reduction"]
    assert isinstance(structural, dict)
    assert structural["mode"] == "proven global-helicity-flip equivalence"
    recurrence = prepared.filters["helicity_recurrence"]
    assert isinstance(recurrence, dict)
    assert recurrence["contract_version"] == HELICITY_RECURRENCE_CONTRACT_VERSION
    assert recurrence["residual_current_count"] == 0
    assert len(prepared.validation_points) == 10
    assert [point.seed for point in prepared.validation_points] == list(
        range(12345, 12355)
    )

    metadata_only_backend = service_module.GenerationBackend(
        GenerationConfig(
            validation=GenerationValidationConfig(enabled=False, samples=25)
        ),
        None,
    )
    metadata_only = metadata_only_backend._prepare_warmup_process(
        service_module._DagProcess(expanded, dag, coverage),
        model,
        index=0,
        phase=PhaseHandle("test-disabled", None, 1),
    )
    assert len(metadata_only.validation_points) == 1


def test_opt_in_relation_discovery_is_scoped_in_generation_filters() -> None:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(
        GenerationConfig(
            relation_discovery=GenerationRelationDiscoveryConfig(
                mode="diagnostic",
                precision_digits=80,
                probe_count=2,
                seed=31,
            )
        ),
        None,
    )
    process_ir = build_process_ir("d d~ > z", color_accuracy="lc")
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    discovery = coverage["relation_discovery"]
    assert isinstance(discovery, dict)
    assert discovery["state"] == "diagnostic-only"
    assert discovery["scope"] == {
        "execution_mode": "compiled",
        "color_accuracy": "lc",
        "representation": "generic-dag",
    }
    assert discovery["probe"] == {
        "status": "completed",
        "precision_digits": 80,
        "probe_count": 2,
        "seed": 31,
        "deterministic": True,
        "candidate_only": True,
    }

    expanded = service_module._ExpandedProcess(
        request=ProcessRequest.parse("d d~ > z", name="ddbar_z"),
        process_ir=process_ir,
        aliases=(),
    )
    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(expanded, dag, coverage),
        model,
        index=0,
        phase=PhaseHandle("test-discovery", None, 1),
    )
    assert prepared.filters["relation_discovery"] == discovery


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
@pytest.mark.parametrize("color_accuracy", ("lc", "nlc", "full"))
def test_relation_discovery_scope_covers_every_dag_mode_and_color(
    execution_mode,
    color_accuracy,
) -> None:
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=color_accuracy),
        generation=GenerationConfig(
            relation_discovery=GenerationRelationDiscoveryConfig(
                mode="diagnostic",
                precision_digits=80,
                probe_count=2,
            )
        ),
        evaluator=EvaluatorConfig(execution_mode=execution_mode),
    )
    backend = service_module.GenerationBackend(config, None)
    process_ir = build_process_ir(
        "d d~ > z",
        color_accuracy=color_accuracy,
    )
    _dag, coverage = backend._compile_concrete_process(
        process_ir,
        BuiltinSMModel(),
    )

    discovery = coverage["relation_discovery"]
    assert isinstance(discovery, dict)
    assert discovery["scope"] == {
        "execution_mode": execution_mode,
        "color_accuracy": color_accuracy,
        "representation": "generic-dag",
    }
