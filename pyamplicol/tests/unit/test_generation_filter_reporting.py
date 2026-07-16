# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pyamplicol.generation.service as service_module
from pyamplicol.api import ProcessRequest
from pyamplicol.config import GenerationConfig, GenerationValidationConfig
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.models import BuiltinSMModel
from pyamplicol.processes.ir import build_process_ir


def test_generation_reports_only_structural_reduction() -> None:
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

    assert set(prepared.filters) == {"structural_helicity_reduction"}
    structural = prepared.filters["structural_helicity_reduction"]
    assert isinstance(structural, dict)
    assert structural["mode"] == "proven global-helicity-flip equivalence"
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
