# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pyamplicol._internal.versions import COMPILED_MODEL_SCHEMA_VERSION
from pyamplicol.api import (
    ColorFlow,
    CompiledModel,
    HelicityConfiguration,
    ModelError,
    ModelSource,
    PhysicsReduction,
    ProcessAlias,
    ProcessPhysics,
    ProcessRequest,
    ProcessSet,
    ReductionGroup,
    ResolvedEvaluation,
)


def test_model_source_kind_resolution_does_not_import_symbolica(tmp_path: Path) -> None:
    sys.modules.pop("symbolica", None)
    ufo = tmp_path / "ufo"
    ufo.mkdir()
    serialized = tmp_path / "model.json"
    serialized.write_text("{}", encoding="utf-8")
    compiled = tmp_path / "sm.pyamplicol-model.json"
    compiled.write_text("{}", encoding="utf-8")

    assert ModelSource.built_in_sm().kind == "built-in-sm"
    assert ModelSource.from_path(ufo).kind == "ufo"
    assert ModelSource.from_path(serialized).kind == "json"
    assert ModelSource.from_path(compiled).kind == "compiled"
    assert "symbolica" not in sys.modules


def test_model_source_validates_restriction_files(tmp_path: Path) -> None:
    ufo = tmp_path / "ufo"
    ufo.mkdir()
    restriction = ufo / "restrict_default.dat"
    restriction.write_text("BLOCK MASS\n", encoding="ascii")

    source = ModelSource.from_path(ufo, restriction="restrict_default.dat")

    assert source.restriction == restriction
    with pytest.raises(ModelError, match="does not exist"):
        ModelSource.from_path(ufo, restriction="restrict_missing.dat")


def test_public_compiled_model_uses_the_canonical_schema() -> None:
    compiled = ModelSource.built_in_sm().compile(use_cache=False)

    assert isinstance(compiled, CompiledModel)
    assert compiled.schema_version == COMPILED_MODEL_SCHEMA_VERSION == 6


def test_process_requests_normalize_and_sets_require_unique_names() -> None:
    request = ProcessRequest.parse("  d   d~   >  z g ", name="dd_zg")
    assert request.expression == "d d~ > z g"
    assert request.name == "dd_zg"
    processes = ProcessSet(
        (request,),
        aliases=(
            ProcessAlias(
                name="physical_alias",
                process_name="dd_zg",
                particle_permutation=(0, 1, 3, 2),
            ),
        ),
    )
    assert processes.aliases[0].process_name == "dd_zg"
    assert processes.aliases[0].particle_permutation == (0, 1, 3, 2)
    assert not hasattr(processes.aliases[0], "helicity_ids")
    assert not hasattr(processes.aliases[0], "color_flow_ids")
    with pytest.raises(ValueError, match="unique"):
        ProcessSet((request, request))
    with pytest.raises(ValueError, match="exactly one arrow"):
        ProcessRequest.parse("d > z > g")


def test_resolved_total_reproduces_fully_summed_values() -> None:
    resolved = ResolvedEvaluation(
        values=(
            ((1.0 + 0j, 2.0 + 0j), (3.0 + 0j, 4.0 + 0j)),
            ((5.0 + 0j, 6.0 + 0j), (7.0 + 0j, 8.0 + 0j)),
        ),
        helicity_ids=("h0", "h1"),
        color_ids=("c0", "c1"),
        accuracy="lc",
    )
    assert resolved.total() == (10.0 + 0j, 26.0 + 0j)


def test_process_physics_metadata_is_deeply_immutable() -> None:
    helicity = HelicityConfiguration("h0", 0, (1, -1), True, False, "h0", 1.0)
    flow = ColorFlow("c0", 0, (1, 2), True, "c0", 1.0)
    physics = ProcessPhysics(
        process_id="ddbar_z",
        process="d d~ > z",
        color_accuracy="lc",
        helicity_coverage="all",
        color_coverage="all",
        color_kind="flow",
        structural_zero_helicity_count=0,
        external_particles=(),
        helicities=(helicity,),
        color_flows=(flow,),
        contracted_color_components=(),
        reduction=PhysicsReduction(
            "lc-diagonal",
            (ReductionGroup("g0", "h0", "c0", ("h0",), ("c0",)),),
        ),
        model_parameters=(),
        selector_capabilities=("helicity", "color_flow"),
    )
    assert physics.helicity_ids == ("h0",)
    assert physics.color_ids == ("c0",)
    with pytest.raises(FrozenInstanceError):
        flow.id = "changed"  # type: ignore[misc]
