# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol import (
    CompiledModel,
    Generator,
    ModelError,
    ModelSource,
    ProcessSet,
    Runtime,
)
from pyamplicol._internal.sm_heft import (
    CANONICAL_SM_HEFT_SOURCE,
    SM_HEFT_ALIASES,
    extend_sm_model_payload,
    packaged_sm_source_path,
    source_digest,
)
from pyamplicol.api.models import _compiled_model_payload
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    ModelConfig,
    ProcessConfig,
    RunConfig,
)


@pytest.fixture(scope="module")
def compiled_sm_heft() -> CompiledModel:
    return ModelSource.built_in_sm_heft().compile(use_cache=False)


def _named_records(
    payload: Mapping[str, object], field: str
) -> dict[str, Mapping[str, object]]:
    records = payload[field]
    assert isinstance(records, list)
    return {
        str(record["name"]): record for record in records if isinstance(record, Mapping)
    }


def test_packaged_sm_heft_has_canonical_aliases() -> None:
    assert CANONICAL_SM_HEFT_SOURCE in SM_HEFT_ALIASES


def test_sm_heft_extension_is_independent_and_has_constant_coupling() -> None:
    base = json.loads(packaged_sm_source_path().read_text(encoding="utf-8"))
    extended = extend_sm_model_payload(base)

    assert extended is not base
    assert extended["name"] == CANONICAL_SM_HEFT_SOURCE
    assert "GH" not in _named_records(base, "parameters")
    assert not any(
        name.startswith("V_HEFT") for name in _named_records(base, "vertex_rules")
    )

    parameters = _named_records(extended, "parameters")
    couplings = _named_records(extended, "couplings")
    gh_expression = str(parameters["GH"]["expression"])
    assert "P(" not in gh_expression
    assert "Momentum(" not in gh_expression
    assert {name for name in couplings if name.startswith("GC_HEFT_")} == {
        "GC_HEFT_HGG",
        "GC_HEFT_HGGG",
        "GC_HEFT_HGGGG",
    }
    assert all(
        "P(" not in str(couplings[name]["expression"])
        for name in ("GC_HEFT_HGG", "GC_HEFT_HGGG", "GC_HEFT_HGGGG")
    )


def test_compiled_sm_heft_contains_all_scalar_gluon_vertices(
    compiled_sm_heft: CompiledModel,
) -> None:
    private = _compiled_model_payload(compiled_sm_heft)
    heft_terms = tuple(
        term for term in private.ir.vertex_terms if term.vertex.startswith("V_HEFT_")
    )

    assert compiled_sm_heft.name == CANONICAL_SM_HEFT_SOURCE
    assert compiled_sm_heft.source.kind == "built-in-sm-heft"
    assert compiled_sm_heft.supported
    assert compiled_sm_heft.capabilities.form_factor_count == 0
    assert compiled_sm_heft.capabilities.maximum_valence == 5
    observed_terms = [
        (term.vertex, term.color_index, term.lorentz_index) for term in heft_terms
    ]
    assert observed_terms == [
        ("V_HEFT_HGG", 0, 0),
        ("V_HEFT_HGGG", 0, 0),
        ("V_HEFT_HGGGG", 0, 0),
        ("V_HEFT_HGGGG", 1, 1),
        ("V_HEFT_HGGGG", 2, 2),
    ]
    contact_ids = {term.id for term in heft_terms if len(term.particles) > 3}
    lowered_ids = {
        term_id
        for kernel in private.ir.oriented_kernels
        if "::contact-" in kernel.vertex and "final" in kernel.vertex
        for term_id in (kernel.term_ids or (kernel.term_id,))
    }
    assert contact_ids <= lowered_ids


def test_sm_heft_extension_rejects_record_collisions() -> None:
    base = json.loads(packaged_sm_source_path().read_text(encoding="utf-8"))
    base["parameters"].append({"name": "GH"})

    with pytest.raises(ValueError, match="already defines HEFT parameters"):
        extend_sm_model_payload(base)


def test_default_restriction_is_the_plain_selector() -> None:
    plain = ModelSource.built_in_sm_heft()

    assert plain.restriction is None
    assert ModelSource.built_in_sm_heft(restriction="default") == plain
    for config in (
        ModelConfig(source="built-in-sm-heft"),
        ModelConfig(source="sm_heft", restriction="default"),
    ):
        assert ModelSource.from_config(config) == plain
    assert source_digest() == source_digest("default")


@pytest.mark.parametrize(
    "restriction",
    (
        "no_such_card",
        "restrict_no_b_mass.json",
        "../sm/restrict_no_b_mass",
        packaged_sm_source_path().with_name("restrict_no_b_mass.json"),
    ),
)
def test_sm_heft_restrictions_name_packaged_cards(restriction: str | Path) -> None:
    with pytest.raises(ModelError, match="restrict"):
        ModelSource(kind="built-in-sm-heft", restriction=restriction)


def _b_mass(compiled: CompiledModel) -> tuple[object, object]:
    parameters = {
        parameter.name: parameter
        for parameter in _compiled_model_payload(compiled).ir.parameters
    }
    return parameters["MB"].nature, parameters["MB"].value


def test_restricted_sm_heft_has_its_own_identity_and_a_massless_b(
    compiled_sm_heft: CompiledModel,
) -> None:
    restricted = ModelSource.from_config(
        ModelConfig(source="built-in-sm-heft", restriction="no_b_mass")
    ).compile(use_cache=False)

    assert restricted.name == CANONICAL_SM_HEFT_SOURCE
    assert restricted.source.kind == "built-in-sm-heft"
    assert compiled_sm_heft.source.restriction == "default"
    assert restricted.source.restriction == "no_b_mass"
    assert compiled_sm_heft.source.digest == source_digest()
    assert restricted.source.digest == source_digest("no_b_mass") != source_digest()
    assert _b_mass(compiled_sm_heft) == ("external", (4.7, 0.0))
    assert _b_mass(restricted) == ("internal", (0.0, 0.0))


def _heft_config(execution_mode: str = "compiled") -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full"),
        evaluator=EvaluatorConfig(backend="jit", execution_mode=execution_mode),
        generation=GenerationConfig(workers=1, emit_api_bundle=False),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": 1, "QED": 0, "HIG": 1},
        ),
    )


def test_no_b_mass_heft_b_quark_scatters_like_a_light_quark(tmp_path: Path) -> None:
    """Without its mass and Yukawa coupling, the b couples to H only through
    the effective gluon vertex, like the d quark: b g > H b = d g > H d."""

    output = tmp_path / "heft_no_b_mass"
    Generator(_heft_config()).generate(
        ProcessSet.from_expressions(
            ("b g > h b", "d g > h d"), names=("bg_hb", "dg_hd")
        ),
        output,
        model=ModelSource.built_in_sm_heft(restriction="no_b_mass"),
    )
    manifest = json.loads((output / "artifact.json").read_text(encoding="utf-8"))
    assert manifest["model"]["restriction"] == "no_b_mass"
    assert manifest["model"]["content_sha256"] == source_digest("no_b_mass")
    # sqrt(s) = 1000 GeV, m_H = 125 GeV: |p| = 492.1875 along two exact directions.
    beams = (
        (Decimal(500), Decimal(0), Decimal(0), Decimal(500)),
        (Decimal(500), Decimal(0), Decimal(0), Decimal(-500)),
    )
    points = tuple(
        (
            *beams,
            (Decimal("507.8125"), -x, -y, -z),
            (Decimal("492.1875"), x, y, z),
        )
        for x, y, z in (
            (Decimal("295.3125"), Decimal(0), Decimal("393.75")),
            (Decimal("-236.25"), Decimal("177.1875"), Decimal("-393.75")),
            (Decimal(0), Decimal("492.1875"), Decimal(0)),
        )
    )
    b, d = (
        Runtime.load(output, process=name).evaluate(points, precision=61)
        for name in ("bg_hb", "dg_hd")
    )
    for b_value, d_value in zip(b, d, strict=True):
        assert abs(Decimal(b_value) / Decimal(d_value) - 1) <= Decimal("1e-30")


def test_prepared_execution_refuses_a_restricted_sm_heft(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="uses the default restriction"):
        Generator(_heft_config("recurrence")).generate(
            ProcessSet.from_expressions(("g g > h",), names=("gg_h",)),
            tmp_path / "heft_recurrence",
            model=ModelSource.built_in_sm_heft(restriction="no_b_mass"),
        )
