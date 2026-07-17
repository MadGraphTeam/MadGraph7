# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math

import pytest

import pyamplicol.models as models
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model, Vertex
from pyamplicol.models.builtin.model import BuiltinModel
from pyamplicol.models.loading import compile_model_source


def test_model_builtin_sm_preserves_production_tables_and_couplings() -> None:
    model = BuiltinSMModel()

    assert len(model.particles) == 24
    assert len(model.vertices) == 211
    assert model.sin_weak == 0.47143025548407230
    assert model.mass(23) == 91.188
    assert model.width(23) == 2.441404
    assert model.mass(24) == 80.419002445756163
    left, right = model.z_fermion_coupling(1)
    prefactor = 1.0 / (model.sin_weak * model.cos_weak)
    assert math.isclose(left, prefactor * (-0.5 + model.sin_weak**2 / 3.0))
    assert math.isclose(right, prefactor * (model.sin_weak**2 / 3.0))
    assert model.leading_color_factor([1, -1, 23, 21, 21]) == 27
    assert model.runtime_normalization_parameter_defaults() == {
        "normalization.alpha_s_me_check": model.alpha_s_me_check,
        "normalization.alpha_ew": model.alpha_ew,
    }


def test_model_builtin_lowering_metadata_uses_owned_symbol_names() -> None:
    model = BuiltinSMModel()

    tensor = model.vertex_lowering_rule(1)
    assert tensor.tensor_names == ("pyamplicol::two_gluon_to_tensor",)
    assert tensor.expression_head == "pyamplicol::two_gluon_to_tensor"
    assert model.vertex_lowering_rule(6).expression_head == ("quark_gluon_weyl_current")
    assert model.vertex_lowering_rule(999).backend == "unimplemented"
    assert model.propagator_lowering_rule(-21).description.endswith(
        "adjacent built-in-SM vertex kernels"
    )
    assert model.propagator_lowering_rule(125).description.endswith(
        "the built-in-SM model"
    )


def test_model_builtin_compiles_to_canonical_records_without_replacing_path() -> None:
    compiled = compile_model_source("built-in-sm", use_cache=False)

    assert compiled.name == "built-in-sm"
    assert len(compiled.ir.particles) == 38
    assert len(compiled.ir.vertex_terms) == 211
    assert compiled.ir.oriented_kernels == ()
    assert compiled.source["kind"] == "built-in-sm"
    assert compiled.source["source_name"] is None


def test_model_builtin_public_name_has_no_legacy_alias() -> None:
    assert models.BuiltinSMModel is BuiltinSMModel
    assert not hasattr(models, "AmplicolSMLeadingColorModel")


def test_shared_model_contract_has_no_builtin_sm_pdg_fallbacks() -> None:
    generic = Model(name="generic-model-contract")

    assert issubclass(BuiltinSMModel, BuiltinModel)
    assert not issubclass(BuiltinModel, BuiltinSMModel)
    with pytest.raises(NotImplementedError, match="massless adjoint-vector role"):
        generic.is_massless_adjoint_vector(21)
    with pytest.raises(NotImplementedError, match="fundamental colored-fermion role"):
        generic.is_fundamental_colored_fermion(1)
    with pytest.raises(NotImplementedError, match="propagator lowering"):
        generic.propagator_lowering_rule(21)
    assert not generic.global_helicity_flip_equivalence_is_proven(
        (Vertex(0, (1, -1, 21)),)
    )
    assert not generic.pure_massless_adjoint_helicity_zero_rule_is_proven(
        object(),
        (Vertex(0, (21, 21, 21)),),
    )


def test_global_helicity_flip_proof_is_owned_by_the_builtin_model() -> None:
    model = BuiltinSMModel()

    assert model.global_helicity_flip_equivalence_is_proven(
        (Vertex(0, (1, -1, 21)), Vertex(6, (1, 21, 1)))
    )
    assert not model.global_helicity_flip_equivalence_is_proven(
        (Vertex(10, (1, -1, 22)),)
    )
