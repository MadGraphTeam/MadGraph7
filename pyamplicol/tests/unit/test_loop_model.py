# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from fractions import Fraction
from graphlib import CycleError
from types import ModuleType, SimpleNamespace

import pytest

from pyamplicol.loops.model import CovariantKernelCatalogue, build_loop_model
from pyamplicol.loops.model_expressions import (
    epsilon_symbol,
    lower_scalar_definitions,
    parse_exact_ufo,
)
from pyamplicol.models.loop_import import capture_raw_loop_model, load_raw_loop_model

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def symbolic():
    return pytest.importorskip("symbolica")


def _scalars():
    module = ModuleType("laurent_fixture")
    module.all_parameters = [
        SimpleNamespace(
            name="x",
            value="0.123456789012345678901234567890123456789012345678901234567890",
        )
    ]
    module.all_CTparameters = [
        SimpleNamespace(name="A", value={-1: "a", 0: "b", 1: "c"}),
        SimpleNamespace(name="B", value={0: "A**2"}),
    ]
    module.all_couplings = [
        SimpleNamespace(name="GC", value={-1: "A", 0: "B"}, order={"QCD": 2})
    ]
    return module


def test_exact_decimal_complex_and_standard_functions(symbolic):
    digits = "0.123456789012345678901234567890123456789012345678901234567890"
    rational = Fraction(digits)
    exact = parse_exact_ufo(digits)
    assert exact == symbolic.E(f"{rational.numerator}/{rational.denominator}")
    rounded = Fraction(float(digits))
    assert exact != symbolic.E(f"{rounded.numerator}/{rounded.denominator}")
    imaginary = symbolic.E("1\U0001d456")
    assert parse_exact_ufo(f"complex(1,{digits})") == 1 + imaginary * exact
    assert parse_exact_ufo(f"{digits}j") == imaginary * exact
    assert parse_exact_ufo("cmath.sqrt (2.0)**2") == 2
    assert parse_exact_ufo("complexconjugate(complex(1,2))") == 1 - 2 * imaginary
    assert parse_exact_ufo("reglog(0)") == 0
    assert parse_exact_ufo("cond(0,2,3)") == 2
    assert parse_exact_ufo("cmath.log(1)") == 0
    assert parse_exact_ufo("1e-60") == symbolic.E("1/10^60")
    with pytest.raises(NotImplementedError):
        parse_exact_ufo("a if b else c")


def test_ct_substitution_precedes_laurent_extraction(symbolic):
    raw = capture_raw_loop_model(_scalars())
    values = lower_scalar_definitions(raw)
    coupling = values[("couplings", 0)].value
    # The +epsilon term in A contributes to the finite part of A squared and
    # of A/epsilon. Taking CT epsilon-zero pieces first loses both terms.
    assert coupling.coefficient(-2) == parse_exact_ufo("a+a**2")
    assert coupling.coefficient(-1) == parse_exact_ufo("b+2*a*b")
    assert coupling.coefficient(0) == parse_exact_ufo("c+b**2+2*a*c")
    assert coupling.coefficient(2) == parse_exact_ufo("c**2")
    assert coupling.series(0).get_absolute_order() == (1, 1)
    assert epsilon_symbol() in coupling.expression.get_all_symbols()
    assert values[("parameters", 0)].value.expression == parse_exact_ufo(
        _scalars().all_parameters[0].value
    )


def test_scalar_cycles_names_and_rounded_runtime_inputs_reject(symbolic):
    module = _scalars()
    module.all_CTparameters[0].value = {0: "B"}
    with pytest.raises(CycleError):
        lower_scalar_definitions(capture_raw_loop_model(module))
    module = _scalars()
    module.all_couplings[0].name = "A"
    with pytest.raises(ValueError, match="ambiguous UFO scalar"):
        lower_scalar_definitions(capture_raw_loop_model(module))
    module = _scalars()
    module.all_parameters[0].value = 0.1
    with pytest.raises(ValueError, match="exact input unavailable"):
        lower_scalar_definitions(capture_raw_loop_model(module))


def test_laurent_coupling_and_parameter_aliases_cannot_hide_poles(symbolic):
    module = _scalars()
    module.all_couplings = [
        SimpleNamespace(name="GC1", value={-1: "a", 0: "b"}),
        SimpleNamespace(name="GC2", value="GC1**2"),
    ]
    module.all_parameters.append(SimpleNamespace(name="alias", value="A"))
    module.all_couplings.append(SimpleNamespace(name="GC3", value="alias"))
    values = lower_scalar_definitions(capture_raw_loop_model(module))
    assert values[("couplings", 1)].value.coefficient(-2) == parse_exact_ufo("a**2")
    assert values[("couplings", 2)].value.coefficient(-1) == parse_exact_ufo("a")


def test_loop_model_ir_claims_no_formal_colour_continuation(symbolic):
    """Formal-N_c identities come per model from a LoopColorDeclaration
    (``parameter_identities``); the IR carries no default (QCD) names."""

    ir = build_loop_model(capture_raw_loop_model(_scalars()))
    assert not hasattr(ir, "color_identities")
    assert not ir.formal_nc_continuation_validated
    assert not ir.scheme_validated and not ir.completeness_validated
    assert not ir.arbitrary_precision_physics_validated


@pytest.fixture(scope="module")
def qcd_model(loop_sm_ufo, symbolic):
    if not symbolic.is_licensed():
        pytest.skip("native local Feynman rules require licensed Symbolica")
    source = str(loop_sm_ufo)
    from ufo_model_loader.model import Model

    ir = build_loop_model(load_raw_loop_model(source))
    # Normalize species/propagators without evaluating normalized scalar values.
    # This intentionally avoids the legacy loader's CT epsilon-zero evaluation.
    structural = Model.from_ufo_model(source)
    structural.wrap_indices_in_lorentz_structures()
    return ir, CovariantKernelCatalogue(ir, json.loads(structural.to_json()))


def test_real_qcd_basis_preserves_four_gluon_groups_and_uv_poles(qcd_model):
    ir, catalogue = qcd_model
    quartic = [entry for entry in ir.kernels if entry.vertex_name == "V_R2RGA"]
    assert len(quartic) == 57
    assert len({entry.key for entry in quartic}) == len(quartic)
    assert {entry.loop_group_index for entry in quartic} == {0, 1}
    assert {len(entry.loop_particles) for entry in quartic} == {1, 6}
    for entry in quartic:
        tensor = catalogue.lower(entry)
        assert tensor.color.rank == tensor.lorentz.rank == 4
        assert tensor.leg_half_edges == (0, 1, 2, 3)
        assert entry.coupling.value.coefficient(-1) == 0
    # The first active basis is Metric(1,4)*Metric(2,3). Materialize it through
    # the native tensor library and check a timelike and a spacelike component.
    local_tensor = catalogue.lower(quartic[0]).lorentz
    assert [str(axis.index) for axis in local_tensor.axes] == [
        "hedge(0,1)",
        "hedge(3,1)",
        "hedge(1,1)",
        "hedge(2,1)",
    ]
    components = local_tensor.permute_axes([0, 2, 3, 1]).to_tensor()
    assert components[[0, 0, 0, 0]] == 1
    assert components[[0, 1, 1, 0]] == -1
    assert components[[0, 1, 0, 1]] == 0
    mass = next(
        entry for entry in ir.counterterms("UV") if entry.vertex_name == "V_UVtMass"
    )
    assert mass.kind == "UVmass"
    assert mass.coupling.value.coefficient(-1) == parse_exact_ufo(
        "cond(MT,0,complex(0,1)*3*CF*MT*G**2/(16*cmath.pi**2))"
    )
    tensors = catalogue.lower(mass)
    assert tensors.color.rank == tensors.lorentz.rank == 2
    assert ir.eligibility == "requires-process-loop-group-matching"
    assert not ir.formal_nc_continuation_validated


def test_all_initial_qcd_components_lower_with_native_tensor_interfaces(qcd_model):
    ir, catalogue = qcd_model
    qcd = [
        entry
        for entry in ir.kernels
        if entry.order and all(name == "QCD" for name, _ in entry.order)
    ]
    assert qcd
    for entry in qcd:
        tensors = catalogue.lower(entry)
        assert len(tensors.leg_half_edges) == len(entry.particle_ids)
        assert "LOOP_LOCAL_ONE" not in tensors.lorentz.to_canonical_string()
        assert "LOOP_LOCAL_ONE" not in tensors.color.to_canonical_string()
        # All native series computations remain exact dependency operations.
        entry.coupling.value.series(0)


def test_real_gluon_two_point_trace_retains_two_adjoint_axes(qcd_model):
    from symbolica import E

    ir, catalogue = qcd_model
    entries = [entry for entry in ir.kernels if entry.vertex_name == "V_R2GG"]
    assert entries and all(entry.color == "Tr(1,2)" for entry in entries)
    for entry in entries:
        color = catalogue.lower(entry).color
        assert color.rank == 2
        components = color.to_tensor().to_dense()
        assert tuple(components.shape) == (8, 8)
        assert all(
            components[a, b] == E("1/2") * int(a == b)
            for a in range(8)
            for b in range(8)
        )


def test_quark_photon_r2_uses_exact_model_mixed_factor(qcd_model):
    ir, catalogue = qcd_model
    entry = next(entry for entry in ir.kernels if entry.vertex_name == "V_R2uuA")
    assert entry.coupling.name == "R2_UUA"
    assert entry.order == (("QCD", 2), ("QED", 1))
    assert entry.coupling.value.coefficient(-1) == 0
    expected = parse_exact_ufo(
        "(2*ee*complex(0,1)/3)*(-G**2*(1+lhv)*(Ncol**2-1)/(32*Ncol*cmath.pi**2))"
    )
    assert (entry.coupling.value.coefficient(0) - expected).expand() == 0
    tensors = catalogue.lower(entry)
    assert tensors.color.rank == 2 and tensors.lorentz.rank == 3
    assert entry.loop_particles == ((2, 12),)


def test_single_point_uv_and_duplicate_vertex_names_keep_distinct_identity(symbolic):
    if not symbolic.is_licensed():
        pytest.skip("native local Feynman rules require licensed Symbolica")
    from symbolica.community import hepkit as hep

    native = hep.Model.phi4()
    species = json.loads(native.to_json())["particles"][0]
    module = ModuleType("single_point_uv")
    particle = SimpleNamespace(**species)
    coefficient = SimpleNamespace(name="CT", value={-1: "1", 0: "2"}, order={"QCD": 2})
    lorentz = SimpleNamespace(name="S", spins=[1], structure="1")
    vertex = SimpleNamespace(
        name="duplicate",
        type="UV",
        particles=[particle],
        color=["1"],
        lorentz=[lorentz],
        loop_particles=[[[particle]]],
        couplings={(0, 0, 0): coefficient},
    )
    module.all_particles = [particle]
    module.all_couplings = [coefficient]
    module.all_lorentz = [lorentz]
    module.all_CTvertices = [vertex, SimpleNamespace(**vars(vertex))]
    ir = build_loop_model(capture_raw_loop_model(module))
    assert len({entry.key for entry in ir.kernels}) == 2
    tensor = CovariantKernelCatalogue(ir, native).lower(ir.kernels[0])
    assert tensor.color.rank == tensor.lorentz.rank == 0
    assert tensor.color.to_expression() == tensor.lorentz.to_expression() == 1
    assert tensor.leg_half_edges == (0,)
