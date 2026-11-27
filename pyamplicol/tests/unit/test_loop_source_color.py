# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501
"""Exact semantic colour views; no physical model continuation is certified here."""

from __future__ import annotations

from decimal import Decimal

import pytest

from pyamplicol.loops.scalar_bindings import (
    NativeScalarBindings,
    prepare_scalar_bindings,
)
from pyamplicol.loops.source_color import prepare_source_color_scalars
from pyamplicol.models.loop_import import load_raw_loop_model

pytestmark = pytest.mark.loop_fast


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "colour_source"
    path.mkdir()
    (path / "__init__.py").write_text(
        "from . import object_library, function_library, parameters\n"
        "from .object_library import all_functions, all_parameters, all_CTparameters, all_couplings\n"
    )
    (path / "object_library.py").write_text("""all_functions=[]
all_parameters=[]
all_CTparameters=[]
all_couplings=[]
class Function:
 def __init__(self,name,arguments,expression):
  self.name=name;self.arguments=arguments;self.expr=expression;all_functions.append(self)
class Parameter:
 def __init__(self,name,nature,type,value):
  self.name=name;self.nature=nature;self.type=type;self.value=value;all_parameters.append(self)
class CTParameter:
 def __init__(self,name,value):
  self.name=name;self.value=value;all_CTparameters.append(self)
class Coupling:
 def __init__(self,name,value,order):
  self.name=name;self.value=value;self.order=order;all_couplings.append(self)
""")
    (path / "function_library.py").write_text("""from .object_library import Function
helper=Function(name='helper',arguments=('x',),expression='cmath.log(x)')
""")
    (
        path / "parameters.py"
    ).write_text("""from .object_library import Parameter, CTParameter, Coupling
a=Parameter(name='a',nature='external',type='real',value='1')
z=Parameter(name='z',nature='external',type='complex',value='complex(1,2)')
MU_R=Parameter(name='MU_R',nature='external',type='real',value='2')
external_colour=Parameter(name='external_colour',nature='external',type='real',value='3')
Ncol=Parameter(name='Ncol',nature='internal',type='real',value='3')
CA=Parameter(name='CA',nature='internal',type='real',value='Ncol')
CF=Parameter(name='CF',nature='internal',type='real',value='4/3')
TF=Parameter(name='TF',nature='internal',type='real',value='1/2')
guardedCA=Parameter(name='guardedCA',nature='internal',type='real',value='3 if z < 0 else 3')
safeCA=Parameter(name='safeCA',nature='internal',type='real',value='guardedCA if a < 0 else 3')
adjustableCA=Parameter(name='adjustableCA',nature='internal',type='real',value='3*a')
derived_mass=Parameter(name='derived_mass',nature='internal',type='real',value='CA*a')
bad=Parameter(name='bad',nature='internal',type='real',value='1 if z < 0 else 2')
safe=Parameter(name='safe',nature='internal',type='real',value='bad if MU_R > 3 else a')
zero=Parameter(name='zero',nature='internal',type='real',value='0 if z < 0 else 0')
conditional=Parameter(name='conditional',nature='internal',type='real',value='CF*safe if a > 0 else CA*safe')
colour_test=Parameter(name='colour_test',nature='internal',type='real',value='1 if CA > 2 else 0')
colour_error=Parameter(name='colour_error',nature='internal',type='real',value='bad if CA > 2 else 0')
opaque=Parameter(name='opaque',nature='internal',type='real',value='helper(CA)')
A=CTParameter(name='A',value={-1:'CF*a',0:'CA*helper(MU_R)'})
B=CTParameter(name='B',value={0:'A**2'})
GC=Coupling(name='GC',value='B+TF*safe',order={'QCD':2})
GCplain=Coupling(name='GCplain',value='a',order={'QCD':2})
GCfixed=Coupling(name='GCfixed',value='a-1',order={'QCD':2})
GCformal=Coupling(name='GCformal',value='CA-3',order={'QCD':2})
GCguard=Coupling(name='GCguard',value='zero',order={'QCD':2})
""")
    return load_raw_loop_model(path, restriction="full")


def identities(nc):
    from symbolica import E

    return {"Ncol": nc, "CA": nc, "CF": (nc**2 - 1) / (2 * nc), "TF": E("1/2")}


def prepare(source, outputs, **options):
    from symbolica import S

    nc = S("source_colour_test::Nc")
    return prepare_source_color_scalars(
        source, outputs, nc=nc, parameter_identities=identities(nc), **options
    )


def continuation(source):
    from symbolica import S

    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.source_color import SourceColorContinuation
    from pyamplicol.models.loop_declaration import LoopSchemeDeclaration

    nc = S("source_colour_contract::Nc")
    model = build_loop_model(
        source, scalar_program=prepare_scalar_bindings(source, (("a", 0),)).program
    )
    return SourceColorContinuation(
        model,
        nc,
        identities(nc),
        frozenset(),
        frozenset(),
        LoopSchemeDeclaration(
            "HV", "4d", "none", "explicit", "real", "none", "synthetic"
        ),
    )


def test_cold_formal_expressions_preserve_diagnostics_without_certifying_grading(
    source,
):
    from dataclasses import replace

    declared = continuation(source)
    outputs = (("colour_error", 0),)
    original = prepare_scalar_bindings(
        source,
        outputs,
        program=declared.model.scalar_program,
        semantic_parameters=declared.parameter_identities,
    )
    formal = declared.formal_scalars(outputs, program=declared.model.scalar_program)
    assert formal.expressions == original.expressions
    assert declared.nc in formal.expressions[0].diagnostic.get_all_symbols()
    with pytest.raises(NotImplementedError, match="Nc-dependent active diagnostic"):
        declared.scalars(outputs, program=declared.model.scalar_program)
    changed = replace(
        declared,
        parameter_identities={
            **declared.parameter_identities,
            "CA": declared.nc + (declared.nc - 3),
        },
    )
    other = changed.formal_scalars(outputs, program=declared.model.scalar_program)
    assert formal.expressions[0].diagnostic != other.expressions[0].diagnostic
    assert formal.expressions[0].diagnostic.replace(
        declared.nc, 3
    ) == other.expressions[0].diagnostic.replace(declared.nc, 3)


def test_continuation_checks_all_ordinary_values_and_keeps_diagnostics(source):
    declared = continuation(source)
    ordinary = prepare_scalar_bindings(source, (("safe", 0), ("safeCA", 0)))
    formal = declared.require_ordinary_scalars(ordinary)
    assert formal.grades == (0,)
    for actual, expected in zip(
        formal.formal.expressions, ordinary.expressions, strict=True
    ):
        assert actual.diagnostic == expected.diagnostic
    # A derived mass must not become fixed-Nc3 numerical data before projection.
    for output in ("derived_mass", "GC"):
        selected = prepare_scalar_bindings(
            source, ((output, 0),), program=ordinary.program
        )
        with pytest.raises(
            NotImplementedError, match=f"ordinary scalar.*{output}.*depends on Nc"
        ):
            declared.require_ordinary_scalars(selected)
    with pytest.raises(TypeError):
        declared.parameter_identities["CA"] = declared.nc + 1


def test_continuation_rejects_unowned_or_undeclared_structures(source):
    from dataclasses import replace

    declared = continuation(source)
    key = ("counterterm_vertices", 9, 0, 0, 0)
    with pytest.raises(NotImplementedError, match=r"selected source kernel.*9"):
        declared.require_kernels((key,))
    with pytest.raises(NotImplementedError, match=r"particle field.*QCD"):
        declared.require_fields(((0, "QCD", 0),))
    with pytest.raises(ValueError, match="no exact source kernel mapping"):
        declared.require_ordinary_kernels((("unexpected_renamed_rule", ()),), ())
    with pytest.raises(ValueError, match="do not belong"):
        replace(declared, kernel_keys=frozenset((key,)))
    with pytest.raises(TypeError, match="explicit loop scheme"):
        replace(declared, scheme=None)
    copied = type(source).from_dict(source.to_dict())
    other = prepare_scalar_bindings(copied, (("a", 0),))
    with pytest.raises(ValueError, match="source owners differ"):
        declared.require_ordinary_scalars(other)


def test_partial_ordinary_rule_mapping_requires_every_nonzero_entry(source):
    from dataclasses import replace

    from pyamplicol.loops.model import KernelBasisEntry

    declared = continuation(source)
    program = declared.model.scalar_program
    order = (("QCD", 2),)

    def entry(index, name):
        record = source.record("couplings", name)
        return KernelBasisEntry(
            "vertices",
            0,
            "two_term_rule",
            "tree",
            index,
            0,
            None,
            (),
            (),
            "1",
            0,
            "scalar",
            "1",
            None,
            program.definitions["couplings", record["id"]],
            order,
            program,
        )

    first = entry(0, "GCplain")
    for omitted_name, fixed, allowed in (
        ("GCfixed", {}, False),
        ("GCfixed", {"a": "1"}, True),
        # A physical-Nc3 zero cannot hide a formally nonzero colour term.
        ("GCformal", {}, False),
        # A zero value with an active diagnostic is also not removable.
        ("GCguard", {}, False),
    ):
        second = entry(1, omitted_name)
        current = replace(
            declared,
            model=replace(declared.model, kernels=(first, second)),
            kernel_keys=frozenset((first.key,)),
        )
        plan = prepare_scalar_bindings(
            source, (("GCplain", 0),), program=program, fixed_parameters=fixed
        )
        if allowed:
            current.require_ordinary_kernels(
                (("two_term_rule", order),), ("GCplain",), scalar_plan=plan
            )
        else:
            with pytest.raises(ValueError, match=r"kernel.*exact-zero proof"):
                current.require_ordinary_kernels(
                    (("two_term_rule", order),), ("GCplain",), scalar_plan=plan
                )


def test_scheme_tie_uses_shared_fixed_inputs_and_rejects_mutation(loop_native, source):
    from dataclasses import replace

    from symbolica import E

    declared = replace(continuation(source), scheme_parameter_values={"a": E("1")})
    owner = prepare_scalar_bindings(
        source, (("GCplain", 0),), program=declared.model.scalar_program
    )
    with pytest.raises(ValueError, match=r"scheme parameter.*must remain fixed"):
        declared.require_whole_rules((), owner)
    fixed = prepare_scalar_bindings(
        source, (("GCplain", 0),), program=owner.program, fixed_parameters={"a": "1"}
    )
    declared.require_whole_rules((), fixed)
    runtime = NativeScalarBindings(fixed)
    before = runtime.snapshot()
    with pytest.raises(ValueError, match="fixed"):
        runtime.update({"z": "3", "a": "2"})
    assert runtime.snapshot() is before


def test_exact_replacement_outputs_share_source_values_and_diagnostics(
    loop_native, source
):
    from symbolica import E, S

    nc = S("source_colour_test::Nc")
    plan = prepare(
        source,
        (("GCplain", 0),),
        exact_outputs={
            ("replacement", 0): nc * S("UFO::safe") ** 2 / S("symbolica::pi"),
            ("replacement", -1): E("0"),
        },
    )
    assert ("safe", 0) in plan.formal.outputs
    guarded = plan.formal.expressions[plan.formal.outputs.index(("safe", 0))]
    assert guarded.diagnostic != 0
    index = plan.formal.outputs.index(("replacement", 0))
    assert (
        plan.formal.expressions[index].value
        - nc * guarded.value**2 / S("symbolica::pi")
    ).expand() == 0
    owner = NativeScalarBindings(plan.coefficients)
    old = owner.snapshot()
    with pytest.raises(ArithmeticError, match="active diagnostics"):
        owner.update({"MU_R": "4"})
    assert owner.snapshot() is old
    owner.update({"MU_R": "4", "z": "2"})
    assert owner.revision == 1
    with pytest.raises(ValueError, match="unbound symbols"):
        prepare(
            source, (("GCplain", 0),), exact_outputs={("bad_exact", 0): S("hidden")}
        )
    # The old value's guard remains active even when no new coefficient uses it.
    guarded = prepare(
        source, (("zero", 0),), exact_outputs={("independent_replacement", 0): nc}
    )
    with pytest.raises(ArithmeticError, match="active diagnostics"):
        NativeScalarBindings(guarded.coefficients)
    NativeScalarBindings(guarded.coefficients, parameters={"z": "2"})


def test_exact_grades_keep_laurent_axes_and_owning_program(source):
    from symbolica import E

    outputs = (("GC", -2), ("GC", -1), ("GC", 0), ("conditional", 0))
    plan = prepare(source, outputs)
    assert plan.grades == (-2, -1, 0, 1, 2)
    assert plan.formal.outputs == outputs
    assert plan.coefficients.program is plan.formal.program
    assert plan.coefficients.external_names == plan.formal.external_names
    original = prepare_scalar_bindings(source, outputs, program=plan.formal.program)
    for output, expected in enumerate(plan.formal.expressions):
        reconstructed = sum(
            (
                plan.nc**grade * plan.coefficients.expressions[columns[output]].value
                for grade, columns in zip(plan.grades, plan.columns, strict=True)
            ),
            E("0"),
        )
        # Symbolica is the independent exact collection/substitution oracle.
        from pyamplicol.loops.physical_color import expand_color_conditionals

        expanded = expand_color_conditionals(expected.value, plan.nc)
        assert (reconstructed - expanded).together().expand() == 0
        assert (
            expected.value.replace(plan.nc, 3) - original.expressions[output].value
        ).together().expand() == 0
        for columns in plan.columns:
            scalar = plan.coefficients.expressions[columns[output]]
            assert scalar.diagnostic == expected.diagnostic
            assert plan.nc not in scalar.value.get_all_symbols()
            assert plan.coefficients.outputs[columns[output]][1] == outputs[output][1]
    assert source.record("parameters", "CF")["fields"]["value"]["expression"] == "4/3"


def test_zero_coefficients_do_not_erase_active_diagnostics(loop_native, source):
    plan = prepare(source, (("zero", 0),))
    assert plan.grades == (0,)
    assert plan.coefficients.expressions[0].value == 0
    with pytest.raises(ArithmeticError, match="active diagnostics"):
        NativeScalarBindings(plan.coefficients)
    runtime = NativeScalarBindings(plan.coefficients, parameters={"z": "2"})
    assert all(Decimal(component) == 0 for component in runtime.values.to_decimal()[0])


def test_semantic_replacement_preserves_its_own_active_diagnostic(loop_native, source):
    from symbolica import S

    nc = S("source_colour_test::Nc")
    original = prepare_scalar_bindings(source, (("guardedCA", 0), ("safeCA", 0)))
    plan = prepare_source_color_scalars(
        source,
        (("safeCA", 0),),
        nc=nc,
        parameter_identities={
            name: expression.value * nc / 3
            for (name, _), expression in zip(
                original.outputs, original.expressions, strict=True
            )
        },
        program=original.program,
    )
    runtime = NativeScalarBindings(plan.coefficients)
    before = runtime.values.to_decimal()
    with pytest.raises(ArithmeticError, match="active diagnostics"):
        runtime.update({"a": "-1"})
    assert runtime.revision == 0 and runtime.values.to_decimal() == before
    runtime.update({"z": "2", "a": "-1"})
    assert runtime.revision == 1 and runtime.values.to_decimal() == before


@pytest.mark.parametrize(
    "output,match",
    [
        ("colour_test", "Nc-dependent active diagnostic"),
        ("colour_error", "Nc-dependent active diagnostic"),
        ("opaque", "opaque.*finite Laurent"),
    ],
)
def test_unsupported_colour_dependence_is_explicit(source, output, match):
    with pytest.raises(NotImplementedError, match=match):
        prepare(source, ((output, 0),))


def test_semantic_identity_is_exact_explicit_and_update_safe(loop_native, source):
    from symbolica import E, S

    nc = S("source_colour_test::Nc")
    outputs = (("CA", 0),)
    for identities_, error, match in [
        ({"missing": nc}, ValueError, "ordinary parameter"),
        ({"GC": nc}, ValueError, "ordinary parameter"),
        ({"CA": E("3.0")}, ValueError, "floating"),
        ({"CA": nc + 1}, ValueError, "differs from source"),
        (
            {"CA": nc + (nc - 3) * S("unknown_colour_input")},
            ValueError,
            "unbound symbols",
        ),
        ({"adjustableCA": nc}, ValueError, "differs from source"),
        ({"external_colour": nc}, ValueError, "must be fixed"),
        ([("CA", nc)], TypeError, "mapping"),
    ]:
        with pytest.raises(error, match=match):
            prepare_source_color_scalars(
                source, outputs, nc=nc, parameter_identities=identities_
            )
    fixed = prepare_source_color_scalars(
        source,
        (("external_colour", 0),),
        nc=nc,
        parameter_identities={"external_colour": nc},
        fixed_parameters={"external_colour": "3"},
    )
    runtime = NativeScalarBindings(fixed.coefficients)
    assert fixed.grades == (1,)
    with pytest.raises(ValueError, match="fixed"):
        runtime.update({"external_colour": "4"})
