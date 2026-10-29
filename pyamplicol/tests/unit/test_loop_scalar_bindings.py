# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501
from __future__ import annotations

import cmath
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from pyamplicol.loops.scalar_bindings import (
    NativeScalarBindings,
    prepare_scalar_bindings,
)
from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

pytestmark = pytest.mark.loop_fast


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "binding_source"
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
reglog=Function(name='reglog',arguments=('z',),expression='0 if z == 0 else (cmath.log(z)-2*cmath.pi*1j if z.real < 0 and z.imag > 0 else cmath.log(z))')
""")
    (
        path / "parameters.py"
    ).write_text("""from .object_library import Parameter, CTParameter, Coupling
x=Parameter(name='x',nature='external',type='real',value='2')
z=Parameter(name='z',nature='external',type='complex',value='complex(1/3,1/7)')
tiny=Parameter(name='tiny',nature='external',type='real',value='1+1/2**90')
high=Parameter(name='high',nature='external',type='real',value='1+1/2**240')
k=Parameter(name='k',nature='internal',type='complex',value='reglog(z)')
y=Parameter(name='y',nature='internal',type='complex',value='cmath.sqrt(x)+k')
tiny_difference=Parameter(name='tiny_difference',nature='internal',type='real',value='tiny-1')
high_difference=Parameter(name='high_difference',nature='internal',type='real',value='high-1')
bad=Parameter(name='bad',nature='internal',type='real',value='1 if z < 0 else 2')
safe=Parameter(name='safe',nature='internal',type='complex',value='bad if x < 0 else y')
a=CTParameter(name='A',value={-1:'x',0:'z'})
b=CTParameter(name='B',value={0:'A**2'})
c=Coupling(name='GC',value='y+x',order={'QED':1})
""")
    return load_raw_loop_model(path, restriction="full")


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "2e-15"),
        ("double-double", None, "2e-30"),
        ("arbitrary", 90, "1e-80"),
    ],
)
def test_exact_defaults_derived_complex_values_and_laurent(
    loop_native, source, arithmetic, precision, tolerance
):
    outputs = (
        ("tiny_difference", 0),
        ("high_difference", 0),
        ("B", -2),
        ("B", -1),
        ("B", 0),
        ("GC", 0),
        ("safe", 0),
    )
    plan = prepare_scalar_bindings(source, outputs)
    runtime = NativeScalarBindings(plan, arithmetic=arithmetic, precision=precision)
    retained = runtime.values
    values = retained.to_decimal()
    with localcontext() as context:
        context.prec = 110
        expected_tiny = Decimal(2) ** -90 if arithmetic != "double" else Decimal(0)
        expected_high = Decimal(2) ** -240 if arithmetic == "arbitrary" else Decimal(0)
        assert abs(Decimal(values[0][0]) - expected_tiny) < Decimal(tolerance) * max(
            expected_tiny, Decimal("1e-100")
        )
        assert abs(Decimal(values[1][0]) - expected_high) < Decimal(tolerance) * max(
            expected_high, Decimal("1e-100")
        )
        expected = [
            (Decimal(4), Decimal(0)),
            (Decimal(4) / 3, Decimal(4) / 7),
            (Decimal(1) / 9 - Decimal(1) / 49, Decimal(2) / 21),
        ]
        for actual, target in zip(values[2:5], expected, strict=True):
            for value, reference in zip(actual, target, strict=True):
                assert abs(Decimal(value) - reference) < Decimal(tolerance)
    # Independent source-language oracle for the complex continuation.
    changed = runtime.update({"x": "4", "z": ("-1", "1")}).values.to_decimal()
    expected = 6 + cmath.log(-1 + 1j) - 2j * cmath.pi
    assert abs(complex(*map(float, changed[5])) - expected) < 3e-15
    assert retained.to_decimal() == values and runtime.revision == 1
    assert runtime.overrides == {"x": ("4", "0"), "z": ("-1", "1")}
    assert changed[6] == runtime.values.to_decimal()[6]


def test_failed_update_is_atomic_and_inactive_dependency_stays_inactive(
    loop_native, source
):
    runtime = NativeScalarBindings(prepare_scalar_bindings(source, (("safe", 0),)))
    previous = runtime.values.to_decimal()
    with pytest.raises(ArithmeticError, match="active diagnostics"):
        runtime.update({"x": "-1"})
    assert runtime.revision == 0 and runtime.overrides == {}
    assert runtime.values.to_decimal() == previous
    runtime.update({"z": "2"})
    assert runtime.revision == 1
    runtime.update({"x": "-1"})
    assert Decimal(runtime.values.to_decimal()[0][0]) == 2
    with pytest.raises(TypeError, match="decimal strings"):
        runtime.update({"x": 0.1})
    with pytest.raises(ValueError, match="imaginary part"):
        runtime.update({"x": ("1", "1e-1000")})
    with pytest.raises(ValueError, match="finite"):
        runtime.update({"z": "nan"})
    with pytest.raises(ValueError, match="unknown external"):
        runtime.update({"y": "1"})


@pytest.mark.parametrize(
    "definitions,match",
    [
        ({"x": {"kind": "expression", "expression": "3"}}, "distinct from source"),
        ({"field": {"kind": "expression", "expression": "field+1"}}, "terminal"),
        (
            {
                "field": {"kind": "expression", "expression": "other"},
                "other": {"kind": "expression", "expression": "1"},
            },
            "terminal",
        ),
        ({"field": {"kind": "laurent", "terms": [(0, "1"), (0, "2")]}}, "unique"),
        ({"field": {"kind": "laurent", "terms": [(0, 0.1)]}}, "exact strings"),
    ],
)
def test_anonymous_source_definition_restrictions(source, definitions, match):
    with pytest.raises(ValueError, match=match):
        prepare_scalar_bindings(
            source, (("field", 0),), anonymous_definitions=definitions
        )


def test_no_external_parameters_and_selector_validation(loop_native, tmp_path):
    path = tmp_path / "constant_source"
    path.mkdir()
    (path / "__init__.py").write_text(
        "from .object_library import all_parameters\nfrom . import parameters\n"
    )
    (path / "object_library.py").write_text("""all_parameters=[]
class Parameter:
 def __init__(self,name,nature,type,value):
  self.name=name;self.nature=nature;self.type=type;self.value=value;all_parameters.append(self)
""")
    (path / "parameters.py").write_text(
        "from .object_library import Parameter\nx=Parameter(name='x',nature='internal',type='real',value='1/3')\n"
    )
    raw = load_raw_loop_model(path, restriction="full")
    runtime = NativeScalarBindings(
        prepare_scalar_bindings(raw, (("x", 0),)), arithmetic="arbitrary", precision=90
    )
    assert runtime.values.rows == 1 and runtime.values.columns == 1
    with localcontext() as context:
        context.prec = 100
        assert abs(
            Decimal(runtime.values.to_decimal()[0][0]) - Decimal(1) / 3
        ) < Decimal("1e-88")
    with pytest.raises(ValueError, match="unknown scalar"):
        prepare_scalar_bindings(raw, (("missing", 0),))
    with pytest.raises(ValueError, match="nonempty"):
        prepare_scalar_bindings(raw, ())


def test_gmu_model_scalar_bindings(loop_gmu_ufo, loop_native):
    configured = str(loop_gmu_ufo)
    raw = load_raw_loop_model(Path(configured), restriction="full")
    names = ["MW", "sw", "cw", "ee"]
    names.extend(record["name"] for record in raw.collections["couplings"][:3])
    ct = raw.collections["counterterm_parameters"][0]["name"]
    outputs = (*((name, 0) for name in names), (ct, -1), (ct, 0))
    plan = prepare_scalar_bindings(raw, outputs)
    for arithmetic, precision in (
        ("double", None),
        ("double-double", None),
        ("arbitrary", 70),
    ):
        runtime = NativeScalarBindings(plan, arithmetic=arithmetic, precision=precision)
        assert runtime.values.rows == 1 and runtime.values.columns == len(outputs)
        assert not runtime.values.nonfinite_rows()


def test_explicit_override_bypasses_invalid_default_and_program_identity(
    loop_native, source
):
    from pyamplicol.loops.scalar_frontend import lower_scalar_program

    payload = source.to_dict()
    # Raw serializations retain exact source expressions independently of the
    # historical Python runtime default. Alter this deliberate diagnostic fixture.
    x = payload["collections"]["parameters"][0]
    x["fields"]["value"] = {
        "kind": "expression",
        "expression": "1 if complex(0,1) < 0 else 2",
    }
    changed = RawLoopModel.from_dict(payload)
    plan = prepare_scalar_bindings(changed, (("x", 0),))
    with pytest.raises(ArithmeticError, match="defaults and diagnostics"):
        NativeScalarBindings(plan)
    runtime = NativeScalarBindings(plan, parameters={"x": "3"})
    assert Decimal(runtime.values.to_decimal()[0][0]) == 3
    with pytest.raises(ValueError, match="different raw source"):
        prepare_scalar_bindings(
            source, (("x", 0),), program=lower_scalar_program(changed)
        )


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "2e-15"),
        ("double-double", None, "2e-30"),
        ("arbitrary", 90, "1e-80"),
    ],
)
def test_fixed_parameters_preserve_exact_laurent_values_and_are_not_adjustable(
    loop_native, source, arithmetic, precision, tolerance
):
    original = source.to_dict()
    with localcontext() as context:
        context.prec = 300
        tiny = str(Decimal(1) + Decimal(2) ** -90)
        high = str(Decimal(1) + Decimal(2) ** -240)
    fixed = {"x": "00.0", "z": ("3", "4"), "tiny": tiny, "high": high}
    outputs = (
        ("B", -2),
        ("B", -1),
        ("B", 0),
        ("tiny_difference", 0),
        ("high_difference", 0),
    )
    plan = prepare_scalar_bindings(source, outputs, fixed_parameters=fixed)
    assert plan.external_names == () and plan.input_symbols == ()
    assert plan.fixed_parameters["z"] == ("3", "4")
    runtime = NativeScalarBindings(plan, arithmetic=arithmetic, precision=precision)
    values = runtime.values.to_decimal()
    assert tuple(map(Decimal, values[0])) == (0, 0)
    assert tuple(map(Decimal, values[1])) == (0, 0)
    assert tuple(map(Decimal, values[2])) == (-7, 24)
    with localcontext() as context:
        context.prec = 110
        # Fixed constants are simplified exactly before execution, so these
        # differences survive even double. No source/card default is rounded first.
        for row, exponent in ((3, 90), (4, 240)):
            expected = Decimal(2) ** -exponent
            assert abs(Decimal(values[row][0]) - expected) < expected * Decimal(
                tolerance
            )
    with pytest.raises(ValueError, match="fixed during generation"):
        runtime.update({"x": "1"})
    assert runtime.revision == 0 and runtime.values.to_decimal() == values
    fixed["x"] = "5"
    assert plan.fixed_parameters["x"] == ("00.0", "0")
    with pytest.raises(TypeError):
        plan.fixed_parameters["x"] = ("2", "0")
    assert source.to_dict() == original


def test_fixed_parameters_preserve_active_diagnostics_and_reject_wrong_types(
    loop_native, source
):
    safe = prepare_scalar_bindings(source, (("safe", 0),), fixed_parameters={"x": "1"})
    NativeScalarBindings(safe)  # Invalid ordering of complex z is inactive.
    active = prepare_scalar_bindings(
        source, (("safe", 0),), fixed_parameters={"x": "-1"}
    )
    with pytest.raises(ArithmeticError, match="active diagnostics"):
        NativeScalarBindings(active)
    for values, error, message in (
        ({"x": ("1", "1e-1000")}, ValueError, "imaginary part"),
        ({"x": "nan"}, ValueError, "finite"),
        ({"x": 0.1}, TypeError, "decimal strings"),
        ({"y": "0"}, ValueError, "unknown external"),
    ):
        with pytest.raises(error, match=message):
            prepare_scalar_bindings(source, (("safe", 0),), fixed_parameters=values)


def test_retained_scalar_identity_has_one_concurrent_publication(source, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    from pyamplicol.loops.scalar_bindings import PreparedScalarBindings

    plan = prepare_scalar_bindings(source, (("safe", 0),))
    construction_started, second_started, release = Event(), Event(), Event()
    records = []

    def paused_construction(cls, original):
        assert original is plan
        record = object()
        records.append(record)
        if len(records) == 1:
            construction_started.set()
            assert release.wait(5)
        return record

    monkeypatch.setattr(
        PreparedScalarBindings, "from_plan", classmethod(paused_construction)
    )

    def second_reader():
        second_started.set()
        return plan.prepared

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(lambda: plan.prepared)
        assert construction_started.wait(5)
        second = pool.submit(second_reader)
        assert second_started.wait(5)
        try:
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()
        assert first.result(timeout=5) is second.result(timeout=5) is plan.prepared
    assert records == [plan.prepared]
