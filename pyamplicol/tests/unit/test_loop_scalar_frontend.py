# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.loops.scalar_frontend import ScalarFunctionLibrary, lower_scalar_program
from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

pytestmark = pytest.mark.loop_fast


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "scalar_helpers"
    path.mkdir()
    (path / "__init__.py").write_text(
        "from . import object_library, function_library, parameters\nfrom .object_library import all_functions, all_parameters, all_CTparameters\n"
    )
    (path / "object_library.py").write_text("""all_functions=[]
all_parameters=[]
all_CTparameters=[]
class Function:
 def __init__(self,name,arguments,expression):
  self.name=name;self.arguments=arguments;self.expr=expression;all_functions.append(self)
class Parameter:
 def __init__(self,name,value):
  self.name=name;self.value=value;all_parameters.append(self)
class CTParameter:
 def __init__(self,name,value):
  self.name=name;self.value=value;all_CTparameters.append(self)
""")
    (path / "function_library.py").write_text("""from .object_library import Function
reglog=Function(name='reglog',arguments=('z',),expression='0 if z == 0 else cmath.log(z)')
gate=Function(name='gate',arguments=('x','y'),expression='1 if x < 0 else (2 if y < 0 else 3)')
reglog=Function(name='reglog',arguments=('z'),expression='0 if z == 0 else (cmath.log(z)-2*cmath.pi*1j if z.real < 0 and z.imag > 0 else cmath.log(z))')
""")
    (
        path / "parameters.py"
    ).write_text("""from .object_library import Parameter, CTParameter
p=Parameter(name='p',value='0.123456789012345678901234567890123456789012345678901234567890')
a=CTParameter(name='A',value={-1:'x',0:'y',1:'z'})
b=CTParameter(name='B',value={0:'A**2'})
""")
    return path


def test_active_bindings_roundtrip_and_laurent(source):
    from symbolica import E

    raw = load_raw_loop_model(source, restriction="full")
    assert raw.metadata["function_bindings"]["reglog"] == 2
    assert not raw.collections["functions"][0]["fields"]["expression_source"]["exact"]
    copied = RawLoopModel.from_dict(json.loads(json.dumps(raw.to_dict())))
    program = lower_scalar_program(copied)
    assert len(program.library.helpers) == 2
    assert program.definitions[("counterterm_parameters", 1)].coefficient(0).value == E(
        "UFO::y^2+2*UFO::x*UFO::z"
    )
    assert program.definitions[("parameters", 0)].value.expression != E(
        str(float("0.12345678901234567890123456789"))
    )


@pytest.mark.parametrize(
    "change,match",
    [
        # A value written after its constructor: the listed refusal.
        ("gate.expr='0'", "ufo-value-writes"),
        (
            "gate=Function(name='gate',arguments=('long'),expression='long')",
            "argument string",
        ),
        (
            "gate=Function(name='gate',arguments=('x',),expression='x+unbound')",
            "free helper symbol",
        ),
        ("gate=Function(name='gate',arguments=('x',),expression='gate(x)')", "cycle"),
    ],
)
def test_invalid_helper_source_rejects(source, change, match):
    from graphlib import CycleError

    with (source / "function_library.py").open("a") as stream:
        stream.write(change + "\n")
    with pytest.raises((ValueError, CycleError), match=match):
        ScalarFunctionLibrary(load_raw_loop_model(source, restriction="full"))


def test_epsilon_helper_argument_is_not_silently_opaque(source):
    with (source / "parameters.py").open("a") as stream:
        stream.write("c=CTParameter(name='C',value={0:'gate(A,0)'})\n")
    with pytest.raises(NotImplementedError, match="epsilon-dependent"):
        lower_scalar_program(load_raw_loop_model(source, restriction="full"))


@pytest.mark.parametrize(
    "arithmetic,precision",
    [("double", None), ("double-double", None), ("arbitrary", 90)],
)
def test_native_helpers_conditions_and_precision(
    loop_native, source, arithmetic, precision
):
    native = loop_native
    from symbolica import S

    library = ScalarFunctionLibrary(load_raw_loop_model(source, restriction="full"))
    x, y = S("UFO::x", "UFO::y")
    expressions = tuple(
        library.lower(text)
        for text in (
            "x and y",
            "x or y",
            "gate(x,y)",
            "reglog(x+1j*y)",
            "1 < 1 + 1/2**240",
            "1 < 1 + 1/2**90",
            "1 if x == y else 0",
        )
    )
    options = {"arithmetic": arithmetic}
    if precision is not None:
        options["precision"] = precision
    kernel = native.CoefficientEvaluator(
        library.evaluator(expressions, (x, y)).save(), **options
    )
    rows = [("0", "2"), ("-1", "1"), ("-1", "-1"), ("1e-200", "0"), ("-1e200", "1e200")]
    batch = native.NativeBatch.from_decimal(
        [(v, "0") for row in rows for v in row], len(rows), 2, **options
    )
    result = kernel.evaluate_batch(batch)
    assert not result.nonfinite_rows()
    values = result.to_decimal()
    width = 2 * len(expressions)
    assert Decimal(values[0][0]) == 0  # and returns the first false operand
    assert Decimal(values[2][0]) == 2  # or returns the second operand
    assert Decimal(values[width][0]) == 1  # and returns y, not Boolean(x)
    assert Decimal(values[width + 2][0]) == -1
    assert Decimal(values[width + 6][1]) < 0  # active CMS reglog, not old principal log
    assert Decimal(values[8][0]) == (1 if arithmetic == "arbitrary" else 0)
    assert Decimal(values[10][0]) == (0 if arithmetic == "double" else 1)
    # Every execution remains native, including empty/changing batches.
    empty = native.NativeBatch.from_decimal([], 0, 2, **options)
    assert kernel.evaluate_batch(empty).rows == 0
    assert kernel.evaluate_batch(batch.take_rows([0])).rows == 1


@pytest.mark.parametrize(
    "arithmetic,precision",
    [("double", None), ("double-double", None), ("arbitrary", 70)],
)
def test_active_invalid_predicate_and_inactive_branch(
    loop_native, source, arithmetic, precision
):
    native = loop_native
    from symbolica import S

    library = ScalarFunctionLibrary(load_raw_loop_model(source, restriction="full"))
    x, y = S("UFO::x", "UFO::y")
    saved = library.evaluator((library.lower("gate(x,y)"),), (x, y)).save()
    # x<0 bypasses y<0 even inside a separately serialized helper body.
    values = [
        ("-1", "0"),
        ("nan", "0"),
        ("1", "0"),
        ("nan", "0"),
        ("1", "1"),
        ("0", "0"),
    ]
    options = {"arithmetic": arithmetic}
    if precision is not None:
        options["precision"] = precision
    batch = native.NativeBatch.from_decimal(values, 3, 2, **options)
    result = native.CoefficientEvaluator(saved, **options).evaluate_batch(batch)
    assert result.nonfinite_rows() == [1, 2]
    assert Decimal(result.to_decimal()[0][0]) == 1


def test_complete_gmu_scalar_source_coverage(loop_gmu_ufo):
    configured = str(loop_gmu_ufo)
    raw = load_raw_loop_model(Path(configured), restriction="full")
    program = lower_scalar_program(raw)
    # The UFOmodeldb issue-1 patch adds CKMc12, CKMc13, CKMc23, CKMs13e and
    # splits R2_G0tt_{d,s,b} into six couplings.
    assert len(program.definitions) == 201 + 107 + 1812
    assert len(program.library.helpers) == 16


def test_conditional_decided_by_its_bindings_forms_only_the_taken_branch(source):
    """Python's conditional is lazy: bound to an exact number (a fixed
    parameter), ``0 if m == 0 else mu**2/m**2`` is 0 at m = 0 (no 0^-2, an
    infinite coefficient no evaluator accepts) and the other branch
    otherwise; a free ``m`` keeps the native conditional."""

    library = ScalarFunctionLibrary(load_raw_loop_model(source, restriction="full"))
    text = "( 0 if m == 0 else mu**2/m**2 )"
    zero = library.lower(text, locals={"m": library.lower("complex(0,0)")})
    assert zero.value == library.zero
    half = library.lower(text, locals={"m": library.lower("complex(0.5,0)")})
    assert half.value == library.lower("4*mu**2").value
    free = library.lower(text).value
    assert free.get_name() == "symbolica::if"
    assert library.decided_number(library.lower("1 if 2 < 1 else 0").value) == 0


def test_namespace_alias_keeps_record_identity(source):
    with (source / "function_library.py").open("a") as stream:
        stream.write("alias=gate\n")
    raw = load_raw_loop_model(source, restriction="full")
    library = ScalarFunctionLibrary(raw)
    assert (
        raw.metadata["function_bindings"]["alias"]
        == raw.metadata["function_bindings"]["gate"]
    )
    assert library.lower("alias(x,y)").value == library.lower("gate(x,y)").value
    assert len(library.function_definitions) == 4


@pytest.mark.parametrize(
    "arithmetic,precision",
    [("double", None), ("double-double", None), ("arbitrary", 90)],
)
def test_native_component_access_preserves_extremes(
    loop_native, source, arithmetic, precision
):
    native = loop_native
    from symbolica import S

    library = ScalarFunctionLibrary(load_raw_loop_model(source, restriction="full"))
    x = S("UFO::x")
    expressions = tuple(
        library.lower(text) for text in ("x.real", "x.imag", "x.real < 0", "x.imag < 0")
    )
    saved = library.evaluator(expressions, (x,)).save()
    options = {"arithmetic": arithmetic}
    if precision is not None:
        options["precision"] = precision
    # Arithmetic projection through z+conj(z) would overflow these finite inputs.
    # Multiplying a subnormal by one half would also erase the smallest values.
    batch = native.NativeBatch.from_decimal(
        [("1.7976931348623157e308", "-1.7976931348623157e308"), ("5e-324", "-5e-324")],
        2,
        1,
        **options,
    )
    result = native.CoefficientEvaluator(saved, **options).evaluate_batch(batch)
    assert not result.nonfinite_rows()
    values, original = result.to_decimal(), batch.to_decimal()
    # One (value, validity) output pair per expression.
    width = 2 * len(expressions)
    for row in range(2):
        assert Decimal(values[width * row][0]) == Decimal(original[row][0])
        assert Decimal(values[width * row + 2][0]) == Decimal(original[row][1])
        assert Decimal(values[width * row + 4][0]) == 0
        assert Decimal(values[width * row + 6][0]) == 1
