# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from pathlib import Path
from types import MappingProxyType

import pytest

from pyamplicol.loops.scalar_bindings import (
    NativeScalarBindings,
    NativeScaleBindings,
    prepare_scalar_bindings,
)
from pyamplicol.models.loop_import import load_raw_loop_model


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "scale_source"
    path.mkdir()
    (path / "__init__.py").write_text(
        "from . import object_library, function_library, parameters\n"
        "from .object_library import all_functions, all_parameters, all_CTparameters\n"
    )
    (path / "object_library.py").write_text("""all_functions=[]
all_parameters=[]
all_CTparameters=[]
class Function:
 def __init__(self,name,arguments,expression):
  self.name=name;self.arguments=arguments;self.expr=expression;all_functions.append(self)
class Parameter:
 def __init__(self,name,nature,type,value):
  self.name=name;self.nature=nature;self.type=type;self.value=value;all_parameters.append(self)
class CTParameter:
 def __init__(self,name,value):
  self.name=name;self.value=value;all_CTparameters.append(self)
""")
    (path / "function_library.py").write_text(
        "from .object_library import Function\n"
        "check=Function(name='check',arguments=('z','s'),"
        "expression='(1 if z < 0 else 2) if s < 1 else 0')\n"
    )
    (
        path / "parameters.py"
    ).write_text("""from .object_library import Parameter, CTParameter
mu=Parameter(name='MU_R',nature='external',type='real',value='5')
x=Parameter(name='x',nature='external',type='real',value='2')
z=Parameter(name='z',nature='external',type='complex',value='complex(1/3,2/7)')
a=Parameter(name='alpha',nature='external',type='real',value='0.118')
log=Parameter(name='L',nature='internal',type='real',value='cmath.log(MU_R**2/x**2)')
safe=Parameter(name='safe',nature='internal',type='real',value='check(z,MU_R)')
delta=Parameter(name='delta',nature='internal',type='real',value='MU_R-1')
ct=CTParameter(name='A',value={-1:'z',0:'z*L'})
sq=CTParameter(name='B',value={0:'A**2'})
""")
    return load_raw_loop_model(path, restriction="full")


MODES = (
    ("double", None, "3e-14"),
    ("double-double", None, "3e-29"),
    ("arbitrary", 90, "1e-84"),
)


@pytest.mark.parametrize("arithmetic,precision,tolerance", MODES)
def test_scale_view_laurent_complex_globals_updates_and_tails(
    source, arithmetic, precision, tolerance
):
    backend = pytest.importorskip("_loop_reduction")
    options = {"arithmetic": arithmetic, "precision": precision}
    global_plan = prepare_scalar_bindings(source, (("z", 0), ("MU_R", 0)))
    state = NativeScalarBindings(global_plan, **options)
    original = state.snapshot()
    outputs = (
        ("MU_R", 0),
        ("alpha", 0),
        ("A", -1),
        ("A", 0),
        ("B", -2),
        ("B", -1),
        ("B", 0),
        ("safe", 0),
    )
    plan = prepare_scalar_bindings(source, outputs, program=global_plan.program)
    view = NativeScaleBindings(state, plan)
    with localcontext() as context:
        context.prec = 120
        for size in (1, 17, 3, 0, 1):
            scale_values = [str((index % 3 + 1) * 4) for index in range(size)]
            ids = [101 + 3 * index for index in range(size)]
            scales = backend.NativeBatch.from_decimal(
                [(value, "0") for value in scale_values],
                size,
                1,
                row_ids=ids,
                **options,
            )
            result = view.evaluate_native(scales)
            assert (result.rows, result.columns, result.row_ids) == (size, 8, ids)
            assert result.precision_bits == original.values.precision_bits
            decimal = result.to_decimal()
            for index, scale in enumerate(scale_values):
                logarithm = (Decimal(scale) / 4).ln()
                z = (Decimal(1) / 3, Decimal(2) / 7)
                z2 = (z[0] ** 2 - z[1] ** 2, 2 * z[0] * z[1])
                expected = (
                    (Decimal(scale).sqrt(), 0),
                    (Decimal(".118"), 0),
                    z,
                    tuple(c * logarithm for c in z),
                    z2,
                    tuple(c * 2 * logarithm for c in z2),
                    tuple(c * logarithm**2 for c in z2),
                    (0, 0),
                )
                for actual, target in zip(
                    decimal[8 * index : 8 * (index + 1)], expected, strict=True
                ):
                    for component, reference in zip(actual, target, strict=True):
                        assert abs(Decimal(component) - reference) < Decimal(tolerance)
        assert state.snapshot() is original
        changed = state.update({"z": ("3", "4"), "x": "4", "MU_R": "7"})
        assert state.snapshot().revision == 1
        assert original.revision == 0 and original.overrides == {}
        assert tuple(map(Decimal, original.values.to_decimal()[1])) == (5, 0)
        assert tuple(map(Decimal, changed.to_decimal()[1])) == (7, 0)
        assert tuple(map(Decimal, state.inputs.to_decimal()[0])) == (7, 0)
        scales = backend.NativeBatch.from_decimal(
            [("16", "0")], 1, 1, row_ids=[99], **options
        )
        result = view.evaluate_native(scales)
        assert result.row_ids == [99]
        assert tuple(map(Decimal, result.to_decimal()[0])) == (4, 0)
        assert tuple(map(Decimal, result.to_decimal()[2])) == (3, 4)
        assert tuple(map(Decimal, result.to_decimal()[3])) == (0, 0)
        # Selected source values do not cause implicit alpha running.
        assert abs(Decimal(result.to_decimal()[1][0]) - Decimal(".118")) < Decimal(
            tolerance
        )


@pytest.mark.parametrize("arithmetic,precision,tolerance", MODES)
def test_original_scale_bits_and_native_validation(
    source, arithmetic, precision, tolerance
):
    backend = pytest.importorskip("_loop_reduction")
    options = {"arithmetic": arithmetic, "precision": precision}
    state = NativeScalarBindings(
        prepare_scalar_bindings(source, (("delta", 0),)), **options
    )
    view = NativeScaleBindings(state)
    with localcontext() as context:
        context.prec = 350
        exponent = {"double": -40, "double-double": -90, "arbitrary": -240}[arithmetic]
        displacement = Decimal(2) ** exponent
        scale = str((1 + displacement) ** 2)
        batch = backend.NativeBatch.from_decimal([(scale, "0")], 1, 1, **options)
        actual = Decimal(view.evaluate_native(batch).to_decimal()[0][0])
        # Cancellation costs bits in double/DD; the signal must still retain
        # its original, distinctly beyond-binary64 DD/AP scale input.
        assert abs(actual - displacement) < displacement * Decimal("1e-3")
    for value in (("0", "0"), ("-1", "0"), ("1", "1e-300")):
        bad = backend.NativeBatch.from_decimal([value], 1, 1, row_ids=[711], **options)
        with pytest.raises(ArithmeticError, match="711"):
            view.evaluate_native(bad)
    foreign = backend.NativeBatch.from_decimal(
        [("1", "0")],
        1,
        1,
        arithmetic="double-double" if arithmetic == "double" else "double",
    )
    with pytest.raises(ValueError):
        view.evaluate_native(foreign)
    with pytest.raises(ValueError):
        view.evaluate_native(foreign.take_rows([]))
    with pytest.raises(ValueError, match="one mu_r2 column"):
        view.evaluate_native(state.inputs)
    with pytest.raises(TypeError, match="NativeBatch"):
        view.evaluate_native(["4"])


def test_scale_diagnostics_atomic_failure_and_single_owner_snapshot(source):
    backend = pytest.importorskip("_loop_reduction")
    plan = prepare_scalar_bindings(source, (("safe", 0), ("x", 0)))
    state = NativeScalarBindings(plan)
    view = NativeScaleBindings(state)
    snapshot = state.snapshot()
    with pytest.raises(ArithmeticError, match="active diagnostics"):
        state.update({"MU_R": ".5"})
    assert state.snapshot() is snapshot
    with pytest.raises(ArithmeticError, match="19"):
        view.evaluate_native(
            backend.NativeBatch.from_decimal(
                [("4", "0"), (".25", "0")], 2, 1, row_ids=[18, 19]
            )
        )
    assert state.snapshot() is snapshot
    with pytest.raises(FrozenInstanceError):
        snapshot.revision = 7
    with pytest.raises(TypeError):
        snapshot.overrides["x"] = ("9", "0")

    # Simulate an owner update while a native evaluation has released the GIL.
    # The current call must consistently use the snapshot taken at entry.
    scale_evaluator = view._scale

    class UpdatingEvaluator:
        def evaluate_batch(self, batch):
            state.update({"x": "9"})
            return scale_evaluator.evaluate_batch(batch)

    view._scale = UpdatingEvaluator()
    batch = backend.NativeBatch.from_decimal([("4", "0")], 1, 1)
    result = view.evaluate_native(batch)
    assert Decimal(result.to_decimal()[1][0]) == 2
    view._scale = scale_evaluator
    assert Decimal(view.evaluate_native(batch).to_decimal()[1][0]) == 9
    assert (
        Decimal(view.evaluate_native(batch, snapshot=snapshot).to_decimal()[1][0]) == 2
    )
    assert state.revision == 1
    other = NativeScalarBindings(plan)
    with pytest.raises(ValueError, match="different global state"):
        view.evaluate_native(batch, snapshot=other.snapshot())


def test_scale_plan_identity_layout_and_fixed_parameter_guards(source):
    pytest.importorskip("_loop_reduction")
    plan = prepare_scalar_bindings(source, (("z", 0),))
    state = NativeScalarBindings(plan)
    with pytest.raises(ValueError, match="identical ScalarProgram"):
        NativeScaleBindings(state, prepare_scalar_bindings(source, (("MU_R", 0),)))
    for wrong in (
        replace(plan, external_names=tuple(reversed(plan.external_names))),
        replace(plan, external_types=("complex", *plan.external_types[1:])),
        replace(plan, input_symbols=tuple(reversed(plan.input_symbols))),
        replace(plan, fixed_parameters=MappingProxyType({"other": ("0", "0")})),
    ):
        with pytest.raises(ValueError, match="layout or fixed"):
            NativeScaleBindings(state, wrong)
    with pytest.raises(ValueError, match="must be real"):
        NativeScaleBindings(state, scale_parameter="z")
    with pytest.raises(ValueError, match="adjustable external"):
        NativeScaleBindings(state, scale_parameter="L")
    fixed = prepare_scalar_bindings(source, (("z", 0),), fixed_parameters={"MU_R": "5"})
    with pytest.raises(ValueError, match="fixed during generation"):
        NativeScaleBindings(NativeScalarBindings(fixed))


def test_positive_scale_extremes_and_precision_mismatch(source):
    backend = pytest.importorskip("_loop_reduction")
    plan = prepare_scalar_bindings(source, (("MU_R", 0),))
    view = NativeScaleBindings(NativeScalarBindings(plan))
    batch = backend.NativeBatch.from_decimal(
        [("1e-300", "0"), ("1e300", "0"), ("5e-324", "0")], 3, 1
    )
    result = view.evaluate_native(batch)
    assert not result.nonfinite_rows()
    assert all(
        Decimal(real) > 0 and Decimal(imaginary) == 0
        for real, imaginary in result.to_decimal()
    )
    for bad in ("nan", "inf", "-inf"):
        invalid = backend.NativeBatch.from_decimal([(bad, "0")], 1, 1, row_ids=[507])
        with pytest.raises(ArithmeticError, match="507"):
            view.evaluate_native(invalid)
    arbitrary = NativeScaleBindings(
        NativeScalarBindings(plan, arithmetic="arbitrary", precision=40)
    )
    foreign = backend.NativeBatch.from_decimal(
        [], 0, 1, arithmetic="arbitrary", precision=41
    )
    with pytest.raises(ValueError):
        arbitrary.evaluate_native(foreign)


def test_gmu_source_counterterm_scale_logarithms():
    backend = pytest.importorskip("_loop_reduction")
    configured = os.environ.get("PYAMPLICOL_EW_UFO")
    if not configured:
        pytest.skip("set PYAMPLICOL_EW_UFO for the full Gmu fixture")
    raw = load_raw_loop_model(Path(configured), restriction="full")
    plan = prepare_scalar_bindings(raw, (("MT", 0), ("epsUV", 0)))
    ct = prepare_scalar_bindings(
        raw, (("G_UVt", -1), ("G_UVt", 0)), program=plan.program
    )
    for arithmetic, precision, tolerance in MODES:
        options = {"arithmetic": arithmetic, "precision": precision}
        state = NativeScalarBindings(plan, **options)
        view = NativeScaleBindings(state, ct)
        scales = backend.NativeBatch.from_decimal(
            [("10000", "0"), ("40000", "0"), ("90000", "0")],
            3,
            1,
            row_ids=[31, 52, 80],
            **options,
        )
        output = view.evaluate_native(scales)
        assert output.row_ids == [31, 52, 80]
        with localcontext() as context:
            context.prec = 120
            mass = Decimal(state.values.to_decimal()[0][0])
            eps_uv = Decimal(state.values.to_decimal()[1][0])
            values = output.to_decimal()
            for index, scale in enumerate((10000, 40000, 90000)):
                pole = Decimal(values[2 * index][0])
                finite = Decimal(values[2 * index + 1][0])
                expected = (Decimal(scale) / mass**2).ln() / eps_uv
                assert abs(finite / pole - expected) < Decimal(tolerance)


@pytest.mark.parametrize("arithmetic,precision,_tolerance", MODES)
def test_partial_scale_view_filters_active_diagnostics_with_original_ids(
    source, arithmetic, precision, _tolerance
):
    backend = pytest.importorskip("_loop_reduction")
    options = {"arithmetic": arithmetic, "precision": precision}
    plan = prepare_scalar_bindings(source, (("safe", 0), ("A", 0)))
    state = NativeScalarBindings(plan, **options)
    view = NativeScaleBindings(state)
    scales = backend.NativeBatch.from_exact_decimal(
        [("4", "0"), (".25", "0"), ("0", "0"), ("9", "0")],
        4,
        1,
        row_ids=[17, 61, 83, 99],
        **options,
    )
    values, failures = view.evaluate_partial_native(scales)
    assert values.row_ids == [17, 99]
    assert {failure.point_id for failure in failures} == {61, 83}
    assert any("active diagnostics" in failure.stage for failure in failures)
    expected = view.evaluate_native(scales.take_rows([0, 3]))
    assert values.to_decimal() == expected.to_decimal()
    empty, failures = view.evaluate_partial_native(scales.take_rows([]))
    assert empty.row_ids == [] and not failures
    failed, failures = view.evaluate_partial_native(scales.take_rows([1, 2]))
    assert failed.rows == 0 and len(failures) == 2
    foreign = NativeScalarBindings(plan, **options)
    with pytest.raises(ValueError, match="different global state"):
        view.evaluate_partial_native(scales.take_rows([]), snapshot=foreign.snapshot())
    wrong = backend.NativeBatch.from_decimal(
        [], 0, 1, arithmetic="arbitrary", precision=40
    )
    with pytest.raises(ValueError):
        view.evaluate_partial_native(wrong)


def test_reconstructed_snapshot_uses_exact_defaults_and_retained_overrides(source):
    backend = pytest.importorskip("_loop_reduction")
    plan = prepare_scalar_bindings(source, (("z", 0), ("x", 0), ("A", 0)))
    with localcontext() as context:
        context.prec = 120
        original_x = str(Decimal(2) + Decimal(2) ** -240)
    source_state = NativeScalarBindings(plan, parameters={"x": original_x})
    original = source_state.snapshot()
    higher = NativeScalarBindings(plan, arithmetic="arbitrary", precision=90)
    unchanged_higher = higher.snapshot()
    source_state.update({"x": "7", "z": ("3", "4")})
    reconstructed = higher.reconstruct_snapshot(source_state, original)
    expected = NativeScalarBindings(
        plan, arithmetic="arbitrary", precision=90, parameters={"x": original_x}
    )
    assert higher.snapshot() is unchanged_higher
    assert reconstructed.revision == original.revision == 0
    assert reconstructed.values.to_decimal() == expected.values.to_decimal()
    # Exact source default 1/3 + 2i/7 and the retained low input bits are
    # reconstructed; widening the primary double values cannot reproduce them.
    assert not all(
        good
        for _id, good in reconstructed.values.agreement_rows(
            original.values, "1e-85", "1e-85"
        )
    )
    scales = backend.NativeBatch.from_decimal(
        [("4", "0")], 1, 1, arithmetic="arbitrary", precision=90
    )
    NativeScaleBindings(higher).evaluate_native(scales, snapshot=reconstructed)
    with pytest.raises(ValueError, match="different global state"):
        higher.reconstruct_snapshot(source_state, higher.snapshot())
    other = NativeScalarBindings(prepare_scalar_bindings(source, (("z", 0),)))
    with pytest.raises(ValueError, match="identical plan"):
        higher.reconstruct_snapshot(other, other.snapshot())
