# SPDX-License-Identifier: 0BSD
"""Explicit source continuation composes paired scalars with complete model R2."""

import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def prepared():
    symbolica = pytest.importorskip("symbolica")
    pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path or not symbolica.is_licensed():
        pytest.skip("licensed NLO source required")
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.model_compatibility import prepare_loop_sm_scalar_declarations
    from pyamplicol.models.loop_import import load_raw_loop_model

    raw = prepare_loop_sm_scalar_declarations(load_raw_loop_model(path)).derived
    return prepare_loop_request_model(
        raw,
        born_order_bounds={"QCD": 0, "QED": 2},
        loop_order_bounds={"QCD": 2, "QED": 2},
    )


def declaration(prepared):
    from symbolica import E, S

    from pyamplicol.loops.source_color import SourceColorContinuation
    from pyamplicol.loops.wavefunction import prepare_external_field_rules
    from pyamplicol.models.loop_declaration import LoopSchemeDeclaration

    nc = S("source_dy_control::Nc")
    # Explicit test model input: standard SUN tensors plus CA/CF/TF/Ncol.
    # Fixed-SU3 four-gluon R2 has deliberately NOT been admitted. This fixture
    # does not certify all entries of loop_sm or a public model-name domain.
    keys = frozenset(
        entry.key
        for entry in prepared.model.kernels
        if not (
            entry.kind == "R2"
            and len(entry.particle_ids) == 4
            and all(
                prepared.model.raw.collections["particles"][i]["fields"]["spin"] == 3
                and prepared.model.raw.collections["particles"][i]["fields"]["color"]
                == 8
                for i in entry.particle_ids
            )
        )
    )
    return SourceColorContinuation(
        prepared.model,
        nc,
        {"Ncol": nc, "CA": nc, "CF": (nc**2 - 1) / (2 * nc), "TF": E("1/2")},
        keys,
        frozenset(
            (rule.particle_id, rule.correction, rule.loop_group_index)
            for rule in prepare_external_field_rules(prepared.model)
        ),
        LoopSchemeDeclaration(
            "HV",
            "physical-four-dimensional",
            "source-4d",
            "source-QCD",
            "real",
            "source-local-rules",
            "explicit test source continuation",
        ),
    )


def compile_program(prepared, **options):
    return prepared.compile(
        (2, -2),
        (11, -11),
        selected_corrections=["QCD"],
        reduction_engine="fast",
        **options,
    ).virtual


@pytest.fixture(scope="module")
def programs(prepared, tmp_path_factory):
    from pyamplicol.loops.portable import write_loop_pacbin

    result = {"dense": compile_program(prepared)}
    for accuracy in ("full", "lc", "nlc"):
        program = compile_program(
            prepared, source_color=declaration(prepared), color_accuracy=accuracy
        )
        path = tmp_path_factory.mktemp(f"source-{accuracy}") / "plan.pacbin"
        write_loop_pacbin(program, path, resolved=True)
        result[accuracy] = (program, path)
    return result


def close(actual, expected, tolerance, ratio=Decimal(1), scale_floor=Decimal("1e-15")):
    with localcontext() as context:
        context.prec = 100
        for a, b in zip(actual.to_decimal(), expected.to_decimal(), strict=True):
            for x, y in zip(a, b, strict=True):
                target = Decimal(y) * ratio
                assert abs(Decimal(x) - target) <= Decimal(tolerance) * max(
                    abs(target), scale_floor
                )


@pytest.mark.parametrize(
    "mode,precision,tolerance,bit",
    [
        ("double", None, "1e-10", 20),
        ("double-double", None, "1e-25", 70),
        ("arbitrary", 70, "1e-57", 160),
    ],
)
def test_source_projection_complete_components_and_saved_views(
    programs, mode, precision, tolerance, bit
):
    from pyamplicol.loops.portable import PortableLoopPlan

    fixture = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/drell_yan_qcd.json"
    )
    points = json.loads(fixture.read_text(), parse_float=str, parse_int=str)["points"][
        :1
    ]
    options = dict(arithmetic=mode, precision=precision)
    dense = (
        programs["dense"].rebind(**options).evaluate_ordered_native(points, ["1000000"])
    )
    for accuracy in ("full", "lc", "nlc"):
        original, path = programs[accuracy]
        expected_view = original.rebind(**options)
        view = PortableLoopPlan.open(path).bind(**options)
        actual = view.evaluate_ordered_native(points, ["1000000"])
        expected = expected_view.evaluate_ordered_native(points, ["1000000"])
        assert view.source_scale_view.global_state is view.bare.model_state
        assert view.color_accuracy == accuracy
        close(actual.born, dense.born, tolerance)
        for a, b, c in zip(
            actual.component_batches,
            expected.component_batches,
            dense.component_batches,
            strict=True,
        ):
            close(a, b, tolerance)
            close(a, c, tolerance, Decimal(9) / 8 if accuracy == "lc" else Decimal(1))
        with localcontext() as context:
            context.prec = 100
            alpha = str(Decimal("0.118") + Decimal(2) ** -bit)
        view.update_parameters({"aS": alpha})
        expected_view.bare.model_state.update({"aS": alpha})
        parameter_changed = view.evaluate_ordered_native(points, ["1000000"])
        assert parameter_changed.virtual.to_decimal() != actual.virtual.to_decimal()
        changed = view.evaluate_ordered_native(points, ["900000"])
        expected_changed = expected_view.evaluate_ordered_native(points, ["900000"])
        assert changed.virtual.to_decimal() != parameter_changed.virtual.to_decimal()
        close(changed.virtual, expected_changed.virtual, tolerance)


def test_selected_source_rule_requires_explicit_continuation(prepared, programs):
    continuation = declaration(prepared)
    key = programs["full"][0].insertions[0].rule.entries[0].key
    missing = replace(continuation, kernel_keys=continuation.kernel_keys - {key})
    with pytest.raises(NotImplementedError, match="selected source kernel"):
        compile_program(prepared, source_color=missing, color_accuracy="lc")


def test_source_formal_full_mixed_orders_match_independent_provider(tmp_path):
    symbolica = pytest.importorskip("symbolica")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path or not symbolica.is_licensed():
        pytest.skip("licensed NLO source required")
    import _loop_reduction as native

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.model_compatibility import prepare_loop_sm_scalar_declarations
    from pyamplicol.loops.portable import write_loop_pacbin
    from pyamplicol.models.loop_import import load_raw_loop_model

    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/uudd_mixed_sm_qcd.json"
    )
    reference = json.loads(fixture.read_text())
    raw = prepare_loop_sm_scalar_declarations(
        load_raw_loop_model(path, restriction="full")
    ).derived
    prepared = prepare_loop_request_model(
        raw,
        born_order_bounds={"QCD": (0, 2), "QED": (0, 2)},
        loop_order_bounds={"QCD": (0, 4), "QED": (0, 2)},
        fixed_parameters=reference["fixed_parameters"],
    )
    program = prepared.compile(
        (2, -2),
        (1, -1),
        selected_corrections=["QCD"],
        parameters=reference["parameters"],
        source_color=declaration(prepared),
        color_accuracy="full",
        reduction_engine="fast",
    ).virtual
    projection = program.color_projection
    assert len(projection.born_degrees) == len(projection.virtual_degrees) == 3
    assert projection.virtual_degrees == (3, 3, 3)
    saved = tmp_path / "mixed-projected.pacbin"
    write_loop_pacbin(program, saved, resolved=True)
    runtime = native.NativeLoopProgram(str(saved), reduction_engine="fast")
    actual = runtime.evaluate(reference["points"][:1], [reference["mu_r2"]])
    for kind, batch, width, reference_key in (
        ("born", actual.born, 1, "born_squared_orders"),
        ("virtual", actual.component_batches[0], 3, "virtual_squared_orders"),
    ):
        for index, sector in enumerate(getattr(program.bare.order_layout, kind)):
            key = next(
                k
                for k, values in reference[reference_key].items()
                if {name: value for name, value in values.items() if value}
                == dict(sector.order)
            )
            expected = (
                [reference["rows"][0]["born"][key]]
                if kind == "born"
                else [
                    reference["rows"][0]["full"][key][pole]
                    for pole in ("finite", "single_pole", "double_pole")
                ]
            )
            for pole, target in enumerate(expected):
                value = Decimal(batch.to_decimal()[index * width + pole][0])
                target = Decimal(str(target))
                assert abs(value - target) <= Decimal("5e-11") * max(
                    abs(target), Decimal("1e-15")
                ), value


def test_source_external_fields_keep_their_colour_grade(tmp_path):
    symbolica = pytest.importorskip("symbolica")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path or not symbolica.is_licensed():
        pytest.skip("licensed NLO source required")
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.portable import PortableLoopPlan, write_loop_pacbin
    from pyamplicol.models.loop_import import load_raw_loop_model

    prepared = prepare_loop_request_model(
        load_raw_loop_model(path),
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/quark_gluon_qcd.json"
    )
    points = json.loads(fixture.read_text(), parse_float=str, parse_int=str)["points"][
        :1
    ]
    dense = prepared.compile(
        (2, -2), (21, 21), selected_corrections=["QCD"], reduction_engine="fast"
    ).virtual
    reference = dense.evaluate_ordered_native(points, ["1000000"])
    # Vanishing Laurent components can contain rounding residue from a sum of
    # terms of Born scale. Compare them using that physical scale, not their
    # accidentally tiny numerical value.
    born_scale = max(abs(Decimal(pair[0])) for pair in reference.born.to_decimal())
    assert any(
        Decimal(pair[0]) != 0 for pair in reference.uv_external_fields.to_decimal()
    )
    for accuracy in ("full", "lc", "nlc"):
        program = prepared.compile(
            (2, -2),
            (21, 21),
            selected_corrections=["QCD"],
            source_color=declaration(prepared),
            color_accuracy=accuracy,
            reduction_engine="fast",
        ).virtual
        assert program.field_currents
        path = tmp_path / f"field-{accuracy}.pacbin"
        write_loop_pacbin(program, path)
        loaded = PortableLoopPlan.open(path).bind(reduction_engine="fast")
        expected = program.evaluate_ordered_native(points, ["1000000"])
        actual = loaded.evaluate_ordered_native(points, ["1000000"])
        for a, b in zip(
            actual.component_batches, expected.component_batches, strict=True
        ):
            close(a, b, "1e-10", scale_floor=born_scale)
        if accuracy == "full":
            for a, b in zip(
                actual.component_batches, reference.component_batches, strict=True
            ):
                close(a, b, "1e-10", scale_floor=born_scale)
        if accuracy == "lc":
            assert all(
                Decimal(pair[0]) == 0 for pair in actual.uv_external_fields.to_decimal()
            )
        else:
            assert any(
                Decimal(pair[0]) != 0 for pair in actual.uv_external_fields.to_decimal()
            )


@pytest.fixture(scope="module")
def four_gluon_programs(tmp_path_factory):
    symbolica = pytest.importorskip("symbolica")
    pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path or not symbolica.is_licensed():
        pytest.skip("licensed NLO source required")
    from symbolica import E

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.portable import write_loop_pacbin
    from pyamplicol.loops.qcd_source_color import prepare_loop_sm_four_gluon_color
    from pyamplicol.models.loop_import import load_raw_loop_model

    original = prepare_loop_request_model(
        load_raw_loop_model(path),
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    options = dict(
        selected_corrections=["QCD"],
        parameters={"aS": "0.118", "MT": "173", "MU_R": "1000"},
        reduction_engine="fast",
        coefficient_engine="jit",
    )
    dense = original.compile((21, 21), (21, 21), **options).virtual
    prepared, continued = prepare_loop_sm_four_gluon_color(
        original, declaration(original)
    )
    assert continued.scheme_parameter_values["lhv"] == E("1")
    assert len(continued.whole_rule_continuations) == 2
    assert (
        prepared.counterterms.source_entries_complete
        == original.counterterms.source_entries_complete
    )
    result = {"dense": dense}
    for accuracy in ("full", "lc", "nlc"):
        program = prepared.compile(
            (21, 21),
            (21, 21),
            source_color=continued,
            color_accuracy=accuracy,
            **options,
        ).virtual
        assert any(
            item.rule.whole_rule_continuation is not None for item in program.insertions
        )
        saved = tmp_path_factory.mktemp(f"source-4g-{accuracy}") / "plan.pacbin"
        write_loop_pacbin(program, saved, resolved=True)
        result[accuracy] = program, saved
    return result


@pytest.mark.parametrize(
    "mode,precision,tolerance,bit",
    [
        ("double", None, "2e-10", 20),
        ("double-double", None, "2e-24", 70),
        ("arbitrary", 70, "2e-55", 160),
    ],
)
def test_complete_four_gluon_source_continuation(
    four_gluon_programs, mode, precision, tolerance, bit
):
    from pyamplicol.loops.portable import PortableLoopPlan

    points = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "tools/loop_reference/gluon_scattering_qcd.json"
        ).read_text(),
        parse_float=str,
        parse_int=str,
    )["points"][:1]
    options = dict(arithmetic=mode, precision=precision)
    dense = (
        four_gluon_programs["dense"]
        .rebind(**options)
        .evaluate_ordered_native(points, ["1000000"])
    )
    scale = max(abs(Decimal(pair[0])) for pair in dense.born.to_decimal())
    for accuracy in ("full", "lc", "nlc"):
        retained, saved = four_gluon_programs[accuracy]
        program = retained.rebind(**options)
        loaded = PortableLoopPlan.open(saved).bind(**options)
        actual = loaded.evaluate_ordered_native(points, ["1000000"])
        expected = program.evaluate_ordered_native(points, ["1000000"])
        for a, b in zip(
            actual.component_batches, expected.component_batches, strict=True
        ):
            close(a, b, tolerance, scale_floor=scale)
        if accuracy == "full":
            for a, b in zip(
                actual.component_batches, dense.component_batches, strict=True
            ):
                close(a, b, tolerance, scale_floor=scale)
        assert loaded.color_projection.virtual_degrees == (5,)
        assert loaded.color_projection.born_degrees == (4,)
        # Independent leading-colour counting for four external adjoints:
        # Born has Nc^4-Nc^2 and its universal double pole Nc^5-Nc^3.
        ratio = Decimal(9) / 8 if accuracy == "lc" else Decimal(1)
        close(actual.born, dense.born, tolerance, ratio, scale_floor=scale)
        close(
            actual.virtual.take_columns([2]), dense.virtual.take_columns([2]),
            tolerance, ratio, scale_floor=scale,
        )
        with localcontext() as context:
            context.prec = 100
            alpha = str(Decimal("0.118") + Decimal(2) ** (-bit))
        loaded.bare.model_state.update({"aS": alpha})
        program.bare.model_state.update({"aS": alpha})
        updated = loaded.evaluate_ordered_native(points, ["1000000"])
        assert updated.virtual.to_decimal() != actual.virtual.to_decimal()
        expected_update = program.evaluate_ordered_native(points, ["1000000"])
        for a, b in zip(
            updated.component_batches, expected_update.component_batches, strict=True
        ):
            close(a, b, tolerance, scale_floor=scale)
