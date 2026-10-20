# SPDX-License-Identifier: 0BSD
"""Source scalar owners, local CTs and particle fields survive graph-free loading."""

import json
import os
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from .test_loop_portable import _batches, _forbid_producer, _rewrite

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def source_control(loop_sm_ufo, loop_native, tmp_path_factory):
    source = str(loop_sm_ufo)
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.portable import write_loop_pacbin
    from pyamplicol.models.loop_import import load_raw_loop_model

    prepared = prepare_loop_request_model(
        load_raw_loop_model(source),
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    program = prepared.compile(
        (21, 21),
        (21, 21),
        selected_corrections=["QCD"],
        parameters={"aS": "0.118000000000000000000000001", "MT": "173", "MU_R": "1000"},
        reduction_engine="fast",
        coefficient_engine="jit",
    )
    path = tmp_path_factory.mktemp("source-portable") / "evaluators.pacbin"
    write_loop_pacbin(program.virtual, path)
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    return (
        program,
        path,
        np.asarray(json.loads(fixture.read_text())["points"], dtype=str),
    )


def _forbid_source(monkeypatch):
    from pyamplicol.loops import scalar_bindings, scalar_frontend
    from pyamplicol.models import loop_import

    def forbidden(*args, **kwargs):
        raise AssertionError("portable source load repeated source lowering")

    _forbid_producer(monkeypatch)
    for owner, name in (
        (scalar_bindings.PreparedScalarBindings, "from_plan"),
        (scalar_bindings, "prepare_scalar_bindings"),
        (scalar_bindings, "lower_scalar_program"),
        (scalar_frontend, "lower_scalar_program"),
        (loop_import, "load_raw_loop_model"),
    ):
        monkeypatch.setattr(owner, name, forbidden)


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "2e-12"),
        ("double-double", None, "generic", "1e-27"),
        ("arbitrary", 70, "generic", "1e-59"),
    ],
)
def test_source_owner_complete_roundtrip(
    loop_reference_path,
    source_control,
    monkeypatch,
    arithmetic,
    precision,
    engine,
    tolerance,
):
    from pyamplicol.loops.portable import PortableLoopPlan

    original, path, points = source_control
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    expected = original.rebind(**options).virtual
    _forbid_source(monkeypatch)
    plan = PortableLoopPlan.open(path)
    view = plan.bind(**options)
    owner = view.bare.model_state
    assert (
        owner
        is view.bare.model_scale_view.global_state
        is view.source_scale_view.global_state
    )
    assert (
        owner.plan
        is view.bare.model_scale_view.plan
        is view.source_scale_view.plan
        is None
    )
    assert len(view.field_currents) == len(expected.field_currents) > 0
    assert all(
        current.coefficient_evaluator is None
        for current in (*view.currents, *view.field_currents)
    )
    assert owner.overrides["aS"] == ("0.118000000000000000000000001", "0")
    scales = ["1000000", "900000", "2100000"]
    actual = view.evaluate_ordered_native(points, scales, point_ids=[91, 7, 30])
    reference = expected.evaluate_ordered_native(points, scales)
    for a, b in zip(_batches(actual), _batches(reference), strict=True):
        b = b.take_rows([0, 1, 2], a.row_ids)
        assert all(ok for _, ok in a.agreement_rows(b, tolerance, tolerance))
    retained = [a.to_decimal() for a in _batches(actual)]
    # Updating the owner cannot change the already captured scalar revision.
    batch = view.bare.prepare_currents_native(points[:1], ["900000"]).batch
    old_ct = [a.to_decimal() for a in view.evaluate_counterterm_currents_native(batch)]
    with localcontext() as context:
        context.prec = 110
        bit = (
            20
            if arithmetic == "double"
            else 70
            if arithmetic == "double-double"
            else 160
        )
        alpha = str(Decimal("0.118000000000000000000000001") * (1 + Decimal(2) ** -bit))
    view.update_parameters({"aS": alpha, "MT": "190.000000000000000000000001"})
    assert old_ct == [
        a.to_decimal() for a in view.evaluate_counterterm_currents_native(batch)
    ]
    rebound = view.rebind(**options, parameters={"MU_R": "1500"})
    assert rebound.bare.model_state.overrides == dict(owner.overrides) | {
        "MU_R": ("1500", "0")
    }
    assert rebound.bare.model_state is not owner
    with pytest.raises(ValueError, match="different global state"):
        rebound.source_scale_view.evaluate_native(
            batch.scales_squared, snapshot=batch.model_snapshot
        )
    expected.bare.model_state.update(rebound.bare.model_state.overrides)
    changed = rebound.evaluate_ordered_native(points[:1], ["1500000"])
    target = expected.evaluate_ordered_native(points[:1], ["1500000"])
    assert changed.born.to_decimal() != actual.born.take_rows([0]).to_decimal()
    with localcontext() as context:
        context.prec = 110
        ratio = Decimal(changed.born.to_decimal()[0][0]) / Decimal(
            actual.born.to_decimal()[0][0]
        )
        exact_ratio = (Decimal(alpha) / Decimal("0.118000000000000000000000001")) ** 2
        assert abs(ratio - exact_ratio) < Decimal(tolerance)
    for a, b in zip(_batches(changed), _batches(target), strict=True):
        assert all(ok for _, ok in a.agreement_rows(b, tolerance, tolerance))
    assert rebound.evaluate_ordered_native(points[:0], []).virtual.rows == 0
    assert [a.to_decimal() for a in _batches(actual)] == retained
    # Explicit bind overlays initial saved values, rather than resetting absent keys.
    initial = plan.bind(**options, parameters={"MU_R": "1700"})
    assert initial.bare.model_state.overrides["MT"] == ("173", "0")
    assert (
        initial.bare.model_state.overrides["aS"]
        == original.source_bare.state.overrides["aS"]
    )
    before = owner.snapshot()
    for bad in ({"aS": ["0.1", "1e-1000"]}, {"MB": "1"}, {"unknown": "1"}):
        with pytest.raises(ValueError):
            view.update_parameters(bad)
        assert owner.snapshot() is before
    with pytest.raises(TypeError, match="mapping"):
        plan.bind(parameters=[("aS", "0.1")])
    if arithmetic == "double":
        reference_path = str(loop_reference_path("gluon-scattering-component-control"))
        view.update_parameters(original.source_bare.state.overrides)
        result = view.evaluate_ordered_native(points, ["1000000"] * 3)
        capture = json.loads(Path(reference_path).read_text())
        for name, key in (
            ("bare", "bare"),
            ("uv", "uv"),
            ("r2", "r2"),
            ("virtual", "full"),
            ("uv_local", "uv_vertex"),
            ("uv_external_fields", "wf"),
        ):
            values = np.asarray(
                getattr(result, name).to_decimal(), dtype=float
            ).reshape(3, 3, 2)[:, :, 0]
            reference_values = [
                [row[key][p] for p in ("finite", "single_pole", "double_pole")]
                for row in capture["rows"]
            ]
            np.testing.assert_allclose(values, reference_values, rtol=5e-11, atol=1e-10)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda d: d["model_binding"]["owner"]["external_types"].__setitem__(
                0, "integer"
            ),
            "names/types",
        ),
        (
            lambda d: d["model_binding"]["owner"].update(defaults_state=None),
            "default program",
        ),
        (lambda d: d["model_binding"]["owner"].update(guards_state=None), "type-guard"),
        (
            lambda d: d["model_binding"]["initial_overrides"].update(
                aS=["0.1", "1e-1000"]
            ),
            "imaginary",
        ),
        (
            lambda d: d["model_binding"]["initial_overrides"].update(MB=["1", "0"]),
            "adjustable",
        ),
        (
            lambda d: d["model_binding"]["initial_overrides"].update(aS=["NaN", "0"]),
            "finite decimal",
        ),
        (
            lambda d: d["currents"][0]["scalar"].update(columns=[999999]),
            "column out of range",
        ),
        (
            lambda d: d["fields"][0]["columns"].append(d["fields"][0]["columns"][0]),
            "field scalar width",
        ),
        (
            lambda d: d["currents"][0].update(
                scalar={
                    "kind": "evaluator",
                    "state": d["model_binding"]["owner"]["values_state"],
                }
            ),
            "mix legacy",
        ),
        (
            lambda d: d["model_binding"]["ordinary_scale"]["outputs"][0].__setitem__(
                0, "different"
            ),
            "ordinary source scale",
        ),
        (
            lambda d: d["model_binding"]["counterterm_scale"].update(scale_index=99999),
            "source scale",
        ),
        (
            lambda d: d["model_binding"]["model_output_indices"].__setitem__(0, 99999),
            "output/input selection",
        ),
    ],
)
def test_native_source_descriptor_rejects_invalid_bindings(
    source_control, tmp_path, mutation, message
):
    from pyamplicol.loops.portable import PortableLoopPlan

    path = tmp_path / "invalid.pacbin"
    _rewrite(source_control[1], path, mutation)
    with pytest.raises(ValueError, match=message):
        PortableLoopPlan.open(path)


def test_source_portable_fresh_process(source_control, tmp_path):
    from pyamplicol.loops.portable import PortableLoopPlan

    _, source, points = source_control
    path, inputs, outputs = [
        tmp_path / p for p in ("owned.pacbin", "inputs.json", "outputs.json")
    ]
    path.write_bytes(source.read_bytes())
    inputs.write_text(json.dumps(points[:1].tolist()))
    code = """import json, sys
from pathlib import Path
import numpy as np
from pyamplicol.loops import physical_process, physical_counterterms
from pyamplicol.loops import scalar_frontend, scalar_bindings
from pyamplicol.models import loop_import
from pyamplicol.loops.portable import PortableLoopPlan
def forbidden(*a, **k): raise AssertionError("source producer invoked")
physical_process.NativeBareProcessProgram.__init__=forbidden
physical_counterterms.NativeModelVirtualProgram.__init__=forbidden
scalar_frontend.lower_scalar_program=scalar_bindings.lower_scalar_program=forbidden
scalar_bindings.prepare_scalar_bindings=loop_import.load_raw_loop_model=forbidden
plan=PortableLoopPlan.open(sys.argv[1])
Path(sys.argv[1]).unlink()
view=plan.bind(reduction_engine="fast")
r=view.evaluate_ordered_native(np.asarray(json.load(open(sys.argv[2])),dtype=str),["1000000"])
values=[x.to_decimal() for x in (r.born,*r.component_batches,r.totals)]
json.dump(values,open(sys.argv[3],"w"))
"""
    environment = dict(os.environ)
    environment.pop("PYAMPLICOL_NLO_UFO_REFERENCE", None)
    subprocess.run(
        [sys.executable, "-c", code, str(path), str(inputs), str(outputs)],
        env=environment,
        check=True,
    )
    expected = (
        PortableLoopPlan.open(source)
        .bind(reduction_engine="fast")
        .evaluate_ordered_native(points[:1], ["1000000"])
    )
    for a, b in zip(json.loads(outputs.read_text()), _batches(expected), strict=True):
        np.testing.assert_allclose(
            np.asarray(a, dtype=float),
            np.asarray(b.to_decimal(), dtype=float),
            rtol=4e-12,
            atol=1e-12,
        )
