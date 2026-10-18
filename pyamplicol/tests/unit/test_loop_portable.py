# SPDX-License-Identifier: 0BSD
"""Graph-free PACBIN execution retains complete physics and rejects bad layouts."""

import io
import json
import os
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def portable_control(tmp_path_factory):
    symbolica = pytest.importorskip("symbolica")
    backend = pytest.importorskip("_loop_reduction")
    if not symbolica.is_licensed() or not hasattr(backend, "PortableLoopPlan"):
        pytest.skip("licensed generation and native portable bridge required")
    source = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not source:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE")
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.physical_counterterms import (
        distinct_flavour_model_virtual_program,
    )
    from pyamplicol.loops.portable import write_loop_pacbin
    from pyamplicol.models.loop_import import load_raw_loop_model

    program = distinct_flavour_model_virtual_program(
        build_loop_model(load_raw_loop_model(source)),
        reduction_engine="fast",
        coefficient_engine="jit",
    )
    path = tmp_path_factory.mktemp("loop-portable") / "evaluators.pacbin"
    write_loop_pacbin(program, path)
    fixture = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    points = np.asarray(json.loads(fixture.read_text())["points"], dtype=str)
    return program, path, points


def _batches(result):
    return result.born, *result.component_batches, result.totals


def _forbid_producer(monkeypatch):
    from pyamplicol.loops import physical_counterterms, physical_process
    from pyamplicol.loops.color_bilinears import PhysicalColorSpace
    from pyamplicol.loops.kernels import CoefficientKernel, PreparedCoefficientDAG
    from pyamplicol.loops.kinematics import LoopKinematicPlan

    def forbidden(*args, **kwargs):
        raise AssertionError("portable load invoked producer work")

    for owner, name in (
        (physical_process.NativeBareProcessProgram, "__init__"),
        (physical_counterterms.NativeModelVirtualProgram, "__init__"),
        (physical_process, "build_color_loop_recurrence"),
        (physical_process, "build_color_tree_recurrence"),
        (physical_process, "build_loop_kinematics"),
        (physical_process, "_saved"),
        (physical_process, "_reduction_numerators"),
        (physical_counterterms, "build_color_tree_recurrence"),
        (physical_counterterms, "_saved"),
        (physical_counterterms, "_saved_scalar_pairs"),
        (PhysicalColorSpace, "overlap"),
        (CoefficientKernel, "evaluator_bytes"),
        (PreparedCoefficientDAG, "from_dag"),
        (LoopKinematicPlan, "evaluator_bytes"),
    ):
        monkeypatch.setattr(owner, name, forbidden)


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "1e-13"),
        ("double-double", None, "generic", "1e-28"),
        ("arbitrary", 80, "generic", "1e-68"),
    ],
)
def test_complete_graph_free_views(
    portable_control, monkeypatch, arithmetic, precision, engine, tolerance
):
    from pyamplicol.loops.portable import PortableLoopPlan

    original, path, points = portable_control
    options = dict(
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
        coefficient_engine="interpreted",
    )
    expected_view = original.rebind(**options)
    expected = expected_view.evaluate_ordered_native(
        points, ["1000000"] * 3, model_values=["0.118", "173"]
    )
    _forbid_producer(monkeypatch)
    plan = PortableLoopPlan.open(path)
    view = plan.bind(**options)
    result = view.evaluate_ordered_native(
        points, ["1000000"] * 3, model_values=["0.118", "173"], point_ids=[91, 4, 70]
    )
    assert result.r2_strategy == "model"
    assert result.batch.point_ids == [91, 4, 70]
    assert len(view.field_currents) == 0  # This control is not a nonzero-field proof.
    for actual, reference in zip(_batches(result), _batches(expected), strict=True):
        reference = reference.take_rows([0, 1, 2], actual.row_ids)
        assert all(
            ok for _, ok in actual.agreement_rows(reference, tolerance, tolerance)
        )
    retained = [value.to_decimal() for value in _batches(result)]
    rebound = view.rebind(**options)
    assert rebound.bare.families[0].reducer is not view.bare.families[0].reducer
    changed = rebound.evaluate_ordered_native(
        points[:1], ["700000"], model_values=["0.12", "190"]
    )
    assert changed.virtual.to_decimal() != result.virtual.take_rows([0]).to_decimal()
    assert (
        rebound.evaluate_ordered_native(
            points[:0], [], model_values=["0.118", "173"]
        ).born.rows
        == 0
    )
    restored = rebound.evaluate_ordered_native(
        points, ["1000000"] * 3, model_values=["0.118", "173"], point_ids=[91, 4, 70]
    )
    for actual, reference in zip(_batches(restored), _batches(result), strict=True):
        # Compare independent numerical workspaces in the requested arithmetic;
        # previously returned buffers must remain byte-for-byte unchanged below.
        assert all(
            ok for _, ok in actual.agreement_rows(reference, tolerance, tolerance)
        )
    assert [value.to_decimal() for value in _batches(result)] == retained
    if arithmetic == "arbitrary":
        with localcontext() as context:
            context.prec = 120
            alpha = str(Decimal("0.118") * (1 + Decimal(2) ** -180))
        exact_change = view.evaluate_ordered_native(
            points[:1], ["1000000"], model_values=[alpha, "173"]
        )
        exact_reference = expected_view.evaluate_ordered_native(
            points[:1], ["1000000"], model_values=[alpha, "173"]
        )
        assert (
            exact_change.virtual.to_decimal()
            != result.virtual.take_rows([0]).to_decimal()
        )
        for actual, reference in zip(
            _batches(exact_change), _batches(exact_reference), strict=True
        ):
            assert all(
                ok for _, ok in actual.agreement_rows(reference, "1e-70", "1e-70")
            )
    for operation in (
        lambda: view.bare.process,
        lambda: view.prepared,
        lambda: view.bare.set_color_metrics((), ()),
        lambda: view.evaluate_ordered_partial_native(points, ["1"] * 3),
    ):
        with pytest.raises(NotImplementedError, match="portable control"):
            operation()
    if arithmetic == "double":
        reference_path = os.environ.get("PYAMPLICOL_UUDD_COMPONENT_REFERENCE")
        if reference_path:
            reference = json.loads(Path(reference_path).read_text())
            for name, key in (
                ("bare", "bare"),
                ("uv", "uv"),
                ("r2", "r2"),
                ("virtual", "full"),
            ):
                actual = np.asarray(
                    getattr(result, name).to_decimal(), dtype=float
                ).reshape(3, 3, 2)[:, :, 0]
                expected_values = [
                    [
                        row[key][pole]
                        for pole in ("finite", "single_pole", "double_pole")
                    ]
                    for row in reference["rows"]
                ]
                np.testing.assert_allclose(
                    actual, expected_values, rtol=4e-11, atol=1e-14
                )


def _rewrite(path, destination, mutate, *, wrong_kind=False):
    from pyamplicol.generation.evaluator_container import (
        PacbinMemberKind,
        PacbinMemberSource,
        PacbinReader,
        write_pacbin_atomic,
    )
    from pyamplicol.loops.portable import _PLAN_MEMBER

    with PacbinReader.open(path) as reader:
        members = [
            (member, reader.read_member(member.logical_path, length=member.length))
            for member in reader.members
        ]
    sources = []
    for member, payload in members:
        kind = member.kind
        if member.logical_path == _PLAN_MEMBER:
            data = json.loads(payload)
            mutate(data)
            payload = json.dumps(data).encode()
            if wrong_kind:
                kind = PacbinMemberKind.SYMBOLICA_EXACT_STATE
        sources.append(
            PacbinMemberSource(member.logical_path, kind, io.BytesIO(payload))
        )
    write_pacbin_atomic(destination, sources)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda d: d.update(schema=99), "schema"),
        (lambda d: d.update(r2_strategy="direct"), "model-R2"),
        (lambda d: d["state_legs"][1].update(offset=999), "offset"),
        (lambda d: d["families"][0]["powers"][0].__setitem__(0, 999), "rank"),
        (lambda d: d["schedules"][0]["output_slots"].append(2**40), "output"),
        (
            lambda d: d["born"]["representative_for"].__setitem__(0, 999),
            "representative",
        ),
        (lambda d: d["families"][0].update(exact_state="missing"), "reference"),
        (lambda d: d["families"][0].update(sector_indices=[999]), "sectors"),
        (lambda d: d["order_layout"]["born"][0]["order"][0].__setitem__(1, -1), "u32"),
        (lambda d: d.update(producer_graph={}), "unknown field"),
    ],
)
def test_native_portable_boundary_rejects_corrupt_layouts(
    portable_control, tmp_path, mutation, message
):
    from pyamplicol.loops.portable import PortableLoopPlan

    path = tmp_path / "invalid.pacbin"
    _rewrite(portable_control[1], path, mutation)
    with pytest.raises(ValueError, match=message):
        PortableLoopPlan.open(path)


def test_member_kind_and_digest_are_native_checks(portable_control, tmp_path):
    from pyamplicol.loops.portable import PortableLoopPlan

    path = tmp_path / "wrong-kind.pacbin"
    _rewrite(portable_control[1], path, lambda d: None, wrong_kind=True)
    with pytest.raises(ValueError, match="member kind"):
        PortableLoopPlan.open(path)
    data = bytearray(portable_control[1].read_bytes())
    data[65] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match=r"digest|hash|SHA"):
        PortableLoopPlan.open(path)


def test_fresh_process_needs_no_source_model_or_graphs(portable_control, tmp_path):
    original, source, points = portable_control
    path = tmp_path / "owned.pacbin"
    path.write_bytes(source.read_bytes())
    inputs = tmp_path / "inputs.json"
    outputs = tmp_path / "outputs.json"
    inputs.write_text(json.dumps(points[:1].tolist()))
    code = """import json, sys
import numpy as np
from pathlib import Path
from pyamplicol.loops import physical_process, physical_counterterms
from pyamplicol.loops.portable import PortableLoopPlan
def forbidden(*a, **k): raise AssertionError("producer called in fresh process")
physical_process.NativeBareProcessProgram.__init__ = forbidden
physical_counterterms.NativeModelVirtualProgram.__init__ = forbidden
physical_process._saved = physical_process._reduction_numerators = forbidden
physical_process.build_color_loop_recurrence = forbidden
physical_process.build_color_tree_recurrence = forbidden
physical_counterterms.build_color_tree_recurrence = forbidden
plan=PortableLoopPlan.open(sys.argv[1])
Path(sys.argv[1]).unlink()
program=plan.bind(reduction_engine="fast")
points=np.asarray(json.load(open(sys.argv[2])),dtype=str)
result=program.evaluate_ordered_native(points, ["1000000"],
                                     model_values=["0.118","173"])
json.dump({"born":result.born.to_decimal(),"virtual":result.virtual.to_decimal()},open(sys.argv[3],"w"))
"""
    environment = dict(os.environ)
    environment.pop("PYAMPLICOL_NLO_UFO_REFERENCE", None)
    subprocess.run(
        [sys.executable, "-c", code, str(path), str(inputs), str(outputs)],
        env=environment,
        check=True,
    )
    actual = json.loads(outputs.read_text())
    assert len(actual["born"]) == 1 and len(actual["virtual"]) == 3
    reference = original.evaluate_ordered_native(
        points[:1], ["1000000"], model_values=["0.118", "173"]
    )
    for name in ("born", "virtual"):
        np.testing.assert_allclose(
            np.asarray(actual[name], dtype=float),
            np.asarray(getattr(reference, name).to_decimal(), dtype=float),
            rtol=4e-12,
            atol=1e-14,
        )


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda d: d["order_layout"]["virtual"][0]["pairs"][0]["loop"][
                0
            ].__setitem__(1, 99),
            "order layout",
        ),
        (
            lambda d: d["order_layout"]["born"].append(d["order_layout"]["born"][0]),
            "order labels",
        ),
        (
            lambda d: d["order_layout"]["virtual"][0]["pairs"].append(
                d["order_layout"]["virtual"][0]["pairs"][0]
            ),
            "order layout",
        ),
        (
            lambda d: d["order_layout"]["correction_names"].append(
                d["order_layout"]["correction_names"][0]
            ),
            "correction names",
        ),
    ],
)
def test_native_order_semantics(portable_control, tmp_path, mutation, message):
    from pyamplicol.loops.portable import PortableLoopPlan

    path = tmp_path / "bad-order.pacbin"
    _rewrite(portable_control[1], path, mutation)
    with pytest.raises(ValueError, match=message):
        PortableLoopPlan.open(path)
