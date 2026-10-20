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

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def portable_control(loop_sm_ufo, loop_native, tmp_path_factory):
    symbolica = pytest.importorskip("symbolica")
    backend = loop_native
    if not symbolica.is_licensed() or not hasattr(backend, "PortableLoopPlan"):
        pytest.skip("licensed generation and native portable bridge required")
    source = str(loop_sm_ufo)
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
    loop_reference_path,
    portable_control,
    monkeypatch,
    arithmetic,
    precision,
    engine,
    tolerance,
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
        reference_path = str(loop_reference_path("uudd-component-control"))
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
                [row[key][pole] for pole in ("finite", "single_pole", "double_pole")]
                for row in reference["rows"]
            ]
            np.testing.assert_allclose(actual, expected_values, rtol=4e-11, atol=1e-14)


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
        (lambda d: d.update(r2_strategy="direct"), "strategy/dimensional"),
        (lambda d: d["state_legs"][1].update(offset=999), "offset"),
        (lambda d: d["schedules"][0]["output_slots"].append(2**40), "output"),
        (
            lambda d: d["born"]["representative_for"].__setitem__(0, 999),
            "representative",
        ),
        (lambda d: d["families"][0].update(exact_state="missing"), "reference"),
        (
            lambda d: d["families"][0].update(sector_indices=[999]),
            "dual Born family selection",
        ),
        (lambda d: d["order_layout"]["born"][0]["order"][0].__setitem__(1, -1), "u32"),
        (lambda d: d.update(producer_graph={}), "unknown field"),
        (
            lambda d: d.update(
                color_projection={
                    "accuracy": "lc",
                    "born_degrees": [],
                    "virtual_degrees": [3],
                }
            ),
            "colour projection",
        ),
        (
            lambda d: d.update(
                color_projection={
                    "accuracy": "nlc",
                    "born_degrees": [2],
                    "virtual_degrees": [],
                }
            ),
            "colour projection",
        ),
        (
            lambda d: d.update(
                color_projection={
                    "accuracy": "guess",
                    "born_degrees": [2],
                    "virtual_degrees": [3],
                }
            ),
            "colour projection",
        ),
        (
            lambda d: d.update(
                color_projection={
                    "accuracy": "lc",
                    "born_degrees": [True],
                    "virtual_degrees": [3],
                }
            ),
            "i64",
        ),
        (
            lambda d: d.update(
                color_projection={
                    "accuracy": "lc",
                    "born_degrees": [2],
                    "virtual_degrees": [3],
                    "nc": 5,
                }
            ),
            "unknown field",
        ),
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


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("direct_gg", marks=pytest.mark.deferred_direct_r2),
        "model_top",
    ],
)
def extended_control(loop_sm_ufo, loop_native, request, tmp_path_factory):
    symbolica = pytest.importorskip("symbolica")
    backend = loop_native
    source = str(loop_sm_ufo)
    if not symbolica.is_licensed() or not hasattr(backend, "PortableLoopPlan"):
        pytest.skip("licensed source model and native portable bridge required")
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.physical_counterterms import (
        gluon_scattering_model_virtual_program,
        top_pair_model_virtual_program,
    )
    from pyamplicol.loops.portable import write_loop_pacbin
    from pyamplicol.models.loop_import import load_raw_loop_model

    direct = request.param == "direct_gg"
    factory = (
        gluon_scattering_model_virtual_program
        if direct
        else top_pair_model_virtual_program
    )
    options = {"r2_strategy": "direct"} if direct else {}
    program = factory(
        build_loop_model(load_raw_loop_model(source)),
        reduction_engine="fast",
        coefficient_engine="jit",
        **options,
    )
    path = tmp_path_factory.mktemp(request.param) / "evaluators.pacbin"
    write_loop_pacbin(program, path)
    name = "gluon_scattering" if direct else "top_pair"
    fixture = (
        Path(__file__).resolve().parents[2] / f"tools/loop_reference/{name}_qcd.json"
    )
    points = np.asarray(json.loads(fixture.read_text())["points"], dtype=str)[:2]
    return program, path, points


def test_direct_sectors_and_nonzero_fields_survive_portable_rebinding(
    extended_control, monkeypatch, loop_reference
):
    from pyamplicol.loops.portable import PortableLoopPlan

    original, path, points = extended_control
    direct = original.r2_strategy == "direct"
    plan = PortableLoopPlan.open(path)
    metadata = json.loads(plan._native.metadata_json())
    assert metadata["dimensional_numerators"] is direct
    assert all(current["kind"] != "R2" for current in metadata["currents"]) is direct
    if direct:
        assert metadata["external_pdgs"] == [21] * 4
        assert any(f["rational_numerators"][1] != "0" for f in metadata["families"])
        assert any(
            any(p[4:] != [0, 0] for p in f["powers"]) for f in metadata["families"]
        )
    else:
        assert len(metadata["fields"]) == len(original.field_currents) == 4
    _forbid_producer(monkeypatch)
    retained = []
    previous = None
    for arithmetic, precision, engine, tolerance in (
        ("double", None, "fast", "2e-11"),
        ("double-double", None, "generic", "1e-25"),
        ("arbitrary", 60, "generic", "1e-48"),
    ):
        options = dict(
            arithmetic=arithmetic, precision=precision, reduction_engine=engine
        )
        reference_view = original.rebind(**options)
        view = plan.bind(**options) if previous is None else previous.rebind(**options)
        expected = reference_view.evaluate_ordered_native(
            points, ["1000000"] * 2, model_values=["0.118", "173"]
        )
        actual = view.evaluate_ordered_native(
            points, ["1000000"] * 2, model_values=["0.118", "173"], point_ids=[61, 8]
        )
        assert actual.r2_strategy == original.r2_strategy
        assert actual.batch.point_ids == [61, 8]
        assert all(
            row and all(lane == point and status == "success" for lane, status in row)
            for point, row in zip(actual.batch.point_ids, actual.statuses, strict=True)
        )
        for result, reference in zip(_batches(actual), _batches(expected), strict=True):
            reference = reference.take_rows([0, 1], result.row_ids)
            assert all(
                ok for _, ok in result.agreement_rows(reference, tolerance, tolerance)
            )
        assert all(
            current.executor is not other.executor
            for current, other in zip(
                view.currents, reference_view.currents, strict=True
            )
        )
        assert all(
            current.coefficient_evaluator is not other.coefficient_evaluator
            for current, other in zip(
                view.field_currents, reference_view.field_currents, strict=True
            )
        )
        assert any(
            Decimal(pair[0]) != 0 for pair in actual.uv_external_fields.to_decimal()
        )
        if direct:
            assert any(f.rational_reducer is not None for f in view.bare.families)
        if arithmetic == "double":
            rows = loop_reference(
                "gluon-scattering-component-control"
                if direct
                else "top-pair-component-control"
            )["rows"][:2]
            for name, key in (
                ("virtual", "full"),
                ("bare", "bare"),
                ("uv", "uv"),
                ("r2", "r2"),
                ("uv_local", "uv_vertex"),
                ("uv_external_fields", "wf"),
            ):
                np.testing.assert_allclose(
                    np.asarray(getattr(actual, name).to_decimal(), dtype=float)[
                        :, 0
                    ].reshape(2, 3),
                    [
                        [
                            row[key][pole]
                            for pole in ("finite", "single_pole", "double_pole")
                        ]
                        for row in rows
                    ],
                    rtol=5e-11,
                    atol=1e-10,
                )
        retained.append((actual, [value.to_decimal() for value in _batches(actual)]))
        with localcontext() as context:
            context.prec = 110
            alpha = (
                str(Decimal("0.118") * (1 + Decimal(2) ** -140))
                if arithmetic == "arbitrary"
                else "0.12"
            )
        changed = view.evaluate_ordered_native(
            points[:1], ["700000"], model_values=[alpha, "173"], point_ids=[8]
        )
        changed_reference = reference_view.evaluate_ordered_native(
            points[:1], ["700000"], model_values=[alpha, "173"]
        )
        for result, reference in zip(
            _batches(changed), _batches(changed_reference), strict=True
        ):
            assert all(
                ok
                for _, ok in result.agreement_rows(
                    reference.take_rows([0], [8]), tolerance, tolerance
                )
            )
        empty = view.evaluate_ordered_native(
            points[:0], [], model_values=["0.118", "173"]
        )
        assert empty.r2_strategy == original.r2_strategy and empty.virtual.rows == 0
        assert all(
            [v.to_decimal() for v in _batches(result)] == values
            for result, values in retained
        )
        previous = view


@pytest.mark.deferred_direct_r2
def test_native_direct_metadata_rejects_mixed_or_uncertified_sectors(
    extended_control, tmp_path
):
    from pyamplicol.loops.portable import PortableLoopPlan

    original, source, _ = extended_control
    if original.r2_strategy != "direct":
        pytest.skip("direct-only capability check")

    def change_squared_order(data):
        # Keep the generic pair-sum invariant while leaving the certified gg order.
        data["order_layout"]["born"][0].update(
            order=[["QCD", 6]], pairs=[[[["QCD", 3]], [["QCD", 3]]]]
        )

    for index, (mutate, message) in enumerate(
        (
            (lambda d: d["external_pdgs"].__setitem__(0, 1), "four-gluon"),
            (lambda d: d["currents"][0].update(kind="R2"), "counterterm"),
            (lambda d: d["families"][0]["powers"][0].__setitem__(5, 3), "sector"),
            (lambda d: d.update(r2_strategy="model"), "strategy/dimensional"),
            (change_squared_order, "four-gluon"),
        )
    ):
        path = tmp_path / f"invalid-direct-{index}.pacbin"
        _rewrite(source, path, mutate)
        with pytest.raises(ValueError, match=message):
            PortableLoopPlan.open(path)


def test_jit_projector_binding_retains_portable_ordered_physics(
    portable_control, monkeypatch
):
    """Only numerical binding changes; saved physical projector bytes remain exact."""
    from pyamplicol.loops.portable import PortableLoopPlan

    original, path, points = portable_control
    baseline = original.evaluate_ordered_native(
        points, ["1000000"] * 3, model_values=["0.118", "173"]
    )
    retained = original.prepared
    _forbid_producer(monkeypatch)
    options = dict(
        arithmetic="double",
        reduction_engine="fast",
        coefficient_engine="jit",
        projector_engine="jit",
    )
    views = (original.rebind(**options), PortableLoopPlan.open(path).bind(**options))
    assert original.bare.projector_engine == "interpreted"
    for view in views:
        assert view.bare.projector_engine == "jit"
        assert all(
            family.projector.engine == "jit" and family.projector.tile_rows == 16
            for family in view.bare.families
        )
        result = view.evaluate_ordered_native(
            points, ["1000000"] * 3, model_values=["0.118", "173"]
        )
        for actual, expected in zip(_batches(result), _batches(baseline), strict=True):
            assert all(
                ok for _, ok in actual.agreement_rows(expected, "5e-10", "1e-10")
            )
        reference = [batch.to_decimal() for batch in _batches(result)]
        tail = points[np.arange(17) % 3]
        ids = [401 + 11 * i for i in range(17)] if view is views[1] else list(range(17))
        point_options = {"point_ids": ids} if view is views[1] else {}
        changed = view.evaluate_ordered_native(
            tail, ["700000"] * 17, model_values=["0.12", "190"], **point_options
        )
        expected = original.evaluate_ordered_native(
            tail, ["700000"] * 17, model_values=["0.12", "190"]
        )
        assert changed.batch.point_ids == ids
        assert changed.statuses == tuple(
            tuple((ids[row], status) for row, status in group)
            for group in expected.statuses
        )
        for actual, expected_batch in zip(
            _batches(changed), _batches(expected), strict=True
        ):
            assert all(
                ok
                for _, ok in actual.agreement_rows(
                    expected_batch.take_rows(
                        list(range(expected_batch.rows)), actual.row_ids
                    ),
                    "5e-10",
                    "1e-10",
                )
            )
        assert (
            view.evaluate_ordered_native(
                points[:0], [], model_values=["0.118", "173"]
            ).born.rows
            == 0
        )
        assert [batch.to_decimal() for batch in _batches(result)] == reference
    assert views[0].prepared is retained
    assert (
        views[0].bare.families[0].projector is not views[1].bare.families[0].projector
    )
    assert views[0].bare._evaluator_states == original.bare._evaluator_states
    with pytest.raises(ValueError, match="requires double"):
        original.rebind(arithmetic="arbitrary", precision=80, projector_engine="jit")


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "1e-13"),
        ("double-double", None, "generic", "1e-28"),
        ("arbitrary", 80, "generic", "1e-68"),
    ],
)
def test_shared_core_complete_coordinator(
    loop_native, portable_control, monkeypatch, arithmetic, precision, engine, tolerance
):
    """One Rust call executes the same saved plan, with no Python event stages."""
    backend = loop_native
    from pyamplicol.loops.portable import PortableLoopPlan

    _, path, points = portable_control
    options = dict(
        arithmetic=arithmetic,
        precision=precision,
        coefficient_engine="interpreted",
        reduction_engine=engine,
    )
    _forbid_producer(monkeypatch)
    oracle = PortableLoopPlan.open(path).bind(**options)
    native = backend.NativeLoopProgram(str(path), **options)
    expected = oracle.evaluate_ordered_native(
        points, ["1000000"] * 3, model_values=["0.118", "173"], point_ids=[91, 4, 70]
    )
    result = native.evaluate(
        points.tolist(), ["1000000"] * 3, ["0.118", "173"], point_ids=[91, 4, 70]
    )
    assert result.r2_strategy == native.r2_strategy == "model"
    assert result.point_ids == [91, 4, 70]
    assert result.statuses == [
        [(row, status) for _, status in statuses]
        for row, statuses in enumerate(expected.statuses)
    ]
    assert native.precision_bits == result.born.precision_bits
    for actual, reference in zip(_batches(result), _batches(expected), strict=True):
        assert all(
            ok
            for _, ok in actual.agreement_rows(
                reference.take_rows(list(range(reference.rows)), actual.row_ids),
                tolerance,
                tolerance,
            )
        )
    retained = [value.to_decimal() for value in _batches(result)]
    rebound = native.rebind(**options)
    changed = rebound.evaluate(points[:1].tolist(), ["700000"], ["0.12", "190"])
    assert (
        changed.component_batches[0].to_decimal()
        != result.component_batches[0].take_rows([0]).to_decimal()
    )
    assert rebound.evaluate([], [], ["0.118", "173"]).born.rows == 0
    again = native.evaluate(
        points.tolist(), ["1000000"] * 3, ["0.118", "173"], point_ids=[91, 4, 70]
    )
    for actual, reference in zip(_batches(again), _batches(result), strict=True):
        assert all(
            ok for _, ok in actual.agreement_rows(reference, tolerance, tolerance)
        )
    assert [value.to_decimal() for value in _batches(result)] == retained
    if arithmetic == "arbitrary":
        with localcontext() as context:
            context.prec = 120
            alpha = str(Decimal("0.118") * (1 + Decimal(2) ** -180))
        changed = native.evaluate(points[:1].tolist(), ["1000000"], [alpha, "173"])
        expected = oracle.evaluate_ordered_native(
            points[:1], ["1000000"], model_values=[alpha, "173"]
        )
        assert (
            changed.component_batches[0].to_decimal()
            != result.component_batches[0].take_rows([0]).to_decimal()
        )
        for actual, reference in zip(
            _batches(changed), _batches(expected), strict=True
        ):
            assert all(
                ok for _, ok in actual.agreement_rows(reference, "1e-70", "1e-70")
            )
    if arithmetic == "double":
        for momenta, scales, ids in (
            (points.tolist(), ["1"] * 3, [1]),
            (points.tolist(), ["1"] * 4, [1, 2, 3]),
            (points.tolist(), ["1"] * 3, [1, 1, 3]),
        ):
            with pytest.raises(ValueError, match=r"identit|shape|overflow"):
                native.evaluate(momenta, scales, ["0.118", "173"], point_ids=ids)
        with pytest.raises(ValueError, match=r"nonfinite|finite|decimal"):
            native.evaluate(points[:1].tolist(), ["nan"], ["0.118", "173"])
        # A failed call cannot expose old buffers or poison the next valid call.
        assert (
            native.evaluate(
                points[:1].tolist(), ["1000000"], ["0.118", "173"]
            ).born.rows
            == 1
        )


def test_shared_core_direct_and_nonzero_fields(
    loop_native, extended_control, monkeypatch
):
    """The same native sequence carries Delta sectors and nonzero LSZ terms."""
    backend = loop_native
    from pyamplicol.loops.portable import PortableLoopPlan

    original, path, points = extended_control
    direct = original.r2_strategy == "direct"
    _forbid_producer(monkeypatch)
    saved = PortableLoopPlan.open(path)
    native = None
    for arithmetic, precision, engine, tolerance in (
        ("double", None, "fast", "2e-11"),
        ("double-double", None, "generic", "1e-25"),
        ("arbitrary", 60, "generic", "1e-48"),
    ):
        options = dict(
            arithmetic=arithmetic, precision=precision, reduction_engine=engine
        )
        oracle = saved.bind(**options)
        native = (
            backend.NativeLoopProgram(str(path), **options)
            if native is None
            else native.rebind(**options)
        )
        expected = oracle.evaluate_ordered_native(
            points[:1], ["1000000"], model_values=["0.118", "173"], point_ids=[63]
        )
        actual = native.evaluate(
            points[:1].tolist(), ["1000000"], ["0.118", "173"], point_ids=[63]
        )
        assert actual.r2_strategy == ("direct" if direct else "model")
        assert actual.point_ids == [63]
        assert actual.statuses == [[(0, status) for _, status in expected.statuses[0]]]
        for value, reference in zip(_batches(actual), _batches(expected), strict=True):
            assert all(
                ok
                for _, ok in value.agreement_rows(
                    reference.take_rows(list(range(reference.rows)), value.row_ids),
                    tolerance,
                    tolerance,
                )
            )
        # Both controls have nonzero field terms; direct also keeps an explicit
        # nonzero dimensional remainder distinct from the N4 result.
        assert any(
            Decimal(pair[0]) != 0 for pair in actual.component_batches[5].to_decimal()
        )
        if direct:
            assert any(
                Decimal(pair[0]) != 0
                for pair in actual.component_batches[3].to_decimal()
            )
        retained = [value.to_decimal() for value in _batches(actual)]
        assert native.evaluate([], [], ["0.118", "173"]).born.rows == 0
        assert [value.to_decimal() for value in _batches(actual)] == retained
