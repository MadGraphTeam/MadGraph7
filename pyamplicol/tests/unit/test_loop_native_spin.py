# SPDX-License-Identifier: 0BSD
"""Saved source classes share native currents, exact ingress and subset rescue."""

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from .test_loop_native_directed import _blocks, _values
from .test_loop_physical_correlations import _close, _conjugate
from .test_loop_portable import _rewrite


@pytest.fixture(scope="module")
def spin_plan(tmp_path_factory):
    symbolica = pytest.importorskip("symbolica")
    native = pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path or not symbolica.is_licensed():
        pytest.skip("licensed source generation and loop_sm reference required")
    from pyamplicol.color.connections import ColorConnection, EmitGluon
    from pyamplicol.color.correlator_matrices import ColorCorrelator
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.physical_correlations import directed_color_legs
    from pyamplicol.loops.physical_model_correlations import NativeDirectedModelProgram
    from pyamplicol.loops.portable import write_loop_pacbin
    from pyamplicol.loops.spin_sources import NativeSpinSourcePlan
    from pyamplicol.models.loop_import import load_raw_loop_model

    prepared = prepare_loop_request_model(
        load_raw_loop_model(path),
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    source = prepared.compile(
        (2, -2),
        (21, 21),
        selected_corrections=["QCD"],
        backend=native,
        parameters={"aS": "0.118", "MT": "173", "MU_R": "1000"},
        coefficient_engine="jit",
        reduction_engine="fast",
    )
    legs = directed_color_legs(source.virtual.bare)
    identity = ColorConnection(legs)
    alpha = ColorConnection(legs, (EmitGluon(0, -1), EmitGluon(0, -2)))
    beta = ColorConnection(legs, (EmitGluon(1, -1), EmitGluon(2, -2)))
    histories = ((identity, identity), (alpha, beta), (beta, alpha))
    ordinary = NativeDirectedModelProgram(source, histories).prepared
    sources = NativeSpinSourcePlan(source.virtual.bare, ((3,), (4,), (3, 4)))
    classes = tuple(
        NativeDirectedModelProgram(
            source, histories, spin_layout=sources.layouts[key]
        ).prepared
        for key in ((3,), (4,), (3, 4))
    )
    target = tmp_path_factory.mktemp("native-spin") / "uugg.pacbin"
    write_loop_pacbin(
        source.virtual,
        target,
        directed=ordinary,
        spin_directed=classes,
        directed_histories=tuple(
            ColorCorrelator(label, bra.emissions, ket.emissions)
            for label, (bra, ket) in zip(
                ("identity", "ordered", "swapped"), histories, strict=True
            )
        ),
    )
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/quark_gluon_qcd.json"
    )
    points = json.loads(fixture.read_text(), parse_float=str, parse_int=str)["points"]
    return target, source, classes, points


def _vector():
    return [["0", "0"], ["0.8", "0"], ["0", "1"], ["-0.6", "0"]]


def _scaled(vector, real, imaginary):
    return [
        [
            str(real * Decimal(a) - imaginary * Decimal(b)),
            str(real * Decimal(b) + imaginary * Decimal(a)),
        ]
        for a, b in vector
    ]


def test_spin_descriptor_classes_and_saved_source_phases(spin_plan, tmp_path):
    import _loop_reduction as native

    path, _, classes, _ = spin_plan
    loaded = native.PortableLoopPlan(str(path))
    data = json.loads(loaded.metadata_json())
    assert [entry["legs"] for entry in data["spin_projections"]] == [[3], [4], [3, 4]]
    assert [len(entry["helicities"]) for entry in data["spin_projections"]] == [8, 8, 4]
    assert len(data["helicities"]) == 16
    for entry, prepared in zip(data["spin_projections"], classes, strict=True):
        assert (
            loaded.evaluator_state(entry["directed"]["born"])
            == prepared.bare.born_state
        )
        for phase in entry["phases"]:
            values = native.NativeBatch.from_decimal(_vector(), 1, 4)
            transformed = native.CoefficientEvaluator(
                loaded.evaluator_state(phase)
            ).evaluate_batch(values)
            # SourceIR's gluon crossing phase is exactly one; complex entries
            # survive the saved numerical program without conjugation.
            assert transformed.to_decimal() == values.to_decimal()
    for index, (mutate, message) in enumerate(
        [
            (
                lambda d: d["spin_projections"].append(d["spin_projections"][0]),
                "duplicate",
            ),
            (lambda d: d["spin_projections"][0].update(legs=[1]), "gluon"),
            (lambda d: d["spin_projections"][0]["helicities"].pop(), "spectator"),
            (lambda d: d["spin_projections"][0]["phases"].clear(), "phase"),
            (
                lambda d: d["spin_projections"][0]["directed"]["histories"][0].update(
                    id="wrong"
                ),
                "history",
            ),
        ]
    ):
        broken = tmp_path / f"bad-{index}.pacbin"
        _rewrite(path, broken, mutate)
        with pytest.raises(ValueError, match=message):
            native.PortableLoopPlan(str(broken))


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "3e-9", 20),
        ("double-double", None, "generic", "3e-23", 70),
        ("arbitrary", 70, "generic", "3e-56", 170),
    ],
)
def test_native_spin_classes_literal_sources_and_exact_owner(
    spin_plan, arithmetic, precision, engine, tolerance, bit
):
    import _loop_reduction as native

    from pyamplicol.loops.spin_sources import NativeSpinSourcePlan

    path, source, classes, points = spin_plan
    runtime = native.NativeLoopProgram(
        str(path), arithmetic=arithmetic, precision=precision, reduction_engine=engine
    )
    snapshot = runtime.source_state.snapshot()
    p = points[0]
    a = [["0", "0"], ["0.8", "0"], ["0", "0"], ["-0.6", "0"]]
    b = [["0", "0"], ["0", "0"], ["1", "0"], ["0", "0"]]
    with localcontext() as ctx:
        ctx.prec = 120
        vector = _vector()
        tiny = Decimal(2) ** -bit
        shifted = [
            [str(Decimal(x) + Decimal(q) / 1000), str(Decimal(y) + Decimal(q) / 500)]
            for (x, y), q in zip(vector, p[2], strict=True)
        ]
        sources = [
            vector,
            _scaled(vector, 2, -3),
            shifted,
            a,
            b,
            _scaled(vector, 1 + tiny, 0),
        ]
        ids = [2**64 - 1, 5, 0, 19, 2**63, 70]
        result = runtime.evaluate_directed(
            [p] * 6, ["1000000"] * 6, point_ids=ids, spin_vectors={3: sources}
        )
        baseline = runtime.evaluate_directed([p], ["1000000"])
        assert result.spin_legs == [3] and baseline.spin_legs == []
        assert result.point_ids == ids
        tolerance = Decimal(tolerance)
        for block, ordinary in zip(_blocks(result), _blocks(baseline), strict=True):
            values = _values(block, (6, block.columns))
            _close(values[1], 13 * values[0], tolerance)
            _close(
                values[3] + values[4],
                _values(ordinary, (1, ordinary.columns))[0],
                tolerance,
            )
        # Ward shifts apply to the complete observable, not to each bare piece.
        for block in (
            result.born,
            result.component_batches[0],
            result.directed_first,
            result.directed_second,
        ):
            values = _values(block, (6, block.columns))
            _close(values[2], values[0], tolerance)
        born = _values(result.born, (6, result.born.columns))
        change = born[5, 0, 0] - born[0, 0, 0]
        expected = born[0, 0, 0] * (2 * tiny + tiny * tiny)
        assert abs(change - expected) < abs(expected) * Decimal("1e-4")
        values = _values(result.component_batches[0], (6, 3, 1, 3))
        _close(values[:, 2], _conjugate(values[:, 1]), tolerance)
        assert abs(values[0, 1, 0, 0, 1]) > Decimal("1e-8")
        joint = runtime.evaluate_directed(
            [p, p],
            ["1000000"] * 2,
            spin_vectors={
                3: [vector, _scaled(vector, 2, -3)],
                4: [vector, _scaled(vector, 1, 2)],
            },
        )
        assert joint.spin_legs == [3, 4]
        for block in _blocks(joint):
            values = _values(block, (2, block.columns))
            _close(values[1], 65 * values[0], tolerance)
        # Compare the actual source-owned retained scheduler in the same carrier.
        # This is independent transport, not an independent physics derivation.
        rebound = source.rebind(
            arithmetic=arithmetic, precision=precision, reduction_engine=engine
        )
        source_plan = NativeSpinSourcePlan(rebound.virtual.bare, ((4,),))
        vectors = source_plan.prepare_decimal({4: vector}, [71])
        reference = (
            classes[1]
            .bind(rebound)
            .evaluate_native([p], ["1000000"], point_ids=[71], spin_sources=vectors)
        )
        fourth = runtime.evaluate_directed(
            [p], ["1000000"], point_ids=[71], spin_vectors={4: [vector]}
        )
        names = (
            "born",
            "virtual",
            "bare",
            "uv",
            "r2",
            "uv_local",
            "uv_external_fields",
            "first",
            "second",
        )
        for actual, name in zip(_blocks(fourth)[:-1], names, strict=True):
            expected = getattr(reference, name)
            _close(
                _values(actual, (1, actual.columns)),
                _values(expected, (1, expected.columns)),
                tolerance,
            )
        alpha = Decimal(snapshot.overrides["aS"][0])
        updated = runtime.source_state.update({"aS": [str(alpha * (1 + tiny)), "0"]})
        changed = runtime.evaluate_directed(
            [p], ["1000000"], spin_vectors={3: [vector]}
        )
        assert changed.source_revision == updated.revision
        assert changed.born.to_decimal() != result.born.take_rows([0]).to_decimal()
        restored = runtime.evaluate_directed(
            [p], ["1000000"], snapshot=snapshot, spin_vectors={3: [vector]}
        )
        assert (
            restored.directed_first.to_decimal()
            == result.directed_first.take_rows([0]).to_decimal()
        )
    empty = runtime.evaluate_directed(
        [], [], spin_vectors={3: [vector], 4: [vector]}, partial=True
    )
    assert empty.spin_legs == [3, 4] and empty.point_ids == empty.failures == []
    assert all(block.rows == 0 for block in _blocks(empty))
    with pytest.raises(ValueError, match="joint source class"):
        runtime.evaluate_directed([p], ["1000000"], spin_vectors={2: [vector]})
    with pytest.raises(ValueError, match="one row"):
        runtime.evaluate_directed([p], ["1000000"], spin_vectors={3: [vector, vector]})


def test_spin_partial_compaction_and_exact_source_replay(spin_plan):
    import _loop_reduction as native

    path, _, _, points = spin_plan
    runtime = native.NativeLoopProgram(str(path), reduction_engine="fast")
    vector = _vector()
    other = _scaled(vector, 2, -3)
    ids = [2**64 - 1, 0, 73]
    batch = [points[0], points[1], points[0]]
    scales = ["1e-400", "900000", "1000000"]
    sources = {3: [vector, other, vector], 4: [other, vector, other]}
    partial = runtime.evaluate_directed(
        batch, scales, point_ids=ids, spin_vectors=sources, partial=True
    )
    assert partial.point_ids == ids[1:]
    assert partial.requested_point_ids == ids and partial.spin_legs == [3, 4]
    assert partial.failures and {failure.point_id for failure in partial.failures} == {
        ids[0]
    }
    good = runtime.evaluate_directed(
        batch[1:],
        scales[1:],
        point_ids=ids[1:],
        spin_vectors={leg: rows[1:] for leg, rows in sources.items()},
    )
    for actual, expected in zip(_blocks(partial), _blocks(good), strict=True):
        assert actual.to_decimal() == expected.to_decimal()
    failed = runtime.evaluate_directed(
        batch[:1],
        scales[:1],
        point_ids=ids[:1],
        spin_vectors={leg: rows[:1] for leg, rows in sources.items()},
        partial=True,
    )
    assert failed.point_ids == [] and failed.failures
    assert all(block.rows == 0 for block in _blocks(failed))
    verified = runtime.evaluate_directed_verified(
        batch,
        scales,
        [("arbitrary", 60), ("arbitrary", 80)],
        "1e-8",
        "1e-9",
        point_ids=ids,
        spin_vectors=sources,
    )
    assert verified.result.point_ids == ids and verified.result.failures == []
    assert verified.result.spin_legs == [3, 4]
    diagnostics = json.loads(verified.diagnostics_json)
    assert [row["status"] for row in diagnostics] == ["rescued", "success", "success"]
    assert diagnostics[0]["attempts"][-1]["agreement"]
    exact = runtime.rebind(arithmetic="arbitrary", precision=80).evaluate_directed(
        batch[:1],
        scales[:1],
        point_ids=ids[:1],
        spin_vectors={leg: rows[:1] for leg, rows in sources.items()},
    )
    for actual, expected in zip(_blocks(verified.result), _blocks(exact), strict=True):
        assert all(
            ok
            for _, ok in actual.take_rows([0]).agreement_rows(expected, "1e-8", "1e-9")
        )
    # Rebinding creates fresh workspaces and retains no event-value cache.
    rebound = runtime.rebind(reduction_engine="fast")
    repeated = rebound.evaluate_directed(
        batch[1:],
        scales[1:],
        point_ids=ids[1:],
        spin_vectors={leg: rows[1:] for leg, rows in sources.items()},
    )
    for actual, expected in zip(_blocks(repeated), _blocks(good), strict=True):
        assert actual.to_decimal() == expected.to_decimal()
    empty = runtime.evaluate_directed_verified(
        [], [], [("arbitrary", 60)], "1e-8", "1e-9", spin_vectors={3: [], 4: [vector]}
    )
    assert empty.result.spin_legs == [3, 4] and empty.result.point_ids == []


def test_joint_sources_reconstruct_the_full_polarization_sum(spin_plan):
    import _loop_reduction as native

    path, _, _, points = spin_plan
    runtime = native.NativeLoopProgram(str(path), reduction_engine="fast")
    a = [["0", "0"], ["0.8", "0"], ["0", "0"], ["-0.6", "0"]]
    b = [["0", "0"], ["0", "0"], ["1", "0"], ["0", "0"]]
    joint = runtime.evaluate_directed(
        [points[0]] * 4,
        ["1000000"] * 4,
        spin_vectors={3: [a, a, b, b], 4: [a, b, a, b]},
    )
    ordinary = runtime.evaluate_directed(points[:1], ["1000000"])
    for actual, expected in zip(_blocks(joint), _blocks(ordinary), strict=True):
        _close(
            _values(actual, (4, actual.columns)).sum(axis=0),
            _values(expected, (1, expected.columns))[0],
            Decimal("3e-9"),
        )
