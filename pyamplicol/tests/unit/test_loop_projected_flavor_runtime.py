# SPDX-License-Identifier: 0BSD
"""Saved projected member selection retains signed axes and identity sources."""

import json
from decimal import Decimal, localcontext

import pytest

from .test_loop_portable import _rewrite
from .test_loop_projected_flavor_transport import _compare
from .test_loop_projected_flavor_transport import saved_pair as _saved_pair


@pytest.fixture(scope="module")
def mapped_pair():
    return _saved_pair.__wrapped__()


@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "3e-9"),
        ("double-double", None, "generic", "3e-24"),
        ("arbitrary", 70, "generic", "3e-54"),
    ],
)
def test_mapped_saved_signed_resolved_against_independent_member(
    mapped_pair, accuracy, arithmetic, precision, engine, tolerance
):
    root, _, native, points = mapped_pair
    mapped = native.NativeVirtualRuntime(
        str(root / f"{accuracy}-ordinary-mapped-artifact"),
        process="ssgg",
        arithmetic=arithmetic,
        precision=precision,
    )
    reference = native.NativeLoopProgram(
        str(root / f"ssgg-{accuracy}-current-ordinary.pacbin"),
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
    )
    metadata = json.loads(mapped.metadata_json())
    assert metadata["process_id"] == "ssgg"
    assert metadata["representative_process_id"] == "ddgg"
    for size in (0, 1, 3) if arithmetic == "double" else (1,):
        batch = [points[1], points[0], points[1]][:size]
        ids, scales = [91, 7, 2**64 - 1][:size], ["900000", "1000000", "900000"][:size]
        for resolved in (False, True):
            evaluate = mapped.evaluate_resolved if resolved else mapped.evaluate
            _compare(
                evaluate(batch, scales, point_ids=ids),
                reference.evaluate(batch, scales, point_ids=ids, resolved=resolved),
                tolerance,
                ids,
            )


def _close_values(left, right, tolerance=Decimal("3e-9")):
    if isinstance(left, tuple):
        assert isinstance(right, tuple) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _close_values(a, b, tolerance)
        return
    for a, b in ((left.real, right.real), (left.imag, right.imag)):
        a, b = Decimal(str(a)), Decimal(str(b))
        assert abs(a - b) <= tolerance * max(abs(a), abs(b), Decimal(1))


@pytest.mark.parametrize("accuracy", ["lc", "nlc", "full"])
def test_public_projected_axes_sources_and_clear(mapped_pair, accuracy):
    from pyamplicol import CorrelatedRequest, Runtime

    from .test_loop_generation_spin_flavor_runtime import _vectors

    root, _, _, points = mapped_pair
    ordinary = Runtime.load(
        root / f"{accuracy}-ordinary-mapped-artifact", process="ssgg"
    )
    independent = Runtime.load(
        root / f"{accuracy}-ordinary-independent-artifact", process="ssgg"
    )
    resolved = ordinary.evaluate_resolved(points[:1], mu_r2="1000000")
    expected = independent.evaluate_resolved(points[:1], mu_r2="1000000")
    assert resolved.born_color_ids == expected.born_color_ids
    assert resolved.virtual_color_ids == expected.virtual_color_ids
    assert len(resolved.virtual_color_ids) == (3 if accuracy == "lc" else 1)
    _close_values(resolved.values, expected.values)
    _close_values(
        resolved.total().values, ordinary.evaluate(points[:1], mu_r2="1000000").values
    )
    source = Runtime.load(root / f"{accuracy}-spin-mapped-artifact", process="ssgg")
    reference = Runtime.load(
        root / f"{accuracy}-spin-independent-artifact", process="ssgg"
    )
    a, b = _vectors()
    requests = {
        "left": CorrelatedRequest("born", {3: a}),
        "joint": CorrelatedRequest("born", {3: a, 4: b}),
    }
    actual = source.evaluate_correlated_many(points[:1], requests, mu_r2="1000000")
    expected = reference.evaluate_correlated_many(points[:1], requests, mu_r2="1000000")
    for key in requests:
        _close_values(actual[key].values, expected[key].values)
    source.set_spin_correlation_vectors({3: a, 4: b})
    before = source.evaluate_correlated(points[:1], mu_r2="1000000")
    source.clear()
    assert (
        source.evaluate_correlated(points[:1], mu_r2="1000000").values == before.values
    )
    assert source.evaluate_correlated([], mu_r2=[]).shape == (0, 1, 3)


def test_projected_mapped_source_subset_rescue(mapped_pair):
    from pyamplicol import CorrelatedRequest, Runtime
    from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy

    from .test_loop_generation_spin_flavor_runtime import _vectors

    root, _, _, points = mapped_pair
    runtime = Runtime.load(root / "lc-spin-mapped-artifact", process="ssgg")
    a, b = _vectors()
    with localcontext() as context:
        context.prec = 100
        result = runtime.evaluate_correlated_many(
            [points[0]] * 3,
            {"joint": CorrelatedRequest("born", {3: a, 4: b})},
            mu_r2=["1000000", "1e-400", "1000000"],
            stability=VirtualStabilityPolicy(
                (
                    VirtualArithmeticSpec("arbitrary", 60),
                    VirtualArithmeticSpec("arbitrary", 80),
                ),
                "1e-8",
                "1e-9",
            ),
        )["joint"]
    assert [row.status for row in result.diagnostics] == [
        "success",
        "rescued",
        "success",
    ]
    assert result.values[0] == result.values[2]


@pytest.mark.parametrize("spin", [False, True])
def test_native_projected_member_rejects_unproved_histories(
    mapped_pair, tmp_path, spin
):
    root, _, native, _ = mapped_pair

    def mutate(plan):
        projection = (
            plan["spin_projections"][0]["directed"]
            if spin
            else plan["directed_projection"]
        )
        projection["histories"][0].update(
            bra=[{"kind": "emit-gluon", "emitter": 0, "emitted": -1}],
            ket=[{"kind": "emit-gluon", "emitter": 0, "emitted": -1}],
        )

    invalid = tmp_path / "unproved.pacbin"
    _rewrite(root / "lc-admitted-spin.pacbin", invalid, mutate)
    with pytest.raises(
        ValueError, match="projected flavour transport requires identity"
    ):
        native.PortableLoopPlan(str(invalid))
