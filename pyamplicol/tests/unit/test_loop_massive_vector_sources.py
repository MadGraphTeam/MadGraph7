# SPDX-License-Identifier: 0BSD
"""Saved actual charged-vector sources retain all three physical states."""

import json
from copy import deepcopy
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from .test_loop_native_directed import _blocks, _values
from .test_loop_physical_correlations import _close
from .test_loop_portable import _rewrite

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def massive_sources(saved_loop_fixture):
    fixture = str(saved_loop_fixture("massive-vector-sources-v43ff063"))
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    verify_native_module(_rusticol)
    assert load_loop_native() is _rusticol._loop
    return Path(fixture), _rusticol._loop


def test_saved_charged_source_identity_states_and_reciprocity(
    massive_sources, tmp_path
):
    root, native = massive_sources
    source = root / "outgoing-spin.pacbin"
    native.PortableLoopPlan(str(source))
    actual = json.loads((root / "model-vector-contracts.json").read_text())

    def malformed_pair(data):
        # Substitute the actual opposite charged SourceIR in the second vector
        # port, then break its declared reciprocal relation. This is a malformed
        # cold descriptor control, not another physical scattering process.
        data["external_pdgs"][3] = -24
        data["state_legs"][3]["massless"] = False
        for saved in data["spin_projections"]:
            if 4 in saved["legs"]:
                contract = deepcopy(actual["-24"])
                contract["identity"]["species_id"] = "foreign-species"
                saved["source_contracts"][saved["legs"].index(4)] = contract

    def different_class_identity(data):
        saved = data["spin_projections"][-1]
        saved["source_contracts"][0]["identity"]["species_id"] = "foreign-species"

    mutations = (
        (lambda d: d["state_legs"][2].update(massless=True), "physical helicity"),
        (
            lambda d: d["spin_projections"][0]["source_contracts"][0]["states"].pop(1),
            "physical vector",
        ),
        (
            lambda d: d["spin_projections"][0]["source_contracts"][0][
                "identity"
            ].update(anti_pdg_label=24),
            "antiparticle relation",
        ),
        (different_class_identity, "across saved joint classes"),
        (malformed_pair, "non-involutive particle/antiparticle"),
    )
    for index, (mutate, message) in enumerate(mutations):
        path = tmp_path / f"invalid-{index}.pacbin"
        _rewrite(source, path, mutate)
        with pytest.raises(ValueError, match=message):
            native.PortableLoopPlan(str(path))


def _wavefunctions(native, point, leg, states, incoming, options):
    values = native.NativeBatch.from_decimal(
        [(str(value), "0") for _ in states for value in point[leg - 1]],
        len(states),
        4,
        arithmetic=options["arithmetic"],
        precision=options["precision"],
    )
    result = native.wavefunctions_batch(
        values, states, ["epsilon" if incoming else "epsilon_bar"] * len(states)
    ).to_decimal()
    return [result[4 * index : 4 * (index + 1)] for index in range(len(states))]


def _scale(vector, real, imaginary):
    return [
        (
            str(real * Decimal(a) - imaginary * Decimal(b)),
            str(real * Decimal(b) + imaginary * Decimal(a)),
        )
        for a, b in vector
    ]


@pytest.mark.parametrize(
    "direction,leg", [("outgoing", 3), ("incoming", 2), ("negative", 3)]
)
@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "8e-8", 20),
        ("double-double", None, "generic", "8e-22", 70),
        ("arbitrary", 70, "generic", "8e-54", 170),
    ],
)
def test_charged_three_state_sources_and_exact_replay(
    massive_sources, direction, leg, arithmetic, precision, engine, tolerance, bit
):
    root, native = massive_sources
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    runtime = native.NativeLoopProgram(
        str(root / f"{direction}-spin.pacbin"), **options
    )
    ordinary = native.NativeLoopProgram(
        str(root / f"{direction}-ordinary.pacbin"), **options
    )
    point = json.loads((root / f"{direction}-ordinary.json").read_text())["point"]
    descriptor = json.loads(runtime.metadata_json())
    contract = next(
        saved["source_contracts"][saved["legs"].index(leg)]
        for saved in descriptor["spin_projections"]
        if leg in saved["legs"]
    )
    assert contract["identity"]["orientation"] == (
        "antiparticle" if direction == "negative" else "particle"
    )
    assert contract["identity"]["pdg_label"] == (-24 if direction == "negative" else 24)
    assert contract["identity"]["anti_pdg_label"] == (
        24 if direction == "negative" else -24
    )
    vectors = _wavefunctions(
        native, point, leg, [-1, 0, 1], direction == "incoming", options
    )
    gluons = _wavefunctions(native, point, 4, [-1, 1], False, options)
    with localcontext() as context:
        context.prec = 120
        tiny = Decimal(2) ** -bit
        changes = {"aS": [str(Decimal("0.118") * (1 + tiny)), "0"]}
        runtime.source_state.update(changes)
        ordinary.source_state.update(changes)
        snapshot = runtime.source_state.snapshot()
        baseline = runtime.evaluate_directed([point], ["1000000"], point_ids=[91])
        inclusive = ordinary.evaluate([point], ["1000000"], point_ids=[91])
        for left, right in zip(
            (baseline.born, *baseline.component_batches),
            (inclusive.born, *inclusive.component_batches),
            strict=True,
        ):
            actual = _values(left, (1, left.columns))[0]
            expected = _values(right, (1, right.columns))[0]
            _close(actual[: len(expected)], expected, Decimal(tolerance))
        ids = [2**64 - 1, 7, 42, 0, 2**63, 9]
        supplied = [
            *vectors,
            _scale(vectors[0], Decimal(2), Decimal(3)),
            [("0", "0")] * 4,
            _scale(vectors[0], 1 + tiny, Decimal(0)),
        ]
        singleton = runtime.evaluate_directed(
            [point] * len(supplied),
            ["1000000"] * len(supplied),
            point_ids=ids,
            spin_vectors={leg: supplied},
        )
        joint = runtime.evaluate_directed(
            [point] * 6,
            ["1000000"] * 6,
            spin_vectors={
                leg: [v for v in vectors for _ in gluons],
                4: [g for _ in vectors for g in gluons],
            },
        )
        assert singleton.point_ids == ids and singleton.spin_legs == [leg]
        assert joint.spin_legs == [leg, 4]
        for single, combined, full in zip(
            _blocks(singleton), _blocks(joint), _blocks(baseline), strict=True
        ):
            values = _values(single, (6, single.columns))
            expected = _values(full, (1, full.columns))[0]
            _close(values[:3].sum(axis=0), expected, Decimal(tolerance))
            _close(values[3], 13 * values[0], Decimal(tolerance))
            _close(values[4], 0 * values[0], Decimal(tolerance))
            _close(
                _values(combined, (6, combined.columns)).sum(axis=0),
                expected,
                Decimal(tolerance),
            )
        born = _values(singleton.born, (6, singleton.born.columns))
        shift = born[5, 0, 0] - born[0, 0, 0]
        expected = born[0, 0, 0] * (2 * tiny + tiny * tiny)
        assert abs(shift - expected) < abs(expected) * Decimal("2e-4")
        # Literal arbitrary four-vectors are checked by exact homogeneity,
        # not by the independent reference's physical three-state projection.
        literal = [("2", "3"), ("5", "-7"), ("11", "13"), ("-17", "19")]
        arbitrary = runtime.evaluate_directed(
            [point] * 2,
            ["1000000"] * 2,
            spin_vectors={leg: [literal, _scale(literal, Decimal(2), Decimal(3))]},
        )
        for block in _blocks(arbitrary):
            values = _values(block, (2, block.columns))
            _close(values[1], 13 * values[0], Decimal(tolerance))
        assert runtime.source_state.snapshot().overrides == snapshot.overrides
        empty = runtime.evaluate_directed(
            [], [], point_ids=[], spin_vectors={leg: [], 4: []}
        )
        assert empty.point_ids == [] and empty.spin_legs == [leg, 4]
        assert all(block.rows == 0 for block in _blocks(empty))
        tail = runtime.evaluate_directed(
            [point], ["1000000"], point_ids=[42], spin_vectors={leg: [vectors[2]]}
        )
        for a, b in zip(_blocks(tail), _blocks(singleton), strict=True):
            _close(
                _values(a, (1, a.columns))[0],
                _values(b, (6, b.columns))[2],
                Decimal(tolerance),
            )


@pytest.mark.parametrize("direction", ["outgoing", "incoming", "negative"])
def test_public_charged_sources_match_independent_physical_projection(
    massive_sources, direction
):
    from pyamplicol import CorrelatedRequest, Runtime

    root, _ = massive_sources
    reference = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/wg_sources.json"
    )
    rows = [
        row
        for row in json.loads(reference.read_text())["rows"]
        if row["direction"] == direction
    ]
    runtime = Runtime.load(root / f"public-{direction}")
    leg = rows[0]["leg"]
    assert set(runtime.inspect()["spin_correlations"]) == {(leg,), (4,), (leg, 4)}
    result = runtime.evaluate_correlated_many(
        [row["point"] for row in rows],
        {"physical": CorrelatedRequest("born", {leg: [row["vector"] for row in rows]})},
        mu_r2="1000000",
    )["physical"]
    for index, row in enumerate(rows):
        for value, expected in zip(
            [result.born_values[index][0], *result.values[index][0]],
            row["cepsilon"],
            strict=True,
        ):
            assert float(value.real) == pytest.approx(expected, rel=8e-9, abs=2e-12)
            assert abs(float(value.imag)) < 2e-12


def test_public_incoming_massive_sources_subset_rescue_and_clear(massive_sources):
    from pyamplicol import CorrelatedRequest, Runtime
    from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy
    from pyamplicol.api.errors import EvaluationError

    root, _ = massive_sources
    runtime = Runtime.load(root / "public-incoming")
    point = json.loads((root / "incoming-ordinary.json").read_text())["point"]
    w = [("2", "3"), ("5", "-7"), ("11", "13"), ("-17", "19")]
    g = [("0", "0"), ("0.8", "0"), ("0", "1"), ("-0.6", "0")]
    points, scales = [point] * 3, ["1000000", "1e-400", "1000000"]
    request = {
        "joint": CorrelatedRequest(
            "born", {2: [w, _scale(w, Decimal(2), Decimal(3)), w], 4: g}
        )
    }
    with pytest.raises(EvaluationError) as failure:
        runtime.evaluate_correlated_many(points, request, mu_r2=scales)
    assert {item.point_id for item in failure.value.failures} == {1}
    rescued = runtime.evaluate_correlated_many(
        points,
        request,
        mu_r2=scales,
        stability=VirtualStabilityPolicy(
            (
                VirtualArithmeticSpec("arbitrary", 60),
                VirtualArithmeticSpec("arbitrary", 80),
            ),
            "1e-8",
            "1e-9",
        ),
    )["joint"]
    assert [item.status for item in rescued.diagnostics] == [
        "success",
        "rescued",
        "success",
    ]
    assert rescued.values[0] == rescued.values[2]
    runtime.set_spin_correlation_vectors({2: w, 4: g})
    first = runtime.evaluate_correlated(points[:1], mu_r2=scales[:1])
    runtime.clear()
    assert (
        runtime.evaluate_correlated(points[:1], mu_r2=scales[:1]).values == first.values
    )
    assert runtime.evaluate_correlated([], mu_r2=[]).shape == (0, 1, 3)
