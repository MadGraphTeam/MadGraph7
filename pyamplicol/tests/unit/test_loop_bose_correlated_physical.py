# SPDX-License-Identifier: 0BSD
"""Actual independent member controls for proved Bose source/history transport.

The retained fixture supplies already generated physical owners. Consistent
omission of both actions is invisible to Bose symmetry; the separate exact
three-cycle recorder checks that coordinate action. Here the momenta-only and
source-only wrong actions must differ from the actual member observable.
"""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.spin_sources import NativeSpinSourcePlan

from .test_loop_physical_correlations import _close, _decimals

RETAINED_BOSE_SOURCES = None
BLOCKS = (
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


@pytest.fixture(scope="module")
def physical_pair():
    if RETAINED_BOSE_SOURCES is None:
        pytest.skip("requires actual independently generated Bose/source owners")
    control = RETAINED_BOSE_SOURCES
    relation = control["relation"].with_bose_correlations(
        *control["programs"],
        representative_directed=control["ordinary"][0],
        member_directed=control["ordinary"][1],
        representative_histories=control["histories"][0],
        member_histories=control["histories"][1],
        representative_spin=control["classes"][0],
        member_spin=control["classes"][1],
        selected_history_ids=control["selected_ids"],
    )
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    points = np.asarray(
        json.loads(path.read_text(), parse_float=str, parse_int=str)["points"][:2],
        dtype=object,
    )
    return control, relation, points[[0, 1, 0]]


def _key(prepared):
    return () if prepared.bare.spin_layout is None else prepared.bare.spin_layout.legs


def _source_rows(source, key, vectors, ids):
    return NativeSpinSourcePlan(
        source.virtual.bare, (key,) if key else ()
    ).prepare_decimal(vectors, ids)


def _selected_columns(control):
    indices = {item.id: index for index, item in enumerate(control["histories"][1])}
    return tuple(indices[name] for name in control["selected_ids"])


def _compare(actual, independent, columns, key, ids, tolerance):
    assert actual.orders == independent.orders
    assert actual.histories == tuple(independent.histories[i] for i in columns)
    assert actual.spin_legs == key
    for name in BLOCKS:
        _close(
            _decimals(getattr(actual, name)),
            _decimals(getattr(independent, name)[:, columns]),
            tolerance,
        )
    assert len(actual.statuses) == len(ids)
    assert all(len(row) == len(actual.channels) for row in actual.statuses)
    assert all(
        row_id == expected and status == "success"
        for row, expected in zip(actual.statuses, ids, strict=True)
        for row_id, status in row
    )
    assert all(0 <= direction < 2 * len(columns) for direction, _, _ in actual.channels)
    _close(
        _decimals(actual.virtual),
        _decimals(actual.first) + _decimals(actual.second),
        tolerance,
    )


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "8e-10", 20),
        ("double-double", None, "generic", "3e-23", 70),
        ("arbitrary", 70, "generic", "8e-54", 170),
    ],
)
def test_actual_bose_sources_histories_all_components_and_both_terms(
    physical_pair,
    arithmetic,
    precision,
    engine,
    tolerance,
    bit,
):
    control, relation, points = physical_pair
    originals = control["programs"]
    snapshots = tuple(source.source_bare.state.snapshot() for source in originals)
    left, right = (
        source.rebind(
            arithmetic=arithmetic, precision=precision, reduction_engine=engine
        )
        for source in originals
    )
    for old, new, snapshot in zip(originals, (left, right), snapshots, strict=True):
        assert new.prepared is old.prepared
        assert new.virtual.bare.backend is old.virtual.bare.backend
        assert old.source_bare.state.snapshot() is snapshot
        new.update_parameters({"aS": "0.118", "MT": "173", "MU_R": "1000"})
    rep = {
        _key(item): item for item in (control["ordinary"][0], *control["classes"][0])
    }
    member = {
        _key(item): item for item in (control["ordinary"][1], *control["classes"][1])
    }
    inverse = tuple(
        relation.mapping.permutation.index(i)
        for i in range(len(relation.mapping.permutation))
    )
    columns = _selected_columns(control)
    ids, scales = [2**64 - 1, 7, 42], ["1000000", "900000", "1000000"]
    vector_a = [["0.1", "0.2"], ["0.8", "-0.3"], ["0.4", "1"], ["-0.6", "0.5"]]
    vector_b = [["-0.3", "0.7"], ["-0.2", "0.8"], ["0.6", "0.1"], ["0.5", "-0.4"]]
    with localcontext() as context:
        context.prec = 120
        tolerance = Decimal(tolerance)
        alpha = str(Decimal("0.118") * (1 + Decimal(2) ** -bit))
        # This is an actual exact global update on independent rebound owners;
        # the retained original snapshots remain untouched.
        for source in (left, right):
            source.update_parameters({"aS": alpha})
        for key in ((), (3,), (4,), (3, 4)):
            representative_key = tuple(sorted(inverse[leg - 1] + 1 for leg in key))
            vectors = {
                leg: [vector_a, vector_b, vector_a] if leg == 3 else vector_b
                for leg in key
            }
            # The member source factory applies its actual SourceIR phase once.
            # Admission proves equality to corresponding representative phases.
            sources = _source_rows(right, key, vectors, ids)
            expected_sources = _source_rows(right, key, vectors, ids)
            mapped = relation.bind(left, directed=rep[representative_key])
            actual = mapped.evaluate_directed_decimal(
                points, scales, point_ids=ids, spin_sources=sources
            )
            expected = (
                member[key]
                .bind(right)
                .evaluate_decimal(
                    points, scales, point_ids=ids, spin_sources=expected_sources
                )
            )
            _compare(actual, expected, columns, key, ids, tolerance)
            for name in BLOCKS:
                assert np.array_equal(
                    getattr(actual, name)[0], getattr(actual, name)[2]
                )
            empty_sources = _source_rows(right, key, {leg: vector_a for leg in key}, [])
            empty = mapped.evaluate_directed_decimal(
                points[:0], [], point_ids=[], spin_sources=empty_sources
            )
            assert empty.born.shape == (0, len(columns), len(relation.orders.born), 2)
            assert empty.virtual.shape == (
                0,
                len(columns),
                len(relation.orders.virtual),
                3,
                2,
            )
            assert empty.statuses == () and empty.spin_legs == key
            if arithmetic == "double" and key == (3,):
                permuted = points[:, relation.mapping.permutation, :]
                # Deliberately inconsistent actions on real prepared programs,
                # not mocked amplitudes. Neither receives a capability claim.
                wrong_momenta_only = (
                    rep[key]
                    .bind(left)
                    .evaluate_decimal(
                        permuted, scales, point_ids=ids, spin_sources=sources
                    )
                )
                wrong_sources_only = (
                    rep[representative_key]
                    .bind(left)
                    .evaluate_decimal(
                        points,
                        scales,
                        point_ids=ids,
                        spin_sources=mapped._spin_sources(sources),
                    )
                )
                witness = _decimals(expected.born[:, columns])
                for wrong in (wrong_momenta_only, wrong_sources_only):
                    difference = _decimals(wrong.born[:, columns]) - witness
                    assert max(abs(value) for value in difference.flat) > Decimal(
                        "1e-6"
                    )
            if arithmetic == "double" and key == (3, 4):
                native = mapped.evaluate_directed_native(
                    points, scales, point_ids=ids, spin_sources=sources
                )
                assert native.batch.point_ids == tuple(ids)
                for name in BLOCKS:
                    values = np.asarray(
                        getattr(native, name).to_decimal(), dtype=object
                    ).reshape(getattr(actual, name).shape)
                    _close(
                        _decimals(values), _decimals(getattr(actual, name)), tolerance
                    )
                assert native.statuses == actual.statuses
        # Later update and different scale reuse the captured source/order plan.
        for source in (left, right):
            source.update_parameters({"aS": "0.12"})
        selected = relation.bind(left, directed=rep[(4,)])
        updated_sources = _source_rows(right, (3,), {3: vector_b}, [91])
        updated = selected.evaluate_directed_decimal(
            points[:1], ["2100000"], point_ids=[91], spin_sources=updated_sources
        )
        independent = (
            member[(3,)]
            .bind(right)
            .evaluate_decimal(
                points[:1],
                ["2100000"],
                point_ids=[91],
                spin_sources=_source_rows(right, (3,), {3: vector_b}, [91]),
            )
        )
        _compare(updated, independent, columns, (3,), [91], tolerance)
    for source, snapshot in zip(originals, snapshots, strict=True):
        assert source.source_bare.state.snapshot() is snapshot
