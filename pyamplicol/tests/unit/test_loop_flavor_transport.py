# SPDX-License-Identifier: 0BSD
"""Identity-port transport against independently generated source members."""

import json
import os
import time
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.generation.process_grouping import LegMapping
from pyamplicol.loops.flavor_mappings import RetainedFlavorPlan, prove_flavor_relabeling
from pyamplicol.loops.flavor_transport import IdentityFlavorTransport

from .test_loop_physical_correlations import _close, _conjugate, _decimals, _histories


@pytest.fixture(scope="module")
def pair():
    symbolica = pytest.importorskip("symbolica")
    backend = pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path or not symbolica.is_licensed():
        pytest.skip("licensed loop_sm source control required")
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.physical_model_correlations import NativeDirectedModelProgram
    from pyamplicol.models.loop_import import load_raw_loop_model

    prepared = prepare_loop_request_model(
        load_raw_loop_model(path),
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    plan = RetainedFlavorPlan(
        prepared.structure,
        prepared.counterterms,
        tuple(rule.name for rule in prepared.structure.native.vertex_rules),
        ("QCD",),
    )
    proof = prove_flavor_relabeling(plan, {1: 3, 3: 1, -1: -3, -3: -1})
    programs, directed = [], []
    for pdg in (1, 3):
        started = time.monotonic()
        source = prepared.compile(
            (2, -2),
            (pdg, -pdg),
            selected_corrections=["QCD"],
            backend=backend,
            parameters={"aS": "0.118", "MT": "173", "MU_R": "1000"},
            coefficient_engine="jit",
            reduction_engine="fast",
        )
        programs.append(source)
        directed.append(
            NativeDirectedModelProgram(source, _histories(source.virtual.bare)).prepared
        )
        process = source.virtual.bare.process
        print(
            f"independent member {pdg}: {len(process.born)} Born, "
            f"{len(process.loops)} loops, {len(process.families)} families; "
            f"{time.monotonic() - started:.3f}s including directed preparation",
            flush=True,
        )
    representative, member = programs
    relation = IdentityFlavorTransport.admit(
        proof,
        representative,
        member,
        LegMapping((0, 1, 2, 3), False, "exact-loop-species"),
    )
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    points = np.asarray(json.loads(path.read_text())["points"][:2], dtype=str)
    return relation, programs, directed, points


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "4e-11", 20),
        ("double-double", None, "generic", "4e-25", 70),
        ("arbitrary", 70, "generic", "4e-57", 170),
    ],
)
def test_independent_member_all_components_orders_complex_histories_and_exact_updates(
    pair, arithmetic, precision, engine, tolerance, bit
):
    relation, programs, directed, points = pair
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    representative, member = (program.rebind(**options) for program in programs)
    mapped = relation.bind(representative, directed=directed[0])
    independent = directed[1].bind(member)
    assert representative.prepared is relation.representative_plan
    assert member.prepared is relation.member_plan
    assert representative.prepared is not member.prepared
    assert mapped.state is representative.source_bare.state
    assert mapped.external_request.external_pdgs == (2, -2, 3, -3)
    assert not relation.symmetry.external_transport_validated
    assert mapped._directed.model is representative.virtual
    scales, ids = ["1000000", "900000"], [2**64 - 1, 7]
    actual = mapped.evaluate_directed_decimal(points, scales, point_ids=ids)
    expected = independent.evaluate_decimal(points, scales, point_ids=ids)
    tolerance = Decimal(tolerance)
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
    with localcontext() as context:
        context.prec = 120
        for name in names:
            _close(
                _decimals(getattr(actual, name)),
                _decimals(getattr(expected, name)),
                tolerance,
            )
        assert actual.orders == expected.orders == relation.orders
        assert actual.histories == expected.histories
        assert actual.statuses == expected.statuses
        assert all(
            lane == row_id
            for row, row_id in zip(actual.statuses, ids, strict=True)
            for lane, _ in row
        )
        assert abs(Decimal(actual.virtual[0, 2, 0, 0, 1])) > Decimal("1e-4")
        _close(
            _decimals(actual.virtual[:, 2]),
            _conjugate(_decimals(actual.virtual[:, 3])),
            tolerance,
        )
        old = Decimal("0.118")
        alpha = old * (1 + Decimal(2) ** -bit)
        mapped.update_parameters({"aS": str(alpha)})
        member.update_parameters({"aS": str(alpha)})
        updated = mapped.evaluate_directed_decimal(
            points[:1], scales[:1], point_ids=[42]
        )
        reference = independent.evaluate_decimal(points[:1], scales[:1], point_ids=[42])
        for name in names:
            _close(
                _decimals(getattr(updated, name)),
                _decimals(getattr(reference, name)),
                tolerance,
            )
        assert Decimal(updated.born[0, 0, 0, 0]) != Decimal(actual.born[0, 0, 0, 0])
        _close(
            _decimals(updated.born),
            _decimals(actual.born[:1]) * (alpha / old) ** 2,
            tolerance,
        )
        _close(
            _decimals(updated.virtual),
            _decimals(actual.virtual[:1]) * (alpha / old) ** 3,
            tolerance,
        )
    # Existing generation ties reject before publication, through either view.
    snapshot = mapped.state.snapshot()
    with pytest.raises(ValueError, match="fixed during generation"):
        mapped.update_parameters({"aS": "0.12", "MB": "1"})
    assert mapped.state.snapshot() is snapshot
    empty = mapped.evaluate_directed_decimal(points[:0], [], point_ids=[])
    assert empty.virtual.shape == (0, 4, len(relation.orders.virtual), 3, 2)
    assert empty.statuses == ()


def test_transport_ordinary_results_and_retained_owner_guards(pair):
    relation, programs, directed, points = pair
    mapped = relation.bind(programs[0], directed=directed[0])
    actual = mapped.evaluate_ordered_decimal(points[:1], ["1000000"])
    expected = programs[1].evaluate_ordered_decimal(points[:1], ["1000000"])
    for name in (
        "born",
        "virtual",
        "bare",
        "uv",
        "r2",
        "uv_local",
        "uv_external_fields",
    ):
        _close(
            _decimals(getattr(actual, name)),
            _decimals(getattr(expected, name)),
            Decimal("4e-11"),
        )
    with pytest.raises(FrozenInstanceError):
        relation.mapping = LegMapping((1, 0, 2, 3), False, "invalid")
    with pytest.raises(ValueError, match="another prepared representative"):
        relation.bind(programs[1])
    with pytest.raises(ValueError, match="different retained model plan"):
        relation.bind(programs[0], directed=directed[1])
    plain = relation.bind(programs[0])
    with pytest.raises(ValueError, match="no retained directed histories"):
        plain.evaluate_directed_decimal(points[:0], [])
    for mapping in (
        LegMapping((1, 0, 2, 3), False, "permuted"),
        LegMapping((0, 1, 2, 3), True, "conjugated"),
    ):
        with pytest.raises(NotImplementedError, match="unchanged external port order"):
            IdentityFlavorTransport.admit(relation.symmetry, *programs, mapping)
    with pytest.raises(ValueError, match="proved species image"):
        IdentityFlavorTransport.admit(
            relation.symmetry, programs[0], programs[0], relation.mapping
        )
    fake_request = replace(
        programs[0].source_bare.external_request, external_pdgs=(2, -2, 3, -3)
    )
    fake = replace(
        programs[0],
        source_bare=replace(programs[0].source_bare, external_request=fake_request),
    )
    with pytest.raises(ValueError, match="actual currents"):
        IdentityFlavorTransport.admit(
            relation.symmetry, fake, programs[1], relation.mapping
        )
    partial_proof = replace(
        relation.symmetry,
        plan=replace(
            relation.symmetry.plan,
            ordinary_rules=relation.symmetry.plan.ordinary_rules[:-1],
        ),
    )
    with pytest.raises(ValueError, match="complete generation allow-list"):
        IdentityFlavorTransport.admit(partial_proof, *programs, relation.mapping)
