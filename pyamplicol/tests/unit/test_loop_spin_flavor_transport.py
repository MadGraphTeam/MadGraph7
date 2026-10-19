# SPDX-License-Identifier: 0BSD
"""Paired model-owned sources across independently generated d/s QCD members."""

import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.generation.process_grouping import GroupMember, LegMapping
from pyamplicol.loops.flavor_mappings import RetainedFlavorPlan, prove_flavor_relabeling
from pyamplicol.loops.flavor_transport import (
    AdmittedFlavorMember,
    IdentityFlavorTransport,
)
from pyamplicol.loops.spin_sources import NativeSpinSourcePlan

from .test_loop_physical_correlations import _close, _conjugate, _decimals, _histories


@pytest.fixture(scope="module")
def source_pair(tmp_path_factory):
    symbolica = pytest.importorskip("symbolica")
    source = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not source or not symbolica.is_licensed():
        pytest.skip("licensed loop_sm source control required")
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.model_compatibility import prepare_loop_sm_scalar_declarations
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.physical_model_correlations import NativeDirectedModelProgram
    from pyamplicol.models.loading import compile_model_source

    root = tmp_path_factory.mktemp("spin-flavor-sources")
    model = compile_model_source(
        source, restriction="no_b_mass", cache_dir=root / "cache"
    )
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
    )
    prepared = prepare_loop_request_model(
        model,
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
    programs, ordinary, classes = [], [], []
    for pdg in (1, 3):
        program = prepared.compile(
            (pdg, -pdg),
            (21, 21),
            selected_corrections=["QCD"],
            backend=load_loop_native(),
            parameters={"aS": "0.118", "MT": "173", "MU_R": "1000"},
            coefficient_engine="jit",
            reduction_engine="fast",
        )
        programs.append(program)
        histories = _histories(program.virtual.bare)
        ordinary.append(NativeDirectedModelProgram(program, histories).prepared)
        sources = NativeSpinSourcePlan(program.virtual.bare, ((3,), (4,), (3, 4)))
        classes.append(
            tuple(
                NativeDirectedModelProgram(
                    program, histories, spin_layout=sources.layouts[key]
                ).prepared
                for key in ((3,), (4,), (3, 4))
            )
        )
        print(
            f"INDEPENDENT_SPIN_MEMBER {pdg}: "
            f"{len(program.virtual.bare.process.loops)} loops",
            flush=True,
        )
    options = dict(
        representative_directed=ordinary[0],
        member_directed=ordinary[1],
        representative_spin=classes[0],
        member_spin=classes[1],
    )
    relation = IdentityFlavorTransport.admit(
        proof,
        *programs,
        LegMapping((0, 1, 2, 3), False, "exact-loop-species"),
        **options,
    )
    fixture = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    points = np.asarray(json.loads(fixture.read_text())["points"][:2], dtype=str)
    return relation, programs, ordinary, classes, points


def test_paired_source_admission_rejects_foreign_classes_histories_phases_and_owners(
    source_pair,
):
    relation, programs, ordinary, classes, _ = source_pair
    kwargs = dict(
        representative_directed=ordinary[0],
        member_directed=ordinary[1],
        representative_spin=classes[0],
        member_spin=classes[1],
    )

    def admit(**changes):
        return IdentityFlavorTransport.admit(
            relation.symmetry, *programs, relation.mapping, **(kwargs | changes)
        )

    first = classes[1][0]
    layout = first.bare.spin_layout
    with pytest.raises(ValueError, match="joint source classes"):
        admit(member_spin=classes[1][:-1])
    with pytest.raises(ValueError, match="repeats"):
        admit(member_spin=(*classes[1], first))
    with pytest.raises(ValueError, match="different retained model"):
        admit(member_spin=classes[0])
    with pytest.raises(ValueError, match="different bare"):
        bad = replace(
            first,
            bare=replace(
                first.bare,
                spin_layout=replace(layout, owner=programs[0].virtual.bare.prepared),
            ),
        )
        admit(member_spin=(bad, *classes[1][1:]))
    for bad_layout, message in (
        (
            replace(layout, phase_states=classes[1][-1].bare.spin_layout.phase_states),
            "crossing phase",
        ),
        (replace(layout, contracts=()), "model-owned SourceIR"),
        (replace(layout, helicities=layout.helicities[:-1]), "spectator layout"),
    ):
        with pytest.raises(ValueError, match=message):
            bad = replace(first, bare=replace(first.bare, spin_layout=bad_layout))
            admit(member_spin=(bad, *classes[1][1:]))
    # Equal substituted dataclasses on BOTH sides must not certify a foreign
    # phase or spectator catalog merely by retaining the real owner pointers.
    for field, value, message in (
        ("phase_states", (), "model-owned crossing phase"),
        ("helicities", layout.helicities[:-1], "owner spectator helicities"),
    ):
        altered = []
        for group in classes:
            first = group[0]
            changed = replace(first.bare.spin_layout, **{field: value})
            bare = replace(
                first.bare,
                spin_layout=changed,
                **({"helicities": value} if field == "helicities" else {}),
            )
            altered.append((replace(first, bare=bare), *group[1:]))
        with pytest.raises(ValueError, match=message):
            admit(representative_spin=altered[0], member_spin=altered[1])
    with pytest.raises(ValueError, match="owner helicities"):
        altered = [
            replace(item, bare=replace(item.bare, helicities=())) for item in ordinary
        ]
        admit(representative_directed=altered[0], member_directed=altered[1])
    with pytest.raises(ValueError, match="histories"):
        bad = replace(
            first, bare=replace(first.bare, histories=first.bare.histories[::-1])
        )
        admit(member_spin=(bad, *classes[1][1:]))
    with pytest.raises(TypeError, match="paired directed"):
        admit(member_directed=None)
    with pytest.raises(ValueError, match="ordinary directed"):
        admit(representative_directed=classes[0][0], member_directed=classes[1][0])
    with pytest.raises(ValueError, match="paired source admission"):
        relation.bind(programs[0], directed=replace(classes[0][0]))
    ordinary_only = IdentityFlavorTransport.admit(
        relation.symmetry, *programs, relation.mapping
    )
    with pytest.raises(ValueError, match="paired source admission"):
        ordinary_only.bind(programs[0], directed=classes[0][0])
    # The exact scalar/rule proof is still authoritative, including UV and R2.
    with pytest.raises(ValueError, match=r"counterterm|UV|R2"):
        prove_flavor_relabeling(relation.symmetry.plan, {2: 4, 4: 2, -2: -4, -4: -2})
    member = GroupMember(
        "ssgg",
        " ".join(
            (*relation.member_request.incoming, ">", *relation.member_request.outgoing)
        ),
        (3, -3, 21, 21),
        2,
        generated=False,
        permutation=(0, 1, 2, 3),
    )
    admission = AdmittedFlavorMember(relation, member)
    assert (
        admission.for_program(programs[0].virtual, spin_directed=classes[0][::-1])[
            "external_pdgs"
        ]
        == member.pdgs
    )
    for actual in ((), classes[0][:-1], classes[1], (*classes[0][:2], classes[0][0])):
        with pytest.raises(ValueError, match="saved spin classes"):
            admission.for_program(programs[0].virtual, spin_directed=actual)
    with pytest.raises(ValueError, match="another saved representative"):
        admission.for_program(programs[1].virtual, spin_directed=classes[1])


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "8e-11", 20),
        ("double-double", None, "generic", "8e-25", 70),
        ("arbitrary", 70, "generic", "8e-57", 170),
    ],
)
def test_independent_spin_members_orders_components_histories_and_exact_inputs(
    source_pair, arithmetic, precision, engine, tolerance, bit
):
    relation, programs, ordinary, classes, points = source_pair
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    left, right = (program.rebind(**options) for program in programs)
    ids, scales = [2**64 - 1, 7, 42], ["1000000", "900000", "1000000"]
    points = points[[0, 1, 0]]
    vector_a = [["0.1", "0.2"], ["0.8", "-0.3"], ["0.4", "1"], ["-0.6", "0.5"]]
    vector_b = [["-0.3", "0.7"], ["-0.2", "0.8"], ["0.6", "0.1"], ["0.5", "-0.4"]]
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
        tolerance = Decimal(tolerance)
        alpha = Decimal("0.118") * (1 + Decimal(2) ** -bit)
        for program in (left, right):
            program.update_parameters({"aS": str(alpha)})
        for a, b in zip(
            (ordinary[0], *classes[0]), (ordinary[1], *classes[1]), strict=True
        ):
            key = () if a.bare.spin_layout is None else a.bare.spin_layout.legs
            vectors = {
                leg: [vector_a, vector_b, vector_a] if leg == 3 else vector_b
                for leg in key
            }
            native_sources = [
                NativeSpinSourcePlan(
                    program.virtual.bare, (key,) if key else ()
                ).prepare_decimal(vectors, ids)
                for program in (left, right)
            ]
            mapped = relation.bind(left, directed=a)
            independent = b.bind(right)
            actual = mapped.evaluate_directed_decimal(
                points, scales, point_ids=ids, spin_sources=native_sources[0]
            )
            expected = independent.evaluate_decimal(
                points, scales, point_ids=ids, spin_sources=native_sources[1]
            )
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
            _close(
                _decimals(actual.virtual[:, 2]),
                _conjugate(_decimals(actual.virtual[:, 3])),
                tolerance,
            )
            assert any(
                abs(Decimal(value)) > Decimal("1e-6")
                for value in actual.virtual[:, 2, :, :, 1].flat
            )
            assert np.array_equal(actual.virtual[0], actual.virtual[2])
            empty_sources = NativeSpinSourcePlan(
                left.virtual.bare, (key,) if key else ()
            ).prepare_decimal({leg: vector_a for leg in key}, [])
            empty = mapped.evaluate_directed_decimal(
                points[:0], [], point_ids=[], spin_sources=empty_sources
            )
            assert empty.virtual.shape == (0, 4, len(relation.orders.virtual), 3, 2)
            assert empty.statuses == ()
        snapshot = left.source_bare.state.snapshot()
        with pytest.raises(ValueError, match="fixed during generation"):
            mapped.update_parameters({"aS": "0.12", "MB": "1"})
        assert left.source_bare.state.snapshot() is snapshot
