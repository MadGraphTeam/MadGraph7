# SPDX-License-Identifier: 0BSD
"""Distinct order sectors retain independent structural colour degrees."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyamplicol.loops.physical_color import project_physical_color


def _fixture(monkeypatch):
    from symbolica import E, S

    from pyamplicol.loops import physical_color

    nc = S("sector_color::Nc")
    b0, b1 = (("QCD", 2),), (("QED", 2),)
    l0, l1 = (("QCD", 4),), (("QCD", 2), ("QED", 2))
    born = ((nc**2 - 1, E("0")), (E("0"), E("1")))
    loop = ((nc**5 + nc**3, nc**3 + nc**2 + nc), (nc**2 + nc, nc + 1))
    bare = SimpleNamespace(
        born_color_tensors=("b0", "b1"),
        loop_color_tensors=("l0", "l1"),
        born_colors=("b0", "b1"),
        born_term_orders=(b0, b1),
        loop_graphs=(SimpleNamespace(order=l0), SimpleNamespace(order=l1)),
        loop_finals=((0,), (1,)),
        initial_average=E("36"),
        order_layout=SimpleNamespace(
            born=tuple(
                SimpleNamespace(pairs=pairs)
                for pairs in (
                    ((b0, b0),),
                    ((b0, b1), (b1, b0)),
                    ((b1, b1),),
                )
            ),
            virtual=tuple(
                SimpleNamespace(
                    pairs=tuple(
                        SimpleNamespace(born=left, loop=right) for left, right in pairs
                    )
                )
                for pairs in (
                    ((b0, l0),),
                    ((b0, l1), (b1, l0)),
                    ((b1, l1),),
                )
            ),
        ),
    )

    def set_metrics(born_metric, loop_metric):
        bare.born_metric, bare.loop_metric = born_metric, loop_metric

    bare.set_color_metrics = set_metrics
    monkeypatch.setattr(
        physical_color,
        "SUNColorSpace",
        lambda _: SimpleNamespace(
            rectangular=lambda left, right: (
                born if right == bare.born_color_tensors else loop
            )
        ),
    )
    return bare, nc, (b0, b1, l0, l1)


@pytest.mark.parametrize("accuracy", ["lc", "nlc", "full"])
def test_ct_poles_share_one_degree_only_within_their_squared_order(
    monkeypatch, accuracy
):
    from symbolica import E

    bare, nc, (b0, b1, l0, l1) = _fixture(monkeypatch)
    projection = project_physical_color(
        bare,
        nc,
        [[nc**8, -(nc**8), nc**4 - nc**2], [nc**2], [nc**99]],
        accuracy,
        additional_pairs=((b0, l1), (b1, l0), (b1, (("QCD", 8),))),
    )
    # The exact-zero Born mixed sector remains on its original axis. A high
    # degree in one virtual sector or a forbidden pairing cannot move another P.
    assert projection.born_degrees == (2, None, 0)
    assert projection.virtual_degrees == (5, 4, 1)
    assert projection.born_degree is projection.virtual_degree is None
    assert bare.born_color_degrees == projection.born_degrees
    assert bare.virtual_color_degrees == projection.virtual_degrees
    expected = {
        "lc": (((9, 0), (0, 1)), ((243, 0), (0, 3)), (81, 0, 0)),
        "nlc": (((8, 0), (0, 1)), ((270, 36), (9, 4)), (72, 9, 0)),
        "full": (((8, 0), (0, 1)), ((270, 39), (12, 4)), (72, 9, 0)),
    }[accuracy]
    assert (
        tuple(tuple(value * 36 for value in row) for row in bare.born_metric)
        == expected[0]
    )
    assert (
        tuple(tuple(value * 36 for value in row) for row in bare.loop_metric)
        == expected[1]
    )
    assert projection.additional_rows == tuple(E(str(value)) for value in expected[2])


def test_additional_multi_order_rows_require_actual_pair_identity(monkeypatch):
    bare, nc, _ = _fixture(monkeypatch)
    with pytest.raises(ValueError, match="exact order pairs"):
        project_physical_color(bare, nc, [[nc]], "lc")
    with pytest.raises(ValueError, match="rows and order pairs differ"):
        project_physical_color(bare, nc, [[nc]], "lc", additional_pairs=())


def test_one_source_vertex_splits_all_actual_alias_orders_and_loop_groups():
    import json
    from types import ModuleType, SimpleNamespace

    from symbolica.community import hepkit as hep

    from pyamplicol.loops.counterterms import prepare_counterterm_model
    from pyamplicol.loops.graphs import _rule_orders
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.models.loop_import import capture_raw_loop_model

    native = hep.Model.qcd_qed()
    module = ModuleType("mixed_ct_order_fixture")
    module.all_particles = [
        SimpleNamespace(**p) for p in json.loads(native.to_json())["particles"]
    ]
    species = {p.pdg_code: p for p in module.all_particles}
    module.all_lorentz = [
        SimpleNamespace(name=f"L{i}", spins=[2, 2, 3], structure="Gamma(3,1,2)")
        for i in range(2)
    ]
    module.all_couplings = [
        SimpleNamespace(name=f"C{i}", value={0: "1"}, order=order)
        for i, order in enumerate(
            (
                {"QCD": 2, "QED": 1},
                {"QED": 3},
                {"QCD": 2, "QED": 1},
            )
        )
    ]
    module.all_vertices = []
    module.all_CTvertices = [
        SimpleNamespace(
            name="mixed",
            particles=[species[-2], species[2], species[22]],
            type="R2",
            color=["Identity(1,2)"],
            lorentz=module.all_lorentz,
            couplings={
                (0, 0, 0): module.all_couplings[0],
                (0, 1, 0): module.all_couplings[1],
                (0, 0, 1): module.all_couplings[2],
            },
            loop_particles=[[[species[2], species[21]]], [[species[2], species[22]]]],
        )
    ]
    source = build_loop_model(capture_raw_loop_model(module))
    prepared = prepare_counterterm_model(source, native)
    actual = _rule_orders(prepared.native)
    assert len(prepared.rules) == 3
    assert {
        (rule.entries[0].loop_group_index, rule.order)
        for rule in prepared.rules.values()
    } == {
        (0, (("QCD", 2), ("QED", 1))),
        (0, (("QED", 3),)),
        (1, (("QCD", 2), ("QED", 1))),
    }
    for rule in prepared.rules.values():
        assert {entry.order for _, entry in rule.couplings} == {rule.order}
        assert (
            tuple(value for value in actual[rule.name] if value[0] != prepared.tag)
            == rule.order
        )
