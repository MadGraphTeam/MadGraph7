# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import pytest

from pyamplicol.loops.orders import BornLoopPair
from tools.loop_reference.feynkit_loop_graphs import build_loop_process

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def hep():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native generator worker requires a licensed Symbolica runtime")
    return pytest.importorskip("symbolica.community.hepkit")


@pytest.fixture(scope="module")
def catalogues(hep):
    model = hep.Model.qcd_qed()
    support = {"QCD": 1, "QED": 1}
    return {
        names: build_loop_process(
            model.process(["u", "u~"], ["d", "d~"]),
            selected_corrections=names,
            perturbative_orders=support,
        )
        for names in (("QCD",), ("QED",), ("QCD", "QED"))
    }


def test_qcd_drell_yan_retains_tree_bridge_and_native_local_factors(hep):
    process = hep.Model.qcd_qed().process(["u", "u~"], ["e-", "e+"])
    ir = build_loop_process(
        process, selected_corrections=["QCD"], perturbative_orders={"QCD": 1}
    )
    assert ir.stage == "graph-oracle"
    assert len(ir.born) == len(ir.loops) == len(ir.families) == 1
    graph = ir.loops[0]
    assert len(graph.loop_edges) == 3
    assert len(graph.bridge_edges) == 1
    bridge = next(edge for edge in graph.edges if edge.id in graph.bridge_edges)
    assert bridge.pdg == 22 and bridge.loop_momentum == (0,)
    assert len(ir.families[0].denominators) == 3
    assert ir.orders.sectors[0].order == (("QCD", 2), ("QED", 4))
    assert graph.id == graph.native.id
    assert graph.overall_factor == graph.native.overall_factor
    assert graph.evaluated_factor == "-1"
    assert graph.symmetry_denominator == graph.native.symmetry_factor
    assert ir.convention.internal_bridges == "retained"
    assert (
        ir.convention.external_wavefunction_corrections
        == "excluded-uv-insertions-required"
    )
    assert all(vertex.numerator for vertex in graph.vertices)
    assert all(edge.numerator for edge in graph.edges)
    assert any(edge.directed and edge.spin == 2 for edge in graph.edges)
    # The native default wrongly imposes 1PI for our intended graph catalogue.
    default = process.generate_diagrams(
        loops=1, coupling_orders={"QCD": 2, "QED": 2}, threads=1, progress=None
    )
    assert not default.diagrams


def test_joint_corrections_include_mixed_boxes_and_exact_interferences(catalogues):
    qcd = catalogues[("QCD",)]
    qed = catalogues[("QED",)]
    joint = catalogues[("QCD", "QED")]
    assert [len(ir.loops) for ir in (qcd, qed, joint)] == [16, 15, 35]
    singleton_ids = {g.id for ir in (qcd, qed) for g in ir.loops}
    joint_only = [graph for graph in joint.loops if graph.id not in singleton_ids]
    assert len(joint_only) == 4
    for graph in joint_only:
        assert graph.loop_species == (1, 2, 21, 22)
        assert graph.order == (("QCD", 2), ("QED", 2))
        assert not graph.eligibility.admits(frozenset({"QCD"}))
        assert not graph.eligibility.admits(frozenset({"QED"}))
        assert graph.eligibility.admits(frozenset({"QCD", "QED"}))
    assert len({graph.id for graph in joint.loops}) == 35
    sector = joint.orders.sector((("QCD", 4), ("QED", 2)))
    assert set(sector.pairs) == {
        BornLoopPair((("QCD", 2),), (("QCD", 2), ("QED", 2))),
        BornLoopPair((("QED", 2),), (("QCD", 4),)),
    }
    assert Counter(graph.order for graph in joint.loops) == {
        (("QCD", 4),): 14,
        (("QCD", 2), ("QED", 2)): 8,
        (("QED", 4),): 13,
    }


def test_native_families_share_denominators_without_merging_graphs(catalogues):
    ir = catalogues[("QCD", "QED")]
    # The dependency's qcd_qed model gives every quark and lepton a symbolic
    # mass, so bubbles of different species occupy distinct families.
    assert len(ir.families) == 16
    by_family = defaultdict(list)
    for graph in ir.loops:
        by_family[graph.family.family_index].append(graph)
        target = ir.families[graph.family.family_index].native
        full = graph.native.propagator_family()
        denominators = [
            denominator
            for edge, denominator in zip(
                graph.native.internal_edges, full.denominators, strict=True
            )
            if edge.id in graph.loop_edges
        ]
        for source, target_index in zip(
            denominators, graph.family.denominator_map, strict=True
        ):
            mapped = graph.family.native.apply(source)
            assert (mapped - target.denominators[target_index]).expand() == 0
    assert any(len({g.loop_species for g in group}) > 1 for group in by_family.values())
    bubbles = [graph for graph in ir.loops if len(graph.loop_edges) == 2]
    assert bubbles and all(graph.bridge_edges for graph in bubbles)
    ghost = next(
        graph
        for graph in bubbles
        if any(edge.spin == -1 and edge.id in graph.loop_edges for edge in graph.edges)
    )
    assert "InternalFermionLoopSign(-1)" in ghost.overall_factor
    assert all(edge.directed for edge in ghost.edges if edge.id in ghost.loop_edges)
    assert {item[3] for item in ghost.half_edges} == {"source", "sink"}
    reasons = {item.reason for item in ir.rejected}
    assert reasons == {
        "external-wavefunction-correction",
        "vev-subtracted-tadpole",
        "massless-tadpole",
        "single-coloured-loop-attachment",
    }


def test_explicit_pairs_are_not_inferred_from_born_order_increments(hep):
    pair = BornLoopPair((("QED", 2),), (("QCD", 4),))
    ir = build_loop_process(
        hep.Model.qcd_qed().process(["u", "u~"], ["d", "d~"]),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
        allowed_pairs=[pair],
    )
    assert len(ir.orders.sectors) == 1
    assert ir.orders.sectors[0].pairs == (pair,)


def test_custom_denominator_is_rejected_before_standard_family_construction(hep):
    payload = json.loads(hep.Model.qcd_qed().to_json())
    propagator = next(p for p in payload["propagators"] if p["particle"] == "g")
    propagator["denominator"] = f"({propagator['denominator']})+1"
    model = hep.Model.from_json(json.dumps(payload))
    with pytest.raises(NotImplementedError, match="custom denominator"):
        build_loop_process(
            model.process(["u", "u~"], ["e-", "e+"]),
            selected_corrections=["QCD"],
            perturbative_orders={"QCD": 1},
        )


def test_inhomogeneous_vertex_orders_are_not_replaced_by_native_maxima(hep):
    payload = json.loads(hep.Model.qcd_qed().to_json())
    vertex = next(
        vertex
        for vertex in payload["vertex_rules"]
        if vertex["particles"] == ["g", "g", "g", "g"]
    )
    qed_coupling = next(
        coupling["name"]
        for coupling in payload["couplings"]
        if coupling["orders"] == [["QED", 1]]
    )
    vertex["couplings"][0][0] = qed_coupling
    model = hep.Model.from_json(json.dumps(payload))
    assert model.vertex_rule(vertex["name"]).coupling_orders() == {"QCD": 2, "QED": 1}
    with pytest.raises(NotImplementedError, match="homogeneous coupling orders"):
        build_loop_process(
            model.process(["u", "u~"], ["e-", "e+"]),
            selected_corrections=["QCD"],
            perturbative_orders={"QCD": 1},
        )


_MADLOOP_PROBE = r"""
import json, logging, sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
logging.disable(logging.CRITICAL)
from models import import_ufo
from madgraph.core import base_objects as base
from madgraph.loop import loop_diagram_generation as loop
import_ufo.ReadWrite = False
model = import_ufo.import_model(sys.argv[2], restrict=False)
# The declared full-EW UFO gluon ghost is PDG 9000005. MadLoop changes its
# imported spin to 1, so identifying ghosts by negative spin would lose it.
allowed = {1, 2, 3, 4, 5, 6, 11, 13, 15, 21, 22, 9000005}
model.set('particles', base.ParticleList(
    p for p in model['particles'] if abs(p.get_pdg_code()) in allowed))
model.set('interactions', base.InteractionList(
    v for v in model['interactions'].get_type('base')
    if all(abs(p.get_pdg_code()) in allowed for p in v['particles'])))
model.actualize_dictionaries()
result = {}
for corrections in (['QCD'], ['QED'], ['QCD', 'QED']):
    legs = base.LegList([base.Leg({'id': pdg, 'state': i >= 2})
                        for i, pdg in enumerate((2, -2, 1, -1))])
    process = base.Process({'legs': legs, 'model': model,
        'orders': {'QCD': 4, 'QED': 4}, 'perturbation_couplings': corrections,
        'squared_orders': {'WEIGHTED': 99}})
    amplitude = loop.LoopAmplitude()
    amplitude.set('process', process)
    amplitude.generate_diagrams()
    diagrams = amplitude['loop_diagrams']
    result['+'.join(corrections)] = {
        'born': len(amplitude['born_diagrams']),
        'orders': [[d['orders'].get('QCD', 0), d['orders'].get('QED', 0)]
                   for d in diagrams],
        'mixed_boxes': sum({21, 22} <= d.get_loop_line_types() for d in diagrams)}
print(json.dumps(result))
"""


def test_optional_madloop_graph_selection_matches_native_catalogue(
    loop_qcd_qed_sm_ufo, madgraph_source, catalogues
):
    source = str(madgraph_source)
    model = str(loop_qcd_qed_sm_ufo)
    result = subprocess.run(
        [sys.executable, "-c", _MADLOOP_PROBE, source, model],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    reference = json.loads(result.stdout)
    for names, native in catalogues.items():
        selected = reference["+".join(names)]
        assert len(native.born) == selected["born"]
        assert Counter(
            (dict(g.order).get("QCD", 0), dict(g.order).get("QED", 0))
            for g in native.loops
        ) == Counter(tuple(order) for order in selected["orders"])
        assert (
            sum({21, 22} <= set(g.loop_species) for g in native.loops)
            == selected["mixed_boxes"]
        )
