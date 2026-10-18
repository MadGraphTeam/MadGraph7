# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from pyamplicol.loops.counterterms import (
    attached_currents,
    match_counterterms,
    prepare_counterterm_model,
)
from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.model import build_loop_model
from pyamplicol.loops.model_correspondence import OrdinaryBasisRepresentative
from pyamplicol.loops.model_expressions import parse_exact_ufo
from pyamplicol.models.loop_import import capture_raw_loop_model, load_raw_loop_model


@pytest.fixture(scope="module")
def hep():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native CT graph generation requires licensed Symbolica")
    return pytest.importorskip("symbolica.community.hepkit")


def _source(hep, specifications, native=None):
    """Small exact graph-selection fixtures; not a physical R2 model."""
    native = hep.Model.qcd_qed() if native is None else native
    payload = json.loads(native.to_json())
    module = ModuleType("ct_selection_fixture")
    module.all_particles = [SimpleNamespace(**p) for p in payload["particles"]]
    species = {p.pdg_code: p for p in module.all_particles}
    module.all_lorentz, module.all_couplings, module.all_CTvertices = [], [], []
    module.all_vertices = []
    for index, spec in enumerate(specifications):
        particles = [species[pdg] for pdg in spec.get("particles", (-2, 2, 22))]
        lorentz = SimpleNamespace(
            name=f"L{index}",
            spins=[p.spin for p in particles],
            structure=spec.get("lorentz", "Gamma(3,1,2)"),
        )
        coupling = SimpleNamespace(
            name=f"C{index}",
            value=spec.get("value", {-1: "a", 0: "b"}),
            order=spec.get("order", {"QCD": 2, "QED": 1}),
        )
        vertex = SimpleNamespace(
            name=f"CT{index}",
            particles=particles,
            type=spec.get("kind", "R2"),
            color=[spec.get("color", "Identity(1,2)")],
            lorentz=[lorentz],
            couplings={(0, 0, 0): coupling},
            loop_particles=[
                [
                    [species[pdg] for pdg in alternative]
                    for alternative in spec.get("alternatives", ((2, 21),))
                ]
            ],
        )
        if "perturbation_type" in spec:
            vertex.perturbation_type = spec["perturbation_type"]
        module.all_lorentz.append(lorentz)
        module.all_couplings.append(coupling)
        if spec.get("kind") == "tree":
            vertex.couplings = {(0, 0): coupling}
            module.all_vertices.append(vertex)
        else:
            module.all_CTvertices.append(vertex)
    return build_loop_model(capture_raw_loop_model(module)), native


def _catalogue(
    prepared,
    incoming=("u", "u~"),
    outgoing=("e-", "e+"),
    corrections=("QCD",),
    **filters,
):
    process = prepared.process(incoming, outgoing, **filters)
    loops = build_loop_process(
        process,
        ordinary_model=prepared.ordinary_model,
        selected_corrections=corrections,
        perturbative_orders={"QCD": 1, "QED": 1},
    )
    ct = prepared.process(incoming, outgoing, counterterms=True, **filters)
    return loops, match_counterterms(
        prepared, ct, loops, selected_corrections=corrections
    )


def test_drell_yan_local_order_exact_coefficients_and_alternative_multiplicity(hep):
    source, ordinary = _source(
        hep,
        [
            {"alternatives": ((2, 21), (2, 21), ())},
            # This equals the full loop order, but not its *local cycle* order.
            {"order": {"QCD": 2, "QED": 2}},
            {"alternatives": ((1, 21),)},
        ],
    )
    prepared = prepare_counterterm_model(source, ordinary)
    loops, ct = _catalogue(prepared)
    assert len(loops.loops) == 1 and len(ct.insertions) == 3
    assert {i.alternative_index for i in ct.insertions} == {0, 1, 2}
    assert len({i.graph.id for i in ct.insertions}) == 1
    for insertion in ct.insertions:
        assert insertion.rule.entries[0].vertex_name == "CT0"
        assert insertion.rule.order == (("QCD", 2), ("QED", 1))
        assert insertion.graph.order == loops.loops[0].order
        assert insertion.witness_loop_ids == (loops.loops[0].id,)
        assert insertion.coefficient_substitutions(-1)[0][1] == parse_exact_ufo("a")
        assert insertion.coefficient_substitutions(0)[0][1] == parse_exact_ufo("b")
        local = (
            next(
                v
                for v in insertion.graph.native.vertices
                if v.id == insertion.vertex_id
            )
            .numerator_expression()
            .to_expression()
        )
        alias = insertion.coefficient_substitutions(0)[0][0]
        assert alias in local.get_all_symbols()
        assert local.replace(alias, 0) == 0
        assert all(prepared.tag not in dict(v.order) for v in insertion.graph.vertices)
    assert not prepared.tensor_convention_validated
    assert (
        not ct.scheme_validated and not ct.external_wavefunction_counterterms_included
    )


def test_uvloop_uses_forbidden_species_but_uvtree_needs_no_loop_witness(hep):
    source, ordinary = _source(
        hep,
        [
            {"kind": "UVloop", "alternatives": ((6,), (1,))},
            {"kind": "UVtree", "alternatives": ((6,), (1,))},
            {"kind": "UV", "alternatives": ((1,),)},
            {"kind": "UVtree", "perturbation_type": "QED"},
        ],
    )
    prepared = prepare_counterterm_model(source, ordinary)
    loops, ct = _catalogue(prepared, particle_veto=["t"])
    assert len(ct.insertions) == 3
    assert {i.rule.entries[0].vertex_name for i in ct.insertions} == {
        "CT0",
        "CT1",
        "CT2",
    }
    assert all(
        i.alternative_index == 1
        for i in ct.insertions
        if i.rule.entries[0].vertex_name != "CT2"
    )
    empty = replace(loops, loops=())
    only_trees = match_counterterms(
        prepared,
        prepared.process(
            ["u", "u~"], ["e-", "e+"], counterterms=True, particle_veto=["t"]
        ),
        empty,
        selected_corrections=["QCD"],
    )
    assert len(only_trees.insertions) == 1
    assert only_trees.insertions[0].rule.kind == "UVtree"
    assert not only_trees.insertions[0].witness_loop_ids


def test_quartic_crossed_loops_match_one_counterterm_without_loop_sign(hep):
    source, ordinary = _source(
        hep,
        [
            {
                "particles": (21, 21, 21, 21),
                "order": {"QCD": 4},
                "color": "Identity(1,2)*Identity(3,4)",
                "lorentz": "Metric(1,2)*Metric(3,4)",
                "alternatives": ((2,),),
            }
        ],
    )
    prepared = prepare_counterterm_model(source, ordinary)
    loops, ct = _catalogue(
        prepared,
        incoming=("g", "g"),
        outgoing=("g", "g"),
        particle_selection=["u", "g"],
    )
    assert len(ct.insertions) == 1
    insertion = ct.insertions[0]
    assert len(insertion.witness_loop_ids) >= 3
    witnesses = [g for g in loops.loops if g.id in insertion.witness_loop_ids]
    assert all("InternalFermionLoopSign(-1)" in g.overall_factor for g in witnesses)
    assert "InternalFermionLoopSign" not in insertion.graph.overall_factor
    assert insertion.graph.evaluated_factor == "1"


def test_joint_correction_selection_is_inherited_from_actual_mixed_loops(hep):
    # Scalar graph-selection model: X is QCD-perturbative, Y QED-perturbative.
    # The mixed X-Y bubble can be admitted only by the joint correction set.
    payload = json.loads(hep.Model.phi4().to_json())
    particle, propagator = payload["particles"][0], payload["propagators"][0]
    payload["particles"] = [
        dict(particle, name=name, antiname=name, pdg_code=pdg, propagator=f"P{name}")
        for name, pdg in (("phi", 100), ("X", 101), ("Y", 102))
    ]
    payload["propagators"] = [
        dict(propagator, name=f"P{name}", particle=name) for name in ("phi", "X", "Y")
    ]
    payload["orders"] = [
        dict(name=name, expansion_order=99, hierarchy=1) for name in ("QCD", "QED")
    ]
    payload["lorentz_structures"] = [
        dict(name="SCALAR", spins=[1, 1, 1], structure="1")
    ]
    payload["couplings"], payload["vertex_rules"] = [], []
    for name, particles, order in (
        ("VX", ["X"] * 3, [["QCD", 1]]),
        ("VY", ["Y"] * 3, [["QED", 1]]),
        ("VP", ["phi"] * 3, [["QED", 1]]),
        ("VPX", ["phi", "X", "X"], [["QCD", 1]]),
        ("VM", ["phi", "X", "Y"], [["QCD", 1], ["QED", 1]]),
    ):
        payload["couplings"].append(
            dict(name=f"C{name}", expression="1", orders=order, value=None)
        )
        payload["vertex_rules"].append(
            dict(
                name=name,
                particles=particles,
                color_structures=["1"],
                lorentz_structures=["SCALAR"],
                couplings=[[f"C{name}"]],
            )
        )
    native = hep.Model.from_json(json.dumps(payload))
    source, ordinary = _source(
        hep,
        [
            {
                "particles": (100, 100),
                "order": {"QCD": 2, "QED": 2},
                "color": "1",
                "lorentz": "1",
                "alternatives": ((101, 102),),
            }
        ],
        native=native,
    )
    prepared = prepare_counterterm_model(source, ordinary)
    _, qcd = _catalogue(prepared, incoming=("phi", "phi"), outgoing=("phi", "phi"))
    _, joint = _catalogue(
        prepared,
        incoming=("phi", "phi"),
        outgoing=("phi", "phi"),
        corrections=("QCD", "QED"),
    )
    assert not qcd.insertions and len(joint.insertions) == 3
    assert all(i.witness_loop_ids for i in joint.insertions)


def test_rooted_current_identity_normalizes_only_artificial_boundary(hep):
    payload = json.loads(hep.Model.qcd_qed().to_json())
    ordinary = next(v for v in payload["vertex_rules"] if v["name"] == "V_98")
    payload["vertex_rules"].append(dict(ordinary, name="V_98_ALTERNATIVE"))
    model = hep.Model.from_json(json.dumps(payload))

    def currents(interaction="V_98", reverse=False, swap_labels=False):
        photon = "current -> root" if reverse else "root -> current"
        electrons = ("e+", "e-") if swap_labels else ("e-", "e+")
        diagram = hep.FeynmanDiagram.from_dot(
            model,
            f'''digraph {{
          ext [style=invis]; root [int_id="V_132"]; current [int_id="{interaction}"];
          ext -> root [particle="u"]; ext -> root [particle="u~"];
          current -> ext [particle="{electrons[0]}"];
          current -> ext [particle="{electrons[1]}"];
          {photon} [particle="a"];
        }}''',
        )
        return attached_currents(
            diagram, [next(v.id for v in diagram.vertices if v.interaction == "V_132")]
        )

    assert currents() == currents(reverse=True)
    # Same physical support is insufficient: internal rule or external state
    # placement changes must survive the native graph canonicalization.
    assert currents() != currents(interaction="V_98_ALTERNATIVE")
    assert currents() != currents(swap_labels=True)


def test_ordinary_species_metadata_ignores_counterterm_tag_and_effective_orders(hep):
    source, ordinary = _source(hep, [{"order": {}}])
    prepared = prepare_counterterm_model(source, ordinary)
    loops, _ = _catalogue(prepared, outgoing=("d", "d~"))
    assert len(loops.loops) == 16
    # CT tags themselves must never become perturbative species membership.
    assert all(
        prepared.tag not in names
        for graph in loops.loops
        for names in graph.eligibility.species_orders
    )
    without_ordinary_metadata = build_loop_process(
        prepared.process(["u", "u~"], ["d", "d~"]),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
    )
    assert any(
        prepared.tag in names
        for graph in without_ordinary_metadata.loops
        for names in graph.eligibility.species_orders
    )


def test_fingerprint_and_correction_selection_reject_mismatched_catalogues(hep):
    source, ordinary = _source(hep, [{}])
    prepared = prepare_counterterm_model(source, ordinary)
    loops, _ = _catalogue(prepared)
    process = prepared.process(["u", "u~"], ["e-", "e+"], counterterms=True)
    with pytest.raises(ValueError, match="correction selection"):
        match_counterterms(prepared, process, loops, selected_corrections=["QED"])
    foreign = build_loop_process(
        ordinary.process(["u", "u~"], ["e-", "e+"]),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
    )
    with pytest.raises(ValueError, match="fingerprint"):
        match_counterterms(prepared, process, foreign, selected_corrections=["QCD"])


def test_real_loop_sm_drell_yan_selects_model_r2_without_uv(hep):
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if path is None:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE to MadLoop loop_sm")
    source = build_loop_model(load_raw_loop_model(path))
    prepared = prepare_counterterm_model(source, hep.Model.qcd_qed())
    loops, ct = _catalogue(prepared)
    assert len(loops.loops) == len(ct.insertions) == 1
    insertion = ct.insertions[0]
    assert insertion.rule.entries[0].vertex_name == "V_R2uuA"
    assert insertion.rule.entries[0].coupling.name == "R2_UUA"
    assert insertion.rule.kind == "R2"
    assert len(prepared.rules) == 74 and prepared.unavailable_entries


def _correspondence_source(hep):
    return _source(
        hep,
        [
            {
                "kind": "tree",
                "order": {"QED": 1},
                "value": "2*complex(0,1)*ee/3",
                "lorentz": "Gamma(3,2,1)",
            },
            {"lorentz": "Gamma(3,2,1)"},
            {"lorentz": "2*Gamma(3,2,1)"},
        ],
    )


def test_explicit_ordinary_correspondence_requires_exact_source_basis_identity(hep):
    source, ordinary = _correspondence_source(hep)
    tree = next(entry for entry in source.kernels if entry.kind == "tree")
    representative = OrdinaryBasisRepresentative(
        tree.key, "V_132", 0, 0, "native-qcd-qed-vector-current-control"
    )
    prepared = prepare_counterterm_model(
        source, ordinary, ordinary_basis_representatives=[representative]
    )
    _, ct = _catalogue(prepared)
    matched = next(i for i in ct.insertions if i.rule.entries[0].vertex_name == "CT1")
    unmatched = next(i for i in ct.insertions if i.rule.entries[0].vertex_name == "CT2")
    assert matched.rule.has_ordinary_basis_correspondence
    assert not unmatched.rule.has_ordinary_basis_correspondence
    assert matched.rule.basis_correspondences[0].representative == representative
    assert matched.rule.entries[0].lorentz == "Gamma(3,2,1)"
    assert (
        matched.rule.basis_correspondences[0].lorentz
        == "Gamma(idx(1,3),idx(1,1),idx(1,2))"
    )
    assert matched.coefficient_substitutions(0)[0][1] == parse_exact_ufo("b")
    # Bind both ordinary and CT through the native Feynman-rule interface in
    # identical local ports. Their geometric tensors agree with couplings removed.
    tensors = []
    for name, alias in (
        ("V_132", ordinary.coupling("GC_2").symbol),
        (matched.rule.name, matched.rule.coupling_symbols[0]),
    ):
        graph = hep.FeynmanDiagram.from_dot(
            prepared.native,
            f'''digraph {{
          ext [style=invis]; v [int_id="{name}"];
          ext -> v [particle="u~"]; ext -> v [particle="u"]; ext -> v [particle="a"];
        }}''',
        ).apply_feynman_rules()
        tensors.append(
            graph.vertices[0].numerator_expression().to_expression().replace(alias, 1)
        )
    assert (tensors[0] - tensors[1]).expand() == 0
    assert not prepared.tensor_convention_validated


def test_ordinary_correspondence_rejects_wrong_ports_scalar_or_missing_provenance(hep):
    source, ordinary = _correspondence_source(hep)
    tree = next(entry for entry in source.kernels if entry.kind == "tree")
    good = OrdinaryBasisRepresentative(
        tree.key, "V_132", 0, 0, "vector-current-control"
    )
    for bad, message in (
        (replace(good, native_rule="V_71"), "ordered particle ports"),
        (replace(good, color_index=1), "basis slot"),
        (replace(good, calibration=""), "calibration provenance"),
        (replace(good, source_key=source.counterterms("R2")[0].key), "source ordinary"),
    ):
        with pytest.raises(ValueError, match=message):
            prepare_counterterm_model(
                source, ordinary, ordinary_basis_representatives=[bad]
            )
    payload = json.loads(ordinary.to_json())
    next(c for c in payload["couplings"] if c["name"] == "GC_2")["expression"] = "1/7"
    with pytest.raises(ValueError, match="exact coupling expression"):
        prepare_counterterm_model(
            source,
            hep.Model.from_json(json.dumps(payload)),
            ordinary_basis_representatives=[good],
        )
    payload = json.loads(ordinary.to_json())
    tensor = next(
        item for item in payload["lorentz_structures"] if item["name"] == "FFV1"
    )
    tensor["structure"] = "0.1*(" + tensor["structure"] + ")"
    with pytest.raises(ValueError, match="floating numeric atoms"):
        prepare_counterterm_model(
            source,
            hep.Model.from_json(json.dumps(payload)),
            ordinary_basis_representatives=[good],
        )


def test_real_loop_sm_drell_yan_source_basis_correspondence(hep):
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if path is None:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE to MadLoop loop_sm")
    source = build_loop_model(load_raw_loop_model(path))
    tree = next(
        e
        for e in source.kernels
        if e.kind == "tree" and e.particle_names == ("u~", "u", "A")
    )
    representative = OrdinaryBasisRepresentative(
        tree.key, "V_132", 0, 0, "native-qcd-qed-vector-current-control"
    )
    prepared = prepare_counterterm_model(
        source, hep.Model.qcd_qed(), ordinary_basis_representatives=[representative]
    )
    _, ct = _catalogue(prepared)
    assert len(ct.insertions) == 1
    insertion = ct.insertions[0]
    assert insertion.rule.has_ordinary_basis_correspondence
    assert insertion.rule.entries[0].coupling.name == "R2_UUA"
    assert insertion.rule.basis_correspondences[0].representative.source_key == tree.key
