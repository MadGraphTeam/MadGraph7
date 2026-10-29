# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from pyamplicol.loops.counterterms import (
    prepare_counterterm_model,
)
from pyamplicol.loops.model import build_loop_model
from pyamplicol.loops.model_correspondence import (
    CountertermPrimitiveValidation,
    OrdinaryBasisRepresentative,
)
from pyamplicol.models.loop_import import capture_raw_loop_model

pytestmark = pytest.mark.loop_fast


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


def test_complete_source_selection_distinguishes_prepared_and_ir_pruning(hep):
    source, native = _source(hep, [{"kind": "R2"}, {"kind": "UVloop"}])
    complete = prepare_counterterm_model(source, native, vertex_spinor_frame="ufo")
    assert complete.source_entries_complete and not complete.unavailable_entries
    assert all(rule.vertex_spinor_frame == "ufo" for rule in complete.rules.values())
    pruned = prepare_counterterm_model(source, native, entries=source.kernels[:1])
    assert not pruned.source_entries_complete
    pruned_ir = prepare_counterterm_model(
        replace(source, kernels=source.kernels[:1]), native
    )
    assert not pruned_ir.source_entries_complete
    with pytest.raises(ValueError, match="cannot mix native representatives"):
        prepare_counterterm_model(
            source,
            native,
            vertex_spinor_frame="ufo",
            tensor_representatives=(object(),),
        )


def test_prepared_counterterm_trace_is_native_tensor_not_scalar(hep):
    source, ordinary = _source(
        hep,
        [dict(particles=(21, 21), lorentz="Metric(1,2)", color="Tr(1,2)")],
    )
    prepared = prepare_counterterm_model(source, ordinary)
    rule = next(iter(prepared.rules.values()))
    graph = hep.FeynmanDiagram.from_dot(
        prepared.native,
        f'digraph local {{ ext [style=invis]; v [int_id="{rule.name}"]; '
        'ext -> v [particle="g"]; ext -> v [particle="g"]; }',
    ).apply_feynman_rules()
    tensor = graph.vertices[0].numerator_expression()
    assert tensor.rank == 4  # two Lorentz and two adjoint ports
    assert "UFO::Tr" not in tensor.to_canonical_string()
    assert rule.entries[0].color == "Tr(1,2)"


def test_primitive_validation_binds_only_exact_local_source_and_ports(hep):
    specification = dict(particles=(21, 21), lorentz="Metric(1,2)", color="Tr(1,2)")
    source, ordinary = _source(hep, [specification, specification])
    entry = source.kernels[0]
    evidence = CountertermPrimitiveValidation(
        entry.key,
        "Tr(1,2)",
        "Metric(1,2)",
        (21, 21),
        (3, 3),
        (8, 8),
        "local-gluon-two-point-aloha-control",
    )
    prepared = prepare_counterterm_model(
        source, ordinary, primitive_validations=[evidence]
    )
    first, second = prepared.rules.values()
    assert (
        first.has_validated_tensor_basis and not first.has_ordinary_basis_correspondence
    )
    assert not second.has_validated_tensor_basis
    assert first.primitive_validations == (evidence,)
    assert first.entries[0] is entry
    assert first.entries[0].coupling.value.expression == entry.coupling.value.expression
    assert not prepared.tensor_convention_validated
    for changed in (
        replace(evidence, color="2*Tr(1,2)"),
        replace(evidence, lorentz="P(1,1)*P(2,1)"),
        replace(evidence, particle_pdgs=(22, 22)),
        replace(evidence, particle_spins=(1, 1)),
        replace(evidence, particle_colors=(1, 1)),
        replace(evidence, source_key=("counterterm_vertices", 900, 0, 0, 0)),
    ):
        with pytest.raises(ValueError, match="primitive validation"):
            prepare_counterterm_model(source, ordinary, primitive_validations=[changed])
    with pytest.raises(ValueError, match="duplicate"):
        prepare_counterterm_model(
            source, ordinary, primitive_validations=[evidence, evidence]
        )
    with pytest.raises(ValueError, match="provenance"):
        replace(evidence, conformance_control=" ")


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


def test_one_source_vertex_splits_all_actual_alias_orders_and_loop_groups(hep):
    from pyamplicol.loops.graphs import _rule_orders

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
