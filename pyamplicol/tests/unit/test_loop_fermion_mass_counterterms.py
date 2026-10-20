# SPDX-License-Identifier: 0BSD
"""Massive Dirac identity two-point counterterms: exact gate and ALOHA control."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.loop_fast

_MASS_VERTICES = {"V_R2CC", "V_R2BB", "V_R2TT", "V_UVcMass", "V_UVbMass", "V_UVtMass"}


def _structure(value, spins=(2, 2)):
    from ufo_model_loader.symbolica_processing import wrap_indices

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    return SimpleNamespace(spins=spins, structure=wrap_indices(parse_exact_ufo(value)))


@pytest.fixture(scope="module")
def source(loop_sm_ufo):
    path = str(loop_sm_ufo)
    from ufo_model_loader.model import Model

    from pyamplicol.loops.model import CovariantKernelCatalogue, build_loop_model
    from pyamplicol.models.loop_import import load_raw_loop_model

    model = build_loop_model(load_raw_loop_model(path))
    structural = Model.from_ufo_model(path, include_counterterm_finite_parts=False)
    structural.wrap_indices_in_lorentz_structures()
    return model, CovariantKernelCatalogue(model, json.loads(structural.to_json()))


@pytest.mark.parametrize("value", ("Identity(1,2)", "Identity(2,1)"))
def test_identity_gate_is_exact_and_two_point_only(value):
    pytest.importorskip("symbolica")
    from pyamplicol.loops.vertex_frames import (
        certified_fermion_mass_source,
        certified_source_dirac_frame,
    )

    assert certified_fermion_mass_source(_structure(value))
    assert certified_source_dirac_frame(_structure(value))
    # The FFS identity keeps its own scalar gate; a scaled identity is a
    # different tensor and is not admitted by this local certificate.
    assert not certified_fermion_mass_source(_structure(value, (2, 2, 1)))
    assert not certified_fermion_mass_source(_structure(f"2*{value}"))


@pytest.mark.parametrize(
    "value",
    (
        "ProjM(2,1)",
        "ProjP(2,1)",
        "ProjM(2,1)+ProjP(2,1)-Identity(2,1)+Gamma5(2,1)",
        "P(-1,1)*Gamma(-1,2,1)",
        "P(-1,2)*Gamma(-1,2,-2)*ProjP(-2,1)",
    ),
)
def test_identity_gate_does_not_infer_chiral_or_momentum_two_points(value):
    pytest.importorskip("symbolica")
    from pyamplicol.loops.vertex_frames import certified_fermion_mass_source

    assert not certified_fermion_mass_source(_structure(value))


def test_actual_mass_two_points_are_certified_without_pdg_or_kind(source):
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases
    from pyamplicol.loops.scalar_frontend import lower_scalar_program
    from pyamplicol.models.loop_import import RawLoopModel

    model, _ = source
    entries = [
        e
        for e in model.kernels
        if e.vertex_name in _MASS_VERTICES and e.lorentz == "Identity(1,2)"
    ]
    # R2 mass parts of c/b/t and the three UV mass counterterms.
    assert {e.vertex_name for e in entries} == _MASS_VERTICES
    evidence = {c.source_key: c for c in certified_source_counterterm_bases(model)}
    for entry in entries:
        record = evidence[entry.key]
        assert (
            record.conformance_control
            == "local-fermion-mass-two-point-source-aloha-control"
        )
        assert record.particle_spins == (2, 2)
        assert record.particle_colors == (-3, 3)
        assert record.particle_pdgs[0] == -record.particle_pdgs[1]
        for changed in (
            replace(entry, lorentz="2*Identity(1,2)"),
            replace(entry, lorentz="ProjM(2,1)"),
            replace(entry, color="2*Identity(1,2)"),
        ):
            changed_model = replace(
                model,
                kernels=tuple(changed if e is entry else e for e in model.kernels),
            )
            assert entry.key not in {
                c.source_key for c in certified_source_counterterm_bases(changed_model)
            }
    payload = model.raw.to_dict()
    for p in payload["collections"]["particles"]:
        if abs(p["fields"]["pdg_code"]) == 6:
            p["fields"]["pdg_code"] = 7654321 * (
                1 if p["fields"]["pdg_code"] > 0 else -1
            )
    raw = RawLoopModel.from_dict(payload)
    renumbered = build_loop_model(raw, scalar_program=lower_scalar_program(raw))
    keys = {e.key for e in entries if e.vertex_name in {"V_R2TT", "V_UVtMass"}}
    for kind in ("UV", "R2", "UVmass"):
        changed = replace(
            renumbered,
            kernels=tuple(
                replace(e, kind=kind) for e in renumbered.kernels if e.key in keys
            ),
        )
        certificates = certified_source_counterterm_bases(changed)
        assert {c.source_key for c in certificates} == keys
        assert all(c.particle_pdgs == (-7654321, 7654321) for c in certificates)


def test_actual_mass_two_points_match_offshell_aloha_in_both_frames(source):
    from symbolica import E
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    from pyamplicol.loops.vertex_frames import native_physical_vertex

    model, catalogue = source
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "tools/loop_reference/fermion_mass_two_point_local.json"
        ).read_text()
    )
    assert fixture["source_lorentz"] == "Identity(1,2)"
    entries = [
        e
        for e in model.kernels
        if e.vertex_name in {"V_R2TT", "V_UVtMass"} and e.lorentz == "Identity(1,2)"
    ]
    assert len(entries) == 2
    for entry in entries:
        kernel = catalogue.lower(entry)
        color = kernel.color.to_tensor().to_dense()
        assert tuple(color.shape) == (3, 3)
        assert all(color[i, j] == int(i == j) for i in range(3) for j in range(3))
        graph = SimpleNamespace(
            edges=tuple(SimpleNamespace(id=i, spin=2) for i in range(2)),
            half_edges=tuple(
                (half, 0, i, "incoming") for i, half in enumerate(kernel.leg_half_edges)
            ),
        )
        physical = native_physical_vertex(graph, 0, kernel.lorentz)
        assert set(physical.axes) == set(kernel.lorentz.axes)

        def contract(expression, states, halves=kernel.leg_half_edges):
            library = t.TensorLibrary.hep_lib_atom()
            contracted = expression
            for leg, (half, components) in enumerate(zip(halves, states, strict=True)):
                axis = next(
                    a
                    for a in expression.axes
                    if a.index == h.Symbols.half_edge()(half, 1)
                ).dual()
                state = t.TensorName(f"fermion_mass_control::state{leg}")
                values = [E(re) + E("1i") * E(im) for re, im in components]
                library.register(t.Tensor.dense(state(axis.representation), values))
                contracted *= state(axis)
            return complex(contracted.to_tensor(library)[0].evaluate({}))

        for row in fixture["rows"]:
            target = complex(*row["expected"])
            first, second = row["first"], row["second"]
            # ALOHA F1/F2 bind the original UFO slots; the physical adapter
            # exchanges them. The identity is endpoint symmetric, so both
            # physical bindings agree; no momentum or projector is involved.
            for value in (
                contract(kernel.lorentz, (first, second)),
                contract(physical, (second, first)),
                contract(physical, (first, second)),
            ):
                assert value == pytest.approx(target, rel=2e-14, abs=2e-15)


def test_quark_scalar_counterterms_use_the_tested_ffs_bases(source):
    """QCD counterterms of quark H/G0/G+- vertices: exact FFS bases only."""
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases

    model, _ = source
    particles = model.raw.collections["particles"]
    entries = [
        e
        for e in model.kernels
        if e.kind != "tree"
        and tuple(particles[i]["fields"]["spin"] for i in e.particle_ids) == (2, 2, 1)
        and tuple(particles[i]["fields"]["color"] for i in e.particle_ids) == (-3, 3, 1)
    ]
    # loop_sm: R2 and UV QCD counterterms of the quark Yukawa/Goldstone vertices.
    assert len(entries) == 44
    evidence = {c.source_key: c for c in certified_source_counterterm_bases(model)}
    for entry in entries:
        assert (
            evidence[entry.key].conformance_control
            == "source-FFS-physical-frame-and-fundamental-color-control"
        )
    # One representative per distinct source tensor for the negative controls.
    for entry in {e.lorentz: e for e in entries}.values():
        for changed in (
            replace(entry, lorentz="2*ProjM(2,1)"),
            replace(entry, lorentz="P(-1,1)*Gamma(-1,2,1)"),
            replace(entry, color="T(3,2,1)"),
        ):
            changed_model = replace(
                model,
                kernels=tuple(changed if e is entry else e for e in model.kernels),
            )
            assert entry.key not in {
                c.source_key for c in certified_source_counterterm_bases(changed_model)
            }
