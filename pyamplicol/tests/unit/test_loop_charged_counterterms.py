# SPDX-License-Identifier: 0BSD
"""Charged FFV source rules retain flavour metadata and the tested local frame."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def source(loop_sm_ufo):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("source local tensors require licensed Symbolica")
    path = str(loop_sm_ufo)
    from ufo_model_loader.model import Model

    from pyamplicol.loops.model import CovariantKernelCatalogue, build_loop_model
    from pyamplicol.models.loop_import import load_raw_loop_model

    model = build_loop_model(load_raw_loop_model(path))
    structural = Model.from_ufo_model(path, include_counterterm_finite_parts=False)
    structural.wrap_indices_in_lorentz_structures()
    return model, CovariantKernelCatalogue(model, json.loads(structural.to_json()))


def test_charged_ffv_admission_preserves_actual_flavours_groups_and_scalars(source):
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases

    model, _ = source
    entries = [
        entry
        for entry in model.kernels
        if entry.vertex_name in {"V_R2dxuW", "V_R2uxdW", "V_R2bxtW", "V_R2txbW"}
    ]
    assert len(entries) == 4
    evidence = {
        item.source_key: item for item in certified_source_counterterm_bases(model)
    }
    for entry in entries:
        record = evidence[entry.key]
        assert record.particle_pdgs[0] != -record.particle_pdgs[1]
        assert record.particle_pdgs[0] < 0 < record.particle_pdgs[1]
        assert record.particle_colors == (-3, 3, 1)
        assert record.particle_spins == (2, 2, 3)
        assert record.color == entry.color
        assert record.lorentz == entry.lorentz
        assert entry.order == (("QCD", 2), ("QED", 1))
        # Local tensor admission never changes or collapses the distinct loop
        # species, CKM/scalar factor, Laurent series or perturbative order.
        assert len(entry.loop_particles) == 1
        loop_pdgs = {
            model.raw.collections["particles"][i]["fields"]["pdg_code"]
            for i in entry.loop_particles[0]
        }
        assert loop_pdgs == {abs(p) for p in record.particle_pdgs[:2]} | {21}
        assert entry.coupling.value.coefficient(-1) == 0
        for changed in (
            replace(entry, lorentz="Gamma(3,1,-1)*ProjM(-1,2)"),
            replace(entry, color="2*Identity(1,2)"),
            replace(entry, particle_ids=entry.particle_ids[::-1]),
        ):
            changed_model = replace(
                model,
                kernels=tuple(
                    changed if item is entry else item for item in model.kernels
                ),
            )
            assert entry.key not in {
                item.source_key
                for item in certified_source_counterterm_bases(changed_model)
            }


def test_local_ffv_admission_does_not_infer_endpoint_direction_from_pdg_sign(source):
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases
    from pyamplicol.models.loop_import import RawLoopModel

    model, _ = source
    payload = model.raw.to_dict()
    # Keep conjugate pairs consistent while reversing their arbitrary numeric
    # signs. Tensor slots and ordered colour representations stay unchanged.
    for record in payload["collections"]["particles"]:
        fields = record["fields"]
        if abs(fields["pdg_code"]) in (1, 2, 5, 6):
            fields["pdg_code"] *= -1
    renumbered = build_loop_model(RawLoopModel.from_dict(payload))
    evidence = {
        item.source_key: item for item in certified_source_counterterm_bases(renumbered)
    }
    entries = [
        entry
        for entry in renumbered.kernels
        if entry.vertex_name in {"V_R2bxtW", "V_R2txbW"}
    ]
    assert len(entries) == 2
    for entry in entries:
        record = evidence[entry.key]
        assert record.particle_pdgs[0] > 0 > record.particle_pdgs[1]
        assert record.particle_colors == (-3, 3, 1)


@pytest.mark.parametrize("vertex", ("V_R2bxtW", "V_R2txbW"))
def test_charged_local_vertex_with_unequal_mass_offshell_aloha_currents(source, vertex):
    from symbolica import E
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    from pyamplicol.loops.vertex_frames import native_physical_vertex

    model, catalogue = source
    entry = next(item for item in model.kernels if item.vertex_name == vertex)
    kernel = catalogue.lower(entry)
    graph = SimpleNamespace(
        edges=tuple(
            SimpleNamespace(id=i, spin=spin) for i, spin in enumerate((2, 2, 3))
        ),
        half_edges=tuple(
            (half, 0, i, "incoming") for i, half in enumerate(kernel.leg_half_edges)
        ),
    )
    physical = native_physical_vertex(graph, 0, kernel.lorentz)
    # Colour is the ordered anti-fundamental/fundamental identity, unchanged by
    # spinor endpoint conversion; no assumption about equal flavours is needed.
    color = kernel.color.to_tensor().to_dense()
    assert tuple(color.shape) == (3, 3)
    assert all(color[i, j] == int(i == j) for i in range(3) for j in range(3))
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/charged_ffv_local.json"
    )
    for row in json.loads(fixture.read_text())["rows"]:
        assert row["masses"][0] != row["masses"][1]
        assert row["yukawas"][0] != row["yukawas"][1]

        def contract(expression, names, values):
            library = t.TensorLibrary.hep_lib_atom()
            contracted = expression
            for half, name in zip(kernel.leg_half_edges, names, strict=True):
                axis = next(
                    a
                    for a in expression.axes
                    if a.index == h.Symbols.half_edge()(half, 1)
                ).dual()
                state = t.TensorName(f"charged_ffv_control::{name}")
                components = [
                    E(str(re)) + E("1i") * E(str(im)) for re, im in values[name]
                ]
                library.register(t.Tensor.dense(state(axis.representation), components))
                contracted *= state(axis)
            return complex(contracted.to_tensor(library)[0].evaluate({}))

        target = complex(*row["expected"])
        original = ("first", "second", "vector")
        assert contract(kernel.lorentz, original, row) == pytest.approx(
            target, rel=2e-13, abs=2e-15
        )
        # ALOHA F1/F2 refer to the original UFO endpoint slots. The physical
        # current adapter exchanges those slots, so bind F2/F1 in that frame.
        # This is the existing explicit endpoint convention, not a flavour- or
        # mass-dependent spinor transformation.
        actual = contract(physical, ("second", "first", "vector"), row)
        assert actual == pytest.approx(target, rel=2e-13, abs=2e-15)
        assert abs(contract(physical, original, row) - target) > 1e-6
