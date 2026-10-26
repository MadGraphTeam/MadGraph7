# SPDX-License-Identifier: 0BSD
"""Exact admitted source products and the two-point typed endpoint relation."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def prepared(loop_sm_ufo):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("source kernels require licensed Symbolica")
    path = str(loop_sm_ufo)
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.models.loop_import import load_raw_loop_model
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    view = prepare_loop_sm_scalar_declarations(load_raw_loop_model(path))
    return prepare_loop_request_model(
        view.derived,
        born_order_bounds={"QCD": 2, "QED": 2},
        loop_order_bounds={"QCD": 4, "QED": 2},
        fixed_parameters={"MB": "0", "ymb": "0", "MTA": "0", "ymtau": "0"},
    )


def test_neutral_certificates_bind_exact_ordered_source_products(prepared):
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases
    from pyamplicol.loops.model_neutral_qcd import certified_neutral_qcd_basis

    entries = [
        entry
        for entry in prepared.model.kernels
        if entry.vertex_name in {"V_GGZ", "V_GGGZ", "V_GGGA"}
    ]
    assert len(entries) == 8
    certified = {
        item.source_key for item in certified_source_counterterm_bases(prepared.model)
    }
    assert {entry.key for entry in entries} <= certified
    assert all(
        certified_neutral_qcd_basis(entry, prepared.model.raw) for entry in entries
    )
    for entry in entries:
        # Keeping source names and keys does not admit a changed tensor product,
        # a changed local phase or a neutral/coloured port interchange.
        for changed in (
            replace(entry, lorentz="2*(" + entry.lorentz + ")"),
            replace(entry, color="complex(0,1)*(" + entry.color + ")"),
            replace(entry, kind="UV"),
            replace(entry, particle_ids=entry.particle_ids[::-1]),
        ):
            assert not certified_neutral_qcd_basis(changed, prepared.model.raw)


def test_source_slash_typed_frame_equals_calibrated_native_representative(prepared):
    from pyamplicol.loops.model import CovariantKernelCatalogue
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases
    from pyamplicol.loops.vertex_frames import native_physical_vertex

    source = prepared.model
    entry = next(entry for entry in source.kernels if entry.vertex_name == "V_R2UU")
    assert entry.key in {
        item.source_key for item in certified_source_counterterm_bases(source)
    }
    target = replace(entry, lorentz="P(-1,1)*Gamma(-1,1,2)")
    extended = replace(source, kernels=(*source.kernels, target))
    catalogue = CovariantKernelCatalogue(
        extended, json.loads(prepared.structure.native.to_json())
    )
    original, expected = catalogue.lower(entry), catalogue.lower(target)
    assert original.leg_half_edges == expected.leg_half_edges
    graph = SimpleNamespace(
        edges=tuple(SimpleNamespace(id=index, spin=2) for index in range(2)),
        half_edges=tuple(
            (half, 0, index, "incoming")
            for index, half in enumerate(original.leg_half_edges)
        ),
    )
    mapped = native_physical_vertex(graph, 0, original.lorentz)
    assert mapped == expected.lorentz
    assert mapped != original.lorentz
    assert set(mapped.axes) == set(original.lorentz.axes)


@pytest.mark.parametrize(
    "value",
    [
        "P(-1,2)*Gamma(-1,2,1)",
        "P(-1,1)*Gamma(-1,1,2)",
        "Identity(2,1)",
        "P(-1,1)*Gamma(-1,2,1)*ProjM(2,1)",
    ],
)
def test_slash_gate_does_not_infer_other_momenta_chirality_or_mass_terms(value):
    pytest.importorskip("symbolica")
    from ufo_model_loader.symbolica_processing import wrap_indices

    from pyamplicol.loops.model_expressions import parse_exact_ufo
    from pyamplicol.loops.vertex_frames import certified_fermion_slash_source

    assert not certified_fermion_slash_source(
        SimpleNamespace(
            spins=(2, 2),
            structure=wrap_indices(parse_exact_ufo(value)),
        )
    )


def test_neutral_certificates_are_structural_in_the_singlet_vector(prepared):
    """m5 (genericity audit): the certificate attests the exact ordered colour
    and Lorentz tensor product, whichever neutral colour-singlet vector carries
    it (coefficients stay the model's). The Z's epsilon products certify for
    any self-conjugate singlet vector, here the photon, a scope the former
    PDG-23 gate excluded; a charged (non-self-conjugate) vector or a scalar in
    the singlet port does not certify."""

    from pyamplicol.loops.model_neutral_qcd import certified_neutral_qcd_basis

    raw = prepared.model.raw
    index = {
        record["fields"]["name"]: position
        for position, record in enumerate(raw.collections["particles"])
    }
    epsilon = [
        entry
        for entry in prepared.model.kernels
        if entry.vertex_name in {"V_GGZ", "V_GGGZ"} and "Epsilon" in entry.lorentz
    ]
    assert epsilon
    for entry in epsilon:
        assert certified_neutral_qcd_basis(entry, raw)
        coloured = entry.particle_ids[:-1]
        assert certified_neutral_qcd_basis(
            replace(entry, particle_ids=(*coloured, index["A"])), raw
        )
        for other in ("W+", "H"):
            assert not certified_neutral_qcd_basis(
                replace(entry, particle_ids=(*coloured, index[other])), raw
            )
