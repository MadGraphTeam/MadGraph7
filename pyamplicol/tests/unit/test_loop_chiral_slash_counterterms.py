# SPDX-License-Identifier: 0BSD
"""Exact chiral-slash admission with original momentum and source ownership."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from .test_loop_adjoint_scalar_counterterms import prepared as _prepared

pytestmark = pytest.mark.loop_fast

prepared = _prepared
_SLASHES = (
    "P(-1,1)*Gamma(-1,2,1)",
    "P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1)",
    "P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1)",
)


def _structure(value, spins=(2, 2)):
    from ufo_model_loader.symbolica_processing import wrap_indices

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    return SimpleNamespace(spins=spins, structure=wrap_indices(parse_exact_ufo(value)))


@pytest.mark.parametrize("value", _SLASHES)
def test_exact_source_slashes_and_only_their_original_ports(value):
    pytest.importorskip("symbolica")
    from pyamplicol.loops.vertex_frames import certified_fermion_slash_source

    assert certified_fermion_slash_source(_structure(value))
    assert not certified_fermion_slash_source(_structure(value, (2, 2, 1)))
    assert not certified_fermion_slash_source(
        _structure(value.replace("P(-1,1)", "P(-1,2)"))
    )
    assert not certified_fermion_slash_source(_structure(f"2*({value})"))


@pytest.mark.parametrize(
    "value",
    (
        "Identity(2,1)",
        "ProjM(2,1)",
        "ProjP(2,1)",
        "P(-1,1)*Gamma(-1,1,2)",
        "P(-1,1)*Gamma(-1,2,1)*ProjM(2,1)",
        "P(-1,1)*ProjM(2,-2)*Gamma(-1,-2,1)",
    ),
)
def test_slash_gate_does_not_infer_other_mass_or_chiral_orderings(value):
    pytest.importorskip("symbolica")
    from pyamplicol.loops.vertex_frames import certified_fermion_slash_source

    assert not certified_fermion_slash_source(_structure(value))


def test_actual_uv_quark_group_has_all_three_independent_certificates(prepared):
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases
    from pyamplicol.loops.model_correspondence import validate_counterterm_primitives

    evidence = validate_counterterm_primitives(
        prepared.model, certified_source_counterterm_bases(prepared.model)
    )
    rule = next(
        r
        for r in prepared.counterterms.rules.values()
        if r.particle_pdgs == (-2, 2)
        and r.kind == "UVloop"
        and {e.lorentz for e in r.entries} == set(_SLASHES)
    )
    assert rule.has_validated_tensor_basis
    assert rule.vertex_spinor_frame == "ufo"
    assert rule.scalar_program is prepared.model.scalar_program
    assert rule.alternatives == ((2, 21),)
    assert rule.order == (("QCD", 2),)
    for entry, certificate in zip(
        rule.entries, rule.primitive_validations, strict=True
    ):
        assert certificate == evidence[entry.key]
        assert entry.scalar_program is prepared.model.scalar_program
        assert entry.scalar_expression().diagnostic == entry.coupling.diagnostic
        assert entry.order == rule.order
        changed = replace(entry, lorentz=entry.lorentz.replace("P(-1,1)", "P(-1,2)"))
        with pytest.raises(ValueError, match="source lorentz expression differs"):
            validate_counterterm_primitives(
                replace(prepared.model, kernels=(changed,)), [certificate]
            )


def test_slash_local_certificate_does_not_depend_on_pdg_or_counterterm_kind(prepared):
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases
    from pyamplicol.loops.scalar_frontend import lower_scalar_program
    from pyamplicol.models.loop_import import RawLoopModel

    original = next(
        r
        for r in prepared.counterterms.rules.values()
        if r.particle_pdgs == (-2, 2)
        and r.kind == "UVloop"
        and {e.lorentz for e in r.entries} == set(_SLASHES)
    )
    keys = {e.key for e in original.entries}
    payload = prepared.model.raw.to_dict()
    for p in payload["collections"]["particles"]:
        if abs(p["fields"]["pdg_code"]) == 2:
            p["fields"]["pdg_code"] = 7654321 * (
                1 if p["fields"]["pdg_code"] > 0 else -1
            )
    raw = RawLoopModel.from_dict(payload)
    source = build_loop_model(raw, scalar_program=lower_scalar_program(raw))
    for kind in ("UV", "R2"):
        changed = replace(
            source,
            kernels=tuple(
                replace(e, kind=kind) for e in source.kernels if e.key in keys
            ),
        )
        certificates = certified_source_counterterm_bases(changed)
        assert {c.source_key for c in certificates} == keys
        assert all(c.particle_pdgs == (-7654321, 7654321) for c in certificates)
