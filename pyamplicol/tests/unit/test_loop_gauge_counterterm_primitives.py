# SPDX-License-Identifier: 0BSD
"""Actual local gauge bases and exact source evidence, independent of CT kind."""

from dataclasses import replace

import pytest

from .test_loop_adjoint_scalar_counterterms import prepared as _prepared

prepared = _prepared

_NAMES = (
    "VV1",
    "VV2",
    "VV3",
    "VV4",
    "VV5",
    "VVV2",
    "VVV3",
    "VVV4",
    "VVV5",
    "VVV6",
    "VVV7",
)


def _entries(prepared):
    return tuple(
        e
        for rule in prepared.counterterms.rules.values()
        for e in rule.entries
        if e.lorentz_name in _NAMES and all(p == 21 for p in rule.particle_pdgs)
    )


def test_actual_gauge_bases_preserve_source_scalar_and_loop_groups(prepared):
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases
    from pyamplicol.loops.model_correspondence import validate_counterterm_primitives

    model = prepared.model
    entries = _entries(prepared)
    assert {e.lorentz_name for e in entries} == set(_NAMES)
    before = model.raw.to_json()
    evidence = validate_counterterm_primitives(
        model, certified_source_counterterm_bases(model)
    )
    for entry in entries:
        certificate = evidence[entry.key]
        assert (certificate.color, certificate.lorentz) == (entry.color, entry.lorentz)
        assert entry.scalar_program is model.scalar_program
        assert entry.scalar_expression().diagnostic == entry.coupling.diagnostic
    for rule in prepared.counterterms.rules.values():
        if any(e in entries for e in rule.entries):
            assert rule.has_validated_tensor_basis
            assert rule.scalar_program is model.scalar_program
            assert rule.vertex_spinor_frame == "ufo"
            for entry in rule.entries:
                assert rule.order == entry.order
                assert rule.alternatives == tuple(
                    tuple(
                        sorted(
                            model.raw.collections["particles"][i]["fields"]["pdg_code"]
                            for i in group
                        )
                    )
                    for group in entry.loop_particles
                )
    assert model.raw.to_json() == before


@pytest.mark.parametrize("name", _NAMES)
def test_gauge_certificate_rejects_changed_expression_at_same_key(prepared, name):
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases
    from pyamplicol.loops.model_correspondence import validate_counterterm_primitives

    model = prepared.model
    entry = next(e for e in _entries(prepared) if e.lorentz_name == name)
    original = next(
        c
        for c in certified_source_counterterm_bases(model)
        if c.source_key == entry.key
    )
    for changed in (
        replace(entry, color=f"2*({entry.color})"),
        replace(entry, lorentz=f"2*({entry.lorentz})"),
    ):
        altered = replace(model, kernels=(changed,))
        assert not certified_source_counterterm_bases(altered)
        with pytest.raises(ValueError, match=r"source .* expression differs"):
            validate_counterterm_primitives(altered, [original])


def test_gauge_local_evidence_is_not_uv_r2_or_pdg_specific(prepared):
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases
    from pyamplicol.loops.scalar_frontend import lower_scalar_program
    from pyamplicol.models.loop_import import RawLoopModel

    payload = prepared.model.raw.to_dict()
    for particle in payload["collections"]["particles"]:
        if particle["fields"]["pdg_code"] == 21:
            particle["fields"]["pdg_code"] = 7654321
    raw = RawLoopModel.from_dict(payload)
    renamed = build_loop_model(raw, scalar_program=lower_scalar_program(raw))
    keys = {e.key for e in _entries(prepared)}
    for kind in ("R2", "UV"):
        changed = replace(
            renamed,
            kernels=tuple(
                replace(e, kind=kind) for e in renamed.kernels if e.key in keys
            ),
        )
        certificates = certified_source_counterterm_bases(changed)
        assert {c.source_key for c in certificates} == keys
        assert all(set(c.particle_pdgs) == {7654321} for c in certificates)
