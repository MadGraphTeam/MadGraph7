# SPDX-License-Identifier: 0BSD
"""Actual adjoint-scalar UFO bases; local conformance is not a scheme proof."""

from dataclasses import replace

import pytest

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def prepared(sgluon_nlo_ufo):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("local source tensors require licensed Symbolica")
    path = str(sgluon_nlo_ufo)
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.models.loop_import import load_raw_loop_model

    raw = load_raw_loop_model(path, restriction="full")
    before = raw.to_json()
    result = prepare_loop_request_model(
        raw,
        born_order_bounds={"QCD": (0, 3), "QED": (0, 0), "NP": (0, 0)},
        loop_order_bounds={"QCD": (0, 5), "QED": (0, 0), "NP": (0, 0)},
        covariant_multiplets={
            23: (23, 250, 9000002, -9000002),
            24: (24, 251, 9000003, -9000004),
            -24: (-24, -251, 9000004, -9000003),
        },
    )
    assert raw.to_json() == before
    assert result.original is raw
    return result


def _scalar_rules(prepared):
    return tuple(
        rule
        for rule in prepared.counterterms.rules.values()
        if 9000001 in rule.particle_pdgs
    )


def test_actual_scalar_bases_keep_source_groups_and_scalar_diagnostics(prepared):
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases
    from pyamplicol.loops.model_correspondence import validate_counterterm_primitives

    model = prepared.model
    before = model.raw.to_json()
    evidence = validate_counterterm_primitives(
        model, certified_source_counterterm_bases(model)
    )
    rules = _scalar_rules(prepared)
    # Observed source content, not an implementation bound.
    assert len(rules) == 17
    assert {rule.kind for rule in rules} == {"R2", "UVloop", "UVmass"}
    actual_keys = set()
    for rule in rules:
        assert rule.has_validated_tensor_basis
        assert rule.vertex_spinor_frame == "ufo"
        assert rule.scalar_program is model.scalar_program
        assert not rule.has_ordinary_basis_correspondence
        for entry, validation in zip(
            rule.entries, rule.primitive_validations, strict=True
        ):
            assert any(entry is item for item in model.kernels)
            assert validation == evidence[entry.key]
            assert validation.source_key == entry.key
            assert validation.particle_pdgs == rule.particle_pdgs
            assert (validation.color, validation.lorentz) == (
                entry.color,
                entry.lorentz,
            )
            assert entry.scalar_program is model.scalar_program
            assert entry.scalar_expression().diagnostic == entry.coupling.diagnostic
            assert rule.order == entry.order
            assert rule.alternatives == tuple(
                tuple(
                    model.raw.collections["particles"][i]["fields"]["pdg_code"]
                    for i in group
                )
                for group in entry.loop_particles
            )
            actual_keys.add(entry.key)
    assert len(actual_keys) == 63
    assert model.raw.to_json() == before


def test_np_zero_is_exact_source_selection_not_particle_or_arity_pruning(prepared):
    admitted = prepared.scope.admitted_keys
    assert all(
        dict(entry.order).get("NP", 0) == dict(entry.order).get("QED", 0) == 0
        for entry in prepared.model.kernels
        if entry.key in admitted
    )
    effective = tuple(
        entry
        for entry in prepared.model.kernels
        if len(entry.particle_ids) == 5 and dict(entry.order).get("NP", 0) > 0
    )
    assert effective
    assert all(entry.key not in admitted for entry in effective)
    # The complete source remains available; excluded effective interactions
    # are not deleted or reinterpreted as renormalizable scalar interactions.
    assert prepared.counterterms.source is prepared.model


@pytest.mark.parametrize("lorentz", ["SS1", "SS2", "VSS1", "VVSS1"])
def test_changed_local_tensor_cannot_reuse_source_key_certificate(prepared, lorentz):
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases
    from pyamplicol.loops.model_correspondence import validate_counterterm_primitives

    model = prepared.model
    entry = next(
        e
        for r in _scalar_rules(prepared)
        for e in r.entries
        if e.lorentz_name == lorentz
    )
    validation = next(
        v
        for v in certified_source_counterterm_bases(model)
        if v.source_key == entry.key
    )
    for changed in (
        replace(entry, lorentz=f"2*({entry.lorentz})"),
        replace(entry, color=f"2*({entry.color})"),
    ):
        altered = replace(
            model, kernels=tuple(changed if e is entry else e for e in model.kernels)
        )
        assert entry.key not in {
            v.source_key for v in certified_source_counterterm_bases(altered)
        }
        with pytest.raises(ValueError, match=r"source .* expression differs"):
            validate_counterterm_primitives(altered, [validation])
    if lorentz in ("VSS1", "VVSS1"):
        changed = replace(entry, particle_ids=entry.particle_ids[::-1])
        altered = replace(
            model, kernels=tuple(changed if e is entry else e for e in model.kernels)
        )
        assert entry.key not in {
            v.source_key for v in certified_source_counterterm_bases(altered)
        }
        with pytest.raises(ValueError, match="ordered particle ports differ"):
            validate_counterterm_primitives(altered, [validation])


def test_local_scalar_certificate_depends_on_ports_not_model_or_pdg_name(prepared):
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases
    from pyamplicol.loops.scalar_frontend import lower_scalar_program
    from pyamplicol.models.loop_import import RawLoopModel

    payload = prepared.model.raw.to_dict()
    for particle in payload["collections"]["particles"]:
        if particle["fields"]["pdg_code"] == 9000001:
            particle["fields"]["pdg_code"] = 7654321
    raw = RawLoopModel.from_dict(payload)
    changed = build_loop_model(raw, scalar_program=lower_scalar_program(raw))
    evidence = {v.source_key: v for v in certified_source_counterterm_bases(changed)}
    for rule in _scalar_rules(prepared):
        for entry in rule.entries:
            assert 7654321 in evidence[entry.key].particle_pdgs
            assert 9000001 not in evidence[entry.key].particle_pdgs
