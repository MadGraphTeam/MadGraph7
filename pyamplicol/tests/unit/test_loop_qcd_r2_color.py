# SPDX-License-Identifier: 0BSD
"""Published HV rational tensors against the complete independent SU(3) UFO."""

import json
from dataclasses import replace

import pytest

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def source(loop_sm_ufo):
    s = pytest.importorskip("symbolica")
    if not s.is_licensed():
        pytest.skip("exact source tensor conformance requires licensed Symbolica")
    path = str(loop_sm_ufo)
    from ufo_model_loader.model import Model

    from pyamplicol.loops.model import CovariantKernelCatalogue, build_loop_model
    from pyamplicol.models.loop_import import load_raw_loop_model

    source = build_loop_model(load_raw_loop_model(path))
    structural = Model.from_ufo_model(path, include_counterterm_finite_parts=False)
    structural.wrap_indices_in_lorentz_structures()
    catalogue = CovariantKernelCatalogue(source, json.loads(structural.to_json()))
    return source, catalogue


@pytest.fixture(scope="module")
def replacements(source):
    from symbolica import S

    from pyamplicol.loops.qcd_r2_color import prepare_qcd_four_gluon_r2

    model, catalogue = source
    nc, coupling = S("qcd_r2_test::Nc", is_real=True), S("UFO::G")
    return tuple(
        prepare_qcd_four_gluon_r2(
            model,
            catalogue,
            tuple(
                e
                for e in model.kernels
                if e.vertex_name == "V_R2RGA" and e.loop_group_index == group
            ),
            nc,
            gauge_coupling=coupling,
            contribution=kind,
            scheme="HV",
        )
        for group, kind in ((0, "fundamental-dirac"), (1, "gauge"))
    )


def test_complete_source_groups_keep_species_orders_and_exact_coefficients(
    source, replacements
):
    model, _ = source
    assert sum(len(r.source_entries) for r in replacements) == 57
    for replacement in replacements:
        first = replacement.source_entries[0]
        assert replacement.scheme == "HV"
        assert replacement.key == (
            first.collection,
            first.vertex_id,
            first.loop_group_index,
            (("QCD", 4),),
        )
        assert all(
            any(e is old for old in model.kernels) for e in replacement.source_entries
        )
        assert replacement.loop_particles is first.loop_particles
        assert replacement.terms
        assert all(
            t.kernel.particle_names == first.particle_names for t in replacement.terms
        )
    # Each alternative retains its species and occurrence. No sum is replaced by Nf.
    assert len(replacements[0].loop_particles) == 6
    assert len(replacements[1].loop_particles) == 1


def test_source_mismatch_partial_group_and_wrong_scheme_fail(source, replacements):
    from pyamplicol.loops.qcd_r2_color import prepare_qcd_four_gluon_r2

    model, catalogue = source
    rule = replacements[1]
    args = dict(gauge_coupling=rule.gauge_coupling, contribution="gauge", scheme="HV")
    with pytest.raises(ValueError, match="complete original"):
        prepare_qcd_four_gluon_r2(
            model, catalogue, rule.source_entries[:-1], rule.nc, **args
        )
    with pytest.raises(ValueError, match="loop species"):
        prepare_qcd_four_gluon_r2(
            model, catalogue, replacements[0].source_entries, rule.nc, **args
        )
    with pytest.raises(NotImplementedError, match="only for HV"):
        prepare_qcd_four_gluon_r2(
            model, catalogue, rule.source_entries, rule.nc, **{**args, "scheme": "FDH"}
        )
    with pytest.raises(ValueError, match="whole source tensor differs"):
        prepare_qcd_four_gluon_r2(
            model,
            catalogue,
            rule.source_entries,
            rule.nc,
            **{**args, "gauge_coupling": 2 * rule.gauge_coupling},
        )


def test_source_names_do_not_certify_or_block_rule(source, replacements):
    from pyamplicol.loops.model import CovariantKernelCatalogue
    from pyamplicol.loops.qcd_r2_color import prepare_qcd_four_gluon_r2

    model, catalogue = source
    rule = replacements[0]
    entries = tuple(
        replace(e, vertex_name="renamed_local_tensor") for e in rule.source_entries
    )
    by_key = {e.key: e for e in entries}
    renamed = replace(
        model,
        name="different_label",
        kernels=tuple(by_key.get(e.key, e) for e in model.kernels),
    )
    updated = CovariantKernelCatalogue(renamed, catalogue._template)
    actual = prepare_qcd_four_gluon_r2(
        renamed,
        updated,
        entries,
        rule.nc,
        gauge_coupling=rule.gauge_coupling,
        contribution="fundamental-dirac",
        scheme="HV",
    )
    assert actual.key == rule.key
    assert actual.loop_particles == rule.loop_particles


def test_prepared_whole_rule_rejects_partial_mismatched_and_unused_groups(
    source, replacements
):
    from symbolica.community import hepkit as h

    from pyamplicol.loops.counterterms import prepare_counterterm_model

    model, _ = source
    ordinary = h.Model.qcd_qed()
    continuation = replacements[0]
    with pytest.raises(ValueError, match="partial group"):
        prepare_counterterm_model(
            model,
            ordinary,
            entries=continuation.source_entries[:-1],
            whole_rule_continuations=[continuation],
        )
    with pytest.raises(ValueError, match="complete source group"):
        prepare_counterterm_model(
            model,
            ordinary,
            whole_rule_continuations=[
                replace(continuation, source_entries=continuation.source_entries[:-1])
            ],
        )
    with pytest.raises(ValueError, match="not selected"):
        prepare_counterterm_model(
            model,
            ordinary,
            entries=[],
            whole_rule_continuations=[continuation],
        )
    with pytest.raises(ValueError, match="duplicate whole-rule"):
        prepare_counterterm_model(
            model,
            ordinary,
            whole_rule_continuations=[continuation, continuation],
        )
