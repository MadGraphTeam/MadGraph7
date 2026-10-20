# SPDX-License-Identifier: 0BSD
"""Published HV rational tensors against the complete independent SU(3) UFO."""

import json
from dataclasses import replace
from itertools import permutations

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


@pytest.mark.parametrize("index", (0, 1))
def test_formal_nc_rule_matches_independent_2009_tensor(source, replacements, index):
    from symbolica import E
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    from pyamplicol.loops.color_bilinears import SUNColorSpace
    from pyamplicol.loops.model import NativeLocalKernel
    from pyamplicol.loops.qcd_r2_color import _metric_components

    rule = replacements[index]
    frame = rule.terms[0].kernel.leg_half_edges
    ports = tuple(h.Symbols.half_edge()(half, 1) for half in frame)
    metric = t.TensorExpression.g(t.Representation.mink(4))
    delta = t.TensorExpression.g(t.Representation.coad(8))
    generator = t.TensorExpression.color_t(8, 3)
    paper, actual = [None] * 3, [None] * 3

    def trace(*legs):
        return t.trace(
            t.Representation.cof(3),
            *(generator(leg, t.AUTO, t.AUTO) for leg in legs),
        )

    def g(a, b, c, d):
        return metric(a, b) * metric(c, d)

    # arXiv:0903.0356v2 Fig. 2, lambda_HV=1. This is a different
    # colour/Lorentz decomposition from the 2020 rule used in production.
    for order in permutations((1, 2, 3)):
        a, b, c, d = (ports[i] for i in (0, *order))
        if index == 0:
            color = 12 * trace(a, b, c, d) / rule.nc
            lorentz = E("5/3") * g(a, c, b, d) - g(a, b, c, d) - g(b, c, a, d)
        else:
            color = (
                delta(a, b) * delta(c, d)
                + delta(a, c) * delta(d, b)
                + delta(a, d) * delta(b, c)
            ) / rule.nc
            color += 16 * (trace(a, c, b, d) + trace(a, d, b, c))
            color -= 7 * (
                trace(a, b, c, d)
                + trace(a, b, d, c)
                + trace(b, a, c, d)
                + trace(b, a, d, c)
            )
            lorentz = g(a, b, c, d)
        local = NativeLocalKernel(color, lorentz, frame, ("g",) * 4)
        for channel, value in enumerate(_metric_components(local)):
            if value != 0:
                term = -rule.nc / 6 * value * color
                paper[channel] = (
                    term if paper[channel] is None else paper[channel] + term
                )
    from symbolica import S

    prefactor = E("1i") * rule.gauge_coupling**4 / (16 * S("symbolica::pi") ** 2)
    for term in rule.terms:
        for channel, value in enumerate(_metric_components(term.kernel)):
            if value != 0:
                value = (
                    (term.coefficient / prefactor).together()
                    * value
                    * term.kernel.color
                )
                actual[channel] = (
                    value if actual[channel] is None else actual[channel] + value
                )
    formal = SUNColorSpace(rule.nc)
    for left, right in zip(paper, actual, strict=True):
        difference = left - right
        assert formal.overlap(difference, difference).together() == 0


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


def test_symbolic_conformance_preserves_active_source_diagnostic(source, replacements):
    from symbolica import S

    from pyamplicol.loops.model import CovariantKernelCatalogue
    from pyamplicol.loops.qcd_r2_color import prepare_qcd_four_gluon_r2
    from pyamplicol.loops.scalar_frontend import SourceScalarDefinition

    model, catalogue = source
    rule = replacements[0]
    first = rule.source_entries[0]
    definition = first.coupling
    diagnostic = S("qcd_r2_test::active_source_diagnostic")
    changed = replace(
        first,
        coupling=SourceScalarDefinition(
            definition.collection,
            definition.id,
            definition.name,
            definition.value,
            diagnostic,
        ),
    )
    entries = tuple(
        changed if entry is first else entry for entry in rule.source_entries
    )
    owner = replace(
        model, kernels=tuple(changed if e is first else e for e in model.kernels)
    )
    updated = CovariantKernelCatalogue(owner, catalogue._template)
    actual = prepare_qcd_four_gluon_r2(
        owner,
        updated,
        entries,
        rule.nc,
        gauge_coupling=rule.gauge_coupling,
        contribution="fundamental-dirac",
        scheme="HV",
    )
    assert actual.source_entries[0] is changed
    assert actual.source_entries[0].scalar_expression(0).diagnostic == diagnostic
    # The record retains the guard; it does not make value-only ingress legal.
    with pytest.raises(NotImplementedError, match="paired scalar value and diagnostic"):
        actual.source_entries[0].scalar_value(0)
    from symbolica.community import hepkit as h

    from pyamplicol.loops.counterterms import (
        CountertermInsertion,
        prepare_counterterm_model,
    )

    prepared = prepare_counterterm_model(
        owner,
        h.Model.qcd_qed(),
        entries=actual.source_entries,
        whole_rule_continuations=[actual],
        vertex_spinor_frame="ufo",
    )
    bound = next(iter(prepared.rules.values()))
    insertion = CountertermInsertion(None, bound, 0, 0, ())
    assert insertion.source_coefficient_expressions(0)[0][1].diagnostic == diagnostic
    assert all(
        value.diagnostic == diagnostic
        for _, value in insertion.coefficient_expressions(0)
    )
    with pytest.raises(NotImplementedError, match="paired scalar value and diagnostic"):
        insertion.coefficient_substitutions(0)


def test_prepared_whole_rule_preserves_source_and_native_tensor(source, replacements):
    from symbolica import E
    from symbolica.community import hepkit as h

    from pyamplicol.loops.counterterms import (
        CountertermInsertion,
        prepare_counterterm_model,
    )

    model, _ = source
    entries = tuple(e for rule in replacements for e in rule.source_entries)
    ordinary = h.Model.qcd_qed()
    prepared = prepare_counterterm_model(
        model,
        ordinary,
        entries=entries,
        whole_rule_continuations=replacements,
        vertex_spinor_frame="ufo",
    )
    assert len(prepared.rules) == 2
    for rule, continuation in zip(prepared.rules.values(), replacements, strict=True):
        assert rule.whole_rule_continuation is continuation
        assert rule.entries == continuation.source_entries
        assert tuple(entry for _, entry in rule.couplings) == rule.entries
        assert rule.order == continuation.source_entries[0].order
        assert len(rule.alternatives) == len(continuation.loop_particles)
        assert rule.has_validated_tensor_basis
        assert not rule.has_ordinary_basis_correspondence
        assert len(rule.native_coefficients) == len(continuation.terms)
        assert rule.active_coupling_symbols == rule.native_coupling_symbols
        assert set(rule.active_coupling_symbols).isdisjoint(rule.coupling_symbols)
        # Relocalize the complete declared tensor through the real native model
        # and compare before any numerical substitution or on-shell projection.
        names = next(
            vertex["particles"]
            for vertex in json.loads(prepared.native.to_json())["vertex_rules"]
            if vertex["name"] == rule.name
        )
        dot = f'digraph local {{ ext [style=invis]; v [int_id="{rule.name}"]; '
        dot += " ".join(f'ext -> v [particle="{name}"];' for name in names) + " }"
        diagram = h.FeynmanDiagram.from_dot(prepared.native, dot).apply_feynman_rules()
        actual = diagram.vertices[0].numerator_expression().to_expression()
        insertion = CountertermInsertion(None, rule, 0, 0, ())
        for symbol, value in insertion.coefficient_expressions(0):
            assert value.diagnostic == 0
            actual = actual.replace(symbol, value.value)
        expected = E("0")
        for term in continuation.terms:
            local = (term.kernel.color * term.kernel.lorentz).to_expression()
            for old, new in zip(
                term.kernel.leg_half_edges, diagram.half_edges, strict=True
            ):
                local = local.replace(
                    h.Symbols.half_edge()(old, 1),
                    h.Symbols.half_edge()(new.id, 1),
                )
            expected += term.coefficient * local
        assert (actual - expected).expand() == 0
        assert all(
            value.value == 0 for _, value in insertion.coefficient_expressions(-1)
        )
        assert len(insertion.source_coefficient_expressions(0)) == len(rule.entries)
        assert tuple(
            value for _, value in insertion.coefficient_substitutions(0)
        ) == tuple(term.coefficient for term in continuation.terms)


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
