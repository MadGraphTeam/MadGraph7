# SPDX-License-Identifier: 0BSD
"""Indexed field bookkeeping using owned UFO rules; no process physics oracle."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyamplicol.loops.flavor_transport import _bose_external_fields
from pyamplicol.loops.wavefunction import match_external_field_counterterms

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def source_plan(loop_sm_ufo):
    """A graph-free loop_sm source owner supplying the real complete UFO rules."""
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.flavor_mappings import RetainedFlavorPlan
    from pyamplicol.models.loop_import import load_raw_loop_model

    prepared = prepare_loop_request_model(
        load_raw_loop_model(loop_sm_ufo),
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    return RetainedFlavorPlan(
        prepared.structure,
        prepared.counterterms,
        tuple(rule.name for rule in prepared.structure.native.vertex_rules),
        ("QCD",),
    )


@pytest.fixture
def field_control(source_plan):
    # The synthetic Born below tests index/multiset bookkeeping, not an amplitude.
    plan = source_plan
    pdgs = (1, -1, 21, 21)
    native = plan.structure.native
    born = SimpleNamespace(
        order=(("QCD", 2),),
        edges=tuple(
            SimpleNamespace(
                external_index=index,
                pdg=pdg,
                spin=native.particle_by_pdg(pdg).spin,
                color=native.particle_by_pdg(pdg).color,
            )
            for index, pdg in enumerate(pdgs)
        ),
    )
    process = SimpleNamespace(
        born=(born,),
        orders=SimpleNamespace(
            correction_names=("QCD",),
            sectors=(SimpleNamespace(pairs=(SimpleNamespace(loop=(("QCD", 4),)),)),),
        ),
    )
    fields = match_external_field_counterterms(
        plan.counterterms.source,
        process,
        selected_corrections=plan.correction_names,
        allowed_amplitude_orders={(("QCD", 4),)},
    ).insertions
    assert fields

    def owner(insertions):
        return SimpleNamespace(
            virtual=SimpleNamespace(
                bare=SimpleNamespace(process=process), external_fields=insertions
            ),
            prepared=SimpleNamespace(metadata=(("external_fields", insertions),)),
        )

    return plan, fields, owner


def test_complete_real_source_fields_keep_their_permuted_index_multiset(field_control):
    plan, fields, owner = field_control
    _bose_external_fields(plan, owner(fields), (0, 1, 3, 2))
    # Reordering records does not alter their multiplicities or identities.
    _bose_external_fields(plan, owner(tuple(reversed(fields))), (0, 1, 3, 2))


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "index", "alternative", "born", "rule"]
)
def test_field_multiset_cannot_hide_incomplete_or_foreign_entries(field_control, fault):
    plan, fields, owner = field_control
    original = fields[0]
    if fault == "missing":
        changed = fields[1:]
    elif fault == "duplicate":
        changed = (*fields, original)
    else:
        if fault == "index":
            value = replace(original, external_index=(original.external_index + 1) % 4)
        elif fault == "alternative":
            value = replace(original, alternative_index=original.alternative_index + 1)
        elif fault == "born":
            value = replace(original, born=SimpleNamespace(**vars(original.born)))
        else:
            value = replace(
                original,
                rule=replace(original.rule, diagnostic=original.rule.diagnostic + 1),
            )
        changed = (value, *fields[1:])
    with pytest.raises(ValueError, match="field"):
        _bose_external_fields(plan, owner(changed), (0, 1, 3, 2))
