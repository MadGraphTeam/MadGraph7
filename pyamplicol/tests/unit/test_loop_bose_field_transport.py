# SPDX-License-Identifier: 0BSD
"""Indexed field bookkeeping using owned UFO rules; no process physics oracle."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyamplicol.loops.flavor_transport import _bose_external_fields
from pyamplicol.loops.wavefunction import match_external_field_counterterms

# An existing source-only qualified owner supplies the real complete UFO rules.
# The synthetic Born below tests index/multiset bookkeeping, not an amplitude.
RETAINED_PLAN = None


@pytest.fixture
def field_control():
    if RETAINED_PLAN is None:
        pytest.skip("requires the retained graph-free loop-model source owner")
    plan = RETAINED_PLAN
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
