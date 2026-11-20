# SPDX-License-Identifier: 0BSD
"""Actual adjoint-scalar UFO request scope: NP = 0 is exact source selection."""

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
