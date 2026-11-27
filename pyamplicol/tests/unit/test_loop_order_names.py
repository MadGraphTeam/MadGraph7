# SPDX-License-Identifier: 0BSD
"""Coupling-order names resolve once against the model's declared spelling
(genericity audit M3): the contract of ``loops.order_names`` and the load-time
refusal of two declared names one case apart. The renamed-model harness
(``test_loop_genericity_renamed_models``) checks the decisions and values end
to end."""

from __future__ import annotations

from pathlib import Path

import pytest

from .loop_renamed_models import write_variant

pytestmark = pytest.mark.loop_fast


def test_order_names_resolve_once_in_any_case():
    from pyamplicol.loops.order_names import (
        CouplingOrderNameError,
        CouplingOrderNames,
    )

    names = CouplingOrderNames.of(("QS", "Qe", "NPprop"))
    assert names.resolve("qe") == names.resolve("QE") == "Qe"
    assert names.resolve_all(("npprop", "QS")) == ("NPprop", "QS")
    assert names.orders((("NPPROP", 1), ("QE", 2), ("QS", 1))) == (
        ("NPprop", 1),
        ("QS", 1),
        ("Qe", 2),
    )
    assert names.mapping({"QE": 0, "QS": 2}) == {"QS": 2, "Qe": 0}
    assert names.declared_name("CT_INSERTION") == "CT_INSERTION"
    with pytest.raises(CouplingOrderNameError, match="not a coupling order"):
        names.resolve("QCD")
    with pytest.raises(CouplingOrderNameError, match="differ only in case"):
        CouplingOrderNames.of(("QCD", "qcd"))
    shipped = CouplingOrderNames.of(("QCD", "QED"))
    orders = (("QCD", 2),)
    assert shipped.identity and shipped.orders(orders) is orders
    # A model without a compiled IR has no known spelling: refused, not an
    # identity translation that would hand tree keys to the loop side.
    with pytest.raises(TypeError, match="need a compiled model"):
        CouplingOrderNames.of_model(object())


def test_orders_one_key_apart_are_refused_at_load(loop_sm_ufo, tmp_path):
    from pyamplicol.models.loading import compile_model_source
    from pyamplicol.models.order_names import CouplingOrderNameError

    source = tmp_path / "loop_sm-qs"
    write_variant(Path(loop_sm_ufo), source, "orders")
    path = source / "coupling_orders.py"
    path.write_text(
        path.read_text(encoding="latin-1").replace("'NPprop'", "'qs'"),
        encoding="latin-1",
    )
    with pytest.raises(CouplingOrderNameError, match="'QS' and 'qs'"):
        compile_model_source(source, cache_dir=tmp_path / "cache", use_cache=False)
