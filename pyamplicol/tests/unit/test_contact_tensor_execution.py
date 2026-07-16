# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.models import compiler_symbolica as _sym
from pyamplicol.models.compiler_contacts import _execute_dense_tensor


def test_rank_zero_plain_expression_bypasses_tensor_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _sym._ensure_symbolica()
    expression = _sym.E("contact_left*contact_right")
    library = _sym.TensorLibrary.hep_lib_atom()

    def unexpected_tensor_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("plain scalar expression must not create a tensor network")

    monkeypatch.setattr(_sym, "TensorNetwork", unexpected_tensor_network)

    assert _execute_dense_tensor(expression, library, axis_labels=()) == (expression,)
