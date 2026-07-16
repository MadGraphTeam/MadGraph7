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


def test_rank_zero_symbol_detection_is_order_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReorderedSymbols:
        def get_all_symbols(
            self, *, include_function_symbols: bool = True
        ) -> list[str]:
            return ["left", "right"] if include_function_symbols else ["right", "left"]

    expression = ReorderedSymbols()

    def unexpected_tensor_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("symbol order must not create a tensor network")

    monkeypatch.setattr(_sym, "TensorNetwork", unexpected_tensor_network)

    assert _execute_dense_tensor(expression, object(), axis_labels=()) == (expression,)
