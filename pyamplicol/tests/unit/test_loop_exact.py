# SPDX-License-Identifier: 0BSD
import pytest

from pyamplicol.loops.exact import require_exact_numbers

pytestmark = pytest.mark.loop_fast


def test_exact_guard_preserves_rational_complex_and_nested_constants():
    sym = pytest.importorskip("symbolica")
    for source in ["1/7", "(2+3i)/11", "sin(1/3)+pi", "f(2/3,x)"]:
        require_exact_numbers(sym.E(source))


def test_exact_guard_rejects_float_kind_even_for_integral_values():
    sym = pytest.importorskip("symbolica")
    function = sym.S("loop_exact_test::f")
    for number in [0.1, 1.0, 1j, sym.Float("0.1", decimal_digits=90)]:
        atom = sym.Expression.num(number)
        for expression in [atom, function(atom)]:
            with pytest.raises(ValueError, match="floating numeric atoms"):
                require_exact_numbers(expression)
