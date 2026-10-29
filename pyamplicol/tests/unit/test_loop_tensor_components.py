# SPDX-License-Identifier: 0BSD
"""Exact 4D component conventions, independent of a dimensional prescription."""

import pytest

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def components():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("exact native tensor components require licensed Symbolica")
    from symbolica.community import tensor

    return symbolica, tensor


def test_native_epsilon_orientation_metric_and_exact_components(components):
    symbolica, t = components
    from pyamplicol.loops.tensor_components import physical_tensor_library

    labels = symbolica.S(*(f"epsilon_test::{label}" for label in "abcd"))
    epsilon = t.TensorExpression.levi_civita(t.Representation.mink(4))(*labels)
    value = epsilon.to_tensor(physical_tensor_library())
    slots = [t.Representation.mink(4)(label) for label in labels]
    value = value.permute_axes([list(value.axes).index(slot) for slot in slots])
    assert value[0, 1, 2, 3] == -symbolica.E("1i")
    assert value[1, 0, 2, 3] == symbolica.E("1i")
    assert value[0, 1, 2, 2] == 0
    assert all(
        component in (0, symbolica.E("1i"), -symbolica.E("1i")) for component in value
    )
    assert sum(component != 0 for component in value) == 24
    # Includes all four (+---) contraction metrics and Idenso's native -i.
    assert (epsilon * epsilon).to_tensor(physical_tensor_library())[0] == 24
