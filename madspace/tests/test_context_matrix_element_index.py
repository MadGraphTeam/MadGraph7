"""Loading matrix elements at an explicit position of a context."""

import numpy as np
import pytest

import madspace as ms


@pytest.fixture
def context():
    return ms.Context(device=ms.cpu_device(), thread_count=1)


def python_function():
    return ms.PythonFunction(
        lambda momenta, pdg_ids: np.ones(len(momenta)),
        ["momenta", "pdg_ids"],
        ["cut"],
        framework="numpy",
    )


def test_empty_slot_is_an_error(context):
    with pytest.raises(RuntimeError, match="out of bounds"):
        context.matrix_element(0)


def test_failed_load_leaves_context_untouched(context):
    with pytest.raises(Exception):
        context.load_matrix_element("missing_library.so", "param_card.dat", index=2)
    # nothing was reserved: the next matrix element is still the first one
    context.add_matrix_element(python_function())
    assert context.matrix_element(0) is not None
    with pytest.raises(RuntimeError, match="out of bounds"):
        context.matrix_element(1)
