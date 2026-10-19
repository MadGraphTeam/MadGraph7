# SPDX-License-Identifier: 0BSD
"""Actual incoming W+ and outgoing W- declared-colour source composition."""

import pytest

from . import test_loop_massive_projected_sources as _projected_checks
from . import test_loop_massive_vector_sources as _massive_checks
from .test_loop_massive_projected_sources import (
    projected_massive,
)
from .test_loop_massive_projected_sources import (
    test_public_massive_projected_identity_source_rescue as _check_public_rescue,
)
from .test_loop_massive_vector_sources import (
    test_charged_three_state_sources_and_exact_replay as _check_three_states,
)

__all__ = ["projected_massive"]


@pytest.mark.parametrize("direction,leg", [("incoming", 2), ("negative", 3)])
@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "8e-8", 20),
        ("double-double", None, "generic", "8e-22", 70),
        ("arbitrary", 70, "generic", "8e-54", 170),
    ],
)
def test_saved_crossed_projected_three_state_sources(
    projected_massive,
    direction,
    leg,
    accuracy,
    arithmetic,
    precision,
    engine,
    tolerance,
    bit,
):
    root, native = projected_massive
    _check_three_states(
        (root / "crossings" / direction / accuracy, native),
        direction,
        leg,
        arithmetic,
        precision,
        engine,
        tolerance,
        bit,
    )


@pytest.mark.parametrize("direction", ["incoming", "negative"])
def test_full_formal_source_completion_preserves_independent_oracle(
    projected_massive, direction
):
    root, native = projected_massive
    _massive_checks.test_public_charged_sources_match_independent_physical_projection(
        (root / "crossings" / direction / "full", native), direction
    )


@pytest.mark.parametrize("direction,leg", [("incoming", 2), ("negative", 3)])
@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
def test_public_crossed_projected_source_rescue(
    projected_massive, direction, leg, accuracy
):
    root, native = projected_massive
    _check_public_rescue(
        (root / "crossings" / direction, native), accuracy, direction, leg
    )


@pytest.mark.parametrize("direction", ["incoming", "negative"])
@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "1e-8"),
        ("double-double", None, "1e-24"),
        ("arbitrary", 70, "1e-60"),
    ],
)
def test_crossed_projected_resolved_reconstruction(
    projected_massive, direction, accuracy, arithmetic, precision, tolerance
):
    root, native = projected_massive
    _projected_checks.test_public_massive_projected_resolved_reconstruction(
        (root / "crossings" / direction, native),
        accuracy,
        arithmetic,
        precision,
        tolerance,
        direction,
    )
