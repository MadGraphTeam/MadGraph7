# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pyamplicol.generation.physics_metadata import _color_id


def test_public_lc_color_ids_are_physical_and_sector_independent() -> None:
    assert _color_id(()) == "flow:singlet"
    assert _color_id((2, 4, 1)) == "flow:2,4,1"
