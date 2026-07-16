# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.processes import enumerate_generic_process_set


def test_generic_selection_accepts_model_driven_pure_singlet_processes() -> None:
    enumeration = enumerate_generic_process_set("e- e+ > mu- mu+")

    assert tuple(entry.process for entry in enumeration.entries) == ("e- e+ > mu- mu+",)


def test_generic_selection_still_rejects_charge_violating_singlets() -> None:
    with pytest.raises(ValueError, match="no valid processes"):
        enumerate_generic_process_set("e- e- > mu- mu+")
