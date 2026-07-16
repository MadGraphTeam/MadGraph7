# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.models.contracts import CompiledParticleRecord
from pyamplicol.models.tensors import normalize_color_expression
from pyamplicol.processes.model import ModelParticleCatalog


def _particle(name: str, color: int) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code=9000001,
        spin=1,
        color=color,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )


def test_model_tensors_reject_unsupported_colored_representations() -> None:
    with pytest.raises(ValueError, match="unsupported UFO color representation 6"):
        normalize_color_expression("1", [6, 1, 1])

    with pytest.raises(ValueError, match="particle 'sextet'"):
        ModelParticleCatalog("synthetic", (_particle("sextet", 6),))


def test_model_tensors_keep_supported_singlet_projection() -> None:
    normalized = normalize_color_expression("1", [1, 1, 1])
    assert normalized.expression == "1"
