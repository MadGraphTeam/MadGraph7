# SPDX-License-Identifier: 0BSD
"""``[loop] external_widths = "zero"``: external states are stable.

A width is a property of a propagating unstable particle; an on-shell
external state has none. With ``"zero"`` every external particle of the
request (any concrete process's legs, particle or antiparticle) takes its
model width parameter fixed to zero in the loop preparation, unless the
card's ``[loop.preparation] fixed_parameters`` already names it. The rule
reads the model's particle records (their ``width`` parameter, an external
parameter that is not already zero) and the requested legs: no particle or
process name. Under the complex-mass scheme it keeps external states stable
(``cms-external-width`` otherwise); in the real on-shell scheme it is the
``coloured-width`` / ``electroweak-width`` remedy for the external
particles' own widths. Widths of particles that only propagate are kept.
The default, ``"model"``, uses the model's widths (those guards then name
the parameters to fix).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

EXTERNAL_WIDTH_POLICIES = ("model", "zero")


def stable_external_widths(
    compiled: Any, pdgs: Iterable[int], fixed: Mapping[str, Any]
) -> dict[str, str]:
    """The width parameters of the external particles ``pdgs`` to fix to
    zero: external parameters of the model, neither zero nor named in
    ``fixed``."""

    from pyamplicol.loops.v2.ew_policy import is_zero

    particles: dict[int, Any] = {}
    for record in compiled.ir.particles:
        particles[int(record.pdg_code)] = record
    parameters = {record.name: record for record in compiled.ir.parameters}
    out: dict[str, str] = {}
    for pdg in {int(pdg) for pdg in pdgs}:
        record = particles.get(pdg) or particles.get(-pdg)
        if record is None or record.width is None:
            continue
        width = str(record.width)
        parameter = parameters.get(width)
        if (
            parameter is None
            or parameter.nature != "external"
            or width in fixed
            or is_zero(width, parameters, fixed)
        ):
            continue
        out[width] = "0"
    return dict(sorted(out.items()))


__all__ = ["EXTERNAL_WIDTH_POLICIES", "stable_external_widths"]
