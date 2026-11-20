# SPDX-License-Identifier: 0BSD
"""Explicit model restrictions for the existing calibrated QCD controls."""

from __future__ import annotations

import json


def qcd_qed_control_model():
    """Return the native QCD/QED model with structurally massless u, d and s.

    The reference controls declare these three flavours massless. Their masses
    must be fixed before graph/family generation, independently of the bundled
    model's adjustable defaults. Charm and bottom retain their symbolic masses
    (bound separately by the controls), and the top mass remains adjustable.
    This does not modify the generic model or infer restrictions from values.
    """
    from symbolica.community import hepkit as h

    payload = json.loads(h.Model.qcd_qed().to_json())
    fixed = {1: "MD", 2: "MU", 3: "MS"}
    for particle in payload["particles"]:
        mass = fixed.get(abs(particle["pdg_code"]))
        if mass is not None and particle["mass"] not in {"ZERO", mass}:
            raise ValueError("QCD control light-flavour mass identity changed")
    for parameter in payload["parameters"]:
        if parameter["name"] in fixed.values():
            parameter.update(
                nature="internal",
                expression="0",
                value=[0, 0],
                lhablock=None,
                lhacode=None,
            )
    payload["restriction"] = "pyamplicol-massless-uds-control"
    # The dependency owns exact-zero propagation into particles, propagator
    # numerators/denominators and topology; no expression rewrite is done here.
    return h.Model.from_json(json.dumps(payload))
