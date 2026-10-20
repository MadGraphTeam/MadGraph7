# SPDX-License-Identifier: 0BSD
"""Model-derived external states and normalization for loop requests.

This boundary prepares physical external data, not a process capability
certificate. Graphs, active local kernels and UV/R2 completeness are checked by
the subsequent model/process compiler.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from math import factorial, prod

from ..processes.model import ufo_goldstone_flag
from .physical_states import ExternalStateLayout, ExternalStateLeg


@dataclass(frozen=True, slots=True)
class LoopExternalRequest:
    incoming: tuple[str, ...]
    outgoing: tuple[str, ...]
    external_pdgs: tuple[int, ...]
    states: ExternalStateLayout
    initial_color_average: int
    identical_final_factor: int

    @property
    def state_kinds(self):
        return tuple(leg.kind for leg in self.states.legs)

    @property
    def initial_average(self):
        """The complete divisor used by the existing physical loop engine."""
        return (
            self.states.initial_spin_average
            * self.initial_color_average
            * self.identical_final_factor
        )

    def to_dict(self):
        """Plain metadata for the existing process artifact, without a new ID."""
        return {
            "incoming": list(self.incoming),
            "outgoing": list(self.outgoing),
            "external_pdgs": list(self.external_pdgs),
            "states": [
                {
                    "kind": leg.kind,
                    "incoming": leg.incoming,
                    "massless": leg.massless,
                    "offset": leg.offset,
                    "components": leg.components,
                    "helicities": list(leg.helicities),
                }
                for leg in self.states.legs
            ],
            "normalization": {
                "initial_spin_average": self.states.initial_spin_average,
                "initial_color_average": self.initial_color_average,
                "identical_final_factor": self.identical_final_factor,
                "divisor": self.initial_average,
                "couplings_in_local_kernels": True,
            },
        }


def prepare_loop_external_request(model, incoming, outgoing):
    """Resolve names/PDGs through the native model, with structural masses.

    An adjustable mass whose default happens to be zero remains massive. The
    exact restriction must be applied before this boundary. The normalization
    agrees with the existing source-model convention: average initial spin and
    colour, divide by identical-final factorials, and retain all local couplings.
    """

    def particles(selectors):
        if isinstance(selectors, (str, bytes)):
            raise TypeError("external particles require a sequence of names or PDGs")
        result = []
        for selector in selectors:
            if isinstance(selector, str):
                particle = model.particle(selector)
            elif type(selector) is int:
                particle = model.particle_by_pdg(selector)
            else:
                raise TypeError("external particles require names or integer PDGs")
            result.append(particle)
        if not result:
            raise ValueError("incoming and outgoing states must both be nonempty")
        return tuple(result)

    initial, final = particles(incoming), particles(outgoing)
    metadata = {p["name"]: p for p in json.loads(model.to_json())["particles"]}
    legs, offset = [], 0
    for index, particle in enumerate((*initial, *final)):
        fields = metadata[particle.name]
        is_initial = index < len(initial)
        if (
            not fields.get("propagating", True)
            or ufo_goldstone_flag(fields)
            or fields.get("ghost_number", 0)
        ):
            raise NotImplementedError(
                "unphysical auxiliary external states are unsupported"
            )
        if abs(particle.color) not in {1, 3, 8}:
            raise NotImplementedError(
                "external colour representation is outside the tree envelope"
            )
        if particle.spin == 1:
            kind = "scalar"
        elif particle.spin == 2:
            if particle.is_self_antiparticle:
                raise NotImplementedError("Majorana external states are unsupported")
            kind = (
                ("v_bar" if particle.is_antiparticle else "u")
                if is_initial
                else ("v" if particle.is_antiparticle else "u_bar")
            )
        elif particle.spin == 3:
            kind = "epsilon" if is_initial else "epsilon_bar"
        else:
            raise NotImplementedError("external spin is outside the tree envelope")
        leg = ExternalStateLeg(index, kind, is_initial, particle.is_massless, offset)
        legs.append(leg)
        offset += leg.components
    return LoopExternalRequest(
        tuple(p.name for p in initial),
        tuple(p.name for p in final),
        tuple(p.pdg_code for p in (*initial, *final)),
        ExternalStateLayout(tuple(legs)),
        prod(abs(p.color) for p in initial),
        prod(factorial(count) for count in Counter(p.pdg_code for p in final).values()),
    )
