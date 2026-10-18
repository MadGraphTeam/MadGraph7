# SPDX-License-Identifier: 0BSD
"""Exact local Bose symmetries used only for counterterm witness matching.

FeynKit instantiates each complete vertex in a fixed external-edge frame.
Symbolica checks its exact equality after native interaction slots are swapped.
No current, graph weight or numerical kernel is identified by this proof.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from itertools import pairwise


def symmetric_boson_slots(model, interactions):
    """Return full-permutation slot classes proved on the actual model rules.

    Only identical self-conjugate bosons are candidates. Adjacent transpositions
    generate the full symmetric group; an orbit under a smaller group would not
    justify erasing individual slot labels. All unproved slots remain distinct.
    The complete colour, Lorentz and coupling product is compared, before any
    kinematic, parameter or on-shell substitution.
    """
    from symbolica.community import hepkit as hep

    result = {}
    for name in sorted(set(interactions)):
        rule = model.vertex_rule(name)
        candidates = defaultdict(list)
        for slot, species in enumerate(rule.particles):
            particle = model.particle(species)
            if (
                particle.spin > 0
                and particle.spin % 2 == 1
                and particle.pdg_code == particle.antiparticle.pdg_code
            ):
                candidates[species].append(slot)
        groups = [slots for slots in candidates.values() if len(slots) > 1]
        if not groups:
            continue
        # Public native interchange owns graph construction and instantiates
        # momentum, Lorentz and colour ports together. No hand-written tensor
        # or momentum relabelling is needed.
        diagram = hep.FeynmanDiagram.from_dot(
            model,
            "digraph { ext [style=invis]; v [int_id="
            + json.dumps(name)
            + "];"
            + "".join(
                "ext -> v [particle=" + json.dumps(species) + "];"
                for species in rule.particles
            )
            + "}",
        )
        encoded = diagram.to_dot()
        original_slots = [
            data["target_slot"] for _, data in json.loads(diagram.to_json())["edges"]
        ]
        original = (
            diagram.apply_feynman_rules()
            .vertices[0]
            .numerator_expression()
            .to_expression()
        )
        labels = list(range(len(rule.particles)))
        for slots in groups:
            proved = True
            for left, right in pairwise(slots):

                def swap(match, left=left, right=right):
                    slot = int(match.group(1))
                    target = right if slot == left else left if slot == right else slot
                    return f"target_slot={target}"

                swapped, count = re.subn(r"\btarget_slot=(\d+)", swap, encoded)
                if count != len(rule.particles):
                    raise ValueError(
                        "native vertex interchange omitted interaction slots"
                    )
                transformed = hep.FeynmanDiagram.from_dot(
                    model, swapped
                ).apply_feynman_rules()
                actual_slots = [
                    data["target_slot"]
                    for _, data in json.loads(transformed.to_json())["edges"]
                ]
                expected_slots = [
                    right if slot == left else left if slot == right else slot
                    for slot in original_slots
                ]
                if actual_slots != expected_slots:
                    raise ValueError(
                        "native vertex interchange changed the requested swap"
                    )
                expression = (
                    transformed.vertices[0].numerator_expression().to_expression()
                )
                if (expression - original).expand() != 0:
                    proved = False
                    break
            if proved:
                for slot in slots:
                    labels[slot] = slots[0]
        if labels != list(range(len(labels))):
            result[name] = tuple(labels)
    return result
