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
from collections.abc import Mapping
from itertools import pairwise


def symmetric_boson_slots(model, interactions, *, coupling_expressions=None):
    """Return full-permutation slot classes proved on the actual model rules.

    Only identical self-conjugate bosons are candidates. Adjacent transpositions
    generate the full symmetric group; an orbit under a smaller group would not
    justify erasing individual slot labels. All unproved slots remain distinct.
    The complete colour, Lorentz and coupling product is compared, before any
    kinematic, parameter or on-shell substitution. An optional exact expansion
    of native coupling aliases is useful when the complete vertex carries
    distinct names for equal source coefficients. Its caller must retain the
    checked source scalar owner, including every active diagnostic; this local
    tensor comparison neither evaluates nor discards scalar diagnostics.
    """
    from symbolica import Replacement
    from symbolica.community import hepkit as hep

    from .exact import require_exact_numbers

    replacements = []
    if coupling_expressions is not None:
        if not isinstance(coupling_expressions, Mapping):
            raise TypeError("native coupling expansions require a mapping")
        available = {coupling.name: coupling.symbol for coupling in model.couplings}
        symbols = set(available.values())
        for name, expression in coupling_expressions.items():
            if name not in available:
                raise ValueError(f"unknown native coupling expansion {name!r}")
            require_exact_numbers(expression)
            if set(expression.get_all_symbols(False)) & symbols:
                raise ValueError("native coupling expansions must be fully resolved")
            replacements.append(Replacement(available[name], expression))

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
        if replacements:
            original = original.replace_multiple(replacements)
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
                if replacements:
                    expression = expression.replace_multiple(replacements)
                if (expression - original).expand() != 0:
                    proved = False
                    break
            if proved:
                for slot in slots:
                    labels[slot] = slots[0]
        if labels != list(range(len(labels))):
            result[name] = tuple(labels)
    return result
