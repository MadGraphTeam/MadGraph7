"""The MLM / CKKW-L clustering diagrams of [amplicol] events.

An [amplicol] event comes from one leading-colour flow, so it can only be
clustered along the tree diagrams contributing to that flow: the planar diagrams
of its colour order. They are enumerated here from the model's 3-point vertices
(export_amplicol.clustering_model, in CLUSTERING_MODEL) within the
coupling orders of the process. Vertices with more legs are left out: the
lines of a 4-point vertex that two 3-point vertices joined by a propagator
reproduce (the 4-gluon, VVVV, VVHH, HHHH vertices of the SM) are lines of some
3-point diagram, but a contact interaction (a four-fermion or ggHH operator,
say) is not, and the topologies only it makes cannot be clustered
(VertexTable.contact_vertices).

A line of a leading-colour diagram carries as many colour lines across it as its
colour representation: none for a singlet, one colour (anticolour) for a
(anti)triplet, one of each for an octet. Counting the colour lines of the event's
flow that join the legs on one side of a line to the other side is what makes a
diagram planar, and it places the colour singlets anywhere along the quark lines.
The leading-colour flows of pyamplicol also keep the U(1) part of a gluon between
two quark lines (-1/N in the colour-flow decomposition), a gluon line with no
colour across it; it joins a quark-antiquark pair to a quark line.
"""

from __future__ import annotations

import json
from functools import lru_cache

# in SubProcesses/, written by the output
CLUSTERING_MODEL = "amplicol_clustering.json"

# the colour lines (colour, anticolour) crossing a line of each representation
_CROSSINGS = {1: (0, 0), 3: (1, 0), -3: (0, 1), 8: (1, 1)}


class VertexTable:
    """The 3-point vertices of a model and its particles, as exported by
    export_amplicol.clustering_model."""

    def __init__(self, data: dict):
        self.particles = {int(pdg): info for pdg, info in data["particles"].items()}
        self.merges: dict[tuple[int, int], list[tuple[int, dict]]] = {}
        for pdgs, orders in data["vertices"]:
            for k in range(3):
                a, b = (pdgs[i] for i in range(3) if i != k)
                # all outgoing: the line joining a and b is the anti of the third
                merge = (self.anti(pdgs[k]), orders)
                known = self.merges.setdefault(tuple(sorted((a, b))), [])
                if merge not in known:
                    known.append(merge)
        # the vertices of more legs that no pair of 3-point vertices reproduces
        self.contact_vertices = [
            list(pdgs) for pdgs in data.get("contact_vertices", [])
            if not self._split_by_3_point(pdgs)
        ]

    def _split_by_3_point(self, pdgs: list[int]) -> bool:
        """Whether two 3-point vertices joined by a line have the legs of this
        4-point vertex."""
        if len(pdgs) != 4:
            return False
        for pair in ((0, 1), (0, 2), (0, 3)):
            a, b = (pdgs[i] for i in pair)
            c, d = (pdgs[i] for i in range(4) if i not in pair)
            lines = {line for line, _ in self.merges.get(tuple(sorted((a, b))), [])}
            if any(self.anti(line) == other
                   for other, _ in self.merges.get(tuple(sorted((c, d))), [])
                   for line in lines):
                return True
        return False

    @classmethod
    def load(cls, path: str) -> VertexTable:
        with open(path) as stream:
            return cls(json.load(stream))

    def anti(self, pdg: int) -> int:
        return self.particles[pdg]["anti"]

    def color(self, pdg: int) -> int:
        return self.particles[pdg]["color"]

    def color_types(self) -> dict[int, int]:
        return {pdg: info["color"] for pdg, info in self.particles.items()}


def _color_bits(flow: list[list[int]], n_in: int, n_ext: int) -> list[tuple[int, int]]:
    """Per colour tag of an LHE colour flow, the bits of the legs holding it as
    colour and as anticolour in the all-outgoing convention."""
    holders: dict[int, list[int]] = {}
    for leg, (color, anticolor) in enumerate(flow[:n_ext]):
        if leg < n_in:
            color, anticolor = anticolor, color
        for tag, slot in ((color, 0), (anticolor, 1)):
            if tag:
                holders.setdefault(tag, [0, 0])[slot] |= 1 << leg
    return [tuple(bits) for bits in holders.values()]


def planar_diagrams(
    table: VertexTable,
    pdgs: list[int],
    n_in: int,
    flow: list[list[int]] | None,
    max_orders: dict[str, int],
) -> list[tuple[tuple[int, int], ...]]:
    """The tree diagrams of the process with these pdg ids (incoming first) in the
    leading-colour flow (LHE colour tags per leg; None: every diagram), within
    max_orders. A diagram is a tuple of internal lines (mask, pdg): the legs on the
    side away from leg 0 as a bit mask, and the outgoing particle they merge into.
    """
    n_ext = len(pdgs)
    out = [table.anti(pdg) if leg < n_in else pdg for leg, pdg in enumerate(pdgs)]
    names = sorted(max_orders)
    limits = tuple(max_orders[name] for name in names)
    merges = {
        key: [(pdg, tuple(orders.get(name, 0) for name in names)) for pdg, orders in value]
        for key, value in table.merges.items()
    }
    tags = _color_bits(flow, n_in, n_ext) if flow is not None else None

    def crossing(mask):
        if tags is None:
            return None
        return (
            sum(1 for c, a in tags if c & mask and not a & mask),
            sum(1 for c, a in tags if a & mask and not c & mask),
        )

    def fits(pdg, cross):
        if cross is None:
            return True
        rep = table.color(pdg)
        if rep not in _CROSSINGS:
            raise NotImplementedError(
                f"the clustering of particles in colour representation {rep} "
                f"(pdg {pdg}) is not supported"
            )
        return _CROSSINGS[rep] == cross

    zero = tuple(0 for _ in names)

    def triplet(pdg):
        return abs(table.color(pdg)) == 3

    def u1_gluon(pdg, cross, pdg1, pdg2):
        # the U(1) part of a gluon between quark lines, which the leading-colour
        # flows keep (-1/N in the colour-flow decomposition): no colour across it
        return (
            cross == (0, 0) and table.color(pdg) == 8
            and table.color(pdg1) == -table.color(pdg2) and triplet(pdg1)
        )

    # currents of the legs in mask: (pdg, orders, lines, whether a U(1) gluon)
    @lru_cache(maxsize=None)
    def currents(mask):
        if mask & (mask - 1) == 0:
            return ((out[mask.bit_length() - 1], zero, (), False),)
        cross = crossing(mask)
        if cross is not None and cross not in _CROSSINGS.values():
            return ()
        low = mask & -mask
        rest = mask ^ low
        result = []
        # every split into a part holding the lowest leg and the rest
        sub = rest
        while True:
            sub = (sub - 1) & rest
            first = low | sub
            second = mask ^ first
            for pdg1, orders1, lines1, u1_1 in currents(first):
                for pdg2, orders2, lines2, u1_2 in currents(second):
                    # a U(1) gluon only attaches to a quark line
                    if (u1_1 and not triplet(pdg2)) or (u1_2 and not triplet(pdg1)):
                        continue
                    key = (pdg1, pdg2) if pdg1 <= pdg2 else (pdg2, pdg1)
                    for pdg, vertex in merges.get(key, ()):
                        orders = tuple(
                            o1 + o2 + o for o1, o2, o in zip(orders1, orders2, vertex)
                        )
                        if any(o > limit for o, limit in zip(orders, limits)):
                            continue
                        u1 = cross is not None and u1_gluon(pdg, cross, pdg1, pdg2)
                        if not u1 and not fits(pdg, cross):
                            continue
                        result.append(
                            (pdg, orders, lines1 + lines2 + ((mask, pdg),), u1)
                        )
            if sub == 0:
                break
        return tuple(result)

    rest = ((1 << n_ext) - 1) ^ 1
    # the line closing on leg 0 is leg 0 itself, not an internal line
    return [
        lines[:-1] for pdg, _, lines, _ in currents(rest) if pdg == table.anti(out[0])
    ]


def madspace_lines(
    diagram: tuple[tuple[int, int], ...], n_ext: int, anti
) -> list[tuple[int, int]]:
    """The lines of a planar_diagrams diagram as MLMClustering takes them,
    (mask, pdg): a final-state line once, a t-channel line from both beams."""
    full = (1 << n_ext) - 1
    lines = []
    for mask, pdg in diagram:
        lines.append((mask, pdg))
        if mask & 2:
            lines.append((full ^ mask, anti(pdg)))
    return lines
