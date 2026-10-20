# SPDX-License-Identifier: 0BSD
"""Loop engine v2 families, skeletons and schedules (design WP1.3).

Oracles: FeynKit families and symmetry factors (O7), Recola tensor-integral
sets and the cost-model counts of DESIGN.md Appendix B (O12). FeynKit is used
here only; the production modules never generate diagrams.
"""

from __future__ import annotations

import collections
import json
import re
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.loops.v2 import families as v2_families
from pyamplicol.loops.v2.families import (
    LoopOrderSelection,
    MassId,
    ModelBlockTable,
    ModelLoopCatalogue,
    canonical_family_key,
    generate_families,
    shifts,
)
from pyamplicol.loops.v2.schedules import (
    DENSE,
    STRUCTURED,
    SkeletonView,
    build_schedules,
    candidate_tries,
    cheapest,
    forward_graph,
    largest_first,
    merged_graph,
    trie_shapes,
)
from pyamplicol.loops.v2.skeletons import DressedColour, PlanarWords, family_skeletons

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "loop" / "v2"


@pytest.fixture(scope="module")
def control():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("loop v2 generation requires a licensed Symbolica runtime")
    from pyamplicol.loops.control_models import qcd_qed_control_model

    model = qcd_qed_control_model()
    return model, ModelLoopCatalogue(model)


def _generate(model, catalogue, incoming, outgoing, born):
    loop = {**born, "QCD": born.get("QCD", 0) + 2}
    selection = LoopOrderSelection(
        frozenset({"QCD"}), frozenset({tuple(sorted(loop.items()))})
    )
    table = ModelBlockTable(model, incoming, outgoing, cap=selection.cap)
    return generate_families(table, catalogue, selection)


def _mass_names(key):
    return tuple((mask, mass.mass) for mask, mass in key)


# --------------------------------------------------------------------------- O7


def _feynkit_ring(graph):
    """Ring of a FeynKit loop graph: attachment leg masks per loop vertex,
    loop-edge masses and species, read along the cycle."""
    loop = set(graph.loop_edges)
    edges = {edge.id: edge for edge in graph.edges}
    trees = collections.defaultdict(list)
    for edge in graph.edges:
        if edge.id not in loop:
            for vertex in (edge.source, edge.target):
                if vertex is not None:
                    trees[vertex].append(edge)

    def legs_behind(vertex, edge):
        mask, stack, seen = 0, [(edge, vertex)], set()
        while stack:
            current, origin = stack.pop()
            if current.id in seen:
                continue
            seen.add(current.id)
            if current.external_index is not None:
                mask |= 1 << current.external_index
                continue
            far = current.target if current.source == origin else current.source
            stack.extend((other, far) for other in trees[far] if other.id != current.id)
        return mask

    incident = collections.defaultdict(list)
    for index in loop:
        incident[edges[index].source].append(edges[index])
        incident[edges[index].target].append(edges[index])
    vertex, order, used = min(incident), [], []
    while len(used) < len(loop):
        order.append(vertex)
        edge = next(e for e in incident[vertex] if e not in used)
        used.append(edge)
        vertex = edge.target if edge.source == vertex else edge.source
    attachments = [
        tuple(sorted(legs_behind(v, edge) for edge in trees[v])) for v in order
    ]
    masses = [re.sub(r".*::", "", edge.mass) for edge in used]
    return attachments, masses, [edge.particle for edge in used]


def _feynkit(process, born):
    from pyamplicol.loops.graphs import build_loop_process

    loop = {**born, "QCD": born.get("QCD", 0) + 2}
    return build_loop_process(
        process,
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
        born_order_bounds={"QED": 0, **born},
        loop_order_bounds={"QED": 0, **loop},
    )


O7_PROCESSES = [
    pytest.param(*case, marks=pytest.mark.loop_fast)
    for case in (
        (["g", "g"], ["g", "g"], {"QCD": 2}),
        (["u", "u~"], ["g", "g"], {"QCD": 2}),
        (["u", "u~"], ["d", "d~"], {"QCD": 2}),
        (["u", "u~"], ["u", "u~"], {"QCD": 2}),
        (["u", "u~"], ["d", "d~", "g"], {"QCD": 3}),
        (["u", "u~"], ["e-", "e+", "g", "g"], {"QCD": 2, "QED": 2}),
    )
] + [
    pytest.param(
        ["g", "g"], ["g", "g", "g"], {"QCD": 3}, marks=pytest.mark.loop_physics
    )
]


@pytest.mark.parametrize(("incoming", "outgoing", "born"), O7_PROCESSES)
def test_family_sets_equal_feynkit_modulo_certified_zeros(
    control, incoming, outgoing, born
):
    model, catalogue = control
    families = _generate(model, catalogue, incoming, outgoing, born)
    oracle = _feynkit(model.process(tuple(incoming), tuple(outgoing)), born)
    found = collections.defaultdict(set)
    for graph in oracle.loops:
        attachments, masses, _species = _feynkit_ring(graph)
        key = canonical_family_key([sum(a) for a in attachments], masses)
        found[key].add(graph.family.family_index)
    # One FeynKit IntegralFamily per canonical key, and the same partition.
    assert all(len(indices) == 1 for indices in found.values())
    assert len(found) == len(oracle.families)
    mine = {_mass_names(key) for key in families.families}
    zeros = {_mass_names(entry.key) for entry in families.scaleless}
    assert mine <= set(found)
    assert set(found) - mine <= zeros
    if (incoming, outgoing) == (["g", "g"], ["g", "g"]):
        # 52 = 4 mass classes x 12 + 4 massless bubbles with a quartic
        # partner; the massless quartic tadpole is logged but FeynKit drops it.
        assert (len(found), len(mine)) == (52, 48)
        assert sorted(len(key) for key in set(found) - mine) == [2, 2, 2, 2]
        assert sorted(len(entry.key) for entry in families.scaleless) == [1, 2, 2, 2, 2]


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    ("incoming", "outgoing", "born"),
    [(["g", "g"], ["g", "g"], {"QCD": 2}), (["u", "u~"], ["g", "g"], {"QCD": 2})],
)
def test_skeleton_weights_equal_feynkit_symmetry_factors(
    control, incoming, outgoing, born
):
    model, catalogue = control
    families = _generate(model, catalogue, incoming, outgoing, born)
    oracle = _feynkit(model.process(tuple(incoming), tuple(outgoing)), born)

    def species_class(name):
        name = families.merged.get(name, name)
        return min(name, catalogue.species(name).antiname)

    zeros = {_mass_names(entry.key) for entry in families.scaleless}
    expected = collections.Counter()
    for graph in oracle.loops:
        attachments, masses, species = _feynkit_ring(graph)
        key = canonical_family_key([sum(a) for a in attachments], masses)
        if key in zeros:
            continue
        ring = frozenset(attachments)
        classes = frozenset(map(species_class, species))
        expected[key, ring, classes, Fraction(1, graph.symmetry_denominator)] += 1
    found = collections.Counter()
    colour = DressedColour()
    for family in families.families.values():
        for skeleton in family_skeletons(families, family, colour, catalogue):
            ring = frozenset(
                tuple(sorted(b.mask for b in c.colour)) for c in skeleton.contents
            )
            classes = frozenset(map(species_class, skeleton.species))
            key = _mass_names(family.key)
            found[key, ring, classes, skeleton.weight] += skeleton.multiplicity
    assert found == expected


@pytest.mark.loop_fast
def test_closed_loop_flavours_merge_only_when_structurally_identical(control):
    model, catalogue = control
    gluons = _generate(model, catalogue, ["g", "g"], ["g", "g"], {"QCD": 2})
    quarks = _generate(model, catalogue, ["u", "u~"], ["g", "g"], {"QCD": 2})
    # Massless u, d, s share every gluon vertex; c, b and t have own masses.
    assert set(gluons.merged) == {"d", "d~", "s", "s~", "u", "u~"}
    assert set(gluons.merged.values()) == {"d", "d~"}
    # An external u line distinguishes u from d and s.
    assert set(quarks.merged) == {"d", "d~", "s", "s~"}


# --------------------------------------------------------------------------- schedules


@pytest.mark.loop_fast
def test_schedules_are_projections_of_one_skeleton_set(control):
    model, catalogue = control
    families = _generate(model, catalogue, ["u", "u~"], ["d", "d~", "g"], {"QCD": 3})
    colour = PlanarWords(len(families.legs))
    rows = lambda mask: 2 ** mask.bit_count()  # noqa: E731
    plan = build_schedules(families, catalogue, colour, policy=cheapest(rows))
    skeletons = [
        skeleton
        for family in families.families.values()
        for skeleton in family_skeletons(families, family, colour, catalogue)
    ]
    forward, merged = forward_graph(skeletons), merged_graph(skeletons)
    for graph in (forward, merged):
        closed = [s for group in graph.closures.values() for s in group]
        assert sorted(map(id, closed)) == sorted(map(id, skeletons))
    # M merges across orderings and families: fewer states than F.
    assert len(merged) < len(forward)
    per_family = collections.Counter(s.family for s in skeletons)
    for key, schedule in plan.schedules.items():
        trie = schedule.trie
        assert len(trie.leaves) == per_family[key]
        assert [node.depth for node in trie.nodes].count(len(key)) == 1
        assert schedule.bucket == (len(key), schedule.family.rank)
        assert schedule.family.rank <= len(key) + 2
        for node in trie.nodes:
            for segment, child in node.edges:
                if node.depth == 1:
                    assert trie.leaves[child][1][1:] == node.suffix
                else:
                    assert trie.nodes[child].suffix == (segment, *node.suffix)
    # Families are reduced together per (N, R) bucket; requests share masters.
    assert sum(map(len, plan.buckets().values())) == len(families.families)
    requests = plan.master_requests()
    assert sum(requests.values()) > len(requests)
    shapes = trie_shapes(plan, rows)
    total = sum(STRUCTURED.close(s.trie.rank) for s in plan.schedules.values())
    total += sum(record["work"] for record in shapes)
    assert total == sum(s.trie.cost(rows) for s in plan.schedules.values())
    # Positions only request block currents the block table provides.
    table = ModelBlockTable(model, ["u", "u~"], ["d", "d~", "g"], cap=(("QCD", 5),))
    assert all(
        block in table.currents(block.mask) for block in families.block_requests()
    )


@pytest.mark.loop_fast
def test_relabelled_frames_are_exact_affine_maps(control):
    model, catalogue = control
    families = _generate(model, catalogue, ["g", "g"], ["g", "g", "g"], {"QCD": 3})
    legs = len(families.legs)

    def momentum(mask, sign=1):  # leg-mask momentum modulo conservation
        vector = [sign * (mask >> leg & 1) for leg in range(legs)]
        return tuple(value - vector[-1] for value in vector)

    def add(*vectors):
        return tuple(map(sum, zip(*vectors, strict=True)))

    for key, family in families.families.items():
        relabel = v2_families.default_relabelling(key)
        assert relabel.frame.reflected and len(set(relabel.propagators)) == len(key)
        old = [momentum(mask) for mask in shifts(key)]
        new_key = relabel.frame.apply(key)
        new = [momentum(mask) for mask in shifts(new_key)]
        shift = momentum(relabel.shift_mask)
        # With q_old = sign q_new - V_shift, (q_old + V_p(i))^2 equals
        # (q_new + V'_i)^2 exactly when V'_i = sign (V_p(i) - V_shift).
        for index, source in enumerate(relabel.propagators):
            assert new_key[index][1] == key[source][1]
            difference = add(old[source], tuple(-v for v in shift))
            assert new[index] == tuple(relabel.sign * v for v in difference)
        assert family.rank == max(
            sum(p.vertex.q_degree for p in families.routing(c)[1])
            for c in family.closures
        )


@pytest.mark.loop_fast
def test_rank_beyond_integrand_reduction_domain_fails(control):
    model, catalogue = control

    class Raised:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def vertices(self, loop_in):
            from dataclasses import replace

            return tuple(
                replace(v, q_degree=v.q_degree + 2)
                for v in self.inner.vertices(loop_in)
            )

        def reflected(self, vertex):
            from dataclasses import replace

            base = self.inner.reflected(replace(vertex, q_degree=vertex.q_degree - 2))
            return replace(base, q_degree=base.q_degree + 2)

    with pytest.raises(ValueError, match="N \\+ 2"):
        _generate(model, Raised(catalogue), ["g", "g"], ["g", "g"], {"QCD": 2})


@pytest.mark.loop_fast
def test_production_modules_never_generate_diagrams():
    source = Path(v2_families.__file__).parent
    forbidden = re.compile(
        r"generate_diagrams|IntegralFamily|find_mappings|build_loop_process|"
        r"FeynmanDiagram|overall_factor|loops\.graphs"
    )
    for path in ("families.py", "skeletons.py", "schedules.py"):
        assert not forbidden.search((source / path).read_text()), path


# --------------------------------------------------------------------------- loop_sm


@pytest.fixture(scope="module")
def loop_sm_models(loop_sm_ufo):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("loop v2 generation requires a licensed Symbolica runtime")
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    models = {}

    def load(restriction):
        if restriction not in models:
            raw = load_raw_loop_model(loop_sm_ufo, restriction=restriction)
            raw = prepare_loop_sm_scalar_declarations(raw).derived
            prepared = prepare_loop_request_model(
                raw,
                born_order_bounds={"QCD": 5, "QED": 2},
                loop_order_bounds={"QCD": 7, "QED": 2},
            )
            data = json.loads(prepared.structure.native.to_json())
            models[restriction] = data, ModelLoopCatalogue(data)
        return models[restriction]

    return load


RECOLA = json.loads((FIXTURES / "recola_tensor_integrals.json").read_text())


def _recola_order(legs, catalogue):
    """Recola's internal leg order (core/currents_rcl.f90:523-563)."""
    crossed = []
    for leg in legs:
        outgoing = catalogue.species(leg.field).antiname
        colour = abs(leg.colour)
        crossed.append((leg.index, colour, outgoing.endswith("~")))
    groups = (
        [i for i, c, _ in crossed if c == 1],
        [i for i, c, anti in crossed if c == 3 and anti],
        [i for i, c, anti in crossed if c == 3 and not anti],
        [i for i, c, _ in crossed if c == 8],
    )
    return [index for group in groups for index in group]


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "entry", RECOLA["processes"], ids=[p["recola"] for p in RECOLA["processes"]]
)
def test_family_sets_equal_recola_tensor_integrals(loop_sm_models, entry):
    model, catalogue = loop_sm_models("no_b_mass")
    families = _generate(
        model, catalogue, entry["incoming"], entry["outgoing"], entry["born_orders"]
    )
    order = _recola_order(families.legs, catalogue)
    top = catalogue.species("t").mass
    recola = set()
    for count, _rank, moms, masses in entry["tensor_integrals"]:
        blocks = collections.defaultdict(int)
        for slot, propagator in enumerate(moms):
            blocks[propagator] |= 1 << order[slot]
        sequence = [*range(1, count), 0]
        masks = [blocks[i] for i in sequence] if count > 1 else [sum(blocks.values())]
        ids = [masses[i] for i in sequence] if count > 1 else masses
        recola.add(
            canonical_family_key(masks, [MassId("0") if i == 1 else top for i in ids])
        )
    mine = set(families.families)
    logged = {entry.key for entry in families.scaleless} | {
        entry.key
        for entry in families.rejected
        if entry.reason == "single-coloured-attachment"
    }
    assert mine <= recola
    assert recola - mine <= logged
    if entry["recola"] == "u u~ -> e- e+ d d~ g":
        assert (len(recola), len(mine)) == (246, 242)


APPENDIX_B = json.loads((FIXTURES / "appendix_b_counts.json").read_text())


def _appendix_b_cases():
    for config, entries in APPENDIX_B["configurations"].items():
        for entry in entries:
            legs = len(entry["incoming"]) + len(entry["outgoing"])
            marks = [
                pytest.mark.loop_nightly if legs >= 7 else pytest.mark.loop_physics
            ]
            yield pytest.param(
                config, entry, marks=marks, id=f"{config}:{entry['process']}"
            )


def _cost_model_view(families, catalogue, entry):
    """The Appendix B cost model's projection of the generated skeletons.

    Its planar blocks are orderings of letters (a lepton pair is one letter);
    ghost loops are not counted, closed quark loops of one mass form one
    orientation-free class and open quark lines keep their flavour. Where a
    skeleton fits a frame in both directions (N <= 2), the model reads it in
    the direction of its first enumerated colour word, and it scores frames
    with its own leg order.
    """
    legs = families.legs
    singlets = sum(1 << leg.index for leg in legs if abs(leg.colour) == 1)
    letters = [1 << leg.index for leg in legs if abs(leg.colour) != 1]
    letters += [singlets] if singlets else []
    order = entry.get("cost_model_legs") or sorted(
        [[i for i in range(len(legs)) if letter >> i & 1] for letter in letters]
    )
    rank = {
        sum(1 << i for i in group): position for position, group in enumerate(order)
    }
    first = min(rank, key=rank.get)

    def letter_rows(letter):
        leg = legs[letter.bit_length() - 1]
        quark = abs(leg.colour) == 3 and leg.mass.massless
        return (
            1
            if letter == 1 << leg.index and quark and not leg.field.endswith("~")
            else 2
        )

    def rows(mask):
        out = 1
        for letter in letters:
            if letter & mask:
                out *= letter_rows(letter)
        return out

    def classes(skeleton):
        names = set(skeleton.species)
        closed = (
            all(abs(catalogue.species(n).colour) == 3 for n in names)
            and len({catalogue.species(n).mass for n in names}) == 1
            and not any(
                abs(catalogue.species(block.field).colour) == 3
                for content in skeleton.contents
                for term in content.terms
                for block in term.blocks
            )
        )

        def species_class(name):
            species = catalogue.species(name)
            if abs(species.colour) != 3:
                return ("g",)
            if closed:
                return "Q", species.mass
            return "q", min(name, species.antiname), species.mass

        return species_class

    def first_word_direction(readings):
        def successor(ring):
            word = [letter for segment in ring for letter in segment.label[0]]
            return rank[word[(word.index(first) + 1) % len(word)]]

        return min(readings, key=successor)

    def ghost_loop(skeleton):
        return all(catalogue.species(n).spin == -1 for n in skeleton.species)

    colour = PlanarWords(len(legs), letters)
    view = SkeletonView(
        catalogue, colour, classes=classes, keep=lambda s: not ghost_loop(s)
    )
    policy = largest_first(
        size=lambda mask: sum(1 for letter in letters if letter & mask),
        first=lambda mask: min(rank[letter] for letter in letters if letter & mask),
    )
    return colour, view, rows, policy, first_word_direction


@pytest.mark.parametrize(("config", "entry"), list(_appendix_b_cases()))
def test_counts_equal_design_appendix_b(loop_sm_models, config, entry):
    restriction = "no_b_mass" if config.endswith("no_b_mass") else "default"
    model, catalogue = loop_sm_models(restriction)
    families = _generate(
        model, catalogue, entry["incoming"], entry["outgoing"], entry["born_orders"]
    )
    histogram = collections.Counter(
        f"{family.size}:{family.rank}" for family in families.families.values()
    )
    assert len(families.families) == entry["families"]
    assert dict(histogram) == entry["families_by_N_rank"]
    colour, view, rows, policy, choose = _cost_model_view(families, catalogue, entry)
    counts = collections.Counter()
    for family in families.families.values():
        viewed = view.select(family_skeletons(families, family, colour, catalogue))
        candidates = candidate_tries(family, viewed, choose)
        counts["skeletons"] += len(viewed)
        counts["trie_nodes_largest"] += len(policy(family, candidates).nodes)
        counts["dress_structured"] += min(t.cost(rows) for _f, t in candidates)
        counts["dress_dense"] += min(t.cost(rows, DENSE) for _f, t in candidates)
    fields = ["skeletons", "dress_structured", "dress_dense"]
    if len(families.legs) < 7:  # see the fixture's trie_nodes_seed_dependent
        fields.append("trie_nodes_largest")
    assert {f: counts[f] for f in fields if f in entry} == {
        f: entry[f] for f in fields if f in entry
    }
