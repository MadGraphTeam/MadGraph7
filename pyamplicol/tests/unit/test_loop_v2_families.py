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
    FamilyFrame,
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
    from tools.loop_reference.feynkit_loop_graphs import build_loop_process

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
        assert len(trie.accumulators) == 1  # one loop order
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
    total = sum(
        STRUCTURED.close(s.trie.rank) * len(s.trie.accumulators)
        for s in plan.schedules.values()
    )
    total += sum(record["work"] for record in shapes)
    assert total == sum(s.trie.cost(rows) for s in plan.schedules.values())
    # A projected view has no interned contents: species fields are absent.
    spins = SkeletonView(
        catalogue, colour, classes=lambda _s: lambda n: catalogue.species(n).spin
    )
    projected = build_schedules(
        families, catalogue, colour, policy=cheapest(rows), view=spins
    )
    assert all(
        record["loop_in"] is record["vertex_templates"] is None
        for record in trie_shapes(projected, rows)
    )
    assert all(isinstance(record["vertex_templates"], int) for record in shapes)
    # Positions only request block currents the block table provides.
    table = ModelBlockTable(model, ["u", "u~"], ["d", "d~", "g"], cap=(("QCD", 5),))
    assert all(
        block in table.currents(block.mask) for block in families.block_requests()
    )


@pytest.mark.loop_fast
def test_schedules_keep_coupling_order_sectors_apart(control):
    model, catalogue = control
    sectors = frozenset({(("QCD", 4),), (("QCD", 2), ("QED", 2))})
    selection = LoopOrderSelection(frozenset({"QCD", "QED"}), sectors)
    table = ModelBlockTable(model, ["u", "u~"], ["d", "d~"], cap=selection.cap)
    families = generate_families(table, catalogue, selection)
    colour = PlanarWords(len(families.legs))
    skeletons = [
        skeleton
        for family in families.families.values()
        for skeleton in family_skeletons(families, family, colour, catalogue)
    ]
    assert {skeleton.orders for skeleton in skeletons} == sectors
    contents = {c for skeleton in skeletons for c in skeleton.contents}
    assert all(term.orders == c.orders for c in contents for term in c.terms)
    # Gluon and photon terms share colour states: orders split the contents.
    by_state = collections.defaultdict(set)
    for c in contents:
        by_state[c.colour, c.loop_in, c.loop_out, c.mask].add(c.orders)
    assert any(len(orders) > 1 for orders in by_state.values())

    def prefix_orders(graph):
        """Orders of every state, propagated from the entries along edges."""
        children = collections.defaultdict(list)
        for parent, child, label in graph.edges:
            children[parent].append((child, label[-1]))  # label ends in orders
        orders = dict.fromkeys(graph.entries, ())
        queue = list(graph.entries)
        while queue:
            state = queue.pop()
            for child, step in children[state]:
                total = v2_families.add_orders(orders[state], step)
                if child not in orders:
                    orders[child] = total
                    queue.append(child)
                assert orders[child] == total  # no state mixes prefix orders
        return orders

    for graph in (forward_graph(skeletons), merged_graph(skeletons)):
        orders = prefix_orders(graph)
        for state, group in graph.closures.items():
            assert {skeleton.orders for skeleton in group} == {orders[state]}
    rows = lambda mask: 2 ** mask.bit_count()  # noqa: E731
    plan = build_schedules(families, catalogue, colour, policy=cheapest(rows))
    split = loop_orders = 0
    for schedule in plan.schedules.values():
        trie = schedule.trie
        leaves = {skeleton.orders for skeleton, _ring in trie.leaves}
        assert sorted(node.orders for node in trie.accumulators) == sorted(leaves)
        for node in trie.nodes:
            for _segment, child in node.edges:
                below = trie.leaves[child][0] if node.depth == 1 else trie.nodes[child]
                assert below.orders == node.orders
        split += len(leaves) > 1
        loop_orders += len(leaves)
    assert split
    # A bucket holds one reduction lane per family and loop order.
    lanes = plan.lanes()
    assert lanes.keys() == plan.buckets().keys()
    assert sum(map(len, lanes.values())) == loop_orders
    for bucket, entries in lanes.items():
        assert {key for key, _orders in entries} == set(plan.buckets()[bucket])


@pytest.mark.loop_fast
def test_relabelled_frames_are_exact_affine_maps(control):
    model, catalogue = control
    families = _generate(model, catalogue, ["g", "g"], ["g", "g", "g"], {"QCD": 3})
    legs = len(families.legs)
    colour = PlanarWords(legs)

    def momentum(mask, sign=1):  # leg-mask momentum modulo conservation
        vector = [sign * (mask >> leg & 1) for leg in range(legs)]
        return tuple(value - vector[-1] for value in vector)

    def add(*vectors):
        return tuple(map(sum, zip(*vectors, strict=True)))

    rows = lambda mask: 2 ** mask.bit_count()  # noqa: E731
    on_key_retest = 0
    for policy in (cheapest(rows), largest_first()):
        plan = build_schedules(families, catalogue, colour, policy=policy)
        assert plan.schedules
        for key, schedule in plan.schedules.items():
            relabel, frame = schedule.relabelling, schedule.trie.frame
            # The re-test is relative to the frame the polynomial is produced
            # in: reversed loop direction, so never the primary reduction.
            assert relabel.source == frame and relabel.sign == -1
            assert relabel.frame.reflected
            assert sorted(relabel.propagators) == list(range(len(key)))
            on_key_retest += frame == FamilyFrame(len(key) // 2, True)
            source = frame.apply(key)
            old = [momentum(mask) for mask in shifts(source)]
            new_key = relabel.apply(key)
            new = [momentum(mask) for mask in shifts(new_key)]
            shift = momentum(relabel.shift_mask)
            # With q_old = sign q_new - V_shift, (q_old + V_p(i))^2 equals
            # (q_new + V'_i)^2 exactly when V'_i = sign (V_p(i) - V_shift).
            for index, origin in enumerate(relabel.propagators):
                assert new_key[index][1] == source[origin][1]
                difference = add(old[origin], tuple(-v for v in shift))
                assert new[index] == tuple(relabel.sign * v for v in difference)
    # Some production frames are the key's own re-test reading, where a
    # key-relative relabelling would have repeated the primary reduction.
    assert on_key_retest
    for family in families.families.values():
        assert family.rank == max(
            sum(p.vertex.q_degree for p in families.routing(c)[1])
            for c in family.closures
        )


@pytest.mark.loop_fast
def test_vertex_momentum_degree_rejects_unknown_momentum_forms(control):
    degree = v2_families._momentum_degree
    # P_in + P_out carries no loop momentum; P_in - P_out carries one.
    assert degree("P(idx(1,3),idx(1,1))+P(idx(1,3),idx(1,2))", (1, 2)) == 0
    assert degree("P(idx(1,3),idx(1,1))-P(idx(1,3),idx(1,2))", (1, 2)) == 1
    for structure in ("PSlash(idx(1,1),idx(1,2),idx(1,3))", "P(idx(1,3),2)"):
        with pytest.raises(NotImplementedError, match="P\\(mu, leg\\)"):
            degree(structure, (1, 2))


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
    # Nothing in the package enumerates loop diagrams or maps integral
    # families (the legacy per-diagram engine was removed; FeynKit's diagrams
    # are a test oracle only, ``tools/loop_reference/feynkit_loop_graphs.py``).
    enumeration = re.compile(r"generate_diagrams|find_mappings|build_loop_process")
    package = Path(v2_families.__file__).parents[2]
    offenders = [
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if enumeration.search(path.read_text())
    ]
    assert offenders == [], offenders


# --------------------------------------------------------------------------- loop_sm


@pytest.fixture(scope="module")
def loop_sm_models(loop_sm_ufo):
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.models.loop_import import load_raw_loop_model
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

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
            facts = prepared.structure.perturbative_species_orders
            assert facts is not None  # the request-scoped dependency facts
            models[restriction] = data, ModelLoopCatalogue(data, species_orders=facts)
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
def test_catalogue_species_facts_match_the_dependency(loop_sm_models):
    data, catalogue = loop_sm_models("default")
    derived = ModelLoopCatalogue(data)  # the catalogue's own is_perturbating rule
    species = [p["name"] for p in data["particles"] if p.get("propagating", True)]
    assert any(catalogue.species(name).perturbative_orders for name in species)
    for name in species:
        assert (
            catalogue.species(name).perturbative_orders
            == derived.species(name).perturbative_orders
        )


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


def _largest_ring(entry):
    return max(int(bucket.split(":")[0]) for bucket in entry["families_by_N_rank"])


def _appendix_b_cases():
    for config, entries in APPENDIX_B["configurations"].items():
        for entry in entries:
            # Heptagon processes run nightly; their largest-first trie nodes
            # depend on the cost model's hash seed (see the fixture).
            heptagons = _largest_ring(entry) >= 7
            marks = [
                pytest.mark.loop_nightly if heptagons else pytest.mark.loop_physics
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


@pytest.fixture(scope="module")
def born_flows(loop_sm_compiled_model):
    """The leading Born colour flows of an Appendix B process (design §7.5)."""

    from pyamplicol.generation.dag_compiler import compile_generic_dag
    from pyamplicol.loops.v2.colour import leading_born_flows
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.processes.model import build_model_process_ir

    compiled = load_compiled_model(loop_sm_compiled_model)

    def flows(entry):
        process = build_model_process_ir(
            " ".join(entry["incoming"]) + " > " + " ".join(entry["outgoing"]),
            compiled.ir,
            color_accuracy="full",
        )
        born = compile_generic_dag(
            process,
            model=CompiledUFOModel(compiled),
            max_coupling_orders=entry["born_orders"],
        )
        return leading_born_flows(born)

    return flows


@pytest.mark.parametrize(("config", "entry"), list(_appendix_b_cases()))
def test_counts_equal_design_appendix_b(loop_sm_models, born_flows, config, entry):
    from pyamplicol.loops.v2.colour import PlanarLeadingColour

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
    # Leading colour: the structural face rule of design §7.5 on the
    # colour-ordered skeletons, against the Born's leading flows.
    planar = PlanarLeadingColour(families.legs, catalogue, born_flows(entry))
    views = {
        "FC": view,
        "LC": SkeletonView(
            catalogue,
            colour,
            classes=view.classes,
            keep=lambda s: view.keep(s) and planar.keep(s),
        ),
    }
    counts = {name: collections.Counter() for name in views}
    for family in families.families.values():
        skeletons = family_skeletons(families, family, colour, catalogue)
        for name, projection in views.items():
            viewed = projection.select(skeletons)
            if not viewed:
                continue
            candidates = candidate_tries(family, viewed, choose)
            count = counts[name]
            count["families"] += 1
            count[f"{family.size}:{family.rank}"] += 1
            count["skeletons"] += len(viewed)
            count["trie_nodes_largest"] += len(policy(family, candidates).nodes)
            count["dress_structured"] += min(t.cost(rows) for _f, t in candidates)
            count["dress_dense"] += min(t.cost(rows, DENSE) for _f, t in candidates)
    fields = ["skeletons", "dress_structured", "dress_dense"]
    if _largest_ring(entry) < 7:  # see the fixture's trie_nodes_seed_dependent
        fields.append("trie_nodes_largest")
    assert {f: counts["FC"][f] for f in fields if f in entry} == {
        f: entry[f] for f in fields if f in entry
    }
    expected = entry["leading_colour"]
    lc = counts["LC"]
    assert lc["families"] == expected["families"]
    if "families_by_N_rank" in expected:
        assert {
            bucket: lc[bucket] for bucket in expected["families_by_N_rank"]
        } == expected["families_by_N_rank"]
    all_gluon = set(entry["incoming"]) | set(entry["outgoing"]) == {"G"}
    # See the fixture's leading_colour note on largest-first ties.
    lc_fields = [f for f in fields if all_gluon or f != "trie_nodes_largest"]
    assert {f: lc[f] for f in lc_fields if f in expected} == {
        f: expected[f] for f in lc_fields if f in expected
    }
