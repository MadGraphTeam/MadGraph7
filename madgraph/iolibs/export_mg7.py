import json
import logging
import os
from collections import defaultdict

from madgraph import MadGraph5Error
from madgraph.various.diagram_symmetry import find_symmetry, IdentifySGConfigTag
from madgraph.iolibs import export_cpp
from madgraph.iolibs import crossing_table
from madgraph.iolibs.group_subprocs import IdentifyConfigTag
from madgraph.core import diagram_generation, helas_objects
from madgraph.core.diagram_generation import DiagramTag

logger = logging.getLogger('madgraph.export_mg7')

class IdentifyTopologyTag(IdentifyConfigTag):
    """ Like IndentifyConfigTag, but ignores spin and color """

    @staticmethod
    def link_from_leg(leg, model):
        # the parent link grew a third element (the bound state) in 3.8.0,
        # so take the leg data and its number positionally
        link = super(
            IdentifyTopologyTag, IdentifyTopologyTag
        ).link_from_leg(leg, model)[0]
        (leg_num1, _, mass, width, _), leg_num2 = link[0], link[1]
        return [((leg_num1, mass, width), leg_num2)]

    @staticmethod
    def vertex_id_from_vertex(vertex, last_vertex, model, ninitial):
        vertex = super(IdentifyTopologyTag, IdentifyTopologyTag).vertex_id_from_vertex(
            vertex, last_vertex, model, ninitial
        )
        if len(vertex) == 1:
            return ((0,),)
        (_, mass, width), _ = vertex
        return ((mass, width), 0)


class IdentifySGTopologyTag(IdentifySGConfigTag):
    """ Like IndentifySGConfigTag, but ignores spin, color and charge """

    @staticmethod
    def link_from_leg(leg, model):
        link = super(
            IdentifySGTopologyTag, IdentifySGTopologyTag
        ).link_from_leg(leg, model)[0]
        (state, _, _, _, mass, width), leg_num = link[0], link[1]
        return [((state, mass, width), leg_num)]

    @staticmethod
    def vertex_id_from_vertex(vertex, last_vertex, model, ninitial):
        vertex = super(IdentifySGTopologyTag, IdentifySGTopologyTag).vertex_id_from_vertex(
            vertex, last_vertex, model, ninitial
        )
        if vertex == (0,):
            return (0,)
        (_, mass, width, qcd, onshell), = vertex
        return ((mass, width, qcd, onshell),)


class OneProcessExporterMG7(export_cpp.OneProcessExporterCPP):

    def __init__(self, matrix_element, cpp_helas_call_writer, merge_same_topologies=True):
        super().__init__(matrix_element, cpp_helas_call_writer)
        self.init_subprocess_metadata(matrix_element, merge_same_topologies)

    def init_subprocess_metadata(self, matrix_element, merge_same_topologies=True):
        """Everything get_subprocess_info reads: the channels, flavors and
        colour flows of `matrix_element` (no C++ is involved, see
        SubprocessMetadataMG7)."""
        self.matrix_element = matrix_element
        self.name = f"P{matrix_element.get('processes')[0].shell_string()}"
        self.model = self.matrix_element.get("processes")[0].get("model")
        self.amplitude = self.matrix_element.get("base_amplitude")
        if merge_same_topologies:
            self.sym_indices, self.sym_perms, _ = find_symmetry(
                self.matrix_element,
                lambda diag: IdentifySGTopologyTag(diag, self.model),
                skip_identical_check=True,
            )
        else:
            self.sym_indices, self.sym_perms, _ = find_symmetry(
                self.matrix_element, lambda diag: IdentifySGConfigTag(diag, self.model)
            )

        self.diagrams = self.amplitude.get("diagrams")
        self.helas_diagrams = self.matrix_element.get("diagrams")
        self.all_flavors, self.all_flavors_pdgs = self.matrix_element.get_external_flavors_with_iden(return_pdgs=True)
        self.all_flavors = [list(flavors) for flavors in self.all_flavors]
        self.all_flavors_pdgs = [list(pdgs) for pdgs in self.all_flavors_pdgs]
        self.expand_flavors_over_processes()
        self.process = self.amplitude.get("process")
        self.legs = self.process.get("legs_with_decays")
        self.color_basis = self.matrix_element.get("color_basis")
        # The basis a color flow is picked among: always the trace one, which
        # is the color basis itself unless the color sum runs on the DDM basis.
        # Everything indexing a color flow -- the color_flows table, the
        # active_colors masks, icolamp -- has to use this one and not the
        # (smaller) basis of the color sum.
        self.color_flow_basis = self.color_basis.get_flow_basis() \
                                if self.color_basis else self.color_basis
        self.set_subprocess_class()
        self.set_topology()
        self.set_flavor_indices()
        self.set_active_flavors()
        self.set_channels_colors_map()

    def generate_process_files(self):
        super().generate_process_files()

    def set_subprocess_class(self):
        is_parts = [
            self.model.get_particle(l.get("id"))
            for l in self.process.get("legs")
            if not l.get("state")
        ]
        fs_parts = [
            self.model.get_particle(l.get("id"))
            for l in self.process.get("legs")
            if l.get("state")
        ]
        self.subprocess_class = (
            tuple(
                (p.get("mass"), l.get("onshell"))
                for (p, l) in zip(is_parts + fs_parts, self.process.get("legs"))
            ),
            self.process.get("id"),
        )

    def set_topology(self):
        """Name every external leg i<k>/o<k> and record the initial/final pdgs.

        Two initial legs for a collision, one for a decay (``t > b w+, ...``,
        which MadSpin hands over as a single flattened matrix element). Legs are
        numbered 1..n with the initial state first, so the outgoing offset is
        the number of initial legs.
        """
        self.edge_names = {}
        self.n_initial = sum(1 for leg in self.legs if not leg.get("state"))
        self.incoming = [None] * self.n_initial
        self.outgoing = [None] * (len(self.legs) - self.n_initial)
        for leg in self.legs:
            number = leg.get("number")
            if leg.get("state"):
                index = number - self.n_initial - 1
                self.edge_names[number] = f"o{index}"
                self.outgoing[index] = leg.get("id")
            else:
                self.edge_names[number] = f"i{number - 1}"
                self.incoming[number - 1] = leg.get("id")
        if any(pdg is None for pdg in self.incoming + self.outgoing):
            raise AssertionError(
                "external legs of %s are not numbered 1..%d with the initial "
                "state first: %s" % (
                    self.name, len(self.legs),
                    [(leg.get("number"), leg.get("state")) for leg in self.legs])
            )

    def expand_flavors_over_processes(self):
        """Add the flavors that live in the *processes* mapped onto this matrix
        element rather than in its merged legs.

        get_external_flavors_with_iden only expands merged legs (pdg 81/82/...),
        i.e. the apply_flavor_grouping=True case. With grouping off there are no
        merged legs and MG5 instead maps every flavor-equivalent process onto a
        single matrix element -- u u~ > e+ e-, u u~ > mu+ mu-, c c~ > e+ e- and
        c c~ > mu+ mu- all share one -- so asking only for the merged expansion
        returns the representative alone and the other channels never make it
        into subprocesses.json (p p > l+ l- came out at 538 pb instead of
        1336 pb, i.e. 4 of 16 channels). madevent walks both sources in
        get_leshouche_lines; use the same shared enumeration here.
        """
        combinations = self.matrix_element.get_flavor_pdg_combinations(self.model)
        # Merged legs: get_external_flavors_with_iden already enumerated
        # everything, and re-expanding here would double count.
        if any(has_merged for _, has_merged in combinations):
            return
        pdg_lists = [pdgs for pdg_lists, _ in combinations for pdgs in pdg_lists]
        if len(pdg_lists) <= 1:
            return
        # Without merged legs every leg trivially takes flavor index 1, so all
        # these processes share the single coupling class and its flavor-index
        # tuple; keep all_flavors aligned with all_flavors_pdgs.
        if len(self.all_flavors) != 1 or len(self.all_flavors[0]) != 1:
            return
        self.all_flavors_pdgs = [pdg_lists]
        self.all_flavors = [self.all_flavors[0] * len(pdg_lists)]

    def set_flavor_indices(self):
        # Flavor combinations are grouped by their initial state: the launcher
        # picks one initial state (PDF-weighted), then a final state within it.
        # A decay has a single initial leg to group on, not a beam pair.
        self.all_flavors_same_initial = []
        self.all_flavors_indices = []
        for i, flavors in enumerate(self.all_flavors_pdgs):
            flv_dict = defaultdict(list)
            for flv in flavors:
                flv_dict[tuple(flv[:self.n_initial])].append(flv)
            indices = []
            for flv in flv_dict.values():
                indices.append(len(self.all_flavors_same_initial))
                self.all_flavors_same_initial.append((i, flv))
            self.all_flavors_indices.append(indices)

    def set_active_flavors(self):
        # Per-diagram flavor validity is precomputed in the diagram flavor store
        # (populate_flavor_validity, triggered via get_external_flavors_with_iden
        # in __init__), so this is a pure read through HelasDiagram.has_flavor.
        self.active_flavors = [[] for d in self.diagrams]
        for indices, flavors in zip(self.all_flavors_indices, self.all_flavors):
            flavor = tuple(flavors[0])
            for active_flavors, diag in zip(
                self.active_flavors, self.matrix_element.get('diagrams')
            ):
                if diag.has_flavor(flavor):
                    active_flavors.extend(indices)

    def diagram_edge_leg_sets(self, diagram, sym_perm=None):
        """For each internal line of `diagram`, in vertex-list order, the
        frozenset of external edge names behind it -- a vertex-order-
        independent identity, unlike diagram.get("vertices") position.
        `sym_perm` translates this diagram's leg numbers to the
        representative's; leave None for the representative itself."""
        def canonical_name(leg_number):
            if sym_perm is not None:
                leg_number = sym_perm[leg_number - 1] + 1
            return self.edge_names[leg_number]

        diagram_edge_names = {}
        edge_leg_sets = {name: frozenset((name,)) for name in self.edge_names.values()}
        leg_sets = []
        diag_vertices = diagram.get("vertices")
        for i_vert, vertex in enumerate(diag_vertices):
            legs = vertex.get("legs")
            input_names = [
                diagram_edge_names.get(leg.get("number"))
                or canonical_name(leg.get("number"))
                for leg in legs[:-1]
            ]
            downstream = frozenset().union(*(edge_leg_sets[name] for name in input_names))
            if i_vert == len(diag_vertices) - 1:
                # Closing vertex: its last leg is a pre-existing external edge,
                # not a new internal line.
                continue
            prop_name = f"p{len(leg_sets)}"
            diagram_edge_names[legs[-1].get("number")] = prop_name
            edge_leg_sets[prop_name] = downstream
            leg_sets.append(downstream)
        return leg_sets

    def propagator_pdg(self, leg, leg_set):
        """Signed pdg id of the internal line `leg`, oriented the way the
        phase-space topology reads it: flowing away from the initial state.

        Madgraph records, on the leg a vertex creates, the pdg of the line
        flowing into the legs that were combined to make it -- `leg_set`.
        That is already the decay orientation while those are all final
        state, but a line holding *every* initial leg is the one madspace
        roots the other way round: its decay products are the complementary
        legs, so what belongs in the LHE is the anti-particle. Without this
        the s-channel W+ of `p p > e+ ve` is written as a W- decaying to
        e+ ve. A leg set holding only some of the initial legs is a
        t-channel, which never becomes a decay and is left as madgraph put
        it.

        Only colour singlets actually depend on this: for a coloured line
        lhe_output.cpp's compute_decay_color infers the orientation from the
        colour flow and flips the pdg back itself.
        """
        part = self.model.get_particle(leg.get("id"))
        if part.get("self_antipart"):
            return part.get("pdg_code")
        sign = 1 if part.get("is_part") else -1
        if sum(1 for name in leg_set if name.startswith("i")) == self.n_initial:
            sign = -sign
        return sign * part.get("pdg_code")

    def diagram_propagator_pdgs(self, diagram, channel_leg_sets, sym_perm):
        """Signed pdg id of each internal line of `diagram`, reordered to
        match `channel_leg_sets` (the order used for
        Topology::Decay::flat_propagator_index) rather than this diagram's
        own vertex order, which need not agree even for a diagram merged
        into the channel by merge_same_topologies."""
        diag_vertices = diagram.get("vertices")
        leg_sets = self.diagram_edge_leg_sets(diagram, sym_perm)
        pdg_by_leg_set = {}
        for i_vert, vertex in enumerate(diag_vertices[:-1]):
            legs = vertex.get("legs")
            pdg_by_leg_set[leg_sets[i_vert]] = self.propagator_pdg(
                legs[-1], leg_sets[i_vert]
            )
        return [pdg_by_leg_set[leg_set] for leg_set in channel_leg_sets]

    def set_channels_colors_map(self):
        if self.color_basis:
            # active_colors ends up in the icolamp mask, which is walked over
            # the color flows, so it must be indexed on the flow basis
            flow_basis = self.color_flow_basis
            diag_jamps = defaultdict(list)
            # Only leading-Nc jamps are planar-compatible with a diagram's own
            # topology; like export_v4's get_icolamp_lines, drop the rest.
            max_Nc = max(
                v[4] - v[5]
                for val in flow_basis.values()
                for v in val
            )
            for ijamp, col_basis_elem in enumerate(sorted(flow_basis.keys())):
                for diag_tuple in flow_basis[col_basis_elem]:
                    if diag_tuple[4] - diag_tuple[5] == max_Nc:
                        diag_jamps[diag_tuple[0]].append(ijamp)
            # kept for the crossed subprocesses, which pick their colour flow
            # in this basis (get_crossed_subprocess_info)
            self.diag_jamps = diag_jamps

        self.channels = []
        # Index-aligned with self.channels; kept separate (not serialized --
        # frozensets aren't JSON-able) and only needed transiently to reorder
        # merged diagrams' propagator_pdgs, see diagram_propagator_pdgs.
        channel_leg_sets = []
        channel_indices = []
        self.diagram_tags = []
        for diagram_index, (sym_index, sym_perm) in enumerate(zip(self.sym_indices, self.sym_perms)):
            if sym_index == 0:
                channel_indices.append(-1)
                continue

            active_colors = diag_jamps[diagram_index] if self.color_basis else [0]
            active_flavors = self.active_flavors[diagram_index]
            if len(active_flavors) == 0:
                raise RuntimeError(
                    f"no valid flavor configurations found for diagram {diagram_index+1}"
                )
            diagram = self.diagrams[diagram_index]
            if sym_index < 0:
                chan_index = channel_indices[-sym_index - 1]
                self.diagram_tags[chan_index].append(
                    IdentifyTopologyTag(diagram, self.model),
                )
                self.channels[chan_index]["diagrams"].append(
                    {
                        "diagram": diagram_index,
                        "permutation": sym_perm,
                        "active_flavors": active_flavors,
                        "active_colors": active_colors,
                        "propagator_pdgs": self.diagram_propagator_pdgs(
                            diagram, channel_leg_sets[chan_index], sym_perm
                        ),
                    }
                )
                channel_indices.append(-1)
                continue

            vertices = []
            propagators = []
            on_shell_propagators = []
            diagram_edge_names = dict(self.edge_names)
            diag_vertices = diagram.get("vertices")
            # Index-aligned with `propagators`: both are filled in vertex-list
            # order and both skip the closing vertex, which is the last one.
            leg_sets = self.diagram_edge_leg_sets(diagram)
            for i_vert, vertex in enumerate(diag_vertices):
                legs = vertex.get("legs")
                # Last amplitude vertex does not create new edges
                vertex_props = [diagram_edge_names[leg.get("number")] for leg in legs[:-1]]

                if i_vert == len(diag_vertices) - 1:
                    vertex_props.append(diagram_edge_names[legs[-1].get("number")])
                else:
                    prop_index = len(propagators)
                    prop_name = f"p{prop_index}"
                    diagram_edge_names[legs[-1].get("number")] = prop_name
                    vertex_props.append(prop_name)
                    propagators.append(
                        self.propagator_pdg(legs[-1], leg_sets[prop_index])
                    )
                    if legs[-1].get("onshell"):
                        on_shell_propagators.append(prop_index)
                vertices.append(vertex_props)

            chan_index = len(self.channels)
            self.diagram_tags.append([IdentifyTopologyTag(diagram, self.model)])
            channel_indices.append(chan_index)
            channel_leg_sets.append(leg_sets)
            self.channels.append(
                {
                    "propagators": propagators,
                    "vertices": vertices,
                    "on_shell_propagators": on_shell_propagators,
                    "diagrams": [
                        {
                            "diagram": diagram_index,
                            "permutation": sym_perm,
                            "active_flavors": active_flavors,
                            "active_colors": active_colors,
                            "propagator_pdgs": propagators,
                        }
                    ],
                }
            )

        self.multi_channel_map = {}
        self.active_color_map = []
        i = 0
        for channel in self.channels:
            for diag in channel["diagrams"]:
                diagram_index = diag["diagram"]
                active_colors = diag["active_colors"]
                self.multi_channel_map[i] = [diagram_index]
                self.active_color_map.append(active_colors)
                i += 1

    @staticmethod
    def get_color_code_tables(color_flow_dicts, legs):
        """(codes, slots) -- the canonical colour-flow code of each flow plus the
        slot structure needed to decode it, or (None, None) when the flows have
        no usable code (a sextet, or an epsilon structure).

        Same encoding as the fortran madevent output (see export_v4:
        _color_flow_code / _color_flow_decode): flip the initial-state pair so
        every colour index connects to an anticolour index, then digit i is the
        anticolour SLOT that colour slot i connects to, and
        code = sum_i digit_i * N^i. `slots` is {"color": [...], "acolor": [...]}
        with 1-based leg numbers, and is flow independent -- it is fixed by the
        colour representations, so one table serves every flow.

        Consumers decode a code back to the per-leg tags rather than looking the
        flow up in the ICOLUP-style "color_flows" table."""
        from madgraph.iolibs.export_v4 import ProcessExporterFortranME as _E
        states = [l.get("state") for l in legs]
        flows = [[tuple(cf[l.get("number")]) for l in legs]
                 for cf in color_flow_dicts]
        if any(c < 0 or a < 0 for fl in flows for c, a in fl):
            return None, None      # sextet: negative tag, not representable
        conns = [_E._color_flow_canon(fl, states) for fl in flows]
        codes = [_E._color_flow_code(c) for c in conns]
        if any(c is None for c in codes) or len(set(codes)) != len(codes):
            return None, None
        colslots, acolslots = _E._color_flow_slots(conns[0])
        if not acolslots or any(_E._color_flow_slots(c) != (colslots, acolslots)
                                for c in conns[1:]):
            return None, None
        return codes, {"color": [l + 1 for l in colslots],
                       "acolor": [l + 1 for l in acolslots]}

    def get_color_flow_dicts(self):
        """(flows, legs): one {leg number: (colour, anticolour)} dict per
        colour flow, in the order the flow index counts them, over the
        external legs `legs`; flows is None without a colour basis."""
        legs = self.process.get_legs_with_decays()
        if not self.color_basis:
            return None, legs
        n_initial = self.matrix_element.get_nexternal_ninitial()[1]
        # First build a color representation dictionnary
        repr_dict = {}
        for leg in legs:
            repr_dict[leg.get("number")] = self.model.get_particle(
                leg.get("id")
            ).get_color() * (-1) ** (1 + leg.get("state"))
        # Get the list of color flows. This is about color flows, so
        # always the trace basis, even when the color sum runs on the DDM
        # one.
        return self.color_flow_basis.\
            color_flow_decomposition(repr_dict, n_initial), legs

    def get_subprocess_info(self, proc_dir, lib_me_path):
        n_external, n_initial = self.matrix_element.get_nexternal_ninitial()
        color_flow_dicts, legs = self.get_color_flow_dicts()
        if color_flow_dicts is not None:
            # And output them properly
            color_flows = [
                [[color_flow_dict[leg.get("number")][i] for i in [0, 1]] for leg in legs]
                for color_flow_dict in color_flow_dicts
            ]
            color_codes, color_slots = self.get_color_code_tables(
                color_flow_dicts, legs)
        else:
            color_flows = [[[0, 0]] * n_external]
            color_codes, color_slots = None, None

        # We need the both particle and antiparticle wf_ids, since the identity
        # depends on the direction of the wf.
        wf_ids = set(
            wf_id
            for d in self.matrix_element.get("diagrams")
            for wf in d.get("wavefunctions")
            for wf_id in [wf.get_pdg_code(), wf.get_anti_pdg_code()]
        )
        leg_ids = set(
            leg_id
            for p in self.matrix_element.get("processes")
            for leg in p.get_legs_with_decays()
            for leg_id in [leg.get("id"), self.model.get_particle(leg.get("id")).get_anti_pdg_code()]
        )
        pdg_color_types = {}
        for part_id in sorted(list(wf_ids.union(leg_ids))):
            pdg_color_types[part_id] = self.model.get_particle(part_id).get_color()
            if abs(part_id) in self.model["merged_particles"]:
                for pdg in self.model["merged_particles"][abs(part_id)]:
                    sign = -1 if part_id < 0 else 1
                    pdg_color_types[sign * pdg] = sign * self.model.get_particle(part_id).get_color()

        has_mirror_all = self.matrix_element.get("has_mirror_process")
        # Whether the beam-swapped initial state is part of the process must be
        # derived from the process definition (per-beam multiparticle content),
        # exactly as madevent's write_mirrorprocs does -- not from the pdg of the
        # matrix-element legs. With flavor merging both initial legs of
        # "u q > u q" (q = u d) carry the same merged pdg (81), yet leg 1 is
        # fixed to u, so "d u > u d" is not part of the process and mirroring the
        # u d flavor would double count it.
        # A decay has a single initial leg, so there is no beam swap to mirror.
        same_initial_multiparticle = self.n_initial == 2 and \
            self.matrix_element.get("processes")[0].has_same_initial_multiparticle()
        flavors = [
            {
                "index": index,
                "options": options,
                "mirror": self.n_initial == 2 and (has_mirror_all or (
                    same_initial_multiparticle and options[0][0] != options[0][1]
                ))
            }
            for index, options in self.all_flavors_same_initial
        ]

        # power of alpha_s in |M|^2 (the QCD coupling order of the amplitude),
        # -1 when it differs between diagrams. The systematics computation uses
        # it to rescale |M|^2 for renormalisation scale variations; never let
        # its computation break the output.
        qcd_orders = set()
        try:
            for diagram in self.helas_diagrams:
                qcd_orders.add(diagram.calculate_orders().get('QCD', 0))
        except Exception as error:
            logger.debug('could not determine the QCD order: %s', error)
            qcd_orders.add(None)
        qcd_power = (qcd_orders.pop()
                     if len(qcd_orders) == 1 and None not in qcd_orders else -1)

        return (
            {
                "incoming": self.incoming,
                "outgoing": self.outgoing,
                "channels": self.channels,
                "me_path": lib_me_path,
                "path": proc_dir,
                "flavors": flavors,
                "qcd_power": qcd_power,
                # ICOLUP-style per-flow tags. Still needed by the LHE writer to
                # reconstruct the colour of INTERNAL (propagator/decay) lines, so
                # it cannot be dropped just because the code gives the external
                # legs.
                "color_flows": color_flows,
                # canonical colour-flow code of each flow + the (flow
                # independent) slot structure to decode it; null when the flows
                # have no usable code, in which case a consumer falls back to
                # "color_flows".
                "color_codes": color_codes,
                "color_slots": color_slots,
                "pdg_color_types": pdg_color_types,
                "diagram_count": len(self.diagrams),
                "helicities": list(self.matrix_element.get_helicity_matrix()),
            },
            self.diagram_tags,
            self.subprocess_class,
        )

    # ------------------------------------------------------------------
    # Crossed subprocesses folded into this matrix element
    # ------------------------------------------------------------------
    @staticmethod
    def crossing_keeps_helicity_states(base_process, crossed_process):
        """Whether every slot of `crossed_process` keeps the helicity states
        of the same slot of `base_process`.

        For a crossed event the backend reports the BASE helicity row whose
        configuration equals the crossed one, slot by slot (selected_hel_code
        in backend/<variant>/SigmaKin.cc), and the entry indexes the base
        helicity table with it. A slot whose crossed particle has a state the
        base slot does not know -- a massive vector moved into a fermion
        slot -- has no such row, so its helicity would come out wrong."""
        model = base_process.get('model')

        def states(process):
            return [set(model.get_particle(leg.get('id')).get_helicity_states())
                    for leg in process.get_legs_with_decays()]
        base, crossed = states(base_process), states(crossed_process)
        return len(base) == len(crossed) and \
            all(c <= b for b, c in zip(base, crossed))

    def class_diagram_validity(self):
        """Per flavor class (the FLAV half of an extended id), the positions
        of the diagrams that have that flavor."""
        diagrams = self.matrix_element.get('diagrams')
        return [set(i for i, diag in enumerate(diagrams)
                    if diag.has_flavor(tuple(flavors[0])))
                for flavors in self.all_flavors]

    def canonical_propagator(self, subset, pdg):
        """A propagator as (the canonical one of its two external-leg subsets,
        the PDG of the particle flowing into that subset), every external leg
        read as outgoing (an incoming particle is its outgoing antiparticle)."""
        nx = len(self.edge_names)
        rest = frozenset(range(1, nx + 1)) - subset
        if (len(subset), sorted(subset)) <= (len(rest), sorted(rest)):
            return subset, pdg
        if getattr(self, '_anti', None) is None:
            self._anti = crossing_table.make_anti(self.model)
        return rest, self._anti(pdg)

    def diagram_signatures(self):
        """Per diagram position, its propagators (canonical_propagator). Read
        with all the legs outgoing, the diagrams of a process and of its
        crossings are the same graphs, only the external legs are renamed;
        so the signature is crossing covariant, W+ and W- exchanges -- the
        same propagators up to the particle -- included.

        MadGraph records, on the leg a vertex creates, the particle flowing
        into the legs it combines (in the all-outgoing reading: the u d~ pair
        of u d~ > w+ is joined by a w-, the one flowing into {u, d~})."""
        signatures = []
        for diagram in self.diagrams:
            subsets, props = {}, []
            for vertex in diagram.get('vertices')[:-1]:
                legs = vertex.get('legs')
                subset = frozenset().union(*[
                    subsets.get(leg.get('number'),
                                frozenset([leg.get('number')]))
                    for leg in legs[:-1]])
                subsets[legs[-1].get('number')] = subset
                props.append(self.canonical_propagator(subset,
                                                       legs[-1].get('id')))
            signatures.append(frozenset(props))
        return signatures

    def crossed_diagram_map(self, crossed_signatures, base_signatures, D):
        """The position of the base diagram each crossed diagram is when
        crossed leg k is fed to base slot D[k] -- the one with the same
        propagators (diagram_signatures) -- or None. The backend fills the
        base's amp2 at the crossed momenta, so this is the amp2 slot of each
        crossed diagram.

        A crossed diagram with no counterpart is one the rows served by D
        never need: a flavor-merged base keeps only the diagrams its own
        flavors use, and a row reaches the others through another D (of the
        u-channel w+ and w- of q q > q q only the one of the flavor order the
        base keeps survives; its crossings need both, one per row). Several
        base diagrams with the same signature are interchangeable (same
        propagators, same particles); the one at the same position is taken
        first."""
        by_signature = defaultdict(list)
        for b, sig in enumerate(base_signatures):
            by_signature[sig].append(b)
        used = set()
        cmap = []
        for i, sig in enumerate(crossed_signatures):
            sig = frozenset(self.canonical_propagator(
                                frozenset(D[l - 1] + 1 for l in subset), pdg)
                            for (subset, pdg) in sig)
            options = [b for b in by_signature.get(sig, []) if b not in used]
            if not options:
                cmap.append(None)
                continue
            b = i if i in options else options[0]
            used.add(b)
            cmap.append(b)
        return cmap

    def crossed_matrix_element(self, record):
        """The matrix element of the crossed process of `record` = (process,
        base_perm, crossed_perm), built for its metadata only: what an
        expanded output would have generated for it (cross_amplitude on this
        base, the merged-flavor trimming of HelasMultiProcess), without the
        colour basis this entry takes from the base."""
        proc, base_perm, crossed_perm = record
        # the amplitude the matrix element was built from, which the
        # expansion crosses too (the rebuilt base_amplitude would lose the
        # merged-flavor content of the vertices, see HelasMatrixElement)
        base = getattr(self.matrix_element, 'crossing_amplitude', None)
        if base is None:
            base = self.matrix_element.get('base_amplitude')
        amplitude = diagram_generation.MultiProcess.cross_amplitude(
            base, proc, base_perm, crossed_perm)
        if 'crossed_processes' in amplitude:
            amplitude.set('crossed_processes', [])
        matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                          gen_color=False)
        matrix_element.get_external_flavors()
        matrix_element.set('base_amplitude',
                           matrix_element.get_base_amplitude())
        return matrix_element

    def get_crossing_table(self, matrix_element):
        """The crossing table the C++ of this matrix element is written from
        (ProcessTables.h, flavorPDG): the one prepare_crossed_subprocesses
        built for the subprocesses.json entries when there is one, so the two
        agree on every row K."""
        table = getattr(self, 'targeted_crossing_table', None)
        if table is not None and matrix_element is self.matrix_element:
            return table
        return super().get_crossing_table(matrix_element)

    def prepare_crossed_subprocesses(self, merge_same_topologies=True,
                                     collect_mirror=True):
        """Build, before any file of this matrix element is written, the
        subprocesses.json entries of the crossed subprocesses folded into it
        (merge_crossing='record'), and the crossing table they are evaluated
        with; get_crossed_subprocess_info then only gives them their paths.

        Per recorded crossed process: a matrix element of its own for its
        metadata (crossed_matrix_element) -- what an expanded output would
        have integrated -- and the crossing table serving each of its flavor
        rows in its own slot order (crossing_table.build_table with target
        rows), which then is the table of this output (get_crossing_table, so
        ProcessTables.h agrees with the entries on every row K).

        Record mode stores a crossing and its beam swap as two records; an
        expanded output folds the second into the first's mirror
        (has_mirror_process) when it collects mirrors (`collect_mirror`, the
        group_subprocesses option), and so is it here: the first gets mirror,
        the second no entry (and no row). Otherwise both get entries of their
        own, as the expanded output writes both.

        Every consistency check of the entries runs here, so that an entry
        which cannot be written stops the output before this directory is."""
        from madgraph.iolibs.export_v4 import ProcessExporterFortran
        me = self.matrix_element
        records = list(me.get('crossed_processes')) \
            if 'crossed_processes' in me else []
        n_initial = self.n_initial
        base_process = me.get('processes')[0]

        def key(row):
            return crossing_table.physical_key(row, n_initial)

        def swapped(k):
            return ((k[0][1], k[0][0]), k[1])

        crossed = []
        for record in records:
            name = record[0].base_string()
            if not self.crossing_keeps_helicity_states(base_process, record[0]):
                raise MadGraph5Error(
                    'crossed process %s moves a leg into a slot of %s with '
                    'other helicity states: it should have been expanded '
                    '(crossing_foldable)' % (name, self.name))
            xme = self.crossed_matrix_element(record)
            xmeta = SubprocessMetadataMG7(xme, merge_same_topologies)
            xinfo, xtags, xclass = xmeta.get_subprocess_info(None, None)
            cover = set()
            for flavor in xinfo['flavors']:
                for option in flavor['options']:
                    cover.add(key(option))
                    if flavor['mirror']:
                        cover.add(swapped(key(option)))
            crossed.append({'name': name, 'xmeta': xmeta,
                            'xinfo': xinfo, 'xtags': xtags, 'xclass': xclass,
                            'cover': cover})

        mirrored, skipped = set(), set()
        if n_initial == 2 and collect_mirror:
            for r, entry in enumerate(crossed):
                if r in skipped or not entry['cover']:
                    continue
                target = set(swapped(k) for k in entry['cover'])
                if target & entry['cover']:
                    continue
                for r2 in range(r + 1, len(crossed)):
                    if r2 not in skipped and crossed[r2]['cover'] == target:
                        mirrored.add(r)
                        skipped.add(r2)
                        break

        inputs = ProcessExporterFortran.crossing_table_inputs(self, me)
        table_records = []
        for r, (proc, dep, seed) in enumerate(inputs['records']):
            targets = [] if r in skipped else \
                [tuple(option) for flavor in crossed[r]['xinfo']['flavors']
                 for option in flavor['options']]
            table_records.append((proc, dep, seed, targets))
        table = crossing_table.build_table(
            inputs['nexternal'], inputs['ninitial'], inputs['labels'],
            ProcessExporterFortran.crossing_base_entries(self, me, 'classes'),
            table_records, inputs['model'], fixed=inputs['fixed'])
        for r, record in enumerate(table.records):
            if r not in skipped and not record.complete():
                raise MadGraph5Error(
                    'the crossing table of %s cannot serve the crossed process '
                    '%s as it comes (%s): output it with --use_crossing=False'
                    % (self.name, crossed[r]['name'],
                       record.unserved[:3] or 'nothing served'))
        self.targeted_crossing_table = table
        self.crossed_entries = [
            entry for r, xentry in enumerate(crossed) if r not in skipped
            for entry in self.crossed_entries_of(xentry, table.records[r],
                                                 r in mirrored)]

    def crossed_entries_of(self, xentry, record, mirrored):
        """The entries of one crossed process (prepare_crossed_subprocesses):
        one per crossing-table row K serving it, as (info, diagram_tags,
        subprocess_class) like get_subprocess_info, without the paths.

        A crossed entry is evaluated by THIS library -- same me_path, flavor
        index the extended id K*nflav + flav of the crossing-table row K that
        serves it. The backend gathers the momenta through the row and hands
        everything back in the BASE numbering: the diagram amp2 (the diagram
        it selects is a base one), the colour-flow index and the helicity code.
        So the entry is written in that numbering too:

          - incoming/outgoing, the flavor rows (in their slot order) and their
            mirror, the channels (topologies, symmetric-diagram permutations,
            propagator pdgs), pdg_color_types and qcd_power are the crossed
            process's own, as an expanded output writes them; each channel
            diagram is then renumbered to the base diagram it is under the row
            (crossed_diagram_map), and its active flavors / colours are the
            base's for that diagram;
          - color_flows is indexed by the BASE flow, each flow crossed onto the
            crossed legs (colour <-> anticolour for a leg changing side);
          - helicities is the base table (crossing_keeps_helicity_states);
          - diagram_count is the base's: the length of the amp2 array.

        One entry per row K: the diagram renumbering depends on it."""
        table = self.targeted_crossing_table
        n_external = self.matrix_element.get_nexternal_ninitial()[0]
        n_initial = self.n_initial
        nflav = len(self.matrix_element.get_external_flavors_with_iden())
        base_flows, _ = self.get_color_flow_dicts()
        validity = self.class_diagram_validity()
        base_signatures = self.diagram_signatures()
        base_channel = set(diag['diagram'] for channel in self.channels
                           for diag in channel['diagrams'])
        name, xinfo, xtags = xentry['name'], xentry['xinfo'], xentry['xtags']
        xmeta = xentry['xmeta']
        xlegs = xmeta.process.get_legs_with_decays()
        xsignatures = xmeta.diagram_signatures()
        served = dict((tuple(a.pdgs), a) for a in record.assignments)

        # the flavor rows of each row K, grouped by (extended id, initial
        # state) -- one per flavor of the base -- in the order of the crossed
        # process's own flavors
        by_row = {}
        for flavor in xinfo['flavors']:
            for option in flavor['options']:
                a = served[tuple(option)]
                mirror = flavor['mirror'] or (
                    mirrored and option[0] != option[1])
                group = (a.index(nflav), tuple(option[:n_initial]), mirror)
                groups = by_row.setdefault(a.K, {})
                groups.setdefault(group, (a.flav, []))[1].append(list(option))

        entries = []
        for K in sorted(by_row):
            perm = table[K]
            cmap = self.crossed_diagram_map(xsignatures, base_signatures,
                                            perm.D)
            flavors = [{"index": index, "options": options, "mirror": mirror}
                       for (index, _, mirror), (_, options)
                       in by_row[K].items()]
            flavor_classes = [flav for flav, _ in by_row[K].values()]

            channels, tags = [], []
            for channel, channel_tags in zip(xinfo['channels'], xtags):
                diagrams, diagram_tags = [], []
                for diag, tag in zip(channel['diagrams'], channel_tags):
                    b = cmap[diag['diagram']]
                    # a diagram these rows never need (no counterpart, or none
                    # of their flavors has it) gets no channel here
                    active = [] if b is None else \
                        [f for f, c in enumerate(flavor_classes)
                         if b in validity[c]]
                    if not active:
                        continue
                    diag = dict(diag)
                    diag['diagram'] = b
                    diag['active_flavors'] = active
                    diag['active_colors'] = list(self.diag_jamps[b]) \
                        if self.color_basis else [0]
                    diagrams.append(diag)
                    diagram_tags.append(tag)
                if diagrams:
                    channel = dict(channel)
                    channel['diagrams'] = diagrams
                    channels.append(channel)
                    tags.append(diagram_tags)

            # The backend picks the event's diagram among the base's channel
            # diagrams, by their amp2: each one these rows can fill needs a
            # channel here, or the event has none to be written with (and its
            # flavor none to be sampled in).
            needed = set(b for c in flavor_classes for b in validity[c]) \
                & base_channel
            reached = set(diag['diagram'] for channel in channels
                          for diag in channel['diagrams'])
            sampled = set(f for channel in channels
                          for diag in channel['diagrams']
                          for f in diag['active_flavors'])
            if needed - reached or len(sampled) != len(flavors):
                raise MadGraph5Error(
                    'crossed process %s (row %d of %s): the base diagrams %s '
                    'have no channel of its own and %d of its %d flavor groups '
                    'none at all; output it with --use_crossing=False'
                    % (name, K, self.name, sorted(needed - reached),
                       len(flavors) - len(sampled), len(flavors)))

            if base_flows is not None:
                flow_dicts = []
                for flow in base_flows:
                    crossed_flow = {}
                    for k in range(len(perm.D)):
                        colour, acolour = flow[perm.D[k] + 1]
                        if perm.SD[k] == -1:
                            colour, acolour = acolour, colour
                        crossed_flow[k + 1] = (colour, acolour)
                    flow_dicts.append(crossed_flow)
                color_flows = [
                    [list(flow[leg.get('number')]) for leg in xlegs]
                    for flow in flow_dicts]
                color_codes, color_slots = self.get_color_code_tables(
                    flow_dicts, xlegs)
            else:
                color_flows = [[[0, 0]] * n_external]
                color_codes = color_slots = None

            info = dict(xinfo)
            info.update({
                "channels": channels,
                "flavors": flavors,
                "color_flows": color_flows,
                "color_codes": color_codes,
                "color_slots": color_slots,
                "diagram_count": len(self.diagrams),
                "helicities": list(self.matrix_element.get_helicity_matrix()),
            })
            entries.append((K, info, tags, xentry['xclass']))
            logger.debug('%s: crossed subprocess %s folded in (row %d, '
                         '%d flavor(s)%s)', self.name, name, K, len(flavors),
                         ', mirrored' if mirrored else '')
        return entries

    def get_crossed_subprocess_info(self, proc_dir, lib_me_path):
        """The subprocesses.json entries of the crossed subprocesses folded into
        this matrix element (prepare_crossed_subprocesses built them), as a
        list of (info, diagram_tags, subprocess_class) like
        get_subprocess_info, evaluated by the library `lib_me_path` of the
        directory `proc_dir`."""
        entries = []
        for K, info, tags, sclass in getattr(self, 'crossed_entries', []):
            info = dict(info)
            info.update({
                "me_path": lib_me_path,
                "path": proc_dir,
                # evaluated by the library of `base` at crossing-table row
                # `row` (the GPU backend cannot, see launch.py)
                "crossing": {"base": proc_dir, "row": K},
            })
            entries.append((info, tags, sclass))
        return entries


class SubprocessMetadataMG7(OneProcessExporterMG7):
    """The subprocesses.json metadata of a matrix element that gets no C++ of
    its own: a crossed subprocess folded into its base
    (OneProcessExporterMG7.get_crossed_subprocess_info)."""

    def __init__(self, matrix_element, merge_same_topologies=True):
        # deliberately not OneProcessExporterCPP.__init__: nothing is written
        self.init_subprocess_metadata(matrix_element, merge_same_topologies)
