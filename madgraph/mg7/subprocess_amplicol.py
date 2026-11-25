"""Subprocesses with AmpliCol matrix elements ('[amplicol]' processes).

Each subprocess is one UMAMI channel of an AmpliCol provider: colour orderings
integrated at leading colour over one colour-ordered phase-space mapping, its
first phase-space channel. A colour ordering with partners (where the colour
singlets attach to the colour lines, the s-channel resonances it maps out) is
copied into the subprocess of each partner, and every copy weighted by its
mapping's density over the partners' densities, as in AmpliCol. Resonances that
cannot go on shell with the param card masses and widths are dropped at launch.
A second pass re- or
unweights the events to full colour with the ratio of the full-colour to the
leading-colour matrix element summed over all colour orderings, at the
helicity of the event. Under lc_flavors = two_lines_resample, the first pass
only samples the flavours with at most two quark lines and the second one
switches events to the others, in proportion to the leading colour of all
flavours at the event (the flavour switch of the ratio library).
MLM and CKKW-L scales cluster each event along the planar diagrams of its
leading-colour flow (amplicol_clustering).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import TYPE_CHECKING

from madgraph.mg7.amplicol_clustering import (
    CLUSTERING_MODEL,
    VertexTable,
    madspace_lines,
    planar_diagrams,
)
from madgraph.mg7.common import lhe_helicities, logger, mlm_clustering_history, ms
from madgraph.mg7.observables import clean_pids
from madgraph.mg7.subprocess_madmatrix import (
    Channel,
    MadgraphSubprocess,
    PhaseSpace,
)

if TYPE_CHECKING:
    from madgraph.mg7.madgraph_process import MadgraphProcess


def is_amplicol(meta: dict) -> bool:
    return meta.get("backend") == "amplicol"


def amplicol_parameters(meta: dict, process: MadgraphProcess) -> dict[str, float]:
    """The values of the mutable model parameters in the param card."""
    parameters = {}
    for name, (block, code) in meta["parameters"].items():
        try:
            parameters[name] = float(
                process.param_card.get_value(block.lower(), tuple(code))
            )
        except (KeyError, ValueError):
            continue
    return parameters


def chain_topology(incoming_masses: list[float], outgoing_masses: list[float]):
    """A t-channel chain without propagators, for the LHE completion."""
    n_out = len(outgoing_masses)
    vertices = [["i0", "p0", "o0"]]
    vertices += [[f"p{i - 1}", f"p{i}", f"o{i}"] for i in range(1, n_out - 1)]
    vertices.append(["i1", f"p{n_out - 2}", f"o{n_out - 1}"])
    propagators = [ms.Propagator() for _ in range(n_out - 1)]
    return ms.Topology(
        ms.Diagram(incoming_masses, outgoing_masses, propagators, vertices)
    )


def channel_topology(
    incoming_masses: list[float],
    outgoing_masses: list[float],
    order: list[int],
    resonances: list[tuple[int, list[int]]],
    process: MadgraphProcess,
) -> tuple[ms.Topology, list[int], int]:
    """The topology of a colour-ordered phase-space channel, a t-channel chain
    through the final-state legs and the resonances (pdg, legs) that decay to
    them, its colour order over the children of that chain, and the number of
    chain propagators (which come first among the propagators)."""
    n_in = len(incoming_masses)
    legs = [frozenset(r_legs) for _, r_legs in resonances]
    top = [i for i, r in enumerate(legs) if not any(r < other for other in legs)]
    children = []
    for slot in order:
        if slot < n_in:
            continue
        child = next((("r", i) for i in top if slot in legs[i]), ("o", slot))
        if child not in children:
            children.append(child)
        elif children[-1] != child:
            raise RuntimeError(f"the resonance legs of {order} are not contiguous")
    chain_count = len(children) - 1

    def ref(child):
        kind, index = child
        return f"o{index - n_in}" if kind == "o" else f"p{chain_count + index}"

    refs = [ref(child) for child in children]
    if len(children) == 1:
        vertices = [["i0", "i1", refs[0]]]
    else:
        vertices = [["i0", "p0", refs[0]]]
        vertices += [[f"p{i - 1}", f"p{i}", refs[i]] for i in range(1, chain_count)]
        vertices.append(["i1", f"p{chain_count - 1}", refs[-1]])
    position = {slot: i for i, slot in enumerate(order)}
    for index, r_legs in enumerate(legs):
        inner = [
            i for i, other in enumerate(legs)
            if other < r_legs and not any(other < mid < r_legs for mid in legs)
        ]
        covered = set().union(*(legs[i] for i in inner))
        decay = [(min(position[s] for s in legs[i]), ref(("r", i))) for i in inner]
        decay += [(position[s], ref(("o", s))) for s in r_legs - covered]
        vertices.append([f"p{chain_count + index}", *(r for _, r in sorted(decay))])
    propagators = [ms.Propagator() for _ in range(chain_count)]
    for pdg, _ in resonances:
        propagators.append(ms.Propagator(
            mass=process.get_mass(abs(pdg)),
            width=process.get_width(abs(pdg)),
            pdg_id=pdg,
        ))
    topology = ms.Topology(
        ms.Diagram(incoming_masses, outgoing_masses, propagators, vertices)
    )
    color_order = []
    for slot in order:
        node = slot if slot < n_in else n_in + children.index(
            next((("r", i) for i in top if slot in legs[i]), ("o", slot))
        )
        if node not in color_order:
            color_order.append(node)
    return topology, color_order, chain_count


def channel_topologies(meta: dict, process: MadgraphProcess) -> list[tuple]:
    """channel_topology of every phase-space channel of a subprocess."""
    incoming_masses = [process.get_mass(pid) for pid in clean_pids(meta["incoming"])]
    outgoing_masses = [process.get_mass(pid) for pid in clean_pids(meta["outgoing"])]
    return [
        channel_topology(
            incoming_masses,
            outgoing_masses,
            channel["order"],
            [(r["pdg"], r["legs"]) for r in channel["resonances"]],
            process,
        )
        for channel in meta["phasespace_channels"]
    ]


def active_resonances(meta: dict, get_mass, get_width) -> dict:
    """The subprocess meta without the s-channel resonances that cannot go on
    shell: a resonance is active if its width is positive and its mass above
    the masses of its decay legs. Phase-space channels that then agree are kept
    apart: each is the channel of a copy of the flavours, in this subprocess or
    another one, and its density counts once in the weights of every copy."""
    leg_masses = [
        get_mass(pid) for pid in clean_pids(meta["incoming"] + meta["outgoing"])
    ]

    def active(resonance):
        pdg = abs(resonance["pdg"])
        return get_width(pdg) > 0 and get_mass(pdg) > sum(
            leg_masses[leg] for leg in resonance["legs"]
        )

    for resonance in meta.get("colored_resonances", []):
        if active(resonance):
            raise RuntimeError(
                f"the coloured s-channel resonance {resonance['pdg']} can go on "
                "shell, which [amplicol] phase-space mappings do not follow yet; "
                "generate the process with on-shell decays or with the madmatrix "
                "backend"
            )
    kept = [
        [i for i, r in enumerate(channel["resonances"]) if active(r)]
        for channel in meta["phasespace_channels"]
    ]
    channels = [
        {**channel, "resonances": [channel["resonances"][i] for i in keep]}
        for channel, keep in zip(meta["phasespace_channels"], kept)
    ]
    flavors = [
        {
            **flavor,
            "resonance_pdgs": [
                [pdgs[i] for i in kept[index]]
                for index, pdgs in zip(flavor["channels"], flavor["resonance_pdgs"])
            ],
        }
        for flavor in meta["flavors"]
    ]
    meta = {**meta, "phasespace_channels": channels, "flavors": flavors}
    if "switch_resonance_pdgs" in meta:
        meta["switch_resonance_pdgs"] = [
            [[pdgs[i] for i in keep] for pdgs, keep in zip(target, kept)]
            for target in meta["switch_resonance_pdgs"]
        ]
    return meta


def quark_lines(pdgs: list[int]) -> int:
    return sum(1 for pid in pdgs if 0 < abs(pid) <= 6) // 2


def with_prior_factors(meta: dict, partner_share: bool) -> dict:
    """Each flavour's prior factor: its factor, divided by its number of
    phase-space channels if partner_share (its expected share in this one)."""
    return {**meta, "flavors": [
        {
            **flavor,
            "prior_factor": flavor.get("factor", len(flavor["options"]))
            / (len(flavor["channels"]) if partner_share else 1),
        }
        for flavor in meta["flavors"]
    ]}


def lc_flavor_mode(run_card) -> str:
    """lc_flavors, two_lines_resample read as all without a full-colour pass
    to switch flavours in."""
    amplicol = run_card["amplicol"]
    mode = amplicol["lc_flavors"]
    if mode == "two_lines_resample" and amplicol.get("color_accuracy") == "lc":
        if not getattr(lc_flavor_mode, "warned", False):
            lc_flavor_mode.warned = True
            logger.warning("color_accuracy = lc: lc_flavors = two_lines_resample has no "
                           "full-colour pass to switch flavours in, using lc_flavors = all")
        return "all"
    return mode


def lc_flavors(meta: dict, mode: str) -> dict:
    """The flavours of the leading-colour pass: all of them, or only those with
    at most two quark lines (two_lines, the others neglected, and
    two_lines_resample, the others reached by the flavour switch)."""
    flavors = meta["flavors"]
    if mode == "all":
        return meta
    kept = [f for f in flavors if quark_lines(f["options"][0]) <= 2]
    if not kept:
        raise RuntimeError(f"lc_flavors = {mode} leaves no flavour")
    return {**meta, "flavors": kept}


@lru_cache(maxsize=None)
def _load_switch(path: str) -> dict:
    with open(path) as stream:
        return json.load(stream)


def flavor_switch(meta: dict, process: MadgraphProcess) -> dict | None:
    """The flavour switch of an [amplicol] subprocess under two_lines_resample
    (its targets, colour flows), None if it switches to no flavour."""
    if not is_amplicol(meta) or lc_flavor_mode(process.run_card) != "two_lines_resample":
        return None
    dropped = any(quark_lines(f["options"][0]) > 2 for f in meta["flavors"])
    if "flavor_switch" not in meta:
        if dropped:
            raise RuntimeError(
                "lc_flavors = two_lines_resample needs an [amplicol] output with the "
                "flavour switch: regenerate the output"
            )
        return None
    # relative to the process directory, as the other paths of the meta
    switch = _load_switch(os.path.abspath(meta["flavor_switch"]))
    if switch["switch_lines"] != 3:
        raise RuntimeError("the flavour switch of the output does not start at 3 lines")
    if not switch["targets"]:
        return None
    return switch


def event_flavors(meta: dict, process: MadgraphProcess) -> list[dict]:
    """The flavours of a subprocess as its events index them (LHE, systematics):
    those of the leading-colour pass, then the targets of the flavour switch."""
    if not is_amplicol(meta):
        return meta["flavors"]
    flavors = lc_flavors(meta, lc_flavor_mode(process.run_card))["flavors"]
    switch = flavor_switch(meta, process)
    if switch is None:
        return flavors
    return flavors + [
        {"index": -1, "options": [pdgs], "weights": [1.0]} for pdgs in switch["targets"]
    ]


def switch_diagrams(meta: dict, diagrams: list) -> tuple[list, list[list[int]]]:
    """The LHE diagrams with those of the switched events appended, and per
    switch target the diagram replacing each diagram: the same phase-space
    channel, its resonance charges on the target's legs."""
    if not diagrams or "switch_resonance_pdgs" not in meta:
        return diagrams, []
    extended = list(diagrams)
    index = {diagram: i for i, diagram in enumerate(extended)}
    switch_map = []
    for target in meta["switch_resonance_pdgs"]:
        row = []
        for channel, _ in diagrams:
            diagram = (channel, tuple(target[channel]))
            if diagram not in index:
                index[diagram] = len(extended)
                extended.append(diagram)
            row.append(index[diagram])
        switch_map.append(row)
    return extended, switch_map


def lhe_diagrams(meta: dict) -> tuple[list[tuple[int, tuple]], list[list[float]]]:
    """The diagrams that the LHE writer takes the resonances of an event from,
    one per phase-space channel and resonance charges, and per flavour a mask
    of those it can be written with. None if there are no resonances."""
    if not any(channel["resonances"] for channel in meta["phasespace_channels"]):
        return [], []
    diagrams = []
    for flavor in meta["flavors"]:
        for channel, pdgs in zip(flavor["channels"], flavor["resonance_pdgs"]):
            if (channel, tuple(pdgs)) not in diagrams:
                diagrams.append((channel, tuple(pdgs)))
    masks = [
        [
            float(diagram in zip(flavor["channels"], map(tuple, flavor["resonance_pdgs"])))
            for diagram in diagrams
        ]
        for flavor in meta["flavors"]
    ]
    return diagrams, masks


def amplicol_subproc_args(meta: dict, process: MadgraphProcess) -> ms.SubprocArgs:
    """LHE completion of an AmpliCol subprocess: its diagrams are the
    phase-space channels, with the resonances they map out."""
    meta = active_resonances(meta, process.get_mass, process.get_width)
    switch = flavor_switch(meta, process)
    meta = lc_flavors(meta, lc_flavor_mode(process.run_card))
    diagrams, _ = lhe_diagrams(meta)
    color_flows = meta["color_flows"]
    colors = sorted({flavor["color_index"] for flavor in meta["flavors"]})
    flavors = meta["flavors"]
    if switch is not None:
        # the switched events: their flavours, colour flows and diagrams
        diagrams, _ = switch_diagrams(meta, diagrams)
        if len(color_flows) != switch["color_flow_offset"]:
            raise RuntimeError("the flavour switch does not match the colour flows")
        color_flows = color_flows + switch["color_flows"]
        colors += list(range(switch["color_flow_offset"], len(color_flows)))
        flavors = flavors + [
            {"options": [pdgs], "weights": [1.0]} for pdgs in switch["targets"]
        ]
    if diagrams:
        topologies = channel_topologies(meta, process)
        particle_count = len(meta["incoming"]) + len(meta["outgoing"])
        by_channel = [
            [(d, pdgs) for d, (c, pdgs) in enumerate(diagrams) if c == channel]
            for channel in range(len(topologies))
        ]
        # the chain propagators come first: one less than the root's children
        chain_counts = [count for _, _, count in topologies]
        layout = dict(
            topologies=[topology for topology, _, _ in topologies],
            permutations=[
                [list(range(particle_count))] * len(channel) for channel in by_channel
            ],
            diagram_indices=[[d for d, _ in channel] for channel in by_channel],
            diagram_color_indices=[[colors] * len(channel) for channel in by_channel],
            diagram_propagator_pdgs=[
                [[0] * count + list(pdgs) for _, pdgs in channel]
                for channel, count in zip(by_channel, chain_counts)
            ],
        )
    else:
        layout = dict(
            topologies=[chain_topology(
                [process.get_mass(pid) for pid in clean_pids(meta["incoming"])],
                [process.get_mass(pid) for pid in clean_pids(meta["outgoing"])],
            )],
            permutations=[[]],
            diagram_indices=[[]],
            diagram_color_indices=[[]],
        )
    return ms.SubprocArgs(
        **layout,
        color_flows=color_flows,
        pdg_color_types={
            int(key): value for key, value in meta["pdg_color_types"].items()
        },
        helicities=lhe_helicities(meta, process.run_card),
        pdg_ids=[flavor["options"] for flavor in flavors],
        pdg_id_weights=[flavor.get("weights", []) for flavor in flavors],
    )


def file_key(path: str) -> tuple:
    """The cache key of a file: a new output into the same directory changes it."""
    stat = os.stat(path)
    return path, stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=None)
def _vertex_table(key: tuple) -> VertexTable:
    table = VertexTable.load(key[0])
    if table.contact_vertices:
        logger.warning(
            "the MLM clustering of [amplicol] events leaves out the diagrams of "
            "the contact vertices %s", table.contact_vertices
        )
    return table


def vertex_table(path: str) -> VertexTable:
    return _vertex_table(file_key(path))


@lru_cache(maxsize=None)
def _max_coupling_orders(key: tuple) -> tuple:
    with open(key[0]) as stream:
        return tuple(sorted(json.load(stream)["max_coupling_orders"].items()))


def max_coupling_orders(path: str) -> tuple:
    return _max_coupling_orders(file_key(path))


@lru_cache(maxsize=None)
def _clustering_diagrams(table_key, pdgs, n_in, flow, orders) -> list:
    return planar_diagrams(
        _vertex_table(table_key), list(pdgs), n_in,
        None if flow is None else [list(tags) for tags in flow], dict(orders),
    )


def clustering_diagrams(table_path, pdgs, n_in, flow, orders) -> list:
    """planar_diagrams, shared by the subprocesses with these arguments (flow
    and orders as tuples)."""
    return _clustering_diagrams(file_key(table_path), pdgs, n_in, flow, orders)


class AmplicolSubprocess(MadgraphSubprocess):
    def __init__(self, process, meta, subproc_id, unmerged_meta=None):
        if process.mlm_clustering and \
                mlm_clustering_history(process.run_card) != "all_diagrams":
            # there is no single-diagram weight to pick a diagram by
            raise RuntimeError(
                "the MLM clustering of [amplicol] subprocesses needs "
                "clustering_history = all_diagrams"
            )
        # before lc_flavors drops the flavours it switches to; the matrix
        # elements loaded by the base class need it
        self.switch = flavor_switch(meta, process)
        super().__init__(process, meta, subproc_id, unmerged_meta)
        self.meta = active_resonances(self.meta, process.get_mass, process.get_width)
        if unmerged_meta is not None and lc_flavor_mode(process.run_card) != "all":
            raise RuntimeError("lc_flavors needs unmerged [amplicol] subprocesses")
        self.meta = with_prior_factors(
            self.meta, process.run_card["amplicol"]["prior_partner_share"]
        )
        self.meta = lc_flavors(self.meta, lc_flavor_mode(process.run_card))
        if self.switch is not None and getattr(process, "custom_functions", None):
            raise RuntimeError(
                "lc_flavors = two_lines_resample changes the flavour of events after "
                "the custom cut, scale, PDF and bias functions saw it: use "
                "lc_flavors = all with [custom_functions]"
            )
        self.check_flavor_blind_cuts()

    def auto_buffered_steps_fraction(self) -> float:
        """No buffered training: the leading-colour matrix element is cheap
        enough to sample every batch afresh."""
        return 0.0

    def check_flavor_blind_cuts(self):
        """AmpliCol groups the flavours and identical-particle orderings that
        share a matrix element into one channel, so its cuts, built for the
        first flavour, must select the same particles for all of them."""
        options = [
            clean_pids(pdgs) for flavor in self.meta["flavors"]
            for pdgs in flavor["options"]
        ]
        # the flavours the events can be switched to as well
        if self.switch is not None:
            options += [clean_pids(pdgs) for pdgs in self.switch["targets"]]
        for item in self.process.cut_data:
            for group in item.observable_kwargs["select_pids"]:
                if len({tuple(pid in group for pid in pdgs) for pdgs in options}) > 1:
                    raise RuntimeError(
                        f"the cut {item.observable_kwargs['name']} does not select "
                        f"the same particles in all flavours of the [amplicol] "
                        f"subprocess {self.meta['incoming']} > {self.meta['outgoing']}"
                    )

    def build_mlm_clustering(self, options: dict):
        """The clustering along the planar diagrams of the colour flow of each
        flavour, one clustering class per flavour and flow."""
        # absolute: the caches outlive the run directory
        table_path = os.path.abspath(os.path.join("SubProcesses", CLUSTERING_MODEL))
        process_path = os.path.abspath(os.path.join(self.meta["path"], "process.json"))
        if not os.path.exists(table_path):
            raise RuntimeError(
                "the output has no vertices for the MLM clustering of its "
                "[amplicol] processes: regenerate it"
            )
        table = vertex_table(table_path)
        orders = max_coupling_orders(process_path)
        n_in = len(self.meta["incoming"])
        n_ext = n_in + len(self.meta["outgoing"])

        def value(pdg, kind, get):
            name = table.particles[pdg][kind]
            return 0.0 if name == "ZERO" else float(get(abs(pdg)))

        classes, keys, flavor_classes = [], {}, []
        for flavor in self.meta["flavors"]:
            pdgs = tuple(flavor["options"][0])
            flow = None
            if self.meta["color_flows"]:
                flow = tuple(map(tuple, self.meta["color_flows"][flavor["color_index"]]))
            key = (pdgs, flow)
            if key not in keys:
                diagrams = clustering_diagrams(table_path, pdgs, n_in, flow, orders)
                if not diagrams:
                    contact = (
                        "; it may come only from a contact vertex of the "
                        f"model ({table.contact_vertices}), which the "
                        "clustering does not support"
                        if table.contact_vertices else ""
                    )
                    raise RuntimeError(
                        f"no diagram of 3-point vertices of {pdgs} has the colour "
                        f"flow {flow} of its [amplicol] events, which the MLM "
                        f"clustering needs{contact}"
                    )
                keys[key] = len(classes)
                classes.append(ms.ClusteringClass(
                    [abs(pdg) for pdg in pdgs],
                    [
                        [
                            ms.ClusteringLine(
                                mask, pdg,
                                value(pdg, "mass", self.process.get_mass),
                                value(pdg, "width", self.process.get_width),
                            )
                            for mask, pdg in madspace_lines(diagram, n_ext, table.anti)
                        ]
                        for diagram in diagrams
                    ],
                ))
            flavor_classes.append(keys[key])
        return ms.MLMClustering(
            classes,
            flavor_classes,
            self.incoming_masses + self.outgoing_masses,
            pdg_color_types=table.color_types(),
            **options,
        )

    def build_merging_cut(self):
        """As for madmatrix, but ptlund takes the flavours of the first flavour
        for all, which only holds when no flavour changes its splittings."""
        merging_cut = getattr(self.process, "merging_cut", None)
        if merging_cut is not None and merging_cut[0] == ms.MergingScale.Measure.pt_lund:
            # the kernel reads the flavours of the incoming and merging legs only
            def splittings(pdgs):
                keep = self.merging_leg_mask(pdgs)
                return tuple(
                    (0 if abs(pdg) in (1, 2, 3) else abs(pdg)) if i < 2 or keep[i]
                    else None
                    for i, pdg in enumerate(pdgs)
                )
            if len({splittings(pdgs) for flavor in self.meta["flavors"]
                    for pdgs in flavor["options"]}) > 1:
                raise RuntimeError(
                    "the ptlund merging cut depends on the flavours of the event, "
                    "which the [amplicol] subprocess "
                    f"{self.meta['incoming']} > {self.meta['outgoing']} groups: "
                    "use ktdurham"
                )
        return super().build_merging_cut()

    def load_matrix_elements(self, api_path_formats, member_metas):
        run_args = self.process.run_card["run"]
        if any(device != "cpu" for device in self.process.device_types):
            raise RuntimeError("[amplicol] processes are only supported on CPUs")
        if run_args["dummy_matrix_element"]:
            raise RuntimeError("[amplicol] processes need their matrix elements")
        gridpack_args = self.process.run_card["gridpack"]
        if gridpack_args["save_gridpack"] and gridpack_args["include_source"]:
            raise RuntimeError(
                "gridpacks of [amplicol] processes cannot include the sources"
            )
        if self.meta["qcd_power"] < 0 and self.process.systematics_enabled():
            raise RuntimeError(
                "the scale and PDF systematics of [amplicol] processes that mix "
                "coupling orders are not supported yet: disable [systematics]"
            )
        self.supports_alpha_s = self.meta.get("supports_alpha_s", True)
        if not self.supports_alpha_s and not self.process.run_card["scale"]["fixed_ren_scale"]:
            raise RuntimeError(
                "the AmpliCol matrix elements of this model cannot take the running "
                "alpha_s, as the model has no SMINPUTS 3 parameter: use a fixed "
                "renormalization scale"
            )
        parameters = amplicol_parameters(self.meta, self.process)
        self.channel_index = self.meta["umami_channel"]
        self.matrix_elements = [
            self.load(self.meta["me_path"], self.meta["artifact"], parameters)
        ]
        self.surrogates = []
        self.ratio_matrix_element = None
        if self.process.run_card["amplicol"]["color_accuracy"] != "lc":
            ratio_parameters = dict(parameters)
            if self.switch is not None:
                ratio_parameters["flavor_switch"] = 1.0
            self.ratio_matrix_element = self.load(
                self.meta["ratio_path"], self.meta["ratio_artifact"], ratio_parameters
            )

    def load(self, path: str, artifact: str, parameters: dict):
        # loaded once for all channels; each load reads the whole pyamplicol artifact
        apis = self.process.amplicol_apis
        if path not in apis:
            positions = self.process.amplicol_positions
            apis[path] = self.process.load_matrix_element(
                path, artifact, parameters,
                position=None if positions is None else positions[path])
        return apis[path]

    def build_multi_channel_data(self):
        return None

    def build_multichannel_phasespace(self) -> PhaseSpace:
        """The first phase-space channel of the UMAMI channel, weighted by the
        density of its mapping over those of each flavour's partners, the other
        phase-space channels, which are sampled by the subprocesses of the
        flavour's copies."""
        if self.process.lepton_density is not None:
            # The dressed-lepton luminosity and its x sampling are wired into
            # the madmatrix subprocesses only; here the beams would silently
            # stay monochromatic.
            raise NotImplementedError(
                "dressed-lepton beams (beam.lepton_density) are not supported "
                "with the AmpliCol output; use the madmatrix one"
            )
        if self.process.upc_flux:
            # the same for the gamma-UPC luminosity
            raise NotImplementedError(
                "gamma-UPC beams (beam.upc_flux) are not supported with the "
                "AmpliCol output; use the madmatrix one"
            )
        flavors = self.meta["flavors"]
        particle_count = len(self.incoming_masses) + len(self.outgoing_masses)
        mappings = []
        for topology, color_order, _ in channel_topologies(self.meta, self.process):
            mappings.append(ms.PhaseSpaceMapping(
                topology,
                self.process.e_cm,
                leptonic=self.process.leptonic,
                invariant_power=self.process.run_card["phasespace"]["invariant_power"],
                t_channel_mode=ms.PhaseSpaceMapping.color_ordered,
                cuts=self.cuts,
                permutations=[list(range(particle_count))],
                color_ordered=ms.ColorOrderedMapping.Options(color_order),
                beam_rapidity=self.process.beam_rapidity,
                mirror_beams=self.mirror_beams,
            ))
        if any(flavor["channels"][0] != 0 for flavor in flavors):
            raise RuntimeError(
                "the [amplicol] flavours of a subprocess must sample its first "
                "phase-space channel: regenerate the output"
            )
        prefix = f"subproc{self.subproc_id}.amplicol0"
        discrete_sym, discrete_flavor = self.build_discrete(1, len(flavors), prefix)
        channel = Channel(
            phasespace_mapping=mappings[0],
            adaptive_mapping=self.build_vegas(mappings[0], prefix),
            discrete_sym=discrete_sym,
            discrete_flavor=discrete_flavor,
            channel_weight_indices=[0],
            name="C0",
            active_flavors=[list(range(len(flavors)))],
        )
        diagrams, diagram_masks = lhe_diagrams(self.meta)
        weights = None
        if len(mappings) > 1 or diagrams:
            weights = ms.MappingChannelWeights(
                mappings[:1],
                [
                    [float(index in flavor["channels"]) for index in range(len(mappings))]
                    for flavor in flavors
                ],
                [channel for channel, _ in diagrams],
                diagram_masks,
                partner_mappings=mappings[1:],
            )
        return PhaseSpace(
            mode="multichannel",
            channels=[channel],
            symfact=[None],
            mapping_chan_weights=weights,
        )

    def build_integrands(self, *args, **kwargs) -> list:
        integrands = super().build_integrands(*args, **kwargs)
        for integrand in integrands:
            if len(self.meta["flavors"]) > 1:
                integrand.set_flavor_prior_factors(
                    [flavor["prior_factor"] for flavor in self.meta["flavors"]]
                )
        return integrands

    def build_flat_phasespace(self) -> PhaseSpace:
        return self.build_multichannel_phasespace()

    def simplify_phasespace(self, multi_phasespace, cross_sections) -> PhaseSpace:
        return multi_phasespace

    def integrand_matrix_elements(self) -> list:
        # without an alpha_s input, the model value of the fixed scale is used
        inputs = [
            ms.MatrixElement.momenta_in,
            *([ms.MatrixElement.alpha_s_in] if self.supports_alpha_s else []),
            ms.MatrixElement.flavor_in,
            ms.MatrixElement.channel_in,
            ms.MatrixElement.random_helicity_in,
        ]
        outputs = [
            ms.MatrixElement.matrix_element_out,
            ms.MatrixElement.color_index_out,
            ms.MatrixElement.helicity_index_out,
        ]
        return [ms.MatrixElement(
            self.matrix_elements[0].index(), self.particle_count, inputs, outputs, 1,
            True, self.process.me_frame, self.process.incoming_count,
            self.channel_index,
        )]

    def has_weighting_steps(self) -> bool:
        return self.ratio_matrix_element is not None

    def parton_density_output(self) -> bool:
        return self.switch is not None

    def weighting_steps(self) -> list:
        """The second pass to full colour."""
        if self.ratio_matrix_element is None:
            return []
        inputs = [ms.MatrixElement.momenta_in, ms.MatrixElement.flavor_in,
                  ms.MatrixElement.channel_in, ms.MatrixElement.helicity_in]
        outputs = [ms.MatrixElement.matrix_element_ratio_out]
        switch_args = {}
        if self.switch is not None:
            inputs += [ms.MatrixElement.parton_densities_in,
                       ms.MatrixElement.random_flavor_in,
                       ms.MatrixElement.random_helicity_in,
                       ms.MatrixElement.random_color_in]
            outputs += [ms.MatrixElement.flavor_index_out,
                        ms.MatrixElement.helicity_index_out,
                        ms.MatrixElement.color_index_out]
            _, switch_map = switch_diagrams(self.meta, lhe_diagrams(self.meta)[0])
            switch_args = dict(
                flavor_target_count=len(self.switch["targets"]),
                # the targets follow the flavours of this subprocess, see event_flavors
                flavor_offset=len(self.meta["flavors"]),
                switch_diagram_map=switch_map,
            )
        ratio = ms.MatrixElement(
            self.ratio_matrix_element.index(),
            self.particle_count,
            inputs,
            outputs,
            1,
            False,
            self.process.me_frame,
            self.process.incoming_count,
            self.channel_index,
        )
        gen_args = self.process.run_card["generation"]
        return [ms.WeightingStep(
            [ratio],
            mode=self.process.run_card["amplicol"]["mode"],
            max_overweight_truncation=gen_args["max_overweight_truncation"],
            freeze_max_weight_after=gen_args["freeze_max_weight_after"],
            **switch_args,
        )]

    def first_pass_settings(self) -> dict | None:
        amplicol = self.process.run_card["amplicol"]
        if self.ratio_matrix_element is None or amplicol["mode"] != "unweight":
            return None
        return amplicol
