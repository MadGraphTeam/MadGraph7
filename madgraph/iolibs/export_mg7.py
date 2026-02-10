import json
import os
from collections import defaultdict

from madgraph.various.diagram_symmetry import find_symmetry, IdentifySGConfigTag


def get_subprocess_info(matrix_element, proc_dir_name, lib_me_path):
    model = matrix_element.get("processes")[0].get("model")
    amplitude = matrix_element.get("base_amplitude")

    process = amplitude.get("process")
    edge_names = {}
    legs = process.get("legs_with_decays")
    incoming = [None] * 2
    outgoing = [None] * (len(legs) - 2)
    for leg in legs:
        number = leg.get("number")
        if leg.get("state"):
            edge_names[number] = f"o{number - 3}"
            outgoing[number - 3] = leg.get("id")
        else:
            edge_names[number] = f"i{number - 1}"
            incoming[number - 1] = leg.get("id")

    sym_indices, sym_perms, _ = find_symmetry(
        matrix_element, lambda diag: IdentifySGConfigTag(diag, model)
    )
    diagrams = amplitude.get("diagrams")
    helas_diagrams = matrix_element.get("diagrams")
    all_flavors = matrix_element.get_external_flavors_unique()


    color_basis = matrix_element.get("color_basis")
    if color_basis:
        diag_jamps = defaultdict(list)
        for ijamp, col_basis_elem in enumerate(sorted(color_basis.keys())):
            for diag_tuple in color_basis[col_basis_elem]:
                diag_jamps[diag_tuple[0]].append(ijamp)

    channels = []
    channel_indices = []
    for diagram_index, (sym_index, sym_perm) in enumerate(zip(sym_indices, sym_perms)):
        if sym_index == 0:
            channel_indices.append(-1)
            continue

        active_colors = diag_jamps[diagram_index] if color_basis else [0]
        if sym_index < 0:
            channels[channel_indices[-sym_index - 1]]["diagrams"].append(
                {
                    "diagram": diagram_index,
                    "permutation": sym_perm,
                    "active_colors": active_colors,
                }
            )
            channel_indices.append(-1)
            continue

        helas_diagram = helas_diagrams[diagram_index]
        active_flavors = [
            i
            for i, flavors in enumerate(all_flavors)
            if helas_diagram.check_flavor(flavors[0], model)
        ]

        diagram = diagrams[diagram_index]
        vertices = []
        propagators = []
        on_shell_propagators = []
        diagram_edge_names = dict(edge_names)
        diag_vertices = diagram.get("vertices")
        for i_vert, vertex in enumerate(diag_vertices):
            legs = vertex.get("legs")
            # Last amplitude vertex does not create new edges
            vertex_props = [diagram_edge_names[leg.get("number")] for leg in legs[:-1]]

            final_part = model.get_particle(legs[-1].get("id"))
            if i_vert == len(diag_vertices) - 1:
                vertex_props.append(diagram_edge_names[legs[-1].get("number")])
            else:
                prop_index = len(propagators)
                prop_name = f"p{prop_index}"
                diagram_edge_names[legs[-1].get("number")] = prop_name
                vertex_props.append(prop_name)
                propagators.append(final_part.get_pdg_code())
                if legs[-1].get("onshell"):
                    on_shell_propagators.append(prop_index)
            vertices.append(vertex_props)

        channel_indices.append(len(channels))
        channels.append(
            {
                "propagators": propagators,
                "vertices": vertices,
                "on_shell_propagators": on_shell_propagators,
                "active_flavors": active_flavors,
                "diagrams": [
                    {
                        "diagram": diagram_index,
                        "permutation": sym_perm,
                        "active_colors": active_colors,
                    }
                ],
            }
        )

    n_external, n_initial = matrix_element.get_nexternal_ninitial()
    if color_basis:
        # First build a color representation dictionnary
        repr_dict = {}
        legs = process.get_legs_with_decays()
        for leg in legs:
            repr_dict[leg.get("number")] = model.get_particle(
                leg.get("id")
            ).get_color() * (-1) ** (1 + leg.get("state"))
        # Get the list of color flows
        color_flow_dicts = color_basis.color_flow_decomposition(repr_dict, n_initial)
        # And output them properly
        color_flows = [[
            [[color_flow_dict[leg.get("number")][i] for i in [0, 1]] for leg in legs]
            for color_flow_dict in color_flow_dicts
        ]] * len(all_flavors) #TODO: this is wrong for multiple flavors!!!
    else:
        color_flows = [[[[0, 0]] * n_external]] * len(all_flavors) #TODO: this is wrong for multiple flavors!!!

    # We need the both particle and antiparticle wf_ids, since the identity
    # depends on the direction of the wf.
    wf_ids = set(
        wf_id
        for d in matrix_element.get("diagrams")
        for wf in d.get("wavefunctions")
        for wf_id in [wf.get_pdg_code(), wf.get_anti_pdg_code()]
    )
    leg_ids = set(
        leg_id
        for p in matrix_element.get("processes")
        for l in p.get_legs_with_decays()
        for leg_id in [l.get("id"), model.get_particle(l.get("id")).get_anti_pdg_code()]
    )
    pdg_color_types = {}
    for part_id in sorted(list(wf_ids.union(leg_ids))):
        pdg_color_types[part_id] = model.get_particle(part_id).get_color()
        if abs(part_id) in model["merged_particles"]:
            for pdg in model["merged_particles"][abs(part_id)]:
                sign = -1 if part_id < 0 else 1
                pdg_color_types[sign * pdg] = sign * model.get_particle(part_id).get_color()

    flavors = [{"index": i, "options": options} for i, options in enumerate(all_flavors)]
    return {
        "incoming": incoming,
        "outgoing": outgoing,
        "channels": channels,
        "path": lib_me_path,
        "flavors": flavors,
        "color_flows": color_flows,
        "pdg_color_types": pdg_color_types,
        "diagram_count": len(diagrams),
        "helicities": list(matrix_element.get_helicity_matrix()),
        "has_mirror_process": matrix_element.get("has_mirror_process"),
    }
