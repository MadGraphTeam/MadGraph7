import json
import os

from madgraph.various.diagram_symmetry import find_symmetry

def get_subprocess_info(matrix_element, proc_dir_name):
    model = matrix_element.get("processes")[0].get("model")
    amplitude = matrix_element.get("base_amplitude")

    process = amplitude.get("process")
    edge_names = {}
    legs = process.get("legs")
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

    sym_indices, sym_perms, _ = find_symmetry(matrix_element)
    print(sym_indices, sym_perms)
    diagrams = amplitude.get("diagrams")

    #TODO: when is the matrix element index not zero?
    me_index = 0

    channels = []
    channel_indices = []
    for diagram_index, (sym_index, sym_perm) in enumerate(zip(sym_indices, sym_perms)):
        if sym_index == 0:
            continue
        if sym_index < 0:
            print(len(channels), sym_index)
            channels[channel_indices[-sym_index - 1]]["diagrams"].append({
                "matrix_element": me_index,
                "diagram": diagram_index,
                "permutation": sym_perm,
            })
            channel_indices.append(-1)
            continue

        diagram = diagrams[diagram_index]
        vertices = []
        propagators = []
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
                prop_name = f"p{len(propagators)}"
                diagram_edge_names[legs[-1].get("number")] = prop_name
                vertex_props.append(prop_name)
                propagators.append(final_part.get_pdg_code())
            vertices.append(vertex_props)

        #TODO: determine active flavors

        #TODO: do we ever have cases where incoming and outgoing are not the same for
        #      all channels? if not, we could move these one level up
        channel_indices.append(len(channels))
        channels.append({
            "incoming": incoming,
            "outgoing": outgoing,
            "propagators": propagators,
            "vertices": vertices,
            "diagrams": [{
                "matrix_element": me_index,
                "diagram": diagram_index + 1,
                "permutation": sym_perm,
            }],
        })

    helicity_count = len([x for x in matrix_element.get_helicity_matrix()])
    flavor_count = 1 #TODO: determine flavor count

    return {
        "channels": channels,
        "path": os.path.join("SubProcesses", proc_dir_name, "api.so"),
        "flavor_count": flavor_count,
        "helicity_count": helicity_count,
        "has_mirror_process": matrix_element.get("has_mirror_process"),
        "crossing": False, #TODO: hardcoded to false for now
    }
