import json
import os

import numpy as np
import torch

import madspace as ms

BATCH_SIZE = 100
CM_ENERGY = 13000.0
rng = np.random.default_rng(5678)

TTGG_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "processes", "ttgg.json"
)


def test_mlm_clustering():
    with open(TTGG_FILE) as f:
        diagrams = json.load(f)

    topologies = [
        ms.Topology(
            ms.Diagram(
                d["incoming_masses"],
                d["outgoing_masses"],
                [ms.Propagator(*p) for p in d["propagators"]],
                d["vertices"],
            )
        )
        for d in diagrams
    ]
    permutations = [d["permutations"] for d in diagrams]
    offset = 0
    diagram_indices = []
    for d in diagrams:
        n_perms = len(d["permutations"])
        diagram_indices.append(list(range(offset, offset + n_perms)))
        offset += n_perms

    clustering = ms.MLMClustering(topologies, permutations, diagram_indices)

    first = diagrams[0]
    mapping = ms.PhaseSpaceMapping(
        topologies[0], CM_ENERGY, permutations=first["permutations"]
    )
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    perm_count = len(first["permutations"])
    condition = (
        []
        if perm_count <= 1
        else [rng.integers(0, perm_count, BATCH_SIZE, dtype=np.int32)]
    )
    p_ext, _, _, _ = mapping.map_forward([r], condition)
    ren_scale, fact_scale1, fact_scale2, outgoing_scales, diagram_index = clustering(
        p_ext
    )
    from icecream import ic

    print(ren_scale, fact_scale1, fact_scale2, outgoing_scales, diagram_index)
