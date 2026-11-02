"""FKSSectorFunction: the sector functions S_ij of fks_Sij.f.

The reference data of genps_fks.f (see test_fks_reference.py) store, for each
point, S_ij of the configuration at the event and at the soft, collinear and
soft-collinear counter-events, as fks_Sij.f gives it on the momenta of
genps_fks.f, together with the momentum of i per unit xi (p_i_fks_ev and
p_i_fks_cnt). The sector function of the configurations of the Born, from the
metadata of the FKS exporter, must give the same on those momenta, each event
in the configuration of its fks_index. On any momenta, the sector functions of
all the FKS pairs of a real process add up to one.
"""

import os

import numpy as np
import pytest
from fks_testing import FKS_DATA as DATA
from fks_testing import FKS_DATA_IDS as IDS
from fks_testing import evaluate, reference
from fks_testing import reference_born_mapping as born_mapping

import madspace as ms
from madspace.fks import fks_sector_function, real_masses

# position k + 2 of the counter-events k = 0, 1, 2 of genps_fks.f
COUNTER_EVENTS = {"soft": 2, "collinear": 3, "soft_collinear": 4}


def fortran_points(data):
    """The momenta, momenta of i per unit xi, fks_index and xi of the events
    and counter-events of the reference points, with S_ij of fks_Sij.f, where
    genps_fks.f generates them."""
    rows = []
    for conf in data["configs"]:
        for point in conf["points"]:
            if point["s_ij_ev"] != -1.0:
                rows.append(
                    (
                        "event",
                        point["p_ev"],
                        point["p_i_hat_ev"],
                        conf["nfks"] - 1,
                        point["xi_ev"],
                        point["s_ij_ev"],
                    )
                )
            for k, (name, position) in enumerate(COUNTER_EVENTS.items()):
                s = point["s_ij_cnt"][k]
                if s == -1.0:
                    continue
                assert point["p_i_hat_cnt"][position][0] >= 0.0
                rows.append(
                    (
                        name,
                        point["p_cnt"][position],
                        point["p_i_hat_cnt"][position],
                        conf["nfks"] - 1,
                        point["xi_cnt"][position],
                        s,
                    )
                )
    return tuple(np.array(column) for column in zip(*rows))


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_matches_fks_sij(path):
    data = reference(path)
    configurations = data["fks_configurations"]
    function = fks_sector_function(born_mapping(data), configurations)
    count = len(configurations["configurations"])
    assert function.configuration_count() == count
    assert function.particle_count() == data["nexternal"]

    kinds, momenta, p_i_hat, fks_index, xi, s_fortran = fortran_points(data)
    assert set(kinds) >= {"event", "soft"}
    (s,) = evaluate(
        function, momenta, p_i_hat, fks_index=fks_index if count > 1 else None
    )
    # fks_Sij.f takes the direction of i from its momentum, the energies from
    # frames it boosts into: the same to rounding. Below xi = 1e-6 (tiny) it
    # takes the direction from p_i_fks_cnt, that of i at the soft (or
    # soft-collinear) counter-event, where the real emission is evaluated in
    # the soft limit too. That differs from the direction at the event by
    # O(xi), as does S_ij, which here comes from the event.
    small = (xi > 0.0) & (xi <= 1e-6)
    np.testing.assert_allclose(s[~small], s_fortran[~small], rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(s[small], s_fortran[small], rtol=1e-6)
    assert ((s >= 0.0) & (s <= 1.0)).all()

    # each configuration on its own, without fks_index, as in the batch
    for c in range(count):
        part = fks_index == c
        if count == 1 or not part.any():
            continue
        arguments = function_arguments(data, c)
        single = ms.FKSSectorFunction(*([value] for value in arguments.values()))
        np.testing.assert_allclose(
            evaluate(single, momenta[part], p_i_hat[part])[0],
            s[part],
            rtol=1e-14,
            atol=0,
        )


def function_arguments(data, c):
    configuration = data["fks_configurations"]["configurations"][c]
    born_masses = born_mapping(data).external_masses()
    return dict(
        i_fks=configuration["i_fks"] - 1,
        j_fks=configuration["j_fks"] - 1,
        fks_j_from_i=[[j - 1 for j in js] for js in configuration["fks_j_from_i"]],
        colors=configuration["real_colors"],
        pdgs=configuration["real_pdgs"],
        masses=real_masses(born_masses, configuration),
    )


def all_sectors(configuration):
    """The sectors (i, j) of every FKS pair of the real process of a
    configuration: each pair once, as fks_j_from_i first gives it, and also
    the other way round for two final-state gluons or photons, which share
    their pair through h_damp."""
    aorg = [abs(pdg) == 21 or pdg == 22 for pdg in configuration["real_pdgs"]]
    sectors = []
    for k, js in enumerate(configuration["fks_j_from_i"]):
        for l in (j - 1 for j in js):
            if (k, l) in sectors or (l, k) in sectors:
                continue
            sectors.append((k, l))
            if aorg[k] and aorg[l] and l >= 2:
                sectors.append((l, k))
    return sectors


def real_processes():
    """One configuration of each real process of the reference data, with the
    reference events of all its configurations."""
    cases = []
    for path in DATA:
        data = reference(path)
        seen = {}
        for conf in data["configs"]:
            configuration = data["fks_configurations"]["configurations"][
                conf["nfks"] - 1
            ]
            key = tuple(configuration["real_pdgs"])
            seen.setdefault(key, (data, conf["nfks"] - 1, []))[2].extend(
                point["p_ev"] for point in conf["points"]
            )
        for key, (data, c, momenta) in seen.items():
            cases.append(
                pytest.param(
                    data,
                    c,
                    np.array(momenta),
                    id=f"{os.path.basename(path)[:-5]}-{'_'.join(map(str, key))}",
                )
            )
    return cases


@pytest.mark.parametrize("data, c, momenta", real_processes())
def test_partition_of_unity(data, c, momenta):
    """The sector functions of all the FKS pairs of a real process add up to
    one on its events, also where the configurations of the Born keep only
    some of them (the others are those of identical particles, or belong to
    other Borns)."""
    arguments = function_arguments(data, c)
    sectors = all_sectors(data["fks_configurations"]["configurations"][c])
    count = len(sectors)
    function = ms.FKSSectorFunction(
        [i for i, _ in sectors],
        [j for _, j in sectors],
        [arguments["fks_j_from_i"]] * count,
        [arguments["colors"]] * count,
        [arguments["pdgs"]] * count,
        [arguments["masses"]] * count,
    )
    batch = len(momenta)
    total = np.zeros(batch)
    for index, (i, _) in enumerate(sectors):
        # away from the soft limit, any momentum along that of i will do
        (s,) = evaluate(
            function,
            momenta,
            momenta[:, i],
            fks_index=np.full(batch, index) if count > 1 else None,
        )
        assert ((s >= 0.0) & (s <= 1.0)).all()
        total += s
    np.testing.assert_allclose(total, 1.0, rtol=1e-13)


def test_invalid_configurations():
    pdgs = [2, -2, 11, -11, 21]
    pairs = [[], [], [], [], [0, 1]]
    colors = [3, -3, 1, 1, 8]
    masses = [0.0] * 5
    with pytest.raises(ValueError, match="FKS pairs"):
        # (i, j) = (4, 2) is not an FKS pair
        ms.FKSSectorFunction([4], [2], [pairs], [colors], [pdgs], [masses])
    with pytest.raises(ValueError, match="final-state"):
        ms.FKSSectorFunction([0], [4], [pairs], [colors], [pdgs], [masses])
    with pytest.raises(ValueError, match="twice"):
        # a quark-gluon pair in both orientations
        ms.FKSSectorFunction(
            [4], [0], [[[4], [], [], [], [0, 1]]], [colors], [pdgs], [masses]
        )
    with pytest.raises(ValueError, match="number of particles"):
        ms.FKSSectorFunction(
            [4, 4],
            [0, 1],
            [pairs, pairs + [[]]],
            [colors, colors + [1]],
            [pdgs, pdgs + [22]],
            [masses, masses + [0.0]],
        )
    born = ms.PhaseSpaceMapping(
        [0.0] * 4, 1000.0, mode=ms.PhaseSpaceMapping.TChannelMode.rambo
    )
    finite = {
        "nincoming": 2,
        "born_pdgs": [2, -2, 11, -11],
        "configurations": [
            {
                "i_fks": 5,
                "j_fks": 1,
                "ij_pdg": 2,
                "real_pdgs": pdgs,
                "real_colors": colors,
                "fks_j_from_i": [[] for _ in pdgs],
            }
        ],
    }
    with pytest.raises(ValueError, match="no FKS pairs"):
        fks_sector_function(born, finite)
