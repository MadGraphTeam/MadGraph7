"""FKSEikonals: the reduced eikonal factors of the soft limit, as
eikonal_reduced in fks_singular.f.

The reference data (see test_fks_reference.py) store, for each point whose soft
counter-event genps_fks.f generates, eikonal_reduced at that counter-event for
the pairs (m, n) of sbornsoft. The eikonals of the configurations of the Born,
from the metadata of the FKS exporter, must have the same pairs in the same
order and give the same factors on those momenta. On events of the FKS
mappings, the factors at the event are the eikonal factors times xi^2 (1 - y),
and they tend to those at the soft counter-event as xi goes to zero.
"""

import numpy as np
import pytest
from fks_testing import FKS_DATA, FKS_DATA_IDS, evaluate, reference
from fks_testing import reference_born_mapping as born_mapping

import madspace as ms
from madspace.fks import fks_configuration_mapping, fks_eikonals, real_masses

SOFT = 2  # position of the soft counter-event (k = 0) of genps_fks.f
TINY = 1e-6  # tiny of eikonal_reduced


def fortran_points(data):
    """The momenta, momenta of i per unit xi, y, fks_index and pairs of the
    soft counter-events of the reference points, with eikonal_reduced."""
    rows = []
    for conf in data["configs"]:
        for point in conf["points"]:
            if not point["eik_pairs"]:
                continue
            rows.append(
                (
                    point["p_cnt"][SOFT],
                    point["p_i_hat_cnt"][SOFT],
                    point["y_ev"],
                    conf["nfks"] - 1,
                    [(m - 1, n - 1) for m, n in point["eik_pairs"]],
                    point["eik_soft"],
                )
            )
    return rows


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_matches_eikonal_reduced(path):
    data = reference(path)
    configurations = data["fks_configurations"]
    eikonals = fks_eikonals(born_mapping(data), configurations)
    count = len(configurations["configurations"])
    assert eikonals.configuration_count() == count
    rows = fortran_points(data)
    momenta, p_i_hat, y, fks_index, pairs, fortran = zip(*rows)
    (eik,) = evaluate(
        eikonals,
        np.array(momenta),
        np.array(p_i_hat),
        np.array(y),
        fks_index=np.array(fks_index),
    )
    assert eik.shape == (len(rows), eikonals.max_pair_count())
    for k, (c, pairs_c, fortran_c) in enumerate(zip(fks_index, pairs, fortran)):
        # the pairs of sbornsoft, in its order
        assert eikonals.pairs(c) == pairs_c
        size = len(pairs_c)
        # eikonal_reduced takes the direction of i from the soft-collinear
        # counter-event where 1 - y < 1e-6 (tiny)
        rtol = 1e-12 if 1 - y[k] >= TINY else 1e-5
        np.testing.assert_allclose(eik[k, :size], fortran_c, rtol=rtol)
        np.testing.assert_array_equal(eik[k, size:], 0.0)


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_pairs_of_each_configuration(path):
    """The pairs of every configuration, also those without a soft limit (a
    quark as i): every unordered pair of the partners of i, a particle with
    itself where it is massive, and their positions in the Born."""
    data = reference(path)
    configurations = data["fks_configurations"]["configurations"]
    eikonals = fks_eikonals(born_mapping(data), data["fks_configurations"])
    born_masses = born_mapping(data).external_masses()
    for c, configuration in enumerate(configurations):
        i_fks = configuration["i_fks"] - 1
        partners = [j - 1 for j in configuration["fks_j_from_i"][i_fks]]
        masses = real_masses(born_masses, configuration)
        expected = [
            (m, n)
            for k, m in enumerate(partners)
            for n in partners[: k + 1]
            if m != n or masses[m] != 0.0
        ]
        assert eikonals.pairs(c) == expected
        born = [(m - (m > i_fks), n - (n > i_fks)) for m, n in expected]
        assert eikonals.born_pairs(c) == born
    largest = max(len(eikonals.pairs(c)) for c in range(len(configurations)))
    assert eikonals.max_pair_count() == max(largest, 1)


def direct_eikonals(momenta, p_i_hat, y, pairs):
    """(p_m p_n) (1 - y) / ((p_m p_i_hat) (p_n p_i_hat)) without the special
    form for a massless j."""

    def dot(a, b):
        return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)

    return np.stack(
        [
            dot(momenta[:, m], momenta[:, n])
            * (1 - y)
            / (dot(momenta[:, m], p_i_hat) * dot(momenta[:, n], p_i_hat))
            for m, n in pairs
        ],
        axis=1,
    )


LIMIT_CASES = ["pp_ttx_P0_gg_ttx", "epem_uuxg_P0_epem_uuxg", "pp_epemj_P0_ug_uemep"]


@pytest.mark.parametrize("name", LIMIT_CASES)
def test_soft_limit(name):
    """On events of the FKS mappings, the factors are those of the momenta
    (the form for a massless j is exact at the event). As xi goes to zero they
    tend to those at the soft counter-event, linearly in xi: |E(xi) / E(0) - 1|
    / xi is the same for an event at xi_hat = 1e-4, 1e-6 and 1e-8."""
    data = reference(FKS_DATA[FKS_DATA_IDS.index(name)])
    configurations = data["fks_configurations"]
    born = born_mapping(data)
    mapping = fks_configuration_mapping(born, configurations)
    eikonals = fks_eikonals(born, configurations)
    count = len(configurations["configurations"])
    names = list(mapping.counter_event_types().keys())
    kinematics = ms.FKSKinematics(mapping)
    rng = np.random.default_rng(31)
    batch = 400
    r = 0.05 + 0.9 * rng.random((batch, mapping.random_dim()))
    index = rng.integers(0, count, batch).astype(np.int32)
    with_pairs = np.array([len(eikonals.pairs(c)) > 0 for c in index])
    slopes = []
    for r_xi in [1e-2, 1e-3, 1e-4]:
        # the random number of xi_hat, the first of the emission
        r[:, -3] = r_xi
        out = dict(zip(names, evaluate(kinematics, r, fks_index=index)))
        assert ((out["weight"] > 0) & (out["soft_det"] > 0)).all()
        at_event, at_soft = (
            evaluate(
                eikonals,
                out[f"{prefix}momenta"],
                out[f"{prefix}p_i_hat"],
                out["y"],
                fks_index=index,
            )[0]
            for prefix in ["", "soft_"]
        )
        slope = np.zeros(batch)
        for c in range(count):
            part = index == c
            size = len(eikonals.pairs(c))
            if not part.any() or size == 0:
                continue
            direct = direct_eikonals(
                out["momenta"][part],
                out["p_i_hat"][part],
                out["y"][part],
                eikonals.pairs(c),
            )
            np.testing.assert_allclose(at_event[part, :size], direct, rtol=1e-10)
            relative = np.abs(at_event[part, :size] / at_soft[part, :size] - 1)
            slope[part] = relative.max(axis=1) / out["xi"][part]
        slopes.append(slope[with_pairs])
    assert len(slopes[0]) > 0.5 * batch
    # O(xi) corrections to the slopes at xi up to 1e-4, then rounding
    np.testing.assert_allclose(slopes[1], slopes[0], rtol=5e-2, atol=1e-3)
    np.testing.assert_allclose(slopes[2], slopes[1], rtol=1e-3, atol=1e-3)


def test_invalid_configurations():
    pairs = [[], [], [], [], [0, 1]]
    masses = [0.0] * 5
    with pytest.raises(ValueError, match="FKS pair"):
        # j = 2 is not a partner of i = 4
        ms.FKSEikonals([4], [2], [pairs], [masses])
    with pytest.raises(ValueError, match="final-state"):
        ms.FKSEikonals([1], [0], [pairs], [masses])
    with pytest.raises(ValueError, match="number of particles"):
        ms.FKSEikonals([4, 4], [0, 1], [pairs, pairs + [[]]], [masses, masses + [0]])
