"""The resonance-local recoil of an FKS emission off a final-state parton
(FKSRadiation with a resonance), and the resonances of MadGraph
(madspace.fks.fks_resonances with fks_minimal_masses).

The reference data of test_data/fks_resonance come from the MCcntRefactor_Sfun
branch of MadGraph7 (see fks_reference/make_fks_resonance_reference.py), one
file per Born configuration. test_fks_radiation.py and
test_fks_counter_events.py compare their kinematics, with the members of the
resonance of the data. Here fks_resonances, from the topology of the Born
configuration and the cuts of the run_card, must give the resonances that the
branch installs (set_granny, select_fks_recoil). The map must keep the
momentum K of the resonance and every momentum outside it, have xi and y of
the rest frame of K, be the map of the whole final state where all of it is
the resonance and cover the phase space of the real emission once, and an
FKSConfigurationMapping must give every event the resonance of its
configuration and of the permutation of its Born.
"""

import math

import numpy as np
import pytest
import torch
from fks_testing import (
    FKS_RESONANCE_DATA,
    data_id,
    forest_topology,
    reference,
    resonance_members,
)
from madspace_testing import default_context, generator

import madspace as ms
from madspace.fks import fks_minimal_masses, fks_resonances

SQRT_S = 500.0
M_Z, W_Z = 91.188, 2.4952
M_B = 4.7


def to_torch(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64).to(
        torch.get_default_device()
    )


def to_torch_int(x):
    return torch.as_tensor(x, dtype=torch.int32).to(torch.get_default_device())


def to_numpy(x):
    return torch.as_tensor(x).cpu().numpy()


def minkowski(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


@pytest.mark.parametrize(
    "path", FKS_RESONANCE_DATA, ids=map(data_id, FKS_RESONANCE_DATA)
)
def test_resonances_of_set_granny(path):
    """fks_resonances on the Born configuration of the data, with the cuts of
    its run_card: the members of the resonance of every FKS configuration
    are those of select_fks_recoil, the decay products of the grandmother
    where set_granny finds it a resonance."""
    data = reference(path)
    configurations = data["fks_configurations"]
    conf = data["configs"][0]
    born_masses = list(np.delete(np.array(conf["masses"]), conf["i_fks"] - 1))
    minimal = fks_minimal_masses(configurations, born_masses, **data["cuts"])
    resonances = fks_resonances(forest_topology(data), configurations, minimal)
    assert len(resonances) == len(data["configs"])
    for conf, resonance in zip(data["configs"], resonances):
        assert resonance == [resonance_members(conf)], conf["nfks"]
        granny = conf["granny"]
        chain = [k for k, f in enumerate(granny["granny_chain_real_final"]) if f]
        assert resonance[0] == (chain if granny["granny_is_res"] else [])
        # the momentum K of the resonance, the members of the event and of the
        # Born (where the event is generated)
        points = [p for p in conf["points"] if p["jac"] > 0]
        k = np.array([p["resonance_momentum"] for p in points])
        if not resonance[0]:
            assert (k == 0).all()
            continue
        members = resonance[0]
        p_ev = np.array([p["p_ev"] for p in points])
        sqrt_s = np.sqrt(np.array([p["shat_ev"] for p in points]))[:, None]
        np.testing.assert_allclose(
            p_ev[:, members].sum(1) / sqrt_s, k / sqrt_s, rtol=0, atol=1e-13
        )
        i_fks = conf["i_fks"] - 1
        born_members = [m if m < i_fks else m - 1 for m in members if m != i_fks]
        p_born = np.array([p["p_born"] for p in points])
        np.testing.assert_allclose(
            p_born[:, born_members].sum(1) / sqrt_s, k / sqrt_s, rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(
            [p["resonance_mass2"] for p in points], minkowski(k, k), rtol=1e-12
        )


def test_reference_covers_the_resonances():
    """Resonances of a massless and a massive emitter, next to a t-channel too,
    and a Z boson that cannot be on shell with the jet cuts."""
    kinds = set()
    for path in FKS_RESONANCE_DATA:
        data = reference(path)
        t_channel = any(d1 in (1, 2) for _, d1, _, _, _ in data["born_forest"])
        # the resonances without cuts
        configurations = data["fks_configurations"]
        conf = data["configs"][0]
        masses = list(np.delete(np.array(conf["masses"]), conf["i_fks"] - 1))
        uncut = fks_resonances(
            forest_topology(data),
            configurations,
            fks_minimal_masses(configurations, masses),
        )
        for conf, resonance in zip(data["configs"], uncut):
            j = conf["j_fks"] - 1
            if resonance_members(conf):
                kinds.add("massive" if conf["masses"][j] > 0 else "massless")
                if t_channel:
                    kinds.add("t-channel")
            elif resonance[0]:
                kinds.add("off shell by the cuts")
    assert kinds == {"massless", "massive", "t-channel", "off shell by the cuts"}


def test_minimal_masses_of_set_tau_min():
    """Jets, photons and charged leptons with their pair mass cuts, which take
    the first lepton of opposite charge, of the same flavor or with mll; and
    the position of the i_fks of the last configuration, which keeps the value
    of an earlier configuration with another i_fks, else 0."""
    leptons = {
        "nincoming": 2,
        "configurations": [
            {
                "i_fks": 9,
                "j_fks": 7,
                "real_pdgs": [11, -11, -13, 11, -11, 22, 2, -2, 21],
            }
        ],
    }
    masses = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    cuts = dict(ptj=10.0, ptgmin=20.0, ptl=5.0, mll=12.0, mll_sf=30.0)
    # mu+ and e- with mll, e+ with mll_sf from the e- before it
    expected = [6.0, 6.0, 15.0, 20.0, 10.0, 10.0]
    assert fks_minimal_masses(leptons, masses, **cuts) == [expected]

    # u u~ > a g: i is not the last particle
    real = [2, -2, 22, 21, 21]
    same_i = {
        "nincoming": 2,
        "configurations": [
            {"i_fks": 4, "j_fks": 1, "real_pdgs": real},
            {"i_fks": 4, "j_fks": 2, "real_pdgs": real},
        ],
    }
    born_masses = [0.0] * 4
    assert (
        fks_minimal_masses(same_i, born_masses, ptj=10.0, ptgmin=20.0)
        == [[20.0, 0.0]] * 2
    )
    other_i = {
        "nincoming": 2,
        "configurations": [{"i_fks": 5, "j_fks": 4, "real_pdgs": real}]
        + same_i["configurations"],
    }
    assert (
        fks_minimal_masses(other_i, born_masses, ptj=10.0, ptgmin=20.0)
        == [[20.0, 10.0]] * 3
    )
    # massive particles with a cut by their PDG code (|pdg|): the transverse
    # mass, and with a pair mass cut half of it where it is larger
    tops = {
        "nincoming": 2,
        "configurations": [{"i_fks": 5, "j_fks": 3, "real_pdgs": [21, 21, 6, -6, 21]}],
    }
    top_masses = [0.0, 0.0, 173.0, 173.0]
    pt_cut = dict(pdg=6, ptmin=100.0, mxxmin=0.0, part_antipart=False)
    assert fks_minimal_masses(tops, top_masses, pdg_cuts=[pt_cut]) == [
        [math.hypot(100.0, 173.0)] * 2
    ]
    pair_cut = dict(pt_cut, mxxmin=400.0, part_antipart=True)
    assert fks_minimal_masses(tops, top_masses, pdg_cuts=[pair_cut]) == [[200.0] * 2]


def z_topology(masses):
    """e+ e- > a* > e+ e- (Z > q q~): the Z decays into the last two outgoing
    particles, after a t-like chain of massless propagators."""
    return ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            list(masses),
            [
                ms.Propagator(mass=0.0, width=0.0),
                ms.Propagator(mass=0.0, width=0.0),
                ms.Propagator(mass=M_Z, width=W_Z, pdg_id=23),
            ],
            [
                ["i0", "i1", "p0"],
                ["p0", "o0", "p1"],
                ["p1", "o1", "p2"],
                ["p2", "o2", "o3"],
            ],
        )
    )


def born_points(masses, batch, seed, permutations=None):
    """Born points of the Z topology at SQRT_S, and their weights."""
    mapping = ms.PhaseSpaceMapping(
        z_topology(masses),
        SQRT_S,
        leptonic=True,
        permutations=permutations or [],
    )
    r = torch.rand((batch, mapping.random_dim()), generator=generator(seed))
    out = mapping.map_forward([r.to(torch.get_default_device())])
    return to_numpy(out[0]), to_numpy(out[-1])


def forward(mapping, r, born):
    inputs = [to_torch(r[:, 0]), to_torch(r[:, 1]), to_torch(r[:, 2]), to_torch(born)]
    momenta, det = mapping.map_forward(inputs)
    return to_numpy(momenta), to_numpy(det)


def inverse(mapping, momenta):
    r_xi, r_y, r_phi, born, det = mapping.map_inverse([to_torch(momenta)])
    r = np.stack([to_numpy(r_xi), to_numpy(r_y), to_numpy(r_phi)], 1)
    return r, to_numpy(born), to_numpy(det)


def rest_frame(p, k):
    """p in the rest frame of k (boostx)."""
    m = np.sqrt(minkowski(k, k))
    e = minkowski(k, p) / m
    c = (p[..., 0] + e) / (k[..., 0] + m)
    return np.concatenate([e[..., None], p[..., 1:] - c[..., None] * k[..., 1:]], -1)


# (Born masses, j_fks): a massless and a massive emitter of the Z, i last
EMITTERS = [([0.0, 0.0, 0.0, 0.0], 4), ([0.0, 0.0, M_B, M_B], 5)]
EMITTER_IDS = ["massless", "massive"]


@pytest.mark.parametrize("masses,j_fks", EMITTERS, ids=EMITTER_IDS)
def test_resonance_local_recoil(masses, j_fks):
    """The Z and every other momentum stay, xi and y are those of the rest
    frame of the Z, and the inverse gives back the Born and the random
    numbers."""
    batch = 2000
    born, _ = born_points(masses, batch, seed=1)
    r = 0.02 + 0.96 * np.random.default_rng(2).random((batch, 3))
    members = [4, 5, 6]
    mapping = ms.FKSRadiation(
        6, 6, j_fks, leptonic=True, emitter_mass=masses[j_fks - 2], resonance=members
    )
    assert mapping.resonance() == members
    momenta, det = forward(mapping, r, born)
    assert (det > 0).all()
    np.testing.assert_array_equal(momenta[:, :4], born[:, :4])
    k = born[:, 4] + born[:, 5]
    np.testing.assert_allclose(
        momenta[:, 4:].sum(1) / SQRT_S, k / SQRT_S, rtol=0, atol=1e-13
    )
    real_masses = list(masses) + [0.0]
    m2 = minkowski(momenta, momenta) - np.array([0, 0] + real_masses) ** 2
    np.testing.assert_allclose(m2 / SQRT_S**2, 0.0, atol=1e-12)

    # xi and y in the rest frame of the Z, from the kinematics of an FKSMapping
    born_mapping = ms.PhaseSpaceMapping(z_topology(masses), SQRT_S, leptonic=True)
    fks = ms.FKSMapping(born_mapping, 6, j_fks, resonance=members)
    rng = np.random.default_rng(3)
    r_all = rng.random((batch, fks.random_dim()))
    names = list(fks.counter_event_types().keys())
    runtime = ms.FunctionRuntime(ms.FKSKinematics(fks).function(), default_context())
    out = dict(zip(names, [to_numpy(t) for t in runtime.call([to_torch(r_all)])]))
    p = out["momenta"]
    k = p[:, 4:].sum(1)
    m_k = np.sqrt(minkowski(k, k))
    p_i, p_j = rest_frame(p[:, 6], k), rest_frame(p[:, j_fks], k)
    np.testing.assert_allclose(out["xi"], 2 * p_i[:, 0] / m_k, rtol=1e-12)
    cos = (p_i[:, 1:] * p_j[:, 1:]).sum(1) / np.linalg.norm(p_i[:, 1:], axis=1)
    cos /= np.linalg.norm(p_j[:, 1:], axis=1)
    np.testing.assert_allclose(out["y"], cos, rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        out["xi"][:, None] * out["p_i_hat"], p[:, 6], rtol=0, atol=1e-12 * SQRT_S
    )
    # the soft counter-event is the Born, with i soft along p_i_hat
    soft = out["soft_momenta"]
    np.testing.assert_allclose(
        soft[:, :6] / SQRT_S, out["born_momenta"] / SQRT_S, rtol=0, atol=1e-13
    )

    r_inv, born_inv, det_inv = inverse(mapping, momenta)
    np.testing.assert_allclose(r_inv[:, :2], r[:, :2], rtol=0, atol=1e-9)
    np.testing.assert_allclose(born_inv / SQRT_S, born / SQRT_S, rtol=0, atol=1e-12)
    np.testing.assert_allclose(det * det_inv, 1.0, rtol=1e-9)


def test_recoil_of_a_decay_chain():
    """With the Z and the particle before it in the resonance, the Z recoils
    as a whole: its mass and its decay products stay as they are in its rest
    frame."""
    batch = 1000
    masses = [0.0, 0.0, 0.0, 0.0]
    born, _ = born_points(masses, batch, seed=4)
    r = np.random.default_rng(5).random((batch, 3))
    mapping = ms.FKSRadiation(6, 6, 3, leptonic=True, resonance=[3, 4, 5, 6])
    momenta, _ = forward(mapping, r, born)
    np.testing.assert_array_equal(momenta[:, :3], born[:, :3])
    z_born, z_real = born[:, 4] + born[:, 5], momenta[:, 4] + momenta[:, 5]
    np.testing.assert_allclose(
        minkowski(z_real, z_real), minkowski(z_born, z_born), rtol=1e-11
    )
    np.testing.assert_allclose(
        minkowski(momenta[:, 4], momenta[:, 5]) / SQRT_S**2,
        minkowski(born[:, 4], born[:, 5]) / SQRT_S**2,
        rtol=0,
        atol=1e-13,
    )


@pytest.mark.parametrize("masses,j_fks", EMITTERS, ids=EMITTER_IDS)
def test_resonance_of_the_whole_final_state(masses, j_fks):
    """A resonance of the whole final state has the frame of the Born."""
    batch = 1000
    born, _ = born_points(masses, batch, seed=6)
    r = np.random.default_rng(7).random((batch, 3))
    m_j = masses[j_fks - 2]
    plain = ms.FKSRadiation(6, 6, j_fks, leptonic=True, emitter_mass=m_j)
    whole = ms.FKSRadiation(
        6, 6, j_fks, leptonic=True, emitter_mass=m_j, resonance=[2, 3, 4, 5, 6]
    )
    momenta, det = forward(whole, r, born)
    momenta_ref, det_ref = forward(plain, r, born)
    np.testing.assert_allclose(
        momenta / SQRT_S, momenta_ref / SQRT_S, rtol=0, atol=1e-13
    )
    np.testing.assert_allclose(det, det_ref, rtol=1e-12)


def assert_same_mean(weight, weight_ref):
    mean, err = weight.mean(), weight.std() / math.sqrt(len(weight))
    mean_ref = weight_ref.mean()
    err_ref = weight_ref.std() / math.sqrt(len(weight_ref))
    assert abs(mean - mean_ref) < 5 * math.hypot(err, err_ref)


@pytest.mark.parametrize("masses,j_fks", EMITTERS, ids=EMITTER_IDS)
def test_volume(masses, j_fks):
    """A Born of the Z topology times the emission in the rest frame of the Z
    covers the phase space with one more massless particle once."""
    batch = 400000
    born, det_born = born_points(masses, batch, seed=8)
    r = np.random.default_rng(9).random((batch, 3))
    mapping = ms.FKSRadiation(
        6,
        6,
        j_fks,
        leptonic=True,
        emitter_mass=masses[j_fks - 2],
        resonance=[4, 5, 6],
    )
    _, det = forward(mapping, r, born)
    rambo = ms.PhaseSpaceMapping(
        [0.0, 0.0] + list(masses) + [0.0], SQRT_S, mode="rambo", leptonic=True
    )
    r_real = torch.rand((batch, rambo.random_dim()), generator=generator(10))
    det_real = to_numpy(rambo.map_forward([r_real.to(torch.get_default_device())])[-1])
    assert_same_mean(det_born * det, det_real)


def test_events_take_the_resonance_of_their_permutation():
    """An FKSConfigurationMapping with two permutations of the Born, where the Z
    decays into the last two or into the first two outgoing particles: each
    event is that of the FKSMapping of its configuration with the Born mapping
    of its permutation alone and its resonance."""
    masses = [0.0, 0.0, 0.0, 0.0]
    permutations = [[0, 1, 2, 3, 4, 5], [0, 1, 4, 5, 2, 3]]
    topology = z_topology(masses)
    configurations = {
        "nincoming": 2,
        "configurations": [{"i_fks": 7, "j_fks": j} for j in (3, 4, 5, 6)],
    }
    minimal = [[0.0] * 4] * 4
    resonances = fks_resonances(topology, configurations, minimal, permutations)
    assert resonances == [
        [[], [2, 3, 6]],
        [[], [2, 3, 6]],
        [[4, 5, 6], []],
        [[4, 5, 6], []],
    ]
    born = ms.PhaseSpaceMapping(
        topology, SQRT_S, leptonic=True, permutations=permutations
    )
    pairs = [(6, j) for j in (2, 3, 4, 5)]
    mapping = ms.FKSConfigurationMapping(born, pairs, resonances=resonances)
    assert mapping.resonances() == resonances
    batch = 2000
    rng = np.random.default_rng(11)
    r = 0.02 + 0.96 * rng.random((batch, mapping.random_dim()))
    index = rng.integers(0, 4, batch).astype(np.int32)
    perm = rng.integers(0, 2, batch).astype(np.int32)
    momenta, x1, x2, det = mapping.map_forward(
        [to_torch(r)], [to_torch_int(perm), to_torch_int(index)]
    )
    outputs = [to_numpy(t) for t in (momenta, x1, x2, det)]
    for p, permutation in enumerate(permutations):
        single = ms.PhaseSpaceMapping(
            topology, SQRT_S, leptonic=True, permutations=[permutation]
        )
        for c, (i_fks, j_fks) in enumerate(pairs):
            events = np.flatnonzero((index == c) & (perm == p))
            fks = ms.FKSMapping(single, i_fks, j_fks, resonance=resonances[c][p])
            ref = [to_numpy(t) for t in fks.map_forward([to_torch(r)])]
            for ours, theirs in zip(outputs, ref):
                scale = np.abs(theirs).max()
                np.testing.assert_allclose(
                    ours[events], theirs[events], rtol=1e-13, atol=1e-13 * scale
                )
    r_inv, det_inv = mapping.map_inverse(
        [momenta, x1, x2], [to_torch_int(perm), to_torch_int(index)]
    )
    np.testing.assert_allclose(to_numpy(r_inv), r, rtol=0, atol=1e-9)
    np.testing.assert_allclose(outputs[3] * to_numpy(det_inv), 1.0, rtol=1e-9)


def test_lepton_beams():
    """With lepton beams the members of the resonance of a final-state emitter
    keep their momentum K and the other particles their momenta, beside an
    incoming emitter without the projection of the event, and the inverse
    gives back the random numbers."""
    born = ms.PhaseSpaceMapping(
        z_topology([0.0] * 4), SQRT_S, leptonic=True, sqrt_s_hat_condition=True
    )
    luminosity = ms.LeptonLuminosity(SQRT_S**2, 0.975, s_hat_min_condition=True)
    pairs = [(6, 0), (6, 2), (6, 4), (6, 5)]
    resonances = [[[]], [[]], [[4, 5, 6]], [[4, 5, 6]]]
    mapping = ms.FKSConfigurationMapping(
        born, pairs, luminosity=luminosity, resonances=resonances
    )
    batch = 2000
    rng = np.random.default_rng(12)
    r = 0.02 + 0.96 * rng.random((batch, mapping.random_dim()))
    index = rng.integers(0, 4, batch).astype(np.int32)
    names = list(mapping.counter_event_types().keys())
    runtime = ms.FunctionRuntime(
        ms.FKSKinematics(mapping).function(), default_context()
    )
    out = dict(
        zip(
            names,
            [to_numpy(t) for t in runtime.call([to_torch(r), to_torch_int(index)])],
        )
    )
    p, born_p = out["momenta"], out["born_momenta"]
    ok = out["weight"] > 0
    resonant = ok & (index >= 2)
    assert resonant.any()
    np.testing.assert_array_equal(p[resonant, :4], born_p[resonant, :4])
    k_real, k_born = p[resonant, 4:].sum(1), born_p[resonant, 4:].sum(1)
    np.testing.assert_allclose(k_real / SQRT_S, k_born / SQRT_S, rtol=0, atol=1e-13)
    inputs = [out[name] for name in ("momenta", "x1", "x2")]
    inputs += [out["one_minus_x1"], out["one_minus_x2"]]
    r_inv, det_inv = mapping.map_inverse(
        [to_torch(v) for v in inputs], [to_torch_int(index)]
    )
    np.testing.assert_allclose(to_numpy(r_inv)[ok], r[ok], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        out["weight"][ok] * to_numpy(det_inv)[ok], 1.0, rtol=1e-9
    )


@pytest.mark.parametrize(
    "j_fks,resonance",
    [
        (4, [4, 6]),
        (4, [4, 5]),
        (4, [3, 5, 6]),
        (4, [0, 4, 5, 6]),
        (4, [4, 4, 5, 6]),
        (4, [4, 5, 6, 7]),
        (0, [4, 5, 6]),
    ],
    ids=[
        "no recoil",
        "without i",
        "without j",
        "incoming",
        "twice",
        "out",
        "initial j",
    ],
)
def test_invalid_resonances(j_fks, resonance):
    with pytest.raises(ValueError):
        ms.FKSRadiation(6, 6, j_fks, leptonic=j_fks >= 2, resonance=resonance)


def test_invalid_resonances_of_the_mappings():
    masses = [0.0, 0.0, 0.0, 0.0]
    permutations = [[0, 1, 2, 3, 4, 5], [0, 1, 4, 5, 2, 3]]
    born = ms.PhaseSpaceMapping(
        z_topology(masses), SQRT_S, leptonic=True, permutations=permutations
    )
    with pytest.raises(ValueError):
        ms.FKSMapping(born, 6, 4, resonance=[4, 5, 6])
    pairs = [(6, 4), (6, 5)]
    for resonances in [
        [[[4, 5, 6], []]],
        [[[4, 5, 6]], [[4, 5, 6]]],
        [[[4, 5, 6], []], [[4, 6], []]],
    ]:
        with pytest.raises(ValueError):
            ms.FKSConfigurationMapping(born, pairs, resonances=resonances)
