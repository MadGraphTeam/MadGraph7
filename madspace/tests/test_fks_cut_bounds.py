"""Lower bounds on the partonic energy of the FKS phase space from the cuts.

fks_sqrt_s_hat_min turns the cuts on the Born-level objects into a lower bound
on sqrt(s_hat) of the real emission. madspace.subtraction.fks.fks_mappings puts
it where it belongs: a floor of the Born mapping for a final-state emitter
(s_hat is that of the Born), s_hat_min for an incoming one. fks_configuration_mapping does the
same in one pass, the floor read per event by the Born mapping. The bound must
hold for every real emission that passes the cuts, whatever i does, or it would
remove part of the cross section. Here the cuts are applied as an NLO integrand would apply them:
to jets clustered from the final-state partons, and to the other particles.
"""

import json
import math
import os

import numpy as np
import pytest
import torch
from madspace_testing import default_context, to_numpy

import madspace as ms
from madspace.subtraction.fks import fks_configuration_mapping, fks_mappings

O = ms.Observable
SQRT_S = 2000.0
BATCH = 50000
DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data", "fks")
QCD = {1, 2, 3, 4, 5, -1, -2, -3, -4, -5, 21}


def reference(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def cut_items(pids, pt_jet, pt_lepton, m_ll, pt_top=0.0, sqrt_s_min=0.0):
    items = []
    jets = [[pid for pid in O.jet_pids if pid in pids[2:]]]
    if pt_jet > 0 and jets[0]:
        items.append(ms.CutItem(O(pids, O.obs_pt, jets), min=pt_jet))
    if pt_lepton > 0:
        items.append(ms.CutItem(O(pids, O.obs_pt, [[11, -11]]), min=pt_lepton))
    if m_ll > 0:
        items.append(ms.CutItem(O(pids, O.obs_pair_mass, [[11, -11]]), min=m_ll))
    if pt_top > 0:
        items.append(ms.CutItem(O(pids, O.obs_pt, [[6, -6]]), min=pt_top))
    if sqrt_s_min > 0:
        items.append(ms.CutItem(O(pids, O.obs_sqrt_s, []), min=sqrt_s_min))
    return ms.Cuts(items)


def test_bound():
    born = [2, 21, 2, 11, -11]
    massless = [0.0, 0.0, 0.0]
    # transverse masses 30 + 2 * 20
    assert ms.fks_sqrt_s_hat_min(cut_items(born, 30.0, 20.0, 0.0), massless) == 70.0
    # the pair mass cut, with the transverse mass of the jet
    assert ms.fks_sqrt_s_hat_min(cut_items(born, 30.0, 20.0, 100.0), massless) == 130.0
    # the sqrt(s_hat) cut
    bound = ms.fks_sqrt_s_hat_min(
        cut_items(born, 30.0, 20.0, 100.0, sqrt_s_min=300.0), massless
    )
    assert bound == 300.0
    # massive particles: transverse masses, and their masses with a pair cut on
    # others
    top = 173.0
    ttx = [21, 21, 6, -6]
    bound = ms.fks_sqrt_s_hat_min(
        cut_items(ttx, 0.0, 0.0, 0.0, pt_top=50.0), [top, top]
    )
    assert bound == pytest.approx(2 * math.hypot(top, 50.0), rel=1e-15)
    assert ms.fks_sqrt_s_hat_min(ms.Cuts(4), [top, top]) == 2 * top
    # all pairs of four leptons above 50: sqrt(6) * 50
    four = [11, -11, 11, -11, 13, -13]
    cuts = ms.Cuts(
        [ms.CutItem(O(four, O.obs_pair_mass, [[11, -11, 13, -13]]), min=50.0)]
    )
    assert ms.fks_sqrt_s_hat_min(cuts, [0.0] * 4) == pytest.approx(math.sqrt(6) * 50.0)


# --- the cuts as an NLO integrand applies them -------------------------------


def pt(p):
    return np.hypot(p[..., 1], p[..., 2])


def rapidity_phi(p):
    pt_ = np.maximum(pt(p), 1e-300)
    eta = np.arcsinh(p[..., 3] / pt_)
    return eta, np.arctan2(p[..., 2], p[..., 1])


def jets(partons, radius=0.4):
    """Final-state partons (batch, n, 4), the pairs closer than radius merged
    (enough for up to two partons, as in a real emission of one jet)."""
    if partons.shape[1] < 2:
        return [partons[:, k] for k in range(partons.shape[1])]
    assert partons.shape[1] == 2
    eta, phi = rapidity_phi(partons)
    dphi = np.abs(phi[:, 0] - phi[:, 1])
    dphi = np.minimum(dphi, 2 * np.pi - dphi)
    close = np.hypot(eta[:, 0] - eta[:, 1], dphi) < radius
    merged = partons[:, 0] + partons[:, 1]
    first = np.where(close[:, None], merged, partons[:, 0])
    second = np.where(close[:, None], 0.0, partons[:, 1])
    return [first, second]


def passes(momenta, real_pdgs, n_jets, pt_jet, pt_lepton, m_ll, pt_top=0.0):
    """IR-safe cuts on a real emission: at least n_jets jets above pt_jet,
    leptons above pt_lepton with m_ll above its cut, tops above pt_top."""
    final = np.arange(2, len(real_pdgs))
    partons = [k for k in final if real_pdgs[k] in QCD]
    ok = np.ones(len(momenta), dtype=bool)
    if n_jets > 0:
        passing = sum(pt(j) >= pt_jet for j in jets(momenta[:, partons]))
        ok &= passing >= n_jets
    leptons = [k for k in final if abs(real_pdgs[k]) == 11]
    for k in leptons:
        ok &= pt(momenta[:, k]) >= pt_lepton
    if m_ll > 0 and len(leptons) == 2:
        q = momenta[:, leptons[0]] + momenta[:, leptons[1]]
        ok &= q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(-1) >= m_ll**2
    for k in final:
        if abs(real_pdgs[k]) == 6:
            ok &= pt(momenta[:, k]) >= pt_top
    return ok


def sqrt_s_hat(momenta):
    q = momenta[:, 0] + momenta[:, 1]
    return np.sqrt(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(-1))


CASES = {
    # (reference, Born masses, cut parameters)
    "pp-epemj-ug": ("pp_epemj_P0_ug_uemep.json", [0.0] * 5, (30.0, 20.0, 60.0, 0.0)),
    "pp-epemj-uux": ("pp_epemj_P0_uux_emepg.json", [0.0] * 5, (30.0, 20.0, 60.0, 0.0)),
    "pp-ttx-gg": (
        "pp_ttx_P0_gg_ttx.json",
        [0.0, 0.0, 173.0, 173.0],
        (0.0, 0.0, 0.0, 80.0),
    ),
}


def born_factory(masses):
    def born(cuts):
        return ms.PhaseSpaceMapping(
            masses, SQRT_S, cuts=cuts, mode=ms.PhaseSpaceMapping.TChannelMode.rambo
        )

    return born


@pytest.mark.parametrize("case", CASES)
def test_no_passing_event_below_the_bound(case):
    """The real emissions of every FKS configuration, sampled without the
    bound: those that pass the cuts all lie above it."""
    name, masses, (pt_jet, pt_lepton, m_ll, pt_top) = CASES[case]
    configurations = reference(name)["fks_configurations"]
    born_pdgs = configurations["born_pdgs"]
    n_jets = sum(pid in QCD for pid in born_pdgs[2:])
    cuts = cut_items(born_pdgs, pt_jet, pt_lepton, m_ll, pt_top=pt_top)
    bound = ms.fks_sqrt_s_hat_min(cuts, masses[2:])
    assert bound > sum(masses)
    mappings = fks_mappings(born_factory(masses)(None), configurations)
    rng = np.random.default_rng(21)
    closest = np.inf
    for mapping, conf in zip(mappings, configurations["configurations"]):
        r = rng.random((BATCH, mapping.random_dim()))
        momenta, _, _, det = (
            to_numpy(t) for t in mapping.map_forward([torch.as_tensor(r)])
        )
        ok = passes(momenta, conf["real_pdgs"], n_jets, pt_jet, pt_lepton, m_ll, pt_top)
        ok &= det > 0
        assert ok.any()
        energy = sqrt_s_hat(momenta)[ok]
        assert (energy >= bound * (1 - 1e-12)).all()
        closest = min(closest, energy.min() / bound)
    # and it is not a vacuous bound: passing events come close to it (for
    # e+ e- j the smallest energy that passes is about 8% above it, since it
    # takes the leptons apart from the pair mass cut)
    assert closest < 1.25


@pytest.mark.parametrize("case", CASES)
def test_bounded_configurations(case):
    """With the cuts, final-state emitters get a Born with a sqrt(s_hat) floor
    and incoming ones s_hat_min: their events lie above the bound, and the
    cross section with the cuts is the same as without the bound."""
    name, masses, (pt_jet, pt_lepton, m_ll, pt_top) = CASES[case]
    configurations = reference(name)["fks_configurations"]
    born_pdgs = configurations["born_pdgs"]
    n_jets = sum(pid in QCD for pid in born_pdgs[2:])
    cuts = cut_items(born_pdgs, pt_jet, pt_lepton, m_ll, pt_top=pt_top)
    bound = ms.fks_sqrt_s_hat_min(cuts, masses[2:])
    factory = born_factory(masses)
    bounded = fks_mappings(factory, configurations, cuts=cuts)
    free = fks_mappings(factory(None), configurations)
    rng = np.random.default_rng(22)
    for mapping, plain, conf in zip(bounded, free, configurations["configurations"]):
        if conf["j_fks"] > 2:
            assert mapping.born().cuts().only_sqrt_s_min()
        else:
            assert mapping.born().cuts().empty()
        means = []
        for m in (mapping, plain):
            r = rng.random((BATCH, m.random_dim()))
            momenta, _, _, det = (
                to_numpy(t) for t in m.map_forward([torch.as_tensor(r)])
            )
            ok = passes(
                momenta, conf["real_pdgs"], n_jets, pt_jet, pt_lepton, m_ll, pt_top
            )
            if m is mapping:
                assert (sqrt_s_hat(momenta)[det > 0] >= bound * (1 - 1e-12)).all()
            w = det * ok
            means.append((w.mean(), w.std() / math.sqrt(len(w))))
        (mean, err), (mean_free, err_free) = means
        assert abs(mean - mean_free) < 5 * math.hypot(err, err_free)


@pytest.mark.parametrize("case", CASES)
def test_configurations_in_one_pass(case):
    """With the cuts, the FKSConfigurationMapping gives every event, forward,
    inverse and with its counter-events, those of the FKSMapping of its
    configuration from fks_mappings: the floor of the Born of a final-state
    emitter as a condition of the Born mapping, read per event."""
    name, masses, (pt_jet, pt_lepton, m_ll, pt_top) = CASES[case]
    configurations = reference(name)["fks_configurations"]
    cuts = cut_items(configurations["born_pdgs"], pt_jet, pt_lepton, m_ll, pt_top)
    born = ms.PhaseSpaceMapping(
        masses,
        SQRT_S,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
        sqrt_s_hat_min_condition=True,
    )
    mapping = fks_configuration_mapping(born, configurations, cuts=cuts)
    assert mapping.sqrt_s_hat_floor() == ms.fks_sqrt_s_hat_min(cuts, masses[2:])
    references = fks_mappings(born_factory(masses), configurations, cuts=cuts)
    count = len(references)
    batch = 400 * count
    rng = np.random.default_rng(23)
    r = 0.02 + 0.96 * rng.random((batch, mapping.random_dim()))
    index = rng.integers(0, count, batch).astype(np.int32)
    index_torch = torch.as_tensor(index, dtype=torch.int32).to(
        torch.get_default_device()
    )
    outputs = mapping.map_forward([torch.as_tensor(r)], [index_torch])
    momenta, x1, x2, det = (to_numpy(t) for t in outputs)
    names = list(mapping.counter_event_types().keys())
    runtime = ms.FunctionRuntime(
        ms.FKSKinematics(mapping).function(), default_context()
    )
    kinematics = [to_numpy(t) for t in runtime.call([torch.as_tensor(r), index_torch])]
    # the references on the whole batch, so that every event is computed in the
    # same SIMD lane, compared at the events of their configuration
    for k, reference_mapping in enumerate(references):
        events = np.flatnonzero(index == k)
        ref = reference_mapping.map_forward([torch.as_tensor(r)])
        for ours, theirs in zip([momenta, x1, x2, det], ref):
            np.testing.assert_array_equal(ours[events], to_numpy(theirs)[events])
        single = ms.FunctionRuntime(
            ms.FKSKinematics(reference_mapping).function(), default_context()
        )
        ref = single.call([torch.as_tensor(r)])
        for name, ours, theirs in zip(names, kinematics, ref):
            np.testing.assert_array_equal(
                ours[events], to_numpy(theirs)[events], err_msg=name
            )
    bound = mapping.sqrt_s_hat_floor()
    assert (sqrt_s_hat(momenta)[det > 0] >= bound * (1 - 1e-12)).all()
    # the inverse is that of the FKSMapping of each event too, and a round trip
    # to their conditioning (3.5e-9 in r_xi for an incoming emitter near the
    # edge of its range)
    r_inv, det_inv = (
        to_numpy(t) for t in mapping.map_inverse(list(outputs[:3]), [index_torch])
    )
    for k, reference_mapping in enumerate(references):
        events = np.flatnonzero(index == k)
        ref = reference_mapping.map_forward([torch.as_tensor(r)])
        ref_inv = reference_mapping.map_inverse(list(ref[:3]))
        for ours, theirs in zip([r_inv, det_inv], ref_inv):
            np.testing.assert_array_equal(ours[events], to_numpy(theirs)[events])
    ok = det > 0
    np.testing.assert_allclose(r_inv[ok], r[ok], rtol=0, atol=1e-8)
    # the floor needs the condition
    with pytest.raises(ValueError):
        fks_configuration_mapping(born_factory(masses)(None), configurations, cuts=cuts)


def test_born_floor_only_for_final_emitters():
    pids = [2, -2, 11, -11]
    born = born_factory([0.0] * 4)
    floor = ms.Cuts([ms.CutItem(O(pids, O.obs_sqrt_s, []), min=100.0)])
    ms.FKSMapping(born(floor), 4, 2)
    with pytest.raises(ValueError):
        ms.FKSMapping(born(floor), 4, 0)
    with pytest.raises(ValueError):
        ms.FKSMapping(born(cut_items(pids, 0.0, 20.0, 0.0)), 4, 2)
    # cuts need a function that builds the Born for them
    configurations = reference("pp_epem_P0_uux_emep.json")["fks_configurations"]
    with pytest.raises(ValueError):
        fks_mappings(born(None), configurations, cuts=cut_items(pids, 0.0, 20.0, 0.0))
