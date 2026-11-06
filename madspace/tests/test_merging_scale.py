"""CKKW-L merging scales (ktdurham, ptlund) against a statement-by-statement
port of the merging cuts and RHOPYTHIA of Template/LO/SubProcesses/cuts.f."""

import math

import numpy as np
import phasespace_helpers
import pytest
import torch

import madspace as ms

CM_ENERGY = 13000.0
SEED = 20261102
N_EVENTS = 400


def sample(masses, leptonic=False, n=N_EVENTS, seed=SEED):
    mapping = ms.PhaseSpaceMapping(masses, CM_ENERGY, leptonic=leptonic)
    p_ext, det = phasespace_helpers.sample(mapping, seed=seed, n=n)
    keep = torch.isfinite(p_ext).all(dim=2).all(dim=1) & (det > 0)
    return p_ext[keep]


# --------------------------------------------------------------------------
# reference: Template/LO/SubProcesses/cuts.f
# --------------------------------------------------------------------------


def ref_kt_durham(p, merging, d_parameter, hadronic):
    """DURHAM_KT CUT block: the smallest of the pT (hadronic) and the pair kt."""
    jets = [p[i] for i in range(2, len(p)) if merging[i]]
    ptmin = math.inf
    for i in range(len(jets)):
        pi = jets[i]
        pt1 = math.hypot(pi[1], pi[2])
        if hadronic:
            ptmin = min(ptmin, pt1)
        for j in range(i + 1, len(jets)):
            pj = jets[j]
            pabs1 = math.sqrt(pi[1] ** 2 + pi[2] ** 2 + pi[3] ** 2)
            pabs2 = math.sqrt(pj[1] ** 2 + pj[2] ** 2 + pj[3] ** 2)
            if pabs1 * pabs2 != 0:
                costh = (pi[1] * pj[1] + pi[2] * pj[2] + pi[3] * pj[3]) / (
                    pabs1 * pabs2
                )
            else:
                costh = 1.0
            pt2 = math.hypot(pj[1], pj[2])
            if not hadronic:
                ptnow = math.sqrt(2 * min(pi[0] ** 2, pj[0] ** 2) * (1 - costh))
            else:
                eta1 = 0.5 * math.log((pi[0] + pi[3]) / (pi[0] - pi[3]))
                eta2 = 0.5 * math.log((pj[0] + pj[3]) / (pj[0] - pj[3]))
                cos_dphi = (pi[1] * pj[1] + pi[2] * pj[2]) / (pt1 * pt2)
                dphi = math.acos(max(-1.0, min(1.0, cos_dphi)))
                ptnow = math.sqrt(
                    min(pt1**2, pt2**2)
                    * ((eta1 - eta2) ** 2 + dphi**2)
                    / d_parameter**2
                )
            ptmin = min(ptmin, ptnow)
    return ptmin


def msq(p):
    return p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2


def dot(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def rho_pythia(prad, pemt, prec, flav_rad, flav_emt, rad_type, rec_type):
    """RHOPYTHIA, statement by statement."""
    m0c, m0b, m0t, m0x = 1.5, 4.7, 172.5, 400.0
    prec_aft = list(prec)
    emtsign = -1 if rad_type == -1 else 1
    q = [prad[k] + emtsign * pemt[k] for k in range(4)]
    q2 = emtsign * msq(q)
    allowed = not q2 < 0

    id_bef = 0
    if abs(flav_emt) in (21, 22):
        id_bef = flav_rad
    if rad_type == 1 and flav_emt == -flav_rad:
        id_bef = 21
    if rad_type == 1 and abs(flav_emt) < 10 and flav_rad == 21:
        id_bef = flav_emt
    if rad_type == -1 and abs(flav_emt) < 10 and flav_rad == 21:
        id_bef = -flav_emt
    if rad_type == -1 and abs(flav_emt) < 10 and flav_rad == flav_emt:
        id_bef = 21
    if flav_emt == 24:
        id_bef = flav_rad + 1
    if flav_emt == -24:
        id_bef = flav_rad - 1

    m2_rad_aft, m2_emt_aft = msq(prad), msq(pemt)
    if m2_rad_aft < 1e-4:
        m2_rad_aft = 0.0
    if m2_emt_aft < 1e-4:
        m2_emt_aft = 0.0
    m2_rad_bef = 0.0
    if (
        abs(flav_rad) not in (21, 22)
        and abs(flav_emt) != 24
        and abs(flav_rad) != abs(flav_emt)
    ):
        m2_rad_bef = m2_rad_aft
    elif abs(flav_emt) == 24:
        if id_bef != 0:
            m2_rad_bef = {4: m0c**2, 5: m0b**2, 6: m0t**2, 9000001: m0x**2}.get(
                abs(id_bef), 0.0
            )
    elif rad_type == -1:
        if abs(flav_rad) == 21 and abs(flav_emt) == 21:
            m2_rad_bef = m2_emt_aft

    m2_final = msq([prad[k] + prec[k] + pemt[k] for k in range(4)])
    if rad_type == 1 and m2_final < 0:
        allowed = False
    rescale = 1.0
    if rad_type == 1 and rec_type == -1:
        m2_ar = m2_final - 2 * q2 + 2 * m2_rad_bef
        r = (q2 - m2_rad_bef) / (m2_ar - m2_rad_bef)
        rescale = (1 - r) / (1 + r)
        prec_aft = [x * rescale for x in prec_aft]
    if rescale < 0:
        allowed = False
    s = [prad[k] + prec_aft[k] + pemt[k] for k in range(4)]
    m2_dip = msq(s)
    x1 = 2 * dot(s, prad) / m2_dip
    x2 = 2 * dot(s, prec_aft) / m2_dip
    if rad_type == 1 and not (0 <= x1 <= 1 and 0 <= x2 <= 1):
        allowed = False
    lam = math.sqrt(
        max((q2 - m2_rad_aft - m2_emt_aft) ** 2 - 4 * m2_rad_aft * m2_emt_aft, 0)
    )
    k1 = (q2 - lam + (m2_emt_aft - m2_rad_aft)) / (2 * q2)
    k3 = (q2 - lam - (m2_emt_aft - m2_rad_aft)) / (2 * q2)
    if rad_type == -1:
        q_br = [prad[k] + prec[k] - pemt[k] for k in range(4)]
        q_ar = [prad[k] + prec[k] for k in range(4)]
        z = msq(q_br) / msq(q_ar)
    else:
        z = 1 / (1 - k1 - k3) * (x1 / (2 - x2) - k3)
    if z < 0 or z > 1:
        allowed = False
    temp = z * (1 - z) * (q2 - m2_rad_bef) if rad_type == 1 else (1 - z) * q2
    if rad_type != 1:
        if (
            (abs(flav_rad) == 4 or abs(flav_emt) == 4)
            and temp >= 0
            and math.sqrt(temp) <= 2 * m0c**2
        ):
            temp = (1 - z) * (q2 + m0c**2)
        if (
            (abs(flav_rad) == 5 or abs(flav_emt) == 5)
            and temp >= 0
            and math.sqrt(temp) <= 2 * m0b**2
        ):
            temp = (1 - z) * (q2 + m0b**2)
    if not allowed:
        temp = 1e15
    return math.sqrt(max(temp, 0.0))


def ref_pt_lund(p, pdgs, merging, lpp1, lpp2):
    """PTLUND CUT block, with the special case of two massless outgoing legs."""
    hadronic = lpp1 or lpp2
    jets = [(p[i], pdgs[i]) for i in range(2, len(p)) if merging[i]]
    massless = [f for _, f in jets if abs(f) == 21 or 1 <= abs(f) <= 5]
    if len(massless) == 0:
        return math.inf
    if len(massless) == 2 and len(jets) == 2 and len(p) == 4:
        if not hadronic:
            return math.inf
        return min(math.hypot(q[1], q[2]) for q, _ in jets)
    pinc, finc = [p[0], p[1]], [pdgs[0], pdgs[1]]
    ptminsave = math.inf
    for i, (pi, fi) in enumerate(jets):
        ptmin = math.inf
        if hadronic and abs(fi) < 30:
            ptmin = min(ptmin, rho_pythia(pinc[0], pi, pinc[1], finc[0], fi, -1, -1))
            ptmin = min(ptmin, rho_pythia(pinc[1], pi, pinc[0], finc[1], fi, -1, -1))
        for j, (pj, fj) in enumerate(jets):
            for k, (pk, _) in enumerate(jets):
                if len({i, j, k}) < 3:
                    continue
                t = rho_pythia(pj, pi, pk, fj, fi, 1, 1)
                if abs(fi) < 30:
                    ptmin = min(ptmin, t)
                t = rho_pythia(pi, pj, pk, fi, fj, 1, 1)
                if abs(fj) < 30:
                    ptmin = min(ptmin, t)
        if hadronic:
            for j, (pj, fj) in enumerate(jets):
                if abs(fi) > 1000000:
                    break
                if i != j:
                    t = rho_pythia(pj, pi, pinc[0], fj, fi, 1, -1)
                    if lpp1:
                        ptmin = min(ptmin, t)
                    t = rho_pythia(pj, pi, pinc[1], fj, fi, 1, -1)
                    if lpp2:
                        ptmin = min(ptmin, t)
        ptminsave = min(ptminsave, ptmin)
    return ptminsave


def evaluate(scale, p_ext):
    return scale(p_ext).reshape(-1).double().numpy()


def compare(computed, reference):
    reference = np.array(reference)
    none = np.isinf(reference)
    assert np.all(computed[none] >= 1e29)
    np.testing.assert_allclose(computed[~none], reference[~none], rtol=1e-8, atol=1e-8)


# --------------------------------------------------------------------------
# ktdurham
# --------------------------------------------------------------------------

KT = ms.MergingScale.Measure.kt_durham
LUND = ms.MergingScale.Measure.pt_lund


@pytest.mark.parametrize("d_parameter", [0.4, 1.0])
def test_kt_durham_hadronic_matches_cuts_f(d_parameter):
    # g g > e+ e- g g g, the leptons are not merging legs
    pdgs = [21, 21, -11, 11, 21, 21, 21]
    merging = [False, False, False, False, True, True, True]
    p_ext = sample([0.0] * 7)
    scale = ms.MergingScale(KT, pdgs, merging, d_parameter=d_parameter)
    reference = [ref_kt_durham(p.tolist(), merging, d_parameter, True) for p in p_ext]
    compare(evaluate(scale, p_ext), reference)


def test_kt_durham_leptonic_matches_cuts_f():
    # e+ e- > u u~ g g
    pdgs = [-11, 11, 2, -2, 21, 21]
    merging = [False, False, True, True, True, True]
    p_ext = sample([0.0] * 6, leptonic=True)
    scale = ms.MergingScale(KT, pdgs, merging, hadronic1=False, hadronic2=False)
    reference = [ref_kt_durham(p.tolist(), merging, 0.4, False) for p in p_ext]
    compare(evaluate(scale, p_ext), reference)


def test_kt_durham_uses_the_true_rapidity_of_massive_legs():
    # g g > t t~ g with the tops among the merging legs (madevent's default list)
    pdgs = [21, 21, 6, -6, 21]
    merging = [False, False, True, True, True]
    p_ext = sample([0.0, 0.0, 173.0, 173.0, 0.0])
    scale = ms.MergingScale(KT, pdgs, merging)
    reference = [ref_kt_durham(p.tolist(), merging, 0.4, True) for p in p_ext]
    compare(evaluate(scale, p_ext), reference)


def test_without_massless_merging_legs_there_is_no_merging_scale():
    # g g > t t~: the tops are merging legs, but none is a massless parton
    pdgs = [21, 21, 6, -6]
    scale = ms.MergingScale(KT, pdgs, [False, False, True, True])
    assert not scale.active()
    p_ext = sample([0.0, 0.0, 173.0, 173.0])
    assert np.all(evaluate(scale, p_ext) >= 1e29)
    cuts = ms.Cuts(4, merging_cut=ms.MergingCut(scale, 50.0))
    assert torch.all(cuts(p_ext).reshape(-1) == 1.0)


def test_max_jet_flavor_six_makes_the_top_massless():
    pdgs = [21, 21, 6, -6]
    assert ms.MergingScale(
        KT, pdgs, [False, False, True, True], max_jet_flavor=6
    ).active()


def test_d_parameter_must_be_positive():
    with pytest.raises(ValueError):
        ms.MergingScale(
            KT, [21, 21, 21, 21], [False, False, True, True], d_parameter=0.0
        )


# --------------------------------------------------------------------------
# ptlund
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pdgs,masses",
    [
        ([21, 21, 21, 21, 21], [0.0] * 5),
        ([2, -2, 21, 21, 21], [0.0] * 5),
        ([2, 21, 2, 21, 1, -1], [0.0] * 6),
        ([21, 21, 5, -5, 21], [0.0, 0.0, 4.7, 4.7, 0.0]),
        ([21, 21, 6, -6, 21, 21], [0.0, 0.0, 173.0, 173.0, 0.0, 0.0]),
        ([2, -1, 24, 21, 21], [0.0, 0.0, 80.4, 0.0, 0.0]),
    ],
)
def test_pt_lund_matches_cuts_f(pdgs, masses):
    merging = [False, False] + [abs(f) in (1, 2, 3, 4, 5, 6, 21) for f in pdgs[2:]]
    p_ext = sample(masses, n=150)
    scale = ms.MergingScale(LUND, pdgs, merging)
    reference = [ref_pt_lund(p.tolist(), pdgs, merging, True, True) for p in p_ext]
    compare(evaluate(scale, p_ext), reference)


def test_pt_lund_with_a_single_hadronic_beam():
    pdgs = [2, -11, 2, -11, 21]
    merging = [False, False, True, False, True]
    p_ext = sample([0.0] * 5, n=150)
    scale = ms.MergingScale(LUND, pdgs, merging, hadronic1=True, hadronic2=False)
    reference = [ref_pt_lund(p.tolist(), pdgs, merging, True, False) for p in p_ext]
    compare(evaluate(scale, p_ext), reference)


def test_pt_lund_of_two_massless_legs_is_their_pt():
    pdgs = [21, 21, 21, 21]
    merging = [False, False, True, True]
    p_ext = sample([0.0] * 4)
    computed = evaluate(ms.MergingScale(LUND, pdgs, merging), p_ext)
    pt = torch.hypot(p_ext[:, 2, 1], p_ext[:, 2, 2]).double().numpy()
    np.testing.assert_allclose(computed, pt, rtol=1e-10)
    leptonic = ms.MergingScale(LUND, pdgs, merging, hadronic1=False, hadronic2=False)
    assert not leptonic.active()


# --------------------------------------------------------------------------
# the cut
# --------------------------------------------------------------------------


@pytest.mark.parametrize("measure", [KT, LUND])
def test_cut_masks_events_below_the_merging_scale(measure):
    pdgs = [21, 21, 21, 21, 21]
    merging = [False, False, True, True, True]
    p_ext = sample([0.0] * 5)
    scale = ms.MergingScale(measure, pdgs, merging)
    values = evaluate(scale, p_ext)
    cut = float(np.median(values))
    mask = ms.Cuts(5, merging_cut=ms.MergingCut(scale, cut))(p_ext).reshape(-1)
    np.testing.assert_array_equal(mask.numpy(), (values >= cut).astype(float))
    # combined with ordinary cut items, both have to pass
    O = ms.Observable
    cuts = ms.Cuts(
        [ms.CutItem(O(pdgs, O.obs_pt, [[21]]), min=10.0)],
        merging_cut=ms.MergingCut(scale, cut),
    )
    pt_ok = (torch.hypot(p_ext[:, 2:, 1], p_ext[:, 2:, 2]) >= 10.0).all(dim=1)
    expected = (torch.from_numpy(values >= cut) & pt_ok).double()
    np.testing.assert_array_equal(cuts(p_ext).reshape(-1).numpy(), expected.numpy())


def test_hadronic_kt_durham_floors_the_pt_of_merging_legs():
    pdgs = [21, 21, -11, 11, 21, 21]
    merging = [False, False, False, False, True, True]
    cut = ms.MergingCut(ms.MergingScale(KT, pdgs, merging), 30.0)
    assert ms.Cuts(6, merging_cut=cut).pt_min() == [0.0, 0.0, 30.0, 30.0]
    # neither ptlund nor a lepton collider bound the pT
    lund = ms.MergingCut(ms.MergingScale(LUND, pdgs, merging), 30.0)
    assert ms.Cuts(6, merging_cut=lund).pt_min() == [0.0] * 4
    lep = ms.MergingCut(
        ms.MergingScale(KT, pdgs, merging, hadronic1=False, hadronic2=False), 30.0
    )
    assert ms.Cuts(6, merging_cut=lep).pt_min() == [0.0] * 4


def test_merging_cut_must_match_the_particle_count():
    scale = ms.MergingScale(KT, [21] * 5, [False, False, True, True, True])
    with pytest.raises(ValueError):
        ms.Cuts(6, merging_cut=ms.MergingCut(scale, 30.0))
