"""FKS with lepton PDFs: the luminosity of generate_ee_tau_y, the emission off
an incoming lepton without the projection of the event (generate_noevpr_
kinematics of genps_fks.f) and the FKS mapping of lepton beams.

The reference data of lepton collisions (see test_fks_reference.py) give at
every point of an incoming emitter the momentum fractions of the luminosity
and, for the event and its soft, collinear and soft-collinear counter-events,
the energy squared s_rec of the Born, the weight of generate_momenta_born at
it and the Born momenta in their rest frame. The luminosity must give the
momentum fractions of genps_fks.f from its random numbers, and the emission,
from the random numbers of the radiation and those Born momenta, the momenta
of the event and counter-events. Their Jacobians, with that of the Born at
s_rec, the luminosity and the flux, must be those of genps_fks.f. The FKS
mapping of lepton beams must be invertible, with the volume of the real phase
space, and the Born of its collinear counter-event must be that of the
collinear limit (p_born_coll).
"""

import math

import numpy as np
import pytest
import torch
from fks_testing import FKS_DATA, FKS_DATA_IDS, evaluate, reference
from madspace_testing import default_context

import madspace as ms
from madspace.fks import fks_configuration_mapping

# get_ee_expo of the built-in lepton PDFs of MadGraph (pdg2pdf.f)
BETA = 0.975
STINY, CTINY = 1e-6, 5e-7
# position k + 2 of the counter-events k = 0, 1, 2 of genps_fks.f
SOFT, COLLINEAR, SOFT_COLLINEAR = 2, 3, 4


def is_lepton(path):
    return abs(reference(path)["lpp"][0]) in (3, 4)


LEPTON_DATA = [path for path in FKS_DATA if is_lepton(path)]
LEPTON_IDS = [FKS_DATA_IDS[FKS_DATA.index(path)] for path in LEPTON_DATA]


def to_torch(x):
    return torch.as_tensor(np.asarray(x)).to(torch.get_default_device())


def to_torch_int(x):
    return torch.as_tensor(x, dtype=torch.int32).to(torch.get_default_device())


def to_numpy(x):
    return torch.as_tensor(x).cpu().numpy()


def noevpr_rows(data):
    """The configurations and points of an incoming emitter without the
    projection of the event."""
    return [
        (conf, point)
        for conf in data["configs"]
        for point in conf["points"] + conf["limit_points"]
        if "noevpr" in point
    ]


def flux_factor(shat, nexternal, nin=2):
    """compute_flux of genps_fks.f for massless incoming partons."""
    pi2 = 2 * math.pi
    return 1 / (2 * shat) / pi2 ** (3 * (nexternal - nin) - 4) * pi2**3


def s_lab(data):
    return 4 * data["ebeam"][0] * data["ebeam"][1]


def test_lepton_data_exist():
    """The reference data cover lepton collisions with incoming emitters."""
    assert LEPTON_DATA
    for path in LEPTON_DATA:
        assert noevpr_rows(reference(path))


@pytest.mark.parametrize("path", LEPTON_DATA, ids=LEPTON_IDS)
def test_luminosity_matches_generate_x_ee(path):
    """x1 and x2 of generate_x_ee from the random numbers of tau and y, with
    x_min = tau_Born_lower_bound for x1 and that over x1 for x2, and the
    Jacobian of generate_x_ee times (1 - x1)^beta (1 - x2)^beta."""
    data = reference(path)
    rows = noevpr_rows(data)
    ndim = data["ndim"]
    r = np.array([point["x"][ndim - 5 : ndim - 3] for _, point in rows])
    tau = np.array([point["noevpr"]["tau_born_lower_bound"] for _, point in rows])
    luminosity = ms.LeptonLuminosity(s_lab(data), BETA, s_hat_min_condition=True)
    x1, x2, s_hat, omx1, omx2, det = map(
        to_numpy,
        luminosity.map_forward(
            [to_torch(r[:, 0]), to_torch(r[:, 1])], [to_torch(tau * s_lab(data))]
        ),
    )
    x = np.array([point["noevpr"]["x"] for _, point in rows])
    omx = np.array([point["noevpr"]["one_minus_x"] for _, point in rows])
    np.testing.assert_allclose(np.stack([omx1, omx2], 1), omx, rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.stack([x1, x2], 1), 1 - omx, rtol=1e-15, atol=0)
    # genps_fks.f takes the momentum fractions from tau and the rapidity
    # (get_tau_y_from_x12), which it sets to zero below 1e-12
    np.testing.assert_allclose(np.stack([x1, x2], 1), x, rtol=0, atol=3e-12)
    np.testing.assert_allclose(s_hat, x1 * x2 * s_lab(data), rtol=1e-15)
    a = 1 / (1 - BETA)
    jac = a * (1 - tau) ** (1 - BETA) * a * (1 - tau / x1) ** (1 - BETA)
    np.testing.assert_allclose(det, jac * (omx1 * omx2) ** BETA, rtol=1e-13)
    # and back, from 1 - x
    r1, r2, det_inv = map(
        to_numpy,
        luminosity.map_inverse(
            [to_torch(v) for v in (x1, x2, s_hat, omx1, omx2)],
            [to_torch(tau * s_lab(data))],
        ),
    )
    np.testing.assert_allclose(np.stack([r1, r2], 1), r, rtol=1e-13)
    np.testing.assert_allclose(det * det_inv, 1.0, rtol=1e-13)


def fixed_emission(n_final):
    """fks_initial_fixed_radiate and fks_initial_fixed_boost as a function of
    xi_hat, y, r_phi, sqrt(s), s_born_min, beam_sign and the final state of
    the Born in its rest frame."""
    vec = ms.batch_four_vec_array(n_final)
    names = ["xi_hat", "y", "r_phi", "sqrt_s", "s_born_min", "beam_sign"]
    fb = ms.FunctionBuilder(
        ms.NamedTypes([(name, ms.batch_float) for name in names] + [("born", vec)]),
        ms.NamedTypes(
            [
                ("p_i", ms.batch_four_vec),
                ("xi", ms.batch_float),
                ("xi_max", ms.batch_float),
                ("sqrt_s_rec", ms.batch_float),
                ("det_xi", ms.batch_float),
                ("p_i_hat", ms.batch_four_vec),
                ("final", vec),
            ]
        ),
    )
    p_i, xi, xi_max, sqrt_s_rec, det_xi, p_i_hat = fb.fks_initial_fixed_radiate(
        *[fb.input(k) for k in range(6)]
    )
    final = fb.fks_initial_fixed_boost(fb.input(6), p_i_hat, xi)
    for k, value in enumerate([p_i, xi, xi_max, sqrt_s_rec, det_xi, p_i_hat, final]):
        fb.output(k, value)
    return ms.FunctionRuntime(fb.function(), default_context())


def configuration_cases():
    cases = []
    for path in LEPTON_DATA:
        data = reference(path)
        for conf in data["configs"]:
            if any("noevpr" in p for p in conf["points"] + conf["limit_points"]):
                name = f"{FKS_DATA_IDS[FKS_DATA.index(path)]}-fks{conf['nfks']}"
                cases.append(pytest.param(path, conf["nfks"], id=name))
    return cases


@pytest.mark.parametrize("path,nfks", configuration_cases())
def test_emission_matches_genps_fks(path, nfks):
    """From the random numbers of the radiation and the Born of
    generate_momenta_born at s_rec, the event and the counter-events of
    generate_noevpr_kinematics: i at fixed sqrt(s), the Born boosted against
    it, xi, xi_max and i per unit xi, and the Jacobians, which with the
    luminosity, the Born at s_rec and the flux at sqrt(s) are those of
    genps_fks.f."""
    data = reference(path)
    conf = data["configs"][nfks - 1]
    points = [p for p in conf["points"] + conf["limit_points"] if "noevpr" in p]
    ndim, nex = data["ndim"], data["nexternal"]
    i_fks, j_fks = conf["i_fks"] - 1, conf["j_fks"] - 1
    assert j_fks < 2

    def field(key, position=None):
        values = np.array([p[key] for p in points])
        return values if position is None else values[:, position]

    x = field("x")
    r_xi, r_y, r_phi = x[:, ndim - 3], x[:, ndim - 2], x[:, ndim - 1]
    xi_hat = STINY + (1 - STINY) * r_xi**2
    y = 1 - 2 * (CTINY + (1 - CTINY) * r_y**2)
    det_sample = 8 * (1 - STINY) * (1 - CTINY) * r_xi * r_y
    shat = field("shat_ev")
    sqrt_s = np.sqrt(shat)
    tau = np.array([p["noevpr"]["tau_born_lower_bound"] for p in points])
    s_born_min = tau * s_lab(data)
    sign = np.full(len(points), 1.0 if j_fks == 0 else -1.0)
    one = np.ones(len(points))
    # luminosity of generate_x_ee, as in test_luminosity_matches_generate_x_ee
    xs = np.array([p["noevpr"]["x"] for p in points])
    a = 1 / (1 - BETA)
    jac_lumi = a * (1 - tau) ** (1 - BETA) * a * (1 - tau / xs[:, 0]) ** (1 - BETA)
    runtime = fixed_emission(nex - 3)
    # event, soft, collinear and soft-collinear counter-event
    cases = [
        ("event", xi_hat, y, field("p_ev"), field("jac")),
        ("soft", 0 * one, y, field("p_cnt", SOFT), field("jac_cnt", SOFT)),
        (
            "collinear",
            xi_hat,
            one,
            field("p_cnt", COLLINEAR),
            field("jac_cnt", COLLINEAR),
        ),
        (
            "soft_collinear",
            0 * one,
            one,
            field("p_cnt", SOFT_COLLINEAR),
            field("jac_cnt", SOFT_COLLINEAR),
        ),
    ]
    for k, (name, xh, yv, p_ref, jac_ref) in enumerate(cases):
        rec = [p["noevpr"]["born_rec"][k] for p in points]
        born = np.array([r["p_born"] for r in rec])
        s_rec = np.array([r["s_rec"] for r in rec])
        born_weight = np.array([r["weight"] for r in rec])
        p_i, xi, xi_max, sqrt_s_rec, det_xi, p_i_hat, final = map(
            to_numpy,
            runtime.call(
                [to_torch(v) for v in (xh, yv, r_phi, sqrt_s, s_born_min, sign)]
                + [to_torch(born[:, 2:])]
            ),
        )
        # the Born of genps_fks.f is at the energy that the emission leaves
        np.testing.assert_allclose(sqrt_s_rec**2, s_rec, rtol=1e-14, err_msg=name)
        incoming = np.zeros((len(points), 2, 4))
        incoming[:, :, 0] = sqrt_s[:, None] / 2
        incoming[:, 0, 3] = sqrt_s / 2
        incoming[:, 1, 3] = -sqrt_s / 2
        ours = np.concatenate([incoming, final], axis=1)
        ours = np.insert(ours, i_fks, p_i, axis=1)
        ok = jac_ref > 0
        assert ok.any(), name
        error = np.abs(ours - np.array(p_ref)).max((1, 2)) / sqrt_s
        assert error[ok].max() < 1e-13, name
        np.testing.assert_allclose(
            xi[ok], 0.0 if name.startswith("soft") else field("xi_ev")[ok], rtol=1e-14
        )
        np.testing.assert_allclose(xi_max[ok], field("xiimax_ev")[ok], rtol=1e-14)
        hat = field("p_i_hat_ev") if k == 0 else field("p_i_hat_cnt", k + 1)
        hat_error = np.abs(p_i_hat - hat).max(1) / sqrt_s
        assert hat_error[ok].max() < 1e-14, name
        # the Jacobian of genps_fks.f has neither the factors 1 - stiny and
        # 1 - ctiny of the sampling of xi_hat and y nor xi_max
        jac = (
            jac_lumi
            * det_xi
            * det_sample
            / (xi_max * (1 - STINY) * (1 - CTINY))
            * born_weight
            * flux_factor(shat, nex)
        )
        np.testing.assert_allclose(jac[ok], jac_ref[ok], rtol=1e-13, err_msg=name)


@pytest.mark.parametrize("path", LEPTON_DATA, ids=LEPTON_IDS)
def test_collinear_born_of_genps_fks(path):
    """p_born_coll of genps_fks.f: the collinear counter-event without i,
    with j - i for j."""
    data = reference(path)
    for conf, point in noevpr_rows(data):
        if point["jac_cnt"][COLLINEAR] <= 0:
            continue
        i_fks, j_fks = conf["i_fks"] - 1, conf["j_fks"] - 1
        p = np.array(point["p_cnt"][COLLINEAR])
        born = np.delete(p, i_fks, axis=0)
        born[j_fks] = p[j_fks] - p[i_fks]
        np.testing.assert_allclose(
            point["p_born_coll"], born, rtol=0, atol=1e-12 * p[0, 0]
        )


def lepton_mapping(data, cuts_floor=0.0):
    """The FKS mapping of lepton beams of the configurations of reference
    data, with a RAMBO Born mapping of their masses."""
    conf = data["configs"][0]
    masses = np.delete(np.array(conf["masses"]), conf["i_fks"] - 1)
    sqrt_s = math.sqrt(s_lab(data))
    born = ms.PhaseSpaceMapping(
        list(masses),
        sqrt_s,
        leptonic=True,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
        sqrt_s_hat_condition=True,
    )
    luminosity = ms.LeptonLuminosity(s_lab(data), BETA, s_hat_min_condition=True)
    mapping = fks_configuration_mapping(
        born, data["fks_configurations"], luminosity=luminosity
    )
    return born, luminosity, mapping


@pytest.mark.parametrize("path", LEPTON_DATA, ids=LEPTON_IDS)
def test_mapping_round_trip(path):
    """The FKS mapping of lepton beams and its inverse."""
    data = reference(path)
    _, _, mapping = lepton_mapping(data)
    count = len(data["configs"])
    rng = np.random.default_rng(11)
    batch = 2000
    r = 0.02 + 0.96 * rng.random((batch, mapping.random_dim()))
    index = rng.integers(0, count, batch).astype(np.int32)
    p, x1, x2, omx1, omx2, weight = map(
        to_numpy, mapping.map_forward([to_torch(r)], [to_torch_int(index)])
    )
    ok = weight > 0
    assert ok.mean() > 0.9
    # most momentum fractions are 1 to a double, not 1 - x
    assert (x1 == 1.0).any() and (omx1 > 0).all() and (omx2 > 0).all()
    np.testing.assert_allclose(x1, 1 - omx1, rtol=1e-15, atol=0)
    # four-momentum conservation in the lab and the momentum fractions
    total = p[:, 0] + p[:, 1] - p[:, 2:].sum(1)
    assert np.abs(total[ok]).max() < 1e-12 * math.sqrt(s_lab(data))
    np.testing.assert_allclose(p[ok, 0, 0], x1[ok] * data["ebeam"][0], rtol=1e-14)
    np.testing.assert_allclose(p[ok, 1, 0], x2[ok] * data["ebeam"][1], rtol=1e-14)
    r_back, det_inv = map(
        to_numpy,
        mapping.map_inverse(
            [to_torch(v) for v in (p, x1, x2, omx1, omx2)], [to_torch_int(index)]
        ),
    )
    # the RAMBO Born mapping of a 2 -> 2 Born resolves its random numbers
    # only to a few 1e-11 on some points, also for a final-state emitter
    np.testing.assert_allclose(r_back[ok], r[ok], rtol=0, atol=1e-10)
    np.testing.assert_allclose(weight[ok] * det_inv[ok], 1.0, rtol=1e-9)


@pytest.mark.parametrize("path", LEPTON_DATA, ids=LEPTON_IDS)
def test_counter_events(path):
    """The counter-events of an incoming emitter keep the momentum fractions
    and the partonic energy of the event, i is soft or along the beam of j,
    and collinear_born_momenta is the collinear counter-event without i with
    j - i, which is the Born at s_rec boosted along the beam."""
    data = reference(path)
    _, _, mapping = lepton_mapping(data)
    names = list(mapping.counter_event_types().keys())
    count = len(data["configs"])
    rng = np.random.default_rng(13)
    batch = 500
    r = 0.02 + 0.96 * rng.random((batch, mapping.random_dim()))
    index = rng.integers(0, count, batch).astype(np.int32)
    out = dict(zip(names, evaluate(ms.FKSKinematics(mapping), r, fks_index=index)))
    j = np.array([c["j_fks"] - 1 for c in data["configs"]])[index]
    i = np.array([c["i_fks"] - 1 for c in data["configs"]])[index]
    rows = np.arange(batch)
    incoming = j < 2
    for name in ["soft", "collinear", "soft_collinear"]:
        np.testing.assert_array_equal(out[f"{name}_x1"], out["x1"])
        np.testing.assert_array_equal(out[f"{name}_x2"], out["x2"])
        p = out[f"{name}_momenta"]
        ok = out[f"{name}_det"] > 0
        assert ok.any()
        if name != "collinear":
            assert np.all(p[rows, i][ok, 0] == 0.0)
    # the collinear counter-event of an incoming emitter: i along the beam
    p_c = out["collinear_momenta"]
    ok = (out["collinear_det"] > 0) & incoming
    assert ok.any()
    p_i = p_c[rows, i]
    assert (np.abs(p_i[ok, 1:3]).max(1) <= 1e-12 * p_i[ok, 0]).all()
    expected = np.array([np.delete(p_c[k], i[k], axis=0) for k in range(batch)])
    for k in np.flatnonzero(ok):
        jb = j[k]
        expected[k, jb] = p_c[k, jb] - p_c[k, i[k]]
    coll = out["collinear_born_momenta"]
    scale = np.abs(p_c).max((1, 2))
    error = np.abs(coll - expected).max((1, 2)) / scale
    assert error[ok].max() < 1e-13
    # the others take the Born
    other = ~incoming
    np.testing.assert_array_equal(coll[other], out["born_momenta"][other])


def mean_and_error(values):
    return values.mean(), values.std() / math.sqrt(len(values))


@pytest.mark.parametrize("path", LEPTON_DATA, ids=LEPTON_IDS)
def test_volume(path):
    """Per configuration, the FKS mapping of lepton beams covers the real
    phase space at the momentum fractions of the luminosity once: its mean
    weight is that of the luminosity times a RAMBO mapping of the real
    process at the partonic energy."""
    data = reference(path)
    _, luminosity, mapping = lepton_mapping(data)
    conf = data["configs"][0]
    sqrt_s = math.sqrt(s_lab(data))
    real = ms.PhaseSpaceMapping(
        list(conf["masses"]),
        sqrt_s,
        leptonic=True,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
        sqrt_s_hat_condition=True,
    )
    rng = np.random.default_rng(17)
    batch = 200000
    # the reference: the luminosity without a bound (the Born is massless and
    # has no cuts) and the real phase space at its energy
    for nfks in sorted({c["nfks"] for c in data["configs"]}):
        r = rng.random((batch, mapping.random_dim()))
        index = np.full(batch, nfks - 1, dtype=np.int32)
        *_, weight = map(
            to_numpy, mapping.map_forward([to_torch(r)], [to_torch_int(index)])
        )
        r_lumi = rng.random((batch, 2))
        x1, x2, s_hat, _, _, det_lumi = map(
            to_numpy,
            luminosity.map_forward(
                [to_torch(r_lumi[:, 0]), to_torch(r_lumi[:, 1])],
                [to_torch(np.zeros(batch))],
            ),
        )
        r_real = rng.random((batch, real.random_dim()))
        *_, det_real = map(
            to_numpy,
            real.map_forward([to_torch(r_real)], [to_torch(np.sqrt(s_hat))]),
        )
        ours, ours_error = mean_and_error(weight)
        theirs, theirs_error = mean_and_error(det_lumi * det_real)
        assert abs(ours - theirs) < 5 * math.hypot(ours_error, theirs_error)


def test_invalid_lepton_mappings():
    """A lepton luminosity needs a leptonic Born mapping with the sqrt_s_hat
    condition and the s_hat_min condition of the luminosity, and a Born
    mapping with the sqrt_s_hat condition needs the luminosity."""
    masses = [0.0] * 4
    configurations = [(4, 0), (4, 2)]

    def born(**options):
        return ms.PhaseSpaceMapping(
            masses, 1000.0, mode=ms.PhaseSpaceMapping.TChannelMode.rambo, **options
        )

    luminosity = ms.LeptonLuminosity(1e6, BETA, s_hat_min_condition=True)
    fixed = born(leptonic=True, sqrt_s_hat_condition=True)
    mapping = ms.FKSConfigurationMapping(fixed, configurations, luminosity=luminosity)
    assert mapping.random_dim() == fixed.random_dim() + 5
    for bad_born in [born(), born(leptonic=True)]:
        with pytest.raises(ValueError, match="lepton luminosity"):
            ms.FKSConfigurationMapping(bad_born, configurations, luminosity=luminosity)
    with pytest.raises(ValueError, match="s_hat_min condition"):
        ms.FKSConfigurationMapping(
            fixed, configurations, luminosity=ms.LeptonLuminosity(1e6, BETA)
        )
    with pytest.raises(ValueError, match="needs a lepton luminosity"):
        ms.FKSConfigurationMapping(fixed, configurations)
    with pytest.raises(ValueError, match="lepton beams"):
        ms.FKSMapping(fixed, 4, 0)
    with pytest.raises(ValueError, match="sqrt_s_hat condition"):
        born(sqrt_s_hat_condition=True)
    for bad in [dict(beta=1.0), dict(s_hat_min=1e6), dict(s_hat_min=-1.0)]:
        with pytest.raises(ValueError, match="LeptonLuminosity"):
            ms.LeptonLuminosity(1e6, **{"beta": BETA, **bad})
