"""FKSKinematics: the real emission of an FKSMapping, its Born and its FKS
counter-events, against genps_fks.f.

Each reference point (see test_fks_reference.py) gives a Born, in its
center-of-mass frame, with its momentum fractions and rapidity, and the last
three entries of the unit-cube vector. The Born is boosted into the lab and
mapped back to the random numbers of a RAMBO Born mapping, so that
FKSKinematics, run on them and on the three random numbers of genps_fks.f, has
the same Born. The FKSMapping comes from the metadata of the FKS exporter
(fks_configurations.json, stored with the reference data). Its event and
counter-events, boosted back into the frame of the Born, must then be those of
genps_fks.f (the rad_ entries, and the cut_ ones with the lower bound on the
partonic energy), with their Jacobians in the conventions of
test_fks_kernels.py and test_fks_radiation.py, and its counter-events must be
generated exactly where genps_fks.f generates them. Lepton PDFs have their
own mapping, see test_fks_lepton.py.
"""

import glob
import json
import math
import os

import numpy as np
import pytest
import torch
from fks_testing import massive_fortran_precision
from madspace_testing import default_context

import madspace as ms
from madspace.fks import fks_mappings

STINY, CTINY = 1e-6, 5e-7
DATA = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_data", "fks", "*.json"
        )
    )
)
# position k + 2 of the counter-events k = 0, 1, 2 of genps_fks.f
POSITIONS = {"soft": 2, "collinear": 3, "soft_collinear": 4}


def to_torch(x):
    return torch.as_tensor(np.asarray(x)).to(torch.get_default_device())


def to_numpy(x):
    return torch.as_tensor(x).cpu().numpy()


def boost_z(p, masses, y):
    """Boost along z by the rapidity y, in light-cone components with the
    smaller one from the mass shell."""
    q = p.copy()
    trans2 = np.asarray(masses) ** 2 + p[..., 1] ** 2 + p[..., 2] ** 2
    forward = p[..., 3] >= 0
    big = np.where(forward, p[..., 0] + p[..., 3], p[..., 0] - p[..., 3])
    small = trans2 / np.maximum(big, 1e-300)
    plus = np.where(forward, big, small) * np.exp(y)[:, None]
    minus = np.where(forward, small, big) * np.exp(-y)[:, None]
    q[..., 0] = 0.5 * (plus + minus)
    q[..., 3] = 0.5 * (plus - minus)
    return q


def flux_factor(shat, nexternal, nin=2):
    """compute_flux of genps_fks.f for massless incoming partons."""
    pi2 = 2 * math.pi
    return 1 / (2 * shat) / pi2 ** (3 * (nexternal - nin) - 4) * pi2**3


def cases():
    out = []
    for path in DATA:
        with open(path) as f:
            data = json.load(f)
        if abs(data["lpp"][0]) in (3, 4):
            # lepton PDFs, whose momentum fractions close to 1 the luminosity
            # of a hadronic Born mapping resolves poorly: see test_fks_lepton.py
            continue
        nin = data["nincoming"]
        for conf in data["configs"]:
            points = [p for p in conf["points"] if p["jac"] > 0 and "rad_jac" in p]
            if not points:
                continue
            j = conf["j_fks"]
            kind = "initial" if j <= nin else "final"
            if j > nin and conf["masses"][j - 1] > 0:
                kind = "massive"
            name = f"{os.path.basename(path)[:-5]}-fks{conf['nfks']}-{kind}"
            for prefix in ["rad", "cut"] if kind == "initial" else ["rad"]:
                out.append(
                    pytest.param(data, conf, points, prefix, id=f"{name}-{prefix}")
                )
    return out


@pytest.mark.parametrize("data,conf,points,prefix", cases())
def test_matches_genps_fks(data, conf, points, prefix):
    nex = data["nexternal"]
    i_fks, j_fks = conf["i_fks"] - 1, conf["j_fks"] - 1
    leptonic = data["lpp"] == [0, 0]
    masses = np.array(conf["masses"])
    born_masses = np.delete(masses, i_fks)
    s_lab = 4 * data["ebeam"][0] * data["ebeam"][1]
    s_hat_min = points[0]["tau_lower_bound"] * s_lab if prefix == "cut" else 0.0

    def field(key):
        if key.startswith("rad_"):
            key = prefix + key[3:]
        return np.array([p[key] for p in points])

    born = ms.PhaseSpaceMapping(
        list(born_masses),
        math.sqrt(s_lab),
        leptonic=leptonic,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
    )
    # the configuration from the metadata of the FKS exporter (fks.py)
    mapping = fks_mappings(born, data["fks_configurations"], s_hat_min)[
        conf["nfks"] - 1
    ]
    assert (mapping.i_fks(), mapping.j_fks()) == (i_fks, j_fks)
    kinematics = ms.FKSKinematics(mapping)
    names = list(mapping.counter_event_types().keys())

    # the Born of genps_fks.f in the lab, and the random numbers that give it
    y_born = field("ycm_born")
    x_born = field("xbjrk_born")
    born_lab = boost_z(field("p_born"), born_masses, y_born)
    r_born, _ = born.map_inverse(
        [to_torch(born_lab), to_torch(x_born[:, 0]), to_torch(x_born[:, 1])]
    )
    r = np.concatenate([to_numpy(r_born), field("x")[:, -3:]], axis=1)
    runtime = ms.FunctionRuntime(kinematics.function(), default_context())
    out = dict(zip(names, [to_numpy(t) for t in runtime.call([to_torch(r)])]))
    batch = len(points)
    for key, value in out.items():
        if value.ndim == 0:
            out[key] = np.full(batch, value)

    # The Born comes back through the round trip of the Born mapping and the
    # boosts of PhaseSpaceMapping, so it differs from that of genps_fks.f by up
    # to 2e-13 of its energy, and the emission off it by about as much. All
    # comparisons below scale with this deviation.
    sqrt_s_born = np.sqrt(field("shat_born"))
    born_again = boost_z(out["born_momenta"], born_masses, -y_born)
    born_dev = np.abs(born_again - field("p_born")).max((1, 2)) / sqrt_s_born
    assert (born_dev <= 3e-13).all()
    # relative to that, the deviation of the direction of the Born emitter
    jbar = j_fks if j_fks < i_fks else j_fks - 1
    direction_dev = sqrt_s_born / field("p_born")[:, jbar, 0]
    if not leptonic:
        np.testing.assert_allclose(out["born_x1"], x_born[:, 0], rtol=1e-12)
        np.testing.assert_allclose(out["born_x2"], x_born[:, 1], rtol=1e-12)

    # the event, in the frame of the Born
    p_ev = field("rad_p_ev")
    scale = p_ev[..., 0].max(1)
    prec_k = prec_jac = np.zeros(batch)
    if j_fks >= 2 and masses[j_fks] > 0:
        jb = j_fks if j_fks < i_fks else j_fks - 1
        prec_k, prec_jac = massive_fortran_precision(
            p_ev, field("p_born")[:, jb], i_fks, j_fks, masses[j_fks]
        )
        prec_k = prec_k * np.sqrt(field("shat_born")) / scale
    event = boost_z(out["momenta"], masses, -y_born)
    diff = np.abs(event - p_ev).max((1, 2)) / scale
    assert (diff <= 1e-13 + 10 * born_dev + 10 * prec_k).all()

    def assert_close(ours, theirs, rtol):
        """to rtol, and to the deviation of the Born absolute"""
        assert (np.abs(ours - theirs) <= rtol * np.abs(theirs) + 10 * born_dev).all()

    assert_close(out["xi"], field("rad_xi_ev"), 1e-12)
    assert_close(out["xi_hat"], field("rad_xi_hat_ev"), 1e-12)
    assert_close(out["y"], field("rad_y_ev"), 1e-12)
    assert_close(out["xi_max"], field("rad_xiimax_ev"), 1e-12)
    if not leptonic:
        assert_close(out["x1"], field("rad_xbjrk_ev")[:, 0], 1e-12)
        assert_close(out["x2"], field("rad_xbjrk_ev")[:, 1], 1e-12)

    # i per unit xi: p_i = xi p_i_hat (the event of genps_fks.f is compared
    # above, its p_i_fks_ev adds nothing; its counter-events below, where
    # p_i_hat is all that is left of i)
    np.testing.assert_allclose(
        out["xi"][:, None] * out["p_i_hat"],
        out["momenta"][:, i_fks],
        rtol=0,
        atol=1e-13 * scale.max(),
    )

    # the radiation Jacobian: genps_fks.f leaves out xi, the width of the range
    # of xi (xi_max - xi_min, xi_norm for a massive emitter) and the factors
    # 1 - stiny (not for the second solution of a massive emitter) and 1 - ctiny
    xi_ev, xi_hat_ev = field("rad_xi_ev"), field("rad_xi_hat_ev")
    second = field("isolsign") == -1
    xi_min = np.zeros(batch)
    if j_fks < 2 and s_hat_min > 0:
        xi_min = np.maximum(1 - field("shat_born") / s_hat_min, 0.0)
    if j_fks >= 2 and masses[j_fks] > 0:
        width = np.where(second, 2 * field("rad_xiimax_ev") - xi_ev, xi_ev) / xi_hat_ev
    else:
        width = field("rad_xiimax_ev") - xi_min
    units = (1 - np.where(second, 0.0, STINY)) * (1 - CTINY)
    # xi_max = 1 - M_rec^2 / s_hat (final-state emitter) takes the deviation of
    # the Born relative to xi_max, which goes to 0 at the edge of the Born phase
    # space
    rtol_jac = 1e-11 + 50 * prec_jac + 10 * born_dev / field("rad_xiimax_ev")
    rad_weight = out["weight"] / out["born_weight"]
    fortran = field("rad_jac") / flux_factor(field("rad_shat_ev"), nex)
    expected = fortran * xi_ev * width * units
    assert (np.abs(rad_weight / expected - 1) <= rtol_jac).all()

    # the counter-events, where genps_fks.f generates them
    jac_cnt = field("rad_jac_cnt")
    for name, pos in POSITIONS.items():
        valid = jac_cnt[:, pos] > 0
        if name + "_det" not in out:
            assert not valid.any()
            continue
        det = out[name + "_det"] / out["born_weight"]
        np.testing.assert_array_equal(det > 0, valid)
        if not valid.any():
            continue
        p_cnt = field("rad_p_cnt")[:, pos]
        cnt = boost_z(out[name + "_momenta"], masses, -y_born)
        # in units of its hardest particle: in the frame of the Born, the
        # collinear counter-event of an incoming emitter has energies up to
        # 1 / (1 - xi) times those of the Born at its own xi (that at y = 1),
        # far above those of the event (1393 GeV against 60 GeV on one point)
        scale_cnt = p_cnt[..., 0].max(1)
        diff = np.abs(cnt - p_cnt).max((1, 2)) / scale_cnt
        assert (diff <= 1e-13 + 10 * born_dev)[valid].all()
        # i per unit xi, finite where i is soft (p_i_fks_cnt). Its direction is
        # set relative to the Born emitter, whose own direction carries the
        # deviation of the Born divided by its energy (a soft emitter amplifies
        # it).
        hat = boost_z(out[name + "_p_i_hat"][:, None], [0.0], -y_born)[:, 0]
        hat_scale_cnt = np.sqrt(field("rad_shat_cnt")[:, pos]) / 2
        diff = np.abs(hat - field("rad_p_i_hat_cnt")[:, pos]).max(1) / hat_scale_cnt
        assert (diff <= 1e-13 + 10 * born_dev * direction_dev)[valid].all()
        # its xi and the upper end of the range of xi at its y
        for key, fortran_key in [("_xi", "rad_xi_cnt"), ("_xi_max", "rad_xiimax_cnt")]:
            theirs = field(fortran_key)[:, pos]
            close = np.abs(out[name + key] - theirs) <= (
                1e-12 * np.abs(theirs) + 10 * born_dev
            )
            assert close[valid].all(), key
        if not leptonic:
            x_cnt = field("rad_xbjrk_cnt")[:, pos]
            dx1 = np.abs(out[name + "_x1"] / x_cnt[:, 0] - 1)
            dx2 = np.abs(out[name + "_x2"] / x_cnt[:, 1] - 1)
            assert (np.maximum(dx1, dx2) <= 1e-12 + 10 * born_dev)[valid].all()
        width_cnt = width
        if j_fks < 2:
            width_cnt = field("rad_xiimax_cnt")[:, pos] - xi_min
        fortran_cnt = jac_cnt[:, pos] / flux_factor(field("rad_shat_cnt")[:, pos], nex)
        deviation = np.abs(det / (fortran_cnt * width_cnt * units) - 1)
        assert (deviation <= rtol_jac)[valid].all()


def test_counter_events_of_each_emitter():
    """soft, collinear and soft-collinear for every emitter, each with momenta,
    momentum fractions and Jacobian, so that the FKS configurations of a Born
    share one layout."""
    names = ["soft", "collinear", "soft_collinear"]
    born = ms.PhaseSpaceMapping([0.0, 0.0, 173.0, 173.0], 1000.0, mode="rambo")
    for j_fks in [0, 2]:
        mapping = ms.FKSMapping(born, 4, j_fks)
        assert list(mapping.radiation().counter_events()) == names
        keys = list(mapping.counter_event_types().keys())
        assert keys[13:] == [
            f"{name}_{k}"
            for name in names
            for k in ["momenta", "x1", "x2", "det", "p_i_hat", "xi", "xi_max"]
        ]
    born = ms.PhaseSpaceMapping([0.0, 0.0, 0.0, 0.0], 1000.0, mode="rambo")
    keys = list(ms.FKSMapping(born, 4, 2).counter_event_types().keys())
    assert keys[:13] == [
        "momenta",
        "x1",
        "x2",
        "weight",
        "born_momenta",
        "born_x1",
        "born_x2",
        "born_weight",
        "xi",
        "xi_hat",
        "y",
        "xi_max",
        "p_i_hat",
    ]


def test_no_collinear_counter_events_of_a_massive_emitter():
    """they keep the layout with the event and det = 0"""
    born = ms.PhaseSpaceMapping([0.0, 0.0, 173.0, 173.0], 1000.0, mode="rambo")
    mapping = ms.FKSMapping(born, 4, 2)
    r = np.random.default_rng(5).random((500, mapping.random_dim()))
    names = list(mapping.counter_event_types().keys())
    runtime = ms.FunctionRuntime(
        ms.FKSKinematics(mapping).function(), default_context()
    )
    out = dict(zip(names, [to_numpy(t) for t in runtime.call([to_torch(r)])]))
    for name in ["collinear", "soft_collinear"]:
        np.testing.assert_array_equal(out[name + "_det"], 0.0)
        for k in ["momenta", "x1", "x2"]:
            np.testing.assert_array_equal(out[f"{name}_{k}"], out[k])
    assert (out["soft_det"] > 0).any()


@pytest.mark.parametrize(
    "masses,leptonic,i_fks,j_fks",
    [
        ([0.0, 0.0, 0.0], True, 5, 2),
        ([173.0, 173.0], True, 4, 3),
        ([0.0, 0.0], False, 4, 0),
        ([173.0, 173.0], False, 4, 1),
    ],
    ids=["ee-final", "ee-massive", "pp-initial", "pp-ttx-initial"],
)
def test_event_and_born_as_the_mappings(masses, leptonic, i_fks, j_fks):
    """The event is that of FKSMapping and the Born that of the Born mapping,
    the soft counter-event is the Born with a zero i."""
    batch = 1000
    born = ms.PhaseSpaceMapping(
        [0.0, 0.0] + masses, 1000.0, leptonic=leptonic, mode="rambo"
    )
    mapping = ms.FKSMapping(born, i_fks, j_fks)
    r = np.random.default_rng(3).random((batch, mapping.random_dim()))
    names = list(mapping.counter_event_types().keys())
    runtime = ms.FunctionRuntime(
        ms.FKSKinematics(mapping).function(), default_context()
    )
    out = dict(zip(names, [to_numpy(t) for t in runtime.call([to_torch(r)])]))
    momenta, x1, x2, weight = (to_numpy(t) for t in mapping.map_forward([to_torch(r)]))
    np.testing.assert_array_equal(out["momenta"], momenta)
    np.testing.assert_array_equal(out["weight"], weight)
    born_momenta, *_, born_weight = (
        to_numpy(t) for t in born.map_forward([to_torch(r[:, :-3])])
    )
    np.testing.assert_array_equal(out["born_momenta"], born_momenta)
    np.testing.assert_array_equal(out["born_weight"], born_weight)
    soft = out["soft_momenta"]
    np.testing.assert_allclose(soft[:, i_fks] / 1000.0, 0.0, atol=1e-13)
    np.testing.assert_allclose(
        np.delete(soft, i_fks, axis=1) / 1000.0, born_momenta / 1000.0, atol=1e-12
    )
    # it exists except for the second solution of a massive emitter, which
    # needs y < 0
    assert (out["soft_det"] >= 0).all()
    assert (out["soft_det"][out["y"] >= 0] > 0).all()
    # the weight of the event is xi times its Jacobian, which is that of the
    # soft counter-event at xi = 0
    assert (out["weight"] > 0).all()
