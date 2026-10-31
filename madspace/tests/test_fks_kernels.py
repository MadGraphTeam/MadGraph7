"""FKS radiation kernels against genps_fks.f of MadGraph.

The reference points in test_data/fks come from fks_reference/ (see
test_fks_reference.py). For a massless final-state emitter, the Born of a point
and the last three entries of its unit-cube vector must give the event and the
counter-events of genps_fks.f, and the radiation Jacobian of genps_fks.f up to
two conventions of the kernels:

* det_xi includes xi_max (genps_fks.f integrates over xi_hat) and the
  Jacobian of the map is xi det_xi (genps_fks.f leaves out the factor xi),
* fks_xi_y_sample has the exact Jacobian, with the factors 1 - stiny and
  1 - ctiny that genps_fks.f drops.

The momenta agree to 1e-12 of sqrt(s) rather than to machine precision because
genps_fks.f computes the angle of i to i + j with the law of cosines, which
loses up to half the digits of a small angle. The kernel uses the exact sine
and cosine of that angle.
"""

import glob
import json
import math
import os

import numpy as np
import pytest
import torch
from madspace_testing import default_context, generator

import madspace as ms

STINY, CTINY = 1e-6, 5e-7
DATA = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_data", "fks", "*.json"
        )
    )
)
F, V = ms.batch_float, ms.batch_four_vec


def run(build, in_types, out_types, inputs):
    fb = ms.FunctionBuilder(
        ms.NamedTypes([(f"in{i}", t) for i, t in enumerate(in_types)]),
        ms.NamedTypes([(f"out{i}", t) for i, t in enumerate(out_types)]),
    )
    outputs = build(fb, *[fb.input(i) for i in range(len(in_types))])
    for i, value in enumerate(outputs):
        fb.output(i, value)
    runtime = ms.FunctionRuntime(fb.function(), default_context())
    tensors = [torch.as_tensor(np.asarray(x), dtype=torch.float64) for x in inputs]
    return [torch.from_dlpack(t).cpu().numpy() for t in runtime.call(tensors)]


def xi_y_sample(r_xi, r_y):
    return run(
        lambda fb, a, b: fb.fks_xi_y_sample(a, b), [F, F], [F, F, F], [r_xi, r_y]
    )


def xi_y_sample_inverse(xi_hat, y):
    return run(
        lambda fb, a, b: fb.fks_xi_y_sample_inverse(a, b),
        [F, F],
        [F, F, F],
        [xi_hat, y],
    )


def radiate(xi_hat, y, r_phi, sqrt_s, p_jbar):
    return run(
        lambda fb, *args: fb.fks_final_massless_radiate(*args),
        [F, F, F, F, V],
        [V, V, F, F, F],
        [xi_hat, y, r_phi, sqrt_s, p_jbar],
    )


def cluster(p_i, p_j, sqrt_s):
    return run(
        lambda fb, *args: fb.fks_final_massless_cluster(*args),
        [V, V, F],
        [V, F, F, F, F, F, F],
        [p_i, p_j, sqrt_s],
    )


def recoil_boost(p, k, sqrt_s, inverse=False):
    n = p.shape[1]
    op = "fks_recoil_boost_inverse" if inverse else "fks_recoil_boost"
    return run(
        lambda fb, *args: [getattr(fb, op)(*args)],
        [ms.batch_four_vec_array(n), V, F],
        [ms.batch_four_vec_array(n)],
        [p, k, sqrt_s],
    )[0]


def real_from_born(p_born, i_fks, j_fks, nin, xi_hat, y, r_phi, sqrt_s):
    """Born momenta (batch, n, 4) and FKS variables -> real momenta (batch, n+1, 4).
    i_fks, j_fks are the 0-based positions in the real process, j a massless
    final-state parton."""
    jb = j_fks if j_fks < i_fks else j_fks - 1
    p_i, p_j, xi, xi_max, det_xi = radiate(xi_hat, y, r_phi, sqrt_s, p_born[:, jb])
    spect = [b for b in range(nin, p_born.shape[1]) if b != jb]
    real_born = p_born.copy()
    if spect:
        real_born[:, spect] = recoil_boost(p_born[:, spect], p_i + p_j, sqrt_s)
    real_born[:, jb] = p_j
    real = np.insert(real_born, i_fks, 0.0, axis=1)
    real[:, i_fks] = p_i
    return real, xi, xi_max, det_xi


def flux_factor(shat, nexternal, nin=2):
    """compute_flux of genps_fks.f for massless incoming partons."""
    pi2 = 2 * math.pi
    return 1 / (2 * shat) / pi2 ** (3 * (nexternal - nin) - 4) * pi2**3


def massless_final_points():
    cases = []
    for path in DATA:
        with open(path) as f:
            data = json.load(f)
        nin = data["nincoming"]
        for conf in data["configs"]:
            j = conf["j_fks"]
            if j <= nin or conf["masses"][j - 1] > 0:
                continue
            points = [p for p in conf["points"] if p["jac"] > 0 and "rad_jac" in p]
            if points:
                cases.append(
                    (os.path.basename(path)[:-5], conf["nfks"], data, conf, points)
                )
    return cases


CASES = massless_final_points()


def test_reference_has_massless_final_emitters():
    assert len(CASES) > 0


@pytest.mark.parametrize(
    "name,nfks,data,conf,points", CASES, ids=[f"{c[0]}-fks{c[1]}" for c in CASES]
)
def test_final_massless_matches_genps_fks(name, nfks, data, conf, points):
    nin, nex = data["nincoming"], data["nexternal"]
    i_fks, j_fks = conf["i_fks"] - 1, conf["j_fks"] - 1
    p_born = np.array([p["p_born"] for p in points])
    sqrt_s = np.sqrt([p["shat_born"] for p in points])
    r = np.array([p["x"][-3:] for p in points])
    scale = sqrt_s[:, None, None]

    xi_hat, y, det_sample = xi_y_sample(r[:, 0], r[:, 1])
    real, xi, xi_max, det_xi = real_from_born(
        p_born, i_fks, j_fks, nin, xi_hat, y, r[:, 2], sqrt_s
    )
    np.testing.assert_allclose(xi, [p["xi_ev"] for p in points], rtol=1e-13)
    np.testing.assert_allclose(xi_max, [p["xiimax_ev"] for p in points], rtol=1e-13)
    np.testing.assert_allclose(y, [p["y_ev"] for p in points], rtol=0, atol=1e-15)
    p_ev = np.array([p["p_ev"] for p in points])
    np.testing.assert_allclose(real / scale, p_ev / scale, rtol=0, atol=1e-12)

    fortran = np.array([p["rad_jac"] for p in points]) / flux_factor(
        np.array([p["shat_ev"] for p in points]), nex
    )
    ours = det_xi / xi_max * det_sample / ((1 - STINY) * (1 - CTINY))
    np.testing.assert_allclose(ours, fortran, rtol=1e-12)

    # counter-events: position k + 2 for k = 0 (soft), 1 (collinear), 2 (both)
    for pos, xh, yy in [
        (2, np.zeros_like(xi_hat), y),
        (3, xi_hat, np.ones_like(y)),
        (4, np.zeros_like(xi_hat), np.ones_like(y)),
    ]:
        valid = np.array([p["rad_jac_cnt"][pos] > 0 for p in points])
        if not valid.any():
            continue
        cnt, _, _, det_cnt = real_from_born(
            p_born, i_fks, j_fks, nin, xh, yy, r[:, 2], sqrt_s
        )
        p_cnt = np.array([p["p_cnt"][pos] for p in points])
        np.testing.assert_allclose(
            (cnt / scale)[valid], (p_cnt / scale)[valid], rtol=0, atol=1e-12
        )
        fortran_cnt = np.array([p["rad_jac_cnt"][pos] for p in points]) / flux_factor(
            np.array([p["shat_cnt"][pos] for p in points]), nex
        )
        ours_cnt = det_cnt / xi_max * det_sample / ((1 - STINY) * (1 - CTINY))
        np.testing.assert_allclose(ours_cnt[valid], fortran_cnt[valid], rtol=1e-12)

    # and back: cluster the event onto the Born and the unit-cube numbers
    p_jbar, xi_hat2, y2, r_phi2, _, _, det_inv = cluster(
        p_ev[:, i_fks], p_ev[:, j_fks], sqrt_s
    )
    r_xi2, r_y2, _ = xi_y_sample_inverse(xi_hat2, y2)
    np.testing.assert_allclose(
        np.stack([r_xi2, r_y2, r_phi2], 1), r, rtol=0, atol=1e-12
    )
    jb = j_fks if j_fks < i_fks else j_fks - 1
    np.testing.assert_allclose(
        p_jbar / sqrt_s[:, None], p_born[:, jb] / sqrt_s[:, None], atol=1e-14
    )
    np.testing.assert_allclose(det_inv * det_xi, 1.0, rtol=1e-12)


def random_born(n_out, batch, sqrt_s=1000.0, seed=1):
    mapping = ms.PhaseSpaceMapping(
        [0.0] * (n_out + 2), sqrt_s, mode="rambo", leptonic=True
    )
    r = torch.rand(
        (batch, mapping.random_dim()), generator=generator(seed), device="cpu"
    )
    out = mapping.map_forward([r.to(torch.get_default_device())])
    momenta, det = out[0], out[-1]
    return torch.as_tensor(momenta).cpu().numpy(), torch.as_tensor(det).cpu().numpy()


def test_xi_y_sample_round_trip():
    rng = np.random.default_rng(2)
    r = rng.random((1000, 2))
    xi_hat, y, det = xi_y_sample(r[:, 0], r[:, 1])
    assert (xi_hat >= STINY).all() and (xi_hat <= 1).all()
    assert (y >= -1).all() and (y <= 1 - 2 * CTINY).all()
    r_xi, r_y, det_inv = xi_y_sample_inverse(xi_hat, y)
    np.testing.assert_allclose(np.stack([r_xi, r_y], 1), r, rtol=0, atol=1e-13)
    # y close to 1 carries 1 - y only to a relative precision of eps / (1 - y),
    # which limits r_y and det_inv for small r_y
    np.testing.assert_allclose(det * det_inv, 1.0, rtol=1e-10)
    eps = 1e-6
    xp, yp, _ = xi_y_sample(r[:, 0] + eps, r[:, 1] + eps)
    xm, ym, _ = xi_y_sample(r[:, 0] - eps, r[:, 1] - eps)
    fd = (xp - xm) / (2 * eps) * (ym - yp) / (2 * eps)
    np.testing.assert_allclose(det, fd, rtol=1e-8)


def test_final_massless_round_trip():
    batch = 2000
    p_born, _ = random_born(3, batch)
    sqrt_s = np.full(batch, 1000.0)
    rng = np.random.default_rng(3)
    r = rng.random((batch, 3))
    xi_hat, y, _ = xi_y_sample(r[:, 0], r[:, 1])
    real, xi, xi_max, det_xi = real_from_born(
        p_born, 5, 4, 2, xi_hat, y, r[:, 2], sqrt_s
    )

    # momentum conservation and mass shells
    scale = 1000.0
    np.testing.assert_allclose(
        real[:, 2:].sum(1) / scale, real[:, :2].sum(1) / scale, atol=1e-13
    )
    m2 = real[..., 0] ** 2 - (real[..., 1:] ** 2).sum(-1)
    np.testing.assert_allclose(m2 / scale**2, 0.0, atol=1e-12)

    p_jbar, xi_hat2, y2, r_phi2, xi2, xi_max2, det_inv = cluster(
        real[:, 5], real[:, 4], sqrt_s
    )
    spect = recoil_boost(real[:, 2:4], real[:, 4] + real[:, 5], sqrt_s, inverse=True)
    np.testing.assert_allclose(p_jbar / scale, p_born[:, 4] / scale, atol=1e-12)
    np.testing.assert_allclose(spect / scale, p_born[:, 2:4] / scale, atol=1e-12)
    np.testing.assert_allclose(xi_hat2, xi_hat, rtol=1e-10)
    np.testing.assert_allclose(y2, y, rtol=0, atol=1e-10)
    dphi = (r_phi2 - r[:, 2] + 0.5) % 1.0 - 0.5
    np.testing.assert_allclose(dphi, 0.0, atol=1e-10)
    np.testing.assert_allclose(det_xi * det_inv, 1.0, rtol=1e-10)


def test_final_massless_limits():
    batch = 50
    p_born, _ = random_born(3, batch, seed=4)
    sqrt_s = np.full(batch, 1000.0)
    rng = np.random.default_rng(5)
    xi_hat, y, r_phi = rng.random(batch), 2 * rng.random(batch) - 1, rng.random(batch)

    # soft: i vanishes and the Born is recovered
    soft, xi, _, det_soft = real_from_born(
        p_born, 5, 4, 2, np.zeros(batch), y, r_phi, sqrt_s
    )
    assert (xi == 0).all() and np.isfinite(det_soft).all()
    np.testing.assert_allclose(soft[:, 5], 0.0, atol=1e-12)
    np.testing.assert_allclose(soft[:, :5], p_born, rtol=0, atol=1e-10)

    # collinear: i parallel to j, i + j along the Born emitter, same recoil
    coll, *_ = real_from_born(p_born, 5, 4, 2, xi_hat, np.ones(batch), r_phi, sqrt_s)
    k = coll[:, 4] + coll[:, 5]
    np.testing.assert_allclose(k / 1000.0, p_born[:, 4] / 1000.0, atol=1e-10)
    np.testing.assert_allclose(coll[:, 2:4], p_born[:, 2:4], rtol=0, atol=1e-9)
    cross = np.cross(coll[:, 4, 1:], coll[:, 5, 1:])
    np.testing.assert_allclose(cross / 1000.0**2, 0.0, atol=1e-10)


@pytest.mark.parametrize("n_out", [2, 3])
def test_final_massless_volume(n_out):
    """Born (n_out-body RAMBO) times the FKS emission off its last particle covers
    the (n_out + 1)-body phase space once."""
    batch = 200000
    p_born, det_born = random_born(n_out, batch, seed=6 + n_out)
    sqrt_s = np.full(batch, 1000.0)
    rng = np.random.default_rng(7)
    r = rng.random((batch, 3))
    xi_hat, y, det_sample = xi_y_sample(r[:, 0], r[:, 1])
    _, _, xi, _, det_xi = radiate(xi_hat, y, r[:, 2], sqrt_s, p_born[:, -1])
    weight = det_born * xi * det_xi * det_sample

    _, det_real = random_born(n_out + 1, batch, seed=8 + n_out)
    mean, err = weight.mean(), weight.std() / math.sqrt(batch)
    mean_ref, err_ref = det_real.mean(), det_real.std() / math.sqrt(batch)
    assert abs(mean - mean_ref) < 5 * math.hypot(err, err_ref)


# Emission off an incoming parton. The reference is the FKS map of genps_fks.f
# without (the rad_ entries) and with (cut_) the lower bound on the partonic
# energy.


def initial_radiate(
    xi_hat, y_hat, r_phi, sqrt_s, x1_born, x2_born, beam_sign, s_hat_min=0.0
):
    return run(
        lambda fb, *args: fb.fks_initial_radiate(*args, s_hat_min, beam_sign),
        [F, F, F, F, F, F],
        [V, V, V, F, F, F, F, F, F, F],
        [xi_hat, y_hat, r_phi, sqrt_s, x1_born, x2_born],
    )


def initial_recoil_boost(p, xi, y, r_phi, beam_sign):
    n = p.shape[1]
    return run(
        lambda fb, *args: [fb.fks_initial_recoil_boost(*args, beam_sign)],
        [ms.batch_four_vec_array(n), F, F, F],
        [ms.batch_four_vec_array(n)],
        [p, xi, y, r_phi],
    )[0]


def initial_real_from_born(
    p_born, i_fks, beam_sign, xi_hat, y_hat, r_phi, x_born, s_hat_min=0.0
):
    """Born momenta (batch, n, 4) in the Born c.m. frame -> real momenta, momentum
    fractions, xi, y, the width xi_max - xi_min of its range and det_xi."""
    sqrt_s = np.sqrt(
        (p_born[:, 0] + p_born[:, 1])[:, 0] ** 2
        - ((p_born[:, 0] + p_born[:, 1])[:, 1:] ** 2).sum(1)
    )
    p1, p2, p_i, x1, x2, xi, y, xi_min, xi_max, det_xi = initial_radiate(
        xi_hat, y_hat, r_phi, sqrt_s, x_born[0], x_born[1], beam_sign, s_hat_min
    )
    final = initial_recoil_boost(p_born[:, 2:], xi, y, r_phi, beam_sign)
    real = np.concatenate([p1[:, None], p2[:, None], final], axis=1)
    real = np.insert(real, i_fks, 0.0, axis=1)
    real[:, i_fks] = p_i
    return real, (x1, x2), xi, y, xi_max - xi_min, det_xi


def initial_points():
    cases = []
    for path in DATA:
        with open(path) as f:
            data = json.load(f)
        for conf in data["configs"]:
            if conf["j_fks"] > data["nincoming"]:
                continue
            points = [p for p in conf["points"] if p["jac"] > 0 and "rad_jac" in p]
            if points:
                name = os.path.basename(path)[:-5]
                cases.append(
                    pytest.param(data, conf, points, id=f"{name}-fks{conf['nfks']}")
                )
    return cases


@pytest.mark.parametrize("prefix", ["rad", "cut"], ids=["uncut", "bounded"])
@pytest.mark.parametrize("data,conf,points", initial_points())
def test_initial_matches_genps_fks(data, conf, points, prefix):
    """Event and counter-events of an incoming emitter, with their momentum
    fractions and radiation Jacobians, without (rad_) and with (cut_) the lower
    bound tau_lower_bound s_lab on the partonic energy of the real emission.
    genps_fks.f leaves the width xi_max - xi_min of the range of xi out of its
    Jacobian (see the top of the file)."""
    nex = data["nexternal"]
    i_fks = conf["i_fks"] - 1
    beam_sign = 1.0 if conf["j_fks"] == 1 else -1.0
    s_lab = 4 * data["ebeam"][0] * data["ebeam"][1]

    def field(key):
        if key.startswith("rad_"):
            key = prefix + key[3:]
        return np.array([p[key] for p in points])

    # s_hat_min is one per configuration (the same for all its points)
    taus = {p["tau_lower_bound"] for p in points}
    assert len(taus) == 1
    s_hat_min = taus.pop() * s_lab if prefix == "cut" else 0.0

    p_born = field("p_born")
    x_born = tuple(field("xbjrk_born").T)
    r = field("x")[:, -3:]
    xi_hat, y, det_sample = xi_y_sample(r[:, 0], r[:, 1])
    det_fortran_units = det_sample / ((1 - STINY) * (1 - CTINY))

    # event and counter-events: position k + 2 for k = 0 (soft), 1 (collinear),
    # 2 (both); for an incoming emitter only y = 1 is a collinear limit
    for pos, xh, yy in [
        (None, xi_hat, y),
        (2, np.zeros_like(xi_hat), y),
        (3, xi_hat, np.ones_like(y)),
        (4, np.zeros_like(xi_hat), np.ones_like(y)),
    ]:
        if pos is None:
            valid = np.ones(len(points), dtype=bool)
            p_ref, x_ref = field("rad_p_ev"), field("rad_xbjrk_ev")
            jac_ref, shat_ref = field("rad_jac"), field("rad_shat_ev")
        else:
            valid = field("rad_jac_cnt")[:, pos] > 0
            p_ref = field("rad_p_cnt")[:, pos]
            x_ref = field("rad_xbjrk_cnt")[:, pos]
            jac_ref = field("rad_jac_cnt")[:, pos]
            shat_ref = field("rad_shat_cnt")[:, pos]
        if not valid.any():
            continue
        real, x, xi, _, width, det_xi = initial_real_from_born(
            p_born, i_fks, beam_sign, xh, yy, r[:, 2], x_born, s_hat_min
        )
        scale = p_ref[:, :, 0].max(1)[:, None, None]
        np.testing.assert_allclose(
            (real / scale)[valid], (p_ref / scale)[valid], rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(np.stack(x, 1)[valid], x_ref[valid], rtol=1e-13)
        fortran = jac_ref / flux_factor(shat_ref, nex)
        ours = det_xi / width * det_fortran_units
        np.testing.assert_allclose(ours[valid], fortran[valid], rtol=1e-12)


def test_reference_has_initial_emitters():
    assert len(initial_points()) > 0


def test_initial_limits():
    """Soft: the Born with a zero i, same momentum fractions. Collinear: i along
    the beam of j, x of that beam rescaled by 1 / (1 - xi), the final state of the
    Born and the other beam unchanged."""
    batch = 50
    p_born, _ = random_born(3, batch, seed=60)
    rng = np.random.default_rng(61)
    u = rng.random(batch)
    x_born = (0.01**u, 0.01 ** (1 - u))
    xi_hat, y, r_phi = rng.random(batch), 2 * rng.random(batch) - 1, rng.random(batch)
    for beam_sign, beam in [(1.0, 0), (-1.0, 1)]:
        soft, x, xi, _, _, det_soft = initial_real_from_born(
            p_born, 4, beam_sign, np.zeros(batch), y, r_phi, x_born
        )
        assert (xi == 0).all() and np.isfinite(det_soft).all()
        np.testing.assert_allclose(soft[:, 4], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.delete(soft, 4, axis=1), p_born, atol=1e-10)
        np.testing.assert_allclose(np.stack(x, 1), np.stack(x_born, 1), rtol=1e-14)

        coll, x, xi, _, _, _ = initial_real_from_born(
            p_born, 4, beam_sign, xi_hat, np.ones(batch), r_phi, x_born
        )
        np.testing.assert_allclose(coll[:, 4, 1:3], 0.0, atol=1e-10)
        assert (np.sign(coll[:, 4, 3]) == beam_sign).all()
        np.testing.assert_allclose(x[beam], x_born[beam] / (1 - xi), rtol=1e-13)
        np.testing.assert_allclose(x[1 - beam], x_born[1 - beam], rtol=1e-13)
        np.testing.assert_allclose(
            np.delete(coll, [0, 1, 4], axis=1), p_born[:, 2:], atol=1e-10
        )
