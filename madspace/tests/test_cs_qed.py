"""The operators of the QED corrections of the Catani-Seymour dipoles,
CSInsertion and CSCollinear with qed: the charges in place of the colors,
Q_f^2 in place of the Casimir of a charged particle, and the photon, which
has no charge but splits into the light fermions, with the weights of its
spectators (Dittmaier, arXiv:hep-ph/9904440; Dittmaier, Kabelschacht,
Kasprzik, arXiv:0802.1405).

A Born with two charged particles is the QCD Born of a quark pair of the
same kinematics with C_F in place of Q_f^2 (e+ e- > Z H against d d~ > Z H,
Z Z > W+ W- against Z Z > t t~ of the mass of the W). The terms of a
photon follow from those of the same Born with a neutral particle in its
place, and the K and P operators of a photon of the beam in closed form.
"""

import math

import numpy as np
import pytest
from test_cs_collinear import born_kinematics
from test_cs_insertion import born_points, evaluate

import madspace as ms

ALPHA = 1 / 128.0
A = ALPHA / (2 * math.pi)
CF, TR = 4.0 / 3.0, 0.5
N_F, N_L = 5.0, 3.0
# sum_f N_c Q_f^2 over the light fermions
PAIRS = 3 * (2 * 4 / 9 + 3 / 9) + N_L
GAMMA_PHOTON, K_PHOTON = -2 / 3 * PAIRS, -10 / 9 * PAIRS


def qed_insertion(pdgs, masses, momenta, born, mu, **parameters):
    generator = ms.CSInsertion(
        pdgs,
        masses,
        born_orders=born.shape[1],
        qed=True,
        n_f=N_F,
        n_l=N_L,
        **parameters,
    )
    batch = len(momenta)
    out = evaluate(generator, momenta, born, np.full(batch, ALPHA), np.full(batch, mu))
    return out["insertion"], out["poles"]


def qcd_insertion(pdgs, masses, momenta, colors, mu, **parameters):
    generator = ms.CSInsertion(
        pdgs, masses, born_orders=colors.shape[1], n_f=N_F, **parameters
    )
    batch = len(momenta)
    out = evaluate(
        generator, momenta, colors, np.full(batch, ALPHA), np.full(batch, mu)
    )
    return out["insertion"], out["poles"]


def pair(n, j, k, value, batch):
    out = np.zeros((batch, 1, n, n))
    out[:, 0, j, k] = out[:, 0, k, j] = value
    return out


def decay_points(masses, batch, seed):
    """Two-body decays at rest of the first particle in random directions."""
    m, m1, m2 = masses
    p = math.sqrt((m**2 - (m1 + m2) ** 2) * (m**2 - (m1 - m2) ** 2)) / (2 * m)
    rng = np.random.default_rng(seed)
    cos = rng.uniform(-1, 1, batch)
    phi = rng.uniform(0, 2 * math.pi, batch)
    sin = np.sqrt(1 - cos**2)
    direction = np.stack([sin * np.cos(phi), sin * np.sin(phi), cos], 1)
    out = np.zeros((batch, 3, 4))
    out[:, 0, 0] = m
    out[:, 1, 0] = math.sqrt(p**2 + m1**2)
    out[:, 1, 1:] = p * direction
    out[:, 2, 0] = math.sqrt(p**2 + m2**2)
    out[:, 2, 1:] = -p * direction
    return out


CASES = {
    # e+ e- > Z H against d d~ > Z H
    "massless": ([11, -11, 23, 25], [1, -1, 23, 25], [0.0, 0.0, 91.0, 125.0], 2),
    # W+ > e+ nu with the masses of t > b W+, the decaying particle as the
    # spectator of the massive positron
    "decay": ([24, -11, 12], [6, 5, 24], [173.0, 4.7, 80.0], 1),
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_like_qcd(case):
    """Two charged particles of opposite charges flowing out are a quark pair
    with C_F in place of Q^2: I of QED is that of QCD over C_F, its poles
    too, for massless leptons and for the decay of a W into a massive
    positron (the dipole with the decaying particle as the spectator)."""
    qed, qcd, masses, incoming = case
    batch = 6
    if incoming == 1:
        momenta = decay_points(masses, batch, seed=3)
    else:
        momenta = born_points(masses, batch, seed=3, leptonic=True)
    born = np.random.default_rng(4).uniform(0.5, 2.0, (batch, 1))
    colors = pair(len(qed), 0, 1, -CF, batch) * born[:, :, None, None]
    for alpha_ff in (1.0, 0.3):
        options = dict(alpha_ff=alpha_ff, incoming=incoming)
        ours = qed_insertion(qed, masses, momenta, born, 150.0, **options)
        theirs = qcd_insertion(qcd, masses, momenta, colors, 150.0, **options)
        for a, b in zip(ours, theirs):
            np.testing.assert_allclose(a, b / CF, rtol=1e-13, atol=1e-15)


def test_photon():
    """A final-state photon of u u~ > a Z adds to I its splitting into the
    light fermions with the spectators u and u~ at weight 1/2 each:
    (alpha / 2 pi) B (gamma_a / eps + sum_k w_k [gamma_a log(mu^2 / s_ak) +
    gamma_a + K_a]), against the same Born with a neutral particle in its
    place."""
    masses = [0.0] * 4
    momenta = born_points(masses, 5, seed=7, leptonic=True)
    born = np.random.default_rng(8).uniform(0.5, 2.0, (5, 1))
    mu = 120.0
    with_photon = qed_insertion([2, -2, 22, 23], masses, momenta, born, mu)
    without = qed_insertion([2, -2, 23, 23], masses, momenta, born, mu)
    s = [
        2
        * (
            momenta[:, 2, 0] * momenta[:, k, 0]
            - (momenta[:, 2, 1:] * momenta[:, k, 1:]).sum(-1)
        )
        for k in (0, 1)
    ]
    finite = (
        A
        * born[:, 0]
        * sum(
            0.5 * (GAMMA_PHOTON * np.log(mu**2 / sk) + GAMMA_PHOTON + K_PHOTON)
            for sk in s
        )
    )
    np.testing.assert_allclose(
        with_photon[0][:, 0] - without[0][:, 0], finite, rtol=1e-12, atol=1e-16
    )
    poles = with_photon[1] - without[1]
    np.testing.assert_allclose(poles[:, 0, 0], 0.0, atol=1e-16)
    np.testing.assert_allclose(
        poles[:, 0, 1], A * GAMMA_PHOTON * born[:, 0], rtol=1e-13
    )


def collinear_weights(pdgs, masses, momenta, born, colors, x, r, mu_f, qed):
    batch = len(r)
    generator = ms.CSCollinear(
        pdgs, masses, born_orders=1, n_f=N_F, qed=qed, n_l=N_L if qed else 0.0
    )
    tile = lambda v: np.full(batch, float(v))
    inputs = [np.broadcast_to(momenta, (batch, *momenta.shape)).copy(), born]
    if not qed:
        inputs.append(colors)
    inputs += [tile(ALPHA), tile(mu_f), tile(mu_f), tile(x[0]), tile(x[1]), r]
    out = evaluate(generator, *inputs)
    return out["z"], out["weights"]


def test_collinear_like_qcd():
    """K and P of an incoming u u~ pair in QED are those of QCD with Q_u^2
    in place of C_F for u -> u and N_c Q_u^2 in place of T_R for the photon
    that splits into the u (and g -> u in QCD)."""
    masses = [0.0, 0.0, 91.0, 125.0]
    momenta, x1, x2 = born_kinematics(masses, seed=5)
    r = np.random.default_rng(6).random(8)
    born = np.full((8, 1), 1.3)
    colors = pair(4, 0, 1, -CF * 1.3, 8)
    for pdgs in ([2, -2, 23, 25], [-1, 1, 23, 25]):
        q2 = (2 / 3 if abs(pdgs[0]) == 2 else 1 / 3) ** 2
        _, ours = collinear_weights(
            pdgs, masses, momenta, born, colors, (x1, x2), r, 90.0, True
        )
        _, theirs = collinear_weights(
            pdgs, masses, momenta, born, colors, (x1, x2), r, 90.0, False
        )
        np.testing.assert_allclose(ours[..., 0], theirs[..., 0] * q2 / CF, rtol=1e-12)
        np.testing.assert_allclose(
            ours[..., 1], theirs[..., 1] * 3 * q2 / TR, rtol=1e-12
        )


def test_collinear_photon():
    """K and P of the incoming photons of a a > mu+ mu-: from the fermions
    of the beam per unit Q_f^2, (alpha / 2 pi) (1 - x_i) [c_P P(z) + B (P(z)
    log((1 - z) / z) + z)] with P(z) = (1 + (1 - z)^2) / z and c_P = -B sum_j
    w_j log(mu_F^2 / s_aj) over the muons, none from the photon at z, and at
    the end point c_P gamma_a + 16/9 sum_f N_c Q_f^2 B."""
    masses = [0.0] * 4
    momenta, x1, x2 = born_kinematics(masses, seed=9)
    r = np.random.default_rng(10).random(6)
    born = np.full((6, 1), 0.7)
    mu_f = 80.0
    z, weights = collinear_weights(
        [22, 22, 13, -13], masses, momenta, born, None, (x1, x2), r, mu_f, True
    )
    for leg, eta in ((0, x1), (1, x2)):
        zl = z[:, leg]
        s = [
            2 * (momenta[leg, 0] * momenta[j, 0] - momenta[leg, 1:] @ momenta[j, 1:])
            for j in (2, 3)
        ]
        c_p = -0.7 * sum(0.5 * np.log(mu_f**2 / sj) for sj in s)
        p = (1 + (1 - zl) ** 2) / zl
        expected = A * (1 - eta) * (c_p * p + 0.7 * (p * np.log((1 - zl) / zl) + zl))
        np.testing.assert_allclose(weights[:, 0, leg, 1], expected, rtol=1e-12)
        np.testing.assert_allclose(weights[:, 0, leg, 0], 0.0, atol=1e-15)
        end = A * (c_p * GAMMA_PHOTON + 16 / 9 * PAIRS * 0.7)
        np.testing.assert_allclose(weights[:, 0, leg, 2], end, rtol=1e-12)


def test_invalid():
    with pytest.raises(ValueError, match="charged particle as its spectator"):
        ms.CSInsertion([23, 23, 22, 25], [91.0, 91.0, 0.0, 125.0], qed=True)
    with pytest.raises(ValueError, match="n_l"):
        ms.CSInsertion([11, -11, 23, 25], [0.0, 0.0, 91.0, 125.0], qed=True, n_l=4.0)
