"""CSCollinear: the K and P operators of the Catani-Seymour dipoles as
weights of the luminosities of the beams at one point of their convolution.

The weights are checked against the K and P operators of CS and CDST as the
NCV code writes them, convolved with a test luminosity by quadrature, their
alpha terms against the integral of the dipoles of CSDipoles over what their
gates remove, as densities in x, and their dependence on the factorization
scales against the collinear counterterm P^{aa'} B.
"""

import math

import numpy as np
import pytest
import torch
from madspace_testing import default_context
from scipy import integrate

import madspace as ms

SQRT_S = 1000.0
ALPHA_S = 0.118
TOP = 173.0
CA, CF, TR = 3.0, 4.0 / 3.0, 0.5
A = ALPHA_S / (2 * math.pi)
METRIC = np.array([1.0, -1.0, -1.0, -1.0])


def to_torch(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


def to_numpy(x):
    return torch.from_dlpack(x).cpu().numpy()


def dot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def evaluate(generator, *inputs):
    runtime = ms.FunctionRuntime(generator.function(), default_context())
    out = runtime.call([to_torch(x) for x in inputs])
    return dict(zip(generator.return_types().keys(), [to_numpy(t) for t in out]))


def collinear(
    pdgs, masses, momenta, born, colors, x, r, mu_f=(100.0, 100.0), **parameters
):
    """z (batch, 2) and the weights (batch, b, 2, 3)."""
    batch = len(r)
    tile = lambda v: np.broadcast_to(np.asarray(v, dtype=float), (batch,)).copy()
    generator = ms.CSCollinear(pdgs, masses, born_orders=born.shape[1], **parameters)
    out = evaluate(
        generator,
        np.broadcast_to(momenta, (batch, *momenta.shape[-2:])).copy(),
        np.broadcast_to(born, (batch, born.shape[1])).copy(),
        np.broadcast_to(colors, (batch, *colors.shape[-3:])).copy(),
        tile(ALPHA_S),
        tile(mu_f[0]),
        tile(mu_f[1]),
        tile(x[0]),
        tile(x[1]),
        np.asarray(r, dtype=float),
    )
    return out["z"], out["weights"]


# --- the K and P operators of CS (eqs. 10.24-10.29) and CDST (eqs. 6.55-6.62)
# as the NCV code writes them: terms (class, kind, coefficient, function) of
# x and of the invariant s, convolved by quadrature


def p_regular(kind, x):
    return {
        "qq": lambda: -CF * (1 + x),
        "gg": lambda: 2 * CA * ((1 - x) / x - 1 + x * (1 - x)),
        "gq": lambda: TR * (x**2 + (1 - x) ** 2),
        "qg": lambda: CF * (1 + (1 - x) ** 2) / x,
    }[kind]()


def p_full(kind, x):
    om = 1 - x
    return {
        "qq": lambda: CF * (1 + x**2) / om,
        "gg": lambda: 2 * CA * (x / om + om / x + x * om),
        "gq": lambda: TR * (x**2 + om**2),
        "qg": lambda: CF * (1 + om**2) / x,
    }[kind]()


class Term:
    """A distribution in x: reg f(x), plus [f(x)]_+, pprod [1/(1-x)]_+ h(x),
    delta delta(1-x) h, with f and h of x and s (s / x inside the
    convolution, s at its end point)."""

    def __init__(self, kind, coef, fn):
        self.kind, self.coef, self.fn = kind, coef, fn

    def convolve(self, g, eta, s):
        f = self.fn
        if self.kind == "reg":
            value = integrate.quad(lambda x: f(x, s / x) * g(x), eta, 1, limit=200)[0]
        elif self.kind == "plus":
            inside = integrate.quad(
                lambda x: f(x, s / x) * g(x) - f(x, s) * g(1.0), eta, 1, limit=200
            )[0]
            below = integrate.quad(lambda x: f(x, s), 0, eta, limit=200)[0]
            value = inside - g(1.0) * below
        elif self.kind == "pprod":
            inside = integrate.quad(
                lambda x: (f(x, s / x) * g(x) - f(1.0, s) * g(1.0)) / (1 - x),
                eta,
                1,
                limit=200,
            )[0]
            value = inside + g(1.0) * f(1.0, s) * math.log(1 - eta)
        else:
            value = g(1.0) * f(1.0, s)
        return self.coef * value


def gamma_of(gluon, n_f=5.0):
    return 11 / 6 * CA - 2 / 3 * TR * n_f if gluon else 1.5 * CF


def massive_terms(kind, cc, t2, gamma_a, m):
    """The terms of a massive final-state quark (CDST 6.55-6.62)."""
    m2 = m * m
    out = []
    if kind in ("qq", "gg"):
        out += [
            Term("plus", -2 * cc, lambda x, s: math.log(1 - x) / (1 - x)),
            Term("reg", -cc, lambda x, s: -2 * math.log(2 - x) / (1 - x)),
            Term(
                "plus",
                -cc,
                lambda x, s: (1 - x) / (2 * (1 - x + m2 / s) ** 2)
                - 2 / (1 - x) * (1 + math.log(1 - x + m2 / s)),
            ),
            Term("pprod", -2 * cc, lambda x, s: math.log(2 + m2 / s - x)),
            Term(
                "pprod",
                -2 * cc,
                lambda x, s: math.log((2 - x) * s / ((2 - x) * s + m2)),
            ),
            Term(
                "delta",
                -cc,
                lambda x, s: -1.5
                + m2 / s * math.log(m2 / (s + m2))
                + 0.5 * m2 / (s + m2),
            ),
            Term(
                "delta",
                -cc / t2 * gamma_a,
                lambda x, s: math.log((s - 2 * m * math.sqrt(s + m2) + 2 * m2) / s)
                + 2 * m / (math.sqrt(s + m2) + m),
            ),
        ]
    if kind in ("qg", "gg"):
        factor = CA / CF if kind == "gg" else 1.0
        out.append(
            Term(
                "reg",
                -cc * factor,
                lambda x, s: 2
                * CF
                / CA
                * m2
                / (x * s)
                * math.log(m2 / ((1 - x) * s + m2)),
            )
        )
    out.append(
        Term(
            "reg",
            -cc / t2,
            lambda x, s: p_regular(kind, x)
            * math.log((1 - x) * s / ((1 - x) * s + m2)),
        )
    )
    return out


def distribution(coef, reg=None, c1=0.0, c_log=0.0, c_ratio=0.0, delta=0.0):
    """reg(x) + c1 [1/(1-x)]_+ + c_log [log(1-x)/(1-x)]_+ + c_ratio
    [log((1-x)/x)/(1-x)]_+ + delta delta(1-x), without s."""
    out = [Term("reg", coef, lambda x, s: reg(x))] if reg is not None else []
    if c1:
        out.append(Term("plus", coef * c1, lambda x, s: 1 / (1 - x)))
    if c_log:
        out.append(Term("plus", coef * c_log, lambda x, s: math.log(1 - x) / (1 - x)))
    if c_ratio:
        out.append(
            Term("plus", coef * c_ratio, lambda x, s: math.log((1 - x) / x) / (1 - x))
        )
    if delta:
        out.append(Term("delta", coef * delta, lambda x, s: 1.0))
    return out


def removed_fi(gluon, x, s, m, alpha, n_f=5.0):
    om = 1 - x
    if om <= alpha:
        return 0.0
    if gluon:
        return (2 * CA * math.log((1 + om) / om) - gamma_of(True, n_f)) / om
    mu2 = m * m / s
    zp = om / (om + mu2)
    return (
        CF
        / om
        * (2 * math.log((om + zp) / om) - 2 * zp + 0.5 * zp**2 - 2 * mu2 * zp / om)
    )


def removed_if(kind, x, s, m, alpha):
    om = 1 - x
    mu2 = m * m / s
    zp = om / (om + mu2)
    if zp <= alpha:
        return 0.0
    lu = math.log(zp / alpha)
    e = math.log(zp * (om + alpha) / (alpha * (om + zp))) / om
    lm = mu2 * math.log((1 - alpha) * (om + mu2) / mu2) / x if m > 0 else 0.0
    return {
        "qq": lambda: CF * (2 * e - (1 + x) * lu),
        "gq": lambda: TR * (x**2 + om**2) * lu,
        "qg": lambda: CF * ((x + 2 * om / x) * lu - 2 * lm),
        "gg": lambda: 2 * CA * (e + (-1 + x * om + om / x) * lu - lm),
    }[kind]()


def kp_terms(pdgs, masses, momenta, born, colors, leg, mu_f, alpha, n_f=5.0):
    """The terms of K + P of the Born leg per class (same flavor, other
    flavor) with their invariants s, as kp_operator of the NCV code."""
    colored = [j for j, p in enumerate(pdgs) if abs(p) <= 6 or p == 21]
    t2 = {True: CA, False: CF}
    gluon = pdgs[leg] == 21
    t2_leg = t2[gluon]
    kinds = {"same": "gg" if gluon else "qq", "other": "qg" if gluon else "gq"}
    other_leg = 1 - leg
    s_of = lambda j: 2 * dot(momenta[leg], momenta[j])
    coef_p = (
        sum(colors[j, leg] * math.log(mu_f**2 / s_of(j)) for j in colored if j != leg)
        / t2_leg
    )
    terms = {"same": [], "other": []}
    for cls, kind in kinds.items():
        reg_p = lambda x, kind=kind: p_regular(kind, x)
        if kind == "qq":
            terms[cls] += [
                (t, 0.0) for t in distribution(coef_p, reg_p, c1=2 * CF, delta=1.5 * CF)
            ]
        elif kind == "gg":
            terms[cls] += [
                (t, 0.0)
                for t in distribution(
                    coef_p, reg_p, c1=2 * CA, delta=gamma_of(True, n_f)
                )
            ]
        else:
            terms[cls] += [(t, 0.0) for t in distribution(coef_p, reg_p)]
        lr = lambda x: math.log((1 - x) / x)
        kbar = {
            "qg": distribution(born, lambda x: p_regular("qg", x) * lr(x) + CF * x),
            "gq": distribution(
                born, lambda x: p_regular("gq", x) * lr(x) + 2 * TR * x * (1 - x)
            ),
            "qq": distribution(
                born,
                lambda x: CF * (-(1 + x) * lr(x) + (1 - x)),
                c_ratio=2 * CF,
                delta=-CF * (5 - math.pi**2),
            ),
            "gg": distribution(
                born,
                lambda x: p_regular("gg", x) * lr(x),
                c_ratio=2 * CA,
                delta=-((50 / 9 - math.pi**2) * CA - 16 / 9 * TR * n_f),
            ),
        }[kind]
        terms[cls] += [(t, 0.0) for t in kbar]
        same = cls == "same"
        for j in colored:
            if j < 2:
                continue
            cc, s, m = colors[j, leg], s_of(j), masses[j]
            cj = t2[pdgs[j] == 21]
            if m == 0 and same:
                terms[cls] += [
                    (t, s)
                    for t in distribution(
                        cc * gamma_of(pdgs[j] == 21, n_f) / cj, c1=1.0, delta=1.0
                    )
                ]
            elif m > 0:
                terms[cls] += [
                    (t, s)
                    for t in massive_terms(kind, cc, t2_leg, gamma_of(gluon, n_f), m)
                ]
            if alpha["fi"] < 1 and same:
                terms[cls].append(
                    (
                        Term(
                            "reg",
                            cc / cj,
                            lambda x, s, g=pdgs[j] == 21, m=m: removed_fi(
                                g, x, s, m, alpha["fi"], n_f
                            ),
                        ),
                        s,
                    )
                )
            if alpha["if"] < 1:
                terms[cls].append(
                    (
                        Term(
                            "reg",
                            cc / t2_leg,
                            lambda x, s, kind=kind, m=m: removed_if(
                                kind, x, s, m, alpha["if"]
                            ),
                        ),
                        s,
                    )
                )
        if other_leg in colored:
            coef = -colors[other_leg, leg] / t2_leg
            reg_k = lambda x, kind=kind: p_regular(kind, x) * math.log(1 - x)
            if kind in ("qq", "gg"):
                terms[cls] += [
                    (t, 0.0)
                    for t in distribution(
                        coef, reg_k, c_log=2 * t2_leg, delta=-t2_leg * math.pi**2 / 3
                    )
                ]
            else:
                terms[cls] += [(t, 0.0) for t in distribution(coef, reg_k)]
            if alpha["ii"] < 1:
                terms[cls].append(
                    (
                        Term(
                            "reg",
                            -coef,
                            lambda x, s, kind=kind: (
                                p_full(kind, x) * math.log((1 - x) / alpha["ii"])
                                if 1 - x > alpha["ii"]
                                else 0.0
                            ),
                        ),
                        0.0,
                    )
                )
    return terms


def luminosity(eta):
    """A luminosity g(x) = f(eta / x) / x of a test density f(y) = y^-0.3
    (1 - y)^3, zero below eta."""
    f = lambda y: y**-0.3 * (1 - y) ** 3
    return lambda x: f(eta / x) / x if x > eta else (f(eta) if x == 1.0 else 0.0)


def born_kinematics(masses, seed):
    """Born momenta of RAMBO at hadron beams and the momentum fractions."""
    mapping = ms.PhaseSpaceMapping(
        list(masses),
        SQRT_S,
        leptonic=False,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
    )
    r = np.random.default_rng(seed).random((1, mapping.random_dim()))
    momenta, x1, x2, _ = mapping.map_forward([to_torch(r)], [])
    return (
        to_numpy(momenta)[0],
        float(to_numpy(x1).ravel()[0]),
        float(to_numpy(x2).ravel()[0]),
    )


PROCESSES = {
    "gg_ttx": ([21, 21, 6, -6], [0.0, 0.0, TOP, TOP]),
    "ug_ug": ([2, 21, 2, 21], [0.0] * 4),
    "uux_ttxg": ([2, -2, 6, -6, 21], [0.0, 0.0, TOP, TOP, 0.0]),
    "ud_enu": ([2, -1, -11, 12], [0.0] * 4),
}


def random_colors(n, pdgs, seed, born_orders=1):
    """Symmetric color-correlated Borns of colored particles with color
    conservation, sum_k <T_j . T_k> = 0 with <T_j . T_j> = C_j B."""
    rng = np.random.default_rng(seed)
    colored = [j for j, p in enumerate(pdgs) if abs(p) <= 6 or p == 21]
    born = rng.uniform(0.5, 2.0, born_orders)
    colors = np.zeros((born_orders, n, n))
    for b in range(born_orders):
        for k, j in enumerate(colored):
            for m in colored[:k]:
                colors[b, j, m] = colors[b, m, j] = -rng.uniform(0.1, 1.0)
        for j in colored:
            colors[b, j, j] = -colors[b, j].sum()
    return born, colors


@pytest.mark.parametrize("name", sorted(PROCESSES))
def test_convolution(name):
    """The mean of the weights over r is the convolution of K + P with the
    luminosities of the beams, against the terms of CS and CDST convolved by
    adaptive quadrature. The nodes in r gather at z = 1, where the plus
    distributions leave logarithms of 1 - z."""
    pdgs, masses = PROCESSES[name]
    momenta, x1, x2 = born_kinematics(masses, seed=4)
    born, colors = random_colors(len(pdgs), pdgs, seed=5)
    nodes, w = np.polynomial.legendre.leggauss(300)
    t = (nodes + 1) / 2
    r = 1 - (1 - t) ** 3
    w = w / 2 * 3 * (1 - t) ** 2
    mu_f = (150.0, 70.0)
    z, weights = collinear(
        pdgs, masses, momenta[None], born[None], colors[None], (x1, x2), r, mu_f=mu_f
    )
    no_alpha = dict(fi=1.0, ii=1.0, **{"if": 1.0})
    for leg, eta in enumerate((x1, x2)):
        if not (abs(pdgs[leg]) <= 6 or pdgs[leg] == 21):
            assert np.all(weights[:, :, leg] == 0)
            continue
        g = luminosity(eta)
        lum = np.array([g(v) for v in z[:, leg]])
        same = np.sum(w * weights[:, 0, leg, 0] * lum) + np.sum(
            w * weights[:, 0, leg, 2]
        ) * g(1.0)
        other = np.sum(w * weights[:, 0, leg, 1] * lum)
        terms = kp_terms(
            pdgs, masses, momenta, born[0], colors[0], leg, mu_f[leg], no_alpha
        )
        theirs = {
            cls: A * sum(t.convolve(g, eta, s) for t, s in terms[cls]) for cls in terms
        }
        np.testing.assert_allclose(same, theirs["same"], rtol=1e-11)
        np.testing.assert_allclose(other, theirs["other"], rtol=1e-11)


@pytest.mark.parametrize("name", ["gg_ttx", "ug_ug", "uux_ttxg"])
def test_factorization_scale(name):
    """From mu_F to mu_F' the weights of a leg change by a log(mu_F'^2 /
    mu_F^2) P^{aa'} sum_j <T_j . T_a'> / T_a'^2, the collinear counterterm
    -a log(mu_F'^2 / mu_F^2) P^{aa'} B with color conservation."""
    pdgs, masses = PROCESSES[name]
    momenta, x1, x2 = born_kinematics(masses, seed=6)
    born, colors = random_colors(len(pdgs), pdgs, seed=7)
    r = np.linspace(0.05, 0.95, 7)
    z, w1 = collinear(
        pdgs,
        masses,
        momenta[None],
        born[None],
        colors[None],
        (x1, x2),
        r,
        mu_f=(100.0, 100.0),
    )
    _, w2 = collinear(
        pdgs,
        masses,
        momenta[None],
        born[None],
        colors[None],
        (x1, x2),
        r,
        mu_f=(300.0, 40.0),
    )
    for leg, (eta, mu) in enumerate(((x1, 300.0), (x2, 40.0))):
        gluon = pdgs[leg] == 21
        t2 = CA if gluon else CF
        same, other = ("gg", "qg") if gluon else ("qq", "gq")
        l = math.log(mu**2 / 100.0**2)
        zl, jac = z[:, leg], 1 - eta
        partners = [
            j for j, p in enumerate(pdgs) if j != leg and (abs(p) <= 6 or p == 21)
        ]
        coef = A * l * sum(colors[0, j, leg] for j in partners) / t2
        np.testing.assert_allclose(
            w2[:, 0, leg, 0] - w1[:, 0, leg, 0],
            coef
            * (np.array([p_regular(same, v) for v in zl]) + 2 * t2 / (1 - zl))
            * jac,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            w2[:, 0, leg, 1] - w1[:, 0, leg, 1],
            coef * np.array([p_regular(other, v) for v in zl]) * jac,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            w2[:, 0, leg, 2] - w1[:, 0, leg, 2],
            coef * (-2 * t2 * (jac / (1 - zl) - math.log(1 - eta)) + gamma_of(gluon)),
            rtol=1e-10,
        )


def g_perp(emitter, spectator):
    """-g_perp^mu nu of a massless emitter in the gauge of the spectator,
    contravariant, batched."""
    pn = dot(emitter, spectator)[:, None, None]
    out = (
        -np.diag(METRIC)
        + (
            emitter[:, :, None] * spectator[:, None, :]
            + spectator[:, :, None] * emitter[:, None, :]
        )
        / pn
    )
    return (
        out
        - dot(spectator, spectator)[:, None, None]
        * emitter[:, :, None]
        * emitter[:, None, :]
        / pn**2
    )


def removed_density(born_masses, i, j, k, kernel, alpha_name, alpha, x_nodes):
    """The density in x of the dipole of CSDipoles over what its gate
    removes at fixed Born, at unit color and averaged over the azimuth:
    int dr_2 dphi det / det_Born (D at alpha = 1 minus D at alpha) / (dx /
    dr_1), with 1 - x = (1 - eta) r_1^2 and the gate in r_2 of the family."""
    born = ms.PhaseSpaceMapping(
        list(born_masses),
        SQRT_S,
        leptonic=False,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
    )
    mapping = ms.CSMapping(born, i, j, k)
    r_born = np.array([0.3, 0.7, 0.4, 0.6])
    momenta_b, x1b, _, det_b = born.map_forward([to_torch(r_born[None])], [])
    momenta_b = to_numpy(momenta_b)[0]
    eta = float(to_numpy(x1b).ravel()[0])
    a = j if j < 2 else k
    s_born = 2 * dot(momenta_b[a], momenta_b[k if j < 2 else j])
    nodes, weights = np.polynomial.legendre.leggauss(64)
    rows, row_weights, index = [], [], []
    for n, x in enumerate(x_nodes):
        r1 = math.sqrt((1 - x) / (1 - eta))
        if alpha_name == "alpha_fi":
            low = 0.0 if 1 - x > alpha else 1.0
        elif alpha_name == "alpha_if":
            m = born_masses[k]
            z_max = (1 - x) / (1 - x + m * m * x / s_born)
            low = math.sqrt(alpha / z_max) if alpha < z_max else 1.0
        else:
            low = math.sqrt(alpha / (1 - x)) if alpha < 1 - x else 1.0
        r2 = low + (1 - low) * (nodes + 1) / 2
        w2 = (1 - low) * weights / 2
        for phi in (0.125, 0.375, 0.625, 0.875):
            for a2, b2 in zip(r2, w2):
                rows.append([*r_born, r1, a2, phi])
                row_weights.append(b2 / 4 / (2 * (1 - eta) * r1))
                index.append(n)
    r = np.array(rows)
    momenta, x1, x2, det = mapping.map_forward([to_torch(r)], [])
    batch = len(r)
    broadcast = lambda t: np.broadcast_to(to_numpy(t), (batch,)).copy()
    values = []
    for parameters in ({}, {alpha_name: alpha}):
        out = evaluate(
            ms.CSDipoles([mapping], [kernel], [1.0], **parameters),
            to_numpy(momenta),
            broadcast(x1),
            broadcast(x2),
        )
        bj = j - (j > i)
        bk = k - (k > i)
        b_mom = out["born_momenta"][:, 0]
        k_t = out["spin_vector"][:, 0] * METRIC
        spin = 0.5 * np.einsum(
            "bm,bmn,bn->b", k_t, g_perp(b_mom[:, bj], b_mom[:, bk]), k_t
        )
        values.append(
            -4
            * math.pi
            * ALPHA_S
            * (out["color_weight"][:, 0] + out["spin_weight"][:, 0] * spin)
        )
    integrand = (
        np.array(row_weights)
        * to_numpy(det)
        / to_numpy(det_b)[0]
        * (values[0] - values[1])
    )
    density = np.bincount(np.array(index), integrand, minlength=len(x_nodes))
    return density, momenta_b, eta


# (family, the kernels of the dipoles with their factors, the class of the
# leg 0 of the Born, its PDG code, the Born masses)
BEAM_CASES = {
    "fi q": (
        "alpha_fi",
        (4, 2, 0),
        {"qg": 1.0},
        0,
        [2, -1, 1, 23],
        [0.0, 0.0, 0.0, 91.188],
    ),
    "fi Q": (
        "alpha_fi",
        (4, 2, 0),
        {"qg": 1.0},
        0,
        [2, -1, 6, 23],
        [0.0, 0.0, TOP, 91.188],
    ),
    "fi g": (
        "alpha_fi",
        (4, 2, 0),
        {"gg": 0.5, "gq": 5.0},
        0,
        [2, -2, 21, 23],
        [0.0, 0.0, 0.0, 91.188],
    ),
    "if q": (
        "alpha_if",
        (4, 0, 2),
        {"qg": 1.0},
        0,
        [2, -1, 1, 23],
        [0.0, 0.0, 0.0, 91.188],
    ),
    "if q Q": (
        "alpha_if",
        (4, 0, 2),
        {"qg": 1.0},
        0,
        [2, -1, 6, 23],
        [0.0, 0.0, TOP, 91.188],
    ),
    "if q from g": (
        "alpha_if",
        (4, 0, 2),
        {"qq": 1.0},
        1,
        [2, -1, 6, 23],
        [0.0, 0.0, TOP, 91.188],
    ),
    "if g": (
        "alpha_if",
        (4, 0, 2),
        {"gg": 1.0},
        0,
        [21, -1, 6, 23],
        [0.0, 0.0, TOP, 91.188],
    ),
    "if g from q": (
        "alpha_if",
        (4, 0, 2),
        {"gq": 1.0},
        1,
        [21, -1, 1, 23],
        [0.0, 0.0, 0.0, 91.188],
    ),
    "ii q": (
        "alpha_ii",
        (4, 0, 1),
        {"qg": 1.0},
        0,
        [2, -2, 11, -11],
        [0.0, 0.0, 0.0, 0.0],
    ),
    "ii q from g": (
        "alpha_ii",
        (4, 0, 1),
        {"qq": 1.0},
        1,
        [2, -2, 11, -11],
        [0.0, 0.0, 0.0, 0.0],
    ),
    "ii g": (
        "alpha_ii",
        (4, 0, 1),
        {"gg": 1.0},
        0,
        [21, 21, 25, 23],
        [0.0, 0.0, 125.0, 91.188],
    ),
    "ii g from q": (
        "alpha_ii",
        (4, 0, 1),
        {"gq": 1.0},
        1,
        [21, 21, 25, 23],
        [0.0, 0.0, 125.0, 91.188],
    ),
}


@pytest.mark.parametrize("case", sorted(BEAM_CASES))
def test_alpha(case):
    """What alpha_fi, alpha_if and alpha_ii remove from the weights of the
    luminosity at z is the density in x = z of the dipoles over what their
    gates remove, times -x^2 (the flux and the luminosity of the real
    emission), for the pair of the Born leg 0 and its partner alone."""
    alpha_name, (i, j, k), kernels, cls, pdgs, masses = BEAM_CASES[case]
    alpha = 0.25
    x_nodes = np.array([0.35, 0.5, 0.62, 0.71])
    density = 0.0
    for kernel, factor in kernels.items():
        d, momenta, eta = removed_density(
            masses, i, j, k, kernel, alpha_name, alpha, x_nodes
        )
        density = density + factor * d
    assert eta < x_nodes.min()
    partner = k if j == 0 else j
    colors = np.zeros((1, 4, 4))
    colors[0, 0, partner] = colors[0, partner, 0] = 1.0
    born = np.array([0.0])
    r = (x_nodes - eta) / (1 - eta)
    z, full = collinear(
        pdgs, masses, momenta[None], born[None], colors[None], (eta, 0.5), r
    )
    _, gated = collinear(
        pdgs,
        masses,
        momenta[None],
        born[None],
        colors[None],
        (eta, 0.5),
        r,
        **{alpha_name: alpha},
    )
    np.testing.assert_allclose(z[:, 0], x_nodes, rtol=1e-14)
    difference = (gated - full)[:, 0, 0, cls] / (1 - eta)
    np.testing.assert_allclose(
        difference, -density * x_nodes**2, rtol=1e-10, atol=1e-14
    )
    # the end point is unchanged, and so is the other flavor for fi (the
    # spectator keeps its flavor)
    np.testing.assert_allclose((gated - full)[:, 0, 0, 2], 0.0, atol=1e-14)
    if alpha_name == "alpha_fi":
        np.testing.assert_allclose((gated - full)[:, 0, 0, 1], 0.0, atol=1e-14)


def test_invalid():
    with pytest.raises(ValueError):
        ms.CSCollinear([21, 21, 6, -6], [0.0, 0.0, TOP, TOP], alpha_if=0.0)
    with pytest.raises(ValueError):
        ms.CSCollinear([21, 21, 6, -6], [0.0, 0.0, TOP, TOP], alpha_ii=1.5)
    with pytest.raises(ValueError):
        ms.CSCollinear([6, 21, 6, 21], [TOP, 0.0, TOP, 0.0])
    with pytest.raises(ValueError):
        ms.CSCollinear([21, 21, 6, -6], [0.0, 0.0, TOP])
    with pytest.raises(ValueError):
        ms.CSCollinear([21, 21, 6, -6], [0.0, 0.0, TOP, TOP], born_orders=0)
