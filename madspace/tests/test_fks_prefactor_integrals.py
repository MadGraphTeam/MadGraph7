"""FKSPrefactors against closed-form plus distributions.

The real emission and its counterterms are integrated over the radiation
variables of the FKS mappings with Gauss-Legendre quadrature, with simple test
functions in place of xi^2 (1 - y) times the real matrix element and of the
collinear remnants. The prefactors multiply the Jacobians of the event and the
counter-events. Those are written here from the sampling of xi_hat and y (with
the ranges of xi that FKSKinematics outputs), so that the phase-space factors
of the dets cancel and the sums are one-dimensional integrals with known
values. With the cuts of FKSSubtraction, the sums must agree with
  int_lo^h dxi [F(xi) - F(0) theta(xi < xi_l)] / xi
      - F(0) log(xicut / xi_l) (xi_l - lo) / xi_l
in xi, xi_l = min(xi_max, xiScut), and the same with 1 - y, delta and deltaS in
1 - y. Their lower ends lo are those of the sampling (stiny xi_max and
2 ctiny), at which the sums agree with the plus distributions
  int_0^h dxi [F(xi) - F(0)] / xi + F(0) log(h / xicut)
up to O(stiny log^2 stiny). The collinear remnants of an initial-state emitter
add log(xi) / xi and the parts of a change of the PDF scheme, whose
plus distribution [1/(1 - z)]_+ in z = 1 - xi gives log(h) for its delta
and plus parts, and log(h)^2 / 2 for its log-plus part.

F(t) = 1 + t + t^2 throughout, so that every integral has a closed form.
"""

import numpy as np
import pytest
from fks_testing import FKS_DATA, FKS_DATA_IDS, evaluate, reference
from fks_testing import reference_born_mapping as born_mapping

import madspace as ms
from madspace.fks import fks_configuration_mapping, fks_prefactors

STINY, CTINY = 1e-6, 5e-7
EPS = np.finfo(np.float64).eps
# the analytic integrals agree with the quadrature to this, relative to the
# size of the terms that cancel in it
RTOL = 1e-12
# the integrals over the sampled ranges agree with the plus distributions up to
# terms of order stiny log^2 stiny and ctiny log ctiny
STINY_TOL = 1e-4

CUTS = [
    dict(xicut=0.5, xi_s_cut=0.5, delta_o=1.0, delta_i=1.0, delta_s=1.0),
    dict(xicut=0.3, xi_s_cut=0.25, delta_o=0.8, delta_i=0.7, delta_s=0.6),
    dict(xicut=0.1, xi_s_cut=0.2, delta_o=0.2, delta_i=0.4, delta_s=0.3),
    dict(xicut=0.9, xi_s_cut=1.0, delta_o=1.8, delta_i=1.5, delta_s=2.0),
]
CUT_IDS = ["default", "params", "small", "large"]


def F(t):
    return 1.0 + t + t * t


def integral(a, b, log_power=0, without_constant=False):
    """int_a^b F(t) log(t)^log_power / t dt (log_power 0 or 1), without the
    constant term of F if without_constant, in closed form (a may be 0 then)."""

    def power(k, t):
        # int t^(k - 1) log(t)^log_power dt
        if k == 0:
            return np.log(t) if log_power == 0 else np.log(t) ** 2 / 2
        if t == 0:
            return 0.0
        if log_power == 0:
            return t**k / k
        return t**k * (k * np.log(t) - 1) / k**2

    return sum(
        power(k, b) - power(k, a) for k in range(3) if k > 0 or not without_constant
    )


def subtracted(h, lo, lim, cut, log_power=0):
    """int_lo^h F log^p / t - F(0) [int_lo^lim log^p / t + c (lim - lo) / lim]
    with c = log(cut / lim) for p = 0, (log^2 cut - log^2 lim) / 2 for p = 1:
    a subtraction at t <= lim with an analytic cut."""
    if log_power == 0:
        c = np.log(cut / lim)
        under = np.log(lim / lo)
    else:
        c = (np.log(cut) ** 2 - np.log(lim) ** 2) / 2
        under = (np.log(lim) ** 2 - np.log(lo) ** 2) / 2
    return integral(lo, h, log_power) - F(0.0) * (under + c * (lim - lo) / lim)


def plus(h, cut, log_power=0):
    """The plus distribution in t with the analytic cut: int_0^h [F(t) - F(0)]
    log^p / t + F(0) (log^(p + 1) h - log^(p + 1) cut) / (p + 1)."""
    edge = (np.log(h) ** (log_power + 1) - np.log(cut) ** (log_power + 1)) / (
        log_power + 1
    )
    return integral(0.0, h, log_power, without_constant=True) + F(0.0) * edge


def nodes(edges, count=32):
    """Gauss-Legendre nodes and weights on [0, 1] with breakpoints at edges,
    refined towards 0 where xi = stiny + r^2 turns over at r ~ sqrt(stiny)."""
    edges = sorted(
        {e for e in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3] if e < max(edges)}
        | set(edges)
    )
    x, w = np.polynomial.legendre.leggauss(count)
    xs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            xs.append(a + (b - a) * (x + 1) / 2)
            ws.append((b - a) * w / 2)
    return np.concatenate(xs), np.concatenate(ws)


def r_of_xi_hat(xi_hat):
    """The random number of xi_hat = stiny + (1 - stiny) r^2."""
    return np.sqrt(np.clip((xi_hat - STINY) / (1 - STINY), 0.0, 1.0))


def r_of_omy(omy, width=2.0):
    """The random number of 1 - y = width (ctiny + (1 - ctiny) r^2)."""
    return np.sqrt(np.clip((omy / width - CTINY) / (1 - CTINY), 0.0, 1.0))


class Process:
    """FKSKinematics and FKSPrefactors of one FKS configuration of reference
    data at a fixed Born point."""

    def __init__(
        self, name, index, born_random, s_hat_min=0.0, initial_map="symmetric"
    ):
        data = reference(FKS_DATA[FKS_DATA_IDS.index(name)])
        self.born = born_mapping(data)
        self.configurations = data["fks_configurations"]
        self.index = index
        self.born_random = np.array(born_random)
        self.mapping = fks_configuration_mapping(
            self.born, self.configurations, s_hat_min=s_hat_min, initial_map=initial_map
        )
        self.names = list(self.mapping.counter_event_types().keys())

    def kinematics(self, r_xi, r_y, r_phi=0.3):
        r_xi, r_y = np.broadcast_arrays(np.asarray(r_xi, float), np.asarray(r_y, float))
        batch = r_xi.size
        r = np.column_stack(
            [
                np.tile(self.born_random, (batch, 1)),
                r_xi.ravel(),
                r_y.ravel(),
                np.full(batch, r_phi),
            ]
        )
        values = evaluate(
            ms.FKSKinematics(self.mapping), r, fks_index=np.full(batch, self.index)
        )
        return dict(zip(self.names, values))

    def prefactors(self, out, cuts):
        values = evaluate(
            fks_prefactors(self.born, self.configurations, **cuts),
            out["xi"],
            out["y"],
            out["xi_max"],
            out["collinear_xi"],
            out["collinear_xi_max"],
            out["collinear_momenta"],
            out["soft_collinear_momenta"],
            fks_index=np.full(len(out["xi"]), self.index),
        )
        names = ["f_r", "f_s", "f_c", "f_dc", "f_sc", "f_dsc", "f_pdf"]
        return dict(zip(names, values))


def conditions(out, cuts):
    """Where FKSSubtraction keeps the soft, collinear and soft-collinear
    counterterms, as 1 or 0."""
    xi_s_cut, delta_s = cuts["xi_s_cut"], cuts["delta_s"]
    in_y = out["y"] > 1 - delta_s
    soft = (
        (out["soft_det"] > 0)
        & (out["xi_hat"] * out["soft_xi_max"] <= xi_s_cut)
        & (out["xi"] <= xi_s_cut)
    )
    collinear = (out["collinear_det"] > 0) & in_y
    soft_collinear = (
        (out["soft_collinear_det"] > 0)
        & in_y
        & (out["xi_hat"] * out["collinear_xi_max"] < xi_s_cut)
        & (out["collinear_xi"] < xi_s_cut)
    )
    return soft * 1.0, collinear * 1.0, soft_collinear * 1.0


def check(numeric, exact, scale, target=None):
    assert abs(numeric - exact) <= RTOL * scale, (numeric, exact)
    if target is not None:
        assert abs(exact - target) <= STINY_TOL * max(1.0, abs(target)), (exact, target)


# massless final-state emitter: e+ e- > u u~ g, emitter u
FINAL = ("epem_uuxg_P0_epem_uuxg", 0, [0.3, 0.6, 0.2, 0.7, 0.4])
# massive final-state emitter: e+ e- > t t~, emitter t
MASSIVE = ("epem_ttx_P0_epem_ttx", 0, [0.3, 0.6])
# incoming emitter: u u~ > e- e+, emitter u, at a Born with x1 = 0.36 (xi_max
# of the collinear counter-event 0.64) and one with x1 = 0.61 (0.39)
INITIAL = "pp_epem_P0_uux_emep"
SMALL_X = [0.5, 0.3, 0.3, 0.6]
LARGE_X = [0.9, 0.95, 0.3, 0.6]


@pytest.mark.parametrize("cuts", CUTS, ids=CUT_IDS)
def test_massless_final_emitter(cuts):
    """Real emission and all counterterms in xi and 1 - y at once: for a
    final-state emitter the range of xi does not depend on y and the
    collinear counter-event shares xi with the event, so the integral is the
    product of the two one-dimensional ones."""
    process = Process(*FINAL)
    h = process.kinematics(0.5, 0.5)["xi_max"][0]
    lim_xi = min(h, cuts["xi_s_cut"])
    r_xi, w_xi = nodes([0.0, r_of_xi_hat(lim_xi / h), 1.0])
    r_y, w_y = nodes([0.0, r_of_omy(min(cuts["delta_s"], 2.0)), 1.0])
    rx, ry = (a.ravel() for a in np.meshgrid(r_xi, r_y, indexing="ij"))
    weight = np.outer(w_xi, w_y).ravel()
    out = process.kinematics(rx, ry)
    f = process.prefactors(out, cuts)
    soft, collinear, soft_collinear = conditions(out, cuts)
    np.testing.assert_allclose(out["collinear_xi"], out["xi"], rtol=1e-14)
    xi, omy = out["xi"], 1 - out["y"]
    # dxi dy / dr of the sampling, for the event and the counter-events alike
    jac = h * 2 * (1 - STINY) * rx * 4 * (1 - CTINY) * ry
    terms = [
        f["f_r"] * F(xi) * F(omy),
        -soft * f["f_s"] * F(0.0) * F(omy),
        -collinear * f["f_c"] * F(xi) * F(0.0),
        soft_collinear * f["f_sc"] * F(0.0) ** 2,
    ]
    numeric = np.sum(weight * jac * sum(terms))
    scale = sum(np.sum(weight * jac * np.abs(t)) for t in terms)
    in_xi = subtracted(h, h * STINY, lim_xi, cuts["xicut"])
    in_y = subtracted(2.0, 2 * CTINY, min(cuts["delta_s"], 2.0), cuts["delta_o"])
    check(
        numeric,
        in_xi * in_y,
        scale,
        plus(h, cuts["xicut"]) * plus(2.0, cuts["delta_o"]),
    )


@pytest.mark.parametrize("r_y", [0.9, 0.3])
def test_massive_final_emitter(r_y):
    """Real emission and soft counterterm in xi at fixed y: for y < 0 the
    second solution continues from xi_max back to xi_m (the emitter at rest)
    without a counterterm, for y >= 0 there is only the first."""
    cuts = CUTS[0]
    process = Process(*MASSIVE)
    ends = process.kinematics([0.5, 1.0], r_y)
    h, xi_m = ends["xi_max"][0], ends["xi"][1]
    xi_norm = 2 * h - xi_m
    rat = h / xi_norm
    assert (rat < 1) == (ends["y"][0] < 0)
    lim = min(h, cuts["xi_s_cut"])
    r, w = nodes([0.0, rat * r_of_xi_hat(lim / h), rat, 1.0])
    out = process.kinematics(r, r_y)
    f = process.prefactors(out, cuts)
    soft, _, _ = conditions(out, cuts)
    assert not soft[r > rat].any()  # soft counter-event of the first solution only
    omy = 1 - out["y"]
    # xi = xi_max (stiny + (1 - stiny) (r / rat)^2) up to rat, then
    # 2 xi_max - xi_norm r
    jac = np.where(r <= rat, h * 2 * (1 - STINY) * r / rat**2, xi_norm)
    # (1 - y) F(xi): no collinear singularity
    terms = [f["f_r"] * omy * F(out["xi"]), -soft * f["f_s"] * omy * F(0.0)]
    numeric = np.sum(w * jac * sum(terms))
    scale = sum(np.sum(w * jac * np.abs(t)) for t in terms)
    second = integral(xi_m, h) if rat < 1 else 0.0
    exact = subtracted(h, h * STINY, lim, cuts["xicut"]) + second
    check(numeric, exact, scale, plus(h, cuts["xicut"]) + second)


@pytest.mark.parametrize("cuts", CUTS, ids=CUT_IDS)
@pytest.mark.parametrize("born", [SMALL_X, LARGE_X], ids=["small_x", "large_x"])
@pytest.mark.parametrize("r_y", [0.2, 0.5, 0.9])
def test_initial_emitter_soft(r_y, born, cuts):
    """Real emission and soft counterterm in xi at fixed y, up to the
    y-dependent end of the range of xi."""
    process = Process(INITIAL, 0, born)
    h = process.kinematics(0.5, r_y)["xi_max"][0]
    lim = min(h, cuts["xi_s_cut"])
    r, w = nodes([0.0, r_of_xi_hat(lim / h), 1.0])
    out = process.kinematics(r, r_y)
    f = process.prefactors(out, cuts)
    soft, _, _ = conditions(out, cuts)
    np.testing.assert_array_equal(out["soft_xi_max"], out["xi_max"])
    omy = 1 - out["y"]
    jac = h * 2 * (1 - STINY) * r
    terms = [f["f_r"] * omy * F(out["xi"]), -soft * f["f_s"] * omy * F(0.0)]
    numeric = np.sum(w * jac * sum(terms))
    scale = sum(np.sum(w * jac * np.abs(t)) for t in terms)
    exact = subtracted(h, h * STINY, lim, cuts["xicut"])
    check(numeric, exact, scale, plus(h, cuts["xicut"]))


@pytest.mark.parametrize("cuts", CUTS, ids=CUT_IDS)
@pytest.mark.parametrize("r_xi", [0.3, 0.8])
def test_initial_emitter_collinear(r_xi, cuts):
    """Real emission and collinear counterterm in 1 - y at fixed xi_hat. The
    collinear counter-event shares xi_hat with the event, with the range of xi
    at y = 1, so the test function F(1 - y) xi / xi_max(y) xi_max(1) has the
    same limit at the counter-event as the event."""
    process = Process(INITIAL, 0, SMALL_X)
    lim = min(cuts["delta_s"], 2.0)
    r, w = nodes([0.0, r_of_omy(lim), 1.0])
    out = process.kinematics(r_xi, r)
    f = process.prefactors(out, cuts)
    _, collinear, _ = conditions(out, cuts)
    omy = 1 - out["y"]
    h_c = out["collinear_xi_max"]
    jac_xi = 2 * (1 - STINY) * r_xi
    jac_y = 4 * (1 - CTINY) * r
    terms = [
        # f_r T dxi/dr with T = xi F(1 - y) h_c / xi_max(y)
        f["f_r"] * out["xi"] * F(omy) * h_c,
        -collinear * f["f_c"] * out["collinear_xi"] * F(0.0) * h_c,
    ]
    numeric = np.sum(w * jac_y * jac_xi * sum(terms)) / (h_c[0] * jac_xi)
    scale = sum(np.sum(w * jac_y * np.abs(t)) for t in terms) / h_c[0]
    exact = subtracted(2.0, 2 * CTINY, lim, cuts["delta_i"])
    check(numeric, exact, scale, plus(2.0, cuts["delta_i"]))


@pytest.mark.parametrize("cuts", CUTS, ids=CUT_IDS)
@pytest.mark.parametrize("born", [SMALL_X, LARGE_X], ids=["small_x", "large_x"])
def test_initial_emitter_asymmetric(born, cuts):
    """With the asymmetric map the range of xi, up to 1 - x of the emitter,
    does not depend on y, and the collinear counter-event shares xi with the
    event: the real emission and all counterterms in xi and 1 - y at once, the
    product of the two one-dimensional integrals as for a massless final-state
    emitter, with delta_I."""
    process = Process(INITIAL, 0, born, initial_map="asymmetric")
    ends = process.kinematics(0.5, [0.1, 0.9])
    h = ends["xi_max"][0]
    np.testing.assert_array_equal(ends["xi_max"], h)
    np.testing.assert_allclose(h, 1 - ends["born_x1"][0], rtol=1e-15)
    lim_xi = min(h, cuts["xi_s_cut"])
    r_xi, w_xi = nodes([0.0, r_of_xi_hat(lim_xi / h), 1.0])
    r_y, w_y = nodes([0.0, r_of_omy(min(cuts["delta_s"], 2.0)), 1.0])
    rx, ry = (a.ravel() for a in np.meshgrid(r_xi, r_y, indexing="ij"))
    weight = np.outer(w_xi, w_y).ravel()
    out = process.kinematics(rx, ry)
    f = process.prefactors(out, cuts)
    soft, collinear, soft_collinear = conditions(out, cuts)
    np.testing.assert_allclose(out["collinear_xi"], out["xi"], rtol=1e-14)
    xi, omy = out["xi"], 1 - out["y"]
    jac = h * 2 * (1 - STINY) * rx * 4 * (1 - CTINY) * ry
    terms = [
        f["f_r"] * F(xi) * F(omy),
        -soft * f["f_s"] * F(0.0) * F(omy),
        -collinear * f["f_c"] * F(xi) * F(0.0),
        soft_collinear * f["f_sc"] * F(0.0) ** 2,
    ]
    numeric = np.sum(weight * jac * sum(terms))
    scale = sum(np.sum(weight * jac * np.abs(t)) for t in terms)
    in_xi = subtracted(h, h * STINY, lim_xi, cuts["xicut"])
    in_y = subtracted(2.0, 2 * CTINY, min(cuts["delta_s"], 2.0), cuts["delta_i"])
    check(
        numeric,
        in_xi * in_y,
        scale,
        plus(h, cuts["xicut"]) * plus(2.0, cuts["delta_i"]),
    )


def test_initial_emitter_below_threshold():
    """With s_hat_min above the Born, xi starts at xi_min = 1 - s_born /
    s_hat_min: no soft counterterm, and the collinear one on the range of y
    that remains, 1 - y from 0 to its width."""
    cuts = CUTS[0]
    # at the Born with the larger x2, where x2 <= 1 restricts y
    plain = Process(INITIAL, 0, LARGE_X)
    born = plain.kinematics(0.5, 0.5)["born_momenta"][0]
    p = born[0] + born[1]
    s_born = p[0] ** 2 - np.sum(p[1:] ** 2)
    process = Process(INITIAL, 0, LARGE_X, s_hat_min=1.3 * s_born)
    xi_min = 1 - 1 / 1.3
    # soft direction at fixed y: the real emission alone
    r_y = 0.3
    h = process.kinematics(0.5, r_y)["xi_max"][0]
    r, w = nodes([0.0, 1.0])
    out = process.kinematics(r, r_y)
    f = process.prefactors(out, cuts)
    assert (out["soft_det"] == 0).all()
    np.testing.assert_allclose(
        out["xi"], xi_min + (h - xi_min) * out["xi_hat"], rtol=1e-14
    )
    omy = 1 - out["y"]
    jac = (h - xi_min) * 2 * (1 - STINY) * r
    numeric = np.sum(w * jac * f["f_r"] * omy * F(out["xi"]))
    check(numeric, integral(xi_min + (h - xi_min) * STINY, h), numeric)
    # collinear direction at fixed xi_hat, 1 - y up to the width of the range
    r_xi = 0.6
    width = 1 - process.kinematics(r_xi, 1.0)["y"][0]
    assert width < 2
    lim = min(cuts["delta_s"], width)
    r, w = nodes([0.0, r_of_omy(lim, width), 1.0])
    out = process.kinematics(r_xi, r)
    f = process.prefactors(out, cuts)
    _, collinear, _ = conditions(out, cuts)
    omy = 1 - out["y"]
    np.testing.assert_allclose(
        omy, width * (CTINY + (1 - CTINY) * r**2), rtol=1e-12, atol=4 * EPS
    )
    width_c = out["collinear_xi_max"] - xi_min
    jac_y = width * 2 * (1 - CTINY) * r
    # T = xi F(1 - y) width_c / width(y), with dxi/dr = width(y) 2 (1 - stiny) r_xi
    terms = [
        f["f_r"] * out["xi"] * F(omy) * width_c,
        -collinear * f["f_c"] * out["collinear_xi"] * F(0.0) * width_c,
    ]
    numeric = np.sum(w * jac_y * sum(terms)) / width_c[0]
    scale = sum(np.sum(w * jac_y * np.abs(t)) for t in terms) / width_c[0]
    lo = width * CTINY
    exact = integral(lo, width) - F(0.0) * (
        np.log(lim / lo)
        + np.log(cuts["delta_i"] / cuts["delta_s"]) / cuts["delta_s"] * (lim - lo)
    )
    check(numeric, exact, scale)


@pytest.mark.parametrize("cuts", CUTS, ids=CUT_IDS)
@pytest.mark.parametrize("born", [SMALL_X, LARGE_X], ids=["small_x", "large_x"])
def test_initial_emitter_remnants(born, cuts):
    """The collinear remnants in xi at the collinear counter-event (z = 1 -
    xi): the MSbar parts with [1/xi] and [log(xi)/xi] and the analytic cut,
    and the parts of a change of the PDF scheme with [1/(1 - z)]_+,
    [log(1 - z)/(1 - z)]_+ and delta(1 - z) on the range of z of the Born,
    also where xi_max of the collinear counter-event is below xiScut
    (large_x)."""
    process = Process(INITIAL, 0, born)
    r_y = 0.1
    h = process.kinematics(0.5, r_y)["collinear_xi_max"][0]
    lim = min(h, cuts["xi_s_cut"])
    r, w = nodes([0.0, r_of_xi_hat(lim / h), 1.0])
    out = process.kinematics(r, r_y)
    f = process.prefactors(out, cuts)
    _, collinear, soft_collinear = conditions(out, cuts)
    assert collinear.all()

    def s(momenta):
        p = momenta[:, 0] + momenta[:, 1]
        return p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=1)

    # the prefactors per unit 32 pi^2 / (s deltaS)
    unit_c = s(out["collinear_momenta"]) * cuts["delta_s"] / (32 * np.pi**2)
    unit_sc = s(out["soft_collinear_momenta"]) * cuts["delta_s"] / (32 * np.pi**2)
    f_dc = f["f_dc"] * unit_c
    f_dsc = f["f_dsc"] * unit_sc[:, None]
    f_pdf = f["f_pdf"] * unit_sc[:, None]
    xi_c = out["collinear_xi"]
    log_xi = np.log(xi_c)
    jac = h * 2 * (1 - STINY) * r
    sc = soft_collinear
    f0 = F(0.0)
    cases = {
        # D_xi: f_dc D(xi) - f_dsc1 D(0) - f_dsc2 D(0)
        "msbar": (
            [f_dc * F(xi_c), -sc * (f_dsc[:, 0] + f_dsc[:, 1]) * f0],
            subtracted(h, h * STINY, lim, cuts["xicut"]),
            plus(h, cuts["xicut"]),
        ),
        # D_lxi: f_dc D(xi) log xi - f_dsc1 D(0) log xi - f_dsc3 D(0)
        "msbar_log": (
            [f_dc * F(xi_c) * log_xi, -sc * (f_dsc[:, 0] * log_xi + f_dsc[:, 2]) * f0],
            subtracted(h, h * STINY, lim, cuts["xicut"], 1),
            plus(h, cuts["xicut"], 1),
        ),
        # K_p: f_dc K(xi) - f_dsc1 K(0) + f_pdf_p K(0)
        "scheme_plus": (
            [f_dc * F(xi_c), -sc * (f_dsc[:, 0] - f_pdf[:, 1]) * f0],
            subtracted(h, h * STINY, lim, 1.0),
            plus(h, 1.0),
        ),
        # K_l: f_dc K(xi) log xi - f_dsc1 K(0) log xi + f_pdf_l K(0)
        "scheme_log": (
            [f_dc * F(xi_c) * log_xi, -sc * (f_dsc[:, 0] * log_xi - f_pdf[:, 2]) * f0],
            subtracted(h, h * STINY, lim, 1.0, 1),
            plus(h, 1.0, 1),
        ),
        # K_d: f_pdf_d K
        "scheme_delta": ([sc * f_pdf[:, 0]], (lim - h * STINY) / lim, 1.0),
    }
    for name, (terms, exact, target) in cases.items():
        numeric = np.sum(w * jac * sum(terms))
        scale = sum(np.sum(w * jac * np.abs(t)) for t in terms)
        try:
            check(numeric, exact, scale, target)
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from None
