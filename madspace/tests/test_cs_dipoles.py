"""CSDipoles: the Catani-Seymour dipoles of a real process against its real
matrix element in the soft and collinear limits.

The reference data (test_data/cs, from cs_reference/make_cs_reference.py)
have real emissions along rays into the limits of each pair (i, j) of
dipoles, with the real matrix element and the Born blocks of every dipole of
the real process from pyAmpliCol: the color-correlated Born of its emitter
and spectator and, for a gluon emitter, its spin tensor. The sum of the
dipoles has to approach the real matrix element, (R - sum D) / R -> 0, for
every family, massless and massive, with the spin correlations of gluon
emitters (also where the joint color and spin correlation of g g > t t~ g
does not factorize, and with the product of its marginals in its place). The
weights of the dipoles are checked against the kernels of the CS and CDST
papers as the NCV code writes them, and the alpha parameters against the
variables of the families.
"""

import glob
import json
import math
import os

import numpy as np
import pytest
import torch
from madspace_testing import default_context

import madspace as ms

DATA = sorted(
    glob.glob(os.path.join(os.path.dirname(__file__), "test_data", "cs", "*.json"))
)
IDS = [os.path.basename(path)[:-5] for path in DATA]
CA, CF, TR = 3.0, 4.0 / 3.0, 0.5


def load(path):
    with open(path) as f:
        return json.load(f)


def real_masses(data, pdgs):
    masses = {int(k): v for k, v in data["setup"]["masses"].items()}
    return np.array([masses.get(abs(p), 0.0) for p in pdgs])


def dipoles_of(data, real):
    return [(n, d) for n, d in enumerate(data["cs"]) if d["real"] == real]


def dipole_generator(data, real, **parameters):
    """The CSDipoles of a real process of the reference data."""
    setup = data["setup"]
    pdgs = data["reals"][real]["pdgs"]
    masses = real_masses(data, pdgs)
    mappings, kernels, symmetry = [], [], []
    for _, d in dipoles_of(data, real):
        born = ms.PhaseSpaceMapping(
            list(np.delete(masses, d["i"])),
            setup["sqrt_s"],
            leptonic=setup["leptonic"],
            mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
        )
        mappings.append(ms.CSMapping(born, d["i"], d["j"], d["k"]))
        kernels.append(d["kernel"])
        symmetry.append(d["symmetry"])
    return ms.CSDipoles(mappings, kernels, symmetry, **parameters)


def evaluate(generator, points):
    runtime = ms.FunctionRuntime(generator.function(), default_context())
    names = list(generator.return_types().keys())
    inputs = [
        torch.as_tensor(np.array([p[key] for p in points]))
        for key in ("momenta", "x1", "x2")
    ]
    out = runtime.call(inputs)
    return dict(zip(names, [torch.from_dlpack(t).cpu().numpy() for t in out]))


def born_line(data, d):
    """The PDG codes of the Born of a dipole in the order of its mapping,
    the real process without i."""
    pdgs = data["borns"][d["born"]]["pdgs"]
    line = [0] * len(pdgs)
    for n, m in enumerate(d["born_order"]):
        line[m] = pdgs[n]
    return line


def resolved(data, d, momenta):
    """The jet function of the tests on the Born of a dipole: its massless
    colored final-state partons have a transverse momentum above 1% of
    sqrt(s) (hadron beams) and pair invariants above 1e-3 s, so that its
    Born has no soft or collinear singularity of its own."""
    setup = data["setup"]
    line = born_line(data, d)
    masses = real_masses(data, line)
    partons = [
        n
        for n in range(setup["nincoming"], len(line))
        if (abs(line[n]) <= 6 or line[n] == 21) and masses[n] == 0.0
    ]
    s = setup["sqrt_s"] ** 2
    if not setup["leptonic"]:
        for n in partons:
            if np.hypot(*momenta[n, 1:3]) < 1e-2 * setup["sqrt_s"]:
                return False
    for k, a in enumerate(partons):
        for b in partons[k + 1 :]:
            if 2 * dot(momenta[a], momenta[b]) < 1e-3 * s:
                return False
    return True


def dipole_values(data, points, out, jets=True, factorized=False):
    """Each dipole -g_s^2 (c <T_j . T_k> + s <T_j . T_k>(k_t)) at the
    points, (points, dipoles), zero where the jet function of the tests
    rejects its Born (with jets). factorized takes <T_j . T_k> B(k_t) / B
    for the joint color and spin correlation."""
    g2 = 4 * math.pi * data["setup"]["alpha_s"]
    values = np.zeros(out["color_weight"].shape)
    dipoles = [d for _, d in dipoles_of(data, points[0]["real"])]
    for p, point in enumerate(points):
        for m, entry in enumerate(point["dipoles"]):
            if jets and not resolved(data, dipoles[m], out["born_momenta"][p, m]):
                continue
            value = out["color_weight"][p, m] * entry["color_correlated"]
            if entry["spin_correlated"] is not None:
                k_t = out["spin_vector"][p, m]
                tensor = np.array(entry["spin_correlated"])
                if factorized:
                    tensor = np.array(entry["born_spin_correlated"]) * (
                        entry["color_correlated"] / entry["born"]
                    )
                value += out["spin_weight"][p, m] * (k_t @ tensor @ k_t)
            values[p, m] = -g2 * value
    return values


def points_by_real(data):
    reals = {}
    for point in data["points"]:
        reals.setdefault(point["real"], []).append(point)
    return reals


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_projection(path):
    """CSDipoles projects the real emission onto the Borns of the reference
    data, at which their blocks are."""
    data = load(path)
    for real, points in points_by_real(data).items():
        out = evaluate(dipole_generator(data, real), points)
        stored = np.array([[e["born_momenta"] for e in p["dipoles"]] for p in points])
        scale = data["setup"]["sqrt_s"]
        np.testing.assert_allclose(
            out["born_momenta"] / scale, stored / scale, rtol=0, atol=1e-12
        )


def ray_ratios(data, factorized=False):
    """(R - sum D) / R along each ray, by lambda from 0.1 down."""
    rays = {}
    for real, points in points_by_real(data).items():
        out = evaluate(dipole_generator(data, real), points)
        values = dipole_values(data, points, out, factorized=factorized)
        for p, point in enumerate(points):
            ray = point["ray"]
            if ray["limit"] == "none":
                continue
            key = (real, ray["dipole"], ray["limit"], ray["random"][-1])
            difference = abs(point["real_me"] - values[p].sum()) / point["real_me"]
            rays.setdefault(key, []).append((ray["lam"], difference))
    return {key: [r for _, r in sorted(v, reverse=True)] for key, v in rays.items()}


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_limits(path):
    """(R - sum D) / R falls along every ray into a soft or collinear limit
    of a pair of dipoles, from lambda = 0.1 to 1e-4 (lambda^2 in the variable
    of the limit) by a factor 10 down to the rounding of the momenta (1e-8
    for a gluon of 1e-8 of the energy), to below 1e-3. The dipoles whose
    Born the jet function rejects are left out, as their Born is singular in
    the limits of other pairs (e+ e- > u u~ g at u collinear to g)."""
    data = load(path)
    for key, ratios in ray_ratios(data).items():
        assert ratios[-1] < max(ratios[0] / 10, 1e-6), (key, ratios)
        assert ratios[-1] < 1e-3, (key, ratios)


def g_perp(emitter, spectator):
    """-g_perp^mu nu, the sum over the polarizations of a massless vector
    boson of momentum emitter in the gauge of spectator, contravariant."""
    metric = np.diag([1.0, -1.0, -1.0, -1.0])
    pn = dot(emitter, spectator)
    return (
        -metric
        + (np.outer(emitter, spectator) + np.outer(spectator, emitter)) / pn
        - dot(spectator, spectator) * np.outer(emitter, emitter) / pn**2
    )


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_spin_tensors(path):
    """The spin tensors of a gluon emitter sum over its polarizations to the
    color-correlated Born and to the Born, -g_perp^mu nu S_mu,nu, in the
    gauge of the spectator: the azimuthal average of a dipole of the real
    over its emission at fixed Born takes <T_j . T_k>(k_t) to <T_j . T_k>
    (-k_t^2) / 2, and so does the product of the marginals, <T_j . T_k>
    B(k_t) / B."""
    data = load(path)
    count = 0
    for point in data["points"]:
        dipoles = [d for _, d in dipoles_of(data, point["real"])]
        for d, entry in zip(dipoles, point["dipoles"]):
            if entry["spin_correlated"] is None:
                continue
            born = np.array(entry["born_momenta"])
            bj = d["j"] - (d["j"] > d["i"])
            bk = d["k"] - (d["k"] > d["i"])
            metric = g_perp(born[bj], born[bk])
            joint = np.sum(metric * np.array(entry["spin_correlated"]))
            summed = np.sum(metric * np.array(entry["born_spin_correlated"]))
            np.testing.assert_allclose(joint, entry["color_correlated"], rtol=1e-9)
            np.testing.assert_allclose(summed, entry["born"], rtol=1e-9)
            count += 1
    if any(pdg == 21 for born in data["borns"] for pdg in born["pdgs"]):
        assert count


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_factorized_spin(path):
    """The product of the marginals, <T_j . T_k> B(k_t) / B, in place of
    the joint color and spin correlation also cancels the limits of the real
    matrix element: the sum over the spectators has the same collinear
    limit, and the difference of a dipole averages to zero over its azimuth
    (test_spin_tensors), so that the integrated terms are those of the joint
    one. For g g > t t~ g the two differ by up to 25% per dipole at the
    points away from the limits."""
    data = load(path)
    for key, ratios in ray_ratios(data, factorized=True).items():
        assert ratios[-1] < max(ratios[0] / 10, 1e-6), (key, ratios)
        assert ratios[-1] < 1e-3, (key, ratios)


def dot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def kallen(a, b, c):
    return a * a + b * b + c * c - 2 * (a * b + a * c + b * c)


def transverse(p, a, b):
    """The part of p orthogonal to a and b."""
    aa, ab, bb = dot(a, a), dot(a, b), dot(b, b)
    pa, pb = dot(p, a), dot(p, b)
    det = aa * bb - ab * ab
    return p - ((pa * bb - pb * ab) / det) * a - ((pb * aa - pa * ab) / det) * b


def ncv_kernels(family, kernel, p, d, kappa):
    """(s, A, C) of the dipole d at the real momenta p as the NCV code
    (physics/schemes/cs/local.py) writes the kernels of CS and CDST: the
    dipole -g_s^2 / s (A <T_j . T_k> + 2 Re(C (e-.k_t)(e+.k_t)* / k_t^2
    <-|T_j . T_k|+>)), with masses m_j and m_k."""
    i, j, k = d["i"], d["j"], d["k"]
    pi, pj, pk = p[i], p[j], p[k]
    m_j, m_k = d["m_j"], d["m_k"]
    if family == "ff":
        s = dot(pi, pj)
        y = s / (s + dot(pi, pk) + dot(pj, pk))
        z_i = dot(pi, pk) / dot(pi + pj, pk)
        z_j = 1 - z_i
        q2 = dot(pi + pj + pk, pi + pj + pk)
        mu_j2, mu_k2 = m_j**2 / q2, m_k**2 / q2
        a = 1 - mu_j2 - mu_k2
        v = math.sqrt(max((2 * mu_k2 + a * (1 - y)) ** 2 - 4 * mu_k2, 0)) / (
            a * (1 - y)
        )
        if kernel == "qg":
            A = 2 / (1 - z_j * (1 - y)) - math.sqrt(kallen(1, mu_j2, mu_k2)) / a / v * (
                1 + z_j + m_j**2 / s
            )
            return s, A, 0.0
        zz = (z_i - (1 - v) / 2) * (z_j - (1 - v) / 2)
        zpm = (1 - v**2) / 4
        if kernel == "gg":
            A = 2 * (
                1 / (1 - z_i * (1 - y))
                + 1 / (1 - z_j * (1 - y))
                + (zz + kappa * zpm - 2) / v
            )
            return s, A, -4 * zz / v
        return s, TR / CA * (1 - 2 * (zz + kappa * zpm)) / v, 4 * TR / CA * zz / v
    if family == "fi":
        pij_pa = dot(pi + pj, pk)
        s = dot(pi, pj)
        x = 1 - s / pij_pa
        z_i = dot(pi, pk) / pij_pa
        z_j = 1 - z_i
        if kernel == "qg":
            return x * s, 2 / (2 - z_j - x) - 1 - z_j - m_j**2 / s, 0.0
        if kernel == "gg":
            A = 2 / (z_j + 1 - x) + 2 / (z_i + 1 - x) - 4 + 2 * z_i * z_j
            return x * s, A, -4 * z_i * z_j
        return x * s, TR / CA * (1 - 2 * z_i * z_j), 4 * TR / CA * z_i * z_j
    pa = pj
    if family == "if":
        pa_ik = dot(pa, pi + pk)
        x = 1 - dot(pi, pk) / pa_ik
        u = dot(pa, pi) / pa_ik
        kappa_j = 1 - m_k**2 / (2 * pa_ik) * u / ((1 - x) * (1 - u))
    else:
        x = 1 - (dot(pi, pa) + dot(pi, pk)) / dot(pa, pk)
        u, kappa_j = 0.0, 1.0
    s = x * dot(pa, pi)
    if kernel == "qg":
        return s, 2 / (1 - x + u) - 1 - x, 0.0
    if kernel == "qq":
        return s, TR / CF * (1 - 2 * x * (1 - x)), 0.0
    if kernel == "gq":
        return (
            s,
            CF / CA * (x + 2 * (1 - x) / x * kappa_j),
            -4 * CF / CA * (1 - x) / x * kappa_j,
        )
    A = 2 * (1 / (1 - x + u) - 1 + x * (1 - x) + (1 - x) / x * kappa_j)
    return s, A, -4 * (1 - x) / x * kappa_j


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_kernels(path):
    """Each dipole is that of the kernels of the NCV code, with the spin part
    through the part of p_i transverse to the Born emitter and spectator,
    at the points away from the limits, for kappa 2/3 and 0: -g_s^2 / s ((A + C / 2) <T_j . T_k> - C <T_j . T_k>(e)) with the
    unit vector e of k_t."""
    data = load(path)
    g2 = 4 * math.pi * data["setup"]["alpha_s"]
    for kappa in (2 / 3, 0.0):
        for real, points in points_by_real(data).items():
            points = [p for p in points if p["ray"]["limit"] == "none"]
            pdgs = data["reals"][real]["pdgs"]
            masses = real_masses(data, pdgs)
            out = evaluate(dipole_generator(data, real, kappa=kappa), points)
            values = dipole_values(data, points, out, jets=False)
            for q, point in enumerate(points):
                p = np.array(point["momenta"])
                for m, (n, d) in enumerate(dipoles_of(data, real)):
                    d = dict(d, m_j=masses[d["j"]], m_k=masses[d["k"]])
                    s, A, C = ncv_kernels(d["family"], d["kernel"], p, d, kappa)
                    entry = point["dipoles"][m]
                    expected = (A + C / 2) * entry["color_correlated"]
                    if entry["spin_correlated"] is not None:
                        born = np.array(entry["born_momenta"])
                        bj = d["j"] - (d["j"] > d["i"])
                        bk = d["k"] - (d["k"] > d["i"])
                        k_t = transverse(p[d["i"]], born[bj], born[bk])
                        e = k_t / math.sqrt(-dot(k_t, k_t))
                        expected -= C * (e @ np.array(entry["spin_correlated"]) @ e)
                    expected *= -g2 * d["symmetry"] / s
                    np.testing.assert_allclose(
                        values[q, m], expected, rtol=1e-9, err_msg=str((real, d))
                    )


def gate_variable(family, p, d, m_k):
    """The variable that alpha bounds and its bound at alpha = 1 (y and y_+
    for ff, 1 - x for fi, u for if, v for ii)."""
    i, j, k = d["i"], d["j"], d["k"]
    pi, pj, pk = p[i], p[j], p[k]
    if family == "ff":
        y = dot(pi, pj) / (dot(pi, pj) + dot(pi, pk) + dot(pj, pk))
        q2 = dot(pi + pj + pk, pi + pj + pk)
        m_j2 = dot(pj, pj)
        mu_k = m_k / math.sqrt(q2)
        y_plus = 1 - 2 * mu_k * (1 - mu_k) / (1 - m_j2 / q2 - mu_k**2)
        return y / y_plus
    if family == "fi":
        return dot(pi, pj) / dot(pi + pj, pk)
    if family == "if":
        return dot(pj, pi) / dot(pj, pi + pk)
    return dot(pj, pi) / dot(pj, pk)


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_alpha(path):
    """With alpha < 1 a dipole vanishes where the variable of its family is
    above alpha and is unchanged below."""
    data = load(path)
    alpha = dict(alpha_ff=0.03, alpha_fi=0.02, alpha_if=0.025, alpha_ii=0.015)
    for real, points in points_by_real(data).items():
        full = evaluate(dipole_generator(data, real), points)
        cut = evaluate(dipole_generator(data, real, **alpha), points)
        masses = real_masses(data, data["reals"][real]["pdgs"])
        inside = outside = 0
        for q, point in enumerate(points):
            p = np.array(point["momenta"])
            for m, (n, d) in enumerate(dipoles_of(data, real)):
                variable = gate_variable(d["family"], p, d, masses[d["k"]])
                bound = alpha[f"alpha_{d['family']}"]
                for name in ("color_weight", "spin_weight"):
                    if variable < bound:
                        assert cut[name][q, m] == full[name][q, m]
                    else:
                        assert cut[name][q, m] == 0.0
                inside += variable < bound
                outside += variable >= bound
        assert inside and outside


def test_invalid_dipoles():
    born = ms.PhaseSpaceMapping(
        [0.0] * 4, 1000.0, mode=ms.PhaseSpaceMapping.TChannelMode.rambo
    )
    mapping = ms.CSMapping(born, 4, 2, 3)
    with pytest.raises(ValueError):
        ms.CSDipoles([mapping], ["qq"], [1.0])
    with pytest.raises(ValueError):
        ms.CSDipoles([mapping], ["qg"], [1.0, 1.0])
    with pytest.raises(ValueError):
        ms.CSDipoles([mapping], ["qg"], [1.0], alpha_ff=0.0)
    with pytest.raises(ValueError):
        ms.CSDipoles([], [], [])
