"""CSInsertion: the insertion operator I of the Catani-Seymour dipoles.

Its parts are checked one by one: the closed form of e+ e- > q q~, the
dependence on the scale mu that (mu^2)^eps of the normalization fixes, the
alpha terms against the integral of the dipoles of CSDipoles over what their
gate removes, the kappa terms against the integral of the difference of the
dipoles at two kappa (both finite and in four dimensions), and the poles
against those of the virtual of MadLoop on the provider outputs
(MADSPACE_FKS_OUTPUTS, as for test_fks_fortran_provider.py). In a decay the
alpha terms of the dipole of a final-state quark with the decaying particle
as the spectator are checked in the same way, and the poles on the decays of
a top into a massive and a massless b.
"""

import math
import os

import numpy as np
import pytest
import torch
from fks_testing import reference
from madspace_testing import default_context

import madspace as ms

SQRT_S = 1000.0
ALPHA_S = 0.118
TOP = 173.0
CA, CF, TR = 3.0, 4.0 / 3.0, 0.5
A = ALPHA_S / (2 * math.pi)
OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
FKS_DATA = os.path.join(os.path.dirname(__file__), "test_data", "fks_integrand")


def to_torch(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


def to_numpy(x):
    return torch.from_dlpack(x).cpu().numpy()


def dot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def evaluate(generator, *inputs):
    """The outputs of a FunctionGenerator by name."""
    runtime = ms.FunctionRuntime(generator.function(), default_context())
    out = runtime.call([to_torch(x) for x in inputs])
    return dict(zip(generator.return_types().keys(), [to_numpy(t) for t in out]))


def insertion(pdgs, masses, momenta, color_correlated, ren_scale, **parameters):
    """The finite part and the poles of <I> at the momenta (batch, n, 4) with
    the color-correlated Borns (batch, b, n, n)."""
    color_correlated = np.asarray(color_correlated, dtype=float)
    batch = len(momenta)
    generator = ms.CSInsertion(
        pdgs, masses, born_orders=color_correlated.shape[1], **parameters
    )
    out = evaluate(
        generator,
        momenta,
        color_correlated,
        np.full(batch, ALPHA_S),
        np.broadcast_to(np.asarray(ren_scale, dtype=float), (batch,)).copy(),
    )
    return out["insertion"], out["poles"]


def born_points(masses, batch, seed, leptonic=True):
    """Born momenta of RAMBO in the lab frame."""
    mapping = ms.PhaseSpaceMapping(
        list(masses),
        SQRT_S,
        leptonic=leptonic,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
    )
    r = np.random.default_rng(seed).random((batch, mapping.random_dim()))
    momenta, *_ = mapping.map_forward([to_torch(r)], [])
    return to_numpy(momenta)


def pair_colors(n, j, k, batch=1):
    """A color-correlated Born with <T_j . T_k> B = 1 for the emitter j and
    spectator k alone, the ordered pair."""
    out = np.zeros((batch, 1, n, n))
    out[:, 0, j, k] = 1.0
    return out


def test_epem_qq():
    """e+ e- > q q~: <I> = a C_F B [2 / eps^2 + (3 + 2 L) / eps + 10 - pi^2 +
    3 L + L^2] with L = log(mu^2 / s) (CS)."""
    momenta = born_points([0.0] * 4, 3, seed=1)
    colors = np.zeros((3, 1, 4, 4))
    colors[:, 0, 2, 3] = colors[:, 0, 3, 2] = -CF
    colors[:, 0, 2, 2] = colors[:, 0, 3, 3] = CF
    for mu in (SQRT_S, 91.188, 3000.0):
        finite, poles = insertion(
            [-11, 11, 1, -1], [0.0] * 4, momenta, colors, mu, alpha_ff=1.0
        )
        l = math.log(mu**2 / SQRT_S**2)
        np.testing.assert_allclose(poles[:, 0, 0], 2 * A * CF, rtol=1e-14)
        np.testing.assert_allclose(poles[:, 0, 1], A * CF * (3 + 2 * l), rtol=1e-13)
        np.testing.assert_allclose(
            finite[:, 0], A * CF * (10 - math.pi**2 + 3 * l + l * l), rtol=1e-13
        )


SCALE_PROCESSES = [
    # every combination of masses of emitter and spectator, incoming ones
    ([21, 2, 6, -6, 2], [0.0, 0.0, TOP, TOP, 0.0], False),
    ([-11, 11, 6, -6, 21], [0.0, 0.0, TOP, TOP, 0.0], True),
    ([-11, 11, 2, -2, 21, 21], [0.0] * 6, True),
]


@pytest.mark.parametrize("pdgs, masses, hadronic", SCALE_PROCESSES)
def test_scale_dependence(pdgs, masses, hadronic):
    """<I> is (mu^2)^eps times a function of eps: from mu to mu' the finite
    part gains the single pole times L = log(mu'^2 / mu^2) and half the double
    pole times L^2, the single pole the double pole times L. This fixes the
    log mu terms of Gamma_j and gamma_j against the poles."""
    n = len(pdgs)
    momenta = born_points(masses, 4, seed=2, leptonic=not hadronic)
    rng = np.random.default_rng(3)
    colors = rng.normal(size=(4, 2, n, n))
    for kappa, alpha_ff in ((2.0 / 3.0, 1.0), (0.0, 0.2)):
        parameters = dict(kappa=kappa, alpha_ff=alpha_ff)
        finite, poles = insertion(pdgs, masses, momenta, colors, 200.0, **parameters)
        finite2, poles2 = insertion(pdgs, masses, momenta, colors, 50.0, **parameters)
        l = math.log(50.0**2 / 200.0**2)
        np.testing.assert_allclose(poles2[..., 0], poles[..., 0], rtol=1e-13)
        np.testing.assert_allclose(
            poles2[..., 1], poles[..., 1] + poles[..., 0] * l, rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            finite2,
            finite + poles[..., 1] * l + 0.5 * poles[..., 0] * l * l,
            rtol=1e-12,
            atol=1e-13,
        )


def g_perp(emitter, spectator):
    """-g_perp^mu nu of the polarizations of a massless emitter in the gauge
    of the spectator, contravariant, batched."""
    metric = np.diag([1.0, -1.0, -1.0, -1.0])
    pn = dot(emitter, spectator)[:, None, None]
    out = (
        -metric
        + (
            emitter[:, :, None] * spectator[:, None, :]
            + spectator[:, :, None] * emitter[:, None, :]
        )
        / pn
    )
    return out - (
        dot(spectator, spectator)[:, None, None]
        * emitter[:, :, None]
        * emitter[:, None, :]
        / pn**2
    )


class Dipole:
    """A final-final dipole of e+ e- > X at a fixed Born, its emission on
    Gauss-Legendre nodes in the random numbers of CSMapping."""

    def __init__(self, born_masses, kernel):
        self.born = ms.PhaseSpaceMapping(
            list(born_masses),
            SQRT_S,
            leptonic=True,
            mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
        )
        self.mapping = ms.CSMapping(self.born, 4, 2, 3)
        self.kernel = kernel
        self.r_born = np.array([0.3, 0.7])

    def born_momenta(self):
        momenta, *_ = self.born.map_forward([to_torch(self.r_born[None])], [])
        return to_numpy(momenta)

    def integral(self, r1_low, nodes, alpha_ff, kappa):
        """int det D / det_Born over r_1 in (r1_low, 1) at unit color,
        averaged over the azimuth: B_mu nu = -g_perp_mu nu B / 2 (k_t
        lowered)."""
        x, w = np.polynomial.legendre.leggauss(nodes)
        # r_1 = 1 - (1 - r1_low) t^2 keeps the square root of the velocity of
        # a massive spectator at y_+ (r_1 = 1) out of the integrand
        t = (x + 1) / 2
        r1 = 1 - (1 - r1_low) * t * t
        w1 = (1 - r1_low) * 2 * t * w / 2
        # r_2 on intervals that shrink towards both ends, where the soft
        # limits of i and (for g -> g g) of j leave peaks of the width of y
        ends = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]
        ends += [0.99999, 0.999999, 1.0]
        x2, v2 = np.polynomial.legendre.leggauss(16)
        r2 = np.concatenate(
            [a + (b - a) * (x2 + 1) / 2 for a, b in zip(ends[:-1], ends[1:])]
        )
        w2 = np.concatenate([(b - a) * v2 / 2 for a, b in zip(ends[:-1], ends[1:])])
        phi = (np.arange(4) + 0.5) / 4
        g1, g2, g3 = np.meshgrid(r1, r2, phi, indexing="ij")
        weight = (w1[:, None, None] * w2[None, :, None] / 4 + 0 * g3).ravel()
        r = np.concatenate(
            [
                np.tile(self.r_born, (weight.size, 1)),
                np.stack([g1.ravel(), g2.ravel(), g3.ravel()], -1),
            ],
            -1,
        )
        momenta, x1, x2, det = self.mapping.map_forward([to_torch(r)], [])
        *_, det_born = self.born.map_forward([to_torch(self.r_born[None])], [])
        batch = weight.size
        broadcast = lambda t: np.broadcast_to(to_numpy(t), (batch,)).copy()
        generator = ms.CSDipoles(
            [self.mapping], [self.kernel], [1.0], alpha_ff=alpha_ff, kappa=kappa
        )
        out = evaluate(generator, to_numpy(momenta), broadcast(x1), broadcast(x2))
        born = out["born_momenta"][:, 0]
        k_t = out["spin_vector"][:, 0] * np.array([1.0, -1.0, -1.0, -1.0])
        spin = 0.5 * np.einsum("bm,bmn,bn->b", k_t, g_perp(born[:, 2], born[:, 3]), k_t)
        dipole = (
            -4
            * math.pi
            * ALPHA_S
            * (out["color_weight"][:, 0] + out["spin_weight"][:, 0] * spin)
        )
        return np.sum(weight * to_numpy(det) / to_numpy(det_born)[0] * dipole)


# the emitter 2 and spectator 3 of e+ e- > X, by the kernels of the emitter
# with their factors in I (1/2 for g -> g g, n_f for g -> q q~)
FINAL_FINAL = {
    "q": ([0.0] * 4, [-11, 11, 1, -1], {"qg": 1.0}),
    "Q": ([0.0, 0.0, TOP, 0.0], [-11, 11, 6, -1], {"qg": 1.0}),
    "q K": ([0.0, 0.0, 0.0, TOP], [-11, 11, 1, -6], {"qg": 1.0}),
    "Q K": ([0.0, 0.0, TOP, TOP], [-11, 11, 6, -6], {"qg": 1.0}),
    "g": ([0.0] * 4, [-11, 11, 21, -1], {"gg": 0.5, "gq": 5.0}),
    "g K": ([0.0, 0.0, 0.0, TOP], [-11, 11, 21, -6], {"gg": 0.5, "gq": 5.0}),
    # a light massive spectator, near the massless limit of the closed form
    "Q b": ([0.0, 0.0, TOP, 4.7], [-11, 11, 6, -5], {"qg": 1.0}),
    "q b": ([0.0, 0.0, 0.0, 4.7], [-11, 11, 1, -5], {"qg": 1.0}),
    "g b": ([0.0, 0.0, 0.0, 4.7], [-11, 11, 21, -5], {"gg": 0.5, "gq": 5.0}),
}


@pytest.mark.parametrize("case", sorted(FINAL_FINAL))
@pytest.mark.parametrize(
    "alpha_ff, kappa", [(0.3, 2.0 / 3.0), (0.03, 0.0), (1e-4, 2.0 / 3.0)]
)
def test_alpha_final_final(case, alpha_ff, kappa):
    """What alpha_ff removes from <I> is the integral of the dipoles over
    alpha_ff y_+ < y < y_+ (r_1 > sqrt(alpha_ff) in CSMapping), here for the
    ordered pair of emitter and spectator alone at unit color."""
    masses, pdgs, kernels = FINAL_FINAL[case]
    removed = 0.0
    for kernel, factor in kernels.items():
        dipole = Dipole(masses, kernel)
        full = dipole.integral(math.sqrt(alpha_ff), 96, 1.0, kappa)
        gated = dipole.integral(math.sqrt(alpha_ff), 96, alpha_ff, kappa)
        removed += factor * (full - gated)
    momenta = Dipole(masses, "qg").born_momenta()
    colors = pair_colors(4, 2, 3)
    finite, _ = insertion(pdgs, masses, momenta, colors, 100.0, kappa=kappa)
    finite_alpha, _ = insertion(
        pdgs, masses, momenta, colors, 100.0, kappa=kappa, alpha_ff=alpha_ff
    )
    # <I> loses the integral of the dipoles, whose integral by quadrature
    # loses digits at small alpha, where a gluon j is soft
    rtol = 1e-11 if alpha_ff > 1e-3 else 1e-9
    np.testing.assert_allclose(finite_alpha - finite, -removed, rtol=rtol)


@pytest.mark.parametrize("alpha_ff", [0.9, 0.1, 1e-3, 1e-6])
def test_alpha_massless(alpha_ff):
    """Without masses alpha_ff removes C_F [log^2 alpha + 3/2 (log alpha + 1
    - alpha)] for a quark and C_A log^2 alpha - gamma_g (log(1 / alpha) - 1 +
    alpha) for a gluon (Nagy), per unit a <T_j . T_k> / T_j^2."""
    momenta = Dipole([0.0] * 4, "qg").born_momenta()
    colors = pair_colors(4, 2, 3)
    l = math.log(alpha_ff)
    gamma_g = 11 / 6 * CA - 2 / 3 * TR * 5
    expected = {
        1: CF * (l * l + 1.5 * (l + 1 - alpha_ff)),
        21: CA * l * l - gamma_g * (-l - 1 + alpha_ff),
    }
    for pdg, removed in expected.items():
        pdgs = [-11, 11, pdg, -1]
        finite, _ = insertion(pdgs, [0.0] * 4, momenta, colors, 100.0)
        finite_alpha, _ = insertion(
            pdgs, [0.0] * 4, momenta, colors, 100.0, alpha_ff=alpha_ff
        )
        t2 = CA if pdg == 21 else CF
        np.testing.assert_allclose(
            finite_alpha - finite, A * removed / t2, rtol=1e-13, atol=1e-16
        )


@pytest.mark.parametrize("case", ["q", "Q K", "g", "g K"])
def test_kappa(case):
    """The kappa terms of the final-state gluon kernels vanish in the soft
    and collinear limits: the difference of <I> at two kappa is the integral
    of the difference of the dipoles over all of their emission, zero for a
    quark emitter or a massless spectator."""
    masses, pdgs, kernels = FINAL_FINAL[case]
    difference = 0.0
    for kernel, factor in kernels.items():
        dipole = Dipole(masses, kernel)
        difference += factor * (
            dipole.integral(0.0, 64, 1.0, 0.0) - dipole.integral(0.0, 64, 1.0, 0.9)
        )
    momenta = Dipole(masses, "qg").born_momenta()
    colors = pair_colors(4, 2, 3)
    finite, _ = insertion(pdgs, masses, momenta, colors, 100.0, kappa=0.0)
    finite_kappa, _ = insertion(pdgs, masses, momenta, colors, 100.0, kappa=0.9)
    np.testing.assert_allclose(
        finite - finite_kappa, difference, rtol=1e-11, atol=1e-16
    )
    if case != "g K":
        assert abs(difference) < 1e-14


W_MASS = 80.419


def decay_born(bottom):
    """The mapping of t > b w+ at rest."""
    diagram = ms.Diagram([TOP], [bottom, W_MASS], [], [["i0", "o0", "o1"]])
    return ms.PhaseSpaceMapping(ms.Topology(diagram), TOP, leptonic=True)


class DecayDipole(Dipole):
    """The dipole of the b of t > b w+ with the top as the spectator at a
    fixed Born, its emission on Gauss-Legendre nodes."""

    def __init__(self, bottom):
        self.born = decay_born(bottom)
        self.mapping = ms.CSMapping(self.born, 3, 1, 0)
        self.kernel = "qg"
        self.r_born = np.array([0.3, 0.7])

    def integral(self, r1_low, nodes, alpha_ff, kappa):
        """int det D / det_Born over r_1 in (r1_low, 1) at unit color."""
        x, w = np.polynomial.legendre.leggauss(nodes)
        t = (x + 1) / 2
        r1 = 1 - (1 - r1_low) * t * t
        w1 = (1 - r1_low) * 2 * t * w / 2
        ends = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 1.0]
        x2, v2 = np.polynomial.legendre.leggauss(16)
        r2 = np.concatenate(
            [a + (b - a) * (x2 + 1) / 2 for a, b in zip(ends[:-1], ends[1:])]
        )
        w2 = np.concatenate([(b - a) * v2 / 2 for a, b in zip(ends[:-1], ends[1:])])
        g1, g2 = np.meshgrid(r1, r2, indexing="ij")
        weight = (w1[:, None] * w2[None, :]).ravel()
        r = np.concatenate(
            [
                np.tile(self.r_born, (weight.size, 1)),
                np.stack([g1.ravel(), g2.ravel(), np.full(weight.size, 0.3)], -1),
            ],
            -1,
        )
        momenta, x1, x2, det = self.mapping.map_forward([to_torch(r)], [])
        *_, det_born = self.born.map_forward([to_torch(self.r_born[None])], [])
        batch = weight.size
        broadcast = lambda t: np.broadcast_to(to_numpy(t), (batch,)).copy()
        generator = ms.CSDipoles(
            [self.mapping], [self.kernel], [1.0], alpha_ff=alpha_ff, kappa=kappa
        )
        out = evaluate(generator, to_numpy(momenta), broadcast(x1), broadcast(x2))
        dipole = -4 * math.pi * ALPHA_S * out["color_weight"][:, 0]
        return np.sum(weight * to_numpy(det) / to_numpy(det_born)[0] * dipole)


@pytest.mark.parametrize("bottom", [4.7, 0.0])
@pytest.mark.parametrize("alpha_ff", [0.3, 0.03, 1e-4])
def test_alpha_decay(bottom, alpha_ff):
    """What alpha_ff removes from <I> of t > b w+ is the integral of the dipole
    of the b with the top as the spectator over alpha_ff y_+ < y < y_+, the
    final-final part with the W as the spectator and the soft terms of the
    top."""
    dipole = DecayDipole(bottom)
    removed = dipole.integral(math.sqrt(alpha_ff), 96, 1.0, 2.0 / 3.0)
    removed -= dipole.integral(math.sqrt(alpha_ff), 96, alpha_ff, 2.0 / 3.0)
    momenta = dipole.born_momenta()
    colors = pair_colors(3, 1, 0)
    pdgs, masses = [6, 5, 24], [TOP, bottom, W_MASS]
    finite, _ = insertion(pdgs, masses, momenta, colors, 100.0, incoming=1)
    finite_alpha, _ = insertion(
        pdgs, masses, momenta, colors, 100.0, alpha_ff=alpha_ff, incoming=1
    )
    rtol = 1e-11 if alpha_ff > 1e-3 else 1e-9
    np.testing.assert_allclose(finite_alpha - finite, -removed, rtol=rtol)


def test_decay_pairs():
    """In a decay the decaying particle is no emitter: <I> of t > b w+ has the
    pair of the b with the top as the spectator alone, and that of a colorless
    decaying particle is that of its final state."""
    momenta = DecayDipole(4.7).born_momenta()
    pdgs, masses = [6, 5, 24], [TOP, 4.7, W_MASS]
    both, _ = insertion(pdgs, masses, momenta, pair_colors(3, 1, 0), 100.0, incoming=1)
    reverse, poles = insertion(
        pdgs, masses, momenta, pair_colors(3, 0, 1), 100.0, incoming=1
    )
    assert both[0, 0] != 0.0
    np.testing.assert_array_equal(reverse, 0.0)
    np.testing.assert_array_equal(poles, 0.0)
    # H > b b~ at rest as a decay and e+ e- > b b~ with the same b b~
    energy = 62.5
    momentum = math.sqrt(energy**2 - 4.7**2)
    b_pair = np.array([[energy, 0.0, 0.0, momentum], [energy, 0.0, 0.0, -momentum]])
    higgs = np.concatenate([[[2 * energy, 0.0, 0.0, 0.0]], b_pair])[None]
    leptons = np.array([[energy, 0.0, 0.0, energy], [energy, 0.0, 0.0, -energy]])
    pair = np.concatenate([leptons, b_pair])[None]
    colors = np.zeros((1, 1, 4, 4))
    colors[0, 0, 2, 3] = colors[0, 0, 3, 2] = -CF
    for alpha_ff in (1.0, 0.1):
        decay = insertion(
            [25, 5, -5],
            [2 * energy, 4.7, 4.7],
            higgs,
            colors[:, :, 1:, 1:],
            100.0,
            alpha_ff=alpha_ff,
            incoming=1,
        )
        scattering = insertion(
            [-11, 11, 5, -5],
            [0.0, 0.0, 4.7, 4.7],
            pair,
            colors,
            100.0,
            alpha_ff=alpha_ff,
        )
        for ours, theirs in zip(decay, scattering):
            np.testing.assert_allclose(ours, theirs, rtol=1e-14)


def provider_points(name):
    """The number of light flavors of the output, and per subprocess the
    provider, the masses of its Born and the Born kinematics of the
    reference points."""
    from madgraph.mg7.fks_provider import FortranProvider

    data = reference(os.path.join(FKS_DATA, f"{name}.json"))
    out = []
    for channel in data["channels"]:
        momenta = np.array([p["p_born"] for p in channel["points"] if "p_born" in p])
        squares = dot(momenta[0], momenta[0])
        masses = list(np.where(squares > 1.0, np.sqrt(np.abs(squares)), 0.0))
        directory = os.path.join(OUTPUTS, name, "SubProcesses", channel["subprocess"])
        provider = FortranProvider(
            directory,
            rebuild=not os.path.isfile(os.path.join(directory, "fks_provider")),
        )
        out.append((provider, masses, momenta))
    return data["n_f"], out


def assert_poles(provider, masses, momenta, n_f, rtol=1e-10):
    """The poles of <I> with the color-correlated Borns of the provider cancel
    those of its virtual at the momenta."""
    with provider:
        configurations = provider.fks_configurations
        pdgs = configurations["born_pdgs"]
        colors = provider.color_correlated(momenta)
        for mu in (91.188, 500.0):
            generator = ms.CSInsertion(
                pdgs,
                masses,
                born_orders=colors.shape[1],
                alpha_ff=0.5,
                n_f=n_f,
                incoming=configurations["nincoming"],
            )
            out = evaluate(
                generator,
                momenta,
                colors,
                np.full(len(momenta), 1 / (4 * math.pi)),
                np.full(len(momenta), mu),
            )
            values, codes = provider.virtual(momenta, mu)
            positions = [provider.position(o) for o in provider.orders["nlo"]]
            for pole, key in ((0, "double"), (1, "single")):
                theirs = values[key][:, positions]
                np.testing.assert_allclose(
                    out["poles"][..., pole],
                    -theirs,
                    rtol=rtol,
                    atol=1e-12 * np.abs(theirs).max(),
                )


@pytest.mark.skipif(not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs")
@pytest.mark.parametrize("name", ["epem_jj", "pp_epem", "pp_ttx", "t_bw"])
def test_poles(name):
    """The poles of <I> with the color-correlated Borns of the Fortran
    provider cancel those of the virtual of MadLoop, at g_s = 1 per Born order
    and its NLO order, for massless and massive partons, with the n_f light
    flavors of the output (4 here, with a massive b), and for the decay of a
    top into a massive b."""
    n_f, providers = provider_points(name)
    for provider, masses, momenta in providers:
        assert_poles(provider, masses, momenta, n_f)
        assert len(momenta)


@pytest.mark.skipif(not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs")
def test_poles_massless_decay():
    """The poles of <I> of t > b w+ with a massless b (the output t_bw_mb0 of
    make_fks_cross_section_reference.py) at Born points of the decay at
    rest, with the W mass of MadGraph (from M_Z, G_F and alpha)."""
    from madgraph.mg7.fks_provider import FortranProvider

    root = os.path.join(OUTPUTS, "t_bw_mb0", "SubProcesses")
    (name,) = [d for d in os.listdir(root) if d.startswith("P")]
    directory = os.path.join(root, name)
    provider = FortranProvider(
        directory, rebuild=not os.path.isfile(os.path.join(directory, "fks_provider"))
    )
    # the masses in the order of the Born of the output (t > w+ b)
    pdg_masses = {6: TOP, 5: 0.0, 24: 80.4190024}
    masses = [pdg_masses[pdg] for pdg in provider.fks_configurations["born_pdgs"]]
    diagram = ms.Diagram(masses[:1], masses[1:], [], [["i0", "o0", "o1"]])
    born = ms.PhaseSpaceMapping(ms.Topology(diagram), TOP, leptonic=True)
    r = np.random.default_rng(3).random((8, born.random_dim()))
    momenta, *_ = born.map_forward([to_torch(r)], [])
    # MadLoop to 3e-10 at these points
    assert_poles(provider, masses, to_numpy(momenta), 5.0, rtol=1e-9)


def test_invalid():
    with pytest.raises(ValueError):
        ms.CSInsertion([21, 21, 6, -6], [0.0, 0.0, TOP, TOP], alpha_ff=0.0)
    with pytest.raises(ValueError):
        ms.CSInsertion([6, 21, 6, 21], [TOP, 0.0, TOP, 0.0])
    with pytest.raises(ValueError):
        ms.CSInsertion([21, 21, 6, -6], [0.0, 0.0, TOP])
    with pytest.raises(ValueError):
        ms.CSInsertion([21, 21, 21, 21], [0.0, 0.0, 10.0, 0.0])
    with pytest.raises(ValueError):
        ms.CSInsertion([21, 21, 6, -6], [0.0, 0.0, TOP, TOP], born_orders=0)
    # decays: one or two incoming particles, no decaying gluon, no final-state
    # gluon with a colored decaying particle
    with pytest.raises(ValueError):
        ms.CSInsertion([6, 5, 24], [TOP, 4.7, W_MASS], incoming=3)
    with pytest.raises(ValueError):
        ms.CSInsertion([21, 1, -1], [10.0, 0.0, 0.0], incoming=1)
    with pytest.raises(ValueError, match="gluon"):
        ms.CSInsertion([6, 5, 24, 21], [TOP, 4.7, W_MASS, 0.0], incoming=1)
