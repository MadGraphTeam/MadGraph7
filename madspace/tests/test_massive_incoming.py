"""Massive incoming particles in PhaseSpaceMapping and DifferentialCrossSection.

The mapping builds massive incoming particles (W and Z beams, muons with a
mass, massive partons) on their mass shells in the partonic frame
(com_p_in_massive). The propagator chain scatters off them with the massive
versions of its blocks (unclamped |t| range, shifted |t| sampling, massive
rapidity and ETmin bounds). Without a t-channel chain (rambo, s-channel,
2 -> 1) the final state and the Jacobian are those of massless incoming
particles. Chili and the color-ordered chain build the final state against the
massless pair with the same sum and put it on its mass shells afterwards
(incoming_on_shell), with the same final state and Jacobian too. Every mode
has the phase-space volume of massless beams, an inverse that takes the massive
momenta, and s_hat from the threshold (m1 + m2)^2 on. The flux
1/(2 sqrt(lambda(s_hat, m1^2, m2^2))) is that of DifferentialCrossSection with
the same incoming_masses.
"""

import math
from decimal import Decimal, getcontext

import numpy as np
import pytest
from madspace_testing import default_context, to_numpy
from pytest import approx

import madspace as ms

# numpy inputs, which madspace runs on the CPU
pytestmark = pytest.mark.cpu_only

PSM = ms.PhaseSpaceMapping
SQRT_S = 1000.0
M_IN = [80.4, 91.2]
M_OUT = [173.0, 173.0, 10.0]
N = 4000

MODES = [
    ("rambo", False),
    ("propagator", False),
    ("color_ordered", False),
    ("chili", False),
    ("rambo", True),
    ("propagator", True),
    ("color_ordered", True),
]
MODE_IDS = [
    f"{mode}-{'leptonic' if leptonic else 'hadronic'}" for mode, leptonic in MODES
]
# the modes whose final state does not depend on the incoming masses
SAME = [case for case in MODES if case[0] != "propagator"]
SAME_IDS = [name for name, case in zip(MODE_IDS, MODES) if case[0] != "propagator"]


def make(m_in, mode, leptonic, m_out=M_OUT, sqrt_s=SQRT_S):
    extra = {}
    if mode == "color_ordered":
        extra["color_ordered"] = ms.ColorOrderedMapping.Options(
            [0] + [i + 2 for i in range(len(m_out))] + [1]
        )
    return PSM(
        list(m_in) + list(m_out),
        sqrt_s,
        leptonic=leptonic,
        mode=getattr(PSM, mode),
        **extra,
    )


def draw(mapping, seed, n=N):
    rng = np.random.default_rng(seed)
    inputs = [rng.random((n, mapping.random_dim()))]
    if mapping.discrete_dim():
        inputs.append(rng.integers(0, 2, (n, mapping.discrete_dim()), dtype=np.int32))
    return inputs


def forward(mapping, inputs):
    p, x1, x2, det = map(to_numpy, mapping.map_forward(inputs, []))
    n = p.shape[0]
    return p, np.broadcast_to(x1, (n,)), np.broadcast_to(x2, (n,)), det


def mass2(p):
    return p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2 - p[..., 3] ** 2


def rapidity(p):
    return 0.5 * np.log((p[..., 0] + p[..., 3]) / (p[..., 0] - p[..., 3]))


@pytest.mark.parametrize("mode,leptonic", SAME, ids=SAME_IDS)
def test_same_final_state_and_jacobian(mode, leptonic):
    """Without a propagator chain only the incoming momenta differ from the
    massless mapping: on their mass shells along the beams, with the same sum
    and in the same order of rapidity, the first one ahead."""
    massless = make([0.0, 0.0], mode, leptonic)
    massive = make(M_IN, mode, leptonic)
    assert massive.massive_incoming() and not massless.massive_incoming()
    assert massive.incoming_masses() == M_IN
    inputs = draw(massive, 11)
    p0, x1_0, x2_0, det0 = forward(massless, inputs)
    p, x1, x2, det = forward(massive, inputs)
    ok = det0 > 0
    assert ok.mean() > 0.3

    # (chili leaves x and the momenta of unphysical points, of no weight, NaN)
    assert det == approx(det0, rel=1e-12)
    assert x1[ok] == approx(x1_0[ok], rel=1e-14) and x2[ok] == approx(
        x2_0[ok], rel=1e-14
    )
    assert p[ok, 2:] == approx(p0[ok, 2:], rel=1e-12, abs=1e-10 * SQRT_S)

    p_in = p[ok, :2]
    for leg, mass in enumerate(M_IN):
        assert mass2(p_in[:, leg]) == approx(mass**2, rel=1e-8)
    assert np.all(p_in[..., 1:3] == 0.0)
    assert p_in.sum(axis=1) == approx(
        p0[ok, :2].sum(axis=1), rel=1e-13, abs=1e-10 * SQRT_S
    )
    assert p_in.sum(axis=1) == approx(
        p[ok, 2:].sum(axis=1), rel=1e-12, abs=1e-10 * SQRT_S
    )
    assert np.all(rapidity(p_in[:, 0]) > rapidity(p_in[:, 1]))


@pytest.mark.parametrize("leptonic", [False, True], ids=["hadronic", "leptonic"])
def test_asymmetric_and_mirrored_beams(leptonic):
    """With a beam rapidity and mirrored events the incoming momenta are on
    their mass shells in the lab too, and the final state and the Jacobian are
    still those of the massless mapping (rambo)."""
    kwargs = dict(
        leptonic=leptonic, mode=PSM.rambo, beam_rapidity=0.2, mirror_beams=True
    )
    massless = PSM([0.0, 0.0] + M_OUT, SQRT_S, **kwargs)
    massive = PSM(M_IN + M_OUT, SQRT_S, **kwargs)
    rng = np.random.default_rng(19)
    r = rng.random((N, massive.random_dim()))
    mirror = rng.integers(0, 2, N, dtype=np.int32)
    p0, _, _, det0 = map(to_numpy, massless.map_forward([r], [mirror]))
    p, x1, x2, det = map(to_numpy, massive.map_forward([r], [mirror]))
    ok = det0 > 0
    assert det == approx(det0, rel=1e-12)
    assert p[ok, 2:] == approx(p0[ok, 2:], rel=1e-12, abs=1e-10 * SQRT_S)
    for leg, mass in enumerate(M_IN):
        assert mass2(p[ok, leg]) == approx(mass**2, rel=1e-8)
    assert p[ok, :2].sum(axis=1) == approx(
        p0[ok, :2].sum(axis=1), rel=1e-13, abs=1e-10 * SQRT_S
    )
    # a mirrored event has its first leg from beam 2
    ahead = rapidity(p[ok, 0]) > rapidity(p[ok, 1])
    assert np.all(ahead == (mirror[ok] == 0))
    result = massive.map_inverse(
        [
            p,
            np.ascontiguousarray(np.broadcast_to(x1, (N,))),
            np.ascontiguousarray(np.broadcast_to(x2, (N,))),
        ],
        [mirror],
    )
    assert to_numpy(result[0])[ok] == approx(r[ok], abs=1e-7)


def jet_cuts(n_out, pt_min=30.0, eta_max=2.0):
    O = ms.Observable
    pids = [24, -24] + [21] * n_out
    return ms.Cuts(
        [
            ms.CutItem(O(pids, O.obs_pt, [O.jet_pids]), min=pt_min),
            ms.CutItem(O(pids, O.obs_eta_abs, [O.jet_pids]), max=eta_max),
        ]
    )


@pytest.mark.parametrize("cuts", [False, True], ids=["no-cuts", "cuts"])
@pytest.mark.parametrize("leptonic", [False, True], ids=["hadronic", "leptonic"])
def test_propagator_chain_off_massive_beams(leptonic, cuts):
    """The propagator chain scatters off the massive beams: the incoming
    momenta are on their mass shells before the chain, and every |t| of it is
    a transfer from them. The phase-space volume (with cuts, whose |t| windows
    are not used for massive beams) is that of massless beams."""
    m_out = [0.0, 0.0, 0.0]
    kwargs = dict(leptonic=leptonic, mode=PSM.propagator, invariant_power=0.0)
    if cuts:
        kwargs["cuts"] = jet_cuts(len(m_out))
    massive = PSM(M_IN + m_out, SQRT_S, **kwargs)
    massless = PSM([0.0, 0.0] + m_out, SQRT_S, **kwargs)
    n = 100000
    inputs = draw(massive, 20, n)
    p, x1, x2, det = forward(massive, inputs)
    _, _, _, det0 = forward(massless, draw(massless, 21, n))
    ok = det > 0
    assert ok.mean() > 0.2
    for leg, mass in enumerate(M_IN):
        assert mass2(p[ok, leg]) == approx(mass**2, rel=1e-8)
    assert p[ok, :2].sum(axis=1) == approx(
        p[ok, 2:].sum(axis=1), rel=1e-12, abs=1e-10 * SQRT_S
    )
    error = math.hypot(det.std(), det0.std()) / math.sqrt(n)
    assert abs(det.mean() - det0.mean()) < 5.0 * error

    # the inverse, through the same (massive) |t| ranges and clamps
    result = massive.map_inverse(
        [p, np.ascontiguousarray(x1), np.ascontiguousarray(x2)], []
    )
    r_inv, det_inv = to_numpy(result[0]), to_numpy(result[-1])
    assert r_inv[ok] == approx(inputs[0][ok], abs=1e-7)
    assert det_inv[ok] * det[ok] == approx(1.0, rel=1e-6)

    # the transfer from the first beam to the particle the chain peels first
    # (the last outgoing one) reaches down to its kinematic minimum with the
    # beam masses, below which a massless beam would put it
    if not cuts and leptonic:
        q = p[ok, 0] - p[ok, 4]
        t_abs = -mass2(q)
        s = SQRT_S**2
        m1, m2 = M_IN
        lam = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)
        e1, p1 = (s + m1**2 - m2**2) / (2 * SQRT_S), math.sqrt(lam) / (2 * SQRT_S)
        # |t| = 2 (E1 E - p1 p_z) - m1^2 >= 2 E (E1 - p1) - m1^2 for a massless
        # particle of energy E, so |t| > -m1^2 always and t_min < 0 is reached
        assert t_abs.min() < 0.0
        assert np.all(t_abs >= -(m1**2) * (1 + 1e-9))
        energy = p[ok, 4, 0]
        assert np.all(t_abs >= 2 * energy * (e1 - p1) - m1**2 - 1e-9 * s)


@pytest.mark.parametrize("mode,leptonic", MODES, ids=MODE_IDS)
def test_inverse_takes_the_massive_momenta(mode, leptonic):
    mapping = make(M_IN, mode, leptonic)
    inputs = draw(mapping, 12)
    p, x1, x2, det = forward(mapping, inputs)
    ok = det > 0
    result = mapping.map_inverse(
        [p, np.ascontiguousarray(x1), np.ascontiguousarray(x2)], []
    )
    r_inv, det_inv = to_numpy(result[0]), to_numpy(result[-1])
    assert r_inv[ok] == approx(inputs[0][ok], abs=1e-7)
    assert det_inv[ok] * det[ok] == approx(1.0, rel=1e-7)


@pytest.mark.parametrize("mode", ["rambo", "propagator"])
def test_s_hat_from_the_threshold_on(mode):
    """With a massless final state the luminosity mapping starts s_hat at
    (m1 + m2)^2, and its weight is that of the massless incoming mapping above
    the threshold."""
    m_out = [0.0, 0.0]
    massive = make(M_IN, mode, False, m_out)
    massless = make([0.0, 0.0], mode, False, m_out)
    threshold = sum(M_IN) ** 2
    p, x1, x2, det = forward(massive, draw(massive, 13, 40000))
    s_hat = x1 * x2 * SQRT_S**2
    assert np.all(det > 0.0)
    assert s_hat.min() >= threshold * (1.0 - 1e-12)
    assert mass2(p[:, 0]) == approx(M_IN[0] ** 2, rel=1e-8)

    _, x1_0, x2_0, det0 = forward(massless, draw(massless, 14, 40000))
    above = det0 * (x1_0 * x2_0 * SQRT_S**2 > threshold)
    mean, mean0 = det.mean(), above.mean()
    error = math.hypot(det.std(), above.std()) / math.sqrt(len(det))
    assert abs(mean - mean0) < 5.0 * error


def test_chili_below_threshold_has_no_weight():
    """chili samples x1, x2 itself, below the threshold too: no weight there."""
    mapping = make(M_IN, "chili", False, [0.0, 0.0, 0.0])
    p, x1, x2, det = forward(mapping, draw(mapping, 15))
    below = x1 * x2 * SQRT_S**2 <= sum(M_IN) ** 2
    assert below.any() and (~below).any()
    assert np.all(det[below] == 0.0)
    above = ~below & (det > 0)
    assert above.any()
    assert mass2(p[above, 1]) == approx(M_IN[1] ** 2, rel=1e-8)


def two_to_one(m_in, mass):
    return ms.Topology(ms.Diagram(list(m_in), [mass], [], [["i0", "i1", "o0"]]))


def test_fixed_s_hat():
    """A 2 -> 1 process and fixed beams give s_hat, below the threshold an
    empty mapping."""
    assert PSM(two_to_one([80.4, 80.4], 125.0), SQRT_S).empty()
    assert make(M_IN, "rambo", True, sqrt_s=170.0, m_out=[10.0, 10.0]).empty()
    assert not make(M_IN, "rambo", True, sqrt_s=500.0, m_out=[10.0, 10.0]).empty()

    mapping = PSM(two_to_one([80.4, 80.4], 500.0), SQRT_S)
    massless = PSM(two_to_one([0.0, 0.0], 500.0), SQRT_S)
    assert not mapping.empty()
    r = np.random.default_rng(16).random((N, 1))
    p, x1, x2, det = forward(mapping, [r])
    _, _, _, det0 = forward(massless, [r])
    assert det == approx(det0, rel=1e-14)
    assert x1 * x2 * SQRT_S**2 == approx(500.0**2)
    assert mass2(p[:, 0]) == approx(80.4**2, rel=1e-8)
    assert p[:, 0] + p[:, 1] == approx(p[:, 2], rel=1e-12, abs=1e-10)


def run(fb, args):
    runtime = ms.FunctionRuntime(fb.function(), default_context())
    return [np.from_dlpack(out) for out in runtime.call(args)]


def on_shell(p, masses):
    vec = ms.batch_four_vec_array(p.shape[1])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("p", vec)]),
        ms.NamedTypes([("p_out", vec), ("above_threshold", ms.batch_float)]),
    )
    out = fb.incoming_on_shell(fb.input(0), ms.Value(list(masses)))
    fb.output(0, out[0])
    fb.output(1, out[1])
    return run(fb, [p])


def exact_pair(plus, minus, m1, m2):
    """E and p_z of the on-shell pair with the light-cone sums plus, minus, to
    40 digits"""
    getcontext().prec = 40
    plus, minus, m1, m2 = map(Decimal, (plus, minus, m1, m2))
    s = plus * minus
    root = ((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)).sqrt()
    u1 = (1 + (m1 * m1 - m2 * m2) / s + root / s) / 2
    u2 = (1 - (m1 * m1 - m2 * m2) / s + root / s) / 2
    p1 = (plus * u1, m1 * m1 / (plus * u1))
    p2 = (m2 * m2 / (minus * u2), minus * u2)
    return [[float((a + b) / 2), float((a - b) / 2)] for a, b in (p1, p2)]


@pytest.mark.parametrize(
    "masses", [(80.4, 91.2), (0.10566, 0.10566)], ids=["wz", "mumu"]
)
def test_on_shell_keeps_its_precision(masses):
    """Up to |y| = 10: each energy and p_z to rounding (light-cone components),
    and the masses 0 take the massless pair back. The way back takes the sums
    E -+ p_z of the pair from the massive momenta, whose small light-cone
    component a momentum at rapidity y holds to eps exp(2 |y|) only, which is
    why the mapping goes back in the partonic frame."""
    rng = np.random.default_rng(17)
    n = 200
    sqrt_s = 10 ** rng.uniform(np.log10(sum(masses)) + 0.01, 4.0, n)
    y = rng.uniform(-10.0, 10.0, n)
    plus, minus = sqrt_s * np.exp(y), sqrt_s * np.exp(-y)
    p = np.zeros((n, 3, 4))
    p[:, 0, 0] = p[:, 0, 3] = plus / 2
    p[:, 1, 0], p[:, 1, 3] = minus / 2, -minus / 2
    p[:, 2, 0], p[:, 2, 3] = (plus + minus) / 2, (plus - minus) / 2
    p_out, above = on_shell(p, masses)
    assert np.all(above == 1.0)
    assert np.all(p_out[:, 2] == p[:, 2])
    assert np.all(p_out[:, :2, 1:3] == 0.0)
    exact = np.array([exact_pair(a, b, *masses) for a, b in zip(plus, minus)])
    assert p_out[:, :2, 0] == approx(exact[..., 0], rel=1e-14)
    assert p_out[:, :2, 3] == approx(exact[..., 1], rel=1e-14)

    back, _ = on_shell(p_out, (0.0, 0.0))
    tolerance = 1e-14 + 1e-15 * np.exp(2.0 * np.abs(y))
    assert np.all(
        np.abs(back[:, :2] - p[:, :2]) <= tolerance[:, None, None] * p[:, 2:, :1]
    )
    p_rest = p.copy()
    p_rest[:, 0, 0] = p_rest[:, 0, 3] = sqrt_s / 2
    p_rest[:, 1, 0], p_rest[:, 1, 3] = sqrt_s / 2, -sqrt_s / 2
    back, _ = on_shell(on_shell(p_rest, masses)[0], (0.0, 0.0))
    assert back[:, :2] == approx(p_rest[:, :2], rel=1e-14, abs=0.0)

    # below the threshold
    p[:, 0, 0] = p[:, 0, 3] = p[:, 1, 0] = 0.4 * sum(masses)
    p[:, 1, 3] = -p[:, 1, 0]
    assert np.all(on_shell(p, masses)[1] == 0.0)


def test_massive_flux():
    names = ["x1", "x2", "pdf1", "pdf2", "me"]
    e_cm2 = SQRT_S**2

    def flux(masses):
        fb = ms.FunctionBuilder(
            ms.NamedTypes([(name, ms.batch_float) for name in names]),
            ms.NamedTypes([("result", ms.batch_float)]),
        )
        args = [fb.input(i) for i in range(len(names))]
        if masses is None:
            fb.output(0, fb.diff_cross_section(*args, e_cm2))
        else:
            fb.output(
                0, fb.diff_cross_section_massive(*args, e_cm2, ms.Value(list(masses)))
            )
        return run(fb, [x1, x2, pdf1, pdf2, me])[0]

    rng = np.random.default_rng(18)
    x1, x2 = rng.random((2, N)) ** 2
    pdf1, pdf2, me = rng.random((3, N))
    s_hat = x1 * x2 * e_cm2
    m1, m2 = M_IN
    lam = (s_hat - (m1 + m2) ** 2) * (s_hat - (m1 - m2) ** 2)
    above = s_hat > (m1 + m2) ** 2
    assert above.any() and (~above).any()
    gev2_to_pb = flux(None) * 2.0 * s_hat * x1 * x2 / (me * pdf1 * pdf2)
    expected = np.where(
        above,
        gev2_to_pb * me * pdf1 * pdf2 / (2.0 * x1 * x2 * np.sqrt(np.abs(lam))),
        0.0,
    )
    assert flux(M_IN) == approx(expected, rel=1e-13)
    assert flux([-m1, m2]) == approx(expected, rel=1e-13)
    assert flux([0.0, 0.0]) == approx(flux(None), rel=1e-14)


def flat_matrix_element(n_particles):
    return ms.MatrixElement(
        0xBADCAFE,
        n_particles,
        ms.Integrand.matrix_element_inputs,
        ms.Integrand.matrix_element_outputs,
        1,
        True,
    )


def cross_section(**kwargs):
    return ms.DifferentialCrossSection(
        matrix_element=flat_matrix_element(2 + len(M_OUT)),
        cm_energy=SQRT_S,
        running_coupling=None,
        energy_scale=ms.CachedScale(),
        **kwargs,
    )


def test_cross_section_incoming_masses():
    assert cross_section().incoming_masses() == []
    assert cross_section(incoming_masses=[0.0, 0.0]).incoming_masses() == []
    assert cross_section(incoming_masses=[-80.4, 91.2]).incoming_masses() == M_IN
    with pytest.raises(ValueError):
        cross_section(incoming_masses=[80.4])
    with pytest.raises(ValueError):
        cross_section(incoming_masses=M_IN, input_momentum_fraction=False)


def test_integrand_needs_the_masses_of_the_mapping():
    mapping = make(M_IN, "propagator", True)
    with pytest.raises(ValueError, match="incoming_masses"):
        ms.Integrand(mapping, [cross_section()])
    with pytest.raises(ValueError, match="incoming_masses"):
        ms.Integrand(
            make([0.0, 0.0], "propagator", True), [cross_section(incoming_masses=M_IN)]
        )
    ms.Integrand(mapping, [cross_section(incoming_masses=M_IN)])


def kernel(name, inputs, outputs, *args):
    """Runs one kernel on numpy arrays: inputs [(name, type, array)], then the
    constant arguments args after them"""
    fb = ms.FunctionBuilder(
        ms.NamedTypes([(n, t) for n, t, _ in inputs]),
        ms.NamedTypes([(n, ms.batch_float) for n in outputs]),
    )
    out = getattr(fb, name)(*[fb.input(i) for i in range(len(inputs))], *args)
    out = out if isinstance(out, (list, tuple)) else [out]
    for i in range(len(outputs)):
        fb.output(i, out[i])
    return run(fb, [np.ascontiguousarray(a) for _, _, a in inputs])


def boost_z(p, y):
    """p boosted along z by the rapidity -y (into the frame moving with y)"""
    q = p.copy()
    ch, sh = np.cosh(y), np.sinh(y)
    q[..., 0] = ch * p[..., 0] - sh * p[..., 3]
    q[..., 3] = ch * p[..., 3] - sh * p[..., 0]
    return q


def partonic_2to2(m_out, n=50000, seed=22):
    """2 -> 2 events off the massive beams, without cuts, in the partonic frame:
    pa, pb, p1, p2, x1, x2"""
    mapping = PSM(M_IN + m_out, SQRT_S, leptonic=False, mode=PSM.propagator)
    p, x1, x2, det = forward(mapping, draw(mapping, seed, n))
    ok = det > 0
    p, x1, x2 = p[ok], x1[ok], x2[ok]
    p = np.stack([boost_z(p[:, i], 0.5 * np.log(x1 / x2)) for i in range(4)], axis=1)
    return p[:, 0], p[:, 1], p[:, 2], p[:, 3], x1, x2


def t_abs(pa, p1):
    return -mass2(pa - p1)


FOUR = ms.batch_four_vec
FLOAT = ms.batch_float


def test_massive_rapidity_clamp_keeps_what_passes():
    """Every event whose outgoing lab rapidities pass the bounds has its |t|
    inside the range of t_rapidity_clamp_massive, which is narrower than the
    kinematic one for many events."""
    m_out = [173.0, 10.0]
    pa, pb, p1, p2, x1, x2 = partonic_2to2(m_out)
    n = len(x1)
    m1, m2 = np.full(n, m_out[0]), np.full(n, m_out[1])
    t_min, t_max = kernel(
        "t_inv_min_max_massive",
        [("pa", FOUR, pa), ("pb", FOUR, pb), ("m1", FLOAT, m1), ("m2", FLOAT, m2)],
        ["t_min", "t_max"],
    )
    y_boost = 0.5 * np.log(x1 / x2)
    for y_max1, y_max2 in [(1.0, 2.0), (2.5, 0.5), (-1.0, 1.5), (0.8, -1.0)]:
        lo, hi = kernel(
            "t_rapidity_clamp_massive",
            [
                ("t_min", FLOAT, t_min),
                ("t_max", FLOAT, t_max),
                ("pa", FOUR, pa),
                ("pb", FOUR, pb),
                ("m1", FLOAT, m1),
                ("m2", FLOAT, m2),
                ("x1", FLOAT, x1),
                ("x2", FLOAT, x2),
            ],
            ["lo", "hi"],
            1.0,
            y_max1,
            y_max2,
        )
        y1 = rapidity(p1) + y_boost
        y2 = rapidity(p2) + y_boost
        passes = (y_max1 < 0) | (np.abs(y1) <= y_max1)
        passes &= (y_max2 < 0) | (np.abs(y2) <= y_max2)
        t = t_abs(pa, p1)
        tolerance = 1e-9 * SQRT_S**2
        assert passes.sum() > 100
        assert np.all(t[passes] >= lo[passes] - tolerance)
        assert np.all(t[passes] <= hi[passes] + tolerance)
        assert np.mean(hi - lo < 0.9 * (t_max - t_min)) > 0.2


def test_massive_etmin_bound_keeps_what_passes():
    """The same for the ETmin lower bound of t_inv_min_max_cut_massive."""
    m_out = [173.0, 10.0]
    pa, pb, p1, p2, x1, x2 = partonic_2to2(m_out, seed=23)
    n = len(x1)
    m1, m2 = np.full(n, m_out[0]), np.full(n, m_out[1])
    et = lambda p: np.sqrt(mass2(p) + p[:, 1] ** 2 + p[:, 2] ** 2)
    for et1, et2 in [(200.0, 50.0), (180.0, 150.0), (0.0, 120.0)]:
        e1, e2 = np.full(n, et1), np.full(n, et2)
        lo, hi = kernel(
            "t_inv_min_max_cut_massive",
            [
                ("pa", FOUR, pa),
                ("pb", FOUR, pb),
                ("m1", FLOAT, m1),
                ("m2", FLOAT, m2),
                ("etmin_1", FLOAT, e1),
                ("etmin_2", FLOAT, e2),
            ],
            ["lo", "hi"],
        )
        t_min, t_max = kernel(
            "t_inv_min_max_massive",
            [("pa", FOUR, pa), ("pb", FOUR, pb), ("m1", FLOAT, m1), ("m2", FLOAT, m2)],
            ["t_min", "t_max"],
        )
        passes = (et(p1) >= et1) & (et(p2) >= et2)
        t = t_abs(pa, p1)
        assert passes.sum() > 100
        assert np.all(t[passes] >= lo[passes] - 1e-9 * SQRT_S**2)
        assert np.all(hi == t_max)
        assert np.mean(lo > t_min) > 0.2


def test_massive_kernels_have_the_massless_limit():
    """With massless beams the massive kernels give the massless ranges."""
    m_out = [173.0, 10.0]
    mapping = PSM([0.0, 0.0] + m_out, SQRT_S, leptonic=False, mode=PSM.propagator)
    p, x1, x2, det = forward(mapping, draw(mapping, 24, 20000))
    ok = det > 0
    p, x1, x2 = p[ok], x1[ok], x2[ok]
    p = np.stack([boost_z(p[:, i], 0.5 * np.log(x1 / x2)) for i in range(4)], axis=1)
    n = len(x1)
    momenta = [("pa", FOUR, p[:, 0]), ("pb", FOUR, p[:, 1])]
    masses = [("m1", FLOAT, np.full(n, m_out[0])), ("m2", FLOAT, np.full(n, m_out[1]))]
    for name in ["t_inv_min_max"]:
        massless = kernel(name, momenta + masses, ["lo", "hi"])
        massive = kernel(name + "_massive", momenta + masses, ["lo", "hi"])
        assert massive[0] == approx(massless[0], rel=1e-9, abs=1e-9)
        assert massive[1] == approx(massless[1], rel=1e-12)
    ranges = kernel("t_inv_min_max", momenta + masses, ["lo", "hi"])
    clamp_inputs = [("t_min", FLOAT, ranges[0]), ("t_max", FLOAT, ranges[1])]
    clamp_inputs += momenta + masses + [("x1", FLOAT, x1), ("x2", FLOAT, x2)]
    massless = kernel("t_rapidity_clamp", clamp_inputs, ["lo", "hi"], 1.0, 1.2, 0.7)
    massive = kernel(
        "t_rapidity_clamp_massive", clamp_inputs, ["lo", "hi"], 1.0, 1.2, 0.7
    )
    assert massive[0] == approx(massless[0], rel=1e-9, abs=1e-6)
    assert massive[1] == approx(massless[1], rel=1e-9, abs=1e-6)
    # the ETmin bound: the same lower end where the massless one narrows the
    # range (where it leaves nothing, it returns the kinematic range, and the
    # massive one, with no upper end, may still narrow it)
    etmin = [
        ("etmin_1", FLOAT, np.full(n, 200.0)),
        ("etmin_2", FLOAT, np.full(n, 50.0)),
    ]
    massless = kernel("t_inv_min_max_cut", momenta + masses + etmin, ["lo", "hi"])
    massive = kernel(
        "t_inv_min_max_cut_massive", momenta + masses + etmin, ["lo", "hi"]
    )
    narrowed = massless[0] > ranges[0]
    assert narrowed.mean() > 0.2
    assert massive[0][narrowed] == approx(massless[0][narrowed], rel=1e-9, abs=1e-6)
    assert np.all(massive[0] >= ranges[0])
