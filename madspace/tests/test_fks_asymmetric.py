"""The asymmetric map of an emission off an incoming parton.

FKSRadiation with InitialMap asymmetric is FKSISRMapping 2 of the
MCcntRefactor_Sfun branch of MadGraph7: only the emitter changes its momentum
fraction, and the final state of the Born recoils against i by a Lorentz map
that keeps the direction of the spectator. Its reference data from the branch
(test_data/fks_asymmetric) are compared with those of the other emitters in
test_fks_radiation.py and test_fks_counter_events.py, its round trip and
volume in test_fks_radiation.py, test_fks_mapping.py and test_fks_channels.py.
Here the Lorentz map (fks_initial_light_cone_map) is compared with the
formulas of the branch and with 50-digit values, and the counter-events with
those of the symmetric map, which they are.
"""

import math

import mpmath
import numpy as np
import pytest
import torch
from madspace_testing import default_context

import madspace as ms

EPS = np.finfo(np.float64).eps
F = ms.batch_float
SQRT_S = 1000.0


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


def light_cone_map(p, masses, xi, y, r_phi, beam_sign, inverse=False, fixed=False):
    n = p.shape[1]
    vec = ms.batch_four_vec_array(n)

    def build(fb, *args):
        name = "fks_initial_" + ("fixed_" if fixed else "") + "light_cone_map"
        name += "_inverse" if inverse else ""
        p_in, *rest = args
        return [getattr(fb, name)(p_in, ms.Value(list(masses)), *rest, beam_sign)]

    return run(build, [vec, F, F, F], [vec], [p, xi, y, r_phi])[0]


def branch_map(p, xi, y, r_phi, beam_sign, inverse=False):
    """prepare_isr_recoil_boost and apply_isr_recoil_boost of the branch (without
    the mass shell of its inverse)."""
    xi, y, phi = xi[:, None], y[:, None], 2 * math.pi * r_phi[:, None]
    z = 1 - xi
    a = 1 + xi * (1 - y) / (2 * z)
    b_norm = -xi * np.sqrt((1 - y) * (1 + y)) / (2 * np.sqrt(z))
    b1, b2 = b_norm * np.cos(phi), b_norm * np.sin(phi)
    p_plus = p[..., 0] + beam_sign * p[..., 3]
    p_minus = p[..., 0] - beam_sign * p[..., 3]
    t1, t2 = p[..., 1], p[..., 2]
    if inverse:
        p_plus = p_plus / a
        t1, t2 = t1 - b1 * p_plus, t2 - b2 * p_plus
        p_minus = a * p_minus - 2 * (b1 * t1 + b2 * t2) - (b1**2 + b2**2) * p_plus
    else:
        p_minus = (p_minus + 2 * (b1 * t1 + b2 * t2) + (b1**2 + b2**2) * p_plus) / a
        t1, t2 = t1 + b1 * p_plus, t2 + b2 * p_plus
        p_plus = a * p_plus
    out = np.empty_like(p)
    out[..., 0] = 0.5 * (p_plus + p_minus)
    out[..., 1], out[..., 2] = t1, t2
    out[..., 3] = beam_sign * 0.5 * (p_plus - p_minus)
    return out


def random_final_state(masses, batch, seed):
    """Momenta of the given masses in random directions, energies up to SQRT_S."""
    rng = np.random.default_rng(seed)
    n = len(masses)
    k = SQRT_S * rng.random((batch, n))
    cos_theta = 2 * rng.random((batch, n)) - 1
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = 2 * math.pi * rng.random((batch, n))
    p = np.empty((batch, n, 4))
    p[..., 0] = np.sqrt(k**2 + np.asarray(masses) ** 2)
    p[..., 1] = k * sin_theta * np.cos(phi)
    p[..., 2] = k * sin_theta * np.sin(phi)
    p[..., 3] = k * cos_theta
    return p


MASSES = [0.0, 173.0, 0.0, 91.0]


@pytest.mark.parametrize("beam_sign", [1.0, -1.0])
def test_light_cone_map_matches_branch(beam_sign):
    """The formulas of the branch, directly without masses and to rounding
    with the mass shell, both ways."""
    batch = 2000
    p = random_final_state(MASSES, batch, seed=1)
    rng = np.random.default_rng(2)
    xi, y, r_phi = rng.random(batch), 2 * rng.random(batch) - 1, rng.random(batch)
    scale = p[..., 0].max(1)[:, None, None]
    for inverse in [False, True]:
        ref = branch_map(p, xi, y, r_phi, beam_sign, inverse)
        out_scale = np.maximum(scale, ref[..., 0].max(1)[:, None, None])
        linear = light_cone_map(p, [-1.0] * 4, xi, y, r_phi, beam_sign, inverse)
        np.testing.assert_allclose(linear / out_scale, ref / out_scale, atol=1e-13)
        shell = light_cone_map(p, MASSES, xi, y, r_phi, beam_sign, inverse)
        np.testing.assert_allclose(shell / out_scale, ref / out_scale, atol=1e-12)


def test_light_cone_map_of_the_born():
    """The total momentum of the final state of the Born goes to that of the
    real emission without i, and a momentum along the spectator keeps its
    direction."""
    batch = 500
    rng = np.random.default_rng(3)
    xi_hat, y, r_phi = rng.random(batch), 2 * rng.random(batch) - 1, rng.random(batch)
    x_born = (0.01 ** rng.random(batch), 0.5 * np.ones(batch))
    for beam_sign in [1.0, -1.0]:
        p_in1, p_in2, p_i, x1, x2, xi, y_out, *_ = run(
            lambda fb, *args: fb.fks_initial_asymmetric_radiate(*args, 0.0, beam_sign),
            [F] * 6,
            [ms.batch_four_vec] * 3 + [F] * 7 + [ms.batch_four_vec],
            [xi_hat, y, r_phi, np.full(batch, SQRT_S), *x_born],
        )
        np.testing.assert_array_equal(y_out, y)
        total = np.zeros((batch, 2, 4))
        total[:, 0, 0] = SQRT_S
        # along the spectator, with the energy of the Born's
        total[:, 1, 0] = 0.5 * SQRT_S
        total[:, 1, 3] = -beam_sign * 0.5 * SQRT_S
        out = light_cone_map(total, [SQRT_S, 0.0], xi, y, r_phi, beam_sign)
        recoil = p_in1 + p_in2 - p_i
        np.testing.assert_allclose(out[:, 0] / SQRT_S, recoil / SQRT_S, atol=1e-13)
        a = 1 + xi * (1 - y) / (2 * (1 - xi))
        np.testing.assert_allclose(out[:, 1, 1:3], 0.0, atol=0)
        np.testing.assert_allclose(out[:, 1, 0], 0.5 * SQRT_S / a, rtol=1e-14)
        np.testing.assert_allclose(out[:, 1, 3], -beam_sign * out[:, 1, 0], rtol=0)
        # the spectator keeps its momentum, the emitter has its own / (1 - xi)
        spectator, emitter = (p_in2, p_in1) if beam_sign > 0 else (p_in1, p_in2)
        np.testing.assert_allclose(spectator[:, 0], 0.5 * SQRT_S, rtol=1e-15)
        np.testing.assert_allclose(emitter[:, 0] * (1 - xi), 0.5 * SQRT_S, rtol=1e-14)
        x_spectator = x2 if beam_sign > 0 else x1
        np.testing.assert_array_equal(x_spectator, x_born[1 if beam_sign > 0 else 0])


def exact_map(p, mass, xi, y, phi, beam_sign, inverse=False):
    """fks_initial_light_cone_map of one momentum in 50 digits, from the given
    double inputs, with q^- from the mass shell."""
    with mpmath.workdps(50):
        p = [mpmath.mpf(float(v)) for v in p]
        xi, y, phi, m = (mpmath.mpf(float(v)) for v in (xi, y, phi, mass))
        z = 1 - xi
        a = 1 + xi * (1 - y) / (2 * z)
        b_norm = -xi * mpmath.sqrt((1 - y) * (1 + y)) / (2 * mpmath.sqrt(z))
        b = (b_norm * mpmath.cos(phi), b_norm * mpmath.sin(phi))
        q3 = beam_sign * p[3]
        k = mpmath.sqrt(p[1] ** 2 + p[2] ** 2 + p[3] ** 2)
        # on the mass shell, as the kernel takes the inputs
        e = mpmath.sqrt(k**2 + m**2)
        plus = e + q3
        if inverse:
            plus = plus / a
            t = (p[1] - b[0] * plus, p[2] - b[1] * plus)
        else:
            t = (p[1] + b[0] * plus, p[2] + b[1] * plus)
            plus = a * plus
        minus = (m**2 + t[0] ** 2 + t[1] ** 2) / plus
        return [
            float((plus + minus) / 2),
            float(t[0]),
            float(t[1]),
            float(beam_sign * (plus - minus) / 2),
        ]


@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
def test_light_cone_map_precision(inverse):
    """Momenta close to the axis, with i hard (1 - xi down to 1e-6), where A
    and |b| become large: with the mass shell each momentum keeps its relative
    precision against 50-digit values, in both directions, where the formulas
    without it lose up to A eps of the energy (the small light-cone component of
    the real emission, multiplied by A in the inverse)."""
    rng = np.random.default_rng(4)
    batch = 200
    beam_sign = 1.0
    masses = [0.0, 173.0]
    p = np.empty((batch, 2, 4))
    angle = 10.0 ** rng.uniform(-8, 0, (batch, 2))
    k = SQRT_S * 10.0 ** rng.uniform(-3, 0, (batch, 2))
    # along +z and -z
    direction = np.where(rng.random((batch, 2)) < 0.5, 1.0, -1.0)
    phi = 2 * math.pi * rng.random((batch, 2))
    p[..., 1] = k * np.sin(angle) * np.cos(phi)
    p[..., 2] = k * np.sin(angle) * np.sin(phi)
    p[..., 3] = direction * k * np.cos(angle)
    p[..., 0] = np.sqrt(k**2 + np.asarray(masses) ** 2)
    xi = 1 - 10.0 ** rng.uniform(-6, 0, batch)
    y = 1 - 2 * 10.0 ** rng.uniform(-6, 0, batch)
    r_phi = rng.random(batch)
    out = light_cone_map(p, masses, xi, y, r_phi, beam_sign, inverse)
    exact = np.array(
        [
            [
                exact_map(
                    p[b, n],
                    masses[n],
                    xi[b],
                    y[b],
                    2 * math.pi * r_phi[b],
                    beam_sign,
                    inverse,
                )
                for n in range(2)
            ]
            for b in range(batch)
        ]
    )
    deviation = np.abs(out - exact).max(-1) / exact[..., 0]
    assert deviation.max() <= 1e-13


def counter_events(initial_map, seed, j_fks, s_hat_min=0.0):
    born = ms.PhaseSpaceMapping(
        [0.0, 0.0, 173.0, 173.0],
        SQRT_S,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
    )
    mapping = ms.FKSMapping(born, 4, j_fks, s_hat_min, initial_map)
    names = list(mapping.counter_event_types().keys())
    r = np.random.default_rng(seed).random((2000, mapping.random_dim()))
    runtime = ms.FunctionRuntime(
        ms.FKSKinematics(mapping).function(), default_context()
    )
    out = [torch.as_tensor(t).cpu().numpy() for t in runtime.call([torch.as_tensor(r)])]
    return dict(zip(names, out))


@pytest.mark.parametrize("s_hat_min", [0.0, 600.0**2], ids=["uncut", "bounded"])
@pytest.mark.parametrize("j_fks", [0, 1])
def test_counter_events_are_symmetric(j_fks, s_hat_min):
    """At the same random numbers the counter-events are those of the symmetric
    map, momenta, momentum fractions and i per unit xi, where that has them
    (with a bound its range of y can leave out y = 1). Without a bound the
    collinear ones have the same Jacobians (with one, the symmetric map adds the
    width of its range of y). The soft one has xi_max = 1 - x of the emitter for
    every y, and its Jacobian the width of that range of xi."""
    sym = counter_events("symmetric", 5, j_fks, s_hat_min)
    asym = counter_events("asymmetric", 5, j_fks, s_hat_min)
    x_emitter = asym["born_x1"] if j_fks == 0 else asym["born_x2"]
    xi_min = np.zeros_like(x_emitter)
    if s_hat_min > 0:
        s_born = asym["born_x1"] * asym["born_x2"] * SQRT_S**2
        xi_min = np.maximum(1 - s_born / s_hat_min, 0.0)
    for name in ["soft", "collinear", "soft_collinear"]:
        exists = sym[name + "_det"] > 0
        assert exists.any()
        np.testing.assert_array_equal(asym[name + "_det"][exists] > 0, True)
        scale = sym[name + "_momenta"][..., 0].max(1)[:, None, None]
        np.testing.assert_allclose(
            (asym[name + "_momenta"] / scale)[exists],
            (sym[name + "_momenta"] / scale)[exists],
            atol=1e-14,
        )
        for key in ["_x1", "_x2"]:
            np.testing.assert_allclose(
                asym[name + key][exists], sym[name + key][exists], rtol=1e-14
            )
        hat_scale = np.abs(sym[name + "_p_i_hat"]).max(1)[:, None]
        np.testing.assert_allclose(
            (asym[name + "_p_i_hat"] / hat_scale)[exists],
            (sym[name + "_p_i_hat"] / hat_scale)[exists],
            atol=1e-14,
        )
    np.testing.assert_allclose(asym["soft_xi_max"], 1 - x_emitter, rtol=1e-15)
    for name in ["collinear", "soft_collinear"]:
        exists = sym[name + "_det"] > 0
        for key in ["_xi", "_xi_max"]:
            np.testing.assert_allclose(
                asym[name + key][exists], sym[name + key][exists], rtol=1e-14
            )
        if s_hat_min == 0:
            np.testing.assert_allclose(
                asym[name + "_det"][exists], sym[name + "_det"][exists], rtol=1e-13
            )
    exists = sym["soft_det"] > 0
    ratio = (1 - x_emitter - xi_min) / (sym["soft_xi_max"] - xi_min)
    np.testing.assert_allclose(
        asym["soft_det"][exists], (sym["soft_det"] * ratio)[exists], rtol=1e-13
    )


@pytest.mark.parametrize("beam_sign", [1.0, -1.0])
def test_fixed_light_cone_map(beam_sign):
    """Without the projection of the event (lepton beams) the Born in its rest
    frame at s_rec = (1 - xi) s goes to the center-of-mass frame of the real
    emission at s: fks_initial_light_cone_map and the boost by sqrt(1 - xi)
    of the branch, the total momentum to (sqrt(s), 0) - p_i of
    fks_initial_fixed_radiate, and back."""
    batch = 1000
    rng = np.random.default_rng(6)
    xi_hat, y, r_phi = rng.random(batch), 2 * rng.random(batch) - 1, rng.random(batch)
    p_i, xi, *_ = run(
        lambda fb, *args: fb.fks_initial_fixed_radiate(*args, 0.0, beam_sign),
        [F] * 4,
        [ms.batch_four_vec] + [F] * 4 + [ms.batch_four_vec],
        [xi_hat, y, r_phi, np.full(batch, SQRT_S)],
    )
    sqrt_s_rec = SQRT_S * np.sqrt(1 - xi)
    born = random_final_state(MASSES, batch, seed=7)
    out = light_cone_map(born, MASSES, xi, y, r_phi, beam_sign, fixed=True)
    lc = branch_map(born, xi, y, r_phi, beam_sign)
    plus = (lc[..., 0] + beam_sign * lc[..., 3]) * np.sqrt(1 - xi)[:, None]
    minus = (lc[..., 0] - beam_sign * lc[..., 3]) / np.sqrt(1 - xi)[:, None]
    ref = lc.copy()
    ref[..., 0] = 0.5 * (plus + minus)
    ref[..., 3] = beam_sign * 0.5 * (plus - minus)
    scale = np.maximum(born[..., 0].max(1), out[..., 0].max(1))[:, None, None]
    np.testing.assert_allclose(out / scale, ref / scale, atol=1e-12)
    total = np.zeros((batch, 1, 4))
    total[:, 0, 0] = sqrt_s_rec
    mapped = light_cone_map(total, [-1.0], xi, y, r_phi, beam_sign, fixed=True)[:, 0]
    expected = -p_i
    expected[:, 0] += SQRT_S
    np.testing.assert_allclose(mapped / SQRT_S, expected / SQRT_S, atol=1e-13)
    back = light_cone_map(
        out, MASSES, xi, y, r_phi, beam_sign, inverse=True, fixed=True
    )
    np.testing.assert_allclose(back / scale, born / scale, atol=1e-13)
