"""Discrete sampling where the option probabilities can be exactly zero.

The probabilities of the flavour sampler are parton densities (times an
active-flavour mask), and modern sets return exactly 0.0 below the c/b
thresholds and at large x, so a draw whose selected option has probability zero
is routine rather than pathological. kernel_sample_discrete_probs answers that
with det = 0, a sentinel meaning "zero-weight event". These tests pin what the
consumers of that sentinel are allowed to see, in particular the two densities
the MadNIS loss divides by.
"""

import numpy as np
import madspace as ms

N_OPTIONS = 4

# row 0: a healthy prior
# row 1: every option's density exactly zero -- the degenerate draw
# row 2: a single non-zero option
PRIOR = np.array([
    [0.3, 0.2, 0.4, 0.1],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.7],
])
RANDOM = np.array([0.42, 0.42, 0.42])
UNUSED = np.zeros(len(RANDOM))
DEGENERATE = 1


_prefix_count = 0


def sampler_and_context():
    # the default context is shared between tests, so every sampler needs its
    # own name for the probability global
    global _prefix_count
    _prefix_count += 1
    ctx = ms.default_context()
    sampler = ms.DiscreteSampler([N_OPTIONS], f"test_zero_prob{_prefix_count}_", [0])
    sampler.initialize_globals(ctx)
    return sampler, ctx


def run(builder_fn, output_names, inputs):
    sampler, ctx = sampler_and_context()
    fb = ms.FunctionBuilder(
        ms.NamedTypes([
            ("r", ms.batch_float),
            ("prior", ms.batch_float_array(N_OPTIONS)),
            ("f", ms.batch_float),
        ]),
        ms.NamedTypes([(name, ms.batch_float) for name in output_names]),
    )
    builder_fn(fb, sampler)
    runtime = ms.FunctionRuntime(fb.function(), ctx)
    return dict(zip(output_names, (ms.Tensor.numpy(t) for t in runtime.call(inputs))))


def test_forward_det_is_zero_or_at_least_one():
    """The invariant Integrand::build_channel_part splits the sentinel on.

    det is 1/p, so a det that came from an actual choice is >= 1 and the only
    other value it can take is the 0 sentinel. That is what makes max(det, 1)
    and min(det, 1) an exact split into a neutral density factor and a 0/1
    weight mask.
    """
    def build(fb, sampler):
        det = sampler.build_forward(fb, [fb.input(0)], [fb.input(1)])["det"]
        fb.output(0, det)
        fb.output(1, fb.max(det, ms.Value(1.0)))
        fb.output(2, fb.min(det, ms.Value(1.0)))

    out = run(build, ["det", "prob_factor", "weight_mask"], [RANDOM, PRIOR, UNUSED])
    det = out["det"]
    assert det[DEGENERATE] == 0.0
    regular = np.delete(np.arange(len(RANDOM)), DEGENERATE)
    assert np.all(det[regular] >= 1.0)

    # the split: bit for bit exact on a regular det, neutral/voiding on the
    # sentinel
    assert np.array_equal(out["prob_factor"][regular], det[regular])
    assert np.array_equal(out["weight_mask"][regular], np.ones(len(regular)))
    assert out["prob_factor"][DEGENERATE] == 1.0
    assert out["weight_mask"][DEGENERATE] == 0.0


def test_inverse_det_is_never_zero():
    """The inverse det is the MadNIS g, which the loss divides by.

    A zero there would put 0/0 into f/g, so a draw the sampler could not have
    made returns the marginal density over the discrete step, 1, instead of the
    zero probability.
    """
    def build(fb, sampler):
        index = sampler.build_forward(fb, [fb.input(0)], [fb.input(1)])[0]
        fb.output(0, sampler.build_inverse(fb, [index], [fb.input(1)])["det"])

    out = run(build, ["g"], [RANDOM, PRIOR, UNUSED])
    assert np.all(out["g"] > 0.0)
    assert out["g"][DEGENERATE] == 1.0


def test_madnis_terms_stay_finite_for_a_degenerate_draw():
    """End of the chain: the three MadnisLoss kernels, wired as the integrand
    wires them. Without the split, adaptive_prob is +inf and madnis_softclip
    returns NaN for the whole batch.
    """
    mean, norm, threshold = ms.Value(0.5), ms.Value(1.0), ms.Value(100.0)

    def build(fb, sampler):
        r, prior, f_raw = fb.input(0), fb.input(1), fb.input(2)
        forward = sampler.build_forward(fb, [r], [prior])
        det = forward["det"]
        g = sampler.build_inverse(fb, [forward[0]], [prior])["det"]
        # Integrand: adaptive_prob = norm / max(det, 1), weight *= min(det, 1)
        q = fb.div(norm, fb.max(det, ms.Value(1.0)))
        f = fb.mul(f_raw, fb.min(det, ms.Value(1.0)))
        fb.output(0, f)
        fb.output(1, q)
        fb.output(2, fb.madnis_abs_weight(f, q))
        fb.output(3, fb.madnis_softclip(f, q, mean, threshold))
        fb.output(4, fb.madnis_variance(f, g, q, mean))

    f_raw = np.array([1.5, 2.5, 3.5])
    out = run(
        build,
        ["f", "q", "abs_weight", "softclip", "variance"],
        [RANDOM, PRIOR, f_raw],
    )
    for name, values in out.items():
        assert np.all(np.isfinite(values)), f"{name} is not finite: {values}"

    # the degenerate row contributes nothing to the integral estimate, and its
    # variance term is the ordinary penalty for a sample with zero integrand
    assert out["f"][DEGENERATE] == 0.0
    assert out["q"][DEGENERATE] == 1.0
    assert out["abs_weight"][DEGENERATE] == 0.0
    # every other row is untouched by the split
    keep = np.delete(np.arange(len(f_raw)), DEGENERATE)
    assert np.all(out["f"][keep] == f_raw[keep])
