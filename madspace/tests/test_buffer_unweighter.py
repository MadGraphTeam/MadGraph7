"""Partial unweighting of the MadNIS replay buffer, with signed weights.

BufferUnweighter keeps an event of a batch with probability
min(1, |w| / w_max), gives it the weight sign(w) * max(|w|, w_max), and
multiplies its ``adaptive_prob`` by its acceptance probability times
N / N_kept, so that training on the buffered events stays unbiased for any
w_max. w_max is the quantile of |w| over the batch.

It used to be the quantile of the *signed* weights. Nothing changes for a
positive integrand, but an interference has negative weights: for a batch that
is negative throughout the signed quantile is negative and every event was kept
at its full weight (no unweighting at all), and for a mostly negative batch
with a short positive tail it came out far too small. The result stayed
unbiased (the rescaling above holds for any w_max), only the buffer filled up
with far fewer effective events. The all-negative and mixed cases below fail
with the signed quantile.

Each output event carries its input ``index``, so the per-event relations are
checked exactly; the kept count and the estimator of the mean weight are
binomial, and checked to 5 sigma.
"""

import numpy as np

import madspace as ms

QUANTILE = 0.95
N = 200_000


def run(weights, prob, quantile=QUANTILE):
    types = ms.NamedTypes(
        [
            ("weight", ms.batch_float),
            ("adaptive_prob", ms.batch_float),
            ("index", ms.batch_float),
        ]
    )
    runtime = ms.FunctionRuntime(
        ms.BufferUnweighter(types, quantile).function(), ms.default_context()
    )
    index = np.arange(len(weights), dtype=np.float64)
    uw, prob_out, index_out = (
        ms.Tensor.numpy(t) for t in runtime.call([weights, prob, index])
    )
    return uw, prob_out, index_out.astype(np.int64)


def expected_max_weight(weights, quantile=QUANTILE):
    """What op_quantile picks: element floor(q N) of the sorted |w|."""
    position = min(int(quantile * len(weights)), len(weights) - 1)
    return np.sort(np.abs(weights))[position]


def check(weights, prob):
    """The per-event relations exactly, the kept count and the mean weight
    statistically; returns the kept fraction."""
    uw, prob_out, index = run(weights, prob)
    w, q = weights[index], prob[index]
    w_max = expected_max_weight(weights)
    assert w_max > 0

    # the sign is kept, the weight is max(|w|, w_max), and w_max is reached
    np.testing.assert_array_equal(np.sign(uw), np.sign(w))
    np.testing.assert_array_equal(np.abs(uw), np.maximum(np.abs(w), w_max))
    assert np.abs(uw).min() == w_max

    # the density of the kept events: acceptance probability times N / N_kept
    np.testing.assert_allclose(
        prob_out, w / uw * q * (len(weights) / len(index)), rtol=1e-14
    )

    # the kept count is a sum of Bernoulli draws with p = min(1, |w| / w_max)
    p = np.minimum(1.0, np.abs(weights) / w_max)
    assert abs(len(index) - p.sum()) < 5 * np.sqrt(np.sum(p * (1 - p))) + 1

    # sum_kept uw / N estimates the mean weight (each event contributes
    # sign(w) max(|w|, w_max) with probability p, i.e. w on average)
    spread = np.sqrt(np.sum((1 - p) * np.abs(weights) * w_max)) / len(weights)
    assert abs(uw.sum() / len(weights) - weights.mean()) < 5 * spread
    return len(index) / len(weights)


def batch(rng, scale=1.0):
    return scale * rng.lognormal(0.0, 1.0, N)


def test_positive_batch():
    rng = np.random.default_rng(1)
    weights = batch(rng)
    kept = check(weights, rng.uniform(0.5, 1.5, N))
    assert kept < 0.5


def test_negative_batch_is_unweighted():
    """A channel where the interference is negative throughout: the signed
    quantile was negative, so every event was kept at its full weight."""
    rng = np.random.default_rng(2)
    weights = -batch(rng)
    kept = check(weights, rng.uniform(0.5, 1.5, N))
    assert kept < 0.5


def test_mixed_batch_is_unweighted():
    """85% large negative weights and a short positive tail: the signed 95%
    quantile came from the positive tail, about 30 times below the |w| one,
    and nearly every negative event was kept as an over-weight."""
    rng = np.random.default_rng(3)
    negative = rng.uniform(size=N) < 0.85
    weights = np.where(negative, -batch(rng, 10.0), batch(rng, 1.0))
    kept = check(weights, rng.uniform(0.5, 1.5, N))
    assert kept < 0.5
