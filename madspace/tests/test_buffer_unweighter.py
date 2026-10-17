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

import torch
from madspace_testing import default_context, generator, rand, randn

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
        ms.BufferUnweighter(types, quantile).function(), default_context()
    )
    index = torch.arange(len(weights), dtype=torch.float64)
    uw, prob_out, index_out = (
        torch.from_dlpack(t) for t in runtime.call([weights, prob, index])
    )
    return uw, prob_out, index_out.long()


def expected_max_weight(weights, quantile=QUANTILE):
    """What op_quantile picks: element floor(q N) of the sorted |w|."""
    position = min(int(quantile * len(weights)), len(weights) - 1)
    return torch.sort(weights.abs()).values[position]


def check(weights, prob):
    """The per-event relations exactly, the kept count and the mean weight
    statistically; returns the kept fraction."""
    uw, prob_out, index = run(weights, prob)
    w, q = weights[index], prob[index]
    w_max = expected_max_weight(weights)
    assert w_max > 0

    # the sign is kept, the weight is max(|w|, w_max), and w_max is reached
    assert torch.equal(torch.sign(uw), torch.sign(w))
    assert torch.equal(uw.abs(), torch.maximum(w.abs(), w_max))
    assert uw.abs().min() == w_max

    # the density of the kept events: acceptance probability times N / N_kept
    torch.testing.assert_close(
        prob_out, w / uw * q * (len(weights) / len(index)), rtol=1e-14, atol=0
    )

    # the kept count is a sum of Bernoulli draws with p = min(1, |w| / w_max)
    p = torch.clamp(weights.abs() / w_max, max=1.0)
    assert abs(len(index) - p.sum()) < 5 * torch.sqrt(torch.sum(p * (1 - p))) + 1

    # sum_kept uw / N estimates the mean weight (each event contributes
    # sign(w) max(|w|, w_max) with probability p, i.e. w on average)
    spread = torch.sqrt(torch.sum((1 - p) * weights.abs() * w_max)) / len(weights)
    assert abs(uw.sum() / len(weights) - weights.mean()) < 5 * spread
    return len(index) / len(weights)


def batch(g, scale=1.0):
    return scale * torch.exp(randn(N, generator=g))


def test_positive_batch():
    g = generator(1)
    weights = batch(g)
    kept = check(weights, 0.5 + rand(N, generator=g))
    assert kept < 0.5


def test_negative_batch_is_unweighted():
    """A channel where the interference is negative throughout: the signed
    quantile was negative, so every event was kept at its full weight."""
    g = generator(2)
    weights = -batch(g)
    kept = check(weights, 0.5 + rand(N, generator=g))
    assert kept < 0.5


def test_mixed_batch_is_unweighted():
    """85% large negative weights and a short positive tail: the signed 95%
    quantile came from the positive tail, about 30 times below the |w| one,
    and nearly every negative event was kept as an over-weight."""
    g = generator(3)
    negative = rand(N, generator=g) < 0.85
    weights = torch.where(negative, -batch(g, 10.0), batch(g, 1.0))
    kept = check(weights, 0.5 + rand(N, generator=g))
    assert kept < 0.5
