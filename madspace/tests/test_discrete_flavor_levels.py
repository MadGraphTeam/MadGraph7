"""DiscreteFlavorLevels: flavours sampled as initial state, final-state content
and flavour, q(f) = q_A(i) q_B(c | i) q_C(f | c).

- Untrained (the last layers start at zero), it reproduces the flat flavour
  prior exactly when the prior is the parton densities of the initial state
  times a constant weight per flavour, as the integrand's is.
- The forward det is 1 / q of the sampled flavour, and the inverse gives q.
- The flavours are drawn with these probabilities.
- A logit bound c keeps each level within exp(+-2c) of its prior, however
  large the learned logits.
"""

import pytest
import torch
from madspace_testing import default_context

import madspace as ms

# initial states 0, 1, 2; contents 0, 1 of initial 0, content 2 of 1, 3 of 2
INITIALS = [0, 0, 0, 1, 1, 2]
CONTENTS = [0, 0, 1, 2, 2, 3]
WEIGHTS = [1.0, 2.0, 4.0, 1.0, 3.0, 2.0]
CONDITION_DIM = 3
_prefix_count = 0


def levels(prior_floor=0.0, logit_bound=0.0):
    global _prefix_count
    _prefix_count += 1
    sampler = ms.DiscreteFlavorLevels(
        INITIALS,
        CONTENTS,
        WEIGHTS,
        f"test_flavor_levels{_prefix_count}",
        CONDITION_DIM,
        prior_floor,
        logit_bound,
    )
    sampler.initialize_globals(default_context(), 1)
    return sampler


def prior(n, seed):
    """Parton densities per initial state times the flavour weights."""
    g = torch.Generator().manual_seed(seed)
    densities = torch.rand((n, 3), generator=g, dtype=torch.float64) + 0.05
    return densities[:, INITIALS] * torch.tensor(WEIGHTS, dtype=torch.float64)


def evaluate(sampler, random, condition, flavor_prior, flavors):
    count = len(WEIGHTS)
    fb = ms.FunctionBuilder(
        ms.NamedTypes(
            [
                ("r", ms.batch_float),
                ("condition", ms.batch_float_array(CONDITION_DIM)),
                ("prior", ms.batch_float_array(count)),
                ("flavor", ms.batch_int),
            ]
        ),
        ms.NamedTypes(
            [
                ("index", ms.batch_int),
                ("det", ms.batch_float),
                ("prob", ms.batch_float),
            ]
        ),
    )
    conditions = [fb.input(1), fb.input(2)]
    forward = sampler.build_forward(fb, [fb.input(0)], conditions)
    fb.output(0, forward[0])
    fb.output(1, forward["det"])
    fb.output(2, sampler.build_inverse(fb, [fb.input(3)], conditions)["det"])
    runtime = ms.FunctionRuntime(fb.function(), default_context())
    index, det, prob = (
        torch.from_dlpack(t)
        for t in runtime.call([random, condition, flavor_prior, flavors])
    )
    return index, det, prob


def inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    random = torch.rand(n, generator=g, dtype=torch.float64)
    condition = torch.rand((n, CONDITION_DIM), generator=g, dtype=torch.float64)
    flavors = torch.randint(0, len(WEIGHTS), (n,), generator=g, dtype=torch.int32)
    return random, condition, prior(n, seed), flavors


@pytest.mark.parametrize("prior_floor", [0.0, 0.1])
def test_untrained_levels_reproduce_the_flat_prior(prior_floor):
    random, condition, flavor_prior, flavors = inputs(2000, 1)
    index, det, prob = evaluate(
        levels(prior_floor), random, condition, flavor_prior, flavors
    )
    flat = flavor_prior / flavor_prior.sum(dim=1, keepdim=True)
    rows = torch.arange(len(random))
    torch.testing.assert_close(1.0 / det, flat[rows, index.long()])
    torch.testing.assert_close(prob, flat[rows, flavors.long()])


def test_forward_and_inverse_agree():
    random, condition, flavor_prior, _ = inputs(2000, 2)
    sampler = levels(0.05)
    index, det, _ = evaluate(
        sampler,
        random,
        condition,
        flavor_prior,
        torch.zeros(len(random), dtype=torch.int32),
    )
    _, _, prob = evaluate(sampler, random, condition, flavor_prior, index)
    torch.testing.assert_close(prob * det, torch.ones_like(det))


def test_flavors_follow_their_probabilities():
    n = 200_000
    random, condition, _, _ = inputs(n, 3)
    flavor_prior = prior(1, 4).expand(n, -1).contiguous()
    index, _, _ = evaluate(
        levels(), random, condition, flavor_prior, torch.zeros(n, dtype=torch.int32)
    )
    counts = torch.bincount(index.long(), minlength=len(WEIGHTS)).double()
    expected = n * flavor_prior[0] / flavor_prior[0].sum()
    assert torch.all(torch.abs(counts - expected) < 5 * expected.sqrt())


def test_a_content_shares_its_initial_state():
    with pytest.raises(ValueError, match="share their initial state"):
        ms.DiscreteFlavorLevels([0, 1], [0, 0], [1.0, 1.0], "bad", 2)
    with pytest.raises(ValueError, match="needs a condition"):
        ms.DiscreteFlavorLevels([0, 1], [0, 1], [1.0, 1.0], "bad", 0)


def test_logit_bound_limits_the_learned_correction():
    n = 20_000
    random, condition, flavor_prior, _ = inputs(n, 4)
    sampler = levels(logit_bound=1.0)
    context = default_context()
    gen = torch.Generator().manual_seed(5)
    for name in context.global_names():
        if name.startswith(f"test_flavor_levels{_prefix_count}"):
            param = context.get_global(name).torch()
            param += 30.0 * torch.randn(param.shape, generator=gen, dtype=param.dtype)
    index, det, _ = evaluate(
        sampler, random, condition, flavor_prior, torch.zeros(n, dtype=torch.int32)
    )
    flat = flavor_prior / flavor_prior.sum(dim=1, keepdim=True)
    ratio = 1.0 / det / flat[torch.arange(n), index.long()]
    # two learned levels, each within exp(+-2c) of its prior
    assert ratio.min() >= torch.exp(torch.tensor(-4.0)) * (1 - 1e-12)
    assert ratio.max() <= torch.exp(torch.tensor(4.0)) * (1 + 1e-12)
    assert ratio.max() / ratio.min() > 2.0
    with pytest.raises(ValueError, match="logit_bound"):
        ms.DiscreteFlavorLevels(INITIALS, CONTENTS, WEIGHTS, "bad", 3, 0.0, -1.0)
