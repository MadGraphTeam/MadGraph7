"""Regression tests for seeded random number generation.

These guard the seeding plumbing that reproducible event generation rests on: a
non-zero ``Context`` seed must make every random draw a deterministic function of
that seed, distinct seeds must produce independent streams, and a seed of 0 (the
run_card default) must keep the historical non-deterministic behaviour.

Threads: the raw ``FunctionRuntime`` path splits a batch over the thread pool and
takes each chunk's stream from whichever worker runs it, so it is only guaranteed
reproducible with a single-threaded pool. Reproducibility at higher thread counts
during *event generation* comes from the generator seeding each job by its logical
identity instead of by thread; that path needs a compiled matrix element and so is
not covered here.
"""

import numpy as np
import pytest

import madspace as ms

DIM = 4
BATCH = 20000


def random_function(dim=DIM):
    """Build a function whose only output is ``dim`` uniform randoms per event."""
    input_types = ms.NamedTypes([("batch_size", ms.Type([ms.batch_size]))])
    output_types = ms.NamedTypes([("r", ms.batch_float_array(dim))])
    fb = ms.FunctionBuilder(input_types, output_types)
    fb.output(0, fb.random(fb.input(0), dim))
    return fb.function()


def draw(seed, batch_size=BATCH, thread_count=1):
    """Draw one batch of randoms from a fresh context with the given seed."""
    context = ms.Context(device=ms.cpu_device(), thread_count=thread_count, seed=seed)
    runtime = ms.FunctionRuntime(random_function(), context)
    return np.array(runtime(np.array([batch_size], dtype=np.int64)), copy=True)


def test_context_exposes_seed():
    assert ms.Context(thread_count=1).seed == 0
    assert ms.Context(thread_count=1, seed=12345).seed == 12345


def test_same_seed_is_reproducible():
    assert np.array_equal(draw(12345), draw(12345))


def test_same_seed_is_reproducible_across_repeated_runs():
    first = draw(7)
    for _ in range(3):
        assert np.array_equal(first, draw(7))


def test_different_seeds_differ():
    assert not np.array_equal(draw(12345), draw(12346))


def test_seed_zero_is_non_deterministic():
    # 0 is the run_card default and must keep drawing a fresh stream each run
    assert not np.array_equal(draw(0), draw(0))


def test_draws_are_uniform():
    r = draw(2024)
    assert r.shape == (BATCH, DIM)
    assert np.all((r >= 0) & (r < 1))
    assert r.mean() == pytest.approx(0.5, abs=0.02)


def test_different_seeds_are_independent():
    # correlation of two independent streams is ~1/sqrt(n) == 0.004 here, so a
    # 0.05 bound is far outside the noise while leaving no room for a shared stream
    a, b = draw(1).ravel(), draw(2).ravel()
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.05


def test_nearby_seeds_are_independent():
    # adjacent integer seeds must not give correlated streams: seeds go through
    # std::seed_seq precisely so that seed=1 and seed=2 are not near each other
    a, b = draw(1).ravel(), draw(2).ravel()
    assert not np.array_equal(a, b)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.05
