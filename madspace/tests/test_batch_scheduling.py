"""Unit tests for compute_generation_batch_event_count, the pure function that
sizes a channel's next steady-state generation batch (see
generator_data.hpp/.cpp). The result is a raw (pre-unweighting) event count;
device-specific job splitting happens later, at dispatch time.
"""

import math

import madspace as ms


def make_config():
    config = ms.GeneratorConfig()
    config.finish_remaining_fraction = 0.05
    config.max_batch_fraction = 0.6
    config.batch_overshoot_sigma = 1.0
    return config


def batch_events(
    count_target,
    count_unweighted,
    count_opt,
    cross_section_count,
    cross_section_rel_error,
    config,
):
    return ms.compute_generation_batch_event_count(
        count_target,
        count_unweighted,
        count_opt,
        cross_section_count,
        cross_section_rel_error,
        config,
    )


def test_confident_channel_hits_cap():
    # A converged channel (tiny relative error), far from its target, should
    # be allowed to step close to the max_batch_fraction cap instead of
    # crawling towards the target.
    config = make_config()
    events = batch_events(100000, 0.0, 0, 10000, 0.001, config)
    # efficiency defaults to 1 (count_opt == 0), so no raw/unweighted inflation.
    expected = math.ceil(0.6 * 100000)
    assert events == expected


def test_uncertain_channel_is_throttled():
    # A channel with a large cross-section relative error should get a much
    # smaller batch than a converged one, since the safe lower bound on its
    # target sits far below the point estimate.
    config = make_config()
    confident_events = batch_events(100000, 0.0, 0, 10000, 0.001, config)
    uncertain_events = batch_events(100000, 0.0, 0, 10000, 0.5, config)
    assert uncertain_events < confident_events


def test_no_data_is_cautious_not_full_target():
    # No cross-section data yet (pessimistic rel_error) must NOT be mistaken
    # for "already done" just because the confidence bound collapses to zero
    # -- it should fall through to a small exploratory request, not the full
    # target in one blind shot. This is deliberately independent of the
    # finish_remaining_fraction short-circuit, which only looks at how much
    # is left, not at how well it's known.
    config = make_config()
    no_data_events = batch_events(100000, 0.0, 0, 0, 0.0, config)
    confident_events = batch_events(100000, 0.0, 0, 10000, 0.001, config)
    assert 0 < no_data_events < confident_events


def test_few_events_left_finishes_outright():
    # Once the remaining deficit has shrunk below finish_remaining_fraction of
    # the target, finish outright with the true deficit rather than
    # continuing to throttle.
    config = make_config()
    events = batch_events(100000, 97000.0, 0, 10000, 0.5, config)
    assert events == 3000


def test_mid_generation_is_meaningfully_throttled():
    # Away from both the cap regime and the finish-outright regime, the
    # requested amount should still be nonzero, well below the naive full
    # deficit, and shrink relative to a more certain estimate.
    config = make_config()
    events = batch_events(100000, 50000.0, 0, 10000, 0.3, config)
    assert 0 < events < 50000


def test_always_at_least_one_event():
    # Even when the confidence bound collapses to zero (no data yet, far from
    # target), the result must be strictly positive so a full exploratory
    # batch still gets dispatched at the device level.
    config = make_config()
    events = batch_events(100000, 0.0, 0, 0, 0.0, config)
    assert events >= 1


def test_never_overshoots_true_remaining():
    # The requested event count must never exceed the true remaining deficit.
    config = make_config()
    config.max_batch_fraction = 1.0
    config.batch_overshoot_sigma = 0.0
    events = batch_events(100000, 99500.0, 0, 10000, 0.001, config)
    assert events <= 500


def test_zero_efficiency_does_not_diverge():
    # count_opt > 0 but zero observed successes must not blow up the raw event
    # request (division by zero) or the caller (a real device batch) would
    # never make progress against a runaway target.
    config = make_config()
    events = batch_events(100000, 0.0, 5000, 10000, 0.5, config)
    assert events < 100000 * 5000
