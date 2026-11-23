"""A MadNIS training without buffered steps (buffered_steps_fraction 0, the
[amplicol] default) allocates and fills no replay buffer, whatever its
buffer_capacity."""

import pytest
from madspace_testing import new_context
from test_madnis_channel_dropping import build_integrands

import madspace as ms

pytestmark = pytest.mark.cpu_only


def training(tmp_path, fraction):
    context = new_context()
    integrands, cwnet = build_integrands(context, tmp_path, [[0], [0], [1]])
    config = ms.MadnisConfig()
    config.batches = 2
    config.cpu_generator_batch_size = 100
    config.batch_size_offset = 64
    config.batch_size_per_channel = 16
    config.buffer_capacity = 60000
    config.minimum_buffer_size = 10
    config.buffer_skip_batches = 0
    config.buffered_steps_fraction = fraction
    opt_context = new_context()
    opt_context.copy_globals_from(context)
    return ms.MadnisTraining(context, opt_context, config, integrands, cwnet, 1234)


def test_no_buffered_steps_means_no_buffer(tmp_path):
    train = training(tmp_path, 0.0)
    assert train.config.buffer_capacity == 0
    # trains as before
    train.train_step(0)
    train.train_step(1)


def test_buffered_steps_keep_the_buffer(tmp_path):
    train = training(tmp_path, 0.5)
    assert train.config.buffer_capacity == 60000
