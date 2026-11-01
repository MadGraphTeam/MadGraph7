"""MultiChannelMapping: each slice of the batch goes through its own channel
mapping, forward and inverse."""

import numpy as np
import torch
from madspace_testing import to_numpy

import madspace as ms

PER_CHANNEL = 200


def test_channels_forward_and_inverse():
    masses = [0.0, 0.0, 0.0, 0.0, 0.0]
    channels = [
        ms.PhaseSpaceMapping(masses, 1000.0, mode=ms.PhaseSpaceMapping.rambo),
        ms.PhaseSpaceMapping(masses, 1000.0, mode=ms.PhaseSpaceMapping.propagator),
    ]
    mapping = ms.MultiChannelMapping(channels)
    count = len(channels)
    r = 0.02 + 0.96 * np.random.default_rng(3).random(
        (count * PER_CHANNEL, channels[0].random_dim())
    )
    # a list of batch sizes, read on the host
    sizes = np.full(count, PER_CHANNEL, dtype=np.int64)
    outputs = mapping.map_forward([torch.as_tensor(r)], [sizes])
    for k, channel in enumerate(channels):
        part = slice(k * PER_CHANNEL, (k + 1) * PER_CHANNEL)
        ref = channel.map_forward([torch.as_tensor(r[part])])
        for ours, theirs in zip(outputs, ref):
            np.testing.assert_array_equal(to_numpy(ours)[part], to_numpy(theirs))

    # the inverse returns the random numbers, each slice from its own channel
    r_inv, det_inv = mapping.map_inverse(list(outputs[:3]), [sizes])
    det = to_numpy(outputs[3])
    ok = det > 0
    assert ok.mean() > 0.9
    np.testing.assert_allclose(to_numpy(r_inv)[ok], r[ok], rtol=0, atol=1e-8)
    np.testing.assert_allclose(det[ok] * to_numpy(det_inv)[ok], 1.0, rtol=1e-8)
