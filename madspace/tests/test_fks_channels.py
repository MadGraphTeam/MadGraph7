"""The FKS configurations of a Born as channels.

All FKS configurations of one Born are FKSMappings of the same Born mapping,
with the same inputs, outputs and conditions, massive and massless emitters
alike. As channels of a MultiChannelMapping or, as FKSKinematics, of a
MultiChannelFunction, each slice of the batch is that of its own
configuration.
"""

import numpy as np
import pytest
import torch
from madspace_testing import default_context

import madspace as ms

SQRT_S = 1000.0
PER_CHANNEL = 300


def to_torch(x):
    return torch.as_tensor(np.asarray(x)).to(torch.get_default_device())


def to_numpy(x):
    return torch.as_tensor(x).cpu().numpy()


def batch_sizes(count):
    # a list of batch sizes is read on the host
    return np.full(count, PER_CHANNEL, dtype=np.int64)


# (final-state masses of the Born, leptonic, i_fks, emitters): every emitter of
# t t~ g, and of u u~ g / u u~ > d d~ g with only massless ones
LAYOUTS = [
    ([173.0, 173.0], False, 4, [0, 1, 2, 3]),
    ([173.0, 173.0], True, 4, [2, 3]),
    ([0.0, 0.0], True, 4, [2, 3]),
    ([0.0, 0.0, 0.0], False, 5, [0, 1, 2, 3, 4]),
]
IDS = ["pp-ttx", "ee-ttx", "ee-uux", "pp-3j"]


def configurations(masses, leptonic, i_fks, emitters):
    born = ms.PhaseSpaceMapping(
        [0.0, 0.0] + masses,
        SQRT_S,
        leptonic=leptonic,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
    )
    return [ms.FKSMapping(born, i_fks, j) for j in emitters]


@pytest.mark.parametrize("masses,leptonic,i_fks,emitters", LAYOUTS, ids=IDS)
def test_kinematics_of_each_configuration(masses, leptonic, i_fks, emitters):
    mappings = configurations(masses, leptonic, i_fks, emitters)
    names = list(mappings[0].counter_event_types().keys())
    for mapping in mappings:
        assert list(mapping.counter_event_types().keys()) == names
    count = len(mappings)
    r = np.random.default_rng(11).random(
        (count * PER_CHANNEL, mappings[0].random_dim())
    )
    channels = ms.MultiChannelFunction([ms.FKSKinematics(m) for m in mappings])
    runtime = ms.FunctionRuntime(channels.function(), default_context())
    out = [to_numpy(t) for t in runtime.call([to_torch(r), batch_sizes(count)])]
    assert len(out) == len(names)
    for k, mapping in enumerate(mappings):
        part = slice(k * PER_CHANNEL, (k + 1) * PER_CHANNEL)
        single = ms.FunctionRuntime(
            ms.FKSKinematics(mapping).function(), default_context()
        )
        ref = [to_numpy(t) for t in single.call([to_torch(r[part])])]
        for name, ours, theirs in zip(names, out, ref):
            assert ours[part].shape == theirs.shape, name
            np.testing.assert_array_equal(ours[part], theirs, err_msg=name)
        # a massive emitter has only the soft counter-event
        massive = mapping.radiation().emitter_mass() > 0
        det = dict(zip(names, out))
        for name in ["collinear", "soft_collinear"]:
            assert (det[name + "_det"][part] > 0).any() != massive


@pytest.mark.parametrize("masses,leptonic,i_fks,emitters", LAYOUTS, ids=IDS)
def test_mapping_round_trip(masses, leptonic, i_fks, emitters):
    mappings = configurations(masses, leptonic, i_fks, emitters)
    count = len(mappings)
    channels = ms.MultiChannelMapping(mappings)
    # away from the edges of the unit cube, as in test_fks_mapping.py
    r = 0.02 + 0.96 * np.random.default_rng(12).random(
        (count * PER_CHANNEL, mappings[0].random_dim())
    )
    sizes = batch_sizes(count)
    momenta, x1, x2, det = channels.map_forward([to_torch(r)], [sizes])
    for k, mapping in enumerate(mappings):
        part = slice(k * PER_CHANNEL, (k + 1) * PER_CHANNEL)
        ref = mapping.map_forward([to_torch(r[part])])
        for ours, theirs in zip([momenta, x1, x2, det], ref):
            np.testing.assert_array_equal(to_numpy(ours)[part], to_numpy(theirs))
    # to the conditioning of the Born mappings, as in test_fks_mapping.py
    r_inv, det_inv = channels.map_inverse([momenta, x1, x2], [sizes])
    np.testing.assert_allclose(to_numpy(r_inv), r, rtol=0, atol=1e-9)
    np.testing.assert_allclose(to_numpy(det) * to_numpy(det_inv), 1.0, rtol=1e-9)
