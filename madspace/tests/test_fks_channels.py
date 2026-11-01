"""The FKS configurations of a Born, as channels or by a discrete index.

All FKS configurations of one Born are FKSMappings of the same Born mapping,
with the same inputs, outputs and conditions, massive and massless emitters
alike. As channels of a MultiChannelMapping or, as FKSKinematics, of a
MultiChannelFunction, each slice of the batch is that of its own
configuration. In an FKSConfigurationMapping each event is that of the
configuration of its fks_index.
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


def fks_indices(count, batch, seed):
    return np.random.default_rng(seed).integers(0, count, batch).astype(np.int32)


def to_torch_int(x):
    return torch.as_tensor(x, dtype=torch.int32).to(torch.get_default_device())


def by_configuration(index, count, compare):
    """compare(k, events) for the events of each configuration k"""
    for k in range(count):
        compare(k, np.flatnonzero(index == k))


@pytest.mark.parametrize("masses,leptonic,i_fks,emitters", LAYOUTS, ids=IDS)
def test_configuration_of_each_event(masses, leptonic, i_fks, emitters):
    configs = configurations(masses, leptonic, i_fks, emitters)
    mapping = ms.FKSConfigurationMapping(configs)
    count = len(configs)
    assert mapping.random_dim() == configs[0].random_dim()
    assert mapping.particle_count() == configs[0].particle_count()
    batch = count * PER_CHANNEL
    r = 0.02 + 0.96 * np.random.default_rng(13).random((batch, mapping.random_dim()))
    index = fks_indices(count, batch, seed=14)

    momenta, x1, x2, det = mapping.map_forward([to_torch(r)], [to_torch_int(index)])
    outputs = [to_numpy(t) for t in (momenta, x1, x2, det)]

    def same_as_configuration(k, events):
        ref = configs[k].map_forward([to_torch(r[events])])
        for ours, theirs in zip(outputs, ref):
            np.testing.assert_array_equal(ours[events], to_numpy(theirs))

    by_configuration(index, count, same_as_configuration)

    # the inverse with the configuration of each event, to the conditioning of
    # the Born mappings as in test_fks_mapping.py
    r_inv, det_inv = mapping.map_inverse([momenta, x1, x2], [to_torch_int(index)])
    np.testing.assert_allclose(to_numpy(r_inv), r, rtol=0, atol=1e-9)
    np.testing.assert_allclose(outputs[3] * to_numpy(det_inv), 1.0, rtol=1e-9)

    # the kinematics, counter-events included
    names = list(mapping.counter_event_types().keys())
    kinematics = ms.FKSKinematics(mapping)
    assert isinstance(kinematics.mapping(), ms.FKSConfigurationMapping)
    runtime = ms.FunctionRuntime(kinematics.function(), default_context())
    out = [to_numpy(t) for t in runtime.call([to_torch(r), to_torch_int(index)])]
    assert len(out) == len(names)

    def same_kinematics(k, events):
        single = ms.FunctionRuntime(
            ms.FKSKinematics(configs[k]).function(), default_context()
        )
        ref = [to_numpy(t) for t in single.call([to_torch(r[events])])]
        for name, ours, theirs in zip(names, out, ref):
            np.testing.assert_array_equal(ours[events], theirs, err_msg=name)

    by_configuration(index, count, same_kinematics)


def test_configuration_without_events():
    configs = configurations([173.0, 173.0], False, 4, [0, 1, 2, 3])
    mapping = ms.FKSConfigurationMapping(configs)
    r = np.random.default_rng(15).random((100, mapping.random_dim()))
    index = np.full(100, 2, dtype=np.int32)
    momenta, x1, x2, det = mapping.map_forward([to_torch(r)], [to_torch_int(index)])
    ref = configs[2].map_forward([to_torch(r)])
    np.testing.assert_array_equal(to_numpy(momenta), to_numpy(ref[0]))
    np.testing.assert_array_equal(to_numpy(det), to_numpy(ref[3]))


def test_configurations_of_one_born():
    born = ms.PhaseSpaceMapping([0.0, 0.0, 173.0, 173.0], SQRT_S, mode="rambo")
    other = ms.PhaseSpaceMapping([0.0, 0.0, 0.0, 0.0], SQRT_S, mode="rambo")
    with pytest.raises(ValueError):
        ms.FKSConfigurationMapping([])
    with pytest.raises(ValueError):
        ms.FKSConfigurationMapping(
            [ms.FKSMapping(born, 4, 2), ms.FKSMapping(other, 4, 2)]
        )
