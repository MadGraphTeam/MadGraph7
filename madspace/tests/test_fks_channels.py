"""The FKS configurations of a Born, as channels or by a discrete index.

All FKS configurations of one Born are FKSMappings of the same Born mapping,
with the same inputs, outputs and conditions, massive and massless emitters
alike. As channels of a MultiChannelMapping or, as FKSKinematics, of a
MultiChannelFunction, each slice of the batch is that of its own
configuration. An FKSConfigurationMapping builds all configurations of a Born
in one pass, with i, j and the emitter of each event read at its fks_index:
event by event, everything is that of the FKSMapping of its configuration.
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


def born_mapping(masses, leptonic, **options):
    return ms.PhaseSpaceMapping(
        [0.0, 0.0] + masses,
        SQRT_S,
        leptonic=leptonic,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
        **options,
    )


def configurations(masses, leptonic, i_fks, emitters, **options):
    born = born_mapping(masses, leptonic, **options)
    return [ms.FKSMapping(born, i_fks, j) for j in emitters]


def configuration_mapping(masses, leptonic, i_fks, emitters, **options):
    return ms.FKSConfigurationMapping(
        born_mapping(masses, leptonic, **options), [(i_fks, j) for j in emitters]
    )


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


def as_numpy(values):
    return [to_numpy(t) for t in values]


def compare_events(mapping, configs, r, index, conditions=()):
    """Forward, inverse and the kinematics of mapping at the fks_index of each
    event against the FKSMapping of its configuration, the other conditions
    (one array per event each) passed to both."""
    count = len(configs)
    extra = [to_torch_int(c) for c in conditions]
    momenta, x1, x2, det = mapping.map_forward(
        [to_torch(r)], extra + [to_torch_int(index)]
    )
    outputs = as_numpy((momenta, x1, x2, det))

    # the references on the whole batch, so that every event is computed in the
    # same SIMD lane, compared at the events of their configuration
    def same_as_configuration(k, events):
        ref = configs[k].map_forward([to_torch(r)], extra)
        for ours, theirs in zip(outputs, as_numpy(ref)):
            np.testing.assert_array_equal(ours[events], theirs[events])

    by_configuration(index, count, same_as_configuration)

    # the inverse with the configuration of each event, to the conditioning of
    # the Born mappings as in test_fks_mapping.py
    r_inv, det_inv = mapping.map_inverse(
        [momenta, x1, x2], extra + [to_torch_int(index)]
    )
    np.testing.assert_allclose(to_numpy(r_inv), r, rtol=0, atol=1e-9)
    np.testing.assert_allclose(outputs[3] * to_numpy(det_inv), 1.0, rtol=1e-9)

    # the kinematics, counter-events included
    names = list(mapping.counter_event_types().keys())
    assert names == list(configs[0].counter_event_types().keys())
    kinematics = ms.FKSKinematics(mapping)
    assert isinstance(kinematics.mapping(), ms.FKSConfigurationMapping)
    runtime = ms.FunctionRuntime(kinematics.function(), default_context())
    out = as_numpy(runtime.call([to_torch(r)] + extra + [to_torch_int(index)]))
    assert len(out) == len(names)

    def same_kinematics(k, events):
        single = ms.FunctionRuntime(
            ms.FKSKinematics(configs[k]).function(), default_context()
        )
        ref = as_numpy(single.call([to_torch(r)] + extra))
        for name, ours, theirs in zip(names, out, ref):
            np.testing.assert_array_equal(ours[events], theirs[events], err_msg=name)

    by_configuration(index, count, same_kinematics)


@pytest.mark.parametrize("masses,leptonic,i_fks,emitters", LAYOUTS, ids=IDS)
def test_configuration_of_each_event(masses, leptonic, i_fks, emitters):
    configs = configurations(masses, leptonic, i_fks, emitters)
    mapping = configuration_mapping(masses, leptonic, i_fks, emitters)
    count = len(configs)
    assert mapping.configurations() == [(i_fks, j) for j in emitters]
    assert mapping.random_dim() == configs[0].random_dim()
    assert mapping.particle_count() == configs[0].particle_count()
    batch = count * PER_CHANNEL
    r = 0.02 + 0.96 * np.random.default_rng(13).random((batch, mapping.random_dim()))
    compare_events(mapping, configs, r, fks_indices(count, batch, seed=14))


@pytest.mark.parametrize(
    "masses,i_fks,emitters",
    [([173.0, 173.0], 4, [0, 1, 2, 3]), ([0.0, 0.0, 0.0], 5, [0, 1, 2, 3, 4])],
    ids=["pp-ttx", "pp-3j"],
)
def test_mirrored_and_boosted_beams(masses, i_fks, emitters):
    """mirror_index and the fks_index of each event together, with beams of
    different energies"""
    options = dict(mirror_beams=True, beam_rapidity=0.3)
    configs = configurations(masses, False, i_fks, emitters, **options)
    mapping = configuration_mapping(masses, False, i_fks, emitters, **options)
    count = len(configs)
    batch = count * PER_CHANNEL
    rng = np.random.default_rng(16)
    r = 0.02 + 0.96 * rng.random((batch, mapping.random_dim()))
    mirror = rng.integers(0, 2, batch).astype(np.int32)
    compare_events(mapping, configs, r, fks_indices(count, batch, 17), [mirror])


def test_configuration_without_events():
    masses, emitters = [173.0, 173.0], [0, 1, 2, 3]
    configs = configurations(masses, False, 4, emitters)
    mapping = configuration_mapping(masses, False, 4, emitters)
    r = np.random.default_rng(15).random((100, mapping.random_dim()))
    index = np.full(100, 2, dtype=np.int32)
    momenta, x1, x2, det = mapping.map_forward([to_torch(r)], [to_torch_int(index)])
    ref = configs[2].map_forward([to_torch(r)])
    np.testing.assert_array_equal(to_numpy(momenta), to_numpy(ref[0]))
    np.testing.assert_array_equal(to_numpy(det), to_numpy(ref[3]))


def test_invalid_configurations():
    born = born_mapping([173.0, 173.0], False)
    leptonic = born_mapping([173.0, 173.0], True)
    O = ms.Observable
    floor = ms.Cuts([ms.CutItem(O([21, 21, 6, -6], O.obs_sqrt_s, []), min=400.0)])
    with_cuts = born_mapping([173.0, 173.0], False, cuts=floor)
    for args in [
        (born, []),
        (born, [(4, 4)]),
        (born, [(1, 2)]),
        (born, [(5, 2)]),
        (leptonic, [(4, 0)]),
        (with_cuts, [(4, 2)]),
    ]:
        with pytest.raises(ValueError):
            ms.FKSConfigurationMapping(*args)
    # a floor needs the sqrt_s_hat_min condition of the Born mapping, unless
    # every emitter is incoming
    with pytest.raises(ValueError):
        ms.FKSConfigurationMapping(born, [(4, 0), (4, 2)], sqrt_s_hat_floor=400.0)
    ms.FKSConfigurationMapping(born, [(4, 0), (4, 1)], sqrt_s_hat_floor=400.0)
    with_condition = born_mapping([173.0, 173.0], False, sqrt_s_hat_min_condition=True)
    mapping = ms.FKSConfigurationMapping(
        with_condition, [(4, 0), (4, 2)], sqrt_s_hat_floor=400.0
    )
    assert mapping.sqrt_s_hat_floor() == 400.0
