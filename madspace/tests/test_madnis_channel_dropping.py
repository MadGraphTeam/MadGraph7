"""MadNIS channel dropping keeps the last channel of every flavor and of every
resonance (``channel_resonances``), however small its integral."""

import pytest
from madspace_testing import new_context

import madspace as ms

CM_ENERGY = 1000.0
CHANNEL_COUNT = 3

pytestmark = pytest.mark.cpu_only


def topology():
    # e+ e- > d d~ through an s-channel photon
    propagator = ms.Propagator(mass=0.0, width=0.0, pdg_id=22)
    diagram = ms.Diagram(
        [0.0] * 2,
        [0.0] * 2,
        [propagator],
        [["i0", "i1", "p0"], ["o0", "o1", "p0"]],
    )
    return ms.Topology.topologies(diagram)[0]


def build_integrands(context, tmp_path, channel_flavors):
    # RunningCoupling only reads these two keys of an LHAPDF .info file
    alphas_file = tmp_path / "alphas.info"
    alphas_file.write_text(
        "AlphaS_Qs: [1.0, 10.0, 100.0, 1000.0, 10000.0]\n"
        "AlphaS_Vals: [0.3, 0.2, 0.12, 0.1, 0.08]\n"
    )
    alphas_grid = ms.AlphaSGrid(str(alphas_file))
    alphas_grid.initialize_globals(context)
    cwnet = ms.ChannelWeightNetwork(
        channel_count=CHANNEL_COUNT, particle_count=4, prefix="test.cwnet"
    )
    cwnet.initialize_globals(context, 1)
    integrands = []
    for index in range(CHANNEL_COUNT):
        mapping = ms.PhaseSpaceMapping(
            topology(), CM_ENERGY, permutations=[[0, 1, 2, 3]], leptonic=True
        )
        # 0xBADCAFE: the built-in flat test matrix element, no library needed
        matrix_element = ms.MatrixElement(
            0xBADCAFE,
            4,
            ms.Integrand.matrix_element_inputs,
            ms.Integrand.matrix_element_outputs,
            1,
            True,
        )
        diff_xs = ms.DifferentialCrossSection(
            matrix_element=matrix_element,
            cm_energy=CM_ENERGY,
            running_coupling=None,
            energy_scale=ms.CachedScale(),
        )
        flow = ms.Flow(input_dim=mapping.random_dim(), prefix=f"test.flow{index}")
        flow.initialize_globals(context, 1)
        integrands.append(
            ms.Integrand(
                mapping,
                [diff_xs],
                adaptive_map=flow,
                running_coupling=ms.RunningCoupling(alphas_grid),
                energy_scale=ms.EnergyScale(4, 91.188),
                chan_weight_net=cwnet,
                madnis_training=True,
                pid_options=[[11, -11, 1, -1], [11, -11, 2, -2]],
                channel_indices=[index],
                active_flavors=[channel_flavors[index]],
            )
        )
    return integrands, cwnet


def active_channels_after_dropping(tmp_path, channel_flavors, channel_resonances):
    context = new_context()
    integrands, cwnet = build_integrands(context, tmp_path, channel_flavors)
    config = ms.MadnisConfig()
    config.batches = 2
    config.log_interval = 1
    config.channel_dropping_interval = 1
    # drop every channel the guards allow
    config.channel_dropping_threshold = 1.0
    config.cpu_generator_batch_size = 100
    config.batch_size_offset = 64
    config.batch_size_per_channel = 16
    config.generator_target_size_factor = 2
    opt_context = new_context()
    opt_context.copy_globals_from(context)
    training = ms.MadnisTraining(
        context, opt_context, config, integrands, cwnet, 1234, 0, channel_resonances
    )
    training.train_step(0)
    return training.active_channels()


def test_last_channel_of_a_flavor_is_kept(tmp_path):
    active = active_channels_after_dropping(tmp_path, [[0], [0], [1]], [])
    assert len(active) == 2
    assert 2 in active


def test_last_channel_of_a_resonance_is_kept(tmp_path):
    active = active_channels_after_dropping(tmp_path, [[0], [0], [0]], [[0], [0], [1]])
    assert len(active) == 2
    assert 2 in active
