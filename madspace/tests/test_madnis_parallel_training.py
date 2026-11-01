"""MultiMadnisTraining runs the trainings of several subprocesses in parallel.

Each training has an optimizer context of its own, and the trainings share the
generator contexts, which are frozen meanwhile. The CPU sample generation is
seeded per subprocess, so the trained weights do not depend on how many
trainings ran at the same time, and every generator context gets them back.
"""

import pytest
import torch

import madspace as ms

CM_ENERGY = 1000.0
SUBPROCESS_COUNT = 3

pytestmark = pytest.mark.cpu_only


def build_integrand(context, subproc_index, alphas_grid):
    pids = [11, -11, 1, -1]
    obs = ms.Observable(pids, ms.Observable.obs_eta, [[1]])
    mapping = ms.PhaseSpaceMapping(
        [0.0] * 4,
        CM_ENERGY,
        leptonic=True,
        cuts=ms.Cuts([ms.CutItem(obs, max=0.5)]),
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
    prefix = f"subproc{subproc_index}"
    flow = ms.Flow(input_dim=mapping.random_dim(), prefix=f"{prefix}.flow")
    flow.initialize_globals(context, 1)
    cwnet = ms.ChannelWeightNetwork(
        channel_count=1, particle_count=4, prefix=f"{prefix}.cwnet"
    )
    cwnet.initialize_globals(context, 1)
    integrand = ms.Integrand(
        mapping,
        [diff_xs],
        adaptive_map=flow,
        running_coupling=ms.RunningCoupling(alphas_grid),
        energy_scale=ms.EnergyScale(4, 91.188),
        chan_weight_net=cwnet,
        madnis_training=True,
        drop_cuts_and_rescale=True,
        channel_indices=[0],
    )
    return integrand, cwnet


def training_config():
    config = ms.MadnisConfig()
    config.batches = 30
    config.log_interval = 10
    config.integration_history_length = 10
    config.channel_dropping_interval = 10
    config.cpu_generator_batch_size = 200
    config.batch_size_offset = 64
    config.batch_size_per_channel = 64
    config.generator_target_size_factor = 2
    config.buffer_capacity = 0
    return config


def train(contexts, max_parallel_trainings, tmp_path):
    """Trains SUBPROCESS_COUNT subprocesses on the CPU contexts *contexts*,
    returns the active channels and the globals of every context."""
    # RunningCoupling only reads these two keys of an LHAPDF .info file
    alphas_file = tmp_path / "alphas.info"
    alphas_file.write_text(
        "AlphaS_Qs: [1.0, 10.0, 100.0, 1000.0, 10000.0]\n"
        "AlphaS_Vals: [0.3, 0.2, 0.12, 0.1, 0.08]\n"
    )
    alphas_grid = ms.AlphaSGrid(str(alphas_file))
    alphas_grid.initialize_globals(contexts[0])
    training_args = []
    for subproc_index in range(SUBPROCESS_COUNT):
        integrand, cwnet = build_integrand(contexts[0], subproc_index, alphas_grid)
        training_args.append(
            ms.TrainingArgs(
                config=training_config(),
                integrands=[integrand],
                cwnet=cwnet,
                multiplicity=4 + subproc_index,
            )
        )
    for context in contexts[1:]:
        context.copy_globals_from(contexts[0])
    initial = {
        name: contexts[0].get_global(name).torch().clone()
        for name in contexts[0].global_names()
    }

    parallel_config = ms.MadnisParallelConfig()
    parallel_config.max_parallel_trainings = max_parallel_trainings
    parallel_config.check_interval = 0.01
    # start the trainings whatever the occupancy
    parallel_config.target_occupancy = 1e6
    training = ms.MultiMadnisTraining(
        generator_contexts=contexts,
        training_args=training_args,
        parallel_config=parallel_config,
        verbosity=ms.Verbosity.silent,
        seed=7,
    )
    training.train()
    trained = [
        {name: context.get_global(name).torch().clone() for name in context.global_names()}
        for context in contexts
    ]
    for context in contexts:
        assert not context.frozen()
    return training.active_channels(), initial, trained


def test_parallel_training_matches_sequential(tmp_path):
    sequential = train([ms.Context(ms.cpu_device(), 4)], 1, tmp_path)
    parallel = train([ms.Context(ms.cpu_device(), 4)], SUBPROCESS_COUNT, tmp_path)
    for active_channels, initial, (trained,) in (sequential, parallel):
        assert active_channels == [[0]] * SUBPROCESS_COUNT
        for subproc_index in range(SUBPROCESS_COUNT):
            name = f"subproc{subproc_index}.flow"
            changed = [
                key
                for key in trained
                if key.startswith(name) and not torch.equal(trained[key], initial[key])
            ]
            assert changed, f"the flow of subprocess {subproc_index} was not trained"
    (_, _, (trained_sequential,)), (_, _, (trained_parallel,)) = sequential, parallel
    assert trained_sequential.keys() == trained_parallel.keys()
    for name in trained_sequential:
        assert torch.equal(trained_sequential[name], trained_parallel[name]), name


def test_every_generator_context_gets_the_trained_globals(tmp_path):
    contexts = [ms.Context(ms.cpu_device(), 2) for _ in range(2)]
    _, _, (trained_first, trained_second) = train(contexts, SUBPROCESS_COUNT, tmp_path)
    assert trained_first.keys() == trained_second.keys()
    for name in trained_first:
        assert torch.equal(trained_first[name], trained_second[name]), name
