import json
import math
import os
from glob import glob

import numpy as np
import pytest
from pytest import approx

import madspace as ms

BATCH_SIZE = 1000
CM_ENERGY = 13000.0
INVARIANT_POWER = 0.2
rng = np.random.default_rng(1234)


def load_processes():
    proc_files = glob(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "processes", "*.json")
    )
    ret = []
    for file in proc_files:
        proc_name = os.path.splitext(os.path.basename(file))[0]
        with open(file) as f:
            diagrams = json.load(f)
        ret.extend(
            pytest.param(diagram, id=f"{proc_name}-{i}")
            for i, diagram in enumerate(diagrams)
        )
    return ret


@pytest.fixture(params=load_processes())
def process(request):
    return request.param


@pytest.fixture
def mapping(process):
    diagram = ms.Diagram(
        process["incoming_masses"],
        process["outgoing_masses"],
        [ms.Propagator(*prop) for prop in process["propagators"]],
        process["vertices"],
    )
    topology = ms.Topology(diagram)
    return ms.PhaseSpaceMapping(
        topology,
        CM_ENERGY,
        permutations=process["permutations"],
        invariant_power=INVARIANT_POWER,
    )


@pytest.fixture
def masses(process):
    return process["incoming_masses"] + process["outgoing_masses"]


@pytest.fixture
def permutation_count(process):
    return len(process["permutations"])


@pytest.fixture
def mapping_with_propagators(process):
    # assign a distinct nonzero pdg_id to each distinct nonzero propagator
    # mass, so returned propagators can be mapped back to a mass
    mass_by_pid = {}
    propagators = []
    for mass, width in process["propagators"]:
        pid = 0
        if mass != 0.0:
            pid = next((p for p, m in mass_by_pid.items() if m == mass), None)
            if pid is None:
                pid = len(mass_by_pid) + 1
                mass_by_pid[pid] = mass
        propagators.append(ms.Propagator(mass, width, 0, 0.0, 0.0, pid))
    diagram = ms.Diagram(
        process["incoming_masses"],
        process["outgoing_masses"],
        propagators,
        process["vertices"],
    )
    topology = ms.Topology(diagram)
    mapping = ms.PhaseSpaceMapping(
        topology, CM_ENERGY, return_propagators=True, invariant_power=INVARIANT_POWER
    )
    return mapping, mass_by_pid


def test_process_masses(mapping, masses, permutation_count):
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    condition = (
        []
        if permutation_count <= 1
        else [rng.integers(0, permutation_count, BATCH_SIZE, dtype=np.int32)]
    )
    p_ext, x1, x2, det = mapping.map_forward([r], condition)
    m_ext_true = np.full((BATCH_SIZE, len(masses)), masses)
    m_ext = np.sqrt(
        np.maximum(0, p_ext[:, :, 0] ** 2 - np.sum(p_ext[:, :, 1:] ** 2, axis=2))
    )
    assert m_ext == approx(m_ext_true, abs=1e-3, rel=1e-3)


def test_process_incoming(mapping, masses, permutation_count):
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    condition = (
        []
        if permutation_count <= 1
        else [rng.integers(0, permutation_count, BATCH_SIZE, dtype=np.int32)]
    )
    p_ext, x1, x2, det = mapping.map_forward([r], condition)
    zeros = np.zeros(BATCH_SIZE)
    p_a = p_ext[:, 0]
    p_b = p_ext[:, 1]
    e_beam = 0.5 * CM_ENERGY

    assert p_a[:, 0] == approx(p_a[:, 3]) and p_b[:, 0] == approx(-p_b[:, 3])
    assert p_a[:, 1] == approx(zeros) and p_a[:, 2] == approx(zeros)
    assert p_b[:, 1] == approx(zeros) and p_b[:, 2] == approx(zeros)
    assert np.all(x1 >= 0) and np.all(x1 <= 1)
    assert np.all(x2 >= 0) and np.all(x2 <= 1)
    assert p_a[:, 0] == approx(e_beam * x1)
    assert p_b[:, 0] == approx(e_beam * x2)


def test_process_momentum_conservation(mapping, masses, permutation_count):
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    condition = (
        []
        if permutation_count <= 1
        else [rng.integers(0, permutation_count, BATCH_SIZE, dtype=np.int32)]
    )
    p_ext, x1, x2, det = mapping.map_forward([r], condition)
    p_in = np.sum(p_ext[:, :2], axis=1)
    p_out = np.sum(p_ext[:, 2:], axis=1)

    assert p_out == approx(p_in, rel=1e-6, abs=1e-10)


def test_process_inverse(mapping, masses, permutation_count):
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    condition = (
        []
        if permutation_count <= 1
        else [rng.integers(0, permutation_count, BATCH_SIZE, dtype=np.int32)]
    )
    *map_out, det = mapping.map_forward([r], condition)
    r_inv, det_inv = mapping.map_inverse(map_out, condition)
    assert r_inv == approx(r, abs=1e-3, rel=1e-3)
    assert det_inv == approx(1 / det, rel=1e-3)


def test_process_propagators(mapping_with_propagators):
    mapping, mass_by_pid = mapping_with_propagators
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    result = mapping.map_forward([r])

    p_out = result.momenta[:, 2:]
    n_out = p_out.shape[1]
    # pid and momentum mask are diagram-level constants, broadcast over the batch
    pids_and_masks = result.propagator_pids_and_masks[0]
    invariants = result.propagator_invariants
    virtualities = result.propagator_virtualities

    for i, pid_and_mask in enumerate(pids_and_masks):
        pid = int(pid_and_mask) >> 16
        momentum_mask = int(pid_and_mask) & 0xFFFF

        bits = ((momentum_mask >> np.arange(n_out)) & 1).astype(bool)
        p_sum = np.where(bits[None, :, None], p_out, 0.0).sum(axis=1)
        invariant_from_momenta = p_sum[:, 0] ** 2 - np.sum(p_sum[:, 1:] ** 2, axis=1)
        assert invariant_from_momenta == approx(invariants[:, i], rel=1e-5, abs=1e-5)

        if pid == 0:
            # pid == 0 stands for a massless particle (or an unused slot), so
            # mass == 0 and virtuality == invariant
            assert invariants[:, i] == approx(virtualities[:, i], rel=1e-5, abs=1e-5)
        else:
            virtuality_from_mass = invariant_from_momenta - mass_by_pid[pid] ** 2
            assert virtuality_from_mass == approx(
                virtualities[:, i], rel=1e-5, abs=1e-5
            )
