import json
import os
import tempfile

import numpy as np
import pytest
from pytest import approx

import madspace as ms

HERE = os.path.dirname(os.path.realpath(__file__))
# Copy of PROC_tt_decay/SubProcesses/subprocesses.json (a hadronically decaying
# ttbar process), checked in here since the PROC_tt_decay output directory
# itself isn't guaranteed to exist/persist.
TT_DECAY_SUBPROCESSES = os.path.join(
    HERE, "test_data", "lhe", "tt_decay_subprocesses.json"
)

CM_ENERGY = 13000.0
BW_CUTOFF = 15.0
PARTICLE_COUNT = 8  # 2 incoming + 6 outgoing, for the tt_decay process below
RESONANCE_PDGS = {24, -24, 6, -6}
POLE_MASS_WIDTH = {6: (172.5, 1.4915), 24: (80.379, 2.085)}

# Masses/widths for the particles appearing in PROC_tt_decay's subprocesses.json.
# 81/-81 are MadGraph's generic light-quark placeholder PDG IDs.
MASSES = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 4.7, 6: 172.5, 21: 0.0, 24: 80.379}
WIDTHS = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.4915, 21: 0.0, 24: 2.085}


def clean_pid(pid):
    pid = abs(pid)
    if pid == 81:
        return 1
    if pid == 82:
        return 11
    return pid


@pytest.fixture(scope="module")
def subproc_meta():
    with open(TT_DECAY_SUBPROCESSES) as f:
        return json.load(f)[0]


def build_topology_and_args(meta, bw_cutoff=BW_CUTOFF):
    """Rebuild the ms.Topology/ms.SubprocArgs for the single channel/diagram
    described in PROC_tt_decay's subprocesses.json, mirroring what
    MadgraphSubprocess.build_multi_channel_data/build_lhe_completer do in the
    madevent.py template, without needing a full MadgraphProcess/run_card setup.
    """
    channel = meta["channels"][0]
    propagators = []
    for i, signed_pid in enumerate(channel["propagators"]):
        pid = clean_pid(signed_pid)
        mass = MASSES[pid]
        width = WIDTHS[pid]
        if i in channel["on_shell_propagators"]:
            e_min, e_max = mass - bw_cutoff * width, mass + bw_cutoff * width
        else:
            e_min, e_max = 0.0, 0.0
        propagators.append(
            ms.Propagator(
                mass=mass,
                width=width,
                integration_order=0,
                e_min=e_min,
                e_max=e_max,
                pdg_id=signed_pid,
            )
        )

    incoming_masses = [MASSES[clean_pid(pid)] for pid in meta["incoming"]]
    outgoing_masses = [MASSES[clean_pid(pid)] for pid in meta["outgoing"]]
    diagram = ms.Diagram(
        incoming_masses, outgoing_masses, propagators, channel["vertices"]
    )
    topo = ms.Topology.topologies(diagram)[0]

    diagrams = channel["diagrams"]
    permutations = [d["permutation"] for d in diagrams]
    diagram_indices = [d["diagram"] for d in diagrams]
    diagram_color_indices = [d["active_colors"] for d in diagrams]

    subproc_args = ms.SubprocArgs(
        process_id=0,
        topologies=[topo],
        permutations=[permutations],
        diagram_indices=[diagram_indices],
        diagram_color_indices=[diagram_color_indices],
        color_flows=meta["color_flows"],
        pdg_color_types={
            int(key): value for key, value in meta["pdg_color_types"].items()
        },
        helicities=meta["helicities"],
        pdg_ids=[flavor["options"] for flavor in meta["flavors"]],
    )
    return topo, permutations, subproc_args


@pytest.fixture(scope="module")
def topology_and_args(subproc_meta):
    return build_topology_and_args(subproc_meta)


@pytest.fixture(scope="module")
def mapping(topology_and_args):
    topo, permutations, _ = topology_and_args
    # e_min/e_max on the on-shell propagators above force this mapping to only
    # ever sample points where the top/antitop and W+/W- invariant masses fall
    # inside the same Breit-Wigner window used by the LHECompleter below, so
    # every event is guaranteed to have all four resonances identified.
    return ms.PhaseSpaceMapping(topo, CM_ENERGY, permutations=permutations)


@pytest.fixture(scope="module")
def lhe_completer(topology_and_args):
    _, _, subproc_args = topology_and_args
    return ms.LHECompleter([subproc_args], bw_cutoff=BW_CUTOFF)


def sample_external_momenta(mapping, batch_size, seed):
    rng = np.random.default_rng(seed)
    r = rng.random((batch_size, mapping.random_dim()))
    p_ext, _, _, _ = mapping.map_forward([r], [])
    return p_ext


def build_event(momenta_row):
    event = ms.LHEEvent()
    event.particles = [
        ms.LHEParticle(p_x=px, p_y=py, p_z=pz, energy=e)
        for e, px, py, pz in momenta_row
    ]
    return event


def external_particles(event):
    """The 8 original external particles, in [incoming..., outgoing...] order.

    complete_event_data() inserts intermediate resonances right after the two
    incoming particles, pushing the original outgoing particles to the tail of
    event.particles, so they always occupy its last 6 slots.
    """
    return event.particles[:2] + event.particles[len(event.particles) - 6 :]


@pytest.fixture(scope="module")
def events(lhe_completer, mapping):
    p_ext = sample_external_momenta(mapping, 300, seed=1234)
    rand_gen = ms.MixMaxRandom(2024)
    result = []
    for row in p_ext:
        event = build_event(row)
        lhe_completer.complete_event_data(event, 0, 0, 0, 0, 0, rand_gen)
        result.append(event)
    return result


def momentum(particle):
    return np.array([particle.energy, particle.px, particle.py, particle.pz])


def test_particle_count_includes_all_resonances(events):
    # On-shell mapping guarantees t, tbar, W+ and W- are always identified.
    for event in events:
        assert len(event.particles) == PARTICLE_COUNT + 4


def test_particles_fully_populated(events):
    for event in events:
        for particle in event.particles:
            values = [
                particle.px,
                particle.py,
                particle.pz,
                particle.energy,
                particle.mass,
                particle.spin,
            ]
            assert np.isfinite(values).all()
            assert particle.mass >= -1e-6
            assert particle.energy > 0
            assert particle.pdg_id != 0
            assert particle.status_code in (
                ms.LHEParticle.status_incoming,
                ms.LHEParticle.status_outgoing,
                ms.LHEParticle.status_intermediate_resonance,
            )
            if particle.status_code == ms.LHEParticle.status_incoming:
                assert particle.mother1 == 0 and particle.mother2 == 0
            else:
                # every non-incoming particle is attached either directly to
                # the two beams, or to a single earlier resonance
                assert (particle.mother1, particle.mother2) == (1, 2) or (
                    particle.mother1 == particle.mother2 and particle.mother1 >= 3
                )
            if particle.status_code == ms.LHEParticle.status_intermediate_resonance:
                assert particle.pdg_id in RESONANCE_PDGS
                assert particle.spin == 9
                assert particle.lifetime == 0


def test_resonance_content(events):
    for event in events:
        resonance_pdgs = sorted(
            p.pdg_id
            for p in event.particles
            if p.status_code == ms.LHEParticle.status_intermediate_resonance
        )
        assert resonance_pdgs == sorted(RESONANCE_PDGS)


def test_resonance_masses_within_breit_wigner_window(events):
    for event in events:
        for particle in event.particles:
            if particle.status_code != ms.LHEParticle.status_intermediate_resonance:
                continue
            mass, width = POLE_MASS_WIDTH[abs(particle.pdg_id)]
            lo, hi = mass - BW_CUTOFF * width, mass + BW_CUTOFF * width
            assert lo < particle.mass < hi


def test_mother_daughter_momentum_conservation(events):
    """Every resonance's momentum must equal the sum of its direct daughters',
    and everything attached directly to the beams must sum to the incoming
    momentum -- i.e. no propagator momentum is "stranded" or double counted.
    """
    for event in events:
        particles = event.particles
        incoming = [
            p for p in particles if p.status_code == ms.LHEParticle.status_incoming
        ]
        roots = [
            p
            for p in particles
            if p.status_code != ms.LHEParticle.status_incoming
            and (p.mother1, p.mother2) == (1, 2)
        ]
        for index, particle in enumerate(particles):
            if particle.status_code != ms.LHEParticle.status_intermediate_resonance:
                continue
            mother_index = index + 1  # LHE mother indices are 1-based
            daughters = [
                p
                for p in particles
                if (p.mother1, p.mother2) == (mother_index, mother_index)
            ]
            assert len(daughters) > 0, f"resonance {particle.pdg_id} has no daughters"
            total = sum((momentum(d) for d in daughters), np.zeros(4))
            assert total == approx(momentum(particle), rel=1e-6, abs=1e-6)

        total_roots = sum((momentum(p) for p in roots), np.zeros(4))
        total_incoming = sum((momentum(p) for p in incoming), np.zeros(4))
        assert total_roots == approx(total_incoming, rel=1e-6, abs=1e-6)


def test_resonance_colors_consistent_with_daughters(events, subproc_meta):
    """A resonance's (color, anti_color) must be exactly the color line that
    survives after cancelling matched color/anti-color pairs between its
    direct daughters -- i.e. no color line is left dangling at a resonance.
    """
    pdg_color_types = {
        int(key): value for key, value in subproc_meta["pdg_color_types"].items()
    }
    for event in events:
        particles = event.particles
        for index, particle in enumerate(particles):
            if particle.status_code != ms.LHEParticle.status_intermediate_resonance:
                continue
            mother_index = index + 1
            daughters = [
                p
                for p in particles
                if (p.mother1, p.mother2) == (mother_index, mother_index)
            ]
            colors = [d.color for d in daughters if d.color != 0]
            anti_colors = [d.anti_color for d in daughters if d.anti_color != 0]
            for i, color in enumerate(colors):
                for j, anti_color in enumerate(anti_colors):
                    if color == anti_color:
                        colors[i] = 0
                        anti_colors[j] = 0
            colors = [c for c in colors if c != 0]
            anti_colors = [c for c in anti_colors if c != 0]

            color_type = pdg_color_types[particle.pdg_id]
            if color_type == 1:
                expected = (0, 0)
            elif color_type == 3:
                expected = (colors[0], 0)
            elif color_type == -3:
                expected = (0, anti_colors[0])
            elif color_type == 8:
                expected = (colors[0], anti_colors[0])
            else:
                raise AssertionError(f"unexpected color type {color_type}")
            assert (particle.color, particle.anti_color) == expected


def test_external_color_flows_match_input_across_subprocesses(
    topology_and_args, mapping, subproc_meta
):
    """Regression test for a color-offset bug where a second subprocess's
    external colors were read from the wrong location in the flattened color
    table. Uses two (identical) subprocesses so the offset for subproc index 1
    is actually exercised.
    """
    _, _, subproc_args = topology_and_args
    lhe_completer = ms.LHECompleter([subproc_args, subproc_args], bw_cutoff=BW_CUTOFF)
    p_ext = sample_external_momenta(mapping, 20, seed=99)
    rand_gen = ms.MixMaxRandom(7)
    color_flows = subproc_meta["color_flows"]

    for subprocess_index in (0, 1):
        for color_index in range(len(color_flows)):
            for row in p_ext:
                event = build_event(row)
                lhe_completer.complete_event_data(
                    event, subprocess_index, 0, color_index, 0, 0, rand_gen
                )
                external = external_particles(event)
                for i in range(PARTICLE_COUNT):
                    expected = tuple(color_flows[color_index][i])
                    actual = (external[i].color, external[i].anti_color)
                    assert actual == expected


def test_external_spins_match_helicity_table(lhe_completer, mapping, subproc_meta):
    helicities = subproc_meta["helicities"]
    p_ext = sample_external_momenta(mapping, 5, seed=55)
    rand_gen = ms.MixMaxRandom(3)
    for helicity_index in [0, 3, 10]:
        for row in p_ext:
            event = build_event(row)
            lhe_completer.complete_event_data(
                event, 0, 0, 0, 0, helicity_index, rand_gen
            )
            external = external_particles(event)
            for i in range(PARTICLE_COUNT):
                assert external[i].spin == helicities[helicity_index][i]


def test_external_flavors_are_valid_options(lhe_completer, mapping, subproc_meta):
    p_ext = sample_external_momenta(mapping, 30, seed=77)
    rand_gen = ms.MixMaxRandom(9)
    for flavor_index, flavor in enumerate(subproc_meta["flavors"]):
        options = [tuple(option) for option in flavor["options"]]
        for row in p_ext:
            event = build_event(row)
            lhe_completer.complete_event_data(event, 0, 0, 0, flavor_index, 0, rand_gen)
            pdgs = tuple(p.pdg_id for p in external_particles(event))
            assert pdgs in options


def test_save_load_roundtrip(lhe_completer, mapping):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "lhe_completer.json")
        lhe_completer.save(save_path)
        loaded = ms.LHECompleter.load(save_path)
        assert loaded.max_particle_count == lhe_completer.max_particle_count

        p_ext = sample_external_momenta(mapping, 20, seed=321)
        for row in p_ext:
            event_a, event_b = build_event(row), build_event(row)
            lhe_completer.complete_event_data(
                event_a, 0, 0, 0, 0, 0, ms.MixMaxRandom(11)
            )
            loaded.complete_event_data(event_b, 0, 0, 0, 0, 0, ms.MixMaxRandom(11))
            assert len(event_a.particles) == len(event_b.particles)
            for pa, pb in zip(event_a.particles, event_b.particles):
                assert pa.pdg_id == pb.pdg_id
                assert pa.status_code == pb.status_code
                assert (pa.mother1, pa.mother2) == (pb.mother1, pb.mother2)
                assert (pa.color, pa.anti_color) == (pb.color, pb.anti_color)
                assert pa.mass == approx(pb.mass)
                assert momentum(pa) == approx(momentum(pb))


def test_wrong_particle_count_raises(lhe_completer):
    event = ms.LHEEvent()
    event.particles = [ms.LHEParticle()] * 3
    with pytest.raises(RuntimeError):
        lhe_completer.complete_event_data(event, 0, 0, 0, 0, 0, ms.MixMaxRandom(1))


def test_invalid_color_index_raises(lhe_completer, mapping):
    p_ext = sample_external_momenta(mapping, 1, seed=1)
    event = build_event(p_ext[0])
    with pytest.raises(RuntimeError):
        lhe_completer.complete_event_data(event, 0, 0, 999, 0, 0, ms.MixMaxRandom(1))
