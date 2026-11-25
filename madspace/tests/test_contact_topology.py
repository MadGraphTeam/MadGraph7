"""Contact interaction: both beams and both final-state particles meet at one
vertex, with no t-channel propagator (e.g. a four-fermion operator in
q q~ > t t~). The topology must reduce to the root decaying into the two
outgoing particles, as for a pure s-channel diagram."""

import math

import pytest
import torch
from madspace_testing import generator, rand

import madspace as ms

M_TOP = 173.0
SQRT_S = 1000.0
BATCH_SIZE = 20000


def contact_diagram(masses):
    return ms.Diagram([0.0, 0.0], masses, [], [["o1", "o0", "i0", "i1"]])


@pytest.mark.parametrize("masses", [[M_TOP, M_TOP], [0.0, 0.0], [M_TOP, 0.0]])
def test_contact_topology_structure(masses):
    topos = ms.Topology.topologies(contact_diagram(masses))
    assert len(topos) == 1
    topo = topos[0]
    assert topo.t_propagator_count == 0
    assert topo.t_integration_order == []
    root = topo.decays[0]
    assert sorted(root.child_indices) == sorted(topo.outgoing_indices)
    assert all(topo.decays[i].child_indices == [] for i in root.child_indices)


@pytest.mark.parametrize("masses", [[M_TOP, M_TOP], [0.0, 0.0], [M_TOP, 0.0]])
def test_contact_two_body_volume(masses):
    """At fixed sqrt(s) the weights integrate to the two-body phase space
    volume, lambda^(1/2)(s, m1^2, m2^2) / (8 pi s)."""
    topo = ms.Topology.topologies(contact_diagram(masses))[0]
    mapping = ms.PhaseSpaceMapping(topo, SQRT_S, leptonic=True)
    r = rand((BATCH_SIZE, mapping.random_dim()), generator=generator(7))
    p_ext, x1, x2, det = mapping.map_forward([r])

    assert torch.all(torch.isfinite(det))
    assert torch.all(det > 0)
    # momentum conservation and on-shell final state
    torch.testing.assert_close(
        p_ext[:, 0] + p_ext[:, 1],
        p_ext[:, 2] + p_ext[:, 3],
        rtol=1e-9,
        atol=1e-7,
    )
    m_out = torch.sqrt(
        torch.clamp(p_ext[:, 2:, 0] ** 2 - torch.sum(p_ext[:, 2:, 1:] ** 2, dim=2), 0)
    )
    torch.testing.assert_close(
        m_out, torch.tensor(masses, dtype=m_out.dtype).expand_as(m_out),
        rtol=1e-6, atol=1e-3,
    )

    s = SQRT_S**2
    m1, m2 = masses
    lam = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)
    expected = math.sqrt(lam) / (8 * math.pi * s)
    # two-body phase space is flat in the solid angle: the weight is constant
    assert det.mean().item() == pytest.approx(expected, rel=1e-6)


def test_quartic_vertex_in_t_chain_is_rejected():
    """A t-propagator next to a vertex with two outgoing lines is not a chain
    the t-channel mappings handle: a clear error, not an IndexError."""
    propagator = ms.Propagator(mass=0.0, width=0.0, integration_order=0)
    diagram = ms.Diagram(
        [0.0, 0.0],
        [0.0, 0.0, 0.0],
        [propagator],
        [["i0", "o0", "o1", "p0"], ["p0", "o2", "i1"]],
    )
    with pytest.raises(ValueError, match="more than three"):
        ms.Topology.topologies(diagram)
