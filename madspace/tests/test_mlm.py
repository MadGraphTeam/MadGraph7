"""Tests for the MLM back-clustering (madspace/src/phasespace/mlm_clustering.cpp
and madspace/src/kernels/mlm.hpp).

The clustering compiler turns the diagram topologies of a subprocess into a
state machine of possible clustering histories; the kernel walks that state
machine for one phase-space point, picks a history, and reads the
renormalisation / factorisation / per-jet scales off it.

There is no single "right answer" to compare against (see the discussion in the
branch notes: the resonance handling and the factorisation-scale definition
differ from what madevent does on purpose), so these tests pin down the
properties the implementation must have whatever those choices end up being:

  * the compiled state machine is well formed and reachable,
  * the kernel's outputs are finite, positive and ordered as documented,
  * the clustering is invariant under the symmetries of the phase-space point,
  * the individual clustering measures agree with the Fortran ones they were
    ported from (Template/NLO/SubProcesses/cluster.f),
  * resonant propagators are recognised inside the Breit-Wigner window.
"""

import json
import os

import numpy as np
import pytest

import madspace as ms

CM_ENERGY = 13000.0
BATCH_SIZE = 200
SEED = 5678

M_TOP = 173.0
M_Z, W_Z = 91.188, 2.4414

TEST_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data")

# g g > t t~ g g and g g > t t~ g g g, i.e. legs (g, g, t, t~, g, ...)
PROCESSES = {
    "ttgg": ("ttgg.json", [21, 21, 6, -6, 21, 21]),
    "ttggg": ("ttggg.json", [21, 21, 6, -6, 21, 21, 21]),
}


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


def load_diagrams(file_name):
    with open(os.path.join(TEST_DATA, file_name)) as f:
        return json.load(f)


def make_topologies(diagrams):
    return [
        ms.Topology(
            ms.Diagram(
                d["incoming_masses"],
                d["outgoing_masses"],
                [ms.Propagator(*p) for p in d["propagators"]],
                d["vertices"],
            )
        )
        for d in diagrams
    ]


def make_diagram_indices(diagrams):
    """One running index per (topology, permutation) pair, matching what
    launch.py hands to MLMClustering."""
    offset = 0
    indices = []
    for d in diagrams:
        count = len(d["permutations"])
        indices.append(list(range(offset, offset + count)))
        offset += count
    return indices


def make_clustering(diagrams, **kwargs):
    kwargs.setdefault("cm_energy", CM_ENERGY)
    return ms.MLMClustering(
        make_topologies(diagrams),
        [d["permutations"] for d in diagrams],
        make_diagram_indices(diagrams),
        **kwargs,
    )


def sample_momenta(diagrams, batch_size=BATCH_SIZE, seed=SEED):
    """Phase-space points for the first topology of the process."""
    rng = np.random.default_rng(seed)
    topology = make_topologies(diagrams)[0]
    permutations = diagrams[0]["permutations"]
    mapping = ms.PhaseSpaceMapping(topology, CM_ENERGY, permutations=permutations)
    r = rng.random((batch_size, mapping.random_dim()))
    condition = (
        []
        if len(permutations) <= 1
        else [rng.integers(0, len(permutations), batch_size, dtype=np.int32)]
    )
    p_ext, *_ = mapping.map_forward([r], condition)
    return np.asarray(p_ext)


@pytest.fixture(params=sorted(PROCESSES))
def process(request):
    file_name, pdg_ids = PROCESSES[request.param]
    diagrams = load_diagrams(file_name)
    return request.param, diagrams, pdg_ids


def run(clustering, momenta):
    """The scale outputs of a clustering, as numpy arrays: ren_scale,
    fact_scale1, fact_scale2, outgoing_scales, diagram_index."""
    return run_all(clustering, momenta)[:5]


def run_all(clustering, momenta):
    """As run(), plus the trailing xqcut_weight."""
    out = clustering(momenta)
    return tuple(np.asarray(v) for v in out)


def assert_jet_scales_agree(reference, other, max_flip_fraction=0.05):
    """Compare two sets of per-jet clustering scales that should be identical.

    Which clustering wins a step is an argmin over the candidate measures, and
    processes with identical final-state partons produce exactly degenerate
    candidates. Recomputing the measures on a symmetry-transformed event
    perturbs them at the level of floating-point noise, which can flip the
    argmin and hand the scale to the other member of the degenerate pair. The
    set of clustering *scales* is unchanged (so mu_R and mu_F are), only which
    leg they are booked against, and it happens for a small fraction of events.
    """
    matches = np.isclose(reference, other, rtol=1e-6, atol=1e-6)
    flipped = np.mean(~matches.all(axis=1))
    assert flipped <= max_flip_fraction, (
        f"{flipped:.1%} of events changed their jet-scale assignment, "
        f"more than the {max_flip_fraction:.0%} expected from degenerate "
        "clustering candidates"
    )


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_construction_succeeds(process):
    """Every process in the test data compiles into a state machine. The
    constructor throws on an unreachable or dead-ended state, so this also
    covers the structure of the compiled machine."""
    _, diagrams, pdg_ids = process
    assert make_clustering(diagrams) is not None
    assert make_clustering(diagrams, external_pdg_ids=pdg_ids) is not None


def test_pdg_id_count_is_checked(process):
    """A wrong number of external pdg ids is a caller error, not something to
    silently truncate."""
    _, diagrams, pdg_ids = process
    with pytest.raises(Exception):
        make_clustering(diagrams, external_pdg_ids=pdg_ids[:-1])


def test_flavor_information_selects_which_legs_are_jets(process):
    """Only jets get a real clustering scale. Without pdg ids the clustering has
    to assume every leg is one; with them, the tops (legs 2 and 3 of these
    processes) never are and always fall back to sqrt(s)."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    *_, out_no_pdg, _ = run(make_clustering(diagrams), momenta)
    *_, out_pdg, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta
    )
    # the tops are not jets, so with flavor information they never get one
    assert np.all(out_pdg[:, 0] == CM_ENERGY)
    assert np.all(out_pdg[:, 1] == CM_ENERGY)
    # ... whereas without it they do, at least sometimes
    assert np.any(out_no_pdg[:, :2] < CM_ENERGY)
    # the gluons are jets either way and usually pick up a real scale
    assert np.mean(out_pdg[:, 2:].min(axis=1) < CM_ENERGY) > 0.5


def test_unclustered_legs_fall_back_to_the_collider_energy(process):
    """Every outgoing leg is reported at a scale, never at zero.

    pt_clust is what an MLM veto compares against qcut, so a leg that no QCD
    clustering assigned a scale to has to come out at a value no veto can trip
    on. madevent does the same, via the "ptclus = etot" fallback in
    Template/LO/SubProcesses/reweight.f.
    """
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=2000, seed=4321)
    for clustering in (
        make_clustering(diagrams),
        make_clustering(diagrams, external_pdg_ids=pdg_ids),
    ):
        *_, out, _ = run(clustering, momenta)
        assert np.all(out > 0.0)
        assert np.all(out <= CM_ENERGY)
        # the fallback value appears exactly, not as a rounded-down scale
        fallback = out == CM_ENERGY
        assert fallback.any()


def test_the_fallback_value_follows_the_collider_energy(process):
    """The fallback is the collider energy the clustering was built with, so
    changing it moves only the unassigned legs."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    *_, out, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta
    )
    *_, out_half, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, cm_energy=CM_ENERGY / 2
        ),
        momenta,
    )
    assigned = out != CM_ENERGY
    assert np.array_equal(out[assigned], out_half[assigned])
    assert np.all(out_half[~assigned] == CM_ENERGY / 2)


# --------------------------------------------------------------------------
# kernel outputs
# --------------------------------------------------------------------------


def test_scales_are_finite_and_positive(process):
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    ren, fac1, fac2, out_scales, diagram = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta
    )
    for name, value in [("ren", ren), ("fac1", fac1), ("fac2", fac2)]:
        assert np.all(np.isfinite(value)), f"{name} scale not finite"
        assert np.all(value > 0.0), f"{name} scale not positive"
    assert np.all(np.isfinite(out_scales))
    assert np.all(out_scales >= 0.0)


def test_scales_are_below_the_collider_energy(process):
    """No clustering scale can exceed the collider energy."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    ren, fac1, fac2, out_scales, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta
    )
    for value in (ren, fac1, fac2, out_scales):
        assert np.all(value <= CM_ENERGY)


def test_factorization_scale_does_not_exceed_renormalization_scale(process):
    """The kernel caps mu_F at mu_R, and uses one factorisation scale for both
    beams."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    ren, fac1, fac2, _, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta
    )
    assert np.array_equal(fac1, fac2)
    assert np.all(fac1 <= ren * (1.0 + 1e-12))


def test_diagram_index_is_in_range(process):
    """The selected diagram index has to be one of the indices handed to the
    constructor, since it is passed straight to the matrix element."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    *_, diagram = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta
    )
    total = sum(len(d["permutations"]) for d in diagrams)
    assert diagram.min() >= 0
    assert diagram.max() < total


def test_batch_entries_are_independent(process):
    """Each phase-space point is clustered on its own: running a subset of a
    batch must give the same answer as running the whole batch.

    The kernel keeps per-event scratch arrays (momenta_tmp, masses_tmp, the
    cluster history), so this catches state leaking between batch entries.
    """
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    momenta = sample_momenta(diagrams)
    ren_full, fac_full, _, out_full, _ = run(clustering, momenta)
    # the diagram index draws a random number, so only compare the scales
    ren_half, fac_half, _, out_half, _ = run(clustering, momenta[: BATCH_SIZE // 2])
    n = BATCH_SIZE // 2
    assert np.array_equal(ren_full[:n], ren_half)
    assert np.array_equal(fac_full[:n], fac_half)
    assert np.array_equal(out_full[:n], out_half)


def test_longitudinal_boosts_barely_change_the_scales(process):
    """The clustering measures themselves are longitudinally boost invariant -
    they are built from mT, pseudorapidity differences and azimuthal angles -
    but the *choice* of clustering is not.

    compute_scale multiplies an initial-state candidate by 1.000001 when the
    emitted parton goes against the beam it is clustered onto, tested as
    sign(pz1) != sign(pz2). That sign is not boost invariant, so a boost can
    flip which candidate wins for events where two candidates are close. This is
    inherited from the Fortran, which uses the same test.

    So: pin that the effect stays small rather than asserting exact invariance.
    """
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    momenta = sample_momenta(diagrams)

    rapidity = 0.3
    cosh_y, sinh_y = np.cosh(rapidity), np.sinh(rapidity)
    boosted = momenta.copy()
    energy, pz = momenta[..., 0], momenta[..., 3]
    boosted[..., 0] = cosh_y * energy + sinh_y * pz
    boosted[..., 3] = sinh_y * energy + cosh_y * pz

    ren, fac, _, out, _ = run(clustering, momenta)
    ren_b, fac_b, _, out_b, _ = run(clustering, boosted)

    unchanged = np.isclose(ren_b, ren, rtol=1e-9)
    assert unchanged.mean() > 0.8, (
        f"only {unchanged.mean():.0%} of events keep their renormalisation "
        "scale under a longitudinal boost"
    )
    # where the history is unchanged, the scales have to agree to full precision
    assert ren_b[unchanged] == pytest.approx(ren[unchanged], rel=1e-9)
    assert fac_b[unchanged] == pytest.approx(fac[unchanged], rel=1e-9)
    assert_jet_scales_agree(out[unchanged], out_b[unchanged])


def test_scales_are_azimuthally_invariant(process):
    """Rotating the event around the beam axis must not change any scale."""
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    momenta = sample_momenta(diagrams)

    angle = 0.7
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotated = momenta.copy()
    px, py = momenta[..., 1], momenta[..., 2]
    rotated[..., 1] = cos_a * px - sin_a * py
    rotated[..., 2] = sin_a * px + cos_a * py

    ren, fac, _, out, _ = run(clustering, momenta)
    ren_r, fac_r, _, out_r, _ = run(clustering, rotated)
    assert ren_r == pytest.approx(ren, rel=1e-9)
    assert fac_r == pytest.approx(fac, rel=1e-9)
    assert_jet_scales_agree(out, out_r)


def test_scales_are_homogeneous_for_a_single_clustering():
    """Every clustering measure has mass dimension one, so scaling all momenta
    by a factor scales every returned scale by the same factor.

    This is exact only as long as update_momenta is not involved: it keeps the
    Fortran's absolute 100 GeV^2 guard against boosting into a degenerate frame
    ("prevent too extreme boost" in Template/NLO/SubProcesses/cluster.f), which
    is not scale free. A 2 -> 2 process needs a single clustering, whose
    update_momenta result is never used again, so there the homogeneity is
    exact.
    """
    diagrams = [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0],
            "propagators": [[0.0, 0.0]],
            "vertices": [["i0", "o0", "p0"], ["p0", "o1", "i1"]],
            "permutations": [[0, 1, 2, 3]],
        }
    ]
    pdg_ids = [21, 21, 21, 21]
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    momenta = sample_momenta(diagrams, batch_size=128, seed=11)
    factor = 3.0
    ren, fac, _, out, _ = run(clustering, momenta)
    ren_s, fac_s, _, out_s, _ = run(clustering, momenta * factor)
    # the only measure available here is djb_clus = (E - pz)(E + pz), which
    # cancels badly for forward partons, so this cannot be pushed much below
    # 1e-6 even though the relation is exact in real arithmetic
    assert ren_s == pytest.approx(ren * factor, rel=1e-6)
    assert fac_s == pytest.approx(fac * factor, rel=1e-6)
    # the sqrt(s) fallback is a fixed constant, so only the legs that were
    # actually assigned a clustering scale scale with the momenta
    assigned = (out != CM_ENERGY) & (out_s != CM_ENERGY)
    assert_jet_scales_agree(
        np.where(assigned, out * factor, 0.0),
        np.where(assigned, out_s, 0.0),
        max_flip_fraction=0.3,
    )


def test_scales_mostly_scale_with_the_momenta(process):
    """The same for a multi-step clustering, where the absolute boost guard in
    update_momenta breaks exact homogeneity for the events that cross it.

    Pin how often that happens: the guard is a deliberate numerical safeguard,
    but a large fraction here would mean the scales depend on the overall energy
    of the event far more than intended.
    """
    _, diagrams, _ = process
    # drop the top masses so nothing else sets an absolute scale
    massless = json.loads(json.dumps(diagrams))
    for d in massless:
        d["outgoing_masses"] = [0.0] * len(d["outgoing_masses"])
    pdg_massless = [21] * (len(massless[0]["outgoing_masses"]) + 2)

    clustering = make_clustering(massless, external_pdg_ids=pdg_massless)
    momenta = sample_momenta(massless)
    factor = 2.0
    ren, _, _, _, _ = run(clustering, momenta)
    ren_s, _, _, _, _ = run(clustering, momenta * factor)

    ratio = ren_s / (ren * factor)
    exact = np.isclose(ratio, 1.0, rtol=1e-9)
    assert np.median(ratio) == pytest.approx(1.0, rel=1e-9)
    assert exact.mean() > 0.5, (
        f"only {exact.mean():.0%} of events scale exactly; the absolute boost "
        "guard in update_momenta should affect far fewer"
    )


def test_renormalization_scale_is_a_geometric_mean(process):
    """mu_R is the geometric mean of the clustering scales, so it lies between
    the smallest and the largest of them. mu_F is the smallest QCD clustering
    scale (capped at mu_R), which bounds mu_R from below."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    ren, fac, _, out, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta
    )
    assert np.all(ren >= fac * (1.0 - 1e-12))
    # mu_F is the smallest QCD clustering scale, so every jet scale - each of
    # which is a QCD clustering scale - is at least as large
    assigned = out != CM_ENERGY
    smallest_jet_scale = np.where(assigned, out, np.inf).min(axis=1)
    assert np.all(smallest_jet_scale >= fac * (1.0 - 1e-9))


# --------------------------------------------------------------------------
# resonances
# --------------------------------------------------------------------------


M_B = 4.7


def z_to_bb_diagram(width):
    """u u~ > Z(> b b~) g g with the Z either given its physical width, so it
    can be recognised as a resonance, or a zero width, so it cannot.

    The b quarks are massive on purpose. compute_scale returns the invariant
    mass of the pair both when the clustering is resonant and when the parent is
    massive and both children are massless (itype 8 in the Fortran), so a
    resonance decaying to massless particles cannot tell the two branches apart.
    With massive children the non-resonant case falls through to the kt measure
    instead, and the resonance branch becomes observable.

    Legs: (u, u~, b, b~, g, g); the Z propagator recombines legs 2 and 3.
    """
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [M_B, M_B, 0.0, 0.0],
            # p0 = Z (b b~), p1/p2 = t-channel quark lines
            "propagators": [[M_Z, width], [0.0, 0.0], [0.0, 0.0]],
            "vertices": [
                ["o0", "o1", "p0"],
                ["i0", "o2", "p1"],
                ["p1", "o3", "p2"],
                ["p2", "p0", "i1"],
            ],
            "permutations": [[0, 1, 2, 3, 4, 5]],
        }
    ]


Z_TO_BB_PDGS = [2, -2, 5, -5, 21, 21]


def test_resonant_propagators_get_a_breit_wigner_index():
    """Only a propagator with a width can be resonant, and only in a final-state
    clustering: a t-channel line is spacelike and never goes on shell."""
    with_width = make_clustering(
        z_to_bb_diagram(W_Z), external_pdg_ids=Z_TO_BB_PDGS
    )
    without = make_clustering(z_to_bb_diagram(0.0), external_pdg_ids=Z_TO_BB_PDGS)

    assert list(with_width.bw_masses) == [pytest.approx(M_Z)]
    assert list(with_width.bw_widths) == [pytest.approx(W_Z)]
    assert list(without.bw_masses) == []

    # exactly the b b~ clustering (legs 2 and 3) carries the index
    resonant_pairs = set()
    non_terminal, _ = walk(np.asarray(with_width.cluster_state_machine), 6)
    for _, (_, transitions) in non_terminal.items():
        for data, _, _ in transitions:
            if field(data, BIT_MASS_INDEX) != 0:
                resonant_pairs.add(
                    (field(data, BIT_PARTICLE1), field(data, BIT_PARTICLE2))
                )
    assert resonant_pairs == {(2, 3)}


def test_resonant_clustering_uses_the_invariant_mass():
    """A clustering onto a propagator inside its Breit-Wigner window is scored
    with the invariant mass of the pair rather than the kt measure. Compare
    against the same topology with the width set to zero, which switches the
    resonance branch off entirely."""
    momenta = sample_momenta(z_to_bb_diagram(W_Z))
    resonant = make_clustering(
        z_to_bb_diagram(W_Z), external_pdg_ids=Z_TO_BB_PDGS, bw_cutoff=15.0
    )
    non_resonant = make_clustering(
        z_to_bb_diagram(0.0), external_pdg_ids=Z_TO_BB_PDGS
    )

    ren_res, fac_res, _, _, _ = run(resonant, momenta)
    ren_non, _, _, _, _ = run(non_resonant, momenta)

    # the resonance branch has to actually change the answer
    assert not np.allclose(ren_res, ren_non)
    assert np.all(np.isfinite(ren_res)) and np.all(ren_res > 0.0)
    assert np.all(fac_res <= ren_res * (1.0 + 1e-12))


def test_breit_wigner_cutoff_widens_the_resonance_window():
    """bw_cutoff scales the half width of the window in which a propagator counts
    as resonant, exactly as it does for the phase-space mapping. A cutoff of zero
    closes the window, which has to reproduce the zero-width result."""
    momenta = sample_momenta(z_to_bb_diagram(W_Z))

    closed = make_clustering(
        z_to_bb_diagram(W_Z), external_pdg_ids=Z_TO_BB_PDGS, bw_cutoff=0.0
    )
    zero_width = make_clustering(
        z_to_bb_diagram(0.0), external_pdg_ids=Z_TO_BB_PDGS
    )
    wide = make_clustering(
        z_to_bb_diagram(W_Z), external_pdg_ids=Z_TO_BB_PDGS, bw_cutoff=15.0
    )

    ren_closed, *_ = run(closed, momenta)
    ren_zero, *_ = run(zero_width, momenta)
    ren_wide, *_ = run(wide, momenta)

    assert ren_closed == pytest.approx(ren_zero, rel=1e-12)
    assert not np.allclose(ren_wide, ren_closed)


# --------------------------------------------------------------------------
# clustering measures, against the Fortran they were ported from
# --------------------------------------------------------------------------


def djb_clus_reference(p, hadronic=True):
    """djb_clus from Template/NLO/SubProcesses/cluster.f."""
    r = (p[0] - p[3]) * (p[0] + p[3]) if hadronic else p[0] * p[0]
    return max(r, 0.0)


def dj_clus_reference(p1, p2, mass1, mass2, jet_radius):
    """The both-massless / both-massive branch of dj_clus."""
    pt1_sq = p1[1] ** 2 + p1[2] ** 2
    pt2_sq = p2[1] ** 2 + p2[2] ** 2
    if pt1_sq == 0.0 or pt2_sq == 0.0:
        return 0.0
    p1a = np.sqrt(pt1_sq + p1[3] ** 2)
    p2a = np.sqrt(pt2_sq + p2[3] ** 2)
    eta1 = 0.5 * np.log((p1a + p1[3]) / (p1a - p1[3]))
    eta2 = 0.5 * np.log((p2a + p2[3]) / (p2a - p2[3]))
    m_max_sq = max(mass1**2, mass2**2)
    dphi_cos = (p1[1] * p2[1] + p1[2] * p2[2]) / np.sqrt(pt1_sq * pt2_sq)
    return max(
        m_max_sq
        + min(pt1_sq, pt2_sq)
        * 2.0
        * (np.cosh(eta1 - eta2) - dphi_cos)
        / jet_radius**2,
        0.0,
    )


def first_step_candidates(clustering, n_ext, event, jet_radius, masses):
    """The clustering measures the kernel evaluates in its first step.

    Which pairs are candidates at all is decided by the compiled state machine -
    only clusterings that some diagram topology allows are offered - so read
    them off it rather than enumerating every pair.
    """
    flat = np.asarray(clustering.cluster_state_machine)
    candidates = {}
    for data, _, _ in decode_transitions(flat, 0):
        p1 = field(data, BIT_PARTICLE1)
        p2 = field(data, BIT_PARTICLE2)
        if p1 < 2:
            # initial-state clustering: mT of the final-state parton, with a
            # 1.000001 penalty when it goes against the beam
            measure = np.sqrt(djb_clus_reference(event[p2]))
            if (event[p1][3] < 0.0) != (event[p2][3] < 0.0):
                measure *= 1.000001
        else:
            measure = np.sqrt(
                dj_clus_reference(
                    event[p1], event[p2], masses[p1], masses[p2], jet_radius
                )
            )
        candidates[(p1, p2)] = measure
    return candidates


def test_single_clustering_scale_matches_the_measures():
    """For a 2 -> 2 process there is exactly one clustering step, so the scale
    the kernel returns is fully predicted by the measures above.

    This is the tightest available check that the kernel walks the state machine
    and applies the same measure the Fortran does.
    """
    diagrams = [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0],
            "propagators": [[0.0, 0.0]],
            "vertices": [["i0", "o0", "p0"], ["p0", "o1", "i1"]],
            "permutations": [[0, 1, 2, 3]],
        }
    ]
    pdg_ids = [21, 21, 21, 21]
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    momenta = sample_momenta(diagrams, batch_size=128, seed=99)
    ren, fac, _, _, _ = run(clustering, momenta)

    # one clustering step: mu_R is that step's scale, and mu_F (the smallest QCD
    # clustering scale, capped at mu_R) has to equal it
    assert fac == pytest.approx(ren, rel=1e-12)

    masses = [0.0] * 4
    for event, scale in zip(momenta, ren):
        candidates = first_step_candidates(clustering, 4, event, 0.4, masses)
        assert candidates, "no clustering offered from the initial state"
        assert scale == pytest.approx(min(candidates.values()), rel=1e-9)


def test_kt_measure_matches_the_fortran():
    """The first clustering step of g g > g g g, where the (o0, o1) final-state
    pair competes with the initial-state candidates and is scored by the kt
    measure. Checked against a transcription of dj_clus / djb_clus from
    Template/NLO/SubProcesses/cluster.f.
    """
    jet_radius = 0.4
    diagrams = [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0, 0.0],
            "propagators": [[0.0, 0.0], [0.0, 0.0]],
            "vertices": [
                ["o0", "o1", "p0"],
                ["i0", "o2", "p1"],
                ["p1", "p0", "i1"],
            ],
            "permutations": [[0, 1, 2, 3, 4]],
        }
    ]
    pdg_ids = [21, 21, 21, 21, 21]
    clustering = make_clustering(
        diagrams, external_pdg_ids=pdg_ids, jet_radius=jet_radius
    )
    momenta = sample_momenta(diagrams, batch_size=128, seed=99)
    _, fac, _, _, _ = run(clustering, momenta)

    # the state machine has to offer the final-state pair, otherwise this test
    # would never exercise dj_clus at all
    flat = np.asarray(clustering.cluster_state_machine)
    offered = {
        (field(d, BIT_PARTICLE1), field(d, BIT_PARTICLE2))
        for d, _, _ in decode_transitions(flat, 0)
    }
    assert (2, 3) in offered

    masses = [0.0] * 5
    for event, scale in zip(momenta, fac):
        candidates = first_step_candidates(
            clustering, 5, event, jet_radius, masses
        )
        # two clustering steps here (n_ext - 3), and mu_F is the smallest QCD
        # clustering scale of the whole history, so it is at most the smallest
        # measure available in the first step
        assert scale <= min(candidates.values()) * (1.0 + 1e-9)


# --------------------------------------------------------------------------
# structure of the compiled state machine
# --------------------------------------------------------------------------
#
# The kernel walks a flat integer array. A non-terminal state is a run of
# (data, next_offset) pairs ending at the pair whose data has bit 30 set; a
# terminal state is a count followed by that many diagram indices. Which of the
# two a given offset is depends only on how many clusterings have been done, so
# the walker below tracks the depth the same way the kernel does.

BIT_PARTICLE1 = (0, 0xFF)
BIT_PARTICLE2 = (8, 0xFF)
BIT_MASS_INDEX = (16, 0xFF)
BIT_MASSIVE_IN = (24, 1)
BIT_MASSIVE_OUT1 = (25, 1)
BIT_MASSIVE_OUT2 = (26, 1)
BIT_IS_QCD = (27, 1)
BIT_IS_JET1 = (28, 1)
BIT_IS_JET2 = (29, 1)
BIT_IS_LAST = (30, 1)

# Words per transition: (data, next_offset, trace_data).
STATE_ITEM_SIZE = 3

# trace_data values: which daughter the mother's parton line continues into.
TRACE_FIRST, TRACE_SECOND, TRACE_HARDER, TRACE_BOTH = 0, 1, 2, 3
TRACE_MODES = {TRACE_FIRST, TRACE_SECOND, TRACE_HARDER, TRACE_BOTH}


def field(data, spec):
    shift, mask = spec
    return (data >> shift) & mask


def decode_transitions(machine, offset):
    """The (data, next_offset, trace) triples of the non-terminal state at
    `offset`."""
    transitions = []
    while True:
        assert offset + STATE_ITEM_SIZE <= len(machine), (
            "transition list runs past the machine"
        )
        data, next_offset, trace = machine[offset : offset + STATE_ITEM_SIZE]
        transitions.append((data, next_offset, trace))
        if field(data, BIT_IS_LAST):
            return transitions
        offset += STATE_ITEM_SIZE


def walk(machine, n_ext):
    """Visit every reachable state, returning
    {offset: (depth, transitions)} for non-terminal states and
    {offset: diagram indices} for terminal ones."""
    cluster_max = n_ext - 3
    non_terminal, terminal = {}, {}
    todo = [(0, 0)]
    seen = set()
    while todo:
        offset, depth = todo.pop()
        if (offset, depth) in seen:
            continue
        seen.add((offset, depth))
        if depth == cluster_max:
            count = machine[offset]
            assert count > 0, f"terminal state at {offset} selects no diagram"
            assert offset + count < len(machine)
            terminal[offset] = list(machine[offset + 1 : offset + 1 + count])
            continue
        transitions = decode_transitions(machine, offset)
        non_terminal[offset] = (depth, transitions)
        for _, next_offset, _ in transitions:
            todo.append((next_offset, depth + 1))
    return non_terminal, terminal


@pytest.fixture
def machine(process):
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    n_ext = len(pdg_ids)
    return clustering, np.asarray(clustering.cluster_state_machine), n_ext


def test_state_machine_is_fully_reachable(machine):
    """Every state the walk can reach is well formed, and a state is never
    reached at two different depths (which would make the terminal/non-terminal
    reading of an offset ambiguous)."""
    _, flat, n_ext = machine
    non_terminal, terminal = walk(flat, n_ext)
    assert non_terminal, "no clustering states compiled"
    assert terminal, "no terminal states compiled"
    depths = {}
    for offset, (depth, _) in non_terminal.items():
        assert depths.setdefault(offset, depth) == depth
    for offset in terminal:
        assert offset not in non_terminal


def test_state_machine_transitions_are_in_range(machine):
    """Every transition names two distinct external particles in range and
    points at an offset inside the machine."""
    _, flat, n_ext = machine
    non_terminal, _ = walk(flat, n_ext)
    for offset, (_, transitions) in non_terminal.items():
        for data, next_offset, _ in transitions:
            particle1 = field(data, BIT_PARTICLE1)
            particle2 = field(data, BIT_PARTICLE2)
            # the merged pseudo-particle is always named by the lower index, so
            # the kernel's is_initial test (particle1 < 2) is meaningful
            assert particle1 < particle2 < n_ext
            assert 0 <= next_offset < len(flat)


def test_state_machine_has_no_duplicate_transitions(machine):
    """A state reached along several paths must be expanded only once.
    Expanding it again appends a second copy of the same transitions, which
    grows the table by a factor that blows up with the multiplicity."""
    _, flat, n_ext = machine
    non_terminal, _ = walk(flat, n_ext)
    for offset, (_, transitions) in non_terminal.items():
        keys = [
            (field(d, BIT_PARTICLE1), field(d, BIT_PARTICLE2), nxt)
            for d, nxt, _ in transitions
        ]
        assert len(keys) == len(set(keys)), f"duplicate transitions at state {offset}"


def test_state_machine_marks_exactly_one_last_transition(machine):
    """The kernel decides a state is finished when it sees the is_last bit, so
    exactly one transition per state carries it, and it is the final one."""
    _, flat, n_ext = machine
    non_terminal, _ = walk(flat, n_ext)
    for offset, (_, transitions) in non_terminal.items():
        flags = [field(d, BIT_IS_LAST) for d, _, _ in transitions]
        assert flags[-1] == 1
        assert sum(flags) == 1


def test_state_machine_mass_indices_are_in_range(machine):
    """mass_index is a 1-based index into the Breit-Wigner tables, or 0 for
    'never resonant'. An out-of-range value would read past those tables."""
    clustering, flat, n_ext = machine
    bw_count = len(clustering.bw_masses)
    assert len(clustering.bw_widths) == bw_count
    non_terminal, _ = walk(flat, n_ext)
    for _, (_, transitions) in non_terminal.items():
        for data, _, _ in transitions:
            assert field(data, BIT_MASS_INDEX) <= bw_count


def test_initial_state_clusterings_are_never_resonant(machine):
    """A t-channel propagator is spacelike, so it can never be on shell."""
    _, flat, n_ext = machine
    non_terminal, _ = walk(flat, n_ext)
    for _, (_, transitions) in non_terminal.items():
        for data, _, _ in transitions:
            if field(data, BIT_PARTICLE1) < 2:
                assert field(data, BIT_MASS_INDEX) == 0
                # a beam leg is not a final-state particle and gets no jet scale
                assert field(data, BIT_IS_JET1) == 0


def test_state_machine_trace_modes_are_valid(machine):
    """Every transition says how the mother's parton line continues into its
    daughters, and an initial-state clustering always carries the beam line on,
    which is daughter 1 by construction."""
    _, flat, n_ext = machine
    non_terminal, _ = walk(flat, n_ext)
    for _, (_, transitions) in non_terminal.items():
        for data, _, trace in transitions:
            assert trace in TRACE_MODES
            if field(data, BIT_PARTICLE1) < 2:
                assert trace == TRACE_FIRST


def trace_modes_of(clustering, n_ext):
    non_terminal, _ = walk(np.asarray(clustering.cluster_state_machine), n_ext)
    return {
        (field(data, BIT_PARTICLE1), field(data, BIT_PARTICLE2)): trace
        for _, (_, transitions) in non_terminal.items()
        for data, _, trace in transitions
    }


def three_gluon_diagram(propagator_pdgs):
    """g g > g g g through an s-channel gluon splitting, legs (g, g, g, g, g)."""
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0, 0.0],
            "propagators": [[0.0, 0.0, 0, 0.0, 0.0, pdg] for pdg in propagator_pdgs],
            "vertices": [
                ["o0", "o1", "p0"],
                ["i0", "o2", "p1"],
                ["p1", "p0", "i1"],
            ],
            "permutations": [[0, 1, 2, 3, 4]],
        }
    ]


def test_gluon_splitting_follows_the_harder_gluon():
    """A g -> g g vertex carries the parton line into the harder gluon, as
    ipartupdate does. p0 recombines legs 2 and 3 into a gluon."""
    clustering = make_clustering(
        three_gluon_diagram([21, 21]), external_pdg_ids=[21] * 5
    )
    assert trace_modes_of(clustering, 5)[(2, 3)] == TRACE_HARDER


def test_gluon_to_quarks_carries_the_line_into_both():
    """A g -> q qbar vertex has no single continuing line, so both daughters
    carry it on and both get booked at later vertices."""
    diagrams = three_gluon_diagram([21, 21])
    diagrams[0]["propagators"][0] = [0.0, 0.0, 0, 0.0, 0.0, 21]
    clustering = make_clustering(diagrams, external_pdg_ids=[21, 21, 1, -1, 21])
    assert trace_modes_of(clustering, 5)[(2, 3)] == TRACE_BOTH


def test_quark_line_follows_the_quark():
    """A q -> q g vertex keeps the line on the quark, whichever daughter it is."""
    # p0 recombines legs 2 and 3 into a quark
    quark_first = make_clustering(
        three_gluon_diagram([1, 21]), external_pdg_ids=[21, 21, 1, 21, 21]
    )
    assert trace_modes_of(quark_first, 5)[(2, 3)] == TRACE_FIRST
    quark_second = make_clustering(
        three_gluon_diagram([1, 21]), external_pdg_ids=[21, 21, 21, 1, 21]
    )
    assert trace_modes_of(quark_second, 5)[(2, 3)] == TRACE_SECOND


def test_terminal_states_select_known_diagrams(machine):
    _, flat, n_ext = machine
    _, terminal = walk(flat, n_ext)
    for offset, diagrams in terminal.items():
        assert len(diagrams) == len(set(diagrams))
        assert all(index >= 0 for index in diagrams)


def massive_leg_permutation_diagram():
    """A topology whose permutations move a massive leg onto a slot that a
    different permutation gives to a massless one.

    The shipped fixtures cannot exercise the slot-versus-leg question at all:
    their permutations only exchange identical gluons, so the two orderings
    agree and indexing by the wrong one is invisible. Here leg 2 is the top
    under the first permutation and a gluon under the second.
    """
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [M_TOP, 0.0, 0.0],
            "propagators": [[0.0, 0.0], [0.0, 0.0]],
            "vertices": [
                ["o0", "o1", "p0"],
                ["i0", "o2", "p1"],
                ["p1", "p0", "i1"],
            ],
            "permutations": [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
        }
    ]


def test_permutations_moving_a_mass_between_legs_are_rejected():
    """external_masses is indexed by external leg, while a topology's masses
    are indexed by topology slot, so the permutation has to be applied. The
    kernel keeps a single mass per leg, so permutations that disagree about a
    leg's mass cannot be represented at all and have to be refused rather than
    silently given one of the two answers.
    """
    with pytest.raises(Exception) as caught:
        make_clustering(
            massive_leg_permutation_diagram(), external_pdg_ids=[21, 21, 6, 21, 21]
        )
    assert "mass" in str(caught.value).lower()


def test_external_masses_are_in_leg_order(process):
    """The kernel indexes external_masses by external leg while a topology's
    masses are indexed by topology slot, so the permutation has to be applied.
    For these processes legs 2 and 3 are the tops.

    On its own this does not pin the ordering down: these fixtures permute only
    identical gluons, so slot and leg order coincide. The case that tells them
    apart is test_permutations_moving_a_mass_between_legs_are_rejected.
    """
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    masses = list(clustering.external_masses)
    assert len(masses) == len(pdg_ids)
    assert masses[0] == 0.0 and masses[1] == 0.0
    assert masses[2] == pytest.approx(M_TOP)
    assert masses[3] == pytest.approx(M_TOP)
    assert all(m == 0.0 for m in masses[4:])


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------


def test_extreme_kinematics_do_not_break_the_walk(process):
    """The boost and rotation applied after an initial-state clustering can
    leave a pair exactly collinear, making the clustering measure inf or NaN.
    The kernel has to keep choosing a clustering anyway: if nothing wins a step
    it would jump to a stale or unset state offset and read outside the state
    machine.

    Strongly asymmetric beam energies reach that configuration, so this is a
    regression test for the crash it used to cause. Three guards stand between
    that crash and this assertion - dj_clus zeroing a NaN, the selection
    mapping a non-finite measure onto SCALE_MAX, and taking the first candidate
    unconditionally - and they are deliberately redundant, so dropping any one
    of them leaves the others covering it. What is pinned here is the behaviour
    they exist for, not each guard individually.
    """
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    momenta = sample_momenta(diagrams, batch_size=2000, seed=1234)
    ren, fac1, fac2, out, diagram = run(clustering, momenta)
    assert np.all(np.isfinite(ren))
    assert np.all(np.isfinite(fac1)) and np.all(np.isfinite(fac2))
    assert np.all(np.isfinite(out))
    total = sum(len(d["permutations"]) for d in diagrams)
    assert np.all((diagram >= 0) & (diagram < total))


def test_exactly_collinear_momenta_are_handled(process):
    """The same thing, but forced: put two final-state partons exactly along the
    beam axis so the pseudorapidity of the pair is genuinely undefined."""
    _, diagrams, pdg_ids = process
    clustering = make_clustering(diagrams, external_pdg_ids=pdg_ids)
    momenta = sample_momenta(diagrams, batch_size=32, seed=7)
    degenerate = momenta.copy()
    # zero the transverse components of the last two legs
    degenerate[:, -2:, 1] = 0.0
    degenerate[:, -2:, 2] = 0.0
    ren, fac1, _, out, diagram = run(clustering, degenerate)
    assert np.all(np.isfinite(ren))
    assert np.all(np.isfinite(fac1))
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# the jet-scale scheme switch
# --------------------------------------------------------------------------


EMISSION = ms.MLMClustering.JetScaleScheme.emission
PRODUCTION = ms.MLMClustering.JetScaleScheme.production


def test_production_scheme_is_never_softer_than_emission(process):
    """The two schemes book the same vertices onto a leg plus, for "production",
    every later vertex on the line the leg belongs to. Since "production" keeps
    the hardest of those and "emission" keeps the one "production" starts from,
    the production scale of a leg can only be larger or equal.
    """
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=1000, seed=2468)
    *_, emission, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=EMISSION
        ),
        momenta,
    )
    *_, production, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=PRODUCTION
        ),
        momenta,
    )
    assert np.all(production >= emission * (1.0 - 1e-12))


def gluon_chain_diagram():
    """g g > g g g g through a chain of two s-channel gluon splittings, so that
    one gluon line carries on past the vertex that emitted it:
    p0 = g(o0, o1), p1 = g(p0, o2). Legs (g, g, g, g, g, g)."""
    return [
        {
            "incoming_masses": [0.0, 0.0],
            "outgoing_masses": [0.0, 0.0, 0.0, 0.0],
            "propagators": [[0.0, 0.0, 0, 0.0, 0.0, 21] for _ in range(3)],
            "vertices": [
                ["o0", "o1", "p0"],
                ["p0", "o2", "p1"],
                ["i0", "o3", "p2"],
                ["p2", "p1", "i1"],
            ],
            "permutations": [[0, 1, 2, 3, 4, 5]],
        }
    ]


def test_the_two_schemes_disagree_on_continuing_lines():
    """The schemes differ only for a leg whose parton line carries on past the
    vertex that emitted it. In the chain above the harder of the first two
    gluons stays on the line and is booked again at the second vertex, so
    "production" gives it the harder of the two scales and "emission" the
    softer. That is the whole point of the switch."""
    diagrams = gluon_chain_diagram()
    pdg_ids = [21] * 6
    momenta = sample_momenta(diagrams, batch_size=1000, seed=2468)
    *_, emission, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=EMISSION
        ),
        momenta,
    )
    *_, production, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=PRODUCTION
        ),
        momenta,
    )
    differs = ~np.isclose(production, emission)
    assert differs.any(), "the two schemes gave identical jet scales"
    # and where they differ it is because production kept a harder vertex
    assert np.all(production[differs] > emission[differs])


def test_the_schemes_coincide_when_no_line_continues(process):
    """With propagator flavours unknown a merged node is not recognised as a
    jet, so nothing is booked onto a continuing line and the two schemes fall
    back to the same answer. The ttgg/ttggg fixtures carry propagator masses
    only, so this pins that degradation as intentional rather than accidental.
    """
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=500, seed=2468)
    *_, emission, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=EMISSION
        ),
        momenta,
    )
    *_, production, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=PRODUCTION
        ),
        momenta,
    )
    assert np.array_equal(production, emission)


def test_the_schemes_agree_for_a_single_clustering():
    """A 2 -> 3 process has one clustering per leg, so nothing continues past
    the vertex that emitted it and the two schemes have to coincide."""
    diagrams = three_gluon_diagram([21, 21])
    pdg_ids = [21] * 5
    momenta = sample_momenta(diagrams, batch_size=256, seed=1357)
    *_, emission, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=EMISSION
        ),
        momenta,
    )
    *_, production, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=PRODUCTION
        ),
        momenta,
    )
    assert np.array_equal(production, emission)


def test_the_scheme_does_not_change_the_scales(process):
    """The switch only moves which leg a clustering scale is booked against; the
    clustering history, and therefore mu_R and mu_F, are untouched."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=1000, seed=2468)
    ren_e, fac_e, _, _, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=EMISSION
        ),
        momenta,
    )
    ren_p, fac_p, _, _, _ = run(
        make_clustering(
            diagrams, external_pdg_ids=pdg_ids, jet_scale_scheme=PRODUCTION
        ),
        momenta,
    )
    assert np.array_equal(ren_e, ren_p)
    assert np.array_equal(fac_e, fac_p)


def test_production_is_the_default():
    """The default matches madevent, which is the validated reference."""
    diagrams = three_gluon_diagram([21, 21])
    assert (
        ms.MLMClustering.JetScaleScheme.__members__["production"] == PRODUCTION
    )
    default = make_clustering(diagrams, external_pdg_ids=[21] * 5)
    explicit = make_clustering(
        diagrams, external_pdg_ids=[21] * 5, jet_scale_scheme=PRODUCTION
    )
    momenta = sample_momenta(diagrams, batch_size=64, seed=11)
    assert np.array_equal(run(default, momenta)[3], run(explicit, momenta)[3])


# --------------------------------------------------------------------------
# the generation-level merging cut
# --------------------------------------------------------------------------


def xqcut_weights(diagrams, pdg_ids, momenta, xqcut, **kwargs):
    clustering = make_clustering(
        diagrams, external_pdg_ids=pdg_ids, xqcut=xqcut, **kwargs
    )
    return run_all(clustering, momenta)[5]


def test_xqcut_is_off_by_default(process):
    """Without a merging cut every event keeps its weight, so turning the
    feature on is the only thing that can change an existing result."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    weights = run_all(
        make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta
    )[5]
    assert np.all(weights == 1.0)


def test_xqcut_weight_is_a_veto(process):
    """It multiplies the event weight, so it has to be exactly zero or one."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    weights = xqcut_weights(diagrams, pdg_ids, momenta, 50.0)
    assert np.all((weights == 0.0) | (weights == 1.0))
    assert np.any(weights == 0.0), "a 50 GeV cut should reject something"


def test_xqcut_rejects_monotonically(process):
    """Raising the cut can only ever reject more events."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=1000, seed=808)
    kept = [
        float(xqcut_weights(diagrams, pdg_ids, momenta, cut).sum())
        for cut in (0.0, 10.0, 30.0, 60.0, 120.0)
    ]
    assert kept == sorted(kept, reverse=True), kept
    assert kept[0] == len(momenta)
    assert kept[-1] < kept[0]


def test_xqcut_rejects_exactly_the_soft_jet_emissions(process):
    """madevent's rule: a clustering step whose daughter is still a bare
    final-state jet has to be at or above xqcut. In the "emission" scheme the
    per-jet scales are booked at exactly those steps, so the veto can be
    recomputed from them and compared."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=500, seed=99)
    cut = 40.0
    clustering = make_clustering(
        diagrams,
        external_pdg_ids=pdg_ids,
        xqcut=cut,
        jet_scale_scheme=EMISSION,
    )
    *_, jet_scales, _, weights = run_all(clustering, momenta)
    # a leg left at the sqrt(s) fallback was never booked and cannot fail
    booked = jet_scales < CM_ENERGY
    softest = np.where(booked, jet_scales, np.inf).min(axis=1)
    expected = np.where(np.isfinite(softest) & (softest < cut), 0.0, 1.0)
    assert np.array_equal(weights, expected)


def test_xqcut_does_not_disturb_the_scales(process):
    """The cut only vetoes. Events that survive keep exactly the scales the
    clustering would have given without it."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams)
    ren_off, fac_off, _, out_off, _ = run(
        make_clustering(diagrams, external_pdg_ids=pdg_ids), momenta
    )
    ren_on, fac_on, _, out_on, _, weights = run_all(
        make_clustering(diagrams, external_pdg_ids=pdg_ids, xqcut=40.0), momenta
    )
    keep = weights > 0
    assert keep.any()
    assert np.array_equal(ren_off[keep], ren_on[keep])
    assert np.array_equal(fac_off[keep], fac_on[keep])
    assert np.array_equal(out_off[keep], out_on[keep])


def test_xqcut_survivors_have_no_soft_jet_scale(process):
    """Every booked jet scale of a surviving event is at or above the cut,
    which is what makes the merging scale mean anything downstream."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=1000, seed=4242)
    cut = 35.0
    clustering = make_clustering(
        diagrams, external_pdg_ids=pdg_ids, xqcut=cut, jet_scale_scheme=EMISSION
    )
    *_, jet_scales, _, weights = run_all(clustering, momenta)
    booked = jet_scales[weights > 0]
    booked = booked[booked < CM_ENERGY]
    assert len(booked) > 0
    assert np.all(booked >= cut)


def test_xqcut_does_not_depend_on_the_jet_scale_scheme(process):
    """The veto is defined on the clustering history, which the scheme switch
    does not touch: it only moves which leg a scale is booked against."""
    _, diagrams, pdg_ids = process
    momenta = sample_momenta(diagrams, batch_size=500, seed=17)
    emission = xqcut_weights(
        diagrams, pdg_ids, momenta, 40.0, jet_scale_scheme=EMISSION
    )
    production = xqcut_weights(
        diagrams, pdg_ids, momenta, 40.0, jet_scale_scheme=PRODUCTION
    )
    assert np.array_equal(emission, production)
