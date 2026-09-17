"""2 -> 1 phase space and asymmetric beams (PhaseSpaceMapping beam_rapidity)."""

import math

import numpy as np
import pytest
from pytest import approx

import madspace as ms

N = 2000


@pytest.fixture
def rng():
    return np.random.default_rng(4321)


def boost_z(p, y):
    q = p.copy()
    q[..., 0] = p[..., 0] * math.cosh(y) + p[..., 3] * math.sinh(y)
    q[..., 3] = p[..., 3] * math.cosh(y) + p[..., 0] * math.sinh(y)
    return q


def two_to_one_topology(mass):
    return ms.Topology(ms.Diagram([0.0, 0.0], [mass], [], [["i0", "i1", "o0"]]))


def test_two_to_one(rng):
    mass, sqrt_s = 91.188, 13000.0
    mapping = ms.PhaseSpaceMapping(two_to_one_topology(mass), sqrt_s)
    assert mapping.random_dim() == 1
    r = rng.random((N, 1))
    p, x1, x2, det = map(np.asarray, mapping.map_forward([r], []))

    tau = mass**2 / sqrt_s**2
    assert x1 * x2 == approx(tau)
    assert p[:, 0] + p[:, 1] == approx(p[:, 2], abs=1e-9)
    assert p[:, 0, 0] == approx(x1 * sqrt_s / 2)
    assert p[:, 1, 0] == approx(x2 * sqrt_s / 2)
    mass2 = p[:, 2, 0] ** 2 - np.sum(p[:, 2, 1:] ** 2, axis=1)
    assert mass2 == approx(mass**2)
    # dPhi_1 = 2 pi delta(s_hat - m^2), dx1 dx2 = |ln tau| / s ds_hat dr
    assert det == approx(2 * math.pi * abs(math.log(tau)) / sqrt_s**2)

    r_inv, det_inv = mapping.map_inverse([p, x1, x2], [])
    assert np.asarray(r_inv) == approx(r)
    assert np.asarray(det_inv) * det == approx(1.0)

    flat = ms.PhaseSpaceMapping([0.0, 0.0, mass], sqrt_s)
    assert np.asarray(flat.map_forward([r], [])[0]) == approx(p)


def test_two_to_one_needs_pdf():
    with pytest.raises(ValueError):
        ms.PhaseSpaceMapping(two_to_one_topology(125.0), 13000.0, leptonic=True)
    with pytest.raises(ValueError):
        ms.PhaseSpaceMapping(two_to_one_topology(0.0), 13000.0)


@pytest.mark.parametrize("leptonic", [False, True], ids=["hadronic", "leptonic"])
@pytest.mark.parametrize("mode", ["rambo", "propagator"])
def test_beam_rapidity(rng, mode, leptonic):
    e1, e2 = 7000.0, 4000.0
    e_cm = 2 * math.sqrt(e1 * e2)
    y0 = 0.5 * math.log(e1 / e2)
    masses = [0.0, 0.0, 0.0, 0.0, 173.0]
    kwargs = dict(leptonic=leptonic, mode=getattr(ms.PhaseSpaceMapping, mode))
    sym = ms.PhaseSpaceMapping(masses, e_cm, **kwargs)
    lab = ms.PhaseSpaceMapping(masses, e_cm, beam_rapidity=y0, **kwargs)
    r = rng.random((N, sym.random_dim()))
    p_sym, x1_sym, x2_sym, det_sym = map(np.asarray, sym.map_forward([r], []))
    p_lab, x1, x2, det = map(np.asarray, lab.map_forward([r], []))
    x1, x2 = np.broadcast_to(x1, (N,)), np.broadcast_to(x2, (N,))
    ok = det_sym > 0

    # same point, same weight: the lab frame is a longitudinal boost away
    assert det == approx(det_sym)
    assert p_lab[ok] == approx(boost_z(p_sym[ok], y0), abs=1e-8)
    # the incoming partons carry the momentum fractions of their own beam
    assert p_lab[ok, 0, 0] == approx(x1[ok] * e1)
    assert p_lab[ok, 1, 0] == approx(x2[ok] * e2)

    r_inv, det_inv = lab.map_inverse([p_lab, x1, x2], [])
    assert np.asarray(r_inv)[ok] == approx(r[ok], abs=1e-6)
    assert np.asarray(det_inv)[ok] * det[ok] == approx(1.0, rel=1e-5)


def test_beam_rapidity_cut_in_lab_frame(rng):
    """An eta cut is applied to the boosted (lab) momenta."""
    e1, e2 = 7000.0, 1000.0
    e_cm = 2 * math.sqrt(e1 * e2)
    y0 = 0.5 * math.log(e1 / e2)
    pids = [2, -2, 11, -11]
    cuts = ms.Cuts([
        ms.CutItem(observable=ms.Observable(pids, "eta", [[11, -11]]), min=0.0)
    ])
    mapping = ms.PhaseSpaceMapping(
        [0.0] * 4, e_cm, cuts=cuts, beam_rapidity=y0, mode=ms.PhaseSpaceMapping.rambo
    )
    free = ms.PhaseSpaceMapping(
        [0.0] * 4, e_cm, beam_rapidity=y0, mode=ms.PhaseSpaceMapping.rambo
    )
    r = rng.random((N, mapping.random_dim()))
    p, _, _, det = map(np.asarray, mapping.map_forward([r], []))
    p_free, _, _, det_free = map(np.asarray, free.map_forward([r], []))
    eta = np.arctanh(p_free[:, 2:, 3] / np.linalg.norm(p_free[:, 2:, 1:], axis=2))
    passed = (eta > 0).all(axis=1) & (det_free > 0)
    assert passed.any() and not passed.all()
    assert ((det > 0) == passed).all()


@pytest.mark.parametrize("pdf_asymmetric_only", [False, True], ids=["energies", "pdfs"])
def test_mirror_beams(rng, pdf_asymmetric_only):
    """mirror_index 1: leg 1 from beam 2, i.e. rotated in the beams' frame"""
    e1, e2 = (6500.0, 6500.0) if pdf_asymmetric_only else (7000.0, 4000.0)
    e_cm = 2 * math.sqrt(e1 * e2)
    y0 = 0.5 * math.log(e1 / e2)
    masses = [0.0, 0.0, 0.0, 0.0]
    plain = ms.PhaseSpaceMapping(masses, e_cm, beam_rapidity=y0)
    mirror = ms.PhaseSpaceMapping(masses, e_cm, beam_rapidity=y0, mirror_beams=True)
    assert mirror.mirror_beams() and not plain.mirror_beams()
    r = rng.random((N, plain.random_dim()))
    p, x1, x2, det = map(np.asarray, plain.map_forward([r], []))
    zeros, ones = np.zeros(N, dtype=np.int32), np.ones(N, dtype=np.int32)
    p0, x1_0, x2_0, det0 = map(np.asarray, mirror.map_forward([r], [zeros]))
    p1, x1_1, x2_1, det1 = map(np.asarray, mirror.map_forward([r], [ones]))
    ok = det > 0
    assert p0[ok] == approx(p[ok], abs=1e-8)
    assert det1 == approx(det)
    assert x1_1 == approx(x1)
    assert x2_1 == approx(x2)
    # leg 1 now moves along -z with its fraction of beam 2, leg 2 along +z
    assert (p1[ok, 0, 3] < 0).all()
    assert p1[ok, 0, 0] == approx(x1[ok] * e2)
    assert (p1[ok, 1, 3] > 0).all()
    assert p1[ok, 1, 0] == approx(x2[ok] * e1)
    # which is the beams'-frame rotation of the unmirrored event
    rot = boost_z(p[ok], -y0)
    rot[..., 2:] *= -1
    assert p1[ok] == approx(boost_z(rot, y0), abs=1e-8)
    r_inv, det_inv = mirror.map_inverse([p1, x1_1, x2_1], [ones])
    assert np.asarray(r_inv)[ok] == approx(r[ok], abs=1e-6)


# --- the mirror and the cuts -------------------------------------------------
#
# The initial-state mirror is the rotation by pi about x, (E, px, py, pz) ->
# (E, px, -py, -pz), which moves each leg onto the other beam. With mirror_beams
# the mapping draws the orientation first, so the cuts and the written event see
# the same momenta. Without it the Integrand mirrors accepted events afterwards,
# which only reproduces the same sample if no cut can tell the two orientations
# apart; Integrand refuses the combination otherwise.


def mirror(p):
    """Rotate by pi about the x axis, as kernel_mirror_momenta does."""
    q = p.copy()
    q[..., 2:] *= -1
    return q


def eta(p):
    return np.arctanh(p[..., 3] / np.linalg.norm(p[..., 1:], axis=-1))


PIDS = [2, -2, 11, -11]


def lepton_cut(observable, **bounds):
    return ms.Cuts([
        ms.CutItem(
            observable=ms.Observable(
                PIDS, observable, [[11, -11]], name=f"lepton-{observable}"
            ),
            **bounds,
        )
    ])


@pytest.mark.parametrize(
    "observable, invariant",
    [
        ("e", True),
        ("px", True),
        ("py", False),
        ("pz", False),
        ("mass", True),
        ("pt", True),
        ("p_mag", True),
        ("phi", False),
        ("theta", False),
        ("y", False),
        ("y_abs", True),
        ("eta", False),
        ("eta_abs", True),
    ],
)
def test_observable_mirror_invariance(observable, invariant):
    obs = ms.Observable(PIDS, observable, [[11, -11]])
    assert obs.mirror_invariant() == invariant


@pytest.mark.parametrize(
    "observable, invariant",
    [("delta_eta", False), ("delta_phi", False), ("delta_r", True),
     ("pair_mass", True)],
)
def test_pairwise_observable_mirror_invariance(observable, invariant):
    obs = ms.Observable(PIDS, observable, [[11, -11], [11, -11]])
    assert obs.mirror_invariant() == invariant


def test_sqrt_s_is_mirror_invariant():
    assert ms.Observable(PIDS, "sqrt_s", []).mirror_invariant()


def test_ordering_observable_decides_mirror_invariance():
    """Sorting by a quantity that flips picks another particle out of the event."""
    kwargs = dict(select_pids=[[11, -11]], order_indices=[1])
    by_pt = ms.Observable(PIDS, "pt", order_observable="pt", **kwargs)
    by_eta = ms.Observable(PIDS, "pt", order_observable="eta", **kwargs)
    assert by_pt.mirror_invariant()
    assert not by_eta.mirror_invariant()


def test_cut_on_absent_particle_is_mirror_invariant():
    """It matched nothing, so it is the constant 0 whatever the orientation."""
    obs = ms.Observable(PIDS, "eta", [[5, -5]])
    assert obs.mirror_invariant()
    assert ms.Cuts([ms.CutItem(observable=obs, min=0.0)]).mirror_invariant()


def test_cuts_report_the_offending_cuts_by_name():
    cuts = ms.Cuts([
        ms.CutItem(
            observable=ms.Observable(PIDS, "pt", [[11, -11]], name="lepton-pt"),
            min=10.0,
        ),
        ms.CutItem(
            observable=ms.Observable(PIDS, "eta", [[11, -11]], name="lepton-eta"),
            min=0.0,
        ),
        ms.CutItem(
            observable=ms.Observable(PIDS, "y", [[11, -11]], name="lepton-y"),
            min=0.0,
        ),
    ])
    assert not cuts.mirror_invariant()
    assert cuts.non_mirror_invariant_cuts() == ["lepton-eta", "lepton-y"]
    mapping = ms.PhaseSpaceMapping([0.0] * 4, 13000.0, cuts=cuts)
    assert mapping.cuts().non_mirror_invariant_cuts() == ["lepton-eta", "lepton-y"]


@pytest.mark.parametrize(
    "observable, bounds, separates",
    [("eta", dict(min=0.0), True), ("eta_abs", dict(max=1.0), False)],
    ids=["eta", "eta_abs"],
)
def test_mirror_beams_cuts_the_written_orientation(rng, observable, bounds, separates):
    """With mirror_beams the cuts act on the momenta the mapping returns.

    A signed eta cut tells the two orientations apart -- the configuration the
    post-cut mirror may not be used for -- while |eta| cannot, which is why the
    post-cut path is sound for the cuts mg7 ships.
    """
    e_cm, mode = 13000.0, ms.PhaseSpaceMapping.rambo
    cut = ms.PhaseSpaceMapping(
        [0.0] * 4, e_cm, cuts=lepton_cut(observable, **bounds),
        mirror_beams=True, mode=mode
    )
    free = ms.PhaseSpaceMapping([0.0] * 4, e_cm, mirror_beams=True, mode=mode)
    r = rng.random((N, cut.random_dim()))
    accepted, momenta, physical = {}, {}, {}
    for index in (0, 1):
        condition = np.full(N, index, dtype=np.int32)
        accepted[index] = np.asarray(cut.map_forward([r], [condition])[3]) > 0
        p, _, _, det = map(np.asarray, free.map_forward([r], [condition]))
        momenta[index], physical[index] = p, det > 0

    ok = physical[0]
    assert ok.any()
    # orientation 1 is the pi rotation of orientation 0
    assert momenta[1][ok] == approx(mirror(momenta[0][ok]), abs=1e-7)
    # and both are cut on the momenta that come out, not on the other orientation
    for index in (0, 1):
        obs = eta(momenta[index][:, 2:])
        if observable == "eta_abs":
            passed = (np.abs(obs) < bounds["max"]).all(axis=1)
        else:
            passed = (obs > bounds["min"]).all(axis=1)
        assert (accepted[index][ok] == (passed & physical[index])[ok]).all()
    # which matters only if the cut can tell the two orientations apart
    differ = (accepted[0][ok] != accepted[1][ok]).any()
    assert differ == separates


# --- the mirror and the matrix element ---------------------------------------


@pytest.fixture(scope="module")
def running_coupling(tmp_path_factory):
    """A RunningCoupling over a hand-written alpha_s grid: the Integrand needs
    one to build, and nothing here depends on the values."""
    info = tmp_path_factory.mktemp("alphas") / "grid.info"
    info.write_text(
        "AlphaS_Qs: [1.0, 10.0, 100.0, 1000.0, 10000.0]\n"
        "AlphaS_Vals: [0.4, 0.2, 0.12, 0.09, 0.08]\n"
    )
    return ms.RunningCoupling(ms.AlphaSGrid(str(info)))


PID_OPTIONS = [[2, -2], [-2, 2]]


def build_integrand(running_coupling, cuts=None, mirror_beams=False,
                    flavor_mirror=(True, True)):
    """A 2 -> 2 integrand over a beam-swapped pair of flavors. The matrix
    element is never called; only the compute graph is built."""
    e_cm = 13000.0
    me = ms.MatrixElement(
        matrix_element_index=0,
        particle_count=4,
        inputs=[
            ms.MatrixElement.momenta_in,
            ms.MatrixElement.flavor_in,
            ms.MatrixElement.alpha_s_in,
        ],
        outputs=[
            ms.MatrixElement.matrix_element_out,
            ms.MatrixElement.diagram_amp2_out,
            ms.MatrixElement.color_index_out,
            ms.MatrixElement.helicity_index_out,
            ms.MatrixElement.diagram_index_out,
        ],
        diagram_count=2,
        sample_random_inputs=True,
    )
    diff_xs = ms.DifferentialCrossSection(
        matrix_element=me,
        cm_energy=e_cm,
        running_coupling=running_coupling,
        energy_scale=ms.CachedScale(),
        pid_options=PID_OPTIONS,
    )
    mapping = ms.PhaseSpaceMapping(
        [0.0] * 4, e_cm, cuts=cuts, mirror_beams=mirror_beams
    )
    return ms.Integrand(
        mapping=mapping,
        diff_xs=[diff_xs],
        pid_options=PID_OPTIONS,
        running_coupling=running_coupling,
        energy_scale=ms.EnergyScale(4),
        flavor_mirror=list(flavor_mirror),
        channel_indices=[0],
    )


def test_post_cut_mirror_accepts_mirror_invariant_cuts(running_coupling):
    build_integrand(running_coupling, cuts=lepton_cut("eta_abs", max=2.5))


def test_post_cut_mirror_refuses_a_cut_that_is_not_mirror_invariant(running_coupling):
    """The event is written mirrored, so a cut the mirror changes would be
    applied to an orientation nobody keeps."""
    with pytest.raises(ValueError, match="lepton-eta"):
        build_integrand(running_coupling, cuts=lepton_cut("eta", min=0.0))


def test_mirror_beams_allows_any_cut(running_coupling):
    """Drawing the orientation before the mapping puts the cuts on the written
    event, so the restriction does not apply."""
    build_integrand(
        running_coupling, cuts=lepton_cut("eta", min=0.0), mirror_beams=True
    )


def test_unmirrored_flavors_allow_any_cut(running_coupling):
    build_integrand(
        running_coupling, cuts=lepton_cut("eta", min=0.0), flavor_mirror=(False, False)
    )


def instruction_names(function):
    return [instruction.instruction.name for instruction in function.instructions]


def find_instruction(function, name):
    names = instruction_names(function)
    assert names.count(name) == 1, f"expected one {name}, got {names.count(name)}"
    return function.instructions[names.index(name)]


def test_matrix_element_sees_the_written_momenta(running_coupling):
    """On the post-cut path the matrix element is evaluated on the mirrored
    momenta, the ones the event is written with -- not on the momenta as
    generated. |M|^2 is invariant under the rotation for a Lorentz invariant
    matrix element, but not for a polarised one evaluated in a frame that holds
    the polarised particle at rest."""
    function = build_integrand(running_coupling).function()
    names = instruction_names(function)
    # the orientation is drawn after the cuts here
    assert names.index("mirror_momenta") > names.index("nonzero")
    mirrored = find_instruction(function, "mirror_momenta").outputs[0]
    matrix_element = find_instruction(function, "matrix_element")
    assert any(
        str(value) == str(mirrored) for value in matrix_element.inputs
    ), "the matrix element is not evaluated on the mirrored momenta"


def test_mirror_beams_needs_no_second_mirror(running_coupling):
    """With mirror_beams the mapping already returns the written orientation,
    so there is nothing left for the Integrand to pick between."""
    function = build_integrand(running_coupling, mirror_beams=True).function()
    names = instruction_names(function)
    # the only mirror is the one inside the mapping, before the cuts
    assert names.index("mirror_momenta") < names.index("nonzero")
    mirrored = find_instruction(function, "mirror_momenta").outputs[0]
    matrix_element = find_instruction(function, "matrix_element")
    assert not any(str(value) == str(mirrored) for value in matrix_element.inputs)
