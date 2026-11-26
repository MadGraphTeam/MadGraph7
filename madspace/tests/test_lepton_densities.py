"""Dressed electrons and positrons at NLO: the luminosity of their densities
with the scale of each beam (EELuminosity with beam_scales, as DressedLeptons
of the NLO drivers builds it), that of the FKS integrand (NLOLuminosity with
lepton_luminosity, and fks_luminosities for lpp 3 and -3), the CSMapping of
a Born of dressed leptons, which keeps 1 - x of the beams, and the beams the
NLO drivers take.

The densities are those of isronlyll (ISR) and ilc500ll (ISR and the
beamstrahlung of the ILC) of the lep_densities of MadGraph, tabulated for
madspace by vendor/eepdfgrid/gen_eepdf_lhagrid.py. The luminosity with the
scales of the beams must be the sum over the components of the products of
the densities of each beam at its scale (EEPartonDensity), that with one
scale where the two are equal, and the full densities those reduced by
(1 - x)^p over (1 - x1)^p (1 - x2)^p.
"""

import copy
import functools
import json
import os

import numpy as np
import pytest
from madgraph.mg7.nlo.core import LEPTON_POWER, DressedLeptons, check_run_card
from madspace_testing import default_context

import madspace as ms
from madspace.nlo.fks import fks_luminosities

TESTS = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(os.path.dirname(TESTS))
SOURCE = os.path.join(ROOT, "Template", "Common", "Source")
SETS = ["isronlyll", "ilc500ll"]
ENERGY = 500.0


def run_card(pdlabel, lpp=(-3, 3)):
    return {
        "lpp1": lpp[0],
        "lpp2": lpp[1],
        "pdlabel": pdlabel,
        "ebeam1": ENERGY / 2,
        "ebeam2": ENERGY / 2,
    }


@functools.cache
def leptons(pdlabel, tmp_root):
    """The DressedLeptons of a set for e+ on the first beam and e- on the
    second, in a context of their own."""
    context = ms.Context(ms.cpu_device())
    directory = os.path.join(tmp_root, pdlabel)
    os.makedirs(directory, exist_ok=True)
    return DressedLeptons(SOURCE, directory, run_card(pdlabel), context), context


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return str(tmp_path_factory.mktemp("lepton_densities"))


def call(function, context, *args):
    runtime = ms.FunctionRuntime(function.function(), context)
    tensors = [np.ascontiguousarray(a) for a in args]
    outputs = [np.array(np.from_dlpack(t)) for t in runtime.call(tensors)]
    return dict(zip(function.return_types().keys(), outputs))


def points(count=200, seed=1):
    """Momentum fractions close to one and away from it, with 1 - x exact,
    and scales of the two beams from 3 to 3000 GeV."""
    rng = np.random.default_rng(seed)
    omx = 10.0 ** rng.uniform(-12, -0.3, (2, count))
    q = 10.0 ** rng.uniform(0.5, 3.5, (2, count))
    return 1 - omx[0], omx[0], 1 - omx[1], omx[1], q[0], q[1]


@pytest.mark.parametrize("pdlabel", SETS)
def test_beam_scales(pdlabel, tmp_root):
    """The luminosity with the scales of the beams is the sum over the
    components of the products of the densities of the beams at their
    scales, and that with one scale where both are the same."""
    dressed, context = leptons(pdlabel, tmp_root)
    x1, omx1, x2, omx2, q1, q2 = points()
    r = np.full(len(x1), 0.5)
    luminosity = call(dressed.full, context, x1, omx1, x2, omx2, q1, q2, r)
    # the densities of each beam per component, each in its own context with
    # the grid of the set
    offsets = dressed.offsets(0.0)
    densities = []
    for beam, (x, omx, q) in enumerate([(x1, omx1, q1), (x2, omx2, q2)]):
        own = ms.Context(ms.cpu_device())
        dressed.grid.initialize_globals(own, "beam")
        density = ms.EEPartonDensity(
            dressed.grid,
            dressed.pids[beam],
            offsets[beam],
            alpha_em=dressed.meta["alphaem"],
            lepton_mass=dressed.meta["me"],
            prefix="beam",
        )
        density.initialize_globals(own)
        densities.append(
            [
                call(density, own, x, omx, q, np.full(len(x), n, dtype=np.int32))["pdf"]
                for n in range(len(dressed.components))
            ]
        )
    expected = sum(f1 * f2 for f1, f2 in zip(*densities))
    assert np.all(expected > 0)
    np.testing.assert_allclose(luminosity["luminosity"], expected, rtol=1e-13)
    single = dressed.luminosity(0.0, beam_scales=False)
    one = call(single, context, x1, omx1, x2, omx2, q1, r)
    both = call(dressed.full, context, x1, omx1, x2, omx2, q1, q1, r)
    np.testing.assert_array_equal(both["luminosity"], one["luminosity"])


@pytest.mark.parametrize("pdlabel", SETS)
def test_reduced(pdlabel, tmp_root):
    """The reduced luminosity is the full one times (1 - x1)^p (1 - x2)^p of
    the power of the sampling."""
    dressed, context = leptons(pdlabel, tmp_root)
    x1, omx1, x2, omx2, q1, q2 = points(seed=2)
    r = np.full(len(x1), 0.5)
    full = call(dressed.full, context, x1, omx1, x2, omx2, q1, q2, r)
    reduced = call(dressed.reduced, context, x1, omx1, x2, omx2, q1, q2, r)
    np.testing.assert_allclose(
        reduced["luminosity"],
        full["luminosity"] * (omx1 * omx2) ** LEPTON_POWER,
        rtol=1e-12,
    )


def test_nlo_luminosity(tmp_root):
    """NLOLuminosity of dressed leptons: the luminosity of the leptons of the
    beams for their flavor, with its factor, none for the other flavors (the
    photons of compute_eepdf of MadFKS), summed or per flavor."""
    dressed, context = leptons("ilc500ll", tmp_root)
    x1, omx1, x2, omx2, q1, q2 = points(count=50, seed=3)
    r = np.full(len(x1), 0.5)
    expected = call(dressed.full, context, x1, omx1, x2, omx2, q1, q2, r)["luminosity"]
    flavors = [[[-11, 11], [22, 22], [-11, 22]], [[22, 11], [-11, 11]]]
    factors = [[2.0, 1.0, 1.0], [1.0, 0.5]]
    options = dict(
        factors=factors, lepton_luminosity=dressed.full, lepton_pdgs=[-11, 11]
    )
    summed = ms.NLOLuminosity(flavors, sum_flavors=True, **options)
    assert summed.lepton()
    assert list(summed.arg_types().keys())[:4] == [
        "x1",
        "x2",
        "one_minus_x1",
        "one_minus_x2",
    ]
    for configuration, factor in ((0, 2.0), (1, 0.5)):
        index = np.full(len(x1), configuration, dtype=np.int32)
        out = call(summed, context, x1, x2, omx1, omx2, q1, q2, index)
        np.testing.assert_allclose(out["luminosity"], factor * expected, rtol=1e-15)
    flavored = ms.NLOLuminosity(flavors, **options)
    for configuration, flavor, factor in ((0, 0, 2.0), (0, 1, 0.0), (1, 1, 0.5)):
        index = np.full(len(x1), configuration, dtype=np.int32)
        picked = np.full(len(x1), flavor, dtype=np.int32)
        out = call(flavored, context, x1, x2, omx1, omx2, q1, q2, picked, index)
        np.testing.assert_allclose(out["luminosity"], factor * expected, rtol=1e-15)


def test_nlo_luminosity_invalid(tmp_root):
    dressed, _ = leptons("isronlyll", tmp_root)
    single = dressed.luminosity(0.0, beam_scales=False)
    with pytest.raises(ValueError, match="scales of the beams"):
        ms.NLOLuminosity([[[-11, 11]]], lepton_luminosity=single)
    grid = ms.PdfGrid(dressed_grid_path(tmp_root))
    with pytest.raises(ValueError, match="no grid of hadrons"):
        ms.NLOLuminosity([[[-11, 11]]], grid, lepton_luminosity=dressed.full)


def dressed_grid_path(tmp_root):
    return os.path.join(tmp_root, "isronlyll", "isronlyll.dat")


def configurations():
    """The FKS configurations of e+ e- > t t~ of the reference data, with the
    flavors of a current exporter."""
    path = os.path.join(TESTS, "test_data", "fks", "epem_ttx_P0_epem_ttx.json")
    with open(path) as f:
        result = copy.deepcopy(json.load(f)["fks_configurations"])
    result["born_flavors"] = [list(result["born_pdgs"])]
    result["real_flavors"] = [[list(c["real_pdgs"])] for c in result["configurations"]]
    return result


def test_fks_luminosities(tmp_root):
    """fks_luminosities of lpp -3 and 3: the luminosity of the leptons of the
    beams, for the real emission and the n-body part, and the beams it
    rejects."""
    dressed, context = leptons("isronlyll", tmp_root)
    entries = configurations()
    real, born = fks_luminosities(
        entries, beams=(-3, 3), sum_flavors=True, lepton_luminosity=dressed.full
    )
    assert real.lepton() and born.lepton()
    x1, omx1, x2, omx2, q1, q2 = points(count=20, seed=4)
    r = np.full(len(x1), 0.5)
    expected = call(dressed.full, context, x1, omx1, x2, omx2, q1, q2, r)["luminosity"]
    index = np.zeros(len(x1), dtype=np.int32)
    for luminosity in (real, born):
        out = call(luminosity, context, x1, x2, omx1, omx2, q1, q2, index)
        np.testing.assert_allclose(out["luminosity"], expected, rtol=1e-15)
    # the beams the other way round have no flavor of their leptons
    swapped, _ = fks_luminosities(
        entries, beams=(3, -3), sum_flavors=True, lepton_luminosity=dressed.full
    )
    out = call(swapped, context, x1, x2, omx1, omx2, q1, q2, index)
    np.testing.assert_array_equal(out["luminosity"], 0.0)
    for beams, luminosity in (
        ((-3, 3), None),
        ((-3, 0), dressed.full),
        ((-3, 1), dressed.full),
        ((0, 0), dressed.full),
    ):
        with pytest.raises(ValueError):
            fks_luminosities(entries, beams=beams, lepton_luminosity=luminosity)


TOP = 173.0


def born_mapping(power=LEPTON_POWER):
    return ms.PhaseSpaceMapping(
        [0.0, 0.0, TOP, TOP], ENERGY, leptonic=False, lepton_sampling_power=power
    )


def test_cs_mapping():
    """A CSMapping on a Born of dressed leptons keeps the momentum fractions
    of the Born and their 1 - x, exact where 1 - x has no digit in x, with
    momentum conservation in the lab frame, and its CSDipoles take them."""
    born = born_mapping()
    mapping = ms.CSMapping(born, 4, 2, 3)
    rng = np.random.default_rng(5)
    r = rng.random((500, mapping.random_dim()))
    context = default_context()
    momenta, *fractions, det = call_mapping(mapping, context, r)
    _, *born_fractions, _ = call_mapping(
        born, context, np.ascontiguousarray(r[:, : born.random_dim()])
    )
    assert len(fractions) == 4
    for ours, theirs in zip(fractions, born_fractions, strict=True):
        np.testing.assert_array_equal(ours, theirs)
    x1, x2, omx1, omx2 = fractions
    assert np.any(omx1 < 1e-16) and np.all(det > 0)
    total = momenta[:, 2:].sum(1)
    incoming = momenta[:, 0] + momenta[:, 1]
    np.testing.assert_allclose(total, incoming, atol=1e-9 * ENERGY)
    np.testing.assert_allclose(incoming[:, 0], (x1 + x2) * ENERGY / 2, rtol=1e-12)
    dipoles = ms.CSDipoles([mapping], ["qg"], [1.0])
    assert list(dipoles.arg_types().keys()) == ["momenta", "x1", "x2", "omx1", "omx2"]


def call_mapping(mapping, context, r):
    """The outputs of the forward map, its Jacobian last."""
    runtime = ms.FunctionRuntime(mapping.forward_function(), context)
    return [np.array(np.from_dlpack(t)) for t in runtime.call([r])]


def decay_chain():
    """e+ e- > Z > q q~, q~ > q~ g, with the invariant of the q~ after that
    of the root."""
    return ms.Topology(
        ms.Diagram(
            [0.0, 0.0],
            [0.0, 0.0, 0.0],
            [ms.Propagator(mass=91.188, width=2.44), ms.Propagator()],
            [["i0", "i1", "p0"], ["p0", "o0", "p1"], ["p1", "o1", "o2"]],
        )
    )


@pytest.mark.parametrize(
    "topology, swap",
    [
        ([0.0, 0.0, TOP, TOP], 2),
        ([0.0, 0.0, 0.0, 0.0, 0.0], 2),
        (decay_chain(), 3),
    ],
    ids=["ttx", "t_channel", "decays"],
)
def test_mapping_inverse(topology, swap):
    """The inverse of the dressed-lepton PhaseSpaceMapping takes the outputs
    of the forward map back to its random numbers, with the reciprocal of
    its Jacobian, also where 1 - x has no digit in x. The random numbers are
    those of the invariants, then of the drawn fraction and of the beam that
    carries it, of which the inverse returns 1/4 or 3/4."""
    mapping = ms.PhaseSpaceMapping(
        topology, ENERGY, leptonic=False, lepton_sampling_power=LEPTON_POWER
    )
    context = default_context()
    rng = np.random.default_rng(6)
    r = rng.random((500, mapping.random_dim()))
    # s_hat at s and the drawn fraction at one, down to 1 - x of 1e-280
    r[:100, 0] = 10.0 ** rng.uniform(-7, -1, 100)
    r[100:200, swap - 1] = 10.0 ** rng.uniform(-7, -1, 100)
    momenta, x1, x2, omx1, omx2, det = call_mapping(mapping, context, r)
    assert np.any(omx1 < 1e-100) and np.any(omx2 < 1e-100) and np.all(det > 0)
    inverse = ms.FunctionRuntime(mapping.inverse_function(), context)
    back, density = [
        np.array(np.from_dlpack(t)) for t in inverse.call([momenta, x1, x2, omx1, omx2])
    ]
    # the beam choice where the beams differ, also where both x are one
    differ = omx1 != omx2
    assert differ.mean() > 0.99 and np.any((x1 == x2) & differ)
    np.testing.assert_array_equal(
        back[differ, swap], np.where(r[differ, swap] < 0.5, 0.25, 0.75)
    )
    # the other random numbers where 1 - x has not underflowed (where both
    # random numbers that bring x to one are small together)
    others = [i for i in range(mapping.random_dim()) if i != swap]
    normal = np.minimum(omx1, omx2) > 1e-290
    assert normal.mean() > 0.95
    np.testing.assert_allclose(
        back[normal][:, others], r[normal][:, others], rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(density * det, 1.0, rtol=1e-6)


def test_cs_mapping_invalid():
    """The incoming leptons radiate no QCD, so that a dipole of dressed
    leptons has its emitter and its spectator in the final state."""
    with pytest.raises(ValueError, match="dressed leptons"):
        ms.CSMapping(born_mapping(), 4, 2, 0)


def test_run_card():
    """The NLO drivers take dressed electrons and positrons on both beams,
    without the densities of eMELA, and no muons or other mixtures."""
    base = dict(ickkw=0, dynamical_scale_choice=-1, pdfscheme=0)
    for lpp in ((-3, 3), (3, -3), (3, 3), (0, 0), (1, 1), (1, -1)):
        check_run_card({**base, **run_card("isronlyll", lpp)})
    for lpp in ((4, -4), (-3, 0), (3, 1), (2, 1)):
        with pytest.raises(NotImplementedError):
            check_run_card({**base, **run_card("isronlyll", lpp)})
    with pytest.raises(NotImplementedError, match="eMELA"):
        check_run_card({**base, **run_card("emela", (-3, 3))})
