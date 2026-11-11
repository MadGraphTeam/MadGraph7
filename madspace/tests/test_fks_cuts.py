"""The generation cuts of the FKS integrand (FKSCuts), those of cuts.f of
MadFKS (passcuts_leptons and passcuts_jets).

FKSCuts is checked against the cuts that sigint applied at the points of the
integrand reference (test_data/fks_integrand), at the event and at the soft
counter-event, and against madfks_cuts below on random events and
counter-events of FKS configurations with up to four final-state partons and
with leptons, for the kT, Cambridge/Aachen, anti-kT and a generalized kT
algorithm. madfks_cuts is cuts.f in Python, with a clustering written as
fastjet clusters, which test_clustering_is_fastjet checks where the Python
bindings of fastjet are installed.
"""

import functools
import math
import os

import numpy as np
import pytest
from fks_testing import evaluate, reference, reference_born_mapping
from test_fks_integrand import DATA as INTEGRAND_DATA
from test_fks_integrand import IDS as INTEGRAND_IDS
from test_fks_integrand import channels

import madspace as ms
from madspace.fks import fks_configuration_mapping, fks_cuts

KINEMATICS = ("momenta", "soft_momenta", "collinear_momenta", "soft_collinear_momenta")
LEPTONS = {11, 13, 15}


def mg_eta(p):
    """eta of kin_functions.f."""
    theta = math.acos(p[3] / math.sqrt(p[1] ** 2 + p[2] ** 2 + p[3] ** 2))
    if abs(theta) < 1e-5:
        return 25.0
    if abs(theta - math.pi) < 1e-5:
        return -25.0
    return -math.log(math.tan(theta / 2))


def mg_delta_phi(p1, p2):
    """DELTA_PHI of kin_functions.f."""
    cosine = (p1[1] * p2[1] + p1[2] * p2[2]) / (
        math.hypot(p1[1], p1[2]) * math.hypot(p2[1], p2[2])
    )
    return math.acos(min(max(cosine, -0.99999999), 0.99999999))


def jet_eta(p):
    """PseudoJet::pseudorapidity of fastjet."""
    if p[3] == 0:
        return 0.0
    theta = math.atan(math.hypot(p[1], p[2]) / p[3])
    if theta < 0:
        theta += math.pi
    return -math.log(math.tan(theta / 2))


def genkt_jets(partons, power, radius):
    """The inclusive jets of the pp generalized kT algorithm of fastjet with
    the E scheme (rapidity and azimuth in [0, 2 pi)), without the partons of
    zero transverse momentum."""

    def rap_phi(p):
        rapidity = 0.5 * math.log((p[0] + p[3]) / (p[0] - p[3]))
        return rapidity, math.atan2(p[2], p[1]) % (2 * math.pi)

    pseudo = [np.array(p, dtype=float) for p in partons if p[1] ** 2 + p[2] ** 2 > 0]
    jets = []
    while pseudo:
        kt2 = [p[1] ** 2 + p[2] ** 2 for p in pseudo]
        best = (kt2[0] ** power, 0, None)
        for i, p in enumerate(pseudo):
            beam = kt2[i] ** power
            if beam < best[0]:
                best = (beam, i, None)
            y_i, phi_i = rap_phi(p)
            for j in range(i + 1, len(pseudo)):
                y_j, phi_j = rap_phi(pseudo[j])
                dphi = abs(phi_i - phi_j)
                dphi = 2 * math.pi - dphi if dphi > math.pi else dphi
                d = min(beam, kt2[j] ** power) * ((y_i - y_j) ** 2 + dphi**2)
                d /= radius**2
                if d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        if j is None:
            jets.append(pseudo.pop(i))
        else:
            pseudo[i] = pseudo[i] + pseudo.pop(j)
    return jets


def madfks_cuts(p, pdgs, cuts):
    """passcuts_leptons and passcuts_jets of cuts.f on one kinematics in the
    lab frame."""
    if not p[0][0] > 0 or not np.all(np.abs(p) <= 1e32):
        return False
    final = range(2, len(pdgs))
    for i in final:
        if abs(pdgs[i]) not in LEPTONS:
            continue
        if cuts["ptl"] > 0 and math.hypot(p[i][1], p[i][2]) < cuts["ptl"]:
            return False
        if cuts["etal"] > 0 and abs(mg_eta(p[i])) > cuts["etal"]:
            return False
        if pdgs[i] > 0:
            continue
        for j in final:
            if pdgs[j] not in LEPTONS:
                continue
            r2 = mg_delta_phi(p[i], p[j]) ** 2 + (mg_eta(p[i]) - mg_eta(p[j])) ** 2
            pair = p[i] + p[j]
            m2 = pair[0] ** 2 - pair[1] ** 2 - pair[2] ** 2 - pair[3] ** 2
            pairs = [("drll", "mll")]
            if pdgs[i] == -pdgs[j]:
                pairs.append(("drll_sf", "mll_sf"))
            for dr, mass in pairs:
                if cuts[dr] > 0 and r2 < cuts[dr] ** 2:
                    return False
                if cuts[mass] > 0 and m2 < cuts[mass] ** 2:
                    return False
    partons = [
        p[i]
        for i in final
        if 1 <= abs(pdgs[i]) <= cuts["maxjetflavor"] or abs(pdgs[i]) == 21
    ]
    if cuts["ptj"] > 0 and len(partons) > 1:
        if sum(abs(q[0] / p[0][0]) < 1e-8 for q in partons) > 1:
            return False
        jets = [
            jet
            for jet in genkt_jets(partons, cuts["jetalgo"], cuts["jetradius"])
            if jet[1] ** 2 + jet[2] ** 2 >= cuts["ptj"] ** 2
            and (cuts["etaj"] <= 0 or abs(jet_eta(jet)) <= cuts["etaj"])
        ]
        if len(jets) not in (len(partons), len(partons) - 1):
            return False
    return True


@pytest.mark.parametrize("path", INTEGRAND_DATA, ids=INTEGRAND_IDS)
def test_against_madfks(path):
    """At the event and at the soft counter-event of each point with a valid
    Born, FKSCuts passes where sigint passed (passcuts_n1body and
    passcuts_nbody), and with event projection the collinear counter-event
    takes the cut of the soft one."""
    for channel in channels(path):
        out, index = channel.event
        function = fks_cuts(channel.born, channel.configurations, **channel.cuts())
        for name, key in (("momenta", "event"), ("soft_momenta", "soft")):
            (mask,) = evaluate(function, out[name], fks_index=index)
            np.testing.assert_array_equal(
                mask.astype(bool), channel.field("cuts", key), err_msg=name
            )
        np.testing.assert_array_equal(
            channel.field("cuts", "collinear"), channel.field("cuts", "soft")
        )


def test_reference_cuts_are_active():
    """The references with cuts fail some of the jet and lepton cuts, at the
    Born and at the event alone."""
    for name, kinds in (
        ("epem_jj_kt", {"soft", "event"}),
        ("epem_jj_antikt", {"soft", "event"}),
        ("pp_epem_cuts", {"soft", "event"}),
    ):
        path = INTEGRAND_DATA[INTEGRAND_IDS.index(name)]
        failed = set()
        for channel in channels(path):
            soft, event = channel.field("cuts", "soft"), channel.field("cuts", "event")
            failed |= {"soft"} if not soft.all() else set()
            failed |= {"event"} if (soft & ~event).any() else set()
        assert failed == kinds, name


# configurations with up to four final-state partons, with leptons
RANDOM_DATA = [
    "epem_uuxg_P0_epem_uuxg",
    "pp_epemj_P0_ug_uemep",
    "uux_uux_mix_P0_uux_uux",
]
# kT, Cambridge/Aachen, anti-kT and a generalized kT, in units of sqrt(s_hat)
# for the transverse momenta and masses
RANDOM_CUTS = [
    dict(jetalgo=1.0, jetradius=0.7, ptj=0.05, etaj=2.5),
    dict(jetalgo=0.0, jetradius=1.0, ptj=0.03),
    dict(jetalgo=-1.0, jetradius=0.4, ptj=0.05, etaj=1.5, ptl=0.02, etal=2.5),
    dict(jetalgo=0.5, jetradius=0.6, ptj=0.02, drll=0.4, mll=0.05, mll_sf=0.1),
    dict(jetalgo=-1.0, jetradius=1.2, ptj=0.1, maxjetflavor=2, drll_sf=0.8),
]
ENERGIES = ("ptj", "ptl", "mll", "mll_sf")


@functools.lru_cache(maxsize=None)
def random_kinematics(name, count=500, seed=4711):
    """FKSKinematics of a kinematics reference on random numbers, at random
    configurations, with the PDG codes of the real process of each event and
    sqrt(s) of the run."""
    data = reference(
        os.path.join(os.path.dirname(__file__), "test_data", "fks", name + ".json")
    )
    born = reference_born_mapping(data)
    configurations = data["fks_configurations"]
    mapping = fks_configuration_mapping(born, configurations, 0.0)
    kinematics = ms.FKSKinematics(mapping)
    rng = np.random.default_rng(seed)
    r = rng.random((count, born.random_dim() + 3))
    index = rng.integers(len(configurations["configurations"]), size=count)
    names = list(mapping.counter_event_types().keys())
    out = dict(zip(names, evaluate(kinematics, r, fks_index=index)))
    pdgs = [configurations["configurations"][k]["real_pdgs"] for k in index]
    sqrt_s = 2 * math.sqrt(data["ebeam"][0] * data["ebeam"][1])
    return born, configurations, out, index, pdgs, sqrt_s


@pytest.mark.parametrize("name", RANDOM_DATA)
@pytest.mark.parametrize(
    "cuts", RANDOM_CUTS, ids=["kt", "cambridge", "antikt", "genkt", "antikt_wide"]
)
def test_against_python(name, cuts):
    """On random events and counter-events (with a zero and with two
    collinear partons), FKSCuts is madfks_cuts, and the cuts pass and fail."""
    born, configurations, out, index, pdgs, sqrt_s = random_kinematics(name)
    scale = sqrt_s / 10 if "pp_" in name or "uux" in name else sqrt_s
    cuts = dict(cuts)
    for key in ENERGIES:
        if key in cuts:
            cuts[key] *= scale
    full = dict(
        jetalgo=1.0,
        jetradius=0.7,
        ptj=0.0,
        etaj=-1.0,
        maxjetflavor=4,
        ptl=0.0,
        etal=-1.0,
        drll=0.0,
        drll_sf=0.0,
        mll=0.0,
        mll_sf=0.0,
    )
    full.update(cuts)
    function = fks_cuts(born, configurations, **full)
    outcomes = set()
    for kinematics in KINEMATICS:
        (mask,) = evaluate(function, out[kinematics], fks_index=index)
        weight = out[{"momenta": "weight"}.get(kinematics, kinematics[:-7] + "det")]
        for k in np.flatnonzero(weight != 0):
            expected = madfks_cuts(out[kinematics][k], pdgs[k], full)
            assert bool(mask[k]) == expected, (kinematics, k)
            outcomes.add(expected)
    assert outcomes == {True, False}


def test_clustering_is_fastjet():
    """genkt_jets gives the inclusive jets of fastjet on random partons, for
    each algorithm."""
    fastjet = pytest.importorskip("fastjet")
    rng = np.random.default_rng(42)
    for power, radius in ((1.0, 0.7), (0.0, 1.0), (-1.0, 0.4), (0.5, 0.6)):
        definition = fastjet.JetDefinition(fastjet.genkt_algorithm, radius, power)
        for _ in range(200):
            count = rng.integers(2, 7)
            momenta = []
            for _ in range(count):
                pt = rng.exponential(30.0)
                y, phi = rng.normal(0, 1.5), rng.uniform(0, 2 * math.pi)
                momenta.append(
                    [pt * math.cosh(y), pt * math.cos(phi), pt * math.sin(phi)]
                    + [pt * math.sinh(y)]
                )
            ours = sorted(
                (math.hypot(j[1], j[2]) for j in genkt_jets(momenta, power, radius)),
                reverse=True,
            )
            sequence = fastjet.ClusterSequence(
                [fastjet.PseudoJet(p[1], p[2], p[3], p[0]) for p in momenta],
                definition,
            )
            theirs = sorted(
                (jet.pt() for jet in sequence.inclusive_jets(0.0)), reverse=True
            )
            np.testing.assert_allclose(ours, theirs, rtol=1e-12)
