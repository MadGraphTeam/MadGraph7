"""The fixed-order histograms of the FKS integrand (FKSJets, FKSHistograms,
FKSObservableHistograms and fks_histograms), with the event, its
counter-events and the n-body part of each point in the same histograms.

FKSHistograms is checked against HwU_fill and HwU_add_points of HwU.f in
Python on random points: the kinematics of a point that fall into one bin add
up, and the square of the sum enters the error. FKSJets is checked against
genkt_jets of test_cuts (the clustering of fastjet) on random events and
counter-events. FKSObservableHistograms is checked against the analyses of
MadFKS at the points of the integrand reference (test_data/fks_integrand):
outfun of madfks_plot.f fills each weight line of sigint with its momenta in
the lab frame, the n-body part at the soft counter-event, which has the
momenta of the Born. The sums and squares per bin are those of the weight
lines of each point binned in Python with the momenta of MadFKS, where
madspace bins its own kinematics with the weight lines summed per kinematics.
"""

import math

import numpy as np
import pytest
import torch
from fks_testing import evaluate
from madspace_testing import default_context
from test_fks_cuts import genkt_jets, jet_eta, random_kinematics
from test_fks_integrand import DATA, IDS, boost_z, channels

import madspace as ms
from madspace.subtraction.fks import (
    fks_configurations,
    fks_histogram_jet_count,
    fks_histogram_results,
    fks_histograms,
)

# the weight lines of sigint at each kinematics of FKSObservableHistograms
KINEMATICS_TYPES = {
    "event": (1,),
    "soft": (4,),
    "collinear": (5,),
    "soft_collinear": (6,),
    "nbody": (2, 3, 14, 15),
}
KINEMATICS_MOMENTA = {
    "event": "momenta",
    "soft": "soft_momenta",
    "collinear": "collinear_momenta",
    "soft_collinear": "soft_collinear_momenta",
    "nbody": "born_momenta",
}


def hwu_bin(x, low, high, bins):
    """The bin of x with an underflow bin 0 and an overflow bin bins + 1, None
    where x is not finite."""
    if not math.isfinite(x):
        return None
    r = (x - low) / ((high - low) / bins)
    if r < 0:
        return 0
    if r >= bins:
        return bins + 1
    return int(math.floor(r)) + 1


def hwu_point(fills, low, high, bins, sums, squares):
    """HwU_fill of each (x, w) of one point and HwU_add_points."""
    point = {}
    for x, w in fills:
        b = hwu_bin(x, low, high, bins)
        if b is not None:
            point[b] = point.get(b, 0.0) + w
    for b, w in point.items():
        sums[b] += w
        squares[b] += w * w


def call(function, *args):
    """The outputs of a FunctionGenerator on numpy arguments, the integer
    ones as int32."""
    device = torch.get_default_device()
    tensors = [
        torch.as_tensor(
            np.asarray(a),
            dtype=torch.int32 if np.asarray(a).dtype.kind == "i" else torch.float64,
        ).to(device)
        for a in args
    ]
    runtime = ms.FunctionRuntime(function.function(), default_context())
    return [torch.as_tensor(out).cpu().numpy() for out in runtime.call(tensors)]


def test_merging_is_hwu():
    """The kinematics of a point in one bin add up before the square, a value
    that is not finite or a kinematics without the jets the histogram needs
    does not count, and the values outside the range go to the underflow and
    overflow bins."""
    rng = np.random.default_rng(3)
    count, kinematics = 400, 5
    specs = [(0.0, 10.0, 7, 0), (-1.0, 1.0, 3, 1), (2.0, 3.0, 1, 2)]
    histograms = ms.FKSHistograms(
        [ms.FKSHistograms.HistItem(*spec) for spec in specs], kinematics
    )
    weights = rng.normal(size=(kinematics, count))
    jets = rng.integers(0, 4, size=(kinematics, count)).astype(np.int32)
    args = []
    for k in range(kinematics):
        args += [weights[k], jets[k]]
    values = []
    for low, high, bins, _ in specs:
        x = rng.uniform(
            low - 0.5 * (high - low), high + 0.5 * (high - low), (kinematics, count)
        )
        # several kinematics of a point in one bin, and values that do not count
        x[1, ::3] = x[0, ::3]
        x[4, ::5] = x[2, ::5]
        x[3, ::7] = np.nan
        x[2, ::11] = np.inf
        values.append(x)
        args += list(x)
    outputs = call(histograms, *args)
    for h, (low, high, bins, needed) in enumerate(specs):
        sums, squares = np.zeros(bins + 2), np.zeros(bins + 2)
        for e in range(count):
            fills = [
                (values[h][k, e], weights[k, e])
                for k in range(kinematics)
                if jets[k, e] >= needed
            ]
            hwu_point(fills, low, high, bins, sums, squares)
        ours, ours_squares = outputs[2 * h].reshape(-1), outputs[2 * h + 1].reshape(-1)
        np.testing.assert_allclose(ours, sums, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ours_squares, squares, rtol=1e-12, atol=1e-12)
        assert (squares > 0).sum() > bins / 2


def test_histogram_errors():
    item = ms.FKSHistograms.HistItem
    with pytest.raises(ValueError, match="no histograms"):
        ms.FKSHistograms([], 5)
    with pytest.raises(ValueError, match="needs bins"):
        ms.FKSHistograms([item(0.0, 1.0, 0)], 5)
    with pytest.raises(ValueError, match="needs bins"):
        ms.FKSHistograms([item(1.0, 1.0, 3)], 5)
    with pytest.raises(ValueError, match="kinematics"):
        ms.FKSHistograms([item(0.0, 1.0, 3)], 0)
    with pytest.raises(ValueError, match="kinematics"):
        ms.FKSHistograms([item(0.0, 1.0, 3)], 65)


def python_jets(p, pdgs, incoming, jetalgo, jetradius, ptj, etaj, maxjetflavor):
    """The momenta with jets of FKSJets in Python: the other particles, then
    the jets of genkt_jets with pt >= ptj and |eta| <= etaj by decreasing pt,
    and the number of jets."""
    kept, partons = [], []
    for i, (q, pdg) in enumerate(zip(p, pdgs, strict=True)):
        flavor = abs(pdg)
        if i >= incoming and (1 <= flavor <= maxjetflavor or flavor == 21):
            partons.append(q)
        else:
            kept.append(q)
    jets = [
        j
        for j in genkt_jets(partons, jetalgo, jetradius)
        if math.hypot(j[1], j[2]) >= ptj and (etaj <= 0 or abs(jet_eta(j)) <= etaj)
    ]
    jets.sort(key=lambda j: -math.hypot(j[1], j[2]))
    return kept, jets


@pytest.mark.parametrize("name", ["epem_uuxg_P0_epem_uuxg", "pp_epemj_P0_ug_uemep"])
@pytest.mark.parametrize(
    "jets",
    [
        dict(jetalgo=-1.0, jetradius=0.4, ptj=0.02, etaj=2.5),
        dict(jetalgo=1.0, jetradius=0.7, ptj=0.0),
        dict(jetalgo=0.0, jetradius=1.0, ptj=0.05, etaj=1.5),
    ],
    ids=["antikt", "kt", "cambridge"],
)
def test_jets_against_python(name, jets):
    """FKSJets on random events and counter-events, and on their Borns, are
    the other particles and the jets of the clustering of fastjet by
    decreasing pt, with zero momenta in the empty slots, in a layout that
    every configuration and the Born share."""
    born, configurations, out, index, pdgs, sqrt_s = random_kinematics(name)
    jets = dict(jets)
    jets["ptj"] = jets["ptj"] * sqrt_s
    options = dict(jetalgo=-1.0, jetradius=0.4, ptj=0.0, etaj=-1.0, maxjetflavor=4)
    options.update(jets)
    entries = fks_configurations(born, configurations)
    incoming = configurations["nincoming"]
    for is_born, kinematics in (
        (False, "momenta"),
        (False, "soft_momenta"),
        (False, "collinear_momenta"),
        (True, "born_momenta"),
    ):
        function = ms.FKSJets(entries, born=is_born, **options)
        assert function.born() == is_born
        assert function.jet_particle_count() == len(function.pids())
        jet_momenta, jet_count = evaluate(function, out[kinematics], fks_index=index)
        slots = function.slot_count()
        assert list(function.pids()[-slots:]) == [21] * slots
        for k in range(len(index)):
            config = entries[index[k]]
            process = config.born_pdgs() if is_born else list(config.pdgs)
            kept, theirs = python_jets(out[kinematics][k], process, incoming, **options)
            assert jet_count[k] == len(theirs)
            layout = jet_momenta[k]
            np.testing.assert_allclose(layout[: len(kept)], kept, rtol=0, atol=1e-12)
            ours = layout[len(kept) :]
            np.testing.assert_array_equal(ours[len(theirs) :], 0.0)
            # by decreasing pt, jets of equal pt (back to back) in any order
            pt = np.hypot(ours[: len(theirs), 1], ours[: len(theirs), 2])
            assert (np.diff(pt) <= 1e-9 * sqrt_s).all()
            np.testing.assert_allclose(
                canonical(ours[: len(theirs)]), canonical(theirs), rtol=1e-12, atol=1e-9
            )


def canonical(jets):
    """Jets in an order that does not depend on the order of jets of equal
    pt."""
    jets = np.array(jets, dtype=float).reshape(-1, 4)
    order = np.lexsort((jets[:, 3], jets[:, 2], jets[:, 1], np.round(jets[:, 0], 6)))
    return jets[order]


def test_jet_errors():
    configuration = ms.FKSConfiguration(
        i_fks=4,
        j_fks=3,
        pdgs=[11, -11, 1, -1, 21],
        colors=[1, 1, 3, -3, 8],
        charges=[-1.0, 1.0, -1 / 3, 1 / 3, 0.0],
        masses=[0.0] * 5,
        fks_j_from_i=[[], [], [], [], [3, 4]],
    )
    with pytest.raises(ValueError, match="radius must be positive"):
        ms.FKSJets([configuration], jetradius=0.0)
    other = ms.FKSConfiguration(
        i_fks=4,
        j_fks=3,
        pdgs=[11, -11, 13, -1, 21],
        colors=[1, 1, 1, -3, 8],
        charges=[-1.0, 1.0, -1.0, 1 / 3, 0.0],
        masses=[0.0] * 5,
        fks_j_from_i=[[], [], [], [], [3, 4]],
    )
    with pytest.raises(ValueError):
        ms.FKSJets([configuration, other])


def lab(c):
    """The momenta of a weight line of MadFKS in the lab frame, as outfun of
    madfks_plot.f boosts them: those of the dump are in the reduced partonic
    center-of-mass frame, and y_lab = y - y_bst."""
    return boost_z(np.array(c["momenta"])[None], np.array([-c["y_bst"]]))[0]


def particle(p, pdgs, pdg):
    return next(q for q, code in zip(p, pdgs, strict=True) if code == pdg)


def rapidity(q):
    return 0.5 * math.log((q[0] + q[3]) / (q[0] - q[3]))


# per reference: (name, observable of Observable, observable in Python on the
# momenta of MadFKS with its PDG codes and the jets, min, max, bins), with no
# bin edge at a value of the Born (a jet energy of sqrt(s) / 2)
JET = list(ms.Observable.jet_pids)
OBSERVABLES = {
    "pp_epem": [
        (
            "m_ll",
            dict(observable="mass", select_pids=[[11, -11]], sum_momenta=True),
            lambda p, pdgs, jets: math.sqrt(
                max(
                    (particle(p, pdgs, 11) + particle(p, pdgs, -11))[0] ** 2
                    - ((particle(p, pdgs, 11) + particle(p, pdgs, -11))[1:] ** 2).sum(),
                    0.0,
                )
            ),
            20.0,
            300.0,
            14,
        ),
        (
            "y_em",
            dict(observable="y", select_pids=[[11]]),
            lambda p, pdgs, jets: rapidity(particle(p, pdgs, 11)),
            -4.0,
            4.0,
            16,
        ),
        (
            "pt_j1",
            dict(
                observable="pt",
                select_pids=[JET],
                order_observable="pt",
                order_indices=[1],
            ),
            lambda p, pdgs, jets: math.hypot(jets[0][1], jets[0][2]) if jets else None,
            0.0,
            100.0,
            10,
        ),
    ],
    "epem_jj": [
        # the most energetic jet: two jets of e+ e- have the same pt
        (
            "e_j1",
            dict(
                observable="e",
                select_pids=[JET],
                order_observable="e",
                order_indices=[1],
            ),
            lambda p, pdgs, jets: max(j[0] for j in jets) if jets else None,
            0.0,
            600.0,
            13,
        ),
        (
            "pt_j2",
            dict(
                observable="pt",
                select_pids=[JET],
                order_observable="pt",
                order_indices=[2],
            ),
            lambda p, pdgs, jets: (
                math.hypot(jets[1][1], jets[1][2]) if len(jets) > 1 else None
            ),
            0.0,
            400.0,
            8,
        ),
    ],
    "pp_ttx": [
        (
            "pt_t",
            dict(observable="pt", select_pids=[[6]]),
            lambda p, pdgs, jets: math.hypot(*particle(p, pdgs, 6)[1:3]),
            0.0,
            500.0,
            10,
        ),
    ],
    "t_bw": [
        (
            "e_b",
            dict(observable="e", select_pids=[[5]]),
            lambda p, pdgs, jets: particle(p, pdgs, 5)[0],
            0.0,
            100.0,
            20,
        ),
    ],
}


def observables_of(path):
    name = IDS[DATA.index(path)]
    for key, observables in OBSERVABLES.items():
        if name == key or name.startswith(key + "_"):
            return observables
    return []


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_against_madfks(path):
    """At the points of the integrand reference, the histograms of madspace
    with the weight lines of MadFKS summed per kinematics are those of the
    analyses of MadFKS: each weight line binned with its own momenta in the
    lab frame, those of a point added up per bin before the square."""
    observables = observables_of(path)
    if not observables:
        pytest.skip("no observables for this reference")
    for channel in channels(path):
        run = channel.run_card
        incoming = channel.incoming
        jets = dict(
            jetalgo=float(run.get("jetalgo", -1.0)),
            jetradius=float(run.get("jetradius", 0.4)),
            ptj=max(float(run.get("ptj", 0.0)), 5.0),
            etaj=float(run.get("etaj", -1.0)),
            maxjetflavor=int(float(run.get("maxjetflavor", 4))),
        )
        specs = [
            dict(name=name, observable=kwargs, min=low, max=high, bin_count=bins)
            for name, kwargs, _, low, high, bins in observables
        ]
        function = fks_histograms(channel.born, channel.configurations, specs, **jets)
        out, _ = channel.event
        weights = {name: np.zeros(len(channel.points)) for name in KINEMATICS_TYPES}
        sums = [np.zeros(s["bin_count"] + 2) for s in specs]
        squares = [np.zeros(s["bin_count"] + 2) for s in specs]
        index = np.zeros(len(channel.points), dtype=np.int32)
        for k, point in enumerate(channel.points):
            index[k] = point["nfks_picked"] - 1
            fills = [[] for _ in specs]
            for c in point["contributions"]:
                kinematics = next(
                    (n for n, types in KINEMATICS_TYPES.items() if c["type"] in types),
                    None,
                )
                if kinematics is None:
                    continue
                weights[kinematics][k] += c["weight"]
                p = lab(c)
                _, found = python_jets(p, c["pdg"], incoming, **jets)
                for h, (_, _, value, *_) in enumerate(observables):
                    x = value(p, c["pdg"], found)
                    fills[h].append((math.nan if x is None else x, c["weight"]))
            for h, spec in enumerate(specs):
                hwu_point(
                    fills[h],
                    spec["min"],
                    spec["max"],
                    spec["bin_count"],
                    sums[h],
                    squares[h],
                )
        args = []
        for name in ms.FKSObservableHistograms.kinematics:
            args += [out[KINEMATICS_MOMENTA[name]], weights[name]]
        outputs = evaluate(function, *args, fks_index=index)
        for h in range(len(specs)):
            scale = np.abs(sums[h]).max()
            np.testing.assert_allclose(
                outputs[2 * h].reshape(-1), sums[h], rtol=1e-10, atol=1e-12 * scale
            )
            np.testing.assert_allclose(
                outputs[2 * h + 1].reshape(-1),
                squares[h],
                rtol=1e-10,
                atol=1e-12 * scale**2,
            )
            assert np.count_nonzero(sums[h]) > 1


def test_results_are_info_json():
    """fks_histogram_results gives the weighted histograms of info.json that
    MadBoard draws, normalized as EventGenerator::histograms."""
    specs = [dict(name="a", min=0.0, max=1.0, bin_count=3)]
    values = np.array([0.5, 2.0, -1.0, 4.0, 0.0])
    square_values = np.array([0.25, 5.0, 3.0, 9.0, 0.0])
    (result,) = fks_histogram_results(specs, [values, square_values], 10)
    assert set(result) == {
        "name",
        "min",
        "max",
        "bin_count",
        "bin_values",
        "bin_errors",
    }
    assert result["name"] == "a" and result["bin_count"] == 3
    np.testing.assert_allclose(result["bin_values"], values / 10)
    expected = np.sqrt(np.maximum(square_values - values**2 / 10, 0)) / 10
    np.testing.assert_allclose(result["bin_errors"], expected)
    assert len(result["bin_values"]) == len(result["bin_errors"]) == 5


def test_jet_count_of_observables():
    """The jets an observable needs, from its keyword arguments."""
    lepton = [11, -11]
    assert fks_histogram_jet_count(dict(select_pids=[lepton])) == 0
    ordered = dict(select_pids=[JET, JET], order_observable="pt", order_indices=[1, 3])
    assert fks_histogram_jet_count(ordered) == 3
    assert fks_histogram_jet_count(dict(select_pids=[JET], sum_observable=True)) == 0
    with pytest.raises(ValueError, match="no value per event"):
        fks_histogram_jet_count(dict(select_pids=[JET]))
    with pytest.raises(ValueError, match="softest"):
        fks_histogram_jet_count(dict(select_pids=[JET], order_indices=[-1]))


def test_histograms_errors():
    channel = channels(DATA[IDS.index("pp_epem")])[0]
    spec = dict(
        name="pt",
        observable=dict(observable="pt", select_pids=[[11]]),
        min=0.0,
        max=1.0,
        bin_count=2,
    )
    with pytest.raises(TypeError, match="unknown jet parameter"):
        fks_histograms(channel.born, channel.configurations, [spec], ptl=1.0)
    entries = fks_configurations(channel.born, channel.configurations)
    wrong = ms.Observable([2, -2, 11, -11, 21, 21], "pt", [[11]])
    item = ms.FKSObservableHistograms.HistItem(wrong, 0.0, 1.0, 2)
    with pytest.raises(ValueError, match="momenta with jets"):
        ms.FKSObservableHistograms(entries, [item])
