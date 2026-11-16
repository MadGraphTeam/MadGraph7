"""The cuts, scales, jets and histograms of the NLO integrands, independent of
the subtraction scheme.

The C++ classes live in the madspace module: NLOCuts (the generation cuts of
cuts.f of MadFKS), NLOScale (the scales of setscales.f), NLOJets (the QCD
partons of a kinematics clustered into jets) and NLOHistograms (the
histograms with the event and the counter-events of each point together).
They take the kinematics in one of several flavor layouts, the PDG codes of
its particles, chosen per event by layout_index.

nlo_scale builds the NLOScale from the scales by their names in the run card
of MadFKS. jet_observable builds an Observable on the momenta with jets of an
NLOJets from the keyword arguments of a histogram, histogram_jet_count gives
the jets it needs, and histogram_results turns the sums of the histograms
into the weighted histograms of info.json that MadBoard draws.

The subtraction schemes are submodules:

fks: the FKS subtraction (Frixione, Kunszt, Signer, arXiv:hep-ph/9512328; MadFKS,
arXiv:0908.4272) on the FKS configurations of an NLO output.
cs: the Catani-Seymour dipoles (arXiv:hep-ph/9605323, with the masses of
arXiv:hep-ph/0201036) of the real processes of an NLO output.

Their C++ classes also live in the madspace module, those of FKS with the
prefix FKS and those of Catani-Seymour with CS.
"""

import math

from .._madspace_py import NLOScale, Observable

# the names of the run card of MadFKS for the scales of NLOScale, with the
# factorization scale of both beams under the names of the run card
SCALE_NAMES = {
    "dynamical_scale_choice": ["dynamical_scale_choice"],
    "fixed_ren_scale": ["fixed_ren_scale"],
    "fixed_fac_scale": ["fixed_fac_scale"],
    "mur_ref_fixed": ["mur_ref_fixed"],
    "muf_ref_fixed": ["muf1_ref_fixed", "muf2_ref_fixed"],
    "muf1_ref_fixed": ["muf1_ref_fixed"],
    "muf2_ref_fixed": ["muf2_ref_fixed"],
    "mur_over_ref": ["mur_over_ref"],
    "muf_over_ref": ["muf1_over_ref", "muf2_over_ref"],
    "muf1_over_ref": ["muf1_over_ref"],
    "muf2_over_ref": ["muf2_over_ref"],
}

# the jet definition of NLOJets by its names in the run card
JET_NAMES = ("jetalgo", "jetradius", "ptj", "etaj", "maxjetflavor")


def nlo_scale(particle_count, incoming=2, **scales):
    """The NLOScale of kinematics of particle_count particles, the scales of
    setscales.f of MadFKS (FKSCrossSection takes it for the real processes
    and the Born alike).

    Args:
        particle_count: the number of particles, incoming included.
        incoming: the number of incoming particles, one for a decay.
        scales: the scales by their names in the run card of MadFKS
            (dynamical_scale_choice, fixed_ren_scale, fixed_fac_scale,
            mur_ref_fixed, muf_ref_fixed, mur_over_ref, muf_over_ref, the
            last two of each also per beam as muf1_ and muf2_), the defaults
            of the run card by default.

    Raises:
        TypeError: for a scale it does not know.
    """
    options = {}
    for name, value in scales.items():
        if name not in SCALE_NAMES:
            raise TypeError(f"nlo_scale: unknown scale parameter {name}")
        for option in SCALE_NAMES[name]:
            options[option] = value
    return NLOScale(particle_count, **options, incoming=incoming)


def histogram_jet_count(observable):
    """The jets an Observable on the momenta with jets of NLOJets needs, from
    its keyword arguments: the largest position of a jet in an ordered
    selection (2 for the second jet), none for selections that sum over the
    jets.

    Raises:
        ValueError: for a selection of jets that is neither ordered from the
            hardest nor summed, which has no value per event, or counted
            from the softest, which takes the empty slots.
    """
    jet_pid = Observable.jet_pids
    count = 0
    summed = observable.get("sum_momenta", False) or observable.get(
        "sum_observable", False
    )
    order = list(observable.get("order_indices") or [])
    for k, selection in enumerate(observable.get("select_pids") or []):
        if 21 not in selection and not set(selection) & set(jet_pid):
            continue
        position = order[k] if k < len(order) else 0
        if position < 0:
            raise ValueError(
                f"observable {observable.get('name', '')}: jets counted from the "
                "softest take the empty jet slots"
            )
        if position == 0 and not summed:
            raise ValueError(
                f"observable {observable.get('name', '')}: jets without an order "
                "have no value per event"
            )
        count = max(count, position)
    return count


def jet_observable(jets, histogram):
    """The Observable of a histogram on the momenta with jets of an NLOJets,
    with the jets as gluons (21), so that it takes the jets where it selects
    gluons or the jet particles.

    Args:
        jets: the NLOJets, for its pids and its incoming particles.
        histogram: a dict with name and observable, the keyword arguments of
            Observable without pids (those of parse_observable of the MG7
            launcher).
    """
    options = dict(histogram["observable"])
    options.setdefault("name", histogram.get("name", ""))
    pids = list(jets.pids())
    incoming = jets.incoming()
    if incoming != 2:
        # madspace takes the first two momenta for the beams, a decay
        # selects over all of them, its incoming particle in no group
        pids = [0] * incoming + pids[incoming:]
        options["ignore_incoming"] = False
    return Observable(pids, **options)


def histogram_results(histograms, sums, point_count):
    """The weighted histograms of info.json (its "histograms", which MadBoard
    draws and the HwU writer of the MG7 launcher takes) from the sums of
    NLOHistograms over point_count points, as EventGenerator::histograms
    normalizes those of ObservableHistograms: the cross section in a bin is
    the sum over the points, its error sqrt(sum w^2 - (sum w)^2 / N) / N,
    with the underflow and overflow bins first and last. The histograms of
    several channels add up, their errors in quadrature.

    Args:
        histograms: the histograms (name, min, max, bin_count).
        sums: the outputs of NLOHistograms summed over the batches, values
            and square_values of each histogram in turn.
        point_count: the number of points.
    """
    results = []
    count = float(point_count)
    for k, histogram in enumerate(histograms):
        values = [float(v) for v in sums[2 * k]]
        squares = [float(v) for v in sums[2 * k + 1]]
        errors = [
            math.sqrt(max(square - value * value / count, 0.0)) / count
            for value, square in zip(values, squares, strict=True)
        ]
        results.append(
            {
                "name": histogram.get("name", ""),
                "min": histogram["min"],
                "max": histogram["max"],
                "bin_count": histogram["bin_count"],
                "bin_values": [value / count for value in values],
                "bin_errors": errors,
            }
        )
    return results
