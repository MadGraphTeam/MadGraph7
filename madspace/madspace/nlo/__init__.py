"""The cuts, scales, jets and histograms of the NLO integrands, independent of
the subtraction scheme.

The C++ classes live in the madspace module: NLOCuts (the generation cuts of
cuts.f of MadFKS), NLOScale (the scales of setscales.f), NLOJets (the QCD
partons of a kinematics clustered into jets), NLOHistograms (the
histograms with the event and the counter-events of each point together)
and NLOLuminosity (the parton luminosity of groups of flavors, those of an
FKS configuration or of a Catani-Seymour subprocess). The first four take
the kinematics in one of several flavor layouts, the PDG codes of its
particles, chosen per event by layout_index.

nlo_scale builds the NLOScale from the scales by their names in the run card
of MadFKS, and nlo_luminosity the NLOLuminosity of groups of flavors on the
beams of the run card. jet_observable builds an Observable on the momenta with jets of an
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

from .._madspace_py import NLOLuminosity, NLOScale, Observable

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

# the leptons of beams of dressed electrons, positrons, muons and antimuons by
# lpp of the run card
LEPTON_BEAMS = {3: 11, -3: -11, 4: 13, -4: -13}


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


def beam_flavors(flavors, beams, nincoming=2):
    """The PDG codes of the two incoming partons of each flavor, those of a
    beam of antiprotons (-1) charge conjugated, the decaying particle and 0
    for a decay."""
    if nincoming == 1:
        return [[flavor[0], 0] for flavor in flavors]
    result = []
    for flavor in flavors:
        pair = []
        for pdg, beam in zip(flavor[:2], beams):
            conjugate = beam == -1 and abs(pdg) not in (21, 22)
            pair.append(-pdg if conjugate else pdg)
        result.append(pair)
    return result


def nlo_luminosity(
    groups,
    beams=(1, 1),
    pdf_grid=None,
    pdf_grid2=None,
    nincoming=2,
    factors=None,
    sum_flavors=False,
    all_groups=False,
    lepton_luminosity=None,
    pdf_prefix="",
):
    """The NLOLuminosity of groups of flavors on the beams of a run card, as
    the parton_lum_*.f files of MadFKS: f_1(x_1) f_2(x_2) of each flavor of a
    group times its factor, of the flavor of the event or summed over the
    group, at the group of the event or, with all_groups, at every group from
    one evaluation of the densities.

    Args:
        groups: the flavors of each group, lists with the PDG codes of the
            incoming partons first (the real flavors of an FKS configuration,
            the flavors of a Catani-Seymour subprocess, or those of the
            partons that split into them at the points of a convolution).
        beams: lpp1 and lpp2 of the run card, 1 for protons, -1 for
            antiprotons, 0 for leptons without densities, 2 and -2 for
            elastic photons from protons and antiprotons and 3, -3, 4, -4 for
            dressed electrons, positrons, muons and antimuons (both of one
            kind, as MadFKS has no DIS at NLO), ignored for a decay.
        pdf_grid: the PdfGrid of the hadron beams, None for none.
        pdf_grid2: the PdfGrid of the second beam where it differs.
        nincoming: the number of incoming particles, one for a decay.
        factors: the factor of each flavor of each group, one by default.
        sum_flavors: whether the luminosities sum over the flavors, without a
            flavor index.
        all_groups: whether the luminosity is that of every group (batch,
            groups), without a group index, which needs sum_flavors.
        lepton_luminosity: for dressed leptons, the EELuminosity of their
            densities with the scales of the beams (beam_scales), full or
            reduced as the momentum fractions are sampled.
        pdf_prefix: the prefix of the globals of pdf_grid, for a member of a
            set of PDFs (see PdfGrid.initialize_globals).

    Raises:
        ValueError: for other beams or a hadron with a lepton, for hadrons
            without a grid or a grid without hadrons, for dressed leptons
            without their luminosity or a luminosity without them, and as
            NLOLuminosity.
    """
    beams = (0, 0) if nincoming == 1 else tuple(int(beam) for beam in beams)
    if any(beam not in (-2, -1, 0, 1, 2, *LEPTON_BEAMS) for beam in beams):
        raise ValueError(
            f"nlo_luminosity: beams {beams} are not protons, antiprotons, "
            "photons or leptons"
        )
    kinds = [
        {1: "hadron", 2: "photon", 0: "lepton"}.get(abs(beam), "dressed")
        for beam in beams
    ]
    if kinds[0] != kinds[1]:
        raise ValueError(
            f"nlo_luminosity: beams {beams} are of two kinds, MadFKS has no DIS "
            "at NLO"
        )
    densities = [kind == "hadron" for kind in kinds]
    if any(densities) != (pdf_grid is not None):
        raise ValueError(
            "nlo_luminosity: hadron beams need a PdfGrid and lepton beams none"
        )
    dressed = kinds[0] == "dressed"
    if dressed != (lepton_luminosity is not None):
        raise ValueError(
            "nlo_luminosity: dressed leptons need their EELuminosity and other "
            "beams none"
        )
    options = dict(
        pdf_grid=pdf_grid,
        pdf_grid2=pdf_grid2,
        densities=densities,
        sum_flavors=sum_flavors,
        pdf_prefix=pdf_prefix,
        photon_beams=[kind == "photon" for kind in kinds],
        all_groups=all_groups,
    )
    if factors is not None:
        options.update(factors=factors)
    if dressed:
        options.update(
            lepton_luminosity=lepton_luminosity,
            lepton_pdgs=[LEPTON_BEAMS[beam] for beam in beams],
        )
    flavors = [beam_flavors(group, beams, nincoming) for group in groups]
    return NLOLuminosity(flavors, **options)


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
