"""The fixed-order NLO run of madspace on an aMC@NLO output with the
Catani-Seymour subtraction, on the core of nlo/core.py, with the matrix
elements of the Fortran provider (nlo/provider.py) as UMAMI libraries
(nlo/libraries.py) and the PDFs, alpha_s, scales and cuts of the run card.

    python -m madgraph.mg7.nlo.cs OUTPUT [--name RUN] [--accuracy ACC]

The n-body stream of a subprocess (a P* directory) has the Born, the
virtual (MadLoop, with the virtual trick of the FKS driver), the insertion
operator I (CSInsertion) and the K and P operators (CSCollinear) of each of
its Born channels, with the multichannel weight of MadFKS (amp2 of the Born
diagram of the channel over that of all Born diagrams). The K and P
operators take one more random number, the point z = x_i + (1 - x_i) r of
the convolution of both beams, with the densities of the partons that
split into those of the Born at x_i / z. The cuts act on the Born, with no
unresolved parton, as those of the FKS driver on its soft counter-event,
and the scales are those of the Born. With hadron beams the Born mappings of
both streams sample s_hat flat below the bound on sqrt(s_hat) of the cuts on
the Born and falling above it, with the shift of the FKS driver (s_hat_shift
of PhaseSpaceMapping).

The I operator is that of CDR with the normalization (4 pi)^eps / Gamma(1 -
eps) of the virtual of MadLoop at the renormalization scale, with the n_f
light flavors of the model (coupl.inc). The corrections are those of QCD
and of QED that the configurations of the output have (their splitting
types), each to the Born orders whose correction (two more powers of g_s or
of e and the same others) is an NLO order of the output, which then sums
them. The operators of QED (CSInsertion and CSCollinear with qed) take the
Born and the charges in place of the colors, with the photon splitting into
the n_f light quarks and the n_l massless leptons, the QED coupling of the
model (that of the counterterms of MadFKS), and K and P the photon of the
PDFs and the quarks weighted by their charges. The virtual is the sum of the
NLO orders, each with its powers of g_s.

The real stream of a real process (a real matrix element of the cs block of
madspace.nlo.cs, with its flavor lines) is the real emission minus
its dipoles (CSDipoles), with one channel per dipole and Born channel of its
Born: the CSMapping of the dipole on the Born diagram of the channel. The
channels of the real stream (channels) are, by default (pairs), one per
diagram of the real process (a PhaseSpaceMapping of its topology from
real_channels_structure.json of the output) and one per singular pair
{i, j} of its dipoles (the CSMapping of the first dipole of the pair on the
first Born diagram of its Born, since the spectator fixes only the recoil
and the Born peaks are those of the diagrams); reals takes the diagrams
alone, as at LO, and combined all dipoles on all Born diagrams as well (as
MoCaNLO). The weight of every channel is then the density of its mapping
over the sum of those of all channels at the event (as
MappingChannelWeights), which is the C++ integrand of a channel
(CSRealChannelIntegrand). Real-stream errors at the same points per channel
(pairs / reals / combined): p p > t t~ 0.29 / 0.32 / 0.20 with 66 / 46 /
154 channels, p p > e+ e- j 2.9 / 4.8 / 2.5 with 672 / 544 / 1568. Each
dipole takes the blocks of the Born of its own subprocess, the spin
correlation in the factorized
form <T_j . T_k> B(k_t) / B, with B(k_t) the Born of the provider with the
azimuthal part of an FKS configuration whose mother is the emitter. MadFKS
keeps one FKS configuration of identical partons, so a gluon without one
takes that of an identical gluon on the same side with the momenta of the
two exchanged, at which the Born is the same. A dipole of QED takes the
charges of its emitter and spectator times the Born (a photon that splits
the weight of its spectator), the factor of its kernel over that of QCD
(qed_kernel_factor), e^2 in place of g_s^2 and only the Born orders whose
QED correction the output has, as one of QCD only those with a QCD
correction. The real emission has the cuts and scales of the real process,
each dipole those of its Born, and both the densities at the momentum
fractions of the real emission.

The technical cut is the cutoff of the CSMapping of every channel: a channel
covers only the events where both variables of its dipole (y and z_i, 1 - x
and z, v and 1 - x - v) are at or above technical_cut (one of a diagram
those where the variables of all dipoles are), and its events are
never cut afterwards, as the FKS configurations of MadFKS cover xi and 1 - y
from 1e-6 up (genps_fks.f). The real emission minus the dipoles then has the
cut sum_c w_c Theta_c of the channel weights w_c and of the regions Theta_c
of the channels, zero where i is soft below the cut and of the order of the
cut where it is collinear, since there the weights of the other channels
vanish with the invariant. It keeps the real emission of the Fortran in
double precision away from the soft limit, towards which it loses digits as
1 / xi^2 (by 1e-4 to 1e-3 at an energy fraction xi = 5e-7 of the soft gluon
of g g > t t~ g, depending on the angle).

A decay has the CSMapping of a final-state emitter with the decaying
particle as the spectator (fd, Basso, Dittmaier, Huss, Oggero,
arXiv:1507.04676): the final-final map with the other final-state particles
as the spectator, the decaying particle at rest, for quark and gluon
emitters. Its dipoles take the soft terms of the decaying particle and
alpha_ff, and CSInsertion their integral.
A decay has the flux 1 / (2 M) and gives a width in GeV, as the FKS driver.

Beams of dressed electrons and positrons (lpp 3 and -3) take the densities
of pdlabel among the lep_densities of the output, reduced by (1 - x)^p of
the power p of the sampling of their momentum fractions in PhaseSpaceMapping
(lepton_sampling_power), with the real emission and its dipoles at the
momentum fractions of the Born, which the inverse maps of the densities of
the channels take with 1 - x. Relative errors of their real stream at the
same points: e+ e- > j j with ISR 2.1% (9.3% with the dipole channels of the
first driver), e+ e- > t t~ 0.37% (0.40%). Beams of elastic photons from
protons (lpp
2 and -2) take the equivalent-photon density of epa_proton of MadFKS
(ElasticPhotonDensity). Neither the leptons nor the photons radiate QCD, so
that their Borns have no K and P operators. The luminosities of all beams
are those of NLOLuminosity (nlo_luminosity of madspace.nlo), which the FKS
integrand takes too: the Born and the real over their flavors, and those of
K and P per beam at x_i / z with every group of partons at once.

The cuts are those of cuts.f with the photon isolation and the
recombination of photons (NLOCuts), all photons to be isolated.

The scale of the user (dynamical_scale_choice 0 or 10, user_dynamical_scale
of dummy_fct.f) comes from the provider, at the Born with a last particle
without momentum, as at the soft counter-event of MadFKS.

The channels of both streams run in the C++ event generator of madspace
(CSIntegrand, with the virtual trick of the n-body stream in
NLOIterationState, see nlo.core), also with a MadNIS training
(madnis_batches), with the histograms of CSHistograms (CSObservableHistograms
of madspace).

Not covered yet: PDF schemes other than MSbar, densities of eMELA, photons of gamma-UPC, the
scale and PDF variations of reweight_scale and reweight_PDF and the Sudakov
approximation of the EW corrections of --ewsudakov, which the run leaves out
with a warning. The run refuses the settings of MadFKS it would otherwise
ignore (check_run_card of nlo.core).
"""

import argparse
import json
import logging
import math
import os

import numpy as np

from madgraph.mg7.nlo.provider import FortranProvider
from madgraph.mg7.nlo.core import (
    LEPTON_POWER,
    NLORun,
    Runtime,
    born_cuts,
    born_topology,
    integer,
    madspace,
    pdf_reweight,
    real,
    real_topology,
    run_cuts,
    scale_parameters,
    scale_variations,
)

logger = logging.getLogger("madgraph.mg7.nlo.cs")

FAMILIES = ("ff", "fi", "if", "ii")
STREAMS = ("nbody", "real")
CHANNELS = ("pairs", "reals", "combined")


def spin_configuration(provider, leg, mother=21):
    """The first FKS configuration of a provider whose mother is a gluon (or
    the boson mother) at the Born leg leg (from 0), None without one."""
    for c, configuration in enumerate(provider.configurations):
        i_fks, j_fks = configuration["i_fks"], configuration["j_fks"]
        if configuration["ij_pdg"] == mother and j_fks - 1 - (j_fks > i_fks) == leg:
            return c
    return None


def identical_spin_configuration(provider, pdgs, incoming, leg, mother=21):
    """For a gluon (or the boson mother) at the Born leg leg without an FKS
    configuration, the first other one on the same side (incoming or
    outgoing) with one and that configuration, as (leg, configuration), None
    without one. MadFKS keeps one configuration of identical partons, and the
    Born with the two exchanged is the same."""
    side = leg < incoming
    for other, pdg in enumerate(pdgs):
        if other == leg or pdg != mother or (other < incoming) != side:
            continue
        configuration = spin_configuration(provider, other, mother)
        if configuration is not None:
            return other, configuration
    return None


# the factors of the kernels of CSDipoles for the QED splittings, in which a
# photon takes the place of the gluon. With the Borns averaged over their own
# colors, as the provider gives them, the kernels of QCD have T_R for g -> q
# q~ and for an incoming g -> q~ q, and C_F for an incoming q -> g q, and
# those of QED N_c Q_f^2 for gamma -> f f~ and an incoming gamma -> f~ f (the
# colors of the pair, and of the antifermion of the Born over the colorless
# photon of the real) and Q_f^2 for an incoming f -> gamma f. The weights of
# CSDipoles divide by the Casimir of the Born emitter, C_A for a gluon and
# C_F for a quark, which a photon has not (its spectators share it) and a
# fermion has as Q_f^2.
CA, CF, TR = 3.0, 4.0 / 3.0, 0.5


def qed_kernel_factor(kernel, incoming, flavor):
    """The factor of the weights of CSDipoles for a QED splitting (kernel of
    madspace.nlo.cs) of an incoming or final emitter, with flavor the
    emitted fermion."""
    from madspace.nlo.cs import COLORS, charge

    colors = COLORS.get(abs(flavor), 1)
    q2 = charge(flavor) ** 2
    if kernel == "qg":
        return 1.0
    if kernel == "gq" and not incoming:
        # gamma -> f f~: N_c Q_f^2 over T_R / C_A
        return colors * q2 * CA / TR
    if kernel == "gq":
        # f -> gamma f: Q_f^2 over C_F / C_A
        return q2 * CA / CF
    # gamma -> f~ f: N_c Q_f^2 / Q_f^2 over T_R / C_F
    return colors * CF / TR


def qcd_powers(orders, kind):
    """The powers of g_s of the Born or NLO orders (kind) of
    fks_configurations.json, zero without a QCD split order."""
    split = orders["split_orders"]
    if "QCD" not in split:
        return [0] * len(orders[kind])
    q = split.index("QCD")
    return [int(o[q]) for o in orders[kind]]


def correction_orders(orders, groups=("QCD",)):
    """Per group of corrections (QCD, QED) and Born order of
    fks_configurations.json the index of the NLO order of its correction, two
    more powers of g_s or of e and the same others, None where the NLO orders
    have none or the output not the group.

    Raises:
        NotImplementedError: for NLO orders that are no correction of a Born
            order.
    """
    split = orders["split_orders"]
    nlo = [list(o) for o in orders["nlo"]]
    out, covered = {}, set()
    for group in ("QCD", "QED"):
        out[group] = []
        for born in orders["born"]:
            target = list(born)
            if group not in groups or group not in split:
                out[group].append(None)
                continue
            target[split.index(group)] += 2
            index = nlo.index(target) if target in nlo else None
            out[group].append(index)
            if index is not None:
                covered.add(index)
    if covered != set(range(len(nlo))):
        raise NotImplementedError(
            "nlo.cs: NLO orders other than the corrections of the Born orders"
        )
    return out


class Luminosity:
    """The luminosity of groups of flavors (lists with the PDG codes of the
    incoming partons first) on the beams of a subprocess, each summed over
    its flavors times their factors (NLOLuminosity with all_groups): the
    densities of hadrons, the reduced densities of dressed leptons (at 1 - x
    of omx), the densities of elastic photons, or one per flavor for leptons
    without densities and for a decay."""

    def __init__(self, groups, beams, incoming, couplings, context, factors=None):
        from madspace.nlo import nlo_luminosity

        leptons = couplings.leptons if incoming == 2 else None
        self.dressed = leptons is not None
        function = nlo_luminosity(
            groups,
            beams=beams,
            pdf_grid=couplings.pdf if abs(beams[0]) == 1 else None,
            nincoming=incoming,
            factors=factors,
            sum_flavors=True,
            all_groups=True,
            lepton_luminosity=None if leptons is None else leptons.reduced,
        )
        self.runtime = Runtime(function, context)


def convolution_groups(flavors, leg, groups, n_f):
    """The groups of flavors of the K and P operators of the beam leg at x_i
    / z, with their factors: the flavors of the Born, then per group of
    corrections those with the partons of the beam that split into the
    parton of the Born (QCD: the gluon for a quark, the light quarks for a
    gluon, QED: the photon for a quark, the light quarks with Q_q^2 for a
    photon), the parton of the other beam that of the Born."""
    from madspace.nlo.cs import charge

    quarks = [q for k in range(1, int(n_f) + 1) for q in (k, -k)]
    out, factors = [[list(f[:2]) for f in flavors]], [[1.0] * len(flavors)]
    for group in groups:
        mother = 21 if group == "QCD" else 22
        weights = {q: 1.0 if group == "QCD" else charge(q) ** 2 for q in quarks}
        pairs, values = [], []
        for flavor in flavors:
            parton = flavor[leg]
            partons = quarks if parton == mother else [mother]
            for split in partons:
                pair = list(flavor[:2])
                pair[leg] = split
                pairs.append(pair)
                values.append(weights[split] if parton == mother else 1.0)
        out.append(pairs)
        factors.append(values)
    return out, factors


class CSGeneratorIntegrand:
    """The integrand of a channel of the CS integrands for the C++ event
    generator (CSIntegrand of madspace), with channel_integrand and
    channel_dimension of the integrand."""

    def generator_histograms(self, histograms, jets):
        """The CSObservableHistograms of the histograms with the jets of the
        run card, those of histogram_runtime."""
        return self.histogram_runtime(histograms, jets).generator()

    def generator_integrand(
        self,
        channel,
        context,
        prefix,
        fraction,
        min_fraction,
        bins,
        histograms,
        jets,
        flow=None,
        discrete=None,
        madnis_training=False,
    ):
        """The CSIntegrand of a channel for the C++ event generator (see
        nlo.core): channel_integrand with a VEGAS grid of bins bins and, for
        the n-body stream, the virtual trick from fraction down to
        min_fraction, the histograms with the jets of the run card
        (generator_histograms), its globals with the prefix defined in the
        context. With a flow, the integrand takes it instead of the VEGAS grid
        and keeps the globals of the integrand before it, for the MadNIS
        training with madnis_training. The CS integrands have no discrete
        choice."""
        ms = madspace()
        if discrete is not None:
            raise ValueError("nlo.cs: the CS integrands have no discrete choice")
        adaptive_map = flow
        if flow is None:
            adaptive_map = ms.VegasMapping(
                self.channel_dimension(channel), bins, prefix
            )
        observables = None
        if histograms and not madnis_training:
            observables = self.generator_histograms(histograms, jets)
        generator = ms.CSIntegrand(
            self.channel_integrand(channel, context),
            adaptive_map,
            prefix,
            virtual_fraction=fraction,
            min_virtual_fraction=min_fraction,
            histograms=observables,
            madnis_training=madnis_training,
        )
        if flow is None:
            generator.initialize_globals(context)
        return generator


class CSBornIntegrand(CSGeneratorIntegrand):
    """The n-body stream of the Catani-Seymour integrand of one subprocess
    (a P* directory) of an aMC@NLO output, per Born channel of MadFKS (an
    integrand of nlo.core, without a discrete choice)."""

    choice_key = "choice_probabilities"
    # the pole check of the setup of the run (pole_check of nlo.core)
    takes_virtual = True

    def __init__(
        self,
        directory,
        run_card,
        param_card,
        couplings,
        context,
        n_f,
        alpha,
        kappa,
        n_l=0.0,
    ):
        ms = madspace()
        from madspace.nlo import nlo_scale

        self.name = os.path.basename(os.path.normpath(directory))
        self.directory = directory
        self.context = context
        self.provider = FortranProvider(directory)
        self.configurations = self.provider.fks_configurations
        if "ewsudakov" in self.configurations:
            logger.warning(
                "nlo.cs: no Sudakov approximation of the EW corrections yet, the "
                "run leaves out those of --ewsudakov"
            )
        self.incoming = self.configurations["nincoming"]
        self.born_pdgs = list(self.configurations["born_pdgs"])
        self.masses = [
            float(param_card["mass"].get(str(abs(pdg)), 0.0)) for pdg in self.born_pdgs
        ]
        # dressed leptons sample their momentum fractions in the mappings
        self.leptons = couplings.leptons if self.incoming == 2 else None
        if self.incoming == 1:
            # a decay at rest, without beams
            self.energy, leptonic = self.masses[0], True
        else:
            self.energy = 2 * math.sqrt(
                real(run_card["ebeam1"]) * real(run_card["ebeam2"])
            )
            leptonic = integer(run_card["lpp1"]) == 0
        self.beams = (
            (0, 0)
            if leptonic
            else (integer(run_card["lpp1"]), integer(run_card["lpp2"]))
        )
        self.lepton_power = 0.0 if self.leptons is None else LEPTON_POWER
        orders = self.configurations["orders"]
        groups = {
            t
            for c in self.configurations["configurations"]
            for t in c.get("splitting_type", ["QCD"])
        }
        self.corrections = correction_orders(orders, groups)
        self.qcd_powers = qcd_powers(orders, "born")
        self.nlo_powers = qcd_powers(orders, "nlo")
        # the QED coupling of the counterterms and the factors of tagged
        # photons
        _, self.alpha, _ = self.provider.couplings()
        self.tagged = self.provider.tagged_factors()
        self.nlo_count = len(orders["nlo"])
        self.channels = self.provider.channels()
        # s_hat flat below the bound of the Born cuts and falling above, also
        # for the Born of the dipoles of the real emission
        self.s_hat_shift = None
        if not leptonic and self.leptons is None:
            cuts = born_cuts(self.born_pdgs, run_card)
            floor = 0.0
            if cuts is not None:
                floor = ms.fks_sqrt_s_hat_min(cuts, self.masses[self.incoming :])
            self.s_hat_shift = floor**2 if floor > 0 else None
        self.channel_mappings = [
            ms.PhaseSpaceMapping(
                born_topology(self.masses, self.incoming, channel),
                self.energy,
                leptonic=leptonic,
                lepton_sampling_power=self.lepton_power,
                s_hat_shift=self.s_hat_shift,
            )
            for channel in self.channels
        ]
        self.born_dim = self.channel_mappings[0].random_dim()
        n = len(self.born_pdgs)
        scale = nlo_scale(n, self.incoming, **scale_parameters(run_card))
        self.user_scale = scale.user()
        self.scale = Runtime(scale, context)
        # the cuts of the Born, with no unresolved parton
        self.cuts = Runtime(
            ms.NLOCuts(
                [self.born_pdgs],
                incoming=self.incoming,
                unresolved=0,
                **run_cuts(run_card),
            ),
            context,
        )
        b = len(orders["born"])
        options = dict(born_orders=b, n_f=n_f)
        # the operators of the QCD and of the QED corrections, of the groups
        # with corrections of a Born order
        self.groups = [
            group
            for group in ("QCD", "QED")
            if any(n is not None for n in self.corrections[group])
        ]
        qed = dict(qed=True, n_l=n_l)
        self.insertion = self.insertion_qed = None
        if "QCD" in self.groups:
            self.insertion = Runtime(
                ms.CSInsertion(
                    self.born_pdgs,
                    self.masses,
                    kappa=kappa,
                    alpha_ff=alpha["ff"],
                    incoming=self.incoming,
                    **options,
                ),
                context,
            )
        if "QED" in self.groups:
            if self.tagged is not None:
                raise NotImplementedError(
                    "nlo.cs: no QED corrections with tagged photons"
                )
            self.insertion_qed = Runtime(
                ms.CSInsertion(
                    self.born_pdgs,
                    self.masses,
                    kappa=kappa,
                    alpha_ff=alpha["ff"],
                    incoming=self.incoming,
                    **options,
                    **qed,
                ),
                context,
            )
        self.flavors = [list(f) for f in self.configurations["born_flavors"]]
        self.couplings = couplings
        self.luminosity = Luminosity(
            [self.flavors], self.beams, self.incoming, couplings, context
        )
        # the K and P operators of hadrons, per beam at x_i / z (leptons and
        # photons radiate no QCD, so that their Borns have none)
        self.hadrons = self.incoming == 2 and abs(self.beams[0]) == 1
        self.convolution = None
        if self.hadrons:
            alphas = dict(
                alpha_fi=alpha["fi"], alpha_if=alpha["if"], alpha_ii=alpha["ii"]
            )
            self.collinear = self.collinear_qed = None
            if "QCD" in self.groups:
                self.collinear = Runtime(
                    ms.CSCollinear(self.born_pdgs, self.masses, **alphas, **options),
                    context,
                )
            if "QED" in self.groups:
                self.collinear_qed = Runtime(
                    ms.CSCollinear(
                        self.born_pdgs, self.masses, **alphas, **options, **qed
                    ),
                    context,
                )
            self.convolution = []
            for leg in range(2):
                groups, factors = convolution_groups(
                    self.flavors, leg, self.groups, n_f
                )
                self.convolution.append(
                    Luminosity(
                        groups, self.beams, 2, couplings, context, factors=factors
                    )
                )

    @property
    def channel_count(self):
        return len(self.channel_mappings)

    def channel_dimension(self, index):
        # the Born and, for hadrons, the point of the convolution of K, P
        return self.born_dim + self.hadrons

    def channel_name(self, index):
        return f"{self.name} n-body channel {index + 1}"

    def choice_count(self, index):
        return 1

    def libraries(self, context):
        """The MatrixElementApi of the matrix elements of the provider as
        UMAMI libraries in the context (ProviderLibraries of
        nlo/libraries.py, added once)."""
        from madgraph.mg7.nlo.libraries import ProviderLibraries

        if getattr(self, "_libraries", None) is None:
            self._libraries = ProviderLibraries(self.provider, self.name)
            self._apis = self._libraries.add_to(context)
        return self._apis

    def born_library(self, context):
        """The Born of the provider as a MatrixElement with the spin legs and
        vectors, for the dipoles of the real stream."""
        ms = madspace()
        element = ms.MatrixElement
        apis = self.libraries(context)
        outputs = [element.squared_orders_out, element.diagram_amp2_out]
        if self._libraries.has_color:
            outputs.append(element.color_correlated_out)
        outputs.append(element.spin_correlated_out)
        inputs = [
            element.momenta_in,
            element.alpha_s_in,
            element.g_ew_in,
            element.spin_legs_in,
            element.spin_vectors_in,
        ]
        return element(apis["born"], inputs, outputs)

    def channel_integrand(self, channel, context):
        """The C++ integrand of a channel (CSBornChannelIntegrand of
        madspace) with the matrix elements of the provider as UMAMI libraries
        in the context (ProviderLibraries of nlo/libraries.py, added once),
        the same integrand as evaluate at the same random numbers and points
        with the virtual. The libraries evaluate at the e of the model, so
        that the blocks are those of the provider."""
        ms = madspace()
        element = ms.MatrixElement
        apis = self.libraries(context)
        outputs = [element.squared_orders_out, element.diagram_amp2_out]
        if self._libraries.has_color:
            outputs.append(element.color_correlated_out)
        born = element(
            apis["born_plain"],
            [element.momenta_in, element.alpha_s_in, element.g_ew_in],
            outputs,
        )
        loop = element(
            apis["loop"],
            [
                element.momenta_in,
                element.alpha_s_in,
                element.g_ew_in,
                element.scale_in,
            ],
            [element.loop_out],
        )
        reference_scale = None
        if self.user_scale:
            reference_scale = ms.CustomFunction(
                apis["born_scale"], ms.CustomFunction.scale, len(self.born_pdgs)
            )

        def corrected(group):
            orders = self.corrections.get(group) or [None] * len(self.qcd_powers)
            return [group in self.groups and n is not None for n in orders]

        def function(runtime):
            return None if runtime is None else runtime.function

        hadrons = self.hadrons
        return ms.CSBornChannelIntegrand(
            self.channel_mappings[channel],
            self.cuts.function,
            self.scale.function,
            self.couplings.running,
            self.luminosity.runtime.function,
            born,
            loop,
            apis["born_plain"].color_representations(),
            channel,
            self._libraries.e,
            self.alpha,
            [float(p) for p in self.qcd_powers],
            [float(p) for p in self.nlo_powers],
            function(self.insertion),
            function(self.insertion_qed),
            corrected("QCD"),
            corrected("QED"),
            function(self.collinear) if hadrons else None,
            function(self.collinear_qed) if hadrons else None,
            [lum.runtime.function for lum in self.convolution or []],
            self.energy,
            reference_scale=reference_scale,
        )

    def histogram_runtime(self, histograms, jets):
        """The histograms with the jets of the run card, filled with the Born
        alone."""
        runtime = CSHistograms(self.context, histograms, jets, 1)
        runtime.born = runtime.jets(self, self.born_pdgs)
        runtime.layouts = [runtime.born]
        return runtime

    def matrix_element_calls(self):
        return self.provider.statistics

    def close(self):
        self.provider.close()


class CSDipole:
    """One dipole of a real process (an entry of the cs block of
    madspace.nlo.cs) with the CSBornIntegrand of the subprocess of its
    Born: the Born legs (from 0, in the order of the matrix element) of its
    emitter and its spectator and, for a gluon emitter (a photon in QED), the
    FKS configuration of the spin correlation, with swap the leg of the
    identical boson whose configuration it takes (see
    identical_spin_configuration), None for its own. A QED dipole (group QED)
    takes the charges of its emitter and spectator, or the weight of its
    spectator, in place of the color-correlated Born, the factor of its
    kernel (qed_kernel_factor) and the Born orders with a QED correction
    (orders), real_pdgs the flavors of its real process."""

    def __init__(self, entry, born, real_pdgs=None):
        from madspace.nlo.cs import charge

        self.entry = entry
        self.born = born
        self.order = list(entry["born_order"])
        i = entry["i"]
        emitter = entry["j"] - (entry["j"] > i)
        spectator = entry["k"] - (entry["k"] > i)
        self.legs = self.order.index(emitter), self.order.index(spectator)
        self.group = entry.get("group", "QCD")
        self.weight = entry.get("weight")
        # the Born orders whose correction of the group the NLO orders have
        self.orders = [n is not None for n in born.corrections.get(self.group, [])] or [
            True
        ] * len(born.qcd_powers)
        self.factor, self.charges = 1.0, None
        mother = 21
        if self.group == "QED":
            mother = 22
            if born.tagged is not None:
                raise NotImplementedError("nlo.cs: no QED dipoles with tagged photons")
            # the charges flowing out of the Born
            outgoing = [
                (-1 if n < born.incoming else 1) * charge(pdg)
                for n, pdg in enumerate(born.born_pdgs)
            ]
            self.charges = outgoing[self.legs[0]] * outgoing[self.legs[1]]
            flavor = real_pdgs[i] if real_pdgs is not None else 0
            self.factor = qed_kernel_factor(
                entry["kernel"], entry["j"] < born.incoming, flavor
            )
        self.gluon = born.born_pdgs[self.legs[0]] == mother
        self.configuration, self.swap = 0, None
        if self.gluon:
            self.configuration = spin_configuration(born.provider, self.legs[0], mother)
            if self.configuration is None:
                identical = identical_spin_configuration(
                    born.provider, born.born_pdgs, born.incoming, self.legs[0], mother
                )
                if identical is None:
                    raise NotImplementedError(
                        f"nlo.cs: no FKS configuration of {born.name} with the "
                        f"boson {self.legs[0] + 1} or an identical one as its "
                        "mother"
                    )
                self.swap, self.configuration = identical


class CSRealIntegrand(CSGeneratorIntegrand):
    """The real stream of the Catani-Seymour integrand of one real process of
    an aMC@NLO output, the real emission minus its dipoles, per dipole and
    Born channel of its Born (an integrand of nlo.core, without a discrete
    choice). subprocesses are the CSBornIntegrand of the P* directories by
    name, whose providers, scales and cuts the dipoles take."""

    choice_key = "choice_probabilities"

    def __init__(
        self,
        index,
        block,
        subprocesses,
        run_card,
        param_card,
        context,
        alpha,
        kappa,
        technical_cut,
        channels="pairs",
    ):
        ms = madspace()
        from madspace.nlo import nlo_scale

        if channels not in CHANNELS:
            raise ValueError(f"nlo.cs: channels of {CHANNELS} expected, not {channels}")
        entry = block["reals"][index]
        source = subprocesses[entry["subprocess"]]
        self.source = source
        self.context = context
        self.configuration = entry["configuration"]
        self.name = f"{source.name} real {self.configuration + 1}"
        configurations = source.configurations
        real_pdgs = configurations["configurations"][self.configuration]["real_pdgs"]
        # the flavor layout of the real process of the configuration, for the
        # cuts and the jets
        self.real_pdgs = list(real_pdgs)
        self.pdgs = list(entry["pdgs"])
        if len(self.pdgs) != len(real_pdgs):
            raise ValueError(f"nlo.cs: the real {self.pdgs} is not {real_pdgs}")
        self.flavors = [list(f) for f in entry["flavors"]]
        self.incoming = source.incoming
        self.born_pdgs = source.born_pdgs
        self.energy = source.energy
        self.beams = source.beams
        leptonic = self.beams == (0, 0)
        orders = configurations["orders"]
        self.nlo_count = len(orders["nlo"])
        self.real_powers = qcd_powers(orders, "nlo")
        self.technical_cut = technical_cut
        self.masses = [
            float(param_card["mass"].get(str(abs(pdg)), 0.0)) for pdg in self.pdgs
        ]
        n = len(self.pdgs)
        self.scale = Runtime(
            nlo_scale(n, self.incoming, **scale_parameters(run_card)), context
        )
        # the cuts of the real emission, with one unresolved parton
        self.cuts = Runtime(
            ms.NLOCuts([self.real_pdgs], incoming=self.incoming, **run_cuts(run_card)),
            context,
        )
        self.luminosity_function = Luminosity(
            [self.flavors], self.beams, self.incoming, source.couplings, context
        )
        self.dipoles, self.channels, self.dimensions = [], [], []
        mappings, kernels, symmetry = [], [], []
        channel_mappings, discrete_mappings = [], []
        # with channels="pairs" one channel per singular pair {i, j} of the
        # dipoles, on the first Born diagram of the first dipole of the pair
        pairs = set()
        for d in block["cs"]:
            if d["real"] != index:
                continue
            born = subprocesses[block["borns"][d["born"]]["subprocess"]]
            dipole = CSDipole(d, born, self.pdgs)
            line_masses = list(np.delete(self.masses, d["i"]))
            for b, channel in enumerate(born.channels):
                topology = born_topology(
                    line_masses, self.incoming, channel, dipole.order
                )
                born_mapping = ms.PhaseSpaceMapping(
                    topology,
                    self.energy,
                    leptonic=leptonic,
                    lepton_sampling_power=source.lepton_power,
                    s_hat_shift=born.s_hat_shift,
                )
                mapping = ms.CSMapping(
                    born_mapping, d["i"], d["j"], d["k"], cutoff=technical_cut
                )
                if b == 0:
                    mappings.append(mapping)
                if channels == "reals":
                    # the channels of the diagrams alone
                    continue
                if channels == "pairs":
                    pair = frozenset((d["i"], d["j"]))
                    if b > 0 or pair in pairs:
                        continue
                    pairs.add(pair)
                channel_mappings.append(mapping)
                discrete_mappings.append(born_mapping)
                self.channels.append((len(self.dipoles), b))
                self.dimensions.append(mapping.random_dim())
            self.dipoles.append(dipole)
            kernels.append(d["kernel"])
            symmetry.append(d["symmetry"])
        # the channels of the diagrams of the real emission
        path = os.path.join(source.directory, "real_channels_structure.json")
        with open(path) as stream:
            structure = json.load(stream)
        real_me = configurations["configurations"][self.configuration]["real_me"]
        for c, diagram in enumerate(structure["reals"][real_me - 1]["diagrams"]):
            mapping = ms.PhaseSpaceMapping(
                real_topology(diagram, self.masses, self.incoming),
                self.energy,
                leptonic=leptonic,
                lepton_sampling_power=source.lepton_power,
                s_hat_shift=source.s_hat_shift,
            )
            channel_mappings.append(mapping)
            discrete_mappings.append(mapping)
            self.channels.append((None, c))
            self.dimensions.append(mapping.random_dim())
        self.channel_mappings = channel_mappings
        # the mappings of the Borns and of the real emission have no discrete
        # choices, which would change their outputs and their densities
        if any(m.discrete_dim() for m in discrete_mappings):
            raise NotImplementedError("nlo.cs: no channels with discrete choices yet")
        self.subtraction = Runtime(
            ms.CSDipoles(
                mappings,
                kernels,
                symmetry,
                kappa=kappa,
                **{f"alpha_{family}": alpha[family] for family in FAMILIES},
            ),
            context,
        )

    @property
    def channel_count(self):
        return len(self.channels)

    def channel_dimension(self, index):
        return self.dimensions[index]

    def channel_name(self, index):
        m, b = self.channels[index]
        if m is None:
            return f"{self.name} diagram {b + 1}"
        return f"{self.name} dipole {m + 1} channel {b + 1}"

    def choice_count(self, index):
        return 1

    def channel_integrand(self, channel, context):
        """The C++ integrand of a channel (CSRealChannelIntegrand of
        madspace) with the matrix elements of the providers as UMAMI libraries
        in the context, the same integrand as evaluate with the combined
        channels at the same random numbers. The libraries evaluate at the e
        of the model, so that the blocks are those of the providers."""
        ms = madspace()
        element = ms.MatrixElement
        source = self.source
        apis = source.libraries(context)
        borns = []
        for dipole in self.dipoles:
            born = dipole.born
            born_apis = born.libraries(context)
            reference = None
            if born.user_scale:
                reference = ms.CustomFunction(
                    born_apis["born_scale"],
                    ms.CustomFunction.scale,
                    len(born.born_pdgs),
                )
            charge = 0.0
            if dipole.group == "QED":
                charge = -dipole.weight if dipole.weight is not None else dipole.charges
            spin_leg = -1
            if dipole.gluon:
                spin_leg = dipole.legs[0] if dipole.swap is None else dipole.swap
            borns.append(
                ms.CSRealDipole(
                    born.born_library(context),
                    born.cuts.function,
                    born.scale.function,
                    dipole.order,
                    dipole.legs[0],
                    dipole.legs[1],
                    born_apis["born"].color_representations(),
                    [float(p) for p in born.qcd_powers],
                    list(dipole.orders),
                    qed=dipole.group == "QED",
                    charge=float(charge),
                    alpha=float(born.alpha),
                    factor=float(dipole.factor),
                    spin_leg=spin_leg,
                    swap=-1 if dipole.swap is None else dipole.swap,
                    reference_scale=reference,
                )
            )
        real_me = source.configurations["configurations"][self.configuration]["real_me"]
        real = element(
            apis["reals"][real_me - 1],
            [element.momenta_in, element.alpha_s_in, element.g_ew_in],
            [element.squared_orders_out],
        )
        reference = None
        if source.user_scale:
            reference = ms.CustomFunction(
                apis["scale"], ms.CustomFunction.scale, len(self.pdgs)
            )
        dipole_channels = [
            mapping
            for mapping, (m, _) in zip(self.channel_mappings, self.channels)
            if m is not None
        ]
        real_channels = [
            mapping
            for mapping, (m, _) in zip(self.channel_mappings, self.channels)
            if m is None
        ]
        return ms.CSRealChannelIntegrand(
            self.subtraction.function,
            borns,
            dipole_channels,
            real_channels,
            channel,
            self.cuts.function,
            self.scale.function,
            source.couplings.running,
            self.luminosity_function.runtime.function,
            real,
            [float(p) for p in self.real_powers],
            source._libraries.e,
            self.energy,
            reference_scale=reference,
        )

    def histogram_runtime(self, histograms, jets):
        return CSRealHistograms(self, histograms, jets)

    def matrix_element_calls(self):
        return {}

    def close(self):
        pass


class CSHistograms:
    """The histograms of a stream of the CS integrand, the kinematics of each
    point in the same histograms: the jets of NLOJets on each kinematics, in
    its flavor layout, and the observables of the histograms on them
    (jet_observable of madspace.nlo). A kinematics of a subprocess has as
    many jet slots as its real processes have partons, so that the
    observables of the real emission also take its Born."""

    def __init__(self, context, histograms, jets, kinematics_count):
        from madspace.nlo import histogram_jet_count

        self.context = context
        self.histograms = histograms
        self.jet_options = jets
        # the observables of the histograms by the PDG codes of the jets
        self.functions = {}
        # the jets of the kinematics, in their order
        self.layouts = []
        self.real = kinematics_count > 1
        self.jet_counts = [
            histogram_jet_count(dict(histogram["observable"]))
            for histogram in histograms
        ]

    def jets(self, subprocess, layout):
        """The PDG codes of the jets and the NLOJets of a flavor layout of the
        subprocess, with the observables of the histograms on its momenta
        with jets."""
        ms = madspace()
        from madspace.nlo import jet_observable

        maxjetflavor = self.jet_options.get("maxjetflavor", 4)
        slots = max(
            ms.NLOJets.parton_count(c["real_pdgs"], subprocess.incoming, maxjetflavor)
            for c in subprocess.configurations["configurations"]
        )
        function = ms.NLOJets(
            [list(layout)], slots, subprocess.incoming, **self.jet_options
        )
        pids = tuple(function.pids())
        if pids not in self.functions:
            self.functions[pids] = [
                jet_observable(function, histogram) for histogram in self.histograms
            ]
        return pids, function

    def generator(self):
        """The same histograms as CSObservableHistograms of madspace, for the
        event generator, on the kinematics of layouts."""
        ms = madspace()
        items = [
            ms.CSObservableHistograms.HistItem(
                [self.functions[pids][i] for pids, _ in self.layouts],
                histogram["min"],
                histogram["max"],
                histogram["bin_count"],
                self.jet_counts[i],
            )
            for i, histogram in enumerate(self.histograms)
        ]
        return ms.CSObservableHistograms(
            [function for _, function in self.layouts], items, self.real
        )


class CSRealHistograms(CSHistograms):
    """The histograms of the real stream of a real process, the real emission
    and the Borns of its dipoles of each point in the same histograms."""

    def __init__(self, integrand, histograms, jets):
        super().__init__(
            integrand.context, histograms, jets, 1 + len(integrand.dipoles)
        )
        self.real_jets = self.jets(integrand.source, integrand.real_pdgs)
        born_layouts = {}
        self.born_jets = []
        for dipole in integrand.dipoles:
            name = dipole.born.name
            if name not in born_layouts:
                born_layouts[name] = self.jets(dipole.born, dipole.born.born_pdgs)
            self.born_jets.append(born_layouts[name])
        self.layouts = [self.real_jets, *self.born_jets]


class CSRun(NLORun):
    """A fixed-order NLO run of madspace with the Catani-Seymour integrand on
    an aMC@NLO output (see the module and NLORun of nlo.core), with the
    alpha of the four families of dipoles (ff, fi, if, ii, by name or one
    for all), kappa of the final-state gluon kernels with a massive
    spectator, the technical cut of the real stream, the streams to
    integrate (nbody, real or both) and the channels of the real stream
    (pairs, reals or combined, see the module)."""

    scheme = "cs"
    work_directory = "madspace_cs"
    born_class = CSBornIntegrand
    real_class = CSRealIntegrand

    def __init__(
        self,
        output,
        alpha=1.0,
        kappa=2.0 / 3.0,
        technical_cut=1e-6,
        streams=STREAMS,
        channels="pairs",
        **options,
    ):
        super().__init__(output, **options)
        if not isinstance(alpha, dict):
            alpha = {family: alpha for family in FAMILIES}
        self.alpha = {family: float(alpha.get(family, 1.0)) for family in FAMILIES}
        self.kappa = kappa
        self.technical_cut = technical_cut
        if channels not in CHANNELS:
            raise ValueError(f"nlo.cs: channels of {CHANNELS} expected, not {channels}")
        self.real_channels = channels
        unknown = set(streams) - set(STREAMS)
        if unknown or not streams:
            raise ValueError(f"nlo.cs: streams of {STREAMS} expected, not {streams}")
        self.streams = tuple(streams)
        self.subprocesses = {}
        if scale_variations(self.run_card) or pdf_reweight(self.run_card):
            logger.warning(
                "nlo.cs: no scale and PDF variations yet, the run gives the "
                "central cross section"
            )
            self.run_card.update(reweight_scale=False, reweight_pdf=False)

    def make_integrands(self, directories, n_f, n_l):
        from madspace.nlo.cs import read_cs_subtraction

        for directory in directories:
            born = self.born_class(
                directory,
                self.run_card,
                self.param_card,
                self.couplings,
                self.context,
                n_f,
                self.alpha,
                self.kappa,
                n_l,
            )
            self.subprocesses[born.name] = born
        integrands = []
        if "nbody" in self.streams:
            integrands += list(self.subprocesses.values())
        if "real" in self.streams:
            masses = {int(k): float(v) for k, v in self.param_card["mass"].items()}
            block = read_cs_subtraction(self.output, masses)
            integrands += [
                self.real_class(
                    index,
                    block,
                    self.subprocesses,
                    self.run_card,
                    self.param_card,
                    self.context,
                    self.alpha,
                    self.kappa,
                    self.technical_cut,
                    self.real_channels,
                )
                for index in range(len(block["reals"]))
            ]
        return integrands

    def matrix_element_calls(self):
        return {
            name: born.matrix_element_calls()
            for name, born in self.subprocesses.items()
        }

    def close(self):
        super().close()
        for born in self.subprocesses.values():
            born.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", help="the aMC@NLO output")
    parser.add_argument("--name", default="madspace_cs", help="the run name")
    parser.add_argument("--accuracy", type=float, help="req_acc_FO of the run")
    parser.add_argument("--grid-points", type=int, help="points per grid iteration")
    parser.add_argument("--grid-iterations", type=int, help="grid iterations")
    parser.add_argument("--points", type=int, help="points per channel and iteration")
    parser.add_argument("--max-iterations", type=int, help="integration iterations")
    parser.add_argument("--seed", type=int, help="the seed (iseed of the run card)")
    parser.add_argument(
        "--threads", type=int, default=-1, help="threads of the event generator"
    )
    parser.add_argument(
        "--madnis-batches",
        type=int,
        default=0,
        help="batches of a MadNIS training per subprocess",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha of the dipoles")
    parser.add_argument(
        "--technical-cut",
        type=float,
        default=1e-6,
        help="the technical cut of the real stream",
    )
    parser.add_argument(
        "--channels",
        default="pairs",
        choices=CHANNELS,
        help="the channels of the real stream: the diagrams of the real "
        "emission with one dipole per singular pair (pairs), the diagrams "
        "alone (reals) or with all dipoles on all Born diagrams (combined)",
    )
    parser.add_argument(
        "--streams",
        default=",".join(STREAMS),
        help="the streams to integrate, nbody, real or both (nbody,real)",
    )
    parser.add_argument(
        "--histograms", help="a TOML file with [histograms] as run_card.toml"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run = CSRun(
        args.output,
        alpha=args.alpha,
        technical_cut=args.technical_cut,
        streams=tuple(args.streams.split(",")),
        channels=args.channels,
        name=args.name,
        accuracy=args.accuracy,
        grid_points=args.grid_points,
        grid_iterations=args.grid_iterations,
        points=args.points,
        max_iterations=args.max_iterations,
        seed=args.seed,
        histograms=args.histograms,
        threads=args.threads,
        madnis_batches=args.madnis_batches,
    )
    result = run.run()
    unit = result["unit"]
    print(f"NLO: {result['nlo']['sigma']:.6e} +- {result['nlo']['error']:.2e} {unit}")
    print(f"LO:  {result['lo']['sigma']:.6e} +- {result['lo']['error']:.2e} {unit}")
    return result


if __name__ == "__main__":
    main()
