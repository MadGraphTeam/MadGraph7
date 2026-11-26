"""The fixed-order NLO run of madspace on an aMC@NLO output with the
Catani-Seymour subtraction, on the core of nlo/core.py, with the matrix
elements of the Fortran provider (nlo/provider.py) and the PDFs, alpha_s,
scales and cuts of the run card.

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
154 channels, p p > e+ e- j 2.9 / 4.8 / 2.5 with 672 / 544 / 1568. With
channels="dipoles", the scheme of the first driver and that of dressed
leptons (their mappings have no inverse), the channels are those of all
dipoles on all Born diagrams, the weight of a channel |c| of its dipole
over the sum over the dipoles of the real, c the color weight of CSDipoles
at alpha = 1, times amp2 of its Born diagram over that of all Born diagrams
at the Born of the dipole, so that the weights add up to one at every
event. Each dipole takes the blocks
of the Born of its own subprocess, the spin correlation in the factorized
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
as the spectator, the decaying particle at rest. Its dipoles take the soft
terms of the decaying particle and alpha_ff, and CSInsertion their integral.
A decay has the flux 1 / (2 M) and gives a width in GeV, as the FKS driver.

Beams of dressed electrons and positrons (lpp 3 and -3) take the densities
of pdlabel among the lep_densities of the output, reduced by (1 - x)^p of
the power p of the sampling of their momentum fractions in PhaseSpaceMapping
(lepton_sampling_power), with the real emission and its dipoles at the
momentum fractions of the Born, and their real stream the channels of
channels="dipoles". Beams of elastic photons from protons (lpp
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

Not covered yet: a final-state gluon with a colored decaying particle, PDF
schemes other than MSbar, densities of eMELA, photons of gamma-UPC, the
scale and PDF variations of reweight_scale and reweight_PDF and the Sudakov
approximation of the EW corrections of --ewsudakov, which the run leaves out
with a warning.
"""

import argparse
import json
import logging
import math
import os

import numpy as np

from madgraph.mg7.nlo.libraries import transverse_square
from madgraph.mg7.nlo.provider import (
    FortranProvider,
    boost_z,
    partonic_rapidity,
)
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

INV_GEV2_TO_PB = 0.38937937217186e9
FAMILIES = ("ff", "fi", "if", "ii")
STREAMS = ("nbody", "real")
CHANNELS = ("pairs", "reals", "combined", "dipoles")


def minkowski(a, b):
    """The Minkowski product of the four-vectors a and b (..., 4)."""
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def dipole_variables(momenta, entry):
    """The two variables of the CSMapping of a dipole (an entry of the cs
    block) at real emissions (batch, n + 1, 4): y and z_i (ff, and fd with the
    recoil system K = p_a - p_i - p_j as the spectator), 1 - x and z (fi, if),
    v and 1 - x - v (ii), the lower ends of which its cutoff raises."""
    i, j, k = entry["i"], entry["j"], entry["k"]

    def product(a, b):
        return minkowski(momenta[:, a], momenta[:, b])

    if entry["family"] in ("ff", "fd"):
        pipj = product(i, j)
        if entry["family"] == "ff":
            pipk, pjpk = product(i, k), product(j, k)
        else:
            recoil = momenta[:, k] - momenta[:, i] - momenta[:, j]
            pipk = minkowski(momenta[:, i], recoil)
            pjpk = minkowski(momenta[:, j], recoil)
        return pipj / (pipj + pipk + pjpk), pipk / (pipk + pjpk)
    if entry["family"] in ("fi", "if"):
        a, f = (k, j) if entry["family"] == "fi" else (j, k)
        pipa, pfpa, pipf = product(i, a), product(f, a), product(i, f)
        return pipf / (pipa + pfpa), pipa / (pipa + pfpa)
    pipa, pipb, papb = product(i, j), product(i, k), product(j, k)
    return pipa / papb, pipb / papb


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

    def __call__(self, x1, x2, mu1, mu2, omx=None):
        """The luminosity of each group (batch, groups) at the momentum
        fractions x1, x2 and the factorization scales mu1, mu2, for dressed
        leptons with 1 - x of omx (omx1, omx2)."""
        fractions = (x1, x2, *omx) if self.dressed else (x1, x2)
        return self.runtime(*fractions, mu1, mu2)["luminosity"]


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


def mapping_outputs(out, count):
    """The momenta, momentum fractions (x1, x2, broadcast to count points),
    1 - x of dressed leptons (omx1, omx2, None without) and Jacobian of the
    forward map of a PhaseSpaceMapping or CSMapping."""
    out = [np.array(np.from_dlpack(t)) for t in out]
    momenta, x1, x2, det = out[0], out[1], out[2], out[-1]
    omx = None if len(out) == 4 else (out[3], out[4])
    x1 = np.broadcast_to(x1, count).copy()
    x2 = np.broadcast_to(x2, count).copy()
    return momenta, x1, x2, omx, det


class Densities:
    """The densities f(x, q) (not x f) of the light partons of a beam, lpp 1
    for protons and -1 for antiprotons, by PDG code of the parton, with the
    photon where the grid has one."""

    def __init__(self, pdf, lpp, n_f, context):
        ms = madspace()
        self.pids = [21] + [q for k in range(1, int(n_f) + 1) for q in (k, -k)]
        self.quark_count = len(self.pids) - 1
        if 22 in pdf.pids:
            self.pids.append(22)
        sign = -1 if lpp == -1 else 1
        grid_pids = [p if p in (21, 22) else sign * p for p in self.pids]
        self.runtime = Runtime(ms.PartonDensity(pdf, grid_pids), context)
        self.column = {p: k for k, p in enumerate(self.pids)}

    def __call__(self, x, q):
        """f (batch, partons) at the momentum fractions x and the scales q,
        zero for x > 1."""
        inside = x <= 1
        safe = np.where(inside, x, 0.5)
        values = self.runtime(safe, q)["pdf"] / safe[:, None]
        return np.where(inside[:, None], values, 0.0)

    def quarks(self, values):
        """The sum over the light quarks and antiquarks of values (batch,
        partons)."""
        return values[:, 1 : 1 + self.quark_count].sum(1)


class CSBornIntegrand:
    """The n-body stream of the Catani-Seymour integrand of one subprocess
    (a P* directory) of an aMC@NLO output, per Born channel of MadFKS (an
    integrand of nlo.core, without a discrete choice)."""

    choice_key = "choice_probabilities"

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
        self.forward = [
            ms.FunctionRuntime(m.forward_function(), context)
            for m in self.channel_mappings
        ]
        self.born_dim = self.channel_mappings[0].random_dim()
        n = len(self.born_pdgs)
        scale = nlo_scale(n, self.incoming, **scale_parameters(run_card))
        self.user_scale = scale.user()
        self.scale = Runtime(scale, context)
        self.alpha_s = Runtime(couplings.running, context)
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

    def flux(self, x1, x2):
        """The flux factor of the events with the momentum fractions x1, x2
        and the conversion to pb, 1 / (2 M) for a decay (a width in GeV)."""
        if self.incoming == 1:
            return np.full(len(x1), 1 / (2 * self.energy))
        s_hat = x1 * x2 * self.energy**2
        return np.divide(
            INV_GEV2_TO_PB, 2 * s_hat, out=np.zeros(len(s_hat)), where=s_hat > 0
        )

    def passes_cuts(self, momenta):
        """Whether the Born momenta (batch, n, 4) pass the cuts."""
        layout = np.zeros(len(momenta), dtype=np.int32)
        return self.cuts(momenta, layout_index=layout)["mask"] != 0

    def luminosities(self, x1, x2, mu1, mu2, z=None, omx=None):
        """The luminosity of the Born over its flavors and, with the points z
        (batch, 2) of the convolution, those of the K and P operators of each
        beam per group of corrections: the same parton at x_i / z, the
        partons that split into it at x_i / z (QCD: the gluon for a quark,
        the quarks for a gluon, QED: the photon for a quark, the quarks times
        Q_q^2 for a photon), each over z and times the density of the Born
        parton of the other beam (batch, 2, 2). Dressed leptons take 1 - x of
        omx."""
        born = self.luminosity(x1, x2, mu1, mu2, omx)[:, 0]
        if z is None or self.convolution is None:
            return born, None
        convolved = {group: np.zeros((len(x1), 2, 2)) for group in self.groups}
        for leg in range(2):
            shifted = [x1, x2][leg] / z[:, leg]
            inside = shifted <= 1
            fractions = [x1, x2]
            fractions[leg] = np.where(inside, shifted, 0.5)
            values = self.convolution[leg](*fractions, mu1, mu2)
            values = np.where(inside[:, None], values / z[:, leg][:, None], 0.0)
            for k, group in enumerate(self.groups):
                convolved[group][:, leg, 0] = values[:, 0]
                convolved[group][:, leg, 1] = values[:, 1 + k]
        return born, convolved

    def references(self, momenta, valid=None):
        """The reference scale of the user (dynamical_scale_choice 0 or 10)
        at momenta of the Born or of the real process, from the provider, as
        the arguments of the scales it takes, none for the other scales."""
        if not self.user_scale:
            return []
        return [self.provider.reference_scales(momenta, valid)]

    def evaluate(self, index, x, choice, choice_weight, trick, rng):
        """The weights of a channel at the points x of its random numbers (see
        SubprocessIntegrand.evaluate of nlo/fks.py), with the virtual on
        the fraction of the points of trick."""
        count = len(x)
        out = self.forward[index].call([np.ascontiguousarray(x[:, : self.born_dim])])
        momenta, x1, x2, omx, det = mapping_outputs(out, count)
        scales = self.scale(momenta, *self.references(momenta, det != 0))
        mu_r = scales["ren_scale"]
        alpha_s = self.alpha_s(mu_r)["alpha_s"]
        passes = self.passes_cuts(momenta)
        valid = passes & (det != 0)
        b = len(self.qcd_powers)
        n = len(self.born_pdgs)
        born = np.zeros((count, b))
        colors = np.zeros((count, b, n, n))
        amp2 = np.zeros((count, self.provider.channel_count))
        virtual = np.zeros((count, self.nlo_count))
        evaluated = valid & (rng.random(count) < trick.fraction)
        points = np.flatnonzero(valid)
        if len(points):
            cm = boost_z(
                momenta[points], -partonic_rapidity(momenta[points], self.incoming)
            )
            configuration = np.zeros(len(points), dtype=np.int32)
            per_position, _, amp2[points] = self.provider.born(cm, configuration)
            born[points] = self.provider.born_orders_of(per_position)
            colors[points] = self.provider.color_correlated(cm)
            take = evaluated[points]
            if take.any():
                virtual[points[take]] = self.provider.virtual_orders(
                    cm[take], mu_r[points][take]
                )
        total = amp2.sum(1)
        channel_weight = np.where(
            total > 0, amp2[:, index] / np.where(total > 0, total, 1), 0.0
        )
        used, approximation = trick.inputs(virtual, born, evaluated)
        # the operators of each group of corrections, I and the weights of K
        # and P, with alpha_s or alpha
        alpha = np.full(count, self.alpha)
        insertions, collinears = {}, {}
        if self.insertion is not None:
            insertions["QCD"] = self.insertion(momenta, colors, alpha_s, mu_r)[
                "insertion"
            ]
        if self.insertion_qed is not None:
            insertions["QED"] = self.insertion_qed(momenta, born, alpha, mu_r)[
                "insertion"
            ]
        z = None
        r = np.ascontiguousarray(x[:, self.born_dim]) if self.hadrons else None
        scale_args = (scales["fact_scale1"], scales["fact_scale2"], x1, x2, r)
        if self.hadrons and self.collinear is not None:
            collinear = self.collinear(momenta, born, colors, alpha_s, *scale_args)
            z, collinears["QCD"] = collinear["z"], collinear["weights"]
        if self.hadrons and self.collinear_qed is not None:
            collinear = self.collinear_qed(momenta, born, alpha, *scale_args)
            z, collinears["QED"] = collinear["z"], collinear["weights"]
        born_lum, convolved = self.luminosities(
            x1, x2, scales["fact_scale1"], scales["fact_scale2"], z, omx
        )
        g_s = np.sqrt(4 * math.pi * alpha_s)
        norm = np.where(valid, det * channel_weight * self.flux(x1, x2), 0.0)

        def weight(virtual_part):
            total, lo = np.zeros(count), np.zeros(count)
            for k, power in enumerate(self.qcd_powers):
                coupling = g_s**power
                lo += born_lum * coupling * born[:, k]
                total += born_lum * coupling * born[:, k]
                for group in self.groups:
                    if self.corrections[group][k] is None:
                        continue
                    total += born_lum * coupling * insertions[group][:, k]
                    if group in collinears:
                        weights, lums = collinears[group], convolved[group]
                        kp = (
                            weights[:, k, :, 0] * lums[:, :, 0]
                            + weights[:, k, :, 1] * lums[:, :, 1]
                            + weights[:, k, :, 2] * born_lum[:, None]
                        )
                        total += coupling * kp.sum(1)
            # the virtual of each NLO order, with its powers of g_s
            for n, power in enumerate(self.nlo_powers):
                total += born_lum * g_s**power * virtual_part[:, n]
            return norm * total, norm * lo

        nlo, lo = weight(used)
        without, _ = weight(approximation)
        return dict(
            nlo=nlo,
            lo=lo,
            residual=nlo - without,
            momenta=momenta,
            virtual=virtual,
            born=born,
            evaluated=evaluated,
        )

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
        return runtime

    def histogram_sums(self, runtime, result, jacobian, choice):
        inputs = {"weight0": result["nlo"] * jacobian}
        runtime.kinematics(runtime.born, result["momenta"], inputs, 0)
        return runtime.sums(inputs)

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

    def correlated(self, provider, cm, plain):
        """The Born correlated in color (QCD) or charge (QED) of the emitter
        and the spectator per Born order (points, b), -weight B for a photon
        that splits."""
        if self.group == "QCD":
            return provider.color_correlated(cm)[:, :, self.legs[0], self.legs[1]]
        if self.weight is not None:
            return -self.weight * plain
        return self.charges * plain


class CSRealIntegrand:
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
        if channels != "dipoles" and subprocesses[entry["subprocess"]].lepton_power:
            # the mappings of dressed leptons have no inverse for the densities
            channels = "dipoles"
        self.channels_kind = channels
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
        self.alpha_s = source.alpha_s
        self.luminosity_function = Luminosity(
            [self.flavors], self.beams, self.incoming, source.couplings, context
        )
        self.dipoles, self.channels, self.forward, self.dimensions = [], [], [], []
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
                self.forward.append(
                    ms.FunctionRuntime(mapping.forward_function(), context)
                )
                self.dimensions.append(mapping.random_dim())
            self.dipoles.append(dipole)
            kernels.append(d["kernel"])
            symmetry.append(d["symmetry"])
        # the channels of the diagrams of the real emission, and the inverse
        # maps of all channels for their densities
        self.inverse = None
        if channels in ("combined", "reals", "pairs"):
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
                self.forward.append(
                    ms.FunctionRuntime(mapping.forward_function(), context)
                )
                self.dimensions.append(mapping.random_dim())
        self.channel_mappings = channel_mappings
        if channels != "dipoles":
            self.inverse = [
                ms.FunctionRuntime(mapping.inverse_function(), context)
                for mapping in channel_mappings
            ]
            # the mappings of the Borns and of the real emission have no
            # discrete choices, which would change their outputs
            if any(m.discrete_dim() for m in discrete_mappings):
                raise NotImplementedError(
                    "nlo.cs: no channels with discrete choices yet"
                )
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
        # the weights of the channels, without the gates of alpha
        self.ungated = self.subtraction
        if any(alpha[family] != 1.0 for family in FAMILIES):
            self.ungated = Runtime(
                ms.CSDipoles(mappings, kernels, symmetry, kappa=kappa), context
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

    def luminosity(self, x1, x2, mu1, mu2, omx=None):
        """The luminosity of the real over its flavor lines, at the momentum
        fractions of the real and the factorization scales mu1, mu2, for
        dressed leptons with 1 - x of omx."""
        return self.luminosity_function(x1, x2, mu1, mu2, omx)[:, 0]

    def covered(self, index, momenta):
        """Whether real emissions are in the region of the channel index, both
        variables of its dipole at or above the technical cut, those of every
        dipole for the channel of a diagram of the real emission."""
        keep = np.ones(len(momenta), dtype=bool)
        m = self.channels[index][0]
        dipoles = self.dipoles if m is None else [self.dipoles[m]]
        with np.errstate(divide="ignore", invalid="ignore"):
            for dipole in dipoles:
                for variable in dipole_variables(momenta, dipole.entry):
                    keep &= variable >= self.technical_cut
        return keep

    def densities(self, momenta, fractions, valid):
        """The densities of the mappings of all channels at real emissions
        (batch, channels), as MappingChannelWeights: zero where a mapping
        cannot produce the event (a random number of its inverse outside the
        unit hypercube)."""
        count = len(momenta)
        inputs = [np.ascontiguousarray(momenta)]
        inputs += [np.ascontiguousarray(np.broadcast_to(f, count)) for f in fractions]
        densities = np.zeros((count, len(self.inverse)))
        for k, inverse in enumerate(self.inverse):
            out = inverse.call(inputs)
            random = np.array(np.from_dlpack(out[0])).reshape(count, -1)
            density = np.array(np.from_dlpack(out[-1]))
            density = np.broadcast_to(density, count)
            good = valid & np.isfinite(density) & (density > 0)
            good &= ((random >= 0) & (random <= 1)).all(axis=1)
            densities[:, k] = np.where(good, density, 0.0)
        return densities

    def dipole_value(self, dipole, cm, k_t, color_weight, spin_weight, born):
        """-D / g_s^2 per Born order of a dipole (points, b) at its Born in
        the partonic center-of-mass frame cm, with k_t in the same frame,
        the weights of CSDipoles and born the output of the provider."""
        provider = dipole.born.provider
        plain = provider.born_orders_of(born[0])
        colors = dipole.correlated(provider, cm, plain)
        color_weight = dipole.factor * color_weight
        spin_weight = dipole.factor * spin_weight
        value = color_weight[:, None] * colors
        if dipole.gluon:
            configuration = np.full(len(cm), dipole.configuration)
            if dipole.swap is None:
                perp = provider.spin_correlated(cm, k_t, configuration, born=born)
            else:
                # the Born with the emitter and its identical gluon exchanged
                exchanged = cm.copy()
                exchanged[:, [dipole.legs[0], dipole.swap]] = cm[
                    :, [dipole.swap, dipole.legs[0]]
                ]
                perp = provider.spin_correlated(exchanged, k_t, configuration)
            # -k_t^2 as the square of the part of k_t transverse to the gluon
            # in its frame, as the libraries take it (equal for a massless
            # emitter, without the cancellation of the Minkowski square)
            square = transverse_square(cm[:, dipole.legs[0]], k_t)
            ratio = np.divide(
                colors, plain, out=np.zeros_like(colors), where=plain != 0
            )
            value += (spin_weight * square)[:, None] * ratio * perp
        return value

    def evaluate(self, index, x, choice, choice_weight, trick, rng):
        """The weights of a channel at the points x of its random numbers (see
        SubprocessIntegrand.evaluate of nlo/fks.py and weights)."""
        count = len(x)
        out = self.forward[index].call([np.ascontiguousarray(x)])
        momenta, x1, x2, omx, det = mapping_outputs(out, count)
        return self.weights(index, momenta, x1, x2, det, covered=det != 0, omx=omx)

    def weights(self, index, momenta, x1, x2, det, covered=None, omx=None):
        """The weights of the channel index at real emissions (batch, n + 1, 4)
        in the lab frame, with their momentum fractions and the Jacobian det
        of their mapping, and the parts of the histograms: the real emission
        and its weight, the Borns of the dipoles in the order of their matrix
        elements and their weights. covered says which events are in the
        region of the channel (all those of its mapping), from the variables
        of its dipole (covered) if None. Dressed leptons take 1 - x of omx
        (omx1, omx2)."""
        count = len(momenta)
        m_channel, b_channel = self.channels[index]
        valid = (det != 0) & np.isfinite(det)
        valid &= np.isfinite(momenta).all(axis=(1, 2))
        momenta = np.where(valid[:, None, None], momenta, 0.0)
        incoming = momenta[:, : self.incoming].sum(1)
        valid &= minkowski(incoming, incoming) > 0
        if covered is None:
            covered = self.covered(index, momenta)
        fractions = (x1, x2) if omx is None else (x1, x2, *omx)
        projected = self.subtraction(momenta, *fractions)
        if m_channel is None:
            # the channel of a diagram covers the events where the variables of
            # all dipoles are at or above the technical cut
            covered = covered & (projected["covered"] != 0).all(axis=1)
        keep = valid & covered
        if self.inverse is None:
            ungated = self.ungated(momenta, *fractions)["color_weight"]
            rho = np.abs(ungated)
            total = rho.sum(1)
            # not where the ungated dipoles are infinite (the momenta of the
            # points that are not valid, set to zero)
            channel_weight = np.divide(
                rho[:, m_channel],
                total,
                out=np.zeros(count),
                where=np.isfinite(total) & (total > 0),
            )
            amplitude_weight = np.zeros(count)
        else:
            # the density of the mapping of the channel over that of all
            densities = self.densities(momenta, fractions, valid)
            total = densities.sum(1)
            channel_weight = np.divide(
                densities[:, index],
                total,
                out=np.zeros(count),
                where=np.isfinite(total) & (total > 0),
            )
            amplitude_weight = np.ones(count)
        # the real emission
        fks_index = np.full(count, self.configuration, dtype=np.int32)
        scales = self.scale(momenta, *self.source.references(momenta, valid))
        layout = np.zeros(count, dtype=np.int32)
        passes = self.cuts(momenta, layout_index=layout)["mask"] != 0
        points = np.flatnonzero(keep & passes)
        real_weight = np.zeros(count)
        if len(points):
            cm = boost_z(
                momenta[points], -partonic_rapidity(momenta[points], self.incoming)
            )
            matrix_element = self.source.provider.real_orders(cm, fks_index[points])
            alpha_s = self.alpha_s(scales["ren_scale"][points])["alpha_s"]
            g_s = np.sqrt(4 * math.pi * alpha_s)
            weight = sum(
                g_s**power * matrix_element[:, o]
                for o, power in enumerate(self.real_powers)
            )
            lum = self.luminosity(
                x1[points],
                x2[points],
                scales["fact_scale1"][points],
                scales["fact_scale2"][points],
                None if omx is None else (omx[0][points], omx[1][points]),
            )
            real_weight[points] = lum * weight
        # the dipoles, each at its Born
        d = len(self.dipoles)
        n = len(self.pdgs) - 1
        born_momenta = np.zeros((count, d, n, 4))
        dipole_weights = np.zeros((count, d))
        for m, dipole in enumerate(self.dipoles):
            born = dipole.born
            me = projected["born_momenta"][:, m][:, dipole.order]
            me = np.where(valid[:, None, None], me, 0.0)
            born_momenta[:, m] = me
            color_weight = projected["color_weight"][:, m]
            spin_weight = projected["spin_weight"][:, m]
            active = keep & ((color_weight != 0) | (spin_weight != 0))
            active &= born.passes_cuts(me)
            amplitudes = self.inverse is None and m == m_channel
            needed = active | (keep if amplitudes else False)
            points = np.flatnonzero(needed)
            if not len(points):
                continue
            y = -partonic_rapidity(me[points], self.incoming)
            cm = boost_z(me[points], y)
            evaluated = born.provider.born(
                cm, np.full(len(points), dipole.configuration)
            )
            if amplitudes:
                amp2 = evaluated[2]
                amp2_total = amp2.sum(1)
                amplitude_weight[points] = np.divide(
                    amp2[:, b_channel],
                    amp2_total,
                    out=np.zeros(len(points)),
                    where=amp2_total > 0,
                )
            take = active[points]
            if not take.any():
                continue
            points, y, cm = points[take], y[take], cm[take]
            k_t = boost_z(projected["spin_vector"][points, m], y)
            value = self.dipole_value(
                dipole,
                cm,
                k_t,
                color_weight[points],
                spin_weight[points],
                tuple(part[take] for part in evaluated),
            )
            born_scales = born.scale(me[points], *born.references(me[points]))
            alpha_s = born.alpha_s(born_scales["ren_scale"])["alpha_s"]
            g_s = np.sqrt(4 * math.pi * alpha_s)
            # -D, the term the real emission subtracts, with g_s^2 or e^2 of
            # its splitting for the Born orders whose correction the NLO
            # orders have
            coupling = 4 * math.pi * born.alpha if dipole.group == "QED" else g_s**2
            weight = sum(
                coupling * g_s**power * value[:, o]
                for o, power in enumerate(born.qcd_powers)
                if dipole.orders[o]
            )
            lum = self.luminosity(
                x1[points],
                x2[points],
                born_scales["fact_scale1"],
                born_scales["fact_scale2"],
                None if omx is None else (omx[0][points], omx[1][points]),
            )
            dipole_weights[points, m] = lum * weight
        norm = np.where(
            keep,
            det * channel_weight * amplitude_weight * self.source.flux(x1, x2),
            0.0,
        )
        real_weight *= norm
        dipole_weights *= norm[:, None]
        zero = np.zeros(count)
        return dict(
            nlo=real_weight + dipole_weights.sum(1),
            lo=zero,
            residual=zero,
            momenta=momenta,
            real_weight=real_weight,
            born_momenta=born_momenta,
            dipole_weights=dipole_weights,
            virtual=np.zeros((count, self.nlo_count)),
            born=np.zeros((count, 1)),
            evaluated=np.zeros(count, dtype=bool),
        )

    def channel_integrand(self, channel, context):
        """The C++ integrand of a channel (CSRealChannelIntegrand of
        madspace) with the matrix elements of the providers as UMAMI libraries
        in the context, the same integrand as evaluate with the combined
        channels at the same random numbers. The libraries evaluate at the e
        of the model, so that the blocks are those of the providers."""
        ms = madspace()
        element = ms.MatrixElement
        if self.inverse is None:
            raise NotImplementedError(
                "nlo.cs: the C++ integrand takes the densities of the channels, "
                "which dressed leptons and channels='dipoles' have not"
            )
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

    def histogram_sums(self, runtime, result, jacobian, choice):
        return runtime(result, jacobian)

    def matrix_element_calls(self):
        return {}

    def close(self):
        pass


class CSHistograms:
    """The histograms of a stream of the CS integrand, the kinematics of each
    point in the same histograms (NLOHistograms): the jets of NLOJets on each
    kinematics, in its flavor layout, and the observables of the histograms
    on them (jet_observable of madspace.nlo). A kinematics of a subprocess
    has as many jet slots as its real processes have partons, so that the
    observables of the real emission also take its Born."""

    def __init__(self, context, histograms, jets, kinematics_count):
        ms = madspace()
        from madspace.nlo import histogram_jet_count

        self.context = context
        self.histograms = histograms
        self.jet_options = jets
        self.observables = {}
        items = [
            ms.NLOHistograms.HistItem(
                histogram["min"],
                histogram["max"],
                histogram["bin_count"],
                histogram_jet_count(dict(histogram["observable"])),
            )
            for histogram in histograms
        ]
        self.binning = Runtime(ms.NLOHistograms(items, kinematics_count), context)

    def jets(self, subprocess, layout):
        """The NLOJets of a flavor layout of the subprocess, with the
        observables of the histograms on its momenta with jets."""
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
        if pids not in self.observables:
            self.observables[pids] = [
                Runtime(jet_observable(function, histogram), self.context)
                for histogram in self.histograms
            ]
        return Runtime(function, self.context), pids

    def kinematics(self, jets, momenta, inputs, k):
        """The jets and the observables of kinematics k into inputs."""
        runtime, pids = jets
        layout = np.zeros(len(momenta), dtype=np.int32)
        out = runtime(momenta, layout_index=layout)
        inputs[f"jet_count{k}"] = out["jet_count"].astype(np.int32)
        for i, observable in enumerate(self.observables[pids]):
            value = observable(out["jet_momenta"])
            inputs[f"value{i}_{k}"] = np.asarray(next(iter(value.values()))).reshape(
                len(momenta)
            )

    def sums(self, inputs):
        return list(self.binning.by_name(inputs).values())


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

    def __call__(self, result, jacobian):
        inputs = {"weight0": result["real_weight"] * jacobian}
        self.kinematics(self.real_jets, result["momenta"], inputs, 0)
        for m, jets in enumerate(self.born_jets):
            inputs[f"weight{m + 1}"] = result["dipole_weights"][:, m] * jacobian
            self.kinematics(jets, result["born_momenta"][:, m], inputs, m + 1)
        return self.sums(inputs)


class CSRun(NLORun):
    """A fixed-order NLO run of madspace with the Catani-Seymour integrand on
    an aMC@NLO output (see the module and NLORun of nlo.core), with the
    alpha of the four families of dipoles (ff, fi, if, ii, by name or one
    for all), kappa of the final-state gluon kernels with a massive
    spectator, the technical cut of the real stream, the streams to
    integrate (nbody, real or both) and the channels of the real stream
    (pairs, reals, combined or dipoles, see the module)."""

    scheme = "cs"
    work_directory = "madspace_cs"

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
            born = CSBornIntegrand(
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
                CSRealIntegrand(
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
        "alone (reals), with all dipoles on all Born diagrams (combined), or "
        "all dipoles with the weights of the first driver (dipoles)",
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
    )
    result = run.run()
    unit = result["unit"]
    print(f"NLO: {result['nlo']['sigma']:.6e} +- {result['nlo']['error']:.2e} {unit}")
    print(f"LO:  {result['lo']['sigma']:.6e} +- {result['lo']['error']:.2e} {unit}")
    return result


if __name__ == "__main__":
    main()
