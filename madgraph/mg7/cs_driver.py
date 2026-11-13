"""The fixed-order NLO run of madspace on an aMC@NLO output with the
Catani-Seymour subtraction, on the core of nlo_driver.py, with the matrix
elements of the Fortran provider (fks_provider.py) and the PDFs, alpha_s,
scales and cuts of the run card.

    python -m madgraph.mg7.cs_driver OUTPUT [--name RUN] [--accuracy ACC]

The n-body stream of a subprocess (a P* directory) has the Born, the
virtual (MadLoop, with the virtual trick of the FKS driver), the insertion
operator I (CSInsertion) and the K and P operators (CSCollinear) of each of
its Born channels, with the multichannel weight of MadFKS (amp2 of the Born
diagram of the channel over that of all Born diagrams). The K and P
operators take one more random number, the point z = x_i + (1 - x_i) r of
the convolution of both beams, with the densities of the partons that
split into those of the Born at x_i / z. The cuts act on the Born, as the
FKS driver cuts its soft counter-event, and the scales are those of the
Born.

The I operator is that of CDR with the normalization (4 pi)^eps / Gamma(1 -
eps) of the virtual of MadLoop at the renormalization scale, with the n_f
light flavors of the model (coupl.inc). The Catani-Seymour corrections are
those of QCD to every Born order (an NLO order is a Born order with two more
powers of g_s).

The real stream of a real process (a real matrix element of the cs block of
madspace.cs, with its flavor lines) is the real emission minus its dipoles
(CSDipoles), with one channel per dipole and Born channel of its Born: the
CSMapping of the dipole on the Born diagram of the channel. The weight of a
channel is |c| of its dipole over the sum over the dipoles of the real, c
the color weight of CSDipoles at alpha = 1, times amp2 of its Born diagram
over that of all Born diagrams at the Born of the dipole, so that the
weights add up to one at every event. Each dipole takes the blocks of the
Born of its own subprocess, the spin correlation in the factorized form
<T_j . T_k> B(k_t) / B, with B(k_t) the Born of the provider with the
azimuthal part of an FKS configuration whose mother is the emitter. The
real emission has the cuts and scales of the real process, each dipole those
of its Born, and both the densities at the momentum fractions of the real
emission.

The technical cut is the cutoff of the CSMapping of every channel: a channel
covers only the events where both variables of its dipole (y and z_i, 1 - x
and z, v and 1 - x - v) are at or above technical_cut, and its events are
never cut afterwards, as the FKS configurations of MadFKS cover xi and 1 - y
from 1e-6 up (genps_fks.f). The real emission minus the dipoles then has the
cut sum_c w_c Theta_c of the channel weights w_c and of the regions Theta_c
of the channels, zero where i is soft below the cut and of the order of the
cut where it is collinear, since there the weights of the other channels
vanish with the invariant. It keeps the real emission of the Fortran in
double precision away from the soft limit, towards which it loses digits as
1 / xi^2 (by 1e-4 to 1e-3 at an energy fraction xi = 5e-7 of the soft gluon
of g g > t t~ g, depending on the angle).

Not covered yet: decays, PDF schemes other than MSbar, user scales and the
photon cuts.
"""

import argparse
import logging
import math
import os

import numpy as np

from madgraph.mg7.fks_driver import HISTOGRAM_KINEMATICS
from madgraph.mg7.fks_provider import (
    FortranProvider,
    boost_z,
    partonic_rapidity,
)
from madgraph.mg7.nlo_driver import (
    NLORun,
    Runtime,
    born_topology,
    integer,
    madspace,
    real,
    run_cuts,
    scale_parameters,
)

logger = logging.getLogger("madgraph.mg7.cs_driver")

INV_GEV2_TO_PB = 0.38937937217186e9
FAMILIES = ("ff", "fi", "if", "ii")
STREAMS = ("nbody", "real")


def minkowski(a, b):
    """The Minkowski product of the four-vectors a and b (..., 4)."""
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def dipole_variables(momenta, entry):
    """The two variables of the CSMapping of a dipole (an entry of the cs
    block) at real emissions (batch, n + 1, 4): y and z_i (ff), 1 - x and z
    (fi, if), v and 1 - x - v (ii), the lower ends of which its cutoff
    raises."""
    i, j, k = entry["i"], entry["j"], entry["k"]

    def product(a, b):
        return minkowski(momenta[:, a], momenta[:, b])

    if entry["family"] == "ff":
        pipj, pipk, pjpk = product(i, j), product(i, k), product(j, k)
        return pipj / (pipj + pipk + pjpk), pipk / (pipk + pjpk)
    if entry["family"] in ("fi", "if"):
        a, f = (k, j) if entry["family"] == "fi" else (j, k)
        pipa, pfpa, pipf = product(i, a), product(f, a), product(i, f)
        return pipf / (pipa + pfpa), pipa / (pipa + pfpa)
    pipa, pipb, papb = product(i, j), product(i, k), product(j, k)
    return pipa / papb, pipb / papb


def spin_configuration(provider, leg):
    """The first FKS configuration of a provider whose mother is a gluon at
    the Born leg leg (from 0), None without one."""
    for c, configuration in enumerate(provider.configurations):
        i_fks, j_fks = configuration["i_fks"], configuration["j_fks"]
        if configuration["ij_pdg"] == 21 and j_fks - 1 - (j_fks > i_fks) == leg:
            return c
    return None


def qcd_orders(orders):
    """The NLO order of each Born order of fks_configurations.json, which has
    two more powers of g_s (the first split order) and the same others.

    Raises:
        NotImplementedError: for NLO orders other than the QCD corrections
            of the Born orders.
    """
    if orders["split_orders"][0] != "QCD":
        raise NotImplementedError("cs_driver: the first split order is not QCD")
    nlo = [list(o) for o in orders["nlo"]]
    out = []
    for born in orders["born"]:
        target = [born[0] + 2] + list(born[1:])
        if target not in nlo:
            raise NotImplementedError(
                f"cs_driver: the Born order {born} has no QCD correction"
            )
        out.append(nlo.index(target))
    if sorted(out) != list(range(len(nlo))):
        raise NotImplementedError("cs_driver: NLO orders other than QCD corrections")
    return out


class Densities:
    """The densities f(x, q) (not x f) of the light partons of a beam, lpp 1
    for protons and -1 for antiprotons, by PDG code of the parton."""

    def __init__(self, pdf, lpp, n_f, context):
        ms = madspace()
        self.pids = [21] + [q for k in range(1, int(n_f) + 1) for q in (k, -k)]
        sign = -1 if lpp == -1 else 1
        grid_pids = [p if p == 21 else sign * p for p in self.pids]
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
        return values[:, 1:].sum(1)


class CSBornIntegrand:
    """The n-body stream of the Catani-Seymour integrand of one subprocess
    (a P* directory) of an aMC@NLO output, per Born channel of MadFKS (an
    integrand of nlo_driver, without a discrete choice)."""

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
    ):
        ms = madspace()
        from madspace.fks import fks_cuts, fks_scale

        self.name = os.path.basename(os.path.normpath(directory))
        self.context = context
        self.provider = FortranProvider(directory)
        self.configurations = self.provider.fks_configurations
        self.incoming = self.configurations["nincoming"]
        if self.incoming != 2:
            raise NotImplementedError("cs_driver: no decays yet")
        self.born_pdgs = list(self.configurations["born_pdgs"])
        self.masses = [
            float(param_card["mass"].get(str(abs(pdg)), 0.0)) for pdg in self.born_pdgs
        ]
        self.energy = 2 * math.sqrt(real(run_card["ebeam1"]) * real(run_card["ebeam2"]))
        leptonic = integer(run_card["lpp1"]) == 0
        self.beams = (
            (0, 0)
            if leptonic
            else (integer(run_card["lpp1"]), integer(run_card["lpp2"]))
        )
        orders = self.configurations["orders"]
        self.nlo_of_born = qcd_orders(orders)
        self.qcd_powers = [int(o[0]) for o in orders["born"]]
        self.nlo_count = len(orders["nlo"])
        self.channels = self.provider.channels()
        self.channel_mappings = [
            ms.PhaseSpaceMapping(
                born_topology(self.masses, self.incoming, channel),
                self.energy,
                leptonic=leptonic,
            )
            for channel in self.channels
        ]
        self.forward = [
            ms.FunctionRuntime(m.forward_function(), context)
            for m in self.channel_mappings
        ]
        self.born_dim = self.channel_mappings[0].random_dim()
        n = len(self.born_pdgs)
        self.scale = Runtime(
            fks_scale(
                self.configurations, **scale_parameters(run_card)
            ).with_particle_count(n),
            context,
        )
        self.alpha_s = Runtime(couplings.running, context)
        # the cuts of the Born as FKSCuts of the real with a parton of zero
        # momentum at i_fks of the first FKS configuration, as the soft
        # counter-event of the FKS driver
        self.cut_position = self.configurations["configurations"][0]["i_fks"] - 1
        self.cuts = Runtime(
            fks_cuts(
                self.channel_mappings[0], self.configurations, **run_cuts(run_card)
            ),
            context,
        )
        b = len(orders["born"])
        options = dict(born_orders=b, n_f=n_f)
        self.insertion = Runtime(
            ms.CSInsertion(
                self.born_pdgs,
                self.masses,
                kappa=kappa,
                alpha_ff=alpha["ff"],
                **options,
            ),
            context,
        )
        self.densities = None
        if self.beams != (0, 0):
            self.collinear = Runtime(
                ms.CSCollinear(
                    self.born_pdgs,
                    self.masses,
                    alpha_fi=alpha["fi"],
                    alpha_if=alpha["if"],
                    alpha_ii=alpha["ii"],
                    **options,
                ),
                context,
            )
            self.densities = [
                Densities(couplings.pdf, lpp, n_f, context) for lpp in self.beams
            ]
        self.flavors = [list(f) for f in self.configurations["born_flavors"]]

    @property
    def channel_count(self):
        return len(self.channel_mappings)

    def channel_dimension(self, index):
        # the Born and, with densities, the point of the convolution of K, P
        return self.born_dim + (self.densities is not None)

    def channel_name(self, index):
        return f"{self.name} n-body channel {index + 1}"

    def choice_count(self, index):
        return 1

    def soft_momenta(self, momenta):
        """The Born momenta (batch, n, 4) with a parton of zero momentum at
        i_fks of the first FKS configuration (batch, n + 1, 4)."""
        return np.insert(momenta, self.cut_position, 0.0, axis=1)

    def luminosities(self, x1, x2, mu1, mu2, z=None):
        """The luminosity of the Born over its flavors and, with the points z
        (batch, 2) of the convolution, those of the K and P operators of each
        beam: the same parton at x_i / z, the partons that split into it at
        x_i / z, each times the density of the Born parton of the other beam
        (batch, 2, 2)."""
        if self.densities is None:
            return np.full(len(x1), float(len(self.flavors))), None
        x = (x1, x2)
        mu = (mu1, mu2)
        at_born = [d(xi, m) for d, xi, m in zip(self.densities, x, mu)]
        born = np.zeros(len(x1))
        for flavor in self.flavors:
            a, b = flavor[0], flavor[1]
            born += (
                at_born[0][:, self.densities[0].column[a]]
                * at_born[1][:, self.densities[1].column[b]]
            )
        if z is None:
            return born, None
        convolved = np.zeros((len(x1), 2, 2))
        for leg in range(2):
            d = self.densities[leg]
            values = d(x[leg] / z[:, leg], mu[leg]) / z[:, leg][:, None]
            other = self.densities[1 - leg]
            for flavor in self.flavors:
                parton, partner = flavor[leg], flavor[1 - leg]
                at_partner = at_born[1 - leg][:, other.column[partner]]
                same = values[:, d.column[parton]]
                splitting = (
                    d.quarks(values) if parton == 21 else values[:, d.column[21]]
                )
                convolved[:, leg, 0] += same * at_partner
                convolved[:, leg, 1] += splitting * at_partner
        return born, convolved

    def evaluate(self, index, x, choice, choice_weight, trick, rng):
        """The weights of a channel at the points x of its random numbers (see
        SubprocessIntegrand.evaluate of fks_driver.py), with the virtual on
        the fraction of the points of trick."""
        count = len(x)
        out = self.forward[index].call([np.ascontiguousarray(x[:, : self.born_dim])])
        momenta, x1, x2, det = [np.array(np.from_dlpack(t)) for t in out]
        x1 = np.broadcast_to(x1, count).copy()
        x2 = np.broadcast_to(x2, count).copy()
        scales = self.scale(momenta)
        mu_r = scales["ren_scale"]
        alpha_s = self.alpha_s(mu_r)["alpha_s"]
        fks_index = np.zeros(count, dtype=np.int32)
        passes = self.cuts(self.soft_momenta(momenta), fks_index=fks_index)["mask"] != 0
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
            cm = boost_z(momenta[points], -partonic_rapidity(momenta[points], 2))
            per_position, _, amp2[points] = self.provider.born(cm, fks_index[points])
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
        insertion = self.insertion(momenta, colors, alpha_s, mu_r)["insertion"]
        z = weights = None
        if self.densities is not None:
            collinear = self.collinear(
                momenta,
                born,
                colors,
                alpha_s,
                scales["fact_scale1"],
                scales["fact_scale2"],
                x1,
                x2,
                np.ascontiguousarray(x[:, self.born_dim]),
            )
            z, weights = collinear["z"], collinear["weights"]
        born_lum, convolved = self.luminosities(
            x1, x2, scales["fact_scale1"], scales["fact_scale2"], z
        )
        g_s = np.sqrt(4 * math.pi * alpha_s)
        s_hat = x1 * x2 * self.energy**2
        norm = np.where(valid, INV_GEV2_TO_PB * det * channel_weight / (2 * s_hat), 0.0)

        def weight(virtual_part):
            total, lo = np.zeros(count), np.zeros(count)
            for k, power in enumerate(self.qcd_powers):
                coupling = g_s**power
                nlo = coupling * g_s**2
                parts = born[:, k] + insertion[:, k]
                lo += born_lum * coupling * born[:, k]
                total += born_lum * coupling * parts
                total += born_lum * nlo * virtual_part[:, self.nlo_of_born[k]]
                if weights is not None:
                    kp = (
                        weights[:, k, :, 0] * convolved[:, :, 0]
                        + weights[:, k, :, 1] * convolved[:, :, 1]
                        + weights[:, k, :, 2] * born_lum[:, None]
                    )
                    total += coupling * kp.sum(1)
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

    def histogram_runtime(self, histograms, jets):
        """The FKSObservableHistograms of the histograms with the jets of the
        run card, filled with the Born alone."""
        from madspace.fks import fks_histograms

        return Runtime(
            fks_histograms(
                self.channel_mappings[0], self.configurations, histograms, **jets
            ),
            self.context,
        )

    def histogram_sums(self, runtime, result, jacobian, choice):
        momenta = result["momenta"]
        soft = self.soft_momenta(momenta)
        inputs = {name: soft for name, _ in HISTOGRAM_KINEMATICS}
        inputs["born_momenta"] = momenta
        zero = np.zeros(len(momenta))
        inputs.update({weight: zero for _, weight in HISTOGRAM_KINEMATICS})
        inputs["nbody_weight"] = result["nlo"] * jacobian
        fks_index = np.zeros(len(momenta), dtype=np.int32)
        return list(runtime.by_name(inputs, fks_index=fks_index).values())

    def matrix_element_calls(self):
        return self.provider.statistics

    def close(self):
        self.provider.close()


class CSDipole:
    """One dipole of a real process (an entry of the cs block of madspace.cs)
    with the CSBornIntegrand of the subprocess of its Born: the Born legs
    (from 0, in the order of the matrix element) of its emitter and its
    spectator and, for a gluon emitter, the FKS configuration of the spin
    correlation."""

    def __init__(self, entry, born):
        self.entry = entry
        self.born = born
        self.order = list(entry["born_order"])
        i = entry["i"]
        emitter = entry["j"] - (entry["j"] > i)
        spectator = entry["k"] - (entry["k"] > i)
        self.legs = self.order.index(emitter), self.order.index(spectator)
        self.gluon = born.born_pdgs[self.legs[0]] == 21
        self.configuration = 0
        if self.gluon:
            self.configuration = spin_configuration(born.provider, self.legs[0])
            if self.configuration is None:
                raise NotImplementedError(
                    f"cs_driver: no FKS configuration of {born.name} with the "
                    f"gluon {self.legs[0] + 1} as its mother"
                )


class CSRealIntegrand:
    """The real stream of the Catani-Seymour integrand of one real process of
    an aMC@NLO output, the real emission minus its dipoles, per dipole and
    Born channel of its Born (an integrand of nlo_driver, without a discrete
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
    ):
        ms = madspace()
        from madspace.fks import fks_cuts, fks_scale

        entry = block["reals"][index]
        source = subprocesses[entry["subprocess"]]
        self.source = source
        self.context = context
        self.configuration = entry["configuration"]
        self.name = f"{source.name} real {self.configuration + 1}"
        configurations = source.configurations
        real_pdgs = configurations["configurations"][self.configuration]["real_pdgs"]
        self.pdgs = list(entry["pdgs"])
        if len(self.pdgs) != len(real_pdgs):
            raise ValueError(f"cs_driver: the real {self.pdgs} is not {real_pdgs}")
        self.flavors = [list(f) for f in entry["flavors"]]
        self.incoming = source.incoming
        self.born_pdgs = source.born_pdgs
        self.energy = source.energy
        self.beams = source.beams
        leptonic = self.beams == (0, 0)
        orders = configurations["orders"]
        self.nlo_count = len(orders["nlo"])
        self.real_powers = [int(o[0]) for o in orders["nlo"]]
        self.technical_cut = technical_cut
        self.masses = [
            float(param_card["mass"].get(str(abs(pdg)), 0.0)) for pdg in self.pdgs
        ]
        n = len(self.pdgs)
        self.scale = Runtime(
            fks_scale(configurations, **scale_parameters(run_card)).with_particle_count(
                n
            ),
            context,
        )
        self.cuts = Runtime(
            fks_cuts(source.channel_mappings[0], configurations, **run_cuts(run_card)),
            context,
        )
        self.alpha_s = source.alpha_s
        self.densities = source.densities
        self.dipoles, self.channels, self.forward, self.dimensions = [], [], [], []
        mappings, kernels, symmetry = [], [], []
        for d in block["cs"]:
            if d["real"] != index:
                continue
            born = subprocesses[block["borns"][d["born"]]["subprocess"]]
            dipole = CSDipole(d, born)
            line_masses = list(np.delete(self.masses, d["i"]))
            for b, channel in enumerate(born.channels):
                topology = born_topology(line_masses, 2, channel, dipole.order)
                born_mapping = ms.PhaseSpaceMapping(
                    topology, self.energy, leptonic=leptonic
                )
                mapping = ms.CSMapping(
                    born_mapping, d["i"], d["j"], d["k"], cutoff=technical_cut
                )
                if b == 0:
                    mappings.append(mapping)
                self.channels.append((len(self.dipoles), b))
                self.forward.append(
                    ms.FunctionRuntime(mapping.forward_function(), context)
                )
                self.dimensions.append(mapping.random_dim())
            self.dipoles.append(dipole)
            kernels.append(d["kernel"])
            symmetry.append(d["symmetry"])
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
        return f"{self.name} dipole {m + 1} channel {b + 1}"

    def choice_count(self, index):
        return 1

    def luminosity(self, x1, x2, mu1, mu2):
        """The luminosity of the real over its flavor lines, at the momentum
        fractions of the real and the factorization scales mu1, mu2."""
        if self.densities is None:
            return np.full(len(x1), float(len(self.flavors)))
        f1 = self.densities[0](x1, mu1)
        f2 = self.densities[1](x2, mu2)
        c1, c2 = self.densities[0].column, self.densities[1].column
        return sum(f1[:, c1[a]] * f2[:, c2[b]] for a, b, *_ in self.flavors)

    def covered(self, index, momenta):
        """Whether real emissions are in the region of the channel index, both
        variables of its dipole at or above the technical cut."""
        keep = np.ones(len(momenta), dtype=bool)
        dipole = self.dipoles[self.channels[index][0]]
        with np.errstate(divide="ignore", invalid="ignore"):
            for variable in dipole_variables(momenta, dipole.entry):
                keep &= variable >= self.technical_cut
        return keep

    def dipole_value(self, dipole, cm, k_t, color_weight, spin_weight, born):
        """-D / g_s^2 per Born order of a dipole (points, b) at its Born in
        the partonic center-of-mass frame cm, with k_t in the same frame,
        the weights of CSDipoles and born the output of the provider."""
        provider = dipole.born.provider
        plain = provider.born_orders_of(born[0])
        colors = provider.color_correlated(cm)[:, :, dipole.legs[0], dipole.legs[1]]
        value = color_weight[:, None] * colors
        if dipole.gluon:
            perp = provider.spin_correlated(
                cm,
                k_t,
                np.full(len(cm), dipole.configuration),
                born=born,
            )
            square = -minkowski(k_t, k_t)
            ratio = np.divide(
                colors, plain, out=np.zeros_like(colors), where=plain != 0
            )
            value += (spin_weight * square)[:, None] * ratio * perp
        return value

    def evaluate(self, index, x, choice, choice_weight, trick, rng):
        """The weights of a channel at the points x of its random numbers (see
        SubprocessIntegrand.evaluate of fks_driver.py and weights)."""
        count = len(x)
        out = self.forward[index].call([np.ascontiguousarray(x)])
        momenta, x1, x2, det = [np.array(np.from_dlpack(t)) for t in out]
        x1 = np.broadcast_to(x1, count).copy()
        x2 = np.broadcast_to(x2, count).copy()
        return self.weights(index, momenta, x1, x2, det, covered=det != 0)

    def weights(self, index, momenta, x1, x2, det, covered=None):
        """The weights of the channel index at real emissions (batch, n + 1, 4)
        in the lab frame, with their momentum fractions and the Jacobian det
        of their mapping, and the parts of the histograms: the real emission
        and its weight, the Borns of the dipoles in the order of their matrix
        elements and their weights. covered says which events are in the
        region of the channel (all those of its mapping), from the variables
        of its dipole (covered) if None."""
        count = len(momenta)
        m_channel, b_channel = self.channels[index]
        valid = (det != 0) & np.isfinite(det)
        valid &= np.isfinite(momenta).all(axis=(1, 2))
        momenta = np.where(valid[:, None, None], momenta, 0.0)
        incoming = momenta[:, 0] + momenta[:, 1]
        s_hat = minkowski(incoming, incoming)
        valid &= s_hat > 0
        if covered is None:
            covered = self.covered(index, momenta)
        keep = valid & covered
        projected = self.subtraction(momenta, x1, x2)
        ungated = self.ungated(momenta, x1, x2)["color_weight"]
        rho = np.abs(ungated)
        total = rho.sum(1)
        channel_weight = np.divide(
            rho[:, m_channel], total, out=np.zeros(count), where=total > 0
        )
        # the real emission
        fks_index = np.full(count, self.configuration, dtype=np.int32)
        scales = self.scale(momenta)
        passes = self.cuts(momenta, fks_index=fks_index)["mask"] != 0
        points = np.flatnonzero(keep & passes)
        real_weight = np.zeros(count)
        if len(points):
            cm = boost_z(momenta[points], -partonic_rapidity(momenta[points], 2))
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
            )
            real_weight[points] = lum * weight
        # the dipoles, each at its Born
        d = len(self.dipoles)
        n = len(self.pdgs) - 1
        born_momenta = np.zeros((count, d, n, 4))
        dipole_weights = np.zeros((count, d))
        amplitude_weight = np.zeros(count)
        first = np.zeros(count, dtype=np.int32)
        for m, dipole in enumerate(self.dipoles):
            born = dipole.born
            me = projected["born_momenta"][:, m][:, dipole.order]
            me = np.where(valid[:, None, None], me, 0.0)
            born_momenta[:, m] = me
            color_weight = projected["color_weight"][:, m]
            spin_weight = projected["spin_weight"][:, m]
            active = keep & ((color_weight != 0) | (spin_weight != 0))
            active &= born.cuts(born.soft_momenta(me), fks_index=first)["mask"] != 0
            needed = active | (keep if m == m_channel else False)
            points = np.flatnonzero(needed)
            if not len(points):
                continue
            y = -partonic_rapidity(me[points], 2)
            cm = boost_z(me[points], y)
            evaluated = born.provider.born(
                cm, np.full(len(points), dipole.configuration)
            )
            if m == m_channel:
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
            born_scales = born.scale(me[points])
            alpha_s = born.alpha_s(born_scales["ren_scale"])["alpha_s"]
            g_s = np.sqrt(4 * math.pi * alpha_s)
            # -D, the term the real emission subtracts
            weight = sum(
                g_s ** (power + 2) * value[:, o]
                for o, power in enumerate(born.qcd_powers)
            )
            lum = self.luminosity(
                x1[points],
                x2[points],
                born_scales["fact_scale1"],
                born_scales["fact_scale2"],
            )
            dipole_weights[points, m] = lum * weight
        norm = np.where(
            keep,
            INV_GEV2_TO_PB
            * det
            * channel_weight
            * amplitude_weight
            / (2 * np.where(keep, s_hat, 1.0)),
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

    def histogram_runtime(self, histograms, jets):
        return CSRealHistograms(self, histograms, jets)

    def histogram_sums(self, runtime, result, jacobian, choice):
        return runtime(result, jacobian)

    def matrix_element_calls(self):
        return {}

    def close(self):
        pass


class CSRealHistograms:
    """The histograms of the real stream of a real process, the real emission
    and the Borns of its dipoles of each point in the same histograms
    (FKSHistograms): the jets of FKSJets on each kinematics, those of the
    Born for the dipoles, and the observables of the histograms on them, as
    fks_histograms builds them."""

    def __init__(self, integrand, histograms, jets):
        ms = madspace()
        from madspace.fks import fks_configurations, fks_histogram_jet_count

        self.context = integrand.context
        self.configuration = integrand.configuration
        self.observables = {}

        def layout(subprocess, born):
            entries = fks_configurations(
                subprocess.channel_mappings[0], subprocess.configurations
            )
            runtime = Runtime(ms.FKSJets(entries, born, **jets), self.context)
            pids = tuple(runtime.function.pids())
            if pids not in self.observables:
                self.observables[pids] = []
                for histogram in histograms:
                    options = dict(histogram["observable"])
                    options.setdefault("name", histogram.get("name", ""))
                    self.observables[pids].append(
                        Runtime(ms.Observable(list(pids), **options), self.context)
                    )
            return runtime, pids

        self.real_jets = layout(integrand.source, False)
        born_layouts = {}
        self.born_jets = []
        for dipole in integrand.dipoles:
            name = dipole.born.name
            if name not in born_layouts:
                born_layouts[name] = layout(dipole.born, True)
            self.born_jets.append(born_layouts[name])
        items = []
        for histogram in histograms:
            options = dict(histogram["observable"])
            items.append(
                ms.FKSHistograms.HistItem(
                    histogram["min"],
                    histogram["max"],
                    histogram["bin_count"],
                    fks_histogram_jet_count(options),
                )
            )
        self.binning = Runtime(
            ms.FKSHistograms(items, 1 + len(integrand.dipoles)), self.context
        )

    def kinematics(self, jets, momenta, fks_index, inputs, k):
        runtime, pids = jets
        out = runtime(momenta, fks_index=fks_index)
        inputs[f"jet_count{k}"] = out["jet_count"].astype(np.int32)
        for i, observable in enumerate(self.observables[pids]):
            value = observable(out["jet_momenta"])
            inputs[f"value{i}_{k}"] = np.asarray(next(iter(value.values()))).reshape(
                len(momenta)
            )

    def __call__(self, result, jacobian):
        momenta = result["momenta"]
        count = len(momenta)
        inputs = {"weight0": result["real_weight"] * jacobian}
        configuration = np.full(count, self.configuration, dtype=np.int32)
        self.kinematics(self.real_jets, momenta, configuration, inputs, 0)
        zero = np.zeros(count, dtype=np.int32)
        for m, jets in enumerate(self.born_jets):
            inputs[f"weight{m + 1}"] = result["dipole_weights"][:, m] * jacobian
            born = result["born_momenta"][:, m]
            self.kinematics(jets, born, zero, inputs, m + 1)
        return list(self.binning.by_name(inputs).values())


class CSRun(NLORun):
    """A fixed-order NLO run of madspace with the Catani-Seymour integrand on
    an aMC@NLO output (see the module and NLORun of nlo_driver), with the
    alpha of the four families of dipoles (ff, fi, if, ii, by name or one
    for all), kappa of the final-state gluon kernels with a massive
    spectator, the technical cut of the real stream and the streams to
    integrate (nbody, real or both)."""

    scheme = "cs"
    work_directory = "madspace_cs"

    def __init__(
        self,
        output,
        alpha=1.0,
        kappa=2.0 / 3.0,
        technical_cut=1e-6,
        streams=STREAMS,
        **options,
    ):
        super().__init__(output, **options)
        if not isinstance(alpha, dict):
            alpha = {family: alpha for family in FAMILIES}
        self.alpha = {family: float(alpha.get(family, 1.0)) for family in FAMILIES}
        self.kappa = kappa
        self.technical_cut = technical_cut
        unknown = set(streams) - set(STREAMS)
        if unknown or not streams:
            raise ValueError(f"cs_driver: streams of {STREAMS} expected, not {streams}")
        self.streams = tuple(streams)
        self.subprocesses = {}

    def make_integrands(self, directories, n_f, n_l):
        from madspace.cs import read_cs_subtraction

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
