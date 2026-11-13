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

Not covered yet: the real stream (the real emission minus the dipoles), so
a run gives the n-body part alone, decays, PDF schemes other than MSbar,
user scales and the photon cuts.
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
        self.channel_mappings = [
            ms.PhaseSpaceMapping(
                born_topology(self.masses, self.incoming, channel),
                self.energy,
                leptonic=leptonic,
            )
            for channel in self.provider.channels()
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


class CSRun(NLORun):
    """A fixed-order NLO run of madspace with the Catani-Seymour integrand on
    an aMC@NLO output (see the module and NLORun of nlo_driver), with the
    alpha of the four families of dipoles (ff, fi, if, ii, by name or one
    for all) and kappa of the final-state gluon kernels with a massive
    spectator."""

    scheme = "cs"
    work_directory = "madspace_cs"

    def __init__(self, output, alpha=1.0, kappa=2.0 / 3.0, **options):
        super().__init__(output, **options)
        if not isinstance(alpha, dict):
            alpha = {family: alpha for family in ("ff", "fi", "if", "ii")}
        self.alpha = {
            family: float(alpha.get(family, 1.0)) for family in ("ff", "fi", "if", "ii")
        }
        self.kappa = kappa

    def make_integrands(self, directories, n_f, n_l):
        return [
            CSBornIntegrand(
                directory,
                self.run_card,
                self.param_card,
                self.couplings,
                self.context,
                n_f,
                self.alpha,
                self.kappa,
            )
            for directory in directories
        ]


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
        "--histograms", help="a TOML file with [histograms] as run_card.toml"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run = CSRun(
        args.output,
        alpha=args.alpha,
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
