"""Cross sections of the FKS integrand of madspace with the Fortran test
provider (fks_fortran.py), for the comparison with those of MG5aMC
(test_data/fks_cross_section, from
fks_reference/make_fks_cross_section_reference.py).

SubprocessIntegrand integrates one subprocess of an aMC@NLO output, channel
by channel over the Born configurations of MadFKS (born_conf.inc). In a
channel the Born comes from the PhaseSpaceMapping of the topology of its
diagram and the radiation and the FKS configuration (picked uniformly) from
FKSKinematics, with the bound on sqrt(s_hat) that the cuts on the leptons
and jets of the Born imply (fks_sqrt_s_hat_min). The provider gives the
blocks at these kinematics where the cuts let a contribution through (the
real where the event passes them, the Born where the event or a
counter-event does), the virtual at the renormalization scale of the Born.
FKSCrossSection takes them with the scales, alpha_s, PDFs and cuts of the
run card, and the weight of the channel is amp2 of its Born diagram over the
sum over the Born diagrams at the Born, as in MadFKS. A VEGAS grid over the
random numbers adapts in the first iterations of each channel, the later
ones add up with their variances, and the cross section of the subprocess
is the sum over its channels. Next to the NLO cross section it gives that
of the Born (lo_weight).

The PDFs are the nodes of the NNPDF grid of MadGraph (nn23nlo) as an LHAPDF
grid and alpha_s the 2-loop running from alpha_s(m_Z) of the run, as
test_fks_couplings.py builds them. Their interpolation differs from that of
MadFKS by up to 3e-3 point by point at large x and low scales.
"""

import math
import os

import numpy as np
import torch
from fks_fortran import FortranProvider, fks_blocks
from madspace_testing import default_context
from test_fks_couplings import (
    delete_globals,
    scale_parameters,
    write_alpha_s_grid,
    write_lhapdf_grid,
)

import madspace as ms
from madspace.fks import (
    fks_configuration_mapping,
    fks_cross_section,
    fks_cuts,
    fks_scale,
    fks_splitting_kernels,
)

# the cuts of FKSCuts by their names in the run card
CUTS = (
    "jetalgo",
    "jetradius",
    "ptj",
    "etaj",
    "maxjetflavor",
    "ptl",
    "etal",
    "drll",
    "drll_sf",
    "mll",
    "mll_sf",
)


class Runtime:
    """A FunctionGenerator built once, called on numpy arrays."""

    def __init__(self, function):
        self.function = function
        self.runtime = ms.FunctionRuntime(function.function(), default_context())
        self.names = list(function.arg_types().keys())
        self.returns = list(function.return_types().keys())

    def __call__(self, *args, **named):
        device = torch.get_default_device()
        tensors = [torch.as_tensor(np.asarray(a)).to(device) for a in args]
        for name in ("flavor_index", "fks_index"):
            if named.get(name) is not None:
                index = torch.as_tensor(np.asarray(named[name]), dtype=torch.int32)
                tensors.append(index.to(device))
        batch = len(args[0])
        outputs = []
        for out in self.runtime.call(tensors):
            out = torch.as_tensor(out).cpu().numpy()
            outputs.append(np.full(batch, out) if out.ndim == 0 else out)
        return dict(zip(self.returns, outputs))

    def by_name(self, inputs, fks_index=None):
        args = [inputs[name] for name in self.names if name != "fks_index"]
        return self(*args, fks_index=fks_index)


class Vegas:
    """A separable VEGAS grid on the unit hypercube, of bin_count bins per
    dimension, adapted to the squares of the weights (alpha the damping)."""

    def __init__(self, dimension, bin_count=50, alpha=1.5):
        self.edges = np.tile(np.linspace(0.0, 1.0, bin_count + 1), (dimension, 1))
        self.bin_count, self.alpha = bin_count, alpha

    def sample(self, u):
        """The points of the grid at the uniform numbers u (n, dimension),
        their Jacobians and bins."""
        scaled = u * self.bin_count
        bins = np.minimum(scaled.astype(int), self.bin_count - 1)
        dims = np.arange(u.shape[1])
        low = self.edges[dims, bins]
        width = self.edges[dims, bins + 1] - low
        x = low + (scaled - bins) * width
        return x, np.prod(width * self.bin_count, axis=1), bins

    def adapt(self, bins, weights):
        n = self.bin_count
        for d in range(self.edges.shape[0]):
            sums = np.bincount(bins[:, d], weights=weights**2, minlength=n)
            sums = np.convolve(sums, [1, 1, 1], "same") / np.convolve(
                np.ones(n), [1, 1, 1], "same"
            )
            total = sums.sum()
            if total <= 0:
                continue
            r = sums / total
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(r > 0, ((r - 1) / np.log(r)) ** self.alpha, 0.0)
            r = np.nan_to_num(r, nan=1.0)
            if r.sum() <= 0:
                continue
            cumulative = np.concatenate([[0.0], np.cumsum(r)]) / r.sum()
            targets = np.linspace(0.0, 1.0, n + 1)
            self.edges[d] = np.interp(targets, cumulative, self.edges[d])


def born_topology(masses, nincoming, channel):
    """The Topology of the Born diagram of a channel of the provider
    (FortranProvider.channels), with the masses of the Born."""

    def line(label):
        if label < 0:
            return f"p{-label - 1}"
        if label <= nincoming:
            return f"i{label - 1}"
        return f"o{label - nincoming - 1}"

    vertices, used = [], set()
    for index, d1, d2, _, _ in channel["forest"]:
        vertices.append([line(index), line(d1), line(d2)])
        used.update([d1, d2])
    labels = list(range(1, len(masses) + 1))
    labels += [-k for k in range(1, len(channel["forest"]) + 1)]
    vertices.append([line(label) for label in labels if label not in used])
    propagators = [ms.Propagator(mass=m, width=w) for m, w in channel["props"]]
    diagram = ms.Diagram(masses[:nincoming], masses[nincoming:], propagators, vertices)
    return ms.Topology(diagram)


def born_masses(configurations, param_card):
    return [
        float(param_card["mass"].get(str(abs(pdg)), 0.0))
        for pdg in configurations["born_pdgs"]
    ]


def born_cuts(born_pdgs, run_card):
    """The cuts of the run card on the final-state jets and leptons of the
    Born as madspace Cuts, for the floor on sqrt(s_hat) they imply: ptj on
    the jets (where the Born has two QCD partons or more, as cuts.f), ptl on
    the charged leptons and mll_sf on the pairs of a lepton and its
    antilepton."""
    observable = ms.Observable
    final = born_pdgs[2:]
    items = []
    jets = [pid for pid in observable.jet_pids if pid in final]
    qcd = sum(pid in observable.jet_pids for pid in final)
    if float(run_card["ptj"]) > 0 and qcd > 1:
        cut = observable(born_pdgs, observable.obs_pt, [jets])
        items.append(ms.CutItem(cut, min=float(run_card["ptj"])))
    leptons = [pid for pid in (11, -11, 13, -13, 15, -15) if pid in final]
    if float(run_card["ptl"]) > 0 and leptons:
        cut = observable(born_pdgs, observable.obs_pt, [leptons])
        items.append(ms.CutItem(cut, min=float(run_card["ptl"])))
    for lepton in (11, 13, 15):
        if float(run_card["mll_sf"]) > 0 and {lepton, -lepton} <= set(final):
            cut = observable(born_pdgs, observable.obs_pair_mass, [[lepton, -lepton]])
            items.append(ms.CutItem(cut, min=float(run_card["mll_sf"])))
    return ms.Cuts(items) if items else None


def run_cuts(run_card):
    cuts = {name: float(run_card[name]) for name in CUTS}
    cuts["maxjetflavor"] = int(cuts["maxjetflavor"])
    cuts["jetalgo"] = int(cuts["jetalgo"])
    return cuts


class Couplings:
    """The PdfGrid of nn23nlo and the AlphaSGrid of the 2-loop running from
    alpha_s(m_Z), uploaded to the default context."""

    def __init__(self, directory, asmz, hadrons):
        os.makedirs(directory, exist_ok=True)
        alpha_s_path = os.path.join(directory, f"alpha_s_{asmz}.info")
        write_alpha_s_grid(alpha_s_path, asmz)
        self.alpha_s = ms.AlphaSGrid(alpha_s_path)
        self.pdf = None
        if hadrons:
            pdf_path = os.path.join(directory, "NNPDF23nlo_as_0119_qed_0000.dat")
            if not os.path.isfile(pdf_path):
                write_lhapdf_grid(pdf_path)
            self.pdf = ms.PdfGrid(pdf_path)
        context = default_context()
        delete_globals(context)
        if self.pdf is not None:
            self.pdf.initialize_globals(context)
        self.alpha_s.initialize_globals(context)
        self.running = ms.RunningCoupling(self.alpha_s)


class SubprocessIntegrand:
    """The FKS integrand of one subprocess (a P* directory) of an aMC@NLO
    output, per Born channel, with the run card and param card of a
    reference of test_data/fks_cross_section."""

    def __init__(self, directory, reference, couplings, virtual=True, **parameters):
        self.provider = FortranProvider(directory)
        self.configurations = self.provider.fks_configurations
        run, param = reference["run_card"], reference["param_card"]
        self.incoming = self.configurations["nincoming"]
        masses = born_masses(self.configurations, param)
        if self.incoming == 1:
            energy, leptonic = masses[0], True
        else:
            energy = 2 * math.sqrt(float(run["ebeam1"]) * float(run["ebeam2"]))
            leptonic = int(run["lpp1"]) == 0
        self.beams = (0, 0) if leptonic else (int(run["lpp1"]), int(run["lpp2"]))
        self.cuts = None
        if self.incoming == 2:
            self.cuts = born_cuts(self.configurations["born_pdgs"], run)
        self.channels = []
        for channel in self.provider.channels():
            topology = born_topology(masses, self.incoming, channel)
            self.channels.append(
                ms.PhaseSpaceMapping(
                    topology,
                    energy,
                    leptonic=leptonic,
                    sqrt_s_hat_min_condition=self.cuts is not None,
                )
            )
        born = self.channels[0]
        self.alpha = 1 / float(param["sminputs"]["1"])
        self.virtual = virtual
        scales = scale_parameters(run)
        scale = fks_scale(self.configurations, **scales)
        self.born_scale = Runtime(
            scale.with_particle_count(len(self.configurations["born_pdgs"]))
        )
        self.kernels = Runtime(fks_splitting_kernels(born, self.configurations))
        self.pass_cuts = Runtime(fks_cuts(born, self.configurations, **run_cuts(run)))
        self.cross_section = Runtime(
            fks_cross_section(
                born,
                self.configurations,
                n_l=reference["n_l"],
                cuts=run_cuts(run),
                scales=scales,
                running_coupling=couplings.running,
                pdf_grid=couplings.pdf if self.beams != (0, 0) else None,
                beams=self.beams,
                sum_flavors=True,
                n_f=reference["n_f"],
                **parameters,
            )
        )
        self.kinematics = [
            Runtime(
                ms.FKSKinematics(
                    fks_configuration_mapping(b, self.configurations, cuts=self.cuts)
                )
            )
            for b in self.channels
        ]
        self.dimension = born.random_dim() + 3
        self.configuration_count = len(self.configurations["configurations"])

    def close(self):
        self.provider.close()

    def weights(self, channel, random, fks_index):
        """The NLO and LO weights of a channel at random numbers (n, dimension)
        and FKS configurations, per unit random volume."""
        out = self.kinematics[channel](random, fks_index=fks_index)
        kernels = self.kernels(
            out["collinear_momenta"],
            out["p_i_hat"],
            out["born_momenta"],
            fks_index=fks_index,
        )
        inputs = dict(out)
        ren_scale = None
        if self.virtual:
            ren_scale = self.born_scale(out["born_momenta"])["ren_scale"]
        passes = {
            name: self.pass_cuts(out[name], fks_index=fks_index)["mask"] != 0
            for name in ("momenta", "soft_momenta", "collinear_momenta")
        }
        born_mask = passes["momenta"] | passes["soft_momenta"]
        born_mask |= passes["collinear_momenta"]
        blocks = fks_blocks(
            self.provider,
            out,
            fks_index,
            kernels["k_perp"],
            ren_scale,
            real_mask=passes["momenta"],
            born_mask=born_mask,
        )
        inputs.update(blocks)
        if not self.virtual:
            inputs["virtual"] = np.zeros_like(inputs["matrix_element"])
        count = len(fks_index)
        amp2 = inputs["amp2"]
        total = amp2.sum(1)
        weight = np.where(
            total > 0, amp2[:, channel] / np.where(total > 0, total, 1), 0
        )
        inputs.update(
            channel_weight=weight,
            configuration_weight=np.full(count, float(self.configuration_count)),
            alpha=np.full(count, self.alpha),
        )
        result = self.cross_section.by_name(inputs, fks_index=fks_index)
        return result["cross_section"], result["lo_weight"]

    def integrate(self, points, adapt=4, iterations=6, seed=4711, bins=50, k=None):
        """The NLO and LO cross sections with their errors: per channel
        adapt iterations to adapt VEGAS, then iterations of points each,
        added with their variances. With a K-factor k also the integral of
        NLO - k LO on the same points (difference), whose error is that of
        the ratio of the two."""
        rng = np.random.default_rng(seed)
        totals = np.zeros(3)
        variances = np.zeros(3)
        for channel in range(len(self.channels)):
            vegas = Vegas(self.dimension, bins)
            estimates = []
            for iteration in range(adapt + iterations):
                u = rng.random((points, self.dimension))
                x, jacobian, grid_bins = vegas.sample(u)
                fks_index = rng.integers(self.configuration_count, size=points)
                nlo, lo = self.weights(channel, x, fks_index)
                nlo, lo = nlo * jacobian, lo * jacobian
                if iteration < adapt:
                    vegas.adapt(grid_bins, nlo)
                    continue
                difference = nlo - (k or 0.0) * lo
                estimates.append(
                    [(w.mean(), w.var(ddof=1) / points) for w in (nlo, lo, difference)]
                )
            values = np.array(estimates)
            totals += values[:, :, 0].mean(0)
            variances += values[:, :, 1].sum(0) / iterations**2
        errors = np.sqrt(variances)
        result = dict(
            nlo=dict(sigma=totals[0], error=errors[0]),
            lo=dict(sigma=totals[1], error=errors[1]),
        )
        if k is not None:
            result["difference"] = dict(sigma=totals[2], error=errors[2])
        return result
