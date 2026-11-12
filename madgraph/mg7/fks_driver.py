"""The fixed-order NLO run of madspace on an aMC@NLO output: the FKS
integrand of madspace (FKSKinematics, FKSCrossSection) with the matrix
elements of the Fortran provider (fks_provider.py) and the PDFs, alpha_s,
scales and cuts of the run card, integrated per subprocess and per Born
channel of MadFKS, with its histograms in the info.json of the run, which
MadBoard reads.

    python -m madgraph.mg7.fks_driver OUTPUT [--name RUN] [--accuracy ACC]

The output is an aMC@NLO output of a process with [QCD] whose libraries are
built, as a fixed-order run of MG5aMC (or its compile FO) leaves them. The
integration follows the fixed-order run of MadFKS (driver_mintFO.f with
mint):

- A channel is a Born configuration of a subprocess (born_conf.inc, its
  Born diagram as a madspace Topology) with the multichannel weight of
  MadFKS, amp2 of its diagram over that of all Born diagrams at the Born.
- The Born of a channel comes from the PhaseSpaceMapping of its topology and
  the radiation from FKSKinematics, with the bound on sqrt(s_hat) the cuts
  on the Born imply, under a VEGAS grid over their random numbers. The FKS
  configuration comes from an adaptive choice (as MC_integer), its
  probability proportional to the root mean square of its integrand.
- The matrix elements come from the provider where the cuts let a
  contribution through. The virtual (MadLoop) is evaluated on a fraction of
  the points of a channel, with the average of V / B per NLO order times
  the Born on all of them (the virtual trick, types 14 and 15 of MadFKS),
  the fraction updated as update_virtual_fraction of mint.
- Every iteration adapts the grids, the choice of the configurations, the
  fraction and the averages to its points, as mint does. After the grid
  iterations the integration iterations add up their points, each
  iteration with its own variance, the points of an iteration shared among
  the channels by their standard deviations. Towards req_acc_FO the grid
  iterations double their points and each integration iteration has the
  points the errors so far imply are missing (at most four times those
  before). Where req_acc_FO is -1 the run takes niters_FO iterations of
  npoints_FO points per channel.

The run writes Events/RUN/info.json as the LO generator of madspace does
(status, process and channels with mean and error, histograms, run_times),
with the LO cross section, the cross sections per subprocess and the calls
of the matrix elements in "fks", and the histograms as HwU in
MADatNLO_madspace.HwU. The histograms are those of --histograms (a TOML file
with [histograms] and [multiparticles] as the run_card.toml of MG7, which
can be one) or, by default, those MG7 writes for the final state (the pt
and eta of each object, the masses of pairs, sqrt_s).

Not covered yet: the resonance-local recoil of a final-state emitter inside
a resonance of the Born, PDF schemes other than MSbar, lepton beams with
densities, user scales, FxFx and the photon cuts.
"""

import argparse
import json
import logging
import math
import os
import time

import numpy as np

from madgraph.mg7 import fks_couplings
from madgraph.mg7.fks_provider import FortranProvider, fks_blocks

logger = logging.getLogger("madgraph.mg7.fks_driver")

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
# the kinematics of FKSObservableHistograms and their weights
HISTOGRAM_KINEMATICS = (
    ("momenta", "event_weight"),
    ("soft_momenta", "soft_weight"),
    ("collinear_momenta", "collinear_weight"),
    ("soft_collinear_momenta", "soft_collinear_weight"),
    ("born_momenta", "nbody_weight"),
)
# the [multiparticles] names of MG7 of particles by |pdg|, the observables
# telling no particle from its antiparticle
GROUPS = {
    5: "b",
    6: "t",
    11: "e",
    12: "ve",
    13: "mu",
    14: "vm",
    15: "ta",
    16: "vt",
    22: "a",
    23: "z",
    24: "w",
    25: "h",
}


def madspace():
    """madspace, imported where it is used, so that the module loads (and
    --help answers) without it."""
    import madspace as ms

    return ms


class Runtime:
    """A FunctionGenerator of madspace built once and called on numpy
    arrays, its outputs by name."""

    def __init__(self, function, context):
        ms = madspace()
        self.function = function
        self.runtime = ms.FunctionRuntime(function.function(), context)
        self.names = list(function.arg_types().keys())
        self.returns = list(function.return_types().keys())

    def __call__(self, *args, fks_index=None):
        tensors = [np.ascontiguousarray(a) for a in args]
        if fks_index is not None:
            tensors.append(np.ascontiguousarray(fks_index, dtype=np.int32))
        batch = len(args[0])
        outputs = []
        for out in self.runtime.call(tensors):
            out = np.from_dlpack(out)
            outputs.append(np.full(batch, out) if out.ndim == 0 else np.array(out))
        return dict(zip(self.returns, outputs))

    def by_name(self, inputs, fks_index=None):
        args = [inputs[name] for name in self.names if name != "fks_index"]
        return self(*args, fks_index=fks_index)


class Vegas:
    """A separable VEGAS grid on the unit hypercube with bin_count bins per
    dimension, adapted to the squares of the weights (alpha the damping)."""

    def __init__(self, dimension, bin_count=50, alpha=1.5):
        self.edges = np.tile(np.linspace(0.0, 1.0, bin_count + 1), (dimension, 1))
        self.bin_count, self.alpha = bin_count, alpha
        self.sums = np.zeros((dimension, bin_count))

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

    def add(self, bins, weights):
        for d in range(self.edges.shape[0]):
            self.sums[d] += np.bincount(
                bins[:, d], weights=weights**2, minlength=self.bin_count
            )

    def adapt(self):
        n = self.bin_count
        for d in range(self.edges.shape[0]):
            sums = np.convolve(self.sums[d], [1, 1, 1], "same")
            sums /= np.convolve(np.ones(n), [1, 1, 1], "same")
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
        self.sums[:] = 0.0


class DiscreteChoice:
    """The adaptive choice of one of count options (the FKS configuration, as
    MC_integer of MadFKS): probabilities proportional to the root mean
    square of the integrand of each option (the weight times the
    probability of its choice), which minimizes the variance, mixed with
    the uniform choice by floor, so that every option keeps being
    sampled."""

    def __init__(self, count, floor=0.1):
        self.probabilities = np.full(count, 1.0 / count)
        self.floor = floor
        self.squares = np.zeros(count)
        self.counts = np.zeros(count)

    def sample(self, rng, points):
        index = rng.choice(len(self.probabilities), size=points, p=self.probabilities)
        return index, 1.0 / self.probabilities[index]

    def add(self, index, weights):
        n = len(self.probabilities)
        self.squares += np.bincount(index, weights=weights**2, minlength=n)
        self.counts += np.bincount(index, minlength=n)

    def adapt(self):
        seen = self.counts > 0
        if not seen.any():
            return
        target = np.zeros_like(self.squares)
        target[seen] = np.sqrt(self.squares[seen] / self.counts[seen])
        if target.sum() > 0:
            target /= target.sum()
            uniform = np.full_like(target, 1.0 / len(target))
            self.probabilities = (1 - self.floor) * target + self.floor * uniform
        self.squares[:] = 0.0
        self.counts[:] = 0.0


class VirtualTrick:
    """The virtual on a fraction of the points of a channel (the virtual
    trick of MadFKS): where it is evaluated (V - r B) / f, plus r B
    everywhere, with r the average of V / B per NLO order and B the sum of
    the Born orders."""

    def __init__(self, orders, fraction, min_fraction):
        self.ratios = np.zeros(orders)
        self.fraction = fraction
        self.min_fraction = min_fraction
        self.virtual_sums = np.zeros(orders)
        self.born_sum = 0.0

    def inputs(self, virtual, born, evaluated):
        approximation = self.ratios[None, :] * born.sum(1)[:, None]
        residual = np.where(
            evaluated[:, None], (virtual - approximation) / self.fraction, 0.0
        )
        return approximation + residual, approximation

    def add(self, virtual, born, evaluated):
        self.virtual_sums += virtual[evaluated].sum(0)
        self.born_sum += born[evaluated].sum()

    def adapt(self, error_virtual, error_total):
        if self.born_sum != 0:
            self.ratios = self.virtual_sums / self.born_sum
        self.virtual_sums[:] = 0.0
        self.born_sum = 0.0
        if error_total > 0:
            factor = max(min(2.0 * error_virtual / error_total, 2.0), 0.25)
            self.fraction = max(min(self.fraction * factor, 1.0), self.min_fraction)


def born_topology(masses, nincoming, channel):
    """The Topology of the Born diagram of a channel of the provider
    (FortranProvider.channels), with the masses of the Born."""
    ms = madspace()

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


def born_cuts(born_pdgs, run_card):
    """The cuts of the run card on the final-state jets and leptons of the
    Born as madspace Cuts, for the bound on sqrt(s_hat) they imply: ptj on
    the jets (where the Born has two QCD partons or more, as cuts.f), ptl on
    the charged leptons and mll_sf on the pairs of a lepton and its
    antilepton. None without such cuts."""
    ms = madspace()
    observable = ms.Observable
    final = list(born_pdgs[2:])
    items = []
    jets = [pid for pid in observable.jet_pids if pid in final]
    qcd = sum(pid in observable.jet_pids for pid in final)
    if real(run_card["ptj"]) > 0 and qcd > 1:
        cut = observable(born_pdgs, observable.obs_pt, [jets])
        items.append(ms.CutItem(cut, min=real(run_card["ptj"])))
    leptons = [pid for pid in (11, -11, 13, -13, 15, -15) if pid in final]
    if real(run_card["ptl"]) > 0 and leptons:
        cut = observable(born_pdgs, observable.obs_pt, [leptons])
        items.append(ms.CutItem(cut, min=real(run_card["ptl"])))
    for lepton in (11, 13, 15):
        if real(run_card["mll_sf"]) > 0 and {lepton, -lepton} <= set(final):
            cut = observable(born_pdgs, observable.obs_pair_mass, [[lepton, -lepton]])
            items.append(ms.CutItem(cut, min=real(run_card["mll_sf"])))
    return ms.Cuts(items) if items else None


def first(value):
    """The first entry of a list of the run card (lhaid,
    dynamical_scale_choice), the value itself otherwise."""
    if isinstance(value, str) and "," in value:
        value = value.split(",")
    return value[0] if isinstance(value, (list, tuple)) else value


def flag(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", ".true.", "t", "1")
    return bool(value)


def integer(value):
    return int(float(first(value)))


def real(value):
    return float(first(value))


def run_cuts(run_card):
    """The cuts of FKSCuts by their names, from the run card."""
    cuts = {name: real(run_card[name]) for name in CUTS}
    cuts["maxjetflavor"] = integer(cuts["maxjetflavor"])
    cuts["jetalgo"] = integer(cuts["jetalgo"])
    return cuts


def scale_parameters(run_card):
    """The scales of the run card by the names of fks_scale."""
    return dict(
        dynamical_scale_choice=integer(run_card["dynamical_scale_choice"]),
        fixed_ren_scale=flag(run_card["fixed_ren_scale"]),
        fixed_fac_scale=flag(run_card["fixed_fac_scale"]),
        mur_ref_fixed=real(run_card["mur_ref_fixed"]),
        muf_ref_fixed=real(run_card["muf_ref_fixed"]),
        mur_over_ref=real(run_card["mur_over_ref"]),
        muf_over_ref=real(run_card["muf_over_ref"]),
    )


def read_param_card(path):
    """The blocks of a param card (name -> key -> value, the widths under
    decay)."""
    blocks, block = {}, None
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            fields = line.split()
            if fields[0].lower() == "block":
                block = fields[1].lower()
                blocks[block] = {}
            elif fields[0].lower() == "decay":
                blocks.setdefault("decay", {})[fields[1]] = float(fields[2])
                block = None
            elif block:
                blocks[block][" ".join(fields[:-1])] = float(fields[-1])
    return blocks


def read_fks_params(path):
    """VirtualFraction and MinVirtualFraction of FKS_params.dat."""
    values = dict(VirtualFraction=1.0, MinVirtualFraction=0.005)
    if not os.path.isfile(path):
        return values
    with open(path) as f:
        lines = [line.strip() for line in f]
    for k, line in enumerate(lines[:-1]):
        key = line.lstrip("#")
        if line.startswith("#") and key in values:
            values[key] = float(lines[k + 1].lower().replace("d", "e"))
    return values


def light_flavors(output):
    """NF and NL of coupl.inc, the light quarks and massless leptons of the
    model."""
    values = {}
    with open(os.path.join(output, "Source", "coupl.inc")) as f:
        for line in f:
            for key in ("NF", "NL"):
                head = f"PARAMETER({key}="
                if head in line.replace(" ", ""):
                    text = line.replace(" ", "").split(head)[1].split(")")[0]
                    values[key.lower()] = float(text.lower().replace("d", "e"))
    return values["nf"], values["nl"]


def read_run_card(output, values=None):
    """The run card of an aMC@NLO output by lower-case names, updated by
    values, with maxjetflavor of Source/run_card.inc where the card hides
    it and values do not set it."""
    from madgraph.various.banner import RunCardNLO

    card = RunCardNLO(os.path.join(output, "Cards", "run_card.dat"))
    run_card = {key.lower(): card[key] for key in card}
    path = os.path.join(output, "Source", "run_card.inc")
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                if line.split("=")[0].strip().upper() == "MAXJETFLAVOR":
                    run_card["maxjetflavor"] = int(line.split("=")[1].strip())
    run_card.update({key.lower(): value for key, value in (values or {}).items()})
    return run_card


def check_run_card(run_card):
    """Raises NotImplementedError for the settings the driver does not
    cover."""
    if integer(run_card.get("ickkw", 0)) != 0:
        raise NotImplementedError("fks_driver: no FxFx merging (ickkw)")
    if integer(run_card["dynamical_scale_choice"]) in (0, 10):
        raise NotImplementedError("fks_driver: no user scales")
    beams = (integer(run_card["lpp1"]), integer(run_card["lpp2"]))
    if any(abs(beam) > 1 for beam in beams):
        raise NotImplementedError(f"fks_driver: beams {beams}")
    if integer(run_card.get("pdfscheme", 0)) != 0:
        raise NotImplementedError("fks_driver: only the MSbar PDF scheme")


class Couplings:
    """The PdfGrid and AlphaSGrid of the run, uploaded to the context: those
    of the LHAPDF set of lhaid, or the nodes of the grid of nn23nlo with the
    2-loop running of alpha_s. Lepton beams take alpha_s(m_Z) of the param
    card."""

    def __init__(self, directory, run_card, param_card, context):
        ms = madspace()
        os.makedirs(directory, exist_ok=True)
        hadrons = integer(run_card["lpp1"]) != 0
        self.pdf = None
        pdlabel = str(run_card["pdlabel"]).lower()
        if hadrons and pdlabel == "lhapdf":
            lhaid = integer(run_card["lhaid"])
            grid, info = fks_couplings.lhapdf_set(lhaid)
            self.pdf = ms.PdfGrid(grid)
            self.alpha_s = ms.AlphaSGrid(info)
            self.asmz = fks_couplings.lhapdf_alpha_s_mz(info)
        else:
            if hadrons and pdlabel != "nn23nlo":
                raise NotImplementedError(f"fks_driver: pdlabel {pdlabel}")
            self.asmz = (
                fks_couplings.NNPDF_ALPHA_S
                if hadrons
                else float(param_card["sminputs"]["3"])
            )
            path = os.path.join(directory, f"alpha_s_{self.asmz}.info")
            fks_couplings.write_alpha_s_grid(path, self.asmz)
            self.alpha_s = ms.AlphaSGrid(path)
            if hadrons:
                grid = os.path.join(directory, "NNPDF23nlo_as_0119_qed_0000.dat")
                if not os.path.isfile(grid):
                    fks_couplings.write_lhapdf_grid(grid)
                self.pdf = ms.PdfGrid(grid)
        for name in (
            "pdf_logx",
            "pdf_logq2",
            "pdf_coefficients",
            "alpha_s_logq2",
            "alpha_s_coefficients",
        ):
            if context.global_exists(name):
                context.delete_global(name)
        if self.pdf is not None:
            self.pdf.initialize_globals(context)
        self.alpha_s.initialize_globals(context)
        self.running = ms.RunningCoupling(self.alpha_s)


class SubprocessIntegrand:
    """The FKS integrand of one subprocess (a P* directory) of an aMC@NLO
    output, per Born channel of MadFKS."""

    def __init__(self, directory, run_card, param_card, couplings, context, n_f, n_l):
        ms = madspace()
        from madspace.fks import (
            fks_configuration_mapping,
            fks_cross_section,
            fks_cuts,
            fks_minimal_masses,
            fks_resonances,
            fks_scale,
            fks_splitting_kernels,
        )

        self.name = os.path.basename(os.path.normpath(directory))
        self.provider = FortranProvider(directory)
        self.configurations = self.provider.fks_configurations
        self.incoming = self.configurations["nincoming"]
        born_pdgs = self.configurations["born_pdgs"]
        masses = [
            float(param_card["mass"].get(str(abs(pdg)), 0.0)) for pdg in born_pdgs
        ]
        if self.incoming == 1:
            energy, leptonic = masses[0], True
        else:
            energy = 2 * math.sqrt(real(run_card["ebeam1"]) * real(run_card["ebeam2"]))
            leptonic = integer(run_card["lpp1"]) == 0
        beams = (integer(run_card["lpp1"]), integer(run_card["lpp2"]))
        self.beams = (0, 0) if leptonic else beams
        real_pdgs = {
            p for c in self.configurations["configurations"] for p in c["real_pdgs"]
        }
        if 22 in real_pdgs and real(run_card["ptgmin"]) > 0:
            raise NotImplementedError("fks_driver: no isolated photons (ptgmin)")
        self.cuts = born_cuts(born_pdgs, run_card) if self.incoming == 2 else None
        minimal = fks_minimal_masses(
            self.configurations,
            masses,
            **{key: real(run_card[key]) for key in ("ptj", "ptl", "mll", "mll_sf")},
            maxjetflavor=integer(run_card["maxjetflavor"]),
        )
        self.channel_mappings = []
        for channel in self.provider.channels():
            topology = born_topology(masses, self.incoming, channel)
            resonances = fks_resonances(topology, self.configurations, minimal)
            if any(members for entry in resonances for members in entry):
                raise NotImplementedError(
                    f"fks_driver: {self.name} has a resonance-local recoil"
                )
            self.channel_mappings.append(
                ms.PhaseSpaceMapping(
                    topology,
                    energy,
                    leptonic=leptonic,
                    sqrt_s_hat_min_condition=self.cuts is not None,
                )
            )
        born = self.channel_mappings[0]
        self.alpha = 1 / float(param_card["sminputs"]["1"])
        scales = scale_parameters(run_card)
        scale = fks_scale(self.configurations, **scales)
        self.born_scale = Runtime(scale.with_particle_count(len(born_pdgs)), context)
        self.kernels = Runtime(
            fks_splitting_kernels(born, self.configurations), context
        )
        cuts = run_cuts(run_card)
        self.pass_cuts = Runtime(fks_cuts(born, self.configurations, **cuts), context)
        self.cross_section = Runtime(
            fks_cross_section(
                born,
                self.configurations,
                n_l=n_l,
                cuts=cuts,
                scales=scales,
                running_coupling=couplings.running,
                pdf_grid=couplings.pdf if self.beams != (0, 0) else None,
                beams=self.beams,
                sum_flavors=True,
                n_f=n_f,
            ),
            context,
        )
        self.kinematics = [
            Runtime(
                ms.FKSKinematics(
                    fks_configuration_mapping(b, self.configurations, cuts=self.cuts)
                ),
                context,
            )
            for b in self.channel_mappings
        ]
        self.dimension = born.random_dim() + 3
        self.configuration_count = len(self.configurations["configurations"])
        self.nlo_count = len(self.configurations["orders"]["nlo"])
        self.born = born

    def close(self):
        self.provider.close()

    def evaluate(self, channel, x, fks_index, configuration_weight, trick, rng):
        """The weights of a channel at the points x (n, dimension) of its
        random numbers, the FKS configurations and the weights of their
        choice, with the virtual on the fraction of the points of trick:
        the NLO weight per point, its LO part and its virtual residual (that
        of the points with the virtual), the outputs of FKSKinematics and
        FKSCrossSection by name, the virtual and the Born of the points with
        the virtual."""
        out = self.kinematics[channel](x, fks_index=fks_index)
        kernels = self.kernels(
            out["collinear_momenta"],
            out["p_i_hat"],
            out["born_momenta"],
            fks_index=fks_index,
        )
        ren_scale = self.born_scale(out["born_momenta"])["ren_scale"]
        passes = {
            name: self.pass_cuts(out[name], fks_index=fks_index)["mask"] != 0
            for name in ("momenta", "soft_momenta", "collinear_momenta")
        }
        born_mask = passes["momenta"] | passes["soft_momenta"]
        born_mask |= passes["collinear_momenta"]
        count = len(fks_index)
        # the n-body part, the virtual with it, passes the cuts where the soft
        # counter-event does
        evaluated = passes["soft_momenta"] & (out["born_weight"] != 0)
        evaluated &= rng.random(count) < trick.fraction
        blocks = fks_blocks(
            self.provider,
            out,
            fks_index,
            kernels["k_perp"],
            ren_scale,
            real_mask=passes["momenta"],
            born_mask=born_mask,
            virtual_mask=evaluated,
        )
        amp2 = blocks["amp2"]
        total = amp2.sum(1)
        channel_weight = np.where(
            total > 0, amp2[:, channel] / np.where(total > 0, total, 1), 0.0
        )
        virtual, approximation = trick.inputs(
            blocks["virtual"], blocks["born"], evaluated
        )
        inputs = dict(out)
        inputs.update(blocks)
        inputs.update(
            virtual=virtual,
            channel_weight=channel_weight,
            configuration_weight=configuration_weight,
            alpha=np.full(count, self.alpha),
        )
        result = self.cross_section.by_name(inputs, fks_index=fks_index)
        inputs["virtual"] = approximation
        without = self.cross_section.by_name(inputs, fks_index=fks_index)
        residual = result["cross_section"] - without["cross_section"]
        return dict(
            nlo=result["cross_section"],
            lo=result["lo_weight"],
            residual=residual,
            outputs=result,
            kinematics=out,
            virtual=blocks["virtual"],
            born=blocks["born"],
            evaluated=evaluated,
        )


class Channel:
    """The state of one channel of one subprocess: its VEGAS grid, the
    choice of its FKS configurations, its virtual trick and the sums of its
    integration iterations, which add up the points of all of them, each
    iteration with its own variance."""

    def __init__(self, integrand, index, fks_params, bins):
        self.integrand, self.index = integrand, index
        self.vegas = Vegas(integrand.dimension, bins)
        self.choice = DiscreteChoice(integrand.configuration_count)
        self.trick = VirtualTrick(
            integrand.nlo_count,
            fks_params["VirtualFraction"],
            fks_params["MinVirtualFraction"],
        )
        self.reset()
        self.std = None

    def reset(self):
        self.count = 0
        # the sums of the NLO weight, its LO part and its virtual residual,
        # and the sums over the iterations of their products about the means
        # of each iteration
        self.sums = np.zeros(3)
        self.covariance = np.zeros((3, 3))
        self.histograms = None

    @property
    def name(self):
        return f"{self.integrand.name} channel {self.index + 1}"

    def run(self, points, rng, keep, histograms, batch):
        """An iteration of points points of the channel, after which its
        VEGAS grid, the choice of its configurations and its virtual trick
        adapt to them, as mint does after each iteration. With keep the
        points add to the estimates of the channel, with the variance of
        this iteration, and to its histograms."""
        weights, residuals = [], []
        count, sums, products = 0, np.zeros(3), np.zeros((3, 3))
        for start in range(0, points, batch):
            n = min(batch, points - start)
            u = rng.random((n, self.integrand.dimension))
            x, jacobian, bins = self.vegas.sample(u)
            fks_index, configuration_weight = self.choice.sample(rng, n)
            result = self.integrand.evaluate(
                self.index, x, fks_index, configuration_weight, self.trick, rng
            )
            nlo = result["nlo"] * jacobian
            lo = result["lo"] * jacobian
            residual = result["residual"] * jacobian
            weights.append(nlo)
            residuals.append(residual)
            self.trick.add(result["virtual"], result["born"], result["evaluated"])
            self.vegas.add(bins, nlo)
            self.choice.add(fks_index, nlo / configuration_weight)
            if not keep:
                continue
            values = np.stack([nlo, lo, residual])
            count += n
            sums += values.sum(1)
            products += values @ values.T
            if histograms is not None:
                outputs = dict(result["outputs"])
                for name in [w for _, w in HISTOGRAM_KINEMATICS]:
                    outputs[name] = outputs[name] * jacobian
                sums_of = histograms.by_name(
                    {**result["kinematics"], **outputs},
                    fks_index=fks_index,
                )
                sums_of = [np.asarray(v).reshape(-1) for v in sums_of.values()]
                if self.histograms is None:
                    self.histograms = [np.zeros_like(v) for v in sums_of]
                for total, value in zip(self.histograms, sums_of):
                    total += value
        weights = np.concatenate(weights)
        residuals = np.concatenate(residuals)
        error_total = weights.std() / math.sqrt(len(weights))
        error_virtual = residuals.std() / math.sqrt(len(residuals))
        self.vegas.adapt()
        self.choice.adapt()
        self.trick.adapt(error_virtual, error_total)
        if count:
            self.count += count
            self.sums += sums
            self.covariance += products - np.outer(sums, sums) / count
        self.std = weights.std()

    def result(self, k):
        """The estimate and its error of the k-th sum (NLO, LO, residual)."""
        if self.count == 0:
            return 0.0, 0.0
        coefficients = np.zeros(3)
        coefficients[k] = 1.0
        return self.combination(coefficients)

    def combination(self, coefficients):
        """The estimate and its error of a linear combination of the NLO
        weight, its LO part and its virtual residual on the same points."""
        if self.count == 0:
            return 0.0, 0.0
        c = np.asarray(coefficients, dtype=float)
        variance = max(c @ self.covariance @ c, 0.0) / self.count**2
        return c @ self.sums / self.count, math.sqrt(variance)


def parse_observable(name, order_observable, multiparticles):
    """The keyword arguments of the Observable of madspace of a histogram of
    the run_card.toml of MG7 by its name, as MadgraphProcess.parse_observable
    reads it: [multiparticles] groups (with _k for the k-th hardest of a
    group by order_observable) joined by "-", then "sum" for the observable
    of the summed momenta (before the observable) or the sum of the
    observable (after it), then the observable."""
    parts = name.split("-")
    options = {"name": name, "ignore_incoming": True}
    if len(parts) == 1:
        return {**options, "observable": parts[0], "select_pids": []}
    if parts[-1] == "sum":
        options["sum_observable"] = True
        observable, selection = parts[-2], parts[:-2]
    elif parts[-2] == "sum":
        options["sum_momenta"] = True
        observable, selection = parts[-1], parts[:-2]
    else:
        observable, selection = parts[-1], parts[:-1]
    select, order = [], []
    for group in selection:
        pieces = group.split("_")
        if pieces[-1].isnumeric():
            order.append(int(pieces[-1]))
            select.append(list(multiparticles["_".join(pieces[:-1])]))
        else:
            order.append(0)
            select.append(list(multiparticles[group]))
    ordered = any(order)
    return {
        **options,
        "observable": observable,
        "select_pids": select,
        "order_observable": order_observable if ordered else None,
        "order_indices": order if ordered else [],
    }


def read_histograms(path):
    """The histograms of a TOML file in the format of the run_card.toml of
    MG7: [histograms] by name (see parse_observable) with min, max and
    bin_count, and order_by, over the groups of [multiparticles]. The
    histogram of the event weight (weight) has no meaning at fixed order and
    is left out."""
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    entries = dict(data.get("histograms", {}))
    order = entries.pop("order_by", "pt")
    multiparticles = data.get("multiparticles", {})
    return [
        {
            "name": name,
            "min": float(entry["min"]),
            "max": float(entry["max"]),
            "bin_count": int(entry["bin_count"]),
            "observable": parse_observable(name, order, multiparticles),
        }
        for name, entry in entries.items()
        if name != "weight"
    ]


def default_histograms(configurations, energy, leptonic, decay, maxjetflavor):
    """The histograms that MG7 writes by default for the final state of the
    Born (RunCardMG7.set_default_histograms): the pt and eta of every
    final-state object, the jets (the gluon and the quarks up to
    maxjetflavor) ordered by pt and the other particles grouped by |pdg|,
    the invariant mass of every pair of them (at most ten) and sqrt_s (not
    for a decay), with the ranges of MG7 for the energy of the collision."""
    from madgraph.various.banner import RunCardMG7

    final = configurations["born_pdgs"][configurations["nincoming"] :]
    jet = [21] + [q for k in range(1, maxjetflavor + 1) for q in (k, -k)]
    groups, multiparticles = {}, {"jet": jet}
    for pdg in final:
        if pdg in jet:
            group = "jet"
        else:
            group = GROUPS.get(abs(pdg), f"pdg{abs(pdg)}")
            multiparticles[group] = [abs(pdg), -abs(pdg)]
        groups[group] = groups.get(group, 0) + 1
    slots = []
    for group, count in groups.items():
        count = min(count, RunCardMG7.max_histogram_slots)
        slots += [group] if count == 1 else [f"{group}_{k + 1}" for k in range(count)]
    rounded = RunCardMG7._round_scale
    if decay:
        pt_max, mass_max = rounded(energy / 2, True), rounded(energy, True)
    elif leptonic:
        pt_max, mass_max = rounded(energy / 2, False), rounded(energy, True)
    else:
        pt_max, mass_max = rounded(energy / 20, False), rounded(energy / 10, True)
    bins, eta = RunCardMG7.histogram_bin_count, RunCardMG7.histogram_eta_max
    ranges = {f"{slot}-pt": (0.0, pt_max) for slot in slots}
    ranges.update({f"{slot}-eta": (-eta, eta) for slot in slots})
    pairs = [f"{a}-{b}-pair_mass" for k, a in enumerate(slots) for b in slots[k + 1 :]]
    pairs = pairs[: RunCardMG7.max_histogram_pairs]
    ranges.update({name: (0.0, mass_max) for name in pairs})
    if not decay:
        ranges["sqrt_s"] = (0.0, mass_max)
    return [
        {
            "name": name,
            "min": low,
            "max": high,
            "bin_count": bins,
            "observable": parse_observable(name, "pt", multiparticles),
        }
        for name, (low, high) in ranges.items()
    ]


class FKSRun:
    """A fixed-order NLO run of madspace on an aMC@NLO output (see the
    module).

    Args:
        output: the aMC@NLO output.
        name: the name of the run, its directory in Events.
        accuracy: req_acc_FO, that of the run card by default. Zero or
            negative runs grid_iterations of grid_points and max_iterations
            of points per channel, from the run card by default (its
            npoints_FO_grid, niters_FO_grid, npoints_FO and niters_FO).
        grid_points, grid_iterations, points, max_iterations: the iterations
            of the run. Towards an accuracy the grid iterations double their
            points from grid_points on, as the first step of mint does, and
            points is the least number of points per channel of an
            iteration, 2000, 4, 4000 and 50 by default.
        seed: the seed, iseed of the run card by default, drawn where it is
            zero.
        histograms: a TOML file with the histograms (see read_histograms),
            those of default_histograms without.
        batch: the points of one call of the integrand.
        bins: the bins per dimension of the VEGAS grids.
        run_card, param_card: values in place of those of the cards of the
            output (run_card by name, param_card as read_param_card gives
            it).
        run_path: the directory of info.json and the HwU file, the run in
            Events by default.
    """

    def __init__(
        self,
        output,
        name="madspace_fo",
        accuracy=None,
        grid_points=None,
        grid_iterations=None,
        points=None,
        max_iterations=None,
        seed=None,
        histograms=None,
        batch=2000,
        bins=50,
        run_card=None,
        param_card=None,
        run_path=None,
    ):
        self.output = os.path.realpath(output)
        self.run_card = read_run_card(self.output, run_card)
        check_run_card(self.run_card)
        self.param_card = param_card or read_param_card(
            os.path.join(self.output, "Cards", "param_card.dat")
        )
        self.fks_params = read_fks_params(
            os.path.join(self.output, "Cards", "FKS_params.dat")
        )
        card = self.run_card
        accuracy = real(card["req_acc_fo"]) if accuracy is None else accuracy
        self.accuracy = accuracy
        fixed_points = accuracy <= 0
        self.grid_points = grid_points or (
            integer(card["npoints_fo_grid"]) if fixed_points else 2000
        )
        self.grid_iterations = grid_iterations or (
            integer(card["niters_fo_grid"]) if fixed_points else 4
        )
        self.points = points or (integer(card["npoints_fo"]) if fixed_points else 4000)
        self.max_iterations = max_iterations or (
            integer(card["niters_fo"]) if fixed_points else 50
        )
        self.fixed_points = fixed_points
        self.seed = integer(card.get("iseed", 0)) if seed is None else seed
        if not self.seed:
            # iseed 0 draws a seed, which info.json keeps
            self.seed = int(np.random.SeedSequence().generate_state(1)[0]) % 2**31 + 1
        self.rng = np.random.default_rng(self.seed)
        self.batch, self.bins = batch, bins
        self.name = name
        self.run_path = run_path or os.path.join(self.output, "Events", name)
        os.makedirs(self.run_path, exist_ok=True)
        self.histogram_file = histograms
        self.iterations = 0
        self.start = time.time()
        self.cpu_start = time.process_time()

    def setup(self):
        ms = madspace()
        # a context of its own, so that its globals (the PDF and alpha_s
        # grids) stay out of the default context of madspace
        self.context = ms.Context(ms.cpu_device())
        self.couplings = Couplings(
            os.path.join(self.output, "SubProcesses", "madspace_fks"),
            self.run_card,
            self.param_card,
            self.context,
        )
        n_f, n_l = light_flavors(self.output)
        root = os.path.join(self.output, "SubProcesses")
        names = sorted(
            d
            for d in os.listdir(root)
            if d.startswith("P") and os.path.isdir(os.path.join(root, d))
        )
        self.integrands = [
            SubprocessIntegrand(
                os.path.join(root, name),
                self.run_card,
                self.param_card,
                self.couplings,
                self.context,
                n_f,
                n_l,
            )
            for name in names
        ]
        self.channels = [
            Channel(integrand, k, self.fks_params, self.bins)
            for integrand in self.integrands
            for k in range(len(integrand.channel_mappings))
        ]
        leading = self.integrands[0]
        energy = (
            leading.channel_mappings[0].external_masses()[0]
            if leading.incoming == 1
            else 2
            * math.sqrt(real(self.run_card["ebeam1"]) * real(self.run_card["ebeam2"]))
        )
        if self.histogram_file:
            self.histograms = read_histograms(self.histogram_file)
        else:
            self.histograms = default_histograms(
                leading.configurations,
                energy,
                leading.beams == (0, 0),
                leading.incoming == 1,
                integer(self.run_card["maxjetflavor"]),
            )
        self.histogram_runtimes = {}
        if self.histograms:
            from madspace.fks import fks_histograms

            jets = {
                key: real(self.run_card[key])
                for key in ("jetalgo", "jetradius", "ptj", "etaj", "maxjetflavor")
            }
            jets["jetalgo"] = integer(jets["jetalgo"])
            jets["maxjetflavor"] = integer(jets["maxjetflavor"])
            for integrand in self.integrands:
                self.histogram_runtimes[integrand.name] = Runtime(
                    fks_histograms(
                        integrand.born,
                        integrand.configurations,
                        self.histograms,
                        **jets,
                    ),
                    self.context,
                )

    def close(self):
        for integrand in getattr(self, "integrands", []):
            integrand.close()

    def total(self, k):
        coefficients = np.zeros(3)
        coefficients[k] = 1.0
        return self.combination(coefficients)

    def combination(self, coefficients, subprocess=None):
        """The estimate and its error of a linear combination of the NLO
        cross section, its LO part and its virtual residual (see
        Channel.combination), summed over the channels of a subprocess (by
        the name of its P* directory) or of all. The difference NLO - k LO is
        combination([1, -k, 0])."""
        channels = [
            c
            for c in self.channels
            if subprocess is None or c.integrand.name == subprocess
        ]
        values = [c.combination(coefficients) for c in channels]
        mean = sum(v for v, _ in values)
        return mean, math.sqrt(sum(e**2 for _, e in values))

    def run(self):
        """The grid iterations, then the integration iterations to the
        accuracy, then the outputs of the run. Returns the result."""
        self.setup()
        try:
            self.write_status("survey")
            for iteration in range(self.grid_iterations):
                points = self.grid_points
                if not self.fixed_points:
                    points *= 2**iteration
                for channel in self.channels:
                    channel.run(points, self.rng, False, None, self.batch)
                logger.info("grid iteration %d done", iteration + 1)
            for channel in self.channels:
                channel.reset()
            self.write_status("integration")
            total = self.points * len(self.channels)
            for iteration in range(self.max_iterations):
                for channel, points in zip(self.channels, self.shares(total)):
                    histograms = self.histogram_runtimes.get(channel.integrand.name)
                    channel.run(points, self.rng, True, histograms, self.batch)
                self.iterations = iteration + 1
                mean, error = self.total(0)
                logger.info("iteration %d: %.6e +- %.2e", iteration + 1, mean, error)
                self.write_status("integration")
                if self.fixed_points:
                    continue
                if abs(error) <= self.accuracy * abs(mean):
                    break
                total = self.next_points(mean, error)
            self.write_status("done")
            self.write_hwu()
            return self.result()
        finally:
            self.close()

    def next_points(self, mean, error):
        """The points of the next iteration towards the accuracy: those the
        errors so far imply are missing, by a tenth more, at least points
        per channel and at most four times the points so far."""
        done = sum(channel.count for channel in self.channels)
        least = self.points * len(self.channels)
        if mean == 0:
            return least
        needed = done * (error / (self.accuracy * abs(mean))) ** 2
        return round(min(max(1.1 * (needed - done), least), 4 * done))

    def shares(self, total):
        """The points of an iteration per channel, total in all, by the
        standard deviations of the channels (at least a tenth of an even
        share), points each with fixed points."""
        count = len(self.channels)
        if self.fixed_points:
            return [self.points] * count
        std = np.array([c.std or 0.0 for c in self.channels])
        if std.sum() <= 0:
            return [round(total / count)] * count
        share = std / std.sum()
        share = np.maximum(share, 0.1 / count)
        share /= share.sum()
        return [max(round(s * total), self.batch // 10) for s in share]

    def channel_status(self):
        status = []
        for channel in self.channels:
            mean, error = channel.result(0)
            status.append(
                dict(
                    subprocess=self.integrands.index(channel.integrand),
                    name=channel.name,
                    mean=mean,
                    error=error,
                    count=channel.count,
                    virtual_fraction=channel.trick.fraction,
                    configuration_probabilities=channel.choice.probabilities.tolist(),
                )
            )
        return status

    def histogram_results(self):
        from madspace.fks import fks_histogram_results

        results = None
        for channel in self.channels:
            if channel.histograms is None or channel.count == 0:
                continue
            entries = fks_histogram_results(
                self.histograms, channel.histograms, channel.count
            )
            if results is None:
                results = entries
                continue
            for total, entry in zip(results, entries):
                total["bin_values"] = [
                    a + b for a, b in zip(total["bin_values"], entry["bin_values"])
                ]
                total["bin_errors"] = [
                    math.hypot(a, b)
                    for a, b in zip(total["bin_errors"], entry["bin_errors"])
                ]
        return results or []

    def result(self):
        nlo, nlo_error = self.total(0)
        lo, lo_error = self.total(1)
        subprocesses = {}
        for integrand in self.integrands:
            entry = {}
            for key, coefficients in (("nlo", [1, 0, 0]), ("lo", [0, 1, 0])):
                sigma, error = self.combination(coefficients, integrand.name)
                entry[key] = dict(sigma=sigma, error=error)
            subprocesses[integrand.name] = entry
        return dict(
            nlo=dict(sigma=nlo, error=nlo_error),
            lo=dict(sigma=lo, error=lo_error),
            subprocesses=subprocesses,
            unit="GeV" if self.integrands[0].incoming == 1 else "pb",
        )

    def matrix_element_calls(self):
        calls = {}
        for integrand in self.integrands:
            calls[integrand.name] = integrand.provider.statistics
        return calls

    def write_status(self, status):
        nlo, error = self.total(0)
        count = sum(channel.count for channel in self.channels)
        info = dict(
            status=status,
            seed=self.seed,
            process=dict(
                name="NLO",
                subprocess=0,
                mean=nlo,
                error=error,
                count=count,
                count_unweighted=0,
                iterations=self.iterations,
                done=status == "done",
            ),
            channels=self.channel_status(),
            run_times=dict(
                wall_time_sec=time.time() - self.start,
                cpu_time_sec=time.process_time() - self.cpu_start,
            ),
            histograms=self.histogram_results(),
            fks=dict(
                result=self.result(),
                accuracy=self.accuracy,
                matrix_elements=self.matrix_element_calls(),
            ),
        )
        path = os.path.join(self.run_path, "info.json")
        with open(path + ".tmp", "w") as f:
            json.dump(info, f, indent=1)
        os.replace(path + ".tmp", path)

    def write_hwu(self):
        from madgraph.mg7.hwu_output import to_hwu

        histograms = self.histogram_results()
        if not histograms:
            return
        text = to_hwu(histograms, None, "NLO")
        if text:
            path = os.path.join(self.run_path, "MADatNLO_madspace.HwU")
            with open(path, "w") as f:
                f.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", help="the aMC@NLO output")
    parser.add_argument("--name", default="madspace_fo", help="the run name")
    parser.add_argument("--accuracy", type=float, help="req_acc_FO of the run")
    parser.add_argument("--grid-points", type=int, help="points per grid iteration")
    parser.add_argument("--grid-iterations", type=int, help="grid iterations")
    parser.add_argument("--points", type=int, help="points per channel and iteration")
    parser.add_argument("--max-iterations", type=int, help="integration iterations")
    parser.add_argument("--seed", type=int, help="the seed (iseed of the run card)")
    parser.add_argument(
        "--histograms", help="a TOML file with [histograms] as run_card.toml"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run = FKSRun(
        args.output,
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
    for name, entry in result["subprocesses"].items():
        nlo = entry["nlo"]
        print(f"  {name}: {nlo['sigma']:.6e} +- {nlo['error']:.2e}")
    return result


if __name__ == "__main__":
    main()
