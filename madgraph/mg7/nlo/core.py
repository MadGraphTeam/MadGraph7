"""The scheme-independent core of the fixed-order NLO runs of madspace on an
aMC@NLO output, which the integrands of a subtraction scheme plug into (FKS
in nlo/fks.py, Catani-Seymour in nlo/cs.py): the cards of the output,
the PDFs and alpha_s, the VEGAS grids, the adaptive discrete choice, the
virtual trick, the channels with their iterations to the accuracy, and the
outputs of the run (info.json, which MadBoard reads, and the histograms as
HwU).

A run (NLORun) integrates channels. An integrand of a scheme gives the
channels of one subprocess (a P* directory of the output) and

- channel_count, channel_dimension(index) (the random numbers of a channel)
  and channel_name(index),
- choice_count(index), the options of the discrete choice of a channel (the
  FKS configurations), and choice_key, its name in info.json,
- nlo_count, the NLO orders, for the virtual trick,
- evaluate(index, x, choice, choice_weight, trick, rng): the NLO weight per
  point, its LO part and its virtual residual (nlo, lo, residual), the
  virtual and the Born of the points with the virtual (virtual, born,
  evaluated), as SubprocessIntegrand.evaluate of nlo/fks.py, optionally
  the part of the weight that does not depend on the choice (nbody), and
  with
  scale_variations, the pairs of factors of mu_R and mu_F of the run card
  (scale_variations of this module), and pdf_members, the number of
  members of the PDF set of reweight_PDF (0 without), the NLO weight at each
  scale variation, then with each member (variations), and with
  ew_sudakov_modes, the modes of the Sudakov approximation of the EW
  corrections (0 without), its weight per mode (ew_sudakov),
- histogram_runtime(histograms, jets) and histogram_sums(runtime, result,
  jacobian, choice), the weights of the histograms of a batch,
- name, incoming, beams, energy, born_pdgs, matrix_element_calls() and
  close().

The integration follows the fixed-order run of MadFKS (driver_mintFO.f with
mint): every iteration adapts the grids, the choice and the virtual trick
of each channel to its points. After the grid iterations the integration
iterations add up their points, each iteration with its own variance, the
points of an iteration shared among the channels by their standard
deviations. Towards req_acc_FO the grid iterations double their points and
each integration iteration has the points the errors so far imply are
missing (at most four times those before). Where req_acc_FO is -1 the run
takes niters_FO iterations of npoints_FO points per channel.

With generator the run goes through the C++ event generator of madspace
instead (EventGenerator.integrate): an integrand of a scheme then also gives
generator_integrand(index, context, prefix, fraction, min_fraction, bins,
histograms, jets), a GeneratorIntegrand of madspace with its VEGAS grid, its
discrete choice and its virtual trick as globals of the context, which the
iterations adapt in C++ (FKSIntegrand for FKS).

With madnis_batches as well, the grid iterations adapt the VEGAS grids, the
discrete choice and the virtual trick, a MadNIS training of that many
batches per subprocess turns each grid into a Flow, and the integration
iterations run with the flows (generator_integrand with flow and
madnis_training). The configurations keep the discrete choice of the state,
which goes on adapting. With madnis_discrete a DiscreteFlow with the
probabilities of the grid iterations as its prior chooses them instead. Its
training integrand sigma p_k holds the n-body part times the probability the
flow itself learns, so the flow chases its own probabilities: it lowers p_k
where the real part of configuration k is negative and makes the weights
negative there (e+ e- -> j j: the spread of the weights back at that of
VEGAS after 3000 batches).
"""

import json
import logging
import math
import os
import time

import numpy as np

from madgraph.mg7.nlo import grids

logger = logging.getLogger("madgraph.mg7.nlo.core")

# the cuts of NLOCuts by their names in the run card
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
# the photon cuts and recombination of the run card, numbers and switches,
# with the defaults of the NLO run card for older cards
PHOTON_CUTS = {
    "ptgmin": 20.0,
    "etagamma": -1.0,
    "r0gamma": 0.4,
    "xn": 1.0,
    "epsgamma": 1.0,
    "rphreco": 0.1,
    "etaphreco": -1.0,
}
PHOTON_SWITCHES = {
    "isoem": True,
    "gamma_is_j": True,
    "lepphreco": True,
    "quarkphreco": True,
}
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
    arrays, its outputs by name, with its index (fks_index or layout_index)
    as the last argument where it has one."""

    def __init__(self, function, context):
        ms = madspace()
        self.function = function
        self.runtime = ms.FunctionRuntime(function.function(), context)
        self.names = list(function.arg_types().keys())
        self.returns = list(function.return_types().keys())

    def __call__(self, *args, fks_index=None, layout_index=None):
        tensors = [np.ascontiguousarray(a) for a in args]
        for index in (fks_index, layout_index):
            if index is not None:
                tensors.append(np.ascontiguousarray(index, dtype=np.int32))
        batch = len(args[0])
        outputs = []
        for out in self.runtime.call(tensors):
            out = np.from_dlpack(out)
            outputs.append(np.full(batch, out) if out.ndim == 0 else np.array(out))
        return dict(zip(self.returns, outputs))

    def by_name(self, inputs, fks_index=None, layout_index=None):
        indices = ("fks_index", "layout_index")
        args = [inputs[name] for name in self.names if name not in indices]
        return self(*args, fks_index=fks_index, layout_index=layout_index)


# the damping exponent of the VEGAS grids, as of the classic VEGAS
VEGAS_ALPHA = 1.5


class Vegas:
    """A separable VEGAS grid on the unit hypercube with bin_count bins per
    dimension, adapted to the absolute values of the weights (alpha the
    damping), as MINT of MadFKS. One rare large weight dominates the sums
    of the squares of its bins in every dimension, so that grids adapted to
    the squares of the NLO weights sampled the bulk ever more sparsely (p p
    > e+ e- j: errors four times larger at the same points)."""

    def __init__(self, dimension, bin_count=50, alpha=VEGAS_ALPHA):
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
                bins[:, d], weights=np.abs(weights), minlength=self.bin_count
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
    """The adaptive choice of one of count options (the FKS configuration), as
    MC_integer of MadFKS: probabilities proportional to the mean absolute
    integrand of each option, mixed with the uniform choice by floor, so
    that every option keeps being sampled. The integrand of an option is the
    part of the weight that depends on the choice (for FKS the event and its
    counter-events, without the n-body part) times the probability of its
    choice."""

    def __init__(self, count, floor=0.1):
        self.probabilities = np.full(count, 1.0 / count)
        self.floor = floor
        self.sums = np.zeros(count)
        self.counts = np.zeros(count)

    def sample(self, rng, points):
        index = rng.choice(len(self.probabilities), size=points, p=self.probabilities)
        return index, 1.0 / self.probabilities[index]

    def add(self, index, integrands):
        n = len(self.probabilities)
        self.sums += np.bincount(index, weights=np.abs(integrands), minlength=n)
        self.counts += np.bincount(index, minlength=n)

    def adapt(self):
        seen = self.counts > 0
        if not seen.any():
            return
        target = np.zeros_like(self.sums)
        target[seen] = self.sums[seen] / self.counts[seen]
        if target.sum() > 0:
            target /= target.sum()
            uniform = np.full_like(target, 1.0 / len(target))
            self.probabilities = (1 - self.floor) * target + self.floor * uniform
        self.sums[:] = 0.0
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


def born_topology(masses, nincoming, channel, order=None):
    """The Topology of the Born diagram of a channel of the provider
    (FortranProvider.channels), with the masses of the Born. With order the
    leg n of the provider is the particle order[n] of the topology, whose
    masses are then in that order (the Born of a dipole of nlo/cs.py)."""
    ms = madspace()
    order = list(range(len(masses))) if order is None else list(order)

    def line(label):
        if label < 0:
            return f"p{-label - 1}"
        position = order[label - 1]
        if position < nincoming:
            return f"i{position}"
        return f"o{position - nincoming}"

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
    """The cuts of the run card on the final-state jets, leptons and photons
    of the Born as madspace Cuts, for the bound on sqrt(s_hat) they imply:
    ptj on the jets (where the Born has two QCD partons or more, as cuts.f),
    ptl on the charged leptons, mll_sf on the pairs of a lepton and its
    antilepton and ptgmin on the photons where they are isolated (gamma_is_j
    off, cuts.f asks for as many isolated photons as the Born has). None
    without such cuts."""
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
    ptgmin = real(run_card.get("ptgmin", PHOTON_CUTS["ptgmin"]))
    gamma_is_j = flag(run_card.get("gamma_is_j", PHOTON_SWITCHES["gamma_is_j"]))
    if ptgmin > 0 and not gamma_is_j and 22 in final:
        cut = observable(born_pdgs, observable.obs_pt, [[22]])
        items.append(ms.CutItem(cut, min=ptgmin))
    return ms.Cuts(items) if items else None


def first(value):
    """The first entry of a list of the run card (lhaid,
    dynamical_scale_choice), the value itself otherwise."""
    if isinstance(value, str) and "," in value:
        value = value.split(",")
    return value[0] if isinstance(value, (list, tuple)) else value


def scale_variations(run_card):
    """The pairs of factors of mu_R and mu_F of the scale variations of the
    run card (reweight_scale with rw_rscale and rw_fscale), in the order of
    reweight_scale of fks_singular.f (mu_F outer, mu_R inner), none without
    reweight_scale."""
    if not flag(first(run_card.get("reweight_scale", False))):
        return []

    def factors(name):
        value = run_card.get(name, "1.0")
        if isinstance(value, str):
            value = value.replace(",", " ").replace("[", " ").replace("]", " ")
            value = value.split()
        if not isinstance(value, (list, tuple)):
            value = [value]
        return [real(v) for v in value]

    return [(r, f) for f in factors("rw_fscale") for r in factors("rw_rscale")]


def pdf_reweight(run_card):
    """Whether the run card reweights with the members of the PDF set
    (reweight_PDF, of the first set of lhaid)."""
    return flag(first(run_card.get("reweight_pdf", False)))


def flag(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", ".true.", "t", "1")
    return bool(value)


def integer(value):
    return int(float(first(value)))


def real(value):
    return float(first(value))


def run_cuts(run_card):
    """The cuts of NLOCuts by their names, from the run card, with the photon
    cuts and recombination."""
    cuts = {name: real(run_card[name]) for name in CUTS}
    cuts["maxjetflavor"] = integer(cuts["maxjetflavor"])
    cuts["jetalgo"] = integer(cuts["jetalgo"])
    for name, default in PHOTON_CUTS.items():
        cuts[name] = real(run_card.get(name, default))
    for name, default in PHOTON_SWITCHES.items():
        cuts[name] = flag(run_card.get(name, default))
    return cuts


def scale_parameters(run_card):
    """The scales of the run card by the names of nlo_scale."""
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


# the leptons of the beams of dressed electrons and positrons by lpp of the
# run card, the power of the sampling of their momentum fractions, get_ee_expo
# of the densities of MadFKS (pdg2pdf.f), and the alpha_s(m_Z) that pdfwrap.f
# of MadFKS sets for every set of lepton densities
DRESSED_LEPTONS = {3: 11, -3: -11}
LEPTON_POWER = 0.975
DRESSED_ALPHA_S = 0.118


def beam_kind(run_card):
    """What the beams of the run card are: "hadrons" (protons and
    antiprotons, lpp 1 and -1), "photons" (the elastic photons of protons, lpp
    2 and -2), "dressed" (electrons and positrons with the densities of
    pdlabel, lpp 3 and -3) or "leptons" (without densities, lpp 0)."""
    beam = abs(integer(run_card["lpp1"]))
    return {1: "hadrons", 2: "photons", 3: "dressed"}.get(beam, "leptons")


def check_run_card(run_card):
    """Raises NotImplementedError for the settings the driver does not
    cover."""
    if integer(run_card.get("ickkw", 0)) != 0:
        raise NotImplementedError("nlo.core: no FxFx merging (ickkw)")
    from madgraph.various.banner import is_upc_label

    beams = (integer(run_card["lpp1"]), integer(run_card["lpp2"]))
    hadrons = [abs(beam) == 1 for beam in beams]
    photons = [abs(beam) == 2 for beam in beams]
    dressed = [beam in DRESSED_LEPTONS for beam in beams]
    if not (beams == (0, 0) or all(hadrons) or all(photons) or all(dressed)):
        raise NotImplementedError(f"nlo.core: beams {beams}")
    pdlabel = str(run_card["pdlabel"]).lower()
    if all(dressed) and pdlabel.startswith("emela"):
        raise NotImplementedError("nlo.core: no densities of eMELA")
    if all(photons) and is_upc_label(pdlabel):
        raise NotImplementedError("nlo.core: no photons of gamma-UPC")
    if integer(run_card.get("pdfscheme", 0)) != 0:
        raise NotImplementedError("nlo.core: only the MSbar PDF scheme")


class DressedLeptons:
    """The densities of dressed electrons and positrons of the beams (lpp 3
    and -3), those of the set pdlabel of lep_densities in the Source
    directory source, tabulated for madspace in directory (see
    madgraph.mg7.lepton_density) and uploaded to the context, with pdgs the
    leptons of the beams. full and reduced are their luminosity sum_n
    f1_n(x1) f2_n(x2) (ee_comp_prod of pdg2pdf.f) at the scale of each beam:
    full that of the densities themselves, reduced that of the densities
    times (1 - x1)^p (1 - x2)^p of the power p = LEPTON_POWER of the sampling
    of the momentum fractions."""

    prefix = "ee"

    def __init__(self, source, directory, run_card, context):
        ms = madspace()
        from madgraph import MG5DIR
        from madgraph.mg7.lepton_density import LeptonDensity

        self.pdgs = [DRESSED_LEPTONS[integer(run_card[f"lpp{k}"])] for k in (1, 2)]
        energy = 2 * math.sqrt(real(run_card["ebeam1"]) * real(run_card["ebeam2"]))
        density = LeptonDensity(
            str(run_card["pdlabel"]).lower(),
            source,
            MG5DIR,
            ms,
            context,
            energy,
            beam_pdgs=self.pdgs,
            prefix=self.prefix,
            cache_dir=directory,
        )
        self.grid, self.meta = density.grid, density.meta
        # the ids of the components of each beam in the grid
        self.pids = [density._pids(pdg) for pdg in self.pdgs]
        self.components = density.components
        self.full = self.luminosity(0.0)
        self.reduced = self.luminosity(LEPTON_POWER)

    def offsets(self, power):
        """p - kappa0 of each component of each beam for the power p."""
        meta = self.meta
        return [
            [
                power - (kappa["c_b"] * meta["b_exp"] + kappa["c_1"])
                for kappa in (meta["kappa"][f"{n}_{pdg}"] for n in self.components)
            ]
            for pdg in self.pdgs
        ]

    def luminosity(self, power, beam_scales=True):
        """The EELuminosity of the densities times (1 - x)^power on each
        beam, with the scale of each beam or one for both."""
        offsets = self.offsets(power)
        return madspace().EELuminosity(
            self.grid,
            self.pids[0],
            self.pids[1],
            offsets[0],
            offsets[1],
            alpha_em=self.meta["alphaem"],
            lepton_mass=self.meta["me"],
            prefix=self.prefix,
            beam_scales=beam_scales,
        )


class Couplings:
    """The PdfGrid and AlphaSGrid of the run, uploaded to the context: those
    of the LHAPDF set of lhaid, or a built-in set of MadFKS (grids.BUILTIN,
    its grid or its table) with the running of alpha_s of pdfwrap.f. Lepton
    beams take alpha_s(m_Z) of the param card, dressed ones that of pdfwrap.f
    of MadFKS (DRESSED_ALPHA_S) and the densities of their leptons (leptons,
    a DressedLeptons, None for the other beams). Beams of elastic photons
    take the set of pdlabel and its alpha_s as hadrons do (setrun.f of
    MadFKS). With reweight_PDF (only for
    an LHAPDF set, as MG5aMC demands) members holds the RunningCoupling,
    the PdfGrid and the prefix of the globals of every member of its set
    (grids.lhapdf_members), pdf_set its name and error type, as
    pdf_variations of fks_cross_section takes them."""

    member_prefix = "pdf_member"

    def __init__(self, directory, run_card, param_card, context):
        ms = madspace()
        os.makedirs(directory, exist_ok=True)
        output = os.path.dirname(os.path.dirname(os.path.realpath(directory)))
        hadrons = beam_kind(run_card) == "hadrons"
        # elastic photons take the set of pdlabel for alpha_s, as hadrons do
        from_set = hadrons or beam_kind(run_card) == "photons"
        self.pdf = None
        self.members, self.pdf_set = [], None
        pdlabel = str(run_card["pdlabel"]).lower()
        reweight = hadrons and pdf_reweight(run_card)
        if reweight and pdlabel != "lhapdf":
            raise ValueError(
                "nlo.core: reweight_PDF needs the sets of LHAPDF, as in MG5aMC"
            )
        if from_set and pdlabel == "lhapdf":
            lhaid = integer(run_card["lhaid"])
            grid, info = grids.lhapdf_set(lhaid)
            self.pdf = ms.PdfGrid(grid)
            self.alpha_s = ms.AlphaSGrid(info)
            self.asmz = grids.lhapdf_alpha_s_mz(info)
        else:
            if from_set and pdlabel not in grids.BUILTIN:
                raise NotImplementedError(f"nlo.core: pdlabel {pdlabel}")
            loops = 2
            if from_set:
                self.asmz, loops = grids.BUILTIN[pdlabel]
            elif beam_kind(run_card) == "dressed":
                self.asmz = DRESSED_ALPHA_S
            else:
                self.asmz = float(param_card["sminputs"]["3"])
            path = os.path.join(directory, f"alpha_s_{self.asmz}_{loops}.info")
            grids.write_alpha_s_grid(path, self.asmz, loops)
            self.alpha_s = ms.AlphaSGrid(path)
            if hadrons:
                grid = os.path.join(directory, f"{pdlabel}_0000.dat")
                if not os.path.isfile(grid):
                    grids.write_builtin_grid(grid, pdlabel, output, directory)
                self.pdf = ms.PdfGrid(grid)
        grid_names = ("pdf_logx", "pdf_logq2", "pdf_coefficients")
        for name in (
            *grid_names,
            *(f"{DressedLeptons.prefix}.{name}" for name in grid_names),
            "alpha_s_logq2",
            "alpha_s_coefficients",
        ):
            if context.global_exists(name):
                context.delete_global(name)
        if self.pdf is not None:
            self.pdf.initialize_globals(context)
        self.alpha_s.initialize_globals(context)
        self.running = ms.RunningCoupling(self.alpha_s)
        if reweight:
            pdf_set = grids.lhapdf_members(integer(run_card["lhaid"]))
            self.pdf_set = dict(name=pdf_set["name"], error_type=pdf_set["error_type"])
            for member, (path, alpha_s) in enumerate(pdf_set["members"]):
                prefix = f"{self.member_prefix}{member}"
                for name in (*grid_names, "alpha_s_logq2", "alpha_s_coefficients"):
                    if context.global_exists(f"{prefix}.{name}"):
                        context.delete_global(f"{prefix}.{name}")
                grid, alpha_s_grid = ms.PdfGrid(path), ms.AlphaSGrid(alpha_s)
                grid.initialize_globals(context, prefix)
                alpha_s_grid.initialize_globals(context, prefix)
                self.members.append(
                    (ms.RunningCoupling(alpha_s_grid, prefix), grid, prefix)
                )
        self.leptons = None
        if beam_kind(run_card) == "dressed":
            self.leptons = DressedLeptons(
                os.path.join(output, "Source"), directory, run_card, context
            )


class Channel:
    """The state of one channel of one subprocess: its VEGAS grid, its
    discrete choice (of the FKS configurations), its virtual trick and the
    sums of its integration iterations, which add up the points of all of
    them, each iteration with its own variance."""

    def __init__(self, integrand, index, fks_params, bins):
        self.integrand, self.index = integrand, index
        self.dimension = integrand.channel_dimension(index)
        self.vegas = Vegas(self.dimension, bins)
        self.choice = DiscreteChoice(integrand.choice_count(index))
        self.trick = VirtualTrick(
            integrand.nlo_count,
            fks_params["VirtualFraction"],
            fks_params["MinVirtualFraction"],
        )
        self.reset()
        self.std = None

    def reset(self):
        self.count = 0
        # the sums of the NLO weight, its LO part, its virtual residual and the
        # NLO weight at each scale variation and with each PDF member, and the
        # sums over the iterations of their products about the means of each
        # iteration
        size = 3 + len(getattr(self.integrand, "scale_variations", []))
        size += getattr(self.integrand, "pdf_members", 0)
        size += getattr(self.integrand, "ew_sudakov_modes", 0)
        self.sums = np.zeros(size)
        self.covariance = np.zeros((size, size))
        self.histograms = None

    @property
    def name(self):
        return self.integrand.channel_name(self.index)

    def run(self, points, rng, keep, histograms, batch):
        """An iteration of points points of the channel, after which its
        VEGAS grid, its discrete choice and its virtual trick adapt to them,
        as mint does after each iteration. With keep the
        points add to the estimates of the channel, with the variance of
        this iteration, and to its histograms."""
        weights, residuals = [], []
        size = len(self.sums)
        count, sums, products = 0, np.zeros(size), np.zeros((size, size))
        for start in range(0, points, batch):
            n = min(batch, points - start)
            u = rng.random((n, self.dimension))
            x, jacobian, bins = self.vegas.sample(u)
            choice, choice_weight = self.choice.sample(rng, n)
            result = self.integrand.evaluate(
                self.index, x, choice, choice_weight, self.trick, rng
            )
            nlo = result["nlo"] * jacobian
            lo = result["lo"] * jacobian
            residual = result["residual"] * jacobian
            weights.append(nlo)
            residuals.append(residual)
            self.trick.add(result["virtual"], result["born"], result["evaluated"])
            self.vegas.add(bins, nlo)
            # the n-body part does not depend on the choice
            nbody = result.get("nbody", 0.0) * jacobian
            self.choice.add(choice, (nlo - nbody) / choice_weight)
            if not keep:
                continue
            variations = result.get("variations", np.zeros((n, 0))) * jacobian[:, None]
            sudakov = result.get("ew_sudakov", np.zeros((n, 0))) * jacobian[:, None]
            values = np.vstack([nlo, lo, residual, variations.T, sudakov.T])
            count += n
            sums += values.sum(1)
            products += values @ values.T
            if histograms is not None:
                sums_of = self.integrand.histogram_sums(
                    histograms, result, jacobian, choice
                )
                sums_of = [np.asarray(v).reshape(-1) for v in sums_of]
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
        """The estimate and its error of the k-th sum (NLO, LO, residual, then
        the scale variations, the PDF members and the modes of the Sudakov
        approximation)."""
        if self.count == 0:
            return 0.0, 0.0
        coefficients = np.zeros(len(self.sums))
        coefficients[k] = 1.0
        return self.combination(coefficients)

    def combination(self, coefficients):
        """The estimate and its error of a linear combination of the NLO
        weight, its LO part and its virtual residual on the same points."""
        if self.count == 0:
            return 0.0, 0.0
        c = np.zeros(len(self.sums))
        c[: len(coefficients)] = coefficients
        variance = max(c @ self.covariance @ c, 0.0) / self.count**2
        return c @ self.sums / self.count, math.sqrt(variance)


class GeneratorChannel:
    """A channel of a run through the C++ event generator, with the
    interface of Channel that the outputs of NLORun read: its results from
    the FKSIterationState of its ChannelEventGenerator, its histograms from
    the sums of the generator."""

    class Trick:
        def __init__(self, fraction):
            self.fraction = fraction

    class Choice:
        def __init__(self, probabilities):
            self.probabilities = np.asarray(probabilities)

    def __init__(self, channel, generator):
        self.integrand, self.index = channel.integrand, channel.index
        self.name = channel.name
        self.state = generator.iteration_state()
        self.count = self.state.count()
        self.trick = self.Trick(self.state.virtual_fraction())
        self.choice = self.Choice(self.state.configuration_probabilities())
        self.std = self.state.iteration_std()
        histograms = generator.histograms()
        self.histograms = None
        if histograms:
            self.histograms = []
            for histogram in histograms:
                self.histograms.append(np.asarray(histogram.bin_values))
                self.histograms.append(np.asarray(histogram.bin_errors))

    def result(self, k):
        coefficients = np.zeros(k + 1)
        coefficients[k] = 1.0
        return self.combination(coefficients)

    def combination(self, coefficients):
        if self.count == 0:
            return 0.0, 0.0
        width = self.state.value_count()
        c = [float(v) for v in coefficients][:width]
        return tuple(self.state.combination(c))


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
    maxjetflavor) ordered by pt, also a single one, and the other particles
    grouped by |pdg|,
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
        # jets always by their order, the real emission has one more of them
        if count == 1 and group != "jet":
            slots.append(group)
        else:
            slots += [f"{group}_{k + 1}" for k in range(count)]
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


class NLORun:
    """A fixed-order NLO run of madspace on an aMC@NLO output (see the
    module), with the integrands of a scheme from make_integrands, scheme
    the name of the results of the run in info.json and work_directory the
    directory in SubProcesses of the grids of the PDFs and alpha_s.

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
        generator: run the iterations in the C++ event generator of madspace
            (see the module), on threads threads (all cores by default).
        madnis_batches: with generator, the batches of a MadNIS training per
            subprocess between the grid and the integration iterations, none
            by default.
        madnis_discrete: train a DiscreteFlow of the configurations as well
            (see the module), off by default.
    """

    scheme = None
    work_directory = "madspace_nlo"

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
        generator=False,
        threads=-1,
        madnis_batches=0,
        madnis_discrete=False,
    ):
        self.output = os.path.realpath(output)
        self.generator = generator
        self.threads = threads
        self.madnis_batches = madnis_batches
        self.madnis_discrete = madnis_discrete
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
        self.context = ms.Context(
            ms.cpu_device(), self.threads if self.generator else -1
        )
        self.couplings = Couplings(
            os.path.join(self.output, "SubProcesses", self.work_directory),
            self.run_card,
            self.param_card,
            self.context,
        )
        n_f, n_l = light_flavors(self.output)
        # setcuts.f of MadFKS raises maxjetflavor to the light flavors of the
        # model, so that a massless b is a jet
        if integer(self.run_card["maxjetflavor"]) < int(n_f):
            self.run_card["maxjetflavor"] = int(n_f)
        root = os.path.join(self.output, "SubProcesses")
        names = sorted(
            d
            for d in os.listdir(root)
            if d.startswith("P") and os.path.isdir(os.path.join(root, d))
        )
        self.integrands = self.make_integrands(
            [os.path.join(root, name) for name in names], n_f, n_l
        )
        self.channels = [
            Channel(integrand, k, self.fks_params, self.bins)
            for integrand in self.integrands
            for k in range(integrand.channel_count)
        ]
        leading = self.integrands[0]
        if self.histogram_file:
            self.histograms = read_histograms(self.histogram_file)
        else:
            self.histograms = default_histograms(
                dict(born_pdgs=leading.born_pdgs, nincoming=leading.incoming),
                leading.energy,
                leading.beams == (0, 0),
                leading.incoming == 1,
                integer(self.run_card["maxjetflavor"]),
            )
        self.histogram_runtimes = {}
        if self.histograms:
            jets = self.jets()
            try:
                for integrand in self.integrands:
                    self.histogram_runtimes[integrand.name] = (
                        integrand.histogram_runtime(self.histograms, jets)
                    )
            except ValueError as error:
                # the real processes of QED corrections differ in more than
                # their QCD partons (a photon, a lepton pair from a photon),
                # which the default histograms do not cover
                if self.histogram_file:
                    raise
                logger.warning("no histograms: %s", error)
                self.histograms, self.histogram_runtimes = None, {}

    def make_integrands(self, directories, n_f, n_l):
        """The integrands of the subprocesses of the output (their P*
        directories), with the n_f light quarks and n_l massless leptons of
        the model, on self.context with self.couplings."""
        raise NotImplementedError

    def close(self):
        for integrand in getattr(self, "integrands", []):
            integrand.close()

    def total(self, k):
        coefficients = np.zeros(k + 1)
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
        if self.generator:
            return self.run_generator()
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

    def run_generator(self):
        """The run of run through the C++ event generator of madspace: the
        GeneratorIntegrand of every channel in a ChannelEventGenerator, all of
        them integrated by EventGenerator.integrate. The channels of the run
        then read their results from the generator (GeneratorChannel)."""
        ms = madspace()
        self.setup()
        try:
            for integrand in self.integrands:
                if not hasattr(integrand, "generator_integrand"):
                    raise NotImplementedError(
                        f"the event generator of madspace does not run the "
                        f"{self.scheme} integrands yet"
                    )
            self.write_status("survey")
            config = ms.GeneratorConfig()
            config.cpu_batch_size = self.batch
            # the grids adapt as those of the Python iterations
            config.vegas_damping = VEGAS_ALPHA
            config.vegas_absolute_weights = True
            jets = self.jets() if self.histograms else None
            integration = ms.IntegrationConfig()
            integration.grid_points = self.grid_points
            integration.grid_iterations = self.grid_iterations
            integration.points = self.points
            integration.max_iterations = self.max_iterations
            integration.accuracy = 0.0 if self.fixed_points else self.accuracy
            integrands, generators, event_generator = self.generators(config, jets)
            if self.madnis_batches:
                grid = ms.IntegrationConfig()
                grid.grid_points = self.grid_points
                grid.grid_iterations = self.grid_iterations
                grid.max_iterations = 0
                grid.accuracy = integration.accuracy
                event_generator.integrate(grid)
                fractions = [
                    generator.iteration_state().virtual_fraction()
                    for generator in generators
                ]
                self.write_status("madnis")
                flows = self.train_madnis(integrands, fractions)
                # other random numbers than those the flows learnt from
                integrands, generators, event_generator = self.generators(
                    config, jets, flows, fractions, self.seed + 1
                )
                integration.grid_iterations = 0
            event_generator.integrate(integration)
            self.channels = [
                GeneratorChannel(channel, generator)
                for channel, generator in zip(self.channels, generators)
            ]
            self.iterations = max(
                generator.status().iterations for generator in generators
            ) - integration.grid_iterations
            self.write_status("done")
            self.write_hwu()
            return self.result()
        finally:
            self.close()

    def generators(self, config, jets, flows=None, fractions=None, seed=None):
        """The generator integrands of the channels, their ChannelEventGenerators
        and the EventGenerator of them all: with the VEGAS grids, or with the
        flows (Flow, DiscreteFlow or None) of train_madnis and the fractions of
        the virtual the grid iterations reached."""
        ms = madspace()
        integrands, generators = [], []
        for number, channel in enumerate(self.channels):
            flow, discrete = (None, None) if flows is None else flows[number]
            fraction = self.fks_params["VirtualFraction"]
            if fractions is not None:
                fraction = fractions[number]
            integrand = channel.integrand.generator_integrand(
                channel.index,
                self.context,
                f"nlo{number}",
                fraction,
                self.fks_params["MinVirtualFraction"],
                self.bins,
                self.histograms,
                jets,
                flow,
                discrete,
            )
            integrands.append(integrand)
            generators.append(
                ms.ChannelEventGenerator(
                    contexts=[self.context],
                    integrand=integrand,
                    event_file="",
                    weight_file="",
                    config=config,
                    subprocess_index=self.integrands.index(channel.integrand),
                    name=channel.name,
                    histograms=None,
                )
            )
        event_generator = ms.EventGenerator(
            contexts=[self.context],
            channels=generators,
            seed=self.seed if seed is None else seed,
            config=config,
        )
        return integrands, generators, event_generator

    def train_madnis(self, integrands, fractions):
        """The MadNIS training of the channels after the grid iterations: per
        channel a Flow from its VEGAS grid (integrands of generators) and,
        with madnis_discrete and several FKS configurations, a DiscreteFlow
        of them conditioned on the random numbers with their probabilities as
        its prior, trained per subprocess for madnis_batches batches. Gives
        (flow, discrete) per channel."""
        ms = madspace()
        flows = []
        training = {}
        for number, (channel, integrand) in enumerate(zip(self.channels, integrands)):
            prefix = f"nlo{number}"
            vegas = integrand.adaptive_map()
            dimension = vegas.dimension()
            flow = ms.Flow(input_dim=dimension, prefix=f"{prefix}.flow")
            flow.initialize_from_vegas(self.context, vegas.grid_name(), self.seed)
            discrete = None
            count = integrand.configuration_count()
            if self.madnis_discrete and count > 1:
                discrete = ms.DiscreteFlow(
                    option_counts=[count],
                    prefix=f"{prefix}.discrete_flow",
                    dims_with_prior=[0],
                    condition_dim=dimension,
                )
                discrete.initialize_globals(self.context, self.seed)
            flows.append((flow, discrete))
            subprocess = self.integrands.index(channel.integrand)
            training.setdefault(subprocess, []).append(
                channel.integrand.generator_integrand(
                    channel.index,
                    self.context,
                    prefix,
                    fractions[number],
                    self.fks_params["MinVirtualFraction"],
                    self.bins,
                    None,
                    None,
                    flow,
                    discrete,
                    madnis_training=True,
                )
            )
        config = ms.MadnisConfig()
        config.batches = self.madnis_batches
        config.cpu_generator_batch_size = self.batch
        optimizer_context = ms.Context(ms.cpu_device(), self.threads)
        optimizer_context.copy_globals_from(self.context)
        madnis = ms.MultiMadnisTraining(
            generator_context=self.context,
            optimizer_context=optimizer_context,
            training_args=[
                ms.TrainingArgs(config=config, integrands=items, cwnet=None)
                for _, items in sorted(training.items())
            ],
            verbosity="log",
            seed=self.seed,
        )
        madnis.train()
        return flows

    def jets(self):
        """The jet parameters of the run card for the histograms."""
        jets = {
            key: real(self.run_card[key])
            for key in ("jetalgo", "jetradius", "ptj", "etaj", "maxjetflavor")
        }
        jets["jetalgo"] = integer(jets["jetalgo"])
        jets["maxjetflavor"] = integer(jets["maxjetflavor"])
        return jets

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
                    **{
                        channel.integrand.choice_key: (
                            channel.choice.probabilities.tolist()
                        )
                    },
                )
            )
        return status

    def histogram_results(self):
        from madspace.nlo import histogram_results

        results = None
        for channel in self.channels:
            if channel.histograms is None or channel.count == 0:
                continue
            entries = histogram_results(
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
        result = dict(
            nlo=dict(sigma=nlo, error=nlo_error),
            lo=dict(sigma=lo, error=lo_error),
            subprocesses=subprocesses,
            unit="GeV" if self.integrands[0].incoming == 1 else "pb",
        )
        variations = getattr(self.integrands[0], "scale_variations", [])
        if variations:
            # the cross section at each pair of factors, and the envelope in
            # percent of the central one, as MG5aMC prints it
            entries = []
            for k, (ren, fact) in enumerate(variations):
                sigma, error = self.total(3 + k)
                entries.append(dict(mu_r=ren, mu_f=fact, sigma=sigma, error=error))
            values = [entry["sigma"] for entry in entries]
            up = (max(values) / nlo - 1) * 100 if nlo else 0.0
            down = (1 - min(values) / nlo) * 100 if nlo else 0.0
            result["scale_variations"] = entries
            result["scale_uncertainty"] = dict(up=up, down=down)
        members = getattr(self.integrands[0], "pdf_members", 0)
        modes = getattr(self.integrands[0], "ew_sudakov_modes", 0)
        if modes:
            # the Sudakov approximation of the EW corrections per mode (sud_mod
            # 0 and 1 of compute_ewsudakov), and the NLO with all of them, as
            # the integrand of MadFKS sums them, in total and per subprocess
            first = 3 + len(variations) + members
            with_modes = np.zeros(first + modes)
            with_modes[0] = 1.0
            with_modes[first:] = 1.0

            def estimate(coefficients, subprocess=None):
                sigma, error = self.combination(coefficients, subprocess)
                return dict(sigma=sigma, error=error)

            def per_mode(subprocess=None):
                entries = []
                for k in range(modes):
                    coefficients = np.zeros(first + modes)
                    coefficients[first + k] = 1.0
                    entries.append(dict(estimate(coefficients, subprocess), sud_mod=k))
                return entries

            result["ew_sudakov"] = per_mode()
            result["nlo_with_ew_sudakov"] = estimate(with_modes)
            for name, entry in subprocesses.items():
                entry["ew_sudakov"] = per_mode(name)
                entry["nlo_with_ew_sudakov"] = estimate(with_modes, name)
        if members:
            # the cross section with each member and the uncertainty of LHAPDF
            # in percent of its central value, as MG5aMC prints it
            first = 3 + len(variations)
            entries = []
            for k in range(members):
                sigma, error = self.total(first + k)
                entries.append(dict(member=k, sigma=sigma, error=error))
            pdf_set = self.couplings.pdf_set
            central, plus, minus = grids.pdf_uncertainty(
                [entry["sigma"] for entry in entries], pdf_set["error_type"]
            )
            result["pdf_variations"] = entries
            result["pdf_uncertainty"] = dict(
                pdf_set,
                central=central,
                up=abs(plus / central) * 100 if central else 0.0,
                down=abs(minus / central) * 100 if central else 0.0,
            )
        return result

    def matrix_element_calls(self):
        return {
            integrand.name: integrand.matrix_element_calls()
            for integrand in self.integrands
        }

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
            **{
                self.scheme: dict(
                    result=self.result(),
                    accuracy=self.accuracy,
                    matrix_elements=self.matrix_element_calls(),
                )
            },
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
