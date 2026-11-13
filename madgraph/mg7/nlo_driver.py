"""The scheme-independent core of the fixed-order NLO runs of madspace on an
aMC@NLO output, which the integrands of a subtraction scheme plug into (FKS
in fks_driver.py, Catani-Seymour in cs_driver.py): the cards of the output,
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
  evaluated), as SubprocessIntegrand.evaluate of fks_driver.py,
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
"""

import json
import logging
import math
import os
import time

import numpy as np

from madgraph.mg7 import fks_couplings

logger = logging.getLogger("madgraph.mg7.nlo_driver")

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
        raise NotImplementedError("nlo_driver: no FxFx merging (ickkw)")
    if integer(run_card["dynamical_scale_choice"]) in (0, 10):
        raise NotImplementedError("nlo_driver: no user scales")
    beams = (integer(run_card["lpp1"]), integer(run_card["lpp2"]))
    if any(abs(beam) > 1 for beam in beams):
        raise NotImplementedError(f"nlo_driver: beams {beams}")
    if integer(run_card.get("pdfscheme", 0)) != 0:
        raise NotImplementedError("nlo_driver: only the MSbar PDF scheme")


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
                raise NotImplementedError(f"nlo_driver: pdlabel {pdlabel}")
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
        # the sums of the NLO weight, its LO part and its virtual residual,
        # and the sums over the iterations of their products about the means
        # of each iteration
        self.sums = np.zeros(3)
        self.covariance = np.zeros((3, 3))
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
        count, sums, products = 0, np.zeros(3), np.zeros((3, 3))
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
            self.choice.add(choice, nlo / choice_weight)
            if not keep:
                continue
            values = np.stack([nlo, lo, residual])
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
            os.path.join(self.output, "SubProcesses", self.work_directory),
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
            jets = {
                key: real(self.run_card[key])
                for key in ("jetalgo", "jetradius", "ptj", "etaj", "maxjetflavor")
            }
            jets["jetalgo"] = integer(jets["jetalgo"])
            jets["maxjetflavor"] = integer(jets["maxjetflavor"])
            for integrand in self.integrands:
                self.histogram_runtimes[integrand.name] = (
                    integrand.histogram_runtime(self.histograms, jets)
                )

    def make_integrands(self, directories, n_f, n_l):
        """The integrands of the subprocesses of the output (their P*
        directories), with the n_f light quarks and n_l massless leptons of
        the model, on self.context with self.couplings."""
        raise NotImplementedError

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
                    **{
                        channel.integrand.choice_key: (
                            channel.choice.probabilities.tolist()
                        )
                    },
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
