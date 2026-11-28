"""The Catani-Seymour integrands of the Python driver (see the package):
evaluate of a channel of the n-body stream of a subprocess and of the real
stream of a real process at given random numbers, with the virtual trick,
the matrix elements of the providers called on the points the cuts let
through, and the weights of the real stream at given real emissions
(weights), the sums of the histograms of both."""

import math

import numpy as np
from madgraph.mg7.nlo.core import Runtime, madspace, mapping_outputs
from madgraph.mg7.nlo.cs import CSBornIntegrand, CSRealIntegrand, CSRun
from madgraph.mg7.nlo.libraries import transverse_square
from madgraph.mg7.nlo.provider import boost_z, partonic_rapidity

INV_GEV2_TO_PB = 0.38937937217186e9


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


def luminosity(function, x1, x2, mu1, mu2, omx=None):
    """The luminosity of each group (batch, groups) of a Luminosity of
    madgraph/mg7/nlo/cs.py at the momentum fractions x1, x2 and the
    factorization scales mu1, mu2, for dressed leptons with 1 - x of omx
    (omx1, omx2)."""
    fractions = (x1, x2, *omx) if function.dressed else (x1, x2)
    return function.runtime(*fractions, mu1, mu2)["luminosity"]


def correlated(dipole, provider, cm, plain):
    """The Born correlated in color (QCD) or charge (QED) of the emitter
    and the spectator per Born order (points, b), -weight B for a photon
    that splits."""
    if dipole.group == "QCD":
        return provider.color_correlated(cm)[:, :, dipole.legs[0], dipole.legs[1]]
    if dipole.weight is not None:
        return -dipole.weight * plain
    return dipole.charges * plain


def histogram_inputs(runtime, layout, momenta, inputs, k):
    """The jets of a layout of CSHistograms (the PDG codes of its jets and
    its NLOJets) at the momenta of kinematics k and the observables of the
    histograms on them, into inputs."""
    pids, function = layout
    cache = runtime.__dict__.setdefault("reference_runtimes", {})
    if id(function) not in cache:
        cache[id(function)] = Runtime(function, runtime.context)
    if pids not in cache:
        cache[pids] = [Runtime(f, runtime.context) for f in runtime.functions[pids]]
    out = cache[id(function)](momenta, layout_index=np.zeros(len(momenta), np.int32))
    inputs[f"jet_count{k}"] = out["jet_count"].astype(np.int32)
    for i, observable in enumerate(cache[pids]):
        value = observable(out["jet_momenta"])
        inputs[f"value{i}_{k}"] = np.asarray(next(iter(value.values()))).reshape(
            len(momenta)
        )


def histogram_bins(runtime, inputs):
    """The bins of the histograms of CSHistograms (NLOHistograms) of the
    kinematics of its layouts, their values and errors."""
    ms = madspace()
    if getattr(runtime, "reference_binning", None) is None:
        items = [
            ms.NLOHistograms.HistItem(
                histogram["min"], histogram["max"], histogram["bin_count"], count
            )
            for histogram, count in zip(runtime.histograms, runtime.jet_counts)
        ]
        runtime.reference_binning = Runtime(
            ms.NLOHistograms(items, len(runtime.layouts)), runtime.context
        )
    return list(runtime.reference_binning.by_name(inputs).values())


class CSBornReference(CSBornIntegrand):
    """CSBornIntegrand of madgraph/mg7/nlo/cs.py with evaluate, the reference
    of CSBornChannelIntegrand, and the parts the dipoles of the real stream
    of the reference take from the Born of their subprocess."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ms = madspace()
        self.forward = [
            ms.FunctionRuntime(m.forward_function(), self.context)
            for m in self.channel_mappings
        ]
        self.alpha_s = Runtime(self.couplings.running, self.context)

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
        born = luminosity(self.luminosity, x1, x2, mu1, mu2, omx)[:, 0]
        if z is None or self.convolution is None:
            return born, None
        convolved = {group: np.zeros((len(x1), 2, 2)) for group in self.groups}
        for leg in range(2):
            shifted = [x1, x2][leg] / z[:, leg]
            inside = shifted <= 1
            fractions = [x1, x2]
            fractions[leg] = np.where(inside, shifted, 0.5)
            values = luminosity(self.convolution[leg], *fractions, mu1, mu2)
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
        FKSReferenceIntegrand.evaluate), with the virtual on
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

    def histogram_sums(self, runtime, result, jacobian, choice):
        """The histograms of runtime (histogram_runtime) of a batch of
        evaluate with the Jacobian of its grid."""
        inputs = {"weight0": result["nlo"] * jacobian}
        histogram_inputs(runtime, runtime.born, result["momenta"], inputs, 0)
        return histogram_bins(runtime, inputs)


class CSRealReference(CSRealIntegrand):
    """CSRealIntegrand of madgraph/mg7/nlo/cs.py with evaluate and weights,
    the reference of CSRealChannelIntegrand, on the CSBornReference of its
    subprocesses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ms = madspace()
        self.alpha_s = self.source.alpha_s
        # the forward maps of the channels and the inverse maps of all of
        # them for their densities
        self.forward = [
            ms.FunctionRuntime(m.forward_function(), self.context)
            for m in self.channel_mappings
        ]
        self.inverse = [
            ms.FunctionRuntime(m.inverse_function(), self.context)
            for m in self.channel_mappings
        ]

    def luminosity(self, x1, x2, mu1, mu2, omx=None):
        """The luminosity of the real over its flavor lines, at the momentum
        fractions of the real and the factorization scales mu1, mu2, for
        dressed leptons with 1 - x of omx."""
        return luminosity(self.luminosity_function, x1, x2, mu1, mu2, omx)[:, 0]

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
        colors = correlated(dipole, provider, cm, plain)
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
        FKSReferenceIntegrand.evaluate and weights)."""
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
        m_channel, _ = self.channels[index]
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
        # the density of the mapping of the channel over that of all
        densities = self.densities(momenta, fractions, valid)
        total = densities.sum(1)
        channel_weight = np.divide(
            densities[:, index],
            total,
            out=np.zeros(count),
            where=np.isfinite(total) & (total > 0),
        )
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
            points = np.flatnonzero(active)
            if not len(points):
                continue
            y = -partonic_rapidity(me[points], self.incoming)
            cm = boost_z(me[points], y)
            evaluated = born.provider.born(
                cm, np.full(len(points), dipole.configuration)
            )
            k_t = boost_z(projected["spin_vector"][points, m], y)
            value = self.dipole_value(
                dipole,
                cm,
                k_t,
                color_weight[points],
                spin_weight[points],
                evaluated,
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
        norm = np.where(keep, det * channel_weight * self.source.flux(x1, x2), 0.0)
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

    def histogram_sums(self, runtime, result, jacobian, choice):
        """The histograms of runtime (histogram_runtime) of a batch of
        evaluate (or weights) with the Jacobian of its grid: the real
        emission and the Borns of the dipoles of each point."""
        inputs = {"weight0": result["real_weight"] * jacobian}
        histogram_inputs(runtime, runtime.real_jets, result["momenta"], inputs, 0)
        for m, jets in enumerate(runtime.born_jets):
            inputs[f"weight{m + 1}"] = result["dipole_weights"][:, m] * jacobian
            histogram_inputs(runtime, jets, result["born_momenta"][:, m], inputs, m + 1)
        return histogram_bins(runtime, inputs)


class CSReferenceRun(CSRun):
    """CSRun with the integrands of the reference."""

    born_class = CSBornReference
    real_class = CSRealReference
