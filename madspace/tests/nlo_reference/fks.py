"""The FKS integrand of the Python driver (see the package): evaluate of a
channel of a subprocess at given random numbers, FKS configurations and
weights of their choice, with the virtual trick, the matrix elements of the
provider called on the points the cuts let through."""

import math

import numpy as np
from madgraph.mg7.nlo.fks import FKSRun, SubprocessIntegrand
from madgraph.mg7.nlo.provider import ew_sudakov_inputs, fks_blocks


class FKSReferenceIntegrand(SubprocessIntegrand):
    """SubprocessIntegrand of madgraph/mg7/nlo/fks.py with evaluate, the
    reference of FKSChannelIntegrand."""

    def evaluate(self, channel, x, fks_index, configuration_weight, trick, rng):
        """The weights of a channel at the points x (n, dimension) of its
        random numbers, the FKS configurations and the weights of their
        choice, with the virtual on the fraction of the points of trick:
        the NLO weight per point, its LO part, its n-body part and its
        virtual residual (that of the points with the virtual), the outputs
        of FKSKinematics and
        FKSCrossSection by name, the virtual and the Born of the points with
        the virtual."""
        out = self.kinematics[channel](x, fks_index=fks_index)
        if self.resonance and "resonance_momentum" not in out:
            # a channel without resonances
            out["resonance_momentum"] = np.zeros((len(fks_index), 4))
        resonance = [out["resonance_momentum"]] if self.resonance else []
        kernels = self.kernels(
            out["collinear_momenta"],
            out["p_i_hat"],
            out["born_momenta"],
            *resonance,
            fks_index=fks_index,
        )
        references = {}
        if self.user_scale:
            # the reference scales of the user at the event and its
            # counter-events, the Born at the soft counter-event
            for name, momenta, valid in (
                ("reference_scale", "momenta", out["weight"] != 0),
                (
                    "soft_reference_scale",
                    "soft_momenta",
                    (out["soft_det"] != 0) | (out["born_weight"] != 0),
                ),
                (
                    "collinear_reference_scale",
                    "collinear_momenta",
                    out["collinear_det"] != 0,
                ),
            ):
                references[name] = self.provider.reference_scales(out[momenta], valid)
        user = [references["soft_reference_scale"]] if self.user_scale else []
        ren_scale = self.born_scale(out["born_momenta"], *user)["ren_scale"]
        passes = {
            name: self.pass_cuts(out[name], layout_index=fks_index)["mask"] != 0
            for name in ("momenta", "soft_momenta", "collinear_momenta")
        }
        # the soft counter-event only where it exists, at xi = 0, as
        # FKSCrossSection
        passes["soft_momenta"] &= out["soft_xi"] == 0
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
        alpha = self.alpha
        if self.tagged is not None:
            # the couplings of tagged photons in alpha(0), the others in G_mu
            for name in ("matrix_element", "virtual"):
                blocks[name] = blocks[name] * self.tagged["nlo"]
            for name in ("born", "spin_correlated"):
                blocks[name] = blocks[name] * self.tagged["born"]
            b = self.born_count
            factors = self.tagged["born"][:b, None, None]
            blocks["color_correlated"] = blocks["color_correlated"] * factors
            alpha = alpha * self.tagged["alpha"]
        amp2 = blocks["amp2"]
        total = amp2.sum(1)
        channel_weight = np.where(
            total > 0, amp2[:, channel] / np.where(total > 0, total, 1), 0.0
        )
        # the Born orders, without the Borns of the extra counterterms
        born = blocks["born"][:, : self.born_count]
        virtual, approximation = trick.inputs(blocks["virtual"], born, evaluated)
        inputs = dict(out)
        inputs.update(blocks)
        inputs.update(references)
        inputs.update(
            virtual=virtual,
            channel_weight=channel_weight,
            configuration_weight=configuration_weight,
            alpha=np.full(count, alpha),
        )
        if self.ew_sudakov is not None:
            # at the Born of the points that take it, as compute_ewsudakov
            # with the Born
            nbody = born_mask & (out["born_weight"] != 0)
            inputs["ew_sudakov"] = self.ew_sudakov_factor * ew_sudakov_corrections(
                self.provider,
                self.ew_sudakov,
                out["born_momenta"],
                fks_index,
                ren_scale,
                nbody,
            )
        result = self.cross_section.by_name(inputs, fks_index=fks_index)
        inputs["virtual"] = approximation
        without = self.cross_section.by_name(inputs, fks_index=fks_index)
        residual = result["cross_section"] - without["cross_section"]
        return dict(
            nlo=result["cross_section"],
            lo=result["lo_weight"],
            nbody=result["nbody_weight"],
            residual=residual,
            outputs=result,
            kinematics=out,
            virtual=blocks["virtual"],
            born=born,
            evaluated=evaluated,
            variations=np.hstack(
                [
                    result.get("scale_variations", np.zeros((count, 0))),
                    result.get("pdf_variations", np.zeros((count, 0))),
                ]
            ),
            ew_sudakov=result.get("ew_sudakov_weights", np.zeros((count, 0))),
        )


def ew_sudakov_corrections(provider, runtime, momenta, fks_index, ren_scale, mask):
    """The outputs of the EWSudakov runtime (batch, modes) at the Born
    kinematics momenta (batch, n, 4) of the configurations fks_index, with
    the matrix elements of the provider at g = 1 and the LO2 part at
    alpha_s = 1 / (4 pi) and the renormalization scale of the Born, zero
    where mask is false."""
    count = len(momenta)
    out = np.zeros((count, runtime.function.mode_count()))
    points = np.flatnonzero(mask)
    if len(points) == 0:
        return out
    p = np.ascontiguousarray(momenta[points])
    inputs = ew_sudakov_inputs(provider, runtime.function, p, fks_index[points])
    out[points] = runtime(
        p,
        inputs["diagonal"],
        inputs["lo2"],
        inputs["derivatives"],
        inputs["interference"],
        np.full(len(points), 1 / (4 * math.pi)),
        np.asarray(ren_scale)[points],
    )["sudakov"]
    return out


class FKSReferenceRun(FKSRun):
    """FKSRun with the integrands of the reference."""

    integrand_class = FKSReferenceIntegrand
