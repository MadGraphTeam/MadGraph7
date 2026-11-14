#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * The renormalization and factorization scales of one kinematics of an NLO
 * integrand, those of setscales.f of MadFKS [2] with the parameters of its run
 * card. The reference scale of scale_global_reference is, for
 * `dynamical_scale_choice`
 *
 * @f[
 *   1:\ \sum_i \frac{E_i\,p_{\mathrm{T},i}}{\sqrt{p_{\mathrm{T},i}^2 + p_{z,i}^2}},
 *   \qquad
 *   2:\ \sum_i \sqrt{(E_i + p_{z,i})(E_i - p_{z,i})},
 *   \qquad
 *   3 \text{ or } -1:\ \frac{1}{2} \sum_i \sqrt{(E_i + p_{z,i})(E_i - p_{z,i})},
 * @f]
 *
 * over the outgoing particles, the first in the partonic center-of-mass frame
 * (a particle without transverse momentum adds nothing), or `mur_ref_fixed`
 * for -2. For a decay it is the mass @f$M@f$ of the decaying particle for
 * every choice, as setscales.f takes it for @f$\mu_R@f$ and the Ellis-Sexton
 * scale. It is raised to @ref min_scale, as minscaleR and minscaleF of
 * setscales.f. A fixed scale (`fixed_ren_scale`, `fixed_fac_scale`) replaces
 * it, and each scale, fixed or not, is multiplied by its factor:
 *
 * @f[
 *   \mu_R = \text{mur\_over\_ref}\ \mu_\mathrm{ref},
 *   \qquad
 *   \mu_{F,k} = \text{muf}k\text{\_over\_ref}\ \mu_\mathrm{ref}.
 * @f]
 *
 * The FKS integrand (@ref FKSCrossSection) takes the scales of the event, of
 * the Born (for the soft and soft-collinear counterterms and the n-body part,
 * as MadFKS at the soft counter-event) and of the collinear counter-event.
 * The Ellis-Sexton scale of MadFKS has no counterpart: the integrated
 * counterterms are written at @f$\mu_R@f$ and @f$\mu_F@f$, and the virtual
 * comes at @f$\mu_R@f$. The
 * MadFKS run card has one factorization scale (`muf_ref_fixed`,
 * `muf_over_ref`), and sigint of MadFKS takes that of the first beam for both.
 * A decay has no densities, and its @f$\mu_F@f$ enters nothing. MadFKS takes
 * @f$H_T/2@f$ without the first outgoing particle there (scale_global_reference
 * starts at the third particle), madspace the mass.
 *
 * `batch` is the leading batch dimension and `n` the number of particles.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – the kinematics, the incoming
 *   first, in any frame boosted along the beams.
 *
 * **Returns**
 * - `ren_scale` – `float`, shape `(batch,)` – renormalization scale.
 * - `fact_scale1`, `fact_scale2` – `float`, shape `(batch,)` – factorization
 *   scales of the two beams.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 */
class NLOScale : public FunctionGenerator {
public:
    /**
     * @param particle_count          Number of particles, incoming included.
     * @param dynamical_scale_choice  -1 or 3 for @f$H_T/2@f$, 2 for @f$H_T@f$,
     *                                1 for @f$E_T@f$, -2 for `mur_ref_fixed`.
     * @param fixed_ren_scale         Whether @f$\mu_R@f$ is fixed.
     * @param fixed_fac_scale         Whether the @f$\mu_F@f$ are fixed.
     * @param mur_ref_fixed           The fixed @f$\mu_R@f$.
     * @param muf1_ref_fixed          The fixed @f$\mu_F@f$ of the first beam.
     * @param muf2_ref_fixed          The fixed @f$\mu_F@f$ of the second beam.
     * @param mur_over_ref            Factor on @f$\mu_R@f$.
     * @param muf1_over_ref           Factor on @f$\mu_F@f$ of the first beam.
     * @param muf2_over_ref           Factor on @f$\mu_F@f$ of the second beam.
     * @param incoming                Number of incoming particles, two or one
     *                                for a decay.
     * @throws std::invalid_argument without an outgoing particle, for another
     *         number of incoming particles or another dynamical_scale_choice
     *         (0 and 10, the scale of the user, are not part of it).
     */
    NLOScale(
        std::size_t particle_count,
        int dynamical_scale_choice = -1,
        bool fixed_ren_scale = false,
        bool fixed_fac_scale = false,
        double mur_ref_fixed = 91.118,
        double muf1_ref_fixed = 91.118,
        double muf2_ref_fixed = 91.118,
        double mur_over_ref = 1.,
        double muf1_over_ref = 1.,
        double muf2_over_ref = 1.,
        std::size_t incoming = 2
    );
    /// The floor on the dynamical reference scale, minscaleR and minscaleF of
    /// setscales.f, in GeV.
    static constexpr double min_scale = 2.;
    /// Number of particles, incoming included.
    std::size_t particle_count() const { return _particle_count; }
    /// Number of incoming particles, one for a decay.
    std::size_t incoming() const { return _incoming; }
    /// The options of the kernel: dynamical_scale_choice, fixed_ren_scale,
    /// fixed_fac_scale, mur_ref_fixed, muf1_ref_fixed, muf2_ref_fixed,
    /// mur_over_ref, muf1_over_ref, muf2_over_ref and @ref min_scale.
    const std::vector<double>& options() const { return _options; }
    /// The same scales for kinematics of another number of particles.
    NLOScale with_particle_count(std::size_t particle_count) const;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _particle_count;
    std::size_t _incoming;
    std::vector<double> _options;
};

} // namespace madspace
