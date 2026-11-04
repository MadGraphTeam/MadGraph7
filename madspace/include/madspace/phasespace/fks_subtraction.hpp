#pragma once

#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/fks_configuration.hpp"
#include "madspace/phasespace/fks_limits.hpp"
#include "madspace/phasespace/fks_sector.hpp"
#include "madspace/util.hpp"

namespace madspace {

/**
 * The real emission of the FKS configurations of a Born and its local soft,
 * collinear and soft-collinear counterterms, as compute_real_emission and
 * compute_soft_counter_term and its siblings in fks_singular.f [2, 3] without
 * the collinear remnants, each event in the configuration of its `fks_index`
 * (see @ref FKSConfigurationMapping). It takes the event and counter-events of
 * @ref FKSKinematics, the real matrix element at the event and the Born
 * blocks of @ref FKSLimits, for QCD splittings.
 *
 * With the symmetry factor @f$\sigma@f$ of the configuration
 * (FKSConfiguration::symmetry_factor()), the sector functions @f$S@f$ of
 * @ref FKSSectorFunction at the event and the counter-events, the limits
 * @f$L@f$ of @ref FKSLimits and the prefactors @f$f@f$ of
 * @ref FKSPrefactors, the contributions are
 * @f[
 *   \text{real} = \sigma S |M_{n+1}|^2, \quad
 *   \text{soft} = -\sigma S_s f_s L_s, \quad
 *   \text{collinear} = -\sigma S_c f_c L_c, \quad
 *   \text{soft\_collinear} = \sigma S_{sc} f_{sc} L_{sc},
 * @f]
 * per unit `weight` of the event and per unit `det` of each counter-event, so
 * that their sum times those, the flux and the parton luminosity of each is
 * the subtracted real emission. As in MadFKS, the soft counterterm contributes
 * where @f$\hat\xi \xi_\mathrm{max,s} \le \xi_\mathrm{S,cut}@f$ and
 * @f$\xi \le \xi_\mathrm{S,cut}@f$, the collinear one where
 * @f$y > 1 - \delta_S@f$ and the soft-collinear one where
 * @f$\hat\xi \xi_\mathrm{max,c} < \xi_\mathrm{S,cut}@f$,
 * @f$\xi_c < \xi_\mathrm{S,cut}@f$ and @f$y > 1 - \delta_S@f$, with the
 * upper ends @f$\xi_\mathrm{max,s}@f$ and @f$\xi_\mathrm{max,c}@f$ of the
 * range of @f$\xi@f$ at the soft and collinear counter-events and
 * @f$\xi_c@f$ of the collinear one. Each contribution is zero where its
 * momenta are not generated (zero `weight` or `det`).
 *
 * Close to the limits the real emission is that of sreal and fks_Sij.f.
 * Below `tiny` in @f$1 - y@f$ the collinear limit of @ref FKSLimits at the
 * event over @f$\xi^2 (1 - y)@f$, else below `tiny` in @f$\xi@f$ the soft
 * one, with the direction of i at the soft counter-event for the eikonal
 * factors, in place of `matrix_element`, which is not used there. For
 * @f$\xi \le@f$ `tiny` the sector function at the event takes the direction
 * of i at the soft counter-event, at the soft-collinear one where also
 * @f$1 - y <@f$ `tiny` and j is massless. This keeps the precision of the
 * real matrix element near the limits out of the result, at the price of an
 * error of the order of `tiny`. The maps of @f$y@f$ of genps_fks.f and
 * @ref FKSRadiation have @f$1 - y \ge 10^{-6}@f$, so with the default `tiny`
 * only the soft switch acts at their events.
 *
 * `batch` is the leading batch dimension and `n` the number of particles of
 * the real process, incoming included.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – momenta of the real
 *   emission, in the order of the real process of the configuration of the
 *   event, in any frame.
 * - `weight` – `float`, shape `(batch,)` – the weight of the event, zero
 *   where it is not generated.
 * - `xi`, `xi_hat`, `y`, `xi_max` – `float`, shape `(batch,)` – the FKS
 *   variables of the event.
 * - `p_i_hat` – `float`, shape `(batch, 4)` – momentum of i per unit
 *   @f$\xi@f$ at the event.
 * - `soft_momenta`, `collinear_momenta`, `soft_collinear_momenta` – `float`,
 *   shape `(batch, n, 4)` – momenta of the counter-events.
 * - `soft_det`, `collinear_det`, `soft_collinear_det` – `float`, shape
 *   `(batch,)` – their Jacobians, zero where they are not generated.
 * - `soft_p_i_hat`, `collinear_p_i_hat`, `soft_collinear_p_i_hat` – `float`,
 *   shape `(batch, 4)` – momentum of i per unit @f$\xi@f$ at them.
 * - `soft_xi_max` – `float`, shape `(batch,)` – upper end of the range of
 *   @f$\xi@f$ at the soft counter-event.
 * - `collinear_xi`, `collinear_xi_max` – `float`, shape `(batch,)` –
 *   @f$\xi@f$ and the upper end of its range at the collinear counter-event.
 * - `matrix_element` – `float`, shape `(batch,)` – the real matrix element at
 *   the event, not used close to the limits (see `tiny`).
 * - `born`, `color_correlated`, `spin_correlated` – the Born and its
 *   color-correlated and spin-correlated versions, as for @ref FKSLimits.
 * - `alpha_s` – `float`, shape `(batch,)` – the strong coupling.
 * - `fks_index` – `int`, shape `(batch,)` – the configuration of the event,
 *   from 0 to the number of configurations minus one.
 *
 * **Returns**
 * - `real` – `float`, shape `(batch,)` – the real emission, per unit
 *   `weight`.
 * - `soft`, `collinear`, `soft_collinear` – `float`, shape `(batch,)` – the
 *   counterterms, per unit `det` of their counter-events.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] S. Frixione, Z. Kunszt, A. Signer, "Three-jet cross sections to
 *   next-to-leading order", https://arxiv.org/abs/hep-ph/9512328
 * - [3] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 */
class FKSSubtraction : public FunctionGenerator {
public:
    /**
     * @param configurations  The FKS configurations, in the order of
     *                        `fks_index`, with QCD splittings.
     * @param xicut           The cut of the plus distribution in @f$\xi@f$,
     *                        see @ref FKSPrefactors.
     * @param xi_s_cut        The upper end of the soft counterterms in
     *                        @f$\xi@f$ (xiScut).
     * @param delta_o         The cut of the plus distribution in @f$1 - y@f$
     *                        for a final-state emitter (deltaO).
     * @param delta_i         The same for an initial-state emitter (deltaI).
     * @param delta_s         The lower end of the collinear counterterms in
     *                        @f$1 - y@f$ (deltaS).
     * @param fks_a           Power of the energies in d_kl, see
     *                        @ref FKSSectorFunction.
     * @param fks_b           Power of the angle in d_kl.
     * @param use_energy      Energies of particles that are not color octets
     *                        in d_kl (useenergy).
     * @param use_beta        Velocities of massive particles that are not
     *                        color octets in d_kl (usebeta).
     * @param h_damp_one      h_damp is 1 below it and 0 above 1 minus it
     *                        (one_h_damp).
     * @param h_damp_power    Power of h_damp (a_h_damp).
     * @param tiny            Below it in @f$\xi@f$ or @f$1 - y@f$ the real
     *                        emission is its limit, as tiny of sreal (zero
     *                        for the real matrix element everywhere).
     * @throws std::invalid_argument for a splitting that is not one of QCD
     *         or parameters out of their range.
     */
    FKSSubtraction(
        const std::vector<FKSConfiguration>& configurations,
        double xicut = 0.5,
        double xi_s_cut = 0.5,
        double delta_o = 1.,
        double delta_i = 1.,
        double delta_s = 1.,
        double fks_a = 1.5,
        double fks_b = 1.5,
        bool use_energy = true,
        bool use_beta = true,
        double h_damp_one = 0.01,
        double h_damp_power = 1.,
        double tiny = 1e-6
    );
    /// Number of configurations.
    std::size_t configuration_count() const { return _symmetry.size(); }
    /// Number of particles of the real processes, incoming included.
    std::size_t particle_count() const { return _limits.particle_count(); }
    /// The sector functions.
    const FKSSectorFunction& sector_function() const { return _sector_function; }
    /// The soft and collinear limits.
    const FKSLimits& limits() const { return _limits; }
    /// The prefactors.
    const FKSPrefactors& prefactors() const { return _prefactors; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    FKSSectorFunction _sector_function;
    FKSLimits _limits;
    FKSPrefactors _prefactors;
    std::vector<double> _symmetry;
    std::vector<me_int_t> _massless;
    double _xi_s_cut, _delta_s, _tiny;
};

} // namespace madspace
