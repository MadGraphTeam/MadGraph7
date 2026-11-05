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
 * collinear and soft-collinear counterterms with the collinear remnants of an
 * initial-state emitter, as compute_real_emission and
 * compute_soft_counter_term and its siblings in fks_singular.f [2, 3, 4], each
 * event in the configuration of its `fks_index` (see
 * @ref FKSConfigurationMapping). It takes the event and counter-events of
 * @ref FKSKinematics, the real matrix element at the event and the Born
 * blocks of @ref FKSLimits, for QCD and QED splittings. All contributions come
 * per NLO order (see @ref FKSCouplingOrders), the real matrix element too.
 *
 * With the symmetry factor @f$\sigma@f$ of the configuration
 * (FKSConfiguration::symmetry_factor()), the sector functions @f$S@f$ of
 * @ref FKSSectorFunction at the event and the counter-events, the limits
 * @f$L@f$ of @ref FKSLimits and the prefactors @f$f@f$ of
 * @ref FKSPrefactors, the contributions are
 * @f[
 *   \text{real} = \sigma S |M_{n+1}|^2, \quad
 *   \text{soft} = -\sigma S_s f_s L_s, \quad
 *   \text{collinear} = \sigma (R_c - S_c f_c L_c), \quad
 *   \text{soft\_collinear} = \sigma (S_{sc} f_{sc} L_{sc} + R_{sc}),
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
 * The collinear remnants @f$R_c@f$ and @f$R_{sc}@f$ of an initial-state
 * emitter are those of sreal_deg with the prefactors @f$f_{dc}@f$,
 * @f$f_{dsc}@f$ and those of the PDF scheme of @ref FKSPrefactors. With
 * @f$N = g^2 r B / (8 \pi^2 z)@f$, the coupling @f$g^2@f$ of the splitting
 * type (see @ref FKSLimits), the identical-particle ratio @f$r@f$ and the Born
 * @f$B@f$ of the order of the counterterms,
 * @f[
 *   D_\xi = N \Big[(1 - z) P(z) \log\frac{s \delta_I}{2 \mu_F^2} -
 *     (1 - z) P'(z)\Big], \qquad
 *   D_{l\xi} = 2 N (1 - z) P(z),
 * @f]
 * with the O(epsilon) part @f$P'@f$ of the splitting function, the
 * factorization scale @f$\mu_F@f$ of the beam of the emitter and @f$s@f$ of
 * the counter-event, and the parts @f$K_{p,l,d} = -N k_{p,l,d}@f$ of the
 * change of the PDF scheme from MSbar (xkplus, xklog, xkdelta, zero in MSbar,
 * with their QED parts for a QED splitting),
 * @f[
 *   R_c = f_{dc} \big[D_\xi + K_p + (D_{l\xi} + K_l) \log \xi_c\big]
 * @f]
 * at @f$z = 1 - \xi_c@f$, and at @f$z = 1@f$
 * @f[
 *   R_{sc} = -f_{dsc,1} \big[D_\xi + K_p + (D_{l\xi} + K_l) \log \xi_c\big]
 *     - f_{dsc,2} D_\xi - f_{dsc,3} D_{l\xi} + f_{pdf,d} K_d + f_{pdf,p} K_p
 *     + f_{pdf,l} K_l.
 * @f]
 * They are zero where the sector function of their counter-event is.
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
 * `batch` is the leading batch dimension, `n` the number of particles of
 * the real process, incoming included, and `o` the number of NLO orders.
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
 * - `matrix_element` – `float`, shape `(batch, o)` – the real matrix element
 *   at the event per NLO order, not used close to the limits (see `tiny`).
 * - `born`, `color_correlated`, `spin_correlated` – the Born and its
 *   color-correlated and spin-correlated versions per Born order, as for
 *   @ref FKSLimits.
 * - `alpha_s` – `float`, shape `(batch,)` – the strong coupling.
 * - `alpha` – `float`, shape `(batch,)` – the QED coupling.
 * - `fact_scale1`, `fact_scale2` – `float`, shape `(batch,)` – the
 *   factorization scales of the two beams, as @ref EnergyScale returns them.
 * - `fks_index` – `int`, shape `(batch,)` – the configuration of the event,
 *   from 0 to the number of configurations minus one.
 *
 * **Returns**
 * - `real` – `float`, shape `(batch, o)` – the real emission per NLO order,
 *   per unit `weight`.
 * - `soft`, `collinear`, `soft_collinear` – `float`, shape `(batch, o)` – the
 *   counterterms per NLO order, per unit `det` of their counter-events.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] S. Frixione, Z. Kunszt, A. Signer, "Three-jet cross sections to
 *   next-to-leading order", https://arxiv.org/abs/hep-ph/9512328
 * - [3] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 * - [4] R. Frederix, S. Frixione, V. Hirschi, D. Pagani, H.-S. Shao,
 *   M. Zaro, "The automation of next-to-leading order electroweak
 *   calculations", https://arxiv.org/abs/1804.10017
 */
class FKSSubtraction : public FunctionGenerator {
public:
    /// The scheme of the PDFs (pdfscheme of the run_card), which sets the
    /// collinear remnants of an initial-state emitter.
    enum PdfScheme {
        msbar, ///< MSbar.
        dis    ///< DIS.
    };
    /**
     * @param configurations  The FKS configurations, in the order of
     *                        `fks_index`, with QCD or QED splittings.
     * @param orders          The coupling orders of the Born and of the
     *                        real emission.
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
     * @param pdf_scheme      The scheme of the PDFs.
     * @param n_f             The number of light flavors, in the DIS
     *                        remnant of a gluon.
     * @throws std::invalid_argument as @ref FKSLimits or for parameters out
     *         of their range.
     */
    FKSSubtraction(
        const std::vector<FKSConfiguration>& configurations,
        const FKSCouplingOrders& orders,
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
        double tiny = 1e-6,
        PdfScheme pdf_scheme = msbar,
        double n_f = 5.
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
    std::vector<me_int_t> _remnant, _beam, _kind, _coupling;
    nested_vector2<me_int_t> _born_index;
    std::vector<double> _ratio, _ap_factor, _dis_factor, _delta_i;
    double _xi_s_cut, _delta_s, _tiny;
    PdfScheme _pdf_scheme;
};

/**
 * The n-body part of the FKS subtraction without the virtual: the soft and
 * collinear counterterms of @ref FKSSubtraction integrated over the radiation
 * (those of an initial-state emitter up to their remnants), as
 * bornsoftvirtual in fks_singular.f [2, 3, 4] for QCD and QED, per NLO order
 * (see @ref FKSCouplingOrders). It is a function of the Born alone and takes
 * its blocks as @ref FKSLimits.
 *
 * A splitting type (QCD or QED) contributes where a configuration has it, at
 * the NLO orders of the counterterms of the Born orders, with the Born
 * @f$B@f$ of the Born order of the counterterms at each NLO order. With
 * @f$a = \alpha_s / (2 \pi)@f$ for QCD and @f$\alpha / (2 \pi)@f$ for QED,
 * its part is
 * @f[
 *   a B \sum_j Q_j - 2 a \sum_{m < n} L_{mn} I_{mn} - a \sum_m L_{mm} I_{mm},
 * @f]
 * the first sum over the massless particles with Q terms, the second over
 * the pairs of linked particles and the third over the massive linked ones,
 * with the integrated eikonals @f$I_{mn}@f$ of eikonal_Ireg. For QCD the linked
 * particles are the colored ones with the color-correlated Borns
 * @f$L_{mn} = \langle T_m \cdot T_n \rangle B@f$, those of the soft limit of
 * the first configuration with a gluon i, for QED the charged ones with
 * @f$L_{mn} = \eta_m Q_m \eta_n Q_n B@f$, @f$\eta = -1@f$ for an incoming
 * particle, those of the first configuration with a photon i. The Q term of
 * a final-state particle with the factor @f$C_j@f$ and the @f$\gamma_j@f$ and
 * @f$\gamma'_j@f$ of FKS is
 * @f[
 *   Q_j = \gamma'_j - \log\frac{s \delta_O}{2 \mu_R^2} \Big(\gamma_j -
 *     2 C_j \log\frac{2 E_j}{\xi_\mathrm{cut} \sqrt{s}}\Big) +
 *     2 C_j \Big(\log^2\frac{2 E_j}{\sqrt{s}} - \log^2 \xi_\mathrm{cut}\Big)
 *     - 2 \gamma_j \log\frac{2 E_j}{\sqrt{s}},
 * @f]
 * with its energy @f$E_j@f$ in the partonic center-of-mass frame, and that of
 * an incoming one
 * @f$Q_j = -\log(\mu_F^2 / \mu_R^2) (\gamma_j + 2 C_j \log \xi_\mathrm{cut})@f$
 * with the factorization scale of its beam. For QCD these are the gluons and
 * quarks with their color factors. For QED they are the charged particles
 * with @f$C_j = Q_j^2@f$, @f$\gamma_j = 3 Q_j^2 / 2@f$ and
 * @f$\gamma'_j = (13/2 - 2 \pi^2 / 3) Q_j^2@f$ and the photons with
 * @f$C_j = 0@f$, @f$\gamma_j = -2 (3 \sum_q Q_q^2 + n_l) / 3@f$ and
 * @f$\gamma'_j = -23 (3 \sum_q Q_q^2 + n_l) / 9@f$, the sum over the
 * @f$n_f@f$ light quarks, down-type first, and @f$n_l@f$ massless leptons.
 * Tagged photons (FKSConfiguration::tags, identified photons !a!) have no Q
 * terms, as in bornsoftvirtual. The result is in the
 * normalization of the Born, so that @f$B@f$ plus soft_virtual plus the finite
 * part of the virtual in the conventions of MadFKS is the n-body part per unit
 * Born phase space.
 *
 * The Ellis-Sexton scale of MadFKS, the scale of the finite part of the
 * virtual, is @f$\mu_R@f$ here. With the virtual kept at @f$\mu_R@f$, the
 * scales @f$\mu_R'@f$ and @f$\mu_{F,i}'@f$ of the beams add
 * @f[
 *   c_R \log\frac{\mu_R'^2}{\mu_R^2} +
 *     \sum_i c_{F,i} \log\frac{\mu_{F,i}'^2}{\mu_{F,i}^2},
 * @f]
 * with the coefficients of the weight lines of MadFKS,
 * @f$c_R = a b \gamma_g B@f$ of QCD with the power @f$b@f$ of @f$\alpha_s@f$
 * of the Born of the QCD counterterms and @f$c_{F,i}@f$ the sum over QCD and
 * QED of @f$-a (\gamma_i + 2 C_i \log \xi_\mathrm{cut}) B@f$ of the incoming
 * particle of beam @f$i@f$.
 *
 * `batch` is the leading batch dimension, `n` the number of particles of the
 * Born, incoming included, `b` the number of Born orders and `o` that of NLO
 * orders.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – momenta of the Born, in the
 *   order of the Born of the configurations, in any frame.
 * - `born` – `float`, shape `(batch, b)` – the Born per Born order.
 * - `color_correlated` – `float`, shape `(batch, b, n, n)` – its
 *   color-correlated versions per Born order, as for @ref FKSLimits.
 * - `alpha_s` – `float`, shape `(batch,)` – the strong coupling.
 * - `alpha` – `float`, shape `(batch,)` – the QED coupling.
 * - `ren_scale`, `fact_scale1`, `fact_scale2` – `float`, shape `(batch,)` –
 *   the renormalization scale and the factorization scales of the two beams,
 *   as @ref EnergyScale returns them.
 *
 * **Returns**
 * - `soft_virtual` – `float`, shape `(batch, o)` – the n-body part without the
 *   Born and the virtual per NLO order.
 * - `ren_coefficient` – `float`, shape `(batch, o)` – @f$c_R@f$ per NLO order.
 * - `fact_coefficients` – `float`, shape `(batch, o, 2)` – @f$c_{F,i}@f$ of the
 *   two beams per NLO order.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] S. Frixione, Z. Kunszt, A. Signer, "Three-jet cross sections to
 *   next-to-leading order", https://arxiv.org/abs/hep-ph/9512328
 * - [3] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 * - [4] R. Frederix, S. Frixione, V. Hirschi, D. Pagani, H.-S. Shao,
 *   M. Zaro, "The automation of next-to-leading order electroweak
 *   calculations", https://arxiv.org/abs/1804.10017
 */
class FKSSoftVirtual : public FunctionGenerator {
public:
    /**
     * @param configurations  The FKS configurations of the Born, with QCD or
     *                        QED splittings. The linked particles are the
     *                        soft pairs of the first one with a gluon i (QCD)
     *                        and with a photon i (QED).
     * @param orders          The coupling orders of the Born and of the
     *                        real emission.
     * @param xicut           The cut of the plus distribution in @f$\xi@f$,
     *                        see @ref FKSPrefactors.
     * @param delta_o         The cut of the plus distribution in @f$1 - y@f$
     *                        for a final-state emitter (deltaO).
     * @param n_f             The number of light flavors, in @f$\gamma_g@f$,
     *                        @f$\gamma'_g@f$ and those of the photon.
     * @param n_l             The number of massless leptons, in the
     *                        @f$\gamma@f$ and @f$\gamma'@f$ of the photon.
     * @throws std::invalid_argument as @ref FKSLimits, without a configuration
     *         with a gluon or photon i or for parameters out of their range.
     */
    FKSSoftVirtual(
        const std::vector<FKSConfiguration>& configurations,
        const FKSCouplingOrders& orders,
        double xicut = 0.5,
        double delta_o = 1.,
        double n_f = 5.,
        double n_l = 3.
    );
    /// Number of particles of the Born, incoming included.
    std::size_t particle_count() const { return _masses.size(); }
    /// Number of Born orders.
    std::size_t born_order_count() const { return _born_order_count; }
    /// Number of NLO orders.
    std::size_t nlo_order_count() const { return _ren_factor.size(); }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _born_order_count;
    double _xicut, _delta_o;
    std::vector<double> _masses, _ren_factor;
    nested_vector2<double> _casimir, _gamma, _gamma_prime;
    nested_vector2<me_int_t> _pair_m, _pair_n, _born_index;
    nested_vector2<double> _pair_factor;
};

} // namespace madspace
