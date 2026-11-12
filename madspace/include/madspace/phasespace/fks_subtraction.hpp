#pragma once

#include <optional>
#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/fks_configuration.hpp"
#include "madspace/phasespace/fks_couplings.hpp"
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
 * @f$B@f$ of the order of the counterterms, summed over both splitting types
 * of a configuration with an extra counterterm (the second with its Born),
 * @f[
 *   D_\xi = N \Big[(1 - z) P(z) \log\frac{s \delta_I}{2 \mu_F^2} -
 *     (1 - z) P'(z)\Big], \qquad
 *   D_{l\xi} = 2 N (1 - z) P(z),
 * @f]
 * with the O(epsilon) part @f$P'@f$ of the splitting function, the
 * factorization scale @f$\mu_F@f$ of the beam of the emitter and @f$s@f$ of
 * the counter-event, and the parts @f$K_{p,l,d} = -N k_{p,l,d}@f$ of the
 * change of the PDF scheme from MSbar (xkplus, xklog, xkdelta, zero in MSbar,
 * with their QED parts for a QED splitting, see PdfScheme),
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
 *   color-correlated and spin-correlated versions per Born input (per Born
 *   order for the color-correlated ones), as for @ref FKSLimits.
 * - `collinear_born`, `collinear_spin_correlated` – the Born and its
 *   spin-correlated version per Born input at the Born of the collinear
 *   counter-event (`collinear_born_momenta` of
 *   FKSConfigurationMapping::build_counter_events), which the collinear
 *   limits at the event and at the collinear counter-event and the collinear
 *   remnant there take. Present only with `collinear_born` (lepton beams).
 * - `alpha_s` – `float`, shape `(batch,)` – the strong coupling.
 * - `alpha` – `float`, shape `(batch,)` – the QED coupling.
 * - `ren_scale`, `fact_scale1`, `fact_scale2` – `float`, shape `(batch,)` –
 *   the renormalization scale (the Ellis-Sexton scale of the UPC scheme) and
 *   the factorization scales of the two beams, as @ref EnergyScale returns
 *   them.
 * - `resonance_momentum` – `float`, shape `(batch, 4)` – the momentum of the
 *   resonance of the resonance-local recoil of the event, zero where it has
 *   none, for @ref FKSLimits and @ref FKSPrefactors. Present only with
 *   @p resonance.
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
    /**
     * The scheme of the PDFs (pdfscheme of the run_card), which sets the
     * collinear remnants of an initial-state emitter: the parts
     * @f$k_{p,l,d}@f$ of xkplus, xklog and xkdelta in fks_singular.f, per
     * splitting of the mother m of the Born into j and i. The leptonic schemes
     * of eMELA change only m -> m + gluon or photon (q -> q g/a), the delta
     * scheme also a gluon or photon m from a fermion j, and UPC only a photon
     * j into a fermion pair. Their kernels take the color or charge factor of
     * the splitting, @f$C_F@f$ or @f$Q^2@f$ (@f$N_c Q^2@f$ for UPC).
     */
    enum PdfScheme {
        msbar,  ///< MSbar, no change.
        dis,    ///< DIS.
        eta,    ///< Leptonic eta scheme.
        beta,   ///< Leptonic beta scheme.
        mixed,  ///< Leptonic mixed scheme.
        nobeta, ///< Leptonic nobeta scheme, for LL lepton PDFs.
        delta,  ///< Leptonic delta scheme, for NLL lepton PDFs.
        upc     ///< Photons of ultraperipheral collisions of nuclei, with the
                ///< plus part @f$-(1 - x) (x^2 + (1 - x)^2) \log(\mu^2
                ///< \xi_A^2 R_A^2)@f$ for a photon j into a fermion pair, at
                ///< the Ellis-Sexton scale @f$\mu@f$ (the renormalization
                ///< scale) and the nucleus of the beam of j.
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
     * @param upc_radii       For the UPC scheme, @f$\xi_A R_A@f$ of the
     *                        nucleus of each beam in GeV@f$^{-1}@f$ (xiAI
     *                        times RAI of run.inc, from get_nucleus_RA). Not
     *                        used by the other schemes.
     * @param collinear_born  Take the Born of the collinear counter-event
     *                        apart, for lepton beams (see @ref FKSLimits).
     * @param resonance       Take the momentum of the resonance of the
     *                        resonance-local recoil (`resonance_momentum`,
     *                        see @ref FKSLimits and @ref FKSPrefactors).
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
        double n_f = 5.,
        const std::vector<double>& upc_radii = {},
        bool collinear_born = false,
        bool resonance = false
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
    std::vector<me_int_t> _remnant, _beam, _kind;
    nested_vector2<me_int_t> _coupling;
    nested_vector2<double> _ratio, _ap_factor, _scheme_factor;
    std::vector<double> _delta_i;
    double _xi_s_cut, _delta_s, _tiny;
    PdfScheme _pdf_scheme;
    std::vector<double> _upc_radii;
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
 * A decay has no beams: its massive incoming particle has no Q term and its
 * @f$c_F@f$ are zero, and @f$\sqrt{s}@f$ is its mass. MadFKS gives it the
 * coefficient of log(mu_F^2 / Q^2) of an incoming quark in the weight lines
 * (the loop over nincoming of bornsoftvirtual), which it takes back from the
 * scale-independent part, so that the weight is the same at the scales of
 * the run.
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
    std::size_t _incoming;
    double _xicut, _delta_o;
    std::vector<double> _masses, _ren_factor;
    nested_vector2<double> _casimir, _gamma, _gamma_prime;
    nested_vector2<me_int_t> _pair_m, _pair_n, _born_index;
    nested_vector2<double> _pair_factor;
};

/**
 * The weights of the FKS integrand from the outputs of @ref FKSKinematics,
 * @ref FKSSubtraction and @ref FKSSoftVirtual, as sigint of driver_mintFO.f
 * sums the weight lines of MadFKS [2]. The matrix elements and the
 * subtraction come without the strong coupling, as the weight lines (the
 * real, Born and virtual blocks per order at @f$g_s = 1@f$ and the
 * subtraction at @f$\alpha_s = 1 / (4 \pi)@f$), and each contribution takes
 * @f$g_s^p@f$ of its orders at its own kinematics. Each is
 * @f[
 *   w = c \, L \, F \, \frac{\det}{2 \hat s} \sum_o g_s^{p_o} C_o,
 * @f]
 * with the conversion @f$c@f$ from GeV@f$^{-2}@f$ to pb, the parton
 * luminosity @f$L@f$ and the Jacobian of its kinematics, @f$\hat s@f$ of its
 * momenta, its coefficients @f$C_o@f$ and the factor @f$F@f$: the channel
 * weight, for the real emission and its counterterms times the weight of the
 * sampled FKS configuration (one over its probability). The real emission
 * takes the coupling of the event, the soft and soft-collinear counterterms
 * and the n-body part that of the Born and the collinear counterterm that of
 * its counter-event, as sigint. The n-body part is the Born per Born order
 * plus soft_virtual and the virtual per NLO order at the Born (the virtual
 * trick of MadFKS, if any, is the caller's: it passes the virtual it
 * estimates at the event). The luminosities are those of the incoming
 * partons of each contribution at its momentum fractions, of the real
 * process at the event and the counter-events and of the Born for the n-body
 * part, at the factorization scales of its remnants and soft-virtual part.
 *
 * `batch` is the leading batch dimension, `n` the number of particles of the
 * real processes, incoming included, `b` the number of Born orders and `o`
 * that of NLO orders.
 *
 * **Arguments**
 * - `momenta`, `soft_momenta`, `collinear_momenta`, `soft_collinear_momenta`
 *   – `float`, shape `(batch, n, 4)` – the event and the counter-events.
 * - `weight`, `soft_det`, `collinear_det`, `soft_collinear_det` – `float`,
 *   shape `(batch,)` – their Jacobians.
 * - `born_momenta` – `float`, shape `(batch, n - 1, 4)` – the Born.
 * - `born_weight` – `float`, shape `(batch,)` – its Jacobian.
 * - `real`, `soft`, `collinear`, `soft_collinear` – `float`, shape
 *   `(batch, o)` – the outputs of @ref FKSSubtraction.
 * - `born` – `float`, shape `(batch, b)` – the Born per Born order.
 * - `soft_virtual` – `float`, shape `(batch, o)` – the output of
 *   @ref FKSSoftVirtual.
 * - `virtual` – `float`, shape `(batch, o)` – the finite part of the virtual
 *   in the conventions of MadFKS per NLO order, at the renormalization scale
 *   of soft_virtual.
 * - `event_alpha_s`, `born_alpha_s`, `collinear_alpha_s` – `float`, shape
 *   `(batch,)` – the strong coupling at the event, the Born and the collinear
 *   counter-event.
 * - `event_luminosity`, `soft_luminosity`, `collinear_luminosity`,
 *   `soft_collinear_luminosity`, `born_luminosity` – `float`, shape
 *   `(batch,)` – the parton luminosities, @f$f_1(x_1) f_2(x_2)@f$ summed over
 *   the flavors each stands for (one per flavor for lepton beams).
 * - `channel_weight` – `float`, shape `(batch,)` – the channel weight, as the
 *   multichannel factor of include_multichannel_enhance.
 * - `configuration_weight` – `float`, shape `(batch,)` – the weight of the
 *   sampled FKS configuration, one over its probability.
 *
 * **Returns**
 * - `cross_section` – `float`, shape `(batch,)` – the sum of the others, the
 *   integrand in pb (a width in GeV for a decay, with the flux
 *   @f$1 / (2 M)@f$ of the decaying particle in place of
 *   @f$1 / (2 \hat s)@f$).
 * - `event_weight` – `float`, shape `(batch,)` – the real emission, at
 *   `momenta`.
 * - `soft_weight`, `collinear_weight`, `soft_collinear_weight` – `float`,
 *   shape `(batch,)` – the counterterms, at their counter-events.
 * - `nbody_weight` – `float`, shape `(batch,)` – the n-body part, at
 *   `born_momenta`.
 * - `lo_weight` – `float`, shape `(batch,)` – the Born alone, part of
 *   nbody_weight.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 */
class FKSWeights : public FunctionGenerator {
public:
    /**
     * @param configurations  The FKS configurations of the Born, for the
     *                        number of particles.
     * @param orders          The coupling orders of the Born and of the
     *                        real emission, for the powers of @f$g_s@f$.
     * @throws std::invalid_argument for configurations that
     *         check_fks_configurations rejects.
     */
    FKSWeights(
        const std::vector<FKSConfiguration>& configurations,
        const FKSCouplingOrders& orders
    );
    /// Number of particles of the real processes, incoming included.
    std::size_t particle_count() const { return _particle_count; }
    /// Number of incoming particles, one for a decay.
    std::size_t incoming() const { return _incoming; }
    /// Number of Born orders.
    std::size_t born_order_count() const { return _born_power.size(); }
    /// Number of NLO orders.
    std::size_t nlo_order_count() const { return _nlo_power.size(); }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _particle_count;
    std::size_t _incoming;
    std::vector<me_int_t> _born_power, _nlo_power;
};

/**
 * The generation cuts of the FKS integrand on one kinematics, those of
 * cuts.f of MadFKS (passcuts_leptons and passcuts_jets, with the parameters
 * of the run card) [2], which sigint of driver_mintFO.f applies to the
 * event, to the soft counter-event (for the soft and soft-collinear
 * counterterms and the n-body part) and, without event projection, to the
 * collinear counter-event. The cuts on jets are infrared safe: the QCD
 * partons (quarks up to `maxjetflavor` and gluons) are clustered with the pp
 * generalized @f$k_T@f$ algorithm of fastjet (power `jetalgo`, 1 for
 * @f$k_T@f$, 0 for Cambridge/Aachen, -1 for anti-@f$k_T@f$, radius
 * `jetradius`, rapidity and azimuth, E scheme) into jets with
 * @f$p_T \ge@f$ `ptj` and @f$|\eta| \le@f$ `etaj`. With `ptj` > 0 and more
 * than one parton, at most one parton may be soft and all partons, or all
 * but one, must make jets, so a soft or collinear emission never decides.
 * The charged leptons pass `ptl` and `etal`, each pair of opposite charge
 * `drll` and `mll`, and the pairs of the same flavor `drll_sf` and `mll_sf`.
 * A bound of zero (or below for `etaj` and `etal`) is no cut. The cuts on
 * photons, the photon recombination and the cuts per PDG code of cuts.f are
 * not part of it.
 *
 * `batch` is the leading batch dimension and `n` the number of particles of
 * the real processes, incoming included.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – the event or a counter-event
 *   in the lab frame, in the order of the real process of the configuration.
 * - `fks_index` – `int`, shape `(batch,)` – the FKS configuration of the
 *   event.
 *
 * **Returns**
 * - `mask` – `float`, shape `(batch,)` – 1 where the kinematics passes the
 *   cuts, else 0.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 */
class FKSCuts : public FunctionGenerator {
public:
    /**
     * @param configurations  The FKS configurations of the Born, for the
     *                        PDG codes of their real processes.
     * @param jetalgo         Power of the generalized @f$k_T@f$ algorithm.
     * @param jetradius       Its radius.
     * @param ptj             Least transverse momentum of a jet.
     * @param etaj            Largest @f$|\eta|@f$ of a jet.
     * @param maxjetflavor    Heaviest quark flavor of a jet.
     * @param ptl             Least transverse momentum of a charged lepton.
     * @param etal            Largest @f$|\eta|@f$ of a charged lepton.
     * @param drll            Least @f$\Delta R@f$ of a pair of opposite
     *                        charge.
     * @param drll_sf         The same for a pair of the same flavor.
     * @param mll             Least invariant mass of a pair of opposite
     *                        charge.
     * @param mll_sf          The same for a pair of the same flavor.
     * @throws std::invalid_argument for configurations that
     *         check_fks_configurations rejects or with more QCD partons than
     *         the clustering takes.
     */
    FKSCuts(
        const std::vector<FKSConfiguration>& configurations,
        double jetalgo = 1.,
        double jetradius = 0.7,
        double ptj = 0.,
        double etaj = -1.,
        int maxjetflavor = 4,
        double ptl = 0.,
        double etal = -1.,
        double drll = 0.,
        double drll_sf = 0.,
        double mll = 0.,
        double mll_sf = 0.
    );
    /// Most QCD partons of a real process the clustering takes
    /// (FKS_JET_PARTONS_MAX of the kernel).
    static constexpr std::size_t jet_parton_max = 16;
    /// Number of particles of the real processes, incoming included.
    std::size_t particle_count() const { return _particle_count; }
    /// The options of the kernel: jetalgo, jetradius, ptj, etaj,
    /// maxjetflavor, ptl, etal, drll, drll_sf, mll and mll_sf.
    const std::vector<double>& options() const { return _options; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _particle_count;
    std::size_t _incoming;
    nested_vector2<me_int_t> _pdg;
    std::vector<double> _options;
};

/**
 * The FKS integrand from the kinematics of @ref FKSKinematics and the
 * matrix-element blocks: @ref FKSSubtraction for the real emission and its
 * counterterms, @ref FKSSoftVirtual for the n-body part and @ref FKSWeights
 * for the weights in pb, with the couplings and the luminosity of each
 * kinematics. The blocks come without the strong coupling, the real, the
 * Born and its correlated versions and the virtual per order at
 * @f$g_s = 1@f$, as the weight lines of MadFKS, so that each contribution
 * takes @f$g_s^p@f$ at its own kinematics (the subtraction and the n-body
 * part run at @f$\alpha_s = 1 / (4 \pi)@f$). The electroweak couplings are in
 * the blocks and `alpha`.
 *
 * **Arguments**
 * - those of the subtraction without `alpha_s`, with `ren_scale` the
 *   renormalization scale of the Born (also the Ellis-Sexton scale of the
 *   n-body part and of the UPC scheme), `fact_scale1` and `fact_scale2` the
 *   factorization scales of the Born for the n-body part and
 *   `collinear_fact_scale1` and `collinear_fact_scale2` those of the
 *   collinear remnants.
 * - those of the weights that the subtraction does not take: `born_momenta`,
 *   `born_weight`, `virtual` (at `ren_scale`), the couplings, the
 *   luminosities, `channel_weight` and `configuration_weight`. The Born of
 *   the subtraction gives that of the n-body part, its first Born orders
 *   where the subtraction also takes the Borns of extra counterterms.
 * - `fks_index` last, as for the subtraction.
 *
 * With @ref FKSCuts, the luminosity of each contribution takes the cut of its
 * kinematics, as sigint: the real emission that of the event, the soft and
 * soft-collinear counterterms and the n-body part that of the soft
 * counter-event, and the collinear counterterm that of the soft
 * counter-event with event projection and of its own without (where the
 * subtraction takes `collinear_born`). The n-body part takes the soft
 * counter-event of the configuration of the event, where MadFKS takes that
 * of its n-body configuration: for QCD they differ only in the flavor of the
 * soft parton, which the cuts never see.
 *
 * With @ref FKSScale, a @ref RunningCoupling and the @ref FKSLuminosity of
 * the real emission and of the Born, the integrand takes the scales, the
 * couplings and the luminosities itself, as sigint: the scales of the Born
 * for the n-body part and the soft and soft-collinear counterterms (MadFKS
 * takes them at the soft counter-event, whose soft parton adds nothing),
 * those of the collinear counter-event for the collinear counterterm and the
 * collinear remnants, and those of the event for the real emission, with
 * @f$\alpha_s@f$ at the renormalization scale and the luminosity at the
 * momentum fractions and the factorization scales of each. Its arguments
 * then have no scales, couplings and luminosities but the momentum fractions
 * of the kinematics of @ref FKSKinematics in their place: `x1`, `x2`,
 * `soft_x1`, `soft_x2`, `collinear_x1`, `collinear_x2`, `soft_collinear_x1`,
 * `soft_collinear_x2`, `born_x1` and `born_x2`, and with luminosities of one
 * flavor `flavor_index` before `fks_index`, the flavor of the event among the
 * real flavors of its configuration (see @ref FKSLuminosity). The soft-collinear
 * remnants take the factorization scales of the collinear counter-event, which equal
 * those of the Born for the scale choices of @ref FKSScale.
 *
 * **Returns**
 * - those of @ref FKSWeights.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 */
class FKSCrossSection : public FunctionGenerator {
public:
    /**
     * @param subtraction   The real emission and its counterterms.
     * @param soft_virtual  The n-body part without the Born and the virtual.
     * @param weights       The weights of the contributions.
     * @param cuts          The generation cuts, if any.
     * @throws std::invalid_argument if they do not all have the same numbers
     *         of particles and orders.
     */
    FKSCrossSection(
        const FKSSubtraction& subtraction,
        const FKSSoftVirtual& soft_virtual,
        const FKSWeights& weights,
        const std::optional<FKSCuts>& cuts = std::nullopt
    );
    /**
     * With the scales, the couplings and the luminosities inside.
     * @param subtraction       The real emission and its counterterms.
     * @param soft_virtual      The n-body part without the Born and the
     *                          virtual.
     * @param weights           The weights of the contributions.
     * @param cuts              The generation cuts, if any.
     * @param scale             The scales of each kinematics, for any number
     *                          of particles.
     * @param running_coupling  @f$\alpha_s@f$ at the renormalization scale.
     * @param luminosity        The luminosity of the real emission and its
     *                          counter-events, per FKS configuration.
     * @param born_luminosity   That of the n-body part, per FKS configuration
     *                          (for one flavor, by the real flavor), or once
     *                          for the sum over the flavors.
     * @throws std::invalid_argument as the other constructor, or if the
     *         luminosities do not have the FKS configurations of the
     *         subtraction, or one sums over the flavors and the other not.
     */
    FKSCrossSection(
        const FKSSubtraction& subtraction,
        const FKSSoftVirtual& soft_virtual,
        const FKSWeights& weights,
        const std::optional<FKSCuts>& cuts,
        const FKSScale& scale,
        const RunningCoupling& running_coupling,
        const FKSLuminosity& luminosity,
        const FKSLuminosity& born_luminosity
    );
    /// The real emission and its counterterms.
    const FKSSubtraction& subtraction() const { return _subtraction; }
    /// The n-body part without the Born and the virtual.
    const FKSSoftVirtual& soft_virtual() const { return _soft_virtual; }
    /// The weights of the contributions.
    const FKSWeights& weights() const { return _weights; }
    /// The generation cuts, if any.
    const std::optional<FKSCuts>& cuts() const { return _cuts; }
    /// The scales of the kinematics with n + 1 particles, if inside.
    const std::optional<FKSScale>& scale() const { return _scale; }
    /// @f$\alpha_s@f$, if inside.
    const std::optional<RunningCoupling>& running_coupling() const {
        return _running_coupling;
    }
    /// The luminosity of the real emission, if inside.
    const std::optional<FKSLuminosity>& luminosity() const { return _luminosity; }
    /// The luminosity of the n-body part, if inside.
    const std::optional<FKSLuminosity>& born_luminosity() const {
        return _born_luminosity;
    }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    FKSSubtraction _subtraction;
    FKSSoftVirtual _soft_virtual;
    FKSWeights _weights;
    std::optional<FKSCuts> _cuts;
    std::optional<FKSScale> _scale, _born_scale;
    std::optional<RunningCoupling> _running_coupling;
    std::optional<FKSLuminosity> _luminosity, _born_luminosity;
};

} // namespace madspace
