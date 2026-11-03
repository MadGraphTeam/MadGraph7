#pragma once

#include <utility>
#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/fks_configuration.hpp"
#include "madspace/util.hpp"

namespace madspace {

/**
 * The reduced eikonal factors of the soft limit of the FKS configurations of
 * a Born, as eikonal_reduced in fks_singular.f [2, 3], each event in the
 * configuration of its `fks_index` (see @ref FKSConfigurationMapping).
 *
 * Where i becomes soft, the real matrix element times @f$\xi^2 (1 - y)@f$
 * tends to a sum over the pairs @f$(m, n)@f$ of the color (or charge)
 * partners of i,
 * @f[
 *   \xi^2 (1 - y) |M_{n+1}|^2 \to
 *   -2 \sum_{(m, n)} \mathcal{E}_{mn} B_{mn}, \qquad
 *   \mathcal{E}_{mn} = \frac{(p_m p_n) (1 - y)}{(p_m \hat p_i)(p_n \hat p_i)},
 * @f]
 * with the color-linked Borns @f$B_{mn}@f$ in the conventions of MadFKS
 * (sbornsoft) and the momentum of i per unit @f$\xi@f$, @f$\hat p_i@f$
 * (`p_i_hat` of @ref FKSKinematics). For a massless j,
 * @f$(p_j \hat p_i) / (1 - y) = \sqrt{s} E_j / 2@f$ in the partonic
 * center-of-mass frame, so that the factors are finite at @f$y = 1@f$. The
 * pairs are those of sbornsoft, in its order: every unordered pair of the
 * particles that form an FKS pair with i, a particle with itself only where it
 * is massive. They are evaluated on the momenta of the soft counter-event.
 *
 * `batch` is the leading batch dimension, `n` the number of particles of the
 * real process, incoming included, and `k` the largest number of pairs of a
 * configuration (at least one).
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – momenta of the soft
 *   counter-event, in the order of the real process of the configuration of
 *   the event, in any frame.
 * - `p_i_hat` – `float`, shape `(batch, 4)` – momentum of i per unit
 *   @f$\xi@f$ at the soft counter-event.
 * - `y` – `float`, shape `(batch,)` – cosine of the angle between i and j.
 * - `fks_index` – `int`, shape `(batch,)` – the configuration of the event,
 *   from 0 to the number of configurations minus one.
 *
 * **Returns**
 * - `eikonals` – `float`, shape `(batch, k)` – @f$\mathcal{E}_{mn}@f$ of the
 *   pairs of the configuration (see pairs()), zero past them.
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
class FKSEikonals : public FunctionGenerator {
public:
    /**
     * @param configurations  The FKS configurations, in the order of
     *                        `fks_index` (their FKS pairs and masses).
     */
    explicit FKSEikonals(const std::vector<FKSConfiguration>& configurations);
    /// Number of configurations.
    std::size_t configuration_count() const { return _i_fks.size(); }
    /// Number of particles of the real processes, incoming included.
    std::size_t particle_count() const { return _particle_count; }
    /// Number of entries of `eikonals`, the largest number of pairs of a
    /// configuration (at least one).
    std::size_t max_pair_count() const { return _max_pair_count; }
    /// The pairs (m, n) of a configuration, positions in its real process,
    /// in the order of `eikonals`.
    const std::vector<std::pair<std::size_t, std::size_t>>&
    pairs(std::size_t index) const {
        return _pairs.at(index);
    }
    /// The pairs (m, n) of a configuration as positions in its Born, whose
    /// color-linked Borns they multiply (j is at the position of i + j).
    std::vector<std::pair<std::size_t, std::size_t>>
    born_pairs(std::size_t index) const;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _particle_count;
    std::size_t _max_pair_count;
    std::vector<std::size_t> _i_fks;
    std::vector<std::size_t> _j_fks;
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> _pairs;
    nested_vector2<double> _masses;
};

/**
 * The collinear factors of the FKS configurations of a Born, as sborncol_fsr
 * and sborncol_isr in fks_singular.f [2, 3], each event in the configuration of its
 * `fks_index` (see @ref FKSConfigurationMapping).
 *
 * Where i becomes collinear to a massless j, the real matrix element times
 * @f$\xi^2 (1 - y)@f$ tends to the Born times the Altarelli-Parisi kernel of
 * the splitting plus, for a gluon (or photon) i + j, the azimuthal terms with
 * the spin-correlated Born,
 * @f[
 *   \frac{(1 - z) P(z)}{t}, \qquad \frac{Q(z)}{t}, \qquad t = \frac{z s}{4},
 * @f]
 * as AP_reduced and Qterms_reduced_timelike (final-state j) or
 * Qterms_reduced_spacelike (initial-state j) give them, without the couplings
 * (QCD and QED parts apart). The splitting is that of j into i and j for a
 * final-state j, and that of j into i and the mother i + j, which enters the
 * Born, for an initial-state one. Its momentum fraction is @f$z = 1 - E_i /
 * (E_i + E_j)@f$ or @f$z = 1 - \xi = 1 - 2 E_i / \sqrt{s}@f$ in the partonic
 * center-of-mass frame of the counter-event. The azimuthal terms go with the
 * spin-correlated Born contracted with the transverse direction of the emission
 * @f$k_\perp@f$ (xij_kperp of genps_fks.f): the momentum of i per unit @f$\xi@f$ at the
 * event made orthogonal to the Born emitter and to the sum of the incoming
 * Born momenta, @f$k_\perp^2 = -1@f$.
 *
 * `batch` is the leading batch dimension and `n` the number of particles of
 * the real process, incoming included.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – momenta of the collinear or
 *   soft-collinear counter-event, in the order of the real process of the
 *   configuration of the event, in any frame.
 * - `p_i_hat` – `float`, shape `(batch, 4)` – momentum of i per unit
 *   @f$\xi@f$ at the event.
 * - `born_momenta` – `float`, shape `(batch, n - 1, 4)` – momenta of the Born,
 *   in the frame of `p_i_hat`.
 * - `fks_index` – `int`, shape `(batch,)` – the configuration of the event,
 *   from 0 to the number of configurations minus one.
 *
 * **Returns**
 * - `z` – `float`, shape `(batch,)` – momentum fraction of the splitting.
 * - `ap` – `float`, shape `(batch, 2)` – QCD and QED Altarelli-Parisi kernels.
 * - `q_terms` – `float`, shape `(batch, 2)` – QCD and QED azimuthal terms.
 * - `k_perp` – `float`, shape `(batch, 4)` – transverse direction of the
 *   emission, in the frame of the momenta.
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
class FKSSplittingKernels : public FunctionGenerator {
public:
    /**
     * @param configurations  The FKS configurations, in the order of
     *                        `fks_index` (their colors and charges).
     */
    explicit FKSSplittingKernels(const std::vector<FKSConfiguration>& configurations);
    /// Number of configurations.
    std::size_t configuration_count() const { return _i_fks.size(); }
    /// Number of particles of the real processes, incoming included.
    std::size_t particle_count() const { return _particle_count; }
    /// The splitting function of a configuration as in AP_reduced: 0 for
    /// g -> g g, 1 for g/a -> q q~, 2 for q -> q g/a, 3 for q -> g/a q.
    int splitting_kind(std::size_t index) const { return _kind.at(index); }
    /// The QCD and QED factors of the Altarelli-Parisi kernel and of the
    /// azimuthal terms of a configuration.
    const std::vector<double>& factors(std::size_t index) const {
        return _factors.at(index);
    }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _particle_count;
    std::vector<std::size_t> _i_fks;
    std::vector<std::size_t> _j_fks;
    std::vector<int> _kind;
    nested_vector2<double> _factors;
};

/**
 * The soft and collinear limits of the real matrix element of the FKS
 * configurations of a Born from the Born and its color-correlated and
 * spin-correlated versions, as sbornsoft, sborncol_fsr and sborncol_isr in
 * fks_singular.f give them [2, 3], each event in the configuration of its
 * `fks_index` (see @ref FKSConfigurationMapping). These are the limits of
 * @f$\xi^2 (1 - y)@f$ times the real matrix element (sreal at the
 * counter-events), for QCD splittings.
 *
 * With @f$g_s^2 = 4 \pi \alpha_s@f$, the identical-particle ratio @f$r@f$ of
 * the configuration (FKSConfiguration::identical_particle_ratio()), the Born
 * @f$B@f$ and its color-correlated versions
 * @f$B_{mn} = \langle T_m \cdot T_n \rangle B@f$ (all-outgoing color charges,
 * @f$B_{mm} = C_m B@f$),
 * @f[
 *   \text{soft} = -2 r g_s^2 \sum_{(m, n)} w_{mn} \mathcal{E}_{mn} B_{mn},
 *   \qquad
 *   \text{collinear} = r g_s^2 \Big[\frac{(1 - z) P(z)}{t} B +
 *     \frac{Q(z)}{t} \Big(\frac{B}{2} - B_\perp\Big)\Big],
 * @f]
 * with the reduced eikonal factors @f$\mathcal{E}_{mn}@f$ of @ref FKSEikonals
 * at the soft counter-event over the pairs of the configuration in the Born
 * (FKSEikonals::born_pairs()), @f$w_{mn} = 1@f$ and @f$w_{mm} = 1/2@f$, and
 * the QCD kernels of @ref FKSSplittingKernels at the collinear or soft-collinear
 * counter-event. @f$B_\perp@f$ is the Born with the polarization vector of the
 * emitter (its position FKSConfiguration::born_j_fks() in the Born, a gluon
 * where @f$Q \ne 0@f$) replaced by @f$k_\perp@f$ of FKSSplittingKernels, a
 * real vector with @f$k_\perp^2 = -1@f$, with the usual averages and
 * symmetry factors. The azimuthal average of @f$B_\perp@f$ is @f$B / 2@f$.
 * The soft limit is zero where i is a quark, the collinear ones are zero for a
 * massive emitter.
 *
 * `batch` is the leading batch dimension and `n` the number of particles of
 * the real process, incoming included.
 *
 * **Arguments**
 * - `soft_momenta` – `float`, shape `(batch, n, 4)` – momenta of the soft
 *   counter-event, in the order of the real process of the configuration of
 *   the event, in any frame.
 * - `soft_p_i_hat` – `float`, shape `(batch, 4)` – momentum of i per unit
 *   @f$\xi@f$ at the soft counter-event.
 * - `y` – `float`, shape `(batch,)` – @f$y@f$ of the event.
 * - `collinear_momenta`, `soft_collinear_momenta` – `float`, shape
 *   `(batch, n, 4)` – momenta of the collinear and soft-collinear
 *   counter-events, in any frame.
 * - `born` – `float`, shape `(batch,)` – the Born @f$B@f$.
 * - `color_correlated` – `float`, shape `(batch, n - 1, n - 1)` –
 *   @f$B_{mn}@f$ for the particles of the Born.
 * - `spin_correlated` – `float`, shape `(batch,)` – @f$B_\perp@f$, finite.
 * - `alpha_s` – `float`, shape `(batch,)` – the strong coupling.
 * - `fks_index` – `int`, shape `(batch,)` – the configuration of the event,
 *   from 0 to the number of configurations minus one.
 *
 * **Returns**
 * - `soft` – `float`, shape `(batch,)` – the soft limit.
 * - `collinear`, `soft_collinear` – `float`, shape `(batch,)` – the
 *   collinear limit at the collinear and at the soft-collinear counter-event.
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
class FKSLimits : public FunctionGenerator {
public:
    /**
     * @param configurations  The FKS configurations, in the order of
     *                        `fks_index`, with QCD splittings.
     * @throws std::invalid_argument for a splitting that is not one of QCD.
     */
    explicit FKSLimits(const std::vector<FKSConfiguration>& configurations);
    /// Number of configurations.
    std::size_t configuration_count() const { return _collinear_factor.size(); }
    /// Number of particles of the real processes, incoming included.
    std::size_t particle_count() const { return _eikonals.particle_count(); }
    /// Whether a configuration has a soft limit (i is a gluon).
    bool soft(std::size_t index) const { return _pair_count.at(index) > 0; }
    /// Whether a configuration has collinear limits (j is massless).
    bool collinear(std::size_t index) const {
        return _collinear_factor.at(index) != 0.;
    }
    /// The eikonal factors of the soft limits.
    const FKSEikonals& eikonals() const { return _eikonals; }
    /// The kernels of the collinear limits.
    const FKSSplittingKernels& splitting_kernels() const { return _kernels; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    FKSEikonals _eikonals;
    FKSSplittingKernels _kernels;
    std::vector<me_int_t> _fks_i;
    std::vector<me_int_t> _fks_j;
    std::vector<me_int_t> _pair_count;
    nested_vector2<me_int_t> _pair_m;
    nested_vector2<me_int_t> _pair_n;
    nested_vector2<double> _pair_factor;
    std::vector<double> _collinear_factor;
};

/**
 * The prefactors of the real emission and its FKS counterterms, as
 * compute_prefactors_n1body in fks_singular.f [2, 3], each event in the
 * configuration of its `fks_index` (see @ref FKSConfigurationMapping).
 *
 * They are the plus distributions in @f$\xi@f$ and @f$1 - y@f$ with their
 * cuts @f$\xi_\mathrm{cut}@f$, @f$\xi_\mathrm{S,cut}@f$, @f$\delta@f$
 * (@f$\delta_O@f$ for a final-state emitter, @f$\delta_I@f$ for an
 * initial-state one) and @f$\delta_S@f$, for the Jacobians of the event and
 * counter-events without the factor @f$\xi@f$ (`weight / xi` and the `det`s
 * of @ref FKSKinematics, which hold the width of the range of @f$\xi@f$ that
 * MadFKS writes as `xinorm`) and without the symmetry factors. With
 * @f$\xi_l = \min(\xi_\mathrm{max}, \xi_\mathrm{S,cut})@f$,
 * @f$L_\delta = \log(\delta / \delta_S) / \delta_S@f$ and @f$\xi_c@f$,
 * @f$\xi_{l,c}@f$, @f$s_c@f$ those of the collinear counter-event,
 * @f[
 *   f_r = \frac{1}{\xi (1 - y)}, \quad
 *   f_s = \Big(\frac{1}{\xi} + \frac{\log(\xi_\mathrm{cut} / \xi_l)}{\xi_l}\Big)
 *         \frac{1}{1 - y}, \quad
 *   f_c = \Big(\frac{1}{1 - y} + L_\delta\Big) \frac{1}{\xi_c}, \quad
 *   f_{sc} = \Big(\frac{1}{1 - y} + L_\delta\Big)
 *         \Big(\frac{1}{\xi_c} + \frac{\log(\xi_\mathrm{cut} /
 * \xi_{l,c})}{\xi_{l,c}}\Big),
 * @f]
 * and for the collinear remnants @f$f_{dc} = 32 \pi^2 / (s_c \xi_c \delta_S)@f$
 * and the four @f$f_{dsc}@f$ (with @f$32 \pi^2 / s@f$ of the soft-collinear
 * counter-event). They multiply @f$\xi^2 (1 - y)@f$ times the real matrix
 * element and its limits. The collinear ones vanish for a massive emitter.
 * Whether a counterterm contributes (@f$\xi \le \xi_\mathrm{S,cut}@f$,
 * @f$y > 1 - \delta_S@f$) is left to @ref FKSSubtraction, which also adds
 * the symmetry factors.
 *
 * `batch` is the leading batch dimension and `n` the number of particles of
 * the real process, incoming included.
 *
 * **Arguments**
 * - `xi`, `y`, `xi_max` – `float`, shape `(batch,)` – the FKS variables of
 *   the event.
 * - `collinear_xi`, `collinear_xi_max` – `float`, shape `(batch,)` –
 *   @f$\xi@f$ and the upper end of its range at the collinear counter-event.
 * - `collinear_momenta`, `soft_collinear_momenta` – `float`, shape
 *   `(batch, n, 4)` – momenta of the collinear and soft-collinear
 *   counter-events, in any frame.
 * - `fks_index` – `int`, shape `(batch,)` – the configuration of the event,
 *   from 0 to the number of configurations minus one.
 *
 * **Returns**
 * - `f_r`, `f_s`, `f_c`, `f_dc`, `f_sc` – `float`, shape `(batch,)`.
 * - `f_dsc` – `float`, shape `(batch, 4)`.
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
class FKSPrefactors : public FunctionGenerator {
public:
    /**
     * @param configurations  The FKS configurations, in the order of
     *                        `fks_index` (their emitters and masses).
     * @param xicut           The cut of the plus distribution in @f$\xi@f$.
     * @param xi_s_cut        The upper end of the soft counterterm in
     *                        @f$\xi@f$ (xiScut).
     * @param delta_o         The cut of the plus distribution in @f$1 - y@f$
     *                        for a final-state emitter (deltaO).
     * @param delta_i         The same for an initial-state emitter (deltaI).
     * @param delta_s         The lower end of the collinear counterterms in
     *                        @f$1 - y@f$ (deltaS).
     */
    FKSPrefactors(
        const std::vector<FKSConfiguration>& configurations,
        double xicut = 0.5,
        double xi_s_cut = 0.5,
        double delta_o = 1.,
        double delta_i = 1.,
        double delta_s = 1.
    );
    /// Number of configurations.
    std::size_t configuration_count() const { return _j_fks.size(); }
    /// Number of particles of the real processes, incoming included.
    std::size_t particle_count() const { return _particle_count; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _particle_count;
    std::vector<std::size_t> _j_fks;
    nested_vector2<double> _masses;
    double _xicut, _xi_s_cut, _delta_o, _delta_i, _delta_s;
};

} // namespace madspace
