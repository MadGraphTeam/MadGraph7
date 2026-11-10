#pragma once

#include <string>
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
 * center-of-mass frame, so that the factors are finite at @f$y = 1@f$. With
 * the resonance-local recoil (@p resonance, see @ref FKSConfigurationMapping)
 * @f$\xi@f$, @f$y@f$ and @f$\hat p_i@f$ are those of the rest frame of the
 * resonance of momentum @f$K@f$, and @f$(p_j \hat p_i) / (1 - y) =
 * (K p_j) / 2@f$ where the event has one, as eikonal_reduced of MadGraph7
 * takes it. The
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
 * - `resonance_momentum` – `float`, shape `(batch, 4)` – the momentum
 *   @f$K@f$ of the resonance of the event, in the frame of the momenta, zero
 *   where it has none (`resonance_momentum` of
 *   FKSConfigurationMapping::build_counter_events). Present only with
 *   @p resonance.
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
     * @param resonance       Take the momentum of the resonance of the
     *                        resonance-local recoil (`resonance_momentum`).
     */
    explicit FKSEikonals(
        const std::vector<FKSConfiguration>& configurations, bool resonance = false
    );
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
    bool _resonance;
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
 * Born momenta, @f$k_\perp^2 = -1@f$. With the resonance-local recoil
 * (@p resonance, see @ref FKSConfigurationMapping) a final-state j takes the
 * rest frame of the resonance of momentum @f$K@f$ where the event has one, as
 * sborncol_fsr of MadGraph7: the energies
 * there, @f$t = z K^2 / 4@f$ and @f$k_\perp@f$ orthogonal to @f$K@f$ in
 * place of the incoming momenta, the transverse direction in the rest frame of
 * the resonance boosted with the momenta (resonance_collinear_phase of the
 * branch).
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
 * - `resonance_momentum` – `float`, shape `(batch, 4)` – the momentum
 *   @f$K@f$ of the resonance of the event, zero where it has none. Present
 *   only with @p resonance.
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
     * @param resonance       Take the momentum of the resonance of the
     *                        resonance-local recoil (`resonance_momentum`).
     */
    explicit FKSSplittingKernels(
        const std::vector<FKSConfiguration>& configurations, bool resonance = false
    );
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
    bool _resonance;
};

/**
 * The soft and collinear limits of the real matrix element of the FKS
 * configurations of a Born from the Born and its color-correlated and
 * spin-correlated versions, as sbornsoft, sborncol_fsr and sborncol_isr in
 * fks_singular.f give them [2, 3, 4], each event in the configuration of its
 * `fks_index` (see @ref FKSConfigurationMapping). These are the limits of
 * @f$\xi^2 (1 - y)@f$ times the real matrix element (sreal at the
 * counter-events) per NLO order, for QCD and QED splittings.
 *
 * The splitting type of a configuration (FKSConfiguration::splitting_types)
 * sets the coupling, @f$g^2 = 4 \pi \alpha_s@f$ for QCD and @f$4 \pi \alpha@f$
 * for QED, and the kernels. Its limits at an NLO order take the Born of the
 * order that FKSCouplingOrders::counterterm_order() maps to it and are zero
 * where it maps none. A configuration with an extra counterterm (a quark pair
 * from a gluon or a photon, extra_cnt of MadFKS [4]) has the collinear limits
 * of both splitting types, the second one with the Born of the extra
 * counterterm (FKSConfiguration::born_pdgs() of the second type) of the order
 * that FKSCouplingOrders::extra_counterterm_order() maps to the NLO order, and
 * no soft limit. With the identical-particle ratio @f$r@f$ of the
 * configuration and its Born (FKSConfiguration::identical_particle_ratio())
 * and the Born @f$B@f$ of that order,
 * @f[
 *   \text{soft} = -2 r g^2 \sum_{(m, n)} w_{mn} \mathcal{E}_{mn} B_{mn},
 *   \qquad
 *   \text{collinear} = r g^2 \Big[\frac{(1 - z) P(z)}{t} B +
 *     \frac{Q(z)}{t} \Big(\frac{B}{2} - B_\perp\Big)\Big],
 * @f]
 * with the reduced eikonal factors @f$\mathcal{E}_{mn}@f$ of @ref FKSEikonals
 * at the soft counter-event over the pairs of the configuration in the Born
 * (FKSEikonals::born_pairs()), @f$w_{mn} = 1@f$ and @f$w_{mm} = 1/2@f$, and
 * the kernels of the splitting type of @ref FKSSplittingKernels at the
 * collinear or soft-collinear counter-event. For a gluon i
 * @f$B_{mn} = \langle T_m \cdot T_n \rangle B@f$ are the color-correlated
 * Borns (all-outgoing color charges, @f$B_{mm} = C_m B@f$), for a photon i
 * @f$B_{mn} = \eta_m Q_m \eta_n Q_n B@f$ with the charges @f$Q@f$ and
 * @f$\eta = -1@f$ for an incoming particle, 1 otherwise. @f$B_\perp@f$ is the
 * Born with the polarization vector of the emitter (its position
 * FKSConfiguration::born_j_fks() in the Born, a gluon or photon where
 * @f$Q \ne 0@f$) replaced by @f$k_\perp@f$ of FKSSplittingKernels, a real
 * vector with @f$k_\perp^2 = -1@f$, with the usual averages and symmetry
 * factors. The azimuthal average of @f$B_\perp@f$ is @f$B / 2@f$. The soft
 * limit is zero where i is not a gluon or a photon, the collinear ones are
 * zero for a massive emitter.
 *
 * `batch` is the leading batch dimension, `n` the number of particles of the
 * real process, incoming included, `b` the number of Born orders, `B` that of
 * the Born inputs (FKSCouplingOrders::born_input_count(): the Born orders,
 * then the orders of the Born of each extra counterterm) and `o` that of NLO
 * orders.
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
 * - `born` – `float`, shape `(batch, B)` – the Born @f$B@f$ per Born input,
 *   those of the extra counterterms of other configurations unused.
 * - `color_correlated` – `float`, shape `(batch, b, n - 1, n - 1)` –
 *   @f$\langle T_m \cdot T_n \rangle B@f$ for the particles of the Born per
 *   Born order.
 * - `spin_correlated` – `float`, shape `(batch, B)` – @f$B_\perp@f$ per Born
 *   input, finite.
 * - `collinear_born`, `collinear_spin_correlated` – `float`, shape
 *   `(batch, B)` – @f$B@f$ and @f$B_\perp@f$ at the Born of the collinear
 *   counter-event, which the collinear limit there takes. Present only with
 *   `collinear_born` (lepton beams, where the Born of an incoming emitter
 *   differs between the soft and the collinear counter-events, see
 *   @ref FKSConfigurationMapping).
 * - `alpha_s` – `float`, shape `(batch,)` – the strong coupling.
 * - `alpha` – `float`, shape `(batch,)` – the QED coupling.
 * - `resonance_momentum` – `float`, shape `(batch, 4)` – the momentum of the
 *   resonance of the resonance-local recoil of the event, zero where it has
 *   none, for @ref FKSEikonals and @ref FKSSplittingKernels. Present only
 *   with @p resonance.
 * - `fks_index` – `int`, shape `(batch,)` – the configuration of the event,
 *   from 0 to the number of configurations minus one.
 *
 * **Returns**
 * - `soft` – `float`, shape `(batch, o)` – the soft limit per NLO order.
 * - `collinear`, `soft_collinear` – `float`, shape `(batch, o)` – the
 *   collinear limit at the collinear and at the soft-collinear counter-event
 *   per NLO order.
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
class FKSLimits : public FunctionGenerator {
public:
    /**
     * @param configurations  The FKS configurations, in the order of
     *                        `fks_index`, with QCD or QED splittings.
     * @param orders          The coupling orders of the Born and of the
     *                        real emission.
     * @param collinear_born  Take the Born of the collinear counter-event
     *                        apart (`collinear_born` and
     *                        `collinear_spin_correlated`), for lepton beams.
     * @param resonance       Take the momentum of the resonance of the
     *                        resonance-local recoil (`resonance_momentum`).
     * @throws std::invalid_argument for a splitting that is not one of QCD or
     *         QED, an extra counterterm that the coupling orders do not have
     *         or coupling orders without the splitting types.
     */
    FKSLimits(
        const std::vector<FKSConfiguration>& configurations,
        const FKSCouplingOrders& orders,
        bool collinear_born = false,
        bool resonance = false
    );
    /// Number of configurations.
    std::size_t configuration_count() const { return _collinear_factor.size(); }
    /// Number of particles of the real processes, incoming included.
    std::size_t particle_count() const { return _eikonals.particle_count(); }
    /// Number of Born orders.
    std::size_t born_order_count() const { return _born_order_count; }
    /// Number of Born inputs, the Born orders and those of the extra
    /// counterterms.
    std::size_t born_input_count() const { return _born_input_count; }
    /// Number of NLO orders.
    std::size_t nlo_order_count() const { return _nlo_order_count; }
    /// Whether the collinear counter-event takes a Born of its own.
    bool collinear_born() const { return _collinear_born; }
    /// Whether it takes the momentum of the resonance of the resonance-local
    /// recoil.
    bool resonance() const { return _resonance; }
    /// Whether a configuration has a soft limit (i is a gluon or a photon).
    bool soft(std::size_t index) const { return _pair_count.at(index) > 0; }
    /// Whether a configuration has collinear limits (j is massless).
    bool collinear(std::size_t index) const {
        return _collinear_factor.at(index).at(0) != 0.;
    }
    /// Whether the (first) splitting of a configuration is one of QED, else
    /// QCD.
    bool qed(std::size_t index) const { return coupling(index) == 1; }
    /// The coupling of a splitting type of a configuration (0 for the Born, 1
    /// for the extra counterterm): 0 QCD, 1 QED, -1 for none.
    int coupling(std::size_t index, std::size_t splitting = 0) const {
        return static_cast<int>(_coupling.at(index).at(splitting));
    }
    /// For each NLO order, the Born input of the counterterms of a splitting
    /// type of a configuration (0 for the Born, a Born order, 1 for the extra
    /// counterterm), -1 for none.
    const std::vector<me_int_t>&
    born_index(std::size_t index, std::size_t splitting = 0) const {
        return _born_index.at(index).at(splitting);
    }
    /// born_index() of every configuration and both splitting types as a
    /// constant of shape (configurations, 2, NLO orders) for the kernels.
    Value born_index_table() const;
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
    std::size_t _born_order_count;
    std::size_t _born_input_count;
    std::size_t _nlo_order_count;
    bool _collinear_born;
    bool _resonance;
    std::vector<me_int_t> _fks_i;
    std::vector<me_int_t> _fks_j;
    std::vector<me_int_t> _pair_count;
    nested_vector2<me_int_t> _pair_m;
    nested_vector2<me_int_t> _pair_n;
    nested_vector2<double> _pair_factor;
    nested_vector2<double> _collinear_factor;
    nested_vector2<me_int_t> _coupling;
    nested_vector3<me_int_t> _born_index;
};

/**
 * Check FKS configurations with QCD or QED splittings for the limits of the
 * real matrix element and the coupling orders they come with: those of
 * check_fks_configurations(), splitting types whose particles i and j are
 * colored for QCD and photons or charged for QED, a splitting of i and j
 * that gives a Born for each, an extra counterterm of the coupling orders for
 * a quark i only, and the coupling orders of check_fks_coupling_orders() with
 * the splitting types among their split orders.
 *
 * @param configurations  The configurations.
 * @param orders          The coupling orders.
 * @param user            Name of the caller, for the error messages.
 * @return                The number of particles of the real processes.
 */
std::size_t check_splitting_configurations(
    const std::vector<FKSConfiguration>& configurations,
    const FKSCouplingOrders& orders,
    const std::string& user
);

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
 * counter-event), and those of a change of the PDF scheme at the
 * soft-collinear counter-event, @f$32 \pi^2 / (s \delta_S \xi_{l,c})@f$
 * times 1, @f$\log \xi_{l,c}@f$ and @f$\log^2 \xi_{l,c} / 2@f$, which give
 * @f$[1/(1 - z)]_+@f$, @f$[\log(1 - z)/(1 - z)]_+@f$ and
 * @f$\delta(1 - z)@f$ on the range of @f$z = 1 - \xi_c@f$. MadFKS
 * (f_pdfsch_d, f_pdfsch_p, f_pdfsch_l) takes @f$\xi_\mathrm{S,cut}@f$ for
 * @f$\xi_{l,c}@f$, which misses part of them where @f$\xi_\mathrm{max}@f$ of
 * the collinear counter-event is below @f$\xi_\mathrm{S,cut}@f$. They
 * multiply @f$\xi^2 (1 - y)@f$ times the real matrix element and its limits. The
 * collinear ones vanish for a massive emitter. Whether a counterterm contributes
 * (@f$\xi \le \xi_\mathrm{S,cut}@f$,
 * @f$y > 1 - \delta_S@f$) is left to @ref FKSSubtraction, which also adds
 * the symmetry factors.
 *
 * With the resonance-local recoil (@p resonance, see
 * @ref FKSConfigurationMapping) @f$\xi@f$ and @f$y@f$ are those of the rest
 * frame of the resonance of momentum @f$K@f$ where the event has one, while the
 * integrated counterterms stay in the partonic center-of-mass frame. The cuts
 * then follow from those of that frame, as compute_prefactors_n1body of
 * MadGraph7 takes them from
 * resonance_subtraction_scales: with @f$P@f$ the sum of the incoming momenta
 * of the soft counter-event, @f$s = P^2@f$ and
 * @f$a(q) = s (K q) / (K^2 (P q))@f$, the ratio of @f$\xi@f$ in the two frames
 * for a soft momentum along @f$q@f$, @f$\xi_\mathrm{cut} a(\hat p_i)@f$ in
 * @f$f_s@f$, @f$\xi_\mathrm{cut} a(p_j)@f$ in @f$f_{sc}@f$ and the second and
 * fourth @f$f_{dsc}@f$, and @f$\delta s / (K^2 a(p_j)^2)@f$ in @f$L_\delta@f$,
 * with @f$\hat p_i@f$ and @f$p_j@f$ at the soft counter-event.
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
 * - `soft_momenta` – `float`, shape `(batch, n, 4)` – momenta of the soft
 *   counter-event, in the frame of the others. Present only with
 *   @p resonance.
 * - `soft_p_i_hat` – `float`, shape `(batch, 4)` – momentum of i per unit
 *   @f$\xi@f$ at the soft counter-event. Present only with @p resonance.
 * - `resonance_momentum` – `float`, shape `(batch, 4)` – the momentum
 *   @f$K@f$ of the resonance of the event, zero where it has none. Present
 *   only with @p resonance.
 * - `fks_index` – `int`, shape `(batch,)` – the configuration of the event,
 *   from 0 to the number of configurations minus one.
 *
 * **Returns**
 * - `f_r`, `f_s`, `f_c`, `f_dc`, `f_sc` – `float`, shape `(batch,)`.
 * - `f_dsc` – `float`, shape `(batch, 4)`.
 * - `f_pdf_scheme` – `float`, shape `(batch, 3)` – the delta, plus and
 *   log-plus prefactors of a change of the PDF scheme.
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
     * @param resonance       Take the cuts of the resonance-local recoil
     *                        (`soft_momenta`, `soft_p_i_hat` and
     *                        `resonance_momentum`).
     */
    FKSPrefactors(
        const std::vector<FKSConfiguration>& configurations,
        double xicut = 0.5,
        double xi_s_cut = 0.5,
        double delta_o = 1.,
        double delta_i = 1.,
        double delta_s = 1.,
        bool resonance = false
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
    bool _resonance;
};

} // namespace madspace
