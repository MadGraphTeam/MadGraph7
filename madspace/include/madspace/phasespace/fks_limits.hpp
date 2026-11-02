#pragma once

#include <utility>
#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/util.hpp"

namespace madspace {

/**
 * The reduced eikonal factors of the soft limit of the FKS configurations of
 * a Born, as eikonal_reduced in fks_singular.f, each event in the
 * configuration of its `fks_index` (see @ref FKSConfigurationMapping).
 *
 * Where i becomes soft, the real matrix element times @f$\xi^2 (1 - y)@f$
 * tends to a sum over the pairs @f$(m, n)@f$ of the colour (or charge)
 * partners of i,
 * @f[
 *   \xi^2 (1 - y) |M_{n+1}|^2 \to
 *   -2 \sum_{(m, n)} \mathcal{E}_{mn} B_{mn}, \qquad
 *   \mathcal{E}_{mn} = \frac{(p_m p_n) (1 - y)}{(p_m \hat p_i)(p_n \hat p_i)},
 * @f]
 * with the colour-linked Borns @f$B_{mn}@f$ in the conventions of MadFKS
 * (sbornsoft) and the momentum of i per unit @f$\xi@f$, @f$\hat p_i@f$
 * (`p_i_hat` of @ref FKSKinematics). For a massless j,
 * @f$(p_j \hat p_i) / (1 - y) = \sqrt{s} E_j / 2@f$ in the partonic
 * centre-of-mass frame, so that the factors are finite at @f$y = 1@f$. The
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
 * - `fks_index` – `int`, shape `(batch,)` – the configuration of the event.
 *   Present only for several configurations.
 *
 * **Returns**
 * - `eikonals` – `float`, shape `(batch, k)` – @f$\mathcal{E}_{mn}@f$ of the
 *   pairs of the configuration (see pairs()), zero past them.
 */
class FKSEikonals : public FunctionGenerator {
public:
    /**
     * @param i_fks         Position of i in the real process of each
     *                      configuration, 0-based.
     * @param j_fks         Position of j in the real process of each
     *                      configuration, 0-based.
     * @param fks_j_from_i  For each configuration and each particle i of its
     *                      real process, the particles j it forms an FKS
     *                      pair with, 0-based (FKS_J_FROM_I_D).
     * @param masses        Masses of the particles of the real process of each
     *                      configuration.
     */
    FKSEikonals(
        const std::vector<std::size_t>& i_fks,
        const std::vector<std::size_t>& j_fks,
        const nested_vector3<std::size_t>& fks_j_from_i,
        const nested_vector2<double>& masses
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
    /// colour-linked Borns they multiply (j is at the position of i + j).
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
 * and sborncol_isr in fks_singular.f, each event in the configuration of its
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
 * Born, for an initial-state one; its momentum fraction is @f$z = 1 - E_i /
 * (E_i + E_j)@f$ or @f$z = 1 - \xi = 1 - 2 E_i / \sqrt{s}@f$ in the partonic
 * centre-of-mass frame of the counter-event. The azimuthal terms go with the
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
 * - `fks_index` – `int`, shape `(batch,)` – the configuration of the event.
 *   Present only for several configurations.
 *
 * **Returns**
 * - `z` – `float`, shape `(batch,)` – momentum fraction of the splitting.
 * - `ap` – `float`, shape `(batch, 2)` – QCD and QED Altarelli-Parisi kernels.
 * - `q_terms` – `float`, shape `(batch, 2)` – QCD and QED azimuthal terms.
 * - `k_perp` – `float`, shape `(batch, 4)` – transverse direction of the
 *   emission, in the frame of the momenta.
 */
class FKSSplittingKernels : public FunctionGenerator {
public:
    /**
     * @param i_fks    Position of i in the real process of each
     *                 configuration, 0-based.
     * @param j_fks    Position of j in the real process of each
     *                 configuration, 0-based.
     * @param colors   Colour representation of each particle of the real
     *                 process of each configuration (1, 3, -3, 8).
     * @param charges  Electric charge of each particle of the real process of
     *                 each configuration.
     */
    FKSSplittingKernels(
        const std::vector<std::size_t>& i_fks,
        const std::vector<std::size_t>& j_fks,
        const nested_vector2<int>& colors,
        const nested_vector2<double>& charges
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
};

} // namespace madspace
