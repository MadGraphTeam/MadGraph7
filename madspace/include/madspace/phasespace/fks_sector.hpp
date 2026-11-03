#pragma once

#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/fks_configuration.hpp"
#include "madspace/util.hpp"

namespace madspace {

/**
 * The FKS sector functions @f$S_{ij}@f$ of fks_Sij.f [2, 3], for one FKS
 * configuration or for several, each event in the configuration of its
 * `fks_index` (see @ref FKSConfigurationMapping).
 *
 * The FKS pairs @f$(k, l)@f$ of the real process of a configuration partition
 * its phase space,
 * @f[
 *   S_{ij} = h \Big/ \sum_{kl} \frac{d_{ij}}{d_{kl}}, \qquad
 *   d_{kl} = \Big(\frac{4 E_k E_l}{\hat s}\Big)^a
 *            (1 - \beta_k \beta_l \cos\theta_{kl})^b,
 * @f]
 * with energies and angles in the partonic center-of-mass frame of the
 * momenta (velocities only for massive particles that are not color
 * octets, energies always for octets) and @f$h = h_\mathrm{damp}(E_i / (E_i +
 * E_j))@f$ for a final-state pair @f$(i, j)@f$ of two gluons or photons,
 * @f$h_\mathrm{damp}(z) + h_\mathrm{damp}(1 - z) = 1@f$. The sector functions
 * of the configurations of one real process add up to one. i enters with the
 * direction of its momentum per unit @f$\xi@f$ (`p_i_hat` of @ref FKSKinematics)
 * and the energy of its momentum, so that the same function gives @f$S_{ij}@f$
 * at the event and at the counter-events: at a soft one the pairs without i
 * drop out. As in fks_Sij.f, @f$S_{ij}@f$ is zero where an energy or the
 * angle of a pair in a denominator falls below @f$10^{-8}@f$. Below
 * @f$\xi = 10^{-6}@f$, fks_Sij.f takes the direction of i from the soft
 * counter-event, where it evaluates the real emission in the soft limit. Here
 * it is always that of the event, and the two differ by @f$O(\xi)@f$.
 *
 * The momenta can be in any frame. The configurations (see
 * @ref FKSConfiguration) come from madspace.fks, which builds them from the
 * metadata of the FKS exporter. The parameters are those of fks_powers.inc.
 *
 * `batch` is the leading batch dimension and `n` the number of particles of
 * the real process, incoming included.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – momenta of the real emission
 *   or of a counter-event, in the order of the real process of the
 *   configuration of the event.
 * - `p_i_hat` – `float`, shape `(batch, 4)` – momentum of i per unit
 *   @f$\xi@f$.
 * - `fks_index` – `int`, shape `(batch,)` – the configuration of the event,
 *   from 0 to the number of configurations minus one.
 *
 * **Returns**
 * - `s` – `float`, shape `(batch,)` – @f$S_{ij}@f$.
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
class FKSSectorFunction : public FunctionGenerator {
public:
    /**
     * @param configurations  The FKS configurations, in the order of
     *                        `fks_index` (their FKS pairs, colors, PDG codes
     *                        and masses).
     * @param fks_a           Power of the energies in d_kl (fks_powers.inc).
     * @param fks_b           Power of the angle in d_kl.
     * @param use_energy      Energies of particles that are not color octets
     *                        in d_kl (useenergy).
     * @param use_beta        Velocities of massive particles that are not
     *                        color octets in d_kl (usebeta).
     * @param h_damp_one      h_damp is 1 below it and 0 above 1 minus it
     *                        (one_h_damp).
     * @param h_damp_power    Power of h_damp (a_h_damp).
     */
    FKSSectorFunction(
        const std::vector<FKSConfiguration>& configurations,
        double fks_a = 1.5,
        double fks_b = 1.5,
        bool use_energy = true,
        bool use_beta = true,
        double h_damp_one = 0.01,
        double h_damp_power = 1.
    );
    /// Number of configurations.
    std::size_t configuration_count() const { return _i_fks.size(); }
    /// Number of particles of the real processes, incoming included.
    std::size_t particle_count() const { return _particle_count; }
    /// The FKS pairs (k, l) of the real process of a configuration, each
    /// once, in the orientation fks_j_from_i first gives it, ordered by k and
    /// then by l.
    std::vector<std::pair<std::size_t, std::size_t>> pairs(std::size_t index) const;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _particle_count;
    std::vector<me_int_t> _i_fks;
    std::vector<me_int_t> _j_fks;
    nested_vector3<me_int_t> _pairs;
    nested_vector2<me_int_t> _octet;
    nested_vector2<me_int_t> _aorg;
    nested_vector2<double> _masses;
    std::vector<double> _options;
};

} // namespace madspace
