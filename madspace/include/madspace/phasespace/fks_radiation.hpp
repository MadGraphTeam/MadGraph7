#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * FKS emission of a massless parton off a massless final-state parton.
 *
 * Maps a Born phase-space point and three random numbers to the phase space of
 * the real emission, in the FKS parametrization of MadGraph's genps_fks.f
 * [1, 2]. The radiated parton @f$i@f$ and the emitter @f$j@f$ of the real
 * process replace the Born emitter @f$\bar\jmath@f$. The variables are the
 * energy fraction and the angle of @f$i@f$,
 * @f[
 *   \xi = \frac{2 E_i}{\sqrt{s}}, \qquad
 *   y = \cos\theta_{ij},
 * @f]
 * and the azimuth @f$\phi@f$ of @f$i@f$ about the direction of
 * @f$\bar\jmath@f$. The sum @f$i + j@f$ points along @f$\bar\jmath@f$ and
 * the other final-state particles are boosted along it, at fixed partonic
 * energy @f$\sqrt{s}@f$ and fixed incoming momenta. With
 * @f$\xi = \hat\xi\,\xi_\mathrm{max}@f$, where @f$\xi_\mathrm{max} = 1 -
 * M_\mathrm{rec}^2/s@f$ and @f$M_\mathrm{rec}@f$ is the mass of the system
 * recoiling against @f$\bar\jmath@f$, the random numbers are mapped as
 * @f[
 *   \hat\xi = s_0 + (1 - s_0)\,r_\xi^2, \qquad
 *   y = 1 - 2\left[c_0 + (1 - c_0)\,r_y^2\right], \qquad
 *   \phi = 2\pi\,r_\phi,
 * @f]
 * with @f$s_0 = 10^{-6}@f$ and @f$c_0 = 5 \cdot 10^{-7}@f$, importance
 * sampling towards the soft and collinear limits. The building blocks are the
 * instructions `fks_xi_y_sample`, `fks_final_massless_radiate` and
 * `fks_recoil_boost`, and their inverses.
 *
 * The weight is the Jacobian of the full map, so that
 * @f$\mathrm{d}\Phi_{n+1} = w\,\mathrm{d}\Phi_n\,\mathrm{d}r_\xi\,
 * \mathrm{d}r_y\,\mathrm{d}r_\phi@f$ in the normalization of
 * @ref PhaseSpaceMapping. It includes the factor @f$\xi@f$ that genps_fks.f
 * leaves to the FKS distributions.
 *
 * All momenta are in the partonic centre-of-mass frame of the Born, where the
 * two incoming momenta add up to @f$(\sqrt{s}, 0, 0, 0)@f$. Particles are
 * numbered from 0, incoming first, in the order of the real process. The
 * Born momenta are those of the real process without @f$i@f$, with
 * @f$\bar\jmath@f$ in the place of @f$j@f$.
 *
 * `batch` is the leading batch dimension. The Born momenta are inputs: they
 * are supplied on the forward call and recovered by the inverse.
 *
 * **Inputs**
 * - `r_xi` – `float`, shape `(batch,)` – random number for @f$\hat\xi@f$.
 * - `r_y` – `float`, shape `(batch,)` – random number for @f$y@f$.
 * - `r_phi` – `float`, shape `(batch,)` – random number for @f$\phi@f$.
 * - `born_momenta` – `float`, shape `(batch, n, 4)` – momenta of the Born.
 *
 * **Outputs**
 * - `momenta` – `float`, shape `(batch, n + 1, 4)` – momenta of the real
 *   emission.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] S. Frixione, Z. Kunszt, A. Signer, "Three-jet cross sections to
 *   next-to-leading order", https://arxiv.org/abs/hep-ph/9512328
 * - [2] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272 (Sec. 5)
 */
class FKSRadiation : public Mapping {
public:
    /// @param born_particle_count  Number of particles of the Born, incoming
    ///                             included.
    /// @param i_fks                Position of the radiated parton i in the
    ///                             real process.
    /// @param j_fks                Position of the emitter j in the real
    ///                             process, a massless final-state parton.
    FKSRadiation(std::size_t born_particle_count, std::size_t i_fks, std::size_t j_fks);
    /// Number of uniform random inputs consumed by the forward mapping (3).
    std::size_t random_dim() const { return 3; }
    std::size_t born_particle_count() const { return _born_particle_count; }
    std::size_t i_fks() const { return _i_fks; }
    std::size_t j_fks() const { return _j_fks; }

private:
    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    std::size_t born_index(std::size_t real_index) const {
        return real_index < _i_fks ? real_index : real_index - 1;
    }

    std::size_t _born_particle_count;
    std::size_t _i_fks;
    std::size_t _j_fks;
};

} // namespace madspace
