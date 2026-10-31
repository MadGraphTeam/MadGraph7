#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * FKS emission of a massless parton off a massless final-state parton or an
 * incoming parton.
 *
 * Maps a Born phase-space point and three random numbers to the phase space of
 * the real emission, in the FKS parametrization of MadGraph's genps_fks.f
 * [1, 2]. The variables are the energy fraction and the angle of the radiated
 * parton @f$i@f$ to the emitter @f$j@f$ in the centre-of-mass frame of the real
 * emission,
 * @f[
 *   \xi = \frac{2 E_i}{\sqrt{s}}, \qquad
 *   y = \cos\theta_{ij},
 * @f]
 * and an azimuth @f$\phi@f$, with @f$\xi = \hat\xi\,\xi_\mathrm{max}@f$.
 *
 * For a final-state emitter, @f$i@f$ and @f$j@f$ replace the Born emitter
 * @f$\bar\jmath@f$ and @f$\phi@f$ is the azimuth of @f$i@f$ about it. The sum
 * @f$i + j@f$ points along @f$\bar\jmath@f$ and the other final-state
 * particles are boosted along it, at fixed partonic energy @f$\sqrt{s}@f$ and
 * fixed incoming momenta. @f$\xi_\mathrm{max} = 1 - M_\mathrm{rec}^2/s@f$,
 * where @f$M_\mathrm{rec}@f$ is the mass of the system recoiling against
 * @f$\bar\jmath@f$.
 *
 * For an incoming emitter, the real emission has @f$s = \bar s/(1 - \xi)@f$
 * and new momentum fractions @f$x_{1,2}@f$, @f$\phi@f$ is the azimuth of
 * @f$i@f$ about the beam axis, and the final state of the Born is boosted
 * transversely to recoil against @f$i@f$. It keeps zero longitudinal momentum,
 * so the event stays in the centre-of-mass frame of the Born.
 * @f$\xi_\mathrm{max}(y)@f$ is the largest @f$\xi@f$ with @f$x_1 \le 1@f$ and
 * @f$x_2 \le 1@f$. Unlike genps_fks.f, no lower bound on @f$\xi@f$ from the
 * cuts is applied.
 *
 * The random numbers are mapped as
 * @f[
 *   \hat\xi = s_0 + (1 - s_0)\,r_\xi^2, \qquad
 *   y = 1 - 2\left[c_0 + (1 - c_0)\,r_y^2\right], \qquad
 *   \phi = 2\pi\,r_\phi,
 * @f]
 * with @f$s_0 = 10^{-6}@f$ and @f$c_0 = 5 \cdot 10^{-7}@f$, importance
 * sampling towards the soft and collinear limits. The building blocks are the
 * instructions `fks_xi_y_sample`, `fks_final_massless_radiate`,
 * `fks_recoil_boost`, `fks_initial_radiate` and `fks_initial_recoil_boost`,
 * and their inverses.
 *
 * The weight is the Jacobian of the full map, so that
 * @f$\mathrm{d}x_1\,\mathrm{d}x_2\,\mathrm{d}\Phi_{n+1} = w\,
 * \mathrm{d}\bar x_1\,\mathrm{d}\bar x_2\,\mathrm{d}\Phi_n\,
 * \mathrm{d}r_\xi\,\mathrm{d}r_y\,\mathrm{d}r_\phi@f$ in the normalization
 * of @ref PhaseSpaceMapping (without the momentum fractions if @p leptonic). It
 * includes the factor @f$\xi@f$ that genps_fks.f leaves to the FKS
 * distributions.
 *
 * All momenta are in the partonic centre-of-mass frame of the Born, where the
 * two incoming momenta of the Born add up to @f$(\sqrt{\bar s}, 0, 0, 0)@f$.
 * Particles are numbered from 0, incoming first, in the order of the real
 * process. The Born momenta are those of the real process without @f$i@f$,
 * with @f$\bar\jmath@f$ in the place of @f$j@f$.
 *
 * `batch` is the leading batch dimension. The Born momenta and momentum
 * fractions are inputs: they are supplied on the forward call and recovered by
 * the inverse.
 *
 * **Inputs**
 * - `r_xi` – `float`, shape `(batch,)` – random number for @f$\hat\xi@f$.
 * - `r_y` – `float`, shape `(batch,)` – random number for @f$y@f$.
 * - `r_phi` – `float`, shape `(batch,)` – random number for @f$\phi@f$.
 * - `born_momenta` – `float`, shape `(batch, n, 4)` – momenta of the Born.
 * - `x1`, `x2` – `float`, shape `(batch,)` – momentum fractions of the Born.
 *   Absent if @p leptonic.
 *
 * **Outputs**
 * - `momenta` – `float`, shape `(batch, n + 1, 4)` – momenta of the real
 *   emission.
 * - `x1`, `x2` – `float`, shape `(batch,)` – momentum fractions of the real
 *   emission, those of the Born for a final-state emitter. Absent if
 *   @p leptonic.
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
    ///                             process, a massless final-state parton or
    ///                             an incoming parton (0 or 1).
    /// @param leptonic             If true, the mapping has no momentum
    ///                             fractions, which an incoming emitter needs.
    FKSRadiation(
        std::size_t born_particle_count,
        std::size_t i_fks,
        std::size_t j_fks,
        bool leptonic = false
    );
    /// Number of uniform random inputs consumed by the forward mapping (3).
    std::size_t random_dim() const { return 3; }
    std::size_t born_particle_count() const { return _born_particle_count; }
    std::size_t i_fks() const { return _i_fks; }
    std::size_t j_fks() const { return _j_fks; }
    bool leptonic() const { return _leptonic; }
    /// Whether the emitter j is an incoming parton.
    bool initial_state() const { return _j_fks < 2; }

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
    Result
    build_final_forward(FunctionBuilder& fb, const NamedVector<Value>& inputs) const;
    Result
    build_final_inverse(FunctionBuilder& fb, const NamedVector<Value>& inputs) const;
    Result
    build_initial_forward(FunctionBuilder& fb, const NamedVector<Value>& inputs) const;
    Result
    build_initial_inverse(FunctionBuilder& fb, const NamedVector<Value>& inputs) const;

    std::size_t _born_particle_count;
    std::size_t _i_fks;
    std::size_t _j_fks;
    bool _leptonic;
};

} // namespace madspace
