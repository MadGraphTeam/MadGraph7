#pragma once

#include <vector>

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * The insertion operator I of the Catani-Seymour dipoles of a Born process
 * at its events, the integral of the dipoles of its real emissions over the
 * emission at fixed Born kinematics (the parts of the incoming partons that
 * convolve with their densities aside, the K and P operators). It is the
 * operator of Catani and Seymour [1] with the masses of Catani, Dittmaier,
 * Seymour and Trocsanyi [2] (eq. 6.16),
 *
 * @f[ \langle I \rangle = -\frac{\alpha_s}{2\pi}
 *     \frac{(4\pi)^\epsilon}{\Gamma(1 - \epsilon)} \sum_j \frac{1}{T_j^2}
 *     \sum_{k \ne j} \langle T_j \cdot T_k \rangle \Big[ T_j^2
 *     \Big(\frac{\mu^2}{s_{jk}}\Big)^\epsilon \mathcal{V}_j(s_{jk}, m_j, m_k;
 *     \kappa) - T_j^2 \frac{\pi^2}{3} + \Gamma_j(\mu, m_j) + \gamma_j
 *     \log\frac{\mu^2}{s_{jk}} + \gamma_j + K_j \Big], @f]
 *
 * over the colored partons @f$j@f$ and @f$k@f$ of the Born, incoming
 * included, with @f$s_{jk} = 2 p_j p_k@f$, in conventional dimensional
 * regularization with the normalization of the virtual of MadLoop. The
 * scale @f$\mu@f$ is the renormalization scale. An incoming emitter has
 * @f$\kappa = 2/3@f$. Gluons split into the @f$n_f@f$ light quarks only, and
 * the heavy quarks decouple from the virtual. With @f$\alpha_{ff} < 1@f$ the
 * dipoles of a final-state emitter and spectator vanish for
 * @f$y > \alpha_{ff} y_+@f$ (@ref CSDipoles), and I loses their integral over
 * @f$\alpha_{ff} y_+ < y < y_+@f$, the four-dimensional alpha term of Nagy
 * [3].
 *
 * `batch` is the leading batch dimension, `n` the number of particles of the
 * Born, incoming included, and `b` the number of Born orders.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – momenta of the Born, in any
 *   frame.
 * - `color_correlated` – `float`, shape `(batch, b, n, n)` – the
 *   color-correlated Borns @f$\langle T_j \cdot T_k \rangle B@f$ per Born
 *   order, as for @ref FKSSoftVirtual.
 * - `alpha_s` – `float`, shape `(batch,)` – the strong coupling.
 * - `ren_scale` – `float`, shape `(batch,)` – the renormalization scale.
 *
 * **Returns**
 * - `insertion` – `float`, shape `(batch, b)` – the finite part of
 *   @f$\langle I \rangle@f$ per Born order.
 * - `poles` – `float`, shape `(batch, b, 2)` – its coefficients of
 *   @f$1 / \epsilon^2@f$ and @f$1 / \epsilon@f$, which cancel those of the
 *   virtual.
 *
 * **References**
 * - [1] S. Catani, M. H. Seymour, https://arxiv.org/abs/hep-ph/9605323
 *   (Sec. 10)
 * - [2] S. Catani, S. Dittmaier, M. H. Seymour, Z. Trocsanyi,
 *   https://arxiv.org/abs/hep-ph/0201036 (Sec. 6)
 * - [3] Z. Nagy, https://arxiv.org/abs/hep-ph/0307268
 */
class CSInsertion : public FunctionGenerator {
public:
    /// @param pdgs         PDG codes of the Born, two incoming partons or
    ///                     leptons first. Quarks (1 to 6) and gluons are
    ///                     colored.
    /// @param masses       Masses of the Born particles. Incoming partons are
    ///                     massless.
    /// @param born_orders  Number of Born orders.
    /// @param kappa        kappa of the final-state gluon dipoles with a
    ///                     massive spectator, as for @ref CSDipoles.
    /// @param alpha_ff     The alpha of the final-final dipoles, in (0, 1].
    /// @param n_f          Number of light quark flavors.
    CSInsertion(
        const std::vector<int>& pdgs,
        const std::vector<double>& masses,
        std::size_t born_orders = 1,
        double kappa = 2. / 3.,
        double alpha_ff = 1.,
        double n_f = 5.
    );
    /// Number of particles of the Born, incoming included.
    std::size_t particle_count() const { return _masses.size(); }
    /// Number of Born orders.
    std::size_t born_order_count() const { return _born_orders; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _born_orders;
    std::vector<double> _casimir, _gamma, _k_constant, _masses;
    double _kappa, _alpha_ff, _n_f;
};

} // namespace madspace
