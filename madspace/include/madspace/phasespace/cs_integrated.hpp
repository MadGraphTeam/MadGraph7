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

/**
 * The K and P operators of the Catani-Seymour dipoles of a Born process at
 * its events, the parts of the integral of the dipoles of its real emissions
 * that convolve with the densities of the beams, as weights of the
 * luminosities of each beam at one point of the convolution. These are the
 * operators of Catani and Seymour [1] (eqs. 10.24 to 10.29) with the terms
 * of massive final-state quarks of Catani, Dittmaier, Seymour and Trocsanyi
 * [2] (eqs. 6.55 to 6.62), in the MSbar scheme. For the parton @f$a'@f$ of
 * the Born of beam @f$i@f$, with momentum fraction @f$x_i@f$, they enter as
 *
 * @f[ \int_0^1 \mathrm{d}x \sum_a \big[K + P\big]^{a,a'}(x)
 *     \frac{f_a(x_i / x)}{x}, @f]
 *
 * with the parton @f$a@f$ of the hadron that splits into @f$a'@f$ and
 * @f$f_a(y) = 0@f$ for @f$y > 1@f$. One random number @f$r@f$ gives the point
 * @f$z = x_i + (1 - x_i) r@f$ of each beam, and the weights of the
 * luminosities @f$f_{a'}(x_i / z) / z@f$ (the same parton),
 * @f$\sum_a f_a(x_i / z) / z@f$ over the partons @f$a \ne a'@f$ that split into
 * it (the gluon for a quark, the quarks and antiquarks for a gluon) and
 * @f$f_{a'}(x_i)@f$ (the Born) have the convolution as their mean over
 * @f$r@f$. A plus distribution @f$[g(x)]_+@f$ gives
 * @f$[g(z) f(z) - g(z) f(1)] (1 - x_i) - f(1) \int_0^{x_i} g@f$. The terms of
 * a massive quark @f$j@f$ depend on @f$s_{ja} = 2 p_j p_a@f$ of the real
 * emission, @f$s_{ja'} / z@f$ inside the convolution and @f$s_{ja'}@f$ at its
 * end point. P has @f$\log(\mu_F^2 / s_{ja'})@f$ with the factorization scale
 * of the beam. With @f$\alpha < 1@f$ the dipoles of a family vanish outside
 * their gate (@ref CSDipoles), and the weights lose their integral over the
 * rest, a regular function of @f$z@f$, the four-dimensional alpha terms of
 * Nagy [3].
 *
 * `batch` is the leading batch dimension, `n` the number of particles of the
 * Born, incoming included, and `b` the number of Born orders.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – momenta of the Born, in any
 *   frame.
 * - `born` – `float`, shape `(batch, b)` – the Born per Born order.
 * - `color_correlated` – `float`, shape `(batch, b, n, n)` – the
 *   color-correlated Borns @f$\langle T_j \cdot T_k \rangle B@f$ per Born
 *   order, as for @ref FKSSoftVirtual.
 * - `alpha_s` – `float`, shape `(batch,)` – the strong coupling.
 * - `fact_scale1`, `fact_scale2` – `float`, shape `(batch,)` – the
 *   factorization scales of the two beams.
 * - `x1`, `x2` – `float`, shape `(batch,)` – the momentum fractions of the
 *   incoming partons of the Born.
 * - `r` – `float`, shape `(batch,)` – the random number of the convolution.
 *
 * **Returns**
 * - `z` – `float`, shape `(batch, 2)` – the point of the convolution of each
 *   beam.
 * - `weights` – `float`, shape `(batch, b, 2, 3)` – per Born order and beam
 *   the weights of the three luminosities, zero for a colorless particle.
 *
 * **References**
 * - [1] S. Catani, M. H. Seymour, https://arxiv.org/abs/hep-ph/9605323
 *   (Sec. 10)
 * - [2] S. Catani, S. Dittmaier, M. H. Seymour, Z. Trocsanyi,
 *   https://arxiv.org/abs/hep-ph/0201036 (Sec. 6)
 * - [3] Z. Nagy, https://arxiv.org/abs/hep-ph/0307268
 */
class CSCollinear : public FunctionGenerator {
public:
    /// @param pdgs         PDG codes of the Born, the two incoming partons or
    ///                     leptons first. Quarks (1 to 6) and gluons are
    ///                     colored.
    /// @param masses       Masses of the Born particles. Incoming partons are
    ///                     massless.
    /// @param born_orders  Number of Born orders.
    /// @param alpha_fi, alpha_if, alpha_ii  The alpha of the families with an
    ///                     incoming parton, in (0, 1], as for @ref CSDipoles.
    /// @param n_f          Number of light quark flavors.
    CSCollinear(
        const std::vector<int>& pdgs,
        const std::vector<double>& masses,
        std::size_t born_orders = 1,
        double alpha_fi = 1.,
        double alpha_if = 1.,
        double alpha_ii = 1.,
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
    std::vector<double> _casimir, _gamma, _masses, _options;
};

} // namespace madspace
