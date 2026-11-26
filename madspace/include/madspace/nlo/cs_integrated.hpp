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
 * In a decay the decaying particle @f$a@f$ is no emitter. A final-state quark
 * @f$j@f$ has it as the spectator with the dipole of @ref CSDipoles (the
 * final-final dipole with the recoil system @f$K@f$ of the other final-state
 * particles as the spectator, and the soft terms of @f$a@f$). Its integral is
 * the pair term above with @f$s_{jK} = m_a^2 - m_j^2 - m_K^2@f$ and the mass
 * @f$m_K@f$ of the recoil system of the Born (massless only if it is a single
 * massless particle), plus the rest in closed form
 * (see the instruction `cs_insertion`): the pair term of CDST has the mean of
 * the integrals of the eikonal terms of both orderings of the pair, which
 * only their sum needs, the dipole of one ordering its own. A final-state
 * gluon with a colored decaying particle is not covered.
 *
 * The operator of the QED corrections (`qed`) is the same with the charges
 * in place of the colors [5]: @f$Q_j Q_k B@f$ of the charges flowing out of
 * @f$j@f$ and @f$k@f$ in place of @f$\langle T_j \cdot T_k \rangle@f$,
 * @f$Q_j^2@f$ in place of @f$T_j^2@f$ for every charged particle, also a
 * massive W (with @f$\gamma_j@f$ and @f$K_j@f$ of a quark, as its dipoles
 * of @ref CSDipoles), and for a photon, which has no charge but splits into
 * the light fermions @f$f@f$, @f$\gamma_\gamma = -\frac{2}{3} \sum_f N_{c,f}
 * Q_f^2@f$ and @f$K_\gamma = -\frac{10}{9} \sum_f N_{c,f} Q_f^2@f$ with
 * @f$-w_k B@f$ in place of @f$\langle T_j \cdot T_k \rangle B / T_j^2@f$,
 * the weights @f$w_k@f$ of its spectators, one over the number of charged
 * particles (but a decaying one) each [6].
 *
 * `batch` is the leading batch dimension, `n` the number of particles of the
 * Born, incoming included, and `b` the number of Born orders.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – momenta of the Born, in any
 *   frame (the rest frame of the decaying particle for a decay).
 * - `color_correlated` – `float`, shape `(batch, b, n, n)` – the
 *   color-correlated Borns @f$\langle T_j \cdot T_k \rangle B@f$ per Born
 *   order, as for @ref FKSSoftVirtual (QCD).
 * - `born` – `float`, shape `(batch, b)` – the Born per Born order (QED, in
 *   place of `color_correlated`).
 * - `alpha_s` – `float`, shape `(batch,)` – the strong coupling (`alpha`,
 *   the QED coupling, for QED).
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
 * - [1] S. Catani, M. H. Seymour, "A general algorithm for calculating jet
 *   cross sections in NLO QCD", https://arxiv.org/abs/hep-ph/9605323
 *   (Sec. 10)
 * - [2] S. Catani, S. Dittmaier, M. H. Seymour, Z. Trocsanyi, "The dipole
 *   formalism for next-to-leading order QCD calculations with massive
 *   partons", https://arxiv.org/abs/hep-ph/0201036 (Sec. 6)
 * - [3] Z. Nagy, https://arxiv.org/abs/hep-ph/0307268
 * - [4] L. Basso, S. Dittmaier, A. Huss, L. Oggero, "Techniques for the
 *   treatment of IR divergences in decay processes at NLO and application to
 *   the top-quark decay", https://arxiv.org/abs/1507.04676 (decays)
 * - [5] S. Dittmaier, https://arxiv.org/abs/hep-ph/9904440 (QED)
 * - [6] S. Dittmaier, A. Kabelschacht, T. Kasprzik,
 *   https://arxiv.org/abs/0802.1405 (photon splittings)
 */
class CSInsertion : public FunctionGenerator {
public:
    /// @param pdgs         PDG codes of the Born, two incoming partons or
    ///                     leptons first, or the decaying particle. Quarks (1
    ///                     to 6) and gluons are colored.
    /// @param masses       Masses of the Born particles. Incoming partons are
    ///                     massless (but a decaying one).
    /// @param born_orders  Number of Born orders.
    /// @param kappa        kappa of the final-state gluon dipoles with a
    ///                     massive spectator, as for @ref CSDipoles.
    /// @param alpha_ff     The alpha of the final-final dipoles, in (0, 1].
    /// @param n_f          Number of light quark flavors.
    /// @param incoming     Number of incoming particles, 1 for a decay.
    /// @param qed          The operator of the QED corrections, with the Born
    ///                     (`born`) in place of the color-correlated Born and
    ///                     the QED coupling (`alpha`) in place of the strong
    ///                     one.
    /// @param n_l          Number of massless charged leptons, into which a
    ///                     photon splits besides the light quarks (QED).
    CSInsertion(
        const std::vector<int>& pdgs,
        const std::vector<double>& masses,
        std::size_t born_orders = 1,
        double kappa = 2. / 3.,
        double alpha_ff = 1.,
        double n_f = 5.,
        std::size_t incoming = 2,
        bool qed = false,
        double n_l = 0.
    );
    /// Number of particles of the Born, incoming included.
    std::size_t particle_count() const { return _masses.size(); }
    /// Number of Born orders.
    std::size_t born_order_count() const { return _born_orders; }
    /// Whether the operator is that of the QED corrections.
    bool qed() const { return _qed; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _born_orders;
    std::vector<double> _casimir, _gamma, _k_constant, _masses;
    std::vector<double> _self_coupling, _pair_splitting, _charges, _weights;
    double _kappa, _alpha_ff, _n_f;
    std::size_t _incoming;
    bool _qed;
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
 * The operators of the QED corrections (`qed`) take the charges in place of
 * the colors as @ref CSInsertion does, with the kernels of QED [4]:
 * @f$Q_f^2@f$ in place of @f$C_F@f$ for @f$f \to f@f$, @f$N_{c,f} Q_f^2@f$ in
 * place of @f$T_R@f$ for @f$\gamma \to f@f$, @f$f \to \gamma@f$ per unit
 * @f$Q_f^2@f$ (the luminosity of the partons that split into a photon is
 * @f$\sum_f Q_f^2 f_f(x_i / z) / z@f$ over the charged fermions, that of a
 * charged fermion the photon), and for @f$\gamma \to \gamma@f$ only
 * @f$-\frac{2}{3} \sum_f N_{c,f} Q_f^2\,\delta(1 - x)@f$ in P and
 * @f$\frac{16}{9} \sum_f N_{c,f} Q_f^2\,\delta(1 - x)@f$ in
 * @f$\overline{K}@f$, over the light fermions.
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
 *   order, as for @ref FKSSoftVirtual (QCD only).
 * - `alpha_s` – `float`, shape `(batch,)` – the strong coupling (`alpha`,
 *   the QED coupling, for QED).
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
 * - [1] S. Catani, M. H. Seymour, "A general algorithm for calculating jet
 *   cross sections in NLO QCD", https://arxiv.org/abs/hep-ph/9605323
 *   (Sec. 10)
 * - [2] S. Catani, S. Dittmaier, M. H. Seymour, Z. Trocsanyi, "The dipole
 *   formalism for next-to-leading order QCD calculations with massive
 *   partons", https://arxiv.org/abs/hep-ph/0201036 (Sec. 6)
 * - [3] Z. Nagy, https://arxiv.org/abs/hep-ph/0307268
 * - [4] S. Dittmaier, A. Kabelschacht, T. Kasprzik,
 *   https://arxiv.org/abs/0802.1405 (QED)
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
    /// @param qed          The operators of the QED corrections, without the
    ///                     color-correlated Born and with the QED coupling
    ///                     (`alpha`) in place of the strong one.
    /// @param n_l          Number of massless charged leptons (QED).
    CSCollinear(
        const std::vector<int>& pdgs,
        const std::vector<double>& masses,
        std::size_t born_orders = 1,
        double alpha_fi = 1.,
        double alpha_if = 1.,
        double alpha_ii = 1.,
        double n_f = 5.,
        bool qed = false,
        double n_l = 0.
    );
    /// Number of particles of the Born, incoming included.
    std::size_t particle_count() const { return _masses.size(); }
    /// Number of Born orders.
    std::size_t born_order_count() const { return _born_orders; }
    /// Whether the operators are those of the QED corrections.
    bool qed() const { return _qed; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _born_orders;
    std::vector<double> _casimir, _gamma, _masses, _options;
    std::vector<double> _self_coupling, _pair_splitting, _other_coefficient;
    std::vector<double> _charges, _weights;
    bool _qed;
};

} // namespace madspace
