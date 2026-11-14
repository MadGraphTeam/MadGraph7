#pragma once

#include "madspace/phasespace/base.hpp"

#include <string>
#include <tuple>
#include <vector>

namespace madspace {

/**
 * FKS emission of a massless parton off a final-state parton, massless or
 * massive, or an incoming parton.
 *
 * Maps a Born phase-space point and three random numbers to the phase space of
 * the real emission, in the FKS parametrization of MadGraph's genps_fks.f
 * [2, 3]. The variables are the energy fraction and the angle of the radiated
 * parton @f$i@f$ to the emitter @f$j@f$ in the center-of-mass frame of the real
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
 * For a massive final-state emitter (@p emitter_mass > 0) the map follows
 * generate_momenta_massive_final of genps_fks.f. The emitter comes to rest at
 * @f$\xi_m@f$. For @f$y \ge 0@f$, @f$\xi@f$ runs up to
 * @f$\xi_\mathrm{max} = \xi_m@f$. For @f$y < 0@f$ there are two solutions for
 * the momentum of the emitter, which meet at @f$\xi_\mathrm{max} > \xi_m@f$,
 * and @f$\hat\xi@f$ runs along a path of length @f$\xi_\mathrm{norm} =
 * 2\xi_\mathrm{max} - \xi_m@f$, from @f$\xi = 0@f$ to
 * @f$\xi_\mathrm{max}@f$ with the first solution and back to @f$\xi_m@f$
 * with the second. @f$r_\xi@f$ below @f$\xi_\mathrm{max}/\xi_\mathrm{norm}@f$
 * gives the first solution, with the quadratic sampling below, and above it
 * the second one, linearly.
 *
 * A decay (one incoming particle, @p incoming) radiates off final-state
 * emitters only, in the rest frame of the decaying particle at
 * @f$\sqrt{s} = M@f$, and has no momentum fractions.
 * For an incoming emitter, the real emission has @f$s = \bar s/(1 - \xi)@f$
 * and new momentum fractions @f$x_{1,2}@f$, @f$\phi@f$ is the azimuth of
 * @f$i@f$ about the beam axis, and the final state of the Born is boosted
 * transversely to recoil against @f$i@f$. It keeps zero longitudinal momentum,
 * so the event stays in the center-of-mass frame of the Born.
 * @f$\xi_\mathrm{max}(y)@f$ is the largest @f$\xi@f$ with @f$x_1 \le 1@f$ and
 * @f$x_2 \le 1@f$. With a lower bound @f$\hat s_\mathrm{min}@f$ on the
 * partonic energy of the real emission (@p s_hat_min), as genps_fks.f derives
 * from the cuts (tau_lower_bound), @f$\xi \ge \xi_\mathrm{min} = 1 - \bar
 * s/\hat s_\mathrm{min}@f$ and @f$y@f$ is restricted to where
 * @f$\xi_\mathrm{max}(y) \ge \xi_\mathrm{min}@f$, so that the map covers
 * exactly the real emissions with @f$s \ge \hat s_\mathrm{min}@f$ and no
 * sampling is wasted on the others. @f$\hat\xi@f$ and the variable @f$\hat y@f$
 * that the random number gives are then mapped linearly onto the allowed
 * ranges. This is the symmetric map of genps_fks.f (@p initial_map
 * `symmetric`, FKSISRMapping 1).
 *
 * The asymmetric map (`asymmetric`, the default, FKSISRMapping 0 and 2 of
 * MadGraph7) changes only the momentum fraction of the emitter,
 * @f$x_j = \bar x_j/(1 - \xi)@f$, and keeps that of the spectator. The final
 * state of the Born recoils against @f$i@f$ by a Lorentz map that keeps the
 * direction of the spectator (`fks_initial_light_cone_map`), and
 * @f$\xi_\mathrm{max} = 1 - \bar x_j@f$ at every @f$y@f$, so that
 * @f$\xi \ge \xi_\mathrm{min}@f$ leaves the range of @f$y@f$ whole
 * (@f$\hat y = y@f$). Its counter-events are those of the symmetric map. The
 * event has the same @f$\xi@f$, @f$y@f$ and @f$\phi@f$ in the center-of-mass
 * frame of the real emission, but another range of @f$\xi@f$ and other
 * momenta in the frame of the Born.
 *
 * With a resonance (@p resonance, the resonance-local recoil of MadGraph7
 * after [4]) a final-state emitter
 * radiates in the rest frame of the momentum @f$K@f$ of a part of the final
 * state, its members, which contain @f$\bar\jmath@f$: they are boosted into
 * that frame, the emission above maps them with @f$\sqrt{s}@f$ replaced by
 * @f$\sqrt{K^2}@f$, the other members recoil against @f$i + j@f$, and they are
 * boosted back. @f$K@f$ and the momenta of all other particles stay as they
 * are, and @f$\xi@f$, @f$y@f$ and @f$\phi@f$ are those of the rest frame of
 * @f$K@f$ (`fks_resonance_frame`). Since the members of a resonance have a
 * Lorentz invariant phase space at fixed @f$K@f$, the weight is that of the
 * emission at @f$\sqrt{K^2}@f$. madspace.subtraction.fks.fks_resonances gives the
 * resonance of MadGraph, the decay products of the grandmother of i and j
 * where it is a resonance (set_granny).
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
 * `fks_final_massive_sample`, `fks_final_massive_radiate`,
 * `fks_recoil_boost`, `fks_initial_radiate` and `fks_initial_recoil_boost`,
 * `fks_initial_asymmetric_radiate` and `fks_initial_light_cone_map`, and their
 * inverses.
 *
 * The weight is the Jacobian of the full map, so that
 * @f$\mathrm{d}x_1\,\mathrm{d}x_2\,\mathrm{d}\Phi_{n+1} = w\,
 * \mathrm{d}\bar x_1\,\mathrm{d}\bar x_2\,\mathrm{d}\Phi_n\,
 * \mathrm{d}r_\xi\,\mathrm{d}r_y\,\mathrm{d}r_\phi@f$ in the normalization
 * of @ref PhaseSpaceMapping (without the momentum fractions if @p leptonic). It
 * includes the factor @f$\xi@f$ that genps_fks.f leaves to the FKS
 * distributions.
 *
 * All momenta are in the partonic center-of-mass frame of the Born, where the
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
 * **Conditions**
 * - None.
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
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] S. Frixione, Z. Kunszt, A. Signer, "Three-jet cross sections to
 *   next-to-leading order", https://arxiv.org/abs/hep-ph/9512328
 * - [3] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272 (Sec. 5)
 * - [4] T. Ježo, P. Nason, "On the treatment of resonances in next-to-leading
 *   order calculations matched to a parton shower",
 *   https://arxiv.org/abs/1509.09071 (Sec. 3)
 */
class FKSRadiation : public Mapping {
public:
    /**
     * The map of an emission off an incoming parton.
     */
    enum InitialMap {
        /// both momentum fractions change, and the final state of the Born
        /// keeps zero longitudinal momentum (FKSISRMapping 1)
        symmetric,
        /// only the momentum fraction of the emitter changes (FKSISRMapping 0
        /// and 2, the default of MadGraph7)
        asymmetric,
    };

    /// @param born_particle_count  Number of particles of the Born, incoming
    ///                             included.
    /// @param i_fks                Position of the radiated parton i in the
    ///                             real process.
    /// @param j_fks                Position of the emitter j in the real
    ///                             process, a final-state parton or an
    ///                             incoming parton (0 or 1).
    /// @param leptonic             If true, the mapping has no momentum
    ///                             fractions, which an incoming emitter needs.
    /// @param s_hat_min            Lower bound on the partonic energy squared
    ///                             of the real emission, for an incoming
    ///                             emitter (a final-state one leaves it
    ///                             unchanged). 0 for none.
    /// @param emitter_mass         Mass of a final-state emitter j (incoming
    ///                             ones are massless). The Born emitter must
    ///                             have this mass.
    /// @param initial_map          The map of an emission off an incoming
    ///                             parton, see @ref InitialMap.
    /// @param born_masses          The masses of the particles of the Born,
    ///                             for the mass shell of its final state in
    ///                             the asymmetric map. Empty for none (the map
    ///                             is then linear in the momenta).
    /// @param resonance            The positions in the real process of the
    ///                             members of the resonance of a final-state
    ///                             emitter, i and j included, at least one
    ///                             more. Empty for none: the whole final
    ///                             state recoils.
    /// @param incoming             Number of incoming particles, two or one
    ///                             for a decay, which needs @p leptonic and a
    ///                             final-state emitter.
    FKSRadiation(
        std::size_t born_particle_count,
        std::size_t i_fks,
        std::size_t j_fks,
        bool leptonic = false,
        double s_hat_min = 0.,
        double emitter_mass = 0.,
        InitialMap initial_map = asymmetric,
        const std::vector<double>& born_masses = {},
        const std::vector<std::size_t>& resonance = {},
        std::size_t incoming = 2
    );
    /// Number of uniform random inputs consumed by the forward mapping (3).
    std::size_t random_dim() const { return 3; }
    /// Number of particles of the Born, incoming included.
    std::size_t born_particle_count() const { return _born_particle_count; }
    /// Position of the radiated parton i in the real process.
    std::size_t i_fks() const { return _i_fks; }
    /// Position of the emitter j in the real process.
    std::size_t j_fks() const { return _j_fks; }
    /// Number of incoming particles, one for a decay.
    std::size_t incoming() const { return _incoming; }
    /// Whether the mapping has no momentum fractions.
    bool leptonic() const { return _leptonic; }
    /// Lower bound on the partonic energy squared of the real emission.
    double s_hat_min() const { return _s_hat_min; }
    /// Mass of the emitter, zero for a massless one.
    double emitter_mass() const { return _emitter_mass; }
    /// The map of an emission off an incoming parton.
    InitialMap initial_map() const { return _initial_map; }
    /// Whether the emitter j is an incoming parton.
    bool initial_state() const { return _j_fks < _incoming; }
    /// The members of the resonance, positions in the real process in
    /// ascending order, empty for none.
    const std::vector<std::size_t>& resonance() const { return _resonance; }
    /// The FKS counter-events: `soft`, `collinear` and `soft_collinear`, for
    /// every emitter, so that the FKS configurations of a Born share one
    /// layout. A massive emitter has only the soft one (the others have
    /// `det` = 0).
    std::vector<std::string> counter_events() const;
    /**
     * Add the real emission and its FKS counter-events at the same random
     * numbers to @p fb.
     *
     * The counter-events are the same map at @f$\hat\xi = 0@f$ (soft), at
     * @f$y = 1@f$ (collinear, @f$\hat y = 1@f$ for an incoming emitter) or
     * both, with the azimuth and the other variable of the event, as
     * genps_fks.f. Takes the inputs of the forward map and returns
     * - `momenta`, `x1`, `x2` (unless leptonic) and `weight`: the forward map;
     * - `xi`, `xi_hat`, `y`, `xi_max`: the FKS variables of the event (`xi_hat`
     *   as genps_fks.f samples it, `y` the cosine of i to j);
     * - `p_i_hat`: the momentum of i per unit @f$\xi@f$ in the event,
     *   @f$p_i / \xi@f$ (the `p_i_fks_ev` of genps_fks.f);
     * - for each counter-event `k` of counter_events(): `k_momenta`, `k_x1`,
     *   `k_x2` (unless leptonic) and `k_det`, the Jacobian of the map at the
     *   counter-event without the factor @f$\xi@f$ (so that the weight of the
     *   event is @f$\xi@f$ times its own `det`), with the sampling Jacobian of
     *   the event, and `k_p_i_hat`, the momentum of i per unit
     *   @f$\xi@f$ (`p_i_fks_cnt`), finite at @f$\xi = 0@f$: the direction
     *   from which i becomes soft, which the sector functions and the eikonal
     *   factors of the soft limits need, and `k_xi`, `k_xi_max`, its
     *   @f$\xi@f$ (0 where i is soft) and the upper end of the range of
     *   @f$\xi@f$ at its @f$y@f$ (`xi_i_fks_cnt`, `xiimax_cnt`), which the
     *   prefactors of the counterterms need. `k_det` is 0 where genps_fks.f
     *   does not generate the counter-event: the soft one for @f$\xi_\mathrm{min} >
     * 0@f$ and for the second solution of a massive emitter, the collinear one for
     *   @f$y = 1@f$ outside the range of an incoming emitter with
     *   @p s_hat_min. A massive emitter has no collinear counter-events: their
     *   momenta and momentum fractions are those of the event, their `det` 0.
     */
    NamedVector<Value>
    build_counter_events(FunctionBuilder& fb, const NamedVector<Value>& inputs) const;

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
    // the emission at given sampling variables, in the Born c.m. frame
    struct Emission {
        // rat: end of the first solution of a massive emitter in xi_hat, 1
        // for any other emitter; p_i_hat: i per unit xi
        Value momenta, x1, x2, xi, y, xi_max, det_xi, rat, p_i_hat;
    };
    // (xi_hat, y or y_hat, det_sample) from (r_xi, r_y)
    std::tuple<Value, Value, Value> build_sample(
        FunctionBuilder& fb, Value r_xi, Value r_y, Value sqrt_s, Value p_jbar
    ) const;
    Emission build_emission(
        FunctionBuilder& fb,
        Value xi_hat,
        Value y_var,
        Value r_phi,
        Value born,
        Value sqrt_s,
        Value x1,
        Value x2
    ) const;
    NamedVector<Value> build_counter_events_impl(
        FunctionBuilder& fb, const NamedVector<Value>& inputs, bool with_counter_events
    ) const;
    Result
    build_final_inverse(FunctionBuilder& fb, const NamedVector<Value>& inputs) const;
    Result
    build_initial_inverse(FunctionBuilder& fb, const NamedVector<Value>& inputs) const;
    // sqrt(s) of momenta, the incoming first: the mass of the decaying
    // particle or the partonic energy
    Value sqrt_s_of(FunctionBuilder& fb, Value momenta) const;

    std::size_t _born_particle_count;
    std::size_t _i_fks;
    std::size_t _j_fks;
    std::size_t _incoming;
    bool _leptonic;
    double _s_hat_min;
    double _emitter_mass;
    InitialMap _initial_map;
    // the masses of the final state of the Born, -1 where unknown
    std::vector<double> _final_masses;
    // the members of the resonance, and 1 for them, 0 for the other particles,
    // in the order of the Born and of the real process
    std::vector<std::size_t> _resonance;
    std::vector<double> _born_resonance_mask;
    std::vector<double> _real_resonance_mask;
};

} // namespace madspace
