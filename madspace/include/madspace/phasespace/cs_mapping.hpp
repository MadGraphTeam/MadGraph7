#pragma once

#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/phasespace.hpp"

namespace madspace {

/**
 * Phase space of a real emission in one Catani-Seymour dipole: a Born
 * phase-space mapping followed by the emission of a massless parton
 * @f$i@f$ off the emitter @f$j@f$ with the spectator @f$k@f$, in the lab frame.
 *
 * The emission inverts the momentum map of the dipole, the four families of
 * Catani and Seymour [2] with the masses of Catani, Dittmaier, Seymour and
 * Trocsanyi [3]:
 * - final emitter and spectator (`cs_final_final_radiate`), in the variables
 *   @f$y_{ij,k}@f$ and @f$z_i@f$;
 * - final emitter, incoming spectator @f$a@f$ and incoming emitter @f$a@f$,
 *   final spectator (`cs_initial_final_radiate`, the same map for both), in
 *   @f$x@f$ and @f$z@f$ (@f$z_i@f$ and @f$u_i@f$ of [2]);
 * - incoming emitter and spectator (`cs_initial_initial_radiate`), in
 *   @f$x_{i,ab}@f$ and @f$v_i@f$, with the final state Lorentz transformed;
 * - final emitter and the decaying particle of a decay as the spectator
 *   (`cs_final_decay_radiate`, Basso, Dittmaier, Huss, Oggero [4]), the
 *   final-final map with the other final-state particles as the spectator,
 *   in @f$y@f$ and @f$z_i@f$, with those particles Lorentz transformed and
 *   the momentum of the decaying particle fixed.
 *
 * The emitter keeps its mass (@f$m_{ij} = m_j@f$, so not the splitting of a
 * gluon into massive quarks), the spectator may be massive, and incoming
 * partons are massless (but the decaying particle of a decay). The Born mapping gives
 * the Born momenta in the lab frame, they are boosted along the beam axis into the
 * center-of-mass frame of the Born, where the emission adds @f$i@f$, and the real
 * emission is boosted back with the same boost (the instruction `fks_beam_boost`). With
 * an incoming emitter or spectator @f$a@f$ the momentum fraction of @f$a@f$ is that of
 * the Born over @f$x@f$, from the Born fraction (the lower end of
 * @f$x@f$) to one, and the other beam keeps its fraction. The inverse projects
 * the real emission onto the Born in its own center-of-mass frame. The maps
 * are covariant and the azimuth of @f$i@f$ is defined in a frame built from the
 * momenta and the @f$x@f$ axis, which boosts along the beam axis and the
 * mirror of `mirror_beams` leave unchanged, so the mapping commutes with
 * both.
 *
 * The weight is the product of the weights of the two mappings, so that
 * @f$\mathrm{d}x_1\,\mathrm{d}x_2\,\mathrm{d}\Phi_{n+1} = w\,\mathrm{d}r@f$ in
 * the normalization of @ref PhaseSpaceMapping: each dipole covers the phase
 * space of the real emission once, as a @ref PhaseSpaceMapping of the real
 * process does. The last three random numbers are those of the emission,
 * each squared in its variable towards the soft and collinear limits.
 *
 * A cutoff @f$c@f$ raises the lower end of both variables to @f$c@f$,
 * @f$y@f$ and @f$z_i@f$, @f$1 - x@f$ and @f$z@f$, and @f$v_i@f$ and
 * @f$1 - x - v_i@f$ (so @f$1 - x \ge 2 c@f$), the two products of the emitted
 * parton with the emitter and the spectator over their normalizations, so
 * that the mapping does not sample the soft and collinear limits of the
 * dipole below it, the technical cut of a real emission whose matrix element
 * loses its precision there. An empty range has a weight of zero, as the
 * inverse has below the cutoff.
 *
 * Cuts belong to the real emission and the subtraction terms, not to the
 * Born, so the Born mapping must have none, except a floor on
 * @f$\sqrt{\hat s}@f$ when emitter and spectator are in the final state. An
 * incoming emitter or spectator needs the momentum fractions of hadron beams.
 *
 * `batch` is the leading batch dimension and `n` the number of particles of
 * the Born, incoming included.
 *
 * **Inputs**
 * - `random` – `float`, shape `(batch, random_dim())` – the random numbers of
 *   the Born mapping, then those of the two variables of the dipole and of the
 *   azimuth of @f$i@f$.
 * - `discrete` – `int`, shape `(batch, discrete_dim())` – the discrete channel
 *   choices of the Born mapping. Present only when it has any.
 *
 * **Conditions**
 * - those of the Born mapping (`permutation_index`, `mirror_index`).
 *
 * **Outputs**
 * - `momenta` – `float`, shape `(batch, n + 1, 4)` – momenta of the real
 *   emission in the lab frame, in the order of the real process.
 * - `x1`, `x2` – `float`, shape `(batch,)` – momentum fractions of the real
 *   emission (1 for every point if the Born mapping is leptonic).
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] S. Catani, M. H. Seymour, "A general algorithm for calculating jet
 *   cross sections in NLO QCD", https://arxiv.org/abs/hep-ph/9605323 (Sec. 5)
 * - [3] S. Catani, S. Dittmaier, M. H. Seymour, Z. Trocsanyi, "The dipole
 *   formalism for next-to-leading order QCD calculations with massive
 *   partons", https://arxiv.org/abs/hep-ph/0201036 (Sec. 5)
 * - [4] L. Basso, S. Dittmaier, A. Huss, L. Oggero, "Techniques for the
 *   treatment of IR divergences in decay processes at NLO and application to
 *   the top-quark decay", https://arxiv.org/abs/1507.04676 (Sec. 3.2)
 */
class CSMapping : public Mapping {
public:
    /// The family of a dipole by the emitter and the spectator.
    enum Family {
        final_final,
        final_initial,
        initial_final,
        initial_initial,
        final_decay
    };

    /// @param born       Mapping of the Born process, without cuts (but a
    ///                   floor on sqrt(s_hat) for a final-state emitter and
    ///                   spectator).
    /// @param emitted    Position of the emitted massless parton i in the real
    ///                   process, in the final state.
    /// @param emitter    Position of the emitter j in the real process. The
    ///                   Born has the emitter ij there (one before it if it
    ///                   comes after i), with the mass of j.
    /// @param spectator  Position of the spectator k in the real process, 0
    ///                   for the decaying particle of a decay.
    /// @param cutoff     Lower end of the two variables of the dipole, in
    ///                   [0, 1).
    CSMapping(
        const PhaseSpaceMapping& born,
        std::size_t emitted,
        std::size_t emitter,
        std::size_t spectator,
        double cutoff = 0.
    );
    /// Number of continuous random inputs: those of the Born mapping and 3.
    std::size_t random_dim() const { return _born.random_dim() + 3; }
    /// Number of discrete inputs, those of the Born mapping.
    std::size_t discrete_dim() const override { return _born.discrete_dim(); }
    /// Number of particles of the real emission, incoming included.
    std::size_t particle_count() const { return _real_masses.size(); }
    /// Position of the emitted parton i in the real process.
    std::size_t emitted() const { return _emitted; }
    /// Position of the emitter j in the real process.
    std::size_t emitter() const { return _emitter; }
    /// Position of the spectator k in the real process.
    std::size_t spectator() const { return _spectator; }
    /// The family of the dipole.
    Family family() const { return _family; }
    /// The lower end of the two variables of the dipole.
    double cutoff() const { return _cutoff; }
    /// The Born mapping.
    const PhaseSpaceMapping& born() const { return _born; }
    /// The masses of the real emission (those of the Born mapping, zero for
    /// i).
    const std::vector<double>& real_masses() const { return _real_masses; }
    /**
     * Add the projection of a real emission onto the Born of the dipole to
     * @p fb, the momenta of the dipole term (the inverse map without the Born
     * mapping). @p inputs are the outputs of the forward map, in the lab
     * frame. Returns
     * - `born_momenta`, `born_x1`, `born_x2`: the Born of the dipole in the lab
     *   frame and its momentum fractions;
     * - `r_1`, `r_2`, `r_phi`: the random numbers of the emission;
     * - the two variables of the dipole, `y` and `z` for final_final and
     *   final_decay, `x` and `z` for final_initial and initial_final, `x` and
     *   `v` for initial_initial (see the instructions of the families);
     * - `det`: the Jacobian of the emission, that of the inverse map over the
     *   one of the Born mapping.
     */
    NamedVector<Value> build_projection(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions = {}
    ) const;

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
    // the position in the Born of a particle of the real emission other than i
    std::size_t born_position(std::size_t real) const {
        return real < _emitted ? real : real - 1;
    }

    PhaseSpaceMapping _born;
    std::size_t _emitted;
    std::size_t _emitter;
    std::size_t _spectator;
    Family _family;
    std::size_t _incoming;
    std::vector<double> _born_masses;
    std::vector<double> _real_masses;
    double _cutoff;
};

/**
 * The Catani-Seymour dipoles of a real process at its events: for each
 * dipole the Born it projects onto (@ref CSMapping::build_projection) and the
 * weights by which the dipole takes the Born blocks (the instructions
 * `cs_final_final_dipole`, `cs_final_initial_dipole`,
 * `cs_initial_final_dipole`, `cs_initial_initial_dipole` and
 * `cs_final_decay_dipole`),
 *
 * @f[ D = -g_s^2\,S \left[ c\,\langle T_j \cdot T_k \rangle
 *     + s\,\langle T_j \cdot T_k \rangle(k_t) \right], @f]
 *
 * with the color-correlated Born of the Born emitter @f$j@f$ and spectator
 * @f$k@f$ and the same with the real vector @f$k_t@f$ in place of the
 * polarization vector of a gluon emitter. @f$S@f$ is the symmetry factor of
 * the dipole (the identical particles of its Born over those of the real).
 * A dipole vanishes outside the alpha of its family, @f$y < \alpha y_+@f$
 * (ff), @f$1 - x < \alpha@f$ (fi), @f$u < \alpha@f$ (if), @f$v < \alpha@f$
 * (ii). The dipoles of a final-state quark with the decaying particle of a
 * decay as the spectator take @f$y < \alpha_{ff} y_+@f$ of their final-final
 * map, and the soft terms of the decaying particle (`cs_final_decay_dipole`).
 *
 * `batch` is the leading batch dimension, `d` the number of dipoles and `n`
 * the number of particles of their Borns.
 *
 * **Arguments**
 * - `momenta`, `x1`, `x2` – the outputs of the mappings, the real emission in
 *   the lab frame.
 * - the conditions of the mappings (`mirror_index`).
 *
 * **Returns**
 * - `born_momenta` – `float`, shape `(batch, d, n, 4)` – the Born of each
 *   dipole in the lab frame, in the order of its mapping (the real process
 *   without i).
 * - `born_x1`, `born_x2` – `float`, shape `(batch, d)` – their momentum
 *   fractions.
 * - `color_weight`, `spin_weight` – `float`, shape `(batch, d)` – @f$S c@f$
 *   and @f$S s@f$.
 * - `spin_vector` – `float`, shape `(batch, d, 4)` – @f$k_t@f$.
 *
 * **References**
 * - [1] S. Catani, M. H. Seymour, https://arxiv.org/abs/hep-ph/9605323
 * - [2] S. Catani, S. Dittmaier, M. H. Seymour, Z. Trocsanyi,
 *   https://arxiv.org/abs/hep-ph/0201036
 * - [3] Z. Nagy, https://arxiv.org/abs/hep-ph/0307268 (the alpha parameters)
 * - [4] L. Basso, S. Dittmaier, A. Huss, L. Oggero,
 *   https://arxiv.org/abs/1507.04676 (decays)
 */
class CSDipoles : public FunctionGenerator {
public:
    /// @param mappings  One CSMapping per dipole, all of the same real
    ///                  process, with Borns of the same number of particles,
    ///                  beams and conditions.
    /// @param kernels   The kernel of each dipole: qg, gg or gq for a
    ///                  final-state emitter, qg, gg, qq or gq for an
    ///                  incoming one (see madspace.cs), qg with a decaying
    ///                  spectator.
    /// @param symmetry  The symmetry factor of each dipole.
    /// @param alpha_ff, alpha_fi, alpha_if, alpha_ii  The alpha of each
    ///                  family, in (0, 1].
    /// @param kappa     kappa of the final-state gluon kernels with a
    ///                  massive spectator (CDST).
    CSDipoles(
        const std::vector<CSMapping>& mappings,
        const std::vector<std::string>& kernels,
        const std::vector<double>& symmetry,
        double alpha_ff = 1.,
        double alpha_fi = 1.,
        double alpha_if = 1.,
        double alpha_ii = 1.,
        double kappa = 2. / 3.
    );
    /// Number of dipoles.
    std::size_t dipole_count() const { return _mappings.size(); }
    /// The mappings of the dipoles.
    const std::vector<CSMapping>& mappings() const { return _mappings; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<CSMapping> _mappings;
    std::vector<double> _kernels;
    std::vector<double> _symmetry;
    double _alpha[4];
    double _kappa;
};

} // namespace madspace
