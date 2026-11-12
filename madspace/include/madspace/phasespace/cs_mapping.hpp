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
 *   @f$x_{i,ab}@f$ and @f$v_i@f$, with the final state Lorentz transformed.
 *
 * The emitter keeps its mass (@f$m_{ij} = m_j@f$, so not the splitting of a
 * gluon into massive quarks), the spectator may be massive, and incoming
 * partons are massless. The Born mapping gives the Born momenta in the lab
 * frame, they are boosted along the beam axis into the center-of-mass frame of
 * the Born, where the emission adds @f$i@f$, and the real emission is boosted
 * back with the same boost (the instruction `fks_beam_boost`). With an
 * incoming emitter or spectator @f$a@f$ the momentum fraction of @f$a@f$ is
 * that of the Born over @f$x@f$, from the Born fraction (the lower end of
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
 */
class CSMapping : public Mapping {
public:
    /// The family of a dipole by the emitter and the spectator.
    enum Family { final_final, final_initial, initial_final, initial_initial };

    /// @param born       Mapping of the Born process, without cuts (but a
    ///                   floor on sqrt(s_hat) for a final-state emitter and
    ///                   spectator).
    /// @param emitted    Position of the emitted massless parton i in the real
    ///                   process, in the final state.
    /// @param emitter    Position of the emitter j in the real process. The
    ///                   Born has the emitter ij there (one before it if it
    ///                   comes after i), with the mass of j.
    /// @param spectator  Position of the spectator k in the real process.
    CSMapping(
        const PhaseSpaceMapping& born,
        std::size_t emitted,
        std::size_t emitter,
        std::size_t spectator
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
    /// The Born mapping.
    const PhaseSpaceMapping& born() const { return _born; }
    /**
     * Add the projection of a real emission onto the Born of the dipole to
     * @p fb, the momenta of the dipole term (the inverse map without the Born
     * mapping). @p inputs are the outputs of the forward map, in the lab
     * frame. Returns
     * - `born_momenta`, `born_x1`, `born_x2`: the Born of the dipole in the lab
     *   frame and its momentum fractions;
     * - `r_1`, `r_2`, `r_phi`: the random numbers of the emission;
     * - the two variables of the dipole, `y` and `z` for final_final, `x` and
     *   `z` for final_initial and initial_final, `x` and `v` for
     *   initial_initial (see the instructions of the families);
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
};

} // namespace madspace
