#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/fks_radiation.hpp"
#include "madspace/phasespace/phasespace.hpp"

namespace madspace {

/**
 * Phase space of a real emission in one FKS configuration: a Born phase-space
 * mapping followed by the FKS emission of @ref FKSRadiation, in the lab frame.
 *
 * The Born mapping, a @ref PhaseSpaceMapping of the Born process of the FKS
 * configuration, gives the Born momenta in the lab frame and the momentum
 * fractions of the Born. They are boosted along the beam axis into the
 * centre-of-mass frame of the Born, where @ref FKSRadiation adds the radiated
 * parton @f$i@f$ off the emitter @f$j@f$, and the real emission is boosted back
 * into the lab with the same boost (the instruction `fks_beam_boost`, which
 * keeps the precision of the momenta in the centre-of-mass frame for any
 * rapidity). For an incoming emitter the real emission
 * then has the momentum fractions of the real emission, and its incoming
 * momenta are those of the beams times them. The emitter is massive if the
 * Born mapping gives it a mass. As in genps_fks.f, the last three random
 * numbers are those of the emission.
 *
 * The inverse needs the frame of the Born from the real emission: for a
 * final-state emitter it is that of the incoming momenta, for an incoming
 * emitter that of the system recoiling against @f$i@f$ (the instruction
 * `fks_recoil_light_cone`), which has zero longitudinal momentum in the frame
 * of the Born.
 *
 * The weight is the product of the weights of the two mappings, so that
 * @f$\mathrm{d}x_1\,\mathrm{d}x_2\,\mathrm{d}\Phi_{n+1} = w\,\mathrm{d}r@f$ in
 * the normalization of @ref PhaseSpaceMapping: the weight of a
 * @ref PhaseSpaceMapping of the real process, with the factor @f$\xi@f$ that
 * genps_fks.f leaves to the FKS distributions.
 *
 * Cuts belong to the real emission and the counter-events, not to the Born, so
 * the Born mapping must have none. Its `mirror_index` condition is not
 * supported: the emission needs the first incoming parton along @f$+z@f$.
 *
 * `batch` is the leading batch dimension and `n` the number of particles of
 * the Born, incoming included.
 *
 * **Inputs**
 * - `random` – `float`, shape `(batch, random_dim())` – the random numbers of
 *   the Born mapping, then @f$r_\xi@f$, @f$r_y@f$ and @f$r_\phi@f$ of the
 *   emission.
 * - `discrete` – `int`, shape `(batch, discrete_dim())` – the discrete channel
 *   choices of the Born mapping. Present only when it has any.
 *
 * **Conditions**
 * - those of the Born mapping (`permutation_index`).
 *
 * **Outputs**
 * - `momenta` – `float`, shape `(batch, n + 1, 4)` – momenta of the real
 *   emission in the lab frame, in the order of the real process.
 * - `x1`, `x2` – `float`, shape `(batch,)` – momentum fractions of the real
 *   emission (1 if the Born mapping is leptonic).
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272 (Sec. 5)
 */
class FKSMapping : public Mapping {
public:
    /// @param born       Mapping of the Born process, without cuts and
    ///                   without `mirror_beams`.
    /// @param i_fks      Position of the radiated parton i in the real process.
    /// @param j_fks      Position of the emitter j in the real process.
    /// @param s_hat_min  Lower bound on the partonic energy squared of the
    ///                   real emission, for an incoming emitter. See
    ///                   @ref FKSRadiation.
    FKSMapping(
        const PhaseSpaceMapping& born,
        std::size_t i_fks,
        std::size_t j_fks,
        double s_hat_min = 0.
    );
    /// Number of continuous random inputs: those of the Born mapping and 3.
    std::size_t random_dim() const { return _born.random_dim() + 3; }
    /// Number of discrete inputs, those of the Born mapping.
    std::size_t discrete_dim() const override { return _born.discrete_dim(); }
    /// Number of particles of the real emission, incoming included.
    std::size_t particle_count() const { return _real_masses.size(); }
    std::size_t i_fks() const { return _radiation.i_fks(); }
    std::size_t j_fks() const { return _radiation.j_fks(); }
    /// The Born mapping.
    const PhaseSpaceMapping& born() const { return _born; }
    /// The emission.
    const FKSRadiation& radiation() const { return _radiation; }

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
    // light-cone components (up to a factor) whose ratio is exp(2 y) for the
    // rapidity y of the centre-of-mass frame of the Born in the lab frame, from
    // its momentum fractions (the arguments of fks_beam_boost)
    std::pair<Value, Value> born_frame(FunctionBuilder& fb, Value x1, Value x2) const;

    PhaseSpaceMapping _born;
    FKSRadiation _radiation;
    std::vector<double> _born_masses;
    std::vector<double> _real_masses;
};

} // namespace madspace
