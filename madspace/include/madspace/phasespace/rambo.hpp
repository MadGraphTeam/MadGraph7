#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * Phase-space mapping based on the RAMBO algorithm.
 *
 * Builds the momenta of an @p n_particles final state from a hypercube of
 * uniform random numbers. The forward mapping expects `3 * n_particles - 4`
 * inputs named `random_i`, plus a `com_momentum` four-vector when @p com is
 * false, and the conditions `com_energy` (and `mass_i` per particle when
 * @p massless is false). It returns the four-momenta `momentum_i` together
 * with the phase-space weight.
 */
class FastRamboMapping : public Mapping {
public:
    /**
     * Construct the mapping for a fixed multiplicity.
     *
     * @param n_particles Number of final-state particles; must be in [3, 12].
     * @param massless    If true all final-state particles are massless,
     *                    otherwise the masses are read from the conditions.
     * @param com         If true the momenta are generated in the
     *                    centre-of-mass frame, otherwise the total incoming
     *                    momentum is taken from the `com_momentum` input.
     */
    FastRamboMapping(std::size_t n_particles, bool massless, bool com = true);

    /// Number of uniform random inputs consumed by the forward mapping,
    /// equal to `3 * n_particles - 4`.
    std::size_t random_dim() const { return 3 * _n_particles - 4; }

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

    std::size_t _n_particles;
    bool _massless;
    double _com;
};

} // namespace madspace
