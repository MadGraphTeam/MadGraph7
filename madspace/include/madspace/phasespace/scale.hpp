#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * Renormalization and factorization scales for an event.
 *
 * Computes the renormalization scale and the two factorization scales from the
 * event momenta. Each can be fixed to a constant or set to a dynamical choice
 * (Sec. 3.2.6 of [1], matching the choices in [2]):
 *
 * @f[
 *   \mu = \sum_i m_i, \qquad
 *   \mu = E_\mathrm{T} = \sum_i \frac{E_i\,p_{\mathrm{T},i}}{|\vec p_i|},
 * @f]
 * @f[
 *   \mu = H_\mathrm{T} = \sum_i \sqrt{m_i^2 + p_{\mathrm{T},i}^2},
 * @f]
 *
 * with @ref DynamicalScaleType selecting @f$E_\mathrm{T}@f$
 * (`transverse_energy`), @f$H_\mathrm{T}@f$ (`transverse_mass`),
 * @f$H_\mathrm{T}/2@f$ (`half_transverse_mass`) or @f$\sqrt{\hat s}@f$
 * (`partonic_energy`).
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, particle_count, 4)` – the event momenta.
 *
 * **Returns**
 * - `ren_scale` – `float`, shape `(batch,)` – renormalization scale.
 * - `fact_scale1` – `float`, shape `(batch,)` – factorization scale for the
 *   first beam.
 * - `fact_scale2` – `float`, shape `(batch,)` – factorization scale for the
 *   second beam.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.6)
 * - [2] V. Hirschi, O. Mattelaer, "Automated event generation for loop-induced
 *   processes", https://arxiv.org/abs/1507.00020
 */
class EnergyScale : public FunctionGenerator {
public:
    /// Dynamical scale choice; see the class description for the formulas.
    enum DynamicalScaleType {
        transverse_energy,    ///< Total transverse energy @f$E_\mathrm{T}@f$.
        transverse_mass,      ///< Sum of transverse masses @f$H_\mathrm{T}@f$.
        half_transverse_mass, ///< @f$H_\mathrm{T}/2@f$.
        partonic_energy       ///< Partonic energy @f$\sqrt{\hat s}@f$.
    };

    /// Dynamical @ref half_transverse_mass for all three scales.
    /// @param particle_count  Number of external particles.
    EnergyScale(std::size_t particle_count) :
        EnergyScale(particle_count, half_transverse_mass, false, false, 0., 0., 0.) {}
    /// Dynamical scale of type @p type for all three scales.
    /// @param particle_count  Number of external particles.
    /// @param type            The dynamical scale choice.
    EnergyScale(std::size_t particle_count, DynamicalScaleType type) :
        EnergyScale(particle_count, type, false, false, 0., 0., 0.) {}
    /// A single fixed value for all three scales.
    /// @param particle_count  Number of external particles.
    /// @param fixed_scale     The constant scale value.
    EnergyScale(std::size_t particle_count, double fixed_scale) :
        EnergyScale(
            particle_count,
            half_transverse_mass,
            true,
            true,
            fixed_scale,
            fixed_scale,
            fixed_scale
        ) {}
    /**
     * @param particle_count       Number of external particles.
     * @param dynamical_scale_type Scale choice used where a scale is not fixed.
     * @param ren_scale_fixed      Whether the renormalization scale is fixed.
     * @param fact_scale_fixed     Whether the factorization scales are fixed.
     * @param ren_scale            Fixed renormalization-scale value.
     * @param fact_scale1          Fixed factorization scale for the first beam.
     * @param fact_scale2          Fixed factorization scale for the second beam.
     */
    EnergyScale(
        std::size_t particle_count,
        DynamicalScaleType dynamical_scale_type,
        bool ren_scale_fixed,
        bool fact_scale_fixed,
        double ren_scale,
        double fact_scale1,
        double fact_scale2
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    DynamicalScaleType _dynamical_scale_type;
    bool _ren_scale_fixed;
    bool _fact_scale_fixed;
    double _ren_scale;
    double _fact_scale1;
    double _fact_scale2;
};

} // namespace madspace
