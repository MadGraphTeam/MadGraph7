#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/mlm_clustering.hpp"

namespace madspace {

class EnergyScale : public FunctionGenerator {
public:
    enum DynamicalScaleType {
        transverse_energy,
        transverse_mass,
        half_transverse_mass,
        partonic_energy
    };

    EnergyScale(std::size_t particle_count) :
        EnergyScale(particle_count, half_transverse_mass, false, false, 0., 0., 0.) {}
    EnergyScale(
        std::size_t particle_count,
        DynamicalScaleType type,
        double min_scale = 0.,
        double max_scale = 0.
    ) :
        EnergyScale(
            particle_count, type, false, false, 0., 0., 0., min_scale, max_scale
        ) {}
    EnergyScale(
        std::size_t particle_count,
        double fixed_scale,
        double min_scale = 0.,
        double max_scale = 0.
    ) :
        EnergyScale(
            particle_count,
            half_transverse_mass,
            true,
            true,
            fixed_scale,
            fixed_scale,
            fixed_scale,
            min_scale,
            max_scale
        ) {}
    EnergyScale(
        std::size_t particle_count,
        DynamicalScaleType dynamical_scale_type,
        bool ren_scale_fixed,
        bool fact_scale_fixed,
        double ren_scale,
        double fact_scale1,
        double fact_scale2,
        // Floor on mu_R and mu_F. An event below it is vetoed through the
        // scale_weight output and the scales are clamped, so that a pdf is
        // never asked for a density below the bottom of its grid. Zero
        // disables it.
        double min_scale = 0.,
        // Upper end of the same range, normally the top of the PDF grid.
        double max_scale = 0.
    );
    EnergyScale(
        const MLMClustering& clustering,
        double min_scale = 0.,
        double max_scale = 0.
    );

    bool is_mlm() const { return _clustering.has_value(); }

    bool has_scale_range() const {
        return _min_scale > 0. || _max_scale > 0.;
    }

private:
    NamedVector<Value> apply_scale_range(
        FunctionBuilder& fb, NamedVector<Value> scales
    ) const;

    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    DynamicalScaleType _dynamical_scale_type;
    bool _ren_scale_fixed;
    bool _fact_scale_fixed;
    double _ren_scale;
    double _fact_scale1;
    double _fact_scale2;
    double _min_scale;
    double _max_scale;
    std::optional<MLMClustering> _clustering;
};

} // namespace madspace
