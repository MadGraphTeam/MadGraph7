#include "madspace/phasespace/scale.hpp"

using namespace madspace;

EnergyScale::EnergyScale(
    std::size_t particle_count,
    DynamicalScaleType dynamical_scale_type,
    bool ren_scale_fixed,
    bool fact_scale_fixed,
    double ren_scale,
    double fact_scale1,
    double fact_scale2,
    double min_scale
) :
    FunctionGenerator(
        "EnergyScale",
        {{"momenta", batch_four_vec_array(particle_count)}},
        min_scale > 0.
            ? NamedVector<Type>{{"ren_scale", batch_float},
                                {"fact_scale1", batch_float},
                                {"fact_scale2", batch_float},
                                {"scale_weight", batch_float}}
            : NamedVector<Type>{{"ren_scale", batch_float},
                                {"fact_scale1", batch_float},
                                {"fact_scale2", batch_float}}
    ),
    _dynamical_scale_type(dynamical_scale_type),
    _ren_scale_fixed(ren_scale_fixed),
    _fact_scale_fixed(fact_scale_fixed),
    _ren_scale(ren_scale),
    _fact_scale1(fact_scale1),
    _fact_scale2(fact_scale2),
    _min_scale(min_scale) {}

EnergyScale::EnergyScale(const MLMClustering& clustering, double min_scale) :
    FunctionGenerator(
        "EnergyScale",
        clustering.arg_types(),
        min_scale > 0.
            ? [&] {
                  auto types = clustering.return_types();
                  types.push_back("scale_weight", batch_float);
                  return types;
              }()
            : clustering.return_types()
    ),
    _min_scale(min_scale),
    _clustering(clustering) {}

// A floor on the scales the event is evaluated at, applied to whatever
// dynamical scale choice produced them. Below the lowest Q of a PDF grid the
// densities are not defined, and a scale of a few MeV comes back as a NaN that
// no later cut can remove, so the event is both vetoed and clamped: the veto
// weight drops it, the clamp keeps what is computed on the way finite.
// madevent applies the same floor to mu_F, at 2 GeV, in reweight.f.
NamedVector<Value> EnergyScale::apply_min_scale(
    FunctionBuilder& fb, NamedVector<Value> scales
) const {
    if (_min_scale <= 0.) {
        return scales;
    }
    Value weight;
    for (auto name : {"fact_scale1", "fact_scale2", "ren_scale"}) {
        auto& scale = scales.at(name);
        auto pass = fb.cut_one(scale, _min_scale, 1e30);
        weight = weight ? fb.mul(weight, pass) : pass;
        scale = fb.max(scale, fb.full({_min_scale, fb.batch_size({scale})}));
    }
    scales.push_back("scale_weight", weight);
    return scales;
}

NamedVector<Value> EnergyScale::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto momenta = args.at(0);
    if (_clustering) {
        return apply_min_scale(fb, _clustering.value().build_function(fb, args));
    }
    if (_ren_scale_fixed && _fact_scale_fixed) {
        auto batch_size = fb.batch_size({momenta});
        return apply_min_scale(fb, {
            {"ren_scale", fb.full({_ren_scale, batch_size})},
            {"fact_scale1", fb.full({_fact_scale1, batch_size})},
            {"fact_scale2", fb.full({_fact_scale2, batch_size})},
        });
    }
    Value scale;
    switch (_dynamical_scale_type) {
    case transverse_energy:
        scale = fb.scale_transverse_energy(momenta);
        break;
    case transverse_mass:
        scale = fb.scale_transverse_mass(momenta);
        break;
    case half_transverse_mass:
        scale = fb.scale_half_transverse_mass(momenta);
        break;
    case partonic_energy:
        scale = fb.scale_partonic_energy(momenta);
        break;
    default:
        throw std::runtime_error("invalid dynamical scale type");
    }
    auto batch_size = fb.batch_size({momenta});
    return apply_min_scale(fb, {
        {"ren_scale", _ren_scale_fixed ? fb.full({_ren_scale, batch_size}) : scale},
        {"fact_scale1",
         _fact_scale_fixed ? fb.full({_fact_scale1, batch_size}) : scale},
        {"fact_scale2", _fact_scale_fixed ? fb.full({_fact_scale2, batch_size}) : scale}
    });
}
