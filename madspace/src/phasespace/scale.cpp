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
    double min_scale,
    double max_scale
) :
    FunctionGenerator(
        "EnergyScale",
        {{"momenta", batch_four_vec_array(particle_count)}},
        (min_scale > 0. || max_scale > 0.)
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
    _min_scale(min_scale),
    _max_scale(max_scale) {}

EnergyScale::EnergyScale(
    const MLMClustering& clustering, double min_scale, double max_scale
) :
    FunctionGenerator(
        "EnergyScale",
        clustering.arg_types(),
        (min_scale > 0. || max_scale > 0.)
            ? [&] {
                  auto types = clustering.return_types();
                  types.push_back("scale_weight", batch_float);
                  return types;
              }()
            : clustering.return_types()
    ),
    _min_scale(min_scale),
    _max_scale(max_scale),
    _clustering(clustering) {}

// The range of scales an event may be evaluated at, applied to whatever
// dynamical scale choice produced them. A PDF grid only covers a band in Q -
// 1 to 10000 GeV for NNPDF23, say - and outside it the densities are not
// defined: below, a scale of a few MeV, above, one of several TeV. Either
// comes back as a NaN that no later cut can remove, because the pdf is
// evaluated before the veto weight multiplies it and NaN times zero is still
// NaN. So an event outside the range is both vetoed, through the scale_weight
// output, and clamped, which keeps what is computed on the way finite.
//
// madevent applies the same floor to mu_F, at 2 GeV, in reweight.f. The upper
// end has no counterpart there: LHAPDF extrapolates rather than returning a
// hole, so madevent never has to look.
NamedVector<Value> EnergyScale::apply_scale_range(
    FunctionBuilder& fb, NamedVector<Value> scales
) const {
    if (_min_scale <= 0. && _max_scale <= 0.) {
        return scales;
    }
    double low = _min_scale > 0. ? _min_scale : 0.;
    double high = _max_scale > 0. ? _max_scale : 1e30;
    Value weight;
    for (auto name : {"fact_scale1", "fact_scale2", "ren_scale"}) {
        auto& scale = scales.at(name);
        auto pass = fb.cut_one(scale, low, high);
        weight = weight ? fb.mul(weight, pass) : pass;
        auto batch_size = fb.batch_size({scale});
        scale = fb.max(scale, fb.full({low, batch_size}));
        scale = fb.min(scale, fb.full({high, batch_size}));
    }
    scales.push_back("scale_weight", weight);
    return scales;
}

NamedVector<Value> EnergyScale::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto momenta = args.at(0);
    if (_clustering) {
        return apply_scale_range(fb, _clustering.value().build_function(fb, args));
    }
    if (_ren_scale_fixed && _fact_scale_fixed) {
        auto batch_size = fb.batch_size({momenta});
        return apply_scale_range(fb, {
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
    return apply_scale_range(fb, {
        {"ren_scale", _ren_scale_fixed ? fb.full({_ren_scale, batch_size}) : scale},
        {"fact_scale1",
         _fact_scale_fixed ? fb.full({_fact_scale1, batch_size}) : scale},
        {"fact_scale2", _fact_scale_fixed ? fb.full({_fact_scale2, batch_size}) : scale}
    });
}
