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
        {{"ren_scale", batch_float},
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
// evaluated before any veto weight multiplies it and NaN times zero is still
// NaN. So the scales are clamped into the range, which is what LHAPDF's
// freezing does for madevent.
//
// An MLM clustering also vetoes the event, through the scale_weight output:
// madevent's setclscales drops a merged event whose factorisation scale is
// under 2 GeV (reweight.f) rather than evaluating it at a frozen density. No
// other scale choice has that veto in madevent, so none has it here. The upper
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
    std::vector<const char*> names{"fact_scale1", "fact_scale2", "ren_scale"};
    if (_clustering && _clustering->pdf_reweighting()) {
        // The scale the density is actually asked for under pdf reweighting.
        // It sits below the factorisation scale, so clamping that one is not
        // enough to keep the density inside the grid, and madevent applies its
        // own 2 GeV floor to the lowered scale too rather than to the central
        // one.
        names.push_back("pdf_scale1");
        names.push_back("pdf_scale2");
    }
    for (auto name : names) {
        auto& scale = scales.at(name);
        if (_clustering) {
            auto pass = fb.cut_one(scale, low, high);
            weight = weight ? fb.mul(weight, pass) : pass;
        }
        auto batch_size = fb.batch_size({scale});
        scale = fb.max(scale, fb.full({low, batch_size}));
        scale = fb.min(scale, fb.full({high, batch_size}));
    }
    if (weight) {
        scales.push_back("scale_weight", weight);
    }
    return scales;
}

NamedVector<Value> EnergyScale::build_mlm_from_start_state(
    FunctionBuilder& fb,
    Value momenta,
    Value start_state,
    Value flavor_index,
    const std::vector<me_int_t>& leg_flavors
) const {
    return apply_scale_range(
        fb,
        _clustering.value().build_from_start_state(
            fb, momenta, start_state, flavor_index, leg_flavors
        )
    );
}

NamedVector<Value> EnergyScale::build_mlm_with_flavors(
    FunctionBuilder& fb,
    Value momenta,
    Value flavor_index,
    const std::vector<me_int_t>& leg_flavors
) const {
    return apply_scale_range(
        fb,
        _clustering.value().build_with_flavors(fb, momenta, flavor_index, leg_flavors)
    );
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
