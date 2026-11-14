#include "madspace/nlo/scale.hpp"

#include <format>
#include <stdexcept>

using namespace madspace;

NLOScale::NLOScale(
    std::size_t particle_count,
    int dynamical_scale_choice,
    bool fixed_ren_scale,
    bool fixed_fac_scale,
    double mur_ref_fixed,
    double muf1_ref_fixed,
    double muf2_ref_fixed,
    double mur_over_ref,
    double muf1_over_ref,
    double muf2_over_ref,
    std::size_t incoming
) :
    FunctionGenerator(
        "NLOScale",
        NamedVector<Type>{
            {"momenta", batch_four_vec_array(static_cast<int>(particle_count))}
        },
        NamedVector<Type>{
            {"ren_scale", batch_float},
            {"fact_scale1", batch_float},
            {"fact_scale2", batch_float},
        }
    ),
    _particle_count(particle_count),
    _incoming(incoming),
    _options{
        static_cast<double>(dynamical_scale_choice),
        fixed_ren_scale ? 1. : 0.,
        fixed_fac_scale ? 1. : 0.,
        mur_ref_fixed,
        muf1_ref_fixed,
        muf2_ref_fixed,
        mur_over_ref,
        muf1_over_ref,
        muf2_over_ref,
        min_scale
    } {
    if (incoming != 1 && incoming != 2) {
        throw std::invalid_argument(
            "NLOScale: the kinematics needs two incoming particles or one (a "
            "decay)"
        );
    }
    if (particle_count < incoming + 1) {
        throw std::invalid_argument(
            "NLOScale: the kinematics needs an outgoing particle"
        );
    }
    int choice = dynamical_scale_choice;
    if (choice != -2 && choice != -1 && choice != 1 && choice != 2 && choice != 3) {
        throw std::invalid_argument(
            std::format(
                "NLOScale: dynamical_scale_choice {} is not one of -2, -1, 1, 2 and 3",
                choice
            )
        );
    }
}

NLOScale NLOScale::with_particle_count(std::size_t particle_count) const {
    return NLOScale(
        particle_count,
        static_cast<int>(_options.at(0)),
        _options.at(1) != 0.,
        _options.at(2) != 0.,
        _options.at(3),
        _options.at(4),
        _options.at(5),
        _options.at(6),
        _options.at(7),
        _options.at(8),
        _incoming
    );
}

NamedVector<Value> NLOScale::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto scales =
        fb.nlo_scale(args.at("momenta"), _options, static_cast<me_int_t>(_incoming));
    return {
        {"ren_scale", scales.at(0)},
        {"fact_scale1", scales.at(1)},
        {"fact_scale2", scales.at(2)},
    };
}
