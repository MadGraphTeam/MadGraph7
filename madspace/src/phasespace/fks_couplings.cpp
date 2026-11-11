#include "madspace/phasespace/fks_couplings.hpp"

#include <algorithm>
#include <format>
#include <set>
#include <stdexcept>

using namespace madspace;

FKSScale::FKSScale(
    std::size_t particle_count,
    int dynamical_scale_choice,
    bool fixed_ren_scale,
    bool fixed_fac_scale,
    double mur_ref_fixed,
    double muf1_ref_fixed,
    double muf2_ref_fixed,
    double mur_over_ref,
    double muf1_over_ref,
    double muf2_over_ref
) :
    FunctionGenerator(
        "FKSScale",
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
    if (particle_count < 3) {
        throw std::invalid_argument(
            "FKSScale: the kinematics needs two incoming particles and an "
            "outgoing one"
        );
    }
    int choice = dynamical_scale_choice;
    if (choice != -2 && choice != -1 && choice != 1 && choice != 2 && choice != 3) {
        throw std::invalid_argument(
            std::format(
                "FKSScale: dynamical_scale_choice {} is not one of -2, -1, 1, 2 and 3",
                choice
            )
        );
    }
}

FKSScale FKSScale::with_particle_count(std::size_t particle_count) const {
    return FKSScale(
        particle_count,
        static_cast<int>(_options.at(0)),
        _options.at(1) != 0.,
        _options.at(2) != 0.,
        _options.at(3),
        _options.at(4),
        _options.at(5),
        _options.at(6),
        _options.at(7),
        _options.at(8)
    );
}

NamedVector<Value> FKSScale::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto scales = fb.fks_scale(args.at("momenta"), _options);
    return {
        {"ren_scale", scales.at(0)},
        {"fact_scale1", scales.at(1)},
        {"fact_scale2", scales.at(2)},
    };
}

FKSLuminosity::FKSLuminosity(
    const nested_vector3<me_int_t>& flavors,
    const std::optional<PdfGrid>& pdf_grid,
    const std::optional<PdfGrid>& pdf_grid2,
    const std::string& pdf2_prefix,
    const std::array<bool, 2>& densities,
    bool sum_flavors,
    const nested_vector2<double>& factors
) :
    FunctionGenerator(
        "FKSLuminosity",
        [&] {
            NamedVector<Type> args{
                {"x1", batch_float},
                {"x2", batch_float},
                {"fact_scale1", batch_float},
                {"fact_scale2", batch_float},
            };
            if (!sum_flavors) {
                args.push_back("flavor_index", batch_int);
            }
            args.push_back("fks_index", batch_int);
            return args;
        }(),
        NamedVector<Type>{{"luminosity", batch_float}}
    ),
    _sum_flavors(sum_flavors),
    _q_min{0., 0.},
    _q_max{0., 0.} {
    if (flavors.empty()) {
        throw std::invalid_argument("FKSLuminosity: no FKS configurations");
    }
    if (pdf_grid2 && !pdf_grid) {
        throw std::invalid_argument(
            "FKSLuminosity: a grid for the second beam needs one for the first"
        );
    }
    if (pdf_grid && !densities.at(0) && !densities.at(1)) {
        throw std::invalid_argument(
            "FKSLuminosity: a grid needs a beam with densities"
        );
    }
    if (!factors.empty() && factors.size() != flavors.size()) {
        throw std::invalid_argument(
            "FKSLuminosity: the factors need the FKS configurations of the flavors"
        );
    }
    std::size_t flavor_max = 0;
    for (std::size_t c = 0; c < flavors.size(); ++c) {
        const auto& configuration = flavors.at(c);
        if (configuration.empty()) {
            throw std::invalid_argument(
                "FKSLuminosity: an FKS configuration without flavors"
            );
        }
        if (!factors.empty() && factors.at(c).size() != configuration.size()) {
            throw std::invalid_argument(
                "FKSLuminosity: a factor for each flavor of each FKS configuration"
            );
        }
        for (const auto& flavor : configuration) {
            if (flavor.size() != 2) {
                throw std::invalid_argument(
                    "FKSLuminosity: a flavor needs the PDG codes of the two "
                    "incoming partons"
                );
            }
        }
        flavor_max = std::max(flavor_max, configuration.size());
    }
    // the flavors of each beam with densities in the grid, in their order
    std::array<std::vector<int>, 2> pids;
    for (std::size_t beam = 0; beam < 2; ++beam) {
        const auto& grid = beam == 1 && pdf_grid2 ? pdf_grid2 : pdf_grid;
        if (!grid || !densities.at(beam)) {
            continue;
        }
        std::set<int> beam_pids;
        for (const auto& configuration : flavors) {
            for (const auto& flavor : configuration) {
                int pid = static_cast<int>(flavor.at(beam));
                if (std::find(grid->pids.begin(), grid->pids.end(), pid) !=
                    grid->pids.end()) {
                    beam_pids.insert(pid);
                }
            }
        }
        pids.at(beam) = {beam_pids.begin(), beam_pids.end()};
        if (pids.at(beam).empty()) {
            // no density of the beam: PartonDensity still wants a flavor
            pids.at(beam).push_back(grid->pids.at(0));
        }
        _pdfs.at(beam) = PartonDensity(
            *grid, pids.at(beam), false, beam == 1 && pdf_grid2 ? pdf2_prefix : ""
        );
        _q_min.at(beam) = grid->q.front();
        _q_max.at(beam) = grid->q.back();
    }
    // the positions of each flavor in the densities, -2 for none (no density),
    // 0 for a beam without densities (one), -1 after the last
    for (std::size_t c = 0; c < flavors.size(); ++c) {
        std::vector<me_int_t> positions;
        for (const auto& flavor : flavors.at(c)) {
            for (std::size_t beam = 0; beam < 2; ++beam) {
                if (!_pdfs.at(beam)) {
                    positions.push_back(0);
                    continue;
                }
                // the densities have the flavors of the grid only
                const auto& beam_pids = pids.at(beam);
                auto found = std::find(
                    beam_pids.begin(),
                    beam_pids.end(),
                    static_cast<int>(flavor.at(beam))
                );
                positions.push_back(
                    found == beam_pids.end()
                        ? -2
                        : static_cast<me_int_t>(found - beam_pids.begin())
                );
            }
        }
        positions.resize(2 * flavor_max, -1);
        _positions.push_back(positions);
        std::vector<double> weights = factors.empty()
            ? std::vector<double>(flavors.at(c).size(), 1.)
            : factors.at(c);
        weights.resize(flavor_max, 0.);
        _factors.push_back(weights);
    }
}

NamedVector<Value> FKSLuminosity::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto batch_size = fb.batch_size({args.at("x1")});
    std::array<Value, 2> pdfs, xs;
    for (std::size_t beam = 0; beam < 2; ++beam) {
        if (!_pdfs.at(beam)) {
            // a beam without densities: one per flavor
            auto one = fb.full({1., batch_size});
            pdfs.at(beam) = fb.unsqueeze(one);
            xs.at(beam) = one;
            continue;
        }
        Value x = args.at(beam == 0 ? "x1" : "x2");
        Value q = args.at(beam == 0 ? "fact_scale1" : "fact_scale2");
        q = fb.min(
            fb.max(q, fb.full({_q_min.at(beam), batch_size})),
            fb.full({_q_max.at(beam), batch_size})
        );
        pdfs.at(beam) = _pdfs.at(beam)->build_function(fb, {x, q}).at(0);
        xs.at(beam) = x;
    }
    Value flavor = _sum_flavors
        ? fb.full({static_cast<me_int_t>(-1), batch_size})
        : args.at("flavor_index");
    return {
        {"luminosity",
         fb.fks_luminosity(
             pdfs.at(0),
             pdfs.at(1),
             xs.at(0),
             xs.at(1),
             args.at("fks_index"),
             flavor,
             _positions,
             _factors
         )}
    };
}
