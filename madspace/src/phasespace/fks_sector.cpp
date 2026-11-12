#include "madspace/phasespace/fks_sector.hpp"

#include <cstdlib>
#include <stdexcept>

using namespace madspace;

namespace {

bool is_aorg(int pdg) { return std::abs(pdg) == 21 || pdg == 22; }

} // namespace

FKSSectorFunction::FKSSectorFunction(
    const std::vector<FKSConfiguration>& configurations,
    double fks_a,
    double fks_b,
    bool use_energy,
    bool use_beta,
    double h_damp_one,
    double h_damp_power
) :
    FunctionGenerator(
        "FKSSectorFunction",
        [&] {
            std::size_t n =
                check_fks_configurations(configurations, "FKSSectorFunction");
            return NamedVector<Type>{
                {"momenta", batch_four_vec_array(n)},
                {"p_i_hat", batch_four_vec},
                {"fks_index", batch_int},
            };
        }(),
        {{"s", batch_float}}
    ),
    _particle_count(configurations.at(0).particle_count()),
    _incoming(configurations.at(0).incoming),
    _options{
        fks_a, fks_b, use_energy ? 1. : 0., use_beta ? 1. : 0., h_damp_one, h_damp_power
    } {
    std::size_t n = _particle_count;
    for (const auto& config : configurations) {
        _i_fks.push_back(config.i_fks);
        _j_fks.push_back(config.j_fks);
        _masses.push_back(config.masses);
        // each unordered pair once, as the first of its two orientations in
        // fks_j_from_i (ijskip = 1 in fks_Sij.f); the second one is there
        // only for two gluons or photons. A decay leaves out the pairs with
        // the decaying particle, as fks_Sij.f for nincoming != 2.
        nested_vector2<me_int_t> pairs(n, std::vector<me_int_t>(n, 0));
        for (std::size_t k = 0; k < n; ++k) {
            for (auto l : config.fks_j_from_i.at(k)) {
                if (config.incoming == 1 && (k < 1 || l < 1)) {
                    continue;
                }
                if (pairs.at(k).at(l) != 0 || pairs.at(l).at(k) != 0) {
                    if (pairs.at(k).at(l) != 0 || !is_aorg(config.pdgs.at(k)) ||
                        !is_aorg(config.pdgs.at(l))) {
                        throw std::invalid_argument(
                            "FKSSectorFunction: an FKS pair appears twice, and "
                            "not as two gluons or photons"
                        );
                    }
                    continue;
                }
                pairs.at(k).at(l) = 1;
            }
        }
        _pairs.push_back(pairs);
        std::vector<me_int_t> octet, aorg;
        for (std::size_t k = 0; k < n; ++k) {
            octet.push_back(config.colors.at(k) == 8 ? 1 : 0);
            aorg.push_back(is_aorg(config.pdgs.at(k)) ? 1 : 0);
        }
        _octet.push_back(octet);
        _aorg.push_back(aorg);
    }
}

std::vector<std::pair<std::size_t, std::size_t>>
FKSSectorFunction::pairs(std::size_t index) const {
    std::vector<std::pair<std::size_t, std::size_t>> result;
    const auto& pairs = _pairs.at(index);
    for (std::size_t k = 0; k < _particle_count; ++k) {
        for (std::size_t l = 0; l < _particle_count; ++l) {
            if (pairs.at(k).at(l) != 0) {
                result.push_back({k, l});
            }
        }
    }
    return result;
}

NamedVector<Value> FKSSectorFunction::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value index = args.at("fks_index");
    // the pairs as a (c, n, n) tensor, laid out as Value lays out nested
    // vectors: the first index fastest
    std::size_t count = configuration_count(), n = _particle_count;
    std::vector<me_int_t> flat_pairs(count * n * n);
    for (std::size_t c = 0; c < count; ++c) {
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t l = 0; l < n; ++l) {
                flat_pairs.at(c + count * (k + n * l)) = _pairs.at(c).at(k).at(l);
            }
        }
    }
    Value pairs(
        flat_pairs, {static_cast<int>(count), static_cast<int>(n), static_cast<int>(n)}
    );
    auto s = fb.fks_sector_function(
        args.at("momenta"),
        args.at("p_i_hat"),
        index,
        _i_fks,
        _j_fks,
        pairs,
        _octet,
        _aorg,
        _masses,
        _options,
        static_cast<me_int_t>(_incoming)
    );
    return {{"s", s}};
}
