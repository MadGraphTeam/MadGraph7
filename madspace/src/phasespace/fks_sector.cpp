#include "madspace/phasespace/fks_sector.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace madspace;

namespace {

constexpr std::size_t INCOMING_COUNT = 2;

std::size_t checked_particle_count(
    const std::vector<std::size_t>& i_fks,
    const std::vector<std::size_t>& j_fks,
    const nested_vector3<std::size_t>& fks_j_from_i,
    const nested_vector2<int>& colors,
    const nested_vector2<int>& pdgs,
    const nested_vector2<double>& masses
) {
    std::size_t count = i_fks.size();
    if (count == 0) {
        throw std::invalid_argument(
            "FKSSectorFunction: needs at least one configuration"
        );
    }
    if (j_fks.size() != count || fks_j_from_i.size() != count ||
        colors.size() != count || pdgs.size() != count || masses.size() != count) {
        throw std::invalid_argument(
            "FKSSectorFunction: one entry per configuration in every argument"
        );
    }
    std::size_t n = pdgs.at(0).size();
    for (std::size_t c = 0; c < count; ++c) {
        if (fks_j_from_i.at(c).size() != n || colors.at(c).size() != n ||
            pdgs.at(c).size() != n || masses.at(c).size() != n) {
            throw std::invalid_argument(
                "FKSSectorFunction: the real processes of the configurations must "
                "have the same number of particles"
            );
        }
        if (i_fks.at(c) < INCOMING_COUNT || i_fks.at(c) >= n || j_fks.at(c) >= n ||
            i_fks.at(c) == j_fks.at(c)) {
            throw std::invalid_argument(
                "FKSSectorFunction: i_fks must be a final-state particle and "
                "j_fks another particle of the real process"
            );
        }
        for (const auto& js : fks_j_from_i.at(c)) {
            for (auto j : js) {
                if (j >= n) {
                    throw std::invalid_argument(
                        "FKSSectorFunction: FKS pair out of range"
                    );
                }
            }
        }
    }
    return n;
}

bool is_aorg(int pdg) { return std::abs(pdg) == 21 || pdg == 22; }

} // namespace

FKSSectorFunction::FKSSectorFunction(
    const std::vector<std::size_t>& i_fks,
    const std::vector<std::size_t>& j_fks,
    const nested_vector3<std::size_t>& fks_j_from_i,
    const nested_vector2<int>& colors,
    const nested_vector2<int>& pdgs,
    const nested_vector2<double>& masses,
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
            std::size_t n = checked_particle_count(
                i_fks, j_fks, fks_j_from_i, colors, pdgs, masses
            );
            NamedVector<Type> args{
                {"momenta", batch_four_vec_array(n)},
                {"p_i_hat", batch_four_vec},
            };
            args.push_back("fks_index", batch_int);
            return args;
        }(),
        {{"s", batch_float}}
    ),
    _particle_count(pdgs.at(0).size()),
    _masses(masses),
    _options{
        fks_a, fks_b, use_energy ? 1. : 0., use_beta ? 1. : 0., h_damp_one, h_damp_power
    } {
    std::size_t n = _particle_count;
    for (std::size_t c = 0; c < i_fks.size(); ++c) {
        _i_fks.push_back(i_fks.at(c));
        _j_fks.push_back(j_fks.at(c));
        // each unordered pair once, as the first of its two orientations in
        // fks_j_from_i (ijskip = 1 in fks_Sij.f); the second one is there
        // only for two gluons or photons
        nested_vector2<me_int_t> pairs(n, std::vector<me_int_t>(n, 0));
        for (std::size_t k = 0; k < n; ++k) {
            for (auto l : fks_j_from_i.at(c).at(k)) {
                if (pairs.at(k).at(l) != 0 || pairs.at(l).at(k) != 0) {
                    if (pairs.at(k).at(l) != 0 || !is_aorg(pdgs.at(c).at(k)) ||
                        !is_aorg(pdgs.at(c).at(l))) {
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
            octet.push_back(colors.at(c).at(k) == 8 ? 1 : 0);
            aorg.push_back(is_aorg(pdgs.at(c).at(k)) ? 1 : 0);
        }
        _octet.push_back(octet);
        _aorg.push_back(aorg);
        bool own_pair = pairs.at(i_fks.at(c)).at(j_fks.at(c)) != 0 ||
            pairs.at(j_fks.at(c)).at(i_fks.at(c)) != 0;
        if (!own_pair) {
            throw std::invalid_argument(
                "FKSSectorFunction: (i_fks, j_fks) of a configuration must be one "
                "of the FKS pairs of its real process"
            );
        }
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
        _options
    );
    return {{"s", s}};
}
