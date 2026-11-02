#include "madspace/phasespace/fks_limits.hpp"

#include <algorithm>
#include <stdexcept>

using namespace madspace;

namespace {

constexpr std::size_t INCOMING_COUNT = 2;

std::size_t checked_particle_count(
    const std::vector<std::size_t>& i_fks,
    const std::vector<std::size_t>& j_fks,
    const nested_vector3<std::size_t>& fks_j_from_i,
    const nested_vector2<double>& masses
) {
    std::size_t count = i_fks.size();
    if (count == 0) {
        throw std::invalid_argument("FKSEikonals: needs at least one configuration");
    }
    if (j_fks.size() != count || fks_j_from_i.size() != count ||
        masses.size() != count) {
        throw std::invalid_argument(
            "FKSEikonals: one entry per configuration in every argument"
        );
    }
    std::size_t n = masses.at(0).size();
    for (std::size_t c = 0; c < count; ++c) {
        if (fks_j_from_i.at(c).size() != n || masses.at(c).size() != n) {
            throw std::invalid_argument(
                "FKSEikonals: the real processes of the configurations must have "
                "the same number of particles"
            );
        }
        if (i_fks.at(c) < INCOMING_COUNT || i_fks.at(c) >= n || j_fks.at(c) >= n ||
            i_fks.at(c) == j_fks.at(c)) {
            throw std::invalid_argument(
                "FKSEikonals: i_fks must be a final-state particle and j_fks "
                "another particle of the real process"
            );
        }
        const auto& partners = fks_j_from_i.at(c).at(i_fks.at(c));
        if (std::find(partners.begin(), partners.end(), j_fks.at(c)) ==
            partners.end()) {
            throw std::invalid_argument(
                "FKSEikonals: j_fks of a configuration must form an FKS pair with "
                "its i_fks"
            );
        }
        for (auto m : partners) {
            if (m >= n || m == i_fks.at(c)) {
                throw std::invalid_argument("FKSEikonals: FKS pair out of range");
            }
        }
    }
    return n;
}

} // namespace

FKSEikonals::FKSEikonals(
    const std::vector<std::size_t>& i_fks,
    const std::vector<std::size_t>& j_fks,
    const nested_vector3<std::size_t>& fks_j_from_i,
    const nested_vector2<double>& masses
) :
    FunctionGenerator(
        "FKSEikonals",
        [&] {
            std::size_t n = checked_particle_count(i_fks, j_fks, fks_j_from_i, masses);
            NamedVector<Type> args{
                {"momenta", batch_four_vec_array(n)},
                {"p_i_hat", batch_four_vec},
                {"y", batch_float},
            };
            if (i_fks.size() > 1) {
                args.push_back("fks_index", batch_int);
            }
            return args;
        }(),
        [&] {
            // the pairs of sbornsoft: every unordered pair of the partners of
            // i, a particle with itself only where it is massive
            std::size_t largest = 1;
            for (std::size_t c = 0; c < i_fks.size(); ++c) {
                const auto& partners = fks_j_from_i.at(c).at(i_fks.at(c));
                std::size_t count = 0;
                for (std::size_t k = 0; k < partners.size(); ++k) {
                    for (std::size_t l = 0; l <= k; ++l) {
                        if (k != l || masses.at(c).at(partners.at(k)) != 0.) {
                            ++count;
                        }
                    }
                }
                largest = std::max(largest, count);
            }
            return NamedVector<Type>{
                {"eikonals", batch_float_array(static_cast<int>(largest))}
            };
        }()
    ),
    _particle_count(masses.at(0).size()),
    _max_pair_count(1),
    _i_fks(i_fks),
    _j_fks(j_fks),
    _masses(masses) {
    for (std::size_t c = 0; c < i_fks.size(); ++c) {
        const auto& partners = fks_j_from_i.at(c).at(i_fks.at(c));
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        for (std::size_t k = 0; k < partners.size(); ++k) {
            for (std::size_t l = 0; l <= k; ++l) {
                std::size_t m = partners.at(k), n = partners.at(l);
                if (m != n || masses.at(c).at(m) != 0.) {
                    pairs.push_back({m, n});
                }
            }
        }
        _max_pair_count = std::max(_max_pair_count, pairs.size());
        _pairs.push_back(pairs);
    }
}

std::vector<std::pair<std::size_t, std::size_t>>
FKSEikonals::born_pairs(std::size_t index) const {
    std::size_t i = _i_fks.at(index);
    auto born = [i](std::size_t r) { return r < i ? r : r - 1; };
    std::vector<std::pair<std::size_t, std::size_t>> result;
    for (auto [m, n] : _pairs.at(index)) {
        result.push_back({born(m), born(n)});
    }
    return result;
}

NamedVector<Value> FKSEikonals::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    std::size_t count = configuration_count();
    Value index = count > 1 ? args.at("fks_index") : Value(static_cast<me_int_t>(0));
    std::vector<me_int_t> fks_j, pair_count;
    nested_vector2<me_int_t> pair_m, pair_n;
    for (std::size_t c = 0; c < count; ++c) {
        fks_j.push_back(_j_fks.at(c));
        const auto& pairs = _pairs.at(c);
        pair_count.push_back(pairs.size());
        std::vector<me_int_t> ms(_max_pair_count, 0), ns(_max_pair_count, 0);
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            ms.at(k) = pairs.at(k).first;
            ns.at(k) = pairs.at(k).second;
        }
        pair_m.push_back(ms);
        pair_n.push_back(ns);
    }
    auto eikonals = fb.fks_eikonals(
        args.at("momenta"),
        args.at("p_i_hat"),
        args.at("y"),
        index,
        fks_j,
        pair_count,
        pair_m,
        pair_n,
        _masses
    );
    return {{"eikonals", eikonals}};
}
