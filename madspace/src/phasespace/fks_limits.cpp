#include "madspace/phasespace/fks_limits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

using namespace madspace;

namespace {

constexpr std::size_t INCOMING_COUNT = 2;
constexpr double CA = 3., TR = 0.5, CF = 4. / 3.;

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
            args.push_back("fks_index", batch_int);
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
    Value index = args.at("fks_index");
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

namespace {

struct Splitting {
    int kind;
    std::vector<double> factors;
};

// The type and charge of the mother i + j, as get_mother_col_charge
std::pair<int, double>
mother(int i_type, double ch_i, int j_type, double ch_j, bool initial) {
    if (std::abs(i_type) == std::abs(j_type) && std::abs(ch_i) == std::abs(ch_j) &&
        std::abs(i_type) > 1) {
        return {8, 0.};
    }
    if (std::abs(i_type) == std::abs(j_type) && std::abs(ch_i) == std::abs(ch_j) &&
        std::abs(i_type) == 1) {
        return {1, 0.};
    }
    if ((std::abs(i_type) == 3 && j_type == 8) || (ch_i != 0. && ch_j == 0.)) {
        if (!initial) {
            throw std::invalid_argument(
                "FKSSplittingKernels: a quark i with a gluon or photon j in the "
                "final state"
            );
        }
        int type = -i_type;
        return {type == -1 ? 1 : type, -ch_i};
    }
    if ((i_type == 8 && std::abs(j_type) == 3) || (ch_i == 0. && ch_j != 0.)) {
        return {j_type, ch_j};
    }
    throw std::invalid_argument("FKSSplittingKernels: no splitting of i and j");
}

// The splitting function and factors of AP_reduced and
// Qterms_reduced_timelike/spacelike for the particles 1 and 2 of the splitting
Splitting splitting(int col1, int col2, double ch1, double ch2, bool final_state) {
    if (col1 == 8 && col2 == 8) {
        return {0, {2. * CA, 0., -4. * CA, 0.}};
    }
    if ((std::abs(col1) == 3 && std::abs(col2) == 3) || (ch1 != 0. && ch2 != 0.)) {
        double qed = std::abs(col1) * ch1 * ch1;
        if (final_state) {
            return {1, {TR, qed, 4. * TR, 4. * qed}};
        }
        return {1, {TR, qed, 0., 0.}};
    }
    if ((std::abs(col1) == 3 && col2 == 8) || (ch1 != 0. && ch2 == 0.)) {
        return {2, {CF, ch1 * ch1, 0., 0.}};
    }
    if ((col1 == 8 && std::abs(col2) == 3) || (ch1 == 0. && ch2 != 0.)) {
        if (final_state) {
            return {3, {CF, ch2 * ch2, 0., 0.}};
        }
        return {3, {CF, ch2 * ch2, -4. * CF, -4. * ch2 * ch2}};
    }
    throw std::invalid_argument("FKSSplittingKernels: no splitting of i and j");
}

} // namespace

FKSSplittingKernels::FKSSplittingKernels(
    const std::vector<std::size_t>& i_fks,
    const std::vector<std::size_t>& j_fks,
    const nested_vector2<int>& colors,
    const nested_vector2<double>& charges
) :
    FunctionGenerator(
        "FKSSplittingKernels",
        [&] {
            std::size_t count = i_fks.size();
            if (count == 0 || j_fks.size() != count || colors.size() != count ||
                charges.size() != count) {
                throw std::invalid_argument(
                    "FKSSplittingKernels: one entry per configuration (at least "
                    "one) in every argument"
                );
            }
            std::size_t n = colors.at(0).size();
            for (std::size_t c = 0; c < count; ++c) {
                if (colors.at(c).size() != n || charges.at(c).size() != n) {
                    throw std::invalid_argument(
                        "FKSSplittingKernels: the real processes of the "
                        "configurations must have the same number of particles"
                    );
                }
                if (i_fks.at(c) < INCOMING_COUNT || i_fks.at(c) >= n ||
                    j_fks.at(c) >= n || i_fks.at(c) == j_fks.at(c)) {
                    throw std::invalid_argument(
                        "FKSSplittingKernels: i_fks must be a final-state particle "
                        "and j_fks another particle of the real process"
                    );
                }
            }
            NamedVector<Type> args{
                {"momenta", batch_four_vec_array(n)},
                {"p_i_hat", batch_four_vec},
                {"born_momenta", batch_four_vec_array(n - 1)},
            };
            args.push_back("fks_index", batch_int);
            return args;
        }(),
        {{"z", batch_float},
         {"ap", batch_float_array(2)},
         {"q_terms", batch_float_array(2)},
         {"k_perp", batch_four_vec}}
    ),
    _particle_count(colors.at(0).size()),
    _i_fks(i_fks),
    _j_fks(j_fks) {
    for (std::size_t c = 0; c < i_fks.size(); ++c) {
        std::size_t i = i_fks.at(c), j = j_fks.at(c);
        int i_type = colors.at(c).at(i), j_type = colors.at(c).at(j);
        double ch_i = charges.at(c).at(i), ch_j = charges.at(c).at(j);
        bool final_state = j >= INCOMING_COUNT;
        Splitting split;
        if (final_state) {
            mother(i_type, ch_i, j_type, ch_j, false);
            split = splitting(j_type, i_type, ch_j, ch_i, true);
        } else {
            auto [m_type, ch_m] = mother(i_type, ch_i, j_type, ch_j, true);
            split = splitting(m_type, i_type, ch_m, ch_i, false);
        }
        _kind.push_back(split.kind);
        _factors.push_back(split.factors);
    }
}

NamedVector<Value> FKSSplittingKernels::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    std::size_t count = configuration_count();
    Value index = args.at("fks_index");
    std::vector<me_int_t> fks_i, fks_j, kind, born_j;
    for (std::size_t c = 0; c < count; ++c) {
        std::size_t i = _i_fks.at(c), j = _j_fks.at(c);
        fks_i.push_back(i);
        fks_j.push_back(j);
        kind.push_back(_kind.at(c));
        born_j.push_back(j < i ? j : j - 1);
    }
    auto [z, ap, q_terms] = fb.fks_splitting_kernels(
        args.at("momenta"), index, fks_i, fks_j, kind, _factors
    );
    auto k_perp = fb.fks_transverse_vector(
        args.at("p_i_hat"), args.at("born_momenta"), index, born_j
    );
    return {{"z", z}, {"ap", ap}, {"q_terms", q_terms}, {"k_perp", k_perp}};
}

FKSPrefactors::FKSPrefactors(
    const std::vector<std::size_t>& j_fks,
    const nested_vector2<double>& masses,
    double xicut,
    double xi_s_cut,
    double delta_o,
    double delta_i,
    double delta_s
) :
    FunctionGenerator(
        "FKSPrefactors",
        [&] {
            std::size_t count = j_fks.size();
            if (count == 0 || masses.size() != count) {
                throw std::invalid_argument(
                    "FKSPrefactors: one entry per configuration (at least one) in "
                    "every argument"
                );
            }
            std::size_t n = masses.at(0).size();
            for (std::size_t c = 0; c < count; ++c) {
                if (masses.at(c).size() != n) {
                    throw std::invalid_argument(
                        "FKSPrefactors: the real processes of the configurations "
                        "must have the same number of particles"
                    );
                }
                if (j_fks.at(c) >= n) {
                    throw std::invalid_argument(
                        "FKSPrefactors: j_fks must be a particle of the real "
                        "process"
                    );
                }
            }
            if (!(xicut > 0. && xi_s_cut > 0. && delta_o > 0. && delta_i > 0. &&
                  delta_s > 0.)) {
                throw std::invalid_argument(
                    "FKSPrefactors: the parameters of the subtraction must be "
                    "positive"
                );
            }
            NamedVector<Type> args{
                {"xi", batch_float},
                {"y", batch_float},
                {"xi_max", batch_float},
                {"collinear_xi", batch_float},
                {"collinear_xi_max", batch_float},
                {"collinear_momenta", batch_four_vec_array(n)},
                {"soft_collinear_momenta", batch_four_vec_array(n)},
            };
            args.push_back("fks_index", batch_int);
            return args;
        }(),
        {{"f_r", batch_float},
         {"f_s", batch_float},
         {"f_c", batch_float},
         {"f_dc", batch_float},
         {"f_sc", batch_float},
         {"f_dsc", batch_float_array(4)}}
    ),
    _particle_count(masses.at(0).size()),
    _j_fks(j_fks),
    _masses(masses),
    _xicut(xicut),
    _xi_s_cut(xi_s_cut),
    _delta_o(delta_o),
    _delta_i(delta_i),
    _delta_s(delta_s) {}

NamedVector<Value> FKSPrefactors::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    std::size_t count = configuration_count();
    Value index = args.at("fks_index");
    std::vector<me_int_t> collinear;
    std::vector<double> delta;
    for (std::size_t c = 0; c < count; ++c) {
        std::size_t j = _j_fks.at(c);
        collinear.push_back(_masses.at(c).at(j) == 0. ? 1 : 0);
        delta.push_back(j < INCOMING_COUNT ? _delta_i : _delta_o);
    }
    auto [f_r, f_s, f_c, f_dc, f_sc, f_dsc] = fb.fks_prefactors(
        args.at("xi"),
        args.at("y"),
        args.at("xi_max"),
        args.at("collinear_xi"),
        args.at("collinear_xi_max"),
        args.at("collinear_momenta"),
        args.at("soft_collinear_momenta"),
        index,
        collinear,
        delta,
        std::vector<double>{_xicut, _xi_s_cut, _delta_s}
    );
    return {
        {"f_r", f_r},
        {"f_s", f_s},
        {"f_c", f_c},
        {"f_dc", f_dc},
        {"f_sc", f_sc},
        {"f_dsc", f_dsc}
    };
}
