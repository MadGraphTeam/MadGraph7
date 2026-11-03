#include "madspace/phasespace/fks_limits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

using namespace madspace;

namespace {

constexpr std::size_t INCOMING_COUNT = 2;
constexpr double CA = 3., TR = 0.5, CF = 4. / 3.;

// the pairs of sbornsoft: every unordered pair of the partners of i, a
// particle with itself only where it is massive
std::vector<std::pair<std::size_t, std::size_t>>
eikonal_pairs(const FKSConfiguration& config) {
    const auto& partners = config.fks_j_from_i.at(config.i_fks);
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    for (std::size_t k = 0; k < partners.size(); ++k) {
        for (std::size_t l = 0; l <= k; ++l) {
            std::size_t m = partners.at(k), n = partners.at(l);
            if (m == config.i_fks || n == config.i_fks) {
                throw std::invalid_argument(
                    "FKSEikonals: i_fks cannot be its own FKS partner"
                );
            }
            if (m != n || config.masses.at(m) != 0.) {
                pairs.push_back({m, n});
            }
        }
    }
    return pairs;
}

} // namespace

FKSEikonals::FKSEikonals(const std::vector<FKSConfiguration>& configurations) :
    FunctionGenerator(
        "FKSEikonals",
        [&] {
            std::size_t n = check_fks_configurations(configurations, "FKSEikonals");
            return NamedVector<Type>{
                {"momenta", batch_four_vec_array(n)},
                {"p_i_hat", batch_four_vec},
                {"y", batch_float},
                {"fks_index", batch_int},
            };
        }(),
        [&] {
            // checked here too: the order of the two arguments is unspecified
            check_fks_configurations(configurations, "FKSEikonals");
            std::size_t largest = 1;
            for (const auto& config : configurations) {
                largest = std::max(largest, eikonal_pairs(config).size());
            }
            return NamedVector<Type>{
                {"eikonals", batch_float_array(static_cast<int>(largest))}
            };
        }()
    ),
    _particle_count(configurations.at(0).particle_count()),
    _max_pair_count(1) {
    for (const auto& config : configurations) {
        _i_fks.push_back(config.i_fks);
        _j_fks.push_back(config.j_fks);
        _masses.push_back(config.masses);
        _pairs.push_back(eikonal_pairs(config));
        _max_pair_count = std::max(_max_pair_count, _pairs.back().size());
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
    const std::vector<FKSConfiguration>& configurations
) :
    FunctionGenerator(
        "FKSSplittingKernels",
        [&] {
            std::size_t n =
                check_fks_configurations(configurations, "FKSSplittingKernels");
            return NamedVector<Type>{
                {"momenta", batch_four_vec_array(n)},
                {"p_i_hat", batch_four_vec},
                {"born_momenta", batch_four_vec_array(n - 1)},
                {"fks_index", batch_int},
            };
        }(),
        {{"z", batch_float},
         {"ap", batch_float_array(2)},
         {"q_terms", batch_float_array(2)},
         {"k_perp", batch_four_vec}}
    ),
    _particle_count(configurations.at(0).particle_count()) {
    for (const auto& config : configurations) {
        std::size_t i = config.i_fks, j = config.j_fks;
        _i_fks.push_back(i);
        _j_fks.push_back(j);
        int i_type = config.colors.at(i), j_type = config.colors.at(j);
        double ch_i = config.charges.at(i), ch_j = config.charges.at(j);
        Splitting split;
        if (!config.initial_state()) {
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

namespace {

// the configurations, checked to have QCD splittings
std::size_t check_qcd_configurations(
    const std::vector<FKSConfiguration>& configurations, const std::string& user
) {
    std::size_t n = check_fks_configurations(configurations, user);
    for (const auto& config : configurations) {
        int i_type = config.colors.at(config.i_fks);
        int j_type = config.colors.at(config.j_fks);
        if ((i_type != 8 && std::abs(i_type) != 3) ||
            (j_type != 8 && std::abs(j_type) != 3)) {
            throw std::invalid_argument(
                user + ": only QCD splittings, of quarks and gluons, are supported"
            );
        }
        config.born_pdgs();
    }
    return n;
}

} // namespace

FKSLimits::FKSLimits(const std::vector<FKSConfiguration>& configurations) :
    FunctionGenerator(
        "FKSLimits",
        [&] {
            int n =
                static_cast<int>(check_qcd_configurations(configurations, "FKSLimits"));
            return NamedVector<Type>{
                {"soft_momenta", batch_four_vec_array(n)},
                {"soft_p_i_hat", batch_four_vec},
                {"y", batch_float},
                {"collinear_momenta", batch_four_vec_array(n)},
                {"soft_collinear_momenta", batch_four_vec_array(n)},
                {"born", batch_float},
                {"color_correlated",
                 Type{DataType::dt_float64, batch_size, {n - 1, n - 1}}},
                {"spin_correlated", batch_float},
                {"alpha_s", batch_float},
                {"fks_index", batch_int},
            };
        }(),
        [&] {
            // checked here too: the order of the two arguments is unspecified
            check_qcd_configurations(configurations, "FKSLimits");
            return NamedVector<Type>{
                {"soft", batch_float},
                {"collinear", batch_float},
                {"soft_collinear", batch_float}
            };
        }()
    ),
    _eikonals(configurations),
    _kernels(configurations) {
    std::size_t largest = _eikonals.max_pair_count();
    for (std::size_t c = 0; c < configurations.size(); ++c) {
        const auto& config = configurations.at(c);
        _fks_i.push_back(config.i_fks);
        _fks_j.push_back(config.j_fks);
        double ratio = config.identical_particle_ratio();
        // only a gluon i has a soft limit
        auto pairs = _eikonals.born_pairs(c);
        if (config.colors.at(config.i_fks) != 8) {
            pairs.clear();
        }
        std::vector<me_int_t> ms(largest, 0), ns(largest, 0);
        std::vector<double> factors(largest, 0.);
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            auto [m, n] = pairs.at(k);
            ms.at(k) = m;
            ns.at(k) = n;
            factors.at(k) = (m == n ? -1. : -2.) * ratio;
        }
        _pair_count.push_back(pairs.size());
        _pair_m.push_back(ms);
        _pair_n.push_back(ns);
        _pair_factor.push_back(factors);
        _collinear_factor.push_back(config.masses.at(config.j_fks) == 0. ? ratio : 0.);
    }
}

NamedVector<Value> FKSLimits::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value index = args.at("fks_index");
    Value alpha_s = args.at("alpha_s");
    auto eikonals =
        _eikonals
            .build_function(
                fb,
                {args.at("soft_momenta"), args.at("soft_p_i_hat"), args.at("y"), index}
            )
            .at(0);
    auto soft = fb.fks_soft_limit(
        eikonals,
        args.at("color_correlated"),
        alpha_s,
        index,
        _pair_count,
        _pair_m,
        _pair_n,
        _pair_factor
    );
    std::vector<me_int_t> kind;
    nested_vector2<double> factors;
    for (std::size_t c = 0; c < configuration_count(); ++c) {
        kind.push_back(_kernels.splitting_kind(c));
        factors.push_back(_kernels.factors(c));
    }
    NamedVector<Value> result{{"soft", soft}};
    for (std::string name : {"collinear", "soft_collinear"}) {
        auto [z, ap, q] = fb.fks_splitting_kernels(
            args.at(name + "_momenta"), index, _fks_i, _fks_j, kind, factors
        );
        result.push_back(
            name,
            fb.fks_collinear_limit(
                args.at("born"),
                args.at("spin_correlated"),
                ap,
                q,
                alpha_s,
                index,
                _collinear_factor
            )
        );
    }
    return result;
}

FKSPrefactors::FKSPrefactors(
    const std::vector<FKSConfiguration>& configurations,
    double xicut,
    double xi_s_cut,
    double delta_o,
    double delta_i,
    double delta_s
) :
    FunctionGenerator(
        "FKSPrefactors",
        [&] {
            std::size_t n = check_fks_configurations(configurations, "FKSPrefactors");
            if (!(xicut > 0. && xi_s_cut > 0. && delta_o > 0. && delta_i > 0. &&
                  delta_s > 0.)) {
                throw std::invalid_argument(
                    "FKSPrefactors: the parameters of the subtraction must be "
                    "positive"
                );
            }
            return NamedVector<Type>{
                {"xi", batch_float},
                {"y", batch_float},
                {"xi_max", batch_float},
                {"collinear_xi", batch_float},
                {"collinear_xi_max", batch_float},
                {"collinear_momenta", batch_four_vec_array(n)},
                {"soft_collinear_momenta", batch_four_vec_array(n)},
                {"fks_index", batch_int},
            };
        }(),
        {{"f_r", batch_float},
         {"f_s", batch_float},
         {"f_c", batch_float},
         {"f_dc", batch_float},
         {"f_sc", batch_float},
         {"f_dsc", batch_float_array(4)}}
    ),
    _particle_count(configurations.at(0).particle_count()),
    _xicut(xicut),
    _xi_s_cut(xi_s_cut),
    _delta_o(delta_o),
    _delta_i(delta_i),
    _delta_s(delta_s) {
    for (const auto& config : configurations) {
        _j_fks.push_back(config.j_fks);
        _masses.push_back(config.masses);
    }
}

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
