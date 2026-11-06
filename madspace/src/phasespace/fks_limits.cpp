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

std::size_t madspace::check_splitting_configurations(
    const std::vector<FKSConfiguration>& configurations,
    const FKSCouplingOrders& orders,
    const std::string& user
) {
    std::size_t n = check_fks_configurations(configurations, user);
    check_fks_coupling_orders(orders, user);
    for (const auto& config : configurations) {
        if (config.extra_counterterm >= 0) {
            auto extra = static_cast<std::size_t>(config.extra_counterterm);
            if (extra >= orders.extra_born.size()) {
                throw std::invalid_argument(
                    user + ": the coupling orders have no such extra counterterm"
                );
            }
            int pdg_i = config.pdgs.at(config.i_fks);
            if (pdg_i == 21 || pdg_i == 22) {
                throw std::invalid_argument(
                    user +
                    ": an extra counterterm needs a pair of a quark and its "
                    "antiquark (or the same quark incoming)"
                );
            }
        }
        for (std::size_t s = 0; s < config.splitting_types.size(); ++s) {
            const auto& type = config.splitting_types.at(s);
            const auto& names = orders.split_orders;
            if (std::find(names.begin(), names.end(), type) == names.end()) {
                throw std::invalid_argument(
                    user + ": the splitting type " + type + " is not a split order"
                );
            }
            auto colored = [&](std::size_t k) {
                int color = config.colors.at(k);
                return color == 8 || std::abs(color) == 3;
            };
            auto electroweak = [&](std::size_t k) {
                return config.pdgs.at(k) == 22 || config.charges.at(k) != 0.;
            };
            bool valid = type == "QCD"
                ? colored(config.i_fks) && colored(config.j_fks)
                : electroweak(config.i_fks) && electroweak(config.j_fks);
            if (!valid) {
                throw std::invalid_argument(
                    user + ": a " + type +
                    " splitting needs i and j colored (QCD) or photons or "
                    "charged (QED)"
                );
            }
            config.born_pdgs(s);
        }
    }
    return n;
}

FKSLimits::FKSLimits(
    const std::vector<FKSConfiguration>& configurations,
    const FKSCouplingOrders& orders,
    bool collinear_born
) :
    FunctionGenerator(
        "FKSLimits",
        [&] {
            int n = static_cast<int>(
                check_splitting_configurations(configurations, orders, "FKSLimits")
            );
            int b = static_cast<int>(orders.born_count());
            int inputs = static_cast<int>(orders.born_input_count());
            NamedVector<Type> types{
                {"soft_momenta", batch_four_vec_array(n)},
                {"soft_p_i_hat", batch_four_vec},
                {"y", batch_float},
                {"collinear_momenta", batch_four_vec_array(n)},
                {"soft_collinear_momenta", batch_four_vec_array(n)},
                {"born", batch_float_array(inputs)},
                {"color_correlated",
                 Type{DataType::dt_float64, batch_size, {b, n - 1, n - 1}}},
                {"spin_correlated", batch_float_array(inputs)},
            };
            if (collinear_born) {
                types.push_back("collinear_born", batch_float_array(inputs));
                types.push_back("collinear_spin_correlated", batch_float_array(inputs));
            }
            types.push_back("alpha_s", batch_float);
            types.push_back("alpha", batch_float);
            types.push_back("fks_index", batch_int);
            return types;
        }(),
        [&] {
            // checked here too: the order of the two arguments is unspecified
            check_splitting_configurations(configurations, orders, "FKSLimits");
            int o = static_cast<int>(orders.nlo_count());
            return NamedVector<Type>{
                {"soft", batch_float_array(o)},
                {"collinear", batch_float_array(o)},
                {"soft_collinear", batch_float_array(o)}
            };
        }()
    ),
    _eikonals(configurations),
    _kernels(configurations),
    _born_order_count(orders.born_count()),
    _born_input_count(orders.born_input_count()),
    _nlo_order_count(orders.nlo_count()),
    _collinear_born(collinear_born) {
    std::size_t largest = _eikonals.max_pair_count();
    for (std::size_t c = 0; c < configurations.size(); ++c) {
        const auto& config = configurations.at(c);
        _fks_i.push_back(config.i_fks);
        _fks_j.push_back(config.j_fks);
        double ratio = config.identical_particle_ratio();
        bool qed = config.qed_splitting();
        // only a gluon (QCD) or photon (QED) i has a soft limit
        auto pairs = _eikonals.born_pairs(c);
        auto real_pairs = _eikonals.pairs(c);
        bool soft = qed
            ? config.pdgs.at(config.i_fks) == 22
            : config.colors.at(config.i_fks) == 8;
        if (!soft) {
            pairs.clear();
        }
        // eta Q of a particle of the real process, the charge of an outgoing
        // particle
        auto charge = [&](std::size_t r) {
            return (r < 2 ? -1. : 1.) * config.charges.at(r);
        };
        std::vector<me_int_t> ms(largest, 0), ns(largest, 0);
        std::vector<double> factors(largest, 0.);
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            auto [m, n] = pairs.at(k);
            ms.at(k) = m;
            ns.at(k) = n;
            double factor = (m == n ? -1. : -2.) * ratio;
            if (qed) {
                auto [rm, rn] = real_pairs.at(k);
                factor *= charge(rm) * charge(rn);
            }
            factors.at(k) = factor;
        }
        _pair_count.push_back(pairs.size());
        _pair_m.push_back(ms);
        _pair_n.push_back(ns);
        _pair_factor.push_back(factors);
        // per splitting type (the second one for the extra counterterm) the
        // coupling, the factor of the collinear limits and at each NLO order
        // the Born of the counterterms among the Born inputs
        bool massless = config.masses.at(config.j_fks) == 0.;
        std::vector<me_int_t> couplings(2, -1);
        std::vector<double> collinear_factors(2, 0.);
        nested_vector2<me_int_t> sources(
            2, std::vector<me_int_t>(orders.nlo_count(), -1)
        );
        for (std::size_t s = 0; s < config.splitting_types.size(); ++s) {
            const auto& type = config.splitting_types.at(s);
            couplings.at(s) = type == "QED" ? 1 : 0;
            if (massless) {
                collinear_factors.at(s) = config.identical_particle_ratio(s);
            }
            if (s == 0) {
                for (std::size_t b = 0; b < orders.born_count(); ++b) {
                    int o = orders.counterterm_order(b, type);
                    if (o >= 0) {
                        sources.at(0).at(o) = static_cast<me_int_t>(b);
                    }
                }
                continue;
            }
            auto extra = static_cast<std::size_t>(config.extra_counterterm);
            std::size_t offset = orders.extra_born_offset(extra);
            for (std::size_t e = 0; e < orders.extra_born.at(extra).size(); ++e) {
                int o = orders.extra_counterterm_order(extra, e, type);
                if (o >= 0) {
                    sources.at(1).at(o) = static_cast<me_int_t>(offset + e);
                }
            }
        }
        _collinear_factor.push_back(collinear_factors);
        _coupling.push_back(couplings);
        _born_index.push_back(sources);
    }
}

Value FKSLimits::born_index_table() const {
    // first index fastest, as the constants of nested vectors
    std::size_t count = _born_index.size(), orders = _nlo_order_count;
    std::vector<me_int_t> flat(count * 2 * orders);
    for (std::size_t c = 0; c < count; ++c) {
        for (std::size_t s = 0; s < 2; ++s) {
            for (std::size_t o = 0; o < orders; ++o) {
                flat.at(c + count * (s + 2 * o)) = _born_index.at(c).at(s).at(o);
            }
        }
    }
    return Value(flat, {static_cast<int>(count), 2, static_cast<int>(orders)});
}

NamedVector<Value> FKSLimits::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value index = args.at("fks_index");
    Value alpha_s = args.at("alpha_s"), alpha = args.at("alpha");
    auto eikonals =
        _eikonals
            .build_function(
                fb,
                {args.at("soft_momenta"), args.at("soft_p_i_hat"), args.at("y"), index}
            )
            .at(0);
    auto soft = fb.fks_soft_limit(
        eikonals,
        args.at("born"),
        args.at("color_correlated"),
        alpha_s,
        alpha,
        index,
        _pair_count,
        _pair_m,
        _pair_n,
        _pair_factor,
        _coupling,
        born_index_table()
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
        // the collinear counter-event can have a Born of its own
        bool own = _collinear_born && name == "collinear";
        result.push_back(
            name,
            fb.fks_collinear_limit(
                args.at(own ? "collinear_born" : "born"),
                args.at(own ? "collinear_spin_correlated" : "spin_correlated"),
                ap,
                q,
                alpha_s,
                alpha,
                index,
                _collinear_factor,
                _coupling,
                born_index_table()
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
         {"f_dsc", batch_float_array(4)},
         {"f_pdf_scheme", batch_float_array(3)}}
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
    auto [f_r, f_s, f_c, f_dc, f_sc, f_dsc, f_pdf] = fb.fks_prefactors(
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
        {"f_dsc", f_dsc},
        {"f_pdf_scheme", f_pdf}
    };
}
