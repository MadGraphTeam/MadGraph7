#include "madspace/phasespace/fks_subtraction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <stdexcept>

#include "madspace/constants.hpp"

using namespace madspace;

namespace {

constexpr double CA = 3., CF = 4. / 3., TR = 0.5;
constexpr std::size_t QCD = 0, QED = 1;

// the first configuration with a gluon (QCD) or photon (QED) i, whose soft
// pairs the integrated eikonals take (as bornsoftvirtual those of the first
// with need_color_links or need_charge_links), the size for none
std::size_t
linked_configuration(const std::vector<FKSConfiguration>& configurations, bool qed) {
    for (std::size_t c = 0; c < configurations.size(); ++c) {
        const auto& config = configurations.at(c);
        if (qed ? config.pdgs.at(config.i_fks) == 22
                : config.colors.at(config.i_fks) == 8) {
            return c;
        }
    }
    return configurations.size();
}

// the configuration whose real process without i is the Born, the first with
// a gluon or photon i
std::size_t born_configuration(const std::vector<FKSConfiguration>& configurations) {
    std::size_t c = std::min(
        linked_configuration(configurations, false),
        linked_configuration(configurations, true)
    );
    if (c == configurations.size()) {
        throw std::invalid_argument(
            "FKSSoftVirtual: needs a configuration with a gluon or photon i, for "
            "the color or charge links"
        );
    }
    return c;
}

// The factor of the kernels of the PDF scheme (fks_scheme_shapes) of a
// splitting of a kind (fks_ap_shape) and a coupling with the factor ap_factor
// of its Altarelli-Parisi kernel, as xkplus, xklog and xkdelta select them,
// zero where the scheme does not change it
double scheme_factor(
    FKSSubtraction::PdfScheme scheme,
    int kind,
    std::size_t coupling,
    double ap_factor,
    double n_f
) {
    switch (scheme) {
    case FKSSubtraction::dis:
        if (kind == 0) {
            return coupling == QCD ? -2. * n_f * TR : 0.;
        }
        return kind == 3 ? -ap_factor : ap_factor;
    case FKSSubtraction::eta:
    case FKSSubtraction::beta:
    case FKSSubtraction::mixed:
    case FKSSubtraction::nobeta:
        return kind == 2 ? ap_factor : 0.;
    case FKSSubtraction::delta:
        return kind == 2 || kind == 3 ? ap_factor : 0.;
    case FKSSubtraction::upc:
        return kind == 1 && coupling == QED ? ap_factor : 0.;
    default:
        return 0.;
    }
}

} // namespace

FKSSubtraction::FKSSubtraction(
    const std::vector<FKSConfiguration>& configurations,
    const FKSCouplingOrders& orders,
    double xicut,
    double xi_s_cut,
    double delta_o,
    double delta_i,
    double delta_s,
    double fks_a,
    double fks_b,
    bool use_energy,
    bool use_beta,
    double h_damp_one,
    double h_damp_power,
    double tiny,
    PdfScheme pdf_scheme,
    double n_f,
    const std::vector<double>& upc_radii,
    bool collinear_born,
    bool resonance
) :
    FunctionGenerator(
        "FKSSubtraction",
        [&] {
            // the limits check the configurations
            FKSLimits limits(configurations, orders);
            auto n = static_cast<int>(limits.particle_count());
            auto b = static_cast<int>(orders.born_count());
            auto inputs = static_cast<int>(orders.born_input_count());
            auto o = static_cast<int>(orders.nlo_count());
            NamedVector<Type> types{
                {"momenta", batch_four_vec_array(n)},
                {"weight", batch_float},
                {"xi", batch_float},
                {"xi_hat", batch_float},
                {"y", batch_float},
                {"xi_max", batch_float},
                {"p_i_hat", batch_four_vec},
                {"soft_momenta", batch_four_vec_array(n)},
                {"soft_det", batch_float},
                {"soft_p_i_hat", batch_four_vec},
                {"soft_xi_max", batch_float},
                {"collinear_momenta", batch_four_vec_array(n)},
                {"collinear_det", batch_float},
                {"collinear_p_i_hat", batch_four_vec},
                {"collinear_xi", batch_float},
                {"collinear_xi_max", batch_float},
                {"soft_collinear_momenta", batch_four_vec_array(n)},
                {"soft_collinear_det", batch_float},
                {"soft_collinear_p_i_hat", batch_four_vec},
                {"matrix_element", batch_float_array(o)},
                {"born", batch_float_array(inputs)},
                {"color_correlated",
                 Type{DataType::dt_float64, batch_size, {b, n - 1, n - 1}}},
                {"spin_correlated", batch_float_array(inputs)},
            };
            if (collinear_born) {
                types.push_back("collinear_born", batch_float_array(inputs));
                types.push_back("collinear_spin_correlated", batch_float_array(inputs));
            }
            for (const auto& name :
                 {"alpha_s", "alpha", "ren_scale", "fact_scale1", "fact_scale2"}) {
                types.push_back(name, batch_float);
            }
            if (resonance) {
                types.push_back("resonance_momentum", batch_four_vec);
            }
            types.push_back("fks_index", batch_int);
            return types;
        }(),
        [&] {
            auto o = static_cast<int>(orders.nlo_count());
            return NamedVector<Type>{
                {"real", batch_float_array(o)},
                {"soft", batch_float_array(o)},
                {"collinear", batch_float_array(o)},
                {"soft_collinear", batch_float_array(o)}
            };
        }()
    ),
    _sector_function(
        configurations, fks_a, fks_b, use_energy, use_beta, h_damp_one, h_damp_power
    ),
    _limits(configurations, orders, collinear_born, resonance),
    _prefactors(configurations, xicut, xi_s_cut, delta_o, delta_i, delta_s, resonance),
    _xi_s_cut(xi_s_cut),
    _delta_s(delta_s),
    _tiny(tiny),
    _pdf_scheme(pdf_scheme),
    _upc_radii(2, 1.) {
    if (!(tiny >= 0.)) {
        throw std::invalid_argument("FKSSubtraction: tiny cannot be negative");
    }
    if (!(n_f >= 0.)) {
        throw std::invalid_argument("FKSSubtraction: n_f cannot be negative");
    }
    if (pdf_scheme < msbar || pdf_scheme > upc) {
        throw std::invalid_argument("FKSSubtraction: unknown PDF scheme");
    }
    if (pdf_scheme == upc) {
        if (upc_radii.size() != 2 ||
            !std::all_of(upc_radii.begin(), upc_radii.end(), [](double r) {
                return r > 0. && std::isfinite(r);
            })) {
            throw std::invalid_argument(
                "FKSSubtraction: the UPC scheme needs a positive xi_A R_A for "
                "each beam"
            );
        }
        _upc_radii = upc_radii;
    }
    const auto& kernels = _limits.splitting_kernels();
    for (std::size_t c = 0; c < configurations.size(); ++c) {
        const auto& config = configurations.at(c);
        bool massless = config.masses.at(config.j_fks) == 0.;
        int kind = kernels.splitting_kind(c);
        // per splitting type (the second for the extra counterterm) the
        // coupling, the Born inputs, the identical-particle ratio, the color
        // or charge factor of AP_reduced and that of the kernels of the PDF
        // scheme, xkplus, xklog and xkdelta
        std::vector<me_int_t> couplings(2, -1);
        std::vector<double> ratios(2, 0.), ap_factors(2, 0.), scheme_factors(2, 0.);
        for (std::size_t s = 0; s < 2; ++s) {
            int coupling = _limits.coupling(c, s);
            if (coupling < 0) {
                continue;
            }
            std::size_t t = coupling == 1 ? QED : QCD;
            double ap_factor = kernels.factors(c).at(t);
            couplings.at(s) = static_cast<me_int_t>(t);
            ratios.at(s) = config.identical_particle_ratio(s);
            ap_factors.at(s) = ap_factor;
            scheme_factors.at(s) = scheme_factor(pdf_scheme, kind, t, ap_factor, n_f);
        }
        _symmetry.push_back(config.symmetry_factor());
        _massless.push_back(massless ? 1 : 0);
        _remnant.push_back(massless && config.initial_state() ? 1 : 0);
        _beam.push_back(config.initial_state() ? config.j_fks : 0);
        _kind.push_back(kind);
        _coupling.push_back(couplings);
        _ratio.push_back(ratios);
        _ap_factor.push_back(ap_factors);
        _scheme_factor.push_back(scheme_factors);
        _delta_i.push_back(delta_i);
    }
}

NamedVector<Value> FKSSubtraction::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value index = args.at("fks_index");
    // the sector function at the momenta of one kinematics with the direction
    // of i of another (the same by default)
    auto sector = [&](const std::string& prefix, std::string direction = "") {
        direction = direction.empty() ? prefix : direction;
        return _sector_function
            .build_function(
                fb, {args.at(prefix + "momenta"), args.at(direction + "p_i_hat"), index}
            )
            .at(0);
    };
    bool collinear_born = _limits.collinear_born();
    bool resonance = _limits.resonance();
    auto limit_arguments =
        [&](const std::string& soft,
            const std::string& collinear,
            const std::string& soft_collinear) {
            std::vector<Value> arguments{
                args.at(soft + "momenta"),
                args.at("soft_p_i_hat"),
                args.at("y"),
                args.at(collinear + "momenta"),
                args.at(soft_collinear + "momenta"),
                args.at("born"),
                args.at("color_correlated"),
                args.at("spin_correlated"),
            };
            if (collinear_born) {
                arguments.push_back(args.at("collinear_born"));
                arguments.push_back(args.at("collinear_spin_correlated"));
            }
            arguments.push_back(args.at("alpha_s"));
            arguments.push_back(args.at("alpha"));
            if (resonance) {
                arguments.push_back(args.at("resonance_momentum"));
            }
            arguments.push_back(index);
            return arguments;
        };
    // the limits at the event, where the real emission switches to them, with
    // the direction of i of the soft counter-event as eikonal_reduced
    auto event_limits = _limits.build_function(fb, limit_arguments("", "", ""));
    auto real = fb.fks_real_emission(
        args.at("matrix_element"),
        sector(""),
        sector("", "soft_"),
        sector("", "soft_collinear_"),
        event_limits.at("soft"),
        event_limits.at("collinear"),
        args.at("xi"),
        args.at("y"),
        args.at("soft_det"),
        args.at("soft_collinear_det"),
        index,
        _massless,
        std::vector<double>{_tiny}
    );
    auto limits = _limits.build_function(
        fb, limit_arguments("soft_", "collinear_", "soft_collinear_")
    );
    ValueVec prefactor_args{
        args.at("xi"),
        args.at("y"),
        args.at("xi_max"),
        args.at("collinear_xi"),
        args.at("collinear_xi_max"),
        args.at("collinear_momenta"),
        args.at("soft_collinear_momenta"),
    };
    if (resonance) {
        prefactor_args.push_back(args.at("soft_momenta"));
        prefactor_args.push_back(args.at("soft_p_i_hat"));
        prefactor_args.push_back(args.at("resonance_momentum"));
    }
    prefactor_args.push_back(index);
    auto prefactors = _prefactors.build_function(fb, prefactor_args);
    auto s_c = sector("collinear_"), s_sc = sector("soft_collinear_");
    auto [remnant_c, remnant_sc] = fb.fks_collinear_remnants(
        args.at("born"),
        args.at(collinear_born ? "collinear_born" : "born"),
        args.at("alpha_s"),
        args.at("alpha"),
        s_c,
        s_sc,
        args.at("collinear_momenta"),
        args.at("soft_collinear_momenta"),
        args.at("collinear_xi"),
        prefactors.at("f_dc"),
        prefactors.at("f_dsc"),
        prefactors.at("f_pdf_scheme"),
        args.at("ren_scale"),
        args.at("fact_scale1"),
        args.at("fact_scale2"),
        index,
        _remnant,
        _beam,
        _kind,
        _coupling,
        _limits.born_index_table(),
        _ratio,
        _ap_factor,
        _scheme_factor,
        _delta_i,
        std::vector<double>{
            static_cast<double>(_pdf_scheme),
            _upc_radii.at(0) * _upc_radii.at(0),
            _upc_radii.at(1) * _upc_radii.at(1)
        }
    );
    auto [real_out, soft, collinear, soft_collinear] = fb.fks_subtraction(
        real,
        limits.at("soft"),
        limits.at("collinear"),
        limits.at("soft_collinear"),
        sector("soft_"),
        s_c,
        s_sc,
        prefactors.at("f_s"),
        prefactors.at("f_c"),
        prefactors.at("f_sc"),
        remnant_c,
        remnant_sc,
        args.at("xi"),
        args.at("xi_hat"),
        args.at("y"),
        args.at("soft_xi_max"),
        args.at("collinear_xi"),
        args.at("collinear_xi_max"),
        args.at("weight"),
        args.at("soft_det"),
        args.at("collinear_det"),
        args.at("soft_collinear_det"),
        index,
        _symmetry,
        std::vector<double>{_xi_s_cut, _delta_s}
    );
    return {
        {"real", real_out},
        {"soft", soft},
        {"collinear", collinear},
        {"soft_collinear", soft_collinear}
    };
}

FKSSoftVirtual::FKSSoftVirtual(
    const std::vector<FKSConfiguration>& configurations,
    const FKSCouplingOrders& orders,
    double xicut,
    double delta_o,
    double n_f,
    double n_l
) :
    FunctionGenerator(
        "FKSSoftVirtual",
        [&] {
            // the limits check the configurations
            FKSLimits limits(configurations, orders);
            auto n = static_cast<int>(limits.particle_count()) - 1;
            auto b = static_cast<int>(orders.born_count());
            born_configuration(configurations);
            if (!(xicut > 0. && xicut <= 1. && delta_o > 0. && delta_o <= 2. &&
                  n_f >= 0. && n_l >= 0.)) {
                throw std::invalid_argument(
                    "FKSSoftVirtual: xicut must be in (0, 1], delta_o in (0, 2] "
                    "and n_f and n_l not negative"
                );
            }
            return NamedVector<Type>{
                {"momenta", batch_four_vec_array(n)},
                {"born", batch_float_array(b)},
                {"color_correlated", Type{DataType::dt_float64, batch_size, {b, n, n}}},
                {"alpha_s", batch_float},
                {"alpha", batch_float},
                {"ren_scale", batch_float},
                {"fact_scale1", batch_float},
                {"fact_scale2", batch_float},
            };
        }(),
        [&] {
            auto o = static_cast<int>(orders.nlo_count());
            return NamedVector<Type>{
                {"soft_virtual", batch_float_array(o)},
                {"ren_coefficient", batch_float_array(o)},
                {"fact_coefficients", Type{DataType::dt_float64, batch_size, {o, 2}}}
            };
        }()
    ),
    _born_order_count(orders.born_count()),
    _xicut(xicut),
    _delta_o(delta_o) {
    const auto& config = configurations.at(born_configuration(configurations));
    // the factors of the gluon and the quarks (QCD), of the photon (QED)
    double gamma_g = (11. * CA - 2. * n_f) / 6.;
    double gamma_prime_g = (67. / 9. - 2. * PI * PI / 3.) * CA - 23. * n_f / 18.;
    double gamma_q = 1.5 * CF;
    double gamma_prime_q = (6.5 - 2. * PI * PI / 3.) * CF;
    double charges = 0.;
    for (long f = 1; f <= std::lround(n_f); ++f) {
        // down-type for odd f, up-type for even f
        charges += f % 2 == 0 ? 4. / 9. : 1. / 9.;
    }
    double gamma_photon = -(3. * charges + n_l) * 2. / 3.;
    double gamma_prime_photon = -(3. * charges + n_l) * 23. / 9.;
    _casimir.assign(2, {});
    _gamma.assign(2, {});
    _gamma_prime.assign(2, {});
    for (std::size_t r = 0; r < config.particle_count(); ++r) {
        if (r == config.i_fks) {
            continue;
        }
        // the Born of a gluon or photon i: the real process without it
        int color = config.colors.at(r);
        double mass = config.masses.at(r), charge = config.charges.at(r);
        bool gluon = mass == 0. && color == 8;
        bool quark = mass == 0. && std::abs(color) == 3;
        // a tagged photon (an identified one, !a!) has no Q terms
        bool photon = mass == 0. && config.pdgs.at(r) == 22 && !config.tagged(r);
        bool charged = mass == 0. && charge != 0.;
        double q2 = charge * charge;
        _masses.push_back(mass);
        _casimir.at(QCD).push_back(gluon ? CA : (quark ? CF : 0.));
        _gamma.at(QCD).push_back(gluon ? gamma_g : (quark ? gamma_q : 0.));
        _gamma_prime.at(QCD).push_back(
            gluon ? gamma_prime_g : (quark ? gamma_prime_q : 0.)
        );
        _casimir.at(QED).push_back(charged ? q2 : 0.);
        _gamma.at(QED).push_back(photon ? gamma_photon : (charged ? 1.5 * q2 : 0.));
        _gamma_prime.at(QED).push_back(
            photon ? gamma_prime_photon
                   : (charged ? (6.5 - 2. * PI * PI / 3.) * q2 : 0.)
        );
    }
    // the linked pairs of QCD and QED, padded to the same length
    FKSEikonals eikonals(configurations);
    _pair_m.assign(2, {});
    _pair_n.assign(2, {});
    _pair_factor.assign(2, {});
    for (std::size_t t : {QCD, QED}) {
        std::size_t c = linked_configuration(configurations, t == QED);
        if (c == configurations.size()) {
            continue;
        }
        const auto& linked = configurations.at(c);
        auto real_pairs = eikonals.pairs(c);
        auto born_pairs = eikonals.born_pairs(c);
        for (std::size_t k = 0; k < born_pairs.size(); ++k) {
            auto [m, n] = born_pairs.at(k);
            double factor = m == n ? -1. : -2.;
            if (t == QED) {
                auto [rm, rn] = real_pairs.at(k);
                auto charge = [&](std::size_t r) {
                    return (r < 2 ? -1. : 1.) * linked.charges.at(r);
                };
                factor *= charge(rm) * charge(rn);
            }
            _pair_m.at(t).push_back(m);
            _pair_n.at(t).push_back(n);
            _pair_factor.at(t).push_back(factor);
        }
    }
    std::size_t largest = std::max<std::size_t>(
        {1, _pair_factor.at(QCD).size(), _pair_factor.at(QED).size()}
    );
    for (std::size_t t : {QCD, QED}) {
        _pair_m.at(t).resize(largest, 0);
        _pair_n.at(t).resize(largest, 0);
        _pair_factor.at(t).resize(largest, 0.);
    }
    // the Born orders of the QCD and QED counterterms at each NLO order, for
    // the splitting types of the configurations (split_type_used, which has
    // those of the extra counterterms too)
    bool used[2] = {false, false};
    for (const auto& c : configurations) {
        for (const auto& type : c.splitting_types) {
            used[type == "QED" ? QED : QCD] = true;
        }
    }
    _born_index.assign(2, std::vector<me_int_t>(orders.nlo_count(), -1));
    _ren_factor.assign(orders.nlo_count(), 0.);
    auto qcd_position = std::find(
        orders.split_orders.begin(), orders.split_orders.end(), std::string("QCD")
    );
    for (std::size_t t : {QCD, QED}) {
        if (!used[t]) {
            continue;
        }
        for (std::size_t b = 0; b < orders.born_count(); ++b) {
            int o = orders.counterterm_order(b, t == QCD ? "QCD" : "QED");
            if (o < 0) {
                continue;
            }
            _born_index.at(t).at(o) = static_cast<me_int_t>(b);
            if (t == QCD) {
                // the power of alpha_s of the Born, half that of g_s
                int power =
                    orders.born.at(b).at(qcd_position - orders.split_orders.begin());
                _ren_factor.at(o) = 0.5 * power * gamma_g;
            }
        }
    }
}

NamedVector<Value> FKSSoftVirtual::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto [soft_virtual, ren, fact] = fb.fks_soft_virtual(
        args.at("momenta"),
        args.at("born"),
        args.at("color_correlated"),
        args.at("alpha_s"),
        args.at("alpha"),
        args.at("ren_scale"),
        args.at("fact_scale1"),
        args.at("fact_scale2"),
        _casimir,
        _gamma,
        _gamma_prime,
        _masses,
        _pair_m,
        _pair_n,
        _pair_factor,
        _born_index,
        _ren_factor,
        std::vector<double>{_xicut, _delta_o}
    );
    return {
        {"soft_virtual", soft_virtual},
        {"ren_coefficient", ren},
        {"fact_coefficients", fact}
    };
}

namespace {

// the powers of g_s of the orders, zero without a QCD split order
std::vector<me_int_t>
qcd_powers(const FKSCouplingOrders& orders, const nested_vector2<int>& order_powers) {
    auto split = std::find(
        orders.split_orders.begin(), orders.split_orders.end(), std::string("QCD")
    );
    std::vector<me_int_t> powers;
    for (const auto& order : order_powers) {
        powers.push_back(
            split == orders.split_orders.end()
                ? 0
                : order.at(split - orders.split_orders.begin())
        );
    }
    return powers;
}

} // namespace

FKSWeights::FKSWeights(
    const std::vector<FKSConfiguration>& configurations, const FKSCouplingOrders& orders
) :
    FunctionGenerator(
        "FKSWeights",
        [&] {
            auto n = static_cast<int>(
                check_fks_configurations(configurations, "FKSWeights")
            );
            auto b = static_cast<int>(orders.born_count());
            auto o = static_cast<int>(orders.nlo_count());
            return NamedVector<Type>{
                {"momenta", batch_four_vec_array(n)},
                {"weight", batch_float},
                {"soft_momenta", batch_four_vec_array(n)},
                {"soft_det", batch_float},
                {"collinear_momenta", batch_four_vec_array(n)},
                {"collinear_det", batch_float},
                {"soft_collinear_momenta", batch_four_vec_array(n)},
                {"soft_collinear_det", batch_float},
                {"born_momenta", batch_four_vec_array(n - 1)},
                {"born_weight", batch_float},
                {"real", batch_float_array(o)},
                {"soft", batch_float_array(o)},
                {"collinear", batch_float_array(o)},
                {"soft_collinear", batch_float_array(o)},
                {"born", batch_float_array(b)},
                {"soft_virtual", batch_float_array(o)},
                {"virtual", batch_float_array(o)},
                {"event_alpha_s", batch_float},
                {"born_alpha_s", batch_float},
                {"collinear_alpha_s", batch_float},
                {"event_luminosity", batch_float},
                {"soft_luminosity", batch_float},
                {"collinear_luminosity", batch_float},
                {"soft_collinear_luminosity", batch_float},
                {"born_luminosity", batch_float},
                {"channel_weight", batch_float},
                {"configuration_weight", batch_float},
            };
        }(),
        NamedVector<Type>{
            {"cross_section", batch_float},
            {"event_weight", batch_float},
            {"soft_weight", batch_float},
            {"collinear_weight", batch_float},
            {"soft_collinear_weight", batch_float},
            {"nbody_weight", batch_float},
            {"lo_weight", batch_float},
        }
    ),
    _particle_count(check_fks_configurations(configurations, "FKSWeights")),
    _born_power(qcd_powers(orders, orders.born)),
    _nlo_power(qcd_powers(orders, orders.nlo)) {}

NamedVector<Value> FKSWeights::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value channel = args.at("channel_weight");
    // the real emission and its counterterms come with the configuration
    Value emission = fb.mul(channel, args.at("configuration_weight"));
    auto weight =
        [&](Value coefficients,
            const std::string& kinematics,
            const std::string& det,
            const std::string& coupling,
            const std::string& luminosity,
            Value factor,
            const std::vector<me_int_t>& powers) {
            return fb.fks_contribution_weight(
                coefficients,
                args.at(det),
                args.at(kinematics + "momenta"),
                args.at(coupling + "_alpha_s"),
                args.at(luminosity + "luminosity"),
                factor,
                powers
            );
        };
    Value event =
        weight(args.at("real"), "", "weight", "event", "event_", emission, _nlo_power);
    Value soft = weight(
        args.at("soft"), "soft_", "soft_det", "born", "soft_", emission, _nlo_power
    );
    Value collinear = weight(
        args.at("collinear"),
        "collinear_",
        "collinear_det",
        "collinear",
        "collinear_",
        emission,
        _nlo_power
    );
    Value soft_collinear = weight(
        args.at("soft_collinear"),
        "soft_collinear_",
        "soft_collinear_det",
        "born",
        "soft_collinear_",
        emission,
        _nlo_power
    );
    Value lo = weight(
        args.at("born"), "born_", "born_weight", "born", "born_", channel, _born_power
    );
    Value nbody_nlo = weight(
        fb.add(args.at("soft_virtual"), args.at("virtual")),
        "born_",
        "born_weight",
        "born",
        "born_",
        channel,
        _nlo_power
    );
    Value nbody = fb.add(lo, nbody_nlo);
    Value total =
        fb.add(fb.add(fb.add(event, soft), fb.add(collinear, soft_collinear)), nbody);
    return {
        {"cross_section", total},
        {"event_weight", event},
        {"soft_weight", soft},
        {"collinear_weight", collinear},
        {"soft_collinear_weight", soft_collinear},
        {"nbody_weight", nbody},
        {"lo_weight", lo},
    };
}

FKSCuts::FKSCuts(
    const std::vector<FKSConfiguration>& configurations,
    double jetalgo,
    double jetradius,
    double ptj,
    double etaj,
    int maxjetflavor,
    double ptl,
    double etal,
    double drll,
    double drll_sf,
    double mll,
    double mll_sf
) :
    FunctionGenerator(
        "FKSCuts",
        NamedVector<Type>{
            {"momenta",
             batch_four_vec_array(
                 static_cast<int>(check_fks_configurations(configurations, "FKSCuts"))
             )},
            {"fks_index", batch_int},
        },
        NamedVector<Type>{{"mask", batch_float}}
    ),
    _particle_count(check_fks_configurations(configurations, "FKSCuts")),
    _options{
        jetalgo,
        jetradius,
        ptj,
        etaj,
        static_cast<double>(maxjetflavor),
        ptl,
        etal,
        drll,
        drll_sf,
        mll,
        mll_sf
    } {
    for (const auto& configuration : configurations) {
        std::size_t partons = 0;
        for (std::size_t i = 2; i < configuration.pdgs.size(); ++i) {
            int flavor = std::abs(configuration.pdgs.at(i));
            partons += (flavor >= 1 && flavor <= maxjetflavor) || flavor == 21;
        }
        if (partons > jet_parton_max) {
            throw std::invalid_argument(
                "FKSCuts: more QCD partons than the jet clustering takes"
            );
        }
        _pdg.emplace_back(configuration.pdgs.begin(), configuration.pdgs.end());
    }
}

NamedVector<Value> FKSCuts::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    return {
        {"mask", fb.fks_cuts(args.at("momenta"), args.at("fks_index"), _pdg, _options)}
    };
}

namespace {

// the momentum fractions of the kinematics that the luminosities take
const std::vector<std::string> KINEMATICS = {
    "", "soft_", "collinear_", "soft_collinear_", "born_"
};

// the arguments of FKSCrossSection: those of the subtraction without alpha_s
// and with the factorization scales of the Born and of the remnants, then
// those of the weights that the subtraction and the n-body part do not give,
// fks_index last. With the couplings inside, the scales, the couplings and
// the luminosities give way to the momentum fractions.
NamedVector<Type> cross_section_args(
    const FKSSubtraction& subtraction,
    const FKSSoftVirtual& soft_virtual,
    const FKSWeights& weights,
    bool couplings = false
) {
    if (subtraction.particle_count() != weights.particle_count() ||
        soft_virtual.particle_count() + 1 != weights.particle_count() ||
        soft_virtual.born_order_count() != weights.born_order_count() ||
        soft_virtual.nlo_order_count() != weights.nlo_order_count()) {
        throw std::invalid_argument(
            "FKSCrossSection: the subtraction, the soft-virtual part and the "
            "weights need the same numbers of particles and orders"
        );
    }
    NamedVector<Type> args;
    for (auto [type, name] :
         zip(subtraction.arg_types(), subtraction.arg_types().keys())) {
        if (name == "alpha_s" || name == "fact_scale2" || name == "fks_index" ||
            (couplings && (name == "ren_scale" || name == "fact_scale1"))) {
            continue;
        }
        if (name == "fact_scale1") {
            for (auto scale :
                 {"fact_scale1",
                  "fact_scale2",
                  "collinear_fact_scale1",
                  "collinear_fact_scale2"}) {
                args.push_back(scale, batch_float);
            }
            continue;
        }
        args.push_back(name, type);
    }
    for (auto [type, name] : zip(weights.arg_types(), weights.arg_types().keys())) {
        if (name == "real" || name == "soft" || name == "collinear" ||
            name == "soft_collinear" || name == "soft_virtual" || name == "born") {
            continue;
        }
        bool coupling = name.ends_with("_alpha_s") || name.ends_with("_luminosity");
        if (couplings && name == "event_alpha_s") {
            for (const auto& kinematics : KINEMATICS) {
                args.push_back(kinematics + "x1", batch_float);
                args.push_back(kinematics + "x2", batch_float);
            }
        }
        if (couplings && coupling) {
            continue;
        }
        bool present = false;
        for (const auto& key : args.keys()) {
            present = present || key == name;
        }
        if (!present) {
            args.push_back(name, type);
        }
    }
    args.push_back("fks_index", subtraction.arg_types().at("fks_index"));
    return args;
}

} // namespace

FKSCrossSection::FKSCrossSection(
    const FKSSubtraction& subtraction,
    const FKSSoftVirtual& soft_virtual,
    const FKSWeights& weights,
    const std::optional<FKSCuts>& cuts
) :
    FunctionGenerator(
        "FKSCrossSection",
        cross_section_args(subtraction, soft_virtual, weights),
        weights.return_types()
    ),
    _subtraction(subtraction),
    _soft_virtual(soft_virtual),
    _weights(weights),
    _cuts(cuts) {
    if (_cuts && _cuts->particle_count() != weights.particle_count()) {
        throw std::invalid_argument(
            "FKSCrossSection: the cuts need the particles of the weights"
        );
    }
}

FKSCrossSection::FKSCrossSection(
    const FKSSubtraction& subtraction,
    const FKSSoftVirtual& soft_virtual,
    const FKSWeights& weights,
    const std::optional<FKSCuts>& cuts,
    const FKSScale& scale,
    const RunningCoupling& running_coupling,
    const FKSLuminosity& luminosity,
    const FKSLuminosity& born_luminosity
) :
    FunctionGenerator(
        "FKSCrossSection",
        cross_section_args(subtraction, soft_virtual, weights, true),
        weights.return_types()
    ),
    _subtraction(subtraction),
    _soft_virtual(soft_virtual),
    _weights(weights),
    _cuts(cuts),
    _scale(scale.with_particle_count(weights.particle_count())),
    _born_scale(scale.with_particle_count(weights.particle_count() - 1)),
    _running_coupling(running_coupling),
    _luminosity(luminosity),
    _born_luminosity(born_luminosity) {
    if (_cuts && _cuts->particle_count() != weights.particle_count()) {
        throw std::invalid_argument(
            "FKSCrossSection: the cuts need the particles of the weights"
        );
    }
    std::size_t count = subtraction.configuration_count();
    if (luminosity.configuration_count() != count ||
        (born_luminosity.configuration_count() != count &&
         born_luminosity.configuration_count() != 1)) {
        throw std::invalid_argument(
            "FKSCrossSection: the luminosities need the FKS configurations of the "
            "subtraction"
        );
    }
}

NamedVector<Value> FKSCrossSection::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& arguments
) const {
    // the arguments, and with the couplings inside the scales, the couplings
    // and the luminosities of each kinematics
    std::map<std::string, Value> args;
    for (auto [value, name] : zip(arguments, arguments.keys())) {
        args[name] = value;
    }
    auto batch_size = fb.batch_size({args.at("born_weight")});
    if (_scale) {
        auto born = _born_scale->build_function(fb, {args.at("born_momenta")});
        auto collinear = _scale->build_function(fb, {args.at("collinear_momenta")});
        auto event = _scale->build_function(fb, {args.at("momenta")});
        args["ren_scale"] = born.at("ren_scale");
        args["fact_scale1"] = born.at("fact_scale1");
        args["fact_scale2"] = born.at("fact_scale2");
        args["collinear_fact_scale1"] = collinear.at("fact_scale1");
        args["collinear_fact_scale2"] = collinear.at("fact_scale2");
        auto coupling = [&](const NamedVector<Value>& scales) {
            return _running_coupling->build_function(fb, {scales.at("ren_scale")})
                .at(0);
        };
        args["event_alpha_s"] = coupling(event);
        args["born_alpha_s"] = coupling(born);
        args["collinear_alpha_s"] = coupling(collinear);
        auto luminosity =
            [&](const std::string& kinematics, const NamedVector<Value>& scales) {
                bool nbody = kinematics == "born_";
                const auto& function = nbody ? *_born_luminosity : *_luminosity;
                Value index = args.at("fks_index");
                if (function.configuration_count() == 1) {
                    index = fb.full({static_cast<me_int_t>(0), batch_size});
                }
                return function
                    .build_function(
                        fb,
                        {args.at(kinematics + "x1"),
                         args.at(kinematics + "x2"),
                         scales.at("fact_scale1"),
                         scales.at("fact_scale2"),
                         index}
                    )
                    .at(0);
            };
        args["event_luminosity"] = luminosity("", event);
        args["soft_luminosity"] = luminosity("soft_", born);
        args["soft_collinear_luminosity"] = luminosity("soft_collinear_", born);
        args["collinear_luminosity"] = luminosity("collinear_", collinear);
        args["born_luminosity"] = luminosity("born_", born);
    }
    // the subtraction and the n-body part at g_s = 1
    Value alpha_s = fb.full({1. / (4. * PI), batch_size});
    ValueVec subtraction_args;
    for (const auto& name : _subtraction.arg_types().keys()) {
        if (name == "alpha_s") {
            subtraction_args.push_back(alpha_s);
        } else if (name == "fact_scale1" || name == "fact_scale2") {
            subtraction_args.push_back(args.at("collinear_" + name));
        } else {
            subtraction_args.push_back(args.at(name));
        }
    }
    auto subtraction = _subtraction.build_function(fb, subtraction_args);
    // the Born orders among the Borns of the subtraction
    Value born = args.at("born"), color_correlated = args.at("color_correlated");
    std::size_t born_orders = _weights.born_order_count();
    if (_subtraction.arg_types().at("born").shape.at(0) !=
        static_cast<int>(born_orders)) {
        auto borns = fb.unstack(born), colors = fb.unstack(color_correlated);
        born = fb.stack(ValueVec(borns.begin(), borns.begin() + born_orders));
        color_correlated =
            fb.stack(ValueVec(colors.begin(), colors.begin() + born_orders));
    }
    auto soft_virtual = _soft_virtual.build_function(
        fb,
        {args.at("born_momenta"),
         born,
         color_correlated,
         alpha_s,
         args.at("alpha"),
         args.at("ren_scale"),
         args.at("fact_scale1"),
         args.at("fact_scale2")}
    );
    // the cut of the kinematics of each luminosity, as sigint
    std::map<std::string, Value> cut;
    if (_cuts) {
        auto mask = [&](const std::string& momenta) {
            return _cuts->build_function(fb, {args.at(momenta), args.at("fks_index")})
                .at("mask");
        };
        Value event = mask("momenta"), soft = mask("soft_momenta");
        bool projected = true;
        for (const auto& name : _subtraction.arg_types().keys()) {
            projected = projected && name != "collinear_born";
        }
        cut = {
            {"event_luminosity", event},
            {"soft_luminosity", soft},
            {"soft_collinear_luminosity", soft},
            {"born_luminosity", soft},
            {"collinear_luminosity", projected ? soft : mask("collinear_momenta")},
        };
    }
    ValueVec weight_args;
    for (const auto& name : _weights.arg_types().keys()) {
        if (name == "real" || name == "soft" || name == "collinear" ||
            name == "soft_collinear") {
            weight_args.push_back(subtraction.at(name));
        } else if (name == "soft_virtual") {
            weight_args.push_back(soft_virtual.at("soft_virtual"));
        } else if (name == "born") {
            weight_args.push_back(born);
        } else if (cut.count(name) != 0) {
            weight_args.push_back(fb.mul(args.at(name), cut.at(name)));
        } else {
            weight_args.push_back(args.at(name));
        }
    }
    return _weights.build_function(fb, weight_args);
}
