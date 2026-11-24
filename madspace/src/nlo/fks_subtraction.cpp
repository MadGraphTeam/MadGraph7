#include "madspace/nlo/fks_subtraction.hpp"
#include "madspace/nlo/common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <stdexcept>

#include "madspace/constants.hpp"

using namespace madspace;

namespace {

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
            return coupling == QCD ? -2. * n_f * QCD_TR : 0.;
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
    _incoming(configurations.at(0).incoming),
    _xicut(xicut),
    _delta_o(delta_o) {
    const auto& config = configurations.at(born_configuration(configurations));
    // the factors of the gluon and the quarks (QCD), of the photon (QED)
    double gamma_g = (11. * QCD_CA - 2. * n_f) / 6.;
    double gamma_prime_g = (67. / 9. - 2. * PI * PI / 3.) * QCD_CA - 23. * n_f / 18.;
    double gamma_q = 1.5 * QCD_CF;
    double gamma_prime_q = (6.5 - 2. * PI * PI / 3.) * QCD_CF;
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
        _casimir.at(QCD).push_back(gluon ? QCD_CA : (quark ? QCD_CF : 0.));
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
                    return (r < linked.incoming ? -1. : 1.) * linked.charges.at(r);
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
        std::vector<double>{_xicut, _delta_o},
        static_cast<me_int_t>(_incoming)
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
    const std::vector<FKSConfiguration>& configurations,
    const FKSCouplingOrders& orders,
    std::size_t ew_sudakov_modes,
    int ew_sudakov_power
) :
    FunctionGenerator(
        "FKSWeights",
        [&] {
            auto n = static_cast<int>(
                check_fks_configurations(configurations, "FKSWeights")
            );
            auto b = static_cast<int>(orders.born_count());
            auto o = static_cast<int>(orders.nlo_count());
            auto types = NamedVector<Type>{
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
            if (ew_sudakov_modes > 0) {
                types.push_back(
                    "ew_sudakov", batch_float_array(static_cast<int>(ew_sudakov_modes))
                );
            }
            return types;
        }(),
        [&] {
            auto types = NamedVector<Type>{
                {"cross_section", batch_float},
                {"event_weight", batch_float},
                {"soft_weight", batch_float},
                {"collinear_weight", batch_float},
                {"soft_collinear_weight", batch_float},
                {"nbody_weight", batch_float},
                {"lo_weight", batch_float},
            };
            if (ew_sudakov_modes > 0) {
                types.push_back(
                    "ew_sudakov_weights",
                    batch_float_array(static_cast<int>(ew_sudakov_modes))
                );
            }
            return types;
        }()
    ),
    _particle_count(check_fks_configurations(configurations, "FKSWeights")),
    _incoming(configurations.at(0).incoming),
    _born_power(qcd_powers(orders, orders.born)),
    _nlo_power(qcd_powers(orders, orders.nlo)),
    _ew_sudakov_modes(ew_sudakov_modes),
    _ew_sudakov_power(ew_sudakov_power) {
    if (ew_sudakov_power < 0) {
        throw std::invalid_argument(
            "FKSWeights: the power of g_s of the Sudakov approximation must not be "
            "negative"
        );
    }
}

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
                powers,
                static_cast<me_int_t>(_incoming)
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
    NamedVector<Value> result{
        {"cross_section", total},
        {"event_weight", event},
        {"soft_weight", soft},
        {"collinear_weight", collinear},
        {"soft_collinear_weight", soft_collinear},
        {"nbody_weight", nbody},
        {"lo_weight", lo},
    };
    if (_ew_sudakov_modes > 0) {
        // each mode an n-body part of its own, with the coupling, the
        // luminosity and the factor of the Born
        std::vector<me_int_t> power{static_cast<me_int_t>(_ew_sudakov_power)};
        ValueVec modes;
        for (Value mode : fb.unstack(args.at("ew_sudakov"))) {
            modes.push_back(weight(
                fb.stack({mode}),
                "born_",
                "born_weight",
                "born",
                "born_",
                channel,
                power
            ));
        }
        result.push_back("ew_sudakov_weights", fb.stack(modes));
    }
    return result;
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
// the luminosities give way to the momentum fractions, with dressed leptons
// also 1 - x of the beams (the same for every kinematics), the scale of the
// user takes the reference scales of the event, the soft counter-event and the
// collinear counter-event, and luminosities of one flavor take flavor_index
// before fks_index.
NamedVector<Type> cross_section_args(
    const FKSSubtraction& subtraction,
    const FKSSoftVirtual& soft_virtual,
    const FKSWeights& weights,
    bool cuts,
    bool couplings = false,
    bool flavor = false,
    bool lepton = false,
    bool user = false
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
        if (cuts && name == "soft_xi_max") {
            // where the soft counter-event exists, for its cuts
            args.push_back("soft_xi", batch_float);
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
            if (lepton) {
                args.push_back("one_minus_x1", batch_float);
                args.push_back("one_minus_x2", batch_float);
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
    if (user) {
        for (auto name :
             {"reference_scale", "soft_reference_scale", "collinear_reference_scale"}) {
            args.push_back(name, batch_float);
        }
    }
    if (flavor) {
        args.push_back("flavor_index", batch_int);
    }
    args.push_back("fks_index", subtraction.arg_types().at("fks_index"));
    return args;
}

void check_cuts(
    const std::optional<NLOCuts>& cuts,
    const std::optional<NLOCuts>& nbody_cuts,
    const FKSWeights& weights
) {
    for (const auto* item : {&cuts, &nbody_cuts}) {
        if (*item && (*item)->particle_count() != weights.particle_count()) {
            throw std::invalid_argument(
                "FKSCrossSection: the cuts need the particles of the weights"
            );
        }
        if (*item && (*item)->unresolved() != 1) {
            throw std::invalid_argument(
                "FKSCrossSection: the cuts need one unresolved parton, that of the "
                "real emission"
            );
        }
    }
    if (nbody_cuts &&
        (!cuts || nbody_cuts->layouts().size() != cuts->layouts().size())) {
        throw std::invalid_argument(
            "FKSCrossSection: the cuts of the n-body part need the cuts, with a "
            "layout per FKS configuration as they have"
        );
    }
}

} // namespace

FKSCrossSection::FKSCrossSection(
    const FKSSubtraction& subtraction,
    const FKSSoftVirtual& soft_virtual,
    const FKSWeights& weights,
    const std::optional<NLOCuts>& cuts,
    const std::optional<NLOCuts>& nbody_cuts
) :
    FunctionGenerator(
        "FKSCrossSection",
        cross_section_args(subtraction, soft_virtual, weights, cuts.has_value()),
        weights.return_types()
    ),
    _subtraction(subtraction),
    _soft_virtual(soft_virtual),
    _weights(weights),
    _cuts(cuts),
    _nbody_cuts(nbody_cuts) {
    check_cuts(_cuts, _nbody_cuts, weights);
}

FKSCrossSection::FKSCrossSection(
    const FKSSubtraction& subtraction,
    const FKSSoftVirtual& soft_virtual,
    const FKSWeights& weights,
    const std::optional<NLOCuts>& cuts,
    const NLOScale& scale,
    const RunningCoupling& running_coupling,
    const NLOLuminosity& luminosity,
    const NLOLuminosity& born_luminosity,
    const std::optional<NLOCuts>& nbody_cuts,
    const std::vector<std::pair<double, double>>& scale_variations,
    const std::vector<std::tuple<RunningCoupling, NLOLuminosity, NLOLuminosity>>&
        pdf_variations
) :
    FunctionGenerator(
        "FKSCrossSection",
        cross_section_args(
            subtraction,
            soft_virtual,
            weights,
            cuts.has_value(),
            true,
            !luminosity.sum_flavors(),
            luminosity.lepton(),
            scale.user()
        ),
        [&] {
            auto types = weights.return_types();
            if (!scale_variations.empty()) {
                types.push_back(
                    "scale_variations",
                    batch_float_array(static_cast<int>(scale_variations.size()))
                );
            }
            if (!pdf_variations.empty()) {
                types.push_back(
                    "pdf_variations",
                    batch_float_array(static_cast<int>(pdf_variations.size()))
                );
            }
            return types;
        }()
    ),
    _subtraction(subtraction),
    _soft_virtual(soft_virtual),
    _weights(weights),
    _cuts(cuts),
    _nbody_cuts(nbody_cuts),
    _scale(scale.with_particle_count(weights.particle_count())),
    _born_scale(scale.with_particle_count(weights.particle_count() - 1)),
    _running_coupling(running_coupling),
    _luminosity(luminosity),
    _born_luminosity(born_luminosity),
    _scale_variations(scale_variations),
    _pdf_variations(pdf_variations) {
    check_cuts(_cuts, _nbody_cuts, weights);
    for (auto [ren_factor, fact_factor] : scale_variations) {
        if (!(ren_factor > 0.) || !(fact_factor > 0.)) {
            throw std::invalid_argument(
                "FKSCrossSection: the factors of the scale variations must be "
                "positive"
            );
        }
    }
    if (scale.incoming() != weights.incoming()) {
        throw std::invalid_argument(
            "FKSCrossSection: the scales need the incoming particles of the weights"
        );
    }
    std::size_t count = subtraction.configuration_count();
    auto check_luminosities =
        [&](const NLOLuminosity& real, const NLOLuminosity& nbody) {
            if (real.sum_flavors() != nbody.sum_flavors() ||
                real.sum_flavors() != luminosity.sum_flavors()) {
                throw std::invalid_argument(
                    "FKSCrossSection: the luminosities all sum over the flavors or all "
                    "take a flavor index"
                );
            }
            if (real.lepton() != nbody.lepton() ||
                real.lepton() != luminosity.lepton()) {
                throw std::invalid_argument(
                    "FKSCrossSection: the luminosities all take dressed leptons or none"
                );
            }
            // with a flavor index the n-body part takes the Born of the real
            // flavor, per configuration
            if (real.all_groups() || nbody.all_groups() ||
                real.group_count() != count ||
                (nbody.group_count() != count &&
                 (nbody.group_count() != 1 || !real.sum_flavors()))) {
                throw std::invalid_argument(
                    "FKSCrossSection: the luminosities need one group per FKS "
                    "configuration of the subtraction"
                );
            }
        };
    check_luminosities(luminosity, born_luminosity);
    for (const auto& [running_coupling, real, nbody] : pdf_variations) {
        check_luminosities(real, nbody);
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
    // the scales of the event, the Born and the collinear counter-event
    std::map<std::string, NamedVector<Value>> scales;
    if (_scale) {
        // the scale of the user comes as reference scales, the Born that of
        // the soft counter-event, as MadFKS takes it
        auto scale =
            [&](const NLOScale& function,
                const std::string& momenta,
                const std::string& reference) {
                ValueVec inputs{args.at(momenta)};
                if (function.user()) {
                    inputs.push_back(args.at(reference));
                }
                return function.build_function(fb, inputs);
            };
        scales["born"] = scale(*_born_scale, "born_momenta", "soft_reference_scale");
        scales["collinear"] =
            scale(*_scale, "collinear_momenta", "collinear_reference_scale");
        scales["event"] = scale(*_scale, "momenta", "reference_scale");
        args["ren_scale"] = scales["born"].at("ren_scale");
        args["fact_scale1"] = scales["born"].at("fact_scale1");
        args["fact_scale2"] = scales["born"].at("fact_scale2");
        args["collinear_fact_scale1"] = scales["collinear"].at("fact_scale1");
        args["collinear_fact_scale2"] = scales["collinear"].at("fact_scale2");
    }
    // the couplings and luminosities of each kinematics at the scales times
    // factors of mu_R and mu_F
    auto couplings =
        [&](double ren_factor,
            double fact_factor,
            const RunningCoupling& running_coupling,
            const NLOLuminosity& real_luminosity,
            const NLOLuminosity& born_luminosity) {
            std::map<std::string, Value> out;
            auto factor = [&](Value value, double f) {
                return f == 1. ? value : fb.mul(value, fb.full({f, batch_size}));
            };
            auto coupling = [&](const std::string& kinematics) {
                Value ren = factor(scales.at(kinematics).at("ren_scale"), ren_factor);
                return running_coupling.build_function(fb, {ren}).at(0);
            };
            out["event_alpha_s"] = coupling("event");
            out["born_alpha_s"] = coupling("born");
            out["collinear_alpha_s"] = coupling("collinear");
            auto luminosity = [&](const std::string& kinematics,
                                  const std::string& at) {
                bool nbody = kinematics == "born_";
                const auto& function = nbody ? born_luminosity : real_luminosity;
                Value index = args.at("fks_index");
                if (function.group_count() == 1) {
                    index = fb.full({static_cast<me_int_t>(0), batch_size});
                }
                std::vector<Value> inputs{
                    args.at(kinematics + "x1"), args.at(kinematics + "x2")
                };
                if (function.lepton()) {
                    inputs.push_back(args.at("one_minus_x1"));
                    inputs.push_back(args.at("one_minus_x2"));
                }
                inputs.push_back(factor(scales.at(at).at("fact_scale1"), fact_factor));
                inputs.push_back(factor(scales.at(at).at("fact_scale2"), fact_factor));
                if (!function.sum_flavors()) {
                    inputs.push_back(args.at("flavor_index"));
                }
                inputs.push_back(index);
                return function.build_function(fb, inputs).at(0);
            };
            out["event_luminosity"] = luminosity("", "event");
            out["soft_luminosity"] = luminosity("soft_", "born");
            out["soft_collinear_luminosity"] = luminosity("soft_collinear_", "born");
            out["collinear_luminosity"] = luminosity("collinear_", "collinear");
            out["born_luminosity"] = luminosity("born_", "born");
            return out;
        };
    if (_scale) {
        auto central =
            couplings(1., 1., *_running_coupling, *_luminosity, *_born_luminosity);
        for (auto& [name, value] : central) {
            args[name] = value;
        }
    }
    // the subtraction and the n-body part at g_s = 1, the subtraction with the
    // factorization scales of the collinear counter-event times a factor
    Value alpha_s = fb.full({1. / (4. * PI), batch_size});
    auto subtraction_at = [&](double fact_factor) {
        ValueVec subtraction_args;
        for (const auto& name : _subtraction.arg_types().keys()) {
            if (name == "alpha_s") {
                subtraction_args.push_back(alpha_s);
            } else if (name == "fact_scale1" || name == "fact_scale2") {
                Value scale = args.at("collinear_" + name);
                if (fact_factor != 1.) {
                    scale = fb.mul(scale, fb.full({fact_factor, batch_size}));
                }
                subtraction_args.push_back(scale);
            } else {
                subtraction_args.push_back(args.at(name));
            }
        }
        return _subtraction.build_function(fb, subtraction_args);
    };
    auto subtraction = subtraction_at(1.);
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
        // the soft counter-event only where it exists, at xi = 0 (below the
        // lower bound on s_hat it lies at xi_min without a weight)
        Value exists = fb.cut_one(args.at("soft_xi"), 0., 0.);
        Value event = mask("momenta");
        Value soft = fb.mul(mask("soft_momenta"), exists), nbody = soft;
        if (_nbody_cuts) {
            nbody = fb.mul(
                _nbody_cuts
                    ->build_function(
                        fb, {args.at("soft_momenta"), args.at("fks_index")}
                    )
                    .at("mask"),
                exists
            );
        }
        bool projected = true;
        for (const auto& name : _subtraction.arg_types().keys()) {
            projected = projected && name != "collinear_born";
        }
        cut = {
            {"event_luminosity", event},
            {"soft_luminosity", soft},
            {"soft_collinear_luminosity", soft},
            {"born_luminosity", nbody},
            {"collinear_luminosity", projected ? soft : mask("collinear_momenta")},
        };
    }
    auto weights =
        [&](const NamedVector<Value>& subtraction,
            Value soft_virtual,
            const std::map<std::string, Value>& couplings) {
            ValueVec weight_args;
            for (const auto& name : _weights.arg_types().keys()) {
                if (name == "real" || name == "soft" || name == "collinear" ||
                    name == "soft_collinear") {
                    weight_args.push_back(subtraction.at(name));
                } else if (name == "soft_virtual") {
                    weight_args.push_back(soft_virtual);
                } else if (name == "born") {
                    weight_args.push_back(born);
                } else {
                    Value value =
                        couplings.count(name) != 0 ? couplings.at(name) : args.at(name);
                    weight_args.push_back(
                        cut.count(name) != 0 ? fb.mul(value, cut.at(name)) : value
                    );
                }
            }
            return _weights.build_function(fb, weight_args);
        };
    std::map<std::string, Value> central;
    auto result = weights(subtraction, soft_virtual.at("soft_virtual"), central);
    // the members of the PDF variations at the scales of the run
    ValueVec members;
    for (const auto& [running_coupling, real_luminosity, born_luminosity] :
         _pdf_variations) {
        auto member = weights(
            subtraction,
            soft_virtual.at("soft_virtual"),
            couplings(1., 1., running_coupling, real_luminosity, born_luminosity)
        );
        members.push_back(member.at("cross_section"));
    }
    if (_scale_variations.empty()) {
        if (!members.empty()) {
            result.push_back("pdf_variations", fb.stack(members));
        }
        return result;
    }
    // the variations of mu_R and mu_F, the n-body part with the Ellis-Sexton
    // scale at mu_R: the coefficients of log(mu_R'^2 / mu_R^2) and
    // log(mu_F'^2 / mu_F^2) of the soft-virtual part
    auto parts = fb.unstack(soft_virtual.at("soft_virtual"));
    auto ren = fb.unstack(soft_virtual.at("ren_coefficient"));
    auto fact = fb.unstack(soft_virtual.at("fact_coefficients"));
    ValueVec variations;
    std::map<double, NamedVector<Value>> subtractions;
    for (auto [ren_factor, fact_factor] : _scale_variations) {
        if (subtractions.count(fact_factor) == 0) {
            subtractions.emplace(fact_factor, subtraction_at(fact_factor));
        }
        ValueVec shifted;
        for (std::size_t o = 0; o < parts.size(); ++o) {
            auto beams = fb.unstack(fact.at(o));
            Value logs = fb.add(
                fb.mul(ren.at(o), fb.full({2. * std::log(ren_factor), batch_size})),
                fb.mul(
                    fb.add(beams.at(0), beams.at(1)),
                    fb.full({2. * std::log(fact_factor), batch_size})
                )
            );
            shifted.push_back(fb.add(parts.at(o), logs));
        }
        auto varied = weights(
            subtractions.at(fact_factor),
            fb.stack(shifted),
            couplings(
                ren_factor,
                fact_factor,
                *_running_coupling,
                *_luminosity,
                *_born_luminosity
            )
        );
        variations.push_back(varied.at("cross_section"));
    }
    result.push_back("scale_variations", fb.stack(variations));
    if (!members.empty()) {
        result.push_back("pdf_variations", fb.stack(members));
    }
    return result;
}
