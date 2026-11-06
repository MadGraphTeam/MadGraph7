#include "madspace/phasespace/fks_subtraction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
    bool collinear_born
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
    _limits(configurations, orders, collinear_born),
    _prefactors(configurations, xicut, xi_s_cut, delta_o, delta_i, delta_s),
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
    auto prefactors = _prefactors.build_function(
        fb,
        {args.at("xi"),
         args.at("y"),
         args.at("xi_max"),
         args.at("collinear_xi"),
         args.at("collinear_xi_max"),
         args.at("collinear_momenta"),
         args.at("soft_collinear_momenta"),
         index}
    );
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
