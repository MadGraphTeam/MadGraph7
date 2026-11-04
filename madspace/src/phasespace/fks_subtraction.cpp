#include "madspace/phasespace/fks_subtraction.hpp"

#include <stdexcept>

using namespace madspace;

FKSSubtraction::FKSSubtraction(
    const std::vector<FKSConfiguration>& configurations,
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
    double n_f
) :
    FunctionGenerator(
        "FKSSubtraction",
        [&] {
            // the limits check the configurations
            auto n = static_cast<int>(FKSLimits(configurations).particle_count());
            return NamedVector<Type>{
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
                {"matrix_element", batch_float},
                {"born", batch_float},
                {"color_correlated",
                 Type{DataType::dt_float64, batch_size, {n - 1, n - 1}}},
                {"spin_correlated", batch_float},
                {"alpha_s", batch_float},
                {"fact_scale1", batch_float},
                {"fact_scale2", batch_float},
                {"fks_index", batch_int},
            };
        }(),
        {{"real", batch_float},
         {"soft", batch_float},
         {"collinear", batch_float},
         {"soft_collinear", batch_float}}
    ),
    _sector_function(
        configurations, fks_a, fks_b, use_energy, use_beta, h_damp_one, h_damp_power
    ),
    _limits(configurations),
    _prefactors(configurations, xicut, xi_s_cut, delta_o, delta_i, delta_s),
    _xi_s_cut(xi_s_cut),
    _delta_s(delta_s),
    _tiny(tiny),
    _pdf_scheme(pdf_scheme),
    _n_f(n_f) {
    if (!(tiny >= 0.)) {
        throw std::invalid_argument("FKSSubtraction: tiny cannot be negative");
    }
    for (std::size_t c = 0; c < configurations.size(); ++c) {
        const auto& config = configurations.at(c);
        bool massless = config.masses.at(config.j_fks) == 0.;
        _symmetry.push_back(config.symmetry_factor());
        _massless.push_back(massless ? 1 : 0);
        _remnant.push_back(massless && config.initial_state() ? 1 : 0);
        _beam.push_back(config.initial_state() ? config.j_fks : 0);
        _kind.push_back(_limits.splitting_kernels().splitting_kind(c));
        _ratio.push_back(config.identical_particle_ratio());
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
    // the limits at the event, where the real emission switches to them, with
    // the direction of i of the soft counter-event as eikonal_reduced
    auto event_limits = _limits.build_function(
        fb,
        {args.at("momenta"),
         args.at("soft_p_i_hat"),
         args.at("y"),
         args.at("momenta"),
         args.at("momenta"),
         args.at("born"),
         args.at("color_correlated"),
         args.at("spin_correlated"),
         args.at("alpha_s"),
         index}
    );
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
        fb,
        {args.at("soft_momenta"),
         args.at("soft_p_i_hat"),
         args.at("y"),
         args.at("collinear_momenta"),
         args.at("soft_collinear_momenta"),
         args.at("born"),
         args.at("color_correlated"),
         args.at("spin_correlated"),
         args.at("alpha_s"),
         index}
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
    auto counterterm =
        [&](const std::string& name, const std::string& prefactor, Value s) {
            return fb.mul(fb.mul(s, prefactors.at(prefactor)), limits.at(name));
        };
    auto [remnant_c, remnant_sc] = fb.fks_collinear_remnants(
        args.at("born"),
        args.at("alpha_s"),
        s_c,
        s_sc,
        args.at("collinear_momenta"),
        args.at("soft_collinear_momenta"),
        args.at("collinear_xi"),
        prefactors.at("f_dc"),
        prefactors.at("f_dsc"),
        prefactors.at("f_pdf_scheme"),
        args.at("fact_scale1"),
        args.at("fact_scale2"),
        index,
        _remnant,
        _beam,
        _kind,
        _ratio,
        _delta_i,
        std::vector<double>{static_cast<double>(_pdf_scheme), _n_f}
    );
    auto [real_out, soft, collinear, soft_collinear] = fb.fks_subtraction(
        real,
        counterterm("soft", "f_s", sector("soft_")),
        counterterm("collinear", "f_c", s_c),
        counterterm("soft_collinear", "f_sc", s_sc),
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
