#include "madspace/phasespace/cs_integrated.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace madspace;

namespace {

constexpr double CA = 3., CF = 4. / 3., TR = 0.5;
constexpr double PI = 3.14159265358979323846;

NamedVector<Type> insertion_arg_types(
    const std::vector<int>& pdgs,
    const std::vector<double>& masses,
    std::size_t orders,
    const std::string& name = "CSInsertion",
    std::size_t incoming = 2
) {
    if (incoming != 1 && incoming != 2) {
        throw std::invalid_argument(name + ": one or two incoming particles");
    }
    if (pdgs.size() != masses.size() || pdgs.size() < 3) {
        throw std::invalid_argument(
            name +
            ": the Born needs its incoming particles, two outgoing ones for a decay "
            "and one else, and a mass for each"
        );
    }
    if (orders == 0) {
        throw std::invalid_argument(name + ": there has to be a Born order");
    }
    auto n = static_cast<int>(pdgs.size());
    auto b = static_cast<int>(orders);
    return {
        {"momenta", batch_four_vec_array(n)},
        {"color_correlated", Type{DataType::dt_float64, batch_size, {b, n, n}}},
        {"alpha_s", batch_float},
        {"ren_scale", batch_float},
    };
}

// T_j^2, gamma_j and K_j of the particles of the Born (zero for colorless
// ones), after checking their masses
void color_factors(
    const std::vector<int>& pdgs,
    const std::vector<double>& masses,
    double n_f,
    const std::string& name,
    std::vector<double>& casimir,
    std::vector<double>& gamma,
    std::vector<double>& k_constant,
    std::size_t incoming = 2
) {
    if (!(n_f >= 0.)) {
        throw std::invalid_argument(name + ": n_f must not be negative");
    }
    double gamma_q = 1.5 * CF, gamma_g = 11. / 6. * CA - 2. / 3. * TR * n_f;
    double k_q = (3.5 - PI * PI / 6.) * CF;
    double k_g = (67. / 18. - PI * PI / 6.) * CA - 10. / 9. * TR * n_f;
    for (std::size_t j = 0; j < pdgs.size(); ++j) {
        int pdg = std::abs(pdgs[j]);
        bool quark = pdg >= 1 && pdg <= 6, gluon = pdg == 21;
        if (masses[j] < 0. || (gluon && masses[j] != 0.)) {
            throw std::invalid_argument(name + ": invalid mass");
        }
        if (incoming == 2 && j < 2 && (quark || gluon) && masses[j] != 0.) {
            throw std::invalid_argument(name + ": incoming partons are massless");
        }
        if (incoming == 1 && j == 0 && gluon) {
            throw std::invalid_argument(name + ": a gluon does not decay");
        }
        casimir.push_back(quark ? CF : gluon ? CA : 0.);
        gamma.push_back(quark ? gamma_q : gluon ? gamma_g : 0.);
        k_constant.push_back(quark ? k_q : gluon ? k_g : 0.);
    }
}

} // namespace

CSInsertion::CSInsertion(
    const std::vector<int>& pdgs,
    const std::vector<double>& masses,
    std::size_t born_orders,
    double kappa,
    double alpha_ff,
    double n_f,
    std::size_t incoming
) :
    FunctionGenerator(
        "CSInsertion",
        insertion_arg_types(pdgs, masses, born_orders, "CSInsertion", incoming),
        {{"insertion", batch_float_array(static_cast<int>(born_orders))},
         {"poles",
          Type{DataType::dt_float64, batch_size, {static_cast<int>(born_orders), 2}}}}
    ),
    _born_orders(born_orders),
    _masses(masses),
    _kappa(kappa),
    _alpha_ff(alpha_ff),
    _n_f(n_f),
    _incoming(incoming) {
    if (!(alpha_ff > 0. && alpha_ff <= 1.)) {
        throw std::invalid_argument("CSInsertion: alpha_ff has to be in (0, 1]");
    }
    color_factors(
        pdgs, masses, n_f, "CSInsertion", _casimir, _gamma, _k_constant, incoming
    );
    if (incoming == 1 && _casimir[0] != 0.) {
        for (std::size_t j = 1; j < pdgs.size(); ++j) {
            if (_casimir[j] == CA) {
                throw std::invalid_argument(
                    "CSInsertion: no dipoles of a final-state gluon with a colored "
                    "decaying particle"
                );
            }
        }
    }
}

CSCollinear::CSCollinear(
    const std::vector<int>& pdgs,
    const std::vector<double>& masses,
    std::size_t born_orders,
    double alpha_fi,
    double alpha_if,
    double alpha_ii,
    double n_f
) :
    FunctionGenerator(
        "CSCollinear",
        [&] {
            auto types = insertion_arg_types(pdgs, masses, born_orders, "CSCollinear");
            auto b = static_cast<int>(born_orders);
            return NamedVector<Type>{
                {"momenta", types.at("momenta")},
                {"born", batch_float_array(b)},
                {"color_correlated", types.at("color_correlated")},
                {"alpha_s", batch_float},
                {"fact_scale1", batch_float},
                {"fact_scale2", batch_float},
                {"x1", batch_float},
                {"x2", batch_float},
                {"r", batch_float},
            };
        }(),
        {{"z", batch_float_array(2)},
         {"weights",
          Type{
              DataType::dt_float64, batch_size, {static_cast<int>(born_orders), 2, 3}
          }}}
    ),
    _born_orders(born_orders),
    _masses(masses),
    _options{alpha_fi, alpha_if, alpha_ii, n_f} {
    for (double alpha : {alpha_fi, alpha_if, alpha_ii}) {
        if (!(alpha > 0. && alpha <= 1.)) {
            throw std::invalid_argument("CSCollinear: alpha has to be in (0, 1]");
        }
    }
    std::vector<double> k_constant;
    color_factors(pdgs, masses, n_f, "CSCollinear", _casimir, _gamma, k_constant);
}

NamedVector<Value> CSCollinear::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto [z, weights] = fb.cs_collinear(
        args.at("momenta"),
        args.at("born"),
        args.at("color_correlated"),
        args.at("alpha_s"),
        args.at("fact_scale1"),
        args.at("fact_scale2"),
        args.at("x1"),
        args.at("x2"),
        args.at("r"),
        _casimir,
        _gamma,
        _masses,
        _options
    );
    return {{"z", z}, {"weights", weights}};
}

NamedVector<Value> CSInsertion::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto [insertion, poles] = fb.cs_insertion(
        args.at("momenta"),
        args.at("color_correlated"),
        args.at("alpha_s"),
        args.at("ren_scale"),
        _casimir,
        _gamma,
        _k_constant,
        _masses,
        std::vector<double>{_kappa, _alpha_ff, _n_f},
        static_cast<me_int_t>(_incoming)
    );
    return {{"insertion", insertion}, {"poles", poles}};
}
