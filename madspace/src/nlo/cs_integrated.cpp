#include "madspace/nlo/cs_integrated.hpp"
#include "madspace/nlo/common.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace madspace;

namespace {

constexpr double PI = 3.14159265358979323846;

NamedVector<Type> insertion_arg_types(
    const std::vector<int>& pdgs,
    const std::vector<double>& masses,
    std::size_t orders,
    const std::string& name = "CSInsertion",
    std::size_t incoming = 2,
    bool qed = false
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
    if (qed) {
        return {
            {"momenta", batch_four_vec_array(n)},
            {"born", batch_float_array(b)},
            {"alpha", batch_float},
            {"ren_scale", batch_float},
        };
    }
    return {
        {"momenta", batch_four_vec_array(n)},
        {"color_correlated", Type{DataType::dt_float64, batch_size, {b, n, n}}},
        {"alpha_s", batch_float},
        {"ren_scale", batch_float},
    };
}

// the electric charge of a particle of the SM in units of e, zero for the
// neutral ones
double electric_charge(int pdg) {
    int a = std::abs(pdg);
    double q = 0.;
    if (a >= 1 && a <= 6) {
        q = a % 2 == 0 ? 2. / 3. : -1. / 3.;
    } else if (a == 11 || a == 13 || a == 15) {
        q = -1.;
    } else if (a == 24 || a == 37) {
        q = 1.;
    }
    return pdg < 0 ? -q : q;
}

// Q_j^2, gamma_j, K_j, the couplings of a photon to the n_f light quarks and
// n_l massless leptons, the charges flowing out of the particles of the Born
// and the weights of the spectators of a photon, after checking the masses
void charge_factors(
    const std::vector<int>& pdgs,
    const std::vector<double>& masses,
    double n_f,
    double n_l,
    std::size_t incoming,
    std::vector<double>& casimir,
    std::vector<double>& gamma,
    std::vector<double>& k_constant,
    std::vector<double>& pair_splitting,
    std::vector<double>& charges,
    std::vector<double>& weights
) {
    if (!(n_f >= 0. && n_f <= 6.) || !(n_l >= 0. && n_l <= 3.)) {
        throw std::invalid_argument("CSInsertion: n_f in [0, 6] and n_l in [0, 3]");
    }
    // sum_f N_c Q_f^2 over the light fermions
    double pairs = n_l;
    for (int q = 1; q <= static_cast<int>(n_f); ++q) {
        pairs += 3. * electric_charge(q) * electric_charge(q);
    }
    std::size_t charged = 0;
    for (std::size_t j = 0; j < pdgs.size(); ++j) {
        double q = electric_charge(pdgs[j]);
        bool photon = pdgs[j] == 22;
        if (masses[j] < 0. || (photon && masses[j] != 0.)) {
            throw std::invalid_argument("CSInsertion: invalid mass");
        }
        if (incoming == 2 && j < 2 && (q != 0. || photon) && masses[j] != 0.) {
            throw std::invalid_argument("CSInsertion: incoming partons are massless");
        }
        casimir.push_back(q * q);
        gamma.push_back(photon ? -2. / 3. * pairs : 1.5 * q * q);
        k_constant.push_back(photon ? -10. / 9. * pairs : (3.5 - PI * PI / 6.) * q * q);
        pair_splitting.push_back(photon ? pairs : 0.);
        charges.push_back(j < incoming ? -q : q);
        bool spectator = q != 0. && !(incoming == 1 && j == 0);
        weights.push_back(spectator ? 1. : 0.);
        charged += spectator ? 1 : 0;
    }
    for (std::size_t j = 0; j < pdgs.size(); ++j) {
        if (pdgs[j] == 22 && charged == 0) {
            throw std::invalid_argument(
                "CSInsertion: a photon without a charged particle as its spectator"
            );
        }
    }
    for (auto& weight : weights) {
        weight = charged == 0 ? 0. : weight / static_cast<double>(charged);
    }
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
    double gamma_q = 1.5 * QCD_CF, gamma_g = 11. / 6. * QCD_CA - 2. / 3. * QCD_TR * n_f;
    double k_q = (3.5 - PI * PI / 6.) * QCD_CF;
    double k_g = (67. / 18. - PI * PI / 6.) * QCD_CA - 10. / 9. * QCD_TR * n_f;
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
        casimir.push_back(quark ? QCD_CF : gluon ? QCD_CA : 0.);
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
    std::size_t incoming,
    bool qed,
    double n_l
) :
    FunctionGenerator(
        "CSInsertion",
        insertion_arg_types(pdgs, masses, born_orders, "CSInsertion", incoming, qed),
        {{"insertion", batch_float_array(static_cast<int>(born_orders))},
         {"poles",
          Type{DataType::dt_float64, batch_size, {static_cast<int>(born_orders), 2}}}}
    ),
    _born_orders(born_orders),
    _masses(masses),
    _kappa(kappa),
    _alpha_ff(alpha_ff),
    _n_f(n_f),
    _incoming(incoming),
    _qed(qed) {
    if (!(alpha_ff > 0. && alpha_ff <= 1.)) {
        throw std::invalid_argument("CSInsertion: alpha_ff has to be in (0, 1]");
    }
    if (qed) {
        charge_factors(
            pdgs,
            masses,
            n_f,
            n_l,
            incoming,
            _casimir,
            _gamma,
            _k_constant,
            _pair_splitting,
            _charges,
            _weights
        );
        _self_coupling.assign(pdgs.size(), 0.);
        return;
    }
    color_factors(
        pdgs, masses, n_f, "CSInsertion", _casimir, _gamma, _k_constant, incoming
    );
    for (double casimir : _casimir) {
        _self_coupling.push_back(casimir == QCD_CA ? QCD_CA : 0.);
        _pair_splitting.push_back(casimir == QCD_CA ? QCD_TR * n_f : 0.);
    }
    _charges.assign(pdgs.size(), 0.);
    _weights.assign(pdgs.size(), 0.);
}

CSCollinear::CSCollinear(
    const std::vector<int>& pdgs,
    const std::vector<double>& masses,
    std::size_t born_orders,
    double alpha_fi,
    double alpha_if,
    double alpha_ii,
    double n_f,
    bool qed,
    double n_l
) :
    FunctionGenerator(
        "CSCollinear",
        [&] {
            auto types = insertion_arg_types(pdgs, masses, born_orders, "CSCollinear");
            auto b = static_cast<int>(born_orders);
            NamedVector<Type> out{
                {"momenta", types.at("momenta")},
                {"born", batch_float_array(b)},
            };
            if (!qed) {
                out.push_back("color_correlated", types.at("color_correlated"));
            }
            out.push_back(qed ? "alpha" : "alpha_s", batch_float);
            for (auto name : {"fact_scale1", "fact_scale2", "x1", "x2", "r"}) {
                out.push_back(name, batch_float);
            }
            return out;
        }(),
        {{"z", batch_float_array(2)},
         {"weights",
          Type{
              DataType::dt_float64, batch_size, {static_cast<int>(born_orders), 2, 3}
          }}}
    ),
    _born_orders(born_orders),
    _masses(masses),
    _options{alpha_fi, alpha_if, alpha_ii, n_f, qed ? 1. : 0.},
    _qed(qed) {
    for (double alpha : {alpha_fi, alpha_if, alpha_ii}) {
        if (!(alpha > 0. && alpha <= 1.)) {
            throw std::invalid_argument("CSCollinear: alpha has to be in (0, 1]");
        }
    }
    std::vector<double> k_constant;
    if (qed) {
        charge_factors(
            pdgs,
            masses,
            n_f,
            n_l,
            2,
            _casimir,
            _gamma,
            k_constant,
            _pair_splitting,
            _charges,
            _weights
        );
        _self_coupling.assign(pdgs.size(), 0.);
        for (int pdg : pdgs) {
            // gamma -> f with N_c Q_f^2, f -> gamma per unit Q_f^2
            double q = electric_charge(pdg);
            int a = std::abs(pdg);
            double colors = a >= 1 && a <= 6 ? 3. : 1.;
            bool fermion = (a >= 1 && a <= 6) || a == 11 || a == 13 || a == 15;
            _other_coefficient.push_back(
                pdg == 22     ? 1.
                    : fermion ? colors * q * q
                              : 0.
            );
        }
        return;
    }
    color_factors(pdgs, masses, n_f, "CSCollinear", _casimir, _gamma, k_constant);
    for (double casimir : _casimir) {
        bool gluon = casimir == QCD_CA;
        _self_coupling.push_back(gluon ? QCD_CA : 0.);
        _pair_splitting.push_back(gluon ? QCD_TR * n_f : 0.);
        _other_coefficient.push_back(gluon ? QCD_CF : casimir == QCD_CF ? QCD_TR : 0.);
    }
    _charges.assign(pdgs.size(), 0.);
    _weights.assign(pdgs.size(), 0.);
}

NamedVector<Value> CSCollinear::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto batch = fb.batch_size({args.at("momenta")});
    auto n = static_cast<me_int_t>(_masses.size());
    auto b = static_cast<me_int_t>(_born_orders);
    Value color_correlated =
        _qed ? fb.full({0., batch, b, n, n}) : args.at("color_correlated");
    auto [z, weights] = fb.cs_collinear(
        args.at("momenta"),
        args.at("born"),
        color_correlated,
        args.at(_qed ? "alpha" : "alpha_s"),
        args.at("fact_scale1"),
        args.at("fact_scale2"),
        args.at("x1"),
        args.at("x2"),
        args.at("r"),
        _casimir,
        _gamma,
        _self_coupling,
        _pair_splitting,
        _other_coefficient,
        _charges,
        _weights,
        _masses,
        _options
    );
    return {{"z", z}, {"weights", weights}};
}

NamedVector<Value> CSInsertion::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    // the color-correlated Born of QCD or the Born of QED, the other one zero
    auto batch = fb.batch_size({args.at("momenta")});
    auto n = static_cast<me_int_t>(_masses.size());
    auto b = static_cast<me_int_t>(_born_orders);
    Value color_correlated =
        _qed ? fb.full({0., batch, b, n, n}) : args.at("color_correlated");
    Value born = _qed ? args.at("born") : fb.full({0., batch, b});
    auto [insertion, poles] = fb.cs_insertion(
        args.at("momenta"),
        color_correlated,
        born,
        args.at(_qed ? "alpha" : "alpha_s"),
        args.at("ren_scale"),
        _casimir,
        _gamma,
        _k_constant,
        _self_coupling,
        _pair_splitting,
        _charges,
        _weights,
        _masses,
        std::vector<double>{_kappa, _alpha_ff, _n_f, _qed ? 1. : 0.},
        static_cast<me_int_t>(_incoming)
    );
    return {{"insertion", insertion}, {"poles", poles}};
}
