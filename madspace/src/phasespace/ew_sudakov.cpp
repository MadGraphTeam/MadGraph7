#include "madspace/phasespace/ew_sudakov.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <stdexcept>

#include "madspace/constants.hpp"

using namespace madspace;

namespace {

using complex = std::complex<double>;
constexpr complex I{0., 1.};

// sign(a, b) of Fortran
double fsign(double a, double b) { return b >= 0. ? std::abs(a) : -std::abs(a); }

bool any_of(int x, std::initializer_list<int> values) {
    return std::find(values.begin(), values.end(), x) != values.end();
}

// left-handed fermions as pdg * helicity of ewsudakov_functions.f
bool left_lepton(int x) { return any_of(x, {-11, -13, -15}); }
bool left_neutrino(int x) { return any_of(x, {-12, -14, -16}); }
bool left_up(int x) { return any_of(x, {-2, -4, -6}); }
bool left_down(int x) { return any_of(x, {-1, -3, -5}); }

/*
 * The quantum numbers of ewsudakov_functions.f (sdk_*) of a particle with PDG
 * code pdg, helicity hel and ifsign -1 for incoming, 1 for outgoing, with the
 * conventions of the usual model (FAV4 false).
 */
class QuantumNumbers {
public:
    QuantumNumbers(double mw, double mz, int sud_mod) :
        _cw2(mw * mw / (mz * mz)), _sw2(1. - mw * mw / (mz * mz)), _sud_mod(sud_mod) {}

    double sw2() const { return _sw2; }
    double cw2() const { return _cw2; }

    // pdg times the helicity, pdg times ifsign for helicity 0
    static int s_pdg(int pdg, int hel, int ifsign) {
        return hel != 0 ? pdg * hel : pdg * ifsign;
    }

    // sdk_charge
    complex charge(int pdg, int ifsign) const {
        int x = pdg * ifsign;
        double q = 0.;
        if (any_of(x, {11, 13, 15})) {
            q = -1.;
        }
        if (any_of(x, {-11, -13, -15})) {
            q = 1.;
        }
        if (any_of(x, {2, 4, 6})) {
            q = 2. / 3.;
        }
        if (any_of(x, {-2, -4, -6})) {
            q = -2. / 3.;
        }
        if (any_of(x, {1, 3, 5})) {
            q = -1. / 3.;
        }
        if (any_of(x, {-1, -3, -5})) {
            q = 1. / 3.;
        }
        if (x == 251 || x == 24) {
            q = 1.;
        }
        if (x == -251 || x == -24) {
            q = -1.;
        }
        complex c = -q;
        if (std::abs(x) == 251) {
            c *= I;
        }
        return c;
    }

    // sdk_chargesq
    complex charge_squared(int pdg, int ifsign) const {
        complex c = charge(pdg, ifsign);
        c *= c;
        if (std::abs(pdg) == 251) {
            c = -c;
        }
        return c;
    }

    // sdk_tpm, with the PDG code pdgp after the charge is changed
    complex tpm(int pdg, int hel, int ifsign, int pdgp) const {
        int x = s_pdg(pdg, hel, ifsign);
        if (left_neutrino(x) || left_lepton(x) || left_up(x) || left_down(x)) {
            return -fsign(1., ifsign * pdg) / std::sqrt(2. * _sw2);
        }
        complex t = 0.;
        if (std::abs(x) == 251 && pdgp == 25) {
            t = fsign(1., x);
        }
        if (std::abs(x) == 251 && pdgp == 250) {
            t = -I;
        }
        if (t != 0.) {
            t = t / (2. * std::sqrt(_sw2)) * I;
            if (pdgp == 25 && pdg * ifsign == 251) {
                t = -t;
            }
            if (pdgp == 250 && pdg * ifsign == -251) {
                t = -t;
            }
            return t;
        }
        if (std::abs(x) == 250 && std::abs(pdgp * ifsign) == 251) {
            t = I;
        }
        if (std::abs(x) == 25 && std::abs(pdgp * ifsign) == 251) {
            t = fsign(1., pdgp * ifsign);
        }
        if (t != 0.) {
            t = t / (2. * std::sqrt(_sw2)) * I;
            if (std::abs(x) == 25 && pdgp * ifsign == -251) {
                t = -t;
            }
            if (std::abs(x) == 250 && pdgp * ifsign == 251) {
                t = -t;
            }
            return t;
        }
        double ratio = std::sqrt(_cw2 / _sw2);
        double v = 0.;
        if (std::abs(pdg * ifsign) == 24 && pdgp == 22) {
            v = fsign(1., pdg * ifsign);
        }
        if (pdg == 22 && std::abs(pdgp * ifsign) == 24) {
            v = fsign(1., pdgp * ifsign);
        }
        if (std::abs(pdg * ifsign) == 24 && pdgp == 23) {
            v = -fsign(1., pdg * ifsign) * ratio;
        }
        if (pdg == 23 && std::abs(pdgp * ifsign) == 24) {
            v = -fsign(1., pdgp * ifsign) * ratio;
        }
        if (pdg == 23 || pdgp == 23) {
            v = -v;
        }
        return v;
    }

    // sdk_t3_diag
    complex t3(int pdg, int hel, int ifsign) const {
        int x = s_pdg(pdg, hel, ifsign);
        double t = 0.;
        if (left_neutrino(x) || left_up(x)) {
            t = fsign(0.5, ifsign * pdg);
        }
        if (left_lepton(x) || left_down(x)) {
            t = -fsign(0.5, ifsign * pdg);
        }
        if (std::abs(x) == 251) {
            t = fsign(0.5, x);
        }
        if (std::abs(x) == 24) {
            t = fsign(1., ifsign * pdg);
        }
        complex c = -t;
        if (std::abs(x) == 251) {
            c *= I;
        }
        return c;
    }

    // sdk_iz_diag
    complex iz(int pdg, int hel, int ifsign) const {
        complex v = (t3(pdg, hel, ifsign) - _sw2 * charge(pdg, ifsign)) /
            std::sqrt(_sw2 * _cw2);
        v = -v;
        if (std::abs(pdg) == 251) {
            v /= I;
        }
        return v;
    }

    // sdk_iz_nondiag, with the PDG code after the Higgs and neutral Goldstone
    // boson are exchanged
    complex iz_nondiag(int pdg, int ifsign) const {
        complex v = 0.;
        if (pdg == 25) {
            v = -0.5 * I * double(ifsign);
        }
        if (pdg == 250) {
            v = 0.5 * I * double(ifsign);
        }
        return -v / std::sqrt(_sw2 * _cw2);
    }

    // sdk_ia_diag, without the factor i of the charge of a charged Goldstone
    // boson (as sdk_iz_diag): it couples to the photon as its W boson
    complex ia(int pdg, int ifsign) const {
        if (_sud_mod == 1) {
            return 0.;
        }
        complex v = -charge(pdg, ifsign);
        if (std::abs(pdg) == 251) {
            v /= I;
        }
        return v;
    }

    // sdk_iz2_diag
    double iz2(int pdg, int hel, int ifsign) const {
        int x = s_pdg(pdg, hel, ifsign);
        double sw2 = _sw2, cw2 = _cw2;
        if (left_neutrino(x)) {
            return 1. / (4. * sw2 * cw2);
        }
        if (any_of(x, {11, 13, 15})) {
            return sw2 / cw2;
        }
        if (left_lepton(x)) {
            return (cw2 - sw2) * (cw2 - sw2) / (4. * sw2 * cw2);
        }
        if (any_of(x, {2, 4, 6})) {
            return 4. * sw2 / (9. * cw2);
        }
        if (left_up(x)) {
            return (3. * cw2 - sw2) * (3. * cw2 - sw2) / (36. * sw2 * cw2);
        }
        if (any_of(x, {1, 3, 5})) {
            return sw2 / (9. * cw2);
        }
        if (left_down(x)) {
            return (3. * cw2 + sw2) * (3. * cw2 + sw2) / (36. * sw2 * cw2);
        }
        if (std::abs(x) == 251) {
            return (cw2 - sw2) * (cw2 - sw2) / (4. * sw2 * cw2);
        }
        if (std::abs(x) == 250 || std::abs(x) == 25) {
            return 1. / (4. * sw2 * cw2);
        }
        if (std::abs(x) == 24) {
            return cw2 / sw2;
        }
        return 0.;
    }

    // sdk_cew_diag
    complex cew(int pdg, int hel, int ifsign) const {
        int x = s_pdg(pdg, hel, ifsign);
        double sw2 = _sw2, cw2 = _cw2, c = 0.;
        if (left_neutrino(x) || left_lepton(x)) {
            c = (1. + 2. * cw2) / (4. * sw2 * cw2);
        }
        if (any_of(x, {11, 13, 15})) {
            c = 1. / cw2;
        }
        if (any_of(x, {2, 4, 6})) {
            c = 4. / (9. * cw2);
        }
        if (left_up(x) || left_down(x)) {
            c = (sw2 + 27. * cw2) / (36. * sw2 * cw2);
        }
        if (any_of(x, {1, 3, 5})) {
            c = 1. / (9. * cw2);
        }
        if (any_of(std::abs(x), {251, 250, 25})) {
            c = (1. + 2. * cw2) / (4. * sw2 * cw2);
        }
        if (std::abs(x) == 24) {
            c = 2. / sw2;
        }
        if (std::abs(x) == 23) {
            c = 2. * cw2 / sw2;
        }
        if (std::abs(x) == 22) {
            c = 2.;
        }
        complex v = c;
        if (_sud_mod == 1) {
            v -= charge_squared(pdg, ifsign);
        }
        return v;
    }

    // sdk_cew_nondiag, the mixing of the Z boson and the photon
    double cew_nondiag() const { return 2. / _sw2 * std::sqrt(_cw2 * _sw2); }

    // sdk_betaew_diag
    double beta(int pdg) const {
        if (std::abs(pdg) == 24) {
            return 19. / 6. / _sw2 - (_sud_mod == 1 ? 11. / 3. : 0.);
        }
        if (pdg == 23) {
            return (19. - 38. * _sw2 - 22. * _sw2 * _sw2) / 6. / _sw2 / _cw2;
        }
        if (pdg == 22) {
            return -11. / 3. + (_sud_mod == 1 ? 80. / 9. : 0.);
        }
        return 0.;
    }

    // sdk_betaew_nondiag
    double beta_nondiag() const {
        return (19. + 22. * _sw2) / 6. / std::sqrt(_sw2 * _cw2);
    }

private:
    double _cw2, _sw2;
    int _sud_mod;
};

// the coefficients over the logarithms: L, l, then F and G of every pair
struct Coefficients {
    std::vector<complex> values;

    explicit Coefficients(std::size_t pairs) : values(2 + 2 * pairs, 0.) {}
    complex& big_l() { return values.at(0); }
    complex& small_l() { return values.at(1); }
    complex& f(std::size_t pair) { return values.at(2 + 2 * pair); }
    complex& g(std::size_t pair) { return values.at(3 + 2 * pair); }
    bool zero() const {
        return std::all_of(values.begin(), values.end(), [](complex v) {
            return v == 0.;
        });
    }
    // the real and imaginary parts of each, one after the other
    std::vector<double> table() const {
        std::vector<double> t;
        for (auto v : values) {
            t.push_back(v.real());
            t.push_back(v.imag());
        }
        return t;
    }
};

// the position of the pair of particles i and j among the pairs (j < i
// first, as the loops of ewsudakov_functions.f)
std::size_t pair_index(std::size_t i, std::size_t j) {
    if (i < j) {
        std::swap(i, j);
    }
    return i * (i - 1) / 2 + j;
}

// the isospin partner of a fermion, as get_isopart_mass_from_id
int isospin_partner(int pdg) {
    int a = std::abs(pdg);
    return a % 2 == 0 ? a - 1 : a + 1;
}

void check_process(const EWSudakovProcess& process) {
    std::size_t n = process.pdgs.size();
    if (process.incoming != 2 || n < 3) {
        throw std::invalid_argument(
            "EWSudakov: the Born needs two incoming particles and at least one outgoing"
        );
    }
    if (process.helicities.empty()) {
        throw std::invalid_argument("EWSudakov: no helicity combinations");
    }
    for (const auto& hel : process.helicities) {
        if (hel.size() != n) {
            throw std::invalid_argument(
                "EWSudakov: a helicity combination does not fit the Born"
            );
        }
    }
    auto check = [&](const EWSudakovAmplitude& amp, bool goldstone) {
        std::size_t legs = amp.legs.size();
        bool fits = goldstone ? amp.type == "goldstone" && legs >= 1
            : (amp.type == "cew" || amp.type == "iz1") ? legs == 1
            : (amp.type == "iz2" || amp.type == "ipm2")
            ? legs == 2
            : false;
        fits = fits && amp.old_pdgs.size() == legs && amp.new_pdgs.size() == legs &&
            amp.base <= (goldstone ? 0 : process.goldstone.size());
        for (std::size_t k = 0; fits && k < legs; ++k) {
            fits = amp.legs.at(k) < n;
        }
        if (!fits) {
            throw std::invalid_argument(
                "EWSudakov: an amplitude of type '" + amp.type +
                "' does not fit the Born"
            );
        }
    };
    for (const auto& amp : process.goldstone) {
        check(amp, true);
    }
    for (const auto& amp : process.interference) {
        check(amp, false);
    }
}

// a table of three indices as a constant, first index fastest as those of
// nested vectors
Value table(const nested_vector3<double>& values) {
    std::size_t a = values.size(), b = values.at(0).size();
    std::size_t c = values.at(0).at(0).size();
    std::vector<double> flat(a * b * c);
    for (std::size_t i = 0; i < a; ++i) {
        for (std::size_t j = 0; j < b; ++j) {
            for (std::size_t k = 0; k < c; ++k) {
                flat.at(i + a * (j + b * k)) = values.at(i).at(j).at(k);
            }
        }
    }
    return Value(flat, {static_cast<int>(a), static_cast<int>(b), static_cast<int>(c)});
}

// the tables of EWSudakov
struct Tables {
    std::vector<double> signs, par, qcd, options;
    std::vector<std::size_t> bases;
    std::vector<std::pair<std::size_t, std::size_t>> terms;
    nested_vector2<me_int_t> modes;
    nested_vector3<double> diagonal, interference;
};

Tables tables(
    const EWSudakovProcess& process,
    const EWSudakovParameters& parameters,
    const std::vector<EWSudakovMode>& modes
) {
    check_process(process);
    if (modes.empty()) {
        throw std::invalid_argument("EWSudakov: no modes");
    }
    Tables r;
    const auto& p = parameters;
    if (!(p.mw > 0. && p.mz > p.mw && p.mh > 0. && p.mt > 0. && p.mb >= 0. &&
          p.alpha > 0. && (p.scheme == EWSudakovScheme::alpha_mz || p.gf > 0.))) {
        throw std::invalid_argument(
            "EWSudakov: the masses and couplings must be positive, with mz > mw"
        );
    }
    for (const auto& mode : modes) {
        if (mode.sud_mod != 0 && mode.sud_mod != 1) {
            throw std::invalid_argument("EWSudakov: sud_mod must be 0 or 1");
        }
        r.modes.push_back({mode.s_to_rij ? 1 : 0, mode.rij_ge_mw ? 1 : 0});
    }
    std::size_t n = process.pdgs.size();
    std::size_t pairs = n * (n - 1) / 2;
    if (n > 10) {
        throw std::invalid_argument("EWSudakov: at most ten particles");
    }
    for (std::size_t i = 0; i < n; ++i) {
        r.signs.push_back(i < process.incoming ? -1. : 1.);
    }
    auto ifsign = [&](std::size_t i) { return i < process.incoming ? -1 : 1; };
    auto mass = [&](int pdg) {
        int a = std::abs(pdg);
        return a == 6 ? p.mt : a == 5 ? p.mb : 0.;
    };

    // the base amplitude and its PDG codes for each helicity combination
    std::vector<std::vector<int>> base_pdgs{process.pdgs};
    for (const auto& gold : process.goldstone) {
        auto pdgs = process.pdgs;
        for (std::size_t k = 0; k < gold.legs.size(); ++k) {
            pdgs.at(gold.legs.at(k)) = gold.new_pdgs.at(k);
        }
        base_pdgs.push_back(pdgs);
    }
    for (const auto& hel : process.helicities) {
        std::size_t base = 0;
        for (std::size_t k = process.goldstone.size(); k > 0; --k) {
            const auto& legs = process.goldstone.at(k - 1).legs;
            if (std::all_of(legs.begin(), legs.end(), [&](std::size_t leg) {
                    return hel.at(leg) == 0;
                })) {
                base = k;
            }
        }
        r.bases.push_back(base);
    }

    double mw2 = p.mw * p.mw, mz2 = p.mz * p.mz;
    double lzow = std::log(mz2 / mw2);
    std::vector<std::pair<std::size_t, std::size_t>> terms;
    std::vector<std::vector<Coefficients>> term_coefficients;
    for (std::size_t m = 0; m < modes.size(); ++m) {
        QuantumNumbers q(p.mw, p.mz, modes.at(m).sud_mod);
        double sw2 = q.sw2();
        nested_vector2<double> diagonal;
        std::vector<Coefficients> by_term;
        for (std::size_t h = 0; h < process.helicities.size(); ++h) {
            const auto& hel = process.helicities.at(h);
            const auto& pdgs = base_pdgs.at(r.bases.at(h));
            Coefficients c(pairs);
            if (process.has_lo1) {
                for (std::size_t i = 0; i < n; ++i) {
                    int pdg = pdgs.at(i), s = ifsign(i);
                    // get_lsc_diag
                    c.big_l() += -0.5 * q.cew(pdg, hel.at(i), s);
                    c.small_l() += lzow * q.iz2(pdg, hel.at(i), s);
                    // get_ssc_n_diag
                    for (std::size_t j = 0; j < i; ++j) {
                        int pdg_j = pdgs.at(j), s_j = ifsign(j);
                        std::size_t k = pair_index(i, j);
                        complex a = q.ia(pdg, s) * q.ia(pdg_j, s_j);
                        complex z =
                            q.iz(pdg, hel.at(i), s) * q.iz(pdg_j, hel.at(j), s_j);
                        c.f(k) += a + z;
                        c.g(k) += z;
                    }
                    // get_xxc_diag
                    int a = std::abs(pdg);
                    if (a <= 6 || (a >= 11 && a <= 16)) {
                        double m_f = mass(pdg), m_iso = mass(isospin_partner(pdg));
                        double m_y = pdg * hel.at(i) < 0 ? m_iso : m_f;
                        c.small_l() += 1.5 * q.cew(pdg, hel.at(i), s) -
                            (m_f * m_f + m_y * m_y) / (8. * sw2 * mw2);
                    } else if (a >= 22 && a <= 24 && hel.at(i) != 0) {
                        c.small_l() += q.beta(pdg) / 2.;
                    } else if (a == 250 || a == 251 || pdg == 25) {
                        c.small_l() += 2. * q.cew(pdg, hel.at(i), s) -
                            3. / 4. / sw2 * p.mt * p.mt / mw2;
                    }
                }
            }
            diagonal.push_back(c.table());

            for (std::size_t t = 0; t < process.interference.size(); ++t) {
                const auto& amp = process.interference.at(t);
                if (amp.base != r.bases.at(h) || !process.has_lo1) {
                    continue;
                }
                Coefficients ct(pairs);
                std::size_t i0 = amp.legs.at(0);
                auto higgs_pair = [](int old_pdg, int new_pdg) {
                    return (old_pdg == 25 && new_pdg == 250) ||
                        (old_pdg == 250 && new_pdg == 25);
                };
                if (amp.type == "cew") {
                    // get_lsc_nondiag and get_xxc_nondiag
                    int old_pdg = amp.old_pdgs.at(0), new_pdg = amp.new_pdgs.at(0);
                    bool mixing = (old_pdg == 23 && new_pdg == 22) ||
                        (old_pdg == 22 && new_pdg == 23);
                    if (std::abs(hel.at(i0)) == 1 && mixing) {
                        ct.big_l() += -0.5 * q.cew_nondiag();
                        if (old_pdg == 23) {
                            ct.small_l() += q.beta_nondiag();
                        }
                    }
                } else if (amp.type == "iz1") {
                    // get_ssc_n_nondiag_1
                    if (higgs_pair(amp.old_pdgs.at(0), amp.new_pdgs.at(0))) {
                        complex nd = q.iz_nondiag(amp.new_pdgs.at(0), ifsign(i0));
                        for (std::size_t i = 0; i < n; ++i) {
                            if (i == i0) {
                                continue;
                            }
                            complex z = q.iz(pdgs.at(i), hel.at(i), ifsign(i)) * nd;
                            ct.f(pair_index(i, i0)) += z;
                            ct.g(pair_index(i, i0)) += z;
                        }
                    }
                } else if (amp.type == "iz2") {
                    // get_ssc_n_nondiag_2
                    std::size_t i1 = amp.legs.at(1);
                    if (higgs_pair(amp.old_pdgs.at(0), amp.new_pdgs.at(0)) &&
                        higgs_pair(amp.old_pdgs.at(1), amp.new_pdgs.at(1))) {
                        complex z = q.iz_nondiag(amp.new_pdgs.at(0), ifsign(i0)) *
                            q.iz_nondiag(amp.new_pdgs.at(1), ifsign(i1));
                        ct.f(pair_index(i0, i1)) += z;
                        ct.g(pair_index(i0, i1)) += z;
                    }
                } else {
                    // get_ssc_c
                    std::size_t i1 = amp.legs.at(1);
                    ct.f(pair_index(i0, i1)) +=
                        q.tpm(pdgs.at(i0), hel.at(i0), ifsign(i0), amp.new_pdgs.at(0)) *
                        q.tpm(pdgs.at(i1), hel.at(i1), ifsign(i1), amp.new_pdgs.at(1));
                }
                if (m == 0) {
                    terms.push_back({h, t});
                }
                by_term.push_back(ct);
            }
        }
        r.diagonal.push_back(diagonal);
        term_coefficients.push_back(by_term);
    }
    // the interference terms with a coefficient in a mode
    for (std::size_t k = 0; k < terms.size(); ++k) {
        bool zero = true;
        for (const auto& by_term : term_coefficients) {
            zero = zero && by_term.at(k).zero();
        }
        if (!zero) {
            r.terms.push_back(terms.at(k));
        }
    }
    r.interference.assign(modes.size(), {});
    for (std::size_t m = 0; m < modes.size(); ++m) {
        for (std::size_t k = 0; k < terms.size(); ++k) {
            bool kept =
                std::find(r.terms.begin(), r.terms.end(), terms.at(k)) != r.terms.end();
            if (kept) {
                r.interference.at(m).push_back(term_coefficients.at(m).at(k).table());
            }
        }
        if (r.terms.empty()) {
            r.interference.at(m).push_back(Coefficients(pairs).table());
        }
    }
    // the parameter renormalisation, with sud_mod 0 (get_par_ren_alphamz and
    // get_par_ren_gmu), per unit l and with the factor 2 of compute_ewsudakov
    r.par.assign(6, 0.);
    if (process.has_lo1) {
        QuantumNumbers q(p.mw, p.mz, 0);
        double sw2 = q.sw2(), cw2 = q.cw2();
        double mt = p.mt, mh = p.mh, mw = p.mw, mz = p.mz;
        double aewm1 = 1. / p.alpha, ee = std::sqrt(4. * PI * p.alpha);
        double dalpha = -q.beta(22) / aewm1;
        double cew = q.cew(250, 0, 1).real();
        double top = 3. * mt * mt / 2. / mw2 / sw2;
        double dmw2 = (-(q.beta(24) - 4. * cew) - top) * mw2;
        double dmz2 = (-(q.beta(23) - 4. * cew) - top) * mz2;
        double qt = 2. / 3.;
        double dmt =
            (1. / 4. / sw2 + 1. / 8. / sw2 / cw2 + 3. / 2. / cw2 * qt -
             3. / cw2 * qt * qt + 3. / 8. / sw2 * mt * mt / mw2) *
            mt;
        double dmh2 =
            (1. / 2. / sw2 *
                 (9. * mw2 / (mh * mh) * (1. + 1. / 2. / (cw2 * cw2)) -
                  3. / 2. * (1. + 1. / 2. / cw2) + 15. / 4. * mh * mh / mw2) +
             3. / 2. / sw2 * mt * mt / mw2 * (1. - 6. * mt * mt / (mh * mh))) *
            mh * mh;
        double dt = 1. / (ee * std::sqrt(sw2) * mw) *
            (-3. / 2. * mw2 * (mz2 / cw2 + 2. * mw2) -
             mh * mh / 4. * (2. * mw2 + mz2 + 3. * mh * mh) + 6. * std::pow(mt, 4));
        double dheff = ee / (2. * std::sqrt(sw2)) * dt / (mw * mh * mh);
        if (p.scheme == EWSudakovScheme::gmu) {
            r.par.at(0) = p.gf * aewm1 * dalpha +
                p.gf * (-mw2 / (mz2 * mz2) / (1. - mw2 / mz2)) * dmz2 +
                p.gf * (-1. / mw2 + 1. / mz2 / (1. - mw2 / mz2)) * dmw2;
        } else {
            r.par.at(0) = -aewm1 * aewm1 * dalpha;
        }
        r.par.at(1) = dmz2 / (2. * mz);
        r.par.at(2) = dmw2 / (2. * mw);
        r.par.at(3) = dmt;
        r.par.at(4) = dmh2 / (2. * mh);
        r.par.at(5) = dheff;
    }
    // GET_QCD_LO2 with sud_mod 0 or 1, per unit alpha_s / (2 pi): the
    // factors of log(mu_R^2 / m_t^2) and log(s / m_t^2)
    r.qcd.assign(2, 0.);
    if (process.has_lo2) {
        double gluons =
            std::count_if(process.pdgs.begin(), process.pdgs.end(), [](int x) {
                return std::abs(x) == 21;
            });
        r.qcd.at(0) = process.lo2_qcd_power / 12.;
        r.qcd.at(1) = -gluons / 6.;
    }
    r.options = {mw2, mz2, p.mt * p.mt, p.alpha / (4. * PI)};
    return r;
}

} // namespace

EWSudakov::EWSudakov(
    const EWSudakovProcess& process,
    const EWSudakovParameters& parameters,
    const std::vector<EWSudakovMode>& modes
) :
    FunctionGenerator(
        "EWSudakov",
        [&] {
            auto n = static_cast<int>(process.pdgs.size());
            auto h = static_cast<int>(process.helicities.size());
            auto t = static_cast<int>(std::max<std::size_t>(
                tables(process, parameters, modes).terms.size(), 1
            ));
            return NamedVector<Type>{
                {"momenta", batch_four_vec_array(n)},
                {"diagonal", batch_float_array(h)},
                {"lo2", batch_float_array(h)},
                {"derivatives", Type{DataType::dt_float64, batch_size, {h, 6}}},
                {"interference", Type{DataType::dt_float64, batch_size, {t, 2}}},
                {"alpha_s", batch_float},
                {"ren_scale", batch_float},
            };
        }(),
        NamedVector<Type>{
            {"sudakov", batch_float_array(static_cast<int>(modes.size()))}
        }
    ) {
    auto r = tables(process, parameters, modes);
    _signs = r.signs;
    _par = r.par;
    _qcd = r.qcd;
    _options = r.options;
    _bases = r.bases;
    _terms = r.terms;
    _modes = r.modes;
    _diagonal = r.diagonal;
    _interference = r.interference;
}

NamedVector<Value> EWSudakov::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto sudakov = fb.ew_sudakov(
        args.at("momenta"),
        args.at("diagonal"),
        args.at("lo2"),
        args.at("derivatives"),
        args.at("interference"),
        args.at("alpha_s"),
        args.at("ren_scale"),
        _signs,
        _modes,
        table(_diagonal),
        table(_interference),
        _par,
        _qcd,
        _options
    );
    return {{"sudakov", sudakov}};
}
