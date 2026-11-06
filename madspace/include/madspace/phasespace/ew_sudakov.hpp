#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/util.hpp"

namespace madspace {

/**
 * An amplitude of the Sudakov approximation of the EW corrections with
 * particles of the Born replaced, as get_ewsudakov_info of MadGraph's
 * export_fks.py lists them in fks_configurations.json.
 */
struct EWSudakovAmplitude {
    /// Kind: "goldstone" for Goldstone bosons in place of longitudinal W and
    /// Z bosons, or of an interference "cew" (a Z boson and a photon), "iz1"
    /// (a Higgs boson and a neutral Goldstone boson), "iz2" (two of them) or
    /// "ipm2" (two particles with their charge changed).
    std::string type;
    /// The base amplitude of an interference: 0 for the Born, k for the
    /// Goldstone amplitude k (1-based). Zero for a Goldstone amplitude.
    std::size_t base = 0;
    /// The replaced particles, positions in the Born (0-based).
    std::vector<std::size_t> legs;
    /// PDG codes of the replaced particles before and after.
    std::vector<int> old_pdgs, new_pdgs;
};

/**
 * The process of the Sudakov approximation of the EW corrections: the Born
 * and the amplitudes that ewsudakov_wrapper.f of MadGraph takes.
 */
struct EWSudakovProcess {
    /// PDG codes of the Born, the incoming first.
    std::vector<int> pdgs;
    /// Number of incoming particles.
    std::size_t incoming = 2;
    /// Helicities of the particles of the Born, per helicity combination (as
    /// NHEL of the Born).
    nested_vector2<int> helicities;
    /// The Goldstone amplitudes, in their order. A helicity combination takes
    /// the first one whose replaced particles all have helicity 0, the Born
    /// where none has.
    std::vector<EWSudakovAmplitude> goldstone;
    /// The interferences, in their order.
    std::vector<EWSudakovAmplitude> interference;
    /// Whether the Born has a LO1 (its only coupling combination within its
    /// orders) and a LO2 (two powers of QCD less, two of QED more).
    bool has_lo1 = true, has_lo2 = false;
    /// The power of the QCD coupling (g) of the LO2.
    int lo2_qcd_power = 0;
};

/// The scheme of the parameter renormalisation.
enum class EWSudakovScheme { alpha_mz, gmu };

/**
 * The parameters of the Sudakov approximation of the EW corrections: masses,
 * couplings and the scheme of the parameter renormalisation (numder_*.f of
 * MadGraph, alpha_mz for models without Gmu in their names).
 */
struct EWSudakovParameters {
    double mw = 80.419, mz = 91.188, mh = 125., mt = 173.3, mb = 0.;
    /// The QED coupling, @f$1 / \alpha@f$ the aEWM1 of the model and
    /// @f$4 \pi \alpha@f$ the square of the coupling of the logarithms
    /// (gal(1)).
    double alpha = 1. / 132.50698;
    /// The Fermi constant, for the Gmu scheme.
    double gf = 1.16639e-5;
    EWSudakovScheme scheme = EWSudakovScheme::gmu;
};

/**
 * A mode of the Sudakov approximation, as the switches of
 * ewsudakov_functions.f.
 */
struct EWSudakovMode {
    /// sud_mod: 0 drops the logarithms of the photon mass and of the masses of
    /// light charged particles, 1 drops the QED contributions too (compute_ewsudakov
    /// of fks_singular.f takes both).
    int sud_mod = 0;
    /// s_to_rij: the double logarithms of s are taken at the invariants of the
    /// pairs.
    bool s_to_rij = true;
    /// rij_ge_mw: invariants smaller than @f$M_W^2@f$ in size are set to it.
    bool rij_ge_mw = true;
};

/**
 * The Sudakov approximation of the EW corrections to a Born [1, 2, 3], as
 * ewsudakov_wrapper.f and ewsudakov_functions.f of MadGraph, for each of its
 * modes per event at the NLO order of the LO1 with two powers of QED more.
 * It takes the matrix elements of the Born per helicity combination @f$h@f$,
 * all averaged and with the identical-particle factor of the Born:
 *
 * - `diagonal`: @f$|M_h|^2@f$ of the base amplitude of @f$h@f$ (the Born, or
 *   the Goldstone amplitude for longitudinal W and Z bosons) at the LO1,
 * - `lo2`: the same at the LO2,
 * - `derivatives`: the derivatives of `diagonal` with respect to the
 *   parameters of numder_*.f (1 / alpha in alpha_mz or the Fermi constant
 *   in gmu, @f$M_Z@f$, @f$M_W@f$, @f$m_t@f$ with the Yukawa coupling,
 *   @f$M_H@f$ and the tadpole switch),
 * - `interference`: the real and imaginary parts of the interferences
 *   @f$M'_h M_h^*@f$ of the base amplitude with those of the interference
 *   amplitudes of its base (at momenta reshuffled to the masses of the
 *   replaced particles, momentum_reshuffling.f) at the LO1, per interference
 *   term (see interference_terms()),
 *
 * and the strong coupling and the renormalization scale of the LO2 part.
 *
 * With @f$a = \alpha / (4 \pi)@f$, @f$l = a \log(s / M_W^2)@f$ and
 * @f$L = a \log^2(s / M_W^2)@f$, and for every pair of particles the
 * invariant @f$r_{ij} = (p_i \pm p_j)^2@f$ (minus for one incoming), with
 * @f$\ell_{ij} = \log(|r_{ij}| / s)@f$ and @f$i \pi_{ij} = -i \pi@f$ for
 * @f$r_{ij} > 0@f$, zero otherwise,
 * @f[
 *   F_{ij} = 2 l (\ell_{ij} + i \pi_{ij})
 *     + a (\ell_{ij}^2 + 2 \ell_{ij} i \pi_{ij}), \qquad
 *   G_{ij} = 2 a \ell_{ij} \log(M_W^2 / M_Z^2)
 * @f]
 * (the second terms only with s_to_rij), the coefficients of
 * ewsudakov_functions.f of every helicity combination (get_lsc_diag,
 * get_ssc_n_diag, get_xxc_diag) and of every interference term
 * (get_lsc_nondiag and get_xxc_nondiag, get_ssc_n_nondiag_1,
 * get_ssc_n_nondiag_2, get_ssc_c) are linear combinations of @f$L@f$,
 * @f$l@f$, @f$F_{ij}@f$ and @f$G_{ij}@f$ with constant complex factors.
 * The correction is
 * @f[
 *   2 \Re \Big[\sum_h C_h |M_h|^2 + \sum_t C_t M'_t M_t^*\Big]
 *   + l \sum_h \sum_p \kappa_p \partial_p |M_h|^2
 *   + 2 Q \sum_h |M_h|^2_\mathrm{LO2}
 * @f]
 * with the parameter renormalisation @f$\kappa_p@f$ of get_par_ren_alphamz
 * and get_par_ren_gmu (with sud_mod 0 for every mode) and
 * @f$Q = \alpha_s / (2 \pi) [k / 12 \log(\mu_R^2 / m_t^2) - n_g / 6
 * \log(s / m_t^2)]@f$ of GET_QCD_LO2 with the QCD power @f$k@f$ of the LO2
 * and the number @f$n_g@f$ of gluons of the Born. The first two terms need a
 * LO1, the third a LO2. The correction is that of compute_ewsudakov in
 * fks_singular.f without the factor @f$f_b / g^k@f$ of its weight lines.
 *
 * References:
 * - [1] A. Denner, S. Pozzorini, arXiv:hep-ph/0010201
 * - [2] D. Pagani, M. Zaro, arXiv:2110.03714
 * - [3] D. Pagani, T. Vitos, M. Zaro, arXiv:2309.00452
 */
class EWSudakov : public FunctionGenerator {
public:
    /**
     * @param process     The process.
     * @param parameters  The parameters.
     * @param modes       The modes of the outputs, by default the two of
     *                    compute_ewsudakov.
     * @throws std::invalid_argument for an amplitude or helicity combination
     *         that does not fit the Born, a mode with sud_mod other than 0 and
     *         1, no modes or parameters out of their range.
     */
    EWSudakov(
        const EWSudakovProcess& process,
        const EWSudakovParameters& parameters,
        const std::vector<EWSudakovMode>& modes = {{0, true, true}, {1, true, true}}
    );
    /// Number of particles of the Born.
    std::size_t particle_count() const { return _signs.size(); }
    /// Number of helicity combinations.
    std::size_t helicity_count() const { return _bases.size(); }
    /// Number of modes.
    std::size_t mode_count() const { return _modes.size(); }
    /// The base amplitude of each helicity combination, 0 for the Born and k
    /// for the Goldstone amplitude k.
    const std::vector<std::size_t>& helicity_bases() const { return _bases; }
    /**
     * The interference terms, the entries of the input `interference`: the
     * helicity combination and the interference amplitude (positions in the
     * helicities and the interferences of the process), for the
     * interferences of the base amplitude of each helicity combination with a
     * coefficient. Without any, `interference` has one entry, which is not
     * used.
     */
    const std::vector<std::pair<std::size_t, std::size_t>>& interference_terms() const {
        return _terms;
    }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<double> _signs, _par, _qcd, _options;
    std::vector<std::size_t> _bases;
    std::vector<std::pair<std::size_t, std::size_t>> _terms;
    nested_vector2<me_int_t> _modes;
    nested_vector3<double> _diagonal, _interference;
};

} // namespace madspace
