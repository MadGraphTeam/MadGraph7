#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * The generation cuts of an NLO integrand on one kinematics, those of cuts.f
 * of MadFKS (passcuts_leptons and passcuts_jets, with the parameters of its
 * run card) [2]. The kinematics has one of several flavor layouts, the PDG
 * codes of its particles, chosen per event, so that a batch can mix the real
 * processes of several FKS configurations. The cuts on jets are infrared
 * safe: the QCD partons (quarks up to `maxjetflavor` and gluons) are
 * clustered with the pp generalized @f$k_T@f$ algorithm of fastjet [3] (power
 * `jetalgo`, 1 for @f$k_T@f$, 0 for Cambridge/Aachen, -1 for anti-@f$k_T@f$,
 * radius `jetradius`, rapidity and azimuth, E scheme) into jets with
 * @f$p_T \ge@f$ `ptj` and @f$|\eta| \le@f$ `etaj`. With `ptj` > 0 and more
 * partons than `unresolved`, at most `unresolved` partons may be soft and all
 * partons but at most `unresolved` must make jets, so that a soft or
 * collinear emission never decides: one for a real emission and its
 * counter-events, as passcuts_jets, and none for a Born. The charged leptons
 * pass `ptl` and `etal`, each pair of opposite charge `drll` and `mll`, and
 * the pairs of the same flavor `drll_sf` and `mll_sf`. A bound of zero (or
 * below for `etaj` and `etal`) is no cut. The cuts on photons, the photon
 * recombination and the cuts per PDG code of cuts.f are not part of it.
 * @ref FKSCrossSection applies them to the event and the counter-events of the
 * FKS integrand.
 *
 * `batch` is the leading batch dimension and `n` the number of particles of
 * the layouts, incoming included.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – the kinematics in the lab
 *   frame, in the order of its layout.
 * - `layout_index` – `int`, shape `(batch,)` – the flavor layout of the
 *   kinematics.
 *
 * **Returns**
 * - `mask` – `float`, shape `(batch,)` – 1 where the kinematics passes the
 *   cuts, else 0.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 * - [3] M. Cacciari, G. P. Salam, G. Soyez, "FastJet user manual",
 *   https://arxiv.org/abs/1111.6097
 */
class NLOCuts : public FunctionGenerator {
public:
    /**
     * @param layouts       The flavor layouts, the PDG codes of the particles
     *                      of each, incoming included, all of the same length.
     * @param incoming      Number of incoming particles, two or one for a
     *                      decay.
     * @param unresolved    Number of partons that may be unresolved, one for
     *                      a real emission and its counter-events and none
     *                      for a Born.
     * @param jetalgo       Power of the generalized @f$k_T@f$ algorithm.
     * @param jetradius     Its radius.
     * @param ptj           Least transverse momentum of a jet.
     * @param etaj          Largest @f$|\eta|@f$ of a jet.
     * @param maxjetflavor  Heaviest quark flavor of a jet.
     * @param ptl           Least transverse momentum of a charged lepton.
     * @param etal          Largest @f$|\eta|@f$ of a charged lepton.
     * @param drll          Least @f$\Delta R@f$ of a pair of opposite charge.
     * @param drll_sf       The same for a pair of the same flavor.
     * @param mll           Least invariant mass of a pair of opposite charge.
     * @param mll_sf        The same for a pair of the same flavor.
     * @throws std::invalid_argument without layouts, for layouts of different
     *         lengths or without an outgoing particle, for another number of
     *         incoming particles or for more QCD partons than the clustering
     *         takes.
     */
    NLOCuts(
        const nested_vector2<int>& layouts,
        std::size_t incoming = 2,
        std::size_t unresolved = 1,
        double jetalgo = 1.,
        double jetradius = 0.7,
        double ptj = 0.,
        double etaj = -1.,
        int maxjetflavor = 4,
        double ptl = 0.,
        double etal = -1.,
        double drll = 0.,
        double drll_sf = 0.,
        double mll = 0.,
        double mll_sf = 0.
    );
    /// Most QCD partons of a layout the clustering takes
    /// (NLO_JET_PARTONS_MAX of the kernel).
    static constexpr std::size_t jet_parton_max = 16;
    /// Number of particles of the layouts, incoming included.
    std::size_t particle_count() const { return _particle_count; }
    /// Number of incoming particles, one for a decay.
    std::size_t incoming() const { return _incoming; }
    /// Number of partons that may be unresolved.
    std::size_t unresolved() const { return _unresolved; }
    /// The flavor layouts.
    const nested_vector2<me_int_t>& layouts() const { return _layouts; }
    /// The options of the kernel: jetalgo, jetradius, ptj, etaj,
    /// maxjetflavor, ptl, etal, drll, drll_sf, mll, mll_sf and unresolved.
    const std::vector<double>& options() const { return _options; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _particle_count;
    std::size_t _incoming;
    std::size_t _unresolved;
    nested_vector2<me_int_t> _layouts;
    std::vector<double> _options;
};

} // namespace madspace
