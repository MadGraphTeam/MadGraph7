#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * The QCD partons of one kinematics of an NLO integrand clustered into jets,
 * for the observables of its histograms. The kinematics has one of several
 * flavor layouts, chosen per event. The momenta it returns hold the incoming
 * particles and the other final-state particles in their order, then the jets
 * by decreasing @f$p_T@f$, and zero momenta in the slots without a jet. The
 * other final-state particles are the same in every layout, so that one
 * @ref Observable built on @ref pids serves every layout, and kinematics with
 * fewer partons (a Born next to its real emission) take the same slots with
 * @p jet_slots. The QCD partons are the quarks up to `maxjetflavor` and the
 * gluons, clustered with the pp generalized @f$k_T@f$ algorithm of fastjet
 * [2] (power `jetalgo`, radius `jetradius`, E scheme), as @ref NLOCuts
 * clusters them, into jets with @f$p_T \geq@f$ `ptj` and
 * @f$|\eta| \leq@f$ `etaj` (`etaj` @f$\leq 0@f$ for none). Partons without
 * transverse momentum (the soft parton of a soft counter-event) make no jet.
 *
 * `batch` is the leading batch dimension, `n` the number of particles of the
 * layouts and `m` that of the momenta with jets.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – the kinematics in the lab
 *   frame, in the order of its layout.
 * - `layout_index` – `int`, shape `(batch,)` – the flavor layout of the
 *   kinematics.
 *
 * **Returns**
 * - `jet_momenta` – `float`, shape `(batch, m, 4)` – the momenta with jets.
 * - `jet_count` – `int`, shape `(batch,)` – the number of jets.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] M. Cacciari, G. P. Salam, G. Soyez, "FastJet user manual",
 *   https://arxiv.org/abs/1111.6097
 */
class NLOJets : public FunctionGenerator {
public:
    /**
     * @param layouts         The flavor layouts, the PDG codes of the
     *                        particles of each, incoming included, all of the
     *                        same length.
     * @param jet_slots       Number of slots for jets, at least the
     *                        final-state partons of each layout, zero for the
     *                        most of them. A Born next to its real emission
     *                        takes the slots of the real emission.
     * @param incoming        Number of incoming particles, two or one for a
     *                        decay.
     * @param jetalgo         Power of the generalized @f$k_T@f$ algorithm: 1
     *                        for @f$k_T@f$, 0 for Cambridge/Aachen, -1 for
     *                        anti-@f$k_T@f$.
     * @param jetradius       Radius of the jets.
     * @param ptj             Least @f$p_T@f$ of a jet.
     * @param etaj            Largest @f$|\eta|@f$ of a jet, none for
     *                        @f$\leq 0@f$.
     * @param maxjetflavor    Heaviest quark flavor in the jets.
     * @throws std::invalid_argument without layouts, for layouts of different
     *         lengths or that differ outside their QCD partons, for another
     *         number of incoming particles, for fewer slots than partons, for
     *         more partons than the clustering takes (@ref parton_max) or a
     *         radius that is not positive.
     */
    NLOJets(
        const nested_vector2<int>& layouts,
        std::size_t jet_slots = 0,
        std::size_t incoming = 2,
        double jetalgo = -1.,
        double jetradius = 0.4,
        double ptj = 0.,
        double etaj = -1.,
        int maxjetflavor = 4
    );
    /// Number of particles of the kinematics.
    std::size_t particle_count() const { return _particle_count; }
    /// Number of the momenta with jets, the other particles and the slots.
    std::size_t jet_particle_count() const { return _pids.size(); }
    /// Number of slots for jets.
    std::size_t slot_count() const { return _slot_count; }
    /// Number of incoming particles, one for a decay.
    std::size_t incoming() const { return _incoming; }
    /// The PDG codes of the momenta with jets: those of the incoming and the
    /// other final-state particles of the first layout, then 21 for each jet
    /// slot.
    const std::vector<int>& pids() const { return _pids; }
    /// jetalgo, jetradius, ptj and etaj.
    const std::vector<double>& options() const { return _options; }
    /// The most final-state partons of one kinematics, those of the
    /// clustering of @ref NLOCuts (NLOCuts::jet_parton_max).
    static constexpr std::size_t parton_max = 16;
    /**
     * The final-state QCD partons of a layout, quarks up to @p maxjetflavor
     * and gluons.
     *
     * @param layout        The PDG codes of the particles, incoming included.
     * @param incoming      Number of incoming particles.
     * @param maxjetflavor  Heaviest quark flavor in the jets.
     */
    static std::size_t parton_count(
        const std::vector<int>& layout, std::size_t incoming, int maxjetflavor
    );

private:
    // the particles of the kinematics, the jet slots, the PDG codes of the
    // momenta with jets and the partons of each layout
    struct Layout {
        std::size_t particle_count;
        std::size_t slot_count;
        std::vector<int> pids;
        nested_vector2<me_int_t> parton;
    };
    static Layout layout(
        const nested_vector2<int>& layouts,
        std::size_t jet_slots,
        std::size_t incoming,
        int maxjetflavor
    );
    NLOJets(Layout layout, std::size_t incoming, const std::vector<double>& options);
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _particle_count;
    std::size_t _slot_count;
    std::size_t _incoming;
    std::vector<int> _pids;
    nested_vector2<me_int_t> _parton;
    std::vector<double> _options;
};

} // namespace madspace
