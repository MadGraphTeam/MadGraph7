#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/phasespace/base.hpp"

#include <vector>

namespace madspace {

/**
 * Merging scale of an event, as the CKKW-L generation cut of madevent defines it.
 *
 * Evaluates one of the two merging-scale measures madevent cuts on before a
 * CKKW-L merged Pythia8 shower [1, 2] (Template/LO/SubProcesses/cuts.f):
 *
 * - `kt_durham`: the smallest pT to the beam (hadronic beams only) and pairwise
 *   kt of the merging legs, min(pT) sqrt(dy^2 + dphi^2) / D with the true
 *   rapidity y for hadronic beams, sqrt(2 min(E^2) (1 - cos theta)) otherwise.
 * - `pt_lund`: the smallest Pythia8 evolution pT over every ISR and FSR
 *   splitting the merging legs can come from. It depends on the leg flavours.
 *
 * Both are inactive (the scale is a huge number, so no cut ever vetoes the
 * event) when no merging leg is a massless parton, i.e. a gluon or a quark up
 * to the b. A ptlund event with exactly two outgoing legs, both massless
 * merging partons, is only measured by their pT, and not at all at a lepton
 * collider, as in madevent.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – external momenta, incoming first.
 *
 * **Returns**
 * - `scale` – `float`, shape `(batch,)` – the merging scale.
 *
 * **References**
 * - [1] L. Lonnblad, S. Prestel, "Matching Tree-Level Matrix Elements with
 *   Interleaved Showers", https://arxiv.org/abs/1109.4829
 * - [2] T. Sjostrand et al., "An introduction to PYTHIA 8.2",
 *   https://arxiv.org/abs/1410.3012
 */
class MergingScale : public FunctionGenerator {
public:
    /// Which merging-scale definition is evaluated.
    enum Measure { kt_durham, pt_lund };

    /**
     * @param measure         ktdurham or ptlund.
     * @param pdg_ids         Signed pdg id of every external leg, incoming first.
     * @param merging_legs    Whether each external leg is subject to the merging
     *                        cut; the incoming entries are ignored.
     * @param d_parameter     D of the hadronic kt measure.
     * @param hadronic1       Beam 1 is a hadron.
     * @param hadronic2       Beam 2 is a hadron.
     * @param max_jet_flavor  Heaviest quark counted as a jet, which with the b
     *                        quark decides which merging legs are massless.
     */
    MergingScale(
        Measure measure,
        const std::vector<int>& pdg_ids,
        const std::vector<bool>& merging_legs,
        double d_parameter = 0.4,
        bool hadronic1 = true,
        bool hadronic2 = true,
        int max_jet_flavor = 4
    );
    /// The merging-scale definition.
    Measure measure() const { return _measure; }
    /// Whether any event gets a finite merging scale.
    bool active() const { return _mode != 0; }
    /// The merging legs, as external leg indices.
    std::vector<std::size_t> merging_leg_indices() const;
    /// Whether every merging leg's pT is bounded from below by the merging
    /// scale, i.e. a ktdurham measure on a hadronic beam.
    bool bounds_pt() const;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    Measure _measure;
    std::vector<me_int_t> _pdg_ids;
    std::vector<me_int_t> _merging_legs;
    double _d_parameter;
    me_int_t _beam_flags;
    me_int_t _mode;
};

} // namespace madspace
