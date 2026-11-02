#pragma once

#include <variant>
#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/fks_radiation.hpp"
#include "madspace/phasespace/phasespace.hpp"

namespace madspace {

/**
 * Phase space of a real emission in one FKS configuration: a Born phase-space
 * mapping followed by the FKS emission of @ref FKSRadiation, in the lab frame.
 *
 * The Born mapping, a @ref PhaseSpaceMapping of the Born process of the FKS
 * configuration, gives the Born momenta in the lab frame and the momentum
 * fractions of the Born. They are boosted along the beam axis into the
 * centre-of-mass frame of the Born, where @ref FKSRadiation adds the radiated
 * parton @f$i@f$ off the emitter @f$j@f$, and the real emission is boosted back
 * into the lab with the same boost (the instruction `fks_beam_boost`, which
 * keeps the precision of the momenta in the centre-of-mass frame for any
 * rapidity). For an incoming emitter the real emission
 * then has the momentum fractions of the real emission, and its incoming
 * momenta are those of the beams times them. The emitter is massive if the
 * Born mapping gives it a mass. As in genps_fks.f, the last three random
 * numbers are those of the emission.
 *
 * The inverse needs the frame of the Born from the real emission: for a
 * final-state emitter it is that of the incoming momenta, for an incoming
 * emitter that of the system recoiling against @f$i@f$ (the instruction
 * `fks_recoil_light_cone`), which has zero longitudinal momentum in the frame
 * of the Born.
 *
 * The weight is the product of the weights of the two mappings, so that
 * @f$\mathrm{d}x_1\,\mathrm{d}x_2\,\mathrm{d}\Phi_{n+1} = w\,\mathrm{d}r@f$ in
 * the normalization of @ref PhaseSpaceMapping: the weight of a
 * @ref PhaseSpaceMapping of the real process, with the factor @f$\xi@f$ that
 * genps_fks.f leaves to the FKS distributions.
 *
 * Cuts belong to the real emission and the counter-events, not to the Born, so
 * the Born mapping must have none, except a floor on @f$\sqrt{\hat s}@f$ for
 * a final-state emitter (see fks_sqrt_s_hat_min).
 *
 * With `mirror_beams` the Born mapping takes leg 1 from beam 2 for
 * `mirror_index` 1, along @f$-z@f$, with its momentum fraction x1. The emission
 * needs the first incoming parton along @f$+z@f$, so in the centre-of-mass
 * frame of the Born the mirror of the Born mapping (a rotation by @f$\pi@f$
 * about the x axis) turns it around before the emission and back after it,
 * and the frame of the Born follows the beam each leg comes from. The
 * emission and its weight are those without the mirror.
 *
 * The FKS configurations of one Born, each an FKSMapping of the same Born
 * mapping, have the same inputs, outputs and conditions. As channels of a
 * @ref MultiChannelMapping they cover the phase space of the real emission
 * once per configuration, and as @ref FKSKinematics in a
 * @ref MultiChannelFunction they give the event, the Born and the
 * counter-events of each configuration.
 *
 * `batch` is the leading batch dimension and `n` the number of particles of
 * the Born, incoming included.
 *
 * **Inputs**
 * - `random` – `float`, shape `(batch, random_dim())` – the random numbers of
 *   the Born mapping, then @f$r_\xi@f$, @f$r_y@f$ and @f$r_\phi@f$ of the
 *   emission.
 * - `discrete` – `int`, shape `(batch, discrete_dim())` – the discrete channel
 *   choices of the Born mapping. Present only when it has any.
 *
 * **Conditions**
 * - those of the Born mapping (`permutation_index`, `mirror_index`).
 *
 * **Outputs**
 * - `momenta` – `float`, shape `(batch, n + 1, 4)` – momenta of the real
 *   emission in the lab frame, in the order of the real process.
 * - `x1`, `x2` – `float`, shape `(batch,)` – momentum fractions of the real
 *   emission (1 for every point if the Born mapping is leptonic).
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272 (Sec. 5)
 */
class FKSMapping : public Mapping {
public:
    /// @param born       Mapping of the Born process, without cuts (but a
    ///                   floor on sqrt(s_hat) for a final-state emitter).
    /// @param i_fks      Position of the radiated parton i in the real process.
    /// @param j_fks      Position of the emitter j in the real process.
    /// @param s_hat_min  Lower bound on the partonic energy squared of the
    ///                   real emission, for an incoming emitter. See
    ///                   @ref FKSRadiation.
    FKSMapping(
        const PhaseSpaceMapping& born,
        std::size_t i_fks,
        std::size_t j_fks,
        double s_hat_min = 0.
    );
    /// Number of continuous random inputs: those of the Born mapping and 3.
    std::size_t random_dim() const { return _born.random_dim() + 3; }
    /// Number of discrete inputs, those of the Born mapping.
    std::size_t discrete_dim() const override { return _born.discrete_dim(); }
    /// Number of particles of the real emission, incoming included.
    std::size_t particle_count() const { return _real_masses.size(); }
    std::size_t i_fks() const { return _radiation.i_fks(); }
    std::size_t j_fks() const { return _radiation.j_fks(); }
    /// The Born mapping.
    const PhaseSpaceMapping& born() const { return _born; }
    /// The emission.
    const FKSRadiation& radiation() const { return _radiation; }
    /**
     * Add the real emission, its Born and its FKS counter-events to @p fb, at
     * the inputs and conditions of the forward map. All momenta are in the
     * lab frame. Returns
     * - `momenta`, `x1`, `x2`, `weight`: the forward map;
     * - `born_momenta`, `born_x1`, `born_x2`, `born_weight`: the Born mapping;
     * - `xi`, `xi_hat`, `y`, `xi_max`: the FKS variables of the event, see
     *   @ref FKSRadiation::build_counter_events;
     * - for each counter-event `k` of FKSRadiation::counter_events():
     *   `k_momenta`, `k_x1`, `k_x2` and `k_det`, the Jacobian at the
     *   counter-event without the factor @f$\xi@f$, times the weight of the
     *   Born (0 where the counter-event does not exist).
     */
    NamedVector<Value> build_counter_events(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions = {}
    ) const;
    /// Names and types of the results of build_counter_events().
    NamedVector<Type> counter_event_types() const;

private:
    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    NamedVector<Value> build_events(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions,
        bool with_counter_events
    ) const;
    // light-cone components (up to a factor) whose ratio is exp(2 y) for the
    // rapidity y of the centre-of-mass frame of the Born in the lab frame, from
    // its momentum fractions (the arguments of fks_beam_boost)
    std::pair<Value, Value> born_frame(
        FunctionBuilder& fb, Value x1, Value x2, const NamedVector<Value>& conditions
    ) const;
    // between the orientation of the beams (mirror_index) and the one of the
    // emission, the first incoming parton along +z (an involution)
    Value standard_orientation(
        FunctionBuilder& fb, Value momenta, const NamedVector<Value>& conditions
    ) const;

    PhaseSpaceMapping _born;
    FKSRadiation _radiation;
    std::vector<double> _born_masses;
    std::vector<double> _real_masses;
};

/**
 * Lower bound on the partonic centre-of-mass energy of a real emission from
 * the cuts on its Born-level objects.
 *
 * The cuts are those that the n final-state objects of the Born process (jets
 * after clustering, leptons, ...) have to pass in the real emission and its
 * counter-events, indexed like the Born. The bound holds for every real
 * emission that passes them, whatever i does: it is the largest of
 * - the cut on sqrt(s_hat) itself;
 * - the sum of the transverse masses sqrt(m_k^2 + pt_min,k^2) of the
 *   objects, without asking their transverse momenta to balance, which i can
 *   take up (in the partonic centre-of-mass frame sqrt(s_hat) is the sum of
 *   the energies, each object has at least its transverse mass, and a jet is
 *   no lighter than the massless parton it stands for);
 * - the floors of the pair mass cuts: a pair has at least its cut in energy
 *   and the other objects their transverse masses, so cut + their transverse
 *   masses; and since the invariant mass of a sum of future-pointing momenta
 *   is at least that of any part, the invariant mass of the objects from all
 *   pairs together, sum_{a<b} m_ab^2 - (n - 2) sum_k m_k^2;
 * - the sum of the masses.
 * Eta and delta R cuts give none.
 *
 * For a final-state emitter s_hat is that of the Born, and the bound is a
 * floor of the Born mapping (a `sqrt_s` cut, which FKSMapping accepts there).
 * For an incoming emitter the Born has less energy than the real emission, and
 * the bound squared is the `s_hat_min` of the FKSMapping, as the
 * tau_lower_bound of genps_fks.f. An @ref FKSConfigurationMapping takes the
 * bound as its `sqrt_s_hat_floor` and does both, per event.
 *
 * @param cuts             The cuts on the Born-level objects.
 * @param outgoing_masses  The masses of the final-state particles of the
 *                         Born, in the order of the cuts.
 */
double fks_sqrt_s_hat_min(const Cuts& cuts, const std::vector<double>& outgoing_masses);

/**
 * Phase space of a real emission in the FKS configurations of one Born, each
 * event in the configuration of its `fks_index` condition, all in one pass.
 *
 * The configurations share the Born mapping and differ only by the positions
 * @f$i@f$ and @f$j@f$ in the real process. For every event a table read at its
 * `fks_index` gives them, with the mass of the emitter and the two lower
 * bounds on the partonic energy: the floor of the Born mapping (its
 * `sqrt_s_hat_min` condition), @p sqrt_s_hat_floor for a final-state emitter
 * and none for an incoming one, and the `s_hat_min` of the emission,
 * @p s_hat_min, for an incoming emitter at least @p sqrt_s_hat_floor squared.
 * That is the split of madspace.fks.fks_mappings (see fks_sqrt_s_hat_min). The
 * emission is built for every event in one layout, the Born momenta in their
 * order with @f$i@f$ appended, with the formulas of each kind of emitter the
 * configurations have (incoming, massless and massive final-state), and every
 * event takes those of its emitter. The momenta of the real emission are
 * permuted into the order of the real process of the configuration last
 * (`permute_momenta`). Event by event, the results are those of the
 * @ref FKSMapping of the configuration (for a final-state emitter with a
 * Born mapping that has the floor as a cut on sqrt(s_hat)).
 *
 * Drawing `fks_index`, uniformly or with an adaptive discrete sampler, and its
 * weight belong to the integrand, as for the `permutation_index` of a
 * @ref PhaseSpaceMapping: this mapping only routes the events. The
 * alternative, with the configurations as channels, is a
 * @ref MultiChannelMapping of the FKSMappings.
 *
 * `batch` is the leading batch dimension and `n` the number of particles of
 * the Born, incoming included.
 *
 * **Inputs**
 * - `random` – `float`, shape `(batch, random_dim())` – the random numbers of
 *   the Born mapping, then @f$r_\xi@f$, @f$r_y@f$ and @f$r_\phi@f$ of the
 *   emission.
 * - `discrete` – `int`, shape `(batch, discrete_dim())` – the discrete channel
 *   choices of the Born mapping. Present only when it has any.
 *
 * **Conditions**
 * - those of the Born mapping except `sqrt_s_hat_min` (`permutation_index`,
 *   `mirror_index`), then
 * - `fks_index` – `int`, shape `(batch,)` – the configuration of the event,
 *   from 0 to the number of configurations minus one.
 *
 * **Outputs**
 * - `momenta` – `float`, shape `(batch, n + 1, 4)` – momenta of the real
 *   emission in the lab frame, in the order of the real process of the
 *   configuration.
 * - `x1`, `x2` – `float`, shape `(batch,)` – momentum fractions of the real
 *   emission.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation, that of the configuration of the event.
 */
class FKSConfigurationMapping : public Mapping {
public:
    /// The positions of i and of the emitter j in the real process, 0-based.
    using Configuration = std::pair<std::size_t, std::size_t>;

    /**
     * @param born              The @ref PhaseSpaceMapping of the Born process,
     *                          without cuts (they belong to the real emission
     *                          and the counter-events). With a floor, it needs
     *                          the `sqrt_s_hat_min` condition
     *                          (`sqrt_s_hat_min_condition`).
     * @param configurations    (i_fks, j_fks) of each configuration, in the
     *                          order of `fks_index`.
     * @param s_hat_min         Lower bound on the partonic energy squared of
     *                          the real emission, see @ref FKSRadiation.
     * @param sqrt_s_hat_floor  Lower bound on sqrt(s_hat) of every real
     *                          emission that passes the cuts, as
     *                          fks_sqrt_s_hat_min gives it: the floor of the
     *                          Born for a final-state emitter, and raising
     *                          @p s_hat_min for an incoming one.
     */
    FKSConfigurationMapping(
        const PhaseSpaceMapping& born,
        const std::vector<Configuration>& configurations,
        double s_hat_min = 0.,
        double sqrt_s_hat_floor = 0.
    );
    /// The Born mapping.
    const PhaseSpaceMapping& born() const { return _born; }
    /// (i_fks, j_fks) of each configuration.
    const std::vector<Configuration>& configurations() const { return _configurations; }
    /// The lower bound on the partonic energy squared of the real emission.
    double s_hat_min() const { return _s_hat_min; }
    /// The lower bound on sqrt(s_hat) from the cuts.
    double sqrt_s_hat_floor() const { return _sqrt_s_hat_floor; }
    /// Number of random numbers, those of the Born mapping and three.
    std::size_t random_dim() const { return _born.random_dim() + 3; }
    /// Number of discrete channel choices of the Born mapping.
    std::size_t discrete_dim() const override { return _born.discrete_dim(); }
    /// Number of particles of the real emission, incoming included.
    std::size_t particle_count() const { return _born.particle_count() + 1; }
    /**
     * Add the real emission, its Born and its FKS counter-events to @p fb, each
     * event in the configuration of its `fks_index`, at the inputs and
     * conditions of the forward map. Returns those of
     * FKSMapping::build_counter_events.
     */
    NamedVector<Value> build_counter_events(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const;
    /// Names and types of the results of build_counter_events(), those of
    /// FKSMapping::counter_event_types.
    NamedVector<Type> counter_event_types() const;

    /// The kinds of emitter, in the order the formulas are built.
    enum EmitterKind { incoming, massless_final, massive_final };

private:
    // the emission of every kind of emitter the configurations have, in the
    // layout of the Born momenta with i appended, in the Born c.m. frame
    struct Emission {
        Value momenta, x1, x2, xi, y, xi_max, det_xi, rat, p_i_hat;
    };
    // what the emissions of an event need
    struct Context;

    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    NamedVector<Value> build_events(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions,
        bool with_counter_events
    ) const;
    // the conditions of the Born mapping, its floor read at fks_index
    NamedVector<Value>
    born_conditions(FunctionBuilder& fb, const NamedVector<Value>& conditions) const;
    Emission build_emission(
        FunctionBuilder& fb,
        const Context& context,
        EmitterKind kind,
        Value xi_hat,
        Value y_var
    ) const;
    // per event the value of the kind of its emitter
    Value pick(FunctionBuilder& fb, const ValueVec& by_kind, Value slot) const;
    // per event the momenta (rows each) of the kind of its emitter
    Value pick_rows(
        FunctionBuilder& fb, const ValueVec& by_kind, std::size_t rows, Value fks_index
    ) const;
    Emission pick_emission(
        FunctionBuilder& fb,
        const std::vector<Emission>& by_kind,
        Value slot,
        Value fks_index
    ) const;
    bool has(EmitterKind kind) const;

    PhaseSpaceMapping _born;
    std::vector<Configuration> _configurations;
    double _s_hat_min;
    double _sqrt_s_hat_floor;
    std::vector<double> _born_masses;
    // the masses in the layout: those of the Born, then 0 for i
    std::vector<double> _layout_masses;
    // the kinds of emitter the configurations have, and per configuration its
    // position among them
    std::vector<EmitterKind> _kinds;
    std::vector<me_int_t> _kind_slots;
    // per configuration: the emitter mass, the floor of the Born, s_hat_min of
    // the emission, the beam of an incoming emitter (+-1), and whether it has
    // collinear counter-events
    std::vector<double> _emitter_masses;
    std::vector<double> _born_floors;
    std::vector<double> _radiation_s_hat_min;
    std::vector<double> _beam_signs;
    std::vector<double> _collinear_exists;
    // permutations per configuration (output[k] = input[table[c][k]]): the Born
    // emitter first (Born order, n); the emitter and i into place ([Born order,
    // j, i] -> layout, n + 2); the Born emitter into place ([Born order,
    // jbar] -> Born order, n + 1); the layout into the order of the real
    // process and back (n + 1); the emitter first in the layout (n + 1)
    nested_vector2<me_int_t> _emitter_first;
    nested_vector2<me_int_t> _emission_into_place;
    nested_vector2<me_int_t> _jbar_into_place;
    nested_vector2<me_int_t> _real_order;
    nested_vector2<me_int_t> _layout_order;
    nested_vector2<me_int_t> _layout_emitter_first;
};

/**
 * The kinematics an FKS integrand needs: the real emission, its Born and its
 * counter-events, as a function of the random numbers, for one FKS
 * configuration (an @ref FKSMapping) or for several, each event in the
 * configuration of its `fks_index` (an @ref FKSConfigurationMapping).
 *
 * **Arguments**
 * - the inputs of the mapping (`random`, and `discrete` if it has any), then
 *   its conditions (`permutation_index`, and `fks_index` for several
 *   configurations).
 *
 * **Returns**
 * - those of FKSMapping::build_counter_events, in the order of
 *   FKSMapping::counter_event_types.
 */
class FKSKinematics : public FunctionGenerator {
public:
    /// One or several FKS configurations.
    using Configurations = std::variant<FKSMapping, FKSConfigurationMapping>;
    /// @param mapping  The FKS mapping of the configuration, or the
    ///                 FKSConfigurationMapping of several.
    FKSKinematics(const Configurations& mapping);
    /// The mapping.
    const Configurations& mapping() const { return _mapping; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    Configurations _mapping;
};

} // namespace madspace
