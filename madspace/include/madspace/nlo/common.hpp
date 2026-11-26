#pragma once

#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/phasespace.hpp"

namespace madspace {

/// The color factors C_A, C_F and T_R of QCD.
inline constexpr double QCD_CA = 3., QCD_CF = 4. / 3., QCD_TR = 0.5;

/**
 * The light-cone components (up to a common factor) whose ratio is
 * @f$e^{2y}@f$ for the rapidity @f$y@f$ in the lab frame of the
 * center-of-mass frame of the momentum fractions @p x1 and @p x2 of @p born,
 * with its beam rapidity and, with `mirror_beams`, its `mirror_index`
 * condition. The instruction `boost_beam_light_cone` with the arguments `minus`,
 * `plus` boosts the lab frame into that frame, with `plus`, `minus` back.
 *
 * @param fb          The function builder.
 * @param born        The Born mapping.
 * @param x1, x2      The momentum fractions of the first and second incoming
 *                    parton, those of the Born or of a real emission.
 * @param conditions  The conditions of the Born mapping.
 */
std::pair<Value, Value> born_frame_light_cone(
    FunctionBuilder& fb,
    const PhaseSpaceMapping& born,
    Value x1,
    Value x2,
    const NamedVector<Value>& conditions
);

/// The strong coupling at which the NLO libraries give their blocks,
/// @f$\alpha_s = 1 / (4 \pi)@f$, that is @f$g_s = 1@f$.
inline constexpr double NLO_UNIT_ALPHA_S = 1. / (4. * std::numbers::pi);

/**
 * The color-correlated Born @f$\langle T_m \cdot T_n \rangle B@f$ of all
 * legs of a Born from the output `color_correlated` of a UMAMI library, its
 * pairs @f$m < n@f$ of the colored legs: the pairs off the diagonal,
 * @f$C_m B@f$ with the Casimir of the representation of the leg on it and
 * zero for the colorless legs.
 */
class ColorMatrix {
public:
    /// @param color_representations Per Born leg its color representation
    ///                     as UMAMI_META_COLOR_REPRESENTATIONS gives it, 1
    ///                     for a colorless leg.
    /// @param who          The class whose errors name it.
    /// @throws std::invalid_argument if a leg has a representation without a
    ///         Casimir here (but 1, 3, -3 and 8).
    ColorMatrix(const std::vector<int>& color_representations, const std::string& who);
    /// Whether the Born has two colored legs, so that the library has the
    /// output.
    bool has_color() const { return _has_color; }
    /// Number of legs of the Born.
    std::size_t particle_count() const { return _rows.size(); }
    /// Number of pairs of colored legs, the length of the output of the
    /// library per order.
    std::size_t pair_count() const { return _pair_count; }
    /// The matrices, shape `(batch, b, n, n)`, from the Born orders @p born,
    /// shape `(batch, b)`, and the pairs @p pairs, shape `(batch, b, pairs)`.
    Value build(FunctionBuilder& fb, Value born, Value pairs) const;

private:
    // per leg the indices of its row in the pairs of one order followed by
    // the Casimirs times the Born and a zero
    nested_vector2<me_int_t> _rows;
    std::vector<double> _casimirs;
    std::size_t _pair_count;
    bool _has_color;
};

/**
 * The virtual of the virtual trick of MadFKS: with the average @f$r_o@f$ of
 * @f$V_o / B@f$ per NLO order and the fraction @f$f@f$ of the points with
 * the virtual, @f$r_o B + (V_o - r_o B) / f@f$ where the virtual is evaluated
 * and @f$r_o B@f$ elsewhere, with @f$B@f$ the sum of the Born orders.
 *
 * @param fb          The function builder.
 * @param virtual_value The virtual, shape `(batch, o)`, zero where it is not
 *                    evaluated.
 * @param born        The Born orders, shape `(batch, b)`.
 * @param ratio       @f$r_o@f$, shape `(batch, o)`.
 * @param fraction    @f$f@f$, shape `(batch,)`.
 * @param evaluated   One where the virtual is evaluated, zero elsewhere.
 * @return The columns of the virtual of the trick and those of its
 *         approximation @f$r_o B@f$.
 */
std::pair<ValueVec, ValueVec> virtual_trick(
    FunctionBuilder& fb,
    Value virtual_value,
    Value born,
    Value ratio,
    Value fraction,
    Value evaluated
);

} // namespace madspace
