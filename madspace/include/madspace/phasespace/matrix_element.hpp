#pragma once

#include <algorithm>

#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * Calls a matrix element through the UMAMI interface.
 *
 * UMAMI (Unified MAtrix eleMent Interface) is `madspace`'s uniform way of
 * calling matrix-element code — currently the madmatrix backend — for both
 * integration and event generation [1] (Sec. 3.3 of [2]). Which quantities are
 * passed and returned is chosen by @p inputs and @p outputs, given as integer
 * keys so the interface can be extended without breaking binary compatibility.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments** (a subset, selected by @p inputs)
 * - `momenta` – `float`, shape `(batch, particle_count, 4)`.
 * - `alpha_s` – `float`, shape `(batch,)`.
 * - `flavor`, `helicity`, `diagram`, `channel` – `int`, shape `(batch,)`
 *   (`channel` is omitted for a fixed @p channel). Further inputs are named as
 *   in MatrixElementInput.
 * - `random_color`, `random_helicity`, `random_diagram` – `float`, shape
 *   `(batch,)` (omitted when @p sample_random_inputs draws them internally).
 * - `spin_legs` – `int`, shape `(batch, 2)` – the 0-based legs whose
 *   polarization `spin_correlated` replaces, -1 for an unused slot.
 * - `spin_vectors` – `float`, shape `(batch, 2, 4)` – the real vectors that
 *   replace them, in the frame of the momenta.
 * - `g_ew` – `float`, shape `(batch,)` – the electroweak coupling e of the
 *   outputs per squared coupling order, 1 for them without powers of e.
 *
 * **Returns** (a subset, selected by @p outputs)
 * - `matrix_element` – `float`, shape `(batch,)` – the squared matrix element.
 * - `diagram_amp2` – `float`, shape `(batch, diagram_count)` – per-diagram
 *   squared amplitudes.
 * - `color_index`, `helicity_index`, `diagram_index` – `int`, shape `(batch,)` –
 *   the sampled color, helicity and diagram.
 * - the custom outputs `cut`, `ren_scale`, `fact_scale1`, `fact_scale2`, `pdf`,
 *   `bias`, `matrix_element_ratio` – `float`, shape `(batch,)`.
 * - the NLO outputs per squared coupling order `o` (provisional UMAMI keys, see
 *   umami.h): `squared_orders` – `float`, shape `(batch, o)` – the matrix
 *   element per order; `color_correlated` – `float`, shape
 *   `(batch, o, color_pair_count)` – the colour-correlated matrix element of
 *   every pair i < j of the coloured legs, flattened as in umami.h, without
 *   the diagonal; `spin_correlated` – `float`, shape
 *   `(batch, o)` – the matrix element with the polarizations of `spin_legs`
 *   replaced by `spin_vectors`; `loop` – `float`, shape `(batch, o, 3)` – the
 *   finite part and the coefficients of 1/eps and 1/eps^2 of the one-loop
 *   matrix element at the scale `scale`.
 * - the blocks of the Sudakov approximation of the EW corrections per
 *   helicity combination `h` of the Born (provisional UMAMI keys, see
 *   umami.h): `ew_sudakov_diagonal` and `ew_sudakov_lo2` – `float`, shape
 *   `(batch, h)` – the base amplitude squared at the LO1 and the LO2;
 *   `ew_sudakov_derivatives` – `float`, shape `(batch, h, 6)` – the
 *   derivatives of the first with respect to the parameters of its parameter
 *   renormalization; `ew_sudakov_interference` – `float`, shape
 *   `(batch, h, interference_count, 2)` – the interferences with the
 *   interference amplitudes, real and imaginary part.
 *
 * **References**
 * - [1] S. Hageböck et al., "Data-parallel leading-order event generation in
 *   MadGraph5_aMC@NLO", https://arxiv.org/abs/2507.21039
 * - [2] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.3)
 */
class MatrixElement : public FunctionGenerator {
public:
    /// A quantity the matrix element can be asked to consume.
    enum MatrixElementInput {
        momenta_in,           ///< Momenta of all external particles.
        alpha_s_in,           ///< Strong coupling.
        flavor_in,            ///< Flavour assignment.
        random_color_in,      ///< Random number for the color draw.
        random_helicity_in,   ///< Random number for the helicity draw.
        random_diagram_in,    ///< Random number for the diagram draw.
        helicity_in,          ///< Fixed helicity.
        channel_in,           ///< Multi-channel index.
        diagram_in,           ///< Fixed diagram.
        particle_count_in,    ///< Number of external particles (custom functions).
        pdg_ids_in,           ///< PDG ids of the external particles.
        momentum_fraction_in, ///< Momentum fraction of an incoming parton.
        scale_in,             ///< Energy scale, e.g. the factorization scale.
        pdg_id_in,            ///< PDG id of a single parton.
        beam_index_in,        ///< Index of the beam (0 or 1).
        random_flavor_in,     ///< Random number for the flavour draw.
        /// x f(x, mu_F) of both incoming legs for the PDG ids -6..6 (6: gluon).
        parton_densities_in,
        spin_legs_in,         ///< Legs of the spin correlation (provisional).
        spin_vectors_in,      ///< Vectors of the spin correlation (provisional).
        g_ew_in               ///< Electroweak coupling (provisional).
    };

    /// A quantity the matrix element can be asked to produce.
    enum MatrixElementOutput {
        matrix_element_out, ///< Squared matrix element.
        diagram_amp2_out,   ///< Per-diagram squared amplitudes.
        color_index_out,    ///< Sampled color.
        helicity_index_out, ///< Sampled helicity.
        diagram_index_out,  ///< Sampled diagram.
        cut_out,            ///< Custom cut value, 0 rejects the event.
        ren_scale_out,      ///< Custom renormalization scale.
        fact_scale1_out,    ///< Custom factorization scale, first incoming.
        fact_scale2_out,    ///< Custom factorization scale, second incoming.
        pdf_out,            ///< Custom parton density x f(x, Q).
        bias_out,           ///< Custom event bias.
        /// Ratio of the matrix element to a reference matrix element, for a
        /// @ref WeightingStep.
        matrix_element_ratio_out,
        flavor_index_out,     ///< Selected flavour, -1 if unchanged.
        squared_orders_out,   ///< Matrix element per squared order (provisional).
        color_correlated_out, ///< Colour-correlated matrix element (provisional).
        spin_correlated_out,  ///< Spin-correlated matrix element (provisional).
        loop_out,             ///< One-loop matrix element (provisional).
        /// Base amplitudes squared of the EW Sudakov approximation at the LO1
        /// per helicity combination (provisional).
        ew_sudakov_diagonal_out,
        /// The same at the LO2 (provisional).
        ew_sudakov_lo2_out,
        /// Their derivatives with respect to the six parameters of the
        /// parameter renormalization (provisional).
        ew_sudakov_derivatives_out,
        /// Their interferences with the interference amplitudes
        /// (provisional).
        ew_sudakov_interference_out
    };

    /**
     * @param matrix_element_index  Index of the matrix element in the plugin.
     * @param particle_count         Number of external particles.
     * @param inputs                 Quantities to pass in (default: just
     *                               `momenta_in`; required in Python).
     * @param outputs                Quantities to return (default: just
     *                               `matrix_element_out`; required in Python).
     * @param diagram_count          Number of Feynman diagrams.
     * @param sample_random_inputs   If true, the `random_*` inputs are drawn
     *                               internally instead of being passed in.
     * @param me_frame               One-based indices of the external particles
     *                               whose momentum sum defines the rest frame
     *                               the matrix element is evaluated in (the run
     *                               card's me_frame). Empty means no boost.
     * @param incoming_count         How many of the external particles are
     *                               incoming (the first ones). With me_frame
     *                               it fixes the frame the boost starts from,
     *                               see reference_mask() and along_beam(), to
     *                               avoid a Wigner rotation as in madevent.
     * @param channel                If non-negative, `channel_in` is passed this
     *                               fixed value instead of being an input.
     * @param squared_order_count    Number of squared coupling orders of the
     *                               NLO outputs.
     * @param color_pair_count       Number of pairs i < j of the coloured legs,
     *                               n_c (n_c - 1) / 2, the size of
     *                               `color_correlated_out` per order.
     * @param helicity_count         Number of helicity combinations of the
     *                               outputs of the EW Sudakov approximation.
     * @param interference_count     Number of interference amplitudes of
     *                               `ew_sudakov_interference_out`.
     * @throws std::invalid_argument for `spin_vectors_in` with @p me_frame, as
     *         the vectors would need the boost of the momenta, for
     *         `color_correlated_out` without colour pairs, and for the outputs
     *         of the EW Sudakov approximation without helicity combinations
     *         or `ew_sudakov_interference_out` without interferences.
     */
    MatrixElement(
        std::size_t matrix_element_index,
        std::size_t particle_count,
        const std::vector<MatrixElementInput>& inputs = {momenta_in},
        const std::vector<MatrixElementOutput>& outputs = {matrix_element_out},
        std::size_t diagram_count = 1,
        bool sample_random_inputs = false,
        const std::vector<me_int_t>& me_frame = {},
        std::size_t incoming_count = 2,
        me_int_t channel = -1,
        std::size_t squared_order_count = 1,
        std::size_t color_pair_count = 0,
        std::size_t helicity_count = 0,
        std::size_t interference_count = 0
    );
    /**
     * Take @p matrix_element_index, @p particle_count, @p diagram_count (only
     * with `diagram_amp2_out`, one otherwise), @p squared_order_count,
     * @p color_pair_count (only with `color_correlated_out`, from the colour
     * representations of the legs), @p helicity_count (only with the outputs
     * of the EW Sudakov approximation) and @p interference_count (only with
     * `ew_sudakov_interference_out`) from an already-loaded matrix element.
     * @param matrix_element_api    The loaded matrix element.
     * @param inputs                Quantities to pass in.
     * @param outputs               Quantities to return.
     * @param sample_random_inputs  See the other constructor.
     * @param me_frame              See the other constructor.
     * @param incoming_count        See the other constructor.
     * @param channel               See the other constructor.
     */
    MatrixElement(
        const MatrixElementApi& matrix_element_api,
        const std::vector<MatrixElementInput>& inputs = {momenta_in},
        const std::vector<MatrixElementOutput>& outputs = {matrix_element_out},
        bool sample_random_inputs = false,
        const std::vector<me_int_t>& me_frame = {},
        std::size_t incoming_count = 2,
        me_int_t channel = -1
    ) :
        MatrixElement(
            matrix_element_api.index(),
            matrix_element_api.particle_count(),
            inputs,
            outputs,
            std::find(outputs.begin(), outputs.end(), diagram_amp2_out) != outputs.end()
                ? matrix_element_api.diagram_count()
                : 1,
            sample_random_inputs,
            me_frame,
            incoming_count,
            channel,
            matrix_element_api.squared_order_count(),
            std::find(outputs.begin(), outputs.end(), color_correlated_out) !=
                    outputs.end()
                ? matrix_element_api.color_pair_count()
                : 0,
            has_ew_sudakov_output(outputs) ? matrix_element_api.helicity_count() : 0,
            std::find(outputs.begin(), outputs.end(), ew_sudakov_interference_out) !=
                    outputs.end()
                ? matrix_element_api.ew_sudakov_interference_count()
                : 0
        ) {};
    /// Whether the outputs ask for a block of the EW Sudakov approximation.
    static bool has_ew_sudakov_output(const std::vector<MatrixElementOutput>& outputs) {
        return std::any_of(outputs.begin(), outputs.end(), [](auto output) {
            return output == ew_sudakov_diagonal_out || output == ew_sudakov_lo2_out ||
                output == ew_sudakov_derivatives_out ||
                output == ew_sudakov_interference_out;
        });
    }
    /// Index of the matrix element in the plugin.
    std::size_t matrix_element_index() const { return _matrix_element_index; }
    /// Number of Feynman diagrams.
    std::size_t diagram_count() const { return _diagram_count; }
    /// Number of external particles.
    std::size_t particle_count() const { return _particle_count; }
    /// Number of squared coupling orders of the NLO outputs.
    std::size_t squared_order_count() const { return _squared_order_count; }
    /// Number of pairs of coloured legs of the colour-correlated output.
    std::size_t color_pair_count() const { return _color_pair_count; }
    /// Number of helicity combinations of the EW Sudakov outputs.
    std::size_t helicity_count() const { return _helicity_count; }
    /// Number of interference amplitudes of `ew_sudakov_interference_out`.
    std::size_t interference_count() const { return _interference_count; }
    /// The configured input quantities.
    const std::vector<MatrixElementInput>& inputs() const { return _inputs; }
    /// The configured output quantities.
    const std::vector<MatrixElementOutput>& outputs() const { return _outputs; }
    /// Whether the `random_*` inputs are drawn internally.
    bool sample_random_inputs() const { return _sample_random_inputs; }
    /// Whether the input is one of the `random_*` inputs.
    static bool is_random_input(MatrixElementInput input) {
        return input == random_color_in || input == random_helicity_in ||
            input == random_diagram_in || input == random_flavor_in;
    }
    /// The fixed channel passed as `channel_in`, or -1 if it is an input.
    me_int_t channel() const { return _channel; }
    /// The inputs that must be supplied by the caller (excludes the `random_*`
    /// ones when they are drawn internally, and `channel_in` for a fixed channel).
    std::vector<MatrixElementInput> external_inputs() const;
    /// Per-particle 0/1 selector built from me_frame, empty if no boost is
    /// applied.
    const std::vector<me_int_t>& frame_mask() const { return _frame_mask; }
    /// Per-particle 0/1 selector of the incoming particle of a decay, whose
    /// rest frame the me_frame boost starts from; empty when no boost is
    /// applied, when it is already the frame me_frame asks for, and for a
    /// collision, whose me_frame is reached along the beam instead
    /// (along_beam()).
    const std::vector<me_int_t>& reference_mask() const { return _reference_mask; }
    /// Whether the me_frame boost starts with a boost along the beam axis to
    /// zero rapidity of the selected particles, which makes the frame the
    /// same from any frame along the beam (two incoming particles, and a
    /// frame other than their own rest frame).
    bool along_beam() const { return _along_beam; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _matrix_element_index;
    std::size_t _particle_count;
    std::size_t _diagram_count;
    std::size_t _squared_order_count;
    std::size_t _color_pair_count;
    std::size_t _helicity_count;
    std::size_t _interference_count;
    std::vector<MatrixElementInput> _inputs;
    std::vector<MatrixElementOutput> _outputs;
    bool _sample_random_inputs;
    me_int_t _channel;
    std::vector<me_int_t> _frame_mask;
    std::vector<me_int_t> _reference_mask;
    bool _along_beam;
};

} // namespace madspace
