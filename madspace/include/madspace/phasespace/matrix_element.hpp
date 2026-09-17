#pragma once

#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"

namespace madspace {

class MatrixElement : public FunctionGenerator {
public:
    enum MatrixElementInput {
        momenta_in,
        alpha_s_in,
        flavor_in,
        random_color_in,
        random_helicity_in,
        random_diagram_in,
        helicity_in,
        channel_in,
        diagram_in
    };

    enum MatrixElementOutput {
        matrix_element_out,
        diagram_amp2_out,
        color_index_out,
        helicity_index_out,
        diagram_index_out
    };

    /// me_frame: one-based indices of the external particles whose momentum
    /// sum defines the rest frame the matrix element is evaluated in (the run
    /// card's me_frame). Empty means no boost, i.e. the matrix element sees the
    /// momenta in the frame they are generated in.
    ///
    /// incoming_count is how many of the external particles are incoming. The
    /// frame is reached the way madevent reaches it, starting from the rest
    /// frame of the incoming system (the partonic centre of mass of a
    /// collision, the decaying particle's rest frame of a decay) -- boosts do
    /// not commute, so going there straight from the lab frame would leave a
    /// Wigner rotation behind and rotate the polarisation axes.
    MatrixElement(
        std::size_t matrix_element_index,
        std::size_t particle_count,
        const std::vector<MatrixElementInput>& inputs = {momenta_in},
        const std::vector<MatrixElementOutput>& outputs = {matrix_element_out},
        std::size_t diagram_count = 1,
        bool sample_random_inputs = false,
        const std::vector<me_int_t>& me_frame = {},
        std::size_t incoming_count = 2
    );
    MatrixElement(
        const MatrixElementApi& matrix_element_api,
        const std::vector<MatrixElementInput>& inputs = {momenta_in},
        const std::vector<MatrixElementOutput>& outputs = {matrix_element_out},
        bool sample_random_inputs = false,
        const std::vector<me_int_t>& me_frame = {},
        std::size_t incoming_count = 2
    ) :
        MatrixElement(
            matrix_element_api.index(),
            matrix_element_api.particle_count(),
            inputs,
            outputs,
            matrix_element_api.diagram_count(),
            sample_random_inputs,
            me_frame,
            incoming_count
        ) {};
    std::size_t matrix_element_index() const { return _matrix_element_index; }
    std::size_t diagram_count() const { return _diagram_count; }
    std::size_t particle_count() const { return _particle_count; }
    const std::vector<MatrixElementInput>& inputs() const { return _inputs; }
    const std::vector<MatrixElementOutput>& outputs() const { return _outputs; }
    std::vector<MatrixElementInput> external_inputs() const;
    /// Per-particle 0/1 selector built from me_frame, empty if no boost is
    /// applied.
    const std::vector<me_int_t>& frame_mask() const { return _frame_mask; }
    /// Per-particle 0/1 selector of the incoming particles, whose rest frame
    /// the me_frame boost starts from; empty when no boost is applied or when
    /// it is already the frame me_frame asks for.
    const std::vector<me_int_t>& reference_mask() const { return _reference_mask; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _matrix_element_index;
    std::size_t _particle_count;
    std::size_t _diagram_count;
    std::vector<MatrixElementInput> _inputs;
    std::vector<MatrixElementOutput> _outputs;
    bool _sample_random_inputs;
    std::vector<me_int_t> _frame_mask;
    std::vector<me_int_t> _reference_mask;
};

} // namespace madspace
