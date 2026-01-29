#pragma once

#include "madspace/phasespace/matrix_element.h"
#include "madspace/phasespace/pdf.h"
#include "madspace/phasespace/scale.h"

namespace madspace {

class DifferentialCrossSection : public FunctionGenerator {
public:
    DifferentialCrossSection(
        const MatrixElement& matrix_element,
        double cm_energy,
        const RunningCoupling& running_coupling,
        const EnergyScale& energy_scale,
        const nested_vector2<me_int_t>& pid_options = {},
        bool has_pdf1 = false,
        bool has_pdf2 = false,
        const std::optional<PdfGrid>& pdf_grid1 = std::nullopt,
        const std::optional<PdfGrid>& pdf_grid2 = std::nullopt,
        bool has_mirror = false,
        bool input_momentum_fraction = true
    );

    const nested_vector2<me_int_t>& pid_options() const { return _pid_options; }
    bool has_mirror() const { return _has_mirror; }
    bool has_pdf(std::size_t pdf_index) const {
        return _pdfs.at(pdf_index) != std::nullopt;
    }
    const MatrixElement& matrix_element() const { return _matrix_element; }

private:
    ValueVec
    build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    nested_vector2<me_int_t> _pid_options;
    MatrixElement _matrix_element;
    std::array<std::optional<PartonDensity>, 2> _pdfs;
    std::array<std::vector<me_int_t>, 2> _pdf_indices;
    RunningCoupling _running_coupling;
    double _e_cm;
    EnergyScale _energy_scale;
    bool _has_mirror;
    bool _input_momentum_fraction;
};

} // namespace madspace
