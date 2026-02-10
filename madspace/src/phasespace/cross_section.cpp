#include "madspace/phasespace/cross_section.h"

#include <set>

using namespace madspace;

DifferentialCrossSection::DifferentialCrossSection(
    const MatrixElement& matrix_element,
    double cm_energy,
    const RunningCoupling& running_coupling,
    const EnergyScale& energy_scale,
    const nested_vector2<me_int_t>& pid_options,
    bool has_pdf1,
    bool has_pdf2,
    const std::optional<PdfGrid>& pdf_grid1,
    const std::optional<PdfGrid>& pdf_grid2,
    bool has_mirror,
    bool input_momentum_fraction
) :
    FunctionGenerator(
        "DifferentialCrossSection",
        [&] {
            TypeVec arg_types;
            for (auto [arg_type, input] :
                 zip(matrix_element.arg_types(), matrix_element.external_inputs())) {
                if (input != MatrixElement::alpha_s_in) {
                    arg_types.push_back(arg_type);
                }
            }
            if (input_momentum_fraction) {
                arg_types.push_back(batch_float); // x1
                arg_types.push_back(batch_float); // x2
            }
            arg_types.push_back(batch_int); // pdf_id
            if (has_mirror) {
                arg_types.push_back(batch_int); // mirror
            }
            bool uses_cached_pdf = false;
            auto add_pdf_args = [&](auto& pdf_grid, bool has_pdf, int index) {
                if (!pdf_grid && has_pdf) {
                    std::set<int> pids;
                    for (auto& option : pid_options) {
                        pids.insert(option.at(index));
                    }
                    arg_types.push_back(batch_float_array(pids.size())); // pdf cache
                    uses_cached_pdf = true;
                }
            };
            add_pdf_args(pdf_grid1, has_pdf1, 0);
            add_pdf_args(pdf_grid2, has_pdf2, 1);
            if (uses_cached_pdf) {
                arg_types.push_back(batch_float); // renormalization scale
            }
            return arg_types;
        }(),
        matrix_element.return_types()
    ),
    _pid_options(pid_options),
    _matrix_element(matrix_element),
    _running_coupling(running_coupling),
    _e_cm(cm_energy),
    _energy_scale(energy_scale),
    _has_mirror(has_mirror),
    _input_momentum_fraction(input_momentum_fraction) {
    auto init_pdf = [&](auto& pdf_grid, bool has_pdf, int index) {
        if (has_pdf) {
            if (pdf_grid) {
                std::vector<int> pids;
                for (auto& option : pid_options) {
                    pids.push_back(option.at(index));
                }
                _pdfs.at(index) = PartonDensity(pdf_grid.value(), pids, true);
            } else {
                std::set<int> pids;
                for (auto& option : pid_options) {
                    pids.insert(option.at(index));
                }
                for (auto& option : pid_options) {
                    _pdf_indices.at(index).push_back(
                        std::distance(pids.begin(), pids.find(option.at(index)))
                    );
                }
            }
        }
    };
    init_pdf(pdf_grid1, has_pdf1, 0);
    init_pdf(pdf_grid2, has_pdf2, 1);
}

ValueVec DifferentialCrossSection::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    std::size_t arg_index = 0;
    int alpha_s_index = -1;
    Value momenta;
    ValueVec matrix_args;
    for (auto input : _matrix_element.external_inputs()) {
        if (input == MatrixElement::alpha_s_in) {
            alpha_s_index = arg_index;
        } else {
            Value arg = args.at(arg_index++);
            if (input == MatrixElement::momenta_in) {
                momenta = arg;
            }
            matrix_args.push_back(arg);
        }
    }

    std::array<Value, 2> x1x2;
    if (_input_momentum_fraction) {
        x1x2 = {args.at(arg_index), args.at(arg_index + 1)};
        arg_index += 2;
    } else {
        x1x2 = fb.momenta_to_x1x2(momenta, _e_cm);
    }
    auto pdf_flavor_id = args.at(arg_index++);
    if (_has_mirror) {
        Value mirror_id = args.at(arg_index++);
    }
    // TODO: need to use mirror_id if we have two different PDFs
    ValueVec scales;
    Value ren_scale;
    bool use_cached_pdf =
        _pdf_indices.at(0).size() > 0 || _pdf_indices.at(1).size() > 0;
    if (!use_cached_pdf) {
        scales = _energy_scale.build_function(fb, {momenta});
        ren_scale = scales.at(0);
    }

    std::array<Value, 2> pdf_outputs{1., 1.};
    for (std::size_t i = 0; auto [pdf_output, pdf, x, pdf_indices] :
                            zip(pdf_outputs, _pdfs, x1x2, _pdf_indices)) {
        if (pdf) {
            pdf_output = pdf.value()
                             .build_function(fb, {x, scales.at(i + 1), pdf_flavor_id})
                             .at(0);
        } else if (pdf_indices.size() > 0) {
            pdf_output = fb.gather(
                fb.gather_int(pdf_flavor_id, pdf_indices), args.at(arg_index++)
            );
        }
    }

    if (use_cached_pdf) {
        ren_scale = args.at(arg_index++);
    }

    if (alpha_s_index != -1) {
        matrix_args.insert(
            matrix_args.begin() + alpha_s_index,
            _running_coupling.build_function(fb, {ren_scale}).at(0)
        );
    }

    auto me_result = _matrix_element.build_function(fb, matrix_args);
    auto search = std::find(
        _matrix_element.outputs().begin(),
        _matrix_element.outputs().end(),
        MatrixElement::matrix_element_out
    );
    if (search == _matrix_element.outputs().end()) {
        throw std::runtime_error("matrix element missing in return values");
    }
    std::size_t me_index = search - _matrix_element.outputs().begin();
    me_result.at(me_index) = fb.diff_cross_section(
        x1x2.at(0),
        x1x2.at(1),
        pdf_outputs.at(0),
        pdf_outputs.at(1),
        me_result.at(me_index),
        _e_cm * _e_cm
    );
    return me_result;
}
