#include "madspace/phasespace/vegas.h"

using namespace madspace;

VegasHistogram::VegasHistogram(std::size_t dimension, std::size_t bin_count) :
    FunctionGenerator(
        "VegasHistogram",
        {batch_float_array(dimension), batch_float},
        {single_float_array_2d(dimension, bin_count),
         single_int_array_2d(dimension, bin_count)}
    ),
    _bin_count(bin_count) {}

ValueVec
VegasHistogram::build_function_impl(FunctionBuilder& fb, const ValueVec& args) const {
    auto [values, counts] =
        fb.vegas_histogram(args.at(0), args.at(1), static_cast<me_int_t>(_bin_count));
    return {values, counts};
}

Mapping::Result VegasMapping::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto grid = fb.global(
        _grid_name,
        DataType::dt_float,
        {static_cast<int>(_dimension), static_cast<int>(_bin_count) + 1}
    );
    auto [output, dets] = fb.vegas_forward(inputs.at(0), grid);
    return {{output}, fb.reduce_product(dets)};
}

Mapping::Result VegasMapping::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto grid = fb.global(
        _grid_name,
        DataType::dt_float,
        {static_cast<int>(_dimension), static_cast<int>(_bin_count) + 1}
    );
    auto [output, dets] = fb.vegas_inverse(inputs.at(0), grid);
    return {{output}, fb.reduce_product(dets)};
}

void VegasMapping::initialize_globals(ContextPtr context) const {
    context->define_global(
        _grid_name, DataType::dt_float, {_dimension, _bin_count + 1}
    );
    initialize_vegas_grid(context, _grid_name);
}

void madspace::initialize_vegas_grid(ContextPtr context, const std::string& grid_name) {
    bool is_cpu = context->device() == cpu_device();
    auto grid_global = context->global(grid_name);
    if (grid_global.shape().size() != 3 || grid_global.size(0) != 1 ||
        grid_global.size(2) < 2 || grid_global.dtype() != DataType::dt_float) {
        throw std::runtime_error("Invalid vegas grid type");
    }
    Tensor grid;
    if (is_cpu) {
        grid = grid_global;
    } else {
        grid = Tensor(DataType::dt_float, grid_global.shape());
    }

    auto grid_view = grid.view<double, 3>()[0];
    auto bin_count = grid.size(2) - 1;
    double increment = 1. / bin_count;
    for (std::size_t i = 0; i < grid_view.size(); ++i) {
        for (std::size_t j = 0; j < bin_count; ++j) {
            grid_view[i][j] = increment * j;
        }
        grid_view[i][bin_count] = 1.;
    }

    if (!is_cpu) {
        grid_global.copy_from(grid);
    }
}
