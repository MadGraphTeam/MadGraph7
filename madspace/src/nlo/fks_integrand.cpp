#include "madspace/nlo/fks_integrand.hpp"
#include "madspace/nlo/common.hpp"

#include <algorithm>
#include <cstdlib>
#include <format>
#include <map>
#include <numbers>
#include <stdexcept>

using namespace madspace;

namespace {

using Input = MatrixElement::MatrixElementInput;
using Output = MatrixElement::MatrixElementOutput;

// the names of the arguments of the cross section that the integrand gives
// besides the kinematics
const std::vector<std::string> block_names{
    "matrix_element",
    "born",
    "color_correlated",
    "spin_correlated",
    "alpha",
    "virtual",
    "channel_weight",
    "configuration_weight",
    "fks_index",
};

// the reference scales of the user at the event and the soft and collinear
// counter-events
const std::vector<std::string> reference_names{
    "reference_scale",
    "soft_reference_scale",
    "collinear_reference_scale",
};

// the kinematics the integrand returns for the histograms
const std::vector<std::string> kinematics_names{
    "momenta",
    "soft_momenta",
    "collinear_momenta",
    "soft_collinear_momenta",
    "born_momenta",
};

bool has_key(const NamedVector<Type>& types, const std::string& name) {
    auto keys = types.keys();
    return std::find(keys.begin(), keys.end(), name) != keys.end();
}

bool has_output(const MatrixElement& element, Output output) {
    auto& outputs = element.outputs();
    return std::find(outputs.begin(), outputs.end(), output) != outputs.end();
}

void check_library(
    const MatrixElement& element,
    const std::vector<Input>& inputs,
    const std::vector<Output>& outputs,
    const std::string& what
) {
    bool fits = element.inputs() == inputs;
    for (auto output : outputs) {
        fits = fits && has_output(element, output);
    }
    if (!fits) {
        throw std::invalid_argument(
            std::format(
                "FKSChannelIntegrand: the {} library does not have the inputs and "
                "outputs of the integrand",
                what
            )
        );
    }
}

int real_order_count(const std::vector<MatrixElement>& reals) {
    return reals.empty() ? 1 : static_cast<int>(reals.at(0).squared_order_count());
}

std::size_t
count_born_inputs(const MatrixElement& born, const std::vector<MatrixElement>& extras) {
    std::size_t count = born.squared_order_count();
    for (auto& extra : extras) {
        count += extra.squared_order_count();
    }
    return count;
}

} // namespace

FKSChannelIntegrand::FKSChannelIntegrand(
    const FKSKinematics& kinematics,
    const FKSSplittingKernels& kernels,
    const NLOCuts& cuts,
    const NLOScale& born_scale,
    const FKSCrossSection& cross_section,
    const MatrixElement& born,
    const std::vector<MatrixElement>& reals,
    const std::optional<MatrixElement>& loop,
    const std::vector<me_int_t>& real_indices,
    const std::vector<me_int_t>& spin_legs,
    const std::vector<int>& color_representations,
    std::size_t channel,
    double g_ew,
    double alpha,
    const std::vector<double>& tagged_nlo,
    const std::vector<double>& tagged_born,
    double tagged_alpha,
    const std::optional<CustomFunction>& reference_scale,
    const std::vector<MatrixElement>& extras,
    const std::vector<me_int_t>& extra_indices,
    const std::vector<me_int_t>& extra_spin_legs
) :
    FunctionGenerator(
        "FKSChannelIntegrand",
        NamedVector<Type>{
            {"random", kinematics.arg_types().at("random")},
            {"fks_index", batch_int},
            {"configuration_weight", batch_float},
            {"virtual_mask", batch_float},
            {"virtual_ratio", batch_float_array(real_order_count(reals))},
            {"virtual_fraction", batch_float},
        },
        [&] {
            NamedVector<Type> returns;
            auto types = cross_section.return_types();
            auto names = types.keys();
            for (auto [type, name] : zip(types, names)) {
                returns.push_back(name, type);
            }
            returns.push_back("virtual_residual", batch_float);
            returns.push_back("virtual", batch_float_array(real_order_count(reals)));
            returns.push_back(
                "born", batch_float_array(static_cast<int>(born.squared_order_count()))
            );
            returns.push_back("evaluated", batch_float);
            auto kinematic_types = kinematics.return_types();
            for (auto& name : kinematics_names) {
                returns.push_back(name, kinematic_types.at(name));
            }
            return returns;
        }()
    ),
    _kinematics(kinematics),
    _kernels(kernels),
    _cuts(cuts),
    _born_scale(born_scale),
    _cross_section(cross_section),
    _born(born),
    _reals(reals),
    _loop(loop),
    _real_indices(real_indices),
    _spin_legs(spin_legs),
    _color(color_representations, "FKSChannelIntegrand"),
    _channel(channel),
    _g_ew(g_ew),
    _alpha(alpha),
    _tagged_nlo(tagged_nlo),
    _tagged_born(tagged_born),
    _tagged_alpha(tagged_alpha),
    _reference_scale(reference_scale),
    _extras(extras),
    _extra_spin_legs(extra_spin_legs),
    _born_orders(born.squared_order_count()),
    _nlo_orders(reals.empty() ? 0 : reals.at(0).squared_order_count()),
    _born_particles(born.particle_count()),
    _born_inputs(count_born_inputs(born, extras)) {
    if (reals.empty()) {
        throw std::invalid_argument("FKSChannelIntegrand: no real matrix elements");
    }
    if (real_indices.size() != spin_legs.size() || real_indices.empty()) {
        throw std::invalid_argument(
            "FKSChannelIntegrand: a real matrix element and a spin leg per "
            "configuration"
        );
    }
    for (auto index : real_indices) {
        if (index < 0 || std::cmp_greater_equal(index, reals.size())) {
            throw std::invalid_argument(
                std::format("FKSChannelIntegrand: no real matrix element {}", index)
            );
        }
    }
    for (auto leg : spin_legs) {
        if (leg < -1 || std::cmp_greater_equal(leg, _born_particles)) {
            throw std::invalid_argument(
                std::format("FKSChannelIntegrand: no Born leg {}", leg)
            );
        }
        _has_spin_leg.push_back(leg >= 0 ? 1. : 0.);
    }
    if (color_representations.size() != _born_particles) {
        throw std::invalid_argument(
            "FKSChannelIntegrand: a color representation per Born leg"
        );
    }
    std::vector<Output> born_outputs{
        MatrixElement::squared_orders_out,
        MatrixElement::diagram_amp2_out,
        MatrixElement::spin_correlated_out,
    };
    if (_color.has_color()) {
        born_outputs.push_back(MatrixElement::color_correlated_out);
    }
    check_library(
        born,
        {MatrixElement::momenta_in,
         MatrixElement::alpha_s_in,
         MatrixElement::g_ew_in,
         MatrixElement::spin_legs_in,
         MatrixElement::spin_vectors_in},
        born_outputs,
        "Born"
    );
    if (_color.has_color() && born.color_pair_count() != _color.pair_count()) {
        throw std::invalid_argument(
            "FKSChannelIntegrand: the Born library has another number of color "
            "pairs than the colored legs"
        );
    }
    if (channel >= born.diagram_count()) {
        throw std::invalid_argument(
            std::format("FKSChannelIntegrand: no Born diagram {}", channel)
        );
    }
    for (auto& real : reals) {
        check_library(
            real,
            {MatrixElement::momenta_in,
             MatrixElement::alpha_s_in,
             MatrixElement::g_ew_in},
            {MatrixElement::squared_orders_out},
            "real"
        );
        if (real.squared_order_count() != _nlo_orders ||
            real.particle_count() != _born_particles + 1) {
            throw std::invalid_argument(
                "FKSChannelIntegrand: the reals have the NLO orders and one particle "
                "more than the Born"
            );
        }
    }
    if (loop) {
        check_library(
            *loop,
            {MatrixElement::momenta_in,
             MatrixElement::alpha_s_in,
             MatrixElement::g_ew_in,
             MatrixElement::scale_in},
            {MatrixElement::loop_out},
            "loop"
        );
        if (loop->squared_order_count() != _nlo_orders ||
            loop->particle_count() != _born_particles) {
            throw std::invalid_argument(
                "FKSChannelIntegrand: the loop has the NLO orders and the particles "
                "of the Born"
            );
        }
    }
    if ((!tagged_nlo.empty() && tagged_nlo.size() != _nlo_orders) ||
        (!tagged_born.empty() && tagged_born.size() != _born_inputs) ||
        tagged_nlo.empty() != tagged_born.empty()) {
        throw std::invalid_argument(
            "FKSChannelIntegrand: the factors of tagged photons per NLO order and "
            "per Born input"
        );
    }
    // the extra counterterms
    std::size_t configurations = real_indices.size();
    if ((!extra_indices.empty() && extra_indices.size() != configurations) ||
        extra_spin_legs.size() != extra_indices.size()) {
        throw std::invalid_argument(
            "FKSChannelIntegrand: an extra counterterm and its spin leg per "
            "configuration"
        );
    }
    _extra_tables.assign(extras.size(), std::vector<double>(configurations, 0.));
    for (std::size_t c = 0; c < extra_indices.size(); ++c) {
        auto index = extra_indices.at(c);
        if (index < -1 || std::cmp_greater_equal(index, extras.size())) {
            throw std::invalid_argument(
                std::format("FKSChannelIntegrand: no extra counterterm {}", index)
            );
        }
        auto leg = extra_spin_legs.at(c);
        if ((index >= 0) != (leg >= 0) ||
            std::cmp_greater_equal(leg, _born_particles)) {
            throw std::invalid_argument(
                std::format(
                    "FKSChannelIntegrand: configuration {} needs a Born leg for its "
                    "extra counterterm and only then",
                    c
                )
            );
        }
        if (index >= 0) {
            _extra_tables.at(index).at(c) = 1.;
        }
    }
    for (auto& extra : extras) {
        check_library(
            extra,
            {MatrixElement::momenta_in,
             MatrixElement::alpha_s_in,
             MatrixElement::g_ew_in,
             MatrixElement::spin_legs_in,
             MatrixElement::spin_vectors_in},
            {MatrixElement::squared_orders_out, MatrixElement::spin_correlated_out},
            "extra counterterm"
        );
        if (extra.particle_count() != _born_particles) {
            throw std::invalid_argument(
                "FKSChannelIntegrand: the Born of an extra counterterm has the "
                "particles of the Born"
            );
        }
    }
    // the scale of the user
    if (born_scale.user() && !reference_scale) {
        throw std::invalid_argument(
            "FKSChannelIntegrand: the scales take the scale of the user, which needs "
            "the reference scale"
        );
    }
    if (reference_scale) {
        auto types = reference_scale->arg_types();
        if (reference_scale->type() != CustomFunction::scale || types.size() != 1 ||
            types.at(0).shape != kinematics.return_types().at("momenta").shape) {
            throw std::invalid_argument(
                "FKSChannelIntegrand: the reference scale is a scale of the momenta "
                "of the real process"
            );
        }
    }
    // every argument of the cross section from the kinematics, the blocks or
    // the reference scales
    auto kinematic_types = kinematics.return_types();
    for (auto& name : cross_section.arg_types().keys()) {
        bool reference =
            std::find(reference_names.begin(), reference_names.end(), name) !=
            reference_names.end();
        if (reference && !reference_scale) {
            throw std::invalid_argument(
                std::format(
                    "FKSChannelIntegrand: the cross section takes {}, which needs the "
                    "reference scale",
                    name
                )
            );
        }
        if (!has_key(kinematic_types, name) && !reference &&
            std::find(block_names.begin(), block_names.end(), name) ==
                block_names.end()) {
            throw std::invalid_argument(
                std::format(
                    "FKSChannelIntegrand: the cross section takes {}, which the "
                    "integrand does not give",
                    name
                )
            );
        }
    }
    auto born_type = cross_section.arg_types().at("born");
    if (born_type.shape != std::vector<int>{static_cast<int>(_born_inputs)}) {
        throw std::invalid_argument(
            "FKSChannelIntegrand: the cross section takes other Born inputs than the "
            "Born orders and the Borns of the extra counterterms"
        );
    }
}

NamedVector<Value> FKSChannelIntegrand::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value random = args.at("random");
    Value fks_index = args.at("fks_index");
    auto batch_size = fb.batch_size({random});
    auto full = [&](double value) { return fb.full({value, batch_size}); };
    // a block of a subset of the points, zero for the others
    auto scatter = [&](Value indices, Value source, const std::vector<int>& shape) {
        ValueVec full_args{0., batch_size};
        for (int size : shape) {
            full_args.push_back(static_cast<me_int_t>(size));
        }
        return fb.batch_scatter(indices, fb.full(full_args), source);
    };
    // a constant factor per order, the first dimension after the batch
    auto factors = [&](Value value, const std::vector<double>& per_order) {
        if (per_order.empty()) {
            return value;
        }
        ValueVec columns = fb.unstack(value), scaled;
        for (std::size_t k = 0; k < columns.size(); ++k) {
            // of the shape of the column, a matrix per order for the color
            // correlations
            ValueVec full_args{per_order.at(k), batch_size};
            for (int size : columns.at(k).type.shape) {
                full_args.push_back(static_cast<me_int_t>(size));
            }
            scaled.push_back(fb.mul(columns.at(k), fb.full(full_args)));
        }
        return fb.stack(scaled);
    };

    // the kinematics and k_perp of the collinear counter-event
    std::map<std::string, Value> values;
    auto kinematics = _kinematics.build_function(fb, {random, fks_index});
    auto kinematics_keys = kinematics.keys();
    for (auto [value, name] : zip(kinematics, kinematics_keys)) {
        values[name] = value;
    }
    ValueVec kernel_args{
        values.at("collinear_momenta"), values.at("p_i_hat"), values.at("born_momenta")
    };
    if (has_key(_kernels.arg_types(), "resonance_momentum")) {
        // zero for a channel without resonances
        kernel_args.push_back(
            values.contains("resonance_momentum")
                ? values.at("resonance_momentum")
                : fb.full({0., batch_size, static_cast<me_int_t>(4)})
        );
    }
    kernel_args.push_back(fks_index);
    Value k_perp = _kernels.build_function(fb, kernel_args).at("k_perp");

    // the cuts of the event and of the soft and collinear counter-events
    auto passes = [&](const std::string& momenta) {
        return _cuts.build_function(fb, {values.at(momenta), fks_index}).at("mask");
    };
    Value event_cut = passes("momenta");
    // the soft counter-event only where it exists, at xi = 0 (see
    // FKSCrossSection)
    Value soft_cut =
        fb.mul(passes("soft_momenta"), fb.cut_one(values.at("soft_xi"), 0., 0.));
    Value collinear_cut = passes("collinear_momenta");
    Value born_cut = fb.max(fb.max(event_cut, soft_cut), collinear_cut);

    // the reference scales of the user where the kinematics have a weight,
    // at the soft counter-event also where the Born has one
    if (_reference_scale) {
        auto reference = [&](const std::string& momenta, Value weight) {
            Value points = fb.nonzero(weight);
            Value scale =
                _reference_scale
                    ->build_function(fb, {fb.batch_gather(points, values.at(momenta))})
                    .at("ren_scale");
            return scatter(points, scale, {});
        };
        values["reference_scale"] = reference("momenta", values.at("weight"));
        values["soft_reference_scale"] = reference(
            "soft_momenta",
            fb.max(fb.abs(values.at("soft_det")), fb.abs(values.at("born_weight")))
        );
        values["collinear_reference_scale"] =
            reference("collinear_momenta", values.at("collinear_det"));
    }

    // the reals where the event has a weight and passes the cuts, per real
    // matrix element
    Value real_points = fb.nonzero(fb.mul(values.at("weight"), event_cut));
    Value real_index =
        fb.gather_int(fb.batch_gather(real_points, fks_index), _real_indices);
    Value real_momenta = fb.batch_gather(real_points, values.at("momenta"));
    ValueVec split =
        fb.batch_split_by_index(real_index, static_cast<me_int_t>(_reals.size()));
    ValueVec merged;
    for (std::size_t r = 0; r < _reals.size(); ++r) {
        Value indices = split.at(r);
        Value momenta = fb.batch_gather(indices, real_momenta);
        Value real_batch = fb.batch_size({momenta});
        merged.push_back(
            _reals.at(r)
                .build_function(
                    fb,
                    {momenta,
                     fb.full({NLO_UNIT_ALPHA_S, real_batch}),
                     fb.full({_g_ew, real_batch})}
                )
                .at("squared_orders")
        );
        merged.push_back(indices);
    }
    int o = static_cast<int>(_nlo_orders);
    Value matrix_element = scatter(real_points, fb.batch_merge_by_index(merged), {o});

    // the Born where it has a weight and one of the kinematics passes the
    // cuts, with the spin correlation of the mother of the configuration
    Value born_points = fb.nonzero(fb.mul(values.at("born_weight"), born_cut));
    Value born_momenta = fb.batch_gather(born_points, values.at("born_momenta"));
    Value born_batch = fb.batch_size({born_momenta});
    Value spin_legs = fb.stack(
        {fb.gather_int(fb.batch_gather(born_points, fks_index), _spin_legs),
         fb.full({static_cast<me_int_t>(-1), born_batch})}
    );
    Value spin_vectors = fb.stack(
        {fb.batch_gather(born_points, k_perp),
         fb.full({0., born_batch, static_cast<me_int_t>(4)})}
    );
    auto born_values = _born.build_function(
        fb,
        {born_momenta,
         fb.full({NLO_UNIT_ALPHA_S, born_batch}),
         fb.full({_g_ew, born_batch}),
         spin_legs,
         spin_vectors}
    );
    int b = static_cast<int>(_born_orders);
    int n = static_cast<int>(_born_particles);
    Value born = scatter(born_points, born_values.at("squared_orders"), {b});
    Value amp2 = scatter(
        born_points,
        born_values.at("diagram_amp2"),
        {static_cast<int>(_born.diagram_count())}
    );
    // the color matrix per Born order from the pairs, the Casimirs times the
    // Born on the diagonal and zeros for the colorless legs
    Value color = fb.full({0., batch_size, b, n, n});
    if (_color.has_color()) {
        color = scatter(
            born_points,
            _color.build(
                fb, born_values.at("squared_orders"), born_values.at("color_correlated")
            ),
            {b, n, n}
        );
    }
    Value spin = scatter(born_points, born_values.at("spin_correlated"), {b});
    // the channel weight, zero where the Born is not evaluated
    Value channel_weight = fb.div(
        fb.unstack(amp2).at(_channel), fb.max(fb.reduce_sum(amp2), full(1e-300))
    );

    // the virtual where the soft counter-event passes the cuts, the Born has
    // a weight and the mask asks for it
    Value evaluated_points = fb.nonzero(
        fb.mul(fb.mul(soft_cut, values.at("born_weight")), args.at("virtual_mask"))
    );
    Value loop_momenta = fb.batch_gather(evaluated_points, values.at("born_momenta"));
    Value loop_batch = fb.batch_size({loop_momenta});
    Value evaluated = scatter(evaluated_points, fb.full({1., loop_batch}), {});
    Value virtual_value = fb.full({0., batch_size, static_cast<me_int_t>(o)});
    if (_loop) {
        ValueVec scale_args{values.at("born_momenta")};
        if (_born_scale.user()) {
            scale_args.push_back(values.at("soft_reference_scale"));
        }
        Value ren_scale = _born_scale.build_function(fb, scale_args).at("ren_scale");
        auto loop = _loop->build_function(
            fb,
            {loop_momenta,
             fb.full({NLO_UNIT_ALPHA_S, loop_batch}),
             fb.full({_g_ew, loop_batch}),
             fb.batch_gather(evaluated_points, ren_scale)}
        );
        ValueVec finite;
        for (Value parts : fb.unstack(loop.at("loop"))) {
            finite.push_back(fb.unstack(parts).at(0));
        }
        virtual_value = scatter(evaluated_points, fb.stack(finite), {o});
    }

    // the factors of tagged photons, of the Born orders the first ones of
    // the Born inputs
    Value alpha = full(_alpha * _tagged_alpha);
    matrix_element = factors(matrix_element, _tagged_nlo);
    virtual_value = factors(virtual_value, _tagged_nlo);
    std::vector<double> tagged_orders, tagged_extras;
    if (!_tagged_born.empty()) {
        tagged_orders.assign(_tagged_born.begin(), _tagged_born.begin() + b);
    }
    born = factors(born, tagged_orders);
    spin = factors(spin, tagged_orders);
    color = factors(color, tagged_orders);

    // B / 2 for the mothers without a spin correlation, whose azimuthal
    // terms vanish
    Value has_spin_leg = fb.gather(fks_index, _has_spin_leg);
    Value no_spin_leg = fb.sub(full(1.), has_spin_leg);
    ValueVec spin_columns = fb.unstack(spin), born_columns = fb.unstack(born);
    ValueVec spin_correlated;
    for (std::size_t k = 0; k < spin_columns.size(); ++k) {
        spin_correlated.push_back(fb.add(
            fb.mul(has_spin_leg, spin_columns.at(k)),
            fb.mul(no_spin_leg, fb.mul(full(0.5), born_columns.at(k)))
        ));
    }

    // the Borns of the extra counterterms after the Born orders, where the
    // configuration of the point has one, with the spin correlation at its
    // mother
    ValueVec born_inputs{born}, spin_inputs{fb.stack(spin_correlated)};
    std::size_t column = _born_orders;
    for (std::size_t e = 0; e < _extras.size(); ++e) {
        Value has_extra = fb.gather(fks_index, _extra_tables.at(e));
        Value points =
            fb.nonzero(fb.mul(fb.mul(values.at("born_weight"), born_cut), has_extra));
        Value momenta = fb.batch_gather(points, values.at("born_momenta"));
        Value extra_batch = fb.batch_size({momenta});
        Value legs = fb.stack(
            {fb.gather_int(fb.batch_gather(points, fks_index), _extra_spin_legs),
             fb.full({static_cast<me_int_t>(-1), extra_batch})}
        );
        Value vectors = fb.stack(
            {fb.batch_gather(points, k_perp),
             fb.full({0., extra_batch, static_cast<me_int_t>(4)})}
        );
        auto extra = _extras.at(e).build_function(
            fb,
            {momenta,
             fb.full({NLO_UNIT_ALPHA_S, extra_batch}),
             fb.full({_g_ew, extra_batch}),
             legs,
             vectors}
        );
        std::size_t width = _extras.at(e).squared_order_count();
        int k = static_cast<int>(width);
        if (!_tagged_born.empty()) {
            tagged_extras.assign(
                _tagged_born.begin() + column, _tagged_born.begin() + column + width
            );
        }
        born_inputs.push_back(
            factors(scatter(points, extra.at("squared_orders"), {k}), tagged_extras)
        );
        spin_inputs.push_back(
            factors(scatter(points, extra.at("spin_correlated"), {k}), tagged_extras)
        );
        column += width;
    }

    // the virtual trick: r B everywhere, (V - r B) / f where evaluated
    auto [used, approximations] = virtual_trick(
        fb,
        virtual_value,
        born,
        args.at("virtual_ratio"),
        args.at("virtual_fraction"),
        evaluated
    );

    // the cross section with the virtual of the trick and with its
    // approximation alone
    values["matrix_element"] = matrix_element;
    values["born"] = _extras.empty() ? born : fb.cat(born_inputs);
    values["color_correlated"] = color;
    values["spin_correlated"] =
        _extras.empty() ? spin_inputs.at(0) : fb.cat(spin_inputs);
    values["alpha"] = alpha;
    values["channel_weight"] = channel_weight;
    values["configuration_weight"] = args.at("configuration_weight");
    values["fks_index"] = fks_index;
    auto cross_section = [&](const ValueVec& virtual_value_columns) {
        values["virtual"] = fb.stack(virtual_value_columns);
        ValueVec inputs;
        for (auto& name : _cross_section.arg_types().keys()) {
            inputs.push_back(values.at(name));
        }
        return _cross_section.build_function(fb, inputs);
    };
    auto with_virtual = cross_section(used);
    auto without = cross_section(approximations);

    NamedVector<Value> out;
    auto names = with_virtual.keys();
    for (auto [value, name] : zip(with_virtual, names)) {
        out.push_back(name, value);
    }
    out.push_back(
        "virtual_residual",
        fb.sub(with_virtual.at("cross_section"), without.at("cross_section"))
    );
    out.push_back("virtual", virtual_value);
    out.push_back("born", born);
    out.push_back("evaluated", evaluated);
    for (auto& name : kinematics_names) {
        out.push_back(name, values.at(name));
    }
    return out;
}

namespace {

// the results per point of FKSIntegrand, in the order of the sums of the state
const std::vector<std::string> value_names{
    "weight",
    "lo_weight",
    "virtual_residual",
    "scale_variations",
    "pdf_variations",
    "ew_sudakov_weights",
};

// the kinematics and their weights for the histograms
const std::vector<std::pair<std::string, std::string>> histogram_kinematics{
    {"momenta", "event_weight"},
    {"soft_momenta", "soft_weight"},
    {"collinear_momenta", "collinear_weight"},
    {"soft_collinear_momenta", "soft_collinear_weight"},
    {"born_momenta", "nbody_weight"},
};

NamedVector<Type>
fks_event_types(const FKSChannelIntegrand& integrand, bool madnis_training) {
    if (madnis_training) {
        return {
            {"full_weight", batch_float},
            {"weight", batch_float},
            {"latent", integrand.arg_types().at("random")},
            {"adaptive_prob", batch_float},
            {"fks_index", batch_int},
        };
    }
    auto types = integrand.return_types();
    auto index_map = types.index_map();
    NamedVector<Type> out;
    out.push_back("weight", batch_float);
    for (auto& name : value_names) {
        if (name != "weight" && index_map.contains(name)) {
            out.push_back(name, types.at(name));
        }
    }
    for (std::string name : {"virtual", "born", "evaluated"}) {
        out.push_back(name, types.at(name));
    }
    out.push_back("configuration_weight", batch_float);
    for (auto& [momenta, weight] : histogram_kinematics) {
        out.push_back(momenta, types.at(momenta));
        out.push_back(weight, types.at(weight));
    }
    out.push_back("fks_index", batch_int);
    out.push_back("random", integrand.arg_types().at("random"));
    return out;
}

NamedVector<Type> fks_channel_types(std::size_t random_dim) {
    auto dim = static_cast<int>(random_dim);
    return {
        {"random", batch_float_array(dim)},
        {"x", batch_float_array(dim)},
        {"map_det", batch_float},
        {"fks_index", batch_int},
        {"configuration_weight", batch_float},
        {"virtual_random", batch_float},
        {"indices_acc", batch_int},
    };
}

} // namespace

namespace madspace {

// The channel part, the common part and the concatenation of FKSIntegrand.
class FKSIntegrandPart : public FunctionGenerator {
public:
    enum Part { channel, common, concat };

    FKSIntegrandPart(const FKSIntegrand& integrand, Part part) :
        FunctionGenerator(
            "FKSIntegrandPart",
            [&] {
                if (part == channel) {
                    return NamedVector<Type>{{"batch_size", Type({batch_size})}};
                }
                if (part == common) {
                    return integrand._channel_types;
                }
                NamedVector<Type> args;
                auto keys = integrand._channel_types.keys();
                for (std::string prefix : {"arg1_", "arg2_"}) {
                    for (auto [key, type] : zip(keys, integrand._channel_types)) {
                        args.push_back(prefix + key, type);
                    }
                }
                return args;
            }(),
            part == common ? integrand.return_types() : integrand._channel_types
        ),
        _integrand(integrand),
        _part(part) {}

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override {
        if (_part == channel) {
            return _integrand.build_channel_part(fb, args);
        }
        if (_part == common) {
            return _integrand.build_common_part(fb, args);
        }
        auto keys = return_types().keys();
        std::size_t half = args.size() / 2;
        Value sizes = fb.stack_sizes(
            {fb.batch_size({args.at("arg1_random")}),
             fb.batch_size({args.at("arg2_random")})}
        );
        ValueVec outputs;
        for (std::size_t i = 0; i < half; ++i) {
            auto [cat, cat_sizes] = fb.batch_cat({args.at(i), args.at(half + i)});
            if (keys.at(i) == "indices_acc") {
                outputs.push_back(fb.add_int(fb.offset_indices(sizes, cat_sizes), cat));
            } else {
                outputs.push_back(cat);
            }
        }
        return {keys, outputs};
    }

    const FKSIntegrand& _integrand;
    Part _part;
};

} // namespace madspace

FKSIntegrand::FKSIntegrand(
    const FKSChannelIntegrand& integrand,
    const AdaptiveMapping& adaptive_map,
    const std::string& prefix,
    double virtual_fraction,
    double min_virtual_fraction,
    const std::optional<FKSObservableHistograms>& histograms,
    double floor,
    const AdaptiveDiscrete& discrete,
    bool madnis_training
) :
    FunctionGenerator(
        "FKSIntegrand",
        {{"batch_size", Type({batch_size})}},
        fks_event_types(integrand, madnis_training)
    ),
    _integrand(integrand),
    _adaptive_map(adaptive_map),
    _discrete(discrete),
    _histograms(histograms),
    _probability_name(prefixed_name(prefix, "fks_probabilities")),
    _fraction_name(prefixed_name(prefix, "virtual_fraction")),
    _ratio_name(prefixed_name(prefix, "virtual_ratio")),
    _virtual_fraction(virtual_fraction),
    _min_virtual_fraction(min_virtual_fraction),
    _floor(floor),
    _madnis_training(madnis_training),
    _random_dim(integrand.arg_types().at("random").shape.at(0)),
    _particle_count(integrand.born_particle_count() + 1),
    _configuration_count(integrand.configuration_count()),
    _nlo_orders(integrand.nlo_order_count()),
    _channel_types(fks_channel_types(_random_dim)) {
    std::size_t map_dim = std::visit(
        Overloaded{
            [&](std::monostate) { return _random_dim; },
            [](const VegasMapping& vegas) { return vegas.dimension(); },
            [](const Flow& flow) { return flow.input_dim(); }
        },
        adaptive_map
    );
    if (map_dim != _random_dim) {
        throw std::invalid_argument(
            "FKSIntegrand: the adaptive mapping needs the dimension of the random "
            "numbers"
        );
    }
    if (auto flow = std::get_if<Flow>(&adaptive_map); flow && flow->condition_dim()) {
        throw std::invalid_argument("FKSIntegrand: the flow takes no condition");
    }
    if (auto flow = std::get_if<DiscreteFlow>(&discrete)) {
        if (flow->option_counts() != std::vector<std::size_t>{_configuration_count} ||
            flow->condition_dim() != _random_dim ||
            flow->condition_types().size() != 2) {
            throw std::invalid_argument(
                "FKSIntegrand: the discrete flow needs one dimension of the "
                "configurations, the random numbers as its condition and a prior"
            );
        }
    }
    if (madnis_training && !std::holds_alternative<Flow>(adaptive_map)) {
        throw std::invalid_argument("FKSIntegrand: the training needs a flow");
    }
    if (!(virtual_fraction > 0. && virtual_fraction <= 1.) ||
        !(min_virtual_fraction > 0. && min_virtual_fraction <= 1.)) {
        throw std::invalid_argument(
            "FKSIntegrand: the fractions of the virtual are in (0, 1]"
        );
    }
    if (!(floor >= 0. && floor <= 1.)) {
        throw std::invalid_argument("FKSIntegrand: the floor is in [0, 1]");
    }
    if (histograms) {
        auto index_map = return_types().index_map();
        auto arg_types = histograms->arg_types();
        for (auto [name, type] : zip(arg_types.keys(), arg_types)) {
            if (!index_map.contains(name) ||
                return_types().at(name).shape != type.shape ||
                return_types().at(name).dtype != type.dtype) {
                throw std::invalid_argument(
                    std::format(
                        "FKSIntegrand: the histograms take {}, which the integrand "
                        "does not give",
                        name
                    )
                );
            }
        }
    }
}

void FKSIntegrand::initialize_globals(ContextPtr context) const {
    if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
        vegas->initialize_globals(context);
    }
    if (!context->global_exists(_probability_name)) {
        context->define_global(
            _probability_name, DataType::dt_float64, {_configuration_count}
        );
    }
    if (!context->global_exists(_ratio_name)) {
        context->define_global(_ratio_name, DataType::dt_float64, {_nlo_orders});
    }
    if (!context->global_exists(_fraction_name)) {
        context->define_global(_fraction_name, DataType::dt_float64, {});
    }
    // the outputs of the integration, also for the training
    NLOIterationState(
        fks_event_types(_integrand, false),
        _probability_name,
        _fraction_name,
        _ratio_name,
        _configuration_count,
        _virtual_fraction,
        _min_virtual_fraction,
        _floor
    )
        .write_globals(context);
}

Function FKSIntegrand::channel_function() const {
    return FKSIntegrandPart(*this, FKSIntegrandPart::channel).function();
}

Function FKSIntegrand::common_function() const {
    return FKSIntegrandPart(*this, FKSIntegrandPart::common).function();
}

Function FKSIntegrand::concat_function() const {
    return FKSIntegrandPart(*this, FKSIntegrandPart::concat).function();
}

std::optional<std::string> FKSIntegrand::vegas_grid_name() const {
    if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
        return vegas->grid_name();
    }
    return std::nullopt;
}

std::size_t FKSIntegrand::vegas_dimension() const {
    auto vegas = std::get_if<VegasMapping>(&_adaptive_map);
    return vegas ? vegas->dimension() : 0;
}

std::size_t FKSIntegrand::vegas_bin_count() const {
    auto vegas = std::get_if<VegasMapping>(&_adaptive_map);
    return vegas ? vegas->bin_count() : 0;
}

std::shared_ptr<FunctionGenerator> FKSIntegrand::probability() const {
    return std::make_shared<FKSIntegrandProbability>(*this);
}

const std::vector<std::size_t>& FKSIntegrand::active_flavors() const {
    static const std::vector<std::size_t> none;
    return none;
}

const std::vector<me_int_t>& FKSIntegrand::channel_indices() const {
    static const std::vector<me_int_t> none;
    return none;
}

std::optional<std::pair<Function, std::vector<Histogram>>>
FKSIntegrand::histograms() const {
    if (!_histograms) {
        return std::nullopt;
    }
    std::vector<Histogram> empty;
    for (auto& item : _histograms->histograms()) {
        empty.push_back({
            .name = item.observable.name(),
            .min = item.min,
            .max = item.max,
            .bin_values = std::vector<double>(item.bin_count + 2),
            .bin_errors = std::vector<double>(item.bin_count + 2),
        });
    }
    return std::make_pair(_histograms->function(), empty);
}

std::vector<std::string> FKSIntegrand::histogram_inputs() const {
    if (!_histograms) {
        return {};
    }
    return _histograms->arg_types().keys();
}

std::shared_ptr<IterationState> FKSIntegrand::iteration_state() const {
    if (_madnis_training) {
        return nullptr;
    }
    return std::make_shared<NLOIterationState>(
        return_types(),
        _probability_name,
        _fraction_name,
        _ratio_name,
        _configuration_count,
        _virtual_fraction,
        _min_virtual_fraction,
        _floor,
        std::holds_alternative<std::monostate>(_discrete)
    );
}

NamedVector<Value> FKSIntegrand::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    return build_common_part(fb, build_channel_part(fb, args));
}

NamedVector<Value> FKSIntegrand::build_channel_part(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value batch = args.at("batch_size");
    Value r = fb.random(batch, static_cast<me_int_t>(_random_dim + 2));
    auto [r_choice_rest, virtual_random] = fb.pop(r);
    auto [random, choice_random] = fb.pop(r_choice_rest);
    Value x = random;
    Value map_det = fb.full({1., batch});
    std::visit(
        Overloaded{
            [](std::monostate) {},
            [&](const auto& mapping) {
                auto result = mapping.build_forward(fb, {random}, {});
                x = result.at("data");
                map_det = result.at("det");
            }
        },
        _adaptive_map
    );
    auto probabilities = fb.global(
        _probability_name,
        DataType::dt_float64,
        {static_cast<int>(_configuration_count)}
    );
    Value fks_index, configuration_weight;
    std::visit(
        Overloaded{
            [&](std::monostate) {
                auto [index, weight] =
                    fb.sample_discrete_probs(choice_random, probabilities);
                fks_index = index;
                configuration_weight = weight;
            },
            [&](const DiscreteFlow& flow) {
                Value prior = fb.mul(
                    fb.full({1., batch, static_cast<me_int_t>(_configuration_count)}),
                    probabilities
                );
                auto result = flow.build_forward(fb, {choice_random}, {x, prior});
                fks_index = result.at(0);
                configuration_weight = result.at("det");
            }
        },
        _discrete
    );
    Value indices = fb.nonzero(fb.full({1., batch}));
    return {
        _channel_types.keys(),
        {random, x, map_det, fks_index, configuration_weight, virtual_random, indices}
    };
}

NamedVector<Value> FKSIntegrand::build_common_part(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value x = args.at("x");
    Value det = args.at("map_det");
    Value batch = fb.batch_size({x});
    auto fraction = fb.global(_fraction_name, DataType::dt_float64, {});
    auto ratios =
        fb.global(_ratio_name, DataType::dt_float64, {static_cast<int>(_nlo_orders)});
    // the training takes no virtual
    Value virtual_mask = _madnis_training
        ? fb.full({0., batch})
        : fb.cut_one(args.at("virtual_random"), 0., fraction);
    Value virtual_fraction = fb.mul(fb.full({1., batch}), fraction);
    Value virtual_ratio =
        fb.mul(fb.full({1., batch, static_cast<me_int_t>(_nlo_orders)}), ratios);
    Value configuration_weight = args.at("configuration_weight");
    auto out = _integrand.build_function(
        fb,
        {x,
         args.at("fks_index"),
         configuration_weight,
         virtual_mask,
         virtual_ratio,
         virtual_fraction}
    );
    NamedVector<Value> result;
    if (_madnis_training) {
        Value cross_section = out.at("cross_section");
        result.push_back("full_weight", fb.mul(cross_section, det));
        result.push_back("weight", fb.div(cross_section, configuration_weight));
        result.push_back("latent", x);
        result.push_back(
            "adaptive_prob",
            fb.div(fb.full({1., batch}), fb.mul(det, configuration_weight))
        );
        result.push_back("fks_index", args.at("fks_index"));
        return result;
    }
    auto weighted = [&](const std::string& name) {
        auto& value = out.at(name);
        if (value.type.shape.empty()) {
            return fb.mul(value, det);
        }
        ValueVec columns;
        for (auto& column : fb.unstack(value)) {
            columns.push_back(fb.mul(column, det));
        }
        return fb.stack(columns);
    };
    result.push_back("weight", weighted("cross_section"));
    auto index_map = out.index_map();
    for (auto& name : value_names) {
        if (name != "weight" && index_map.contains(name)) {
            result.push_back(name, weighted(name));
        }
    }
    for (std::string name : {"virtual", "born", "evaluated"}) {
        result.push_back(name, out.at(name));
    }
    result.push_back("configuration_weight", args.at("configuration_weight"));
    for (auto& [momenta, weight] : histogram_kinematics) {
        result.push_back(momenta, out.at(momenta));
        result.push_back(weight, weighted(weight));
    }
    result.push_back("fks_index", args.at("fks_index"));
    result.push_back("random", args.at("random"));
    return result;
}

FKSIntegrandProbability::FKSIntegrandProbability(const FKSIntegrand& integrand) :
    FunctionGenerator(
        "FKSIntegrandProbability",
        {{"latent", batch_float_array(static_cast<int>(integrand._random_dim))},
         {"fks_index", batch_int}},
        {{"prob", batch_float}}
    ),
    _adaptive_map(integrand._adaptive_map),
    _discrete(integrand._discrete),
    _probability_name(integrand._probability_name),
    _configuration_count(integrand._configuration_count) {}

NamedVector<Value> FKSIntegrandProbability::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value latent = args.at("latent");
    Value fks_index = args.at("fks_index");
    Value batch = fb.batch_size({latent});
    ValueVec probs;
    std::visit(
        Overloaded{
            [](std::monostate) {},
            [&](const auto& mapping) {
                probs.push_back(mapping.build_inverse(fb, {latent}, {}).at("det"));
            }
        },
        _adaptive_map
    );
    auto probabilities = fb.global(
        _probability_name,
        DataType::dt_float64,
        {static_cast<int>(_configuration_count)}
    );
    std::visit(
        Overloaded{
            [&](std::monostate) {
                auto [random, prob] =
                    fb.sample_discrete_probs_inverse(fks_index, probabilities);
                probs.push_back(prob);
            },
            [&](const DiscreteFlow& flow) {
                Value prior = fb.mul(
                    fb.full({1., batch, static_cast<me_int_t>(_configuration_count)}),
                    probabilities
                );
                probs.push_back(
                    flow.build_inverse(fb, {fks_index}, {latent, prior}).at("det")
                );
            }
        },
        _discrete
    );
    return {{"prob", fb.product(probs)}};
}
