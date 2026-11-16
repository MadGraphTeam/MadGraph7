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

// alpha_s of the blocks without the strong coupling, g_s = 1
constexpr double unit_alpha_s = 1. / (4. * std::numbers::pi);

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
    double tagged_alpha
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
    _channel(channel),
    _g_ew(g_ew),
    _alpha(alpha),
    _tagged_nlo(tagged_nlo),
    _tagged_born(tagged_born),
    _tagged_alpha(tagged_alpha),
    _born_orders(born.squared_order_count()),
    _nlo_orders(reals.empty() ? 0 : reals.at(0).squared_order_count()),
    _born_particles(born.particle_count()) {
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
    // the coloured legs with the index of their Casimir
    if (color_representations.size() != _born_particles) {
        throw std::invalid_argument(
            "FKSChannelIntegrand: a colour representation per Born leg"
        );
    }
    std::vector<std::size_t> colored;
    std::vector<std::size_t> casimir_index;
    for (std::size_t leg = 0; leg < _born_particles; ++leg) {
        int representation = color_representations.at(leg);
        if (representation == 1) {
            continue;
        }
        double casimir;
        if (std::abs(representation) == 3) {
            casimir = QCD_CF;
        } else if (representation == 8) {
            casimir = QCD_CA;
        } else {
            throw std::invalid_argument(
                std::format(
                    "FKSChannelIntegrand: no Casimir of the colour representation "
                    "{}",
                    representation
                )
            );
        }
        auto found = std::find(_casimirs.begin(), _casimirs.end(), casimir);
        casimir_index.push_back(static_cast<std::size_t>(found - _casimirs.begin()));
        if (found == _casimirs.end()) {
            _casimirs.push_back(casimir);
        }
        colored.push_back(leg);
    }
    // the rows of the colour matrix from the pairs i < j of the coloured legs,
    // pair (a, c) at a (2 n_c - a - 1) / 2 + c - a - 1, the Casimirs times the
    // Born after them and a zero for the colourless legs
    std::size_t n_c = colored.size();
    std::size_t pairs = n_c * (n_c - 1) / 2;
    me_int_t zero = static_cast<me_int_t>(pairs + _casimirs.size());
    _color_rows.assign(_born_particles, std::vector<me_int_t>(_born_particles, zero));
    for (std::size_t a = 0; a < n_c; ++a) {
        _color_rows.at(colored.at(a)).at(colored.at(a)) =
            static_cast<me_int_t>(pairs + casimir_index.at(a));
        for (std::size_t c = a + 1; c < n_c; ++c) {
            auto pair = static_cast<me_int_t>(a * (2 * n_c - a - 1) / 2 + c - a - 1);
            _color_rows.at(colored.at(a)).at(colored.at(c)) = pair;
            _color_rows.at(colored.at(c)).at(colored.at(a)) = pair;
        }
    }
    _has_color = n_c >= 2;
    std::vector<Output> born_outputs{
        MatrixElement::squared_orders_out,
        MatrixElement::diagram_amp2_out,
        MatrixElement::spin_correlated_out,
    };
    if (_has_color) {
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
    if (_has_color && born.color_pair_count() != pairs) {
        throw std::invalid_argument(
            "FKSChannelIntegrand: the Born library has another number of colour "
            "pairs than the coloured legs"
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
        (!tagged_born.empty() && tagged_born.size() != _born_orders) ||
        tagged_nlo.empty() != tagged_born.empty()) {
        throw std::invalid_argument(
            "FKSChannelIntegrand: the factors of tagged photons per NLO and per Born "
            "order"
        );
    }
    if (born_scale.user()) {
        throw std::invalid_argument("FKSChannelIntegrand: no scale of the user yet");
    }
    // every argument of the cross section from the kinematics or the blocks
    auto kinematic_types = kinematics.return_types();
    for (auto& name : cross_section.arg_types().keys()) {
        if (!has_key(kinematic_types, name) &&
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
    if (born_type.shape != std::vector<int>{static_cast<int>(_born_orders)}) {
        throw std::invalid_argument(
            "FKSChannelIntegrand: the cross section takes Borns of extra "
            "counterterms"
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
            // of the shape of the column, a matrix per order for the colour
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
    Value soft_cut = passes("soft_momenta");
    Value collinear_cut = passes("collinear_momenta");
    Value born_cut = fb.max(fb.max(event_cut, soft_cut), collinear_cut);

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
                     fb.full({unit_alpha_s, real_batch}),
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
         fb.full({unit_alpha_s, born_batch}),
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
    // the colour matrix per Born order from the pairs, the Casimirs times the
    // Born on the diagonal and zeros for the colourless legs
    Value color = fb.full({0., batch_size, b, n, n});
    if (_has_color) {
        ValueVec born_columns = fb.unstack(born_values.at("squared_orders"));
        ValueVec pair_columns = fb.unstack(born_values.at("color_correlated"));
        Value zero = fb.full({0., born_batch});
        ValueVec matrices;
        for (std::size_t k = 0; k < born_columns.size(); ++k) {
            ValueVec diagonal;
            for (double casimir : _casimirs) {
                diagonal.push_back(
                    fb.mul(fb.full({casimir, born_batch}), born_columns.at(k))
                );
            }
            diagonal.push_back(zero);
            Value entries = fb.cat({pair_columns.at(k), fb.stack(diagonal)});
            ValueVec rows;
            for (auto& row : _color_rows) {
                rows.push_back(fb.select(entries, row));
            }
            matrices.push_back(fb.stack(rows));
        }
        color = scatter(born_points, fb.stack(matrices), {b, n, n});
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
        Value ren_scale =
            _born_scale.build_function(fb, {values.at("born_momenta")}).at("ren_scale");
        auto loop = _loop->build_function(
            fb,
            {loop_momenta,
             fb.full({unit_alpha_s, loop_batch}),
             fb.full({_g_ew, loop_batch}),
             fb.batch_gather(evaluated_points, ren_scale)}
        );
        ValueVec finite;
        for (Value parts : fb.unstack(loop.at("loop"))) {
            finite.push_back(fb.unstack(parts).at(0));
        }
        virtual_value = scatter(evaluated_points, fb.stack(finite), {o});
    }

    // the factors of tagged photons
    Value alpha = full(_alpha * _tagged_alpha);
    matrix_element = factors(matrix_element, _tagged_nlo);
    virtual_value = factors(virtual_value, _tagged_nlo);
    born = factors(born, _tagged_born);
    spin = factors(spin, _tagged_born);
    color = factors(color, _tagged_born);

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

    // the virtual trick: r B everywhere, (V - r B) / f where evaluated
    Value born_sum = fb.reduce_sum(born);
    ValueVec ratios = fb.unstack(args.at("virtual_ratio"));
    ValueVec virtual_columns = fb.unstack(virtual_value);
    ValueVec approximations, used;
    for (std::size_t k = 0; k < ratios.size(); ++k) {
        Value approximation = fb.mul(ratios.at(k), born_sum);
        Value residual = fb.mul(
            evaluated,
            fb.div(
                fb.sub(virtual_columns.at(k), approximation),
                args.at("virtual_fraction")
            )
        );
        approximations.push_back(approximation);
        used.push_back(fb.add(approximation, residual));
    }

    // the cross section with the virtual of the trick and with its
    // approximation alone
    values["matrix_element"] = matrix_element;
    values["born"] = born;
    values["color_correlated"] = color;
    values["spin_correlated"] = fb.stack(spin_correlated);
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
