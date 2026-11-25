#include "madspace/nlo/cs_integrand.hpp"

#include <format>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

#include "madspace/util.hpp"

using namespace madspace;

namespace {

using Input = MatrixElement::MatrixElementInput;
using Output = MatrixElement::MatrixElementOutput;

// the conversion of GeV^-2 to pb, INV_GEV2_TO_PB of the kernels
constexpr double inv_gev2_to_pb = 0.38937937217186e9;

void check_library(
    const MatrixElement& library,
    const std::vector<Input>& inputs,
    const std::vector<Output>& outputs,
    const std::string& what
) {
    if (library.inputs() != inputs || library.outputs() != outputs) {
        throw std::invalid_argument(
            std::format("CSBornChannelIntegrand: the {} library has other keys", what)
        );
    }
}

void check_operator(
    std::size_t particles, std::size_t orders, std::size_t n, std::size_t b, bool qed
) {
    if (particles != n || orders != b) {
        throw std::invalid_argument(
            std::format(
                "CSBornChannelIntegrand: the {} operators take other particles or "
                "Born orders",
                qed ? "QED" : "QCD"
            )
        );
    }
}

NamedVector<Type> born_arg_types(const PhaseSpaceMapping& mapping, bool convolves) {
    std::size_t dimension = mapping.random_dim() + (convolves ? 1 : 0);
    return {
        {"random", batch_float_array(static_cast<int>(dimension))},
        {"virtual_mask", batch_float},
        {"virtual_ratio", batch_float_array(0)},
        {"virtual_fraction", batch_float},
    };
}

} // namespace

CSBornChannelIntegrand::CSBornChannelIntegrand(
    const PhaseSpaceMapping& mapping,
    const NLOCuts& cuts,
    const NLOScale& scale,
    const RunningCoupling& running_coupling,
    const NLOLuminosity& luminosity,
    const MatrixElement& born,
    const std::optional<MatrixElement>& loop,
    const std::vector<int>& color_representations,
    std::size_t channel,
    double g_ew,
    double alpha,
    const std::vector<double>& born_powers,
    const std::vector<double>& nlo_powers,
    const std::optional<CSInsertion>& insertion,
    const std::optional<CSInsertion>& insertion_qed,
    const std::vector<bool>& qcd_corrected,
    const std::vector<bool>& qed_corrected,
    const std::optional<CSCollinear>& collinear,
    const std::optional<CSCollinear>& collinear_qed,
    const std::vector<NLOLuminosity>& convolution,
    double energy,
    const std::optional<CustomFunction>& reference_scale
) :
    FunctionGenerator(
        "CSBornChannelIntegrand",
        [&] {
            auto types = born_arg_types(mapping, !convolution.empty());
            types.at("virtual_ratio") =
                batch_float_array(static_cast<int>(nlo_powers.size()));
            return types;
        }(),
        [&] {
            int b = static_cast<int>(born_powers.size());
            int o = static_cast<int>(nlo_powers.size());
            int n = static_cast<int>(born.particle_count());
            return NamedVector<Type>{
                {"cross_section", batch_float},
                {"lo_weight", batch_float},
                {"virtual_residual", batch_float},
                {"virtual", batch_float_array(o)},
                {"born", batch_float_array(b)},
                {"evaluated", batch_float},
                {"momenta", batch_four_vec_array(n)},
            };
        }()
    ),
    _mapping(mapping),
    _cuts(cuts),
    _scale(scale),
    _running_coupling(running_coupling),
    _luminosity(luminosity),
    _born(born),
    _loop(loop),
    _color(color_representations, "CSBornChannelIntegrand"),
    _channel(channel),
    _g_ew(g_ew),
    _alpha(alpha),
    _born_powers(born_powers),
    _nlo_powers(nlo_powers),
    _insertion(insertion),
    _insertion_qed(insertion_qed),
    _qcd_corrected(qcd_corrected),
    _qed_corrected(qed_corrected),
    _collinear(collinear),
    _collinear_qed(collinear_qed),
    _convolution(convolution),
    _energy(energy),
    _reference_scale(reference_scale),
    _random_dim(mapping.random_dim() + (convolution.empty() ? 0 : 1)),
    _particles(born.particle_count()) {
    std::size_t b = born_powers.size();
    if (b == 0 || born.squared_order_count() != b) {
        throw std::invalid_argument(
            "CSBornChannelIntegrand: a power of g_s per Born order of the library"
        );
    }
    if (qcd_corrected.size() != b || qed_corrected.size() != b) {
        throw std::invalid_argument(
            "CSBornChannelIntegrand: a correction flag per Born order"
        );
    }
    if (color_representations.size() != _particles ||
        mapping.particle_count() != _particles || cuts.layouts().size() != 1 ||
        cuts.layouts().at(0).size() != _particles ||
        scale.particle_count() != _particles) {
        throw std::invalid_argument(
            "CSBornChannelIntegrand: the mapping, the cuts, the scales and the "
            "colours have other particles than the Born library"
        );
    }
    if (luminosity.group_count() != 1 || !luminosity.all_groups()) {
        throw std::invalid_argument(
            "CSBornChannelIntegrand: the luminosity of the Born is one group with "
            "all_groups"
        );
    }
    std::vector<Output> born_outputs{
        MatrixElement::squared_orders_out, MatrixElement::diagram_amp2_out
    };
    if (_color.has_color()) {
        born_outputs.push_back(MatrixElement::color_correlated_out);
    }
    check_library(
        born,
        {MatrixElement::momenta_in, MatrixElement::alpha_s_in, MatrixElement::g_ew_in},
        born_outputs,
        "Born"
    );
    if (_color.has_color() && born.color_pair_count() != _color.pair_count()) {
        throw std::invalid_argument(
            "CSBornChannelIntegrand: the Born library has another number of colour "
            "pairs than the coloured legs"
        );
    }
    if (channel >= born.diagram_count()) {
        throw std::invalid_argument(
            std::format("CSBornChannelIntegrand: no Born diagram {}", channel)
        );
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
        if (loop->particle_count() != _particles ||
            loop->squared_order_count() != nlo_powers.size()) {
            throw std::invalid_argument(
                "CSBornChannelIntegrand: the loop library has other particles or "
                "NLO orders"
            );
        }
    }
    if (insertion) {
        if (insertion->qed() || !_color.has_color()) {
            throw std::invalid_argument(
                "CSBornChannelIntegrand: the QCD insertion operator needs the "
                "colour correlations"
            );
        }
        check_operator(
            insertion->particle_count(),
            insertion->born_order_count(),
            _particles,
            b,
            false
        );
    }
    if (insertion_qed) {
        if (!insertion_qed->qed()) {
            throw std::invalid_argument(
                "CSBornChannelIntegrand: the QED insertion operator is not that of QED"
            );
        }
        check_operator(
            insertion_qed->particle_count(),
            insertion_qed->born_order_count(),
            _particles,
            b,
            true
        );
    }
    if (collinear || collinear_qed) {
        if (convolution.size() != 2 || mapping.leptonic() || mapping.is_decay()) {
            throw std::invalid_argument(
                "CSBornChannelIntegrand: K and P need hadron beams and a convolution "
                "luminosity per beam"
            );
        }
    } else if (!convolution.empty()) {
        throw std::invalid_argument(
            "CSBornChannelIntegrand: convolution luminosities without K and P"
        );
    }
    if (collinear) {
        if (collinear->qed() || !_color.has_color()) {
            throw std::invalid_argument(
                "CSBornChannelIntegrand: the QCD K and P operators need the colour "
                "correlations"
            );
        }
        check_operator(
            collinear->particle_count(),
            collinear->born_order_count(),
            _particles,
            b,
            false
        );
    }
    if (collinear_qed) {
        if (!collinear_qed->qed()) {
            throw std::invalid_argument(
                "CSBornChannelIntegrand: the QED K and P operators are not those of "
                "QED"
            );
        }
        check_operator(
            collinear_qed->particle_count(),
            collinear_qed->born_order_count(),
            _particles,
            b,
            true
        );
    }
    // the convolutions: the Born and one group per group of corrections
    std::size_t groups = 1 + (collinear ? 1 : 0) + (collinear_qed ? 1 : 0);
    for (auto& lum : convolution) {
        if (!lum.all_groups() || lum.group_count() != groups) {
            throw std::invalid_argument(
                "CSBornChannelIntegrand: a convolution luminosity has all_groups with "
                "the Born and a group per group of corrections"
            );
        }
    }
    if (scale.user() && !reference_scale) {
        throw std::invalid_argument(
            "CSBornChannelIntegrand: the scales of the user need the reference scale"
        );
    }
}

NamedVector<Value> CSBornChannelIntegrand::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value random = args.at("random");
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
    int b = static_cast<int>(_born_powers.size());
    int o = static_cast<int>(_nlo_powers.size());
    int n = static_cast<int>(_particles);

    // the Born of the channel, then the point of the convolution
    ValueVec columns = fb.unstack(random);
    std::size_t born_dim = _mapping.random_dim();
    ValueVec born_random(columns.begin(), columns.begin() + born_dim);
    auto mapped = _mapping.build_forward(fb, {fb.stack(born_random)}, {});
    Value momenta = mapped.at("momenta");
    Value x1 = mapped.at("x1"), x2 = mapped.at("x2");
    Value det = mapped.at("det");
    bool dressed = mapped.index_map().contains("omx1");

    // the reference scale of the user where the Born has a weight
    ValueVec scale_args{momenta};
    if (_reference_scale) {
        Value points = fb.nonzero(det);
        Value reference =
            _reference_scale->build_function(fb, {fb.batch_gather(points, momenta)})
                .at("ren_scale");
        scale_args.push_back(scatter(points, reference, {}));
    }
    auto scales = _scale.build_function(fb, scale_args);
    Value ren_scale = scales.at("ren_scale");
    Value fact_scale1 = scales.at("fact_scale1");
    Value fact_scale2 = scales.at("fact_scale2");
    Value alpha_s = _running_coupling.build_function(fb, {ren_scale}).at("alpha_s");

    // the Born where it has a weight and passes the cuts
    Value layout = fb.full({static_cast<me_int_t>(0), batch_size});
    Value passes = _cuts.build_function(fb, {momenta, layout}).at("mask");
    // cut_one(det, 0, 0) is one where det is zero
    Value valid = fb.sub(passes, fb.mul(passes, fb.cut_one(det, 0., 0.)));
    Value born_points = fb.nonzero(valid);
    Value born_momenta = fb.batch_gather(born_points, momenta);
    Value born_batch = fb.batch_size({born_momenta});
    auto born_values = _born.build_function(
        fb,
        {born_momenta,
         fb.full({NLO_UNIT_ALPHA_S, born_batch}),
         fb.full({_g_ew, born_batch})}
    );
    Value born = scatter(born_points, born_values.at("squared_orders"), {b});
    Value amp2 = scatter(
        born_points,
        born_values.at("diagram_amp2"),
        {static_cast<int>(_born.diagram_count())}
    );
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
    // the channel weight, zero where the Born is not evaluated
    Value channel_weight = fb.div(
        fb.unstack(amp2).at(_channel), fb.max(fb.reduce_sum(amp2), full(1e-300))
    );

    // the virtual where the Born is evaluated and the mask asks for it
    Value evaluated_points = fb.nonzero(fb.mul(valid, args.at("virtual_mask")));
    Value loop_momenta = fb.batch_gather(evaluated_points, momenta);
    Value loop_batch = fb.batch_size({loop_momenta});
    Value evaluated = scatter(evaluated_points, fb.full({1., loop_batch}), {});
    Value virtual_value = fb.full({0., batch_size, static_cast<me_int_t>(o)});
    if (_loop) {
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
    auto [used, approximations] = virtual_trick(
        fb,
        virtual_value,
        born,
        args.at("virtual_ratio"),
        args.at("virtual_fraction"),
        evaluated
    );

    // the insertion operators, I per Born order
    Value alpha = full(_alpha);
    std::optional<Value> insertion, insertion_qed;
    if (_insertion) {
        insertion =
            _insertion->build_function(fb, {momenta, color, alpha_s, ren_scale})
                .at("insertion");
    }
    if (_insertion_qed) {
        insertion_qed =
            _insertion_qed->build_function(fb, {momenta, born, alpha, ren_scale})
                .at("insertion");
    }

    // the luminosity of the Born
    ValueVec fractions{x1, x2};
    if (dressed) {
        fractions.push_back(mapped.at("omx1"));
        fractions.push_back(mapped.at("omx2"));
    }
    ValueVec lum_args = fractions;
    lum_args.push_back(fact_scale1);
    lum_args.push_back(fact_scale2);
    Value born_lum =
        fb.unstack(_luminosity.build_function(fb, lum_args).at("luminosity")).at(0);

    // K and P: the weights of the luminosities of each beam at x_i / z_i over
    // z_i, zero for x_i > z_i
    std::optional<Value> collinear, collinear_qed;
    ValueVec convolved;
    if (!_convolution.empty()) {
        Value r = columns.at(born_dim);
        Value z;
        if (_collinear) {
            auto out = _collinear->build_function(
                fb, {momenta, born, color, alpha_s, fact_scale1, fact_scale2, x1, x2, r}
            );
            z = out.at("z");
            collinear = out.at("weights");
        }
        if (_collinear_qed) {
            auto out = _collinear_qed->build_function(
                fb, {momenta, born, alpha, fact_scale1, fact_scale2, x1, x2, r}
            );
            z = out.at("z");
            collinear_qed = out.at("weights");
        }
        ValueVec z_columns = fb.unstack(z);
        ValueVec x{x1, x2};
        for (std::size_t leg = 0; leg < 2; ++leg) {
            Value shifted = fb.div(x.at(leg), z_columns.at(leg));
            Value inside =
                fb.cut_one(shifted, -std::numeric_limits<double>::infinity(), 1.);
            Value safe = fb.add(
                fb.mul(inside, shifted), fb.mul(fb.sub(full(1.), inside), full(0.5))
            );
            ValueVec shifted_fractions{x1, x2};
            shifted_fractions.at(leg) = safe;
            Value values =
                _convolution.at(leg)
                    .build_function(
                        fb,
                        {shifted_fractions.at(0),
                         shifted_fractions.at(1),
                         fact_scale1,
                         fact_scale2}
                    )
                    .at("luminosity");
            ValueVec scaled;
            for (Value column : fb.unstack(values)) {
                scaled.push_back(fb.mul(inside, fb.div(column, z_columns.at(leg))));
            }
            convolved.push_back(fb.stack(scaled));
        }
    }
    // per beam the convolutions of a group of corrections, its column after
    // the Born (QCD first)
    auto kp = [&](Value weights, std::size_t group, std::size_t k) {
        ValueVec per_order = fb.unstack(weights);
        ValueVec beams = fb.unstack(per_order.at(k));
        Value sum = full(0.);
        for (std::size_t leg = 0; leg < 2; ++leg) {
            ValueVec w = fb.unstack(beams.at(leg));
            ValueVec lums = fb.unstack(convolved.at(leg));
            sum = fb.add(
                sum,
                fb.add(
                    fb.add(
                        fb.mul(w.at(0), lums.at(0)), fb.mul(w.at(1), lums.at(group))
                    ),
                    fb.mul(w.at(2), born_lum)
                )
            );
        }
        return sum;
    };

    // the flux factor with the conversion to pb, 1 / (2 M) for a decay
    Value flux;
    if (_mapping.is_decay()) {
        flux = full(1. / (2. * _energy));
    } else {
        Value s_hat = fb.mul(fb.mul(x1, x2), full(_energy * _energy));
        Value positive = fb.cut_one(
            s_hat,
            std::numeric_limits<double>::min(),
            std::numeric_limits<double>::infinity()
        );
        flux = fb.mul(
            positive,
            fb.div(
                full(inv_gev2_to_pb),
                fb.mul(
                    full(2.), fb.max(s_hat, full(std::numeric_limits<double>::min()))
                )
            )
        );
    }
    Value norm = fb.mul(fb.mul(fb.mul(valid, det), channel_weight), flux);

    // the weight with the virtual columns of the trick or of its approximation
    Value g_s = fb.sqrt(fb.mul(full(4. * std::numbers::pi), alpha_s));
    ValueVec born_columns = fb.unstack(born);
    ValueVec insertion_columns, insertion_qed_columns;
    if (insertion) {
        insertion_columns = fb.unstack(*insertion);
    }
    if (insertion_qed) {
        insertion_qed_columns = fb.unstack(*insertion_qed);
    }
    Value lo = full(0.), nbody = full(0.);
    for (std::size_t k = 0; k < _born_powers.size(); ++k) {
        Value coupling = fb.pow(g_s, full(_born_powers.at(k)));
        Value born_k = fb.mul(fb.mul(born_lum, coupling), born_columns.at(k));
        lo = fb.add(lo, born_k);
        nbody = fb.add(nbody, born_k);
        auto add_group =
            [&](bool corrected,
                const std::optional<Value>& weights,
                const ValueVec& insertions,
                std::size_t group) {
                if (!corrected) {
                    return;
                }
                if (!insertions.empty()) {
                    nbody = fb.add(
                        nbody, fb.mul(fb.mul(born_lum, coupling), insertions.at(k))
                    );
                }
                if (weights) {
                    nbody = fb.add(nbody, fb.mul(coupling, kp(*weights, group, k)));
                }
            };
        add_group(_qcd_corrected.at(k), collinear, insertion_columns, 1);
        add_group(
            _qed_corrected.at(k),
            collinear_qed,
            insertion_qed_columns,
            collinear ? 2 : 1
        );
    }
    auto weight = [&](const ValueVec& virtual_columns) {
        Value total = nbody;
        for (std::size_t q = 0; q < _nlo_powers.size(); ++q) {
            Value coupling = fb.pow(g_s, full(_nlo_powers.at(q)));
            total = fb.add(
                total, fb.mul(fb.mul(born_lum, coupling), virtual_columns.at(q))
            );
        }
        return fb.mul(norm, total);
    };
    Value with_virtual = weight(used);
    Value without = weight(approximations);

    return {
        {"cross_section", with_virtual},
        {"lo_weight", fb.mul(norm, lo)},
        {"virtual_residual", fb.sub(with_virtual, without)},
        {"virtual", virtual_value},
        {"born", born},
        {"evaluated", evaluated},
        {"momenta", momenta},
    };
}
