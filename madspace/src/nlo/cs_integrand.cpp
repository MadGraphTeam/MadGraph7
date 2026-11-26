#include "madspace/nlo/cs_integrand.hpp"

#include <algorithm>
#include <format>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

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
            "colors have other particles than the Born library"
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
            "CSBornChannelIntegrand: the Born library has another number of color "
            "pairs than the colored legs"
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
                "color correlations"
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
                "CSBornChannelIntegrand: the QCD K and P operators need the color "
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

namespace {

// the index of the pair of two colored particles among the pairs a < b of
// the colored particles of a library, -1 if one is colorless
int color_pair(const std::vector<int>& representations, std::size_t j, std::size_t k) {
    std::vector<std::size_t> colored;
    for (std::size_t leg = 0; leg < representations.size(); ++leg) {
        if (representations[leg] != 1) {
            colored.push_back(leg);
        }
    }
    std::size_t a = std::min(j, k), b = std::max(j, k);
    int index = 0;
    for (std::size_t m = 0; m < colored.size(); ++m) {
        for (std::size_t l = m + 1; l < colored.size(); ++l, ++index) {
            if (colored[m] == a && colored[l] == b) {
                return index;
            }
        }
    }
    return -1;
}

// the flux factor of events of momentum fractions x1, x2 with the conversion
// to pb, 1 / (2 M) for a decay
Value flux_factor(
    FunctionBuilder& fb, Value x1, Value x2, double energy, bool decay, Value batch_size
) {
    auto full = [&](double value) { return fb.full({value, batch_size}); };
    if (decay) {
        return full(1. / (2. * energy));
    }
    Value s_hat = fb.mul(fb.mul(x1, x2), full(energy * energy));
    Value positive = fb.cut_one(
        s_hat,
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::infinity()
    );
    return fb.mul(
        positive,
        fb.div(
            full(inv_gev2_to_pb),
            fb.mul(full(2.), fb.max(s_hat, full(std::numeric_limits<double>::min())))
        )
    );
}

} // namespace

CSRealDipole::CSRealDipole(
    const MatrixElement& born,
    const NLOCuts& cuts,
    const NLOScale& scale,
    const std::vector<std::size_t>& order,
    std::size_t emitter,
    std::size_t spectator,
    const std::vector<int>& color_representations,
    const std::vector<double>& born_powers,
    const std::vector<bool>& corrected,
    bool qed,
    double charge,
    double alpha,
    double factor,
    int spin_leg,
    int swap,
    const std::optional<CustomFunction>& reference_scale
) :
    born(born),
    cuts(cuts),
    scale(scale),
    order(order),
    emitter(emitter),
    spectator(spectator),
    color_representations(color_representations),
    born_powers(born_powers),
    corrected(corrected),
    qed(qed),
    charge(charge),
    alpha(alpha),
    factor(factor),
    spin_leg(spin_leg),
    swap(swap),
    reference_scale(reference_scale),
    color_pair(-1) {
    std::size_t n = born.particle_count();
    std::vector<bool> seen(n, false);
    for (auto particle : order) {
        if (particle >= n || seen[particle]) {
            throw std::invalid_argument(
                "CSRealDipole: the order is no permutation of the particles of the "
                "Born"
            );
        }
        seen[particle] = true;
    }
    if (order.size() != n || emitter >= n || spectator >= n || emitter == spectator ||
        color_representations.size() != n) {
        throw std::invalid_argument(
            "CSRealDipole: the order, the emitter, the spectator or the colors do "
            "not fit the particles of the Born library"
        );
    }
    std::size_t b = born_powers.size();
    if (b == 0 || born.squared_order_count() != b || corrected.size() != b) {
        throw std::invalid_argument(
            "CSRealDipole: a power of g_s and a correction flag per Born order of "
            "the library"
        );
    }
    if (cuts.layouts().size() != 1 || cuts.layouts().at(0).size() != n ||
        scale.particle_count() != n) {
        throw std::invalid_argument(
            "CSRealDipole: the cuts and the scales have other particles than the "
            "Born library"
        );
    }
    if (scale.user() && !reference_scale) {
        throw std::invalid_argument(
            "CSRealDipole: the scales of the user need the reference scale"
        );
    }
    if (spin_leg >= static_cast<int>(n) || swap >= static_cast<int>(n) ||
        (swap >= 0 && spin_leg != swap)) {
        throw std::invalid_argument(
            "CSRealDipole: the spin leg is a particle of the Born, the exchanged "
            "particle if there is one"
        );
    }
    ColorMatrix colors(color_representations, "CSRealDipole");
    std::vector<Output> outputs{
        MatrixElement::squared_orders_out, MatrixElement::diagram_amp2_out
    };
    if (colors.has_color()) {
        outputs.push_back(MatrixElement::color_correlated_out);
    }
    outputs.push_back(MatrixElement::spin_correlated_out);
    if (born.inputs() !=
            std::vector<Input>{
                MatrixElement::momenta_in,
                MatrixElement::alpha_s_in,
                MatrixElement::g_ew_in,
                MatrixElement::spin_legs_in,
                MatrixElement::spin_vectors_in
            } ||
        born.outputs() != outputs) {
        throw std::invalid_argument("CSRealDipole: the Born library has other keys");
    }
    if (!qed) {
        color_pair = ::color_pair(color_representations, emitter, spectator);
        if (color_pair < 0 || born.color_pair_count() != colors.pair_count()) {
            throw std::invalid_argument(
                "CSRealDipole: a dipole of QCD needs a colored emitter and "
                "spectator and the color pairs of the library"
            );
        }
    }
}

CSRealChannelIntegrand::CSRealChannelIntegrand(
    const CSDipoles& dipoles,
    const std::vector<CSRealDipole>& borns,
    const std::vector<CSMapping>& dipole_channels,
    const std::vector<PhaseSpaceMapping>& real_channels,
    std::size_t channel,
    const NLOCuts& cuts,
    const NLOScale& scale,
    const RunningCoupling& running_coupling,
    const NLOLuminosity& luminosity,
    const MatrixElement& real,
    const std::vector<double>& real_powers,
    double g_ew,
    double energy,
    const std::optional<CustomFunction>& reference_scale
) :
    FunctionGenerator(
        "CSRealChannelIntegrand",
        [&] {
            std::size_t dimension = 0;
            if (channel < dipole_channels.size()) {
                dimension = dipole_channels.at(channel).random_dim();
            } else if (channel - dipole_channels.size() < real_channels.size()) {
                dimension =
                    real_channels.at(channel - dipole_channels.size()).random_dim();
            } else {
                throw std::invalid_argument(
                    std::format("CSRealChannelIntegrand: no channel {}", channel)
                );
            }
            return NamedVector<Type>{
                {"random", batch_float_array(static_cast<int>(dimension))}
            };
        }(),
        [&] {
            int d = static_cast<int>(dipoles.dipole_count());
            int n = static_cast<int>(real.particle_count());
            return NamedVector<Type>{
                {"cross_section", batch_float},
                {"real_weight", batch_float},
                {"dipole_weights", batch_float_array(d)},
                {"channel_weight", batch_float},
                {"momenta", batch_four_vec_array(n)},
                {"born_momenta", Type(DataType::dt_float64, batch_size, {d, n - 1, 4})}
            };
        }()
    ),
    _dipoles(dipoles),
    _borns(borns),
    _dipole_channels(dipole_channels),
    _real_channels(real_channels),
    _channel(channel),
    _cuts(cuts),
    _scale(scale),
    _running_coupling(running_coupling),
    _luminosity(luminosity),
    _real(real),
    _real_powers(real_powers),
    _g_ew(g_ew),
    _energy(energy),
    _reference_scale(reference_scale),
    _random_dim(arg_types().at("random").shape.at(0)),
    _particles(real.particle_count()),
    _decay(false) {
    std::size_t n = _particles;
    if (borns.size() != dipoles.dipole_count() || borns.empty()) {
        throw std::invalid_argument(
            "CSRealChannelIntegrand: a Born per dipole, and at least one dipole"
        );
    }
    for (auto& mapping : dipoles.mappings()) {
        if (mapping.particle_count() != n) {
            throw std::invalid_argument(
                "CSRealChannelIntegrand: the dipoles have another real process than "
                "the real library"
            );
        }
    }
    _decay = dipoles.mappings().front().born().is_decay();
    for (auto& mapping : dipole_channels) {
        if (mapping.particle_count() != n || mapping.born().is_decay() != _decay) {
            throw std::invalid_argument(
                "CSRealChannelIntegrand: a channel of a dipole has another real "
                "process"
            );
        }
    }
    for (auto& mapping : real_channels) {
        if (mapping.particle_count() != n || mapping.is_decay() != _decay ||
            mapping.channel_count() > 1) {
            throw std::invalid_argument(
                "CSRealChannelIntegrand: a channel of a diagram has another real "
                "process or permutations"
            );
        }
    }
    for (std::size_t m = 0; m < borns.size(); ++m) {
        if (borns[m].born.particle_count() + 1 != n) {
            throw std::invalid_argument(
                "CSRealChannelIntegrand: a Born of a dipole has not one particle "
                "less than the real process"
            );
        }
    }
    if (real.inputs() !=
            std::vector<Input>{
                MatrixElement::momenta_in,
                MatrixElement::alpha_s_in,
                MatrixElement::g_ew_in
            } ||
        real.outputs() != std::vector<Output>{MatrixElement::squared_orders_out} ||
        real.squared_order_count() != real_powers.size() || real_powers.empty()) {
        throw std::invalid_argument(
            "CSRealChannelIntegrand: the real library has other keys or another "
            "number of orders than the powers of g_s"
        );
    }
    if (cuts.layouts().size() != 1 || cuts.layouts().at(0).size() != n ||
        scale.particle_count() != n) {
        throw std::invalid_argument(
            "CSRealChannelIntegrand: the cuts and the scales have other particles "
            "than the real library"
        );
    }
    if (luminosity.group_count() != 1 || !luminosity.all_groups()) {
        throw std::invalid_argument(
            "CSRealChannelIntegrand: the luminosity of the real emission is one "
            "group with all_groups"
        );
    }
    if (scale.user() && !reference_scale) {
        throw std::invalid_argument(
            "CSRealChannelIntegrand: the scales of the user need the reference scale"
        );
    }
}

NamedVector<Value> CSRealChannelIntegrand::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value random = args.at("random");
    auto batch_size = fb.batch_size({random});
    auto full = [&](double value) { return fb.full({value, batch_size}); };
    auto scatter = [&](Value indices, Value source, const std::vector<int>& shape) {
        ValueVec full_args{0., batch_size};
        for (int size : shape) {
            full_args.push_back(static_cast<me_int_t>(size));
        }
        return fb.batch_scatter(indices, fb.full(full_args), source);
    };
    // one where a value is zero
    auto is_zero = [&](Value value) { return fb.cut_one(value, 0., 0.); };
    constexpr double tiny = std::numeric_limits<double>::min();
    constexpr double huge = std::numeric_limits<double>::max();
    std::size_t d = _borns.size();
    std::size_t dipole_channels = _dipole_channels.size();

    // the real emission of the channel
    auto mapped = _channel < dipole_channels
        ? _dipole_channels.at(_channel).build_forward(fb, {random}, {})
        : _real_channels.at(_channel - dipole_channels).build_forward(fb, {random}, {});
    Value momenta = mapped.at("momenta");
    Value det = mapped.at("det");
    NamedVector<Value> event{
        {"momenta", momenta}, {"x1", mapped.at("x1")}, {"x2", mapped.at("x2")}
    };
    bool dressed = mapped.index_map().contains("omx1");
    if (dressed) {
        event.push_back("omx1", mapped.at("omx1"));
        event.push_back("omx2", mapped.at("omx2"));
    }

    // the densities of the mappings of all channels, zero where a mapping cannot
    // produce the event (a random number of its inverse outside the unit
    // hypercube) or where they are not finite
    ValueVec densities;
    auto add_density = [&](const Mapping& mapping) {
        auto inverse = mapping.build_inverse(fb, event, {});
        Value inverse_det = inverse.at("det");
        Value finite = fb.cut_one(inverse_det, tiny, huge);
        Value safe = fb.max(fb.min(inverse_det, full(huge)), full(0.));
        densities.push_back(
            fb.mul(fb.mul(safe, finite), fb.cut_all(inverse.at("random"), 0., 1.))
        );
    };
    for (auto& mapping : _dipole_channels) {
        add_density(mapping);
    }
    for (auto& mapping : _real_channels) {
        add_density(mapping);
    }
    Value total = fb.reduce_sum(fb.stack(densities));
    Value channel_weight = fb.mul(
        fb.cut_one(total, tiny, huge),
        fb.div(densities.at(_channel), fb.max(total, full(tiny)))
    );

    // the dipoles at the event; a channel of a diagram covers the events where
    // the variables of all dipoles are at or above the technical cut
    auto projected = _dipoles.build_function(fb, event);
    Value keep = fb.sub(full(1.), is_zero(det));
    if (_channel >= dipole_channels) {
        for (Value covered : fb.unstack(projected.at("covered"))) {
            keep = fb.mul(keep, covered);
        }
    }
    ValueVec fractions{event.at("x1"), event.at("x2")};
    if (dressed) {
        fractions.push_back(event.at("omx1"));
        fractions.push_back(event.at("omx2"));
    }
    // the luminosity of the real emission at some points and scales
    auto luminosity = [&](Value points, Value mu1, Value mu2) {
        ValueVec lum_args;
        for (Value fraction : fractions) {
            lum_args.push_back(fb.batch_gather(points, fraction));
        }
        lum_args.push_back(mu1);
        lum_args.push_back(mu2);
        return fb.unstack(_luminosity.build_function(fb, lum_args).at("luminosity"))
            .at(0);
    };
    Value layout = fb.full({static_cast<me_int_t>(0), batch_size});

    // the real emission where it passes its cuts
    Value passes = _cuts.build_function(fb, {momenta, layout}).at("mask");
    Value real_points = fb.nonzero(fb.mul(keep, passes));
    Value real_momenta = fb.batch_gather(real_points, momenta);
    Value real_batch = fb.batch_size({real_momenta});
    ValueVec scale_args{real_momenta};
    if (_reference_scale) {
        scale_args.push_back(
            _reference_scale->build_function(fb, {real_momenta}).at("ren_scale")
        );
    }
    auto scales = _scale.build_function(fb, scale_args);
    auto real_values = _real.build_function(
        fb,
        {real_momenta,
         fb.full({NLO_UNIT_ALPHA_S, real_batch}),
         fb.full({_g_ew, real_batch})}
    );
    Value alpha_s =
        _running_coupling.build_function(fb, {scales.at("ren_scale")}).at("alpha_s");
    Value g_s = fb.sqrt(fb.mul(fb.full({4. * std::numbers::pi, real_batch}), alpha_s));
    Value real_value = fb.full({0., real_batch});
    ValueVec real_columns = fb.unstack(real_values.at("squared_orders"));
    for (std::size_t o = 0; o < _real_powers.size(); ++o) {
        real_value = fb.add(
            real_value,
            fb.mul(
                fb.pow(g_s, fb.full({_real_powers.at(o), real_batch})),
                real_columns.at(o)
            )
        );
    }
    Value real_weight = scatter(
        real_points,
        fb.mul(
            luminosity(real_points, scales.at("fact_scale1"), scales.at("fact_scale2")),
            real_value
        ),
        {}
    );

    // each dipole at its Born where the Born passes its cuts and the dipole has
    // a weight
    ValueVec born_lab = fb.unstack(projected.at("born_momenta"));
    ValueVec color_weights = fb.unstack(projected.at("color_weight"));
    ValueVec spin_weights = fb.unstack(projected.at("spin_weight"));
    ValueVec spin_vectors = fb.unstack(projected.at("spin_vector"));
    ValueVec dipole_weights, born_momenta;
    for (std::size_t m = 0; m < d; ++m) {
        auto& dipole = _borns.at(m);
        // the Born in the order of the library
        ValueVec particles = fb.unstack(born_lab.at(m));
        ValueVec ordered;
        for (auto particle : dipole.order) {
            ordered.push_back(particles.at(particle));
        }
        Value me = fb.stack(ordered);
        born_momenta.push_back(me);
        Value weighted = fb.sub(
            full(1.), fb.mul(is_zero(color_weights.at(m)), is_zero(spin_weights.at(m)))
        );
        Value born_passes = dipole.cuts.build_function(fb, {me, layout}).at("mask");
        Value points = fb.nonzero(fb.mul(fb.mul(keep, weighted), born_passes));
        Value born_points = fb.batch_gather(points, me);
        Value born_batch = fb.batch_size({born_points});
        auto constant = [&](double value) { return fb.full({value, born_batch}); };
        // the spin correlation with the vector of the dipole, at the Born with
        // the emitter and an identical boson exchanged where the dipole takes the
        // correlation of that boson
        Value library_momenta = born_points;
        if (dipole.swap >= 0) {
            ValueVec exchanged = ordered;
            std::swap(exchanged.at(dipole.emitter), exchanged.at(dipole.swap));
            library_momenta = fb.batch_gather(points, fb.stack(exchanged));
        }
        Value vector = fb.batch_gather(points, spin_vectors.at(m));
        Value legs = fb.stack(
            {fb.full({static_cast<me_int_t>(dipole.spin_leg), born_batch}),
             fb.full({static_cast<me_int_t>(-1), born_batch})}
        );
        auto values = dipole.born.build_function(
            fb,
            {library_momenta,
             constant(NLO_UNIT_ALPHA_S),
             constant(_g_ew),
             legs,
             fb.stack({vector, fb.full({0., born_batch, static_cast<me_int_t>(4)})})}
        );
        ValueVec plain = fb.unstack(values.at("squared_orders"));
        ValueVec spin = fb.unstack(values.at("spin_correlated"));
        std::vector<ValueVec> pairs;
        if (!dipole.qed) {
            for (Value per_order : fb.unstack(values.at("color_correlated"))) {
                pairs.push_back(fb.unstack(per_order));
            }
        }
        Value color = fb.mul(
            fb.batch_gather(points, color_weights.at(m)), constant(dipole.factor)
        );
        Value spin_weight = fb.mul(
            fb.batch_gather(points, spin_weights.at(m)), constant(dipole.factor)
        );
        // the scales and alpha_s of the Born
        ValueVec born_scale_args{born_points};
        if (dipole.reference_scale) {
            born_scale_args.push_back(
                dipole.reference_scale->build_function(fb, {born_points})
                    .at("ren_scale")
            );
        }
        auto born_scales = dipole.scale.build_function(fb, born_scale_args);
        Value born_alpha_s =
            _running_coupling.build_function(fb, {born_scales.at("ren_scale")})
                .at("alpha_s");
        Value born_g_s = fb.sqrt(fb.mul(constant(4. * std::numbers::pi), born_alpha_s));
        // g_s^2, or e^2 for QED
        Value coupling = dipole.qed
            ? constant(4. * std::numbers::pi * dipole.alpha)
            : fb.mul(born_g_s, born_g_s);
        Value weight = constant(0.);
        for (std::size_t k = 0; k < dipole.born_powers.size(); ++k) {
            if (!dipole.corrected.at(k)) {
                continue;
            }
            // the color correlation of the emitter and the spectator, or the
            // charges times the Born for QED
            Value colors = dipole.qed
                ? fb.mul(constant(dipole.charge), plain.at(k))
                : pairs.at(k).at(dipole.color_pair);
            Value value = fb.mul(color, colors);
            if (dipole.spin_leg >= 0) {
                Value zero = is_zero(plain.at(k));
                Value ratio = fb.mul(
                    fb.sub(constant(1.), zero),
                    fb.div(colors, fb.add(plain.at(k), zero))
                );
                value = fb.add(value, fb.mul(fb.mul(spin_weight, ratio), spin.at(k)));
            }
            weight = fb.add(
                weight,
                fb.mul(
                    fb.mul(
                        coupling, fb.pow(born_g_s, constant(dipole.born_powers.at(k)))
                    ),
                    value
                )
            );
        }
        Value lum = luminosity(
            points, born_scales.at("fact_scale1"), born_scales.at("fact_scale2")
        );
        dipole_weights.push_back(scatter(points, fb.mul(lum, weight), {}));
    }

    // the normalization: the Jacobian, the weight of the channel and the flux
    Value flux =
        flux_factor(fb, event.at("x1"), event.at("x2"), _energy, _decay, batch_size);
    Value norm = fb.mul(fb.mul(fb.mul(keep, det), channel_weight), flux);
    real_weight = fb.mul(norm, real_weight);
    Value total_weight = real_weight;
    for (auto& weight : dipole_weights) {
        weight = fb.mul(norm, weight);
        total_weight = fb.add(total_weight, weight);
    }
    return {
        {"cross_section", total_weight},
        {"real_weight", real_weight},
        {"dipole_weights", fb.stack(dipole_weights)},
        {"channel_weight", channel_weight},
        {"momenta", momenta},
        {"born_momenta", fb.stack(born_momenta)},
    };
}

namespace {

NamedVector<Type> cs_event_types(
    const CSIntegrand::ChannelIntegrand& integrand,
    bool madnis_training,
    bool born_momenta = false
) {
    auto& generator = std::visit(
        [](const auto& item) -> const FunctionGenerator& { return item; }, integrand
    );
    Type random = generator.arg_types().at("random");
    if (madnis_training) {
        return {
            {"full_weight", batch_float},
            {"weight", batch_float},
            {"latent", random},
            {"adaptive_prob", batch_float},
        };
    }
    auto& types = generator.return_types();
    NamedVector<Type> out;
    out.push_back("weight", batch_float);
    if (std::holds_alternative<CSBornChannelIntegrand>(integrand)) {
        for (std::string name :
             {"lo_weight", "virtual_residual", "virtual", "born", "evaluated"}) {
            out.push_back(name, types.at(name));
        }
    } else {
        for (std::string name : {"real_weight", "dipole_weights"}) {
            out.push_back(name, types.at(name));
        }
    }
    out.push_back("momenta", types.at("momenta"));
    if (born_momenta && std::holds_alternative<CSRealChannelIntegrand>(integrand)) {
        out.push_back("born_momenta", types.at("born_momenta"));
    }
    out.push_back("random", random);
    return out;
}

NamedVector<Type> cs_channel_types(std::size_t random_dim) {
    auto dim = static_cast<int>(random_dim);
    return {
        {"random", batch_float_array(dim)},
        {"x", batch_float_array(dim)},
        {"map_det", batch_float},
        {"virtual_random", batch_float},
        {"indices_acc", batch_int},
    };
}

} // namespace

namespace madspace {

// The channel part, the common part and the concatenation of CSIntegrand.
class CSIntegrandPart : public FunctionGenerator {
public:
    enum Part { channel, common, concat };

    CSIntegrandPart(const CSIntegrand& integrand, Part part) :
        FunctionGenerator(
            "CSIntegrandPart",
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

    const CSIntegrand& _integrand;
    Part _part;
};

} // namespace madspace

CSIntegrand::CSIntegrand(
    const ChannelIntegrand& integrand,
    const AdaptiveMapping& adaptive_map,
    const std::string& prefix,
    double virtual_fraction,
    double min_virtual_fraction,
    const std::optional<CSObservableHistograms>& histograms,
    bool madnis_training
) :
    FunctionGenerator(
        "CSIntegrand",
        {{"batch_size", Type({batch_size})}},
        cs_event_types(integrand, madnis_training, histograms.has_value())
    ),
    _integrand(integrand),
    _adaptive_map(adaptive_map),
    _histograms(histograms),
    _fraction_name(prefixed_name(prefix, "virtual_fraction")),
    _ratio_name(prefixed_name(prefix, "virtual_ratio")),
    _virtual_fraction(virtual_fraction),
    _min_virtual_fraction(min_virtual_fraction),
    _madnis_training(madnis_training),
    _nbody(std::holds_alternative<CSBornChannelIntegrand>(integrand)),
    _random_dim(0),
    _particle_count(0),
    _nlo_orders(0) {
    auto& generator = std::visit(
        [](const auto& item) -> const FunctionGenerator& { return item; }, integrand
    );
    _random_dim = generator.arg_types().at("random").shape.at(0);
    _particle_count = generator.return_types().at("momenta").shape.at(0);
    if (auto born = std::get_if<CSBornChannelIntegrand>(&integrand)) {
        _nlo_orders = born->nlo_order_count();
    }
    _channel_types = cs_channel_types(_random_dim);
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
            "CSIntegrand: the adaptive mapping needs the dimension of the random "
            "numbers"
        );
    }
    if (auto flow = std::get_if<Flow>(&adaptive_map); flow && flow->condition_dim()) {
        throw std::invalid_argument("CSIntegrand: the flow takes no condition");
    }
    if (madnis_training && !std::holds_alternative<Flow>(adaptive_map)) {
        throw std::invalid_argument("CSIntegrand: the training needs a flow");
    }
    if (!(virtual_fraction > 0. && virtual_fraction <= 1.) ||
        !(min_virtual_fraction > 0. && min_virtual_fraction <= 1.)) {
        throw std::invalid_argument(
            "CSIntegrand: the fractions of the virtual are in (0, 1]"
        );
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
                        "CSIntegrand: the histograms take {}, which the integrand "
                        "does not give",
                        name
                    )
                );
            }
        }
    }
}

void CSIntegrand::initialize_globals(ContextPtr context) const {
    if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
        vegas->initialize_globals(context);
    }
    if (!_nbody) {
        return;
    }
    if (!context->global_exists(_ratio_name)) {
        context->define_global(_ratio_name, DataType::dt_float64, {_nlo_orders});
    }
    if (!context->global_exists(_fraction_name)) {
        context->define_global(_fraction_name, DataType::dt_float64, {});
    }
    NLOIterationState(
        cs_event_types(_integrand, false),
        "",
        _fraction_name,
        _ratio_name,
        1,
        _virtual_fraction,
        _min_virtual_fraction,
        0.
    )
        .write_globals(context);
}

Function CSIntegrand::channel_function() const {
    return CSIntegrandPart(*this, CSIntegrandPart::channel).function();
}

Function CSIntegrand::common_function() const {
    return CSIntegrandPart(*this, CSIntegrandPart::common).function();
}

Function CSIntegrand::concat_function() const {
    return CSIntegrandPart(*this, CSIntegrandPart::concat).function();
}

std::optional<std::string> CSIntegrand::vegas_grid_name() const {
    if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
        return vegas->grid_name();
    }
    return std::nullopt;
}

std::size_t CSIntegrand::vegas_dimension() const {
    auto vegas = std::get_if<VegasMapping>(&_adaptive_map);
    return vegas ? vegas->dimension() : 0;
}

std::size_t CSIntegrand::vegas_bin_count() const {
    auto vegas = std::get_if<VegasMapping>(&_adaptive_map);
    return vegas ? vegas->bin_count() : 0;
}

std::shared_ptr<FunctionGenerator> CSIntegrand::probability() const {
    return std::make_shared<CSIntegrandProbability>(*this);
}

const std::vector<std::size_t>& CSIntegrand::active_flavors() const {
    static const std::vector<std::size_t> none;
    return none;
}

const std::vector<me_int_t>& CSIntegrand::channel_indices() const {
    static const std::vector<me_int_t> none;
    return none;
}

std::optional<std::pair<Function, std::vector<Histogram>>>
CSIntegrand::histograms() const {
    if (!_histograms) {
        return std::nullopt;
    }
    std::vector<Histogram> empty;
    for (auto& item : _histograms->histograms()) {
        empty.push_back({
            .name = item.observables.at(0).name(),
            .min = item.min,
            .max = item.max,
            .bin_values = std::vector<double>(item.bin_count + 2),
            .bin_errors = std::vector<double>(item.bin_count + 2),
        });
    }
    return std::make_pair(_histograms->function(), empty);
}

std::vector<std::string> CSIntegrand::histogram_inputs() const {
    if (!_histograms) {
        return {};
    }
    return _histograms->arg_types().keys();
}

std::shared_ptr<IterationState> CSIntegrand::iteration_state() const {
    if (_madnis_training) {
        return nullptr;
    }
    return std::make_shared<NLOIterationState>(
        return_types(),
        "",
        _nbody ? _fraction_name : "",
        _nbody ? _ratio_name : "",
        1,
        _virtual_fraction,
        _min_virtual_fraction,
        0.
    );
}

NamedVector<Value> CSIntegrand::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    return build_common_part(fb, build_channel_part(fb, args));
}

NamedVector<Value> CSIntegrand::build_channel_part(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value batch = args.at("batch_size");
    Value r = fb.random(batch, static_cast<me_int_t>(_random_dim + 1));
    auto [random, virtual_random] = fb.pop(r);
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
    Value indices = fb.nonzero(fb.full({1., batch}));
    return {_channel_types.keys(), {random, x, map_det, virtual_random, indices}};
}

NamedVector<Value> CSIntegrand::build_common_part(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value x = args.at("x");
    Value det = args.at("map_det");
    Value batch = fb.batch_size({x});
    NamedVector<Value> out;
    if (auto born = std::get_if<CSBornChannelIntegrand>(&_integrand)) {
        auto fraction = fb.global(_fraction_name, DataType::dt_float64, {});
        auto ratios = fb.global(
            _ratio_name, DataType::dt_float64, {static_cast<int>(_nlo_orders)}
        );
        // the training takes no virtual
        Value virtual_mask = _madnis_training
            ? fb.full({0., batch})
            : fb.cut_one(args.at("virtual_random"), 0., fraction);
        Value virtual_fraction = fb.mul(fb.full({1., batch}), fraction);
        Value virtual_ratio =
            fb.mul(fb.full({1., batch, static_cast<me_int_t>(_nlo_orders)}), ratios);
        out = born->build_function(
            fb, {x, virtual_mask, virtual_ratio, virtual_fraction}
        );
    } else {
        out = std::get<CSRealChannelIntegrand>(_integrand).build_function(fb, {x});
    }
    Value cross_section = out.at("cross_section");
    NamedVector<Value> result;
    if (_madnis_training) {
        result.push_back("full_weight", fb.mul(cross_section, det));
        result.push_back("weight", cross_section);
        result.push_back("latent", x);
        result.push_back("adaptive_prob", fb.div(fb.full({1., batch}), det));
        return result;
    }
    auto weighted = [&](Value value) {
        if (value.type.shape.empty()) {
            return fb.mul(value, det);
        }
        ValueVec columns;
        for (auto& column : fb.unstack(value)) {
            columns.push_back(fb.mul(column, det));
        }
        return fb.stack(columns);
    };
    result.push_back("weight", weighted(cross_section));
    if (_nbody) {
        result.push_back("lo_weight", weighted(out.at("lo_weight")));
        result.push_back("virtual_residual", weighted(out.at("virtual_residual")));
        for (std::string name : {"virtual", "born", "evaluated"}) {
            result.push_back(name, out.at(name));
        }
    } else {
        result.push_back("real_weight", weighted(out.at("real_weight")));
        result.push_back("dipole_weights", weighted(out.at("dipole_weights")));
    }
    result.push_back("momenta", out.at("momenta"));
    if (_histograms && !_nbody) {
        result.push_back("born_momenta", out.at("born_momenta"));
    }
    result.push_back("random", args.at("random"));
    return result;
}

CSIntegrandProbability::CSIntegrandProbability(const CSIntegrand& integrand) :
    FunctionGenerator(
        "CSIntegrandProbability",
        {{"latent", batch_float_array(static_cast<int>(integrand._random_dim))}},
        {{"prob", batch_float}}
    ),
    _adaptive_map(integrand._adaptive_map) {}

NamedVector<Value> CSIntegrandProbability::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value latent = args.at("latent");
    Value prob = fb.full({1., fb.batch_size({latent})});
    std::visit(
        Overloaded{
            [](std::monostate) {},
            [&](const auto& mapping) {
                prob = mapping.build_inverse(fb, {latent}, {}).at("det");
            }
        },
        _adaptive_map
    );
    return {{"prob", prob}};
}
