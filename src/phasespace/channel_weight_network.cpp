#include "madspace/phasespace/channel_weight_network.h"

using namespace madspace;

MomentumPreprocessing::MomentumPreprocessing(std::size_t particle_count) :
    FunctionGenerator(
        "MomentumPreprocessing",
        {batch_four_vec_array(particle_count), batch_float, batch_float},
        {batch_float_array(3 * (particle_count - 2) + 2)}
    ),
    _output_dim(3 * (particle_count - 2) + 2) {}

ValueVec MomentumPreprocessing::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    return {fb.pt_eta_phi_x(args.at(0), args.at(1), args.at(2))};
}

ChannelWeightNetwork::ChannelWeightNetwork(
    std::size_t channel_count,
    std::size_t particle_count,
    std::size_t hidden_dim,
    std::size_t layers,
    MLP::Activation activation,
    const std::string& prefix
) :
    FunctionGenerator(
        "ChannelWeightNetwork",
        {batch_four_vec_array(particle_count),
         batch_float,
         batch_float,
         batch_float_array(channel_count)},
        {batch_float_array(channel_count)}
    ),
    _preprocessing(particle_count),
    _mlp(
        _preprocessing.output_dim(),
        channel_count,
        hidden_dim,
        layers,
        activation,
        prefix
    ),
    _channel_count(channel_count),
    _mask_name(prefixed_name(prefix, "active_channels_mask")) {}

ValueVec ChannelWeightNetwork::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    auto p_ext = args.at(0);
    auto x1 = args.at(1);
    auto x2 = args.at(2);
    auto prior = args.at(3);
    auto mask =
        fb.global(_mask_name, DataType::dt_float, {static_cast<int>(_channel_count)});
    auto net_input = _preprocessing.build_function(fb, {p_ext, x1, x2});
    auto net_output = _mlp.build_function(fb, net_input).at(0);
    return {fb.softmax_prior(net_output, fb.mul(prior, mask))};
}

void ChannelWeightNetwork::initialize_globals(ContextPtr context) const {
    _mlp.initialize_globals(context);

    context->define_global(_mask_name, DataType::dt_float, {_channel_count});
    bool is_cpu = context->device() == cpu_device();
    auto mask_global = context->global(_mask_name);
    Tensor mask;
    if (is_cpu) {
        mask = mask_global;
    } else {
        mask = Tensor(DataType::dt_float, mask_global.shape());
    }
    auto mask_view = mask.view<double, 2>()[0];
    for (std::size_t i = 0; i < mask_view.size(); ++i) {
        mask_view[i] = 1.;
    }
    if (!is_cpu) {
        mask_global.copy_from(mask);
    }
}
