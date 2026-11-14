#include "madspace/subtraction/fks/histograms.hpp"

#include <format>
#include <stdexcept>

using namespace madspace;

namespace {

// the flavor layouts of the configurations: their real processes, or with
// born their Borns
nested_vector2<int>
layouts(const std::vector<FKSConfiguration>& configurations, bool born) {
    nested_vector2<int> result;
    for (const auto& configuration : configurations) {
        result.push_back(born ? configuration.born_pdgs() : configuration.pdgs);
    }
    return result;
}

// the kinematics that FKSObservableHistograms takes, its momenta and weight
struct Kinematics {
    std::string momenta, weight;
    bool born;
};

const std::vector<Kinematics> KINEMATICS = {
    {"momenta", "event_weight", false},
    {"soft_momenta", "soft_weight", false},
    {"collinear_momenta", "collinear_weight", false},
    {"soft_collinear_momenta", "soft_collinear_weight", false},
    {"born_momenta", "nbody_weight", true},
};

// the outputs of the histograms, as those of ObservableHistograms
NamedVector<Type>
histogram_returns(const std::vector<FKSObservableHistograms::HistItem>& histograms) {
    NamedVector<Type> returns;
    for (std::size_t i = 0; i < histograms.size(); ++i) {
        auto bins = static_cast<int>(histograms.at(i).bin_count) + 2;
        returns.push_back(std::format("values{}", i), single_float_array(bins));
        returns.push_back(std::format("square_values{}", i), single_float_array(bins));
    }
    return returns;
}

bool same_type(const Type& a, const Type& b) {
    return a.dtype == b.dtype && a.batch_size == b.batch_size && a.shape == b.shape;
}

// whether an observable has one value per event, of shape (batch,) or
// (batch, 1) as an Observable of one particle gives it
bool one_value(const Type& type) {
    return type.dtype == batch_float.dtype &&
        type.batch_size == batch_float.batch_size &&
        (type.shape.empty() || type.shape == std::vector<int>{1});
}

} // namespace

const std::vector<std::string> FKSObservableHistograms::kinematics = {
    "event", "soft", "collinear", "soft_collinear", "nbody"
};

FKSObservableHistograms::FKSObservableHistograms(
    const std::vector<FKSConfiguration>& configurations,
    const std::vector<HistItem>& histograms,
    double jetalgo,
    double jetradius,
    double ptj,
    double etaj,
    int maxjetflavor
) :
    FunctionGenerator(
        "FKSObservableHistograms",
        [&] {
            std::size_t n =
                check_fks_configurations(configurations, "FKSObservableHistograms");
            NamedVector<Type> args;
            for (const auto& kinematics : KINEMATICS) {
                auto count = static_cast<int>(kinematics.born ? n - 1 : n);
                args.push_back(kinematics.momenta, batch_four_vec_array(count));
                args.push_back(kinematics.weight, batch_float);
            }
            args.push_back("fks_index", batch_int);
            return args;
        }(),
        histogram_returns(histograms)
    ),
    _histograms(histograms),
    _jets(
        layouts(configurations, false),
        0,
        configurations.at(0).incoming,
        jetalgo,
        jetradius,
        ptj,
        etaj,
        maxjetflavor
    ),
    _born_jets(
        layouts(configurations, true),
        _jets.slot_count(),
        configurations.at(0).incoming,
        jetalgo,
        jetradius,
        ptj,
        etaj,
        maxjetflavor
    ),
    _binning(
        [&] {
            std::vector<NLOHistograms::HistItem> items;
            for (const auto& histogram : histograms) {
                items.push_back(
                    {histogram.min,
                     histogram.max,
                     histogram.bin_count,
                     histogram.jet_count}
                );
            }
            return items;
        }(),
        KINEMATICS.size()
    ) {
    if (_born_jets.pids().size() != _jets.pids().size()) {
        throw std::invalid_argument(
            "FKSObservableHistograms: the real processes and the Born differ "
            "outside their QCD partons"
        );
    }
    auto layout = batch_four_vec_array(static_cast<int>(_jets.jet_particle_count()));
    for (const auto& histogram : histograms) {
        const auto& observable = histogram.observable;
        if (observable.arg_types().size() != 1 ||
            !same_type(observable.arg_types().at(0), layout) ||
            observable.return_types().size() != 1 ||
            !one_value(observable.return_types().at(0))) {
            throw std::invalid_argument(
                "FKSObservableHistograms: an observable needs one value per event "
                "of the momenta with jets"
            );
        }
    }
}

NamedVector<Value> FKSObservableHistograms::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    NamedVector<Value> inputs;
    std::vector<Value> jet_momenta;
    for (std::size_t k = 0; k < KINEMATICS.size(); ++k) {
        const auto& kinematics = KINEMATICS.at(k);
        const auto& jets = kinematics.born ? _born_jets : _jets;
        auto out = jets.build_function(
            fb, {args.at(kinematics.momenta), args.at("fks_index")}
        );
        jet_momenta.push_back(out.at(0));
        inputs.push_back(std::format("weight{}", k), args.at(kinematics.weight));
        inputs.push_back(std::format("jet_count{}", k), out.at(1));
    }
    for (std::size_t i = 0; i < _histograms.size(); ++i) {
        const auto& observable = _histograms.at(i).observable;
        bool squeeze = !observable.return_types().at(0).shape.empty();
        for (std::size_t k = 0; k < KINEMATICS.size(); ++k) {
            auto value = observable.build_function(fb, {jet_momenta.at(k)}).at(0);
            inputs.push_back(
                std::format("value{}_{}", i, k), squeeze ? fb.squeeze(value) : value
            );
        }
    }
    return _binning.build_function(fb, inputs);
}
