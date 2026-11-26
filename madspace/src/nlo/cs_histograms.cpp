#include "madspace/nlo/cs_histograms.hpp"

#include <format>
#include <stdexcept>

using namespace madspace;

namespace {

// the outputs of the histograms, as those of ObservableHistograms
NamedVector<Type>
histogram_returns(const std::vector<CSObservableHistograms::HistItem>& histograms) {
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

NamedVector<Type> histogram_args(const std::vector<NLOJets>& jets, bool real) {
    if (jets.empty() || (!real && jets.size() != 1) || (real && jets.size() < 2)) {
        throw std::invalid_argument(
            "CSObservableHistograms: the n-body stream has one kinematics, the "
            "real stream the real emission and the Borns of its dipoles"
        );
    }
    auto momenta = [](const NLOJets& item) {
        return static_cast<int>(item.particle_count());
    };
    if (!real) {
        return {
            {"momenta", batch_four_vec_array(momenta(jets.at(0)))},
            {"weight", batch_float}
        };
    }
    int dipoles = static_cast<int>(jets.size()) - 1;
    int born = momenta(jets.at(1));
    for (std::size_t k = 1; k < jets.size(); ++k) {
        if (momenta(jets.at(k)) != born || momenta(jets.at(0)) != born + 1) {
            throw std::invalid_argument(
                "CSObservableHistograms: the Borns of the dipoles need one particle "
                "less than the real emission"
            );
        }
    }
    return {
        {"momenta", batch_four_vec_array(born + 1)},
        {"real_weight", batch_float},
        {"born_momenta", Type(DataType::dt_float64, batch_size, {dipoles, born, 4})},
        {"dipole_weights", batch_float_array(dipoles)},
    };
}

} // namespace

CSObservableHistograms::CSObservableHistograms(
    const std::vector<NLOJets>& jets, const std::vector<HistItem>& histograms, bool real
) :
    FunctionGenerator(
        "CSObservableHistograms",
        histogram_args(jets, real),
        histogram_returns(histograms)
    ),
    _histograms(histograms),
    _jets(jets),
    _real(real),
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
        jets.size()
    ) {
    for (const auto& histogram : histograms) {
        if (histogram.observables.size() != jets.size()) {
            throw std::invalid_argument(
                "CSObservableHistograms: a histogram needs one observable per "
                "kinematics"
            );
        }
        for (std::size_t k = 0; k < jets.size(); ++k) {
            const auto& observable = histogram.observables.at(k);
            auto layout =
                batch_four_vec_array(static_cast<int>(jets.at(k).jet_particle_count()));
            if (observable.arg_types().size() != 1 ||
                !same_type(observable.arg_types().at(0), layout) ||
                observable.return_types().size() != 1 ||
                !one_value(observable.return_types().at(0))) {
                throw std::invalid_argument(
                    "CSObservableHistograms: an observable needs one value per "
                    "event of the momenta with the jets of its kinematics"
                );
            }
        }
    }
}

NamedVector<Value> CSObservableHistograms::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    std::vector<Value> momenta{args.at("momenta")};
    std::vector<Value> weights{args.at(_real ? "real_weight" : "weight")};
    if (_real) {
        for (auto& born : fb.unstack(args.at("born_momenta"))) {
            momenta.push_back(born);
        }
        for (auto& weight : fb.unstack(args.at("dipole_weights"))) {
            weights.push_back(weight);
        }
    }
    Value layout = fb.full({static_cast<me_int_t>(0), fb.batch_size({momenta.at(0)})});
    NamedVector<Value> inputs;
    std::vector<Value> jet_momenta;
    for (std::size_t k = 0; k < _jets.size(); ++k) {
        auto out = _jets.at(k).build_function(fb, {momenta.at(k), layout});
        jet_momenta.push_back(out.at(0));
        inputs.push_back(std::format("weight{}", k), weights.at(k));
        inputs.push_back(std::format("jet_count{}", k), out.at(1));
    }
    for (std::size_t i = 0; i < _histograms.size(); ++i) {
        for (std::size_t k = 0; k < _jets.size(); ++k) {
            const auto& observable = _histograms.at(i).observables.at(k);
            bool squeeze = !observable.return_types().at(0).shape.empty();
            auto value = observable.build_function(fb, {jet_momenta.at(k)}).at(0);
            inputs.push_back(
                std::format("value{}_{}", i, k), squeeze ? fb.squeeze(value) : value
            );
        }
    }
    return _binning.build_function(fb, inputs);
}
