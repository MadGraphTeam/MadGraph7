#include "madspace/phasespace/fks_histograms.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

using namespace madspace;

namespace {

// whether a particle of a process is a QCD parton of its jets, a final-state
// quark up to maxjetflavor or a gluon, as cuts.f takes them
bool is_parton(int pdg, std::size_t index, std::size_t incoming, int maxjetflavor) {
    int flavor = std::abs(pdg);
    return index >= incoming &&
        ((flavor >= 1 && flavor <= maxjetflavor) || flavor == 21);
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
template <typename H>
NamedVector<Type> histogram_returns(const std::vector<H>& histograms) {
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

FKSJets::Layout FKSJets::layout(
    const std::vector<FKSConfiguration>& configurations, bool born, int maxjetflavor
) {
    std::size_t real_count = check_fks_configurations(configurations, "FKSJets");
    std::size_t incoming = configurations.at(0).incoming;
    Layout layout{born ? real_count - 1 : real_count, 0, {}, {}};
    // the particles outside the partons, the same for every configuration
    std::vector<int> kept;
    for (std::size_t c = 0; c < configurations.size(); ++c) {
        const auto& config = configurations.at(c);
        std::vector<int> real(config.pdgs.begin(), config.pdgs.end());
        std::vector<int> pdgs = born ? config.born_pdgs() : real;
        std::size_t partons = 0;
        for (std::size_t i = 0; i < real.size(); ++i) {
            partons += is_parton(real.at(i), i, incoming, maxjetflavor);
        }
        layout.slot_count = std::max(layout.slot_count, partons);
        std::vector<me_int_t> parton;
        std::vector<int> others;
        for (std::size_t i = 0; i < pdgs.size(); ++i) {
            bool p = is_parton(pdgs.at(i), i, incoming, maxjetflavor);
            parton.push_back(p ? 1 : 0);
            if (!p) {
                others.push_back(pdgs.at(i));
            }
        }
        if (c == 0) {
            kept = others;
        } else if (others.size() != kept.size() ||
                   !std::equal(
                       others.begin() + incoming, others.end(), kept.begin() + incoming
                   )) {
            throw std::invalid_argument(
                "FKSJets: the configurations differ outside their QCD partons"
            );
        }
        layout.parton.push_back(parton);
    }
    if (layout.slot_count > parton_max) {
        throw std::invalid_argument(
            "FKSJets: more QCD partons than the jet clustering takes"
        );
    }
    // the Born of the first configuration outside its partons, then the jets
    auto born_pdgs = configurations.at(0).born_pdgs();
    for (std::size_t i = 0; i < born_pdgs.size(); ++i) {
        if (!is_parton(born_pdgs.at(i), i, incoming, maxjetflavor)) {
            layout.pids.push_back(born_pdgs.at(i));
        }
    }
    layout.pids.insert(layout.pids.end(), layout.slot_count, 21);
    return layout;
}

FKSJets::FKSJets(
    const std::vector<FKSConfiguration>& configurations,
    bool born,
    double jetalgo,
    double jetradius,
    double ptj,
    double etaj,
    int maxjetflavor
) :
    FKSJets(
        layout(configurations, born, maxjetflavor),
        born,
        {jetalgo, jetradius, ptj, etaj}
    ) {}

FKSJets::FKSJets(Layout layout, bool born, const std::vector<double>& options) :
    FunctionGenerator(
        "FKSJets",
        {
            {"momenta", batch_four_vec_array(static_cast<int>(layout.particle_count))},
            {"fks_index", batch_int},
        },
        {
            {"jet_momenta", batch_four_vec_array(static_cast<int>(layout.pids.size()))},
            {"jet_count", batch_int},
        }
    ),
    _born(born),
    _particle_count(layout.particle_count),
    _slot_count(layout.slot_count),
    _pids(layout.pids),
    _parton(layout.parton),
    _options(options) {
    if (!(options.at(1) > 0.)) {
        throw std::invalid_argument("FKSJets: the jet radius must be positive");
    }
}

NamedVector<Value> FKSJets::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto [jet_momenta, jet_count] = fb.fks_jets(
        args.at("momenta"),
        args.at("fks_index"),
        _parton,
        _options,
        static_cast<me_int_t>(_pids.size())
    );
    return {{"jet_momenta", jet_momenta}, {"jet_count", jet_count}};
}

FKSHistograms::FKSHistograms(
    const std::vector<HistItem>& histograms, std::size_t kinematics_count
) :
    FunctionGenerator(
        "FKSHistograms",
        [&] {
            NamedVector<Type> args;
            for (std::size_t k = 0; k < kinematics_count; ++k) {
                args.push_back(std::format("weight{}", k), batch_float);
                args.push_back(std::format("jet_count{}", k), batch_int);
            }
            for (std::size_t i = 0; i < histograms.size(); ++i) {
                for (std::size_t k = 0; k < kinematics_count; ++k) {
                    args.push_back(std::format("value{}_{}", i, k), batch_float);
                }
            }
            return args;
        }(),
        histogram_returns(histograms)
    ),
    _histograms(histograms),
    _kinematics_count(kinematics_count) {
    if (histograms.empty()) {
        throw std::invalid_argument("FKSHistograms: no histograms");
    }
    if (kinematics_count == 0 || kinematics_count > 64) {
        throw std::invalid_argument("FKSHistograms: from 1 to 64 kinematics per point");
    }
    for (const auto& histogram : histograms) {
        if (histogram.bin_count == 0 || !(histogram.max > histogram.min)) {
            throw std::invalid_argument(
                "FKSHistograms: a histogram needs bins and max > min"
            );
        }
    }
}

NamedVector<Value> FKSHistograms::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    ValueVec weights, jets;
    for (std::size_t k = 0; k < _kinematics_count; ++k) {
        weights.push_back(args.at(std::format("weight{}", k)));
        jets.push_back(args.at(std::format("jet_count{}", k)));
    }
    Value weight = fb.stack(weights), jet_count = fb.stack(jets);
    NamedVector<Value> result;
    for (std::size_t i = 0; i < _histograms.size(); ++i) {
        const auto& histogram = _histograms.at(i);
        ValueVec values;
        for (std::size_t k = 0; k < _kinematics_count; ++k) {
            values.push_back(args.at(std::format("value{}_{}", i, k)));
        }
        auto [center, merged] = fb.fks_histogram_weights(
            fb.stack(values),
            weight,
            jet_count,
            std::vector<double>{
                histogram.min,
                histogram.max,
                static_cast<double>(histogram.bin_count),
                static_cast<double>(histogram.jet_count),
            }
        );
        // one histogram of the kinematics of all points: a point adds the
        // squares of its sums per bin, as the merged weights have them
        auto [centers, center_counts] = fb.batch_cat(fb.unstack(center));
        auto [sums, sum_counts] = fb.batch_cat(fb.unstack(merged));
        auto [bins, square_bins] = fb.histogram(
            centers,
            sums,
            histogram.min,
            histogram.max,
            static_cast<me_int_t>(histogram.bin_count)
        );
        result.push_back(std::format("values{}", i), bins);
        result.push_back(std::format("square_values{}", i), square_bins);
    }
    return result;
}

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
    _jets(configurations, false, jetalgo, jetradius, ptj, etaj, maxjetflavor),
    _born_jets(configurations, true, jetalgo, jetradius, ptj, etaj, maxjetflavor),
    _binning(
        [&] {
            std::vector<FKSHistograms::HistItem> items;
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
