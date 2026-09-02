#include "madspace/driver/systematics.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <set>
#include <sstream>

#include "madspace/driver/logger.hpp"

using namespace madspace;
using json = nlohmann::json;

namespace {

bool layout_has(const DataLayout& layout, const std::string& name) {
    for (auto& field : layout.event_layout()) {
        if (field.name == name) {
            return true;
        }
    }
    return false;
}

std::string sanitize_description(const std::string& descr) {
    std::string out;
    for (std::size_t i = 0; i < descr.size(); ++i) {
        if (descr.compare(i, 2, "=>") == 0) {
            out += ";";
            ++i;
        } else if (descr[i] == '>') {
            out += ".gt.";
        } else if (descr[i] == '<') {
            out += ".lt.";
        } else if (descr[i] == '\n') {
            out += " ";
        } else {
            out += descr[i];
        }
    }
    return out;
}

// key of a distinct alpha_s value entering the matrix element re-evaluation
struct AlphaSKey {
    std::size_t alpha_s_index;
    double mur;
    int dyn;
    bool operator<(const AlphaSKey& other) const {
        return std::tie(alpha_s_index, mur, dyn) <
            std::tie(other.alpha_s_index, other.mur, other.dyn);
    }
};

} // namespace

std::string SystematicsCalculator::format_number(double value) {
    std::string str = std::format("{}", value);
    if (str.find('.') == std::string::npos && str.find('e') == std::string::npos &&
        str.find("inf") == std::string::npos && str.find("nan") == std::string::npos) {
        str += ".0";
    }
    return str;
}

std::string SystematicsCalculator::dyn_scale_name(int dyn) {
    switch (dyn) {
    case 1:
        return "sum pt";
    case 2:
        return "HT";
    case 3:
        return "HT/2";
    case 4:
        return "sqrts";
    default:
        return std::format("{}", dyn);
    }
}

double SystematicsCalculator::dynamical_scale(
    int dyn, const std::vector<std::array<double, 4>>& momenta
) {
    // same definitions as kernels/scale.hpp (incoming particles are the first two)
    switch (dyn) {
    case 1: {
        double et_sum = 0.;
        for (std::size_t i = 2; i < momenta.size(); ++i) {
            auto& p = momenta[i];
            double pt2 = p[1] * p[1] + p[2] * p[2];
            double p2 = pt2 + p[3] * p[3];
            if (p2 > 0.) {
                et_sum += p[0] * std::sqrt(pt2 / p2);
            }
        }
        return et_sum;
    }
    case 2:
    case 3: {
        double mt_sum = 0.;
        for (std::size_t i = 2; i < momenta.size(); ++i) {
            auto& p = momenta[i];
            mt_sum += std::sqrt(std::max(0., p[0] * p[0] - p[3] * p[3]));
        }
        return dyn == 2 ? mt_sum : 0.5 * mt_sum;
    }
    case 4: {
        if (momenta.size() < 2) {
            return 0.;
        }
        double e_tot = momenta[0][0] + momenta[1][0];
        double pz_tot = momenta[0][3] + momenta[1][3];
        return std::sqrt(std::max(0., e_tot * e_tot - pz_tot * pz_tot));
    }
    default:
        throw std::invalid_argument(std::format("invalid dynamical scale choice {}", dyn));
    }
}

SystematicsCalculator::SystematicsCalculator(
    const SystematicsConfig& config,
    const std::vector<SubprocessSystArgs>& subproc_args,
    const std::optional<PdfGrid>& nominal_pdf,
    const std::optional<AlphaSGrid>& nominal_alpha_s,
    ContextPtr me_context,
    const std::vector<std::optional<MatrixElement>>& matrix_elements,
    const nested_vector2<me_int_t>& me_flavor_remap
) :
    _config(config),
    _subproc_args(subproc_args),
    _nominal_pdf(nominal_pdf),
    _me_context(me_context) {
    if (!nominal_alpha_s) {
        throw std::invalid_argument(
            "SystematicsCalculator requires the nominal alpha_s grid"
        );
    }
    _alpha_s_grids.push_back(nominal_alpha_s.value());
    if (_config.has_pdf && !_nominal_pdf) {
        throw std::invalid_argument(
            "SystematicsCalculator requires the nominal PDF grid for hadronic beams"
        );
    }
    for (int dyn : _config.dyn_scales) {
        if (dyn < 1 || dyn > 4) {
            throw std::invalid_argument(
                std::format("invalid dynamical scale choice {} (must be 1-4)", dyn)
            );
        }
    }

    // matrix elements for the subprocesses with mixed alpha_s powers
    _matrix_elements.resize(_subproc_args.size());
    if (!matrix_elements.empty()) {
        if (!_me_context) {
            throw std::invalid_argument(
                "SystematicsCalculator: a context is required to evaluate matrix elements"
            );
        }
        if (matrix_elements.size() != _subproc_args.size()) {
            throw std::invalid_argument(
                "SystematicsCalculator: one matrix element per subprocess expected"
            );
        }
        for (std::size_t i = 0; i < matrix_elements.size(); ++i) {
            auto& me = matrix_elements.at(i);
            if (!me || _subproc_args.at(i).qcd_power >= 0) {
                continue;
            }
            std::vector<me_int_t> remap;
            if (i < me_flavor_remap.size()) {
                remap = me_flavor_remap.at(i);
            }
            if (remap.size() < _subproc_args.at(i).beam_pdgs.size()) {
                throw std::invalid_argument(std::format(
                    "SystematicsCalculator: flavor remap of subprocess {} has {} "
                    "entries, {} flavors expected",
                    i,
                    remap.size(),
                    _subproc_args.at(i).beam_pdgs.size()
                ));
            }
            _matrix_elements.at(i) = MatrixElementData{
                .runtime = build_runtime(me->function(), _me_context, false),
                .particle_count = me->particle_count(),
                .flavor_remap = remap,
            };
        }
    }

    _mur_supported = true;
    for (std::size_t i = 0; i < _subproc_args.size(); ++i) {
        if (_subproc_args.at(i).qcd_power < 0 && !_matrix_elements.at(i)) {
            _mur_supported = false;
        }
    }

    // load the varied PDF members; the alpha_s grid of a set is shared by its members
    std::map<std::string, std::size_t> alpha_s_indices;
    for (auto& spec : _config.pdf_members) {
        if (!_config.has_pdf) {
            _warnings.push_back(std::format(
                "PDF variation {} member {} ignored: the beams have no PDF",
                spec.set_name,
                spec.member
            ));
            continue;
        }
        bool same_set = spec.set_lhaid == _config.nominal_lhaid;
        if (!same_set && !_mur_supported) {
            _warnings.push_back(std::format(
                "PDF variation {} member {} ignored: the events have no uniform "
                "alpha_s power and no matrix element to re-evaluate, so the alpha_s "
                "of a different PDF set cannot be applied",
                spec.set_name,
                spec.member
            ));
            continue;
        }
        std::size_t alpha_s_index;
        if (same_set) {
            alpha_s_index = 0;
        } else if (auto search = alpha_s_indices.find(spec.info_file);
                   search != alpha_s_indices.end()) {
            alpha_s_index = search->second;
        } else {
            alpha_s_index = _alpha_s_grids.size();
            _alpha_s_grids.push_back(AlphaSGrid(spec.info_file));
            alpha_s_indices[spec.info_file] = alpha_s_index;
        }
        _members.push_back(spec);
        _member_data.push_back({PdfGrid(spec.grid_file), alpha_s_index});
    }

    build_variations();
    _variation_sums.assign(_variations.size(), 0.);
}

void SystematicsCalculator::build_variations() {
    _variations.clear();
    int id = _config.first_id;

    // scale variations: the (mur, muf) grid at the generation scale, then the
    // same grid (including the central point) for every alternative dynamical
    // scale, as systematics.py does with together = (mur, muf, dyn)
    std::vector<std::pair<double, double>> scales;
    if (_config.together) {
        for (double mur : _config.mur) {
            for (double muf : _config.muf) {
                scales.push_back({mur, muf});
            }
        }
    } else {
        scales.push_back({1., 1.});
        for (double mur : _config.mur) {
            if (mur != 1.) {
                scales.push_back({mur, 1.});
            }
        }
        for (double muf : _config.muf) {
            if (muf != 1.) {
                scales.push_back({1., muf});
            }
        }
    }
    std::vector<int> dyns{-1};
    dyns.insert(dyns.end(), _config.dyn_scales.begin(), _config.dyn_scales.end());
    bool mur_dropped = false;
    for (int dyn : dyns) {
        for (auto [mur, muf] : scales) {
            if (dyn == -1 && mur == 1. && muf == 1.) {
                continue; // the nominal point
            }
            bool changes_mur = mur != 1. || dyn != -1;
            if (changes_mur && !_mur_supported) {
                mur_dropped = true;
                continue;
            }
            if (muf != 1. && !_config.has_pdf) {
                continue;
            }
            _variations.push_back({id++, mur, muf, -1, dyn});
        }
    }
    if (mur_dropped) {
        _warnings.push_back(
            "renormalisation scale variations dropped: the power of alpha_s in |M|^2 "
            "is not the same for all diagrams of at least one subprocess (qcd_power "
            "= -1) and no matrix element is available to re-evaluate it. Only "
            "factorisation scale and PDF member variations are written."
        );
    }

    // PDF variations: when members of the nominal set are requested, the group
    // starts with the nominal member itself so that the group is complete
    bool has_nominal_set_members = std::any_of(
        _members.begin(),
        _members.end(),
        [&](auto& spec) { return spec.set_lhaid == _config.nominal_lhaid; }
    );
    if (has_nominal_set_members) {
        _variations.push_back({id++, 1., 1., -1, -1});
    }
    for (std::size_t i = 0; i < _members.size(); ++i) {
        _variations.push_back({id++, 1., 1., static_cast<int>(i), -1});
    }
}

std::vector<int> SystematicsCalculator::weight_ids() const {
    std::vector<int> ids;
    for (auto& var : _variations) {
        ids.push_back(var.id);
    }
    return ids;
}

std::vector<std::size_t> SystematicsCalculator::scale_variation_indices() const {
    std::vector<std::size_t> indices;
    for (std::size_t k = 0; k < _variations.size(); ++k) {
        if (_variations.at(k).is_scale()) {
            indices.push_back(k);
        }
    }
    return indices;
}

std::vector<PdfGroupInfo> SystematicsCalculator::pdf_groups() const {
    std::map<int, PdfGroupInfo> groups;
    for (std::size_t k = 0; k < _variations.size(); ++k) {
        auto& var = _variations.at(k);
        if (var.is_scale()) {
            continue;
        }
        int set_lhaid, member;
        std::string set_name, error_type;
        if (var.pdf_index == -1) {
            set_lhaid = _config.nominal_lhaid;
            set_name = _config.nominal_set_name;
            error_type = _config.nominal_error_type;
            member = 0;
        } else {
            auto& spec = _members.at(var.pdf_index);
            set_lhaid = spec.set_lhaid;
            set_name = spec.set_name;
            error_type = spec.error_type;
            member = spec.member;
        }
        auto& group = groups[set_lhaid];
        group.set_name = set_name;
        group.set_lhaid = set_lhaid;
        group.error_type = error_type;
        group.members.push_back({member, k});
    }
    std::vector<PdfGroupInfo> result;
    for (auto& [lhaid, group] : groups) {
        std::sort(group.members.begin(), group.members.end());
        result.push_back(group);
    }
    return result;
}

std::tuple<double, double, double> SystematicsCalculator::pdf_uncertainty(
    const std::string& error_type,
    std::optional<double> central_opt,
    const std::vector<std::pair<int, double>>& member_values
) {
    double nan = std::nan("");
    std::vector<double> others;
    std::optional<double> member0;
    for (auto& [member, value] : member_values) {
        if (member == 0) {
            member0 = value;
        } else {
            others.push_back(value);
        }
    }
    double central = member0 ? member0.value() : central_opt.value_or(nan);
    if (error_type == "replicas") {
        if (others.empty()) {
            return {central, nan, nan};
        }
        double mean = 0.;
        for (double v : others) {
            mean += v;
        }
        mean /= others.size();
        double var = 0.;
        for (double v : others) {
            var += (v - mean) * (v - mean);
        }
        double err = others.size() > 1 ? std::sqrt(var / (others.size() - 1)) : 0.;
        return {mean, err, err};
    }
    if (std::isnan(central)) {
        return {nan, nan, nan};
    }
    if (error_type == "hessian") {
        // asymmetric errors from the (2i-1, 2i) eigenvector pairs
        double up = 0., down = 0.;
        for (std::size_t i = 0; i + 1 < others.size(); i += 2) {
            double dp = others[i] - central, dm = others[i + 1] - central;
            double u = std::max({dp, dm, 0.}), d = std::max({-dp, -dm, 0.});
            up += u * u;
            down += d * d;
        }
        return {central, std::sqrt(up), std::sqrt(down)};
    }
    // symmhessian and anything else: add in quadrature
    double sum = 0.;
    for (double v : others) {
        sum += (v - central) * (v - central);
    }
    return {central, std::sqrt(sum), std::sqrt(sum)};
}

double SystematicsCalculator::alpha_s_ratio(
    std::size_t alpha_s_index, double mur, double ren_scale, int qcd_power
) const {
    if (qcd_power == 0 || (mur == 1. && alpha_s_index == 0)) {
        return 1.;
    }
    double alpha_s_nominal = _alpha_s_grids.at(0).interpolate(ren_scale);
    double alpha_s_varied = _alpha_s_grids.at(alpha_s_index).interpolate(mur * ren_scale);
    if (alpha_s_nominal == 0.) {
        return 0.;
    }
    return std::pow(alpha_s_varied / alpha_s_nominal, qcd_power);
}

std::vector<std::array<double, 4>>
SystematicsCalculator::event_momenta(EventBuffer& buffer, std::size_t event_index) const {
    std::vector<std::array<double, 4>> momenta;
    for (std::size_t j = 0; j < buffer.particle_count(); ++j) {
        auto particle = buffer.particle(event_index, j);
        double e = particle.energy();
        if (e == 0. && j >= 2) {
            break; // padding
        }
        momenta.push_back({e, particle.px(), particle.py(), particle.pz()});
    }
    return momenta;
}

std::vector<double> SystematicsCalculator::matrix_elements(
    int subproc,
    EventBuffer& buffer,
    const std::vector<std::size_t>& indices,
    const std::vector<double>& alpha_s
) const {
    auto& me_data = _matrix_elements.at(subproc).value();
    std::size_t count = indices.size();
    std::size_t pc = me_data.particle_count;
    Tensor momenta(DataType::dt_float, {count, pc, 4});
    Tensor alpha_s_tensor(DataType::dt_float, {count});
    Tensor flavor(DataType::dt_int, {count});
    auto mom_view = momenta.view<double, 3>();
    auto alpha_view = alpha_s_tensor.view<double, 1>();
    auto flavor_view = flavor.view<me_int_t, 1>();
    bool has_subproc = layout_has(buffer.layout(), "subprocess_index");
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t event_index = indices.at(i);
        auto event = buffer.event(event_index);
        for (std::size_t j = 0; j < pc; ++j) {
            auto particle = buffer.particle(event_index, j);
            mom_view[i][j][0] = particle.energy();
            mom_view[i][j][1] = particle.px();
            mom_view[i][j][2] = particle.py();
            mom_view[i][j][3] = particle.pz();
        }
        alpha_view[i] = alpha_s.at(i);
        flavor_view[i] = me_data.flavor_remap.at(event.flavor_index());
        (void)has_subproc;
    }
    TensorVec outputs;
    {
        std::lock_guard<std::mutex> lock(_me_mutex);
        outputs = me_data.runtime->run({momenta, alpha_s_tensor, flavor});
    }
    Tensor result = outputs.at(0).cpu().contiguous();
    auto result_view = result.view<double, 1>();
    std::vector<double> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = result_view[i];
    }
    return values;
}

LOReweightInfo
SystematicsCalculator::reweight_info(EventBuffer& buffer, std::size_t event_index) const {
    auto& layout = buffer.layout();
    bool has_beam1 = layout_has(layout, "fact_scale1");
    bool has_beam2 = layout_has(layout, "fact_scale2");
    bool has_subproc = layout_has(layout, "subprocess_index");
    auto event = buffer.event(event_index);
    int subproc = has_subproc ? event.subprocess_index().value() : 0;
    auto& args = _subproc_args.at(subproc);
    auto& pdgs = args.beam_pdgs.at(event.flavor_index());
    LOReweightInfo info{
        .qcd_power = args.qcd_power,
        .ren_scale = event.ren_scale(),
        .has_beam1 = has_beam1,
        .has_beam2 = has_beam2,
        .pdg1 = pdgs.at(0),
        .pdg2 = pdgs.at(1),
        .x1 = has_beam1 ? event.x1().value() : 0.,
        .x2 = has_beam2 ? event.x2().value() : 0.,
        .fact_scale1 = has_beam1 ? event.fact_scale1().value() : 0.,
        .fact_scale2 = has_beam2 ? event.fact_scale2().value() : 0.,
    };
    return info;
}

void SystematicsCalculator::compute(
    EventBuffer& buffer, std::vector<double>& weights
) const {
    std::size_t count = buffer.event_count();
    std::size_t var_count = _variations.size();
    weights.resize(count * var_count);
    if (var_count == 0) {
        return;
    }
    auto& layout = buffer.layout();
    bool has_beam1 = layout_has(layout, "fact_scale1");
    bool has_beam2 = layout_has(layout, "fact_scale2");
    bool has_partial = layout_has(layout, "partial_weight_product");
    bool has_subproc = layout_has(layout, "subprocess_index");
    bool use_pdf = _config.has_pdf && (has_beam1 || has_beam2);
    bool has_dyn = !_config.dyn_scales.empty();

    // The alpha_s dependence of the events of subprocesses with mixed alpha_s
    // powers comes from re-evaluating the matrix element: collect these events
    // per subprocess and evaluate them at the nominal and every varied alpha_s.
    std::vector<AlphaSKey> me_keys;
    for (auto& var : _variations) {
        std::size_t alpha_s_index =
            var.pdf_index >= 0 ? _member_data.at(var.pdf_index).alpha_s_index : 0;
        if (var.mur != 1. || var.dyn != -1 || alpha_s_index != 0) {
            AlphaSKey key{alpha_s_index, var.mur, var.dyn};
            if (std::find_if(me_keys.begin(), me_keys.end(), [&](auto& k) {
                    return !(k < key) && !(key < k);
                }) == me_keys.end()) {
                me_keys.push_back(key);
            }
        }
    }
    // me_ratios[event][key index] = |M|^2(varied alpha_s) / |M|^2(nominal alpha_s)
    std::vector<std::vector<double>> me_ratios(count);
    std::vector<std::vector<double>> dyn_scale_values(count);
    if (has_dyn) {
        for (std::size_t i = 0; i < count; ++i) {
            auto momenta = event_momenta(buffer, i);
            dyn_scale_values[i].resize(5, 0.);
            for (int dyn : _config.dyn_scales) {
                dyn_scale_values[i][dyn] = dynamical_scale(dyn, momenta);
            }
        }
    }
    auto scale_of = [&](std::size_t i, int dyn, double generated) {
        return dyn == -1 ? generated : dyn_scale_values[i][dyn];
    };
    if (!me_keys.empty()) {
        std::map<int, std::vector<std::size_t>> me_events;
        for (std::size_t i = 0; i < count; ++i) {
            auto event = buffer.event(i);
            int subproc = has_subproc ? event.subprocess_index().value() : 0;
            if (_subproc_args.at(subproc).qcd_power < 0 && _matrix_elements.at(subproc)) {
                me_events[subproc].push_back(i);
            }
        }
        for (auto& [subproc, indices] : me_events) {
            std::vector<double> alpha_s(indices.size());
            for (std::size_t n = 0; n < indices.size(); ++n) {
                alpha_s[n] = buffer.event(indices[n]).alpha_qcd();
            }
            auto nominal = matrix_elements(subproc, buffer, indices, alpha_s);
            for (std::size_t n = 0; n < indices.size(); ++n) {
                me_ratios[indices[n]].assign(me_keys.size(), 1.);
            }
            for (std::size_t k = 0; k < me_keys.size(); ++k) {
                auto& key = me_keys[k];
                for (std::size_t n = 0; n < indices.size(); ++n) {
                    auto event = buffer.event(indices[n]);
                    double mu = scale_of(indices[n], key.dyn, event.ren_scale());
                    alpha_s[n] = _alpha_s_grids.at(key.alpha_s_index).interpolate(key.mur * mu);
                }
                auto varied = matrix_elements(subproc, buffer, indices, alpha_s);
                for (std::size_t n = 0; n < indices.size(); ++n) {
                    me_ratios[indices[n]][k] =
                        nominal[n] == 0. ? 0. : varied[n] / nominal[n];
                }
            }
        }
    }
    auto me_key_index = [&](std::size_t alpha_s_index, double mur, int dyn) {
        AlphaSKey key{alpha_s_index, mur, dyn};
        for (std::size_t k = 0; k < me_keys.size(); ++k) {
            if (!(me_keys[k] < key) && !(key < me_keys[k])) {
                return k;
            }
        }
        throw std::logic_error("alpha_s key not found");
    };

    for (std::size_t i = 0; i < count; ++i) {
        auto event = buffer.event(i);
        double w0 = event.weight();
        int subproc = has_subproc ? event.subprocess_index().value() : 0;
        auto& args = _subproc_args.at(subproc);
        bool use_me = args.qcd_power < 0 && _matrix_elements.at(subproc);
        double ren_scale = event.ren_scale();
        double x1 = has_beam1 ? event.x1().value() : 0.;
        double x2 = has_beam2 ? event.x2().value() : 0.;
        double muf1 = has_beam1 ? event.fact_scale1().value() : 0.;
        double muf2 = has_beam2 ? event.fact_scale2().value() : 0.;
        int pdg1 = 0, pdg2 = 0;
        std::size_t pid1 = 0, pid2 = 0;
        double nominal_product = 1.;
        if (use_pdf) {
            auto& pdgs = args.beam_pdgs.at(event.flavor_index());
            pdg1 = pdgs.at(0);
            pdg2 = pdgs.at(1);
            pid1 = has_beam1 ? _nominal_pdf->pid_index(pdg1) : 0;
            pid2 = has_beam2 ? _nominal_pdf->pid_index(pdg2) : 0;
            if (has_partial) {
                nominal_product = event.partial_weight_product();
            } else {
                nominal_product =
                    (has_beam1 ? _nominal_pdf->interpolate(pid1, x1, muf1) : 1.) *
                    (has_beam2 ? _nominal_pdf->interpolate(pid2, x2, muf2) : 1.);
            }
        }

        for (std::size_t k = 0; k < var_count; ++k) {
            auto& var = _variations.at(k);
            double w = w0;
            if (var.is_nominal()) {
                weights[i * var_count + k] = w;
                continue;
            }
            const PdfGrid* grid = _nominal_pdf ? &_nominal_pdf.value() : nullptr;
            std::size_t alpha_s_index = 0;
            std::size_t member_pid1 = pid1, member_pid2 = pid2;
            if (var.pdf_index >= 0) {
                auto& member = _member_data.at(var.pdf_index);
                grid = &member.grid;
                alpha_s_index = member.alpha_s_index;
                if (use_pdf) {
                    member_pid1 = has_beam1 ? grid->pid_index(pdg1) : 0;
                    member_pid2 = has_beam2 ? grid->pid_index(pdg2) : 0;
                }
            }
            // renormalisation scale / alpha_s part
            if (var.mur != 1. || var.dyn != -1 || alpha_s_index != 0) {
                if (use_me) {
                    w *= me_ratios[i].at(me_key_index(alpha_s_index, var.mur, var.dyn));
                } else {
                    double mu_r = scale_of(i, var.dyn, ren_scale);
                    if (var.dyn == -1) {
                        w *= alpha_s_ratio(alpha_s_index, var.mur, ren_scale, args.qcd_power);
                    } else if (args.qcd_power != 0) {
                        double a0 = _alpha_s_grids.at(0).interpolate(ren_scale);
                        double a1 = _alpha_s_grids.at(alpha_s_index).interpolate(var.mur * mu_r);
                        w *= a0 == 0. ? 0. : std::pow(a1 / a0, args.qcd_power);
                    }
                }
            }
            // factorisation scale / PDF part
            if (use_pdf && w != 0.) {
                if (nominal_product == 0.) {
                    w = 0.;
                } else {
                    double scale1 = var.muf * scale_of(i, var.dyn, muf1);
                    double scale2 = var.muf * scale_of(i, var.dyn, muf2);
                    double varied =
                        (has_beam1 ? grid->interpolate(member_pid1, x1, scale1) : 1.) *
                        (has_beam2 ? grid->interpolate(member_pid2, x2, scale2) : 1.);
                    w *= varied / nominal_product;
                }
            }
            weights[i * var_count + k] = w;
        }
    }
}

void SystematicsCalculator::accumulate(
    EventBuffer& buffer, const std::vector<double>& weights
) {
    std::size_t count = buffer.event_count();
    std::size_t var_count = _variations.size();
    double nominal_sum = 0.;
    std::vector<double> sums(var_count, 0.);
    for (std::size_t i = 0; i < count; ++i) {
        nominal_sum += buffer.event(i).weight();
        for (std::size_t k = 0; k < var_count; ++k) {
            sums[k] += weights.at(i * var_count + k);
        }
    }
    std::lock_guard<std::mutex> lock(_accumulate_mutex);
    _nominal_sum += nominal_sum;
    for (std::size_t k = 0; k < var_count; ++k) {
        _variation_sums[k] += sums[k];
    }
    _event_count += count;
}

std::string SystematicsCalculator::initrwgt() const {
    std::string text;
    bool in_scale = false;
    int in_pdf = -1;
    for (auto& var : _variations) {
        bool is_scale = var.is_scale();
        if (is_scale) {
            if (!in_scale) {
                text += "<weightgroup name=\"Central scale variation\" "
                        "combine=\"envelope\">\n";
                in_scale = true;
            }
        } else if (in_scale) {
            text += "</weightgroup> # scale\n";
            in_scale = false;
        }

        std::string tag, info;
        int pdf_lhaid;
        if (!is_scale) {
            // PDF group: the nominal member (pdf_index -1) belongs to the
            // nominal set's group
            int set_lhaid;
            std::string set_name, error_type, description;
            if (var.pdf_index == -1) {
                set_lhaid = _config.nominal_lhaid;
                set_name = _config.nominal_set_name;
                error_type = _config.nominal_error_type;
                description = _config.nominal_description;
                pdf_lhaid = set_lhaid;
            } else {
                auto& spec = _members.at(var.pdf_index);
                set_lhaid = spec.set_lhaid;
                set_name = spec.set_name;
                error_type = spec.error_type;
                description = spec.description;
                pdf_lhaid = spec.set_lhaid + spec.member;
                info = std::format("PDF={} MemberID={}", spec.set_lhaid, spec.member);
            }
            if (in_pdf != set_lhaid) {
                if (in_pdf != -1) {
                    text += "</weightgroup> # PDFSET to PDFSET\n";
                }
                text += std::format(
                    "<weightgroup name=\"{}\" combine=\"{}\"> # {}: {}\n",
                    set_name,
                    error_type,
                    set_lhaid,
                    sanitize_description(description)
                );
                in_pdf = set_lhaid;
            }
        } else {
            if (in_pdf != -1) {
                text += "</weightgroup> # PDF\n";
                in_pdf = -1;
            }
            pdf_lhaid = _config.nominal_lhaid;
            if (var.mur != 1.) {
                info += std::format("MUR={} ", format_number(var.mur));
            }
            if (var.muf != 1.) {
                info += std::format("MUF={} ", format_number(var.muf));
            }
            if (var.dyn != -1) {
                info += std::format("dyn_scale_choice={} ", dyn_scale_name(var.dyn));
            }
        }
        tag = std::format(
            "MUR=\"{}\" MUF=\"{}\" ", format_number(var.mur), format_number(var.muf)
        );
        if (var.dyn != -1) {
            tag += std::format("DYN_SCALE=\"{}\" ", var.dyn);
        }
        tag += std::format("PDF=\"{}\" ", pdf_lhaid);
        text += std::format("<weight id=\"{}\" {}> {} </weight>\n", var.id, tag, info);
    }
    if (in_scale || in_pdf != -1) {
        text += "</weightgroup>\n";
    }
    return text;
}

json SystematicsCalculator::summary() const {
    json variations = json::array();
    for (std::size_t k = 0; k < _variations.size(); ++k) {
        auto& var = _variations.at(k);
        json j = var;
        if (var.pdf_index == -1) {
            j["pdf_set"] = _config.nominal_set_name;
            j["pdf_lhaid"] = _config.nominal_lhaid;
            j["pdf_member"] = 0;
        } else {
            auto& spec = _members.at(var.pdf_index);
            j["pdf_set"] = spec.set_name;
            j["pdf_lhaid"] = spec.set_lhaid;
            j["pdf_member"] = spec.member;
        }
        if (_event_count > 0) {
            // the combined events carry "average" normalised weights (each event
            // weight estimates the total cross section), so the cross section of
            // a variation is the mean of its weights
            j["cross_section"] = _variation_sums.at(k) / _event_count;
        }
        variations.push_back(j);
    }

    json result{
        {"variations", variations},
        {"nominal",
         {{"pdf_set", _config.nominal_set_name},
          {"pdf_lhaid", _config.nominal_lhaid}}},
        {"event_count", _event_count},
        {"warnings", _warnings},
    };
    if (_event_count == 0) {
        return result;
    }
    double nominal_xsec = _nominal_sum / _event_count;
    result["nominal"]["cross_section"] = nominal_xsec;
    std::vector<double> variation_xsecs(_variation_sums.size());
    for (std::size_t k = 0; k < _variation_sums.size(); ++k) {
        variation_xsecs[k] = _variation_sums[k] / _event_count;
    }

    // scale envelope
    auto scale_indices = scale_variation_indices();
    if (!scale_indices.empty()) {
        json scale_ids = json::array();
        double scale_min = nominal_xsec, scale_max = nominal_xsec;
        for (std::size_t k : scale_indices) {
            scale_ids.push_back(_variations.at(k).id);
            scale_min = std::min(scale_min, variation_xsecs.at(k));
            scale_max = std::max(scale_max, variation_xsecs.at(k));
        }
        result["scale"] = {
            {"weight_ids", scale_ids},
            {"min", scale_min},
            {"max", scale_max},
        };
    }

    // PDF uncertainty per set
    json pdf_sets = json::array();
    for (auto& group : pdf_groups()) {
        std::vector<std::pair<int, double>> member_values;
        json ids = json::array();
        for (auto [member, k] : group.members) {
            member_values.push_back({member, variation_xsecs.at(k)});
            ids.push_back(_variations.at(k).id);
        }
        std::optional<double> central;
        if (group.set_lhaid == _config.nominal_lhaid) {
            central = nominal_xsec;
        }
        auto [c, up, down] = pdf_uncertainty(group.error_type, central, member_values);
        json entry{
            {"pdf_set", group.set_name},
            {"pdf_lhaid", group.set_lhaid},
            {"error_type", group.error_type},
            {"weight_ids", ids},
        };
        if (!std::isnan(c)) {
            entry["central"] = c;
        }
        if (!std::isnan(up)) {
            entry["uncertainty_up"] = up;
            entry["uncertainty_down"] = down;
        }
        pdf_sets.push_back(entry);
    }
    result["pdf"] = pdf_sets;
    return result;
}

// ---------------------------------------------------------------------------
// JSON (de)serialisation
// ---------------------------------------------------------------------------

void madspace::to_json(json& j, const PdfMemberSpec& spec) {
    j = json{
        {"set_name", spec.set_name},
        {"set_lhaid", spec.set_lhaid},
        {"member", spec.member},
        {"grid_file", spec.grid_file},
        {"info_file", spec.info_file},
        {"error_type", spec.error_type},
        {"description", spec.description},
    };
}

void madspace::from_json(const json& j, PdfMemberSpec& spec) {
    spec.set_name = j.at("set_name").get<std::string>();
    spec.set_lhaid = j.at("set_lhaid").get<int>();
    spec.member = j.at("member").get<int>();
    spec.grid_file = j.at("grid_file").get<std::string>();
    spec.info_file = j.at("info_file").get<std::string>();
    spec.error_type = j.value("error_type", "");
    spec.description = j.value("description", "");
}

void madspace::to_json(json& j, const SystematicsConfig& config) {
    j = json{
        {"mur", config.mur},
        {"muf", config.muf},
        {"together", config.together},
        {"dyn_scales", config.dyn_scales},
        {"pdf_members", config.pdf_members},
        {"nominal_set_name", config.nominal_set_name},
        {"nominal_lhaid", config.nominal_lhaid},
        {"nominal_error_type", config.nominal_error_type},
        {"nominal_description", config.nominal_description},
        {"has_pdf", config.has_pdf},
        {"write_inputs", config.write_inputs},
        {"first_id", config.first_id},
    };
}

void madspace::from_json(const json& j, SystematicsConfig& config) {
    config.mur = j.at("mur").get<std::vector<double>>();
    config.muf = j.at("muf").get<std::vector<double>>();
    config.together = j.value("together", true);
    config.dyn_scales = j.value("dyn_scales", std::vector<int>{});
    config.pdf_members = j.value("pdf_members", std::vector<PdfMemberSpec>{});
    config.nominal_set_name = j.value("nominal_set_name", "");
    config.nominal_lhaid = j.value("nominal_lhaid", 0);
    config.nominal_error_type = j.value("nominal_error_type", "");
    config.nominal_description = j.value("nominal_description", "");
    config.has_pdf = j.value("has_pdf", true);
    config.write_inputs = j.value("write_inputs", false);
    config.first_id = j.value("first_id", 1);
}

void madspace::to_json(json& j, const SubprocessSystArgs& args) {
    j = json{{"qcd_power", args.qcd_power}, {"beam_pdgs", args.beam_pdgs}};
}

void madspace::from_json(const json& j, SubprocessSystArgs& args) {
    args.qcd_power = j.at("qcd_power").get<int>();
    args.beam_pdgs = j.at("beam_pdgs").get<nested_vector2<int>>();
}

void madspace::to_json(json& j, const Variation& variation) {
    j = json{
        {"id", variation.id},
        {"mur", variation.mur},
        {"muf", variation.muf},
        {"dyn", variation.dyn},
        {"pdf_index", variation.pdf_index},
    };
}
