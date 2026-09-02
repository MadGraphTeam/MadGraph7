#include "madspace/driver/systematics.hpp"

#include <algorithm>
#include <atomic>
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

namespace {

// copy of `grid` restricted to the given PIDs (smaller coefficient tensor)
PdfGrid reduce_grid(const PdfGrid& grid, const std::vector<int>& pids) {
    PdfGrid reduced = grid;
    std::vector<std::size_t> columns;
    for (int pid : pids) {
        auto search = std::find(grid.pids.begin(), grid.pids.end(), pid);
        if (search == grid.pids.end()) {
            throw std::invalid_argument(
                std::format("PID {} not found in pdf grid", pid)
            );
        }
        columns.push_back(search - grid.pids.begin());
    }
    reduced.pids = pids;
    for (auto& row : reduced.values) {
        std::vector<double> new_row;
        for (std::size_t column : columns) {
            new_row.push_back(row.at(column));
        }
        row = new_row;
    }
    return reduced;
}

std::atomic<std::size_t> calculator_counter{0};

} // namespace

SystematicsCalculator::PdfEvaluator SystematicsCalculator::make_pdf_evaluator(
    const PdfGrid& grid,
    const std::vector<int>& pids,
    const std::string& name,
    std::size_t alpha_s_index
) {
    PdfGrid reduced = reduce_grid(grid, pids);
    std::string prefix = std::format("{}.{}", _prefix, name);
    reduced.initialize_globals(_context, prefix);
    PartonDensity density(reduced, pids, true, prefix);
    return PdfEvaluator{
        .runtime = build_runtime(density.function(), _context, false),
        .pids = pids,
        .alpha_s_index = alpha_s_index,
    };
}

SystematicsCalculator::SystematicsCalculator(
    const SystematicsConfig& config,
    const std::vector<SubprocessSystArgs>& subproc_args,
    const std::optional<PdfGrid>& nominal_pdf,
    const std::optional<AlphaSGrid>& nominal_alpha_s,
    ContextPtr context,
    const std::vector<std::optional<MatrixElement>>& matrix_elements,
    const nested_vector2<me_int_t>& me_flavor_remap
) :
    _config(config),
    _subproc_args(subproc_args),
    _context(context ? context : std::make_shared<Context>(cpu_device(), 1)),
    _prefix(std::format("systematics{}", calculator_counter++)) {
    if (!nominal_alpha_s) {
        throw std::invalid_argument(
            "SystematicsCalculator requires the nominal alpha_s grid"
        );
    }
    if (_config.has_pdf && !nominal_pdf) {
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

    // alpha_s of the nominal set: batched RunningCoupling on the context
    auto add_alpha_s = [&](const AlphaSGrid& grid) {
        std::size_t index = _alpha_s_runtimes.size();
        std::string prefix = std::format("{}.alpha_s{}", _prefix, index);
        grid.initialize_globals(_context, prefix);
        RunningCoupling coupling(grid, prefix);
        _alpha_s_runtimes.push_back(build_runtime(coupling.function(), _context, false));
        return index;
    };
    add_alpha_s(nominal_alpha_s.value());

    // the PIDs the events need: both beams of every flavor of every subprocess
    std::vector<int> pids;
    for (auto& args : _subproc_args) {
        for (auto& pdgs : args.beam_pdgs) {
            for (int pdg : pdgs) {
                if (std::find(pids.begin(), pids.end(), pdg) == pids.end()) {
                    pids.push_back(pdg);
                }
            }
        }
    }
    std::sort(pids.begin(), pids.end());
    if (_config.has_pdf) {
        _nominal_pdf = make_pdf_evaluator(nominal_pdf.value(), pids, "nominal", 0);
    }

    // matrix elements for the subprocesses with mixed alpha_s powers
    _matrix_elements.resize(_subproc_args.size());
    if (!matrix_elements.empty()) {
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
                .runtime = build_runtime(me->function(), _context, false),
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

    // the varied PDF members; the alpha_s grid of a set is shared by its members
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
            alpha_s_index = add_alpha_s(AlphaSGrid(spec.info_file));
            alpha_s_indices[spec.info_file] = alpha_s_index;
        }
        _member_pdfs.push_back(make_pdf_evaluator(
            PdfGrid(spec.grid_file),
            pids,
            std::format("member{}", _members.size()),
            alpha_s_index
        ));
        _members.push_back(spec);
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

std::vector<double> SystematicsCalculator::evaluate_pdf(
    const PdfEvaluator& evaluator,
    const std::vector<double>& x,
    const std::vector<double>& q,
    const std::vector<me_int_t>& slots
) const {
    std::size_t count = x.size();
    std::vector<double> values(count, 0.);
    if (count == 0) {
        return values;
    }
    Tensor x_tensor(DataType::dt_float, {count});
    Tensor q_tensor(DataType::dt_float, {count});
    Tensor slot_tensor(DataType::dt_int, {count});
    auto x_view = x_tensor.view<double, 1>();
    auto q_view = q_tensor.view<double, 1>();
    auto slot_view = slot_tensor.view<me_int_t, 1>();
    for (std::size_t i = 0; i < count; ++i) {
        x_view[i] = x[i];
        q_view[i] = q[i];
        slot_view[i] = slots[i];
    }
    TensorVec outputs;
    {
        std::lock_guard<std::mutex> lock(_runtime_mutex);
        outputs = evaluator.runtime->run({x_tensor, q_tensor, slot_tensor});
    }
    Tensor result = outputs.at(0).cpu().contiguous();
    auto result_view = result.view<double, 1>();
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = result_view[i];
    }
    return values;
}

std::vector<double> SystematicsCalculator::evaluate_alpha_s(
    std::size_t alpha_s_index, const std::vector<double>& q
) const {
    std::size_t count = q.size();
    std::vector<double> values(count, 0.);
    if (count == 0) {
        return values;
    }
    Tensor q_tensor(DataType::dt_float, {count});
    auto q_view = q_tensor.view<double, 1>();
    for (std::size_t i = 0; i < count; ++i) {
        q_view[i] = q[i];
    }
    TensorVec outputs;
    {
        std::lock_guard<std::mutex> lock(_runtime_mutex);
        outputs = _alpha_s_runtimes.at(alpha_s_index)->run({q_tensor});
    }
    Tensor result = outputs.at(0).cpu().contiguous();
    auto result_view = result.view<double, 1>();
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = result_view[i];
    }
    return values;
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
        std::lock_guard<std::mutex> lock(_runtime_mutex);
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
    if (var_count == 0 || count == 0) {
        return;
    }
    auto& layout = buffer.layout();
    bool has_beam1 = layout_has(layout, "fact_scale1");
    bool has_beam2 = layout_has(layout, "fact_scale2");
    bool has_partial = layout_has(layout, "partial_weight_product");
    bool has_subproc = layout_has(layout, "subprocess_index");
    bool use_pdf = _config.has_pdf && _nominal_pdf && (has_beam1 || has_beam2);
    bool has_dyn = !_config.dyn_scales.empty();

    // per-event inputs
    struct EventInput {
        double w0, ren_scale, x1, x2, muf1, muf2, alpha_s, nominal_product;
        int subproc, qcd_power;
        me_int_t slot1, slot2;
        bool use_me;
        std::array<double, 5> dyn_scale;
    };
    std::vector<EventInput> inputs(count);
    auto pid_slot = [&](int pdg) -> me_int_t {
        auto& pids = _nominal_pdf->pids;
        return std::find(pids.begin(), pids.end(), pdg) - pids.begin();
    };
    for (std::size_t i = 0; i < count; ++i) {
        auto event = buffer.event(i);
        auto& in = inputs[i];
        in.w0 = event.weight();
        in.subproc = has_subproc ? event.subprocess_index().value() : 0;
        auto& args = _subproc_args.at(in.subproc);
        in.qcd_power = args.qcd_power;
        in.use_me = args.qcd_power < 0 && _matrix_elements.at(in.subproc).has_value();
        in.ren_scale = event.ren_scale();
        in.alpha_s = event.alpha_qcd();
        in.x1 = has_beam1 ? event.x1().value() : 0.;
        in.x2 = has_beam2 ? event.x2().value() : 0.;
        in.muf1 = has_beam1 ? event.fact_scale1().value() : 0.;
        in.muf2 = has_beam2 ? event.fact_scale2().value() : 0.;
        in.slot1 = in.slot2 = 0;
        in.nominal_product = 1.;
        if (use_pdf) {
            auto& pdgs = args.beam_pdgs.at(event.flavor_index());
            in.slot1 = pid_slot(pdgs.at(0));
            in.slot2 = pid_slot(pdgs.at(1));
            if (has_partial) {
                in.nominal_product = event.partial_weight_product();
            }
        }
        in.dyn_scale.fill(0.);
        if (has_dyn) {
            auto momenta = event_momenta(buffer, i);
            for (int dyn : _config.dyn_scales) {
                in.dyn_scale[dyn] = dynamical_scale(dyn, momenta);
            }
        }
    }
    auto scale_of = [&](const EventInput& in, int dyn, double generated) {
        return dyn == -1 ? generated : in.dyn_scale[dyn];
    };

    // Nominal PDF product when the events do not carry it: one batched call
    if (use_pdf && !has_partial) {
        std::vector<double> x, q;
        std::vector<me_int_t> slots;
        for (auto& in : inputs) {
            if (has_beam1) {
                x.push_back(in.x1), q.push_back(in.muf1), slots.push_back(in.slot1);
            }
            if (has_beam2) {
                x.push_back(in.x2), q.push_back(in.muf2), slots.push_back(in.slot2);
            }
        }
        auto values = evaluate_pdf(_nominal_pdf.value(), x, q, slots);
        for (std::size_t i = 0, n = 0; i < count; ++i) {
            double product = 1.;
            if (has_beam1) {
                product *= values[n++];
            }
            if (has_beam2) {
                product *= values[n++];
            }
            inputs[i].nominal_product = product;
        }
    }

    // PDF part: R_pdf[event][variation], one batched call per grid
    // (nominal grid: every scale variation; member grids: the nominal scales)
    std::vector<std::vector<double>> r_pdf(count, std::vector<double>(var_count, 1.));
    if (use_pdf) {
        auto evaluate_variations = [&](const PdfEvaluator& evaluator,
                                       const std::vector<std::size_t>& var_indices) {
            std::vector<double> x, q;
            std::vector<me_int_t> slots;
            for (auto& in : inputs) {
                for (std::size_t k : var_indices) {
                    auto& var = _variations.at(k);
                    if (has_beam1) {
                        x.push_back(in.x1);
                        q.push_back(var.muf * scale_of(in, var.dyn, in.muf1));
                        slots.push_back(in.slot1);
                    }
                    if (has_beam2) {
                        x.push_back(in.x2);
                        q.push_back(var.muf * scale_of(in, var.dyn, in.muf2));
                        slots.push_back(in.slot2);
                    }
                }
            }
            auto values = evaluate_pdf(evaluator, x, q, slots);
            for (std::size_t i = 0, n = 0; i < count; ++i) {
                for (std::size_t k : var_indices) {
                    double product = 1.;
                    if (has_beam1) {
                        product *= values[n++];
                    }
                    if (has_beam2) {
                        product *= values[n++];
                    }
                    r_pdf[i][k] = inputs[i].nominal_product == 0.
                        ? 0.
                        : product / inputs[i].nominal_product;
                }
            }
        };
        std::vector<std::size_t> nominal_grid_vars;
        std::vector<std::vector<std::size_t>> member_vars(_member_pdfs.size());
        for (std::size_t k = 0; k < var_count; ++k) {
            auto& var = _variations.at(k);
            if (var.is_nominal()) {
                continue;
            }
            if (var.pdf_index == -1) {
                if (var.muf != 1. || var.dyn != -1) {
                    nominal_grid_vars.push_back(k);
                }
            } else {
                member_vars.at(var.pdf_index).push_back(k);
            }
        }
        if (!nominal_grid_vars.empty()) {
            evaluate_variations(_nominal_pdf.value(), nominal_grid_vars);
        }
        for (std::size_t m = 0; m < _member_pdfs.size(); ++m) {
            if (!member_vars[m].empty()) {
                evaluate_variations(_member_pdfs[m], member_vars[m]);
            }
        }
    }

    // alpha_s part: the distinct (alpha_s grid, mur, dyn) combinations, one
    // batched alpha_s call each; the matrix element is re-evaluated for the
    // events of the mixed-order subprocesses
    std::vector<AlphaSKey> keys;
    std::vector<std::vector<std::size_t>> key_vars;
    for (std::size_t k = 0; k < var_count; ++k) {
        auto& var = _variations.at(k);
        std::size_t alpha_s_index =
            var.pdf_index >= 0 ? _member_pdfs.at(var.pdf_index).alpha_s_index : 0;
        if (var.mur == 1. && var.dyn == -1 && alpha_s_index == 0) {
            continue;
        }
        AlphaSKey key{alpha_s_index, var.mur, var.dyn};
        std::size_t pos = 0;
        for (; pos < keys.size(); ++pos) {
            if (!(keys[pos] < key) && !(key < keys[pos])) {
                break;
            }
        }
        if (pos == keys.size()) {
            keys.push_back(key);
            key_vars.emplace_back();
        }
        key_vars[pos].push_back(k);
    }
    std::vector<std::vector<double>> r_alpha(count, std::vector<double>(var_count, 1.));
    std::map<int, std::vector<std::size_t>> me_events;
    std::map<int, std::vector<double>> me_nominal;
    for (std::size_t i = 0; i < count; ++i) {
        if (inputs[i].use_me) {
            me_events[inputs[i].subproc].push_back(i);
        }
    }
    if (!keys.empty()) {
        for (auto& [subproc, indices] : me_events) {
            std::vector<double> alpha_s(indices.size());
            for (std::size_t n = 0; n < indices.size(); ++n) {
                alpha_s[n] = inputs[indices[n]].alpha_s;
            }
            me_nominal[subproc] = matrix_elements(subproc, buffer, indices, alpha_s);
        }
    }
    for (std::size_t p = 0; p < keys.size(); ++p) {
        auto& key = keys[p];
        std::vector<double> q(count);
        for (std::size_t i = 0; i < count; ++i) {
            q[i] = key.mur * scale_of(inputs[i], key.dyn, inputs[i].ren_scale);
        }
        auto varied = evaluate_alpha_s(key.alpha_s_index, q);
        std::vector<double> ratio(count, 1.);
        for (std::size_t i = 0; i < count; ++i) {
            auto& in = inputs[i];
            if (in.use_me || in.qcd_power == 0) {
                continue;
            }
            ratio[i] = in.alpha_s == 0. ? 0. : std::pow(varied[i] / in.alpha_s, in.qcd_power);
        }
        for (auto& [subproc, indices] : me_events) {
            std::vector<double> alpha_s(indices.size());
            for (std::size_t n = 0; n < indices.size(); ++n) {
                alpha_s[n] = varied[indices[n]];
            }
            auto me = matrix_elements(subproc, buffer, indices, alpha_s);
            auto& nominal = me_nominal.at(subproc);
            for (std::size_t n = 0; n < indices.size(); ++n) {
                ratio[indices[n]] = nominal[n] == 0. ? 0. : me[n] / nominal[n];
            }
        }
        for (std::size_t k : key_vars[p]) {
            for (std::size_t i = 0; i < count; ++i) {
                r_alpha[i][k] = ratio[i];
            }
        }
    }

    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t k = 0; k < var_count; ++k) {
            weights[i * var_count + k] = inputs[i].w0 * r_alpha[i][k] * r_pdf[i][k];
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
