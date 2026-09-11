#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/driver/backend.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/io.hpp"
#include "madspace/phasespace/matrix_element.hpp"
#include "madspace/phasespace/pdf.hpp"
#include "madspace/util.hpp"

namespace madspace {

// One PDF member entering the variations. The nominal member (the set and member
// the events were generated with) is described by SystematicsConfig::nominal_*
// and must not be listed here.
struct PdfMemberSpec {
    std::string set_name;
    int set_lhaid = 0;    // LHAPDF id of the set (SetIndex in the .info file)
    int member = 0;       // member number inside the set
    std::string grid_file; // <set>_<mmmm>.dat
    std::string info_file; // <set>.info (alpha_s of the set)
    std::string error_type; // ErrorType of the set (replicas, hessian, ...)
    std::string description;
};

struct SystematicsConfig {
    // renormalisation / factorisation scale factors; 1.0 is the nominal point
    std::vector<double> mur{1.};
    std::vector<double> muf{1.};
    // true: all (mur, muf) combinations; false: vary one at a time
    bool together = true;
    // alternative dynamical scale choices (systematics.py numbering: 1 = sum of
    // transverse energies, 2 = sum of transverse masses, 3 = half of it,
    // 4 = partonic centre-of-mass energy); combined with mur/muf
    std::vector<int> dyn_scales;
    // varied PDF members, in output order
    std::vector<PdfMemberSpec> pdf_members;
    // the PDF the events were generated with
    std::string nominal_set_name;
    int nominal_lhaid = 0;
    std::string nominal_error_type;
    std::string nominal_description;
    // false: leptonic beams, no PDF at all (only alpha_s variations)
    bool has_pdf = true;
    // also write the per-event reweighting inputs (x1, x2, scales, pdg ids) to
    // the output (npy columns, LHE <mgrwt> block)
    bool write_inputs = false;
    // first weight id
    int first_id = 1;
};

// Per (unmerged) subprocess information needed to reweight its events.
struct SubprocessSystArgs {
    // power of alpha_s in |M|^2, -1 if it differs between diagrams
    int qcd_power = -1;
    // beam_pdgs[flavor_index] = {pdg of parton 1, pdg of parton 2}
    nested_vector2<int> beam_pdgs;
};

struct Variation {
    int id;
    double mur;
    double muf;
    // index into SystematicsCalculator::members(); -1 for the nominal PDF member
    int pdf_index;
    // dynamical scale choice, -1 for the scale the events were generated with
    int dyn = -1;
    bool is_scale() const {
        return pdf_index == -1 && (mur != 1. || muf != 1. || dyn != -1);
    }
    bool is_nominal() const { return pdf_index == -1 && !is_scale(); }
};

// The variations of one PDF set (for uncertainty bands)
struct PdfGroupInfo {
    std::string set_name;
    int set_lhaid;
    std::string error_type;
    // (member number, index into variations()); member 0 is the central member
    std::vector<std::pair<int, std::size_t>> members;
};

// Computes the variation weights of combined (unweighted) events. The PDFs and
// alpha_s are evaluated with the regular batched madspace functions
// (PartonDensity, RunningCoupling) on `context` (a CPU context; one is created
// when none is given): the grids of the nominal PDF and of every varied member
// are registered as globals of that context under a private prefix.
class SystematicsCalculator {
public:
    // `matrix_elements` (one per subprocess, may be empty/nullopt) enable the
    // exact renormalisation scale variation of subprocesses whose |M|^2 mixes
    // several powers of alpha_s (qcd_power == -1): the matrix element is
    // re-evaluated at the varied alpha_s on `context`. The matrix element must
    // take (momenta, alpha_s, flavor) and return the matrix element only;
    // `me_flavor_remap[subprocess][flavor_index]` is the flavor passed to it.
    SystematicsCalculator(
        const SystematicsConfig& config,
        const std::vector<SubprocessSystArgs>& subproc_args,
        const std::optional<PdfGrid>& nominal_pdf,
        const std::optional<AlphaSGrid>& nominal_alpha_s,
        ContextPtr context = nullptr,
        const std::vector<std::optional<MatrixElement>>& matrix_elements = {},
        const nested_vector2<me_int_t>& me_flavor_remap = {}
    );

    const SystematicsConfig& config() const { return _config; }
    const std::vector<Variation>& variations() const { return _variations; }
    std::size_t weight_count() const { return _variations.size(); }
    std::vector<int> weight_ids() const;
    // members actually used (nominal set member 0 first when it is part of a group)
    const std::vector<PdfMemberSpec>& members() const { return _members; }
    // warnings emitted while building the variation list
    const std::vector<std::string>& warnings() const { return _warnings; }
    // indices into variations() of the scale variations (envelope group)
    std::vector<std::size_t> scale_variation_indices() const;
    // the PDF groups (one per set with members among the variations)
    std::vector<PdfGroupInfo> pdf_groups() const;

    // Compute the weights of all variations for the events of `buffer`. The
    // buffer must carry the combined-event layout (weight, subprocess index,
    // event data, momenta and the partial weight columns). `weights` is resized
    // to event_count * weight_count(), row-major per event, and holds the varied
    // event weights (not the ratios). Thread-safe.
    void compute(EventBuffer& buffer, std::vector<double>& weights) const;
    // The reweighting inputs of one event of `buffer`.
    LOReweightInfo reweight_info(EventBuffer& buffer, std::size_t event_index) const;
    // Accumulate the cross sections per variation (thread-safe).
    void accumulate(EventBuffer& buffer, const std::vector<double>& weights);

    // <initrwgt> header content (without the enclosing tag), following the
    // conventions of systematics.py (weight groups and attributes).
    std::string initrwgt() const;
    // JSON description of the variations and, once events were accumulated,
    // the per-variation cross sections and the scale/PDF uncertainties.
    nlohmann::json summary() const;

    // Combined PDF uncertainty of `member_values` ((member, value) pairs, member
    // 0 = central) for the given LHAPDF error type. Returns (central, up, down);
    // `central` is used when member 0 is absent. NaN when it cannot be computed.
    static std::tuple<double, double, double> pdf_uncertainty(
        const std::string& error_type,
        std::optional<double> central,
        const std::vector<std::pair<int, double>>& member_values
    );
    // Dynamical scale `dyn` (1-4) of an event from its momenta
    // (momenta[particle][component], E px py pz, incoming first)
    static double dynamical_scale(int dyn, const std::vector<std::array<double, 4>>& momenta);
    static std::string dyn_scale_name(int dyn);
    static std::string format_number(double value);

private:
    // batched PDF evaluation of one grid (nominal or member): x f(x, q) for the
    // PIDs used by the events, addressed by their slot in `pids`
    struct PdfEvaluator {
        RuntimePtr runtime;
        std::vector<int> pids;
        std::size_t alpha_s_index; // alpha_s grid of the set
    };
    struct MatrixElementData {
        RuntimePtr runtime;
        std::size_t particle_count;
        std::vector<me_int_t> flavor_remap;
    };

    SystematicsConfig _config;
    std::vector<SubprocessSystArgs> _subproc_args;
    std::vector<PdfMemberSpec> _members;
    std::vector<Variation> _variations;
    std::vector<std::string> _warnings;
    bool _mur_supported;

    ContextPtr _context;
    std::string _prefix;
    std::optional<PdfEvaluator> _nominal_pdf;   // has_pdf only
    std::vector<PdfEvaluator> _member_pdfs;     // one per member
    std::vector<RuntimePtr> _alpha_s_runtimes;  // index 0: nominal set
    std::vector<std::optional<MatrixElementData>> _matrix_elements;
    mutable std::mutex _runtime_mutex;

    std::mutex _accumulate_mutex;
    double _nominal_sum = 0.;
    std::vector<double> _variation_sums;
    std::size_t _event_count = 0;

    void build_variations();
    PdfEvaluator make_pdf_evaluator(
        const PdfGrid& grid,
        const std::vector<int>& pids,
        const std::string& name,
        std::size_t alpha_s_index
    );
    // batched x f(x, q) of `evaluator` for the given points; `slots` index
    // evaluator.pids
    std::vector<double> evaluate_pdf(
        const PdfEvaluator& evaluator,
        const std::vector<double>& x,
        const std::vector<double>& q,
        const std::vector<me_int_t>& slots
    ) const;
    std::vector<double>
    evaluate_alpha_s(std::size_t alpha_s_index, const std::vector<double>& q) const;
    // |M|^2 of the events `indices` of `buffer` (all of subprocess `subproc`) at
    // the given alpha_s values, via the matrix element runtime
    std::vector<double> matrix_elements(
        int subproc,
        EventBuffer& buffer,
        const std::vector<std::size_t>& indices,
        const std::vector<double>& alpha_s
    ) const;
    std::vector<std::array<double, 4>>
    event_momenta(EventBuffer& buffer, std::size_t event_index) const;
};

void to_json(nlohmann::json& j, const PdfMemberSpec& spec);
void from_json(const nlohmann::json& j, PdfMemberSpec& spec);
void to_json(nlohmann::json& j, const SystematicsConfig& config);
void from_json(const nlohmann::json& j, SystematicsConfig& config);
void to_json(nlohmann::json& j, const SubprocessSystArgs& args);
void from_json(const nlohmann::json& j, SubprocessSystArgs& args);
void to_json(nlohmann::json& j, const Variation& variation);

} // namespace madspace
