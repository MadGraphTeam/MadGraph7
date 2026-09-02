#pragma once

#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/driver/thread_pool.hpp"
#include "madspace/phasespace/topology.hpp"
#include "madspace/util.hpp"

namespace madspace {

struct LHEHeader {
    std::string name;
    std::string content;
    bool escape_content;
};

struct LHEProcess {
    double cross_section;
    double cross_section_error;
    double max_weight;
    int process_id;
};

struct LHEMeta {
    int beam1_pdg_id, beam2_pdg_id;
    double beam1_energy, beam2_energy;
    int beam1_pdf_authors, beam2_pdf_authors;
    int beam1_pdf_id, beam2_pdf_id;
    int weight_mode;
    std::vector<LHEProcess> processes;
    std::vector<LHEHeader> headers;
};

struct LHEParticle {
    // particle-level information as defined in arXiv:0109068
    inline static const int status_incoming = -1;
    inline static const int status_outgoing = 1;
    inline static const int status_intermediate_resonance = 2;

    int pdg_id;
    int status_code;
    int mother1, mother2;
    int color, anti_color;
    double px, py, pz, energy, mass;
    double lifetime;
    double spin;
};

struct LHEEvent {
    // event-level information as defined in arXiv:0109068
    int process_id;
    double weight;
    double scale;
    double alpha_qed;
    double alpha_qcd;
    std::vector<LHEParticle> particles;

    void format_to(std::string& buffer) const;
};

class LHECompleter {
public:
    struct SubprocArgs {
        int process_id;
        std::vector<Topology> topologies;
        nested_vector3<std::size_t> permutations;
        nested_vector2<std::size_t> diagram_indices;
        nested_vector3<std::size_t> diagram_color_indices;
        nested_vector2<std::tuple<int, int>> color_flows;
        std::unordered_map<int, int> pdg_color_types;
        nested_vector2<double> helicities;
        nested_vector3<int> pdg_ids;
        // Per-diagram pdg override, indexed like diagram_color_indices then by
        // Decay::flat_propagator_index. Falls back to Decay::pdg_id if empty.
        nested_vector3<int> diagram_propagator_pdgs;
    };

    LHECompleter(const std::vector<SubprocArgs>& subproc_args, double bw_cutoff);
    void complete_event_data(
        LHEEvent& event,
        int subprocess_index,
        int diagram_index,
        int color_index,
        int flavor_index,
        int helicity_index,
        std::mt19937& rand_gen
    );
    std::size_t max_particle_count() const { return _max_particle_count; }
    void save(const std::string& file) const;
    static LHECompleter load(const std::string& file);

private:
    struct SubprocData {
        int process_id;
        std::size_t color_offset, pdg_id_offset, helicity_offset, mass_offset;
        std::size_t particle_count, color_count, flavor_count;
        std::size_t diagram_count, helicity_count;
        // 2 for a collision, 1 for a decay. Decides which leading particles are
        // written as initial state and what the outgoing ones point at.
        std::size_t incoming_count;
    };
    struct PropagatorData {
        int pdg_id;
        int momentum_mask;
        int child_prop_mask;
        double mass, width;
    };
    std::vector<SubprocData> _subproc_data;
    std::vector<int> _process_indices;
    std::vector<double> _masses;
    std::vector<std::tuple<int, int>> _colors;
    std::vector<double> _helicities;
    std::vector<std::array<std::size_t, 2>> _pdg_id_and_count;
    std::vector<int> _pdg_ids;
    std::unordered_map<std::size_t, std::array<std::size_t, 3>>
        _propagator_index_and_count;
    std::vector<PropagatorData> _propagators;
    std::vector<std::tuple<int, int>> _propagator_colors;
    double _bw_cutoff;
    std::size_t _max_particle_count;

    std::size_t append_helicities(const SubprocArgs& args);
    std::size_t append_colors(const SubprocArgs& args, std::size_t particle_count);
    void append_pdg_ids(const SubprocArgs& args, std::size_t particle_count);
    void append_masses(const Topology& first_topo);
    std::pair<std::size_t, std::size_t>
    build_propagators(std::size_t subproc_index, const SubprocArgs& args);
    void init_propagator_data(
        const Topology& topo,
        const SubprocArgs& args,
        const std::vector<std::size_t>& colors,
        const std::vector<std::size_t>& permutation,
        std::vector<double>& e_min,
        std::vector<int>& momentum_masks,
        std::vector<std::tuple<int, int>>& prop_colors,
        std::vector<int>& resonant_prop_indices
    ) const;
    void find_resonant_propagators(
        const Topology& topo,
        const SubprocArgs& args,
        const std::vector<std::size_t>& colors,
        const std::vector<int>& propagator_pdgs,
        std::size_t prop_offset,
        std::vector<double>& e_min,
        std::vector<int>& momentum_masks,
        std::vector<std::tuple<int, int>>& prop_colors,
        std::vector<int>& resonant_prop_indices
    );
    void record_propagator_colors(
        std::size_t subproc_index,
        std::size_t diag_index,
        const std::vector<std::size_t>& colors,
        std::size_t prop_offset,
        const std::vector<std::tuple<int, int>>& prop_colors,
        const std::vector<int>& resonant_prop_indices
    );

    LHECompleter() = default;
    friend void to_json(nlohmann::json& j, const LHECompleter& lhe_completer);
    friend void from_json(const nlohmann::json& j, LHECompleter& lhe_completer);
    friend void
    to_json(nlohmann::json& j, const LHECompleter::SubprocData& subproc_data);
    friend void
    from_json(const nlohmann::json& j, LHECompleter::SubprocData& subproc_data);
    friend void
    to_json(nlohmann::json& j, const LHECompleter::PropagatorData& prop_data);
    friend void
    from_json(const nlohmann::json& j, LHECompleter::PropagatorData& prop_data);
};

void to_json(nlohmann::json& j, const LHECompleter& lhe_completer);
void from_json(const nlohmann::json& j, LHECompleter& lhe_completer);
void to_json(nlohmann::json& j, const LHECompleter::SubprocData& subproc_data);
void from_json(const nlohmann::json& j, LHECompleter::SubprocData& subproc_data);
void to_json(nlohmann::json& j, const LHECompleter::PropagatorData& prop_data);
void from_json(const nlohmann::json& j, LHECompleter::PropagatorData& prop_data);

class LHEFileWriter {
public:
    LHEFileWriter(const std::string& file_name, const LHEMeta& meta);
    void write(const LHEEvent& event);
    void write_string(const std::string& str);
    ~LHEFileWriter();

private:
    std::ofstream _file_stream;
    std::string _buffer;
};

// Writes one event stream into several LHE files at once, dealing the events
// out round-robin. Every file is a complete, self-describing LHE file: it gets
// its own copy of `meta`, so its own <header> and its own <init> block with the
// process cross sections. A consumer can therefore open any one of them by path
// alone, with nothing published beside it.
//
// The point of the fan-out is a reader that wants to consume the events in
// parallel: with N files it opens the one that is its own, instead of reading
// the whole stream and skipping the (N-1)/N of it that belongs to somebody
// else. Round-robin rather than contiguous blocks because the files are then
// balanced to within one event whatever the chunking, and because the split
// cannot correlate with the order the stream happens to arrive in.
//
// Chunked, off-thread formatting is supported through reserve()/file_index():
// reserve() hands out a run of consecutive positions in the stream, file_index()
// says where each of them belongs, and the formatted text can then be handed
// over with write_string() whenever it is ready -- in any order.
class LHEMultiFileWriter {
public:
    LHEMultiFileWriter(const std::vector<std::string>& file_names, const LHEMeta& meta);

    std::size_t file_count() const { return _writers.size(); }
    // Which file the `event_index`-th event of the stream belongs to.
    std::size_t file_index(std::size_t event_index) const {
        return event_index % _writers.size();
    }
    // Claim `count` consecutive positions in the stream; returns the first.
    std::size_t reserve(std::size_t count);
    // Total number of positions claimed so far.
    std::size_t event_count() const { return _event_count; }
    void write_string(std::size_t file, const std::string& str);
    // Convenience for a caller that formats one event at a time.
    void write(const LHEEvent& event);

private:
    std::vector<std::unique_ptr<LHEFileWriter>> _writers;
    std::size_t _event_count = 0;
};

} // namespace madspace
