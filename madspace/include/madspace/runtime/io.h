#pragma once

#include <cstring>
#include <fstream>

#include "madspace/runtime/lhe_output.h"
#include "madspace/runtime/tensor.h"

namespace madspace {

Tensor load_tensor(const std::string& file);
void save_tensor(const std::string& file, Tensor tensor);

using FieldLayout = std::pair<const char*, const char*>;

template <typename T>
class UnalignedRef {
public:
    UnalignedRef(void* ptr) : _ptr(ptr) {}
    T value() const {
        T value;
        std::memcpy(&value, _ptr, sizeof(T));
        return value;
    }
    operator T() const { return value(); }
    UnalignedRef<T> operator=(const T& value) {
        std::memcpy(_ptr, &value, sizeof(T));
        return *this;
    }
    UnalignedRef<T> operator=(const UnalignedRef<T>& value) {
        std::memcpy(_ptr, value._ptr, sizeof(T));
        return *this;
    }

private:
    void* _ptr;
};

struct ParticleRecord {
    static constexpr std::size_t size = 32;
    static constexpr std::array<FieldLayout, 4> layout = {
        {{"energy", "<f8"}, {"px", "<f8"}, {"py", "<f8"}, {"pz", "<f8"}}
    };

    UnalignedRef<double> energy() { return &data[0]; }
    UnalignedRef<double> px() { return &data[8]; }
    UnalignedRef<double> py() { return &data[16]; }
    UnalignedRef<double> pz() { return &data[24]; }

    char* data;
};

constexpr int record_weight = 1;
constexpr int record_subproc_index = 2;
constexpr int record_indices = 4;

template <int fields>
struct EventRecord {
    static constexpr std::size_t size = (fields & record_weight ? 8 : 0) +
        (fields & record_subproc_index ? 4 : 0) + (fields & record_indices ? 16 : 0);
    static constexpr std::size_t subproc_index_offset = fields & record_weight ? 8 : 0;
    static constexpr std::size_t indices_offset =
        subproc_index_offset + (fields & record_subproc_index ? 4 : 0);

    static constexpr std::size_t field_count = (fields & record_weight ? 1 : 0) +
        (fields & record_subproc_index ? 1 : 0) + (fields & record_indices ? 4 : 0);
    static constexpr std::array<FieldLayout, field_count> layout = [] {
        std::array<FieldLayout, field_count> layout;
        std::size_t offset = 0;
        if (fields & record_weight) {
            layout[0] = {"weight", "<f8"};
            offset += 1;
        }
        if (fields & record_subproc_index) {
            layout[offset] = {"subprocess_index", "<i4"};
            offset += 1;
        }
        if (fields & record_indices) {
            layout[offset + 0] = {"diagram_index", "<i4"};
            layout[offset + 1] = {"color_index", "<i4"};
            layout[offset + 2] = {"flavor_index", "<i4"};
            layout[offset + 3] = {"helicity_index", "<i4"};
            offset += 4;
        }
        return layout;
    }();

    UnalignedRef<double> weight() { return &data[0]; }
    UnalignedRef<int> subprocess_index() { return &data[subproc_index_offset + 0]; }
    UnalignedRef<int> diagram_index() { return &data[indices_offset + 0]; }
    UnalignedRef<int> color_index() { return &data[indices_offset + 4]; }
    UnalignedRef<int> flavor_index() { return &data[indices_offset + 8]; }
    UnalignedRef<int> helicity_index() { return &data[indices_offset + 12]; }

    char* data;
};

using EventWeightRecord = EventRecord<record_weight>;
using EventIndicesRecord = EventRecord<record_indices>;
using EventFullRecord =
    EventRecord<record_weight | record_subproc_index | record_indices>;

struct EmptyParticleRecord {
    static constexpr std::size_t size = 0;
    static constexpr std::array<FieldLayout, 0> layout = {};
};

struct PackedLHEParticle {
    static constexpr std::size_t size = 6 * sizeof(int) + 7 * sizeof(double);
    static constexpr std::array<FieldLayout, 13> layout = {{
        {"pdg_id", "<i4"},
        {"status_code", "<i4"},
        {"mother1", "<i4"},
        {"mother2", "<i4"},
        {"color", "<i4"},
        {"anti_color", "<i4"},
        {"px", "<f8"},
        {"py", "<f8"},
        {"pz", "<f8"},
        {"energy", "<f8"},
        {"mass", "<f8"},
        {"lifetime", "<f8"},
        {"spin", "<f8"},
    }};

    void from_lhe_particle(const LHEParticle& particle) {
        pdg_id() = particle.pdg_id;
        status_code() = particle.status_code;
        mother1() = particle.mother1;
        mother2() = particle.mother2;
        color() = particle.color;
        anti_color() = particle.anti_color;
        px() = particle.px;
        py() = particle.py;
        pz() = particle.pz;
        energy() = particle.energy;
        mass() = particle.mass;
        lifetime() = particle.lifetime;
        spin() = particle.spin;
    }

    UnalignedRef<int> pdg_id() { return &data[0]; }
    UnalignedRef<int> status_code() { return &data[4]; }
    UnalignedRef<int> mother1() { return &data[8]; }
    UnalignedRef<int> mother2() { return &data[12]; }
    UnalignedRef<int> color() { return &data[16]; }
    UnalignedRef<int> anti_color() { return &data[20]; }
    UnalignedRef<double> px() { return &data[24]; }
    UnalignedRef<double> py() { return &data[32]; }
    UnalignedRef<double> pz() { return &data[40]; }
    UnalignedRef<double> energy() { return &data[48]; }
    UnalignedRef<double> mass() { return &data[56]; }
    UnalignedRef<double> lifetime() { return &data[64]; }
    UnalignedRef<double> spin() { return &data[72]; }

    char* data;
};

struct PackedLHEEvent {
    static constexpr std::size_t size = 1 * sizeof(int) + 4 * sizeof(double);
    static constexpr std::array<FieldLayout, 5> layout = {{
        {"process_id", "<i4"},
        {"weight", "<f8"},
        {"scale", "<f8"},
        {"alpha_qed", "<f8"},
        {"alpha_qcd", "<f8"},
    }};

    void from_lhe_event(const LHEEvent& event) {
        process_id() = event.process_id;
        weight() = event.weight;
        scale() = event.scale;
        alpha_qed() = event.alpha_qed;
        alpha_qcd() = event.alpha_qcd;
    }

    UnalignedRef<int> process_id() { return &data[0]; }
    UnalignedRef<double> weight() { return &data[4]; }
    UnalignedRef<double> scale() { return &data[12]; }
    UnalignedRef<double> alpha_qed() { return &data[20]; }
    UnalignedRef<double> alpha_qcd() { return &data[28]; }

    char* data;
};

struct DataLayout {
    std::span<const FieldLayout> event_fields;
    std::span<const FieldLayout> particle_fields;
    std::size_t event_size;
    std::size_t particle_size;

    template <typename E, typename P>
    static DataLayout of() {
        return {
            .event_fields = {E::layout.begin(), E::layout.end()},
            .particle_fields = {P::layout.begin(), P::layout.end()},
            .event_size = E::size,
            .particle_size = P::size,
        };
    }
};

class EventBuffer {
public:
    EventBuffer(
        std::size_t event_count, std::size_t particle_count, DataLayout layout
    ) :
        _event_count(event_count),
        _particle_count(particle_count),
        _layout(layout),
        _data(
            event_count * (layout.event_size + particle_count * layout.particle_size)
        ) {}
    char* data() { return _data.data(); }
    const char* data() const { return _data.data(); }
    std::size_t size() const { return _data.size(); }
    std::size_t event_count() const { return _event_count; }
    std::size_t particle_count() const { return _particle_count; }
    const DataLayout& layout() const { return _layout; }

    std::size_t event_size() const {
        return _layout.event_size + _particle_count * _layout.particle_size;
    }

    std::size_t event_offset(std::size_t event_index) const {
        return event_index * event_size();
    }

    std::size_t
    particle_offset(std::size_t event_index, std::size_t particle_index) const {
        return event_offset(event_index) + _layout.event_size +
            particle_index * _layout.particle_size;
    }

    template <typename T>
    T particle(std::size_t event_index, std::size_t particle_index) {
        return {&_data.data()[particle_offset(event_index, particle_index)]};
    }

    template <typename T>
    T event(std::size_t event_index) {
        return {&_data.data()[event_offset(event_index)]};
    }

    void resize(std::size_t event_count) {
        _data.resize(event_count * event_size());
        _event_count = event_count;
    }

    void copy_and_pad(EventBuffer& buffer) {
        if (buffer.particle_count() > particle_count()) {
            throw std::runtime_error("Given buffer contains too many particles");
        }
        resize(buffer.event_count());
        for (std::size_t i = 0; i < event_count(); ++i) {
            std::memcpy(
                &_data[event_offset(i)],
                &buffer._data[buffer.event_offset(i)],
                buffer.event_size()
            );
            std::memset(
                &_data[event_offset(i) + buffer.event_size()],
                0,
                event_size() - buffer.event_size()
            );
        }
    }

private:
    std::size_t _event_count;
    std::size_t _particle_count;
    DataLayout _layout;
    std::vector<char> _data;
};

class EventFile {
public:
    enum Mode { create, append, load };

    EventFile(
        const std::string& file_name,
        DataLayout layout,
        std::size_t particle_count = 0,
        Mode mode = create,
        bool delete_on_close = false
    );

    EventFile(EventFile&& other) noexcept = default;
    EventFile& operator=(EventFile&& other) noexcept = default;
    void seek(std::size_t index);
    void clear();
    std::size_t particle_count() const { return _particle_count; }
    std::size_t event_count() const { return _event_count; }
    ~EventFile();

    void write(EventBuffer& buffer) {
        if (_mode == EventFile::load) {
            throw std::runtime_error("Event file opened in read mode.");
        }
        if (buffer.particle_count() != _particle_count) {
            throw std::invalid_argument("Wrong number of particles");
        }
        _file_stream.write(buffer.data(), buffer.size());
        _current_event += buffer.event_count();
        if (_current_event > _event_count) {
            _event_count = _current_event;
        }
    }

    bool read(EventBuffer& buffer, std::size_t count) {
        if (_current_event == _event_count) {
            return false;
        }
        count = std::min(count, _event_count - _current_event);
        buffer.resize(count);
        if (buffer.particle_count() == _particle_count) {
            _file_stream.read(buffer.data(), buffer.size());
        } else if (buffer.particle_count() > _particle_count) {
            EventBuffer tmp_buffer(count, _particle_count, buffer.layout());
            _file_stream.read(tmp_buffer.data(), tmp_buffer.size());
            buffer.copy_and_pad(tmp_buffer);
        } else {
            throw std::invalid_argument("Wrong number of particles");
        }
        _current_event += count;
        return true;
    }

private:
    std::string _file_name;
    std::size_t _event_count;
    std::size_t _current_event;
    std::size_t _capacity;
    std::size_t _particle_count;
    std::size_t _shape_pos;
    std::fstream _file_stream;
    std::size_t _header_size;
    std::size_t _event_size;
    Mode _mode;
    bool _delete_on_close;
};

} // namespace madspace
