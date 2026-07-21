// SPDX-License-Identifier: 0BSD

#ifndef RUSTICOL_HPP
#define RUSTICOL_HPP

#include "rusticol.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rusticol {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string &message) : std::runtime_error(message) {}
};

inline std::string last_error() {
    std::size_t required = 0;
    rusticol_last_error_message(nullptr, 0, &required);
    if (required == 0) {
        return "unknown Rusticol error";
    }
    std::vector<char> buffer(required);
    rusticol_last_error_message(buffer.data(), buffer.size(), &required);
    return std::string(buffer.data());
}

inline void check(int status) {
    if (status != RUSTICOL_STATUS_OK) {
        throw Error(last_error());
    }
}

inline std::string supported_runtime_capabilities_json() {
    std::size_t required = 0;
    check(rusticol_supported_runtime_capabilities_json(nullptr, 0, &required));
    std::vector<char> buffer(required);
    check(rusticol_supported_runtime_capabilities_json(
        buffer.data(), buffer.size(), &required));
    return std::string(buffer.data());
}

struct ExternalParticle {
    std::size_t index{};
    std::int32_t pdg{};
};

struct HelicityConfiguration {
    std::string id;
    std::vector<std::int32_t> helicities;
};

struct ColorComponent {
    std::string id;
    std::string kind;
    std::vector<std::size_t> word;
};

struct ModelParameter {
    std::string name;
};

class ResolvedEvaluation {
public:
    std::vector<double> values;
    std::size_t point_count{};
    std::vector<HelicityConfiguration> helicities;
    std::vector<ColorComponent> colors;

    double operator()(std::size_t point, std::size_t helicity, std::size_t color) const {
        if (point >= point_count || helicity >= helicities.size() || color >= colors.size()) {
            throw std::out_of_range("resolved Rusticol result index is out of range");
        }
        return values[(point * helicities.size() + helicity) * colors.size() + color];
    }

    std::vector<double> total() const {
        const std::size_t stride = helicities.size() * colors.size();
        std::vector<double> totals(point_count, 0.0);
        for (std::size_t point = 0; point < point_count; ++point) {
            for (std::size_t component = 0; component < stride; ++component) {
                totals[point] += values[point * stride + component];
            }
        }
        return totals;
    }
};

class Runtime {
public:
    Runtime(const std::string &process_dir,
            const std::string &process_key = {},
            const std::string &model_parameters = {}) {
        RusticolRuntimeHandle *loaded = nullptr;
        check(rusticol_runtime_load(
            process_dir.c_str(),
            process_key.empty() ? nullptr : process_key.c_str(),
            model_parameters.empty() ? nullptr : model_parameters.c_str(),
            &loaded));
        handle_ = loaded;
    }

    ~Runtime() {
        if (handle_ != nullptr) {
            rusticol_runtime_free(handle_);
        }
    }

    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    Runtime(Runtime &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}

    Runtime &operator=(Runtime &&other) noexcept {
        if (this != &other) {
            if (handle_ != nullptr) {
                rusticol_runtime_free(handle_);
            }
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    std::string process() const { return get_string(rusticol_runtime_process); }
    std::string process_key() const { return get_string(rusticol_runtime_process_key); }
    std::string color_accuracy() const { return get_string(rusticol_runtime_color_accuracy); }
    std::string execution_mode() const { return get_string(rusticol_runtime_execution_mode); }
    std::string metadata_json() const { return get_string(rusticol_runtime_metadata_json); }
    std::string physics_json() const { return get_string(rusticol_runtime_physics_json); }

    std::vector<ExternalParticle> external_particles() const {
        std::size_t count = 0;
        check(rusticol_runtime_external_count(handle_, &count));
        std::vector<ExternalParticle> result;
        result.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            std::int32_t pdg = 0;
            check(rusticol_runtime_external_pdg(handle_, index, &pdg));
            result.push_back({index, pdg});
        }
        return result;
    }

    std::vector<HelicityConfiguration> helicities() const {
        std::size_t count = 0;
        check(rusticol_runtime_helicity_count(handle_, &count));
        std::vector<HelicityConfiguration> result;
        result.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            std::size_t vector_size = 0;
            check(rusticol_runtime_helicity_vector(handle_, index, nullptr, 0, &vector_size));
            std::vector<std::int32_t> vector(vector_size);
            check(rusticol_runtime_helicity_vector(
                handle_, index, vector.data(), vector.size(), &vector_size));
            result.push_back({get_indexed_string(rusticol_runtime_helicity_id, index), vector});
        }
        return result;
    }

    std::vector<ColorComponent> colors() const {
        std::size_t count = 0;
        check(rusticol_runtime_color_count(handle_, &count));
        std::vector<ColorComponent> result;
        result.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            std::size_t word_size = 0;
            check(rusticol_runtime_color_word(handle_, index, nullptr, 0, &word_size));
            std::vector<std::size_t> word(word_size);
            check(rusticol_runtime_color_word(
                handle_, index, word.data(), word.size(), &word_size));
            result.push_back({
                get_indexed_string(rusticol_runtime_color_id, index),
                get_indexed_string(rusticol_runtime_color_kind, index),
                word,
            });
        }
        return result;
    }

    std::vector<ModelParameter> model_parameters() const {
        std::size_t count = 0;
        check(rusticol_runtime_model_parameter_count(handle_, &count));
        std::vector<ModelParameter> result;
        result.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            result.push_back({get_indexed_string(rusticol_runtime_model_parameter_name, index)});
        }
        return result;
    }

    std::vector<double> evaluate(const std::vector<double> &momenta,
                                 std::size_t point_count) {
        std::vector<double> values(point_count);
        check(rusticol_runtime_evaluate_f64(
            handle_, momenta.data(), momenta.size(), point_count, values.data(), values.size()));
        return values;
    }

    std::vector<double> evaluate_selected(
        const std::vector<double> &momenta,
        std::size_t point_count,
        const std::vector<std::string> &helicity_ids = {},
        const std::vector<std::string> &color_ids = {},
        const std::vector<std::uint32_t> &helicity_by_point = {},
        const std::vector<std::uint32_t> &color_flow_by_point = {}) {
        const auto helicity_ptrs = c_string_pointers(helicity_ids);
        const auto color_ptrs = c_string_pointers(color_ids);
        std::vector<double> values(point_count);
        check(rusticol_runtime_evaluate_selected_f64(
            handle_,
            momenta.data(),
            momenta.size(),
            point_count,
            helicity_ptrs.empty() ? nullptr : helicity_ptrs.data(),
            helicity_ptrs.size(),
            color_ptrs.empty() ? nullptr : color_ptrs.data(),
            color_ptrs.size(),
            helicity_by_point.empty() ? nullptr : helicity_by_point.data(),
            helicity_by_point.size(),
            color_flow_by_point.empty() ? nullptr : color_flow_by_point.data(),
            color_flow_by_point.size(),
            values.data(),
            values.size()));
        return values;
    }

    ResolvedEvaluation evaluate_resolved(
        const std::vector<double> &momenta,
        std::size_t point_count,
        const std::vector<std::string> &helicity_ids = {},
        const std::vector<std::string> &color_ids = {}) {
        const auto helicity_ptrs = c_string_pointers(helicity_ids);
        const auto color_ptrs = c_string_pointers(color_ids);
        std::size_t helicity_count = 0;
        std::size_t color_count = 0;
        check(rusticol_runtime_resolved_shape(
            handle_,
            helicity_ptrs.empty() ? nullptr : helicity_ptrs.data(),
            helicity_ptrs.size(),
            color_ptrs.empty() ? nullptr : color_ptrs.data(),
            color_ptrs.size(),
            &helicity_count,
            &color_count));
        std::vector<double> values(point_count * helicity_count * color_count);
        check(rusticol_runtime_evaluate_resolved_f64(
            handle_,
            momenta.data(),
            momenta.size(),
            point_count,
            helicity_ptrs.empty() ? nullptr : helicity_ptrs.data(),
            helicity_ptrs.size(),
            color_ptrs.empty() ? nullptr : color_ptrs.data(),
            color_ptrs.size(),
            values.data(),
            values.size(),
            &helicity_count,
            &color_count));

        auto all_helicities = helicities();
        auto all_colors = colors();
        return {
            std::move(values),
            point_count,
            select_helicities(all_helicities, helicity_ids),
            select_colors(all_colors, color_ids),
        };
    }

    void set_model_parameter(const std::string &name, double real, double imaginary = 0.0) {
        check(rusticol_runtime_set_model_parameter(handle_, name.c_str(), real, imaginary));
    }

    void set_model_parameters(const std::vector<std::string> &names,
                              const std::vector<double> &real,
                              const std::vector<double> &imaginary) {
        if (names.size() != real.size() || names.size() != imaginary.size()) {
            throw std::invalid_argument("model parameter arrays have different lengths");
        }
        const auto pointers = c_string_pointers(names);
        check(rusticol_runtime_set_model_parameters(
            handle_, pointers.data(), real.data(), imaginary.data(), names.size()));
    }

    void set_model_parameters_json(const std::string &path) {
        check(rusticol_runtime_set_model_parameters_json(handle_, path.c_str()));
    }

    void mute_warnings() { check(rusticol_runtime_mute_warnings(handle_, 1)); }
    void unmute_warnings() { check(rusticol_runtime_mute_warnings(handle_, 0)); }
    std::string take_warnings_json() {
        return get_string_mut(rusticol_runtime_take_warnings_json);
    }

private:
    using StringFunction = int (*)(const RusticolRuntimeHandle *, char *, std::size_t, std::size_t *);
    using MutableStringFunction = int (*)(RusticolRuntimeHandle *, char *, std::size_t, std::size_t *);
    using IndexedStringFunction = int (*)(const RusticolRuntimeHandle *, std::size_t, char *, std::size_t, std::size_t *);

    std::string get_string(StringFunction function) const {
        std::size_t required = 0;
        check(function(handle_, nullptr, 0, &required));
        std::vector<char> buffer(required);
        check(function(handle_, buffer.data(), buffer.size(), &required));
        return std::string(buffer.data());
    }

    std::string get_string_mut(MutableStringFunction function) {
        std::size_t required = 0;
        check(function(handle_, nullptr, 0, &required));
        std::vector<char> buffer(required);
        check(function(handle_, buffer.data(), buffer.size(), &required));
        return std::string(buffer.data());
    }

    std::string get_indexed_string(IndexedStringFunction function, std::size_t index) const {
        std::size_t required = 0;
        check(function(handle_, index, nullptr, 0, &required));
        std::vector<char> buffer(required);
        check(function(handle_, index, buffer.data(), buffer.size(), &required));
        return std::string(buffer.data());
    }

    static std::vector<const char *> c_string_pointers(const std::vector<std::string> &values) {
        std::vector<const char *> pointers;
        pointers.reserve(values.size());
        for (const auto &value : values) {
            pointers.push_back(value.c_str());
        }
        return pointers;
    }

    static std::vector<HelicityConfiguration> select_helicities(
        const std::vector<HelicityConfiguration> &available,
        const std::vector<std::string> &selected) {
        if (selected.empty()) {
            return available;
        }
        std::vector<HelicityConfiguration> result;
        for (const auto &item : available) {
            for (const auto &id : selected) {
                if (item.id == id) {
                    result.push_back(item);
                }
            }
        }
        return result;
    }

    static std::vector<ColorComponent> select_colors(
        const std::vector<ColorComponent> &available,
        const std::vector<std::string> &selected) {
        if (selected.empty()) {
            return available;
        }
        std::vector<ColorComponent> result;
        for (const auto &item : available) {
            for (const auto &id : selected) {
                if (item.id == id) {
                    result.push_back(item);
                }
            }
        }
        return result;
    }

    RusticolRuntimeHandle *handle_{};
};

}  // namespace rusticol

#endif
