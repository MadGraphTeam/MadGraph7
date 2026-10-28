// SPDX-License-Identifier: 0BSD
#ifndef RUSTICOL_VIRTUAL_HPP
#define RUSTICOL_VIRTUAL_HPP
// The shared library's glibc floor (2.38) and the opt-in compile-time check
// RUSTICOL_VIRTUAL_SHARED are described in rusticol_virtual.h.
#include "rusticol_virtual.h"
#include <cstdint>
#include <complex>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rusticol {
// status(): the C function status (rusticol_virtual_function_status), e.g.
// RUSTICOL_VIRTUAL_INVALID_ARGUMENT for a refused model parameter update.
class VirtualError : public std::runtime_error {
public:
    explicit VirtualError(const std::string &message,
                          int status = RUSTICOL_VIRTUAL_RUNTIME_ERROR)
        : std::runtime_error(message), status_(status) {}
    int status() const noexcept { return status_; }
private:
    int status_;
};
namespace virtual_detail {
inline void check(int status) {
    if (status != RUSTICOL_VIRTUAL_OK)
        throw VirtualError(rusticol_virtual_last_error_message(), status);
}
inline const char *optional_string(const std::optional<std::string> &value) {
    return value ? value->c_str() : nullptr;
}
template<class Handle, class Query>
std::string json(Handle *handle, Query query) {
    std::size_t count = 0;
    check(query(handle, nullptr, 0, &count));
    std::vector<char> buffer(count);
    check(query(handle, buffer.data(), buffer.size(), &count));
    return std::string(buffer.data());
}
struct RuntimeDeleter {
    void operator()(rusticol_virtual_handle *p) const noexcept { rusticol_virtual_free(p); }
};
struct CorrelatedDeleter {
    void operator()(rusticol_virtual_correlated_result *p) const noexcept { rusticol_virtual_correlated_result_free(p); }
};
struct ResultDeleter {
    void operator()(rusticol_virtual_result *p) const noexcept { rusticol_virtual_result_free(p); }
};
}

// Move-only ownership. Numerical rows are compact; statuses cover every input.
class VirtualEvaluation {
    std::unique_ptr<rusticol_virtual_result, virtual_detail::ResultDeleter> handle_;
    rusticol_virtual_result_shape_info shape_{};
    explicit VirtualEvaluation(rusticol_virtual_result *p) : handle_(p) {
        virtual_detail::check(rusticol_virtual_result_shape(handle_.get(), &shape_));
    }
    friend class VirtualRuntime;
    template<class T, class Copy>
    std::vector<T> copy(std::size_t count, Copy operation) const {
        std::vector<T> values(count);
        virtual_detail::check(operation(handle_.get(), values.data(), values.size()));
        return values;
    }
public:
    VirtualEvaluation(VirtualEvaluation &&) noexcept = default;
    VirtualEvaluation &operator=(VirtualEvaluation &&) noexcept = default;
    VirtualEvaluation(const VirtualEvaluation &) = delete;
    VirtualEvaluation &operator=(const VirtualEvaluation &) = delete;
    const rusticol_virtual_result_shape_info &shape() const noexcept { return shape_; }
    std::vector<double> born() const {
        return copy<double>(shape_.born_values, rusticol_virtual_result_copy_born);
    }
    // [point,order,epsilon], or [point,order,helicity,colour,epsilon].
    std::vector<double> virtual_values() const {
        return copy<double>(shape_.virtual_values, rusticol_virtual_result_copy_virtual);
    }
    std::vector<std::uint64_t> requested_ids() const {
        return copy<std::uint64_t>(shape_.requested_points, rusticol_virtual_result_copy_requested_ids);
    }
    std::vector<std::uint64_t> successful_ids() const {
        return copy<std::uint64_t>(shape_.successful_points, rusticol_virtual_result_copy_successful_ids);
    }
    std::vector<std::uint32_t> statuses() const {
        return copy<std::uint32_t>(shape_.requested_points, rusticol_virtual_result_copy_statuses);
    }
    std::string diagnostics_json() const {
        return virtual_detail::json(handle_.get(), rusticol_virtual_result_diagnostics_json);
    }
};

// Complex grouped axes: point/history/order/(epsilon), never positive weights.
class CorrelatedVirtualEvaluation {
    std::unique_ptr<rusticol_virtual_correlated_result, virtual_detail::CorrelatedDeleter> handle_;
    rusticol_virtual_correlated_result_shape_info shape_{};
    explicit CorrelatedVirtualEvaluation(rusticol_virtual_correlated_result *p) : handle_(p) {
        virtual_detail::check(rusticol_virtual_correlated_result_shape(handle_.get(), &shape_));
    }
    friend class VirtualRuntime;
    template<class T, class Copy> std::vector<T> copy(std::size_t count, Copy operation) const {
        std::vector<T> values(count);
        virtual_detail::check(operation(handle_.get(), values.data(), values.size()));
        return values;
    }
    template<class Copy> std::vector<std::complex<double>> complex_values(std::size_t count, Copy operation) const {
        const auto pairs = copy<rusticol_virtual_complex64>(count, operation);
        std::vector<std::complex<double>> values; values.reserve(count);
        for (const auto &pair : pairs) values.emplace_back(pair.real, pair.imag);
        return values;
    }
public:
    CorrelatedVirtualEvaluation(CorrelatedVirtualEvaluation &&) noexcept = default;
    CorrelatedVirtualEvaluation &operator=(CorrelatedVirtualEvaluation &&) noexcept = default;
    CorrelatedVirtualEvaluation(const CorrelatedVirtualEvaluation &) = delete;
    CorrelatedVirtualEvaluation &operator=(const CorrelatedVirtualEvaluation &) = delete;
    const rusticol_virtual_correlated_result_shape_info &shape() const noexcept { return shape_; }
    std::vector<std::complex<double>> born() const {
        return complex_values(shape_.born_values, rusticol_virtual_correlated_result_copy_born);
    }
    std::vector<std::complex<double>> virtual_values() const {
        return complex_values(shape_.virtual_values, rusticol_virtual_correlated_result_copy_virtual);
    }
    std::vector<std::uint64_t> requested_ids() const {
        return copy<std::uint64_t>(shape_.requested_points, rusticol_virtual_correlated_result_copy_requested_ids);
    }
    std::vector<std::uint64_t> successful_ids() const {
        return copy<std::uint64_t>(shape_.successful_points, rusticol_virtual_correlated_result_copy_successful_ids);
    }
    std::vector<std::uint32_t> statuses() const {
        return copy<std::uint32_t>(shape_.requested_points, rusticol_virtual_correlated_result_copy_statuses);
    }
    std::string diagnostics_json() const {
        return virtual_detail::json(handle_.get(), rusticol_virtual_correlated_result_diagnostics_json);
    }
    std::string histories_json() const {
        return virtual_detail::json(handle_.get(), rusticol_virtual_correlated_result_histories_json);
    }
    std::string spin_legs_json() const {
        return virtual_detail::json(handle_.get(), rusticol_virtual_correlated_result_spin_legs_json);
    }
};

class VirtualRuntime {
    std::unique_ptr<rusticol_virtual_handle, virtual_detail::RuntimeDeleter> handle_;
public:
    explicit VirtualRuntime(const std::string &path,
                            const std::optional<std::string> &process = std::nullopt,
                            const std::optional<std::string> &parameters_json = std::nullopt,
                            const std::string &r2_strategy = "auto",
                            const std::optional<std::string> &verification_json = std::nullopt) {
        rusticol_virtual_handle *p = nullptr;
        virtual_detail::check(rusticol_virtual_open_with_policy(path.c_str(),
            virtual_detail::optional_string(process), virtual_detail::optional_string(parameters_json),
            r2_strategy.c_str(), virtual_detail::optional_string(verification_json), &p));
        handle_.reset(p);
    }
    VirtualRuntime(VirtualRuntime &&) noexcept = default;
    VirtualRuntime &operator=(VirtualRuntime &&) noexcept = default;
    VirtualRuntime(const VirtualRuntime &) = delete;
    VirtualRuntime &operator=(const VirtualRuntime &) = delete;
    void close() noexcept { handle_.reset(); }
    void clear() { virtual_detail::check(rusticol_virtual_clear(handle_.get())); }
    rusticol_virtual_shape_info shape() const {
        rusticol_virtual_shape_info value{};
        virtual_detail::check(rusticol_virtual_shape(handle_.get(), &value));
        return value;
    }
    std::string metadata_json() const {
        return virtual_detail::json(handle_.get(), rusticol_virtual_metadata_json);
    }
    std::string capabilities_json() const {
        return virtual_detail::json(handle_.get(), rusticol_virtual_capabilities_json);
    }
    rusticol_virtual_correlated_shape_info correlated_shape() const {
        rusticol_virtual_correlated_shape_info value{};
        virtual_detail::check(rusticol_virtual_correlated_shape(handle_.get(), &value));
        return value;
    }
    std::string correlation_histories_json() const {
        return virtual_detail::json(handle_.get(), rusticol_virtual_correlation_histories_json);
    }
    std::string spin_correlation_classes_json() const {
        return virtual_detail::json(handle_.get(), rusticol_virtual_spin_correlation_classes_json);
    }
    // Input components are literal complex values, row/selected-leg/component.
    CorrelatedVirtualEvaluation evaluate_correlated_with_sources(
                               const std::vector<double> &momenta, std::size_t point_count,
                               const std::vector<double> &mu_r2,
                               const std::vector<std::size_t> &legs,
                               const std::vector<std::complex<double>> &vectors,
                               const std::vector<std::uint64_t> &point_ids = {},
                               const std::optional<std::string> &r2_strategy = std::nullopt,
                               const std::optional<std::string> &verification_json = std::nullopt) {
        if (!point_ids.empty() && point_ids.size() != point_count)
            throw VirtualError("point_ids must have one entry per input point");
        std::vector<rusticol_virtual_complex64> pairs; pairs.reserve(vectors.size());
        for (const auto &value : vectors) pairs.push_back({value.real(), value.imag()});
        rusticol_virtual_spin_sources sources{legs.data(), legs.size(), pairs.data(), pairs.size()};
        rusticol_virtual_batch_request request{momenta.data(), momenta.size(), point_count,
            point_ids.empty() ? nullptr : point_ids.data(), mu_r2.data(), mu_r2.size(), 0,
            virtual_detail::optional_string(r2_strategy), virtual_detail::optional_string(verification_json)};
        rusticol_virtual_correlated_result *p = nullptr;
        virtual_detail::check(rusticol_virtual_evaluate_correlated_with_sources(handle_.get(), &request, &sources, &p));
        return CorrelatedVirtualEvaluation(p);
    }
    CorrelatedVirtualEvaluation evaluate_correlated(const std::vector<double> &momenta, std::size_t point_count,
                               const std::vector<double> &mu_r2,
                               const std::vector<std::uint64_t> &point_ids = {},
                               const std::optional<std::string> &r2_strategy = std::nullopt,
                               const std::optional<std::string> &verification_json = std::nullopt) {
        if (!point_ids.empty() && point_ids.size() != point_count)
            throw VirtualError("point_ids must have one entry per input point");
        rusticol_virtual_batch_request request{momenta.data(), momenta.size(), point_count,
            point_ids.empty() ? nullptr : point_ids.data(), mu_r2.data(), mu_r2.size(), 0,
            virtual_detail::optional_string(r2_strategy), virtual_detail::optional_string(verification_json)};
        rusticol_virtual_correlated_result *p = nullptr;
        virtual_detail::check(rusticol_virtual_evaluate_correlated(handle_.get(), &request, &p));
        return CorrelatedVirtualEvaluation(p);
    }
    // "all" (default), or for plan v4 "none" (finite part, poles quiet NaNs),
    // "uv" (the bare loop's UV poles) or "ir" (the renormalised IR poles).
    void set_poles(const std::string &poles) {
        virtual_detail::check(rusticol_virtual_set_poles(handle_.get(), poles.c_str()));
    }
    std::uint64_t update_parameters(const std::string &exact_json) {
        std::uint64_t revision = 0;
        virtual_detail::check(rusticol_virtual_update_parameters(handle_.get(), exact_json.c_str(), &revision));
        return revision;
    }
    // One native bulk call. Empty IDs select 0..point_count; mu_r2 has one or N entries.
    VirtualEvaluation evaluate(const std::vector<double> &momenta, std::size_t point_count,
                               const std::vector<double> &mu_r2,
                               const std::vector<std::uint64_t> &point_ids = {}, bool resolved = false,
                               const std::optional<std::string> &r2_strategy = std::nullopt,
                               const std::optional<std::string> &verification_json = std::nullopt) {
        if (!point_ids.empty() && point_ids.size() != point_count)
            throw VirtualError("point_ids must have one entry per input point");
        rusticol_virtual_batch_request request{
            momenta.data(), momenta.size(), point_count,
            point_ids.empty() ? nullptr : point_ids.data(), mu_r2.data(), mu_r2.size(),
            resolved ? 1u : 0u, virtual_detail::optional_string(r2_strategy),
            virtual_detail::optional_string(verification_json)};
        rusticol_virtual_result *p = nullptr;
        virtual_detail::check(rusticol_virtual_evaluate(handle_.get(), &request, &p));
        return VirtualEvaluation(p);
    }
};
}
#endif
