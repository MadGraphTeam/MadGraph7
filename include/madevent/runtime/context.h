#pragma once

#include <stdint.h>
#include <unordered_map>

#include "madevent/madcode.h"
#include "madevent/runtime/tensor.h"
#include "madevent/runtime/thread_pool.h"
#include "madevent/umami.h"

namespace madevent {

class MatrixElementApi {
public:
    MatrixElementApi(
        const std::string& file, const std::string& param_card, std::size_t index = 0
    );
    MatrixElementApi(MatrixElementApi&&) noexcept = default;
    MatrixElementApi& operator=(MatrixElementApi&&) noexcept = default;
    MatrixElementApi(const MatrixElementApi&) = delete;
    MatrixElementApi& operator=(const MatrixElementApi&) = delete;
    DevicePtr device() const {
        UmamiDevice dev;
        check_umami_status(_get_meta(UMAMI_META_DEVICE, &dev));
        switch (dev) {
        case UMAMI_DEVICE_CPU:
            return cpu_device();
        case UMAMI_DEVICE_CUDA:
            return cuda_device();
        case UMAMI_DEVICE_HIP:
            return hip_device();
        default:
            throw_error("matrix element device not known");
        }
    }
    std::size_t particle_count() const {
        int count;
        check_umami_status(_get_meta(UMAMI_META_PARTICLE_COUNT, &count));
        return count;
    }
    std::size_t diagram_count() const {
        int count;
        check_umami_status(_get_meta(UMAMI_META_DIAGRAM_COUNT, &count));
        return count;
    }
    std::size_t helicity_count() const {
        int count;
        check_umami_status(_get_meta(UMAMI_META_HELICITY_COUNT, &count));
        return count;
    }
    std::size_t index() const { return _index; }
    const std::string& file_name() const { return _file_name; }

    void call(
        UmamiHandle handle,
        size_t count,
        size_t stride,
        size_t offset,
        size_t input_count,
        UmamiInputKey const* input_keys,
        void const* const* inputs,
        size_t output_count,
        UmamiOutputKey const* output_keys,
        void* const* outputs
    ) const {
        check_umami_status(_matrix_element(
            handle,
            count,
            stride,
            offset,
            input_count,
            input_keys,
            inputs,
            output_count,
            output_keys,
            outputs
        ));
    }

    void* process_instance(std::size_t index) const {
        return _instances.get(index).get();
    }

private:
    void check_umami_status(UmamiStatus status) const;
    [[noreturn]] void throw_error(const std::string& message) const;
    std::unique_ptr<void, std::function<void(void*)>> _shared_lib;
    decltype(&umami_get_meta) _get_meta;
    decltype(&umami_initialize) _initialize;
    decltype(&umami_matrix_element) _matrix_element;
    decltype(&umami_free) _free;
    using InstanceType = std::unique_ptr<void, std::function<void(void*)>>;
    ThreadResource<InstanceType> _instances;
    std::string _file_name;
    std::size_t _index;
};

class Context {
    /**
     * Contains global variables and matrix elements
     */
public:
    Context() : _device(cpu_device()) {}
    Context(DevicePtr device) : _device(device) {}
    Context(Context&&) = default;
    Context& operator=(Context&&) = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    const MatrixElementApi&
    load_matrix_element(const std::string& file, const std::string& param_card);
    Tensor define_global(
        const std::string& name,
        DataType dtype,
        const SizeVec& shape,
        bool requires_grad = false
    );
    Tensor global(const std::string& name);
    bool global_requires_grad(const std::string& name);
    bool global_exists(const std::string& name);
    std::vector<std::string> global_names() const;
    void delete_global(const std::string& name);
    const MatrixElementApi& matrix_element(std::size_t index) const;
    void save(const std::string& file) const;
    void load(const std::string& file);
    DevicePtr device() { return _device; }

private:
    // make sure that ThreadPool outlives any Context instance
    inline static ThreadPool& thread_pool_ref = default_thread_pool();
    DevicePtr _device;
    std::unordered_map<std::string, std::tuple<Tensor, bool>> _globals;
    std::vector<MatrixElementApi> _matrix_elements;
    std::vector<std::string> _param_card_paths;
};

using ContextPtr = std::shared_ptr<Context>;

ContextPtr default_context();
ContextPtr default_cuda_context();
ContextPtr default_hip_context();
ContextPtr default_device_context(DevicePtr device);

inline std::string prefixed_name(const std::string& prefix, const std::string& name) {
    return prefix == "" ? name : std::format("{}.{}", prefix, name);
}

} // namespace madevent
