#include "madspace/runtime/runtime_base.h"

#include <cstdlib>
#include <dlfcn.h>
#include <format>
#include <unordered_map>

using namespace madspace;

namespace {

struct LoadedRuntime {
    inline static std::string lib_path = "";
    inline static int vector_size = -1;
    inline static std::unordered_map<DevicePtr, LoadedRuntime*> device_runtimes;

    LoadedRuntime(const std::string& file) {
#ifdef __APPLE__
        std::string so_ext = "dylib";
#else
        std::string so_ext = "so";
#endif
        shared_lib = std::shared_ptr<void>(
            dlopen(std::format("{}/{}.{}", lib_path, file, so_ext).c_str(), RTLD_NOW),
            [](void* lib) { dlclose(lib); }
        );
        if (!shared_lib) {
            throw std::runtime_error(
                std::format("Could not load shared object {}", file)
            );
        }
        get_device = reinterpret_cast<decltype(get_device)>(
            dlsym(shared_lib.get(), "get_device")
        );
        if (get_device == nullptr) {
            throw std::runtime_error(
                std::format("Did not find symbol get_device in shared object {}", file)
            );
        }
        build_runtime = reinterpret_cast<decltype(build_runtime)>(
            dlsym(shared_lib.get(), "build_runtime")
        );
        if (build_runtime == nullptr) {
            throw std::runtime_error(
                std::format(
                    "Did not find symbol build_runtime in shared object {}", file
                )
            );
        }

        device_runtimes[get_device()] = this;
    }

    std::shared_ptr<void> shared_lib;
    DevicePtr (*get_device)();
    Runtime* (*build_runtime)(
        const Function& function, ContextPtr context, bool concurrent
    );
};

const LoadedRuntime& cpu_runtime() {
    static LoadedRuntime runtime = [&] {
        std::vector<int> supported_vector_sizes{1};
#ifdef SIMD_AVAILABLE
#ifdef __APPLE__
        supported_vector_sizes.push_back(2);
#else  // __APPLE__
        if (__builtin_cpu_supports("avx2")) {
            supported_vector_sizes.push_back(4);
        }
        if (__builtin_cpu_supports("avx512f")) {
            supported_vector_sizes.push_back(8);
        }
#endif // __APPLE__
#endif // SIMD_AVAILABLE

        int vector_size = LoadedRuntime::vector_size;
        if (vector_size == -1) {
            if (char* env_var = std::getenv("SIMD_VECTOR_SIZE")) {
                vector_size = std::atoi(env_var);
            } else {
                vector_size = 0;
            }
        }
        if (vector_size <= 0) {
#ifdef __APPLE__
            vector_size = 1;
#else
            // vector_size = supported_vector_sizes.back();
            vector_size = 1;
#endif
        } else if (std::find(
                       supported_vector_sizes.begin(),
                       supported_vector_sizes.end(),
                       vector_size
                   ) == supported_vector_sizes.end()) {
            throw std::runtime_error("unsupported SIMD vector size");
        }

        switch (vector_size) {
        case 2:
            return LoadedRuntime("libmadspace_cpu_neon");
        case 4:
            return LoadedRuntime("libmadspace_cpu_avx2");
        case 8:
            return LoadedRuntime("libmadspace_cpu_avx512");
        default:
            return LoadedRuntime("libmadspace_cpu");
        }
    }();
    return runtime;
}

const LoadedRuntime& cuda_runtime() {
    static LoadedRuntime runtime("libmadspace_cuda");
    return runtime;
}

const LoadedRuntime& hip_runtime() {
    static LoadedRuntime runtime("libmadspace_hip");
    return runtime;
}

} // namespace

RuntimePtr
madspace::build_runtime(const Function& function, ContextPtr context, bool concurrent) {
    auto& loaded_runtime = LoadedRuntime::device_runtimes.at(context->device());
    Runtime* runtime = loaded_runtime->build_runtime(function, context, concurrent);
    runtime->shared_lib = loaded_runtime->shared_lib;
    return RuntimePtr(runtime);
}

DevicePtr madspace::cpu_device() { return cpu_runtime().get_device(); }

DevicePtr madspace::cuda_device() { return cuda_runtime().get_device(); }

DevicePtr madspace::hip_device() { return hip_runtime().get_device(); }

void madspace::set_lib_path(const std::string& lib_path) {
    LoadedRuntime::lib_path = lib_path;
}

void madspace::set_simd_vector_size(int vector_size) {
    LoadedRuntime::vector_size = vector_size;
}
