#include "madevent/runtime/context.h"

#include <dlfcn.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <unordered_map>

#include "madevent/runtime/io.h"

using namespace madevent;
using json = nlohmann::json;

MatrixElementApi::MatrixElementApi(
    const std::string& file, const std::string& param_card, std::size_t index
) :
    _file_name(file), _index(index) {
    _shared_lib = std::unique_ptr<void, std::function<void(void*)>>(
        dlopen(file.c_str(), RTLD_NOW), [](void* lib) { dlclose(lib); }
    );
    if (!_shared_lib) {
        throw std::runtime_error(
            std::format("Could not load shared object {}\n{}", file, dlerror())
        );
    }

    _get_meta = reinterpret_cast<decltype(&umami_get_meta)>(
        dlsym(_shared_lib.get(), "umami_get_meta")
    );
    if (_get_meta == nullptr) {
        throw std::runtime_error(
            std::format("Did not find symbol umami_get_meta in shared object {}", file)
        );
    }

    _initialize = reinterpret_cast<decltype(&umami_initialize)>(
        dlsym(_shared_lib.get(), "umami_initialize")
    );
    if (_initialize == nullptr) {
        throw std::runtime_error(
            std::format(
                "Did not find symbol umami_initialize in shared object {}", file
            )
        );
    }

    _matrix_element = reinterpret_cast<decltype(&umami_matrix_element)>(
        dlsym(_shared_lib.get(), "umami_matrix_element")
    );
    if (_matrix_element == nullptr) {
        throw std::runtime_error(
            std::format(
                "Did not find symbol umami_matrix_element in shared object {}", file
            )
        );
    }

    _free =
        reinterpret_cast<decltype(&umami_free)>(dlsym(_shared_lib.get(), "umami_free"));
    if (_free == nullptr) {
        throw std::runtime_error(
            std::format("Did not find symbol umami_free in shared object {}", file)
        );
    }

    _instances = ThreadResource<InstanceType>(default_thread_pool(), [&] {
        void* instance;
        check_umami_status(_initialize(&instance, param_card.c_str()));
        return InstanceType(instance, [this](void* proc) { _free(proc); });
    });
}

void MatrixElementApi::check_umami_status(UmamiStatus status) const {
    std::string error;
    switch (status) {
    case UMAMI_SUCCESS:
        return;
    case UMAMI_ERROR:
        throw_error("unspecified error");
    case UMAMI_ERROR_NOT_IMPLEMENTED:
        throw_error("functionality not implemented");
    case UMAMI_ERROR_UNSUPPORTED_INPUT:
        throw_error("unsupported input key");
    case UMAMI_ERROR_UNSUPPORTED_OUTPUT:
        throw_error("unsupported output key");
    case UMAMI_ERROR_UNSUPPORTED_META:
        throw_error("unsupported metadata key");
    case UMAMI_ERROR_MISSING_INPUT:
        throw_error("missing input");
    default:
        throw_error("unknown error");
    }
}

void MatrixElementApi::throw_error(const std::string& message) const {
    throw std::runtime_error(
        std::format("Error in call to matrix element API {}: {}", _file_name, message)
    );
}

const MatrixElementApi&
Context::load_matrix_element(const std::string& file, const std::string& param_card) {
    _param_card_paths.push_back(param_card);
    return _matrix_elements.emplace_back(file, param_card, _matrix_elements.size());
}

Tensor Context::define_global(
    const std::string& name, DataType dtype, const SizeVec& shape, bool requires_grad
) {
    SizeVec full_shape{1};
    full_shape.insert(full_shape.end(), shape.begin(), shape.end());
    if (_globals.contains(name)) {
        throw std::invalid_argument(
            std::format("Context already contains a global named {}", name)
        );
    }
    Tensor tensor(dtype, full_shape, _device);
    tensor.zero();
    _globals[name] = {tensor, requires_grad};
    return tensor;
}

Tensor Context::global(const std::string& name) {
    if (auto search = _globals.find(name); search != _globals.end()) {
        return std::get<0>(search->second);
    } else {
        throw std::invalid_argument(
            std::format("Context does not contain a global named {}", name)
        );
    }
}

bool Context::global_exists(const std::string& name) {
    return _globals.find(name) != _globals.end();
}

bool Context::global_requires_grad(const std::string& name) {
    if (auto search = _globals.find(name); search != _globals.end()) {
        return std::get<1>(search->second);
    } else {
        throw std::invalid_argument(
            std::format("Context does not contain a global named {}", name)
        );
    }
}

std::vector<std::string> Context::global_names() const {
    std::vector<std::string> names;
    names.reserve(_globals.size());
    for (auto& [name, value] : _globals) {
        names.push_back(name);
    }
    return names;
}

void Context::delete_global(const std::string& name) { _globals.erase(name); }

const MatrixElementApi& Context::matrix_element(std::size_t index) const {
    if (index >= _matrix_elements.size()) {
        throw std::runtime_error("Matrix element index out of bounds");
    }
    return _matrix_elements[index];
}

void Context::save(const std::string& file) const {
    namespace fs = std::filesystem;
    fs::path parent_path = fs::path(file).parent_path();
    fs::path global_path = parent_path / "globals";
    fs::create_directory(global_path);

    json j_globals = json::array();
    for (auto& [name, tensor_and_grad] : _globals) {
        auto& [tensor, requires_grad] = tensor_and_grad;
        fs::path tensor_file = global_path / name;
        tensor_file += ".npy";
        save_tensor(tensor_file, tensor);
        j_globals.push_back({
            {"name", name},
            {"requires_grad", requires_grad},
            {"file", fs::relative(tensor_file, parent_path)},
        });
    }

    json j_matrix_elements = json::array();
    for (auto [api, param_card] : zip(_matrix_elements, _param_card_paths)) {
        j_matrix_elements.push_back({
            {"api", fs::absolute(api.file_name())},
            {"param_card", fs::absolute(param_card)},
        });
    }

    json j_context = {{"globals", j_globals}, {"matrix_elements", j_matrix_elements}};
    std::ofstream f(file);
    f << j_context.dump(2);
}

void Context::load(const std::string& file) {
    namespace fs = std::filesystem;
    if (_matrix_elements.size() > 0) {
        throw std::runtime_error(
            "loading a context is only possible before loading any matrix elements"
        );
    }

    std::ifstream f(file);
    json j_context = json::parse(f);
    fs::path parent_path = fs::path(file).parent_path();

    for (auto j_matrix_element : j_context.at("matrix_elements")) {
        load_matrix_element(
            j_matrix_element.at("api").get<std::string>(),
            j_matrix_element.at("param_card").get<std::string>()
        );
    }

    for (auto j_global : j_context.at("globals")) {
        Tensor tensor =
            load_tensor(parent_path / j_global.at("file").get<std::string>());
        Tensor global_tensor = define_global(
            j_global.at("name").get<std::string>(),
            tensor.dtype(),
            {tensor.shape().begin() + 1, tensor.shape().end()},
            j_global.at("requires_grad").get<bool>()
        );
        global_tensor.copy_from(tensor);
    }
}

ContextPtr madevent::default_context() {
    static ContextPtr context = default_device_context(cpu_device());
    return context;
}

ContextPtr madevent::default_cuda_context() {
    static ContextPtr context = default_device_context(cuda_device());
    return context;
}

ContextPtr madevent::default_hip_context() {
    static ContextPtr context = default_device_context(hip_device());
    return context;
}

ContextPtr madevent::default_device_context(DevicePtr device) {
    static std::unordered_map<DevicePtr, ContextPtr> default_contexts;
    if (auto search = default_contexts.find(device); search != default_contexts.end()) {
        return search->second;
    } else {
        ContextPtr context = std::make_shared<Context>(device);
        default_contexts[device] = context;
        return context;
    }
}
