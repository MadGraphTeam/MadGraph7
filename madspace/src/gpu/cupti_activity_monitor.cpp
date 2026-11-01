#include "cupti_activity_monitor.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <thread>

extern char** environ;

using namespace madspace;
using namespace madspace::gpu;

namespace {

// madspace_gpu_monitor, installed next to this library
std::string helper_path() {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&cupti_activity_monitor), &info) == 0 ||
        !info.dli_fname) {
        throw std::runtime_error("cannot locate the madspace CUDA library");
    }
    return (std::filesystem::path(info.dli_fname).parent_path() / "madspace_gpu_monitor")
        .string();
}

// Reads the lines of madspace_gpu_monitor (see gpu_monitor_main.cpp), which
// samples the SM activity in a process of its own
class HelperActivityMonitor : public DeviceActivityMonitor {
public:
    explicit HelperActivityMonitor(int device_index) {
        std::string path = helper_path();
        int input[2], output[2];
        if (pipe2(input, O_CLOEXEC) != 0) {
            throw std::runtime_error("pipe failed");
        }
        if (pipe2(output, O_CLOEXEC) != 0) {
            close(input[0]);
            close(input[1]);
            throw std::runtime_error("pipe failed");
        }
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, input[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO);
        std::string index = std::to_string(device_index);
        char* argv[] = {path.data(), index.data(), nullptr};
        // posix_spawn, not fork: this process runs CUDA and many threads
        int status = posix_spawn(&_pid, path.c_str(), &actions, nullptr, argv, environ);
        posix_spawn_file_actions_destroy(&actions);
        close(input[0]);
        close(output[1]);
        _input = input[1];
        _output = output[0];
        if (status != 0) {
            _pid = -1;
            close_pipes();
            throw std::runtime_error(std::format("cannot start {}", path));
        }
        // the first line says whether the sampling runs, after the CUPTI setup
        std::string line = read_line(std::chrono::seconds(30));
        if (line != "ready") {
            stop();
            throw std::runtime_error(
                line.starts_with("error ") ? line.substr(6)
                                           : "no answer from madspace_gpu_monitor"
            );
        }
        fcntl(_output, F_SETFL, fcntl(_output, F_GETFL) | O_NONBLOCK);
    }

    ~HelperActivityMonitor() override { stop(); }

    void update() override {
        char buffer[4096];
        while (true) {
            ssize_t count = read(_output, buffer, sizeof(buffer));
            if (count <= 0) {
                break;
            }
            _pending.append(buffer, count);
        }
        std::size_t begin = 0;
        for (std::size_t end; (end = _pending.find('\n', begin)) != std::string::npos;
             begin = end + 1) {
            double active, elapsed;
            if (std::sscanf(_pending.c_str() + begin, "%lf %lf", &active, &elapsed) == 2) {
                _active_cycles += active;
                _elapsed_cycles += elapsed;
            }
        }
        _pending.erase(0, begin);
    }

    std::optional<double> take_active_fraction() override {
        if (_elapsed_cycles == 0.) {
            return std::nullopt;
        }
        double fraction = _active_cycles / _elapsed_cycles;
        _active_cycles = 0.;
        _elapsed_cycles = 0.;
        return fraction;
    }

private:
    std::string read_line(std::chrono::seconds timeout) {
        std::string line;
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            pollfd fd{_output, POLLIN, 0};
            if (poll(&fd, 1, 100) <= 0) {
                continue;
            }
            char c;
            if (read(_output, &c, 1) != 1) {
                break; // the helper exited
            }
            if (c == '\n') {
                return line;
            }
            line += c;
        }
        return line;
    }

    void close_pipes() {
        if (_input >= 0) {
            close(_input);
            _input = -1;
        }
        if (_output >= 0) {
            close(_output);
            _output = -1;
        }
    }

    void stop() {
        // closing its input ends the helper
        close_pipes();
        if (_pid <= 0) {
            return;
        }
        for (int i = 0; i < 50; ++i) {
            if (waitpid(_pid, nullptr, WNOHANG) == _pid) {
                _pid = -1;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        kill(_pid, SIGKILL);
        waitpid(_pid, nullptr, 0);
        _pid = -1;
    }

    pid_t _pid = -1;
    int _input = -1;
    int _output = -1;
    std::string _pending;
    double _active_cycles = 0.;
    double _elapsed_cycles = 0.;
};

} // namespace

std::unique_ptr<DeviceActivityMonitor>
madspace::gpu::cupti_activity_monitor(int device_index, std::string* error) {
    try {
        return std::make_unique<HelperActivityMonitor>(device_index);
    } catch (const std::exception& e) {
        if (error) {
            *error = e.what();
        }
        return nullptr;
    }
}
