// SPDX-License-Identifier: 0BSD

#include <rusticol.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <time.h>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kPointCount = 10;
constexpr std::size_t kWarmSampleCount = 10;
constexpr std::size_t kRepresentativePoint = 0;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kUnitCouplingAlphaS = 1.0 / (4.0 * kPi);

struct Event {
    std::vector<double> momenta;
    std::vector<std::int32_t> helicities;
};

double process_cpu_seconds() {
    struct timespec value {};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) {
        throw std::runtime_error("clock_gettime(CLOCK_PROCESS_CPUTIME_ID) failed");
    }
    return static_cast<double>(value.tv_sec) +
           static_cast<double>(value.tv_nsec) * 1.0e-9;
}

std::string last_rusticol_error() {
    std::size_t required = 0;
    (void)rusticol_last_error_message(nullptr, 0, &required);
    if (required == 0) {
        return "unknown Rusticol error";
    }
    std::vector<char> buffer(required);
    (void)rusticol_last_error_message(buffer.data(), buffer.size(), &required);
    return std::string(buffer.data());
}

void check_rusticol(const int status) {
    if (status != RUSTICOL_STATUS_OK) {
        throw std::runtime_error(last_rusticol_error());
    }
}

class RuntimeHandle {
  public:
    RuntimeHandle(const std::string &artifact, const std::string &process) {
        check_rusticol(rusticol_runtime_load(
            artifact.c_str(), process.c_str(), nullptr, &handle_));
        if (handle_ == nullptr) {
            throw std::runtime_error("Rusticol returned a null runtime handle");
        }
    }

    ~RuntimeHandle() {
        if (handle_ != nullptr) {
            (void)rusticol_runtime_free(handle_);
        }
    }

    RuntimeHandle(const RuntimeHandle &) = delete;
    RuntimeHandle &operator=(const RuntimeHandle &) = delete;

    RusticolRuntimeHandle *get() const { return handle_; }

  private:
    RusticolRuntimeHandle *handle_ = nullptr;
};

using RuntimeStringGetter = int (*)(
    const RusticolRuntimeHandle *, char *, std::size_t, std::size_t *);

std::string runtime_string(
    const RusticolRuntimeHandle *handle,
    const RuntimeStringGetter getter) {
    std::size_t required = 0;
    check_rusticol(getter(handle, nullptr, 0, &required));
    if (required == 0) {
        throw std::runtime_error("Rusticol returned an empty string capacity");
    }
    std::vector<char> buffer(required);
    check_rusticol(getter(handle, buffer.data(), buffer.size(), &required));
    return std::string(buffer.data());
}

std::string runtime_helicity_id(
    const RusticolRuntimeHandle *handle,
    const std::size_t index) {
    std::size_t required = 0;
    check_rusticol(rusticol_runtime_helicity_id(
        handle, index, nullptr, 0, &required));
    if (required == 0) {
        throw std::runtime_error("Rusticol returned an empty helicity ID capacity");
    }
    std::vector<char> buffer(required);
    check_rusticol(rusticol_runtime_helicity_id(
        handle, index, buffer.data(), buffer.size(), &required));
    return std::string(buffer.data());
}

Event read_event(const std::string &path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open event: " + path);
    }
    std::string line;
    bool in_momenta = false;
    bool in_helicities = false;
    std::size_t declared_helicities = 0;
    Event event;
    while (std::getline(stream, line)) {
        std::istringstream fields(line);
        std::string tag;
        fields >> tag;
        if (tag.empty()) {
            continue;
        }
        if (tag == "BEGIN_MOMENTA") {
            in_momenta = true;
            continue;
        }
        if (tag == "END_MOMENTA") {
            in_momenta = false;
            continue;
        }
        if (tag == "NHELICITIES") {
            fields >> declared_helicities;
            continue;
        }
        if (tag == "BEGIN_HELICITIES") {
            in_helicities = true;
            continue;
        }
        if (tag == "END_HELICITIES") {
            in_helicities = false;
            continue;
        }
        if (in_momenta) {
            double energy = 0.0;
            double px = 0.0;
            double py = 0.0;
            double pz = 0.0;
            std::istringstream row(line);
            if (!(row >> energy >> px >> py >> pz)) {
                throw std::runtime_error("invalid momentum row in " + path);
            }
            event.momenta.insert(event.momenta.end(), {energy, px, py, pz});
        } else if (in_helicities) {
            std::istringstream row(line);
            int value = 0;
            while (row >> value) {
                if (value != -1 && value != 1) {
                    throw std::runtime_error("invalid helicity value in " + path);
                }
                event.helicities.push_back(static_cast<std::int32_t>(value));
            }
        }
    }
    if (event.momenta.empty() || event.momenta.size() % 4 != 0) {
        throw std::runtime_error("event has no complete momenta: " + path);
    }
    if (declared_helicities != 1 ||
        event.helicities.size() != event.momenta.size() / 4) {
        throw std::runtime_error(
            "candidate event must contain exactly one complete helicity: " + path);
    }
    return event;
}

std::uint64_t process_peak_rss_kib() {
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage(RUSAGE_SELF) failed");
    }
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss) / 1024U;
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#endif
}

double evaluate_one(
    RusticolRuntimeHandle *runtime,
    const std::vector<double> &point,
    const std::uint32_t *selected_helicity_index,
    double &minimum_absolute_value,
    double &sink) {
    double value = 0.0;
    check_rusticol(rusticol_runtime_evaluate_selected_f64(
        runtime,
        point.data(),
        point.size(),
        1,
        nullptr,
        0,
        nullptr,
        0,
        selected_helicity_index,
        1,
        nullptr,
        0,
        &value,
        1));
    if (!std::isfinite(value)) {
        throw std::runtime_error("Rusticol returned an invalid scalar result");
    }
    const double absolute = std::abs(value);
    minimum_absolute_value = std::min(minimum_absolute_value, absolute);
    sink += value;
    return value;
}

struct Arguments {
    std::string artifact;
    std::string process;
    double target_seconds = 0.25;
    std::size_t samples = kWarmSampleCount;
    bool first_ready_only = false;
    std::vector<std::string> event_paths;
};

Arguments parse_arguments(int argc, char **argv) {
    if (argc < 6) {
        throw std::runtime_error(
            "usage: fft_gluon_candidate_probe ARTIFACT PROCESS "
            "--target-seconds S --samples N [--first-ready-only] EVENT...");
    }
    Arguments arguments;
    arguments.artifact = argv[1];
    arguments.process = argv[2];
    for (int index = 3; index < argc; ++index) {
        const std::string token = argv[index];
        if (token == "--target-seconds") {
            if (++index >= argc) {
                throw std::runtime_error("--target-seconds requires a value");
            }
            arguments.target_seconds = std::stod(argv[index]);
        } else if (token == "--samples") {
            if (++index >= argc) {
                throw std::runtime_error("--samples requires a value");
            }
            arguments.samples = static_cast<std::size_t>(std::stoull(argv[index]));
        } else if (token == "--first-ready-only") {
            arguments.first_ready_only = true;
        } else if (!token.empty() && token[0] == '-') {
            throw std::runtime_error("unknown option: " + token);
        } else {
            arguments.event_paths.push_back(token);
        }
    }
    if (!std::isfinite(arguments.target_seconds) ||
        arguments.target_seconds < 0.25) {
        throw std::runtime_error("calibration target must be at least 0.25 seconds");
    }
    if (arguments.samples != kWarmSampleCount) {
        throw std::runtime_error("the acceptance probe requires exactly 10 samples");
    }
    if (arguments.event_paths.size() != kPointCount) {
        throw std::runtime_error("the acceptance probe requires exactly 10 events");
    }
    return arguments;
}

void write_first_ready(
    const std::string &process,
    const std::string &execution_mode,
    const std::size_t helicity_count,
    const std::string &selected_helicity_id,
    const double load_seconds,
    const double first_warm_seconds,
    const double warm_up_api_seconds,
    const double minimum_absolute_value,
    const std::uint64_t peak_rss_kib) {
    std::cout << std::setprecision(17) << std::scientific
              << "FFT_CANDIDATE_FIRST_READY_V1\n"
              << "PROCESS " << process << "\n"
              << "EXECUTION_MODE " << execution_mode << "\n"
              << "TIMER_SOURCE process-cpu-time\n"
              << "HELICITY_COVERAGE_COUNT " << helicity_count << "\n"
              << "SELECTED_HELICITY_ID " << selected_helicity_id << "\n"
              << "POINT_COUNT " << kPointCount << "\n"
              << "LOAD_SECONDS " << load_seconds << "\n"
              << "FIRST_WARM_SECONDS " << first_warm_seconds << "\n"
              << "WARM_UP_API_SECONDS " << warm_up_api_seconds << "\n"
              << "MIN_ABSOLUTE_VALUE " << minimum_absolute_value << "\n"
              << "MAX_RSS_KIB " << peak_rss_kib << "\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        std::vector<Event> events;
        events.reserve(arguments.event_paths.size());
        for (const auto &path : arguments.event_paths) {
            events.push_back(read_event(path));
        }
        const auto expected_helicity = events.front().helicities;
        const std::size_t point_size = events.front().momenta.size();
        for (const auto &event : events) {
            if (event.momenta.size() != point_size ||
                event.helicities != expected_helicity) {
                throw std::runtime_error(
                    "all candidate events must share one multiplicity and helicity");
            }
        }

        const double load_start = process_cpu_seconds();
        if (rusticol_abi_version() != RUSTICOL_ABI_VERSION) {
            throw std::runtime_error("Rusticol public C ABI version mismatch");
        }
        RuntimeHandle runtime(arguments.artifact, arguments.process);
        RusticolRuntimeHandle *const handle = runtime.get();
        check_rusticol(rusticol_runtime_set_model_parameter(
            handle,
            "normalization.alpha_s_me_check",
            kUnitCouplingAlphaS,
            0.0));
        const std::string process = runtime_string(handle, rusticol_runtime_process_key);
        const std::string execution_mode =
            runtime_string(handle, rusticol_runtime_execution_mode);
        if (runtime_string(handle, rusticol_runtime_color_accuracy) != "full") {
            throw std::runtime_error("candidate artifact is not full colour");
        }
        if (execution_mode != "on-the-fly" && execution_mode != "recurrence") {
            throw std::runtime_error("candidate artifact has an unsupported execution mode");
        }
        std::size_t external_count = 0;
        check_rusticol(rusticol_runtime_external_count(handle, &external_count));
        if (external_count * 4 != point_size) {
            throw std::runtime_error("candidate event multiplicity does not match artifact");
        }
        std::size_t runtime_helicity_count = 0;
        check_rusticol(rusticol_runtime_helicity_count(
            handle, &runtime_helicity_count));
        std::string selected_helicity_id;
        std::uint32_t selected_helicity_index = 0;
        std::size_t matching_helicity_count = 0;
        for (std::size_t index = 0; index < runtime_helicity_count; ++index) {
            std::size_t required = 0;
            check_rusticol(rusticol_runtime_helicity_vector(
                handle, index, nullptr, 0, &required));
            if (required != expected_helicity.size()) {
                continue;
            }
            std::vector<std::int32_t> helicity(required);
            check_rusticol(rusticol_runtime_helicity_vector(
                handle, index, helicity.data(), helicity.size(), &required));
            if (helicity == expected_helicity) {
                ++matching_helicity_count;
                selected_helicity_id = runtime_helicity_id(handle, index);
                if (index > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error(
                        "selected helicity index exceeds the public C ABI");
                }
                selected_helicity_index = static_cast<std::uint32_t>(index);
            }
        }
        if (matching_helicity_count != 1) {
            throw std::runtime_error(
                "candidate artifact does not expose one matching selected helicity");
        }
        const char *selected_helicity_ids[] = {selected_helicity_id.c_str()};
        const double load_seconds = process_cpu_seconds() - load_start;

        double minimum_absolute_value = std::numeric_limits<double>::infinity();
        double sink = 0.0;
        std::array<double, kPointCount> point_values {};
        std::array<bool, kPointCount> point_value_recorded {};
        RusticolWarmUpResult warm_up {};
        double warm_up_api_seconds = 0.0;
        const double first_warm_start = process_cpu_seconds();
        std::size_t first_evaluated_point = 0;
        if (execution_mode == "on-the-fly") {
            check_rusticol(rusticol_runtime_warm_up_f64(
                handle,
                events.front().momenta.data(),
                events.front().momenta.size(),
                selected_helicity_ids,
                1,
                nullptr,
                0,
                nullptr,
                nullptr,
                &warm_up));
            if (warm_up.schema_version != 1 ||
                warm_up.first_evaluation_completed == 0 ||
                !std::isfinite(warm_up.elapsed_seconds) ||
                !(warm_up.elapsed_seconds > 0.0)) {
                throw std::runtime_error("Rusticol returned an invalid OTF warm-up result");
            }
            warm_up_api_seconds = warm_up.elapsed_seconds;
            first_evaluated_point = 1;
        }
        for (std::size_t point = first_evaluated_point;
             point < events.size();
             ++point) {
            point_values[point] = evaluate_one(
                handle,
                events[point].momenta,
                &selected_helicity_index,
                minimum_absolute_value,
                sink);
            point_value_recorded[point] = true;
        }
        const double first_warm_seconds =
            process_cpu_seconds() - first_warm_start;

        std::uint64_t peak_rss_kib = process_peak_rss_kib();
        if (warm_up.peak_rss_available != 0) {
            peak_rss_kib = std::max(
                peak_rss_kib,
                (warm_up.peak_rss_bytes + 1023U) / 1024U);
        }
        if (!std::isfinite(sink) || !std::isfinite(minimum_absolute_value)) {
            throw std::runtime_error("candidate first-pass sink is invalid");
        }
        if (arguments.first_ready_only) {
            write_first_ready(
                process,
                execution_mode,
                runtime_helicity_count,
                selected_helicity_id,
                load_seconds,
                first_warm_seconds,
                warm_up_api_seconds,
                minimum_absolute_value,
                peak_rss_kib);
            return 0;
        }

        std::size_t calibration_calls = 0;
        double calibration_seconds = 0.0;
        std::size_t repetitions = 1;
        for (;;) {
            const double calibration_start = process_cpu_seconds();
            for (std::size_t repetition = 0;
                 repetition < repetitions;
                 ++repetition) {
                const double value = evaluate_one(
                    handle,
                    events[kRepresentativePoint].momenta,
                    &selected_helicity_index,
                    minimum_absolute_value,
                    sink);
                if (!point_value_recorded[kRepresentativePoint]) {
                    point_values[kRepresentativePoint] = value;
                    point_value_recorded[kRepresentativePoint] = true;
                }
            }
            const double elapsed = process_cpu_seconds() - calibration_start;
            if (elapsed >= arguments.target_seconds) {
                calibration_calls = repetitions;
                calibration_seconds = elapsed;
                break;
            }
            if (repetitions > std::numeric_limits<std::size_t>::max() / 2) {
                throw std::runtime_error("calibration repetition count overflow");
            }
            repetitions *= 2;
        }

        std::array<double, kWarmSampleCount> warm_cells {};
        for (std::size_t sample = 0; sample < arguments.samples; ++sample) {
            const double cell_start = process_cpu_seconds();
            for (std::size_t repetition = 0;
                 repetition < calibration_calls;
                 ++repetition) {
                evaluate_one(
                    handle,
                    events[kRepresentativePoint].momenta,
                    &selected_helicity_index,
                    minimum_absolute_value,
                    sink);
            }
            warm_cells[sample] =
                (process_cpu_seconds() - cell_start) /
                static_cast<double>(calibration_calls);
        }

        peak_rss_kib = std::max(peak_rss_kib, process_peak_rss_kib());
        if (!std::isfinite(sink) || !std::isfinite(minimum_absolute_value)) {
            throw std::runtime_error("candidate benchmark sink is invalid");
        }
        if (!std::all_of(
                point_value_recorded.begin(),
                point_value_recorded.end(),
                [](const bool recorded) { return recorded; }) ||
            !std::all_of(
                point_values.begin(),
                point_values.end(),
                [](const double value) {
                    return std::isfinite(value);
                })) {
            throw std::runtime_error("candidate point-value evidence is invalid");
        }

        std::cout << std::setprecision(17) << std::scientific
                  << "FFT_CANDIDATE_PROBE_V4\n"
                  << "PROCESS " << process << "\n"
                  << "EXECUTION_MODE " << execution_mode << "\n"
                  << "TIMER_SOURCE process-cpu-time\n"
                  << "HELICITY_COVERAGE_COUNT " << runtime_helicity_count << "\n"
                  << "SELECTED_HELICITY_ID " << selected_helicity_id << "\n"
                  << "POINT_COUNT " << events.size() << "\n"
                  << "LOAD_SECONDS " << load_seconds << "\n"
                  << "FIRST_WARM_SECONDS " << first_warm_seconds << "\n"
                  << "WARM_UP_API_SECONDS " << warm_up_api_seconds << "\n";
        for (std::size_t point = 0; point < events.size(); ++point) {
            std::cout << "POINT_VALUE " << (point + 1) << " "
                      << point_values[point] << "\n";
        }
        std::cout << "CALIBRATION_CELL " << (kRepresentativePoint + 1) << " "
                  << calibration_calls << " " << calibration_seconds << "\n";
        for (std::size_t sample = 0; sample < arguments.samples; ++sample) {
            std::cout << "WARM_CELL_SECONDS " << (sample + 1) << " "
                      << (kRepresentativePoint + 1) << " "
                      << warm_cells[sample] << "\n";
        }
        std::cout << "MIN_ABSOLUTE_VALUE " << minimum_absolute_value << "\n"
                  << "MAX_RSS_KIB " << peak_rss_kib << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "fft candidate probe: " << error.what() << "\n";
        return 1;
    }
}
